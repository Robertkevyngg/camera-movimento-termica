"""Gravação em vídeo dos eventos de movimento, em thread própria.

A thread de captura entrega TODOS os quadros (`receive`), inclusive os que a análise
não chegou a usar; aqui eles ficam guardados por alguns segundos ("pré-gravação").
Quando o `CameraManager` detecta movimento, `start` grava esse trecho anterior e tudo o
que vier até `stop`. Os comandos levam o horário do quadro, então o corte é exato mesmo
se a gravação estiver um pouco atrás da câmera.

Dois formatos:
  * JPEG direto (Linux, câmera em MJPG): os quadros JPEG da câmera são copiados para um
    AVI sem recomprimir. Custo quase zero de CPU e a qualidade original; 1080p a 60 fps
    grava sem perdas mesmo em computador simples.
  * Recomprimido (demais casos): cada quadro vira MP4 (mp4v), com data e hora na imagem.

Junto de cada vídeo vai `<vídeo>_tempos.csv` com o horário exato de cada quadro.
Se o computador não acompanhar, quadros excedentes são descartados e contados em
`dropped_frames`: a análise em tempo real nunca espera a gravação.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import queue
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_FILE_SEC = 60.0  # eventos longos são divididos em partes
MAX_AVI_BYTES = 900_000_000  # AVI simples abaixo de 1 GB abre em qualquer player
MAX_PREROLL_BYTES = 400_000_000
MAX_QUEUE_BYTES = 400_000_000
JOIN_TIMEOUT_SEC = 30.0


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _list(kind: bytes, content: bytes) -> bytes:
    return b"LIST" + struct.pack("<I", 4 + len(content)) + kind + content


class MjpegAviWriter:
    """Escreve quadros JPEG prontos num AVI (Motion-JPEG), sem recomprimir."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self._fps, self._width, self._height = fps, width, height
        self._index: list[tuple[int, int]] = []
        self._largest = 0
        self._file = open(path, "wb")  # noqa: SIM115 - fechado em close()
        self._file.write(b"RIFF\0\0\0\0AVI " + self._header(0, 0))
        self._movi_pos = self._file.tell()
        self._file.write(b"LIST\0\0\0\0movi")

    @property
    def size_bytes(self) -> int:
        return self._file.tell()

    def _header(self, frames: int, largest: int) -> bytes:
        avih = struct.pack(
            "<14I", round(1e6 / self._fps), 0, 0, 0x10, frames, 0, 1, largest,
            self._width, self._height, 0, 0, 0, 0,
        )  # fmt: skip  (0x10 = AVIF_HASINDEX)
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0,
            1000, round(self._fps * 1000), 0, frames, largest, 0xFFFFFFFF, 0,
            0, 0, self._width, self._height,
        )  # fmt: skip
        strf = struct.pack(
            "<IiiHH4sIiiII", 40, self._width, self._height, 1, 24, b"MJPG",
            self._width * self._height * 3, 0, 0, 0, 0,
        )  # fmt: skip
        strl = _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf))
        return _list(b"hdrl", _chunk(b"avih", avih) + strl)

    def write(self, jpeg: np.ndarray | bytes) -> None:
        size = len(jpeg)
        self._index.append((self._file.tell() - (self._movi_pos + 8), size))
        self._largest = max(self._largest, size)
        self._file.write(b"00dc" + struct.pack("<I", size))
        self._file.write(jpeg)
        if size % 2:
            self._file.write(b"\0")

    def close(self) -> None:
        movi_end = self._file.tell()
        idx1 = b"".join(struct.pack("<4sIII", b"00dc", 0x10, pos, size) for pos, size in self._index)
        self._file.write(b"idx1" + struct.pack("<I", len(idx1)) + idx1)
        total = self._file.tell()
        self._file.seek(4)
        self._file.write(struct.pack("<I", total - 8))
        self._file.seek(12)
        self._file.write(self._header(len(self._index), self._largest))
        self._file.seek(self._movi_pos + 4)
        self._file.write(struct.pack("<I", movi_end - self._movi_pos - 8))
        self._file.close()


@dataclass
class _Item:
    t: float
    frame: np.ndarray | None
    jpeg: np.ndarray | None

    @property
    def nbytes(self) -> int:
        data = self.jpeg if self.jpeg is not None else self.frame
        return int(data.nbytes) if data is not None else 0


def _lower_thread_priority() -> None:
    """No Linux, a gravação cede CPU à captura e à análise quando o computador apertar."""
    if sys.platform.startswith("linux"):
        with contextlib.suppress(AttributeError, OSError):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)


def _stamp(frame: np.ndarray, text: str) -> None:
    scale = frame.shape[1] / 1000
    thickness = max(1, round(scale * 2))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = round(10 * scale), frame.shape[0] - round(12 * scale)
    pad = max(2, round(4 * scale))
    cv2.rectangle(frame, (x - pad, y - h - pad), (x + w + pad, y + base + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def format_timestamp(moment: datetime) -> str:
    """06/11/2026 14:03:22,123 (horário local com milissegundos)."""
    return moment.strftime("%d/%m/%Y %H:%M:%S,") + f"{moment.microsecond // 1000:03d}"


class EventRecorder:
    """Grava vídeos dos eventos a partir dos quadros entregues pela captura."""

    def __init__(
        self,
        fps_provider: Callable[[], float],
        pre_seconds: float = 2.0,
        clock_origin: datetime | None = None,
    ) -> None:
        """`fps_provider` informa o fps real da câmera quando um vídeo começa.

        `clock_origin` converte os horários de time.monotonic em data/hora local.
        """
        self._fps_provider = fps_provider
        self.pre_seconds = pre_seconds
        self._origin = clock_origin or (datetime.now() - timedelta(seconds=time.monotonic()))
        self._preroll: collections.deque[_Item] = collections.deque()
        self._preroll_bytes = 0
        self._queue: queue.SimpleQueue[_Item | None] = queue.SimpleQueue()
        self._queued_bytes = 0
        self._bytes_lock = threading.Lock()
        self._commands: queue.SimpleQueue[tuple[str, float, Path | None]] = queue.SimpleQueue()
        self._writer: MjpegAviWriter | cv2.VideoWriter | None = None
        self._raw = False
        self._path: Path | None = None
        self._base: Path | None = None
        self._part = 1
        self._file_start = 0.0
        self._size = (0, 0)
        self._times: list[float] = []
        self.recorded_frames = 0
        self.dropped_frames = 0
        self.files: list[Path] = []
        self._thread = threading.Thread(target=self._run, name="gravacao-eventos", daemon=True)
        self._thread.start()

    def moment(self, t: float) -> datetime:
        """Data/hora local de um horário em time.monotonic."""
        return self._origin + timedelta(seconds=t)

    # --- chamado pela thread de captura, para cada quadro ---
    def receive(self, t: float, frame: np.ndarray, jpeg: np.ndarray | None = None) -> None:
        item = _Item(t, None if jpeg is not None else frame, jpeg)
        with self._bytes_lock:
            if self._queued_bytes + item.nbytes > MAX_QUEUE_BYTES:
                self.dropped_frames += 1
                return
            self._queued_bytes += item.nbytes
        self._queue.put(item)

    # --- chamados pela thread de análise ---
    def start(self, t: float, base_path: Path) -> None:
        """Grava o evento que começou no horário t (mais a pré-gravação)."""
        self._commands.put(("start", t, base_path))

    def stop(self, t: float) -> None:
        self._commands.put(("stop", t, None))

    def close(self) -> None:
        """Grava o que estiver na fila e fecha o arquivo."""
        self._queue.put(None)
        self._thread.join(timeout=JOIN_TIMEOUT_SEC)

    # --- thread de gravação ---
    def _run(self) -> None:
        _lower_thread_priority()
        pending: collections.deque[tuple[str, float, Path | None]] = collections.deque()
        while True:
            item = self._queue.get()
            while True:
                try:
                    pending.append(self._commands.get_nowait())
                except queue.Empty:
                    break
            if item is None:
                while pending:
                    self._apply(pending.popleft(), None)
                self._close_file()
                return
            with self._bytes_lock:
                self._queued_bytes -= item.nbytes
            try:
                while pending and pending[0][1] <= item.t:
                    self._apply(pending.popleft(), item)
                if self._writer is None:
                    self._keep_preroll(item)
                    continue
                self._write(item)
                too_big = isinstance(self._writer, MjpegAviWriter) and self._writer.size_bytes >= MAX_AVI_BYTES
                if item.t - self._file_start >= MAX_FILE_SEC or too_big:
                    self._close_file()
                    self._part += 1
                    self._open_file(item.t, item)
            except Exception:
                # Erro de disco (cheio, sem permissão) não pode derrubar a captura.
                logger.exception("Falha ao gravar vídeo; gravação do evento interrompida")
                with contextlib.suppress(Exception):
                    self._close_file()
                self._writer = None

    def _keep_preroll(self, item: _Item) -> None:
        self._preroll.append(item)
        self._preroll_bytes += item.nbytes
        while self._preroll and (
            item.t - self._preroll[0].t > self.pre_seconds or self._preroll_bytes > MAX_PREROLL_BYTES
        ):
            self._preroll_bytes -= self._preroll.popleft().nbytes

    def _apply(self, command: tuple[str, float, Path | None], current: _Item | None) -> None:
        kind, t_cmd, base = command
        self._close_file()
        if kind == "start" and base is not None:
            self._base, self._part = base, 1
            self._open_file(current.t if current is not None else t_cmd, current)

    def _open_file(self, t: float, current: _Item | None = None) -> None:
        """Abre o arquivo e grava a pré-gravação. `current` é o quadro que ainda vai ser
        gravado; serve de referência de formato se a pré-gravação estiver vazia."""
        assert self._base is not None
        first = self._preroll[0] if self._preroll else current
        if first is None:
            return
        self._raw = first.jpeg is not None
        sample = cv2.imdecode(first.jpeg, cv2.IMREAD_COLOR) if first.jpeg is not None else first.frame
        if sample is None:
            logger.error("Quadro inicial inválido; evento não gravado")
            return
        height, width = sample.shape[:2]
        fps = max(1.0, min(120.0, float(self._fps_provider() or 30.0)))
        name = self._base.name + (f"_parte{self._part}" if self._part > 1 else "")
        self._base.parent.mkdir(parents=True, exist_ok=True)
        if self._raw:
            self._path = self._base.with_name(name + ".avi")
            self._writer = MjpegAviWriter(self._path, round(fps, 2), width, height)
        else:
            self._path = self._base.with_name(name + ".mp4")
            writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), round(fps, 2), (width, height))
            if not writer.isOpened():
                self._path = self._base.with_name(name + ".avi")
                writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"MJPG"), round(fps, 2), (width, height))
            if not writer.isOpened():
                logger.error("Não consegui criar o vídeo %s", self._path)
                return
            self._writer = writer
        self._size = (width, height)
        self.files.append(self._path)
        self._file_start = t
        self._times = []
        while self._preroll:
            self._write(self._preroll.popleft())
        self._preroll_bytes = 0

    def _write(self, item: _Item) -> None:
        if self._raw:
            if item.jpeg is None:
                self.dropped_frames += 1
                return
            assert isinstance(self._writer, MjpegAviWriter)
            self._writer.write(item.jpeg)
        else:
            frame = item.frame
            if frame is None and item.jpeg is not None:
                frame = cv2.imdecode(item.jpeg, cv2.IMREAD_COLOR)
            if frame is None:
                self.dropped_frames += 1
                return
            if (frame.shape[1], frame.shape[0]) != self._size:
                frame = cv2.resize(frame, self._size)
            else:
                frame = frame.copy()  # o mesmo quadro pode estar em análise na outra thread
            _stamp(frame, format_timestamp(self.moment(item.t)))
            assert isinstance(self._writer, cv2.VideoWriter)
            self._writer.write(frame)
        self._times.append(item.t)
        self.recorded_frames += 1

    def _close_file(self) -> None:
        writer, self._writer = self._writer, None
        if writer is None:
            return
        if isinstance(writer, MjpegAviWriter):
            writer.close()
        else:
            writer.release()
        if self._times and self._path is not None:
            t0 = self._times[0]
            lines = []
            for i, t in enumerate(self._times, 1):
                elapsed = f"{t - t0:.3f}".replace(".", ",")  # vírgula decimal (Excel em português)
                lines.append(f"{i};{format_timestamp(self.moment(t))};{elapsed}")
            csv_path = self._path.with_name(self._path.stem + "_tempos.csv")
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                f.write("quadro;data_hora;tempo_s\r\n" + "\r\n".join(lines) + "\r\n")
