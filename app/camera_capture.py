"""Captura da câmera física em thread dedicada.

Abrir uma câmera e ler um quadro são chamadas bloqueantes: uma webcam em pouca
luz entrega 5-10 FPS e uma câmera ausente pode travar a abertura por segundos.
Feitas na thread da interface, essas esperas congelam o aplicativo inteiro.
Aqui a thread de captura absorve o bloqueio e a interface só consulta o quadro
mais recente, sem nunca esperar pelo hardware.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CameraOpener = Callable[[int, int, int, int], Any]

HARDWARE_FPS = 30
HARDWARE_BUFFER_FRAMES = 1
DEFAULT_RETRY_INTERVAL_SEC = 1.0
STALE_FRAME_SEC = 2.0
STOP_JOIN_TIMEOUT_SEC = 2.0
READ_LOOP_YIELD_SEC = 0.001
PREDECESSOR_POLL_SEC = 0.1
CAPTURE_ERRORS = (cv2.error, OSError, RuntimeError, TypeError)


def apply_capture_format(cap: Any, width: int, height: int, fps: int) -> None:
    """Pede ao dispositivo o formato desejado (o driver escolhe o mais próximo suportado)."""
    with contextlib.suppress(*CAPTURE_ERRORS):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def open_camera_device(camera_index: int, width: int, height: int, fps: int) -> Any:
    """Abre a câmera já na resolução pedida, em MJPG e com buffer mínimo.

    Capturar na resolução de trabalho evita decodificar 1280x720 para depois
    reduzir a cada quadro, e o buffer curto evita exibir quadros atrasados.
    O formato vai junto com a abertura: configurar depois reinicia o stream do
    driver a cada propriedade e atrasa o primeiro quadro.
    """
    if hasattr(cv2, "CAP_V4L2"):
        open_params = [
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_FRAME_WIDTH, int(width),
            cv2.CAP_PROP_FRAME_HEIGHT, int(height),
            cv2.CAP_PROP_FPS, int(fps),
        ]  # fmt: skip
        cap = None
        with contextlib.suppress(*CAPTURE_ERRORS):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2, open_params)
        if cap is not None and cap.isOpened():
            with contextlib.suppress(*CAPTURE_ERRORS):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, float(HARDWARE_BUFFER_FRAMES))
            return cap
        release_capture(cap)

    # Outros sistemas (ou driver que recusou os parâmetros): abertura genérica.
    cap = None
    with contextlib.suppress(*CAPTURE_ERRORS):
        cap = cv2.VideoCapture(camera_index)
    if cap is None or not cap.isOpened():
        release_capture(cap)
        return None
    with contextlib.suppress(*CAPTURE_ERRORS):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(HARDWARE_BUFFER_FRAMES))
    apply_capture_format(cap, width, height, fps)
    return cap


def release_capture(cap: Any) -> None:
    """Libera um dispositivo de captura ignorando falhas do driver."""
    if cap is not None:
        with contextlib.suppress(*CAPTURE_ERRORS):
            cap.release()


class BackgroundCapture:
    """Lê quadros de uma câmera continuamente e expõe apenas o mais recente."""

    def __init__(
        self,
        opener: CameraOpener,
        camera_index: int,
        width: int,
        height: int,
        fps: int = HARDWARE_FPS,
        retry_interval_sec: float = DEFAULT_RETRY_INTERVAL_SEC,
        stale_frame_sec: float = STALE_FRAME_SEC,
        predecessor: threading.Thread | None = None,
    ) -> None:
        """Configura a captura.

        `predecessor` é a thread de uma captura anterior que não encerrou a tempo
        (driver preso): esta só abre o dispositivo depois que aquela terminar,
        em vez de disputar a mesma câmera.
        """
        self._predecessor = predecessor
        self._opener = opener
        self._camera_index = camera_index
        self._format = (width, height, fps)
        self._retry_interval_sec = max(0.0, retry_interval_sec)
        self._stale_frame_sec = stale_frame_sec
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._sequence: int = 0

    def start(self) -> None:
        """Inicia a thread de captura (sem efeito se já estiver rodando)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"captura-camera-{self._camera_index}", daemon=True
        )
        self._thread.start()
        # Garante a liberação da câmera mesmo se o dono esquecer de chamar stop().
        atexit.register(self.stop)

    def stop(self) -> threading.Thread | None:
        """Encerra a thread; o dispositivo é liberado pela própria thread ao sair.

        Devolve a thread se ela ainda estiver presa no driver após o prazo, para
        que a próxima captura possa esperar por ela; caso contrário, None.
        """
        self._stop_event.set()
        atexit.unregister(self.stop)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                logger.warning(
                    "Thread da câmera #%d não encerrou em %.1fs; driver bloqueado na leitura",
                    self._camera_index,
                    STOP_JOIN_TIMEOUT_SEC,
                )
        self._thread = None
        return thread if thread is not None and thread.is_alive() else None

    @property
    def has_delivered_frame(self) -> bool:
        """True se a câmera já entregou ao menos um quadro desde o início."""
        return self._first_frame_event.is_set()

    def wait_for_frame(self, timeout_sec: float) -> bool:
        """Aguarda o primeiro quadro por até `timeout_sec`; True se chegou."""
        return self._first_frame_event.wait(timeout=max(0.0, timeout_sec))

    def read_latest(self) -> tuple[np.ndarray | None, int]:
        """Devolve (quadro, sequência) sem bloquear; quadro None se não há imagem recente.

        O array devolvido é compartilhado: quem recebe não deve alterá-lo.
        """
        with self._lock:
            if (
                self._latest_frame is None
                or (time.perf_counter() - self._latest_ts) > self._stale_frame_sec
            ):
                return None, self._sequence
            return self._latest_frame, self._sequence

    def _run(self) -> None:
        while self._predecessor is not None and self._predecessor.is_alive():
            if self._stop_event.wait(PREDECESSOR_POLL_SEC):
                return
        self._predecessor = None
        cap: Any = None
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    cap = self._try_open()
                    if cap is None:
                        self._stop_event.wait(self._retry_interval_sec)
                        continue
                frame = self._try_read(cap)
                if frame is None:
                    release_capture(cap)
                    cap = None
                    self._stop_event.wait(self._retry_interval_sec)
                    continue
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.perf_counter()
                    self._sequence += 1
                self._first_frame_event.set()
                self._stop_event.wait(READ_LOOP_YIELD_SEC)
        finally:
            release_capture(cap)

    def _try_open(self) -> Any:
        width, height, fps = self._format
        try:
            cap = self._opener(self._camera_index, width, height, fps)
        except CAPTURE_ERRORS as exc:
            logger.debug("Falha ao abrir a câmera #%d: %s", self._camera_index, exc)
            return None
        except Exception:
            # Erro inesperado não pode matar a thread: sem ela a câmera nunca voltaria.
            logger.exception("Erro inesperado ao abrir a câmera #%d", self._camera_index)
            return None
        if cap is None:
            return None
        try:
            if cap.isOpened():
                return cap
        except CAPTURE_ERRORS as exc:
            logger.debug("Câmera #%d inacessível após abertura: %s", self._camera_index, exc)
        release_capture(cap)
        return None

    def _try_read(self, cap: Any) -> np.ndarray | None:
        try:
            ok, frame = cap.read()
        except CAPTURE_ERRORS as exc:
            logger.debug("Falha de leitura na câmera #%d: %s", self._camera_index, exc)
            return None
        except Exception:
            logger.exception("Erro inesperado ao ler a câmera #%d", self._camera_index)
            return None
        if not ok or frame is None or frame.size == 0:
            return None
        return frame
