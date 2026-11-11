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
import os
import struct
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CameraOpener = Callable[[int, int, int, int], Any]
# Recebe cada quadro capturado: (horário em time.monotonic, imagem BGR, bytes JPEG ou None).
FrameListener = Callable[[float, np.ndarray, "np.ndarray | None"], None]

HARDWARE_FPS = 30
LOW_HARDWARE_FPS = 15
HIGH_HARDWARE_FPS = 60
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


def preferred_backends() -> tuple[int, ...]:
    """Backends de captura a tentar primeiro, conforme o sistema operacional."""
    if sys.platform.startswith("linux"):
        return (cv2.CAP_V4L2,)
    if sys.platform == "win32":
        # DirectShow abre webcams bem mais rápido que o backend padrão (MSMF).
        return (cv2.CAP_DSHOW,)
    return ()


def choose_hardware_fps(target_fps: int) -> int:
    """Taxa pedida ao hardware: reduzida quando o alvo é baixo, para decodificar menos,
    e 60 quando o alvo passa de 30 (câmeras 1080p60, como Logitech StreamCam/Brio)."""
    if target_fps <= LOW_HARDWARE_FPS:
        return LOW_HARDWARE_FPS
    if target_fps <= HARDWARE_FPS:
        return HARDWARE_FPS
    return HIGH_HARDWARE_FPS


# --- Controles do driver de vídeo do Linux (V4L2), sem programas extras -----------------
VIDIOC_G_CTRL = 0xC008561B  # _IOWR('V', 27, struct v4l2_control)
VIDIOC_S_CTRL = 0xC008561C  # _IOWR('V', 28, struct v4l2_control)
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903  # "exposure_dynamic_framerate"


def v4l2_control(camera_index: int, control_id: int, value: int | None = None) -> int | None:
    """Lê (value=None) ou ajusta um controle de /dev/video<camera_index>.

    Devolve o valor vigente, ou None fora do Linux ou se a câmera não tiver o controle.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        import fcntl

        fd = os.open(f"/dev/video{camera_index}", os.O_RDWR | os.O_NONBLOCK)
    except (ImportError, OSError):
        return None
    try:
        if value is not None:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", control_id, int(value)))
        reply = fcntl.ioctl(fd, VIDIOC_G_CTRL, struct.pack("Ii", control_id, 0))
        return int(struct.unpack("Ii", reply)[1])
    except OSError:
        return None
    finally:
        os.close(fd)


def is_v4l2_capture(cap: Any) -> bool:
    """True se `cap` é uma câmera real lida pelo backend V4L2 do Linux."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return bool(cap.getBackendName() == "V4L2")
    except (AttributeError, *CAPTURE_ERRORS):
        return False


def capture_fourcc(cap: Any) -> str:
    """Formato de pixel negociado com a câmera (ex.: 'MJPG', 'YUYV')."""
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    except (TypeError, ValueError, *CAPTURE_ERRORS):
        return ""
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\0 ")


def enable_raw_mjpeg(cap: Any) -> bool:
    """Pede ao OpenCV os quadros ainda em JPEG, como saem da câmera (Linux/V4L2 em MJPG).

    A gravação passa a só copiar esses bytes, sem recomprimir. Se o driver não entregar
    JPEG válido, volta ao modo normal e devolve False.
    """
    if not is_v4l2_capture(cap) or capture_fourcc(cap) != "MJPG":
        return False
    with contextlib.suppress(*CAPTURE_ERRORS):
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        for _ in range(5):
            ok, data = cap.read()
            if (
                ok
                and data is not None
                and data.dtype == np.uint8
                and data.size > 100
                and (data.ndim == 1 or data.shape[0] == 1)
                and bytes(data.reshape(-1)[:2]) == b"\xff\xd8"
            ):
                return True
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    return False


def driver_timestamp(cap: Any, now: float) -> float:
    """Horário em que o driver V4L2 recebeu o quadro, no relógio de time.monotonic.

    Se o computador atrasar, o quadro chega tarde ao programa mas com o horário
    verdadeiro. Fora do V4L2 (ou valor implausível) devolve `now`.
    """
    with contextlib.suppress(TypeError, ValueError, *CAPTURE_ERRORS):
        captured = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if 0.0 <= now - captured < 1.0:
            return captured
    return now


def open_camera_device(camera_index: int, width: int, height: int, fps: int) -> Any:
    """Abre a câmera já na resolução pedida, em MJPG e com buffer mínimo.

    Capturar na resolução de trabalho evita decodificar 1280x720 para depois
    reduzir a cada quadro, e o buffer curto evita exibir quadros atrasados.
    O formato vai junto com a abertura: configurar depois reinicia o stream do
    driver a cada propriedade e atrasa o primeiro quadro.
    """
    open_params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FRAME_WIDTH, int(width),
        cv2.CAP_PROP_FRAME_HEIGHT, int(height),
        cv2.CAP_PROP_FPS, int(fps),
    ]  # fmt: skip
    for backend in preferred_backends():
        cap = None
        with contextlib.suppress(*CAPTURE_ERRORS):
            cap = cv2.VideoCapture(camera_index, backend, open_params)
        if cap is not None and cap.isOpened():
            with contextlib.suppress(*CAPTURE_ERRORS):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, float(HARDWARE_BUFFER_FRAMES))
            return cap
        release_capture(cap)

    # Backend preferido indisponível ou recusou os parâmetros: abertura genérica.
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
        realtime_tuning: bool = False,
    ) -> None:
        """Configura a captura.

        `predecessor` é a thread de uma captura anterior que não encerrou a tempo
        (driver preso): esta só abre o dispositivo depois que aquela terminar,
        em vez de disputar a mesma câmera.

        `realtime_tuning` (câmera real no Linux): recebe os quadros JPEG direto da
        câmera, usa o horário do driver e pede fps fixo (a câmera não baixa o fps
        com pouca luz). O ajuste de fps é desfeito quando a captura termina.

        `frame_listener` (atributo) recebe todos os quadros capturados, inclusive os
        que a análise não chegar a consumir; é por ele que a gravação recebe o vídeo.
        """
        self.frame_listener: FrameListener | None = None
        self._realtime_tuning = realtime_tuning
        self._raw_jpeg = False
        self._uses_driver_clock = False
        self._restore_auto_priority: int | None = None
        self._frame_interval_ema: float | None = None
        self._last_capture_ts: float | None = None
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

    @property
    def camera_fps(self) -> float:
        """Quadros por segundo que a câmera está entregando (média móvel)."""
        with self._lock:
            interval = self._frame_interval_ema
        return 1.0 / interval if interval else 0.0

    @property
    def raw_jpeg(self) -> bool:
        """True se os quadros chegam em JPEG direto da câmera (gravação sem recomprimir)."""
        return self._raw_jpeg

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
                    self._tune_for_realtime(cap)
                frame, jpeg = self._try_read(cap)
                if frame is None:
                    self._restore_tuning()
                    release_capture(cap)
                    cap = None
                    self._stop_event.wait(self._retry_interval_sec)
                    continue
                now = time.monotonic()
                captured_at = driver_timestamp(cap, now) if self._uses_driver_clock else now
                listener = self.frame_listener
                if listener is not None:
                    try:
                        listener(captured_at, frame, jpeg)
                    except Exception:
                        logger.exception("Falha no receptor de quadros; desativado")
                        self.frame_listener = None
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.perf_counter()
                    self._sequence += 1
                    if self._last_capture_ts is not None:
                        dt = captured_at - self._last_capture_ts
                        if dt > 0:
                            ema = self._frame_interval_ema
                            self._frame_interval_ema = dt if ema is None else 0.95 * ema + 0.05 * dt
                    self._last_capture_ts = captured_at
                self._first_frame_event.set()
                self._stop_event.wait(READ_LOOP_YIELD_SEC)
        finally:
            self._restore_tuning()
            release_capture(cap)

    def _tune_for_realtime(self, cap: Any) -> None:
        """Ajustes de tempo real para câmera física no Linux (sem efeito em outros casos)."""
        self._raw_jpeg = False
        self._uses_driver_clock = False
        if not self._realtime_tuning or not is_v4l2_capture(cap):
            return
        self._uses_driver_clock = True
        original = v4l2_control(self._camera_index, V4L2_CID_EXPOSURE_AUTO_PRIORITY)
        if original is not None and original != 0:
            if v4l2_control(self._camera_index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0) == 0:
                self._restore_auto_priority = original
                logger.info("Câmera #%d: fps fixo (não reduz com pouca luz)", self._camera_index)
        self._raw_jpeg = enable_raw_mjpeg(cap)
        if self._raw_jpeg:
            logger.info("Câmera #%d: quadros JPEG direto da câmera", self._camera_index)

    def _restore_tuning(self) -> None:
        if self._restore_auto_priority is not None:
            v4l2_control(
                self._camera_index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, self._restore_auto_priority
            )
            self._restore_auto_priority = None

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

    def _try_read(self, cap: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Lê um quadro. Devolve (imagem BGR, bytes JPEG originais ou None)."""
        try:
            ok, frame = cap.read()
        except CAPTURE_ERRORS as exc:
            logger.debug("Falha de leitura na câmera #%d: %s", self._camera_index, exc)
            return None, None
        except Exception:
            logger.exception("Erro inesperado ao ler a câmera #%d", self._camera_index)
            return None, None
        if not ok or frame is None or frame.size == 0:
            return None, None
        if not self._raw_jpeg:
            return frame, None
        jpeg = frame.reshape(-1)
        try:
            image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        except cv2.error:
            image = None
        if image is None:
            return None, None
        return image, jpeg
