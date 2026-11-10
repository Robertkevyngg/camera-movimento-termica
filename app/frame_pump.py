"""Laço de processamento de vídeo em thread própria.

Capturar, analisar e compor o quadro fora da thread da interface mantém a
janela respondendo a cliques e teclas mesmo quando um quadro demora (CPU fraca,
resolução alta). A interface apenas consulta o quadro pronto mais recente.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import NamedTuple

import numpy as np

from app.camera_manager import CameraManager
from app.config import FrameTelemetry

logger = logging.getLogger(__name__)

MIN_PUMP_SLEEP_SEC = 0.002
PAUSED_POLL_SEC = 0.05
ERROR_BACKOFF_SEC = 0.5
STOP_JOIN_TIMEOUT_SEC = 3.0


class PumpedFrame(NamedTuple):
    """Quadro composto mais recente, sua telemetria e o número de sequência."""

    frame_bgr: np.ndarray
    telemetry: FrameTelemetry
    sequence: int


class FramePump:
    """Executa `CameraManager.step_frame` continuamente no ritmo do FPS alvo."""

    def __init__(self, manager: CameraManager) -> None:
        self._manager = manager
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: PumpedFrame | None = None
        self._sequence = 0

    def start(self) -> None:
        """Inicia a thread de processamento (sem efeito se já estiver rodando)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pipeline-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Encerra a thread; deve ser chamado antes de fechar o CameraManager."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SEC)
        self._thread = None

    def set_paused(self, paused: bool) -> None:
        """Pausa ou retoma o processamento sem encerrar a thread."""
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def poll(self) -> PumpedFrame | None:
        """Devolve o quadro pronto mais recente, sem bloquear (None antes do primeiro)."""
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._paused.is_set():
                self._stop_event.wait(PAUSED_POLL_SEC)
                continue
            started = time.perf_counter()
            try:
                frame_bgr, telemetry = self._manager.step_frame()
            except Exception:
                # Um quadro com erro não pode derrubar o vídeo: registra e tenta de novo.
                logger.exception("Falha ao processar quadro de vídeo")
                self._stop_event.wait(ERROR_BACKOFF_SEC)
                continue
            with self._lock:
                self._sequence += 1
                self._latest = PumpedFrame(frame_bgr, telemetry, self._sequence)
            elapsed = time.perf_counter() - started
            remaining = self._manager.get_frame_interval_sec() - elapsed
            self._stop_event.wait(max(MIN_PUMP_SLEEP_SEC, remaining))
