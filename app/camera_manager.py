"""Orquestrador de captura de câmera, simulação, resolução/FPS e telemetria."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import cv2
import numpy as np

from app.camera_capture import (
    DEFAULT_RETRY_INTERVAL_SEC,
    BackgroundCapture,
    apply_capture_format,
    choose_hardware_fps,
    open_camera_device,
    release_capture,
)
from app.config import (
    AlertLevel,
    CameraSource,
    EventRecord,
    FrameTelemetry,
    SystemConfig,
)
from app.event_recorder import EventRecorder
from app.hud_renderer import HudRenderer
from app.industrial_sim import AnomalyType, IndustrialSceneSimulator
from app.motion_detector import MotionDetector, MotionResult
from app.thermal_simulator import ThermalResult, ThermalSimulator

logger = logging.getLogger(__name__)

MAX_EVENT_LOGS = 200
FPS_WINDOW_SIZE = 30
SYNC_RETRY_INTERVAL_SEC = 0.5
# Tolerância para a câmera entregar o primeiro quadro após (re)abrir. Nesse
# intervalo a interface mostra um quadro de espera em vez de travar; passado o
# prazo, assume-se câmera indisponível e entra o simulador de fallback.
CAMERA_OPEN_WAIT_SEC = 2.0
CONNECTING_FRAME_COLOR = (26, 20, 16)
CONNECTING_TEXT_COLOR = (235, 225, 210)
THERMAL_IMAGE_MODES = ("thermal", "hybrid")


class CameraManager:
    """Controlador central de captura de vídeo, visão térmica, detecção de movimento e eventos."""

    def __init__(
        self,
        initial_config: SystemConfig | None = None,
        capture_factory: Callable[[int], Any] | None = None,
        camera_retry_interval_sec: float | None = None,
        background_capture: bool | None = None,
        event_sink: Callable[[EventRecord], None] | None = None,
        recordings_dir: Path | None = None,
    ) -> None:
        """Cria o controlador.

        `event_sink` recebe cada evento registrado (ex.: gravação em disco).

        `recordings_dir` é a pasta dos vídeos dos eventos de movimento, gravados
        quando `record_events` está ligado; sem ela a gravação fica indisponível.

        `capture_factory` substitui a abertura da câmera (testes). Com a câmera
        real a captura roda em thread dedicada; `background_capture` força ou
        desativa esse modo independentemente da fábrica usada.
        """
        self._lock = RLock()
        self._closed = False
        self._event_sink = event_sink
        self._config: SystemConfig = initial_config or SystemConfig()
        self._capture_factory: Callable[[int], Any] | None = capture_factory
        self._use_background_capture: bool = (
            background_capture if background_capture is not None else capture_factory is None
        )
        default_retry = (
            DEFAULT_RETRY_INTERVAL_SEC
            if self._use_background_capture
            else (SYNC_RETRY_INTERVAL_SEC if capture_factory is None else 0.0)
        )
        self._camera_retry_interval_sec: float = (
            camera_retry_interval_sec if camera_retry_interval_sec is not None else default_retry
        )
        self._last_camera_fail_ts: float = 0.0
        self._cap: Any = None
        self._opened_camera_index: int | None = None
        self._applied_hw_props: tuple[int, int, int] | None = None
        self._bg_capture: BackgroundCapture | None = None
        self._bg_format: tuple[int, int, int, int] | None = None
        self._bg_started_ts: float = 0.0
        self._lingering_capture_thread: Any = None
        self._camera_connecting: bool = False
        self._connecting_frame: np.ndarray | None = None
        self._cam_cache_key: tuple[int, bool] | None = None
        self._cam_cache_seq: int | None = None
        self._cam_cache_frame: np.ndarray | None = None
        self._last_cam_frame_is_new: bool = True

        self._recordings_dir = recordings_dir
        self._recorder: EventRecorder | None = None
        self._recording: bool = False
        self._recording_base: Path | None = None
        self._recording_started_mono: float = 0.0
        self._last_motion_mono: float = 0.0

        self._simulator = IndustrialSceneSimulator()
        self._motion_detector = MotionDetector()
        self._thermal_simulator = ThermalSimulator()
        self._hud_renderer = HudRenderer()

        self._frame_index: int = 0
        self._frame_timestamps: deque[float] = deque(maxlen=FPS_WINDOW_SIZE)
        self._latest_frame_bgr: np.ndarray | None = None
        self._latest_telemetry: FrameTelemetry | None = None
        self._latest_stream_jpeg: bytes | None = None
        self._last_step_monotonic: float = 0.0

        self._events: deque[EventRecord] = deque(maxlen=MAX_EVENT_LOGS)
        self._next_event_id: int = 1
        self._prev_motion_alarm: bool = False
        self._prev_overheat_alarm: bool = False
        self._prev_fallback_warned: bool = False

        self._record_event(
            category="SYSTEM",
            severity="NORMAL",
            message=(
                f"Sistema iniciado em {self._config.width}x{self._config.height} "
                f"@ {self._config.target_fps} FPS (fonte: {self._config.source})"
            ),
        )

    def _invalidate_cache(self) -> None:
        self._latest_frame_bgr = None
        self._latest_telemetry = None
        self._latest_stream_jpeg = None
        self._last_step_monotonic = 0.0

    def get_config(self) -> SystemConfig:
        """Retorna a configuração imutável atual."""
        with self._lock:
            return self._config

    def update_config(self, patch: dict[str, Any]) -> SystemConfig:
        """Valida e aplica um patch de configuração, retornando a nova instância imutável."""
        with self._lock:
            old_cfg = self._config
            new_cfg = old_cfg.apply_patch(patch)
            self._config = new_cfg
            self._invalidate_cache()

            if (old_cfg.width, old_cfg.height) != (new_cfg.width, new_cfg.height):
                self._motion_detector.reset()
                self._thermal_simulator.reset()
                self._record_event(
                    category="SYSTEM",
                    severity="NORMAL",
                    message=f"Resolução alterada para {new_cfg.width}x{new_cfg.height}",
                    details={"width": new_cfg.width, "height": new_cfg.height},
                )

            if old_cfg.target_fps != new_cfg.target_fps:
                self._frame_timestamps.clear()
                self._record_event(
                    category="SYSTEM",
                    severity="NORMAL",
                    message=f"Taxa de quadros alterada para {new_cfg.target_fps} FPS",
                    details={"target_fps": new_cfg.target_fps},
                )

            if old_cfg.vision_mode != new_cfg.vision_mode:
                self._record_event(
                    category="SYSTEM",
                    severity="NORMAL",
                    message=f"Modo de visão alterado para '{new_cfg.vision_mode}'",
                )

            if old_cfg.mirror != new_cfg.mirror:
                self._motion_detector.reset()
                self._thermal_simulator.reset()

            if old_cfg.source != new_cfg.source or old_cfg.camera_index != new_cfg.camera_index:
                self._motion_detector.reset()
                self._thermal_simulator.reset()
                self._prev_fallback_warned = False
                self._last_camera_fail_ts = 0.0
                if new_cfg.source == "simulator" or old_cfg.camera_index != new_cfg.camera_index:
                    self._release_camera()
                self._record_event(
                    category="SYSTEM",
                    severity="NORMAL",
                    message=(
                        f"Fonte de captura alterada para '{new_cfg.source}' "
                        f"(índice #{new_cfg.camera_index})"
                    ),
                )

            return new_cfg

    def trigger_anomaly(self, kind: AnomalyType, duration_frames: int = 45) -> None:
        """Injeta ou limpa uma anomalia industrial (pico térmico ou intrusão)."""
        with self._lock:
            self._simulator.trigger_anomaly(kind, duration_frames=duration_frames)
            self._invalidate_cache()
            severity: AlertLevel = "NORMAL" if kind == "clear" else "ATENCAO"
            self._record_event(
                category="ANOMALY",
                severity=severity,
                message=f"Comando de anomalia: {kind} ({duration_frames} quadros)",
                details={"kind": kind, "duration_frames": duration_frames},
            )

    def reset_detectors(self) -> None:
        """Reinicia o modelo de fundo de movimento e o acumulador térmico."""
        with self._lock:
            self._motion_detector.reset()
            self._thermal_simulator.reset()
            self._simulator.trigger_anomaly("clear")
            self._invalidate_cache()
            self._record_event(
                category="SYSTEM",
                severity="NORMAL",
                message="Sensores de movimento e memória térmica recalibrados",
            )

    def get_frame_interval_sec(self) -> float:
        """Retorna o intervalo alvo entre quadros em segundos (1 / target_fps)."""
        with self._lock:
            return 1.0 / float(max(1, self._config.target_fps))

    def step_frame(self) -> tuple[np.ndarray, FrameTelemetry]:
        """Captura um quadro, executa detecção de movimento + termografia + HUD."""
        with self._lock:
            t_start = time.perf_counter()
            cfg = self._config
            raw_bgr, heat_hint, source_active, hw_fallback, is_new_frame = (
                self._acquire_raw_frame(cfg)
            )

            motion_res = self._motion_detector.process(
                raw_bgr, cfg.motion, cfg.roi, is_new_frame=is_new_frame
            )
            thermal_res = self._thermal_simulator.process(
                raw_bgr,
                cfg.thermal,
                motion_mask=motion_res.motion_mask,
                heat_hint=heat_hint,
                roi=cfg.roi,
                is_new_frame=is_new_frame,
                render_image=cfg.vision_mode in THERMAL_IMAGE_MODES,
            )

            actual_fps = self._update_fps_counter(time.perf_counter())
            alert_level = self._determine_alert_level(motion_res, thermal_res)
            rendered = self._hud_renderer.render(
                raw_bgr=raw_bgr,
                motion=motion_res,
                thermal=thermal_res,
                config=cfg,
                actual_fps=actual_fps,
                alert_level=alert_level,
            )

            proc_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            self._frame_index += 1
            telemetry = self._build_telemetry(
                cfg=cfg,
                source_active=source_active,
                hw_fallback=hw_fallback,
                actual_fps=actual_fps,
                proc_ms=proc_ms,
                motion=motion_res,
                thermal=thermal_res,
                alert_level=alert_level,
            )

            self._evaluate_alarm_transitions(telemetry)
            self._update_recording(cfg, telemetry)
            self._latest_frame_bgr = rendered
            self._latest_telemetry = telemetry
            self._latest_stream_jpeg = None
            self._last_step_monotonic = time.perf_counter()
            return rendered, telemetry

    def get_latest_telemetry(self) -> FrameTelemetry:
        """Retorna a última telemetria ou processa um quadro se ainda não houver ou estiver defasado."""
        with self._lock:
            now = time.perf_counter()
            stale_threshold = max(0.4, self.get_frame_interval_sec())
            if (
                self._latest_telemetry is None
                or (now - self._last_step_monotonic) >= stale_threshold
            ):
                _, tel = self.step_frame()
                return tel
            return self._latest_telemetry

    def get_snapshot_jpeg(self, quality: int = 88) -> bytes:
        """Codifica o último quadro processado em JPEG."""
        with self._lock:
            if self._latest_frame_bgr is None:
                self.step_frame()
            assert self._latest_frame_bgr is not None
            return self.encode_jpeg(self._latest_frame_bgr, quality=quality)

    def get_stream_jpeg(self, quality: int = 80) -> bytes:
        """Retorna um quadro JPEG para o stream MJPEG, deduplicando entre múltiplos clientes no mesmo intervalo de FPS."""
        with self._lock:
            now = time.perf_counter()
            interval = self.get_frame_interval_sec()
            if (
                self._latest_frame_bgr is None
                or (now - self._last_step_monotonic) >= interval * 0.75
            ):
                self.step_frame()
            if self._latest_stream_jpeg is None:
                assert self._latest_frame_bgr is not None
                self._latest_stream_jpeg = self.encode_jpeg(
                    self._latest_frame_bgr, quality=quality
                )
            return self._latest_stream_jpeg

    @staticmethod
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = 82) -> bytes:
        """Codifica um array BGR em bytes JPEG."""
        clamped_q = max(30, min(100, int(quality)))
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), clamped_q]
        )
        if not ok:
            raise RuntimeError("Falha ao codificar quadro em JPEG")
        return buf.tobytes()

    def get_events(self, limit: int = 50) -> list[EventRecord]:
        """Retorna os eventos mais recentes (do mais novo para o mais antigo)."""
        with self._lock:
            items = list(self._events)
            items.reverse()
            return items[: max(1, int(limit))]

    def clear_events(self) -> None:
        """Limpa o histórico de eventos."""
        with self._lock:
            self._events.clear()

    @staticmethod
    def get_camera_device_label(camera_index: int = 0) -> str:
        """Retorna o nome amigável do dispositivo de câmera pessoal (ex: /dev/video0)."""
        if not sys.platform.startswith("linux"):
            return f"Câmera #{camera_index}"
        sys_name_path = Path(f"/sys/class/video4linux/video{camera_index}/name")
        if sys_name_path.exists():
            with contextlib.suppress(OSError):
                raw_name = sys_name_path.read_text(encoding="utf-8", errors="ignore").strip()
                if raw_name:
                    clean_name = raw_name.split(":")[0].strip() or raw_name
                    return f"/dev/video{camera_index} ({clean_name})"
        dev_path = Path(f"/dev/video{camera_index}")
        if dev_path.exists():
            return f"/dev/video{camera_index} (Câmera Pessoal #{camera_index})"
        return f"Câmera #{camera_index}"

    @classmethod
    def list_available_cameras(cls) -> list[dict[str, Any]]:
        """Lista as câmeras pessoais V4L2/USB detectadas no sistema."""
        devices: list[dict[str, Any]] = []
        v4l_root = Path("/sys/class/video4linux")
        if v4l_root.exists():
            for node in sorted(v4l_root.glob("video*")):
                suffix = node.name.removeprefix("video")
                if not suffix.isdigit():
                    continue
                idx = int(suffix)
                sub_idx_file = node / "index"
                if sub_idx_file.exists():
                    with contextlib.suppress(OSError):
                        if sub_idx_file.read_text(encoding="utf-8", errors="ignore").strip() not in ("", "0"):
                            continue
                devices.append(
                    {
                        "index": idx,
                        "device": f"/dev/video{idx}",
                        "label": cls.get_camera_device_label(idx),
                    }
                )
        if not devices:
            # Sem enumeração V4L2 (outros sistemas ou nenhuma câmera): oferece o índice 0.
            is_linux = sys.platform.startswith("linux")
            devices.append(
                {
                    "index": 0,
                    "device": "/dev/video0" if is_linux else "camera:0",
                    "label": cls.get_camera_device_label(0),
                }
            )
        return devices

    def reconnect_camera(self) -> bool:
        """Força a reabertura imediata da câmera pessoal física."""
        with self._lock:
            self._release_camera()
            self._motion_detector.reset()
            self._thermal_simulator.reset()
            self._last_camera_fail_ts = 0.0
            self._prev_fallback_warned = False
            self._invalidate_cache()
            if self._config.source != "camera":
                self._config = self._config.apply_patch({"source": "camera"})
            frame = self._read_physical_camera(self._config, wait_for_open=True)
            ok = frame is not None
            # O quadro lido aqui não foi analisado: o próximo passo deve tratá-lo como novo.
            self._cam_cache_key = None
            self._cam_cache_seq = None
            self._record_event(
                category="SYSTEM",
                severity="NORMAL" if ok else "ATENCAO",
                message=(
                    f"Câmera pessoal {self.get_camera_device_label(self._config.camera_index)} ativa"
                    if ok
                    else f"Não foi possível abrir a câmera #{self._config.camera_index}"
                ),
            )
            return ok

    def close(self) -> None:
        """Libera a câmera física em definitivo; passos posteriores não a reabrem."""
        with self._lock:
            self._closed = True
            if self._recording:
                self._finish_recording(time.monotonic())
            recorder, self._recorder = self._recorder, None
            self._release_camera()
        if recorder is not None:
            recorder.close()  # termina de gravar os quadros que estavam na fila

    def _acquire_raw_frame(
        self, cfg: SystemConfig
    ) -> tuple[np.ndarray, np.ndarray | None, CameraSource, bool, bool]:
        if cfg.source == "camera":
            cam_frame = self._read_physical_camera(cfg)
            if cam_frame is not None:
                if self._camera_connecting:
                    # Primeiro quadro real após a tela de espera: recalibra o fundo.
                    self._camera_connecting = False
                    self._motion_detector.reset()
                    self._thermal_simulator.reset()
                if self._prev_fallback_warned:
                    self._prev_fallback_warned = False
                    self._motion_detector.reset()
                    self._thermal_simulator.reset()
                    self._record_event(
                        category="SYSTEM",
                        severity="NORMAL",
                        message=f"Câmera física #{cfg.camera_index} reconectada e operando",
                    )

                if self._simulator.get_active_anomalies():
                    return self._overlay_anomalies(cam_frame, cfg)
                return cam_frame, None, "camera", False, self._last_cam_frame_is_new

            if self._is_camera_connecting():
                self._camera_connecting = True
                waiting_frame = self._get_connecting_frame(cfg)
                if self._simulator.get_active_anomalies():
                    return self._overlay_anomalies(waiting_frame, cfg)
                return waiting_frame, None, "camera", False, True

            if not self._prev_fallback_warned:
                self._prev_fallback_warned = True
                self._camera_connecting = False
                self._motion_detector.reset()
                self._thermal_simulator.reset()
                self._record_event(
                    category="SYSTEM",
                    severity="ATENCAO",
                    message=(
                        f"Câmera física #{cfg.camera_index} indisponível; "
                        "usando Simulador Industrial como fallback automático"
                    ),
                )
            sim_frame, sim_heat = self._simulator.generate_frame(
                cfg.width, cfg.height, cfg.simulator
            )
            return sim_frame, sim_heat, "simulator", True, True

        sim_frame, sim_heat = self._simulator.generate_frame(
            cfg.width, cfg.height, cfg.simulator
        )
        return sim_frame, sim_heat, "simulator", False, True

    def _open_camera(self, camera_index: int, width: int, height: int, fps: int) -> Any:
        """Abre a câmera real ou, se houver, a fábrica injetada, já no formato pedido."""
        if self._capture_factory is None:
            return open_camera_device(camera_index, width, height, fps)
        cap = self._capture_factory(camera_index)
        if cap is not None and cap.isOpened():
            apply_capture_format(cap, width, height, fps)
        return cap

    def _overlay_anomalies(
        self, frame: np.ndarray, cfg: SystemConfig
    ) -> tuple[np.ndarray, np.ndarray | None, CameraSource, bool, bool]:
        """Desenha as anomalias ativas sobre uma cópia do quadro da câmera."""
        # Copia para não sujar o quadro em cache, que é reutilizado entre passos.
        painted = frame.copy()
        heat_overlay = self._thermal_simulator.compute_luminance_heat(painted, cfg.thermal)
        self._simulator.step_anomalies(painted, heat_overlay, self._frame_index * 0.08)
        return painted, heat_overlay, "camera", False, True

    def _is_camera_connecting(self) -> bool:
        """True enquanto a thread de captura ainda está no prazo para entregar o 1º quadro."""
        return (
            self._bg_capture is not None
            and not self._prev_fallback_warned
            and not self._bg_capture.has_delivered_frame
            and (time.perf_counter() - self._bg_started_ts) < CAMERA_OPEN_WAIT_SEC
        )

    def _get_connecting_frame(self, cfg: SystemConfig) -> np.ndarray:
        """Quadro de espera exibido enquanto a câmera abre (em cache por resolução)."""
        cached = self._connecting_frame
        if cached is None or cached.shape[:2] != (cfg.height, cfg.width):
            cached = np.full((cfg.height, cfg.width, 3), CONNECTING_FRAME_COLOR, dtype=np.uint8)
            scale = max(0.4, cfg.width / 900.0)
            cv2.putText(
                cached,
                f"Conectando a camera #{cfg.camera_index}...",
                (max(8, cfg.width // 14), cfg.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                CONNECTING_TEXT_COLOR,
                max(1, round(scale * 2)),
                cv2.LINE_AA,
            )
            self._connecting_frame = cached
        return cached

    def _read_physical_camera(
        self, cfg: SystemConfig, *, wait_for_open: bool = False
    ) -> np.ndarray | None:
        """Devolve o quadro atual da câmera já na resolução/espelhamento da config.

        `wait_for_open` bloqueia até o primeiro quadro (reconexão pedida pelo
        usuário, que espera um resultado); o laço de vídeo nunca bloqueia.
        """
        if self._closed:
            # Um passo atrasado de outra thread não pode religar a câmera já liberada.
            return None
        if self._use_background_capture:
            return self._read_background_camera(cfg, wait_for_open=wait_for_open)
        frame = self._read_sync_camera(cfg)
        if frame is None:
            return None
        self._last_cam_frame_is_new = True
        return self._fit_camera_frame(frame, cfg)

    def _read_background_camera(
        self, cfg: SystemConfig, *, wait_for_open: bool = False
    ) -> np.ndarray | None:
        """Consulta, sem bloquear, o quadro mais recente da thread de captura."""
        hardware_fps = choose_hardware_fps(cfg.target_fps)
        hw_format = (cfg.camera_index, cfg.width, cfg.height, hardware_fps)
        if self._bg_capture is None or self._bg_format != hw_format:
            self._release_camera()
            self._bg_capture = BackgroundCapture(
                self._open_camera,
                cfg.camera_index,
                cfg.width,
                cfg.height,
                fps=hardware_fps,
                retry_interval_sec=self._camera_retry_interval_sec,
                predecessor=self._lingering_capture_thread,
                realtime_tuning=self._capture_factory is None,
            )
            self._bg_capture.frame_listener = (
                self._recorder.receive if self._recorder is not None else None
            )
            self._bg_format = hw_format
            self._bg_capture.start()
            self._bg_started_ts = time.perf_counter()
            self._frame_timestamps.clear()
        if wait_for_open and not self._bg_capture.has_delivered_frame:
            self._bg_capture.wait_for_frame(CAMERA_OPEN_WAIT_SEC)

        raw_frame, sequence = self._bg_capture.read_latest()
        if raw_frame is None:
            self._cam_cache_key = None
            self._cam_cache_frame = None
            return None

        cache_key = (sequence, cfg.mirror)
        if self._cam_cache_key == cache_key and self._cam_cache_frame is not None:
            self._last_cam_frame_is_new = False
            return self._cam_cache_frame

        self._last_cam_frame_is_new = sequence != self._cam_cache_seq
        self._cam_cache_key = cache_key
        self._cam_cache_seq = sequence
        self._cam_cache_frame = self._fit_camera_frame(raw_frame, cfg)
        return self._cam_cache_frame

    def _read_sync_camera(self, cfg: SystemConfig) -> np.ndarray | None:
        """Leitura direta na thread chamadora (fábricas injetadas e modo sem thread)."""
        try:
            if self._cap is None or self._opened_camera_index != cfg.camera_index:
                now = time.perf_counter()
                if (
                    self._last_camera_fail_ts > 0.0
                    and (now - self._last_camera_fail_ts) < self._camera_retry_interval_sec
                ):
                    return None
                self._release_camera()
                self._cap = self._open_camera(
                    cfg.camera_index, cfg.width, cfg.height, cfg.target_fps
                )
                self._opened_camera_index = cfg.camera_index
                self._applied_hw_props = (cfg.width, cfg.height, cfg.target_fps)
                self._frame_timestamps.clear()

            if self._cap is None or not self._cap.isOpened():
                return self._mark_camera_failure()

            desired_props = (cfg.width, cfg.height, cfg.target_fps)
            if self._applied_hw_props != desired_props:
                apply_capture_format(self._cap, *desired_props)
                self._applied_hw_props = desired_props

            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                return self._mark_camera_failure()
        except (cv2.error, OSError, RuntimeError, TypeError):
            return self._mark_camera_failure()

        self._last_camera_fail_ts = 0.0
        if self._recorder is not None:
            self._recorder.receive(time.monotonic(), frame, None)
        return frame

    def _mark_camera_failure(self) -> None:
        self._release_camera()
        self._last_camera_fail_ts = time.perf_counter()

    @staticmethod
    def _fit_camera_frame(frame: np.ndarray, cfg: SystemConfig) -> np.ndarray:
        """Ajusta o quadro bruto à resolução configurada e ao espelhamento."""
        if frame.shape[:2] != (cfg.height, cfg.width):
            frame = cv2.resize(
                frame, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR
            )
        if cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _release_camera(self) -> None:
        if self._bg_capture is not None:
            self._lingering_capture_thread = self._bg_capture.stop()
            self._bg_capture = None
            self._bg_format = None
        if self._cap is not None:
            release_capture(self._cap)
            self._cap = None
            self._opened_camera_index = None
            self._applied_hw_props = None
        self._cam_cache_key = None
        self._cam_cache_seq = None
        self._cam_cache_frame = None
        self._last_cam_frame_is_new = True

    def _update_fps_counter(self, now: float) -> float:
        if self._frame_timestamps and (now - self._frame_timestamps[-1]) > 2.0:
            self._frame_timestamps.clear()
        self._frame_timestamps.append(now)
        if len(self._frame_timestamps) < 2:
            return float(self._config.target_fps)
        elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if elapsed <= 1e-6:
            return float(self._config.target_fps)
        measured = (len(self._frame_timestamps) - 1) / elapsed
        return round(min(float(self._config.target_fps), measured), 1)

    @staticmethod
    def _determine_alert_level(
        motion: MotionResult, thermal: ThermalResult
    ) -> AlertLevel:
        if thermal.overheat_alarm:
            return "CRITICO"
        if motion.alarm_triggered:
            return "ATENCAO"
        return "NORMAL"

    def _build_telemetry(
        self,
        cfg: SystemConfig,
        source_active: CameraSource,
        hw_fallback: bool,
        actual_fps: float,
        proc_ms: float,
        motion: MotionResult,
        thermal: ThermalResult,
        alert_level: AlertLevel,
    ) -> FrameTelemetry:
        return FrameTelemetry(
            frame_index=self._frame_index,
            timestamp_iso=datetime.now(UTC).isoformat(),
            source_active=source_active,
            hardware_fallback=hw_fallback,
            width=cfg.width,
            height=cfg.height,
            target_fps=cfg.target_fps,
            actual_fps=actual_fps,
            processing_ms=proc_ms,
            vision_mode=cfg.vision_mode,
            motion_detected=motion.motion_detected,
            motion_alarm=motion.alarm_triggered,
            motion_percentage=motion.motion_percentage,
            object_count=motion.object_count,
            bounding_boxes=motion.bounding_boxes,
            min_temp_c=thermal.min_temp_c,
            avg_temp_c=thermal.avg_temp_c,
            max_temp_c=thermal.max_temp_c,
            hotspot_x=thermal.hotspot_x,
            hotspot_y=thermal.hotspot_y,
            coldspot_x=thermal.coldspot_x,
            coldspot_y=thermal.coldspot_y,
            overheat_alarm=thermal.overheat_alarm,
            alert_level=alert_level,
            camera_fps=round(self._bg_capture.camera_fps, 1) if self._bg_capture is not None else 0.0,
            recording=self._recording,
            recording_dropped=self._recorder.dropped_frames if self._recorder is not None else 0,
        )

    def _evaluate_alarm_transitions(self, tel: FrameTelemetry) -> None:
        if tel.overheat_alarm and not self._prev_overheat_alarm:
            self._record_event(
                category="THERMAL",
                severity="CRITICO",
                message=(
                    f"Alarme de superaquecimento: {tel.max_temp_c:.1f}°C "
                    f"(limite: {self._config.thermal.overheat_threshold_c:.1f}°C)"
                ),
                details={
                    "max_temp_c": tel.max_temp_c,
                    "hotspot": [tel.hotspot_x, tel.hotspot_y],
                },
            )
        self._prev_overheat_alarm = tel.overheat_alarm

        if tel.motion_alarm and not self._prev_motion_alarm:
            self._record_event(
                category="MOTION",
                severity="ATENCAO",
                message=(
                    f"Movimento acima do limiar: {tel.motion_percentage:.1f}% "
                    f"({tel.object_count} objeto(s))"
                ),
                details={
                    "motion_percentage": tel.motion_percentage,
                    "object_count": tel.object_count,
                },
            )
        self._prev_motion_alarm = tel.motion_alarm

    def _update_recording(self, cfg: SystemConfig, tel: FrameTelemetry) -> None:
        """Liga/desliga o gravador conforme a configuração e grava cada evento de movimento."""
        wanted = cfg.record_events and cfg.source == "camera" and self._recordings_dir is not None
        if wanted and self._recorder is None:
            self._recorder = EventRecorder(self._camera_fps_estimate, pre_seconds=cfg.record_pre_sec)
            if self._bg_capture is not None:
                self._bg_capture.frame_listener = self._recorder.receive
        elif not wanted and self._recorder is not None:
            if self._recording:
                self._finish_recording(time.monotonic())
            recorder, self._recorder = self._recorder, None
            if self._bg_capture is not None:
                self._bg_capture.frame_listener = None
            # Fechar espera a fila esvaziar: fora do laço de vídeo, para não engasgar a tela.
            threading.Thread(target=recorder.close, name="fecha-gravacao").start()
        if self._recorder is None:
            return
        self._recorder.pre_seconds = cfg.record_pre_sec
        now = time.monotonic()
        if tel.motion_alarm and tel.source_active == "camera":
            self._last_motion_mono = now
            if not self._recording:
                self._begin_recording(now)
        elif self._recording and now - self._last_motion_mono > cfg.record_post_sec:
            self._finish_recording(now)

    def _camera_fps_estimate(self) -> float:
        """FPS real da câmera para o arquivo de vídeo (cai para o alvo antes de medir)."""
        measured = self._bg_capture.camera_fps if self._bg_capture is not None else 0.0
        if measured <= 0.0 and self._latest_telemetry is not None:
            measured = self._latest_telemetry.actual_fps
        return measured if measured > 0.0 else float(self._config.target_fps)

    def _begin_recording(self, now: float) -> None:
        assert self._recorder is not None and self._recordings_dir is not None
        moment = datetime.now()
        stamp = moment.strftime("%H-%M-%S-") + f"{moment.microsecond // 1000:03d}"
        base = self._recordings_dir / moment.strftime("%Y-%m-%d") / f"{stamp}_movimento"
        self._recorder.start(now, base)
        self._recording = True
        self._recording_base = base
        self._recording_started_mono = now
        raw = self._bg_capture is not None and self._bg_capture.raw_jpeg
        self._record_event(
            category="MOTION",
            severity="ATENCAO",
            message=f"Gravando vídeo do movimento: {base.name}{'.avi' if raw else '.mp4'}",
            details={"arquivo": str(base), "jpeg_direto": raw},
        )

    def _finish_recording(self, now: float) -> None:
        if self._recorder is not None:
            self._recorder.stop(now)
        self._recording = False
        duration = now - self._recording_started_mono
        self._record_event(
            category="MOTION",
            severity="NORMAL",
            message=f"Vídeo do movimento salvo ({duration:.1f} s): {self._recording_base}",
            details={
                "arquivo": str(self._recording_base),
                "duracao_s": round(duration, 2),
                "quadros_perdidos": self._recorder.dropped_frames if self._recorder else 0,
            },
        )

    def _record_event(
        self,
        category: Literal["MOTION", "THERMAL", "SYSTEM", "ANOMALY"],
        severity: AlertLevel,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        ev = EventRecord(
            id=self._next_event_id,
            timestamp_iso=datetime.now(UTC).isoformat(),
            category=category,
            severity=severity,
            message=message,
            details=details or {},
        )
        self._next_event_id += 1
        self._events.append(ev)
        if self._event_sink is not None:
            try:
                self._event_sink(ev)
            except Exception:
                # Falha ao persistir um evento não pode interromper o vídeo.
                logger.exception("Falha no destino de eventos; persistência desativada")
                self._event_sink = None
