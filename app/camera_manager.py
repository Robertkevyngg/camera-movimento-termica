"""Orquestrador de captura de câmera, simulação, resolução/FPS e telemetria."""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import cv2
import numpy as np

from app.config import (
    AlertLevel,
    CameraSource,
    EventRecord,
    FrameTelemetry,
    SystemConfig,
)
from app.hud_renderer import HudRenderer
from app.industrial_sim import AnomalyType, IndustrialSceneSimulator
from app.motion_detector import MotionDetector, MotionResult
from app.thermal_simulator import ThermalResult, ThermalSimulator

MAX_EVENT_LOGS = 200
FPS_WINDOW_SIZE = 30


class CameraManager:
    """Controlador central de captura de vídeo, visão térmica, detecção de movimento e eventos."""

    def __init__(
        self,
        initial_config: SystemConfig | None = None,
        capture_factory: Callable[[int], Any] | None = None,
        camera_retry_interval_sec: float | None = None,
    ) -> None:
        self._lock = RLock()
        self._config: SystemConfig = initial_config or SystemConfig()
        self._capture_factory: Callable[[int], Any] = capture_factory or cv2.VideoCapture
        self._camera_retry_interval_sec: float = (
            camera_retry_interval_sec
            if camera_retry_interval_sec is not None
            else (0.5 if capture_factory is None else 0.0)
        )
        self._last_camera_fail_ts: float = 0.0
        self._cap: Any = None
        self._opened_camera_index: int | None = None
        self._applied_hw_props: tuple[int, int, int] | None = None
        self._last_raw_cam_frame: np.ndarray | None = None
        self._last_raw_cam_ts: float = 0.0
        self._last_cam_frame_is_new: bool = True
        self._unconsumed_cam_frame: bool = False

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
            devices.append(
                {
                    "index": 0,
                    "device": "/dev/video0",
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
            frame = self._read_physical_camera(self._config)
            ok = frame is not None
            if ok:
                self._unconsumed_cam_frame = True
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
        """Libera recursos de hardware da câmera física se estiverem abertos."""
        with self._lock:
            self._release_camera()

    def _acquire_raw_frame(
        self, cfg: SystemConfig
    ) -> tuple[np.ndarray, np.ndarray | None, CameraSource, bool, bool]:
        if cfg.source == "camera":
            cam_frame = self._read_physical_camera(cfg)
            if cam_frame is not None:
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
                    heat_overlay = self._thermal_simulator.compute_luminance_heat(
                        cam_frame, cfg.thermal
                    )
                    self._simulator.step_anomalies(
                        cam_frame, heat_overlay, self._frame_index * 0.08
                    )
                    return cam_frame, heat_overlay, "camera", False, True
                return cam_frame, None, "camera", False, self._last_cam_frame_is_new

            if not self._prev_fallback_warned:
                self._prev_fallback_warned = True
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

    def _open_hardware_capture(self, camera_index: int) -> Any:
        if self._capture_factory is cv2.VideoCapture and hasattr(cv2, "CAP_V4L2"):
            with contextlib.suppress(cv2.error, OSError, RuntimeError, TypeError):
                cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
                if cap is not None and cap.isOpened():
                    with contextlib.suppress(cv2.error, OSError, RuntimeError):
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2.0)
                        cap.set(
                            cv2.CAP_PROP_FOURCC,
                            float(cv2.VideoWriter_fourcc(*"MJPG")),
                        )
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
                        cap.set(cv2.CAP_PROP_FPS, 30.0)
                    return cap
                if cap is not None:
                    with contextlib.suppress(cv2.error, OSError, RuntimeError):
                        cap.release()
                    return None
        with contextlib.suppress(cv2.error, OSError, RuntimeError, TypeError):
            return self._capture_factory(camera_index)
        return None

    def _read_physical_camera(self, cfg: SystemConfig) -> np.ndarray | None:
        is_real_v4l2 = self._capture_factory is cv2.VideoCapture
        try:
            if self._cap is None or self._opened_camera_index != cfg.camera_index:
                now = time.perf_counter()
                if (
                    self._last_camera_fail_ts > 0.0
                    and (now - self._last_camera_fail_ts) < self._camera_retry_interval_sec
                ):
                    return None
                self._release_camera()
                self._cap = self._open_hardware_capture(cfg.camera_index)
                self._opened_camera_index = cfg.camera_index
                self._applied_hw_props = None
                self._frame_timestamps.clear()

            if self._cap is None or not self._cap.isOpened():
                self._release_camera()
                self._last_camera_fail_ts = time.perf_counter()
                return None

            if not is_real_v4l2:
                desired_props = (cfg.width, cfg.height, cfg.target_fps)
                if self._applied_hw_props != desired_props:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(desired_props[0]))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(desired_props[1]))
                    self._cap.set(cv2.CAP_PROP_FPS, float(desired_props[2]))
                    self._applied_hw_props = desired_props

            now_read = time.perf_counter()
            if (
                is_real_v4l2
                and self._last_raw_cam_frame is not None
                and (now_read - self._last_raw_cam_ts) < 0.028
            ):
                frame = self._last_raw_cam_frame.copy()
                self._last_cam_frame_is_new = self._unconsumed_cam_frame
                self._unconsumed_cam_frame = False
            else:
                ret, frame = self._cap.read()
                if not ret or frame is None or frame.size == 0:
                    self._release_camera()
                    self._last_camera_fail_ts = time.perf_counter()
                    return None
                self._last_cam_frame_is_new = True
                self._unconsumed_cam_frame = False
                if is_real_v4l2:
                    self._last_raw_cam_frame = frame.copy()
                    self._last_raw_cam_ts = time.perf_counter()
        except (cv2.error, OSError, RuntimeError):
            self._release_camera()
            self._last_camera_fail_ts = time.perf_counter()
            return None

        self._last_camera_fail_ts = 0.0
        if frame.shape[:2] != (cfg.height, cfg.width):
            frame = cv2.resize(
                frame, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR
            )
        if cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _release_camera(self) -> None:
        if self._cap is not None:
            with contextlib.suppress(cv2.error, OSError, RuntimeError):
                self._cap.release()
            self._cap = None
            self._opened_camera_index = None
            self._applied_hw_props = None
            self._last_raw_cam_frame = None
            self._last_raw_cam_ts = 0.0
            self._last_cam_frame_is_new = True
            self._unconsumed_cam_frame = False

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
