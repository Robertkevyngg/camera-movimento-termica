"""Testes unitários e de integração para o CameraManager."""

import numpy as np
import pytest

from app.camera_manager import CameraManager
from app.config import SystemConfig


class FakeVideoCapture:
    """Mock determinístico de cv2.VideoCapture para testes."""

    def __init__(self, index: int = 0, should_open: bool = True, native_size=(480, 640)) -> None:
        self.index = index
        self.should_open = should_open
        self.native_h, self.native_w = native_size
        self.props: dict[int, float] = {}
        self.released = False
        self.read_count = 0

    def isOpened(self) -> bool:
        return self.should_open and not self.released

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = float(value)
        return True

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def read(self):
        if not self.isOpened():
            return False, None
        self.read_count += 1
        frame = np.full((self.native_h, self.native_w, 3), 60, dtype=np.uint8)
        # Objeto animado para gerar movimento
        x0 = (self.read_count * 25) % max(40, self.native_w - 80)
        frame[80:160, x0 : x0 + 60] = (230, 240, 250)
        return True, frame

    def release(self) -> None:
        self.released = True


def test_camera_manager_simulator_mode_and_dynamic_resolution_fps():
    mgr = CameraManager(SystemConfig(source="simulator", width=640, height=480, target_fps=24))
    frame1, tel1 = mgr.step_frame()

    assert frame1.shape == (480, 640, 3)
    assert tel1.width == 640
    assert tel1.height == 480
    assert tel1.target_fps == 24
    assert tel1.source_active == "simulator"
    assert tel1.hardware_fallback is False

    # Mudar resolução e FPS em tempo de execução
    updated_cfg = mgr.update_config({"width": 1280, "height": 720, "target_fps": 60})
    assert updated_cfg.width == 1280
    assert updated_cfg.height == 720
    assert updated_cfg.target_fps == 60
    assert mgr.get_frame_interval_sec() == pytest.approx(1.0 / 60.0)

    frame2, tel2 = mgr.step_frame()
    assert frame2.shape == (720, 1280, 3)
    assert tel2.width == 1280
    assert tel2.height == 720
    assert tel2.target_fps == 60


def test_camera_manager_real_camera_applies_props_and_resizes_output():
    created_caps: list[FakeVideoCapture] = []

    def fake_factory(idx: int):
        cap = FakeVideoCapture(index=idx, should_open=True, native_size=(480, 640))
        created_caps.append(cap)
        return cap

    mgr = CameraManager(
        SystemConfig(source="camera", camera_index=0, width=800, height=600, target_fps=30),
        capture_factory=fake_factory,
    )

    frame, tel = mgr.step_frame()
    assert len(created_caps) == 1
    assert frame.shape == (600, 800, 3)
    assert tel.source_active == "camera"
    assert tel.hardware_fallback is False

    # Mudar resolução para 320x240 e FPS para 15
    mgr.update_config({"resolution_preset": "320x240", "target_fps": 15})
    frame_small, tel_small = mgr.step_frame()
    assert frame_small.shape == (240, 320, 3)
    assert tel_small.width == 320
    assert tel_small.height == 240
    assert tel_small.target_fps == 15

    mgr.close()
    assert created_caps[0].released is True


def test_camera_manager_fallback_when_physical_camera_unavailable():
    mgr = CameraManager(
        SystemConfig(source="camera", camera_index=2, width=640, height=480),
        capture_factory=lambda idx: FakeVideoCapture(index=idx, should_open=False),
    )

    frame, tel = mgr.step_frame()
    assert frame.shape == (480, 640, 3)
    assert tel.source_active == "simulator"
    assert tel.hardware_fallback is True


def test_camera_manager_anomaly_injection_and_events_log():
    mgr = CameraManager(
        SystemConfig(
            source="simulator",
            vision_mode="thermal",
            thermal={"min_temp_c": 20.0, "max_temp_c": 120.0, "overheat_threshold_c": 85.0},
        )
    )
    mgr.step_frame()
    mgr.trigger_anomaly("thermal_spike", duration_frames=10)
    _, tel = mgr.step_frame()

    assert tel.overheat_alarm is True
    assert tel.alert_level == "CRITICO"
    events = mgr.get_events()
    assert len(events) >= 2
    categories = {ev.category for ev in events}
    assert "ANOMALY" in categories
    assert "THERMAL" in categories


def test_camera_manager_jpeg_snapshot_encoding():
    mgr = CameraManager()
    mgr.step_frame()
    jpeg_bytes = mgr.get_snapshot_jpeg()
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 500
    # Assinatura SOI JPEG (0xFF 0xD8)
    assert jpeg_bytes[:2] == b"\xff\xd8"


def test_camera_manager_invalidates_stale_telemetry_and_snapshot_on_config_change():
    import cv2

    mgr = CameraManager(
        SystemConfig(source="simulator", width=640, height=480, vision_mode="normal")
    )
    mgr.step_frame()
    assert mgr.get_latest_telemetry().width == 640

    # Alterar resolução e modo sem chamar step_frame explicitamente
    mgr.update_config({"resolution_preset": "1280x720", "vision_mode": "thermal"})
    tel_after = mgr.get_latest_telemetry()
    assert tel_after.width == 1280
    assert tel_after.height == 720
    assert tel_after.vision_mode == "thermal"

    snap_bytes = mgr.get_snapshot_jpeg()
    decoded = cv2.imdecode(np.frombuffer(snap_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (720, 1280, 3)


def test_camera_manager_physical_camera_disconnect_fallback_and_reconnection():
    state = {"healthy": True}
    created_caps: list[FakeVideoCapture] = []

    class DisconnectableCapture(FakeVideoCapture):
        def read(self):
            if not state["healthy"]:
                return False, None
            return super().read()

    def factory(idx: int):
        cap = DisconnectableCapture(index=idx, should_open=state["healthy"])
        created_caps.append(cap)
        return cap

    mgr = CameraManager(
        SystemConfig(source="camera", camera_index=0, width=640, height=480),
        capture_factory=factory,
    )

    # 1. Primeiro quadro com câmera física saudável
    _, tel1 = mgr.step_frame()
    assert tel1.source_active == "camera"
    assert tel1.hardware_fallback is False

    # 2. Simular desconexão abrupta (unplug USB)
    state["healthy"] = False
    _, tel2 = mgr.step_frame()
    assert tel2.source_active == "simulator"
    assert tel2.hardware_fallback is True
    assert created_caps[0].released is True

    # 3. Simular reconexão da câmera física
    state["healthy"] = True
    _, tel3 = mgr.step_frame()
    assert tel3.source_active == "camera"
    assert tel3.hardware_fallback is False
    assert len(created_caps) == 2

    # 4. Segunda desconexão deve registrar novo evento de aviso
    state["healthy"] = False
    _, tel4 = mgr.step_frame()
    assert tel4.hardware_fallback is True
    fallback_events = [
        ev for ev in mgr.get_events() if "fallback automático" in ev.message
    ]
    assert len(fallback_events) == 2


def test_camera_manager_physical_camera_anomaly_preserves_scene_temperature():
    class ModerateWarmCapture(FakeVideoCapture):
        def read(self):
            self.read_count += 1
            frame = np.full((self.native_h, self.native_w, 3), 90, dtype=np.uint8)
            frame[80:160, 100:160] = (140, 140, 140)
            return True, frame

    mgr = CameraManager(
        SystemConfig(
            source="camera",
            vision_mode="thermal",
            thermal={"kinetic_heat_gain": 0.0, "min_temp_c": 20.0, "max_temp_c": 120.0},
        ),
        capture_factory=lambda idx: ModerateWarmCapture(index=idx, should_open=True),
    )
    _, tel_base = mgr.step_frame()
    mgr.trigger_anomaly("thermal_spike", duration_frames=5)
    _, tel_spike = mgr.step_frame()

    assert tel_spike.source_active == "camera"
    assert tel_spike.max_temp_c > tel_base.max_temp_c
    assert tel_spike.overheat_alarm is True
    assert tel_spike.avg_temp_c >= tel_base.avg_temp_c


def test_camera_manager_stream_jpeg_deduplicates_within_fps_interval():
    mgr = CameraManager(SystemConfig(source="simulator", target_fps=10))
    jpeg1 = mgr.get_stream_jpeg()
    idx1 = mgr.get_latest_telemetry().frame_index

    # Chamada imediata (dentro do intervalo de 100ms) deve reutilizar o mesmo quadro
    jpeg2 = mgr.get_stream_jpeg()
    idx2 = mgr.get_latest_telemetry().frame_index
    assert idx2 == idx1
    assert jpeg2 == jpeg1


def test_camera_manager_personal_camera_discovery_mirror_and_reconnect():
    devices = CameraManager.list_available_cameras()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    assert "index" in devices[0]
    assert "label" in devices[0]
    assert CameraManager.get_camera_device_label(0).startswith(("/dev/video0", "Câmera #0"))

    class AsymmetricCapture(FakeVideoCapture):
        def read(self):
            if not self.isOpened():
                return False, None
            self.read_count += 1
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :40] = (0, 0, 220)  # Lado esquerdo vermelho
            return True, frame

    mgr = CameraManager(
        SystemConfig(source="camera", camera_index=0, width=160, height=120, mirror=False, hud_enabled=False),
        capture_factory=lambda idx: AsymmetricCapture(index=idx, should_open=True),
    )
    frame_no_mirror, _ = mgr.step_frame()
    assert int(frame_no_mirror[60, 10, 2]) == 220
    assert int(frame_no_mirror[60, 150, 2]) == 0

    mgr.update_config({"mirror": True})
    frame_mirrored, tel_mirrored = mgr.step_frame()
    assert int(frame_mirrored[60, 10, 2]) == 0
    assert int(frame_mirrored[60, 150, 2]) == 220
    # Alternar espelhamento em cena estática deve recalibrar fundo sem disparar falso alarme de movimento
    assert tel_mirrored.motion_detected is False
    assert tel_mirrored.motion_alarm is False

    assert mgr.reconnect_camera() is True
    mgr.close()


def test_camera_manager_handles_usb_unplug_exception_at_1080p_and_clean_recovery():
    import cv2

    state = {"raise_err": False}

    class Unpluggable1080pCapture(FakeVideoCapture):
        def read(self):
            if state["raise_err"]:
                raise cv2.error("VIDIOC_DQBUF: No such device")
            frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
            frame[100:250, 80:200] = (90, 120, 180)
            return True, frame

    mgr = CameraManager(
        SystemConfig(source="camera", camera_index=0, width=1920, height=1080, target_fps=30),
        capture_factory=lambda idx: Unpluggable1080pCapture(index=idx, should_open=not state["raise_err"]),
    )
    f1, tel1 = mgr.step_frame()
    assert f1.shape == (1080, 1920, 3)
    assert tel1.source_active == "camera"

    # Simular desconexão USB abrupta lançando cv2.error durante renderização 1920x1080
    state["raise_err"] = True
    f2, tel2 = mgr.step_frame()
    assert f2.shape == (1080, 1920, 3)
    assert tel2.source_active == "simulator"
    assert tel2.hardware_fallback is True

    # Reconectar câmera estática não deve disparar falso alarme de 98% contra o fundo do simulador
    state["raise_err"] = False
    _, tel3 = mgr.step_frame()
    assert tel3.source_active == "camera"
    assert tel3.hardware_fallback is False
    assert tel3.motion_detected is False
    assert tel3.motion_alarm is False
    mgr.close()
