"""Testes dos caminhos de desempenho: análise reduzida, captura em thread e modo leve."""

import threading
import time

import cv2
import numpy as np
import pytest

import app.camera_capture as camera_capture_module
import app.camera_manager as camera_manager_module
from app.camera_capture import BackgroundCapture
from app.camera_manager import CameraManager
from app.cli import build_initial_config, parse_args
from app.config import MotionConfig, SystemConfig, ThermalConfig
from app.frame_utils import ANALYSIS_MAX_WIDTH, compute_analysis_size, gray_percentiles
from app.motion_detector import MotionDetector
from app.thermal_simulator import ThermalSimulator


class SlowCapture:
    """Câmera falsa cuja leitura bloqueia, como uma webcam em pouca luz."""

    def __init__(self, read_delay_sec: float = 0.0, fail_after: int | None = None) -> None:
        self.read_delay_sec = read_delay_sec
        self.fail_after = fail_after
        self.read_count = 0
        self.released = threading.Event()
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:
        return not self.released.is_set()

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def read(self):
        time.sleep(self.read_delay_sec)
        self.read_count += 1
        if self.fail_after is not None and self.read_count > self.fail_after:
            return False, None
        return True, np.full((120, 160, 3), 40 + self.read_count % 50, dtype=np.uint8)

    def release(self) -> None:
        self.released.set()


def _wait_until(predicate, timeout_sec: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_compute_analysis_size_keeps_small_frames_and_caps_large_ones():
    assert compute_analysis_size(320, 240) == (320, 240, 1.0)
    assert compute_analysis_size(160, 120) == (160, 120, 1.0)

    width, height, factor = compute_analysis_size(1280, 720)

    assert (width, height) == (ANALYSIS_MAX_WIDTH, 180)
    assert factor == pytest.approx(4.0)


def test_gray_percentiles_match_numpy_within_one_gray_level():
    gray = np.random.default_rng(7).integers(0, 256, size=(240, 320), dtype=np.uint8)

    fast = gray_percentiles(gray, (5.0, 50.0, 92.0))

    expected = np.percentile(gray, (5.0, 50.0, 92.0))
    assert fast == pytest.approx(tuple(expected), abs=1.0)


def test_gray_percentiles_of_uniform_image_return_its_level():
    gray = np.full((60, 80), 123, dtype=np.uint8)

    assert gray_percentiles(gray, (5.0, 75.0)) == (123.0, 123.0)


def test_motion_detector_reports_boxes_in_full_resolution_coordinates():
    detector = MotionDetector()
    cfg = MotionConfig(sensitivity=75, min_area=400, alarm_motion_pct=1.0)
    background = np.full((720, 1280, 3), 40, dtype=np.uint8)
    for _ in range(3):
        detector.process(background, cfg)
    moved = background.copy()
    cv2.rectangle(moved, (800, 300), (1000, 500), (240, 240, 240), -1)

    result = detector.process(moved, cfg)

    assert result.motion_mask.shape == (720, 1280)
    assert result.motion_detected is True
    box = result.bounding_boxes[0]
    assert abs(box.center_x - 900) <= 24
    assert abs(box.center_y - 400) <= 24
    assert 180 <= box.w <= 280 and 180 <= box.h <= 280
    assert box.area >= 200 * 200 * 0.8


def test_thermal_simulator_maps_hotspot_back_to_full_resolution():
    sim = ThermalSimulator()
    cfg = ThermalConfig(kinetic_heat_gain=0.0)
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    heat_hint = np.full((720, 1280), 0.1, dtype=np.float32)
    cv2.circle(heat_hint, (1000, 200), 30, 0.98, -1)

    result = sim.process(frame, cfg, heat_hint=heat_hint)

    assert result.thermal_bgr is not None
    assert result.thermal_bgr.shape == (720, 1280, 3)
    assert result.temp_map_c.shape == (180, 320)
    assert abs(result.hotspot_x - 1000) <= 36
    assert abs(result.hotspot_y - 200) <= 36
    assert result.overheat_alarm is True


def test_thermal_simulator_skips_colorized_image_when_not_requested():
    sim = ThermalSimulator()
    frame = np.full((240, 320, 3), 90, dtype=np.uint8)

    result = sim.process(frame, ThermalConfig(), render_image=False)

    assert result.thermal_bgr is None
    assert result.min_temp_c <= result.avg_temp_c <= result.max_temp_c


def test_background_capture_delivers_latest_frame_without_blocking_reader():
    cap = SlowCapture(read_delay_sec=0.15)
    capture = BackgroundCapture(lambda *_a: cap, 0, 160, 120)
    capture.start()
    try:
        assert capture.wait_for_frame(3.0) is True

        t0 = time.perf_counter()
        results = [capture.read_latest() for _ in range(50)]
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1  # 50 leituras não esperam pela câmera lenta (0.15 s/quadro)
        frame, sequence = results[-1]
        assert frame is not None and frame.shape == (120, 160, 3)
        assert sequence >= 1
    finally:
        capture.stop()
    assert cap.released.is_set()


def test_background_capture_reopens_device_after_read_failure():
    opened: list[SlowCapture] = []

    def opener(*_args):
        cap = SlowCapture(fail_after=2 if not opened else None)
        opened.append(cap)
        return cap

    capture = BackgroundCapture(opener, 0, 160, 120, retry_interval_sec=0.01)
    capture.start()
    try:
        assert _wait_until(lambda: len(opened) >= 2)
        assert opened[0].released.is_set()
        assert _wait_until(lambda: capture.read_latest()[0] is not None)
    finally:
        capture.stop()


def test_background_capture_reports_no_frame_when_device_cannot_open_or_goes_stale():
    never_opens = BackgroundCapture(lambda *_a: None, 3, 160, 120, retry_interval_sec=0.01)
    never_opens.start()
    try:
        assert never_opens.wait_for_frame(0.1) is False
        assert never_opens.read_latest()[0] is None
    finally:
        never_opens.stop()

    cap = SlowCapture(fail_after=1)
    stale = BackgroundCapture(
        lambda *_a: cap, 0, 160, 120, retry_interval_sec=5.0, stale_frame_sec=0.05
    )
    stale.start()
    try:
        assert stale.wait_for_frame(3.0) is True
        assert _wait_until(lambda: stale.read_latest()[0] is None)
    finally:
        stale.stop()


def test_camera_manager_background_capture_keeps_stepping_while_camera_is_slow():
    opened: list[SlowCapture] = []

    def factory(_index: int) -> SlowCapture:
        opened.append(SlowCapture(read_delay_sec=0.2))
        return opened[-1]

    mgr = CameraManager(
        SystemConfig(source="camera", width=160, height=120, mirror=False),
        capture_factory=factory,
        background_capture=True,
    )
    try:
        t_start = time.perf_counter()
        frame, first = mgr.step_frame()
        assert (time.perf_counter() - t_start) < 0.15  # tela de espera, sem aguardar a câmera
        assert frame.shape == (120, 160, 3)
        assert first.hardware_fallback is False
        assert mgr.reconnect_camera() is True  # reconexão explícita aguarda o 1º quadro

        t0 = time.perf_counter()
        telemetries = [mgr.step_frame()[1] for _ in range(20)]
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0  # leitura síncrona custaria 20 x 0.2 s = 4 s
        assert all(tel.source_active == "camera" for tel in telemetries)
        assert opened[-1].props[cv2.CAP_PROP_FRAME_WIDTH] == 160.0
    finally:
        mgr.close()
    assert all(cap.released.is_set() for cap in opened)


def test_camera_manager_applies_anomaly_while_camera_is_still_connecting():
    mgr = CameraManager(
        SystemConfig(source="camera", width=320, height=240),
        capture_factory=lambda _idx: SlowCapture(read_delay_sec=0.6),
        background_capture=True,
    )
    try:
        mgr.trigger_anomaly("thermal_spike", duration_frames=5)

        _, tel = mgr.step_frame()

        assert tel.hardware_fallback is False
        assert tel.overheat_alarm is True
        assert tel.alert_level == "CRITICO"
    finally:
        mgr.close()


def test_camera_manager_background_capture_falls_back_when_camera_is_missing(monkeypatch):
    monkeypatch.setattr(camera_manager_module, "CAMERA_OPEN_WAIT_SEC", 0.15)
    mgr = CameraManager(
        SystemConfig(source="camera", width=160, height=120),
        capture_factory=lambda _idx: None,
        background_capture=True,
        camera_retry_interval_sec=0.01,
    )
    try:
        t0 = time.perf_counter()
        _, connecting = mgr.step_frame()
        assert (time.perf_counter() - t0) < 0.1  # nunca bloqueia esperando o hardware
        assert connecting.hardware_fallback is False

        time.sleep(0.2)  # prazo de conexão esgotado: assume câmera indisponível
        _, fallback = mgr.step_frame()

        assert fallback.hardware_fallback is True
        assert fallback.source_active == "simulator"
        assert any("indisponível" in ev.message for ev in mgr.get_events())
    finally:
        mgr.close()


def test_camera_manager_only_renders_thermal_image_in_thermal_modes(monkeypatch):
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    seen: list[bool] = []
    original = mgr._thermal_simulator.process

    def spy(*args, **kwargs):
        seen.append(kwargs["render_image"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mgr._thermal_simulator, "process", spy)

    mgr.step_frame()
    mgr.update_config({"vision_mode": "hybrid"})
    frame, _ = mgr.step_frame()

    assert seen == [False, True]
    assert frame.shape == (240, 320, 3)
    mgr.close()


def test_cli_lite_flag_builds_low_cost_profile():
    cfg = build_initial_config(parse_args(["--lite", "--resolution", "1280x720", "--fps", "30"]))

    assert (cfg.width, cfg.height, cfg.target_fps) == (320, 240, 15)
    assert cfg.motion.show_trails is False


def test_background_capture_survives_unexpected_opener_error():
    attempts: list[int] = []

    def opener(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("erro inesperado do driver")
        return SlowCapture()

    capture = BackgroundCapture(opener, 0, 160, 120, retry_interval_sec=0.01)
    capture.start()
    try:
        assert capture.wait_for_frame(3.0) is True
        assert len(attempts) >= 2
    finally:
        capture.stop()


def test_background_capture_waits_for_previous_thread_on_same_device(monkeypatch):
    release_first = threading.Event()
    second_opened = threading.Event()

    def blocked_opener(*_args):
        release_first.wait(5.0)
        return None

    def second_opener(*_args):
        second_opened.set()
        return SlowCapture()

    first = BackgroundCapture(blocked_opener, 7, 160, 120)
    first.start()
    time.sleep(0.05)
    monkeypatch.setattr(camera_capture_module, "STOP_JOIN_TIMEOUT_SEC", 0.05)
    lingering = first.stop()  # driver preso: stop() desiste e devolve a thread viva
    assert lingering is not None
    second = BackgroundCapture(second_opener, 7, 160, 120, predecessor=lingering)
    second.start()
    try:
        assert second_opened.wait(0.3) is False  # dispositivo ainda ocupado pela 1ª thread

        release_first.set()

        assert second.wait_for_frame(3.0) is True
    finally:
        release_first.set()
        second.stop()
