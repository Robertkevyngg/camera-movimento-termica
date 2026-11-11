"""Testes da captura em tempo real (60 FPS, JPEG direto, relógio do driver) e da gravação."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import pytest

from app.camera_capture import (
    BackgroundCapture,
    driver_timestamp,
    enable_raw_mjpeg,
)
from app.camera_manager import CameraManager
from app.config import SystemConfig
from app.event_recorder import EventRecorder, MjpegAviWriter

FPS = 60


def make_jpegs(count: int, width: int = 320, height: int = 240) -> list[np.ndarray]:
    """Quadros JPEG com um retângulo andando, como sairiam de uma câmera MJPG."""
    jpegs = []
    for i in range(count):
        img = np.full((height, width, 3), 120, np.uint8)
        x = (i * 4) % (width - 40)
        cv2.rectangle(img, (x, 60), (x + 40, 180), (30, 30, 200), -1)
        jpegs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].reshape(-1))
    return jpegs


def read_times(csv_path) -> list[float]:
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()[1:]
    return [float(line.split(";")[2].replace(",", ".")) for line in lines]


def wait_until(condition, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_mjpeg_avi_writer_produces_playable_file(tmp_path):
    jpegs = make_jpegs(90)
    path = tmp_path / "teste.avi"
    writer = MjpegAviWriter(path, 59.94, 320, 240)
    for jpeg in jpegs:
        writer.write(jpeg)
    writer.close()

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 90
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(59.94, abs=0.01)
    frames = 0
    while cap.read()[0]:
        frames += 1
    assert frames == 90


def test_recorder_keeps_preroll_and_cuts_at_frame_times(tmp_path):
    """60 FPS com horários sintéticos: o vídeo vai de 2 s antes do início até o fim pedido."""
    jpegs = make_jpegs(8 * FPS)
    recorder = EventRecorder(lambda: FPS, pre_seconds=2.0)
    base = tmp_path / "evento"
    for i, jpeg in enumerate(jpegs):
        t = 100.0 + i / FPS
        if i == 4 * FPS:
            recorder.start(t, base)
        if i == 6 * FPS:
            recorder.stop(t)
        recorder.receive(t, np.zeros((1, 1, 3), np.uint8), jpeg)
    recorder.close()

    video = tmp_path / "evento.avi"
    assert recorder.files == [video]
    assert recorder.dropped_frames == 0
    # 2 s de pré-gravação + 2 s de evento = 4 s a 60 FPS (o quadro do "stop" fica de fora).
    assert recorder.recorded_frames == 4 * FPS + 1
    assert int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) == 4 * FPS + 1
    times = read_times(tmp_path / "evento_tempos.csv")
    gaps = np.diff(times)
    assert len(times) == 4 * FPS + 1
    assert gaps.max() == pytest.approx(1 / FPS, abs=0.002)  # nenhum quadro faltando


def test_recorder_reencodes_when_camera_gives_decoded_frames(tmp_path):
    recorder = EventRecorder(lambda: 30.0, pre_seconds=0.5)
    base = tmp_path / "decodificado"
    frame = np.full((240, 320, 3), 90, np.uint8)
    for i in range(90):
        t = 10.0 + i / 30
        if i == 30:
            recorder.start(t, base)
        recorder.receive(t, frame, None)
    recorder.close()

    assert len(recorder.files) == 1
    video = recorder.files[0]
    assert video.suffix in (".mp4", ".avi")
    assert int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) == recorder.recorded_frames
    # 0,5 s de pré-gravação (15 ou 16 quadros) + 60 quadros até o fim.
    assert 74 <= recorder.recorded_frames <= 76


class FakeV4l2Capture:
    """Imita uma câmera Logitech no V4L2 em MJPG, com horário do driver."""

    def __init__(self, jpegs: list[np.ndarray], realtime: bool = True) -> None:
        self.jpegs = jpegs
        self.n = 0
        self.raw = False
        self.t0: float | None = None
        self.realtime = realtime

    def isOpened(self) -> bool:  # noqa: N802 - mesma API do cv2.VideoCapture
        return True

    def getBackendName(self) -> str:  # noqa: N802
        return "V4L2"

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            self.raw = value == 0
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"MJPG"))
        if prop == cv2.CAP_PROP_POS_MSEC and self.t0 is not None:
            return 1000.0 * (self.t0 + (self.n - 1) / FPS)
        return 0.0

    def read(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        if self.realtime:
            delay = self.t0 + self.n / FPS - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        jpeg = self.jpegs[self.n % len(self.jpegs)]
        self.n += 1
        if self.raw:
            return True, jpeg.reshape(1, -1).copy()
        return True, cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    def release(self) -> None:
        pass


def test_raw_mjpeg_only_on_linux_v4l2(monkeypatch):
    cam = FakeV4l2Capture(make_jpegs(5), realtime=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert enable_raw_mjpeg(cam) is True
    assert cam.raw is True

    other = FakeV4l2Capture(make_jpegs(5), realtime=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert enable_raw_mjpeg(other) is False
    assert other.raw is False


def test_driver_timestamp_is_used_only_when_plausible():
    class Cap:
        def __init__(self, ms: float) -> None:
            self.ms = ms

        def get(self, _prop: int) -> float:
            return self.ms

    now = 1000.0
    assert driver_timestamp(Cap(999_980.0), now) == pytest.approx(999.98)  # 20 ms antes
    assert driver_timestamp(Cap(0.0), now) == now  # driver sem horário
    assert driver_timestamp(Cap(5_000_000.0), now) == now  # outro relógio


def test_background_capture_feeds_recorder_with_every_jpeg(monkeypatch):
    """Em tempo real a 60 FPS: a gravação recebe todos os quadros, com o horário do driver."""
    monkeypatch.setattr(sys, "platform", "linux")
    # Não mexe no driver de uma câmera real que esteja ligada na máquina de teste.
    monkeypatch.setattr("app.camera_capture.v4l2_control", lambda *_args: None)
    jpegs = make_jpegs(30)
    received: list[tuple[float, bool]] = []
    capture = BackgroundCapture(
        lambda *_args: FakeV4l2Capture(jpegs), 0, 320, 240, fps=FPS, realtime_tuning=True
    )
    capture.frame_listener = lambda t, frame, jpeg: received.append((t, jpeg is not None))
    capture.start()
    try:
        assert wait_until(lambda: len(received) >= 60)
        frame, _seq = capture.read_latest()
        assert frame is not None and frame.shape == (240, 320, 3)
        assert capture.raw_jpeg is True
        assert capture.camera_fps == pytest.approx(FPS, rel=0.1)
    finally:
        capture.stop()
    times = np.array([t for t, _ in received])
    assert all(is_jpeg for _, is_jpeg in received)
    assert np.diff(times).max() < 2.5 / FPS  # quadros contínuos, sem buracos


class MovingObjectCapture:
    """Câmera síncrona para o CameraManager: cena parada e depois um objeto grande se mexendo."""

    def __init__(self) -> None:
        self.n = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, *_args) -> bool:
        return True

    def get(self, *_args) -> float:
        return 0.0

    def read(self):
        self.n += 1
        img = np.full((240, 320, 3), 110, np.uint8)
        if 20 <= self.n < 40:
            x = (self.n * 12) % 200
            cv2.rectangle(img, (x, 40), (x + 110, 220), (20, 20, 20), -1)
        return True, img

    def release(self) -> None:
        pass


def test_camera_manager_records_motion_events(tmp_path):
    cfg = SystemConfig(
        source="camera", width=320, height=240, target_fps=30, mirror=False,
        record_events=True, record_pre_sec=0.2, record_post_sec=0.5,
    )  # fmt: skip
    manager = CameraManager(
        cfg, capture_factory=lambda _idx: MovingObjectCapture(), recordings_dir=tmp_path
    )
    try:
        recording_seen = False
        for _ in range(70):
            _, tel = manager.step_frame()
            recording_seen = recording_seen or tel.recording
            time.sleep(1 / 30)
        assert recording_seen
        messages = [ev.message for ev in manager.get_events(limit=50)]
        assert any("Gravando vídeo do movimento" in m for m in messages)
        assert any("Vídeo do movimento salvo" in m for m in messages)
    finally:
        manager.close()

    videos = sorted(tmp_path.rglob("*_movimento.mp4")) + sorted(tmp_path.rglob("*_movimento.avi"))
    assert videos, "nenhum vídeo gravado"
    assert int(cv2.VideoCapture(str(videos[0])).get(cv2.CAP_PROP_FRAME_COUNT)) > 10
    assert list(tmp_path.rglob("*_tempos.csv"))


def test_recording_off_by_default(tmp_path):
    manager = CameraManager(
        SystemConfig(source="camera", width=320, height=240),
        capture_factory=lambda _idx: MovingObjectCapture(),
        recordings_dir=tmp_path,
    )
    try:
        for _ in range(45):
            manager.step_frame()
    finally:
        manager.close()
    assert not list(tmp_path.rglob("*_movimento.*"))
