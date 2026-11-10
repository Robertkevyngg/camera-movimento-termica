"""Testes de persistência local, token do painel Web, thread do pipeline e portabilidade."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.camera_capture as camera_capture_module
from app.api import API_TOKEN_COOKIE, API_TOKEN_HEADER, create_app
from app.camera_capture import choose_hardware_fps, preferred_backends
from app.camera_manager import CameraManager
from app.cli import (
    build_explicit_patch,
    main,
    parse_args,
    resolve_api_token,
    resolve_startup_config,
)
from app.config import EventRecord, SystemConfig
from app.desktop_app import run_native_app
from app.frame_pump import FramePump
from app.persistence import EventLog, get_data_dir, load_config, save_config

TOKEN = "segredo-de-teste"


def _wait_until(predicate, timeout_sec: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_event(event_id: int) -> EventRecord:
    return EventRecord(
        id=event_id,
        timestamp_iso="2026-01-01T00:00:00+00:00",
        category="SYSTEM",
        severity="NORMAL",
        message=f"evento {event_id} " + "x" * 80,
    )


def test_save_and_load_config_round_trip(tmp_path: Path):
    path = tmp_path / "sub" / "config.json"
    original = SystemConfig(source="simulator").apply_patch(
        {"resolution_preset": "320x240", "target_fps": 15, "thermal": {"palette": "turbo"}}
    )

    assert save_config(original, path) is True

    assert load_config(path) == original
    assert not path.with_suffix(".json.tmp").exists()


def test_load_config_ignores_missing_corrupted_and_invalid_files(tmp_path: Path):
    corrupted = tmp_path / "corrompido.json"
    corrupted.write_text("{nao é json", encoding="utf-8")
    invalid = tmp_path / "invalido.json"
    invalid.write_text('{"width": 99999}', encoding="utf-8")

    assert load_config(tmp_path / "inexistente.json") is None
    assert load_config(corrupted) is None
    assert load_config(invalid) is None


def test_save_config_returns_false_when_path_is_not_writable(tmp_path: Path):
    blocker = tmp_path / "arquivo"
    blocker.write_text("x", encoding="utf-8")

    assert save_config(SystemConfig(), blocker / "config.json") is False


def test_event_log_appends_and_drops_older_half_when_full(tmp_path: Path):
    path = tmp_path / "eventos.jsonl"
    log = EventLog(path, max_bytes=2_000)

    for event_id in range(1, 41):
        log.append(_make_event(event_id))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert 1 < len(lines) < 40
    assert EventRecord.model_validate_json(lines[-1]).id == 40
    assert EventRecord.model_validate_json(lines[0]).id > 1


def test_event_log_gives_up_quietly_after_disk_failure(tmp_path: Path):
    blocker = tmp_path / "arquivo"
    blocker.write_text("x", encoding="utf-8")
    log = EventLog(blocker / "eventos.jsonl")

    log.append(_make_event(1))
    log.append(_make_event(2))

    assert log._failed is True


def test_camera_manager_forwards_events_to_sink_and_survives_sink_failure():
    received: list[str] = []

    def failing_sink(event: EventRecord) -> None:
        received.append(event.message)
        if len(received) == 2:
            raise OSError("disco cheio")

    mgr = CameraManager(SystemConfig(source="simulator"), event_sink=failing_sink)

    mgr.update_config({"vision_mode": "thermal"})
    mgr.update_config({"vision_mode": "hybrid"})

    assert len(received) == 2  # iniciado + 1ª troca; depois da falha o destino é desligado
    assert len(mgr.get_events()) == 3
    mgr.close()


def test_get_data_dir_honors_environment_override(isolated_data_dir: Path):
    assert get_data_dir() == isolated_data_dir


def test_startup_config_uses_saved_values_with_typed_flags_on_top(tmp_path: Path):
    path = tmp_path / "config.json"
    saved = SystemConfig(source="simulator").apply_patch(
        {"resolution_preset": "320x240", "target_fps": 15, "vision_mode": "thermal"}
    )
    save_config(saved, path)
    argv = ["--app", "--fps=30", "--palette", "jet"]

    cfg = resolve_startup_config(parse_args(argv), argv, path)

    assert (cfg.width, cfg.height) == (320, 240)  # salvo, não o padrão 640x480 da CLI
    assert cfg.vision_mode == "thermal"
    assert cfg.source == "simulator"
    assert cfg.target_fps == 30  # digitado
    assert cfg.thermal.palette == "jet"


def test_startup_config_falls_back_to_cli_defaults_without_saved_file(tmp_path: Path):
    argv = ["--resolution", "1280x720"]

    cfg = resolve_startup_config(parse_args(argv), argv, tmp_path / "nao-existe.json")

    assert (cfg.width, cfg.height, cfg.target_fps) == (1280, 720, 24)
    assert resolve_startup_config(parse_args(argv), argv, None).width == 1280


def test_explicit_patch_only_contains_typed_flags():
    argv = ["--no-mirror", "--lite", "--resolution", "1280x720"]

    patch = build_explicit_patch(parse_args(argv), argv)

    assert patch["mirror"] is False
    assert patch["resolution_preset"] == "320x240"  # --lite prevalece sobre --resolution
    assert "vision_mode" not in patch and "source" not in patch


def test_main_persists_final_config_and_respects_no_persist(monkeypatch, isolated_data_dir: Path):
    import app.cli as cli_module

    def fake_app(config, max_frames=None, on_exit=None, event_sink=None):
        on_exit(config.apply_patch({"vision_mode": "hybrid"}))
        return 0

    monkeypatch.setattr(cli_module, "run_native_app", fake_app)

    assert main(["--app", "--source", "simulator", "--no-persist"]) == 0
    assert not (isolated_data_dir / "config.json").exists()

    assert main(["--app", "--source", "simulator"]) == 0
    saved = load_config(isolated_data_dir / "config.json")
    assert saved is not None and saved.vision_mode == "hybrid"


def test_api_token_is_generated_only_for_network_hosts(monkeypatch):
    monkeypatch.delenv("VISAO_API_TOKEN", raising=False)

    assert resolve_api_token(parse_args([])) is None
    assert resolve_api_token(parse_args(["--host", "localhost"])) is None
    generated = resolve_api_token(parse_args(["--host", "0.0.0.0"]))
    assert generated is not None and len(generated) >= 24
    assert resolve_api_token(parse_args(["--host", "0.0.0.0", "--api-token", "abc"])) == "abc"

    monkeypatch.setenv("VISAO_API_TOKEN", "do-ambiente")
    assert resolve_api_token(parse_args([])) == "do-ambiente"


def test_api_rejects_requests_without_valid_token():
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    client = TestClient(create_app(mgr, api_token=TOKEN))

    for path in ("/", "/api/health", "/api/config", "/api/snapshot", "/static/app.js"):
        response = client.get(path)
        assert response.status_code == 401, path
    assert client.get("/api/health", headers={API_TOKEN_HEADER: "errado"}).status_code == 401
    assert client.get("/api/health?token=errado").status_code == 401
    assert client.patch("/api/config", json={"target_fps": 5}).status_code == 401
    assert mgr.get_config().target_fps == 24
    mgr.close()


def test_api_accepts_header_and_query_token_then_cookie():
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    client = TestClient(create_app(mgr, api_token=TOKEN))

    assert client.get("/api/health", headers={API_TOKEN_HEADER: TOKEN}).status_code == 200

    first_visit = client.get(f"/?token={TOKEN}")
    assert first_visit.status_code == 200
    set_cookie = first_visit.headers["set-cookie"].lower()
    assert API_TOKEN_COOKIE in set_cookie and "httponly" in set_cookie
    assert "samesite=strict" in set_cookie

    assert client.get("/api/config").status_code == 200  # cookie da primeira visita
    assert client.get("/static/app.js").status_code == 200
    mgr.close()


def test_api_without_token_stays_open_for_local_use():
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    client = TestClient(create_app(mgr))

    assert client.get("/api/health").status_code == 200
    mgr.close()


def test_frame_pump_produces_frames_pauses_and_stops():
    mgr = CameraManager(SystemConfig(source="simulator", width=160, height=120, target_fps=60))
    pump = FramePump(mgr)
    assert pump.poll() is None
    pump.start()
    try:
        assert _wait_until(lambda: (pump.poll() or (0, 0, 0))[2] >= 3)
        latest = pump.poll()
        assert latest.frame_bgr.shape == (120, 160, 3)
        assert latest.telemetry.source_active == "simulator"

        pump.set_paused(True)
        time.sleep(0.1)
        paused_sequence = pump.poll().sequence
        time.sleep(0.15)
        assert pump.poll().sequence == paused_sequence

        pump.set_paused(False)
        assert _wait_until(lambda: pump.poll().sequence > paused_sequence)
    finally:
        pump.stop()
        mgr.close()
    stopped_sequence = pump.poll().sequence
    time.sleep(0.1)
    assert pump.poll().sequence == stopped_sequence


def test_frame_pump_keeps_running_after_step_error(monkeypatch):
    import app.frame_pump as frame_pump_module

    mgr = CameraManager(SystemConfig(source="simulator", width=160, height=120, target_fps=60))
    original_step = mgr.step_frame
    calls = {"count": 0}

    def flaky_step():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("quadro corrompido")
        return original_step()

    monkeypatch.setattr(mgr, "step_frame", flaky_step)
    monkeypatch.setattr(frame_pump_module, "ERROR_BACKOFF_SEC", 0.01)
    pump = FramePump(mgr)
    pump.start()
    try:
        assert _wait_until(lambda: pump.poll() is not None)
    finally:
        pump.stop()
        mgr.close()


def test_run_native_app_reports_final_config_and_events_on_exit(tmp_path: Path):
    final_configs: list[SystemConfig] = []
    events: list[EventRecord] = []
    cfg = SystemConfig(source="simulator", width=320, height=240, target_fps=30)

    try:
        frames = run_native_app(
            config=cfg,
            max_frames=2,
            start_hidden=True,
            snapshot_dir=tmp_path,
            on_exit=final_configs.append,
            event_sink=events.append,
        )
    except RuntimeError:
        pytest.skip("Ambiente gráfico indisponível")

    assert frames == 2
    assert final_configs == [cfg]
    assert events and "Sistema iniciado" in events[0].message


def test_hardware_fps_follows_low_targets():
    assert choose_hardware_fps(15) == 15
    assert choose_hardware_fps(5) == 15
    assert choose_hardware_fps(24) == 30
    assert choose_hardware_fps(60) == 30


def test_camera_backends_and_labels_adapt_to_platform(monkeypatch):
    import cv2

    monkeypatch.setattr(sys, "platform", "win32")
    assert preferred_backends() == (cv2.CAP_DSHOW,)
    assert CameraManager.get_camera_device_label(2) == "Câmera #2"

    monkeypatch.setattr(sys, "platform", "darwin")
    assert preferred_backends() == ()

    monkeypatch.setattr(sys, "platform", "linux")
    assert preferred_backends() == (cv2.CAP_V4L2,)


def test_open_camera_device_falls_back_to_generic_backend(monkeypatch):
    opened_with: list[tuple] = []

    class FakeCap:
        def __init__(self, *args):
            opened_with.append(args)
            self.generic = len(args) == 1

        def isOpened(self):
            return self.generic

        def set(self, *_args):
            return True

        def release(self):
            pass

    monkeypatch.setattr(camera_capture_module.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(sys, "platform", "linux")

    cap = camera_capture_module.open_camera_device(0, 640, 480, 30)

    assert cap is not None and cap.generic is True
    assert len(opened_with) == 2 and len(opened_with[0]) == 3


def test_local_api_rejects_foreign_host_header_dns_rebinding():
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    client = TestClient(create_app(mgr))

    assert client.get("/api/snapshot", headers={"Host": "evil.example:8000"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/api/health", headers={"Host": "localhost:8000"}).status_code == 200
    assert client.get("/api/health", headers={"Host": "[::1]:8000"}).status_code == 200
    mgr.close()


@pytest.mark.parametrize("api_token", [None, TOKEN])
def test_api_rejects_cross_site_requests(api_token):
    mgr = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    client = TestClient(create_app(mgr, api_token=api_token))
    auth = {API_TOKEN_HEADER: TOKEN} if api_token else {}
    events_before = len(mgr.get_events())

    forged_form = client.post("/api/reset", headers={**auth, "Origin": "https://evil.example"})
    forged_image = client.get(
        "/api/telemetry?step=true", headers={**auth, "Sec-Fetch-Site": "cross-site"}
    )
    same_origin = client.post("/api/reset", headers={**auth, "Origin": "http://testserver"})

    assert forged_form.status_code == 403
    assert forged_image.status_code == 403
    assert same_origin.status_code == 200
    assert len(mgr.get_events()) == events_before + 1  # só a requisição legítima recalibrou
    mgr.close()


def test_persisted_files_are_private_to_the_user(tmp_path: Path):
    if sys.platform == "win32":
        pytest.skip("Permissões POSIX não se aplicam ao Windows")
    data_dir = tmp_path / "privado"
    save_config(SystemConfig(), data_dir / "config.json")
    EventLog(data_dir / "eventos.jsonl").append(_make_event(1))

    assert (data_dir.stat().st_mode & 0o777) == 0o700
    assert ((data_dir / "config.json").stat().st_mode & 0o777) == 0o600
    assert ((data_dir / "eventos.jsonl").stat().st_mode & 0o777) == 0o600


def test_event_log_trim_keeps_lines_with_unicode_separators_intact(tmp_path: Path):
    path = tmp_path / "eventos.jsonl"
    log = EventLog(path, max_bytes=1_500)

    for event_id in range(1, 31):
        log.append(_make_event(event_id).model_copy(update={"message": f"a b\x85c {event_id}"}))

    lines = path.read_bytes().split(b"\n")
    parsed = [EventRecord.model_validate_json(line) for line in lines if line]
    assert parsed[-1].id == 30 and len(parsed) < 30


def test_closed_manager_does_not_reopen_camera():
    opened: list[int] = []

    class Cap:
        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            import numpy as np

            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def release(self):
            pass

    def factory(index: int) -> Cap:
        opened.append(index)
        return Cap()

    mgr = CameraManager(SystemConfig(source="camera", width=160, height=120), capture_factory=factory)
    mgr.step_frame()
    mgr.close()

    _, tel = mgr.step_frame()

    assert opened == [0]
    assert tel.source_active == "simulator"
