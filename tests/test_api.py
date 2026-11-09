"""Testes de integração da API FastAPI e endpoints SCADA."""

from fastapi.testclient import TestClient

from app.api import create_app
from app.camera_manager import CameraManager
from app.config import SystemConfig


def _make_client() -> TestClient:
    mgr = CameraManager(SystemConfig(source="simulator", width=640, height=480, target_fps=24))
    app = create_app(mgr)
    return TestClient(app)


def test_health_and_presets_endpoints():
    client = _make_client()

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    body_health = res_health.json()
    assert body_health["success"] is True
    assert body_health["error"] is None
    assert body_health["data"]["status"] == "online"

    res_presets = client.get("/api/presets")
    assert res_presets.status_code == 200
    body_presets = res_presets.json()
    assert body_presets["success"] is True
    assert "640x480" in body_presets["data"]["resolutions"]
    assert "1280x720" in body_presets["data"]["resolutions"]
    assert 30 in body_presets["data"]["fps_presets"]
    assert "inferno" in body_presets["data"]["thermal_palettes"]
    assert "human" in body_presets["data"]["thermal_profiles"]
    assert isinstance(body_presets["data"]["cameras"], list)

    res_reconn = client.post("/api/camera/reconnect")
    assert res_reconn.status_code == 200
    assert res_reconn.json()["success"] is True


def test_get_and_patch_config_resolution_fps_and_thermal():
    client = _make_client()

    res_get = client.get("/api/config")
    assert res_get.status_code == 200
    assert res_get.json()["data"]["width"] == 640
    assert res_get.json()["data"]["target_fps"] == 24

    res_patch = client.patch(
        "/api/config",
        json={
            "resolution_preset": "1280x720",
            "target_fps": 30,
            "vision_mode": "thermal",
            "thermal": {"palette": "turbo", "overheat_threshold_c": 88.0},
            "motion": {"sensitivity": 80, "algorithm": "mog2"},
        },
    )
    assert res_patch.status_code == 200
    data = res_patch.json()["data"]
    assert data["width"] == 1280
    assert data["height"] == 720
    assert data["target_fps"] == 30
    assert data["vision_mode"] == "thermal"
    assert data["thermal"]["palette"] == "turbo"
    assert data["motion"]["algorithm"] == "mog2"


def test_patch_config_rejects_invalid_values_with_error_envelope():
    client = _make_client()

    res_bad_fps = client.patch("/api/config", json={"target_fps": 999})
    assert res_bad_fps.status_code == 422
    body = res_bad_fps.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"] is not None

    res_bad_preset = client.patch(
        "/api/config", json={"resolution_preset": "invalid_preset"}
    )
    assert res_bad_preset.status_code == 422
    assert res_bad_preset.json()["success"] is False


def test_telemetry_anomaly_reset_and_events_workflow():
    client = _make_client()

    res_tel = client.get("/api/telemetry?step=true")
    assert res_tel.status_code == 200
    tel = res_tel.json()["data"]
    assert tel["width"] == 640
    assert tel["height"] == 480
    assert "max_temp_c" in tel

    # Injetar anomalia de pico térmico
    res_anom = client.post(
        "/api/anomaly", json={"kind": "thermal_spike", "duration_frames": 15}
    )
    assert res_anom.status_code == 200
    assert res_anom.json()["success"] is True

    res_tel_after = client.get("/api/telemetry?step=true")
    assert res_tel_after.json()["data"]["overheat_alarm"] is True

    # Consultar eventos
    res_events = client.get("/api/events?limit=10")
    assert res_events.status_code == 200
    ev_data = res_events.json()["data"]
    assert ev_data["total"] >= 2
    assert len(ev_data["items"]) >= 2

    # Resetar sensores
    res_reset = client.post("/api/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["success"] is True

    # Limpar eventos
    res_clear = client.delete("/api/events")
    assert res_clear.status_code == 200
    assert client.get("/api/events").json()["data"]["total"] == 0


def test_snapshot_and_mjpeg_stream_endpoints():
    client = _make_client()

    res_snap = client.get("/api/snapshot")
    assert res_snap.status_code == 200
    assert res_snap.headers["content-type"] == "image/jpeg"
    assert res_snap.content[:2] == b"\xff\xd8"

    res_stream = client.get("/api/stream?max_frames=2")
    assert res_stream.status_code == 200
    assert "multipart/x-mixed-replace" in res_stream.headers["content-type"]
    assert b"--frame" in res_stream.content
    assert b"Content-Type: image/jpeg" in res_stream.content


def test_dashboard_html_served_at_root():
    client = _make_client()
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Sistema de Visão Industrial" in res.text


def test_api_edge_cases_invalid_anomaly_and_lazy_telemetry():
    client = _make_client()

    # Telemetria inicial sem step=true deve inicializar automaticamente o primeiro quadro
    res_lazy = client.get("/api/telemetry")
    assert res_lazy.status_code == 200
    assert res_lazy.json()["success"] is True

    # Anomalia inválida retorna 422
    res_bad_anom = client.post(
        "/api/anomaly", json={"kind": "not_an_anomaly", "duration_frames": 10}
    )
    assert res_bad_anom.status_code == 422
    assert res_bad_anom.json()["success"] is False


def test_concurrent_stream_clients_and_rapid_config_switching():
    from concurrent.futures import ThreadPoolExecutor

    mgr = CameraManager(SystemConfig(source="simulator", width=640, height=480, target_fps=30))
    app = create_app(mgr)
    client = TestClient(app)

    def consume_stream():
        res = client.get("/api/stream?max_frames=3")
        assert res.status_code == 200
        assert b"Content-Length:" in res.content
        return res.status_code

    def switch_resolutions():
        for preset in ("320x240", "800x600", "640x480"):
            r = client.patch("/api/config", json={"resolution_preset": preset, "target_fps": 30})
            assert r.status_code == 200
        return 200

    with ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(consume_stream)
        f2 = pool.submit(consume_stream)
        f3 = pool.submit(switch_resolutions)
        assert f1.result() == 200
        assert f2.result() == 200
        assert f3.result() == 200
