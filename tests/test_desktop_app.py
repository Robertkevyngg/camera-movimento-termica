"""Testes unitários e de integração para o aplicativo desktop nativo (Tkinter SCADA)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.cli as cli_mod
import desktop_app as desktop_launcher
from app.camera_manager import CameraManager
from app.cli import main, parse_args
from app.config import SystemConfig, ThermalConfig
from app.desktop_app import (
    IndustrialVisionDesktopApp,
    canvas_rect_to_roi,
    compute_aspect_fit,
    frame_to_ppm_bytes,
    run_native_app,
)


def test_compute_aspect_fit_calculations():
    w, h, ox, oy = compute_aspect_fit(640, 480, 800, 600)
    assert (w, h) == (800, 600)
    assert (ox, oy) == (0, 0)

    w2, h2, ox2, oy2 = compute_aspect_fit(1280, 720, 800, 600)
    assert w2 == 800
    assert h2 == 450
    assert ox2 == 0
    assert oy2 == 75


def test_frame_to_ppm_bytes_native_and_scaled():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # Canal vermelho em BGR

    ppm_native, size_native = frame_to_ppm_bytes(frame)
    assert size_native == (320, 240)
    assert ppm_native.startswith(b"P6 320 240 255\n")
    header_len = len(b"P6 320 240 255\n")
    assert len(ppm_native) == header_len + (320 * 240 * 3)
    assert ppm_native[header_len : header_len + 3] == bytes([200, 0, 0])

    ppm_scaled, size_scaled = frame_to_ppm_bytes(frame, max_size=(160, 120))
    assert size_scaled == (160, 120)
    assert ppm_scaled.startswith(b"P6 160 120 255\n")


def test_frame_to_ppm_bytes_rejects_invalid_frame():
    with pytest.raises(ValueError, match="Quadro inválido"):
        frame_to_ppm_bytes(np.array([], dtype=np.uint8))


def test_canvas_rect_to_roi_converts_and_validates():
    roi = canvas_rect_to_roi(
        x0=80,
        y0=100,
        x1=480,
        y1=350,
        offset_x=0,
        offset_y=50,
        drawn_w=800,
        drawn_h=500,
    )
    assert roi is not None
    assert roi.enabled is True
    assert roi.x == pytest.approx(0.10, abs=0.02)
    assert roi.y == pytest.approx(0.10, abs=0.02)
    assert roi.w == pytest.approx(0.50, abs=0.02)
    assert roi.h == pytest.approx(0.50, abs=0.02)

    # Clique muito pequeno (< 12px) ou dimensões inválidas retornam None
    assert canvas_rect_to_roi(100, 100, 104, 105, 0, 0, 800, 600) is None
    assert canvas_rect_to_roi(10, 10, 100, 100, 0, 0, 0, 600) is None

    # Arraste inteiramente dentro das barras pretas (letterbox/pillarbox) retorna None
    assert canvas_rect_to_roi(10, 10, 80, 50, offset_x=100, offset_y=60, drawn_w=600, drawn_h=450) is None

    # Arraste na borda direita/inferior (> 0.90 até < 0.95) preserva coordenadas válidas
    edge_roi = canvas_rect_to_roi(740, 460, 795, 495, offset_x=0, offset_y=0, drawn_w=800, drawn_h=500)
    assert edge_roi is not None
    assert edge_roi.x == pytest.approx(0.925, abs=0.01)
    assert edge_roi.x + edge_roi.w <= 1.0001


def test_desktop_app_full_controls_and_telemetry(tmp_path: Path):
    cfg = SystemConfig(
        source="simulator",
        width=640,
        height=480,
        target_fps=24,
        vision_mode="normal",
        thermal=ThermalConfig(min_temp_c=20.0, max_temp_c=120.0, overheat_threshold_c=105.0),
    )
    manager = CameraManager(cfg)
    app = IndustrialVisionDesktopApp(
        manager=manager,
        snapshot_dir=tmp_path / "snapshots",
        start_hidden=True,
    )
    try:
        # 1. Passo inicial de quadro e telemetria
        tel = app.step_once()
        assert tel.frame_index == 1
        assert "640 x 480" in app.kpi_resolution_var.get()
        assert "STATUS:" in app.badge_alert_var.get()
        assert "NORMAL" in app.badge_alert_var.get()

        # 2. Alternar modos de visão
        for mode in ("thermal", "hybrid", "motion_mask", "normal"):
            assert app.set_vision_mode(mode) is True
            assert manager.get_config().vision_mode == mode

        # 3. Alterar resolução por preset e customizada
        app.res_preset_var.set("1280x720")
        assert app.apply_resolution_preset() is True
        assert (manager.get_config().width, manager.get_config().height) == (1280, 720)

        app.width_var.set("800")
        app.height_var.set("600")
        assert app.apply_custom_resolution() is True
        assert (manager.get_config().width, manager.get_config().height) == (800, 600)

        # 4. Alterar FPS, velocidade da simulação e HUD
        assert app.set_fps(30) is True
        assert manager.get_config().target_fps == 30

        app.sim_speed_var.set(1.8)
        assert app.apply_sim_speed() is True
        assert manager.get_config().simulator.speed_multiplier == pytest.approx(1.8)

        app.hud_enabled_var.set(False)
        assert app.apply_hud_setting() is True
        assert manager.get_config().hud_enabled is False

        # Testar espelhamento da câmera pessoal, perfis térmicos e reconexão
        assert app.toggle_mirror() is True
        assert manager.get_config().mirror is False
        assert app.apply_thermal_profile("human") is True
        assert manager.get_config().thermal.max_temp_c == 42.0
        assert app.apply_thermal_profile("industrial") is True
        assert manager.get_config().thermal.max_temp_c == 120.0
        app.reconnect_camera()

        # 5. Alterar cenário e fonte
        app.scenario_var.set("boiler")
        assert app.apply_source_and_scenario() is True
        assert manager.get_config().simulator.scenario == "boiler"

        # 6. Configurações térmicas
        app.palette_var.set("turbo")
        app.temp_min_var.set("25")
        app.temp_max_var.set("150")
        app.temp_alarm_var.set("95")
        app.kinetic_heat_var.set(60)
        app.show_hotspots_var.set(False)
        app.show_isotherms_var.set(True)
        app.show_scale_var.set(False)
        assert app.apply_thermal_settings() is True

        t_cfg = manager.get_config().thermal
        assert t_cfg.palette == "turbo"
        assert t_cfg.min_temp_c == 25.0
        assert t_cfg.max_temp_c == 150.0
        assert t_cfg.overheat_threshold_c == 95.0
        assert t_cfg.kinetic_heat_gain == pytest.approx(0.60)
        assert t_cfg.show_hotspots is False
        assert t_cfg.show_scale_bar is False

        # Reduzir Temp. Máx para 70°C sem editar Alarme (95°C) deve ajustar o alarme automaticamente para 70°C
        app.temp_max_var.set("70")
        assert app.apply_thermal_settings() is True
        assert manager.get_config().thermal.max_temp_c == 70.0
        assert manager.get_config().thermal.overheat_threshold_c == 70.0
        assert float(app.temp_alarm_var.get()) == 70.0

        # 7. Configurações de movimento e ROI + seleção por mouse no Canvas
        app.algorithm_var.set("mog2")
        app.sensitivity_var.set(75)
        app.min_area_var.set("400")
        app.alarm_pct_var.set("5.0")
        app.show_boxes_var.set(True)
        app.show_trails_var.set(False)
        app.roi_enabled_var.set(True)
        assert app.apply_motion_settings() is True

        m_cfg = manager.get_config().motion
        assert m_cfg.algorithm == "mog2"
        assert m_cfg.sensitivity == 75
        assert m_cfg.min_area == 400
        assert m_cfg.alarm_motion_pct == pytest.approx(5.0)
        assert m_cfg.show_trails is False
        assert manager.get_config().roi.enabled is True

        # Simula arraste de mouse no canvas para desenhar ROI
        app._viewport_rect = (640, 480, 0, 0)
        app._on_canvas_press(SimpleNamespace(x=64, y=48))
        app._on_canvas_drag(SimpleNamespace(x=320, y=240))
        app._on_canvas_release(SimpleNamespace(x=384, y=288))
        assert manager.get_config().roi.enabled is True
        assert manager.get_config().roi.x == pytest.approx(0.10, abs=0.02)

        assert app.reset_roi() is True
        assert manager.get_config().roi.enabled is False

        # 8. Injeção de anomalias, snapshot, pausa e recalibração
        app.inject_anomaly("thermal_spike")
        app.step_once()
        saved_path = app.save_snapshot()
        assert saved_path.exists()
        assert saved_path.stat().st_size > 500

        app.toggle_pause()
        assert app.is_paused is True
        app.toggle_pause()
        assert app.is_paused is False

        app.recalibrate()
        app.clear_events()
        assert len(manager.get_events()) == 0

        # 9. Entradas inválidas não devem quebrar a aplicação
        app.width_var.set("invalido")
        assert app.apply_custom_resolution() is False
        assert "Erro" in app.status_message_var.get()

        app.camera_index_var.set("abc")
        assert app.apply_source_and_scenario() is False

        app.temp_min_var.set("nao_numero")
        assert app.apply_thermal_settings() is False

        app.min_area_var.set("xyz")
        assert app.apply_motion_settings() is False

        # Validação Pydantic (ex: largura fora do limite)
        app.width_var.set("99999")
        app.height_var.set("480")
        assert app.apply_custom_resolution() is False
        assert "Erro de validação" in app.status_message_var.get()
    finally:
        app.close()


def test_run_native_app_with_max_frames(tmp_path: Path):
    cfg = SystemConfig(source="simulator", width=320, height=240, target_fps=30)
    frames_done = run_native_app(
        config=cfg,
        max_frames=3,
        start_hidden=True,
        snapshot_dir=tmp_path / "snaps",
    )
    assert frames_done == 3


def test_cli_app_flag_and_desktop_launcher_entrypoint(monkeypatch):
    called = {"native": 0, "gui_max": None}
    monkeypatch.setattr(
        cli_mod,
        "run_native_app",
        lambda config, max_frames=None, **_kw: called.__setitem__("native", called["native"] + 1) or 0,
    )
    monkeypatch.setattr(
        cli_mod,
        "run_desktop_gui",
        lambda config, max_frames=None: called.__setitem__("gui_max", max_frames),
    )
    args = parse_args(["--app", "--resolution", "640x480", "--max-frames", "5"])
    assert args.app is True
    assert args.max_frames == 5
    assert main(["--app", "--resolution", "640x480", "--max-frames", "5"]) == 0
    assert desktop_launcher.launch_desktop(["--resolution", "320x240", "--max-frames", "2"]) == 0
    assert called["native"] == 2

    # --desktop-gui também deve respeitar --max-frames e não ser sobrescrito por launch_desktop
    assert desktop_launcher.launch_desktop(["--desktop-gui", "--max-frames", "4"]) == 0
    assert called["gui_max"] == 4


def test_entry_widgets_do_not_trigger_global_shortcuts_and_canvas_resize(tmp_path: Path):
    manager = CameraManager(SystemConfig(width=640, height=480, vision_mode="normal"))
    app = IndustrialVisionDesktopApp(
        manager=manager,
        snapshot_dir=tmp_path / "snaps",
        start_hidden=True,
    )
    try:
        app.step_once()
        # Simula evento de tecla vindo de um widget Entry: não deve trocar modo nem apagar texto
        fake_entry = SimpleNamespace(winfo_class=lambda: "Entry")
        assert app._is_text_input_event(SimpleNamespace(widget=fake_entry)) is True
        assert app._is_text_input_event(SimpleNamespace(widget=app.canvas)) is False

        app.width_var.set("1280")
        for cb_key in ("1", "2", "3", "4", "t", "i"):
            # Quando o foco está em Entry, o atalho deve ser ignorado
            if not app._is_text_input_event(SimpleNamespace(widget=fake_entry)):
                app.set_vision_mode("thermal")
        assert manager.get_config().vision_mode == "normal"
        assert app.width_var.get() == "1280"

        # Redimensionamento para proporção extrema (ex: altura 160 < 240) deve ajustar viewport sem cortar
        app._on_canvas_configure(SimpleNamespace(width=800, height=160))
        drawn_w, drawn_h, ox, oy = app._viewport_rect
        assert drawn_h == 160
        assert drawn_w == 213
        assert ox == (800 - 213) // 2
        assert oy == 0

        # Durante arraste de ROI ativo, renderizar novo quadro mantém o retângulo vivo
        app._on_canvas_press(SimpleNamespace(x=300, y=20))
        app._on_canvas_drag(SimpleNamespace(x=450, y=140))
        assert app._drag_rect_id is not None
        app.step_once()
        app._on_canvas_release(SimpleNamespace(x=450, y=140))
        assert manager.get_config().roi.enabled is True
    finally:
        app.close()


def test_headless_display_error_closes_manager_and_returns_clean_error(monkeypatch, capsys):
    import tkinter as tk

    import app.desktop_app as desktop_mod

    closed = {"count": 0}
    orig_close = CameraManager.close

    def tracked_close(self):
        closed["count"] += 1
        return orig_close(self)

    monkeypatch.setattr(CameraManager, "close", tracked_close)
    monkeypatch.setattr(
        desktop_mod.tk,
        "Tk",
        lambda *a, **kw: (_ for _ in ()).throw(tk.TclError("no display name and no $DISPLAY")),
    )

    with pytest.raises(RuntimeError, match="Ambiente gráfico"):
        run_native_app(SystemConfig(), max_frames=1)
    assert closed["count"] == 1

    rc = main(["--app", "--max-frames", "1"])
    assert rc == 1
    err_out = capsys.readouterr().err
    assert "Erro ao iniciar aplicativo desktop" in err_out


def test_throttled_step_refreshes_video_every_frame_but_panels_on_interval(tmp_path: Path, monkeypatch):
    import tkinter as tk

    import app.desktop_app as desktop_module

    manager = CameraManager(SystemConfig(source="simulator", width=320, height=240))
    try:
        app = IndustrialVisionDesktopApp(manager=manager, snapshot_dir=tmp_path, start_hidden=True)
    except tk.TclError:
        manager.close()
        pytest.skip("Ambiente gráfico indisponível")
    try:
        panel_updates: list[int] = []
        original_update = app._update_telemetry_widgets
        monkeypatch.setattr(
            app,
            "_update_telemetry_widgets",
            lambda tel: (panel_updates.append(tel.frame_index), original_update(tel))[1],
        )
        monkeypatch.setattr(desktop_module, "TELEMETRY_REFRESH_SEC", 60.0)
        monkeypatch.setattr(desktop_module, "EVENTS_REFRESH_SEC", 60.0)

        first = app.step_once(throttled=True)
        video_item = app._video_item_id
        for _ in range(5):
            app.step_once(throttled=True)

        assert panel_updates == [first.frame_index]
        assert app._video_item_id == video_item
        assert len(app.canvas.find_withtag("video_frame")) == 1

        unthrottled = app.step_once()
        assert panel_updates[-1] == unthrottled.frame_index
    finally:
        app.close()


def test_lite_mode_button_applies_low_cost_profile(tmp_path: Path):
    import tkinter as tk

    manager = CameraManager(SystemConfig(source="simulator", width=640, height=480, target_fps=30))
    try:
        app = IndustrialVisionDesktopApp(manager=manager, snapshot_dir=tmp_path, start_hidden=True)
    except tk.TclError:
        manager.close()
        pytest.skip("Ambiente gráfico indisponível")
    try:
        assert app.apply_lite_mode() is True

        cfg = manager.get_config()
        assert (cfg.width, cfg.height, cfg.target_fps) == (320, 240, 15)
        assert cfg.motion.show_trails is False
        assert app.res_preset_var.get() == "320x240"
    finally:
        app.close()
