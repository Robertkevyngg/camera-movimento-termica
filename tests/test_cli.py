"""Testes unitários para a interface de linha de comando (CLI) e benchmark."""

from pathlib import Path

import cv2
import uvicorn

import app.cli as cli_mod
from app.cli import (
    build_initial_config,
    main,
    parse_args,
    run_benchmark,
    run_desktop_gui,
)
from app.config import SystemConfig


def test_cli_parse_args_and_build_initial_config():
    args = parse_args(
        [
            "--source",
            "simulator",
            "--resolution",
            "1280x720",
            "--fps",
            "30",
            "--mode",
            "thermal",
            "--palette",
            "turbo",
            "--scenario",
            "boiler",
        ]
    )
    cfg = build_initial_config(args)
    assert cfg.source == "simulator"
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.target_fps == 30
    assert cfg.vision_mode == "thermal"
    assert cfg.thermal.palette == "turbo"
    assert cfg.simulator.scenario == "boiler"


def test_cli_run_benchmark_executes_and_saves_snapshot(tmp_path: Path):
    snap_path = tmp_path / "bench_snap.jpg"
    args = parse_args(
        [
            "--source",
            "simulator",
            "--resolution",
            "640x480",
            "--fps",
            "24",
            "--mode",
            "hybrid",
            "--benchmark-frames",
            "5",
            "--save-snapshot",
            str(snap_path),
        ]
    )
    cfg = build_initial_config(args)
    summary = run_benchmark(cfg, frames=args.benchmark_frames, snapshot_path=snap_path)

    assert summary["frames_processed"] == 5
    assert summary["width"] == 640
    assert summary["height"] == 480
    assert snap_path.exists()
    assert snap_path.stat().st_size > 500


def test_cli_main_benchmark_mode_returns_zero(capsys, tmp_path: Path):
    snap = tmp_path / "out.jpg"
    rc = main(
        [
            "--benchmark-frames",
            "4",
            "--resolution",
            "320x240",
            "--fps",
            "15",
            "--mode",
            "thermal",
            "--save-snapshot",
            str(snap),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Benchmark do Sistema de Visão Industrial" in captured.out
    assert snap.exists()


def test_cli_run_desktop_gui_keyboard_controls_and_exit(monkeypatch):
    shown_frames = []
    keys = iter(
        [
            ord("1"),
            ord("2"),
            ord("3"),
            ord("4"),
            ord("R"),
            ord("+"),
            ord("-"),
            ord("P"),
            ord("T"),
            ord("I"),
            ord("C"),
            ord("Q"),
        ]
    )

    monkeypatch.setattr(cv2, "imshow", lambda win, frame: shown_frames.append(frame.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys, ord("q")))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda win, prop: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    cfg = SystemConfig(source="simulator", width=640, height=480, target_fps=30)
    run_desktop_gui(cfg)
    assert len(shown_frames) == 12
    # Após pressionar 'R' no 5º quadro, a resolução deve ter mudado de 640x480 para 800x600
    assert shown_frames[0] == (480, 640, 3)
    assert shown_frames[5] == (600, 800, 3)


def test_cli_main_desktop_gui_and_server_modes(monkeypatch):
    called = {"gui": False, "uvicorn": False}
    monkeypatch.setattr(cli_mod, "run_desktop_gui", lambda cfg: called.__setitem__("gui", True))
    monkeypatch.setattr(
        uvicorn, "run", lambda app, host, port, log_level: called.__setitem__("uvicorn", True)
    )

    assert main(["--desktop-gui", "--resolution", "320x240"]) == 0
    assert called["gui"] is True

    assert main(["--host", "127.0.0.1", "--port", "8765"]) == 0
    assert called["uvicorn"] is True
