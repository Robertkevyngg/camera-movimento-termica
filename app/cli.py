"""Interface de linha de comando (CLI), modo benchmark e janela OpenCV local."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import uvicorn

from app.api import create_app
from app.camera_manager import CameraManager
from app.config import LITE_MODE_PATCH, RESOLUTION_PRESETS, SystemConfig
from app.desktop_app import run_native_app
from app.thermal_simulator import PALETTE_MAP


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema de Visão Industrial — Detecção de Movimento & Termografia Simulada"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host para o servidor Web SCADA (padrão: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Porta HTTP para o painel Web SCADA (padrão: 8000)"
    )
    parser.add_argument(
        "--source",
        choices=["camera", "simulator"],
        default="camera",
        help="Fonte inicial de captura: 'camera' (câmera pessoal) ou 'simulator'",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Índice da câmera física V4L2/USB (padrão: 0)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Desativa o espelhamento horizontal da câmera pessoal",
    )
    parser.add_argument(
        "--resolution",
        choices=list(RESOLUTION_PRESETS.keys()),
        default="640x480",
        help="Preset de resolução inicial",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Taxa de quadros alvo inicial (1-60 FPS)"
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "thermal", "hybrid", "motion_mask"],
        default="normal",
        help="Modo de visão inicial",
    )
    parser.add_argument(
        "--palette",
        choices=list(PALETTE_MAP.keys()),
        default="inferno",
        help="Paleta da câmera térmica simulada",
    )
    parser.add_argument(
        "--scenario",
        choices=["conveyor", "machinery", "perimeter", "boiler"],
        default="conveyor",
        help="Cenário do simulador industrial",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=0,
        help="Se > 0, executa N quadros em modo de teste/benchmark e encerra",
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default="",
        help="Caminho opcional para salvar um snapshot JPEG após o benchmark",
    )
    parser.add_argument(
        "--desktop-gui",
        action="store_true",
        help="Abre janela simples OpenCV Desktop em vez do servidor Web SCADA",
    )
    parser.add_argument(
        "--app",
        "--local-app",
        dest="app",
        action="store_true",
        help="Abre o aplicativo Desktop local completo (interface nativa SCADA Tkinter)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Modo leve para computadores modestos (320x240 @ 15 FPS, sem trilhas)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limite opcional de quadros para encerramento automático em modo GUI/App",
    )
    return parser.parse_args(argv)


def build_initial_config(args: argparse.Namespace) -> SystemConfig:
    """Constrói um SystemConfig imutável a partir dos argumentos CLI."""
    width, height = RESOLUTION_PRESETS[args.resolution]
    base = SystemConfig(
        source=args.source,
        camera_index=args.camera_index,
        mirror=not getattr(args, "no_mirror", False),
        width=width,
        height=height,
        target_fps=args.fps,
        vision_mode=args.mode,
        thermal={"palette": args.palette},
        simulator={"scenario": args.scenario},
    )
    return base.apply_patch(LITE_MODE_PATCH) if getattr(args, "lite", False) else base


def run_benchmark(
    config: SystemConfig, frames: int, snapshot_path: Path | None = None
) -> dict[str, Any]:
    """Executa N quadros sincronicamente para validação local e coleta métricas."""
    manager = CameraManager(config)
    try:
        latencies: list[float] = []
        max_temps: list[float] = []
        motion_pcts: list[float] = []
        last_tel = None

        for idx in range(max(1, frames)):
            if idx == max(1, frames // 2):
                manager.trigger_anomaly("thermal_spike", duration_frames=5)
            _, tel = manager.step_frame()
            latencies.append(tel.processing_ms)
            max_temps.append(tel.max_temp_c)
            motion_pcts.append(tel.motion_percentage)
            last_tel = tel

        if snapshot_path is not None:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(manager.get_snapshot_jpeg(quality=90))

        assert last_tel is not None
        return {
            "frames_processed": frames,
            "width": last_tel.width,
            "height": last_tel.height,
            "target_fps": last_tel.target_fps,
            "avg_processing_ms": round(sum(latencies) / len(latencies), 2),
            "peak_temp_c": round(max(max_temps), 1),
            "peak_motion_pct": round(max(motion_pcts), 2),
            "source_active": last_tel.source_active,
        }
    finally:
        manager.close()


def run_desktop_gui(
    config: SystemConfig, max_frames: int | None = None
) -> None:
    """Executa loop interativo em janela nativa OpenCV Desktop."""
    manager = CameraManager(config)
    presets = list(RESOLUTION_PRESETS.keys())
    palettes = list(PALETTE_MAP.keys())
    res_key = f"{config.width}x{config.height}"
    res_idx = presets.index(res_key) if res_key in presets else 1
    pal_idx = palettes.index(config.thermal.palette)
    win_name = "Visao Industrial Local (1-4: Modo | R: Resolucao | +/-: FPS | P: Paleta | T: Pico Termico | Q: Sair)"

    frames_shown = 0
    try:
        while True:
            t0 = time.perf_counter()
            frame, _ = manager.step_frame()
            cv2.imshow(win_name, frame)
            frames_shown += 1

            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((manager.get_frame_interval_sec() - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("1"):
                manager.update_config({"vision_mode": "normal"})
            elif key == ord("2"):
                manager.update_config({"vision_mode": "thermal"})
            elif key == ord("3"):
                manager.update_config({"vision_mode": "hybrid"})
            elif key == ord("4"):
                manager.update_config({"vision_mode": "motion_mask"})
            elif key in (ord("r"), ord("R")):
                res_idx = (res_idx + 1) % len(presets)
                manager.update_config({"resolution_preset": presets[res_idx]})
            elif key in (ord("+"), ord("=")):
                cur = manager.get_config().target_fps
                manager.update_config({"target_fps": min(60, cur + 5)})
            elif key in (ord("-"), ord("_")):
                cur = manager.get_config().target_fps
                manager.update_config({"target_fps": max(1, cur - 5)})
            elif key in (ord("p"), ord("P")):
                pal_idx = (pal_idx + 1) % len(palettes)
                manager.update_config({"thermal": {"palette": palettes[pal_idx]}})
            elif key in (ord("t"), ord("T")):
                manager.trigger_anomaly("thermal_spike", duration_frames=45)
            elif key in (ord("i"), ord("I")):
                manager.trigger_anomaly("intrusion", duration_frames=45)
            elif key in (ord("m"), ord("M")):
                cur_mirror = manager.get_config().mirror
                manager.update_config({"mirror": not cur_mirror})
            elif key in (ord("c"), ord("C")):
                cur_src = manager.get_config().source
                manager.update_config(
                    {"source": "camera" if cur_src == "simulator" else "simulator"}
                )

            if max_frames is not None and frames_shown >= max_frames:
                break
            with contextlib.suppress(cv2.error, RuntimeError):
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        manager.close()
        with contextlib.suppress(cv2.error, RuntimeError):
            cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal da aplicação."""
    args = parse_args(argv)
    cfg = build_initial_config(args)

    if args.benchmark_frames > 0:
        snap = Path(args.save_snapshot) if args.save_snapshot else None
        summary = run_benchmark(cfg, frames=args.benchmark_frames, snapshot_path=snap)
        print("=== Benchmark do Sistema de Visão Industrial ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    max_f = args.max_frames if args.max_frames > 0 else None

    if args.app:
        try:
            run_native_app(cfg, max_frames=max_f)
        except RuntimeError as exc:
            print(f"Erro ao iniciar aplicativo desktop: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.desktop_gui:
        if max_f is not None:
            run_desktop_gui(cfg, max_frames=max_f)
        else:
            run_desktop_gui(cfg)
        return 0

    manager = CameraManager(cfg)
    fastapi_app = create_app(manager)
    uvicorn.run(fastapi_app, host=args.host, port=args.port, log_level="info")
    return 0
