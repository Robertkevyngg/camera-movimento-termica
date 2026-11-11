"""Interface de linha de comando (CLI), modo benchmark e janela OpenCV local."""

from __future__ import annotations

import argparse
import contextlib
import os
import secrets
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
from app.persistence import (
    CONFIG_FILE_NAME,
    RECORDINGS_DIR_NAME,
    EVENTS_FILE_NAME,
    EventLog,
    get_data_dir,
    load_config,
    save_config,
)
from app.thermal_simulator import PALETTE_MAP


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
API_TOKEN_ENV = "VISAO_API_TOKEN"
GENERATED_TOKEN_BYTES = 24


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema de Visão Industrial — Detecção de Movimento & Termografia Simulada",
        allow_abbrev=False,
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
        "--record",
        action="store_true",
        help="Grava em vídeo cada evento de movimento (câmera física; até 1080p a 60 FPS)",
    )
    parser.add_argument(
        "--recordings-dir",
        default="",
        help="Pasta dos vídeos gravados (padrão: subpasta 'gravacoes' da pasta de dados)",
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
        "--no-persist",
        action="store_true",
        help="Não carrega nem salva a configuração e o histórico de eventos em disco",
    )
    parser.add_argument(
        "--api-token",
        default="",
        help=(
            "Token exigido pelo painel Web (ou variável VISAO_API_TOKEN). "
            "Fora de 127.0.0.1, um token é gerado automaticamente se nenhum for informado"
        ),
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
        record_events=getattr(args, "record", False),
        vision_mode=args.mode,
        thermal={"palette": args.palette},
        simulator={"scenario": args.scenario},
    )
    return base.apply_patch(LITE_MODE_PATCH) if getattr(args, "lite", False) else base


def _flag_given(argv: Sequence[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def build_explicit_patch(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    """Patch só com as opções que o usuário digitou, para sobrepor a configuração salva."""
    flag_patches: tuple[tuple[str, dict[str, Any]], ...] = (
        ("--source", {"source": args.source}),
        ("--camera-index", {"camera_index": args.camera_index}),
        ("--no-mirror", {"mirror": False}),
        ("--resolution", {"resolution_preset": args.resolution}),
        ("--fps", {"target_fps": args.fps}),
        ("--record", {"record_events": True}),
        ("--mode", {"vision_mode": args.mode}),
        ("--palette", {"thermal": {"palette": args.palette}}),
        ("--scenario", {"simulator": {"scenario": args.scenario}}),
        ("--lite", LITE_MODE_PATCH),
    )
    patch: dict[str, Any] = {}
    for flag, flag_patch in flag_patches:
        if _flag_given(argv, flag):
            patch = {**patch, **flag_patch}
    return patch


def resolve_startup_config(
    args: argparse.Namespace, argv: Sequence[str], config_path: Path | None
) -> SystemConfig:
    """Configuração inicial: a salva em disco (se houver) com as opções digitadas por cima."""
    saved = load_config(config_path) if config_path is not None else None
    if saved is None:
        return build_initial_config(args)
    return saved.apply_patch(build_explicit_patch(args, argv))


def resolve_recordings_dir(args: argparse.Namespace) -> Path:
    """Pasta dos vídeos de eventos: a informada ou `gravacoes` na pasta de dados."""
    if args.recordings_dir:
        return Path(args.recordings_dir).expanduser()
    return get_data_dir() / RECORDINGS_DIR_NAME


def resolve_api_token(args: argparse.Namespace) -> str | None:
    """Token do painel Web: o informado, ou um gerado quando o host é acessível pela rede."""
    token = args.api_token or os.environ.get(API_TOKEN_ENV, "")
    if token:
        return token
    if args.host in LOOPBACK_HOSTS:
        return None
    return secrets.token_urlsafe(GENERATED_TOKEN_BYTES)


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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    if args.benchmark_frames > 0:
        cfg = build_initial_config(args)
        snap = Path(args.save_snapshot) if args.save_snapshot else None
        summary = run_benchmark(cfg, frames=args.benchmark_frames, snapshot_path=snap)
        print("=== Benchmark do Sistema de Visão Industrial ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    max_f = args.max_frames if args.max_frames > 0 else None
    data_dir = None if args.no_persist else get_data_dir()
    config_path = data_dir / CONFIG_FILE_NAME if data_dir is not None else None
    event_sink = EventLog(data_dir / EVENTS_FILE_NAME).append if data_dir is not None else None
    cfg = resolve_startup_config(args, raw_argv, config_path)
    recordings_dir = resolve_recordings_dir(args)
    if cfg.record_events:
        print(f"Vídeos dos movimentos em: {recordings_dir}", flush=True)

    def persist_config(final_cfg: SystemConfig) -> None:
        if config_path is not None:
            save_config(final_cfg, config_path)

    if args.app:
        try:
            run_native_app(
                cfg,
                max_frames=max_f,
                on_exit=persist_config,
                event_sink=event_sink,
                recordings_dir=recordings_dir,
            )
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

    manager = CameraManager(cfg, event_sink=event_sink, recordings_dir=recordings_dir)
    api_token = resolve_api_token(args)
    if api_token and not (args.api_token or os.environ.get(API_TOKEN_ENV)):
        print(
            "Painel acessível pela rede: acesso protegido por token gerado.\n"
            f"  Abra: http://{args.host}:{args.port}/?token={api_token}",
            flush=True,
        )
    fastapi_app = create_app(manager, api_token=api_token)
    try:
        # Com token, o log de acesso registraria a URL "?token=..." da primeira visita.
        uvicorn.run(
            fastapi_app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=api_token is None,
        )
    finally:
        persist_config(manager.get_config())
        manager.close()
    return 0
