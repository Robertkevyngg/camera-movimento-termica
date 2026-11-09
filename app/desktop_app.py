"""Aplicativo Desktop Nativo (Tkinter SCADA) para o Sistema de Visão Industrial."""

from __future__ import annotations

import contextlib
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from app.camera_manager import CameraManager
from app.config import (
    FPS_PRESETS,
    RESOLUTION_PRESETS,
    EventRecord,
    FrameTelemetry,
    ROIConfig,
    SystemConfig,
)
from app.industrial_sim import AnomalyType
from app.thermal_simulator import PALETTE_MAP

BG_MAIN = "#0b0f17"
BG_CARD = "#111827"
BG_INPUT = "#1f2937"
FG_PRIMARY = "#f8fafc"
FG_MUTED = "#94a3b8"
ACCENT_BLUE = "#0284c7"
ACCENT_GREEN = "#059669"
ACCENT_WARN = "#d97706"
ACCENT_CRIT = "#dc2626"


def compute_aspect_fit(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int, int, int]:
    """Calcula dimensões e deslocamento (w, h, ox, oy) preservando a proporção."""
    safe_src_w, safe_src_h = max(1, int(src_w)), max(1, int(src_h))
    safe_max_w, safe_max_h = max(64, int(max_w)), max(64, int(max_h))
    scale = min(safe_max_w / safe_src_w, safe_max_h / safe_src_h)
    drawn_w, drawn_h = max(1, round(safe_src_w * scale)), max(1, round(safe_src_h * scale))
    return drawn_w, drawn_h, max(0, (safe_max_w - drawn_w) // 2), max(0, (safe_max_h - drawn_h) // 2)


def frame_to_ppm_bytes(
    frame_bgr: np.ndarray, max_size: tuple[int, int] | None = None
) -> tuple[bytes, tuple[int, int]]:
    """Converte um quadro BGR OpenCV para bytes PPM (P6) compatíveis com tk.PhotoImage."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        raise ValueError("Quadro inválido para conversão PPM")
    src_h, src_w = frame_bgr.shape[:2]
    target_bgr = frame_bgr
    if max_size is not None:
        drawn_w, drawn_h, _, _ = compute_aspect_fit(src_w, src_h, max_size[0], max_size[1])
        if (drawn_w, drawn_h) != (src_w, src_h):
            interp = cv2.INTER_AREA if drawn_w < src_w else cv2.INTER_LINEAR
            target_bgr = cv2.resize(frame_bgr, (drawn_w, drawn_h), interpolation=interp)
    out_h, out_w = target_bgr.shape[:2]
    rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    return f"P6 {out_w} {out_h} 255\n".encode("ascii") + rgb.tobytes(), (out_w, out_h)


def canvas_rect_to_roi(
    x0: float, y0: float, x1: float, y1: float, offset_x: int, offset_y: int, drawn_w: int, drawn_h: int
) -> ROIConfig | None:
    """Converte um retângulo arrastado no canvas em uma ROIConfig normalizada."""
    if drawn_w <= 0 or drawn_h <= 0 or abs(x1 - x0) < 12 or abs(y1 - y0) < 12:
        return None
    left, right = max(0.0, min(float(drawn_w), min(x0, x1) - offset_x)), max(0.0, min(float(drawn_w), max(x0, x1) - offset_x))
    top, bottom = max(0.0, min(float(drawn_h), min(y0, y1) - offset_y)), max(0.0, min(float(drawn_h), max(y0, y1) - offset_y))
    if (right - left) < 12.0 or (bottom - top) < 12.0:
        return None
    norm_w, norm_h = round(max(0.05, min(1.0, (right - left) / drawn_w)), 3), round(max(0.05, min(1.0, (bottom - top) / drawn_h)), 3)
    norm_x, norm_y = round(min(0.949, min(1.0 - norm_w, max(0.0, left / drawn_w))), 3), round(min(0.949, min(1.0 - norm_h, max(0.0, top / drawn_h))), 3)
    return ROIConfig(enabled=True, x=norm_x, y=norm_y, w=norm_w, h=norm_h)


def _make_btn(parent: tk.Widget, text: str, command: Any, bg: str = BG_INPUT, **kwargs: Any) -> tk.Button:
    """Cria um botão estilizado no padrão SCADA."""
    return tk.Button(parent, text=text, command=command, bg=bg, fg=FG_PRIMARY, activebackground=ACCENT_BLUE, activeforeground=FG_PRIMARY, relief=tk.FLAT, **kwargs)


def _make_entry(parent: tk.Widget, var: tk.Variable, width: int, on_enter: Any) -> tk.Entry:
    """Cria um campo de entrada estilizado que aplica alterações ao pressionar Enter."""
    entry = tk.Entry(parent, textvariable=var, width=width, bg=BG_INPUT, fg=FG_PRIMARY)
    entry.bind("<Return>", lambda _e: on_enter())
    return entry


class IndustrialVisionDesktopApp:
    """Aplicação Desktop Local nativa para captura da câmera pessoal, movimento e simulação térmica."""

    def __init__(self, manager: CameraManager | None = None, snapshot_dir: Path | None = None, start_hidden: bool = False) -> None:
        self.manager = manager or CameraManager()
        self.snapshot_dir = snapshot_dir or (Path.cwd() / "snapshots")
        self.is_paused = False
        self._closed = False
        self._after_id: str | None = None
        self._photo_ref: tk.PhotoImage | None = None
        self._last_frame_bgr: np.ndarray | None = None
        self._last_events_sig: tuple[int, ...] | None = None
        self._viewport_rect: tuple[int, int, int, int] = (640, 480, 0, 0)
        self._drag_start: tuple[float, float] | None = None
        self._drag_rect_id: int | None = None
        self.root = tk.Tk()
        if start_hidden:
            self.root.withdraw()
        self._configure_window()
        self._init_variables()
        self._build_layout()
        self._bind_shortcuts()
        self.sync_variables_from_config(self.manager.get_config())

    def _configure_window(self) -> None:
        self.root.title("Câmera Pessoal — Análise de Movimento & Simulador de Câmera Térmica")
        self.root.geometry("1360x860")
        self.root.minsize(1100, 720)
        self.root.configure(bg=BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self.root)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_INPUT, foreground=FG_PRIMARY, padding=(10, 6), font=("DejaVu Sans", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", ACCENT_BLUE)])

    def _init_variables(self) -> None:
        cfg = self.manager.get_config()
        dev_lbl = self.manager.get_camera_device_label(cfg.camera_index)
        self.badge_source_var = tk.StringVar(master=self.root, value=f"FONTE: CÂMERA PESSOAL (#{cfg.camera_index})")
        self.badge_alert_var = tk.StringVar(master=self.root, value="STATUS: NORMAL")
        self.status_message_var = tk.StringVar(
            master=self.root,
            value=f"Câmera pessoal {dev_lbl} pronta — Mova-se em frente à câmera ou arraste no vídeo para Zona ROI",
        )
        self.stream_meta_var = tk.StringVar(
            master=self.root, value=f"{cfg.width}x{cfg.height} @ {cfg.target_fps} FPS"
        )
        self.pause_btn_var = tk.StringVar(master=self.root, value="Pausar Vídeo")

        self.kpi_fps_var = tk.StringVar(master=self.root, value=f"{cfg.target_fps}.0 / {cfg.target_fps} FPS")
        self.kpi_latency_var = tk.StringVar(master=self.root, value="Processamento: 0.0 ms")
        self.kpi_motion_var = tk.StringVar(master=self.root, value="0.0%")
        self.kpi_objects_var = tk.StringVar(master=self.root, value="0 objeto(s) em movimento")
        self.kpi_temp_max_var = tk.StringVar(master=self.root, value="22.0 °C")
        self.kpi_temp_range_var = tk.StringVar(master=self.root, value="Mín: 22.0°C | Méd: 22.0°C")
        self.kpi_resolution_var = tk.StringVar(master=self.root, value=f"{cfg.width} x {cfg.height}")
        self.kpi_coords_var = tk.StringVar(master=self.root, value="Hotspot: (0, 0)")

        self.camera_device_var = tk.StringVar(master=self.root, value=dev_lbl)
        self.source_var = tk.StringVar(master=self.root, value=cfg.source)
        self.scenario_var = tk.StringVar(master=self.root, value=cfg.simulator.scenario)
        self.camera_index_var = tk.StringVar(master=self.root, value=str(cfg.camera_index))
        self.mirror_var = tk.BooleanVar(master=self.root, value=cfg.mirror)
        self.res_preset_var = tk.StringVar(master=self.root, value=f"{cfg.width}x{cfg.height}")
        self.width_var = tk.StringVar(master=self.root, value=str(cfg.width))
        self.height_var = tk.StringVar(master=self.root, value=str(cfg.height))
        self.fps_var = tk.IntVar(master=self.root, value=cfg.target_fps)
        self.fps_label_var = tk.StringVar(master=self.root, value=f"{cfg.target_fps} FPS")
        self.sim_speed_var = tk.DoubleVar(master=self.root, value=cfg.simulator.speed_multiplier)
        self.sim_speed_label_var = tk.StringVar(master=self.root, value=f"{cfg.simulator.speed_multiplier:.1f}x")
        self.hud_enabled_var = tk.BooleanVar(master=self.root, value=cfg.hud_enabled)

        self.palette_var = tk.StringVar(master=self.root, value=cfg.thermal.palette)
        self.temp_min_var = tk.StringVar(master=self.root, value=str(cfg.thermal.min_temp_c))
        self.temp_max_var = tk.StringVar(master=self.root, value=str(cfg.thermal.max_temp_c))
        self.temp_alarm_var = tk.StringVar(master=self.root, value=str(cfg.thermal.overheat_threshold_c))
        kin_pct = round(cfg.thermal.kinetic_heat_gain * 100)
        self.kinetic_heat_var = tk.IntVar(master=self.root, value=kin_pct)
        self.kinetic_label_var = tk.StringVar(master=self.root, value=f"{kin_pct}%")
        self.show_hotspots_var = tk.BooleanVar(master=self.root, value=cfg.thermal.show_hotspots)
        self.show_isotherms_var = tk.BooleanVar(master=self.root, value=cfg.thermal.show_isotherms)
        self.show_scale_var = tk.BooleanVar(master=self.root, value=cfg.thermal.show_scale_bar)

        self.algorithm_var = tk.StringVar(master=self.root, value=cfg.motion.algorithm)
        self.sensitivity_var = tk.IntVar(master=self.root, value=cfg.motion.sensitivity)
        self.sensitivity_label_var = tk.StringVar(master=self.root, value=f"{cfg.motion.sensitivity}%")
        self.min_area_var = tk.StringVar(master=self.root, value=str(cfg.motion.min_area))
        self.alarm_pct_var = tk.StringVar(master=self.root, value=str(cfg.motion.alarm_motion_pct))
        self.show_boxes_var = tk.BooleanVar(master=self.root, value=cfg.motion.show_boxes)
        self.show_trails_var = tk.BooleanVar(master=self.root, value=cfg.motion.show_trails)
        self.roi_enabled_var = tk.BooleanVar(master=self.root, value=cfg.roi.enabled)

    def _build_layout(self) -> None:
        self._build_topbar()
        main_body = tk.Frame(self.root, bg=BG_MAIN)
        main_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        main_body.columnconfigure(0, weight=7)
        main_body.columnconfigure(1, weight=4)
        main_body.rowconfigure(0, weight=1)
        self._build_viewport_column(main_body)
        self._build_controls_column(main_body)

    def _build_topbar(self) -> None:
        topbar = tk.Frame(self.root, bg=BG_CARD, padx=14, pady=10)
        topbar.pack(fill=tk.X, padx=12, pady=(10, 4))
        title_box = tk.Frame(topbar, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)
        tk.Label(
            title_box, text="CÂMERA PESSOAL — ANÁLISE DE MOVIMENTO & SIMULADOR TÉRMICO", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 13, "bold")
        ).pack(anchor="w")
        tk.Label(title_box, textvariable=self.status_message_var, bg=BG_CARD, fg=FG_MUTED, font=("DejaVu Sans", 9)).pack(anchor="w")

        actions = tk.Frame(topbar, bg=BG_CARD)
        actions.pack(side=tk.RIGHT)
        self.lbl_badge_source = tk.Label(actions, textvariable=self.badge_source_var, bg=BG_INPUT, fg="#38bdf8", padx=10, pady=5, font=("DejaVu Sans", 9, "bold"))
        self.lbl_badge_source.pack(side=tk.LEFT, padx=4)
        self.lbl_badge_alert = tk.Label(actions, textvariable=self.badge_alert_var, bg=ACCENT_GREEN, fg=FG_PRIMARY, padx=10, pady=5, font=("DejaVu Sans", 9, "bold"))
        self.lbl_badge_alert.pack(side=tk.LEFT, padx=4)
        tk.Button(actions, textvariable=self.pause_btn_var, command=self.toggle_pause, bg=BG_INPUT, fg=FG_PRIMARY, relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        _make_btn(actions, "Recalibrar Fundo", self.recalibrate, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        _make_btn(actions, "Salvar Foto JPEG", self.save_snapshot, bg=ACCENT_BLUE, padx=12, pady=5, font=("DejaVu Sans", 9, "bold")).pack(side=tk.LEFT, padx=4)

    def _build_viewport_column(self, parent: tk.Frame) -> None:
        col = tk.Frame(parent, bg=BG_MAIN)
        col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        col.rowconfigure(0, weight=5)
        col.rowconfigure(2, weight=2)
        col.columnconfigure(0, weight=1)

        video_card = tk.Frame(col, bg=BG_CARD, padx=10, pady=8)
        video_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        video_card.rowconfigure(1, weight=1)
        video_card.columnconfigure(0, weight=1)

        mode_bar = tk.Frame(video_card, bg=BG_CARD)
        mode_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.mode_buttons: dict[str, tk.Button] = {}
        for key, label in [
            ("normal", "1. Câmera Pessoal (Normal)"),
            ("thermal", "2. Simular Câmera Térmica"),
            ("hybrid", "3. Fusão (Câmera + Térmica)"),
            ("motion_mask", "4. Rastreador de Movimento"),
        ]:
            btn = _make_btn(mode_bar, label, lambda m=key: self.set_vision_mode(m), padx=10, pady=4, font=("DejaVu Sans", 9, "bold"))
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self.mode_buttons[key] = btn
        tk.Label(mode_bar, textvariable=self.stream_meta_var, bg=BG_CARD, fg="#38bdf8", font=("DejaVu Sans Mono", 9, "bold")).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(video_card, bg="#05070b", highlightthickness=1, highlightbackground="#1e293b", takefocus=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        quick_bar = tk.Frame(video_card, bg=BG_CARD, pady=4)
        quick_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        tk.Label(quick_bar, text="Ações da Câmera Pessoal:", bg=BG_CARD, fg=FG_MUTED, font=("DejaVu Sans", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        _make_btn(quick_bar, "Simular Câmera Térmica (2)", lambda: self.set_vision_mode("thermal"), bg=ACCENT_CRIT, padx=8, pady=3).pack(side=tk.LEFT, padx=3)
        _make_btn(quick_bar, "Fusão Térmica + Movimento (3)", lambda: self.set_vision_mode("hybrid"), bg=ACCENT_WARN, padx=8, pady=3).pack(side=tk.LEFT, padx=3)
        _make_btn(quick_bar, "Espelhar Webcam (M)", self.toggle_mirror, padx=8, pady=3).pack(side=tk.LEFT, padx=3)
        _make_btn(quick_bar, "Perfil Corporal (20–42°C)", lambda: self.apply_thermal_profile("human"), bg=ACCENT_BLUE, padx=8, pady=3).pack(side=tk.LEFT, padx=3)

        self._build_kpi_row(col)
        self._build_events_panel(col)

    def _build_kpi_row(self, parent: tk.Frame) -> None:
        kpi_frame = tk.Frame(parent, bg=BG_MAIN)
        kpi_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        cards = [
            ("FPS da Câmera & Latência", self.kpi_fps_var, self.kpi_latency_var),
            ("Movimentos Capturados", self.kpi_motion_var, self.kpi_objects_var),
            ("Simulação Térmica (Hotspot)", self.kpi_temp_max_var, self.kpi_temp_range_var),
            ("Resolução & Ponto Quente", self.kpi_resolution_var, self.kpi_coords_var),
        ]
        for idx, (title, main_var, sub_var) in enumerate(cards):
            kpi_frame.columnconfigure(idx, weight=1)
            card = tk.Frame(kpi_frame, bg=BG_CARD, padx=10, pady=8)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 4, 0))
            tk.Label(card, text=title, bg=BG_CARD, fg=FG_MUTED, font=("DejaVu Sans", 8)).pack(anchor="w")
            tk.Label(card, textvariable=main_var, bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 12, "bold")).pack(anchor="w", pady=2)
            tk.Label(card, textvariable=sub_var, bg=BG_CARD, fg=FG_MUTED, font=("DejaVu Sans", 8)).pack(anchor="w")

    def _build_events_panel(self, parent: tk.Frame) -> None:
        ev_card = tk.Frame(parent, bg=BG_CARD, padx=10, pady=8)
        ev_card.grid(row=2, column=0, sticky="nsew")
        ev_card.rowconfigure(1, weight=1)
        ev_card.columnconfigure(0, weight=1)
        hdr = tk.Frame(ev_card, bg=BG_CARD)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        tk.Label(hdr, text="Registro de Movimentos Capturados e Alertas Térmicos", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 10, "bold")).pack(side=tk.LEFT)
        _make_btn(hdr, "Limpar Histórico", self.clear_events, padx=8, pady=2, font=("DejaVu Sans", 8)).pack(side=tk.RIGHT)
        self.events_listbox = tk.Listbox(
            ev_card, bg="#090d14", fg=FG_PRIMARY, selectbackground=ACCENT_BLUE, highlightthickness=0, borderwidth=0, font=("DejaVu Sans Mono", 9), height=6
        )
        self.events_listbox.grid(row=1, column=0, sticky="nsew")

    def _build_controls_column(self, parent: tk.Frame) -> None:
        right_col = tk.Frame(parent, bg=BG_CARD, padx=10, pady=10)
        right_col.grid(row=0, column=1, sticky="nsew")
        notebook = ttk.Notebook(right_col)
        notebook.pack(fill=tk.BOTH, expand=True)
        tab_cam = tk.Frame(notebook, bg=BG_CARD, padx=12, pady=12)
        tab_thermal = tk.Frame(notebook, bg=BG_CARD, padx=12, pady=12)
        tab_motion = tk.Frame(notebook, bg=BG_CARD, padx=12, pady=12)
        notebook.add(tab_cam, text="1. Câmera Pessoal & FPS")
        notebook.add(tab_thermal, text="2. Simulador Térmico")
        notebook.add(tab_motion, text="3. Movimento & ROI")
        self._build_camera_tab(tab_cam)
        self._build_thermal_tab(tab_thermal)
        self._build_motion_tab(tab_motion)

    def _build_camera_tab(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Câmera Pessoal, Resolução & FPS", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 6))
        tk.Label(parent, text="Câmera Pessoal Ativa:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        tk.Label(parent, textvariable=self.camera_device_var, bg=BG_INPUT, fg="#38bdf8", padx=8, pady=5, font=("DejaVu Sans Mono", 9, "bold")).pack(fill=tk.X, pady=(2, 8))

        cam_row = tk.Frame(parent, bg=BG_CARD)
        cam_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(cam_row, text="Índice (/dev/videoN):", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        _make_entry(cam_row, self.camera_index_var, 5, self.apply_source_and_scenario).pack(side=tk.LEFT, padx=6)
        _make_btn(cam_row, "Trocar", self.apply_source_and_scenario, padx=8).pack(side=tk.LEFT, padx=(0, 4))
        _make_btn(cam_row, "Reconectar Webcam", self.reconnect_camera, bg=ACCENT_GREEN, padx=8).pack(side=tk.LEFT)

        tk.Checkbutton(
            parent, text="Espelhar Câmera Pessoal (Modo Webcam)", variable=self.mirror_var, command=self.apply_mirror_setting, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(parent, text="Preset de Resolução da Câmera:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        res_cb = ttk.Combobox(parent, textvariable=self.res_preset_var, values=list(RESOLUTION_PRESETS.keys()), state="readonly")
        res_cb.pack(fill=tk.X, pady=(2, 8))
        res_cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_resolution_preset())

        custom_res = tk.Frame(parent, bg=BG_CARD)
        custom_res.pack(fill=tk.X, pady=(0, 10))
        tk.Label(custom_res, text="Largura:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        _make_entry(custom_res, self.width_var, 6, self.apply_custom_resolution).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(custom_res, text="Altura:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        _make_entry(custom_res, self.height_var, 6, self.apply_custom_resolution).pack(side=tk.LEFT, padx=(4, 8))
        _make_btn(custom_res, "Aplicar Resolução", self.apply_custom_resolution, bg=ACCENT_BLUE, padx=8).pack(side=tk.LEFT)

        self._build_fps_and_speed_controls(parent)

    def _build_fps_and_speed_controls(self, parent: tk.Frame) -> None:
        fps_hdr = tk.Frame(parent, bg=BG_CARD)
        fps_hdr.pack(fill=tk.X)
        tk.Label(fps_hdr, text="Taxa de Quadros da Câmera (FPS):", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        tk.Label(fps_hdr, textvariable=self.fps_label_var, bg=BG_CARD, fg="#38bdf8", font=("DejaVu Sans", 9, "bold")).pack(side=tk.RIGHT)
        tk.Scale(
            parent, from_=1, to=60, orient=tk.HORIZONTAL, variable=self.fps_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda val: self.set_fps(int(float(val)))
        ).pack(fill=tk.X, pady=(2, 6))

        chips = tk.Frame(parent, bg=BG_CARD)
        chips.pack(fill=tk.X, pady=(0, 10))
        for fps_val in FPS_PRESETS:
            _make_btn(chips, f"{fps_val} FPS", lambda f=fps_val: self.set_fps(f), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT, padx=2)

        tk.Checkbutton(
            parent, text="Exibir HUD de Telemetria sobre o Vídeo", variable=self.hud_enabled_var, command=self.apply_hud_setting, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(parent, text="Fonte de Captura (camera = Webcam Pessoal):", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        src_cb = ttk.Combobox(parent, textvariable=self.source_var, values=["camera", "simulator"], state="readonly")
        src_cb.pack(fill=tk.X, pady=(2, 4))
        src_cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_source_and_scenario())

    def _build_thermal_tab(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Simulador Térmico da Câmera Pessoal", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 6))
        tk.Label(parent, text="Perfis Rápidos de Temperatura:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        prof_row = tk.Frame(parent, bg=BG_CARD)
        prof_row.pack(fill=tk.X, pady=(2, 8))
        _make_btn(prof_row, "Corporal (20–42°C)", lambda: self.apply_thermal_profile("human"), bg=ACCENT_BLUE, padx=6, pady=3, font=("DejaVu Sans", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        _make_btn(prof_row, "Ambiente (18–65°C)", lambda: self.apply_thermal_profile("ambient"), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT, padx=(0, 4))
        _make_btn(prof_row, "Alta Temp (22–120°C)", lambda: self.apply_thermal_profile("industrial"), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT)

        tk.Label(parent, text="Paleta da Câmera Térmica Simulada:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        pal_cb = ttk.Combobox(parent, textvariable=self.palette_var, values=list(PALETTE_MAP.keys()), state="readonly")
        pal_cb.pack(fill=tk.X, pady=(2, 8))
        pal_cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_thermal_settings())

        temp_grid = tk.Frame(parent, bg=BG_CARD)
        temp_grid.pack(fill=tk.X, pady=(0, 6))
        for idx, (lbl, var) in enumerate([("Temp. Mín (°C):", self.temp_min_var), ("Temp. Máx (°C):", self.temp_max_var), ("Alarme (°C):", self.temp_alarm_var)]):
            tk.Label(temp_grid, text=lbl, bg=BG_CARD, fg=FG_MUTED).grid(row=idx, column=0, sticky="w", pady=3)
            _make_entry(temp_grid, var, 10, self.apply_thermal_settings).grid(row=idx, column=1, sticky="w", padx=8, pady=3)
        _make_btn(parent, "Aplicar Escala de Temperatura", self.apply_thermal_settings, bg=ACCENT_BLUE, padx=10, pady=4).pack(anchor="w", pady=(2, 8))

        kin_hdr = tk.Frame(parent, bg=BG_CARD)
        kin_hdr.pack(fill=tk.X)
        tk.Label(kin_hdr, text="Calor Gerado pelos Movimentos:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        tk.Label(kin_hdr, textvariable=self.kinetic_label_var, bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.RIGHT)
        tk.Scale(
            parent, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.kinetic_heat_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda _v: self.apply_thermal_settings()
        ).pack(fill=tk.X, pady=(2, 8))

        for text, var in [
            ("Exibir Miras Hotspot / Coldspot", self.show_hotspots_var),
            ("Exibir Contorno de Isoterma Crítica", self.show_isotherms_var),
            ("Exibir Barra de Escala Térmica (°C)", self.show_scale_var),
        ]:
            tk.Checkbutton(parent, text=text, variable=var, command=self.apply_thermal_settings, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD).pack(anchor="w", pady=2)

    def _build_motion_tab(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Análise de Movimentos & Zona ROI", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 8))
        tk.Label(parent, text="Algoritmo de Detecção de Movimento:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
        alg_cb = ttk.Combobox(parent, textvariable=self.algorithm_var, values=["frame_diff", "mog2", "knn"], state="readonly")
        alg_cb.pack(fill=tk.X, pady=(2, 10))
        alg_cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_motion_settings())

        sens_hdr = tk.Frame(parent, bg=BG_CARD)
        sens_hdr.pack(fill=tk.X)
        tk.Label(sens_hdr, text="Sensibilidade aos Movimentos:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
        tk.Label(sens_hdr, textvariable=self.sensitivity_label_var, bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.RIGHT)
        tk.Scale(
            parent, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.sensitivity_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda _v: self.apply_motion_settings()
        ).pack(fill=tk.X, pady=(2, 10))

        m_grid = tk.Frame(parent, bg=BG_CARD)
        m_grid.pack(fill=tk.X, pady=(0, 8))
        tk.Label(m_grid, text="Área Mínima (px²):", bg=BG_CARD, fg=FG_MUTED).grid(row=0, column=0, sticky="w", pady=3)
        _make_entry(m_grid, self.min_area_var, 10, self.apply_motion_settings).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        tk.Label(m_grid, text="Limiar de Alarme (% área):", bg=BG_CARD, fg=FG_MUTED).grid(row=1, column=0, sticky="w", pady=3)
        _make_entry(m_grid, self.alarm_pct_var, 10, self.apply_motion_settings).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        _make_btn(parent, "Aplicar Filtros de Movimento", self.apply_motion_settings, bg=ACCENT_BLUE, padx=10, pady=4).pack(anchor="w", pady=(2, 10))

        for text, var in [
            ("Destacar Objetos em Movimento (Boxes)", self.show_boxes_var),
            ("Rastrear Trilha de Movimento", self.show_trails_var),
            ("Restringir Análise à Zona ROI", self.roi_enabled_var),
        ]:
            tk.Checkbutton(parent, text=text, variable=var, command=self.apply_motion_settings, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD).pack(anchor="w", pady=2)
        _make_btn(parent, "Resetar Zona ROI Padrão (80% Central)", self.reset_roi, padx=10, pady=4).pack(anchor="w", pady=(8, 4))

    @staticmethod
    def _is_text_input_event(event: Any) -> bool:
        widget = getattr(event, "widget", None)
        if widget is None or not hasattr(widget, "winfo_class"):
            return False
        return str(widget.winfo_class()) in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox", "Text"}

    def _bind_shortcuts(self) -> None:
        bindings: list[tuple[str, Any]] = [
            ("1", lambda: self.set_vision_mode("normal")),
            ("2", lambda: self.set_vision_mode("thermal")),
            ("3", lambda: self.set_vision_mode("hybrid")),
            ("4", lambda: self.set_vision_mode("motion_mask")),
            ("m", self.toggle_mirror),
            ("M", self.toggle_mirror),
            ("t", lambda: self.inject_anomaly("thermal_spike")),
            ("T", lambda: self.inject_anomaly("thermal_spike")),
            ("i", lambda: self.inject_anomaly("intrusion")),
            ("I", lambda: self.inject_anomaly("intrusion")),
        ]
        for key, action in bindings:
            self.root.bind(f"<KeyPress-{key}>", lambda e, fn=action: None if self._is_text_input_event(e) else fn())
        self.root.bind("<Escape>", lambda _e: self.close())

    def sync_variables_from_config(self, cfg: SystemConfig) -> None:
        """Sincroniza todas as variáveis da interface com o SystemConfig atual."""
        self.source_var.set(cfg.source)
        self.scenario_var.set(cfg.simulator.scenario)
        self.camera_index_var.set(str(cfg.camera_index))
        self.camera_device_var.set(self.manager.get_camera_device_label(cfg.camera_index))
        self.mirror_var.set(cfg.mirror)
        preset_key = f"{cfg.width}x{cfg.height}"
        if preset_key in RESOLUTION_PRESETS:
            self.res_preset_var.set(preset_key)
        self.width_var.set(str(cfg.width))
        self.height_var.set(str(cfg.height))
        self.fps_var.set(cfg.target_fps)
        self.fps_label_var.set(f"{cfg.target_fps} FPS")
        self.sim_speed_var.set(cfg.simulator.speed_multiplier)
        self.sim_speed_label_var.set(f"{cfg.simulator.speed_multiplier:.1f}x")
        self.hud_enabled_var.set(cfg.hud_enabled)

        self.palette_var.set(cfg.thermal.palette)
        self.temp_min_var.set(str(cfg.thermal.min_temp_c))
        self.temp_max_var.set(str(cfg.thermal.max_temp_c))
        self.temp_alarm_var.set(str(cfg.thermal.overheat_threshold_c))
        kin_pct = round(cfg.thermal.kinetic_heat_gain * 100)
        self.kinetic_heat_var.set(kin_pct)
        self.kinetic_label_var.set(f"{kin_pct}%")
        self.show_hotspots_var.set(cfg.thermal.show_hotspots)
        self.show_isotherms_var.set(cfg.thermal.show_isotherms)
        self.show_scale_var.set(cfg.thermal.show_scale_bar)

        self.algorithm_var.set(cfg.motion.algorithm)
        self.sensitivity_var.set(cfg.motion.sensitivity)
        self.sensitivity_label_var.set(f"{cfg.motion.sensitivity}%")
        self.min_area_var.set(str(cfg.motion.min_area))
        self.alarm_pct_var.set(str(cfg.motion.alarm_motion_pct))
        self.show_boxes_var.set(cfg.motion.show_boxes)
        self.show_trails_var.set(cfg.motion.show_trails)
        self.roi_enabled_var.set(cfg.roi.enabled)
        for mode_key, btn in self.mode_buttons.items():
            btn.configure(bg=ACCENT_BLUE if mode_key == cfg.vision_mode else BG_INPUT)

    def _apply_patch_safely(self, patch: dict[str, Any], success_msg: str | None = None) -> bool:
        try:
            updated = self.manager.update_config(patch)
            self.sync_variables_from_config(updated)
            if success_msg:
                self.status_message_var.set(success_msg)
            return True
        except (ValidationError, ValueError) as exc:
            self.status_message_var.set(f"Erro de validação: {exc}")
            self.sync_variables_from_config(self.manager.get_config())
            return False

    def set_vision_mode(self, mode: str) -> bool:
        return self._apply_patch_safely({"vision_mode": mode}, f"Modo de visão alterado para '{mode}'")

    def toggle_mirror(self) -> bool:
        new_val = not self.manager.get_config().mirror
        return self._apply_patch_safely({"mirror": new_val}, f"Espelhamento da câmera pessoal: {'Ativado' if new_val else 'Desativado'}")

    def apply_mirror_setting(self) -> bool:
        return self._apply_patch_safely({"mirror": bool(self.mirror_var.get())})

    def reconnect_camera(self) -> bool:
        with contextlib.suppress(ValueError, ValidationError):
            typed_idx = int(self.camera_index_var.get().strip())
            if typed_idx != self.manager.get_config().camera_index:
                self.manager.update_config({"camera_index": typed_idx, "source": "camera"})
        ok = self.manager.reconnect_camera()
        cfg = self.manager.get_config()
        self.sync_variables_from_config(cfg)
        dev = self.manager.get_camera_device_label(cfg.camera_index)
        self.status_message_var.set(
            f"Câmera pessoal reconectada: {dev}" if ok else f"Não foi possível abrir {dev}"
        )
        self._refresh_events_listbox(self.manager.get_events(limit=15))
        return ok

    def apply_thermal_profile(self, profile: str) -> bool:
        return self._apply_patch_safely(
            {"thermal_profile": profile},
            f"Perfil térmico aplicado à câmera pessoal: {profile}",
        )

    def apply_resolution_preset(self) -> bool:
        preset = self.res_preset_var.get()
        return self._apply_patch_safely({"resolution_preset": preset}, f"Resolução alterada para {preset}")

    def apply_custom_resolution(self) -> bool:
        try:
            w, h = int(self.width_var.get().strip()), int(self.height_var.get().strip())
        except ValueError:
            self.status_message_var.set("Erro: Largura e altura devem ser números inteiros válidos")
            return False
        return self._apply_patch_safely({"width": w, "height": h}, f"Resolução customizada aplicada: {w}x{h}")

    def set_fps(self, fps: int) -> bool:
        return self._apply_patch_safely({"target_fps": int(fps)}, f"Taxa alvo ajustada: {fps} FPS")

    def apply_sim_speed(self) -> bool:
        spd = round(float(self.sim_speed_var.get()), 2)
        self.sim_speed_label_var.set(f"{spd:.1f}x")
        return self._apply_patch_safely({"simulator": {"speed_multiplier": spd}})

    def apply_hud_setting(self) -> bool:
        return self._apply_patch_safely({"hud_enabled": bool(self.hud_enabled_var.get())})

    def apply_source_and_scenario(self) -> bool:
        try:
            cam_idx = int(self.camera_index_var.get().strip())
        except ValueError:
            self.status_message_var.set("Erro: Índice da câmera inválido")
            return False
        return self._apply_patch_safely(
            {"source": self.source_var.get(), "camera_index": cam_idx, "simulator": {"scenario": self.scenario_var.get()}},
            f"Fonte: {self.source_var.get()} | Câmera #{cam_idx}",
        )

    def apply_thermal_settings(self) -> bool:
        try:
            min_c = float(self.temp_min_var.get().strip())
            max_c = float(self.temp_max_var.get().strip())
            alarm_c = float(self.temp_alarm_var.get().strip())
        except ValueError:
            self.status_message_var.set("Erro: Temperaturas devem ser valores numéricos")
            return False
        cur_th = self.manager.get_config().thermal
        if (
            min_c < max_c
            and alarm_c == cur_th.overheat_threshold_c
            and (min_c != cur_th.min_temp_c or max_c != cur_th.max_temp_c)
        ):
            alarm_c = min(max_c, max(min_c, alarm_c))
        kin_gain = round(float(self.kinetic_heat_var.get()) / 100.0, 2)
        self.kinetic_label_var.set(f"{round(kin_gain * 100)}%")
        thermal_patch = {
            "palette": self.palette_var.get(), "min_temp_c": min_c, "max_temp_c": max_c,
            "overheat_threshold_c": alarm_c, "kinetic_heat_gain": kin_gain,
            "show_hotspots": bool(self.show_hotspots_var.get()),
            "show_isotherms": bool(self.show_isotherms_var.get()),
            "show_scale_bar": bool(self.show_scale_var.get()),
        }
        return self._apply_patch_safely({"thermal": thermal_patch})

    def apply_motion_settings(self) -> bool:
        try:
            min_area = int(self.min_area_var.get().strip())
            alarm_pct = float(self.alarm_pct_var.get().strip())
        except ValueError:
            self.status_message_var.set("Erro: Área mínima e limiar devem ser numéricos")
            return False
        sens = int(self.sensitivity_var.get())
        self.sensitivity_label_var.set(f"{sens}%")
        motion_patch = {
            "algorithm": self.algorithm_var.get(), "sensitivity": sens, "min_area": min_area,
            "alarm_motion_pct": alarm_pct, "show_boxes": bool(self.show_boxes_var.get()),
            "show_trails": bool(self.show_trails_var.get()),
        }
        return self._apply_patch_safely({"motion": motion_patch, "roi": {"enabled": bool(self.roi_enabled_var.get())}})

    def reset_roi(self) -> bool:
        return self._apply_patch_safely(
            {"roi": {"enabled": False, "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}},
            "Zona ROI resetada para o padrão central (80%)",
        )

    def inject_anomaly(self, kind: AnomalyType) -> None:
        self.manager.trigger_anomaly(kind, duration_frames=45)
        self.status_message_var.set(f"Anomalia industrial acionada: {kind}")
        self._refresh_events_listbox(self.manager.get_events(limit=15))

    def recalibrate(self) -> None:
        self.manager.reset_detectors()
        self.status_message_var.set("Sensores de movimento e acumulador térmico recalibrados")
        self._refresh_events_listbox(self.manager.get_events(limit=15))

    def clear_events(self) -> None:
        self.manager.clear_events()
        self._refresh_events_listbox([])
        self.status_message_var.set("Histórico de eventos limpo")

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        self.pause_btn_var.set("Retomar Vídeo" if self.is_paused else "Pausar Vídeo")
        self.status_message_var.set("Captura pausada" if self.is_paused else "Captura retomada em tempo real")

    def save_snapshot(self, target_path: Path | None = None) -> Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_path or (self.snapshot_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        out_path.write_bytes(self.manager.get_snapshot_jpeg(quality=92))
        self.status_message_var.set(f"Snapshot salvo em: {out_path}")
        return out_path

    def _on_canvas_press(self, event: Any) -> None:
        with contextlib.suppress(tk.TclError):
            self.canvas.focus_set()
        self._drag_start = (float(event.x), float(event.y))
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _on_canvas_drag(self, event: Any) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#38bdf8", width=2, dash=(4, 2)
        )

    def _on_canvas_release(self, event: Any) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        drawn_w, drawn_h, ox, oy = self._viewport_rect
        roi = canvas_rect_to_roi(x0, y0, float(event.x), float(event.y), ox, oy, drawn_w, drawn_h)
        if roi is not None:
            self._apply_patch_safely(
                {"roi": roi.model_dump()},
                f"Zona ROI definida no vídeo: x={roi.x:.2f}, y={roi.y:.2f}, w={roi.w:.2f}, h={roi.h:.2f}",
            )

    def _on_canvas_configure(self, event: Any) -> None:
        if self._last_frame_bgr is not None and getattr(event, "width", 0) > 1 and getattr(event, "height", 0) > 1:
            self._render_canvas_frame(self._last_frame_bgr, canvas_size=(int(event.width), int(event.height)))

    def step_once(self) -> FrameTelemetry:
        """Processa um quadro na câmera/simulador e atualiza vídeo, KPIs e eventos."""
        frame_bgr, tel = self.manager.step_frame()
        self._render_canvas_frame(frame_bgr)
        self._update_telemetry_widgets(tel)
        self._refresh_events_listbox(self.manager.get_events(limit=15))
        return tel

    def _render_canvas_frame(
        self, frame_bgr: np.ndarray, canvas_size: tuple[int, int] | None = None
    ) -> None:
        self._last_frame_bgr = frame_bgr
        raw_w, raw_h = canvas_size or (self.canvas.winfo_width(), self.canvas.winfo_height())
        canvas_w = max(64, raw_w) if raw_w > 1 else 640
        canvas_h = max(64, raw_h) if raw_h > 1 else 480
        src_h, src_w = frame_bgr.shape[:2]
        drawn_w, drawn_h, ox, oy = compute_aspect_fit(src_w, src_h, canvas_w, canvas_h)
        self._viewport_rect = (drawn_w, drawn_h, ox, oy)
        ppm_bytes, _ = frame_to_ppm_bytes(frame_bgr, max_size=(canvas_w, canvas_h))
        self._photo_ref = tk.PhotoImage(master=self.root, data=ppm_bytes, format="PPM")
        self.canvas.delete("video_frame")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo_ref, tags=("video_frame",))
        if self._drag_rect_id is not None:
            with contextlib.suppress(tk.TclError):
                self.canvas.tag_raise(self._drag_rect_id)

    def _update_telemetry_widgets(self, tel: FrameTelemetry) -> None:
        self.stream_meta_var.set(f"{tel.width}x{tel.height} @ {tel.actual_fps:.1f} FPS")
        self.kpi_fps_var.set(f"{tel.actual_fps:.1f} / {tel.target_fps} FPS")
        self.kpi_latency_var.set(f"Processamento: {tel.processing_ms:.1f} ms")
        self.kpi_motion_var.set(f"{tel.motion_percentage:.1f}%")
        self.kpi_objects_var.set(f"{tel.object_count} objeto(s) em movimento")
        self.kpi_temp_max_var.set(f"{tel.max_temp_c:.1f} °C")
        self.kpi_temp_range_var.set(f"Mín: {tel.min_temp_c:.1f}°C | Méd: {tel.avg_temp_c:.1f}°C")
        self.kpi_resolution_var.set(f"{tel.width} x {tel.height}")
        self.kpi_coords_var.set(f"Hotspot: ({tel.hotspot_x}, {tel.hotspot_y})")

        if tel.hardware_fallback:
            self.badge_source_var.set("FONTE: FALLBACK (SEM CÂMERA)")
        elif tel.source_active == "camera":
            self.badge_source_var.set(f"FONTE: CÂMERA PESSOAL (#{self.manager.get_config().camera_index})")
        else:
            self.badge_source_var.set("FONTE: MODO TESTE")

        self.badge_alert_var.set(f"STATUS: {tel.alert_level}")
        alert_color = ACCENT_CRIT if tel.alert_level == "CRITICO" else ACCENT_WARN if tel.alert_level == "ATENCAO" else ACCENT_GREEN
        self.lbl_badge_alert.configure(bg=alert_color)

    def _refresh_events_listbox(self, events: list[EventRecord]) -> None:
        sig = tuple(ev.id for ev in events)
        if sig == self._last_events_sig:
            return
        self._last_events_sig = sig
        self.events_listbox.delete(0, tk.END)
        if not events:
            self.events_listbox.insert(tk.END, "Nenhum evento registrado.")
            return
        for ev in events:
            ts = ev.timestamp_iso[11:19] if len(ev.timestamp_iso) >= 19 else ev.timestamp_iso
            self.events_listbox.insert(tk.END, f"[{ts}] [{ev.severity}] [{ev.category}] {ev.message}")

    def close(self) -> None:
        """Encerra o loop de atualização, libera a câmera e fecha a janela."""
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._after_id)
            self._after_id = None
        self.manager.close()
        with contextlib.suppress(tk.TclError):
            self.root.destroy()


def run_native_app(
    config: SystemConfig | None = None,
    max_frames: int | None = None,
    start_hidden: bool = False,
    snapshot_dir: Path | None = None,
) -> int:
    """Executa o aplicativo desktop local completo e retorna o total de quadros exibidos."""
    manager = CameraManager(config or SystemConfig())
    try:
        app = IndustrialVisionDesktopApp(
            manager=manager, snapshot_dir=snapshot_dir, start_hidden=start_hidden
        )
    except tk.TclError as exc:
        manager.close()
        raise RuntimeError(
            "Ambiente gráfico (X11/Wayland) indisponível para abrir o aplicativo desktop local. "
            "Verifique a variável $DISPLAY ou utilize o modo servidor Web ('python main.py')."
        ) from exc

    frames_processed = 0

    def _tick() -> None:
        nonlocal frames_processed
        if app._closed:
            return
        t0 = time.perf_counter()
        if not app.is_paused:
            app.step_once()
            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                app.close()
                return
        elapsed = time.perf_counter() - t0
        delay_ms = max(1, round((app.manager.get_frame_interval_sec() - elapsed) * 1000))
        app._after_id = app.root.after(delay_ms, _tick)

    app._after_id = app.root.after(10, _tick)
    app.root.mainloop()
    return frames_processed
