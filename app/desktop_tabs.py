"""Construção das abas de controle (câmera, termografia e movimento) do app desktop."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from app.config import FPS_PRESETS, RESOLUTION_PRESETS
from app.desktop_widgets import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    BG_CARD,
    BG_INPUT,
    FG_MUTED,
    FG_PRIMARY,
    make_btn,
    make_entry,
)
from app.thermal_simulator import PALETTE_MAP

# As abas só leem variáveis Tk e chamam ações do aplicativo; `Any` evita um
# import circular com app.desktop_app, onde a classe concreta é definida.
DesktopControls = Any


def build_camera_tab(ui: DesktopControls, parent: tk.Frame) -> None:
    tk.Label(parent, text="Câmera Pessoal, Resolução & FPS", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 6))
    tk.Label(parent, text="Câmera Pessoal Ativa:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    tk.Label(parent, textvariable=ui.camera_device_var, bg=BG_INPUT, fg="#38bdf8", padx=8, pady=5, font=("DejaVu Sans Mono", 9, "bold")).pack(fill=tk.X, pady=(2, 8))

    cam_row = tk.Frame(parent, bg=BG_CARD)
    cam_row.pack(fill=tk.X, pady=(0, 6))
    tk.Label(cam_row, text="Índice (/dev/videoN):", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    make_entry(cam_row, ui.camera_index_var, 5, ui.apply_source_and_scenario).pack(side=tk.LEFT, padx=6)
    make_btn(cam_row, "Trocar", ui.apply_source_and_scenario, padx=8).pack(side=tk.LEFT, padx=(0, 4))
    make_btn(cam_row, "Reconectar Webcam", ui.reconnect_camera, bg=ACCENT_GREEN, padx=8).pack(side=tk.LEFT)

    tk.Checkbutton(
        parent, text="Espelhar Câmera Pessoal (Modo Webcam)", variable=ui.mirror_var, command=ui.apply_mirror_setting, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD
    ).pack(anchor="w", pady=(0, 8))

    tk.Label(parent, text="Preset de Resolução da Câmera:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    res_cb = ttk.Combobox(parent, textvariable=ui.res_preset_var, values=list(RESOLUTION_PRESETS.keys()), state="readonly")
    res_cb.pack(fill=tk.X, pady=(2, 8))
    res_cb.bind("<<ComboboxSelected>>", lambda _e: ui.apply_resolution_preset())

    custom_res = tk.Frame(parent, bg=BG_CARD)
    custom_res.pack(fill=tk.X, pady=(0, 10))
    tk.Label(custom_res, text="Largura:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    make_entry(custom_res, ui.width_var, 6, ui.apply_custom_resolution).pack(side=tk.LEFT, padx=(4, 8))
    tk.Label(custom_res, text="Altura:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    make_entry(custom_res, ui.height_var, 6, ui.apply_custom_resolution).pack(side=tk.LEFT, padx=(4, 8))
    make_btn(custom_res, "Aplicar Resolução", ui.apply_custom_resolution, bg=ACCENT_BLUE, padx=8).pack(side=tk.LEFT)

    build_fps_and_speed_controls(ui, parent)


def build_fps_and_speed_controls(ui: DesktopControls, parent: tk.Frame) -> None:
    fps_hdr = tk.Frame(parent, bg=BG_CARD)
    fps_hdr.pack(fill=tk.X)
    tk.Label(fps_hdr, text="Taxa de Quadros da Câmera (FPS):", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    tk.Label(fps_hdr, textvariable=ui.fps_label_var, bg=BG_CARD, fg="#38bdf8", font=("DejaVu Sans", 9, "bold")).pack(side=tk.RIGHT)
    tk.Scale(
        parent, from_=1, to=60, orient=tk.HORIZONTAL, variable=ui.fps_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda val: ui.set_fps(int(float(val)))
    ).pack(fill=tk.X, pady=(2, 6))

    chips = tk.Frame(parent, bg=BG_CARD)
    chips.pack(fill=tk.X, pady=(0, 10))
    for fps_val in FPS_PRESETS:
        make_btn(chips, f"{fps_val} FPS", lambda f=fps_val: ui.set_fps(f), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT, padx=2)

    tk.Checkbutton(
        parent, text="Exibir HUD de Telemetria sobre o Vídeo", variable=ui.hud_enabled_var, command=ui.apply_hud_setting, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD
    ).pack(anchor="w", pady=(0, 8))

    tk.Label(parent, text="Fonte de Captura (camera = Webcam Pessoal):", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    src_cb = ttk.Combobox(parent, textvariable=ui.source_var, values=["camera", "simulator"], state="readonly")
    src_cb.pack(fill=tk.X, pady=(2, 4))
    src_cb.bind("<<ComboboxSelected>>", lambda _e: ui.apply_source_and_scenario())


def build_thermal_tab(ui: DesktopControls, parent: tk.Frame) -> None:
    tk.Label(parent, text="Simulador Térmico da Câmera Pessoal", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 6))
    tk.Label(parent, text="Perfis Rápidos de Temperatura:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    prof_row = tk.Frame(parent, bg=BG_CARD)
    prof_row.pack(fill=tk.X, pady=(2, 8))
    make_btn(prof_row, "Corporal (20–42°C)", lambda: ui.apply_thermal_profile("human"), bg=ACCENT_BLUE, padx=6, pady=3, font=("DejaVu Sans", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
    make_btn(prof_row, "Ambiente (18–65°C)", lambda: ui.apply_thermal_profile("ambient"), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT, padx=(0, 4))
    make_btn(prof_row, "Alta Temp (22–120°C)", lambda: ui.apply_thermal_profile("industrial"), padx=6, pady=3, font=("DejaVu Sans", 8)).pack(side=tk.LEFT)

    tk.Label(parent, text="Paleta da Câmera Térmica Simulada:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    pal_cb = ttk.Combobox(parent, textvariable=ui.palette_var, values=list(PALETTE_MAP.keys()), state="readonly")
    pal_cb.pack(fill=tk.X, pady=(2, 8))
    pal_cb.bind("<<ComboboxSelected>>", lambda _e: ui.apply_thermal_settings())

    temp_grid = tk.Frame(parent, bg=BG_CARD)
    temp_grid.pack(fill=tk.X, pady=(0, 6))
    for idx, (lbl, var) in enumerate([("Temp. Mín (°C):", ui.temp_min_var), ("Temp. Máx (°C):", ui.temp_max_var), ("Alarme (°C):", ui.temp_alarm_var)]):
        tk.Label(temp_grid, text=lbl, bg=BG_CARD, fg=FG_MUTED).grid(row=idx, column=0, sticky="w", pady=3)
        make_entry(temp_grid, var, 10, ui.apply_thermal_settings).grid(row=idx, column=1, sticky="w", padx=8, pady=3)
    make_btn(parent, "Aplicar Escala de Temperatura", ui.apply_thermal_settings, bg=ACCENT_BLUE, padx=10, pady=4).pack(anchor="w", pady=(2, 8))

    kin_hdr = tk.Frame(parent, bg=BG_CARD)
    kin_hdr.pack(fill=tk.X)
    tk.Label(kin_hdr, text="Calor Gerado pelos Movimentos:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    tk.Label(kin_hdr, textvariable=ui.kinetic_label_var, bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.RIGHT)
    tk.Scale(
        parent, from_=0, to=100, orient=tk.HORIZONTAL, variable=ui.kinetic_heat_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda _v: ui.apply_thermal_settings()
    ).pack(fill=tk.X, pady=(2, 8))

    for text, var in [
        ("Exibir Miras Hotspot / Coldspot", ui.show_hotspots_var),
        ("Exibir Contorno de Isoterma Crítica", ui.show_isotherms_var),
        ("Exibir Barra de Escala Térmica (°C)", ui.show_scale_var),
    ]:
        tk.Checkbutton(parent, text=text, variable=var, command=ui.apply_thermal_settings, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD).pack(anchor="w", pady=2)


def build_motion_tab(ui: DesktopControls, parent: tk.Frame) -> None:
    tk.Label(parent, text="Análise de Movimentos & Zona ROI", bg=BG_CARD, fg=FG_PRIMARY, font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(0, 8))
    tk.Label(parent, text="Algoritmo de Detecção de Movimento:", bg=BG_CARD, fg=FG_MUTED).pack(anchor="w")
    alg_cb = ttk.Combobox(parent, textvariable=ui.algorithm_var, values=["frame_diff", "mog2", "knn"], state="readonly")
    alg_cb.pack(fill=tk.X, pady=(2, 10))
    alg_cb.bind("<<ComboboxSelected>>", lambda _e: ui.apply_motion_settings())

    sens_hdr = tk.Frame(parent, bg=BG_CARD)
    sens_hdr.pack(fill=tk.X)
    tk.Label(sens_hdr, text="Sensibilidade aos Movimentos:", bg=BG_CARD, fg=FG_MUTED).pack(side=tk.LEFT)
    tk.Label(sens_hdr, textvariable=ui.sensitivity_label_var, bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.RIGHT)
    tk.Scale(
        parent, from_=1, to=100, orient=tk.HORIZONTAL, variable=ui.sensitivity_var, showvalue=False, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0, troughcolor=BG_INPUT, command=lambda _v: ui.apply_motion_settings()
    ).pack(fill=tk.X, pady=(2, 10))

    m_grid = tk.Frame(parent, bg=BG_CARD)
    m_grid.pack(fill=tk.X, pady=(0, 8))
    tk.Label(m_grid, text="Área Mínima (px²):", bg=BG_CARD, fg=FG_MUTED).grid(row=0, column=0, sticky="w", pady=3)
    make_entry(m_grid, ui.min_area_var, 10, ui.apply_motion_settings).grid(row=0, column=1, sticky="w", padx=8, pady=3)
    tk.Label(m_grid, text="Limiar de Alarme (% área):", bg=BG_CARD, fg=FG_MUTED).grid(row=1, column=0, sticky="w", pady=3)
    make_entry(m_grid, ui.alarm_pct_var, 10, ui.apply_motion_settings).grid(row=1, column=1, sticky="w", padx=8, pady=3)
    make_btn(parent, "Aplicar Filtros de Movimento", ui.apply_motion_settings, bg=ACCENT_BLUE, padx=10, pady=4).pack(anchor="w", pady=(2, 10))

    for text, var in [
        ("Destacar Objetos em Movimento (Boxes)", ui.show_boxes_var),
        ("Rastrear Trilha de Movimento", ui.show_trails_var),
        ("Restringir Análise à Zona ROI", ui.roi_enabled_var),
    ]:
        tk.Checkbutton(parent, text=text, variable=var, command=ui.apply_motion_settings, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_CARD).pack(anchor="w", pady=2)
    make_btn(parent, "Resetar Zona ROI Padrão (80% Central)", ui.reset_roi, padx=10, pady=4).pack(anchor="w", pady=(8, 4))
