"""Tema visual, fábricas de widgets e geometria do vídeo do aplicativo desktop."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import cv2
import numpy as np

from app.config import ROIConfig

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


def make_btn(parent: tk.Widget, text: str, command: Any, bg: str = BG_INPUT, **kwargs: Any) -> tk.Button:
    """Cria um botão estilizado no padrão SCADA."""
    return tk.Button(parent, text=text, command=command, bg=bg, fg=FG_PRIMARY, activebackground=ACCENT_BLUE, activeforeground=FG_PRIMARY, relief=tk.FLAT, **kwargs)


def make_entry(parent: tk.Widget, var: tk.Variable, width: int, on_enter: Any) -> tk.Entry:
    """Cria um campo de entrada estilizado que aplica alterações ao pressionar Enter."""
    entry = tk.Entry(parent, textvariable=var, width=width, bg=BG_INPUT, fg=FG_PRIMARY)
    entry.bind("<Return>", lambda _e: on_enter())
    return entry
