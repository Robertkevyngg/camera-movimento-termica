"""Simulador de cenários industriais sintéticos e injeção de anomalias."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from app.config import SimulatorConfig

AnomalyType = Literal["thermal_spike", "intrusion", "clear"]
VALID_ANOMALIES: tuple[str, ...] = ("thermal_spike", "intrusion", "clear")


def _add_gaussian_hotspot(
    heat_map: np.ndarray,
    center_x: int,
    center_y: int,
    sigma_x: float,
    sigma_y: float,
    peak_intensity: float,
) -> None:
    """Adiciona um pulso gaussiano de calor ao mapa térmico (in-place em buffer novo)."""
    h, w = heat_map.shape
    radius_x = max(4, int(sigma_x * 3.0))
    radius_y = max(4, int(sigma_y * 3.0))
    x0, x1 = max(0, center_x - radius_x), min(w, center_x + radius_x)
    y0, y1 = max(0, center_y - radius_y), min(h, center_y + radius_y)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1, dtype=np.float32) - float(center_x)
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - float(center_y)
    gauss = np.exp(-0.5 * ((xs / max(1.0, sigma_x)) ** 2 + (ys / max(1.0, sigma_y)) ** 2))
    patch = gauss * float(peak_intensity)
    heat_map[y0:y1, x0:x1] = np.maximum(heat_map[y0:y1, x0:x1], patch)


class IndustrialSceneSimulator:
    """Gerador sintético de cenas industriais com suporte a anomalias sob demanda."""

    def __init__(self, seed: int = 42) -> None:
        self._step: int = 0
        self._rng = np.random.default_rng(seed)
        self._anomaly_frames: dict[str, int] = {
            "thermal_spike": 0,
            "intrusion": 0,
        }

    def trigger_anomaly(self, kind: AnomalyType, duration_frames: int = 45) -> None:
        """Ativa ou limpa uma anomalia simulada por N quadros."""
        if kind not in VALID_ANOMALIES:
            raise ValueError(f"Tipo de anomalia inválido: '{kind}'")
        if kind == "clear":
            self._anomaly_frames = {"thermal_spike": 0, "intrusion": 0}
            return
        self._anomaly_frames[kind] = max(1, int(duration_frames))

    def has_active_anomaly(self, kind: str) -> bool:
        """Retorna True se a anomalia especificada estiver ativa."""
        return self._anomaly_frames.get(kind, 0) > 0

    def get_active_anomalies(self) -> tuple[str, ...]:
        """Lista os identificadores de anomalias ativas no momento."""
        return tuple(k for k, v in self._anomaly_frames.items() if v > 0)

    def generate_frame(
        self, width: int, height: int, config: SimulatorConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera um quadro BGR (uint8) e um mapa base de calor (float32 0..1)."""
        self._step += 1
        t = self._step * 0.08 * config.speed_multiplier

        canvas = self._create_industrial_backdrop(width, height)
        heat_map = np.full(
            (height, width), float(config.ambient_heat_factor) * 0.45, dtype=np.float32
        )

        if config.scenario == "conveyor":
            self._render_conveyor(canvas, heat_map, width, height, t)
        elif config.scenario == "machinery":
            self._render_machinery(canvas, heat_map, width, height, t)
        elif config.scenario == "perimeter":
            self._render_perimeter(canvas, heat_map, width, height, t)
        else:
            self._render_boiler(canvas, heat_map, width, height, t)

        self.apply_active_anomalies(canvas, heat_map, t)
        self._decrement_anomalies()

        if config.noise_level > 0.0:
            noise = self._rng.normal(0.0, config.noise_level, size=canvas.shape)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return canvas, np.clip(heat_map, 0.0, 1.0)

    def apply_active_anomalies(
        self, canvas: np.ndarray, heat_map: np.ndarray, t: float
    ) -> None:
        """Aplica anomalias ativas sobre um quadro existente (simulador ou câmera real)."""
        height, width = canvas.shape[:2]
        if self._anomaly_frames.get("intrusion", 0) > 0:
            self._draw_intrusion_anomaly(canvas, heat_map, width, height, t)
        if self._anomaly_frames.get("thermal_spike", 0) > 0:
            self._draw_thermal_spike_anomaly(canvas, heat_map, width, height, t)

    def step_anomalies(
        self, canvas: np.ndarray, heat_map: np.ndarray, t: float
    ) -> None:
        """Aplica anomalias ativas e decrementa os contadores de duração."""
        self.apply_active_anomalies(canvas, heat_map, t)
        self._decrement_anomalies()

    def _decrement_anomalies(self) -> None:
        for key in ("thermal_spike", "intrusion"):
            if self._anomaly_frames[key] > 0:
                self._anomaly_frames[key] -= 1

    def _create_industrial_backdrop(self, width: int, height: int) -> np.ndarray:
        canvas = np.full((height, width, 3), (34, 38, 42), dtype=np.uint8)
        step_x = max(24, width // 12)
        step_y = max(24, height // 9)
        for x in range(0, width, step_x):
            cv2.line(canvas, (x, 0), (x, height), (44, 49, 54), 1)
        for y in range(0, height, step_y):
            cv2.line(canvas, (0, y), (width, y), (44, 49, 54), 1)
        return canvas

    def _render_conveyor(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        belt_y0, belt_y1 = int(h * 0.36), int(h * 0.66)
        cv2.rectangle(canvas, (0, belt_y0), (w, belt_y1), (58, 64, 70), -1)
        cv2.line(canvas, (0, belt_y0), (w, belt_y0), (120, 140, 155), 3)
        cv2.line(canvas, (0, belt_y1), (w, belt_y1), (120, 140, 155), 3)

        roller_spacing = max(28, w // 10)
        offset = int((t * 45) % roller_spacing)
        for rx in range(-roller_spacing + offset, w + roller_spacing, roller_spacing):
            cv2.line(canvas, (rx, belt_y0 + 4), (rx, belt_y1 - 4), (76, 82, 88), 2)

        box_w, box_h = max(36, int(w * 0.09)), max(28, int((belt_y1 - belt_y0) * 0.65))
        cy = (belt_y0 + belt_y1) // 2
        for idx in range(3):
            cx = int(((t * 95 + idx * (w / 3.0)) % (w + box_w * 2)) - box_w)
            x0, y0 = cx - box_w // 2, cy - box_h // 2
            is_hot_part = idx == 1
            color = (70, 145, 235) if is_hot_part else (165, 175, 185)
            cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (220, 230, 240), 2)
            _add_gaussian_hotspot(
                heat, cx, cy, box_w * 0.45, box_h * 0.45, 0.72 if is_hot_part else 0.48
            )

    def _render_machinery(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        centers = [(int(w * 0.32), int(h * 0.50)), (int(w * 0.68), int(h * 0.50))]
        radius = max(32, int(min(w, h) * 0.18))
        for idx, (cx, cy) in enumerate(centers):
            cv2.circle(canvas, (cx, cy), radius + 10, (68, 76, 84), -1)
            cv2.circle(canvas, (cx, cy), radius, (45, 50, 56), -1)
            for blade in range(4):
                angle = t * (2.6 if idx == 0 else -3.2) + blade * (math.pi / 2.0)
                bx = int(cx + math.cos(angle) * (radius - 6))
                by = int(cy + math.sin(angle) * (radius - 6))
                cv2.line(canvas, (cx, cy), (bx, by), (185, 195, 205), 5)
                cv2.circle(canvas, (bx, by), max(5, radius // 7), (110, 170, 230), -1)
            cv2.circle(canvas, (cx, cy), max(8, radius // 4), (140, 160, 190), -1)
            bearing_heat = 0.68 + 0.10 * math.sin(t * 1.5 + idx)
            _add_gaussian_hotspot(heat, cx, cy, radius * 0.65, radius * 0.65, bearing_heat)

    def _render_perimeter(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        zone_x0, zone_y0 = int(w * 0.15), int(h * 0.18)
        zone_x1, zone_y1 = int(w * 0.85), int(h * 0.82)
        cv2.rectangle(canvas, (zone_x0, zone_y0), (zone_x1, zone_y1), (40, 165, 215), 2)

        agv_x = int(w * 0.5 + math.sin(t * 1.1) * (w * 0.28))
        agv_y = int(h * 0.5 + math.cos(t * 0.85) * (h * 0.22))
        agv_r = max(18, int(min(w, h) * 0.065))
        cv2.circle(canvas, (agv_x, agv_y), agv_r, (80, 190, 245), -1)
        cv2.circle(canvas, (agv_x, agv_y), agv_r + 3, (235, 245, 255), 2)
        _add_gaussian_hotspot(heat, agv_x, agv_y, agv_r * 0.9, agv_r * 0.9, 0.66)

    def _render_boiler(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        vx0, vy0 = int(w * 0.25), int(h * 0.22)
        vx1, vy1 = int(w * 0.75), int(h * 0.80)
        cv2.rectangle(canvas, (vx0, vy0), (vx1, vy1), (72, 78, 88), -1)
        cv2.rectangle(canvas, (vx0, vy0), (vx1, vy1), (135, 145, 155), 3)

        cx, cy = (vx0 + vx1) // 2, (vy0 + vy1) // 2
        _add_gaussian_hotspot(heat, cx, cy, (vx1 - vx0) * 0.38, (vy1 - vy0) * 0.35, 0.74)

        for pipe_idx in range(3):
            px = vx0 + int((vx1 - vx0) * (0.25 + pipe_idx * 0.25))
            steam_y = int(vy0 - ((t * 38 + pipe_idx * 25) % max(24, int(h * 0.18))))
            radius = max(8, int(min(w, h) * 0.035))
            cv2.circle(canvas, (px, steam_y), radius, (195, 205, 215), -1)
            _add_gaussian_hotspot(heat, px, steam_y, radius * 1.3, radius * 1.3, 0.65)

    def _draw_intrusion_anomaly(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        obj_w, obj_h = max(48, int(w * 0.14)), max(48, int(h * 0.18))
        cx = int(w * (0.25 + 0.5 * (0.5 + 0.5 * math.sin(t * 3.4))))
        cy = int(h * (0.35 + 0.3 * (0.5 + 0.5 * math.cos(t * 2.7))))
        x0, y0 = max(0, cx - obj_w // 2), max(0, cy - obj_h // 2)
        x1, y1 = min(w - 1, x0 + obj_w), min(h - 1, y0 + obj_h)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (45, 95, 250), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 2)
        _add_gaussian_hotspot(heat, cx, cy, obj_w * 0.5, obj_h * 0.5, 0.78)

    def _draw_thermal_spike_anomaly(
        self, canvas: np.ndarray, heat: np.ndarray, w: int, h: int, t: float
    ) -> None:
        hx = int(w * 0.52 + math.sin(t * 2.0) * (w * 0.06))
        hy = int(h * 0.48 + math.cos(t * 2.0) * (h * 0.06))
        radius = max(16, int(min(w, h) * 0.07))
        cv2.circle(canvas, (hx, hy), radius, (235, 245, 255), -1)
        _add_gaussian_hotspot(heat, hx, hy, radius * 1.4, radius * 1.4, 1.0)
