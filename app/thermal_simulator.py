"""Simulador de câmera térmica infravermelha (termografia industrial)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import ROIConfig, ThermalConfig, ThermalPalette

PALETTE_MAP: dict[ThermalPalette, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "turbo": cv2.COLORMAP_TURBO,
    "plasma": cv2.COLORMAP_PLASMA,
    "bone": cv2.COLORMAP_BONE,
    "magma": cv2.COLORMAP_MAGMA,
}


@dataclass(frozen=True)
class ThermalResult:
    """Resultado imutável do processamento termográfico de um quadro."""

    thermal_bgr: np.ndarray
    temp_map_c: np.ndarray
    min_temp_c: float
    avg_temp_c: float
    max_temp_c: float
    hotspot_x: int
    hotspot_y: int
    coldspot_x: int
    coldspot_y: int
    overheat_alarm: bool


class ThermalSimulator:
    """Converte quadros RGB em imagens termográficas simuladas com telemetria em °C."""

    def __init__(self) -> None:
        self._kinetic_heat: np.ndarray | None = None

    def reset(self) -> None:
        """Limpa o buffer acumulado de calor cinético."""
        self._kinetic_heat = None

    def process(
        self,
        frame_bgr: np.ndarray,
        config: ThermalConfig,
        motion_mask: np.ndarray | None = None,
        heat_hint: np.ndarray | None = None,
        roi: ROIConfig | None = None,
        *,
        is_new_frame: bool = True,
    ) -> ThermalResult:
        """Gera imagem térmica colorizada, mapa de temperatura (°C) e estatísticas."""
        if (
            frame_bgr is None
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] <= 0
            or frame_bgr.shape[1] <= 0
        ):
            raise ValueError("Quadro inválido para simulação térmica")

        height, width = frame_bgr.shape[:2]
        base_heat = self._compute_base_emissivity(frame_bgr, config, heat_hint)
        kinetic_layer = self._update_kinetic_heat(
            height, width, config, motion_mask, is_new_frame=is_new_frame
        )
        combined_norm = np.clip(base_heat + kinetic_layer, 0.0, 1.0)

        temp_span = config.max_temp_c - config.min_temp_c
        temp_map_c = (config.min_temp_c + combined_norm * temp_span).astype(np.float32)

        stats = self._extract_temperature_stats(temp_map_c, width, height, config, roi)
        thermal_bgr = self._render_colormap_and_isotherms(
            combined_norm, temp_map_c, config
        )

        return ThermalResult(
            thermal_bgr=thermal_bgr,
            temp_map_c=temp_map_c,
            min_temp_c=stats["min_c"],
            avg_temp_c=stats["avg_c"],
            max_temp_c=stats["max_c"],
            hotspot_x=stats["hot_x"],
            hotspot_y=stats["hot_y"],
            coldspot_x=stats["cold_x"],
            coldspot_y=stats["cold_y"],
            overheat_alarm=bool(stats["max_c"] >= config.overheat_threshold_c),
        )

    def compute_luminance_heat(
        self, frame_bgr: np.ndarray, config: ThermalConfig
    ) -> np.ndarray:
        """Calcula o mapa base de emissividade térmica [0..1] da câmera pessoal (luminância + tons quentes)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        b_ch = frame_bgr[:, :, 0].astype(np.float32) / 255.0
        g_ch = frame_bgr[:, :, 1].astype(np.float32) / 255.0
        r_ch = frame_bgr[:, :, 2].astype(np.float32) / 255.0
        warmth = np.clip(r_ch - b_ch, 0.0, 1.0) + 0.45 * np.clip(r_ch - g_ch, 0.0, 1.0)

        p_low = float(np.percentile(gray, 5))
        p_med = float(np.percentile(gray, 50))
        p_ref = float(np.percentile(gray, 92))
        span = p_ref - p_low
        if span > 0.02 and p_med < 0.45:
            low_light_gain = float(np.clip((0.45 - p_med) / 0.35, 0.0, 1.0))
            stretched = np.clip((gray - p_low) / max(0.08, span), 0.0, 1.0) * (
                0.55 * low_light_gain
            )
            signal = np.maximum(gray, stretched) + warmth * 0.22
        else:
            signal = gray + warmth * 0.22

        diffused = cv2.GaussianBlur(np.clip(signal, 0.0, 1.0), (9, 9), 0)
        contrasted = 0.35 + (diffused - 0.35) * float(config.emissivity_contrast)
        return np.clip(contrasted, 0.0, 1.0)

    def _compute_base_emissivity(
        self,
        frame_bgr: np.ndarray,
        config: ThermalConfig,
        heat_hint: np.ndarray | None,
    ) -> np.ndarray:
        lum_heat = self.compute_luminance_heat(frame_bgr, config)

        if heat_hint is not None and heat_hint.shape == lum_heat.shape:
            return np.maximum(lum_heat * 0.55, np.clip(heat_hint, 0.0, 1.0))
        return lum_heat

    def _update_kinetic_heat(
        self,
        height: int,
        width: int,
        config: ThermalConfig,
        motion_mask: np.ndarray | None,
        *,
        is_new_frame: bool = True,
    ) -> np.ndarray:
        if self._kinetic_heat is None or self._kinetic_heat.shape != (height, width):
            self._kinetic_heat = np.zeros((height, width), dtype=np.float32)

        if config.kinetic_heat_gain <= 0.0:
            self._kinetic_heat.fill(0.0)
            return self._kinetic_heat

        if not is_new_frame:
            return self._kinetic_heat

        self._kinetic_heat *= float(config.thermal_decay)

        if motion_mask is not None and motion_mask.shape == (height, width):
            motion_norm = (motion_mask > 0).astype(np.float32)
            smoothed_motion = cv2.GaussianBlur(motion_norm, (15, 15), 0)
            self._kinetic_heat = np.clip(
                self._kinetic_heat + smoothed_motion * float(config.kinetic_heat_gain) * 0.42,
                0.0,
                0.85,
            )
        return self._kinetic_heat

    def _extract_temperature_stats(
        self,
        temp_map_c: np.ndarray,
        width: int,
        height: int,
        config: ThermalConfig,
        roi: ROIConfig | None,
    ) -> dict[str, float | int]:
        if roi is not None and roi.enabled:
            rx, ry, rw, rh = roi.to_pixel_rect(width, height)
            sub_map = temp_map_c[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry
        else:
            sub_map = temp_map_c
            offset_x, offset_y = 0, 0

        min_val = float(np.min(sub_map))
        avg_val = float(np.mean(sub_map))
        max_val = float(np.max(sub_map))

        hot_rel_y, hot_rel_x = np.unravel_index(int(np.argmax(sub_map)), sub_map.shape)
        cold_rel_y, cold_rel_x = np.unravel_index(
            int(np.argmin(sub_map)), sub_map.shape
        )

        return {
            "min_c": round(min_val, 1),
            "avg_c": round(avg_val, 1),
            "max_c": round(max_val, 1),
            "hot_x": int(offset_x + hot_rel_x),
            "hot_y": int(offset_y + hot_rel_y),
            "cold_x": int(offset_x + cold_rel_x),
            "cold_y": int(offset_y + cold_rel_y),
        }

    def _render_colormap_and_isotherms(
        self,
        combined_norm: np.ndarray,
        temp_map_c: np.ndarray,
        config: ThermalConfig,
    ) -> np.ndarray:
        gray_u8 = np.clip(combined_norm * 255.0, 0, 255).astype(np.uint8)
        cv_colormap = PALETTE_MAP.get(config.palette, cv2.COLORMAP_INFERNO)
        colored = cv2.applyColorMap(gray_u8, cv_colormap)

        if config.show_isotherms:
            overheat_mask = (temp_map_c >= config.overheat_threshold_c).astype(np.uint8) * 255
            if np.any(overheat_mask):
                contours, _ = cv2.findContours(
                    overheat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(colored, contours, -1, (255, 255, 0), 2)

        return colored
