"""Simulador de câmera térmica infravermelha (termografia industrial)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import ROIConfig, ThermalConfig, ThermalPalette
from app.frame_utils import compute_analysis_size, downscale_for_analysis, gray_percentiles

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
    """Resultado imutável do processamento termográfico de um quadro.

    `thermal_bgr` tem o tamanho do quadro original e é None quando a imagem
    colorizada não foi solicitada (modos que não exibem termografia).
    `temp_map_c` fica na resolução de análise (largura <= ANALYSIS_MAX_WIDTH);
    as coordenadas de hotspot/coldspot já estão na escala do quadro original.
    """

    thermal_bgr: np.ndarray | None
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
        render_image: bool = True,
    ) -> ThermalResult:
        """Gera imagem térmica colorizada, mapa de temperatura (°C) e estatísticas.

        A análise roda em resolução reduzida; `render_image=False` pula a
        colorização quando o modo de visão atual não exibe a imagem térmica.
        """
        if (
            frame_bgr is None
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] <= 0
            or frame_bgr.shape[1] <= 0
        ):
            raise ValueError("Quadro inválido para simulação térmica")

        height, width = frame_bgr.shape[:2]
        small_w, small_h, factor = compute_analysis_size(width, height)
        small_size = (small_w, small_h)
        small_bgr = downscale_for_analysis(frame_bgr, small_size)
        small_hint = (
            downscale_for_analysis(heat_hint, small_size)
            if heat_hint is not None and heat_hint.shape == (height, width)
            else None
        )
        small_mask = (
            cv2.resize(motion_mask, small_size, interpolation=cv2.INTER_NEAREST)
            if motion_mask is not None and motion_mask.shape[:2] != (small_h, small_w)
            else motion_mask
        )

        base_heat = self._compute_base_emissivity(small_bgr, config, small_hint)
        kinetic_layer = self._update_kinetic_heat(
            small_h, small_w, config, small_mask, is_new_frame=is_new_frame
        )
        combined_norm = np.clip(base_heat + kinetic_layer, 0.0, 1.0)

        temp_span = config.max_temp_c - config.min_temp_c
        temp_map_c = (config.min_temp_c + combined_norm * temp_span).astype(np.float32)

        stats = self._extract_temperature_stats(temp_map_c, small_w, small_h, roi, factor)
        thermal_bgr = (
            self._render_colormap_and_isotherms(
                combined_norm, temp_map_c, config, (width, height), factor
            )
            if render_image
            else None
        )

        return ThermalResult(
            thermal_bgr=thermal_bgr,
            temp_map_c=temp_map_c,
            min_temp_c=stats["min_c"],
            avg_temp_c=stats["avg_c"],
            max_temp_c=stats["max_c"],
            hotspot_x=min(width - 1, int(stats["hot_x"])),
            hotspot_y=min(height - 1, int(stats["hot_y"])),
            coldspot_x=min(width - 1, int(stats["cold_x"])),
            coldspot_y=min(height - 1, int(stats["cold_y"])),
            overheat_alarm=bool(stats["max_c"] >= config.overheat_threshold_c),
        )

    def compute_luminance_heat(
        self, frame_bgr: np.ndarray, config: ThermalConfig
    ) -> np.ndarray:
        """Calcula o mapa base de emissividade térmica [0..1] da câmera pessoal (luminância + tons quentes)."""
        gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32) * (1.0 / 255.0)
        b_ch, g_ch, r_ch = cv2.split(frame_bgr)
        # cv2.subtract satura em 0, equivalente a clip(r - b, 0, 1) sem float intermediário.
        warmth_u8 = cv2.addWeighted(
            cv2.subtract(r_ch, b_ch), 1.0, cv2.subtract(r_ch, g_ch), 0.45, 0.0, dtype=cv2.CV_32F
        )
        warmth = warmth_u8 * (0.22 / 255.0)

        p_low, p_med, p_ref = (
            level / 255.0 for level in gray_percentiles(gray_u8, (5.0, 50.0, 92.0))
        )
        span = p_ref - p_low
        if span > 0.02 and p_med < 0.45:
            low_light_gain = float(np.clip((0.45 - p_med) / 0.35, 0.0, 1.0))
            stretched = np.clip((gray - p_low) / max(0.08, span), 0.0, 1.0) * (
                0.55 * low_light_gain
            )
            signal = np.maximum(gray, stretched) + warmth
        else:
            signal = gray + warmth

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
        roi: ROIConfig | None,
        factor: float = 1.0,
    ) -> dict[str, float | int]:
        if roi is not None and roi.enabled:
            rx, ry, rw, rh = roi.to_pixel_rect(width, height)
            sub_map = temp_map_c[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry
        else:
            sub_map = temp_map_c
            offset_x, offset_y = 0, 0

        # minMaxLoc devolve extremos e posições em uma única varredura.
        min_val, max_val, (cold_rel_x, cold_rel_y), (hot_rel_x, hot_rel_y) = cv2.minMaxLoc(
            sub_map
        )
        avg_val = float(cv2.mean(sub_map)[0])

        # Cada pixel de análise cobre um bloco fator x fator: aponta para o centro dele.
        center = (factor - 1.0) / 2.0
        return {
            "min_c": round(float(min_val), 1),
            "avg_c": round(avg_val, 1),
            "max_c": round(float(max_val), 1),
            "hot_x": round((offset_x + hot_rel_x) * factor + center),
            "hot_y": round((offset_y + hot_rel_y) * factor + center),
            "cold_x": round((offset_x + cold_rel_x) * factor + center),
            "cold_y": round((offset_y + cold_rel_y) * factor + center),
        }

    def _render_colormap_and_isotherms(
        self,
        combined_norm: np.ndarray,
        temp_map_c: np.ndarray,
        config: ThermalConfig,
        output_size: tuple[int, int],
        factor: float,
    ) -> np.ndarray:
        gray_u8 = cv2.convertScaleAbs(combined_norm, alpha=255.0)
        cv_colormap = PALETTE_MAP.get(config.palette, cv2.COLORMAP_INFERNO)
        colored = cv2.applyColorMap(gray_u8, cv_colormap)
        if (colored.shape[1], colored.shape[0]) != output_size:
            colored = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)

        if config.show_isotherms:
            _, overheat_mask = cv2.threshold(
                temp_map_c, float(config.overheat_threshold_c) - 1e-4, 255, cv2.THRESH_BINARY
            )
            overheat_u8 = overheat_mask.astype(np.uint8)
            if cv2.countNonZero(overheat_u8):
                contours, _ = cv2.findContours(
                    overheat_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if factor != 1.0:
                    contours = [
                        np.round(contour * factor).astype(np.int32) for contour in contours
                    ]
                cv2.drawContours(colored, contours, -1, (255, 255, 0), 2)

        return colored
