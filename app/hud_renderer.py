"""Compositor de modos de visão e renderizador de HUD industrial SCADA."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AlertLevel, BoundingBox, SystemConfig
from app.motion_detector import MotionResult
from app.thermal_simulator import PALETTE_MAP, ThermalResult

FONT = cv2.FONT_HERSHEY_SIMPLEX


class HudRenderer:
    """Renderiza camadas de visão (RGB, Térmica, Híbrida, Máscara) e telemetria HUD."""

    def render(
        self,
        raw_bgr: np.ndarray,
        motion: MotionResult,
        thermal: ThermalResult,
        config: SystemConfig,
        actual_fps: float,
        alert_level: AlertLevel,
    ) -> np.ndarray:
        """Compõe o quadro final sem mutar os arrays de entrada."""
        base = self._compose_base_frame(raw_bgr, motion, thermal, config)
        if not config.hud_enabled:
            return base

        canvas = base.copy()
        height, width = canvas.shape[:2]

        if config.roi.enabled:
            self._draw_roi_zone(canvas, width, height, config)

        if config.motion.enabled and config.motion.show_boxes:
            self._draw_motion_boxes(canvas, motion.bounding_boxes, motion.trails)

        if (
            config.vision_mode in ("thermal", "hybrid") or thermal.overheat_alarm
        ) and config.thermal.show_hotspots:
            self._draw_thermal_markers(canvas, thermal)

        if config.vision_mode in ("thermal", "hybrid") and config.thermal.show_scale_bar:
            self._draw_thermal_scale_bar(canvas, width, height, config)

        self._draw_top_status_bar(
            canvas, width, height, motion, thermal, config, actual_fps, alert_level
        )
        return canvas

    def _compose_base_frame(
        self,
        raw_bgr: np.ndarray,
        motion: MotionResult,
        thermal: ThermalResult,
        config: SystemConfig,
    ) -> np.ndarray:
        mode = config.vision_mode
        if mode == "thermal":
            return thermal.thermal_bgr.copy()
        if mode == "hybrid":
            blended = cv2.addWeighted(raw_bgr, 0.35, thermal.thermal_bgr, 0.65, 0)
            if np.any(motion.motion_mask):
                contours, _ = cv2.findContours(
                    motion.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(blended, contours, -1, (120, 255, 120), 2)
            return blended
        if mode == "motion_mask":
            dark_bg = (raw_bgr.astype(np.float32) * 0.22).astype(np.uint8)
            mask_color = np.zeros_like(raw_bgr)
            mask_color[:, :, 1] = motion.motion_mask
            mask_color[:, :, 2] = motion.motion_mask // 3
            return cv2.add(dark_bg, mask_color)
        return raw_bgr.copy()

    def _draw_roi_zone(
        self, canvas: np.ndarray, width: int, height: int, config: SystemConfig
    ) -> None:
        rx, ry, rw, rh = config.roi.to_pixel_rect(width, height)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 210, 40), 2)
        cv2.putText(
            canvas,
            "ZONA ROI MONITORADA",
            (rx + 6, max(44, ry + 18)),
            FONT,
            0.45,
            (255, 215, 60),
            1,
            cv2.LINE_AA,
        )

    def _draw_motion_boxes(
        self,
        canvas: np.ndarray,
        boxes: tuple[BoundingBox, ...],
        trails: tuple[tuple[int, int], ...],
    ) -> None:
        if len(trails) >= 2:
            pts = np.array(trails, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(60, 235, 255), thickness=2)
            end_pt = trails[-1]
            start_pt = trails[-2]
            for prev_pt in reversed(trails[:-1]):
                if (prev_pt[0] - end_pt[0]) ** 2 + (prev_pt[1] - end_pt[1]) ** 2 >= 16:
                    start_pt = prev_pt
                    break
            cv2.arrowedLine(
                canvas,
                start_pt,
                end_pt,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.35,
            )

        for idx, box in enumerate(boxes, start=1):
            x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
            color = (50, 230, 90) if idx > 1 else (40, 255, 140)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.circle(canvas, (box.center_x, box.center_y), 3, (0, 255, 255), -1)
            label = f"MOV #{idx} ({int(box.area)}px2)"
            cv2.putText(
                canvas,
                label,
                (x0, max(42, y0 - 6)),
                FONT,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_thermal_markers(
        self, canvas: np.ndarray, thermal: ThermalResult
    ) -> None:
        hx, hy = thermal.hotspot_x, thermal.hotspot_y
        hot_color = (40, 40, 255) if thermal.overheat_alarm else (0, 215, 255)
        cv2.circle(canvas, (hx, hy), 9, hot_color, 2)
        cv2.line(canvas, (hx - 14, hy), (hx + 14, hy), hot_color, 1)
        cv2.line(canvas, (hx, hy - 14), (hx, hy + 14), hot_color, 1)
        cv2.putText(
            canvas,
            f"MAX {thermal.max_temp_c:.1f}C",
            (hx + 12, max(45, hy - 6)),
            FONT,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cx, cy = thermal.coldspot_x, thermal.coldspot_y
        cv2.circle(canvas, (cx, cy), 5, (255, 190, 60), 1)
        cv2.putText(
            canvas,
            f"MIN {thermal.min_temp_c:.1f}C",
            (cx + 8, min(canvas.shape[0] - 10, cy + 14)),
            FONT,
            0.38,
            (255, 210, 120),
            1,
            cv2.LINE_AA,
        )

    def _draw_thermal_scale_bar(
        self, canvas: np.ndarray, width: int, height: int, config: SystemConfig
    ) -> None:
        bar_w = max(12, min(20, width // 36))
        desired_h = max(80, int(height * 0.52))
        y0 = max(36, (height - desired_h) // 2)
        bar_h = max(16, min(desired_h, height - y0 - 6))
        x0 = max(4, width - bar_w - 14)

        grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape((bar_h, 1))
        grad_strip = np.repeat(grad, bar_w, axis=1)
        cmap = PALETTE_MAP.get(config.thermal.palette, cv2.COLORMAP_INFERNO)
        colored_strip = cv2.applyColorMap(grad_strip, cmap)

        canvas[y0 : y0 + bar_h, x0 : x0 + bar_w] = colored_strip
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (220, 225, 230), 1)

        cv2.putText(
            canvas,
            f"{config.thermal.max_temp_c:.0f}C",
            (max(4, x0 - 42), y0 + 10),
            FONT,
            0.38,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{config.thermal.min_temp_c:.0f}C",
            (max(4, x0 - 42), y0 + bar_h),
            FONT,
            0.38,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _draw_top_status_bar(
        self,
        canvas: np.ndarray,
        width: int,
        height: int,
        motion: MotionResult,
        thermal: ThermalResult,
        config: SystemConfig,
        actual_fps: float,
        alert_level: AlertLevel,
    ) -> None:
        bar_h = 30
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_h), (16, 20, 26), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, dst=canvas)

        src_tag = f"CAM#{config.camera_index}" if config.source == "camera" else "SIM"
        summary = (
            f"{src_tag} | MODO:{config.vision_mode.upper()} | "
            f"{width}x{height} @ {actual_fps:.1f}/{config.target_fps}FPS | "
            f"MOV:{motion.motion_percentage:.1f}% ({motion.object_count}) | "
            f"TERM:{thermal.max_temp_c:.1f}C"
        )
        cv2.putText(
            canvas,
            summary,
            (10, 20),
            FONT,
            0.42 if width >= 600 else 0.33,
            (230, 238, 245),
            1,
            cv2.LINE_AA,
        )

        if alert_level == "CRITICO":
            cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (35, 35, 245), 3)
        elif alert_level == "ATENCAO":
            cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (30, 185, 250), 2)
