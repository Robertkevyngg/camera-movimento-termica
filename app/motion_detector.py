"""Motor de detecção de movimento com suporte a múltiplos algoritmos e ROI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import BoundingBox, MotionAlgorithm, MotionConfig, ROIConfig

MAX_TRAIL_POINTS = 32


@dataclass(frozen=True)
class MotionResult:
    """Resultado imutável do processamento de movimento de um quadro."""

    motion_detected: bool
    alarm_triggered: bool
    motion_percentage: float
    object_count: int
    bounding_boxes: tuple[BoundingBox, ...]
    trails: tuple[tuple[int, int], ...]
    motion_mask: np.ndarray


class MotionDetector:
    """Detector de movimento configurável para ambientes industriais."""

    def __init__(self) -> None:
        self._bg_float: np.ndarray | None = None
        self._subtractor: cv2.BackgroundSubtractor | None = None
        self._active_algorithm: MotionAlgorithm | None = None
        self._active_sensitivity: int | None = None
        self._last_shape: tuple[int, int] | None = None
        self._warmup_done: bool = False
        self._last_raw_mask: np.ndarray | None = None
        self._trails: deque[tuple[int, int]] = deque(maxlen=MAX_TRAIL_POINTS)

    def reset(self) -> None:
        """Reinicia o modelo de fundo e o histórico de trilhas."""
        self._bg_float = None
        self._subtractor = None
        self._last_shape = None
        self._warmup_done = False
        self._last_raw_mask = None
        self._trails.clear()

    def process(
        self,
        frame_bgr: np.ndarray,
        config: MotionConfig,
        roi: ROIConfig | None = None,
        *,
        is_new_frame: bool = True,
    ) -> MotionResult:
        """Processa um quadro BGR e retorna métricas, caixas e máscara de movimento."""
        if (
            frame_bgr is None
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] <= 0
            or frame_bgr.shape[1] <= 0
        ):
            raise ValueError("Quadro inválido para detecção de movimento")

        height, width = frame_bgr.shape[:2]
        if not config.enabled:
            return self._empty_result(width, height)

        self._ensure_model_state(width, height, config)
        if (
            not is_new_frame
            and self._last_raw_mask is not None
            and self._last_raw_mask.shape == (height, width)
        ):
            raw_mask = self._last_raw_mask
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (config.blur_size, config.blur_size), 0)
            raw_mask = self._compute_foreground_mask(blurred, config)
            self._last_raw_mask = raw_mask

        filtered_mask, active_area = self._apply_roi_mask(raw_mask, width, height, roi)
        boxes = self._extract_bounding_boxes(filtered_mask, config.min_area)

        if boxes and is_new_frame:
            largest = boxes[0]
            pt = (largest.center_x, largest.center_y)
            if not self._trails or self._trails[-1] != pt:
                self._trails.append(pt)

        motion_pixels = int(np.count_nonzero(filtered_mask))
        motion_pct = round((motion_pixels / max(1, active_area)) * 100.0, 2)
        motion_detected = len(boxes) > 0
        alarm_triggered = motion_detected and (motion_pct >= config.alarm_motion_pct)

        return MotionResult(
            motion_detected=motion_detected,
            alarm_triggered=alarm_triggered,
            motion_percentage=motion_pct,
            object_count=len(boxes),
            bounding_boxes=boxes,
            trails=tuple(self._trails) if config.show_trails else (),
            motion_mask=filtered_mask,
        )

    def _empty_result(self, width: int, height: int) -> MotionResult:
        return MotionResult(
            motion_detected=False,
            alarm_triggered=False,
            motion_percentage=0.0,
            object_count=0,
            bounding_boxes=(),
            trails=(),
            motion_mask=np.zeros((height, width), dtype=np.uint8),
        )

    def _ensure_model_state(
        self, width: int, height: int, config: MotionConfig
    ) -> None:
        shape = (height, width)
        if (
            self._last_shape != shape
            or self._active_algorithm != config.algorithm
            or self._active_sensitivity != config.sensitivity
        ):
            self._bg_float = None
            self._warmup_done = False
            self._last_raw_mask = None
            self._last_shape = shape
            self._active_algorithm = config.algorithm
            self._active_sensitivity = config.sensitivity
            self._subtractor = self._create_subtractor(config)

    def _create_subtractor(
        self, config: MotionConfig
    ) -> cv2.BackgroundSubtractor | None:
        if config.algorithm == "mog2":
            var_threshold = max(6.0, 62.0 - config.sensitivity * 0.52)
            return cv2.createBackgroundSubtractorMOG2(
                history=120, varThreshold=var_threshold, detectShadows=False
            )
        if config.algorithm == "knn":
            dist2_threshold = max(100.0, 900.0 - config.sensitivity * 7.5)
            return cv2.createBackgroundSubtractorKNN(
                history=120, dist2Threshold=dist2_threshold, detectShadows=False
            )
        return None

    def _compute_foreground_mask(
        self, blurred_gray: np.ndarray, config: MotionConfig
    ) -> np.ndarray:
        if config.algorithm == "frame_diff":
            if self._bg_float is None:
                self._bg_float = blurred_gray.astype(np.float32)
                return np.zeros_like(blurred_gray, dtype=np.uint8)

            bg_uint8 = cv2.convertScaleAbs(self._bg_float)
            frame_delta = cv2.absdiff(blurred_gray, bg_uint8)
            cv2.accumulateWeighted(blurred_gray, self._bg_float, 0.08)

            scene_scale = min(1.0, max(0.55, float(np.percentile(blurred_gray, 75)) / 140.0))
            base_thresh = max(5, int(65 - config.sensitivity * 0.56))
            thresh_val = max(
                8 if config.sensitivity < 90 else 5,
                round(base_thresh * scene_scale),
            )
            _, mask = cv2.threshold(frame_delta, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            assert self._subtractor is not None
            if not self._warmup_done:
                warmup_iters = 4 if config.algorithm == "knn" else 1
                for _ in range(warmup_iters):
                    self._subtractor.apply(blurred_gray, learningRate=0.5)
                self._warmup_done = True
                return np.zeros_like(blurred_gray, dtype=np.uint8)
            fg = self._subtractor.apply(blurred_gray)
            _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        dilated = cv2.dilate(opened, kernel_dilate, iterations=2)
        return dilated

    def _apply_roi_mask(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
        roi: ROIConfig | None,
    ) -> tuple[np.ndarray, int]:
        if roi is None or not roi.enabled:
            return mask, width * height

        rx, ry, rw, rh = roi.to_pixel_rect(width, height)
        roi_filtered = np.zeros_like(mask)
        roi_filtered[ry : ry + rh, rx : rx + rw] = mask[ry : ry + rh, rx : rx + rw]
        return roi_filtered, max(1, rw * rh)

    def _extract_bounding_boxes(
        self, mask: np.ndarray, min_area: int
    ) -> tuple[BoundingBox, ...]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: list[BoundingBox] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(
                BoundingBox(
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    area=round(area, 1),
                    center_x=int(x + w // 2),
                    center_y=int(y + h // 2),
                )
            )
        boxes.sort(key=lambda b: b.area, reverse=True)
        return tuple(boxes)
