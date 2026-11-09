"""Testes unitários para o motor de detecção de movimento."""

import cv2
import numpy as np
import pytest

from app.config import MotionConfig, ROIConfig
from app.motion_detector import MotionDetector, MotionResult


def _make_blank_frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 40, dtype=np.uint8)


def test_motion_detector_first_frame_has_no_motion():
    detector = MotionDetector()
    cfg = MotionConfig(algorithm="frame_diff", sensitivity=70, min_area=100)
    frame = _make_blank_frame()

    result = detector.process(frame, cfg)
    assert isinstance(result, MotionResult)
    assert result.motion_detected is False
    assert result.alarm_triggered is False
    assert result.object_count == 0
    assert result.motion_percentage == 0.0
    assert result.motion_mask.shape == (240, 320)


@pytest.mark.parametrize("algorithm", ["frame_diff", "mog2", "knn"])
def test_motion_detector_detects_moving_rectangle_for_all_algorithms(algorithm):
    detector = MotionDetector()
    cfg = MotionConfig(
        algorithm=algorithm,
        sensitivity=75,
        min_area=120,
        alarm_motion_pct=1.0,
    )

    bg = _make_blank_frame(320, 240)
    for _ in range(4):
        static_res = detector.process(bg, cfg)
        assert static_res.motion_detected is False
        assert static_res.motion_percentage == 0.0

    moved = bg.copy()
    cv2.rectangle(moved, (100, 70), (180, 150), (240, 240, 240), -1)
    result = detector.process(moved, cfg)

    assert result.motion_detected is True
    assert result.object_count >= 1
    assert 1.0 < result.motion_percentage < 40.0
    assert result.alarm_triggered is True
    assert len(result.bounding_boxes) >= 1
    box = result.bounding_boxes[0]
    assert box.w >= 40 and box.h >= 40

    # Quadro repetido em cache (is_new_frame=False) deve preservar a detecção sem zerar MOG2/KNN
    cached_res = detector.process(moved, cfg, is_new_frame=False)
    assert cached_res.motion_detected is True
    assert cached_res.object_count == result.object_count
    assert cached_res.motion_percentage == result.motion_percentage


def test_motion_detector_handles_resolution_change_gracefully():
    detector = MotionDetector()
    cfg = MotionConfig(algorithm="frame_diff", sensitivity=70, min_area=100)

    frame_small = _make_blank_frame(320, 240)
    detector.process(frame_small, cfg)

    frame_large = _make_blank_frame(640, 480)
    res = detector.process(frame_large, cfg)
    assert res.motion_mask.shape == (480, 640)
    assert res.motion_detected is False


def test_motion_detector_respects_roi_masking():
    detector = MotionDetector()
    cfg = MotionConfig(algorithm="frame_diff", sensitivity=75, min_area=80)
    roi = ROIConfig(enabled=True, x=0.5, y=0.5, w=0.45, h=0.45)

    bg = _make_blank_frame(400, 300)
    detector.process(bg, cfg, roi)

    # Objeto fora da ROI (canto superior esquerdo)
    outside = bg.copy()
    cv2.rectangle(outside, (20, 20), (90, 90), (250, 250, 250), -1)
    res_outside = detector.process(outside, cfg, roi)
    assert res_outside.motion_detected is False
    assert res_outside.object_count == 0

    # Objeto dentro da ROI (canto inferior direito)
    inside = outside.copy()
    cv2.rectangle(inside, (240, 180), (320, 250), (250, 250, 250), -1)
    res_inside = detector.process(inside, cfg, roi)
    assert res_inside.motion_detected is True
    assert res_inside.object_count == 1


def test_motion_detector_disabled_returns_empty_result():
    detector = MotionDetector()
    cfg = MotionConfig(enabled=False)
    frame = _make_blank_frame(320, 240)
    detector.process(frame, cfg)

    moved = frame.copy()
    cv2.rectangle(moved, (50, 50), (200, 200), (255, 255, 255), -1)
    res = detector.process(moved, cfg)
    assert res.motion_detected is False
    assert res.object_count == 0
    assert res.motion_percentage == 0.0


def test_motion_detector_rejects_invalid_frame():
    detector = MotionDetector()
    cfg = MotionConfig()
    with pytest.raises(ValueError, match="Quadro inválido"):
        detector.process(np.zeros((0, 0, 3), dtype=np.uint8), cfg)
