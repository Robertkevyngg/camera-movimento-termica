"""Testes unitários para o compositor visual e HUD industrial."""

import numpy as np
import pytest

from app.config import BoundingBox, ROIConfig, SystemConfig
from app.hud_renderer import HudRenderer
from app.motion_detector import MotionResult
from app.thermal_simulator import ThermalResult


def _sample_inputs(width: int = 640, height: int = 480):
    raw_bgr = np.full((height, width, 3), 50, dtype=np.uint8)
    thermal_bgr = np.full((height, width, 3), (20, 80, 220), dtype=np.uint8)
    temp_map = np.full((height, width), 45.0, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:180, 120:220] = 255

    motion = MotionResult(
        motion_detected=True,
        alarm_triggered=True,
        motion_percentage=4.2,
        object_count=1,
        bounding_boxes=(
            BoundingBox(
                x=120, y=100, w=100, h=80, area=8000.0, center_x=170, center_y=140
            ),
        ),
        trails=((150, 130), (160, 135), (170, 140)),
        motion_mask=mask,
    )
    thermal = ThermalResult(
        thermal_bgr=thermal_bgr,
        temp_map_c=temp_map,
        min_temp_c=24.5,
        avg_temp_c=45.0,
        max_temp_c=92.3,
        hotspot_x=170,
        hotspot_y=140,
        coldspot_x=30,
        coldspot_y=30,
        overheat_alarm=True,
    )
    return raw_bgr, motion, thermal


@pytest.mark.parametrize("mode", ["normal", "thermal", "hybrid", "motion_mask"])
def test_hud_renderer_renders_all_vision_modes_without_mutating_inputs(mode):
    renderer = HudRenderer()
    raw_bgr, motion, thermal = _sample_inputs()
    raw_copy = raw_bgr.copy()
    thermal_copy = thermal.thermal_bgr.copy()

    cfg = SystemConfig(
        vision_mode=mode,
        hud_enabled=True,
        roi=ROIConfig(enabled=True, x=0.1, y=0.1, w=0.8, h=0.8),
    )

    out = renderer.render(
        raw_bgr=raw_bgr,
        motion=motion,
        thermal=thermal,
        config=cfg,
        actual_fps=24.0,
        alert_level="CRITICO",
    )

    assert out.shape == raw_bgr.shape
    assert out.dtype == np.uint8
    assert out is not raw_bgr
    assert np.array_equal(raw_bgr, raw_copy)
    assert np.array_equal(thermal.thermal_bgr, thermal_copy)


def test_hud_renderer_respects_hud_disabled_flag():
    renderer = HudRenderer()
    raw_bgr, motion, thermal = _sample_inputs(320, 240)
    cfg_no_hud = SystemConfig(
        width=320, height=240, vision_mode="normal", hud_enabled=False
    )
    cfg_with_hud = SystemConfig(
        width=320, height=240, vision_mode="normal", hud_enabled=True
    )

    out_clean = renderer.render(raw_bgr, motion, thermal, cfg_no_hud, 24.0, "NORMAL")
    out_hud = renderer.render(raw_bgr, motion, thermal, cfg_with_hud, 24.0, "CRITICO")

    assert np.array_equal(out_clean, raw_bgr)
    assert not np.array_equal(out_hud, raw_bgr)


def test_hud_renderer_handles_small_frame_scale_bar_and_atencao_alert():
    renderer = HudRenderer()
    raw_bgr, motion, thermal = _sample_inputs(160, 100)
    cfg = SystemConfig(
        width=160,
        height=120,
        vision_mode="thermal",
        hud_enabled=True,
    )
    out = renderer.render(raw_bgr, motion, thermal, cfg, 15.0, "ATENCAO")
    assert out.shape == (100, 160, 3)
