"""Testes unitários para o módulo de configuração imutável."""

import pytest
from pydantic import ValidationError

from app.config import (
    FPS_PRESETS,
    RESOLUTION_PRESETS,
    MotionConfig,
    ROIConfig,
    SystemConfig,
    ThermalConfig,
)


def test_default_system_config_is_valid_and_immutable():
    cfg = SystemConfig()
    assert cfg.width == 640
    assert cfg.height == 480
    assert cfg.target_fps == 24
    assert cfg.vision_mode == "normal"
    assert cfg.source == "camera"
    assert cfg.mirror is True

    with pytest.raises(ValidationError):
        cfg.width = 1280  # type: ignore[misc]


def test_with_resolution_returns_new_instance_without_mutating_original():
    original = SystemConfig()
    updated = original.with_resolution(1280, 720)

    assert original.width == 640
    assert original.height == 480
    assert updated.width == 1280
    assert updated.height == 720
    assert updated is not original


def test_with_preset_resolution_valid_and_invalid():
    cfg = SystemConfig()
    for preset_name, (expected_w, expected_h) in RESOLUTION_PRESETS.items():
        updated = cfg.with_preset_resolution(preset_name)
        assert updated.width == expected_w
        assert updated.height == expected_h

    with pytest.raises(ValueError, match="Preset de resolução inválido"):
        cfg.with_preset_resolution("9999x9999")


def test_with_fps_valid_and_boundary_values():
    cfg = SystemConfig()
    for fps in FPS_PRESETS:
        updated = cfg.with_fps(fps)
        assert updated.target_fps == fps

    assert cfg.with_fps(1).target_fps == 1
    assert cfg.with_fps(60).target_fps == 60

    with pytest.raises(ValidationError):
        cfg.with_fps(0)

    with pytest.raises(ValidationError):
        cfg.with_fps(61)


def test_resolution_bounds_validation():
    with pytest.raises(ValidationError):
        SystemConfig(width=100, height=480)

    with pytest.raises(ValidationError):
        SystemConfig(width=640, height=80)

    with pytest.raises(ValidationError):
        SystemConfig(width=5000, height=480)


def test_roi_validation_rules():
    valid_roi = ROIConfig(enabled=True, x=0.1, y=0.2, w=0.5, h=0.6)
    assert valid_roi.to_pixel_rect(1000, 500) == (100, 100, 500, 300)

    # ROI excedendo os limites normalizados (x + w > 1.0)
    with pytest.raises(ValidationError):
        ROIConfig(enabled=True, x=0.6, y=0.1, w=0.5, h=0.5)

    with pytest.raises(ValidationError):
        ROIConfig(enabled=True, x=0.1, y=0.7, w=0.5, h=0.5)


def test_motion_config_normalizes_even_blur_size_and_validates_sensitivity():
    mc = MotionConfig(blur_size=6, sensitivity=80)
    assert mc.blur_size == 7
    assert mc.sensitivity == 80

    with pytest.raises(ValidationError):
        MotionConfig(sensitivity=0)

    with pytest.raises(ValidationError):
        MotionConfig(sensitivity=101)


def test_thermal_config_temperature_validation():
    tc = ThermalConfig(min_temp_c=20.0, max_temp_c=150.0, overheat_threshold_c=95.0)
    assert tc.min_temp_c == 20.0
    assert tc.max_temp_c == 150.0
    assert tc.overheat_threshold_c == 95.0

    # max_temp_c menor ou igual a min_temp_c deve falhar
    with pytest.raises(ValidationError):
        ThermalConfig(min_temp_c=100.0, max_temp_c=80.0, overheat_threshold_c=90.0)

    # overheat_threshold_c fora da faixa [min_temp_c, max_temp_c] deve falhar
    with pytest.raises(ValidationError):
        ThermalConfig(min_temp_c=20.0, max_temp_c=100.0, overheat_threshold_c=120.0)


def test_apply_patch_nested_immutability():
    cfg = SystemConfig()
    patched = cfg.apply_patch(
        {
            "width": 800,
            "height": 600,
            "target_fps": 30,
            "vision_mode": "thermal",
            "motion": {"sensitivity": 85, "algorithm": "mog2"},
            "thermal": {"palette": "jet", "overheat_threshold_c": 90.0},
            "roi": {"enabled": True, "x": 0.15, "y": 0.15, "w": 0.7, "h": 0.7},
            "simulator": {"scenario": "boiler", "speed_multiplier": 1.5},
        }
    )

    assert cfg.width == 640
    assert cfg.vision_mode == "normal"
    assert cfg.motion.sensitivity == 65
    assert cfg.thermal.palette == "inferno"

    assert patched.width == 800
    assert patched.height == 600
    assert patched.target_fps == 30
    assert patched.vision_mode == "thermal"
    assert patched.motion.sensitivity == 85
    assert patched.motion.algorithm == "mog2"
    assert patched.thermal.palette == "jet"
    assert patched.thermal.overheat_threshold_c == 90.0
    assert patched.roi.enabled is True
    assert patched.simulator.scenario == "boiler"
    assert patched.simulator.speed_multiplier == 1.5


def test_apply_patch_clamps_implicit_overheat_threshold_and_rejects_bad_preset_type():
    cfg = SystemConfig()  # min=22, max=120, overheat=85
    # Reduzir max_temp_c para 70 sem especificar overheat_threshold_c deve ajustar o limiar para 70
    lowered = cfg.apply_patch({"thermal": {"max_temp_c": 70.0}})
    assert lowered.thermal.max_temp_c == 70.0
    assert lowered.thermal.overheat_threshold_c == 70.0

    # Elevar min_temp_c para 95 sem especificar overheat_threshold_c deve ajustar o limiar para 95
    raised = cfg.apply_patch({"thermal": {"min_temp_c": 95.0}})
    assert raised.thermal.min_temp_c == 95.0
    assert raised.thermal.overheat_threshold_c == 95.0

    # Tipo inválido para resolution_preset deve levantar ValueError
    with pytest.raises(ValueError, match="Preset de resolução inválido"):
        cfg.apply_patch({"resolution_preset": 12345})

    # Aplicar perfil térmico corporal (human) e validar rejeição de perfil inválido
    human_cfg = cfg.apply_patch({"thermal_profile": "human", "mirror": False})
    assert human_cfg.mirror is False
    assert human_cfg.thermal.min_temp_c == 20.0
    assert human_cfg.thermal.max_temp_c == 42.0
    assert human_cfg.thermal.overheat_threshold_c == 37.5

    with pytest.raises(ValueError, match="Perfil térmico inválido"):
        cfg.apply_patch({"thermal_profile": "inexistente"})
