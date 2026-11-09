"""Testes unitários para o simulador de câmera térmica infravermelha."""

import cv2
import numpy as np
import pytest

from app.config import ROIConfig, ThermalConfig
from app.thermal_simulator import PALETTE_MAP, ThermalResult, ThermalSimulator


@pytest.mark.parametrize("palette", list(PALETTE_MAP.keys()))
def test_thermal_simulator_supports_all_palettes(palette):
    sim = ThermalSimulator()
    cfg = ThermalConfig(
        palette=palette,
        min_temp_c=20.0,
        max_temp_c=120.0,
        overheat_threshold_c=85.0,
    )
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    cv2.circle(frame, (160, 120), 30, (250, 250, 250), -1)

    res = sim.process(frame, cfg)
    assert isinstance(res, ThermalResult)
    assert res.thermal_bgr.shape == (240, 320, 3)
    assert res.thermal_bgr.dtype == np.uint8
    assert res.temp_map_c.shape == (240, 320)
    assert 20.0 <= res.min_temp_c <= res.avg_temp_c <= res.max_temp_c <= 120.0


def test_thermal_simulator_detects_hotspot_coordinates_and_overheat_alarm():
    sim = ThermalSimulator()
    cfg = ThermalConfig(
        palette="inferno",
        min_temp_c=25.0,
        max_temp_c=125.0,
        overheat_threshold_c=90.0,
        kinetic_heat_gain=0.0,
    )
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    heat_hint = np.full((240, 320), 0.1, dtype=np.float32)
    heat_hint[60, 210] = 0.98  # Ponto quente em x=210, y=60

    res = sim.process(frame, cfg, heat_hint=heat_hint)
    assert res.overheat_alarm is True
    assert res.max_temp_c >= 90.0
    assert abs(res.hotspot_x - 210) <= 2
    assert abs(res.hotspot_y - 60) <= 2


def test_thermal_simulator_kinetic_heat_accumulation_from_motion_mask():
    sim = ThermalSimulator()
    cfg = ThermalConfig(
        min_temp_c=20.0,
        max_temp_c=100.0,
        overheat_threshold_c=80.0,
        kinetic_heat_gain=0.8,
        thermal_decay=0.92,
    )
    frame = np.full((200, 200, 3), 40, dtype=np.uint8)
    motion_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(motion_mask, (70, 70), (130, 130), 255, -1)

    res_no_motion = sim.process(frame, cfg)
    base_max = res_no_motion.max_temp_c

    res_motion = sim.process(frame, cfg, motion_mask=motion_mask)
    assert res_motion.max_temp_c > base_max

    # Ao zerar kinetic_heat_gain, o calor cinético residual deve ser limpo imediatamente
    cfg_zero_kinetic = ThermalConfig(
        min_temp_c=20.0,
        max_temp_c=100.0,
        overheat_threshold_c=80.0,
        kinetic_heat_gain=0.0,
    )
    res_after_zero = sim.process(frame, cfg_zero_kinetic, motion_mask=motion_mask)
    assert res_after_zero.max_temp_c == pytest.approx(base_max, abs=0.1)


def test_thermal_simulator_roi_restricts_hotspot_evaluation():
    sim = ThermalSimulator()
    cfg = ThermalConfig(min_temp_c=20.0, max_temp_c=120.0, overheat_threshold_c=95.0)
    roi = ROIConfig(enabled=True, x=0.5, y=0.5, w=0.45, h=0.45)

    frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    heat_hint = np.full((200, 200), 0.15, dtype=np.float32)
    # Hotspot crítico fora da ROI (canto superior esquerdo)
    heat_hint[20, 20] = 0.99
    # Temperatura moderada dentro da ROI (canto inferior direito)
    heat_hint[150, 160] = 0.50

    res = sim.process(frame, cfg, heat_hint=heat_hint, roi=roi)
    assert res.overheat_alarm is False
    assert res.hotspot_x >= 100 and res.hotspot_y >= 100


def test_thermal_simulator_handles_resolution_change():
    sim = ThermalSimulator()
    cfg = ThermalConfig()
    f1 = np.zeros((240, 320, 3), dtype=np.uint8)
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)

    r1 = sim.process(f1, cfg)
    r2 = sim.process(f2, cfg)
    assert r1.thermal_bgr.shape == (240, 320, 3)
    assert r2.thermal_bgr.shape == (480, 640, 3)


def test_thermal_simulator_reset_and_invalid_frame():
    sim = ThermalSimulator()
    cfg = ThermalConfig()
    sim.reset()
    with pytest.raises(ValueError, match="Quadro inválido"):
        sim.process(np.zeros((0, 0, 3), dtype=np.uint8), cfg)


def test_thermal_simulator_low_light_webcam_and_skin_tone_response():
    sim = ThermalSimulator()
    cfg = ThermalConfig(
        min_temp_c=20.0,
        max_temp_c=42.0,
        overheat_threshold_c=37.5,
        kinetic_heat_gain=0.0,
        emissivity_contrast=1.35,
    )
    # Simula quadro escuro de webcam interna (valores entre 10 e 90) com região de tom quente (pele: R > B)
    frame = np.full((200, 240, 3), 18, dtype=np.uint8)
    frame[60:140, 80:160] = (35, 55, 95)  # BGR com canal vermelho mais alto

    res = sim.process(frame, cfg)
    face_temp_base = float(res.temp_map_c[100, 120])
    assert res.max_temp_c > 30.0
    assert 80 <= res.hotspot_x <= 160
    assert 60 <= res.hotspot_y <= 140

    # Adicionar janela/lâmpada brilhante de 3% no fundo (backlighting) não deve derrubar a temperatura do rosto
    frame_backlit = frame.copy()
    frame_backlit[5:35, 5:55] = (245, 245, 245)
    res_backlit = sim.process(frame_backlit, cfg)
    face_temp_backlit = float(res_backlit.temp_map_c[100, 120])
    assert face_temp_backlit == pytest.approx(face_temp_base, abs=1.0)
