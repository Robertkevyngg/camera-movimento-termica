"""Testes unitários para o simulador de cenários industriais."""

import numpy as np
import pytest

from app.config import SimulatorConfig
from app.industrial_sim import IndustrialSceneSimulator


@pytest.mark.parametrize("scenario", ["conveyor", "machinery", "perimeter", "boiler"])
def test_simulator_generates_frames_for_all_scenarios_and_resolutions(scenario):
    sim = IndustrialSceneSimulator()
    cfg = SimulatorConfig(scenario=scenario, speed_multiplier=1.2, noise_level=1.0)

    for width, height in [(320, 240), (640, 480), (1280, 720)]:
        frame, heat_hint = sim.generate_frame(width=width, height=height, config=cfg)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (height, width, 3)
        assert frame.dtype == np.uint8
        assert isinstance(heat_hint, np.ndarray)
        assert heat_hint.shape == (height, width)
        assert heat_hint.dtype == np.float32
        assert 0.0 <= float(heat_hint.min()) <= float(heat_hint.max()) <= 1.0


def test_simulator_consecutive_frames_have_motion():
    sim = IndustrialSceneSimulator()
    cfg = SimulatorConfig(scenario="conveyor", speed_multiplier=2.0, noise_level=0.0)

    frame1, _ = sim.generate_frame(640, 480, cfg)
    frame2, _ = sim.generate_frame(640, 480, cfg)

    diff = np.abs(frame2.astype(np.int16) - frame1.astype(np.int16))
    assert diff.sum() > 0


def test_simulator_thermal_spike_anomaly_increases_heat_hint():
    sim = IndustrialSceneSimulator()
    cfg = SimulatorConfig(scenario="boiler", noise_level=0.0, ambient_heat_factor=0.2)

    _, baseline_heat = sim.generate_frame(640, 480, cfg)
    baseline_max = float(baseline_heat.max())

    sim.trigger_anomaly("thermal_spike", duration_frames=10)
    assert sim.has_active_anomaly("thermal_spike")

    _, spike_heat = sim.generate_frame(640, 480, cfg)
    assert float(spike_heat.max()) >= max(0.92, baseline_max)

    sim.trigger_anomaly("clear")
    assert not sim.has_active_anomaly("thermal_spike")


def test_simulator_intrusion_anomaly_creates_large_moving_object():
    sim = IndustrialSceneSimulator()
    cfg = SimulatorConfig(scenario="perimeter", noise_level=0.0)

    base_frame, _ = sim.generate_frame(640, 480, cfg)
    sim.trigger_anomaly("intrusion", duration_frames=15)
    assert sim.has_active_anomaly("intrusion")

    intrusion_frame, _ = sim.generate_frame(640, 480, cfg)
    diff_pixels = np.count_nonzero(
        np.abs(intrusion_frame.astype(np.int16) - base_frame.astype(np.int16)).max(axis=2) > 25
    )
    assert diff_pixels > 1500


def test_simulator_invalid_anomaly_raises_value_error():
    sim = IndustrialSceneSimulator()
    with pytest.raises(ValueError, match="Tipo de anomalia inválido"):
        sim.trigger_anomaly("invalid_type")  # type: ignore[arg-type]
