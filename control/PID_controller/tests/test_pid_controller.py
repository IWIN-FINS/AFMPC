from __future__ import annotations

import numpy as np
import pytest

from device_adapter import (
    FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
    FINESUB_V4_PRO1_FORCE_POSITIVE_N,
    finesub_translation_thruster_force_matrix,
)
from pid_controller import PIDConfig, RelativePIDController


def build_controller(**overrides) -> RelativePIDController:
    values = dict(
        dt=0.05,
        reference_position=(0.8, 0.0, 0.0),
        kp=(10.0, 12.0, 14.0),
        ki=(2.0, 2.0, 2.0),
        kd=(3.0, 3.0, 3.0),
        derivative_filter_time_constant=0.1,
        force_min=(-16.0, -16.0, -23.0),
        force_max=(16.0, 16.0, 29.0),
        delta_force_min=(-4.0, -4.0, -5.6),
        delta_force_max=(4.0, 4.0, 5.6),
        thruster_force_matrix=finesub_translation_thruster_force_matrix(),
        thruster_force_min=-FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        thruster_force_max=FINESUB_V4_PRO1_FORCE_POSITIVE_N,
    )
    values.update(overrides)
    return RelativePIDController(PIDConfig(**values))


def test_proportional_sign_moves_vehicle_toward_target() -> None:
    controller = build_controller(ki=(0.0, 0.0, 0.0), kd=(0.0, 0.0, 0.0))
    result = controller.update((1.0, 0.1, -0.1), (0.0, 0.0, 0.0))
    assert result.force[0] > 0.0
    assert result.force[1] > 0.0
    assert result.force[2] < 0.0


def test_first_update_has_no_derivative_kick() -> None:
    controller = build_controller(kp=(0.0, 0.0, 0.0), ki=(0.0, 0.0, 0.0))
    result = controller.update((1.2, 0.3, -0.2), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(result.derivative_term, 0.0)
    np.testing.assert_allclose(result.force, 0.0)


def test_force_rate_limit_is_respected() -> None:
    controller = build_controller(kp=(1000.0, 1000.0, 1000.0))
    result = controller.update((3.0, 2.0, 2.0), np.zeros(3))
    assert np.all(result.force <= np.array([4.0, 4.0, 5.6]) + 1e-10)
    assert np.all(result.force >= np.array([-4.0, -4.0, -5.6]) - 1e-10)


def test_every_output_satisfies_asymmetric_thruster_envelope() -> None:
    controller = build_controller(kp=(1000.0, 1000.0, 1000.0))
    previous = np.zeros(3)
    matrix = finesub_translation_thruster_force_matrix()
    for position in ((3.0, 3.0, 2.0), (-2.0, 3.0, -2.0), (3.0, -3.0, 2.0)):
        result = controller.update(position, previous)
        motor_force = matrix @ result.force
        assert np.all(motor_force <= FINESUB_V4_PRO1_FORCE_POSITIVE_N + 1e-9)
        assert np.all(motor_force >= -FINESUB_V4_PRO1_FORCE_NEGATIVE_N - 1e-9)
        previous = result.force


def test_target_loss_returns_to_latched_baseline_at_slew_limit() -> None:
    controller = build_controller()
    controller.latch_baseline((2.0, -1.0, 0.5))
    safe = controller.safe_force((10.0, 6.0, -8.0))
    np.testing.assert_allclose(safe, (6.0, 2.0, -2.4), atol=1e-10)


def test_bad_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        PIDConfig(dt=0.0).normalized()
    with pytest.raises(ValueError):
        PIDConfig(thruster_force_matrix=np.eye(3)).normalized()


def test_integrator_does_not_run_away_during_saturation() -> None:
    controller = build_controller(
        kp=(1000.0, 0.0, 0.0),
        ki=(20.0, 0.0, 0.0),
        kd=(0.0, 0.0, 0.0),
        integral_limit=(0.4, 0.4, 0.4),
    )
    previous = np.zeros(3)
    for _ in range(200):
        result = controller.update((5.0, 0.0, 0.0), previous)
        previous = result.force
    assert abs(result.integral_state[0]) <= 0.4 + 1e-12
    assert result.saturated[0]
