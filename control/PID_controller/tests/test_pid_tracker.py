from __future__ import annotations

import numpy as np

from camera_transform import camera_to_body_position
from live_integration_example import build_tracker, one_control_update


def test_camera_coordinate_order() -> None:
    np.testing.assert_allclose(
        camera_to_body_position((0.2, -0.1, 1.3)),
        (1.3, 0.2, -0.1),
    )


def test_tracker_output_matches_existing_control_loop_shape() -> None:
    tracker = build_tracker()
    tracker.latch_baseline(np.zeros(3))
    output = one_control_update(tracker, (0.1, -0.05, 1.2), np.zeros(3))
    assert output.pid.force.shape == (3,)
    assert output.measured_state.shape == (6,)
    assert output.thruster_allocation is not None
    assert output.thruster_allocation.throttles.shape == (8,)
    assert isinstance(output.device_command.planar_forward, int)


def test_control_helper_inherits_calibrated_frame_from_hardware_tracker() -> None:
    tracker = build_tracker(calibrated_reference=True)
    tracker.latch_baseline(np.zeros(3))
    output = one_control_update(tracker, (0.0, 0.0, 0.60), np.zeros(3))
    np.testing.assert_allclose(output.measured_state[:3], (0.8576342, -0.055545, -0.1208148), atol=2.0e-6)


def test_default_tracker_uses_conservative_experiment_envelope() -> None:
    tracker = build_tracker()
    config = tracker.controller.config
    np.testing.assert_allclose(config.force_max, (4.730162, 4.997534, 7.06314))
    np.testing.assert_allclose(config.delta_force_max, (1.2, 0.8, 1.0))
    np.testing.assert_allclose(
        tracker.thruster_allocator.translation_channel_limits,
        (0.20, 0.20, 0.20),
    )


def test_hardware_tracker_selects_calibrated_camera_reference() -> None:
    tracker = build_tracker(calibrated_reference=True)
    assert tracker.camera_calibrated
    np.testing.assert_allclose(
        tracker.controller.config.reference_position,
        (0.857634, -0.055545, -0.120815),
    )


def test_no_model_or_optimizer_is_present() -> None:
    tracker = build_tracker()
    assert not hasattr(tracker, "model")
    assert not hasattr(tracker.controller, "solver")
    assert not hasattr(tracker.controller, "horizon")


def test_closed_loop_converges_in_nominal_plant() -> None:
    tracker = build_tracker()
    dt = tracker.controller.config.dt
    position = np.array([1.4, 0.25, -0.18])
    velocity = np.zeros(3)
    force = np.zeros(3)
    tracker.latch_baseline(force)
    mass = np.array([26.1, 26.8, 26.1])
    damping = np.array([93.9, 143.7, 280.9])
    initial_error = np.linalg.norm(position - np.array([0.8, 0.0, 0.0]))
    for _ in range(400):
        output = tracker.update(position, force)
        force = output.pid.force.copy()
        velocity += (-damping * velocity - force) / mass * dt
        position += velocity * dt
    final_error = np.linalg.norm(position - np.array([0.8, 0.0, 0.0]))
    assert final_error < 0.10
    assert final_error < initial_error * 0.2
