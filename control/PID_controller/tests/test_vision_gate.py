from __future__ import annotations

import numpy as np

from PID_controller.vision_gate import PIDVisionGate, VisionGateConfig


def test_gate_requires_stable_startup_samples() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(
            startup_confirmation_samples=3,
            reacquire_confirmation_samples=2,
        )
    )
    assert not gate.update((0.0, 0.0, 1.0), now=0.00).ready
    assert not gate.update((0.0, 0.0, 1.0), now=0.05).ready
    result = gate.update((0.0, 0.0, 1.0), now=0.10)
    assert result.ready
    assert result.reason == "vision_locked"
    np.testing.assert_allclose(result.position_camera_xyz, (0.0, 0.0, 1.0))


def test_large_jump_is_ignored_while_holding_the_last_valid_position() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=2)
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.00).ready
    result = gate.update((0.5, 0.0, 1.0), now=0.05)
    assert result.ready
    assert result.reason == "vision_jump_ignored"
    assert not result.lost
    assert result.locked
    np.testing.assert_allclose(result.position_camera_xyz, (0.0, 0.0, 1.0))


def test_consistent_new_position_reacquires_without_disarm() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=3)
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.00).ready
    assert gate.update((0.5, 0.0, 1.0), now=0.05).reason == "vision_jump_ignored"
    # Repeated delivery of the same camera sample does not fake confirmation.
    assert gate.update((0.5, 0.0, 1.0), now=0.06).reason == "vision_jump_ignored"
    assert gate.update((0.51, 0.0, 1.0), now=0.10).reason == "vision_jump_ignored"
    reacquired = gate.update((0.52, 0.0, 1.0), now=0.15)
    assert reacquired.ready
    assert reacquired.reason == "vision_reacquired"
    np.testing.assert_allclose(reacquired.position_camera_xyz, (0.52, 0.0, 1.0))


def test_locked_track_resumes_immediately_after_an_ignored_jump() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=3)
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.00).ready
    assert gate.update((0.3, 0.0, 1.0), now=0.05).reason == "vision_jump_ignored"
    result = gate.update((0.02, 0.0, 1.0), now=0.10)
    assert result.ready
    assert result.reason == "vision_tracking"


def test_ignored_jump_does_not_contaminate_next_valid_sample() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=2)
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.00).ready
    jump = gate.update((0.8, 0.0, 1.0), now=0.05)
    assert jump.reason == "vision_jump_ignored"
    valid = gate.update((0.02, 0.0, 1.0), now=0.10)
    assert valid.ready
    assert valid.reason == "vision_tracking"
    np.testing.assert_allclose(valid.position_camera_xyz, (0.02, 0.0, 1.0))


def test_forward_range_is_fail_closed() -> None:
    gate = PIDVisionGate(VisionGateConfig(min_forward_m=0.3, max_forward_m=1.4))
    result = gate.update((0.0, 0.0, 2.0), now=0.0)
    assert not result.ready
    assert result.reason == "forward_range"


def test_out_of_range_jump_is_ignored_while_locked() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(
            min_forward_m=0.3,
            max_forward_m=1.4,
            startup_confirmation_samples=1,
        )
    )
    assert gate.update((0.0, 0.0, 0.6), now=0.0).ready
    result = gate.update((0.0, 0.0, 0.1), now=0.05)
    assert result.ready
    assert result.reason == "vision_jump_ignored"
    np.testing.assert_allclose(result.position_camera_xyz, (0.0, 0.0, 0.6))


def test_safety_reset_preserves_reacquisition_history() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=3)
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.0).ready
    gate.reset(preserve_lock_history=True)
    assert not gate.update((0.0, 0.0, 1.0), now=0.1).ready
    assert not gate.update((0.0, 0.0, 1.0), now=0.2).ready
    assert gate.update((0.0, 0.0, 1.0), now=0.3).ready


def test_one_sample_reacquisition_never_accepts_first_implausible_jump() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(
            startup_confirmation_samples=1,
            reacquire_confirmation_samples=1,
            jump_margin_m=0.20,
            max_step_m=0.30,
        )
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.0).ready
    first_jump = gate.update((0.5, 0.0, 1.0), now=0.05)
    assert first_jump.reason == "vision_jump_ignored"
    np.testing.assert_allclose(first_jump.position_camera_xyz, (0.0, 0.0, 1.0))


def test_one_sample_reacquisition_after_real_camera_gap() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(
            startup_confirmation_samples=1,
            reacquire_confirmation_samples=1,
            max_inter_sample_gap_s=1.0,
        )
    )
    assert gate.update((0.0, 0.0, 1.0), now=0.0).ready
    reacquired = gate.update((0.4, 0.0, 1.0), now=1.1)
    assert reacquired.ready
    np.testing.assert_allclose(reacquired.position_camera_xyz, (0.4, 0.0, 1.0))
