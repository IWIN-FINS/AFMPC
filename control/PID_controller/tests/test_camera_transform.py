from __future__ import annotations

import numpy as np

from camera_transform import (
    PID_CONTROL_CAMERA_ORIGIN_IN_BODY,
    PID_CONTROL_REFERENCE_POSITION_BODY,
    PID_CONTROL_ROTATION_BODY_FROM_CAMERA,
    camera_to_body_position,
    camera_to_pid_body_position,
)


def test_aligned_transform_remains_an_explicit_legacy_baseline() -> None:
    np.testing.assert_allclose(
        camera_to_body_position((0.2, -0.1, 1.3)),
        (1.3, 0.2, -0.1),
    )


def test_pid_transform_maps_camera_axes_to_body_axes_without_sign_flips() -> None:
    rotation = PID_CONTROL_ROTATION_BODY_FROM_CAMERA
    # OpenCV camera columns are right, down, forward.  Each corresponding
    # body component must retain its positive FRD sign for the real mount.
    assert rotation[1, 0] > 0.9  # camera-right -> body-right
    assert rotation[2, 1] > 0.9  # camera-down -> body-down
    assert rotation[0, 2] > 0.9  # camera-forward -> body-forward
    assert np.linalg.det(rotation) > 0.99
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)


def test_pid_camera_center_matches_calibrated_reference() -> None:
    position = camera_to_pid_body_position((0.0, 0.0, 0.60))
    np.testing.assert_allclose(position, PID_CONTROL_REFERENCE_POSITION_BODY, atol=2.0e-6)
    np.testing.assert_allclose(
        position,
        PID_CONTROL_ROTATION_BODY_FROM_CAMERA @ np.array((0.0, 0.0, 0.60))
        + PID_CONTROL_CAMERA_ORIGIN_IN_BODY,
    )
