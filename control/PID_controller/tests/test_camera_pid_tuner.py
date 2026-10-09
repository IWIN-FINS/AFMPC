from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # The GUI extra is intentionally optional for PID math.
    cv2 = None

from camera_pid_tuner import (
    BoundingBox,
    CameraIntrinsics,
    ErrorHistory,
    PIDTuningPanel,
    TemplateROITracker,
    camera_position_from_bbox,
    parse_roi,
)
from live_integration_example import build_tracker


def test_camera_pixel_conversion_uses_right_down_forward_order() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=200.0, cx=320.0, cy=240.0)
    position = intrinsics.pixel_to_camera(370.0, 200.0, 2.0)
    np.testing.assert_allclose(position, (1.0, -0.4, 2.0))
    np.testing.assert_allclose(
        camera_position_from_bbox(BoundingBox(360, 190, 20, 20), intrinsics, 2.0),
        position,
    )


def test_error_history_is_bounded_and_returns_four_channels() -> None:
    history = ErrorHistory(maxlen=3)
    for index in range(5):
        history.append(index, (index, 0.0, 0.0, 0.1), (1.0, 2.0, 3.0, 0.2))
    timestamp, error, output = history.arrays()
    assert len(history) == 3
    np.testing.assert_allclose(timestamp, (2.0, 3.0, 4.0))
    assert error.shape == (3, 4)
    assert output.shape == (3, 4)


def test_roi_parser_rejects_invalid_dimensions() -> None:
    assert parse_roi("1,2,30,40") == BoundingBox(1, 2, 30, 40)
    with pytest.raises(Exception):
        parse_roi("1,2,0,40")


@pytest.mark.skipif(cv2 is None, reason="OpenCV GUI extra is not installed")
def test_template_tracker_handles_flat_target_patch() -> None:
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(first, (30, 40), (49, 59), (255, 255, 255), -1)
    tracker = TemplateROITracker(BoundingBox(30, 40, 20, 20), match_threshold=0.5)
    tracker.initialize(first, cv2)
    second = np.zeros_like(first)
    cv2.rectangle(second, (35, 42), (54, 61), (255, 255, 255), -1)
    bbox, score = tracker.update(second, cv2)
    assert bbox == BoundingBox(35, 42, 20, 20)
    assert score is not None and score > 0.9


class _FakeCV2:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.created: list[tuple[str, int, int]] = []

    WINDOW_NORMAL = 0

    def namedWindow(self, *_args) -> None:
        return None

    def resizeWindow(self, *_args) -> None:
        return None

    def createTrackbar(self, name, _window, position, _maximum, _callback) -> None:
        self.values[name] = position
        self.created.append((name, position, _maximum))

    def getTrackbarPos(self, name, _window) -> int:
        return self.values[name]


def test_tuning_panel_updates_gains_without_resetting_pid_history() -> None:
    tracker = build_tracker()
    tracker.latch_baseline(np.zeros(3))
    tracker.update((1.2, 0.0, 0.0), np.zeros(3))
    panel = PIDTuningPanel(tracker, _FakeCV2())
    panel.create(range_m=2.0)
    panel.cv2.values["P_F"] = panel._sliders[0].encode(31.5)
    panel.cv2.values["I_F"] = panel._sliders[1].encode(0.7)
    before = tracker.controller._previous_error.copy()
    reference, range_m, yaw_reference = panel.read(2.0)
    assert tracker.controller.config.kp[0] == pytest.approx(31.5)
    assert tracker.controller.config.ki[0] == pytest.approx(0.7)
    np.testing.assert_allclose(tracker.controller._previous_error, before)
    np.testing.assert_allclose(reference, tracker.controller.config.reference_position)
    assert range_m == pytest.approx(2.0)
    assert yaw_reference is None
