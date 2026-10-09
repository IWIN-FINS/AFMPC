"""Monocular bounding-box geometry matching the fused translation MPC input."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from .controller import wrap_angle


@dataclass(frozen=True)
class VisionConfig:
    """Camera calibration and deliberately simple visual-control gains."""

    range_reference_m: float = 0.50
    target_width_at_reference_px: float = 90.0
    horizontal_half_fov_deg: float = 42.0
    vertical_half_fov_deg: float = 30.0
    width_filter_length: int = 6
    desired_range_m: float = 0.80
    range_deadband_m: float = 0.04
    forward_force_gain_n_m: float = 12.0
    forward_force_limit_n: float = 8.0
    depth_correction_gain: float = 1.0
    max_depth_correction_m: float = 0.35
    max_measurement_age_s: float = 0.25

    def __post_init__(self) -> None:
        positive = {
            "range_reference_m": self.range_reference_m,
            "target_width_at_reference_px": self.target_width_at_reference_px,
            "horizontal_half_fov_deg": self.horizontal_half_fov_deg,
            "vertical_half_fov_deg": self.vertical_half_fov_deg,
            "desired_range_m": self.desired_range_m,
            "forward_force_gain_n_m": self.forward_force_gain_n_m,
            "forward_force_limit_n": self.forward_force_limit_n,
            "depth_correction_gain": self.depth_correction_gain,
            "max_depth_correction_m": self.max_depth_correction_m,
            "max_measurement_age_s": self.max_measurement_age_s,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"vision parameters must be positive: {invalid}")
        if self.width_filter_length < 1:
            raise ValueError("width_filter_length must be at least one")
        if self.range_deadband_m < 0.0:
            raise ValueError("range_deadband_m must be non-negative")
        if not 0.0 < self.horizontal_half_fov_deg < 90.0:
            raise ValueError("horizontal_half_fov_deg must be in (0, 90)")
        if not 0.0 < self.vertical_half_fov_deg < 90.0:
            raise ValueError("vertical_half_fov_deg must be in (0, 90)")


@dataclass(frozen=True)
class VisualObservation:
    """One CSRT bounding-box observation in OpenCV camera coordinates."""

    timestamp: float
    bbox_xywh: tuple[float, float, float, float]
    camera_position_right_down_forward_m: tuple[float, float, float]
    smoothed_width_px: float

    @property
    def right_m(self) -> float:
        return self.camera_position_right_down_forward_m[0]

    @property
    def down_m(self) -> float:
        return self.camera_position_right_down_forward_m[1]

    @property
    def forward_m(self) -> float:
        return self.camera_position_right_down_forward_m[2]

    @property
    def bearing_rad(self) -> float:
        return math.atan2(self.right_m, self.forward_m)


@dataclass(frozen=True)
class VisualControlReference:
    depth_reference_m: float
    yaw_reference_rad: float
    forward_force_n: float
    range_error_m: float
    bearing_error_rad: float
    vertical_error_m: float


class BBoxTargetEstimator:
    """Convert a tracked ROI into range, bearing, and depth/yaw references.

    The geometry is identical to ``rov_track_control3.camera_position_from_bbox``:
    the camera vector order is [right, down, forward], and range is calibrated
    from target width.  No claim is made that this is a metric depth sensor;
    the two range constants must be measured for the selected target.
    """

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._width_history: deque[float] = deque(
            maxlen=self.config.width_filter_length
        )

    def reset(self) -> None:
        self._width_history.clear()

    def update_bbox(
        self,
        bbox_xywh,
        image_width: int,
        image_height: int,
        *,
        timestamp: float | None = None,
    ) -> VisualObservation:
        bbox = np.asarray(bbox_xywh, dtype=float).reshape(-1)
        if bbox.size != 4 or not np.all(np.isfinite(bbox)):
            raise ValueError("bbox_xywh must contain four finite values")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        x, y, width, height = (float(value) for value in bbox)
        if width <= 1.0 or height <= 1.0:
            raise ValueError("tracked bounding box is too small")

        self._width_history.append(width)
        smoothed_width = sum(self._width_history) / len(self._width_history)
        center_x = x + 0.5 * width
        center_y = y + 0.5 * height
        forward = (
            self.config.range_reference_m
            * self.config.target_width_at_reference_px
            / smoothed_width
        )
        focal_x = image_width / (
            2.0 * math.tan(math.radians(self.config.horizontal_half_fov_deg))
        )
        focal_y = image_height / (
            2.0 * math.tan(math.radians(self.config.vertical_half_fov_deg))
        )
        right = (center_x - 0.5 * image_width) * forward / focal_x
        down = (center_y - 0.5 * image_height) * forward / focal_y

        return VisualObservation(
            timestamp=time.monotonic() if timestamp is None else float(timestamp),
            bbox_xywh=(x, y, width, height),
            camera_position_right_down_forward_m=(right, down, forward),
            smoothed_width_px=smoothed_width,
        )

    def is_fresh(
        self,
        observation: VisualObservation | None,
        *,
        now: float | None = None,
    ) -> bool:
        if observation is None:
            return False
        current = time.monotonic() if now is None else float(now)
        age = current - observation.timestamp
        return 0.0 <= age <= self.config.max_measurement_age_s

    def make_reference(
        self,
        observation: VisualObservation,
        *,
        current_depth_m: float,
        current_yaw_rad: float,
    ) -> VisualControlReference:
        depth_correction = float(
            np.clip(
                self.config.depth_correction_gain * observation.down_m,
                -self.config.max_depth_correction_m,
                self.config.max_depth_correction_m,
            )
        )
        bearing = observation.bearing_rad
        range_error = observation.forward_m - self.config.desired_range_m
        if abs(range_error) <= self.config.range_deadband_m:
            effective_range_error = 0.0
        else:
            effective_range_error = math.copysign(
                abs(range_error) - self.config.range_deadband_m,
                range_error,
            )
        forward_force = float(
            np.clip(
                self.config.forward_force_gain_n_m * effective_range_error,
                -self.config.forward_force_limit_n,
                self.config.forward_force_limit_n,
            )
        )
        return VisualControlReference(
            depth_reference_m=float(current_depth_m) + depth_correction,
            yaw_reference_rad=wrap_angle(float(current_yaw_rad) + bearing),
            forward_force_n=forward_force,
            range_error_m=range_error,
            bearing_error_rad=bearing,
            vertical_error_m=observation.down_m,
        )

