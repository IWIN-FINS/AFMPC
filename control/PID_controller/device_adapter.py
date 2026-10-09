"""FineSUB force envelope, device scaling, and eight-motor allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .pid_controller import vector3
except ImportError:
    from pid_controller import vector3


FINESUB_V4_PRO1_FORCE_POSITIVE_N = np.array(
    [8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749]
)
FINESUB_V4_PRO1_FORCE_NEGATIVE_N = np.array(
    [7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750]
)


def finesub_translation_thruster_force_matrix() -> np.ndarray:
    """Map FRD translation force to canonical eight-thruster forces."""
    gain = 1.0 / (2.0 * np.sqrt(2.0))
    return np.array(
        [
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.25],
            [gain, gain, 0.0],
            [gain, -gain, 0.0],
            [-gain, -gain, 0.0],
            [-gain, gain, 0.0],
        ]
    )


def finesub_translation_force_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Conservative independent force bounds for the canonical mixer."""
    matrix = finesub_translation_thruster_force_matrix()
    lower = np.full(3, -np.inf)
    upper = np.full(3, np.inf)
    for axis in range(3):
        for coefficient, positive, negative in zip(
            matrix[:, axis],
            FINESUB_V4_PRO1_FORCE_POSITIVE_N,
            FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        ):
            if abs(coefficient) <= np.finfo(float).eps:
                continue
            if coefficient > 0.0:
                lower[axis] = max(lower[axis], -negative / coefficient)
                upper[axis] = min(upper[axis], positive / coefficient)
            else:
                lower[axis] = max(lower[axis], positive / coefficient)
                upper[axis] = min(upper[axis], -negative / coefficient)
    return lower, upper


@dataclass(frozen=True)
class DeviceCommand:
    planar_forward: int
    planar_right: int
    depth_force: int


@dataclass
class ForceCommandAdapter:
    positive_force_at_limit: object
    signs: object = (1.0, 1.0, 1.0)
    command_limits: object = (99.0, 99.0, 45.0)

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.signs = vector3(self.signs, "signs")
        self.command_limits = vector3(self.command_limits, "command_limits")
        if np.any(self.positive_force_at_limit <= 0.0):
            raise ValueError("positive_force_at_limit must be positive")
        if np.any(self.command_limits <= 0.0):
            raise ValueError("command_limits must be positive")
        if np.any(np.abs(self.signs) != 1.0):
            raise ValueError("signs must contain only +1 or -1")

    def convert(self, force_body: object) -> DeviceCommand:
        force = vector3(force_body, "force_body")
        command = np.rint(
            np.clip(
                self.signs
                * force
                / self.positive_force_at_limit
                * self.command_limits,
                -self.command_limits,
                self.command_limits,
            )
        ).astype(int)
        return DeviceCommand(*(int(value) for value in command))


@dataclass
class YawMomentChannelAdapter:
    """Convert yaw moment in N*m to the normalized FineSUB yaw channel."""

    positive_yaw_moment_at_limit: float = 2.0
    channel_limit: float = 0.20
    sign: float = 1.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.positive_yaw_moment_at_limit)
            or self.positive_yaw_moment_at_limit <= 0.0
        ):
            raise ValueError("positive_yaw_moment_at_limit must be positive")
        if not np.isfinite(self.channel_limit) or self.channel_limit <= 0.0:
            raise ValueError("channel_limit must be positive")
        if self.sign not in (-1.0, 1.0):
            raise ValueError("sign must be +1 or -1")

    def convert(self, yaw_moment: float) -> float:
        moment = float(yaw_moment)
        if not np.isfinite(moment):
            raise ValueError("yaw_moment must be finite")
        return float(
            np.clip(
                self.sign
                * moment
                / self.positive_yaw_moment_at_limit
                * self.channel_limit,
                -self.channel_limit,
                self.channel_limit,
            )
        )


@dataclass(frozen=True)
class ThrusterAllocation:
    throttles: np.ndarray
    translation_channels: np.ndarray
    attitude_channels: np.ndarray


@dataclass
class FineSUBThrusterAllocator:
    """Match the FineSUB firmware mixer and motor order."""

    positive_force_at_limit: object
    # Active MPC experimental translation authority.  Firmware still accepts
    # the wider 0.35/0.50 hard envelope; PID deliberately emits only 0.20.
    translation_channel_limits: object = (0.20, 0.20, 0.20)
    attitude_channel_limits: object = (0.20, 0.20, 0.20)
    deadband: float = 0.01
    enable_depth: bool = True

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.translation_channel_limits = vector3(
            self.translation_channel_limits, "translation_channel_limits"
        )
        self.attitude_channel_limits = vector3(
            self.attitude_channel_limits, "attitude_channel_limits"
        )
        if np.any(self.positive_force_at_limit <= 0.0):
            raise ValueError("positive_force_at_limit must be positive")
        if self.deadband < 0.0:
            raise ValueError("deadband must be nonnegative")
        # Current matrices and motor groups in V4pro1_MPC/V5_SUB.hpp.
        self.upper_matrix = np.array(
            [[-1.0, 1.0, 1.0], [1.0, 1.0, -1.0],
             [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
        )
        self.lower_matrix = np.array(
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]
        )

    def allocate(
        self, force_body: object, attitude_control: object = (0.0, 0.0, 0.0)
    ) -> ThrusterAllocation:
        translation = np.clip(
            vector3(force_body, "force_body")
            / self.positive_force_at_limit
            * self.translation_channel_limits,
            -self.translation_channel_limits,
            self.translation_channel_limits,
        )
        if not self.enable_depth:
            translation[2] = 0.0
        attitude = np.clip(
            vector3(attitude_control, "attitude_control"),
            -self.attitude_channel_limits,
            self.attitude_channel_limits,
        )
        roll, pitch, yaw = attitude
        forward, right, down = translation
        upper = self.upper_matrix @ np.array([roll, pitch, down])
        lower = self.lower_matrix @ np.array([yaw, forward, right])
        upper[np.abs(upper) < self.deadband] = 0.0
        lower[np.abs(lower) < self.deadband] = 0.0
        throttles = np.zeros(8)
        throttles[[2, 3, 4, 7]] = upper
        throttles[[0, 1, 5, 6]] = lower
        return ThrusterAllocation(
            throttles=np.clip(throttles, -1.0, 1.0),
            translation_channels=translation.copy(),
            attitude_channels=attitude.copy(),
        )
