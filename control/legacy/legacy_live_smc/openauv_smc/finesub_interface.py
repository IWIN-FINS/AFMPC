"""Thin adapters around the existing MPC_dual_model FineSUB v3 interface.

This module deliberately uses duck typing: the actual wire protocol and
transport remain owned by ``MPC_dual_model`` so fixes to CRC, session handling,
telemetry decoding, or the firmware mixer are not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .controller import TwoDOFControlOutput
from .model import OpenAUVState


@dataclass
class TelemetryStateEstimator:
    """Build [z, w, psi, r] from FineSUB telemetry.

    FineSUB v3 directly reports depth, yaw, and yaw rate but not heave velocity.
    The missing velocity is obtained by bounded finite difference followed by a
    first-order low-pass filter.  This is intentionally smaller than importing
    the translation MPC's six-state relative-target Kalman filter.
    """

    rate_filter_tau_s: float = 0.25
    max_abs_heave_rate_m_s: float = 1.0
    _last_depth_m: float | None = None
    _last_timestamp: float | None = None
    _heave_rate_m_s: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_filter_tau_s <= 0.0:
            raise ValueError("rate_filter_tau_s must be positive")
        if self.max_abs_heave_rate_m_s <= 0.0:
            raise ValueError("max_abs_heave_rate_m_s must be positive")

    def reset(self) -> None:
        self._last_depth_m = None
        self._last_timestamp = None
        self._heave_rate_m_s = 0.0

    def update(self, telemetry) -> OpenAUVState:
        depth = float(telemetry.depth_m)
        timestamp = float(telemetry.received_monotonic)
        if not all(
            math.isfinite(value)
            for value in (
                depth,
                timestamp,
                telemetry.yaw_rad,
                telemetry.yaw_rate_rad_s,
            )
        ):
            raise ValueError("telemetry state values must be finite")

        if self._last_depth_m is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > 1e-6:
                raw_rate = float(
                    np.clip(
                        (depth - self._last_depth_m) / dt,
                        -self.max_abs_heave_rate_m_s,
                        self.max_abs_heave_rate_m_s,
                    )
                )
                alpha = 1.0 - math.exp(-dt / self.rate_filter_tau_s)
                self._heave_rate_m_s += alpha * (raw_rate - self._heave_rate_m_s)
                self._last_depth_m = depth
                self._last_timestamp = timestamp
        else:
            self._last_depth_m = depth
            self._last_timestamp = timestamp

        return OpenAUVState(
            depth=depth,
            heave_velocity=self._heave_rate_m_s,
            yaw=float(telemetry.yaw_rad),
            yaw_rate=float(telemetry.yaw_rate_rad_s),
        )


@dataclass
class FineSUBCommandMapper:
    """Map visual-range plus 2-DOF SMC outputs to FineSUB high-level channels."""

    hardware_adapter: object

    def convert(
        self,
        control: TwoDOFControlOutput,
        *,
        forward_force_n: float,
        armed: bool,
    ):
        forward = float(forward_force_n)
        if not math.isfinite(forward):
            raise ValueError("forward_force_n must be finite")
        force_body = np.asarray(
            [forward, 0.0, control.heave_force],
            dtype=float,
        )
        return self.hardware_adapter.convert(
            force_body,
            control.yaw_moment,
            armed=armed,
            yaw_direct=True,
        )

