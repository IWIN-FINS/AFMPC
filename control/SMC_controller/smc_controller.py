"""Full-vehicle saturated sliding-mode controller for the FineSUB link.

The controller operates on the same body-FRD target-relative state used by the
translation MPC:

    p_rel = target - vehicle
    v_rel = target_velocity - vehicle_velocity

For a stationary target, a positive body force makes ``v_rel`` negative.  The
translation SMC therefore uses the relative-motion input sign explicitly and
never treats a target-relative position as an absolute vehicle position.

This module contains no transport or motor allocation code.  It produces the
physical wrench ``[Fx, Fy, Fz, N]``; the existing FineSUBHardwareAdapter owns
the final normalized channels and the firmware owns the eight-motor mixer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import SimpleNamespace
import time
from typing import Any

import numpy as np

from MPC_dual_model.camera_transform import rotation_state_body_from_previous, wrap_angle
from MPC_dual_model.finesub_protocol import build_runtime_hardware_adapter
from MPC_dual_model.fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from MPC_dual_model.relative_kalman import KalmanConfig, RelativePositionKalmanFilter


Array = np.ndarray


def _vector(value: Any, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _matrix(value: Any, shape: tuple[int, int], name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def saturation(value: float) -> float:
    """Continuous unit saturation used as the SMC boundary layer."""

    return max(-1.0, min(1.0, float(value)))


@dataclass(frozen=True)
class AxisSMCConfig:
    """Outer position loop and inner velocity sliding-surface parameters."""

    outer_gain: float
    rate_limit: float
    rate_filter_tau: float
    reaching_gain: float
    robust_gain: float
    boundary_layer: float
    delta_input_limit: float
    brake_input_limit: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "outer_gain",
            "rate_limit",
            "rate_filter_tau",
            "reaching_gain",
            "robust_gain",
            "boundary_layer",
            "delta_input_limit",
            "brake_input_limit",
        ):
            _positive(getattr(self, name), name)


@dataclass(frozen=True)
class AxisSMCOutput:
    position_error: float
    rate_command: float
    rate_reference: float
    rate_reference_dot: float
    sliding_variable: float
    desired_acceleration: float


@dataclass(frozen=True)
class ForwardDistanceGuardConfig:
    """Safety envelope around the same forward target used by the tracker.

    ``target_distance_m`` is a consistency check against the MPC/SMC
    ``reference_position`` after it is transformed into camera coordinates.
    The tracker uses that transformed reference as the live target; it never
    substitutes an independent body-frame target.
    """

    target_distance_m: float = 0.60
    approach_start_m: float = 0.80
    hard_minimum_m: float = 0.30
    max_approach_force_n: float = 0.75
    minimum_retreat_force_n: float = 0.75
    alignment_tolerance_m: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "target_distance_m",
            "approach_start_m",
            "hard_minimum_m",
            "max_approach_force_n",
            "minimum_retreat_force_n",
            "alignment_tolerance_m",
        ):
            _positive(getattr(self, name), name)
        if self.approach_start_m <= self.target_distance_m:
            raise ValueError("approach_start_m must exceed target_distance_m")
        if self.hard_minimum_m >= self.target_distance_m:
            raise ValueError("hard_minimum_m must be below target_distance_m")

    def maximum_positive_camera_force(self, distance_m: float) -> float:
        """Return the allowed positive force along camera forward."""

        distance = float(distance_m)
        if not math.isfinite(distance):
            raise ValueError("camera forward distance must be finite")
        if distance <= self.target_distance_m:
            return 0.0
        if distance >= self.approach_start_m:
            return self.max_approach_force_n
        fraction = (
            distance - self.target_distance_m
        ) / (self.approach_start_m - self.target_distance_m)
        return self.max_approach_force_n * float(np.clip(fraction, 0.0, 1.0))


class SaturatedSMCAxis:
    """One bounded cascaded SMC axis.

    The sign of the physical input is applied by the full controller because
    translation state is target-relative while yaw state is vehicle-relative.
    """

    def __init__(self, config: AxisSMCConfig, *, angular_position: bool = False):
        self.config = config
        self.angular_position = bool(angular_position)
        self.rate_reference = 0.0

    def reset(self) -> None:
        self.rate_reference = 0.0

    def compute(
        self,
        position: float,
        rate: float,
        reference: float,
        dt: float,
    ) -> AxisSMCOutput:
        dt = _positive(dt, "dt")
        error = float(reference) - float(position)
        if self.angular_position:
            error = wrap_angle(error)
        cfg = self.config
        rate_command = float(
            np.clip(cfg.outer_gain * error, -cfg.rate_limit, cfg.rate_limit)
        )
        previous_reference = self.rate_reference
        alpha = 1.0 - math.exp(-dt / cfg.rate_filter_tau)
        self.rate_reference += alpha * (rate_command - self.rate_reference)
        rate_reference_dot = (self.rate_reference - previous_reference) / dt
        sliding_variable = self.rate_reference - float(rate)
        desired_acceleration = rate_reference_dot + (
            cfg.reaching_gain * sliding_variable
            + cfg.robust_gain
            * saturation(sliding_variable / cfg.boundary_layer)
        )
        return AxisSMCOutput(
            position_error=error,
            rate_command=rate_command,
            rate_reference=self.rate_reference,
            rate_reference_dot=rate_reference_dot,
            sliding_variable=sliding_variable,
            desired_acceleration=desired_acceleration,
        )


@dataclass(frozen=True)
class SMCControlOutput:
    force: Array
    yaw_moment: float
    translation_axes: tuple[AxisSMCOutput, AxisSMCOutput, AxisSMCOutput]
    yaw_axis: AxisSMCOutput
    yaw_reference: float
    unsaturated_force: Array
    unsaturated_yaw_moment: float


class FullVehicleSMCController:
    """Four-axis SMC with real-vehicle force and slew limits."""

    def __init__(
        self,
        *,
        mass_matrix: Any,
        linear_damping: Any,
        quadratic_damping: Any = (0.0, 0.0, 0.0),
        restoring_force: Any = (0.0, 0.0, 0.0),
        yaw_inertia: float,
        yaw_linear_damping: float,
        yaw_quadratic_damping: float = 0.0,
        yaw_moment_base: float = 0.0,
        translation_config: tuple[AxisSMCConfig, AxisSMCConfig, AxisSMCConfig],
        yaw_config: AxisSMCConfig,
        positive_force_limit: Any,
        negative_force_limit: Any,
        positive_yaw_limit: float,
        negative_yaw_limit: float,
        period_s: float,
    ) -> None:
        self.mass_matrix = _matrix(mass_matrix, (3, 3), "mass_matrix")
        self.linear_damping = _matrix(
            linear_damping, (3, 3), "linear_damping"
        )
        self.quadratic_damping = _vector(
            quadratic_damping, 3, "quadratic_damping"
        )
        self.restoring_force = _vector(
            restoring_force, 3, "restoring_force"
        )
        if not np.allclose(self.mass_matrix, self.mass_matrix.T, atol=1.0e-10):
            raise ValueError("mass_matrix must be symmetric")
        if np.min(np.linalg.eigvalsh(self.mass_matrix)) <= 0.0:
            raise ValueError("mass_matrix must be positive definite")
        damping_symmetric = 0.5 * (
            self.linear_damping + self.linear_damping.T
        )
        if np.min(np.linalg.eigvalsh(damping_symmetric)) < -1.0e-10:
            raise ValueError("linear_damping must have a positive semidefinite symmetric part")
        self.yaw_inertia = _positive(yaw_inertia, "yaw_inertia")
        self.yaw_linear_damping = _nonnegative(
            yaw_linear_damping, "yaw_linear_damping"
        )
        self.yaw_quadratic_damping = _nonnegative(
            yaw_quadratic_damping, "yaw_quadratic_damping"
        )
        self.yaw_moment_base = float(yaw_moment_base)
        if not math.isfinite(self.yaw_moment_base):
            raise ValueError("yaw_moment_base must be finite")
        if len(translation_config) != 3:
            raise ValueError("translation_config must contain three axes")
        self.translation_axes = tuple(
            SaturatedSMCAxis(config) for config in translation_config
        )
        self.yaw_axis = SaturatedSMCAxis(yaw_config, angular_position=True)
        self.positive_force_limit = _vector(
            positive_force_limit, 3, "positive_force_limit"
        )
        self.negative_force_limit = _vector(
            negative_force_limit, 3, "negative_force_limit"
        )
        if np.any(self.positive_force_limit <= 0.0) or np.any(
            self.negative_force_limit <= 0.0
        ):
            raise ValueError("force limits must be positive magnitudes")
        self.positive_yaw_limit = _positive(
            positive_yaw_limit, "positive_yaw_limit"
        )
        self.negative_yaw_limit = _positive(
            negative_yaw_limit, "negative_yaw_limit"
        )
        self.period_s = _positive(period_s, "period_s")
        self._previous_force: Array | None = None
        self._previous_yaw_moment: float | None = None

    @property
    def last_force(self) -> Array:
        if self._previous_force is None:
            return self.restoring_force.copy()
        return self._previous_force.copy()

    @property
    def last_yaw_moment(self) -> float:
        if self._previous_yaw_moment is None:
            return self.yaw_moment_base
        return float(self._previous_yaw_moment)

    def reset(
        self,
        *,
        previous_force: Any | None = None,
        previous_yaw_moment: float | None = None,
    ) -> None:
        for axis in self.translation_axes:
            axis.reset()
        self.yaw_axis.reset()
        self._previous_force = (
            None
            if previous_force is None
            else _vector(previous_force, 3, "previous_force").copy()
        )
        if previous_yaw_moment is None:
            self._previous_yaw_moment = None
        else:
            if not math.isfinite(float(previous_yaw_moment)):
                raise ValueError("previous_yaw_moment must be finite")
            self._previous_yaw_moment = float(previous_yaw_moment)

    def override_previous_force(self, force: Any) -> None:
        """Commit a post-controller safety-limited force for the next slew.

        A safety layer may need to remove an already accumulated approach
        force immediately.  If it only changed the returned command while
        leaving ``_previous_force`` untouched, the next SMC update could slew
        from the stale positive value and continue driving toward the target.
        """

        value = _vector(force, 3, "force")
        self._previous_force = np.clip(
            value,
            -self.negative_force_limit,
            self.positive_force_limit,
        ).copy()

    def override_previous_yaw_moment(self, yaw_moment: float) -> None:
        """Use the measured previous yaw moment as the next slew reference."""

        value = float(yaw_moment)
        if not math.isfinite(value):
            raise ValueError("yaw_moment must be finite")
        self._previous_yaw_moment = float(
            np.clip(value, -self.negative_yaw_limit, self.positive_yaw_limit)
        )

    @staticmethod
    def _slew(
        value: float,
        previous: float | None,
        limit: float,
    ) -> float:
        if previous is None:
            return value
        return float(np.clip(value, previous - limit, previous + limit))

    def compute(
        self,
        position_relative: Any,
        velocity_relative: Any,
        position_reference: Any,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        *,
        dt: float | None = None,
    ) -> SMCControlOutput:
        dt_value = self.period_s if dt is None else _positive(dt, "dt")
        position = _vector(position_relative, 3, "position_relative")
        velocity = _vector(velocity_relative, 3, "velocity_relative")
        reference = _vector(position_reference, 3, "position_reference")
        axes = tuple(
            axis.compute(position[i], velocity[i], reference[i], dt_value)
            for i, axis in enumerate(self.translation_axes)
        )
        desired_acceleration = np.asarray(
            [axis.desired_acceleration for axis in axes], dtype=float
        )
        damping = self.linear_damping @ velocity
        damping += self.quadratic_damping * np.abs(velocity) * velocity
        # For p_rel = p_target - p_vehicle and a stationary target,
        # p_rel_dot = -v_vehicle.  Substituting that relation into the
        # vehicle model gives
        #
        #   p_rel_ddot = M^-1 (tau_h - D p_rel_dot - tau).
        #
        # Therefore the physical force that realizes a requested relative
        # acceleration is tau = tau_h - D p_rel_dot - M a_rel.  This sign must
        # match FixedLinearDampingRelativeModel, whose relative state input
        # matrix is -M^-1.  Using +D p_rel_dot here makes the damping term
        # reinforce relative motion and turns the velocity loop into positive
        # feedback.
        unsaturated_force = (
            self.restoring_force
            - damping
            - self.mass_matrix @ desired_acceleration
        )
        clipped_force = np.clip(
            unsaturated_force,
            -self.negative_force_limit,
            self.positive_force_limit,
        )
        slew_limits = np.asarray(
            [axis.config.delta_input_limit for axis in self.translation_axes],
            dtype=float,
        )
        previous_force = self._previous_force
        if previous_force is not None:
            # Use a separate, faster limit while reducing force magnitude or
            # reversing direction.  A single symmetric slew limit can keep a
            # large approach command alive after the position error changes
            # sign, which is exactly the overshoot seen in the real trace.
            reducing_force = (
                np.abs(clipped_force) < np.abs(previous_force)
            ) | (
                np.sign(clipped_force) != np.sign(previous_force)
            )
            brake_limits = np.asarray(
                [axis.config.brake_input_limit for axis in self.translation_axes],
                dtype=float,
            )
            slew_limits = np.where(reducing_force, brake_limits, slew_limits)
            clipped_force = np.clip(
                clipped_force,
                previous_force - slew_limits,
                previous_force + slew_limits,
            )

        yaw_reference = wrap_angle(
            float(yaw_rad) + math.atan2(float(position[1]), float(position[0]))
        )
        yaw_output = self.yaw_axis.compute(
            float(yaw_rad),
            float(yaw_rate_rad_s),
            yaw_reference,
            dt_value,
        )
        unsaturated_yaw_moment = self.yaw_moment_base + (
            self.yaw_linear_damping * float(yaw_rate_rad_s)
            + self.yaw_quadratic_damping
            * abs(float(yaw_rate_rad_s))
            * float(yaw_rate_rad_s)
            + self.yaw_inertia * yaw_output.desired_acceleration
        )
        yaw_moment = float(
            np.clip(
                unsaturated_yaw_moment,
                -self.negative_yaw_limit,
                self.positive_yaw_limit,
            )
        )
        yaw_moment = self._slew(
            yaw_moment,
            self._previous_yaw_moment,
            self.yaw_axis.config.delta_input_limit,
        )
        yaw_moment = float(
            np.clip(yaw_moment, -self.negative_yaw_limit, self.positive_yaw_limit)
        )
        self._previous_force = clipped_force.copy()
        self._previous_yaw_moment = yaw_moment
        return SMCControlOutput(
            force=clipped_force.copy(),
            yaw_moment=yaw_moment,
            translation_axes=axes,
            yaw_axis=yaw_output,
            yaw_reference=yaw_reference,
            unsaturated_force=unsaturated_force.copy(),
            unsaturated_yaw_moment=float(unsaturated_yaw_moment),
        )


@dataclass
class RelativeStateEstimator:
    """Estimate body-FRD relative velocity from gated position samples."""

    filter_tau_s: float = 0.20
    max_speed_m_s: float = 1.0
    previous_position: Array | None = None
    previous_yaw_rad: float | None = None
    previous_time_s: float | None = None
    velocity: Array | None = None
    position_std: Any = (0.05, 0.05, 0.08)
    velocity_std: Any = (0.30, 0.35, 0.45)
    P: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.filter_tau_s = _positive(self.filter_tau_s, "filter_tau_s")
        self.max_speed_m_s = _positive(self.max_speed_m_s, "max_speed_m_s")
        self.position_std = _vector(self.position_std, 3, "position_std")
        self.velocity_std = _vector(self.velocity_std, 3, "velocity_std")
        if np.any(self.position_std <= 0.0) or np.any(self.velocity_std <= 0.0):
            raise ValueError("position_std and velocity_std must be positive")
        self.P = np.diag(
            np.concatenate((self.position_std**2, self.velocity_std**2))
        )

    def _reset_covariance(self) -> None:
        self.P = np.diag(
            np.concatenate((self.position_std**2, self.velocity_std**2))
        )

    def reset(self) -> None:
        self.previous_position = None
        self.previous_yaw_rad = None
        self.previous_time_s = None
        self.velocity = None
        self._reset_covariance()

    def update(
        self,
        position_relative: Any,
        yaw_rad: float,
        *,
        timestamp_s: float | None = None,
    ) -> tuple[Array, float]:
        position = _vector(position_relative, 3, "position_relative")
        yaw = float(yaw_rad)
        if not math.isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(now):
            raise ValueError("timestamp_s must be finite")
        if self.previous_position is None:
            self.previous_position = position.copy()
            self.previous_yaw_rad = yaw
            self.previous_time_s = now
            self.velocity = np.zeros(3, dtype=float)
            self._reset_covariance()
            return self.velocity.copy(), 0.0

        assert self.previous_yaw_rad is not None
        assert self.previous_time_s is not None
        assert self.velocity is not None
        dt = now - self.previous_time_s
        if dt <= 1.0e-6:
            return self.velocity.copy(), 0.0
        dt = min(dt, 0.50)
        previous_in_current = rotation_state_body_from_previous(
            wrap_angle(yaw - self.previous_yaw_rad)
        )[:3, :3] @ self.previous_position
        raw_velocity = (position - previous_in_current) / dt
        raw_velocity = np.clip(
            raw_velocity,
            -self.max_speed_m_s,
            self.max_speed_m_s,
        )
        alpha = 1.0 - math.exp(-dt / self.filter_tau_s)
        self.velocity += alpha * (raw_velocity - self.velocity)
        # This is a diagnostic covariance for the common guarded runtime log,
        # not a replacement for the MPC Kalman filter.  It deliberately grows
        # with the finite-difference interval and shrinks with the low-pass
        # update so operators can still see estimator health in one schema.
        position_variance = self.position_std**2 + (0.5 * dt * raw_velocity) ** 2
        velocity_variance = self.velocity_std**2 * (1.0 - 0.5 * alpha)
        self.P = np.diag(np.concatenate((position_variance, velocity_variance)))
        self.previous_position = position.copy()
        self.previous_yaw_rad = yaw
        self.previous_time_s = now
        return self.velocity.copy(), dt


class SMCTracker:
    """Adapter exposing the SMC controller through the guarded AUTO API."""

    def __init__(
        self,
        controller: FullVehicleSMCController,
        estimator,
        reference_position: Any,
        *,
        yaw_direct: bool,
        rotation_body_from_camera: Any,
        camera_origin_in_body: Any,
        forward_distance_guard: ForwardDistanceGuardConfig,
    ) -> None:
        self.controller = controller
        self.estimator = estimator
        self.reference_position = _vector(
            reference_position, 3, "reference_position"
        )
        self.rotation_body_from_camera = _matrix(
            rotation_body_from_camera,
            (3, 3),
            "rotation_body_from_camera",
        )
        if not np.allclose(
            self.rotation_body_from_camera.T
            @ self.rotation_body_from_camera,
            np.eye(3),
            atol=1.0e-5,
        ):
            raise ValueError("rotation_body_from_camera must be orthonormal")
        if not np.isclose(
            np.linalg.det(self.rotation_body_from_camera),
            1.0,
            atol=1.0e-5,
        ):
            raise ValueError("rotation_body_from_camera must have determinant +1")
        self.camera_origin_in_body = _vector(
            camera_origin_in_body,
            3,
            "camera_origin_in_body",
        )
        # The third camera basis vector expressed in body FRD is the exact
        # axis used to measure camera forward, including the calibrated mount
        # rotation.  For p_body = R p_camera + t:
        # p_camera[2] = R[:, 2] dot (p_body - t).
        self.camera_forward_body = self.rotation_body_from_camera[:, 2].copy()
        self.forward_distance_guard = forward_distance_guard
        reference_forward = self.camera_forward_distance(self.reference_position)
        if abs(
            reference_forward - forward_distance_guard.target_distance_m
        ) > forward_distance_guard.alignment_tolerance_m:
            raise ValueError(
                "forward distance guard target does not match the SMC/MPC "
                f"reference: {reference_forward:.6f} m vs "
                f"{forward_distance_guard.target_distance_m:.6f} m"
            )
        estimator_model = getattr(estimator, "model", None)
        model_period = (
            float(estimator_model.dt)
            if estimator_model is not None
            else controller.period_s
        )
        self.model = SimpleNamespace(
            dt=model_period,
            translation=SimpleNamespace(
                restoring_force=controller.restoring_force.copy()
            ),
        )
        self.fusion = SimpleNamespace(
            M1=np.zeros(3),
            M2=np.zeros(3),
            C12=np.zeros(3),
            sample_count=0,
            window_count=0,
            window_valid_count=np.zeros(3, dtype=int),
            window_rejected_count=np.zeros(3, dtype=int),
            indistinguishable=np.ones(3, dtype=bool),
            model1_weight=np.zeros(3),
        )
        self.kind = "smc-full"
        self.yaw_direct = bool(yaw_direct)
        self._last_output: SMCControlOutput | None = None
        self.last_forward_distance_m: float | None = None
        self.last_forward_force_before_guard_n: float | None = None
        self.last_forward_force_after_guard_n: float | None = None

    def camera_forward_distance(self, position_body: Any) -> float:
        """Return target distance along the calibrated camera-forward axis."""

        position = _vector(position_body, 3, "position_body")
        return float(
            np.dot(
                self.camera_forward_body,
                position - self.camera_origin_in_body,
            )
        )

    def _guard_force(
        self,
        force: Any,
        position_body: Any,
        reference_position: Any,
    ) -> Array:
        """Limit approach force while preserving the MPC reference target."""

        value = _vector(force, 3, "force")
        reference = _vector(reference_position, 3, "reference_position")
        target_distance = self.camera_forward_distance(reference)
        if abs(
            target_distance - self.forward_distance_guard.target_distance_m
        ) > self.forward_distance_guard.alignment_tolerance_m:
            raise ValueError(
                "forward distance guard target does not match the active "
                f"reference: {target_distance:.6f} m vs "
                f"{self.forward_distance_guard.target_distance_m:.6f} m"
            )
        distance = self.camera_forward_distance(position_body)
        maximum_positive = (
            self.forward_distance_guard.maximum_positive_camera_force(distance)
        )
        if distance < self.forward_distance_guard.hard_minimum_m:
            maximum_positive = min(
                maximum_positive,
                -self.forward_distance_guard.minimum_retreat_force_n,
            )

        projection = float(np.dot(self.camera_forward_body, value))
        guarded = value.copy()
        if projection > maximum_positive:
            guarded -= self.camera_forward_body * (projection - maximum_positive)

        # Keep the safety output inside the same physical force box.  Repeat
        # once after clipping because an oblique camera axis can make clipping
        # one body component slightly increase the forward projection.
        guarded = np.clip(
            guarded,
            -self.controller.negative_force_limit,
            self.controller.positive_force_limit,
        )
        for _ in range(3):
            projection = float(np.dot(self.camera_forward_body, guarded))
            excess = projection - maximum_positive
            if excess <= 1.0e-10:
                break
            guarded = np.clip(
                guarded - self.camera_forward_body * excess,
                -self.controller.negative_force_limit,
                self.controller.positive_force_limit,
            )

        self.last_forward_distance_m = distance
        self.last_forward_force_before_guard_n = float(
            np.dot(self.camera_forward_body, value)
        )
        self.last_forward_force_after_guard_n = float(
            np.dot(self.camera_forward_body, guarded)
        )
        return guarded

    def latch_baseline(
        self,
        force: Any,
        yaw_moment: float,
        _yaw_rad: float,
    ) -> None:
        self.controller.reset(
            previous_force=force,
            previous_yaw_moment=yaw_moment,
        )
        self.estimator.reset()
        self._last_output = None

    def target_lost(self, _force: Any, _yaw_moment: float):
        self.controller.reset(
            previous_force=self.controller.restoring_force,
            previous_yaw_moment=self.controller.yaw_moment_base,
        )
        self.estimator.reset()
        self._last_output = None
        return SimpleNamespace(
            force=self.controller.restoring_force.copy(),
            yaw_moment=(
                float(self.controller.yaw_moment_base)
                if self.yaw_direct
                else 0.0
            ),
        )

    def update(
        self,
        *,
        position_body,
        force_achieved_previous,
        yaw_moment_achieved_previous,
        reference_position,
        yaw_rad,
        yaw_rate_rad_s,
        yaw_delta_rad=0.0,
        measurement_delay_s=0.0,
    ):
        reference = (
            self.reference_position
            if reference_position is None
            else _vector(reference_position, 3, "reference_position")
        )
        # The plant model is driven by the wrench that was actually applied by
        # the lower mixer, not by the command emitted on the preceding host
        # update.  This matters when motor response, per-motor limiting, or
        # telemetry timing makes the achieved wrench differ from the request.
        previous_force = np.clip(
            _vector(force_achieved_previous, 3, "force_achieved_previous"),
            -self.controller.negative_force_limit,
            self.controller.positive_force_limit,
        )
        self.controller.override_previous_force(previous_force)
        self.controller.override_previous_yaw_moment(yaw_moment_achieved_previous)
        delay = float(measurement_delay_s)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("measurement_delay_s must be finite and non-negative")
        if isinstance(self.estimator, RelativePositionKalmanFilter):
            if not self.estimator.initialized:
                estimated_state = self.estimator.initialize(position_body)
            else:
                estimated_state = self.estimator.step(
                    position_body,
                    previous_force,
                    yaw_delta_rad=yaw_delta_rad,
                )
            if delay > 0.0:
                estimated_state = self.estimator.predict_ahead(
                    previous_force,
                    delay,
                    yaw_delta_rad=float(yaw_rate_rad_s) * delay,
                )
            control_position = estimated_state[:3]
            velocity = estimated_state[3:]
            control_dt = self.model.dt
        else:
            velocity, estimator_dt = self.estimator.update(position_body, yaw_rad)
            control_position = _vector(position_body, 3, "position_body")
            estimated_state = np.concatenate((control_position, velocity))
            control_dt = (
                estimator_dt if estimator_dt > 0.0 else self.controller.period_s
            )
        output = self.controller.compute(
            control_position,
            velocity,
            reference,
            yaw_rad,
            yaw_rate_rad_s,
            dt=control_dt,
        )
        guarded_force = self._guard_force(output.force, position_body, reference)
        if not np.array_equal(guarded_force, output.force):
            output = replace(output, force=guarded_force)
        # This must happen even when the force did not change: the next SMC
        # slew must always start from the actually emitted safety-limited
        # force, never from a pre-guard approach command.
        self.controller.override_previous_force(output.force)
        force_delta = output.force - previous_force
        self._last_output = output
        commanded_yaw_moment = (
            float(output.yaw_moment) if self.yaw_direct else 0.0
        )
        state = estimated_state.copy()
        mpc_compatible = SimpleNamespace(
            force=output.force.copy(),
            force_sequence=np.asarray([output.force.copy()]),
            delta_force_sequence=np.asarray([force_delta]),
            predicted_actuator_force_sequence=None,
            predicted_states=np.asarray([state.copy()]),
            slacks=np.zeros(1),
            model1_weight=np.zeros(3),
            frozen_delta_yaw=np.asarray([0.0]),
            frozen_yaw_moments=np.asarray([commanded_yaw_moment]),
            force_reference=reference.copy(),
            model1_base_force=self.controller.restoring_force.copy(),
            yaw_moment=commanded_yaw_moment,
            status="smc",
            iterations=0,
            objective=0.0,
            used_fallback=False,
        )
        return SimpleNamespace(
            estimated_state=state,
            mpc=mpc_compatible,
            yaw_control=SimpleNamespace(
                mode=SimpleNamespace(
                    value=(
                        "direct_smc"
                        if self.yaw_direct
                        else "lower_local_hold"
                    )
                ),
                goal_angle=float(output.yaw_reference),
            ),
            line_of_sight_angle=float(
                math.atan2(float(position_body[1]), float(position_body[0]))
            ),
            smc=output,
        )


def _axis_config(data: dict[str, Any], name: str) -> AxisSMCConfig:
    try:
        return AxisSMCConfig(
            outer_gain=float(data["outer_gain"]),
            rate_limit=float(data["rate_limit"]),
            rate_filter_tau=float(data["rate_filter_tau"]),
            reaching_gain=float(data["reaching_gain"]),
            robust_gain=float(data["robust_gain"]),
            boundary_layer=float(data["boundary_layer"]),
            delta_input_limit=float(data["delta_input_limit"]),
            brake_input_limit=float(
                data.get("brake_input_limit", data["delta_input_limit"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid SMC {name} configuration: {error}") from error


def _axis_configs(data: dict[str, Any], name: str) -> tuple[AxisSMCConfig, ...]:
    values = data.get(name)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"smc_parameters.controller.{name} must contain three axes")
    return tuple(_axis_config(item, f"{name}[{index}]") for index, item in enumerate(values))


def build_smc_tracker(config: dict[str, Any]) -> SMCTracker:
    """Build the full SMC tracker from the selected real-vehicle profile."""

    auto = config.get("auto_runtime")
    if not isinstance(auto, dict):
        raise ValueError("auto_runtime must be an object")
    active_translation = auto.get("active_mpc_parameters")
    active_yaw = auto.get("active_yaw_parameters")
    if not isinstance(active_translation, dict) or not isinstance(active_yaw, dict):
        raise ValueError("selected SMC profile is missing vehicle dynamics")
    if active_translation.get("enabled_for_control") is not True:
        raise ValueError("active_mpc_parameters is not enabled for SMC control")
    if active_yaw.get("enabled_for_control") is not True:
        raise ValueError("active_yaw_parameters is not enabled for SMC control")
    model = active_translation.get("model")
    yaw_dynamics = active_yaw.get("dynamics")
    if not isinstance(model, dict) or not isinstance(yaw_dynamics, dict):
        raise ValueError("selected SMC profile has incomplete dynamics")
    smc = config.get("smc_parameters")
    if not isinstance(smc, dict):
        raise ValueError("smc_parameters must be an object")
    controller_data = smc.get("controller")
    if not isinstance(controller_data, dict):
        raise ValueError("smc_parameters.controller must be an object")
    hardware = build_runtime_hardware_adapter(config)
    period_s = float(config["control"]["period_sec"])
    model_period_s = float(model.get("sample_period_s", period_s))
    if not math.isfinite(model_period_s) or model_period_s <= 0.0:
        raise ValueError("active SMC model sample_period_s must be positive")
    translation_configs = _axis_configs(controller_data, "translation")
    yaw_config_data = controller_data.get("yaw")
    if not isinstance(yaw_config_data, dict):
        raise ValueError("smc_parameters.controller.yaw must be an object")
    yaw_config = _axis_config(yaw_config_data, "yaw")
    reference = smc.get("reference_position")
    if reference is None:
        reference = controller_data.get("reference_position")
    if reference is None:
        raise ValueError("SMC reference_position is missing")
    active_mpc_controller = active_translation.get("controller")
    mpc_reference = (
        active_mpc_controller.get("reference_position")
        if isinstance(active_mpc_controller, dict)
        else None
    )
    if mpc_reference is not None and not np.allclose(
        _vector(reference, 3, "reference_position"),
        _vector(mpc_reference, 3, "active MPC reference_position"),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError(
            "SMC reference_position must match the active MPC "
            "controller.reference_position"
        )
    camera_transform = auto.get("active_camera_transform")
    if not isinstance(camera_transform, dict):
        raise ValueError("auto_runtime.active_camera_transform must be an object")
    rotation_body_from_camera = camera_transform.get("rotation_body_from_camera")
    camera_origin_in_body = camera_transform.get("camera_origin_in_body_frd_m")
    if rotation_body_from_camera is None or camera_origin_in_body is None:
        raise ValueError("active camera transform is incomplete for SMC")
    guard_data = smc.get("forward_distance_guard")
    if not isinstance(guard_data, dict):
        raise ValueError("smc_parameters.forward_distance_guard must be an object")
    try:
        forward_distance_guard = ForwardDistanceGuardConfig(
            target_distance_m=float(guard_data["target_distance_m"]),
            approach_start_m=float(guard_data["approach_start_m"]),
            hard_minimum_m=float(guard_data["hard_minimum_m"]),
            max_approach_force_n=float(guard_data["max_approach_force_n"]),
            minimum_retreat_force_n=float(
                guard_data["minimum_retreat_force_n"]
            ),
            alignment_tolerance_m=float(
                guard_data.get("alignment_tolerance_m", 0.01)
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid smc_parameters.forward_distance_guard: {error}"
        ) from error
    yaw_authority = str(smc.get("yaw_authority", "direct")).strip().lower()
    if yaw_authority not in {"direct", "lower_local_hold"}:
        raise ValueError(
            "smc_parameters.yaw_authority must be direct or lower_local_hold"
        )
    controller = FullVehicleSMCController(
        mass_matrix=model["effective_mass_matrix_kg"],
        linear_damping=model["linear_damping_matrix_n_s_per_m"],
        quadratic_damping=model.get("quadratic_damping_vector", (0.0, 0.0, 0.0)),
        restoring_force=model.get("restoring_force_frd_n", (0.0, 0.0, 0.0)),
        yaw_inertia=float(yaw_dynamics["effective_inertia_kg_m2"]),
        yaw_linear_damping=float(yaw_dynamics["linear_damping_n_m_per_rad_s"]),
        yaw_quadratic_damping=float(
            yaw_dynamics.get("quadratic_damping_n_m_per_rad_s2", 0.0)
        ),
        yaw_moment_base=float(yaw_dynamics.get("yaw_moment_base_n_m", 0.0)),
        translation_config=translation_configs,
        yaw_config=yaw_config,
        positive_force_limit=hardware.positive_force_at_limit,
        negative_force_limit=hardware.negative_force_at_limit,
        positive_yaw_limit=hardware.positive_yaw_moment_at_limit,
        negative_yaw_limit=hardware.negative_yaw_moment_at_limit,
        period_s=period_s,
    )
    kalman_data = active_translation.get("kalman")
    if not isinstance(kalman_data, dict):
        raise ValueError(
            "active_mpc_parameters.kalman must be available for SMC state estimation"
        )
    relative_model = FixedLinearDampingRelativeModel(
        M_t=model["effective_mass_matrix_kg"],
        D_L=model["linear_damping_matrix_n_s_per_m"],
        dt=model_period_s,
        restoring_force=model.get("restoring_force_frd_n", (0.0, 0.0, 0.0)),
    )
    estimator = RelativePositionKalmanFilter(
        relative_model,
        KalmanConfig(**kalman_data),
    )
    tracker = SMCTracker(
        controller,
        estimator,
        reference,
        yaw_direct=yaw_authority == "direct",
        rotation_body_from_camera=rotation_body_from_camera,
        camera_origin_in_body=camera_origin_in_body,
        forward_distance_guard=forward_distance_guard,
    )
    return tracker
