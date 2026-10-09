"""Measurement -> filter/fusion -> yaw PID -> translation MPC -> mixer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from MPC_dual_model.device_adapter import (
    DeviceCommand,
    FineSUBThrusterAllocator,
    ForceCommandAdapter,
    ThrusterAllocation,
)
from MPC_dual_model.fossen_fixed_dl_model import vector3
from MPC_dual_model.model_fusion import FusionConfig, OnlineModelFusion

from .yaw_kalman import RotationAwareKalmanFilter
from .yaw_controller import YawControlResult, YawStateController
from .yaw_mpc_controller import RotationAwareMPCController, YawMPCResult
from .yaw_relative_model import (
    RotationAwareRelativeModel,
    body_to_visibility_position,
    finite_scalar,
    line_of_sight_angle,
    visibility_frame_geometry,
    wrap_angle,
)


DEFAULT_STAIRCASE_HORIZON_CAPS = (5, 5, 5, 5, 5, 4, 3, 1)
DEFAULT_PREDICTION_HORIZON_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)


def build_default_staircase_fusion() -> OnlineModelFusion:
    """Match the current experimental translation fusion profile."""
    return OnlineModelFusion(
        FusionConfig(
            window=len(DEFAULT_STAIRCASE_HORIZON_CAPS),
            prediction_horizon=len(DEFAULT_PREDICTION_HORIZON_WEIGHTS),
            forgetting_factor=0.8,
            weight_update_rate=0.10,
            prediction_horizon_weights=DEFAULT_PREDICTION_HORIZON_WEIGHTS,
            staircase_horizon_caps=DEFAULT_STAIRCASE_HORIZON_CAPS,
            initial_model1_weight=(0.0, 0.0, 0.0),
            minimum_weight=0.01,
        )
    )


@dataclass
class YawMomentChannelAdapter:
    """Convert physical yaw moment N*m to FineSUB normalized yaw input."""

    positive_yaw_moment_at_limit: float = 4.0
    channel_limit: float = 0.20
    sign: float = 1.0
    negative_yaw_moment_at_limit: float | None = None

    def __post_init__(self) -> None:
        self.positive_yaw_moment_at_limit = finite_scalar(
            self.positive_yaw_moment_at_limit,
            "positive_yaw_moment_at_limit",
        )
        self.channel_limit = finite_scalar(self.channel_limit, "channel_limit")
        self.sign = finite_scalar(self.sign, "sign")
        self.negative_yaw_moment_at_limit = (
            self.positive_yaw_moment_at_limit
            if self.negative_yaw_moment_at_limit is None
            else finite_scalar(
                self.negative_yaw_moment_at_limit,
                "negative_yaw_moment_at_limit",
            )
        )
        if self.positive_yaw_moment_at_limit <= 0.0:
            raise ValueError("positive_yaw_moment_at_limit must be positive")
        if self.negative_yaw_moment_at_limit <= 0.0:
            raise ValueError("negative_yaw_moment_at_limit must be positive")
        if self.channel_limit <= 0.0:
            raise ValueError("channel_limit must be positive")
        if abs(self.sign) != 1.0:
            raise ValueError("sign must be +1 or -1")

    def convert(self, yaw_moment: float) -> float:
        moment = finite_scalar(yaw_moment, "yaw_moment")
        signed_moment = self.sign * moment
        scale = (
            self.positive_yaw_moment_at_limit
            if signed_moment >= 0.0
            else self.negative_yaw_moment_at_limit
        )
        raw = (
            signed_moment
            / scale
            * self.channel_limit
        )
        return float(np.clip(raw, -self.channel_limit, self.channel_limit))


@dataclass
class YawTrackerOutput:
    estimated_state: np.ndarray
    yaw_rate: float
    line_of_sight_angle: float
    yaw_channel: float
    yaw_control: YawControlResult
    mpc: YawMPCResult
    device_command: DeviceCommand
    thruster_allocation: ThrusterAllocation | None


@dataclass
class YawSafeControlOutput:
    force: np.ndarray
    yaw_moment: float
    yaw_channel: float
    device_command: DeviceCommand
    status: str


@dataclass
class _PendingRotatingPrediction:
    origin_index: int
    state1: np.ndarray
    state2: np.ndarray
    tau_base: np.ndarray
    steps: int = 0


class RotationAwareMPCTracker:
    """High-level yaw experiment entry point.

    ``yaw_rad`` and ``yaw_rate_rad_s`` must come from the IMU/attitude
    estimator at the camera exposure time.  The actual angle difference is
    used for filtering and historical model scoring; MPC uses yaw rate to form
    its frozen future rotation schedule.
    """

    def __init__(
        self,
        model: RotationAwareRelativeModel,
        estimator: RotationAwareKalmanFilter,
        controller: RotationAwareMPCController,
        yaw_controller: YawStateController,
        force_adapter: ForceCommandAdapter,
        yaw_adapter: YawMomentChannelAdapter,
        fusion: OnlineModelFusion | None = None,
        thruster_allocator: FineSUBThrusterAllocator | None = None,
    ) -> None:
        if estimator.model is not model or controller.model is not model:
            raise ValueError("model, estimator.model, and controller.model must match")
        if yaw_controller.model is not model.yaw:
            raise ValueError("yaw_controller must use model.yaw")
        self.model = model
        self.estimator = estimator
        self.controller = controller
        self.yaw_controller = yaw_controller
        self.force_adapter = force_adapter
        self.yaw_adapter = yaw_adapter
        self.fusion = fusion or build_default_staircase_fusion()
        self.thruster_allocator = thruster_allocator
        self._yaw_previous: float | None = None
        self._previous_achieved_force: np.ndarray | None = None
        self._pending: deque[_PendingRotatingPrediction] = deque()
        self._frame_index = -1

    def latch_baseline(
        self,
        force_achieved,
        yaw_moment_achieved: float = 0.0,
        yaw_rad: float | None = None,
    ) -> None:
        force = vector3(force_achieved, "force_achieved")
        moment = finite_scalar(yaw_moment_achieved, "yaw_moment_achieved")
        self.model.yaw.set_yaw_moment_base(moment)
        self._yaw_previous = (
            None if yaw_rad is None else finite_scalar(yaw_rad, "yaw_rad")
        )
        self.estimator.reset()
        self.fusion.reset()
        self.controller.reset()
        self.yaw_controller.reset(yaw_rad)
        self._pending.clear()
        self._frame_index = -1
        self._previous_achieved_force = force.copy()

    def target_lost(
        self,
        force_achieved_previous,
        yaw_moment_achieved_previous: float,
    ) -> YawSafeControlOutput:
        achieved_force = vector3(
            force_achieved_previous, "force_achieved_previous"
        )
        achieved_moment = finite_scalar(
            yaw_moment_achieved_previous, "yaw_moment_achieved_previous"
        )
        moment = self.yaw_controller.safe_moment(achieved_moment)
        force = self.controller.safe_force(
            achieved_force,
            yaw_moment=moment,
        )

        # A measurement gap has unknown duration in this interface.  Discard
        # all target-dependent state so the next valid camera frame starts a
        # fresh track instead of compressing the whole gap into one dt step or
        # completing a stale yaw goal.
        self.estimator.reset()
        self.fusion.reset()
        self.controller.reset()
        self.yaw_controller.reset()
        self._pending.clear()
        self._yaw_previous = None
        self._previous_achieved_force = None
        self._frame_index = -1

        yaw_channel = self.yaw_adapter.convert(moment)
        return YawSafeControlOutput(
            force=force,
            yaw_moment=moment,
            yaw_channel=yaw_channel,
            device_command=self.force_adapter.convert(force),
            status="target_lost:return_to_restoring_force_and_yaw_baseline",
        )

    def _score_completed_predictions(
        self,
        filtered_position: np.ndarray,
        force_achieved: np.ndarray,
        delta_yaw: float,
    ) -> None:
        retained: deque[_PendingRotatingPrediction] = deque()
        for record in self._pending:
            record.state1 = self.model.predict_model1(
                record.state1,
                force_achieved,
                record.tau_base,
                delta_yaw,
            )
            record.state2 = self.model.predict_model2(
                record.state2,
                force_achieved,
                delta_yaw,
            )
            record.steps += 1
            self.fusion.observe_position(
                actual_position=filtered_position,
                prediction1=record.state1[:3],
                prediction2=record.state2[:3],
                horizon_step=record.steps,
                origin_index=record.origin_index,
                target_index=self._frame_index,
            )
            if record.steps < self.fusion.config.prediction_horizon:
                retained.append(record)
        self._pending = retained

    def _start_prediction(self, state: np.ndarray, tau_base: np.ndarray) -> None:
        self._pending.append(
            _PendingRotatingPrediction(
                origin_index=self._frame_index,
                state1=state.copy(),
                state2=state.copy(),
                tau_base=tau_base.copy(),
            )
        )

    def update(
        self,
        position_body,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        force_achieved_previous,
        yaw_moment_achieved_previous: float,
        reference_position=None,
        roll_pitch_control=(0.0, 0.0),
        rotation_visibility_from_body=None,
        camera_origin_in_body=None,
    ) -> YawTrackerOutput:
        force = vector3(force_achieved_previous, "force_achieved_previous")
        yaw = finite_scalar(yaw_rad, "yaw_rad")
        yaw_rate = finite_scalar(yaw_rate_rad_s, "yaw_rate_rad_s")
        yaw_moment = finite_scalar(
            yaw_moment_achieved_previous, "yaw_moment_achieved_previous"
        )
        roll_pitch = np.asarray(roll_pitch_control, dtype=float).reshape(-1)
        if roll_pitch.shape != (2,) or not np.all(np.isfinite(roll_pitch)):
            raise ValueError("roll_pitch_control must be finite with shape (2,)")
        visibility_rotation, camera_origin = visibility_frame_geometry(
            self.controller.config.rotation_visibility_from_body
            if rotation_visibility_from_body is None
            else rotation_visibility_from_body,
            self.controller.config.camera_origin_in_body
            if camera_origin_in_body is None
            else camera_origin_in_body,
        )
        tau_base_for_interval = (
            force.copy()
            if self._previous_achieved_force is None
            else self._previous_achieved_force.copy()
        )

        self._frame_index += 1
        if not self.estimator.initialized:
            state = self.estimator.initialize(position_body)
            self.fusion.advance_time(self._frame_index)
        else:
            if self._yaw_previous is None:
                raise RuntimeError("previous yaw is unavailable; re-latch or reset tracker")
            delta_yaw = wrap_angle(yaw - self._yaw_previous)
            old_state = self.estimator.x.copy()
            prediction1 = self.model.predict_model1(
                old_state,
                force,
                tau_base_for_interval,
                delta_yaw,
            )
            prediction2 = self.model.predict_model2(
                old_state,
                force,
                delta_yaw,
            )
            weight6 = np.diag(
                np.concatenate(
                    (self.fusion.model1_weight, self.fusion.model1_weight)
                )
            )
            fused_prediction = (
                weight6 @ prediction1 + (np.eye(6) - weight6) @ prediction2
            )
            self.estimator.predict_mean(fused_prediction, delta_yaw)
            state = self.estimator.update(position_body)
            self._score_completed_predictions(state[:3], force, delta_yaw)

        self._yaw_previous = yaw
        self._previous_achieved_force = force.copy()
        self._start_prediction(state, force)
        position_visibility = body_to_visibility_position(
            state[:3],
            visibility_rotation,
            camera_origin,
        )
        alpha = line_of_sight_angle(position_visibility)
        yaw_control = self.yaw_controller.update(
            yaw_angle=yaw,
            yaw_rate=yaw_rate,
            alpha=alpha,
            previous_achieved_moment=yaw_moment,
            horizon=self.controller.config.horizon,
        )
        result = self.controller.solve(
            state=state,
            force_previous=force,
            yaw_prediction=yaw_control.prediction,
            reference_position=reference_position,
            model1_weight=self.fusion.model1_weight,
            rotation_visibility_from_body=visibility_rotation,
            camera_origin_in_body=camera_origin,
        )
        yaw_channel = self.yaw_adapter.convert(result.yaw_moment)
        attitude_control = (roll_pitch[0], roll_pitch[1], yaw_channel)
        allocation = (
            None
            if self.thruster_allocator is None
            else self.thruster_allocator.allocate(
                result.force, attitude_control=attitude_control
            )
        )
        return YawTrackerOutput(
            estimated_state=state,
            yaw_rate=yaw_rate,
            line_of_sight_angle=alpha,
            yaw_channel=yaw_channel,
            yaw_control=yaw_control,
            mpc=result,
            device_command=self.force_adapter.convert(result.force),
            thruster_allocation=allocation,
        )
