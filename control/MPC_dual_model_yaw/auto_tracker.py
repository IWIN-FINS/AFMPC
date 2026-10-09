"""Fail-closed real-vehicle builder for the rotation-aware dual MPC.

The maintained translation AUTO builder remains the single source of truth
for M/D/dt, Kalman, fusion, MPC weights, limits, solver settings and camera
geometry.  This module wraps that validated translation model with explicit
yaw dynamics/control and with the real M1..M8 planar actuator geometry.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from MPC_dual_model.auto_tracker import (
    AutoParameterError,
    build_auto_tracker as build_translation_auto_tracker,
)
from MPC_dual_model.finesub_protocol import build_runtime_hardware_adapter

from .yaw_controller import YawControlConfig, YawStateController
from .yaw_kalman import RotationAwareKalmanFilter
from .yaw_mpc_controller import RotationAwareMPCController, YawMPCConfig
from .yaw_relative_model import LinearYawDynamics, RotationAwareRelativeModel
from .yaw_tracker import RotationAwareMPCTracker, YawMomentChannelAdapter


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoParameterError(f"{name} must be an object")
    return value


def _required(mapping: dict[str, Any], name: str, keys: set[str]) -> None:
    missing = sorted(key for key in keys if mapping.get(key) is None)
    if missing:
        raise AutoParameterError(f"{name} is missing: {', '.join(missing)}")


def _strict_dataclass_kwargs(
    mapping: dict[str, Any], dataclass_type: type, name: str
) -> dict[str, Any]:
    expected = {item.name for item in fields(dataclass_type)}
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing:
        raise AutoParameterError(f"{name} is missing: {', '.join(missing)}")
    if unknown:
        raise AutoParameterError(f"{name} has unknown fields: {', '.join(unknown)}")
    return dict(mapping)


def _shared_translation_mpc_kwargs(base) -> dict[str, Any]:
    """Copy every field shared by the translation and yaw-aware MPC configs.

    The yaw controller has a different actuator constraint representation, so
    its thruster-wrench fields are supplied by the yaw-specific builder below.
    All other overlapping fields must come from the maintained translation
    controller instead of being duplicated here.
    """

    yaw_fields = {item.name for item in fields(YawMPCConfig)}
    return {
        name: getattr(base, name)
        for name in yaw_fields
        if hasattr(base, name)
        and name
        not in {
            "thruster_wrench_matrix",
            "thruster_force_min",
            "thruster_force_max",
        }
    }


def _real_planar_actuator_model(
    runtime_config: dict[str, Any], yaw_active: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real-motor-order [Fx,Fy,Fz,Nz] mapping and force bounds."""

    actuator = _mapping(
        yaw_active.get("actuator_model"),
        "auto_runtime.active_yaw_parameters.actuator_model",
    )
    _required(
        actuator,
        "auto_runtime.active_yaw_parameters.actuator_model",
        {"enabled", "thruster_force_limit_scale"},
    )
    if actuator["enabled"] is not True:
        raise AutoParameterError("yaw planar actuator model is not enabled")
    scale = float(actuator["thruster_force_limit_scale"])
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise AutoParameterError("thruster_force_limit_scale must be in (0, 1]")

    geometry = _mapping(runtime_config.get("thruster_geometry"), "thruster_geometry")
    feedback = _mapping(runtime_config.get("thruster_feedback"), "thruster_feedback")
    prior = _mapping(feedback.get("rpm_force_prior"), "thruster_feedback.rpm_force_prior")
    _required(
        geometry,
        "thruster_geometry",
        {
            "positive_throttle_force_directions_frd_m1_m8",
            "yaw_moment_arm_about_cad_origin_m_per_positive_force_m1_m8",
        },
    )
    _required(
        prior,
        "thruster_feedback.rpm_force_prior",
        {
            "positive_force_limit_prior_n_m1_m8",
            "negative_force_limit_abs_prior_n_m1_m8",
        },
    )
    directions = np.asarray(
        geometry["positive_throttle_force_directions_frd_m1_m8"], dtype=float
    )
    yaw_arms = np.asarray(
        geometry["yaw_moment_arm_about_cad_origin_m_per_positive_force_m1_m8"],
        dtype=float,
    ).reshape(-1)
    positive = np.asarray(
        prior["positive_force_limit_prior_n_m1_m8"], dtype=float
    ).reshape(-1)
    negative = np.asarray(
        prior["negative_force_limit_abs_prior_n_m1_m8"], dtype=float
    ).reshape(-1)
    if directions.shape != (8, 3) or not np.all(np.isfinite(directions)):
        raise AutoParameterError("real M1..M8 force directions must have shape (8, 3)")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-5):
        raise AutoParameterError("real M1..M8 force directions must be unit vectors")
    if any(value.shape != (8,) for value in (yaw_arms, positive, negative)):
        raise AutoParameterError("real M1..M8 yaw arms/force limits must contain 8 values")
    if (
        not np.all(np.isfinite(yaw_arms))
        or not np.all(np.isfinite(positive))
        or not np.all(np.isfinite(negative))
        or np.any(positive <= 0.0)
        or np.any(negative <= 0.0)
    ):
        raise AutoParameterError("real M1..M8 yaw arms/force limits are invalid")

    # Each scalar motor force is expressed along its measured positive-throttle
    # direction.  Keeping M1..M8 order aligns this QP constraint with telemetry
    # and the firmware mixer; the older Unity V/H canonical order did not.
    wrench = np.vstack((directions.T, yaw_arms))
    if np.linalg.matrix_rank(wrench) < 4:
        raise AutoParameterError("real planar M1..M8 wrench matrix is rank deficient")
    return wrench, -scale * negative, scale * positive


def build_auto_tracker(runtime_config: dict[str, Any]) -> RotationAwareMPCTracker:
    """Construct the strict yaw-aware tracker used by real AUTO entries."""

    # This validates and constructs every maintained translation parameter;
    # do not duplicate that mapping here or the two entry paths can drift.
    translation_tracker = build_translation_auto_tracker(runtime_config)
    auto = _mapping(runtime_config.get("auto_runtime"), "auto_runtime")
    yaw_active = _mapping(
        auto.get("active_yaw_parameters"),
        "auto_runtime.active_yaw_parameters",
    )
    if yaw_active.get("enabled_for_control") is not True:
        raise AutoParameterError("active_yaw_parameters is not enabled for control")

    dynamics_data = _mapping(
        yaw_active.get("dynamics"),
        "auto_runtime.active_yaw_parameters.dynamics",
    )
    _required(
        dynamics_data,
        "auto_runtime.active_yaw_parameters.dynamics",
        {
            "effective_inertia_kg_m2",
            "linear_damping_n_m_per_rad_s",
            "quadratic_damping_n_m_per_rad_s2",
        },
    )
    yaw_dynamics = LinearYawDynamics(
        effective_inertia=float(dynamics_data["effective_inertia_kg_m2"]),
        linear_damping=float(dynamics_data["linear_damping_n_m_per_rad_s"]),
        quadratic_damping=float(dynamics_data["quadratic_damping_n_m_per_rad_s2"]),
        dt=translation_tracker.model.dt,
    )
    model = RotationAwareRelativeModel(translation_tracker.model, yaw_dynamics)

    yaw_controller_data = _mapping(
        yaw_active.get("controller"),
        "auto_runtime.active_yaw_parameters.controller",
    )
    yaw_control_config = YawControlConfig(
        **_strict_dataclass_kwargs(
            yaw_controller_data,
            YawControlConfig,
            "auto_runtime.active_yaw_parameters.controller",
        )
    )
    yaw_controller = YawStateController(yaw_dynamics, yaw_control_config)

    base = translation_tracker.controller.config
    wrench, thruster_min, thruster_max = _real_planar_actuator_model(
        runtime_config, yaw_active
    )
    yaw_mpc_kwargs = _shared_translation_mpc_kwargs(base)
    yaw_mpc_kwargs.update(
        thruster_wrench_matrix=wrench,
        thruster_force_min=thruster_min,
        thruster_force_max=thruster_max,
    )
    controller = RotationAwareMPCController(
        model,
        YawMPCConfig(**yaw_mpc_kwargs),
    )
    estimator = RotationAwareKalmanFilter(
        model, translation_tracker.estimator.config
    )
    hardware = build_runtime_hardware_adapter(runtime_config)
    yaw_adapter = YawMomentChannelAdapter(
        positive_yaw_moment_at_limit=hardware.positive_yaw_moment_at_limit,
        negative_yaw_moment_at_limit=hardware.negative_yaw_moment_at_limit,
        channel_limit=hardware.yaw_channel_limit,
        sign=hardware.yaw_sign,
    )
    tracker = RotationAwareMPCTracker(
        model=model,
        estimator=estimator,
        controller=controller,
        yaw_controller=yaw_controller,
        force_adapter=translation_tracker.adapter,
        yaw_adapter=yaw_adapter,
        fusion=translation_tracker.fusion,
        # The real command path uses FineSUBHardwareAdapter and the firmware
        # mixer.  Exposing a second software allocator here would be misleading.
        thruster_allocator=None,
    )
    camera = _mapping(
        auto.get("active_camera_transform"),
        "auto_runtime.active_camera_transform",
    )
    tracker.rotation_body_from_camera = np.asarray(
        camera["rotation_body_from_camera"], dtype=float
    )
    tracker.camera_origin_in_body = np.asarray(
        camera["camera_origin_in_body_frd_m"], dtype=float
    )
    return tracker
