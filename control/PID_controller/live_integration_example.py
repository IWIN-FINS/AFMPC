"""FineSUB integration entry point for the pure PID controller."""

from __future__ import annotations

import numpy as np

try:
    from .camera_transform import (
        PID_CONTROL_REFERENCE_POSITION_BODY,
        camera_to_body_position,
        camera_to_pid_body_position,
    )
    from .device_adapter import (
        FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        FINESUB_V4_PRO1_FORCE_POSITIVE_N,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        YawMomentChannelAdapter,
        finesub_translation_thruster_force_matrix,
    )
    from .pid_controller import PIDConfig, RelativePIDController
    from .pid_tracker import PIDTracker
    from .yaw_pid_controller import YawPIDConfig, YawPIDController
except ImportError:
    from camera_transform import (
        PID_CONTROL_REFERENCE_POSITION_BODY,
        camera_to_body_position,
        camera_to_pid_body_position,
    )
    from device_adapter import (
        FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        FINESUB_V4_PRO1_FORCE_POSITIVE_N,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        YawMomentChannelAdapter,
        finesub_translation_thruster_force_matrix,
    )
    from pid_controller import PIDConfig, RelativePIDController
    from pid_tracker import PIDTracker
    from yaw_pid_controller import YawPIDConfig, YawPIDController


def build_tracker(*, calibrated_reference: bool = False) -> PIDTracker:
    # Keep the PID experiment inside the same conservative force envelope
    # used by the currently authorised MPC runtime.  The physical thruster
    # matrix still supplies the hard absolute envelope below, but the PID
    # experiment must not jump straight to that much larger legacy limit.
    force_min = np.asarray((-5.05068, -4.783308, -6.86714), dtype=float)
    force_max = np.asarray((4.730162, 4.997534, 7.06314), dtype=float)
    reference_position = (
        PID_CONTROL_REFERENCE_POSITION_BODY.copy()
        if calibrated_reference
        else np.asarray((0.80, 0.0, 0.0), dtype=float)
    )
    config = PIDConfig(
        dt=0.05,
        # Keep the user-facing PID target at the established 0.80 m forward
        # standoff.  The calibrated camera mount is applied before this
        # reference in the real hardware/camera-window entry points.
        reference_position=reference_position,
        # Retune after the latest real-vehicle replay: Kp=[12,15,18] left
        # appreciable forward/lateral tracking lag.  The 20 Hz PID loop only
        # receives a new stereo position about every 0.10 s, while rejected
        # jumps are held; the previous full I/D trial therefore increased
        # command reversals instead of adding useful damping.  Keep the
        # higher P authority and add only a small planar I term for the
        # sustained forward/right bias seen in the latest replay.  Depth I
        # stays disabled because its stereo noise accumulated in the held
        # samples; D stays disabled until a variable-rate derivative is used.
        kp=(28.0, 35.0, 42.0),
        ki=(0.10, 0.15, 0.0),
        kd=(0.0, 0.0, 0.0),
        derivative_filter_time_constant=0.35,
        integral_limit=(2.0, 1.5, 1.2),
        force_min=force_min,
        force_max=force_max,
        # Match the active MPC runtime envelope in
        # ``MPC_dual_model/finesub_v4pro1_mpc.json``.  These are per-20 Hz
        # force-step limits, not PID gains; keeping them identical avoids a
        # different acceleration/braking authority when switching models.
        delta_force_min=(-1.2, -0.8, -1.0),
        delta_force_max=(1.2, 0.8, 1.0),
        thruster_force_matrix=finesub_translation_thruster_force_matrix(),
        thruster_force_min=-FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        thruster_force_max=FINESUB_V4_PRO1_FORCE_POSITIVE_N,
    )
    controller = RelativePIDController(config)
    adapter = ForceCommandAdapter(
        positive_force_at_limit=force_max,
        signs=(1.0, 1.0, 1.0),
        command_limits=(99.0, 99.0, 45.0),
    )
    allocator = FineSUBThrusterAllocator(
        positive_force_at_limit=force_max,
        # Active MPC experimental_auto.max_channel_abs = 0.20.
        translation_channel_limits=(0.20, 0.20, 0.20),
        enable_depth=True,
    )
    yaw_controller = YawPIDController(
        YawPIDConfig(
            dt=0.05,
            kp=1.8,
            ki=0.12,
            kd=0.55,
            yaw_moment_min=-2.041126,
            yaw_moment_max=1.807854,
            # Match active MPC yaw slew limits while holding the startup yaw
            # angle.  The yaw reference is latched from the first valid IMU
            # sample; it is not recomputed from the target bearing.
            delta_yaw_moment_min=-0.5,
            delta_yaw_moment_max=0.5,
        )
    )
    yaw_adapter = YawMomentChannelAdapter(
        positive_yaw_moment_at_limit=2.0,
        channel_limit=0.20,
        sign=1.0,
    )
    tracker = PIDTracker(
        controller,
        adapter,
        allocator,
        yaw_controller=yaw_controller,
        yaw_adapter=yaw_adapter,
        # Hold the current yaw angle.  ``PIDTracker.latch_baseline`` latches
        # the first valid IMU yaw as the reference for the yaw PID.
        track_target_bearing=False,
    )
    # Keep the frame selection with the tracker so the small integration
    # helper cannot silently pair a calibrated reference with the old aligned
    # camera axes.  This is metadata only; it does not add a model/observer.
    tracker.camera_calibrated = bool(calibrated_reference)
    return tracker


def one_control_update(
    tracker: PIDTracker,
    position_camera_xyz: object,
    last_achieved_force_body: object,
    imu_yaw_rad: float | None = None,
    imu_yaw_rate_rad_s: float | None = None,
    last_achieved_yaw_moment: float = 0.0,
    reference_yaw_rad: float | None = None,
    calibrated_camera: bool | None = None,
):
    """Call once for every new stereo target position.

    A tracker created with ``build_tracker(calibrated_reference=True)``
    automatically selects the calibrated path.  ``calibrated_camera`` can
    explicitly override that selection; the historical aligned default
    remains available for old offline callers.
    """
    use_calibrated_camera = (
        bool(getattr(tracker, "camera_calibrated", False))
        if calibrated_camera is None
        else bool(calibrated_camera)
    )
    return tracker.update(
        (
            camera_to_pid_body_position(position_camera_xyz)
            if use_calibrated_camera
            else camera_to_body_position(position_camera_xyz)
        ),
        last_achieved_force_body,
        yaw_rad=imu_yaw_rad,
        yaw_rate_rad_s=imu_yaw_rate_rad_s,
        achieved_yaw_moment_previous=last_achieved_yaw_moment,
        reference_yaw_rad=reference_yaw_rad,
    )


# Existing-loop outline:
# tracker = build_tracker(calibrated_reference=True)
# last_force_body = np.zeros(3)
# last_yaw_moment = 0.0
# tracker.latch_baseline(last_force_body, last_yaw_moment, imu_yaw_rad)
# output = one_control_update(
#     tracker, position_camera, last_force_body,
#     imu_yaw_rad, imu_yaw_rate_rad_s, last_yaw_moment,
#     calibrated_camera=True,
# )
# send_xy_2(output.device_command.planar_forward,
#           output.device_command.planar_right)
# send_depth(output.device_command.depth_force)
# send_yaw_direct(output.yaw_channel)
# last_force_body = output.pid.force.copy()
# last_yaw_moment = output.yaw_pid.yaw_moment
#
# Use either device_command (MCU mixes) or thruster_allocation.throttles
# (Python drives eight ESCs), never both.
