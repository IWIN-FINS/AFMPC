#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy joystick/manual + CSRT viewer for FineSUB.

AUTO is permanently unavailable here.  Formal real-vehicle AUTO uses
``finesub_auto_control.py`` and the external red-fish JSONL measurement gate.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np

from MPC_dual_model.finesub_protocol import (
    FineSUBControlCommand,
    FineSUBHardwareAdapter,
)
from MPC_dual_model.finesub_transport import (
    FineSUBConnection,
    load_runtime_config,
    make_transport,
)
from MPC_model1 import live_integration_example as model1_live
from MPC_model2 import live_integration_example as model2_live
from MPC_dual_model import live_integration_example as dual_live
from MPC_dual_model_yaw import live_integration_example as dual_yaw_live


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "MPC_dual_model"
    / "finesub_v4pro1_mpc.json"
)

# Range calibration inherited from the checked tracking program.  Replace
# these two values with a pool measurement of the selected target/ROI.
RANGE_REFERENCE_M = 0.50
TARGET_WIDTH_AT_REFERENCE_PX = 90.0
WIDTH_FILTER_LENGTH = 6

# These half-FOV values match the live MPC configuration.
HORIZONTAL_HALF_FOV_DEG = 42.0
VERTICAL_HALF_FOV_DEG = 30.0

MANUAL_DEADBAND = 0.10
LEGACY_AUTO_AVAILABLE = False

MODEL_MODULES = {
    "model1": model1_live,
    "model2": model2_live,
    "dual": dual_live,
    "dual-yaw": dual_yaw_live,
}


class LiveModelBackend:
    """Normalize the four tracker APIs into one FineSUB hardware interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.module = MODEL_MODULES[name]
        self.tracker = self.module.build_tracker()
        self.yaw_direct = name == "dual-yaw"

    def latch_baseline(self, force, yaw_moment: float, yaw_rad: float) -> None:
        if self.yaw_direct:
            self.tracker.latch_baseline(force, yaw_moment, yaw_rad)
        else:
            self.tracker.latch_baseline(force)

    def execution_feedback(self, telemetry, hardware_adapter):
        """Return the lower-controller output used as the next MPC input."""

        return hardware_adapter.achieved_wrench(telemetry)

    def update(
        self,
        position_camera,
        force,
        yaw_moment: float,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        hardware_adapter,
    ):
        if self.yaw_direct:
            output = self.module.one_control_update(
                tracker=self.tracker,
                position_camera_xyz=position_camera,
                imu_yaw_rad=yaw_rad,
                imu_yaw_rate_rad_s=yaw_rate_rad_s,
                last_achieved_force_body=force,
                last_achieved_yaw_moment=yaw_moment,
            )
            detail = f"{output.yaw_control.mode.value} QP={output.mpc.status}"
        else:
            output = self.module.one_control_update(
                self.tracker,
                position_camera,
                force,
            )
            detail = f"QP={output.mpc.status}"
        command = self.module.to_finesub_command(
            output,
            armed=True,
            adapter=hardware_adapter,
        )
        return command, detail

    def target_lost(self, force, yaw_moment: float, hardware_adapter):
        if self.yaw_direct:
            output = self.tracker.target_lost(force, yaw_moment)
        else:
            output = self.tracker.target_lost(force)
        return self.module.to_finesub_command(
            output,
            armed=True,
            adapter=hardware_adapter,
        )


def zero_command(*, armed: bool) -> FineSUBControlCommand:
    return FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, armed)


def shaped_axis(value: float, deadband: float = MANUAL_DEADBAND) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if abs(value) <= deadband:
        return 0.0
    magnitude = (abs(value) - deadband) / (1.0 - deadband)
    return float(np.copysign(magnitude, value))


def manual_command(
    left_horizontal: float,
    left_vertical: float,
    right_horizontal: float,
    right_vertical: float,
    *,
    armed: bool,
    translation_channel_limits=(0.10, 0.10, 0.10),
    yaw_channel_limit: float = 0.10,
) -> FineSUBControlCommand:
    # FRD convention: stick up -> forward, stick right -> right/down/yaw-right.
    limits = np.asarray(translation_channel_limits, dtype=float)
    if limits.shape != (3,) or np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("translation_channel_limits must be three positive values")
    if not np.isfinite(yaw_channel_limit) or yaw_channel_limit <= 0.0:
        raise ValueError("yaw_channel_limit must be positive")
    return FineSUBControlCommand(
        forward=-float(limits[0]) * shaped_axis(left_vertical),
        right=float(limits[1]) * shaped_axis(left_horizontal),
        down=float(limits[2]) * shaped_axis(right_vertical),
        yaw=float(yaw_channel_limit) * shaped_axis(right_horizontal),
        armed=armed,
    )


def camera_position_from_bbox(
    center_x: float,
    center_y: float,
    smoothed_width: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Estimate OpenCV camera [right, down, forward] position in metres."""

    width = max(float(smoothed_width), 1.0)
    forward = RANGE_REFERENCE_M * TARGET_WIDTH_AT_REFERENCE_PX / width
    focal_x = image_width / (2.0 * np.tan(np.deg2rad(HORIZONTAL_HALF_FOV_DEG)))
    focal_y = image_height / (2.0 * np.tan(np.deg2rad(VERTICAL_HALF_FOV_DEG)))
    right = (center_x - image_width / 2.0) * forward / focal_x
    down = (center_y - image_height / 2.0) * forward / focal_y
    return np.array([right, down, forward], dtype=float)


def create_csrt_tracker():
    import cv2

    legacy = getattr(cv2, "legacy", cv2)
    return legacy.TrackerCSRT_create()


def build_video_pipeline():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    pipeline_description = (
        "udpsrc port=5600 ! "
        "application/x-rtp,encoding-name=H264,payload=96 ! "
        "rtpjitterbuffer latency=40 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink name=mysink max-buffers=1 drop=true"
    )
    pipeline = Gst.parse_launch(pipeline_description)
    appsink = pipeline.get_by_name("mysink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    pipeline.set_state(Gst.State.PLAYING)
    return Gst, pipeline, appsink


def read_gstreamer_frame(Gst, appsink):
    sample = appsink.emit("pull-sample")
    if sample is None:
        return None
    buffer = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    width = caps.get_value("width")
    height = caps.get_value("height")
    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        return np.frombuffer(map_info.data, np.uint8).reshape(
            (height, width, 3)
        ).copy()
    finally:
        buffer.unmap(map_info)


def main(
    model_name: str = "dual-yaw",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        import cv2
        import pygame
    except ImportError as error:
        raise RuntimeError(
            "live control dependencies are missing; install "
            "MPC_dual_model_yaw/requirements-live.txt"
        ) from error

    runtime_config = load_runtime_config(config_path)
    control_config = runtime_config.get("control", {})
    transport_config = runtime_config.get("transport", {})
    adapter_config = runtime_config.get("hardware_adapter", {})
    hardware_adapter = FineSUBHardwareAdapter(**adapter_config)
    backend = LiveModelBackend(model_name)
    connection = FineSUBConnection(
        make_transport(transport_config),
        telemetry_max_age_s=float(
            control_config.get("telemetry_max_age_sec", 0.25)
        ),
        confirmation_max_age_s=float(
            control_config.get("confirmation_max_age_sec", 0.25)
        ),
        reconnect_interval_s=float(
            transport_config.get("reconnect_interval_sec", 1.0)
        ),
    )
    control_period_s = float(control_config.get("period_sec", 0.05))
    connection.connect()

    Gst, pipeline, appsink = build_video_pipeline()
    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print("[JOYSTICK] Detected")
    else:
        print("[JOYSTICK] Not detected")

    vision_tracker = create_csrt_tracker()
    tracking = False
    bbox = None
    width_history: deque[float] = deque(maxlen=WIDTH_FILTER_LENGTH)
    armed = False
    auto_mode = False
    mpc_latched = False
    rearm_required = False
    last_control_time = 0.0
    status_text = "DISARMED"
    command = zero_command(armed=False)

    print("Press S to select ROI, Z to stop tracking, Q to quit")
    print("Button 7 arms/disarms; button 3 AUTO is permanently disabled here")
    print(f"[MPC] Selected model: {model_name}")
    print(f"[CONFIG] {Path(config_path).resolve()}")
    print(
        "This legacy program has no AUTO authority; formal AUTO uses the "
        "separate fail-closed entry"
    )

    try:
        while True:
            frame = read_gstreamer_frame(Gst, appsink)
            if frame is None:
                connection.poll_telemetry()
                continue
            image_height, image_width, _ = frame.shape
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("z"):
                tracking = False
                width_history.clear()
                mpc_latched = False
                print("[TRACK] Stopped")
            if key == ord("s"):
                bbox = cv2.selectROI("Track", frame, False)
                vision_tracker = create_csrt_tracker()
                vision_tracker.init(frame, bbox)
                tracking = True
                width_history.clear()
                mpc_latched = False
                print("[TRACK] Started")

            for event in pygame.event.get():
                if event.type != pygame.JOYBUTTONDOWN:
                    continue
                if event.button == 7:
                    armed = not armed
                    rearm_required = False
                    if not armed:
                        mpc_latched = False
                    command = zero_command(armed=armed)
                    connection.send(command)
                    print("[ARM]", "ON" if armed else "OFF")
                elif event.button == 3:
                    auto_mode = False
                    mpc_latched = False
                    print(
                        "[AUTO BLOCKED] Legacy joystick/CSRT AUTO is disabled; "
                        "use finesub_auto_control.py"
                    )

            left_h = left_v = right_h = right_v = 0.0
            if joystick is not None:
                try:
                    left_h = joystick.get_axis(0)
                    left_v = joystick.get_axis(1)
                    right_h = joystick.get_axis(3)
                    right_v = joystick.get_axis(4)
                except pygame.error:
                    pass

            target_visible = False
            position_camera = None
            estimated_distance = None
            if tracking:
                target_visible, bbox = vision_tracker.update(frame)
                if target_visible:
                    x, y, box_width, box_height = bbox
                    center_x = x + box_width / 2.0
                    center_y = y + box_height / 2.0
                    cv2.rectangle(
                        frame,
                        (int(x), int(y)),
                        (int(x + box_width), int(y + box_height)),
                        (0, 255, 0),
                        2,
                    )
                    width_history.append(box_width)
                    smoothed_width = sum(width_history) / len(width_history)
                    position_camera = camera_position_from_bbox(
                        center_x,
                        center_y,
                        smoothed_width,
                        image_width,
                        image_height,
                    )
                    estimated_distance = float(position_camera[2])

            now = time.monotonic()
            telemetry = connection.fresh_telemetry()
            if telemetry is not None and telemetry.failsafe:
                if armed:
                    print("[SAFETY] Lower-controller failsafe; re-arm required")
                armed = False
                rearm_required = True
                mpc_latched = False
            elif (
                telemetry is not None
                and telemetry.last_command_rejected
                and telemetry.last_command_session == connection.session_id
            ):
                if armed:
                    print(
                        "[SAFETY] Lower controller rejected command: "
                        f"flags=0x{telemetry.reject_flags:08x}"
                    )
                armed = False
                rearm_required = True
                mpc_latched = False
            elif armed and telemetry is None:
                print("[SAFETY] Telemetry stale; re-arm required")
                armed = False
                rearm_required = True
                mpc_latched = False
            elif (
                armed
                and auto_mode
                and telemetry is not None
                and not telemetry.execution_feedback_valid
            ):
                print("[SAFETY] AUTO execution feedback unavailable; re-arm required")
                armed = False
                rearm_required = True
                mpc_latched = False
            elif (
                armed
                and connection.last_confirmed_armed is True
                and not connection.confirmation_fresh()
            ):
                print("[SAFETY] Command confirmation stale; re-arm required")
                armed = False
                rearm_required = True
                mpc_latched = False
            if now - last_control_time >= control_period_s:
                last_control_time = now
                if not armed:
                    command = zero_command(armed=False)
                    status_text = (
                        "DISARMED: RE-ARM REQUIRED"
                        if rearm_required
                        else "DISARMED"
                    )
                elif not connection.armed_confirmation_fresh():
                    command = zero_command(armed=True)
                    mpc_latched = False
                    status_text = "ARMING: WAIT EXECUTION ACK"
                elif not auto_mode:
                    command = manual_command(
                        left_h,
                        left_v,
                        right_h,
                        right_v,
                        armed=True,
                        translation_channel_limits=(
                            hardware_adapter.translation_channel_limits
                        ),
                        yaw_channel_limit=hardware_adapter.yaw_channel_limit,
                    )
                    status_text = "MANUAL"
                elif telemetry is None:
                    # The pre-check above normally handles this branch.  Keep
                    # it as a defensive stop if telemetry changes mid-cycle.
                    armed = False
                    rearm_required = True
                    command = zero_command(armed=False)
                    mpc_latched = False
                    status_text = "AUTO: WAIT TELEMETRY"
                else:
                    achieved_force, achieved_yaw_moment = backend.execution_feedback(
                        telemetry,
                        hardware_adapter,
                    )
                    yaw_rad = telemetry.yaw_rad
                    yaw_rate_rad_s = telemetry.yaw_rate_rad_s
                    if target_visible and position_camera is not None:
                        if not mpc_latched:
                            backend.latch_baseline(
                                achieved_force,
                                achieved_yaw_moment,
                                yaw_rad,
                            )
                            mpc_latched = True
                        command, detail = backend.update(
                            position_camera,
                            achieved_force,
                            achieved_yaw_moment,
                            yaw_rad,
                            yaw_rate_rad_s,
                            hardware_adapter,
                        )
                        status_text = f"AUTO:{model_name} {detail}"
                    else:
                        command = backend.target_lost(
                            achieved_force,
                            achieved_yaw_moment,
                            hardware_adapter,
                        )
                        mpc_latched = False
                        status_text = f"AUTO:{model_name} TARGET LOST"
                connection.send(command)

            telemetry_text = "telemetry=STALE"
            if telemetry is not None:
                valid_rpm = [
                    rpm
                    for index, rpm in enumerate(telemetry.motor_rpm)
                    if telemetry.rpm_valid_mask & (1 << index)
                ]
                rpm_text = (
                    f"{max(valid_rpm):.0f}"
                    if valid_rpm
                    else "INVALID"
                )
                telemetry_text = (
                    f"yaw={np.rad2deg(telemetry.yaw_rad):.1f}deg "
                    f"r={np.rad2deg(telemetry.yaw_rate_rad_s):.1f}deg/s "
                    f"depth={telemetry.depth_m:.2f}m "
                    f"ack={telemetry.last_command_sequence} "
                    f"rpmMax={rpm_text}"
                )
                if telemetry.failsafe:
                    telemetry_text += " FAILSAFE"
                elif telemetry.last_command_rejected:
                    telemetry_text += f" REJECT=0x{telemetry.reject_flags:08x}"
                elif connection.confirmation_fresh():
                    telemetry_text += f" ACK {connection.last_round_trip_ms}ms"

            cv2.putText(
                frame,
                status_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                telemetry_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0) if telemetry is not None else (0, 0, 255),
                2,
            )
            if estimated_distance is not None:
                cv2.putText(
                    frame,
                    f"target z={estimated_distance:.2f}m "
                    f"cmd=({command.forward:+.2f},{command.right:+.2f},"
                    f"{command.down:+.2f},{command.yaw:+.2f})",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Track", frame)
    finally:
        try:
            connection.send(zero_command(armed=False))
        except Exception:
            pass
        connection.close()
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--model",
        choices=tuple(MODEL_MODULES),
        default="dual-yaw",
        help="MPC model used by AUTO mode",
    )
    argument_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="FineSUB transport and hardware calibration JSON",
    )
    arguments = argument_parser.parse_args()
    main(arguments.model, arguments.config)
