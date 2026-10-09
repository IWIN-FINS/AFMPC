#!/usr/bin/env python3
"""CSRT visual tracking + saturated SMC + FineSUB v3 live interface.

The video geometry and lower-controller safety flow follow rov_track_control3.py
and MPC_dual_model.  No propeller can be armed without fresh v3 telemetry and a
confirmed command session.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MPC_WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(MPC_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(MPC_WORKSPACE_ROOT))

from MPC_dual_model.finesub_protocol import (  # noqa: E402
    FineSUBControlCommand,
    FineSUBHardwareAdapter,
)
from MPC_dual_model.finesub_transport import (  # noqa: E402
    FineSUBConnection,
    load_runtime_config,
    make_transport,
)
from openauv_smc import (  # noqa: E402
    BBoxTargetEstimator,
    FineSUBCommandMapper,
    TelemetryStateEstimator,
    VisionConfig,
    build_default_controller,
    build_openauv_model,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "finesub_live.json"


def zero_command(*, armed: bool) -> FineSUBControlCommand:
    return FineSUBControlCommand(
        forward=0.0,
        right=0.0,
        down=0.0,
        yaw=0.0,
        armed=armed,
        yaw_direct=True,
    )


def create_csrt_tracker():
    import cv2

    legacy = getattr(cv2, "legacy", cv2)
    return legacy.TrackerCSRT_create()


def build_video_pipeline(pipeline_description: str):
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_description)
    appsink = pipeline.get_by_name("mysink")
    if appsink is None:
        raise ValueError("GStreamer pipeline must contain appsink name=mysink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    pipeline.set_state(Gst.State.PLAYING)
    return Gst, pipeline, appsink


def try_read_gstreamer_frame(Gst, appsink, timeout_s: float):
    sample = appsink.emit("try-pull-sample", int(timeout_s * Gst.SECOND))
    if sample is None:
        return None
    buffer = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    width = int(caps.get_value("width"))
    height = int(caps.get_value("height"))
    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        return np.frombuffer(map_info.data, np.uint8).reshape(
            (height, width, 3)
        ).copy()
    finally:
        buffer.unmap(map_info)


def build_vision_config(config: dict) -> VisionConfig:
    allowed = set(VisionConfig.__dataclass_fields__)
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown vision config keys: {sorted(unknown)}")
    return VisionConfig(**config)


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV live dependency is missing; install requirements-live.txt"
        ) from error

    runtime = load_runtime_config(config_path)
    transport_config = runtime.get("transport", {})
    control_config = runtime.get("control", {})
    video_config = runtime.get("video", {})
    hardware = FineSUBHardwareAdapter(**runtime.get("hardware_adapter", {}))
    vision = BBoxTargetEstimator(build_vision_config(runtime.get("vision", {})))

    model = build_openauv_model()
    controller = build_default_controller(
        model,
        heave_input_limit=float(hardware.positive_force_at_limit[2]),
        yaw_input_limit=float(hardware.positive_yaw_moment_at_limit),
    )
    command_mapper = FineSUBCommandMapper(hardware)
    state_estimator = TelemetryStateEstimator(
        rate_filter_tau_s=float(
            control_config.get("depth_rate_filter_tau_sec", 0.25)
        ),
        max_abs_heave_rate_m_s=float(
            control_config.get("max_abs_heave_rate_m_s", 1.0)
        ),
    )
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
    period_s = float(control_config.get("period_sec", 0.05))
    if period_s <= 0.0:
        raise ValueError("control.period_sec must be positive")

    default_pipeline = (
        "udpsrc port=5600 ! "
        "application/x-rtp,encoding-name=H264,payload=96 ! "
        "rtpjitterbuffer latency=40 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink name=mysink max-buffers=1 drop=true"
    )
    pipeline_description = str(video_config.get("pipeline", default_pipeline))
    sample_timeout_s = float(video_config.get("sample_timeout_sec", 0.01))

    pipeline = None
    connection.connect()
    Gst, pipeline, appsink = build_video_pipeline(pipeline_description)
    tracker = create_csrt_tracker()
    tracking = False
    latest_observation = None
    armed = False
    auto_mode = False
    rearm_required = False
    command = zero_command(armed=False)
    status = "DISARMED"
    latest_reference = None
    last_control_time = time.monotonic()

    print("Keys: S select ROI, Z stop tracking, A arm/disarm, M AUTO, Q quit")
    print(f"[CONFIG] {Path(config_path).resolve()}")
    print(
        "AUTO requires fresh telemetry, execution feedback, and matching "
        "session/sequence/CRC confirmation"
    )

    try:
        while True:
            frame = try_read_gstreamer_frame(Gst, appsink, sample_timeout_s)
            now = time.monotonic()

            if frame is not None:
                image_height, image_width, _ = frame.shape
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("z"):
                    tracking = False
                    latest_observation = None
                    latest_reference = None
                    vision.reset()
                    controller.reset()
                    print("[TRACK] Stopped")
                elif key == ord("s"):
                    if armed:
                        armed = False
                        rearm_required = True
                        connection.send(zero_command(armed=False))
                        print("[SAFETY] Disarmed before blocking ROI selection")
                    bbox = cv2.selectROI("Track", frame, False)
                    if bbox[2] > 1.0 and bbox[3] > 1.0:
                        tracker = create_csrt_tracker()
                        tracker.init(frame, bbox)
                        tracking = True
                        latest_observation = None
                        vision.reset()
                        controller.reset()
                        print("[TRACK] Started; arm again after checking the ROI")
                elif key == ord("a"):
                    armed = not armed
                    rearm_required = False
                    if not armed:
                        controller.reset()
                    command = zero_command(armed=armed)
                    connection.send(command)
                    print("[ARM]", "ON" if armed else "OFF")
                elif key == ord("m"):
                    auto_mode = not auto_mode
                    controller.reset()
                    print("[MODE]", "AUTO" if auto_mode else "ZERO-HOLD")

                if tracking:
                    visible, bbox = tracker.update(frame)
                    if visible:
                        latest_observation = vision.update_bbox(
                            bbox,
                            image_width,
                            image_height,
                            timestamp=now,
                        )
                        x, y, width, height = bbox
                        cv2.rectangle(
                            frame,
                            (int(x), int(y)),
                            (int(x + width), int(y + height)),
                            (0, 255, 0),
                            2,
                        )

            telemetry = connection.fresh_telemetry()
            if telemetry is not None:
                state = state_estimator.update(telemetry)
            else:
                state = None

            if telemetry is not None and telemetry.failsafe:
                armed = False
                rearm_required = True
                controller.reset()
                status = "DISARMED: LOWER FAILSAFE"
            elif (
                telemetry is not None
                and telemetry.last_command_rejected
                and telemetry.last_command_session == connection.session_id
            ):
                armed = False
                rearm_required = True
                controller.reset()
                status = f"DISARMED: REJECT 0x{telemetry.reject_flags:08x}"
            elif armed and telemetry is None:
                armed = False
                rearm_required = True
                controller.reset()
                status = "DISARMED: TELEMETRY STALE"
            elif (
                armed
                and auto_mode
                and telemetry is not None
                and not telemetry.execution_feedback_valid
            ):
                armed = False
                rearm_required = True
                controller.reset()
                status = "DISARMED: NO EXECUTION FEEDBACK"
            elif (
                armed
                and connection.last_confirmed_armed is True
                and not connection.confirmation_fresh()
            ):
                armed = False
                rearm_required = True
                controller.reset()
                status = "DISARMED: ACK STALE"

            if now - last_control_time >= period_s:
                dt = min(max(now - last_control_time, 1e-3), 0.20)
                last_control_time = now
                observation_fresh = vision.is_fresh(
                    latest_observation,
                    now=now,
                )

                if not armed:
                    command = zero_command(armed=False)
                    status = (
                        "DISARMED: RE-ARM REQUIRED"
                        if rearm_required
                        else "DISARMED"
                    )
                elif not connection.armed_confirmation_fresh():
                    command = zero_command(armed=True)
                    status = "ARMING: WAIT EXECUTION ACK"
                elif not auto_mode:
                    command = zero_command(armed=True)
                    status = "ARMED: ZERO-HOLD"
                elif telemetry is None or state is None:
                    armed = False
                    rearm_required = True
                    command = zero_command(armed=False)
                    controller.reset()
                    status = "DISARMED: TELEMETRY STALE"
                elif not observation_fresh or latest_observation is None:
                    command = zero_command(armed=True)
                    controller.reset()
                    latest_reference = None
                    status = "AUTO: TARGET LOST - ZERO"
                else:
                    latest_reference = vision.make_reference(
                        latest_observation,
                        current_depth_m=state.depth,
                        current_yaw_rad=state.yaw,
                    )
                    control = controller.compute(
                        state,
                        latest_reference.depth_reference_m,
                        latest_reference.yaw_reference_rad,
                        dt,
                    )
                    command = command_mapper.convert(
                        control,
                        forward_force_n=latest_reference.forward_force_n,
                        armed=True,
                    )
                    status = "AUTO: VISION P + DEPTH/YAW SMC"
                connection.send(command)

            if frame is not None:
                telemetry_text = "telemetry=STALE"
                if telemetry is not None:
                    telemetry_text = (
                        f"z={telemetry.depth_m:.2f}m "
                        f"yaw={np.degrees(telemetry.yaw_rad):.1f}deg "
                        f"r={np.degrees(telemetry.yaw_rate_rad_s):.1f}deg/s "
                        f"ack={telemetry.last_command_sequence}"
                    )
                command_text = (
                    f"cmd=({command.forward:+.2f},{command.right:+.2f},"
                    f"{command.down:+.2f},{command.yaw:+.2f})"
                )
                if latest_reference is not None:
                    command_text += (
                        f" rangeErr={latest_reference.range_error_m:+.2f}m"
                        f" bearing={np.degrees(latest_reference.bearing_error_rad):+.1f}deg"
                    )
                for row, (text_value, color) in enumerate(
                    (
                        (status, (0, 255, 255)),
                        (telemetry_text, (0, 255, 0) if telemetry else (0, 0, 255)),
                        (command_text, (0, 255, 0)),
                    )
                ):
                    cv2.putText(
                        frame,
                        text_value,
                        (20, 35 + 30 * row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        color,
                        2,
                    )
                cv2.imshow("Track", frame)
    finally:
        try:
            connection.send(zero_command(armed=False))
        except Exception:
            pass
        connection.close()
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    main(args.config)

