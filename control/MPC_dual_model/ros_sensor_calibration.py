"""Synchronized, disarmed FineSUB and overhead-camera calibration recorder.

This module never exposes an armed or non-zero command option.  By default it
keeps the FineSUB v4 session alive with disarmed zero frames while sampling the
latest raw MCU telemetry and the independent overhead-camera pose into one CSV
time base.  ``--passive-no-finesub`` records only ROS camera data and does not
open or write the FineSUB transport; it is intended to run beside the guarded
single-motor calibration tool.  Run it with the ROS 2 Python environment that
provides ``rclpy`` and ``geometry_msgs``; importing the module itself does not
require ROS.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from .calibration_tool import DEFAULT_CONFIG_PATH, _send_disarmed_burst
from .finesub_protocol import FineSUBControlCommand, FineSUBTelemetry
from .finesub_transport import (
    FineSUBConnection,
    load_runtime_config,
    make_transport,
)


DEFAULT_VISION_TOPIC = "/finsrov/vision/refracted_pose_6d"
DEFAULT_VISION_STATUS_TOPIC = "/finsrov/vision/status"
DEFAULT_REFRACTED_STATUS_TOPIC = "/finsrov/vision/refracted/status"


def _json_vector(values: Any) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class VisionSnapshot:
    received_monotonic: float = float("-inf")
    count: int = 0
    stamp_s: float = float("nan")
    frame_id: str = ""
    position_xyz: tuple[float, float, float] = (float("nan"),) * 3
    quat_xyzw: tuple[float, float, float, float] = (float("nan"),) * 4
    covariance_diag: tuple[float, ...] = (float("nan"),) * 6

    def update(self, message: Any, *, received_monotonic: float | None = None) -> None:
        pose = message.pose.pose
        covariance = list(message.pose.covariance)
        self.received_monotonic = (
            time.monotonic() if received_monotonic is None else float(received_monotonic)
        )
        self.count += 1
        self.stamp_s = _stamp_seconds(message.header.stamp)
        self.frame_id = str(message.header.frame_id)
        self.position_xyz = (
            _finite(pose.position.x),
            _finite(pose.position.y),
            _finite(pose.position.z),
        )
        self.quat_xyzw = (
            _finite(pose.orientation.x),
            _finite(pose.orientation.y),
            _finite(pose.orientation.z),
            _finite(pose.orientation.w),
        )
        self.covariance_diag = tuple(
            _finite(covariance[index]) for index in (0, 7, 14, 21, 28, 35)
        )


@dataclass
class JsonStatusSnapshot:
    received_monotonic: float = float("-inf")
    count: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def update(self, message: Any, *, received_monotonic: float | None = None) -> None:
        self.received_monotonic = (
            time.monotonic() if received_monotonic is None else float(received_monotonic)
        )
        self.count += 1
        try:
            parsed = json.loads(str(message.data))
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        self.data = parsed if isinstance(parsed, dict) else {}


DIRECT_FIELDS = (
    "sent",
    "transport_open",
    "session_confirmed",
    "telemetry_fresh",
    "telemetry_age_s",
    "telemetry_sequence",
    "mcu_tick_ms",
    "state",
    "telemetry_armed",
    "telemetry_failsafe",
    "command_status",
    "reject_flags",
    "last_command_session",
    "last_command_sequence",
    "last_command_crc",
    "command_count",
    "rejected_command_count",
    "round_trip_ms",
    "imu_quat_wxyz",
    "imu_angular_velocity_xyz_rad_s",
    "imu_linear_acceleration_xyz_m_s2",
    "yaw_rad",
    "yaw_rate_rad_s",
    "depth_m",
    "pressure_pa",
    "received_channels_forward_right_down_yaw",
    "applied_motor_throttle_m1_m8",
    "motor_rpm_m1_m8",
    "rpm_valid_mask",
    "execution_feedback_valid",
    "rpm_available",
)

VISION_FIELDS = (
    "vision_fresh",
    "vision_age_s",
    "vision_count",
    "vision_stamp_s",
    "vision_frame_id",
    "vision_position_xyz_m",
    "vision_quat_xyzw",
    "vision_covariance_diag",
    "vision_status_age_s",
    "vision_status_count",
    "vision_detected",
    "vision_detected_ids",
    "vision_world_xy_yaw",
    "vision_detections_detail",
    "vision_pnp_valid",
    "vision_pnp_reprojection_error_px",
    "vision_decision_margin",
    "vision_detection_fps",
    "refracted_status_age_s",
    "refracted_status_count",
    "refracted_mode",
    "refracted_valid",
    "refracted_residual_m",
    "refracted_reject_reason",
)

CSV_FIELDS = (
    "host_time_utc",
    "host_monotonic_s",
    "phase",
    "requested_armed",
    "requested_channels_forward_right_down_yaw",
    *DIRECT_FIELDS,
    *VISION_FIELDS,
)


def _age(received_monotonic: float, now: float) -> float:
    if not math.isfinite(received_monotonic):
        return float("nan")
    return max(0.0, now - received_monotonic)


def _telemetry_fields(
    telemetry: FineSUBTelemetry | None,
    connection: FineSUBConnection | None,
    *,
    sent: bool,
    now: float,
) -> dict[str, Any]:
    if connection is None:
        return {
            "sent": 0,
            "transport_open": 0,
            "session_confirmed": 0,
            "telemetry_fresh": 0,
            "telemetry_age_s": "",
            "round_trip_ms": "",
        }
    row: dict[str, Any] = {
        "sent": int(sent),
        "transport_open": int(bool(getattr(connection, "_opened", False))),
        "session_confirmed": int(connection.session_confirmed),
        "telemetry_fresh": int(telemetry is not None),
        "telemetry_age_s": "",
        "round_trip_ms": "" if connection.last_round_trip_ms is None else connection.last_round_trip_ms,
    }
    if telemetry is None:
        return row
    row.update(
        telemetry_age_s=f"{max(0.0, now - telemetry.received_monotonic):.6f}",
        telemetry_sequence=telemetry.sequence,
        mcu_tick_ms=telemetry.tick_ms,
        state=telemetry.state,
        telemetry_armed=int(telemetry.armed),
        telemetry_failsafe=int(telemetry.failsafe),
        command_status=telemetry.command_status,
        reject_flags=telemetry.reject_flags,
        last_command_session=telemetry.last_command_session,
        last_command_sequence=telemetry.last_command_sequence,
        last_command_crc=telemetry.last_command_crc,
        command_count=telemetry.command_count,
        rejected_command_count=telemetry.rejected_command_count,
        imu_quat_wxyz=_json_vector(telemetry.quat_wxyz),
        imu_angular_velocity_xyz_rad_s=_json_vector(telemetry.angular_velocity_xyz),
        imu_linear_acceleration_xyz_m_s2=_json_vector(telemetry.linear_acceleration_xyz),
        yaw_rad=f"{telemetry.yaw_rad:.9g}",
        yaw_rate_rad_s=f"{telemetry.yaw_rate_rad_s:.9g}",
        depth_m=f"{telemetry.depth_m:.9g}",
        pressure_pa=f"{telemetry.pressure_pa:.9g}",
        received_channels_forward_right_down_yaw=_json_vector(
            (telemetry.forward, telemetry.right, telemetry.down, telemetry.yaw)
        ),
        applied_motor_throttle_m1_m8=_json_vector(telemetry.applied_motor_throttle),
        motor_rpm_m1_m8=_json_vector(telemetry.motor_rpm),
        rpm_valid_mask=telemetry.rpm_valid_mask,
        execution_feedback_valid=int(telemetry.execution_feedback_valid),
        rpm_available=int(telemetry.rpm_available),
    )
    return row


def _vision_fields(
    vision: VisionSnapshot,
    status: JsonStatusSnapshot,
    refracted_status: JsonStatusSnapshot,
    *,
    now: float,
    max_age_s: float,
) -> dict[str, Any]:
    vision_age = _age(vision.received_monotonic, now)
    status_age = _age(status.received_monotonic, now)
    refracted_age = _age(refracted_status.received_monotonic, now)
    data = status.data
    refracted = refracted_status.data
    return {
        "vision_fresh": int(math.isfinite(vision_age) and vision_age <= max_age_s),
        "vision_age_s": "" if not math.isfinite(vision_age) else f"{vision_age:.6f}",
        "vision_count": vision.count,
        "vision_stamp_s": "" if not math.isfinite(vision.stamp_s) else f"{vision.stamp_s:.9f}",
        "vision_frame_id": vision.frame_id,
        "vision_position_xyz_m": _json_vector(vision.position_xyz),
        "vision_quat_xyzw": _json_vector(vision.quat_xyzw),
        "vision_covariance_diag": _json_vector(vision.covariance_diag),
        "vision_status_age_s": "" if not math.isfinite(status_age) else f"{status_age:.6f}",
        "vision_status_count": status.count,
        "vision_detected": int(bool(data.get("detected", False))),
        "vision_detected_ids": json.dumps(data.get("detected_ids", []), separators=(",", ":")),
        "vision_world_xy_yaw": json.dumps(data.get("world_xy_yaw", []), separators=(",", ":")),
        "vision_detections_detail": json.dumps(data.get("detections_detail", []), separators=(",", ":")),
        "vision_pnp_valid": int(bool(data.get("pnp_valid", False))),
        "vision_pnp_reprojection_error_px": data.get("pnp_reprojection_error_px", ""),
        "vision_decision_margin": data.get("decision_margin", ""),
        "vision_detection_fps": data.get("detection_fps", ""),
        "refracted_status_age_s": "" if not math.isfinite(refracted_age) else f"{refracted_age:.6f}",
        "refracted_status_count": refracted_status.count,
        "refracted_mode": refracted.get("mode", ""),
        "refracted_valid": int(bool(refracted.get("constrained_valid", False))),
        "refracted_residual_m": refracted.get("constrained_residual_m", ""),
        "refracted_reject_reason": refracted.get("constrained_reject_reason", ""),
    }


def _connection(config_path: Path) -> FineSUBConnection:
    config = load_runtime_config(config_path)
    control = config.get("control", {})
    transport = config.get("transport", {})
    return FineSUBConnection(
        make_transport(transport),
        telemetry_max_age_s=float(control.get("telemetry_max_age_sec", 0.25)),
        confirmation_max_age_s=float(control.get("confirmation_max_age_sec", 0.25)),
        reconnect_interval_s=float(transport.get("reconnect_interval_sec", 1.0)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python is required; run through FinsSim ros2_ws/scripts/run_ros2_uv.sh"
        ) from error

    if args.duration <= 0.0 or args.rate <= 0.0 or args.vision_max_age <= 0.0:
        raise ValueError("duration, rate, and vision-max-age must be positive")
    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("x", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    handle.flush()

    vision = VisionSnapshot()
    vision_status = JsonStatusSnapshot()
    refracted_status = JsonStatusSnapshot()
    connection = None if args.passive_no_finesub else _connection(Path(args.config))
    zero = FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, armed=False, yaw_direct=True)
    rows = 0
    telemetry_sequences: set[int] = set()

    rclpy.init(args=None)
    node = rclpy.create_node("mpc_disarmed_sensor_calibration_recorder")
    node.create_subscription(
        PoseWithCovarianceStamped,
        args.vision_topic,
        lambda message: vision.update(message),
        20,
    )
    node.create_subscription(
        String,
        args.vision_status_topic,
        lambda message: vision_status.update(message),
        20,
    )
    node.create_subscription(
        String,
        args.refracted_status_topic,
        lambda message: refracted_status.update(message),
        20,
    )

    start = time.monotonic()
    period = 1.0 / args.rate
    next_tick = start
    try:
        while time.monotonic() - start < args.duration:
            # Several camera/status topics can become ready between two CSV
            # rows. Drain the already-queued callbacks so the high-rate pose
            # topic is not starved by lower-rate JSON status callbacks.
            for _ in range(8):
                rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            if connection is None:
                telemetry = None
                sent = False
            else:
                connection.poll_telemetry()
                telemetry = connection.fresh_telemetry()
                sent = connection.send(zero)
            if telemetry is not None:
                telemetry_sequences.add(int(telemetry.sequence))
            row = {field: "" for field in CSV_FIELDS}
            row.update(
                host_time_utc=datetime.now(timezone.utc).isoformat(),
                host_monotonic_s=f"{now:.9f}",
                phase=args.phase,
                requested_armed=0,
                requested_channels_forward_right_down_yaw="[0.0,0.0,0.0,0.0]",
            )
            row.update(_telemetry_fields(telemetry, connection, sent=sent, now=now))
            row.update(
                _vision_fields(
                    vision,
                    vision_status,
                    refracted_status,
                    now=now,
                    max_age_s=args.vision_max_age,
                )
            )
            writer.writerow(row)
            handle.flush()
            rows += 1
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        if connection is not None:
            _send_disarmed_burst(connection, period)
            connection.close()
        node.destroy_node()
        rclpy.shutdown()
        handle.flush()
        handle.close()

    summary = {
        "csv": str(output),
        "duration_s": time.monotonic() - start,
        "rows": rows,
        "telemetry_unique_sequences": len(telemetry_sequences),
        "vision_messages": vision.count,
        "vision_status_messages": vision_status.count,
        "refracted_status_messages": refracted_status.count,
        "nonzero_or_armed_commands_supported": False,
        "passive_no_finesub": bool(args.passive_no_finesub),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record synchronized overhead-camera pose and disarmed FineSUB telemetry"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--phase", default="static_disarmed")
    parser.add_argument("--vision-topic", default=DEFAULT_VISION_TOPIC)
    parser.add_argument("--vision-status-topic", default=DEFAULT_VISION_STATUS_TOPIC)
    parser.add_argument("--refracted-status-topic", default=DEFAULT_REFRACTED_STATUS_TOPIC)
    parser.add_argument("--vision-max-age", type=float, default=0.25)
    parser.add_argument(
        "--passive-no-finesub",
        action="store_true",
        help="record ROS camera/status only; do not open or write the FineSUB transport",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
