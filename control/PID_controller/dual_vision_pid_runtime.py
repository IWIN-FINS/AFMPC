"""双路视觉的纯 PID 实时窗口。

控制输入只有双目视觉 JSONL 中的目标三维坐标 ``[right, down, forward]``。
池顶相机和其 ROS 位姿只用于显示目标位置、估计移动速度以及辅助标记跳变，
绝不会被转换成 PID 的位置输入。这样可以避免把池顶单目像素坐标和人工
``range_m`` 误当成双目深度控制量。

本入口默认 dry-run/disarmed。硬件使能必须显式提供 ``--enable-arm``；
正常情况需在窗口按 ``a``，或额外指定 ``--arm-on-start`` 在本次启动首次
获得连续有效三维测量后请求使能。曾实际使能后失去目标不会自动再使能。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import deque
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

# Keep the PID GUI/control process deterministic.  The stereo producer owns
# the expensive vision compute; this process must not create a large BLAS/TBB
# pool that starves the 20 Hz control loop and OpenCV windows.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np

try:
    from .camera_transform import camera_to_pid_body_position
    from .camera_pid_tuner import ErrorHistory, PIDTuningPanel, render_error_window
    from .hardware_session import (
        PIDHardwareSession,
        build_runtime_hardware_session,
        build_serial_hardware_session,
    )
    from .live_integration_example import build_tracker
    from .pid_tracker import PIDTracker, PIDTrackerOutput
    from .vision_gate import PIDVisionGate, VisionGateConfig
except ImportError:  # direct ``python dual_vision_pid_runtime.py``
    from camera_transform import camera_to_pid_body_position
    from camera_pid_tuner import ErrorHistory, PIDTuningPanel, render_error_window
    from hardware_session import (
        PIDHardwareSession,
        build_runtime_hardware_session,
        build_serial_hardware_session,
    )
    from live_integration_example import build_tracker
    from pid_tracker import PIDTracker, PIDTrackerOutput
    from vision_gate import PIDVisionGate, VisionGateConfig


class DualVisionRuntimeError(RuntimeError):
    """Configuration or live-input error shown to the operator."""


def render_3d_error_norm_window(cv2: Any, samples: Iterable[float], *, width: int = 900, height: int = 520) -> np.ndarray:
    """Draw one Euclidean three-axis position-error curve, never axis traces."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    values = np.asarray(list(samples), dtype=float)
    cv2.putText(canvas, "PID 3D error norm (cm)", (48, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    left, right, top, bottom = 58, width - 20, 58, height - 45
    cv2.rectangle(canvas, (left, top), (right, bottom), (85, 85, 85), 1)
    if values.size:
        ceiling = max(10.0, float(np.max(values)) * 1.2)
        cv2.putText(canvas, f"latest {values[-1]:.1f} cm", (left + 12, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 150, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{ceiling:.0f}", (7, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        xs = np.linspace(left, right, len(values)).astype(int)
        ys = np.clip(bottom - values / ceiling * (bottom - top), top, bottom).astype(int)
        points = np.column_stack((xs, ys)).reshape(-1, 1, 2)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, (220, 150, 220), 2, cv2.LINE_AA)
    return canvas


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class StereoSample:
    frame_index: int
    acquisition_time_s: float
    result_time_s: float
    position_camera_xyz_m: np.ndarray
    confidence: float
    depth_confidence: float
    depth_nis: float
    depth_filter_mode: str = "update"


def is_new_stereo_sample(sample: StereoSample, previous_acquisition_time_s: float | None) -> bool:
    """Accept a camera frame once, even if the producer republishes its result."""
    return previous_acquisition_time_s is None or sample.acquisition_time_s > previous_acquisition_time_s + 1e-6


def retain_initial_arm_request(
    arm_requested: bool,
    pending_arm: bool,
    *,
    has_armed: bool,
    safety_latched: bool,
) -> bool:
    """Keep an initial operator request through empty vision, but not a fault.

    Once an arm command has been transmitted, later vision loss must never
    silently request another arm. A latched hardware stop also requires a new
    explicit operator action.
    """
    return bool((arm_requested or pending_arm) and not has_armed and not safety_latched)


class StereoJsonlTail:
    """Read-only non-blocking tail of the existing stereo pipeline output."""

    def __init__(self, path: str | Path, *, start_at_end: bool = True) -> None:
        self.path = Path(path)
        self.start_at_end = bool(start_at_end)
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._initialized = False

    def poll(self) -> list[dict[str, Any]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        identity = (int(stat.st_dev), int(stat.st_ino))
        replaced = self._identity is not None and identity != self._identity
        truncated = int(stat.st_size) < self._offset
        if not self._initialized or replaced or truncated:
            self._identity = identity
            self._offset = int(stat.st_size) if self.start_at_end and not self._initialized else 0
            self._initialized = True
        if int(stat.st_size) <= self._offset:
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(line_start)
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
            self._offset = handle.tell()
        return records


def _parse_stereo_record(record: dict[str, Any]) -> tuple[StereoSample | None, str]:
    try:
        frame_index = int(record["frame_idx"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_frame_index"
    acquisition = _finite(record.get("frame_ts_mean"))
    result_time = _finite(record.get("result_time"))
    if acquisition is None or result_time is None:
        return None, "missing_timestamp"
    tracks = [item for item in record.get("tracks", []) if isinstance(item, dict)]
    if not tracks:
        return None, "no_target"
    tracks.sort(key=lambda item: _finite(item.get("confidence")) or -1.0, reverse=True)
    track = tracks[0]
    try:
        position = np.asarray(track.get("position", track.get("position_xyz")), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None, "invalid_position"
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return None, "invalid_position"
    if track.get("depth_valid") is not True:
        return None, "invalid_depth"
    depth_filter_mode = str(track.get("depth_filter_mode", "")).strip().lower()
    if depth_filter_mode not in {"update", "weak_update"}:
        return None, "depth_not_updated"
    confidence = _finite(track.get("confidence"))
    depth_confidence = _finite(track.get("depth_confidence"))
    depth_nis = _finite(track.get("depth_nis"))
    if confidence is None or depth_confidence is None or depth_nis is None:
        return None, "missing_quality"
    if result_time < acquisition:
        return None, "negative_pipeline_delay"
    return (
        StereoSample(
            frame_index=frame_index,
            acquisition_time_s=acquisition,
            result_time_s=result_time,
            position_camera_xyz_m=position.copy(),
            confidence=confidence,
            depth_confidence=depth_confidence,
            depth_nis=depth_nis,
            depth_filter_mode=depth_filter_mode,
        ),
        "parsed",
    )


@dataclass(frozen=True)
class StereoQualityConfig:
    max_result_age_s: float = 0.25
    max_pipeline_delay_s: float = 0.15
    min_confidence: float = 0.0
    min_depth_confidence: float = 0.20
    max_depth_nis: float = 25.0
    min_forward_m: float = 0.30
    max_forward_m: float = 1.40


def validate_stereo_sample(
    sample: StereoSample,
    config: StereoQualityConfig,
    *,
    now_s: float,
) -> str | None:
    """Return a rejection reason, or ``None`` for a usable stereo sample."""

    age = float(now_s) - sample.result_time_s
    if not math.isfinite(age) or age < -0.50 or age > config.max_result_age_s:
        return "stereo_stale"
    if sample.result_time_s - sample.acquisition_time_s > config.max_pipeline_delay_s:
        return "stereo_pipeline_delay"
    if sample.confidence < config.min_confidence:
        return "stereo_low_confidence"
    if sample.depth_confidence < config.min_depth_confidence:
        return "stereo_low_depth_confidence"
    if sample.depth_nis > config.max_depth_nis:
        return "stereo_depth_nis_outlier"
    if not config.min_forward_m <= float(sample.position_camera_xyz_m[2]) <= config.max_forward_m:
        return "stereo_forward_range"
    return None


@dataclass
class OverheadMotionSnapshot:
    position_xyz: np.ndarray | None = None
    velocity_xyz: np.ndarray | None = None
    speed_m_s: float | None = None
    stamp_s: float | None = None
    received_monotonic: float = float("-inf")
    jump: bool = False
    frame_id: str = ""


class OverheadMotionMonitor:
    """Pool-top position/velocity monitor; never produces PID coordinates."""

    def __init__(self, *, max_position_jump_m: float = 0.50) -> None:
        self.max_position_jump_m = float(max_position_jump_m)
        self.snapshot = OverheadMotionSnapshot()
        self._previous_position: np.ndarray | None = None
        self._previous_stamp: float | None = None

    def update(
        self,
        position_xyz: object,
        *,
        stamp_s: float | None = None,
        frame_id: str = "",
        received_monotonic: float | None = None,
    ) -> OverheadMotionSnapshot:
        position = np.asarray(position_xyz, dtype=float).reshape(-1)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return self.snapshot
        now = time.monotonic() if received_monotonic is None else float(received_monotonic)
        stamp = now if stamp_s is None or not math.isfinite(float(stamp_s)) else float(stamp_s)
        velocity = None
        jump = False
        if self._previous_position is not None and self._previous_stamp is not None:
            dt = stamp - self._previous_stamp
            if math.isfinite(dt) and dt > 1.0e-6:
                delta = position - self._previous_position
                velocity = delta / dt
                jump = float(np.linalg.norm(delta)) > self.max_position_jump_m
        self._previous_position = position.copy()
        self._previous_stamp = stamp
        self.snapshot = OverheadMotionSnapshot(
            position_xyz=position.copy(),
            velocity_xyz=None if velocity is None else velocity.copy(),
            speed_m_s=None if velocity is None else float(np.linalg.norm(velocity)),
            stamp_s=stamp,
            received_monotonic=now,
            jump=jump,
            frame_id=str(frame_id),
        )
        return self.snapshot


class ROSOverheadSource:
    """Optional ROS 2 subscriber for the pool-top refracted pose topic."""

    def __init__(self, topic: str, monitor: OverheadMotionMonitor) -> None:
        self.topic = str(topic)
        self.monitor = monitor
        self.node: Any | None = None
        self._rclpy: Any | None = None
        self.error: str | None = None
        if not self.topic:
            return
        try:
            import rclpy
            from geometry_msgs.msg import PoseWithCovarianceStamped
        except ImportError:
            self.error = "ROS2 unavailable; overhead display only"
            return
        try:
            rclpy.init(args=None)
            node = rclpy.create_node("pid_overhead_motion_monitor")
            node.create_subscription(
                PoseWithCovarianceStamped,
                self.topic,
                self._callback,
                20,
            )
        except Exception as exc:  # pragma: no cover - host ROS setup dependent.
            self.error = f"ROS2 overhead unavailable: {exc}"
            try:
                rclpy.shutdown()
            except Exception:
                pass
            return
        self._rclpy = rclpy
        self.node = node

    def _callback(self, message: Any) -> None:
        pose = message.pose.pose
        stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1.0e-9
        self.monitor.update(
            (pose.position.x, pose.position.y, pose.position.z),
            stamp_s=stamp,
            frame_id=str(message.header.frame_id),
        )

    def poll(self) -> None:
        if self.node is not None and self._rclpy is not None:
            for _ in range(4):
                self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
        self.node = None
        self._rclpy = None


def _put_text(cv2: Any, image: np.ndarray, text: str, origin: tuple[int, int], color=(230, 230, 230), scale: float = 0.50) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_dual_vision_window(
    cv2: Any,
    frame: np.ndarray,
    *,
    stereo_position: np.ndarray | None,
    stereo_sample: StereoSample | None,
    overhead: OverheadMotionSnapshot,
    error: np.ndarray | None,
    force: np.ndarray | None,
    status: str,
    armed: bool,
) -> np.ndarray:
    """Show the pool-top image as an auxiliary diagnostic, not a controller input."""

    image = np.asarray(frame).copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    state_color = (0, 180, 0) if armed else (0, 180, 255)
    line = 23
    _put_text(cv2, image, f"PID stereo control | {'ARMED' if armed else 'DISARMED'}", (12, line), state_color)
    _put_text(cv2, image, f"status: {status}", (12, line * 2))
    if stereo_position is not None:
        _put_text(cv2, image, "STEREO control [R,D,F] = " + np.array2string(stereo_position, precision=3), (12, line * 3))
    if stereo_sample is not None:
        _put_text(cv2, image, f"stereo conf={stereo_sample.confidence:.2f} depth={stereo_sample.depth_confidence:.2f} frame={stereo_sample.frame_index}", (12, line * 4))
    if overhead.position_xyz is not None:
        _put_text(cv2, image, "TOP auxiliary position = " + np.array2string(overhead.position_xyz, precision=3), (12, line * 5), (170, 220, 255))
        speed = "n/a" if overhead.speed_m_s is None else f"{overhead.speed_m_s:.3f} m/s"
        color = (0, 80, 255) if overhead.jump else (170, 220, 255)
        _put_text(cv2, image, f"TOP auxiliary speed = {speed}" + ("  JUMP" if overhead.jump else ""), (12, line * 6), color)
    else:
        _put_text(cv2, image, "TOP auxiliary pose: waiting (not PID input)", (12, line * 5), (170, 220, 255))
    if error is not None:
        _put_text(cv2, image, "PID error [F,R,D,Y] = " + np.array2string(error, precision=3), (12, line * 7))
    if force is not None:
        _put_text(cv2, image, "PID force [F,R,D] = " + np.array2string(force, precision=2), (12, line * 8))
    _put_text(cv2, image, "a arm (requires --enable-arm) | d/space disarm | r reset | q/ESC quit", (12, image.shape[0] - 14), (220, 220, 220), 0.45)
    return image


def _safe_output(output: PIDTrackerOutput | None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    if output is None:
        return None, None, 0.0
    yaw = 0.0 if output.yaw_pid is None else float(output.yaw_pid.yaw_moment)
    return output.pid.error.copy(), output.pid.force.copy(), yaw


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional GUI dependency.
        raise DualVisionRuntimeError("需要 OpenCV GUI；请先运行 `uv sync --extra gui`") from exc
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return cv2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-jsonl", required=True, help="现有双目 pipeline_results.jsonl，只读尾部")
    parser.add_argument("--camera-device", default="/dev/video2", help="池顶相机，仅用于辅助显示")
    parser.add_argument("--no-camera", action="store_true", help="不打开池顶视频，只显示控制窗口")
    parser.add_argument("--overhead-topic", default="/finsrov/vision/refracted_pose_6d", help="池顶 ROS 位姿 topic")
    parser.add_argument("--history", type=int, default=240)
    parser.add_argument("--trace-jsonl", help="逐拍 PID/遥测 trace 输出文件")
    parser.add_argument("--fixed-parameters", action="store_true", help="锁定代码中的 PID 启动参数，不创建可调滑块")
    parser.add_argument("--port", help="可选串口；不传则 dry-run")
    parser.add_argument("--runtime-config", help="可选 MPC 风格 transport JSON")
    parser.add_argument("--dry-run", action="store_true", help="显式禁止硬件连接")
    parser.add_argument("--enable-arm", action="store_true", help="允许按 a 请求 armed；默认不允许")
    parser.add_argument(
        "--arm-on-start",
        action="store_true",
        help="在握手和新鲜双目确认后自动请求 armed；仅用于已明确授权的实机测试",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-depth-confidence", type=float, default=0.20)
    parser.add_argument("--max-depth-nis", type=float, default=25.0)
    parser.add_argument("--max-result-age", type=float, default=0.25)
    parser.add_argument("--pid-window-x", type=int, default=None)
    parser.add_argument("--pid-window-y", type=int, default=None)
    parser.add_argument("--plot-window-x", type=int, default=None)
    parser.add_argument("--plot-window-y", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.port and args.runtime_config:
        raise DualVisionRuntimeError("--port 与 --runtime-config 不能同时使用")
    if (args.port or args.runtime_config) and args.dry_run:
        raise DualVisionRuntimeError("硬件连接参数不能与 --dry-run 同时使用")
    if args.arm_on_start and not args.enable_arm:
        raise DualVisionRuntimeError("--arm-on-start 必须同时指定 --enable-arm")
    if args.history <= 1 or args.max_frames < 0:
        raise DualVisionRuntimeError("history 必须大于 1，max-frames 不能为负数")
    cv2 = _load_cv2()
    capture = None
    if not args.no_camera:
        capture = cv2.VideoCapture(args.camera_device)
        if not capture.isOpened():
            capture.release()
            raise DualVisionRuntimeError(f"无法打开池顶相机 {args.camera_device}")
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise DualVisionRuntimeError("池顶相机已打开但读不到首帧")
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tail = StereoJsonlTail(args.stereo_jsonl, start_at_end=True)
    quality = StereoQualityConfig(
        max_result_age_s=float(args.max_result_age),
        min_confidence=float(args.min_confidence),
        min_depth_confidence=float(args.min_depth_confidence),
        max_depth_nis=float(args.max_depth_nis),
    )
    gate = PIDVisionGate(
        VisionGateConfig(
            min_forward_m=quality.min_forward_m,
            max_forward_m=quality.max_forward_m,
            jump_margin_m=0.20,
            max_inter_sample_gap_s=1.00,
            startup_confirmation_samples=1,
            reacquire_confirmation_samples=1,
        )
    )
    tracker = build_tracker(calibrated_reference=True)
    # Yaw is enabled and holds the current angle latched from the first valid
    # IMU sample.  The panel leaves the yaw reference unset so the tracker
    # does not replace that hold angle with a target-bearing command.
    panel = PIDTuningPanel(tracker, cv2, fixed_yaw=False)
    if not args.fixed_parameters:
        panel.create(range_m=1.0)
    cv2.namedWindow("PID stereo + top", cv2.WINDOW_NORMAL)
    cv2.namedWindow("PID error", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PID error", 900, 520)
    history = ErrorHistory(args.history)
    norm_history: deque[float] = deque(maxlen=args.history)
    trace_handle = None
    if args.trace_jsonl:
        trace_handle = Path(args.trace_jsonl).open("x", encoding="utf-8", buffering=1)

    def trace(event: str, **fields: Any) -> None:
        if trace_handle is not None:
            trace_handle.write(json.dumps({"event": event, "host_time_s": time.time(), "host_monotonic_s": time.monotonic(), **fields}, allow_nan=False) + "\n")

    trace("start", controller="PID", fixed_parameters=bool(args.fixed_parameters),
          kp=tracker.controller.config.kp.tolist(), ki=tracker.controller.config.ki.tolist(),
          kd=tracker.controller.config.kd.tolist(),
          reference_position_body_frd_m=tracker.controller.config.reference_position.tolist(),
          force_min_n=tracker.controller.config.force_min.tolist(),
          force_max_n=tracker.controller.config.force_max.tolist(),
          delta_force_min_n=tracker.controller.config.delta_force_min.tolist(),
          delta_force_max_n=tracker.controller.config.delta_force_max.tolist())
    overhead_monitor = OverheadMotionMonitor()
    overhead_source = ROSOverheadSource(args.overhead_topic, overhead_monitor)
    if overhead_source.error:
        print(f"[overhead] {overhead_source.error}")
    hardware: PIDHardwareSession | None = None
    arm_requested = bool(
        args.arm_on_start
        and args.enable_arm
        and (args.port or args.runtime_config)
    )
    auto_arm_reacquire = bool(arm_requested)
    has_armed = False
    latest_position: np.ndarray | None = None
    latest_sample: StereoSample | None = None
    latest_seen_monotonic: float | None = None
    last_outer_acquisition_time_s: float | None = None
    last_hardware_acquisition_time_s: float | None = None
    last_force = np.zeros(3, dtype=float)
    last_yaw_moment = 0.0
    initialized = False
    status = "waiting: stereo 3-D input (top camera is auxiliary)"
    if args.runtime_config:
        hardware = build_runtime_hardware_session(args.runtime_config, tracker=tracker, logger=print)
        if not hardware.connect():
            raise DualVisionRuntimeError("hardware session disarm 握手失败")
        status = "hardware connected disarmed; waiting stereo"
    elif args.port:
        hardware = build_serial_hardware_session(args.port, logger=print)
        hardware.tracker = tracker
        if not hardware.connect():
            raise DualVisionRuntimeError("serial session disarm 握手失败")
        status = "hardware connected disarmed; waiting stereo"

    try:
        frame_count = 0
        control_period_s = 0.05  # match the V5 lower-controller/PID dt
        display_period_s = 0.10  # GUI/camera refresh; control remains 20 Hz
        next_cycle = time.monotonic()
        next_display = next_cycle
        windows_positioned = False
        while True:
            now_mono = time.monotonic()
            if now_mono < next_cycle:
                time.sleep(min(next_cycle - now_mono, 0.01))
                continue
            next_cycle = max(next_cycle + control_period_s, now_mono)
            now_wall = time.time()
            overhead_source.poll()
            for record in tail.poll():
                sample, parse_reason = _parse_stereo_record(record)
                if sample is None:
                    status = parse_reason
                    continue
                if not is_new_stereo_sample(sample, last_outer_acquisition_time_s):
                    status = "stereo_duplicate_frame"
                    continue
                last_outer_acquisition_time_s = sample.acquisition_time_s
                reason = validate_stereo_sample(sample, quality, now_s=now_wall)
                if reason is not None:
                    status = reason
                    continue
                # Several completed producer records may arrive in one 20 Hz
                # poll. Preserve their acquisition spacing for the motion
                # gate instead of giving all of them the same loop timestamp.
                sample_monotonic = now_mono - max(0.0, now_wall - sample.acquisition_time_s)
                decision = gate.update(sample.position_camera_xyz_m, now=sample_monotonic)
                status = f"stereo:{decision.reason}"
                # An ignored jump is not a new position measurement. Keep the
                # previous accepted frame's age so repeated outliers cannot
                # indefinitely refresh the control input.
                if (decision.ready and decision.position_camera_xyz is not None
                        and decision.reason != "vision_jump_ignored"):
                    latest_position = decision.position_camera_xyz.copy()
                    latest_sample = sample
                    latest_seen_monotonic = now_mono

            output: PIDTrackerOutput | None = None
            error: np.ndarray | None = None
            force: np.ndarray | None = None
            fresh_hardware_sample = False
            if latest_position is None or latest_seen_monotonic is None or now_mono - latest_seen_monotonic > quality.max_result_age_s:
                status = "stereo stale/lost"
                if hardware is not None:
                    if has_armed and arm_requested and not hardware.safety_latched:
                        # Match the MPC experiment's vision-gap policy: stop
                        # using stale positions, slew toward the latched
                        # baseline, and remain armed only while lower telemetry
                        # positively confirms the armed session.
                        holding = hardware.target_lost(keep_armed=True)
                        status = (
                            "vision_gap:armed_baseline"
                            if holding else "vision_gap:disarmed_hardware_fault"
                        )
                        if not holding:
                            arm_requested = False
                    else:
                        pending_initial_arm = retain_initial_arm_request(
                            arm_requested,
                            auto_arm_reacquire,
                            has_armed=has_armed,
                            safety_latched=hardware.safety_latched,
                        )
                        hardware.target_lost()
                        arm_requested = False
                        # An explicit initial request survives missing vision,
                        # but not an armed session or a newly latched fault.
                        auto_arm_reacquire = bool(
                            args.enable_arm and pending_initial_arm and not hardware.safety_latched
                        )
                        status = "stereo stale/lost: disarmed"
                initialized = False if hardware is not None else initialized
            elif hardware is not None:
                reference, _, fixed_yaw = panel.read(1.0)
                fresh_hardware_sample = (
                    latest_sample is not None
                    and is_new_stereo_sample(latest_sample, last_hardware_acquisition_time_s)
                )
                if fresh_hardware_sample and latest_sample is not None:
                    last_hardware_acquisition_time_s = latest_sample.acquisition_time_s
                # During initial startup, clear a disarmed-session latch only
                # after distinct fresh camera frames. Never auto-rearm after
                # a previously armed session loses vision.
                if auto_arm_reacquire and not arm_requested:
                    result = hardware.step(
                        latest_position,
                        arm_requested=False,
                        reference_position=reference,
                        reference_yaw_rad=fixed_yaw,
                        fresh_vision_sample=fresh_hardware_sample,
                    )
                    if result.status == "disarmed:vision_ready" and not hardware.safety_latched:
                        arm_requested = True
                        auto_arm_reacquire = False
                else:
                    result = hardware.step(
                        latest_position,
                        arm_requested=arm_requested,
                        reference_position=reference,
                        reference_yaw_rad=fixed_yaw,
                        fresh_vision_sample=fresh_hardware_sample,
                    )
                output = result.controller_output
                # A transmitted arm may take a cycle to appear in telemetry.
                # Treat it as an armed attempt immediately: after any later
                # vision loss, only a new explicit operator request may arm.
                if result.transmitted_arm:
                    has_armed = True
                    auto_arm_reacquire = False
                yaw_mode = "direct_pid" if hardware.yaw_direct else "lower_hold"
                status = result.status + f" | yaw={yaw_mode} | top={'JUMP' if overhead_monitor.snapshot.jump else 'ok'}"
                if hardware.safety_latched:
                    arm_requested = False
                    auto_arm_reacquire = False
                if output is not None:
                    last_force = output.pid.force.copy()
                    last_yaw_moment = 0.0 if output.yaw_pid is None else float(output.yaw_pid.yaw_moment)
            else:
                reference, _, fixed_yaw = panel.read(1.0)
                if not initialized:
                    tracker.latch_baseline(last_force, last_yaw_moment, 0.0)
                    initialized = True
                output = tracker.update(
                    camera_to_pid_body_position(latest_position),
                    last_force,
                    reference_position=reference,
                    yaw_rad=0.0,
                    achieved_yaw_moment_previous=last_yaw_moment,
                    reference_yaw_rad=fixed_yaw,
                )
                last_force = output.pid.force.copy()
                last_yaw_moment = 0.0 if output.yaw_pid is None else float(output.yaw_pid.yaw_moment)
                status = "dry-run: stereo PID active" + f" | top={'JUMP' if overhead_monitor.snapshot.jump else 'ok'}"
            error, force, yaw_moment = _safe_output(output)
            if error is not None:
                history.append(now_mono, np.r_[error, 0.0], np.r_[force, yaw_moment])
                norm_history.append(100.0 * float(np.linalg.norm(error)))
            actual_armed = bool(
                hardware is not None
                and arm_requested
                and hardware.connection.latest_telemetry is not None
                and hardware.connection.latest_telemetry.armed
                and hardware.connection.armed_confirmation_fresh()
            )
            if actual_armed:
                has_armed = True
                auto_arm_reacquire = False
            telemetry = None if hardware is None else hardware.connection.latest_telemetry
            trace("control_cycle", status=status, armed=actual_armed,
                  frame_index=None if latest_sample is None else latest_sample.frame_index,
                  vision_acquisition_time_s=None if latest_sample is None else latest_sample.acquisition_time_s,
                  vision_depth_filter_mode=None if latest_sample is None else latest_sample.depth_filter_mode,
                  fresh_hardware_vision_sample=fresh_hardware_sample,
                  position_camera_xyz_m=None if latest_position is None else latest_position.tolist(),
                  position_error_frd_m=None if error is None else error.tolist(),
                  error_norm_m=None if error is None else float(np.linalg.norm(error)),
                  requested_force_frd_n=None if force is None else force.tolist(),
                  kp=tracker.controller.config.kp.tolist(), ki=tracker.controller.config.ki.tolist(),
                  kd=tracker.controller.config.kd.tolist(),
                  reference_position_body_frd_m=tracker.controller.config.reference_position.tolist(),
                  telemetry_armed=None if telemetry is None else telemetry.armed,
                  telemetry_failsafe=None if telemetry is None else telemetry.failsafe,
                  telemetry_execution_feedback_valid=None if telemetry is None else telemetry.execution_feedback_valid,
                  applied_motor_throttle=None if telemetry is None else list(telemetry.applied_motor_throttle),
                  motor_rpm=None if telemetry is None else list(telemetry.motor_rpm))
            if now_mono >= next_display:
                next_display = max(next_display + display_period_s, now_mono)
                if capture is not None:
                    ok, next_frame = capture.read()
                    if ok and next_frame is not None:
                        frame = next_frame
                cv2.imshow(
                    "PID stereo + top",
                    render_dual_vision_window(
                        cv2,
                        frame,
                        stereo_position=latest_position,
                        stereo_sample=latest_sample,
                        overhead=overhead_monitor.snapshot,
                        error=None if error is None else np.r_[error, 0.0],
                        force=force,
                        status=status,
                        armed=actual_armed,
                    ),
                )
                cv2.imshow("PID error", render_3d_error_norm_window(cv2, norm_history))
                if not windows_positioned:
                    if args.pid_window_x is not None and args.pid_window_y is not None:
                        cv2.moveWindow("PID stereo + top", args.pid_window_x, args.pid_window_y)
                    if args.plot_window_x is not None and args.plot_window_y is not None:
                        cv2.moveWindow("PID error", args.plot_window_x, args.plot_window_y)
                    windows_positioned = True
            key = int(cv2.waitKey(1) & 0xFF)
            if key in (27, ord("q")):
                break
            if key in (ord("d"), ord(" ")):
                arm_requested = False
                auto_arm_reacquire = False
                if hardware is not None:
                    hardware.connection.send_disarm()
                status = "disarm requested"
            elif key == ord("a"):
                if hardware is not None and args.enable_arm:
                    arm_requested = True
                    auto_arm_reacquire = not has_armed
                    status = "arm requested; waiting telemetry confirmation"
                else:
                    status = "arm blocked: requires hardware and --enable-arm"
            elif key == ord("r"):
                tracker.controller.reset(keep_baseline=True)
                history.clear()
                initialized = False
                last_force.fill(0.0)
                last_yaw_moment = 0.0
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        arm_requested = False
        if hardware is not None:
            hardware.close()
        trace("stop", reason="operator_or_runtime_exit")
        if trace_handle is not None:
            trace_handle.close()
        overhead_source.close()
        if capture is not None:
            capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except (DualVisionRuntimeError, ValueError) as error:
        print(f"dual_vision_pid_runtime: {error}")
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
