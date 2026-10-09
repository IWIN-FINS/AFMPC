"""池顶相机的纯 PID 实时查看、调参和（可选）硬件接入。

这个文件刻意保持在 ``PID_controller`` 目录内，不调用 MPC、Fossen 模型、
滤波器或其它控制器。它只负责把一个相机测量送入已有的
``PIDTracker``，并把 PID 状态画出来。

默认是 dry-run：不会打开串口，也不会向设备发任何数据。只有显式传入
``--port`` 或 ``--runtime-config`` 才会使用 :mod:`hardware_session`；硬件
进程启动时先 disarm，``a`` 键也只有在同时传入 ``--enable-arm`` 后才会请求解锁。

单目池顶相机没有绝对深度，本工具因此要求一个固定的 ``range_m``（可由
滑块实时调整）。像素坐标按 OpenCV 的 ``[right, down, forward]`` 约定，
再交给已有的 ``camera_transform`` 转成机体 FRD。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Package import.
    from .camera_transform import camera_to_pid_body_position
    from .hardware_session import (
        PIDHardwareSession,
        build_runtime_hardware_session,
        build_serial_hardware_session,
    )
    from .live_integration_example import build_tracker
    from .pid_tracker import PIDTracker, PIDTrackerOutput, SafeControlOutput
except ImportError:  # Direct ``python camera_pid_tuner.py`` execution.
    from camera_transform import camera_to_pid_body_position
    from hardware_session import (
        PIDHardwareSession,
        build_runtime_hardware_session,
        build_serial_hardware_session,
    )
    from live_integration_example import build_tracker
    from pid_tracker import PIDTracker, PIDTrackerOutput, SafeControlOutput


class CameraTunerError(RuntimeError):
    """A user-facing camera/tuner setup error."""


def _load_cv2() -> Any:
    """Import OpenCV lazily so PID math/tests do not need a GUI dependency."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on host package.
        raise CameraTunerError(
            "相机窗口需要 OpenCV；请在 PID_controller 目录运行 "
            "`uv sync --extra gui` 后重试。"
        ) from error
    return cv2


def _finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space rectangle used by the display and template tracker."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding-box width and height must be positive")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + 0.5 * self.width, self.y + 0.5 * self.height)

    def clipped(self, frame_width: int, frame_height: int) -> "BoundingBox | None":
        x0 = max(0, min(int(self.x), frame_width - 1))
        y0 = max(0, min(int(self.y), frame_height - 1))
        x1 = max(x0 + 1, min(int(self.x + self.width), frame_width))
        y1 = max(y0 + 1, min(int(self.y + self.height), frame_height))
        if x0 >= frame_width or y0 >= frame_height or x1 <= x0 or y1 <= y0:
            return None
        return BoundingBox(x0, y0, x1 - x0, y1 - y0)


def parse_roi(value: str) -> BoundingBox:
    """Parse ``x,y,w,h`` from a command-line argument."""
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h") from error
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("ROI 必须是正宽高的 x,y,w,h")
    return BoundingBox(*parts)


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics used to turn a pixel and fixed range into 3-D."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        self.fx = _finite_scalar(self.fx, "fx")
        self.fy = _finite_scalar(self.fy, "fy")
        self.cx = _finite_scalar(self.cx, "cx")
        self.cy = _finite_scalar(self.cy, "cy")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive")

    @classmethod
    def from_frame(
        cls,
        frame_width: int,
        frame_height: int,
        *,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
    ) -> "CameraIntrinsics":
        # If no calibration is supplied, max(width,height) is a conservative
        # FOV approximation. The on-screen warning tells the operator to
        # replace it with the calibrated values before using a real vehicle.
        fallback_focal = float(max(frame_width, frame_height))
        return cls(
            fallback_focal if fx is None else fx,
            fallback_focal if fy is None else fy,
            0.5 * frame_width if cx is None else cx,
            0.5 * frame_height if cy is None else cy,
        )

    def pixel_to_camera(self, u: float, v: float, range_m: float) -> np.ndarray:
        depth = _finite_scalar(range_m, "range_m")
        if depth <= 0.0:
            raise ValueError("range_m must be positive")
        u = _finite_scalar(u, "u")
        v = _finite_scalar(v, "v")
        return np.array(
            [(u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth],
            dtype=float,
        )


def camera_position_from_bbox(
    bbox: BoundingBox, intrinsics: CameraIntrinsics, range_m: float
) -> np.ndarray:
    """Return OpenCV camera coordinates ``[right, down, forward]``."""
    u, v = bbox.center
    return intrinsics.pixel_to_camera(u, v, range_m)


class TemplateROITracker:
    """Small dependency-light ROI tracker based on OpenCV template matching.

    It is intentionally not a new vision model: the operator selects the
    target once and this class only follows that selected image patch. A low
    score is treated as target loss, which lets the PID safety path disarm.
    """

    def __init__(
        self,
        initial_bbox: BoundingBox,
        *,
        match_threshold: float = 0.48,
        search_scale: float = 2.5,
        template_update: float = 0.04,
    ) -> None:
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if search_scale < 1.0 or template_update < 0.0 or template_update > 1.0:
            raise ValueError("invalid template-tracker parameters")
        self.bbox = initial_bbox
        self.match_threshold = float(match_threshold)
        self.search_scale = float(search_scale)
        self.template_update = float(template_update)
        self._template: np.ndarray | None = None
        self.last_score: float | None = None

    @property
    def initialized(self) -> bool:
        return self._template is not None

    @staticmethod
    def _gray(frame: np.ndarray, cv2: Any) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("frame must be a grayscale or BGR image")

    def initialize(self, frame: np.ndarray, cv2: Any) -> BoundingBox | None:
        gray = self._gray(frame, cv2)
        clipped = self.bbox.clipped(gray.shape[1], gray.shape[0])
        if clipped is None:
            return None
        self.bbox = clipped
        patch = gray[clipped.y : clipped.y + clipped.height, clipped.x : clipped.x + clipped.width]
        if patch.size == 0:
            return None
        self._template = patch.astype(np.float32)
        self.last_score = 1.0
        return self.bbox

    def update(self, frame: np.ndarray, cv2: Any) -> tuple[BoundingBox | None, float | None]:
        gray = self._gray(frame, cv2)
        if self._template is None:
            bbox = self.initialize(frame, cv2)
            return bbox, self.last_score
        height, width = gray.shape[:2]
        box = self.bbox
        margin_x = max(box.width, int(box.width * (self.search_scale - 1.0) * 0.5))
        margin_y = max(box.height, int(box.height * (self.search_scale - 1.0) * 0.5))
        x0 = max(0, box.x - margin_x)
        y0 = max(0, box.y - margin_y)
        x1 = min(width, box.x + box.width + margin_x)
        y1 = min(height, box.y + box.height + margin_y)
        search = gray[y0:y1, x0:x1]
        template = self._template
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            self.last_score = None
            return None, None
        search_float = search.astype(np.float32)
        # CCOEFF is undefined for a flat patch (for example a bright marker
        # with a solid interior). SQDIFF remains meaningful in that case.
        if float(np.std(template)) < 1.0e-6:
            result = cv2.matchTemplate(search_float, template, cv2.TM_SQDIFF_NORMED)
            min_score, _, min_location, _ = cv2.minMaxLoc(result)
            score = 1.0 - float(min_score)
            location = min_location
        else:
            result = cv2.matchTemplate(search_float, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            score = float(score)
        self.last_score = score
        if not np.isfinite(score) or score < self.match_threshold:
            return None, self.last_score
        next_box = BoundingBox(
            int(x0 + location[0]),
            int(y0 + location[1]),
            box.width,
            box.height,
        ).clipped(width, height)
        if next_box is None:
            return None, self.last_score
        self.bbox = next_box
        if self.template_update > 0.0:
            patch = gray[
                next_box.y : next_box.y + next_box.height,
                next_box.x : next_box.x + next_box.width,
            ].astype(np.float32)
            if patch.shape == template.shape:
                alpha = self.template_update
                self._template = (1.0 - alpha) * template + alpha * patch
        return next_box, self.last_score


@dataclass(frozen=True)
class ErrorSample:
    timestamp: float
    error: np.ndarray
    output: np.ndarray
    valid: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        error = np.asarray(self.error, dtype=float).reshape(-1)
        output = np.asarray(self.output, dtype=float).reshape(-1)
        if error.shape != (4,) or output.shape != (4,):
            raise ValueError("error and output must contain forward/right/down/yaw")
        if not np.all(np.isfinite(error)) or not np.all(np.isfinite(output)):
            raise ValueError("error and output must be finite")
        object.__setattr__(self, "error", error.copy())
        object.__setattr__(self, "output", output.copy())


class ErrorHistory:
    """Bounded real-time history suitable for an OpenCV plot window."""

    def __init__(self, maxlen: int = 240) -> None:
        if int(maxlen) <= 1:
            raise ValueError("maxlen must be greater than one")
        self._samples: deque[ErrorSample] = deque(maxlen=int(maxlen))

    def append(
        self,
        timestamp: float,
        error: Sequence[float] | None,
        output: Sequence[float] | None = None,
    ) -> None:
        if error is None:
            return
        values = np.asarray(error, dtype=float).reshape(-1)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("error must be a finite 4-vector")
        if output is None:
            output_values = np.zeros(4, dtype=float)
        else:
            output_values = np.asarray(output, dtype=float).reshape(-1)
        self._samples.append(ErrorSample(float(timestamp), values, output_values))

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._samples:
            return np.empty((0,)), np.empty((0, 4)), np.empty((0, 4))
        samples = list(self._samples)
        return (
            np.asarray([sample.timestamp for sample in samples], dtype=float),
            np.vstack([sample.error for sample in samples]),
            np.vstack([sample.output for sample in samples]),
        )


@dataclass(frozen=True)
class _Slider:
    name: str
    group: str
    index: int | None
    scale: float
    minimum: float
    maximum: float

    @property
    def span(self) -> int:
        return int(round((self.maximum - self.minimum) / self.scale))

    def encode(self, value: float) -> int:
        return int(np.clip(round((float(value) - self.minimum) / self.scale), 0, self.span))

    def decode(self, position: int) -> float:
        return float(self.minimum + int(position) * self.scale)


class PIDTuningPanel:
    """OpenCV trackbars that mutate only the existing PID configuration."""

    WINDOW = "PID tuning"

    def __init__(self, tracker: PIDTracker, cv2: Any, *, fixed_yaw: bool = False) -> None:
        self.tracker = tracker
        self.cv2 = cv2
        self.fixed_yaw = bool(fixed_yaw)
        self._sliders = self._build_sliders()
        self._created = False

    def _build_sliders(self) -> list[_Slider]:
        sliders: list[_Slider] = []
        axes = ("forward", "right", "down")
        short = ("F", "R", "D")
        for index, axis in enumerate(axes):
            sliders.extend(
                (
                    _Slider(f"P_{short[index]}", "kp", index, 0.1, 0.0, 100.0),
                    _Slider(f"I_{short[index]}", "ki", index, 0.01, 0.0, 10.0),
                    _Slider(f"D_{short[index]}", "kd", index, 0.1, 0.0, 100.0),
                )
            )
        sliders.extend(
            (
                _Slider("P_Y", "yaw_kp", None, 0.01, 0.0, 10.0),
                _Slider("I_Y", "yaw_ki", None, 0.001, 0.0, 1.0),
                _Slider("D_Y", "yaw_kd", None, 0.01, 0.0, 10.0),
                _Slider("Ref_F", "reference", 0, 0.01, -2.0, 4.0),
                _Slider("Ref_R", "reference", 1, 0.01, -2.0, 2.0),
                _Slider("Ref_D", "reference", 2, 0.01, -2.0, 2.0),
                _Slider("Range_m", "range", None, 0.01, 0.20, 10.0),
            )
        )
        if self.fixed_yaw:
            sliders.append(_Slider("YawRef_deg", "yaw_reference", None, 1.0, -180.0, 180.0))
        return sliders

    def create(self, *, range_m: float) -> None:
        self.cv2.namedWindow(self.WINDOW, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.WINDOW, 500, 500)
        for slider in self._sliders:
            value = self._current_value(slider, range_m)
            self.cv2.createTrackbar(
                slider.name,
                self.WINDOW,
                slider.encode(value),
                slider.span,
                lambda _position: None,
            )
        self._created = True

    def _current_value(self, slider: _Slider, range_m: float) -> float:
        config = self.tracker.controller.config
        if slider.group in ("kp", "ki", "kd"):
            return float(getattr(config, slider.group)[int(slider.index)])
        if slider.group == "yaw_kp":
            return float(self.tracker.yaw_controller.config.kp)
        if slider.group == "yaw_ki":
            return float(self.tracker.yaw_controller.config.ki)
        if slider.group == "yaw_kd":
            return float(self.tracker.yaw_controller.config.kd)
        if slider.group == "reference":
            return float(config.reference_position[int(slider.index)])
        if slider.group == "range":
            return float(range_m)
        return 0.0

    def read(self, range_m: float) -> tuple[np.ndarray, float, float | None]:
        """Read sliders, update gains without resetting PID history.

        Returns ``(reference_position, range_m, fixed_yaw_reference_rad)``.
        """
        if not self._created:
            return (
                np.asarray(self.tracker.controller.config.reference_position, dtype=float),
                float(range_m),
                None,
            )
        config = self.tracker.controller.config
        kp = np.asarray(config.kp, dtype=float).copy()
        ki = np.asarray(config.ki, dtype=float).copy()
        kd = np.asarray(config.kd, dtype=float).copy()
        reference = np.asarray(config.reference_position, dtype=float).copy()
        yaw_reference: float | None = None
        new_range = float(range_m)
        for slider in self._sliders:
            position = self.cv2.getTrackbarPos(slider.name, self.WINDOW)
            value = slider.decode(position)
            if slider.group in ("kp", "ki", "kd"):
                axis = int(slider.index)
                if slider.group == "kp":
                    kp[axis] = value
                elif slider.group == "ki":
                    ki[axis] = value
                else:
                    kd[axis] = value
            elif slider.group == "yaw_kp":
                self.tracker.yaw_controller.config.kp = value
            elif slider.group == "yaw_ki":
                self.tracker.yaw_controller.config.ki = value
            elif slider.group == "yaw_kd":
                self.tracker.yaw_controller.config.kd = value
            elif slider.group == "reference":
                reference[int(slider.index)] = value
            elif slider.group == "range":
                new_range = max(0.05, value)
            elif slider.group == "yaw_reference":
                yaw_reference = math.radians(value)
        config.kp = kp
        config.ki = ki
        config.kd = kd
        config.reference_position = reference
        config.normalized()
        self.tracker.yaw_controller.config.normalized()
        return reference, new_range, yaw_reference


def _put_text(cv2: Any, image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (230, 230, 230), scale: float = 0.52) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_camera_window(
    cv2: Any,
    frame: np.ndarray,
    *,
    bbox: BoundingBox | None,
    score: float | None,
    position_camera: np.ndarray | None,
    error: np.ndarray | None,
    force: np.ndarray | None,
    yaw_error: float | None,
    yaw_moment: float,
    status: str,
    armed: bool,
    calibrated_intrinsics: bool,
) -> np.ndarray:
    """Draw a camera frame and control state; pure display helper for tests."""
    image = np.asarray(frame).copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if bbox is not None:
        cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (0, 220, 0), 2)
        u, v = bbox.center
        cv2.drawMarker(image, (int(round(u)), int(round(v))), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    height = image.shape[0]
    line = 23
    state_color = (0, 180, 0) if armed else (0, 180, 255)
    _put_text(cv2, image, f"PID camera | {'ARMED' if armed else 'DISARMED'}", (12, line), state_color)
    _put_text(cv2, image, f"status: {status}", (12, line * 2))
    if score is not None:
        _put_text(cv2, image, f"ROI score: {score:.2f}", (12, line * 3))
    if position_camera is not None:
        _put_text(cv2, image, "camera [R,D,F] = " + np.array2string(position_camera, precision=2), (12, line * 4))
    if error is not None:
        _put_text(cv2, image, "error [F,R,D] = " + np.array2string(error, precision=3), (12, line * 5))
    if force is not None:
        _put_text(cv2, image, "force [F,R,D] = " + np.array2string(force, precision=2), (12, line * 6))
    if yaw_error is not None:
        _put_text(cv2, image, f"yaw error={math.degrees(yaw_error):+.1f} deg  moment={yaw_moment:+.2f}", (12, line * 7))
    if not calibrated_intrinsics:
        _put_text(cv2, image, "WARNING: using approximate intrinsics; calibrate before hardware", (12, height - 38), (0, 80, 255), 0.48)
    _put_text(cv2, image, "a arm (only --enable-arm) | d/space disarm | r reset | q/ESC quit", (12, height - 14), (220, 220, 220), 0.46)
    return image


def render_error_window(cv2: Any, history: ErrorHistory, *, width: int = 900, height: int = 520) -> np.ndarray:
    """Render bounded error and force traces with OpenCV only."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    margin_left, margin_right = 62, 18
    top, bottom = 42, 30
    split = height // 2
    colors = ((60, 210, 255), (80, 220, 120), (255, 150, 70), (220, 100, 220))
    labels = ("forward", "right", "down", "yaw")
    _, errors, outputs = history.arrays()
    cv2.rectangle(canvas, (margin_left, top), (width - margin_right, split - 16), (75, 75, 75), 1)
    cv2.rectangle(canvas, (margin_left, split + 20), (width - margin_right, height - bottom), (75, 75, 75), 1)
    _put_text(cv2, canvas, "real-time PID error", (margin_left, 24), (240, 240, 240), 0.65)
    _put_text(cv2, canvas, "force / yaw moment output", (margin_left, split + 10), (240, 240, 240), 0.55)
    if len(errors) == 0:
        _put_text(cv2, canvas, "waiting for a valid target...", (margin_left + 15, split - 35), (150, 150, 150))
        return canvas

    def draw_panel(values: np.ndarray, y_top: int, y_bottom: int) -> None:
        maximum = float(np.nanmax(np.abs(values))) if values.size else 1.0
        maximum = max(maximum, 1.0e-6)
        y_mid = (y_top + y_bottom) // 2
        cv2.line(canvas, (margin_left, y_mid), (width - margin_right, y_mid), (80, 80, 80), 1)
        _put_text(cv2, canvas, f"+{maximum:.2g}", (5, y_top + 5), (150, 150, 150), 0.42)
        _put_text(cv2, canvas, f"-{maximum:.2g}", (5, y_bottom), (150, 150, 150), 0.42)
        count = values.shape[0]
        xs = np.linspace(margin_left, width - margin_right, count).astype(int)
        for index in range(4):
            ys = np.clip(y_mid - values[:, index] / maximum * (y_bottom - y_top) * 0.47, y_top, y_bottom).astype(int)
            points = np.column_stack((xs, ys)).reshape(-1, 1, 2)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, colors[index], 2, cv2.LINE_AA)
            _put_text(cv2, canvas, labels[index], (width - 85, y_top + 18 + index * 18), colors[index], 0.45)

    draw_panel(errors, top, split - 16)
    draw_panel(outputs, split + 20, height - bottom)
    return canvas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="池顶相机的 V4L2/OpenCV 索引")
    parser.add_argument("--camera-device", help="直接指定设备路径，例如 /dev/video3；优先于 --camera-index")
    parser.add_argument("--roi", type=parse_roi, help="首帧目标框 x,y,w,h；不传则弹出 selectROI")
    parser.add_argument("--manual-target", nargs=2, type=float, metavar=("U", "V"), help="固定像素目标，跳过 ROI 跟踪")
    parser.add_argument("--range-m", type=float, default=2.0, help="目标前向距离（单目相机需人工提供，默认 2 m）")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.48)
    parser.add_argument("--history", type=int, default=240, help="误差曲线保留的采样点数")
    parser.add_argument("--port", help="可选：V4pro1_MPC USART3 串口；不传则 dry-run")
    parser.add_argument(
        "--runtime-config",
        help="可选：MPC 风格 JSON；按其中 transport 使用 TCP/UDP/serial",
    )
    parser.add_argument("--dry-run", action="store_true", help="显式指定不连接任何硬件传输（默认行为）")
    parser.add_argument("--enable-arm", action="store_true", help="允许按 a 请求 armed；默认永远 disarmed")
    parser.add_argument("--fixed-yaw", action="store_true", help="使用调参窗口中的固定 yaw 参考；默认朝向目标")
    parser.add_argument("--max-frames", type=int, default=0, help="测试/演示时自动退出的帧数，0 表示不限制")
    return parser


def _make_bbox_from_point(u: float, v: float, width: int, height: int) -> BoundingBox:
    size = max(8, min(width, height) // 40)
    return BoundingBox(int(round(u - size / 2)), int(round(v - size / 2)), size, size).clipped(width, height) or BoundingBox(0, 0, 1, 1)


def _safe_output_vectors(output: PIDTrackerOutput | None) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float]:
    if output is None:
        return None, None, None, 0.0
    error = output.pid.error.copy()
    force = output.pid.force.copy()
    yaw_error = None if output.yaw_pid is None else float(output.yaw_pid.angle_error)
    yaw_moment = 0.0 if output.yaw_pid is None else float(output.yaw_pid.yaw_moment)
    return error, force, yaw_error, yaw_moment


def run(args: argparse.Namespace) -> int:
    if args.port or args.runtime_config:
        raise CameraTunerError(
            "旧的单目池顶入口禁止接入实机；请使用 "
            "dual_vision_pid_runtime.py，让双目 3-D JSONL 驱动 PID，"
            "池顶相机仅作位置/速度辅助"
        )
    if args.range_m <= 0.0:
        raise CameraTunerError("--range-m 必须为正数")
    if args.history <= 1:
        raise CameraTunerError("--history 必须大于 1")
    if args.max_frames < 0:
        raise CameraTunerError("--max-frames 不能为负数")
    if args.port and args.runtime_config:
        raise CameraTunerError("--port 与 --runtime-config 不能同时使用")
    if (args.port or args.runtime_config) and args.dry_run:
        raise CameraTunerError("硬件连接参数不能与 --dry-run 同时使用")
    cv2 = _load_cv2()
    camera_source: int | str = args.camera_device if args.camera_device else int(args.camera_index)
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        capture.release()
        raise CameraTunerError(f"无法打开相机 source={camera_source}；请用 --camera-index/--camera-device 选择池顶相机")

    # The camera window is the same live control path as hardware_session;
    # start from the calibrated camera/body frame and its matching standoff
    # reference so dry-run tuning does not change direction when armed.
    tracker = build_tracker(calibrated_reference=True)
    history = ErrorHistory(args.history)
    hardware: PIDHardwareSession | None = None
    arm_requested = False
    last_force = np.zeros(3, dtype=float)
    last_yaw_moment = 0.0
    initialized = False
    hardware_requested = bool(args.port or args.runtime_config)
    status = "dry-run: waiting for target" if not hardware_requested else "hardware: connecting disarmed"
    roi_tracker: TemplateROITracker | None = None
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise CameraTunerError("相机已打开但读不到首帧")
        frame_height, frame_width = first_frame.shape[:2]
        intrinsics = CameraIntrinsics.from_frame(
            frame_width,
            frame_height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
        )
        calibrated_intrinsics = all(value is not None for value in (args.fx, args.fy, args.cx, args.cy))
        if args.manual_target is None:
            initial_bbox = args.roi
            if initial_bbox is None:
                selected = cv2.selectROI("PID camera", first_frame, fromCenter=False, showCrosshair=True)
                initial_bbox = BoundingBox(int(selected[0]), int(selected[1]), int(selected[2]), int(selected[3])) if selected[2] > 0 and selected[3] > 0 else None
            if initial_bbox is None:
                raise CameraTunerError("没有选择 ROI；请重新运行并框选池中目标")
            roi_tracker = TemplateROITracker(initial_bbox, match_threshold=args.match_threshold)
            if roi_tracker.initialize(first_frame, cv2) is None:
                raise CameraTunerError("ROI 超出首帧范围")

        panel = PIDTuningPanel(tracker, cv2, fixed_yaw=args.fixed_yaw)
        try:
            panel.create(range_m=args.range_m)
            cv2.namedWindow("PID camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("PID error", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PID error", 900, 520)
        except Exception as error:
            raise CameraTunerError(
                "无法创建相机窗口；请确认有桌面 DISPLAY，或在带 GUI 的终端运行"
            ) from error

        if args.runtime_config:
            hardware = build_runtime_hardware_session(
                args.runtime_config,
                tracker=tracker,
                logger=lambda message: print(f"[hardware] {message}"),
            )
            if not hardware.connect():
                raise CameraTunerError("runtime-config transport 或 disarm session 握手失败；未发送 armed")
            status = "hardware: runtime-config session established, disarmed"
        elif args.port:
            hardware = build_serial_hardware_session(
                args.port,
                logger=lambda message: print(f"[serial] {message}"),
            )
            # Reuse the exact tracker being tuned; protocol/transport remains
            # the already-tested PID-only hardware bridge.
            hardware.tracker = tracker
            if not hardware.connect():
                raise CameraTunerError("串口连接或 disarm session 握手失败；未发送 armed")
            status = "hardware: session established, disarmed"

        frame = first_frame
        frame_count = 0
        while True:
            reference, range_m, fixed_yaw_reference = panel.read(args.range_m)
            bbox: BoundingBox | None = None
            score: float | None = None
            position_camera: np.ndarray | None = None
            output: PIDTrackerOutput | None = None
            yaw_angle = 0.0
            if args.manual_target is not None:
                u, v = (float(item) for item in args.manual_target)
                bbox = _make_bbox_from_point(u, v, frame_width, frame_height)
                position_camera = intrinsics.pixel_to_camera(u, v, range_m)
                score = 1.0
            elif roi_tracker is not None:
                bbox, score = roi_tracker.update(frame, cv2)
                if bbox is not None:
                    position_camera = camera_position_from_bbox(bbox, intrinsics, range_m)

            if position_camera is None:
                status = "target lost: disarmed" if hardware is not None else "target lost: PID hold baseline"
                if hardware is not None:
                    hardware.target_lost()
                    # A target loss while armed is a latched safety stop.  Do
                    # not let a stale `a` request automatically re-arm when
                    # vision happens to return.
                    arm_requested = False
                    initialized = False
                elif initialized:
                    safe: SafeControlOutput = tracker.target_lost(last_force, last_yaw_moment)
                    last_force = safe.force.copy()
                    last_yaw_moment = safe.yaw_moment
            elif hardware is not None:
                result = hardware.step(
                    position_camera,
                    arm_requested=arm_requested,
                    reference_position=reference,
                    reference_yaw_rad=fixed_yaw_reference,
                )
                output = result.controller_output
                status = result.status
                if hardware.safety_latched:
                    arm_requested = False
                if output is not None:
                    yaw_angle = 0.0 if result.telemetry is None else result.telemetry.yaw_rad
                    last_force = output.pid.force.copy()
                    last_yaw_moment = 0.0 if output.yaw_pid is None else output.yaw_pid.yaw_moment
                    initialized = True
            else:
                # Keep the dry-run plot and the hardware path on the exact
                # same calibrated camera/body axes.  Otherwise tuning in the
                # window would produce a different direction once armed.
                position_body = camera_to_pid_body_position(position_camera)
                if not initialized:
                    tracker.latch_baseline(last_force, last_yaw_moment, yaw_angle)
                    initialized = True
                output = tracker.update(
                    position_body,
                    last_force,
                    reference_position=reference,
                    yaw_rad=yaw_angle,
                    achieved_yaw_moment_previous=last_yaw_moment,
                    reference_yaw_rad=fixed_yaw_reference,
                )
                last_force = output.pid.force.copy()
                last_yaw_moment = 0.0 if output.yaw_pid is None else output.yaw_pid.yaw_moment
                status = "dry-run: PID active"

            error, force, yaw_error, yaw_moment = _safe_output_vectors(output)
            if error is not None:
                history.append(time.monotonic(), np.r_[error, yaw_error if yaw_error is not None else 0.0], np.r_[force, yaw_moment])
            actual_armed = bool(
                hardware is not None
                and arm_requested
                and hardware.connection.latest_telemetry is not None
                and hardware.connection.latest_telemetry.armed
                and hardware.connection.armed_confirmation_fresh()
            )
            camera_view = render_camera_window(
                cv2,
                frame,
                bbox=bbox,
                score=score,
                position_camera=position_camera,
                error=error,
                force=force,
                yaw_error=yaw_error,
                yaw_moment=yaw_moment,
                status=status,
                armed=actual_armed,
                calibrated_intrinsics=calibrated_intrinsics,
            )
            cv2.imshow("PID camera", camera_view)
            cv2.imshow("PID error", render_error_window(cv2, history))
            key = int(cv2.waitKey(1) & 0xFF)
            if key in (27, ord("q")):
                break
            if key in (ord("d"), ord(" ")):
                arm_requested = False
                status = "disarm requested"
                if hardware is not None:
                    hardware.connection.send_disarm()
            elif key == ord("a"):
                if hardware is not None and args.enable_arm:
                    arm_requested = True
                    status = "arm requested; waiting for telemetry confirmation"
                else:
                    status = "arm blocked (requires --port and --enable-arm)"
            elif key == ord("r"):
                tracker.controller.reset(keep_baseline=True)
                if tracker.yaw_controller is not None:
                    tracker.yaw_controller.reset(yaw_angle)
                history.clear()
                initialized = False
                last_force.fill(0.0)
                last_yaw_moment = 0.0
                status = "PID history reset"
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
            ok, next_frame = capture.read()
            if not ok or next_frame is None:
                status = "camera frame lost: disarmed"
                if hardware is not None:
                    hardware.target_lost()
                    arm_requested = False
                break
            frame = next_frame
    finally:
        arm_requested = False
        if hardware is not None:
            hardware.close()
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
    except (CameraTunerError, ValueError) as error:
        print(f"camera_pid_tuner: {error}")
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually.
    raise SystemExit(main())
