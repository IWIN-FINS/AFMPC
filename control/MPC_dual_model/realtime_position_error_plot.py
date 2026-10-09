"""Live position-error plot for FineSUB experimental AUTO JSONL traces."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .auto_readiness import DEFAULT_CONFIG_PATH


AXIS_NAMES = ("Forward", "Right", "Down")
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
ERROR_REFERENCE_LEVELS_CM = (5.0, 10.0)


@dataclass(frozen=True)
class PositionErrorSample:
    time_s: float
    measured_error_m: np.ndarray
    estimated_error_m: np.ndarray
    estimated_velocity_m_s: np.ndarray
    model1_weight: np.ndarray
    wall_time_s: float = float("nan")
    target_absolute_speed_m_s: float = float("nan")
    # SMC diagnostics are optional so the MPC plot can continue to consume
    # the same trace schema.  NaN means that the selected trace did not carry
    # that field (for example, a legacy MPC record).
    smc_desired_acceleration_m_s2: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    smc_sliding_variable: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    requested_force_frd_n: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    achieved_force_frd_n: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )
    camera_forward_distance_m: float = float("nan")
    camera_forward_force_before_guard_n: float = float("nan")
    camera_forward_force_after_guard_n: float = float("nan")
    vision_measurement_age_s: float = float("nan")


@dataclass
class OverheadTargetVelocitySource:
    """Read target speed from a pool-top ROS2 SQLite bag for visualization only.

    The source uses AprilTag 17's world-frame XY velocity from the existing
    overhead-camera loader.  It never feeds the MPC or alters the control
    trace.  A directory is accepted and resolves to its newest ``*.db3`` file.
    """

    path: Path | None
    refresh_interval_s: float = 1.0
    _times_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float), init=False
    )
    _velocity_xy_m_s: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float), init=False
    )
    _signature: tuple[int, int] | None = field(default=None, init=False)
    _last_refresh_monotonic_s: float = field(default=-float("inf"), init=False)
    _next_retry_monotonic_s: float = field(default=-float("inf"), init=False)
    _failure_backoff_s: float = field(default=1.0, init=False)

    def _database_path(self) -> Path | None:
        if self.path is None:
            return None
        candidate = Path(self.path)
        if candidate.is_dir():
            databases = list(candidate.glob("*.db3"))
            if not databases:
                return None
            try:
                return max(
                    databases,
                    key=lambda path: (path.stat().st_mtime_ns, path.name),
                )
            except OSError:
                return None
        return candidate

    def _refresh_if_needed(self) -> None:
        # ros2 bag writes the SQLite file before it writes metadata.yaml.  A
        # live database is not a stable snapshot and can be reported as
        # malformed while the recorder is appending pages.  Do not repeatedly
        # scan that file from the GUI timer; the speed panel will populate once
        # the recorder is stopped and metadata.yaml is finalized.
        if self.path is not None and self.path.is_dir():
            if not (self.path / "metadata.yaml").exists():
                return
        database = self._database_path()
        if database is None:
            return
        now = time.monotonic()
        if now < self._next_retry_monotonic_s:
            return
        if now - self._last_refresh_monotonic_s < self.refresh_interval_s:
            return
        self._last_refresh_monotonic_s = now
        try:
            stat = database.stat()
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            if signature == self._signature and self._times_s.size:
                return
            # Import lazily so the live plot remains usable without a bag or
            # without loading the offline MPC fusion analysis module.
            from .fusion_identifiability_validation import _load_overhead

            overhead = _load_overhead(database, read_only_snapshot=True)
            times = np.asarray(overhead["time"], dtype=float)
            velocity = np.asarray(overhead["target_velocity"], dtype=float)
            valid = (
                np.isfinite(times)
                & np.all(np.isfinite(velocity), axis=1)
            )
            self._times_s = times[valid]
            self._velocity_xy_m_s = velocity[valid]
            self._signature = signature
            self._failure_backoff_s = 1.0
            self._next_retry_monotonic_s = -float("inf")
        except (OSError, ValueError, sqlite3.Error, KeyError, TypeError):
            # A bag can be incomplete while the top-camera recorder is still
            # writing.  Keep the last valid source and leave the panel blank
            # until a readable snapshot is available.  Back off retries so a
            # malformed/incomplete database cannot monopolize the GUI thread.
            self._next_retry_monotonic_s = now + self._failure_backoff_s
            self._failure_backoff_s = min(30.0, self._failure_backoff_s * 2.0)
            return

    def speed_at(self, wall_time_s: float) -> float:
        """Return interpolated target speed, or NaN outside the bag range."""

        self._refresh_if_needed()
        query = float(wall_time_s)
        if (
            not np.isfinite(query)
            or self._times_s.size == 0
            or query < self._times_s[0]
            or query > self._times_s[-1]
        ):
            return float("nan")
        velocity = np.array(
            [
                np.interp(query, self._times_s, self._velocity_xy_m_s[:, axis])
                for axis in range(2)
            ],
            dtype=float,
        )
        return float(np.linalg.norm(velocity))


def load_default_reference(config_path: str | Path) -> np.ndarray:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    reference = config["experimental_auto"]["active_mpc_parameters"][
        "controller"
    ]["reference_position"]
    result = np.asarray(reference, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("configured reference_position must be a finite 3-vector")
    return result


def extract_position_error_sample(
    record: dict,
    default_reference: np.ndarray,
) -> PositionErrorSample | None:
    if record.get("event") != "control_update":
        return None
    reference_value = record.get("reference_position_body_frd_m")
    reference = (
        default_reference
        if reference_value is None
        else np.asarray(reference_value, dtype=float).reshape(-1)
    )
    measured = np.asarray(record.get("position_body_frd_m"), dtype=float).reshape(-1)
    estimated_state = np.asarray(record.get("estimated_state"), dtype=float).reshape(-1)
    if reference.shape != (3,) or measured.shape != (3,) or estimated_state.size < 3:
        return None
    estimated = estimated_state[:3]
    values = np.concatenate((reference, measured, estimated))
    if not np.all(np.isfinite(values)):
        return None
    timestamp = float(record.get("host_monotonic_s", record.get("host_time_s")))
    if not np.isfinite(timestamp):
        return None
    velocity = np.full(3, np.nan)
    if estimated_state.size >= 6 and np.all(np.isfinite(estimated_state[3:6])):
        velocity = estimated_state[3:6].copy()
    weight_value = record.get("model1_weight")
    model1_weight = np.full(3, np.nan)
    if weight_value is not None:
        candidate = np.asarray(weight_value, dtype=float).reshape(-1)
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            model1_weight = candidate
    def optional_vector(name: str) -> np.ndarray:
        value = record.get(name)
        if value is None:
            return np.full(3, np.nan, dtype=float)
        candidate = np.asarray(value, dtype=float).reshape(-1)
        if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
            return np.full(3, np.nan, dtype=float)
        return candidate.copy()

    def optional_scalar(name: str) -> float:
        value = record.get(name)
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return candidate if np.isfinite(candidate) else float("nan")

    return PositionErrorSample(
        time_s=timestamp,
        measured_error_m=measured - reference,
        estimated_error_m=estimated - reference,
        estimated_velocity_m_s=velocity,
        model1_weight=model1_weight,
        wall_time_s=float(record.get("host_time_s", float("nan"))),
        smc_desired_acceleration_m_s2=optional_vector(
            "smc_desired_acceleration_frd_m_s2"
        ),
        smc_sliding_variable=optional_vector("smc_sliding_variable"),
        requested_force_frd_n=optional_vector("requested_force_frd_n"),
        achieved_force_frd_n=optional_vector("achieved_force_previous_frd_n"),
        camera_forward_distance_m=optional_scalar("camera_forward_distance_m"),
        camera_forward_force_before_guard_n=optional_scalar(
            "camera_forward_force_before_guard_n"
        ),
        camera_forward_force_after_guard_n=optional_scalar(
            "camera_forward_force_after_guard_n"
        ),
        vision_measurement_age_s=optional_scalar("vision_measurement_age_s"),
    )


class JsonlTraceFollower:
    """Incrementally follow one trace, or automatically select the latest trace."""

    def __init__(
        self,
        trace_path: str | Path | None,
        log_directory: str | Path,
        default_reference: np.ndarray,
        overhead_source: OverheadTargetVelocitySource | None = None,
        trace_pattern: str = "experimental_auto_*.jsonl",
    ) -> None:
        self.explicit_path = None if trace_path is None else Path(trace_path).resolve()
        self.log_directory = Path(log_directory).resolve()
        self.default_reference = np.asarray(default_reference, dtype=float)
        self.overhead_source = overhead_source
        self.trace_pattern = str(trace_pattern)
        self.path: Path | None = None
        self.offset = 0
        self.partial = ""

    def _latest_trace(self) -> Path | None:
        candidates = list(self.log_directory.glob(self.trace_pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def _selected_path(self) -> Path | None:
        return self.explicit_path or self._latest_trace()

    def poll(self) -> tuple[bool, list[PositionErrorSample]]:
        selected = self._selected_path()
        switched = selected is not None and selected != self.path
        if switched:
            self.path = selected
            self.offset = 0
            self.partial = ""
        if self.path is None or not self.path.exists():
            return switched, []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = ""
            switched = True
        with self.path.open("r", encoding="utf-8") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        if not chunk:
            return switched, []
        text = self.partial + chunk
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.partial = lines.pop()
        else:
            self.partial = ""
        samples: list[PositionErrorSample] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = extract_position_error_sample(record, self.default_reference)
            if sample is not None:
                if self.overhead_source is not None:
                    sample = replace(
                        sample,
                        target_absolute_speed_m_s=self.overhead_source.speed_at(
                            sample.wall_time_s
                        ),
                    )
                samples.append(sample)
        return switched, samples


class LivePositionErrorPlot:
    def __init__(
        self,
        follower: JsonlTraceFollower,
        *,
        window_s: float = 60.0,
        maximum_samples: int = 12000,
        show_target_speed: bool = True,
        forward_only: bool = False,
        error_norm_only: bool = False,
    ) -> None:
        self.follower = follower
        self.window_s = float(window_s)
        self.show_target_speed = bool(show_target_speed)
        self.forward_only = bool(forward_only)
        self.error_norm_only = bool(error_norm_only)
        if self.forward_only and self.error_norm_only:
            raise ValueError("forward_only and error_norm_only are mutually exclusive")
        self.samples: deque[PositionErrorSample] = deque(maxlen=maximum_samples)
        if self.error_norm_only:
            self.figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            self.axes = tuple(axes)
            error_axis, weight_axis = self.axes
            self.error_norm_line, = error_axis.plot(
                [], [], color=AXIS_COLORS[0], linewidth=2.0,
                label="Estimated 3D error norm",
            )
            for level_cm in ERROR_REFERENCE_LEVELS_CM:
                error_axis.axhline(
                    level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":", linewidth=0.9, alpha=0.75,
                    label=f"{level_cm:g} cm",
                )
            error_axis.set_title("Estimated 3D position-error norm")
            error_axis.set_ylabel("Error norm (cm)")
            error_axis.grid(True, alpha=0.25)
            error_axis.legend(loc="upper right")
            self.model1_forward_weight_line, = weight_axis.plot(
                [], [], color=AXIS_COLORS[0], linewidth=2.0,
                label="Forward model-2 fusion weight",
            )
            weight_axis.axhline(1.0, color="black", linewidth=0.7, alpha=0.5)
            weight_axis.axhline(0.5, color="0.5", linewidth=0.7, alpha=0.4)
            weight_axis.set_ylim(-0.03, 1.03)
            weight_axis.set_title("Forward model-2 fusion weight")
            weight_axis.set_ylabel("Weight")
            weight_axis.set_xlabel("Time in current trace (s)")
            weight_axis.grid(True, alpha=0.25)
            weight_axis.legend(loc="lower right")
            self.status_text = self.figure.suptitle("Waiting for MPC control updates")
            self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
            return
        if self.forward_only:
            self.figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            self.axes = tuple(axes)
            error_axis, weight_axis = self.axes
            self.forward_measured_line, = error_axis.plot(
                [], [], "--", color="0.5", linewidth=1.2, label="Measured"
            )
            self.forward_estimated_line, = error_axis.plot(
                [], [], color=AXIS_COLORS[0], linewidth=2.0, label="Kalman estimate"
            )
            error_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            for level_cm in ERROR_REFERENCE_LEVELS_CM:
                for sign in (-1.0, 1.0):
                    error_axis.axhline(
                        sign * level_cm,
                        color="tab:orange" if level_cm == 5.0 else "tab:purple",
                        linestyle=":", linewidth=0.9, alpha=0.75,
                        label=f"±{level_cm:g} cm" if sign > 0 else None,
                    )
            error_axis.set_title("Forward position error")
            error_axis.set_ylabel("Error (cm)")
            error_axis.grid(True, alpha=0.25)
            error_axis.legend(loc="upper right")
            self.model1_forward_weight_line, = weight_axis.plot(
                [], [], color=AXIS_COLORS[0], linewidth=2.0,
                label="Forward model-2 fusion weight",
            )
            weight_axis.axhline(1.0, color="black", linewidth=0.7, alpha=0.5)
            weight_axis.axhline(0.5, color="0.5", linewidth=0.7, alpha=0.4)
            weight_axis.set_ylim(-0.03, 1.03)
            weight_axis.set_title("Forward model-2 fusion weight")
            weight_axis.set_ylabel("Weight")
            weight_axis.set_xlabel("Time in current trace (s)")
            weight_axis.grid(True, alpha=0.25)
            weight_axis.legend(loc="lower right")
            self.status_text = self.figure.suptitle("Waiting for MPC control updates")
            self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
            return
        self.figure, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
        self.axes = tuple(axes.ravel())
        self.measured_lines = []
        self.estimated_lines = []
        for axis_index, axis in enumerate(self.axes[:3]):
            measured, = axis.plot([], [], "--", color="0.65", linewidth=1.0,
                                  label="Measured")
            estimated, = axis.plot([], [], color=AXIS_COLORS[axis_index],
                                   linewidth=1.8, label="Kalman estimate")
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            for level_cm in ERROR_REFERENCE_LEVELS_CM:
                label = f"±{level_cm:g} cm" if axis_index == 0 else None
                axis.axhline(
                    level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                    label=label,
                )
                axis.axhline(
                    -level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                )
            axis.set_title(f"{AXIS_NAMES[axis_index]} error")
            axis.set_ylabel("Error (cm)")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right")
            self.measured_lines.append(measured)
            self.estimated_lines.append(estimated)
        norm_axis = self.axes[3]
        self.norm_line, = norm_axis.plot([], [], color="tab:purple", linewidth=2.0)
        for level_cm in ERROR_REFERENCE_LEVELS_CM:
            norm_axis.axhline(
                level_cm,
                color="tab:orange" if level_cm == 5.0 else "tab:purple",
                linestyle=":",
                linewidth=0.9,
                alpha=0.75,
                label=f"{level_cm:g} cm",
            )
        norm_axis.set_title("Estimated 3D position-error norm")
        norm_axis.set_ylabel("Error norm (cm)")
        norm_axis.grid(True, alpha=0.25)
        norm_axis.legend(loc="upper right")
        weight_axis = self.axes[4]
        self.model1_forward_weight_line, = weight_axis.plot(
            [], [], color=AXIS_COLORS[0], linewidth=2.0, label="Forward"
        )
        weight_axis.axhline(1.0, color="black", linewidth=0.7, alpha=0.5)
        weight_axis.axhline(0.5, color="0.5", linewidth=0.7, alpha=0.4)
        weight_axis.set_ylim(-0.03, 1.03)
        weight_axis.set_title("Forward model-2 fusion weight")
        weight_axis.set_ylabel("Weight")
        weight_axis.grid(True, alpha=0.25)
        weight_axis.legend(loc="lower right")

        velocity_axis = self.axes[5]
        self.target_speed_line, = velocity_axis.plot(
            [], [], color="tab:orange", linewidth=1.8,
            label="Target absolute speed",
        )
        velocity_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        velocity_axis.set_title("Pool-top target absolute speed (visual only)")
        velocity_axis.set_ylabel("Velocity (m/s)")
        velocity_axis.grid(True, alpha=0.25)
        velocity_axis.legend(loc="upper right")
        if not self.show_target_speed:
            velocity_axis.set_visible(False)

        for axis in self.axes[4:]:
            axis.set_xlabel("Time in current trace (s)")
        self.status_text = self.figure.suptitle("Waiting for MPC control updates")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    def _visible_samples(self) -> list[PositionErrorSample]:
        if not self.samples:
            return []
        latest = self.samples[-1].time_s
        minimum = latest - self.window_s
        return [sample for sample in self.samples if sample.time_s >= minimum]

    def refresh(self) -> None:
        switched, new_samples = self.follower.poll()
        if switched:
            self.samples.clear()
        self.samples.extend(new_samples)
        visible = self._visible_samples()
        if not visible:
            selected = self.follower.path
            name = "no trace" if selected is None else selected.name
            self.status_text.set_text(f"Waiting for MPC control updates — {name}")
            return

        time0 = self.samples[0].time_s
        times = np.asarray([sample.time_s - time0 for sample in visible])
        measured_cm = 100.0 * np.vstack(
            [sample.measured_error_m for sample in visible]
        )
        estimated_cm = 100.0 * np.vstack(
            [sample.estimated_error_m for sample in visible]
        )
        target_speed = np.asarray(
            [sample.target_absolute_speed_m_s for sample in visible], dtype=float
        )
        model1_weight = np.vstack([sample.model1_weight for sample in visible])
        if self.error_norm_only:
            norms_cm = np.linalg.norm(estimated_cm, axis=1)
            self.error_norm_line.set_data(times, norms_cm)
            self.axes[0].set_ylim(
                0.0, max(10.0, float(np.nanpercentile(norms_cm, 99.0)) * 1.2),
            )
            self.model1_forward_weight_line.set_data(times, model1_weight[:, 0])
            left = max(float(times[-1]) - self.window_s, float(times[0]))
            right = max(float(times[-1]), left + 1.0)
            for axis in self.axes:
                axis.set_xlim(left, right)
            recent = times >= times[-1] - 5.0
            norm_rms_5s = float(np.sqrt(np.mean(norms_cm[recent] ** 2)))
            weights = model1_weight[recent, 0]
            finite_weights = weights[np.isfinite(weights)]
            weight_mean = float(np.mean(finite_weights)) if finite_weights.size else float("nan")
            sample_age_s = max(0.0, time.monotonic() - self.samples[-1].time_s)
            live_status = "LIVE" if sample_age_s <= 1.0 else f"NO NEW MPC DATA {sample_age_s:.1f}s"
            self.status_text.set_text(
                f"{live_status} — 3D error norm {norms_cm[-1]:.1f} cm   "
                f"5s RMS {norm_rms_5s:.1f} cm   "
                f"Forward model-2 weight (5s mean) {weight_mean:.3f}"
            )
            return
        if self.forward_only:
            self.forward_measured_line.set_data(times, measured_cm[:, 0])
            self.forward_estimated_line.set_data(times, estimated_cm[:, 0])
            values = np.concatenate((measured_cm[:, 0], estimated_cm[:, 0], [-10.0, 10.0]))
            limit = max(10.0, float(np.nanpercentile(np.abs(values), 99.0)) * 1.2)
            self.axes[0].set_ylim(-limit, limit)
            self.model1_forward_weight_line.set_data(times, model1_weight[:, 0])
            left = max(float(times[-1]) - self.window_s, float(times[0]))
            right = max(float(times[-1]), left + 1.0)
            for axis in self.axes:
                axis.set_xlim(left, right)
            recent = times >= times[-1] - 5.0
            forward_rms_5s = float(np.sqrt(np.mean(estimated_cm[recent, 0] ** 2)))
            weights = model1_weight[recent, 0]
            finite_weights = weights[np.isfinite(weights)]
            weight_mean = float(np.mean(finite_weights)) if finite_weights.size else float("nan")
            sample_age_s = max(0.0, time.monotonic() - self.samples[-1].time_s)
            live_status = "LIVE" if sample_age_s <= 1.0 else f"NO NEW MPC DATA {sample_age_s:.1f}s"
            self.status_text.set_text(
                f"{live_status} — Forward error {estimated_cm[-1, 0]:+.1f} cm   "
                f"5s RMS {forward_rms_5s:.1f} cm   "
                f"Forward model-2 weight (5s mean) {weight_mean:.3f}"
            )
            return
        for axis_index, axis in enumerate(self.axes[:3]):
            self.measured_lines[axis_index].set_data(times, measured_cm[:, axis_index])
            self.estimated_lines[axis_index].set_data(times, estimated_cm[:, axis_index])
            values = np.concatenate(
                (
                    measured_cm[:, axis_index],
                    estimated_cm[:, axis_index],
                    [-10.0, 10.0],
                )
            )
            limit = max(10.0, float(np.nanpercentile(np.abs(values), 99.0)) * 1.2)
            axis.set_ylim(-limit, limit)

        norms_cm = np.linalg.norm(estimated_cm, axis=1)
        self.norm_line.set_data(times, norms_cm)
        self.axes[3].set_ylim(
            0.0,
            max(10.0, float(np.nanpercentile(norms_cm, 99.0)) * 1.2),
        )
        self.model1_forward_weight_line.set_data(times, model1_weight[:, 0])
        if self.show_target_speed:
            self.target_speed_line.set_data(times, target_speed)
        finite_speed = target_speed[np.isfinite(target_speed)]
        if self.show_target_speed:
            velocity_limit = (
                max(0.02, float(np.nanpercentile(finite_speed, 99.0)) * 1.2)
                if finite_speed.size
                else 0.02
            )
            self.axes[5].set_ylim(-velocity_limit, velocity_limit)
        left = max(float(times[-1]) - self.window_s, float(times[0]))
        right = max(float(times[-1]), left + 1.0)
        for axis in self.axes:
            axis.set_xlim(left, right)

        current = estimated_cm[-1]
        rms = np.sqrt(np.mean(estimated_cm**2, axis=0))
        p95_norm = float(np.percentile(norms_cm, 95.0))
        recent = times >= times[-1] - 5.0
        forward_mean_5s = float(np.mean(estimated_cm[recent, 0]))
        forward_rms_5s = float(np.sqrt(np.mean(estimated_cm[recent, 0] ** 2)))
        forward_weights = model1_weight[recent, 0]
        finite_forward_weights = forward_weights[np.isfinite(forward_weights)]
        forward_weight_5s = (
            float(np.mean(finite_forward_weights))
            if finite_forward_weights.size
            else float("nan")
        )
        sample_age_s = max(0.0, time.monotonic() - self.samples[-1].time_s)
        live_status = "LIVE" if sample_age_s <= 1.0 else f"NO NEW MPC DATA {sample_age_s:.1f}s"
        trace_name = self.follower.path.name if self.follower.path else "unknown"
        short_trace_name = trace_name.removeprefix("experimental_auto_").removesuffix(
            ".jsonl"
        )
        self.status_text.set_text(
            f"{live_status} — Run {short_trace_name}\nCurrent FRD error="
            f"[{current[0]:+.1f}, {current[1]:+.1f}, {current[2]:+.1f}] cm   "
            f"RMS=[{rms[0]:.1f}, {rms[1]:.1f}, {rms[2]:.1f}] cm   "
            f"3D P95={p95_norm:.1f} cm\n"
            f"Forward 5s mean/RMS={forward_mean_5s:+.1f}/{forward_rms_5s:.1f} cm   "
            f"a2-forward 5s mean={forward_weight_5s:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", help="specific trace; default follows latest")
    parser.add_argument(
        "--log-directory",
        default=str(Path(__file__).resolve().parents[1] / "calibration_logs"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--overhead-bag",
        help=(
            "pool-top ROS2 SQLite bag or directory containing *.db3; "
            "used only for target absolute-speed visualization"
        ),
    )
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--refresh-ms", type=int, default=200)
    parser.add_argument("--save-png", help="optional snapshot path")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--forward-only", action="store_true", help="show only forward error and fusion weight in a vertical layout")
    parser.add_argument("--error-norm-only", action="store_true", help="show only estimated 3D error norm and fusion weight in a vertical layout")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_sec <= 0.0 or args.refresh_ms <= 0:
        raise SystemExit("window-sec and refresh-ms must be positive")
    if args.forward_only and args.error_norm_only:
        raise SystemExit("--forward-only and --error-norm-only are mutually exclusive")
    reference = load_default_reference(args.config)
    overhead_source = (
        None
        if args.overhead_bag is None
        else OverheadTargetVelocitySource(Path(args.overhead_bag).resolve())
    )
    follower = JsonlTraceFollower(
        args.trace_jsonl,
        args.log_directory,
        reference,
        overhead_source=overhead_source,
    )
    plot = LivePositionErrorPlot(
        follower,
        window_s=args.window_sec,
        forward_only=args.forward_only,
        error_norm_only=args.error_norm_only,
    )
    plot.refresh()
    if args.snapshot_only:
        if not args.save_png:
            raise SystemExit("--snapshot-only requires --save-png")
        plot.figure.savefig(args.save_png, dpi=150)
        return 0

    def update(_frame):
        plot.refresh()
        if args.save_png:
            plot.figure.savefig(args.save_png, dpi=120)
        return ()

    animation = FuncAnimation(
        plot.figure,
        update,
        interval=args.refresh_ms,
        cache_frame_data=False,
    )
    plot.figure._finesub_animation = animation  # Keep the timer alive.
    manager = plt.get_current_fig_manager()
    manager.set_window_title("FineSUB MPC Position Error — Live")
    plt.show(block=False)
    window = getattr(manager, "window", None)
    if window is not None:
        def place_window() -> None:
            screen = window.screen()
            available = screen.availableGeometry()
            width = min(1200, max(800, available.width() - 80))
            height = min(800, max(600, available.height() - 80))
            window.setGeometry(
                available.x() + 40,
                available.y() + 40,
                width,
                height,
            )
            window.raise_()
            window.activateWindow()

        from matplotlib.backends.qt_compat import QtCore

        QtCore.QTimer.singleShot(500, place_window)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
