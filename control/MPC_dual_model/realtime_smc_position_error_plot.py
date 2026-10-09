"""Small live SMC position-error plot for FineSUB JSONL traces."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import time
from typing import Iterable

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

from .realtime_position_error_plot import (
    AXIS_COLORS,
    AXIS_NAMES,
    ERROR_REFERENCE_LEVELS_CM,
    JsonlTraceFollower,
    PositionErrorSample,
    load_default_reference,
)


def load_smc_reference(config_path: str | Path) -> np.ndarray:
    """Load the SMC profile reference, or the inherited MPC reference."""

    import json

    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    smc_parameters = config.get("smc_parameters")
    if isinstance(smc_parameters, dict) and "reference_position" in smc_parameters:
        reference = np.asarray(smc_parameters["reference_position"], dtype=float)
        reference = reference.reshape(-1)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                "smc_parameters.reference_position must be a finite 3-vector"
            )
        return reference
    return load_default_reference(config_path)


class LiveSMCPositionErrorPlot:
    """MPC-style four-panel error figure; no camera or ROS subscription."""

    def __init__(
        self,
        follower: JsonlTraceFollower,
        *,
        window_s: float = 60.0,
        maximum_samples: int = 12000,
    ) -> None:
        self.follower = follower
        self.window_s = float(window_s)
        self.samples: deque[PositionErrorSample] = deque(
            maxlen=int(maximum_samples)
        )
        self.figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
        self.axes = tuple(axes.ravel())
        self.measured_lines = []
        self.estimated_lines = []

        for axis_index, axis in enumerate(self.axes[:3]):
            measured, = axis.plot(
                [], [], "--", color="0.60", linewidth=1.0, label="Measured"
            )
            estimated, = axis.plot(
                [], [], color=AXIS_COLORS[axis_index], linewidth=1.8,
                label="Kalman estimate",
            )
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            for level_cm in ERROR_REFERENCE_LEVELS_CM:
                axis.axhline(
                    level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                    label=f"±{level_cm:g} cm" if axis_index == 0 else None,
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
        self.norm_line, = norm_axis.plot(
            [], [], color="tab:purple", linewidth=2.0, label="Estimated 3D norm"
        )
        norm_axis.axhline(5.0, color="tab:orange", linestyle=":", linewidth=0.9,
                          label="5 cm")
        norm_axis.axhline(10.0, color="tab:purple", linestyle=":", linewidth=0.9,
                          label="10 cm")
        norm_axis.set_title("Estimated 3D position-error norm")
        norm_axis.set_ylabel("Error (cm)")
        norm_axis.grid(True, alpha=0.25)
        norm_axis.legend(loc="upper right")
        for axis in self.axes:
            axis.set_xlabel("Time in current SMC trace (s)")
        self.status_text = self.figure.suptitle("Waiting for SMC control updates")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    def _visible_samples(self) -> list[PositionErrorSample]:
        if not self.samples:
            return []
        latest = self.samples[-1].time_s
        minimum = latest - self.window_s
        return [sample for sample in self.samples if sample.time_s >= minimum]

    @staticmethod
    def _symmetric_limit(values: np.ndarray, minimum: float = 10.0) -> float:
        finite = np.abs(np.asarray(values, dtype=float))
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return float(minimum)
        return max(float(minimum), float(np.percentile(finite, 99.0)) * 1.2)

    def refresh(self) -> None:
        switched, new_samples = self.follower.poll()
        if switched:
            self.samples.clear()
        self.samples.extend(new_samples)
        visible = self._visible_samples()
        if not visible:
            selected = self.follower.path
            name = "no trace" if selected is None else selected.name
            self.status_text.set_text(f"Waiting for SMC control updates — {name}")
            return

        time0 = self.samples[0].time_s
        times = np.asarray(
            [sample.time_s - time0 for sample in visible], dtype=float
        )
        measured_cm = 100.0 * np.vstack(
            [sample.measured_error_m for sample in visible]
        )
        estimated_cm = 100.0 * np.vstack(
            [sample.estimated_error_m for sample in visible]
        )
        for axis_index, axis in enumerate(self.axes[:3]):
            self.measured_lines[axis_index].set_data(
                times, measured_cm[:, axis_index]
            )
            self.estimated_lines[axis_index].set_data(
                times, estimated_cm[:, axis_index]
            )
            limit = self._symmetric_limit(
                np.concatenate((measured_cm[:, axis_index], estimated_cm[:, axis_index]))
            )
            axis.set_ylim(-limit, limit)

        norms_cm = np.linalg.norm(estimated_cm, axis=1)
        self.norm_line.set_data(times, norms_cm)
        self.axes[3].set_ylim(
            0.0, max(10.0, float(np.percentile(norms_cm, 99.0)) * 1.2)
        )
        left = max(float(times[-1]) - self.window_s, float(times[0]))
        right = max(float(times[-1]), left + 1.0)
        for axis in self.axes:
            axis.set_xlim(left, right)

        current = estimated_cm[-1]
        rms = np.sqrt(np.mean(estimated_cm**2, axis=0))
        p95_norm = float(np.percentile(norms_cm, 95.0))
        sample_age_s = max(0.0, time.monotonic() - self.samples[-1].time_s)
        live_status = (
            "LIVE" if sample_age_s <= 1.0 else f"NO NEW SMC DATA {sample_age_s:.1f}s"
        )
        trace_name = self.follower.path.name if self.follower.path else "unknown"
        self.status_text.set_text(
            f"{live_status} — {trace_name}\n"
            f"Current FRD error=[{current[0]:+.1f}, {current[1]:+.1f}, {current[2]:+.1f}] cm  "
            f"RMS=[{rms[0]:.1f}, {rms[1]:.1f}, {rms[2]:.1f}] cm  "
            f"3D P95={p95_norm:.1f} cm"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-jsonl",
        help="specific SMC trace; default follows latest smc_*.jsonl",
    )
    parser.add_argument(
        "--log-directory",
        default=str(Path(__file__).resolve().parents[1] / "calibration_logs"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("finesub_v4pro1_smc.json")),
    )
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--refresh-ms", type=int, default=200)
    parser.add_argument("--save-png", help="optional snapshot path")
    parser.add_argument("--snapshot-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_sec <= 0.0 or args.refresh_ms <= 0:
        raise SystemExit("window-sec and refresh-ms must be positive")
    reference = load_smc_reference(args.config)
    follower = JsonlTraceFollower(
        args.trace_jsonl,
        args.log_directory,
        reference,
        trace_pattern="smc_*.jsonl",
    )
    plot = LiveSMCPositionErrorPlot(follower, window_s=args.window_sec)
    plot.refresh()
    if args.snapshot_only:
        if not args.save_png:
            raise SystemExit("--snapshot-only requires --save-png")
        plot.figure.savefig(args.save_png, dpi=150)
        return 0

    from PyQt6 import QtWidgets

    from .realtime_position_error_window import PositionErrorWindow

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PositionErrorWindow(plot, args.refresh_ms)
    window.setWindowTitle("FineSUB SMC Position Error — Live")
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
