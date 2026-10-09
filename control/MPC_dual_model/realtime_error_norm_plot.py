"""Live two-panel plot for a single FineSUB experiment trace.

The upper panel is the Euclidean norm of the estimated three-axis position
error.  The lower panel is the article's moving model-2 weight, stored in the
controller trace as ``model1_weight``.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .realtime_position_error_plot import JsonlTraceFollower, load_default_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--refresh-ms", type=int, default=50)
    parser.add_argument("--save-png", type=Path)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    if args.window_sec <= 0 or args.refresh_ms <= 0:
        parser.error("window-sec and refresh-ms must be positive")
    if args.snapshot_only and args.save_png is None:
        parser.error("--snapshot-only requires --save-png")

    follower = JsonlTraceFollower(
        str(args.trace_jsonl), str(args.trace_jsonl.parent),
        load_default_reference(args.config),
    )
    samples = deque(maxlen=12000)
    figure, (error_axis, weight_axis) = plt.subplots(
        2, 1, figsize=(12.8, 7.2), sharex=True,
    )
    error_line, = error_axis.plot(
        [], [], color="tab:purple", linewidth=2, label="3D error norm",
    )
    weight_line, = weight_axis.plot(
        [], [], color="tab:red", linewidth=2, label="Moving model-2 weight",
    )
    for level in (5.0, 10.0):
        error_axis.axhline(level, color="0.7", linestyle=":", linewidth=1)
    error_axis.set_ylabel("3D position error norm (cm)")
    weight_axis.set_ylabel("Weight")
    weight_axis.set_xlabel("Time in current trace (s)")
    weight_axis.set_ylim(-0.03, 1.03)
    for axis in (error_axis, weight_axis):
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    status = figure.suptitle("Waiting for MPC control updates")
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    def refresh(_frame: int) -> tuple:
        switched, new_samples = follower.poll()
        if switched:
            samples.clear()
        samples.extend(new_samples)
        if not samples:
            return error_line, weight_line, status
        start = samples[0].time_s
        latest = samples[-1].time_s
        visible = [sample for sample in samples if sample.time_s >= latest - args.window_sec]
        times = np.asarray([sample.time_s - start for sample in visible])
        errors = 100.0 * np.asarray([
            np.linalg.norm(sample.estimated_error_m) for sample in visible
        ])
        weights = np.asarray([sample.model1_weight[0] for sample in visible])
        error_line.set_data(times, errors)
        weight_line.set_data(times, weights)
        error_axis.set_ylim(0, max(10.0, float(np.nanmax(errors)) * 1.2))
        left = max(0.0, float(times[-1]) - args.window_sec)
        right = max(left + 1.0, float(times[-1]))
        error_axis.set_xlim(left, right)
        age = max(0.0, time.monotonic() - samples[-1].time_s)
        freshness = "LIVE" if age <= 1.0 else f"NO NEW MPC DATA {age:.1f}s"
        status.set_text(
            f"{freshness}  |  3D error norm {errors[-1]:.1f} cm  |  "
            f"moving model-2 weight {weights[-1]:.3f}"
        )
        return error_line, weight_line, status

    refresh(0)
    if args.snapshot_only:
        figure.savefig(args.save_png, dpi=120)
        plt.close(figure)
        return 0
    animation = FuncAnimation(
        figure, refresh, interval=args.refresh_ms, cache_frame_data=False,
    )
    _ = animation
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
