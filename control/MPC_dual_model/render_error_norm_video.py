"""Render a time-aligned 3D position-error norm and fusion-weight video.

This is offline visualization only; it does not alter or run the controller.
Run with the tracking_depth uv environment, which provides OpenCV and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


WIDTH, HEIGHT = 1280, 720
RED = (40, 40, 210)
GRAY = (190, 190, 190)
BLACK = (35, 35, 35)


def load_samples(trace: Path, config: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with config.open(encoding="utf-8") as stream:
        parameters = json.load(stream)["experimental_auto"]["active_mpc_parameters"]
    default_reference = np.asarray(parameters["controller"]["reference_position"], dtype=float)
    rows = []
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") != "control_update":
                continue
            reference = record.get("reference_position_body_frd_m")
            reference = default_reference if reference is None else np.asarray(reference, dtype=float)
            state = np.asarray(record.get("estimated_state"), dtype=float)
            weight = np.asarray(record.get("model1_weight"), dtype=float)
            timestamp = float(record["host_time_s"])
            if reference.shape != (3,) or state.size < 3 or weight.shape != (3,):
                continue
            values = np.concatenate((reference, state[:3], weight, [timestamp]))
            if not np.all(np.isfinite(values)):
                continue
            # The same sign/reference convention as realtime_position_error_plot.py.
            norm_cm = 100.0 * float(np.linalg.norm(state[:3] - reference))
            rows.append((timestamp, norm_cm, float(weight[0])))
    if not rows:
        raise ValueError(f"no valid control_update samples in {trace}")
    samples = np.asarray(rows, dtype=float)
    if np.any(np.diff(samples[:, 0]) < 0):
        raise ValueError("trace timestamps are not monotonic")
    return samples[:, 0], samples[:, 1], samples[:, 2]


def label(canvas: np.ndarray, text: str, point: tuple[int, int], *, scale: float = 0.65,
          color: tuple[int, int, int] = BLACK, thickness: int = 1) -> None:
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def chart(canvas: np.ndarray, *, title: str, rect: tuple[int, int, int, int],
          times: np.ndarray, values: np.ndarray, now: float, y_max: float,
          ticks: tuple[float, ...], color: tuple[int, int, int]) -> None:
    left, top, right, bottom = rect
    label(canvas, title, (left, top - 14), scale=0.75, thickness=2)
    cv2.rectangle(canvas, (left, top), (right, bottom), GRAY, 1)
    x_min = max(0.0, now - 60.0)
    x_max = max(5.0, now)
    for tick in ticks:
        y = int(round(bottom - tick / y_max * (bottom - top)))
        if top <= y <= bottom:
            cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
            label(canvas, f"{tick:g}", (left - 49, y + 5), scale=0.50)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(left + fraction * (right - left)))
        cv2.line(canvas, (x, top), (x, bottom), (232, 232, 232), 1)
        label(canvas, f"{x_min + fraction * (x_max - x_min):.0f}",
              (x - 10, bottom + 22), scale=0.48)
    valid = (times >= x_min) & (times <= now)
    if np.count_nonzero(valid) >= 2:
        x = left + (times[valid] - x_min) / (x_max - x_min) * (right - left)
        y = bottom - np.clip(values[valid] / y_max, 0.0, 1.0) * (bottom - top)
        points = np.column_stack((x, y)).round().astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), 4, color, -1, cv2.LINE_AA)


def render(trace: Path, config: Path, output: Path, frame_csv: Path,
           start_unix_s: float, frames: int, fps: int) -> None:
    sample_times, norms_cm, weights = load_samples(trace, config)
    relative_times = sample_times - start_unix_s
    # Fixed scale prevents the plot from changing height during the recording.
    norm_y_max = max(10.0, float(np.max(norms_cm)) * 1.1)
    norm_y_max = math.ceil(norm_y_max / 5.0) * 5.0
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with frame_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("frame_index", "frame_unix_s", "latest_sample_unix_s",
                             "sample_age_s", "sample_status",
                             "estimated_3d_error_norm_cm", "forward_model2_weight"))
            for index in range(frames):
                now = index / fps
                last = int(np.searchsorted(relative_times, now, side="right"))
                canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
                label(canvas, "3D tracking error norm and fusion weight", (65, 38),
                      scale=0.90, thickness=2)
                if last:
                    sample_age = max(0.0, start_unix_s + now - sample_times[last - 1])
                    status = "LIVE" if sample_age <= 1.0 else f"NO NEW MPC DATA ({sample_age:.1f} s old)"
                    label(canvas, f"{status}   t={now:.2f} s   |e|={norms_cm[last - 1]:.1f} cm   "
                          f"moving model-2 weight={weights[last - 1]:.3f}",
                          (65, 71), scale=0.58)
                else:
                    sample_age = None
                    status = "WAITING"
                    label(canvas, f"t={now:.2f} s   Waiting for first MPC update",
                          (65, 71), scale=0.58)
                chart(canvas, title="Estimated 3D position-error norm (cm)",
                      rect=(82, 124, 1230, 376), times=relative_times[:last],
                      values=norms_cm[:last], now=now, y_max=norm_y_max,
                      ticks=tuple(np.linspace(0.0, norm_y_max, 5)), color=RED)
                chart(canvas, title="Forward model-2 (moving) fusion weight",
                      rect=(82, 472, 1230, 674), times=relative_times[:last],
                      values=weights[:last], now=now, y_max=1.0,
                      ticks=(0.0, 0.25, 0.5, 0.75, 1.0), color=RED)
                writer.writerow((index, f"{start_unix_s + now:.6f}",
                                 f"{sample_times[last - 1]:.6f}" if last else "",
                                 f"{sample_age:.6f}" if sample_age is not None else "",
                                 status,
                                 f"{norms_cm[last - 1]:.6f}" if last else "",
                                 f"{weights[last - 1]:.6f}" if last else ""))
                assert encoder.stdin is not None
                encoder.stdin.write(canvas.tobytes())
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode plot video")
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--start-unix-s", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.frames <= 0 or args.fps <= 0:
        parser.error("frames and fps must be positive")
    render(args.trace, args.config, args.output, args.frame_csv,
           args.start_unix_s, args.frames, args.fps)


if __name__ == "__main__":
    main()
