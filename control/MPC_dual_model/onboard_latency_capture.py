"""Record host arrival timestamps and a simple stereo target-stem trace."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


CSV_FIELDS = (
    "host_time_utc",
    "host_monotonic_s",
    "gst_pts_s",
    "frame_index",
    "image_width_px",
    "image_height_px",
    "left_stem_x_px",
    "left_stem_y_px",
    "left_stem_width_px",
    "left_stem_height_px",
    "left_stem_area_px",
    "right_stem_x_px",
    "right_stem_y_px",
    "right_stem_width_px",
    "right_stem_height_px",
    "right_stem_area_px",
)


def _stem_from_mask(mask: np.ndarray) -> tuple[float, float, int, int, int]:
    """Return the strongest slender vertical component in ``mask``."""

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[tuple[int, int]] = []
    for index in range(1, count):
        _, _, width, height, area = (int(value) for value in stats[index])
        if height >= 45 and height > 1.8 * width and area >= 250:
            candidates.append((height * area, index))
    if not candidates:
        return math.nan, math.nan, 0, 0, 0
    _, index = max(candidates)
    _, _, width, height, area = (int(value) for value in stats[index])
    x, y = (float(value) for value in centroids[index])
    return x, y, width, height, area


def _stem(eye: np.ndarray) -> tuple[float, float, int, int, int]:
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
    mask = np.asarray((gray < 85) & (hsv[:, :, 1] > 35), dtype=np.uint8) * 255
    mask[:120, :] = 0
    mask[:, :20] = 0
    mask[:, -20:] = 0
    result = _stem_from_mask(mask)
    if math.isfinite(result[0]):
        return result

    # Some target stems are neutral black rather than saturated dark paint.
    # Use an achromatic fallback only when the original conservative detector
    # found nothing; the same lower-frame and edge masks still reject most of
    # the target board and the vertical measuring tapes.
    fallback = np.asarray(gray < 85, dtype=np.uint8) * 255
    fallback[:120, :] = 0
    fallback[:, :20] = 0
    fallback[:, -20:] = 0
    return _stem_from_mask(fallback)


def capture(
    csv_path: Path,
    duration_s: float,
    port: int,
    snapshot_path: Path | None = None,
) -> None:
    Gst.init(None)
    description = (
        f"udpsrc port={port} caps=application/x-rtp,media=video,"
        "encoding-name=H264,payload=96 ! rtpjitterbuffer latency=60 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true sync=false"
    )
    pipeline = Gst.parse_launch(description)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    frame_index = 0
    best_frame: np.ndarray | None = None
    best_detected_eyes = -1
    best_area = -1
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while time.monotonic() - start < duration_s:
                sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
                if sample is None:
                    continue
                arrival = time.monotonic()
                buffer = sample.get_buffer()
                structure = sample.get_caps().get_structure(0)
                width = int(structure.get_value("width"))
                height = int(structure.get_value("height"))
                ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                try:
                    frame = np.frombuffer(mapped.data, np.uint8).reshape(
                        height, width, 3
                    ).copy()
                finally:
                    buffer.unmap(mapped)
                half = width // 2
                left = _stem(frame[:, :half])
                right = _stem(frame[:, half:])
                detected_eyes = int(math.isfinite(left[0])) + int(
                    math.isfinite(right[0])
                )
                detected_area = left[4] + right[4]
                if (detected_eyes, detected_area) > (
                    best_detected_eyes,
                    best_area,
                ):
                    best_frame = frame.copy()
                    best_detected_eyes = detected_eyes
                    best_area = detected_area
                writer.writerow(
                    {
                        "host_time_utc": datetime.now(timezone.utc).isoformat(),
                        "host_monotonic_s": f"{arrival:.9f}",
                        "gst_pts_s": f"{buffer.pts / Gst.SECOND:.9f}",
                        "frame_index": frame_index,
                        "image_width_px": width,
                        "image_height_px": height,
                        "left_stem_x_px": f"{left[0]:.6f}",
                        "left_stem_y_px": f"{left[1]:.6f}",
                        "left_stem_width_px": left[2],
                        "left_stem_height_px": left[3],
                        "left_stem_area_px": left[4],
                        "right_stem_x_px": f"{right[0]:.6f}",
                        "right_stem_y_px": f"{right[1]:.6f}",
                        "right_stem_width_px": right[2],
                        "right_stem_height_px": right[3],
                        "right_stem_area_px": right[4],
                    }
                )
                frame_index += 1
    finally:
        pipeline.set_state(Gst.State.NULL)
    if snapshot_path is not None and best_frame is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(snapshot_path), best_frame):
            raise RuntimeError(f"failed to write snapshot: {snapshot_path}")
    print({"csv": str(csv_path), "duration_s": time.monotonic() - start, "frames": frame_index})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    capture(args.csv, args.duration, args.port, args.snapshot)


if __name__ == "__main__":
    main()
