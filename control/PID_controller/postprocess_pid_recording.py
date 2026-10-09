"""Verify, align and archive a stopped FineSUB PID three-view recording.

The X11 plot recording may start after PID. A trace-derived norm-only plot
covers the complete controller interval; the original X11 recording is kept.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import cv2

from dual_vision_pid_runtime import render_3d_error_norm_window


FPS = 20
SOURCES = (("pool_top.mkv", "pool_top.log", "pool"),
           ("stereo_capture.mp4", "stereo_capture.log", "stereo"),
           ("plot_capture.mp4", "plot_capture.log", "plot_x11"))


def run(command: list[str], log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, check=True)
    else:
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def probe(path: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    streams = json.loads(output.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"video stream missing: {path}")
    return streams[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_timestamps(recording: Path, name: str, log_name: str,
                      label: str, clock_offset: float) -> dict:
    video = recording / name
    info = probe(video)
    if info["avg_frame_rate"] != "20/1":
        raise ValueError(f"source is not 20 fps: {name}: {info}")
    log = (recording / log_name).read_text(errors="replace")
    start_match = re.search(r"Duration: N/A, start: ([0-9.]+)", log)
    if start_match is None:
        raise ValueError(f"missing input timestamp in {log_name}")
    base = float(start_match.group(1)) + (clock_offset if label == "pool" else 0.0)
    frames = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+)", log)
    if not frames:
        raise ValueError(f"missing frame timestamps in {log_name}")
    csv_path = recording / f"{label}_capture_frame_times.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capture_frame_index", "input_pts", "input_pts_time_s", "host_unix_s"))
        for index, pts, seconds in frames:
            writer.writerow((index, pts, seconds, f"{base + float(seconds):.9f}"))
    last_time = base + float(frames[-1][2])
    return {"video": name, "log": log_name, "start_unix_s": base,
            "last_captured_frame_unix_s": last_time,
            "captured_frames": len(frames), "capture_times_csv": csv_path.name,
            "probe": info}


def disarm_summary(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fresh = [row for row in rows if row["telemetry_fresh"] == "1"]
    if len(fresh) < 20:
        raise ValueError("insufficient fresh post-run telemetry")
    # The first fresh packet may precede this diagnostic client's echoed
    # disarm command; require confirmation thereafter, not retroactively.
    confirmed_start = next((i for i, row in enumerate(fresh)
                            if row["session_confirmed"] == "1"), None)
    if confirmed_start is None or len(fresh) - confirmed_start < 20:
        raise ValueError("insufficient session-confirmed post-run telemetry")
    if any(row["telemetry_armed"] != "0" or row["reject_flags"] != "0"
           for row in fresh) or any(row["session_confirmed"] != "1"
                                for row in fresh[confirmed_start:]):
        raise ValueError("post-run disarm telemetry not clean")
    max_motor = max(abs(float(value)) for row in fresh
                    for value in json.loads(row["applied_motor_throttle_m1_m8"]))
    max_rpm = max(abs(float(value)) for row in fresh
                  for value in json.loads(row["motor_rpm_m1_m8"]))
    if max_motor != 0 or max_rpm != 0:
        raise ValueError("post-run motor execution feedback is not zero")
    return {"samples": len(fresh), "armed_rows": 0,
            "max_abs_applied_throttle": max_motor, "max_abs_rpm": max_rpm}


def finish(recording: Path) -> Path:
    recording = recording.resolve()
    trace = [json.loads(line) for line in (recording / "pid_trace.jsonl").open()]
    starts = [row for row in trace if row.get("event") == "start"]
    stops = [row for row in trace if row.get("event") == "stop"]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError("PID trace needs exactly one start and one stop")
    start, stop = starts[0], stops[0]
    start_time, stop_time = float(start["host_time_s"]), float(stop["host_time_s"])
    if stop_time <= start_time:
        raise ValueError("invalid PID interval")
    cycles = [row for row in trace if row.get("event") == "control_cycle"]
    if not cycles:
        raise ValueError("empty PID trace")
    telemetry = disarm_summary(recording / "postrun_disarmed.csv")
    offset = start_time - float(start["host_monotonic_s"])
    sources = {label: source_timestamps(recording, name, log, label, offset)
               for name, log, label in SOURCES}
    for label in ("pool",):
        source = sources[label]
        if source["start_unix_s"] > start_time or source["last_captured_frame_unix_s"] < stop_time:
            raise ValueError(f"{label} does not cover complete PID interval")
    stereo_frames = [json.loads(line) for line in
                     (recording / "stereo_frames.jsonl").open()]
    stereo_times = [float(row["frame_ts_mean"]) for row in stereo_frames]
    stereo_view_info = probe(recording / "stereo_view.mp4")
    if (len(stereo_times) != int(stereo_view_info.get("nb_frames", -1))
            or stereo_times[0] > start_time
            or stereo_times[-1] < stop_time):
        raise ValueError("stereo view/frame timestamps do not cover PID interval")
    plot_gap = max(0.0, sources["plot_x11"]["start_unix_s"] - start_time)
    frame_count = math.ceil((stop_time - start_time) * FPS)
    aligned = recording / "aligned"
    aligned.mkdir(exist_ok=True)

    # Reconstruct the *display* from the recorded PID errors. The raw X11
    # capture stays untouched and the manifest explicitly records its gap.
    render_path = recording / "plot_from_pid_trace_20fps.mp4"
    writer = cv2.VideoWriter(str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (900, 520))
    if not writer.isOpened():
        raise RuntimeError("cannot create trace-derived plot video")
    white_chart_path = recording / "pid_full_error_chart.png"
    white_chart = cv2.imread(str(white_chart_path)) if white_chart_path.exists() else None
    if white_chart is not None and white_chart.shape[:2] != (520, 900):
        raise ValueError("white PID chart must be 900x520 pixels")
    errors: deque[float] = deque(maxlen=240)
    cycle_index = 0
    try:
        with (recording / "plot_trace_frame_times.csv").open("w", newline="") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(("frame_index", "host_unix_s"))
            for frame in range(frame_count):
                now = start_time + frame / FPS
                while cycle_index < len(cycles) and float(cycles[cycle_index]["host_time_s"]) <= now:
                    value = cycles[cycle_index].get("error_norm_m")
                    if value is not None:
                        errors.append(100.0 * float(value))
                    cycle_index += 1
                if white_chart is not None:
                    image = white_chart.copy()
                    cursor_x = round(126 + 747 * frame / max(frame_count - 1, 1))
                    cv2.line(image, (cursor_x, 73), (cursor_x, 406),
                             (70, 70, 220), 1, cv2.LINE_AA)
                else:
                    image = render_3d_error_norm_window(cv2, errors)
                    cv2.putText(image, f"PID trace replay  t={frame / FPS:.2f}s", (60, 505),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (185, 185, 185), 1, cv2.LINE_AA)
                writer.write(image)
                csv_writer.writerow((frame, f"{now:.9f}"))
    finally:
        writer.release()
    if int(probe(render_path).get("nb_frames", -1)) != frame_count:
        raise ValueError("trace plot frame count mismatch")

    for label in ("pool",):
        source = sources[label]
        offset_s = start_time - source["start_unix_s"]
        output = aligned / f"{label}_20fps.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-ss", f"{offset_s:.9f}",
             "-i", str(recording / source["video"]), "-vf", "fps=20,scale=1280:720,setsar=1",
             "-frames:v", str(frame_count), "-r", str(FPS), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
             "-y", str(output)], recording / f"align_{label}.log")
        info = probe(output)
        if int(info.get("nb_frames", -1)) != frame_count or info["avg_frame_rate"] != "20/1":
            raise ValueError(f"incorrect aligned {label} video: {info}")
        source["aligned_video"] = str(output.relative_to(recording))
        source["alignment_offset_s"] = offset_s

    # The X11 stereo window can start after PID. The producer's own view
    # video has one frame per stereo_frames.jsonl row, so align by the actual
    # acquisition clock rather than its nominal 20 fps container timestamps.
    stereo_aligned = aligned / "stereo_20fps.mp4"
    stereo_capture = cv2.VideoCapture(str(recording / "stereo_view.mp4"))
    stereo_writer = cv2.VideoWriter(str(stereo_aligned),
                                    cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                    (1280, 720))
    if not stereo_capture.isOpened() or not stereo_writer.isOpened():
        raise RuntimeError("cannot open stereo view for timestamp alignment")
    source_index = -1
    current_frame = None
    try:
        with (recording / "stereo_view_alignment.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("aligned_frame_index", "aligned_host_unix_s",
                             "source_frame_index", "source_host_unix_s"))
            for frame_index in range(frame_count):
                target_time = start_time + frame_index / FPS
                right = bisect_left(stereo_times, target_time)
                wanted = min((right - 1, right),
                             key=lambda index: abs(stereo_times[index] - target_time)
                             if 0 <= index < len(stereo_times) else float("inf"))
                while source_index < wanted:
                    ok, current_frame = stereo_capture.read()
                    if not ok:
                        raise ValueError("stereo view ended before indexed frame")
                    source_index += 1
                stereo_writer.write(cv2.resize(current_frame, (1280, 720)))
                writer.writerow((frame_index, f"{target_time:.9f}", wanted,
                                 f"{stereo_times[wanted]:.9f}"))
    finally:
        stereo_writer.release()
        stereo_capture.release()
    if int(probe(stereo_aligned).get("nb_frames", -1)) != frame_count:
        raise ValueError("stereo aligned frame count mismatch")
    sources["stereo"]["aligned_video"] = str(stereo_aligned.relative_to(recording))
    sources["stereo"]["alignment_method"] = "nearest producer frame_ts_mean"
    sources["stereo"]["alignment_csv"] = "stereo_view_alignment.csv"

    composite = recording / "pool_stereo_pid_3d_error_norm_vertical_20fps.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "warning",
         "-i", str(aligned / "pool_20fps.mp4"),
         "-i", str(aligned / "stereo_20fps.mp4"),
         "-i", str(render_path),
         "-filter_complex", "[2:v]scale=1280:720[p];[0:v][1:v][p]vstack=inputs=3[v]",
         "-map", "[v]", "-frames:v", str(frame_count), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", "-y", str(composite)],
        recording / "composite.log")
    info = probe(composite)
    if int(info.get("nb_frames", -1)) != frame_count or (info["width"], info["height"]) != (1280, 2160):
        raise ValueError(f"invalid composite: {info}")

    manifest_path = recording / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"status": "stopped_disarmed_verified_archived",
                     "controller_stop_unix_s": stop_time,
                     "frame_zero_unix_s": start_time,
                     "frame_count": frame_count,
                     "frame_k_unix_s_formula": "frame_zero_unix_s + k / 20",
                     "source_video_timestamps": sources,
                     "raw_plot_start_gap_s": plot_gap,
                     "plot_alignment_note": "X11 plot starts late; full-interval third panel is reconstructed from PID trace, not raw screen capture",
                     "derived_plot_video": render_path.name,
                     "derived_plot_style": "white full-run norm chart with current-time cursor" if white_chart is not None else "dark rolling norm chart",
                     "composite_video": composite.name,
                     "pid_control_cycles": len(cycles),
                     "pid_armed_cycles": sum(bool(row.get("armed")) for row in cycles),
                     "postrun_disarmed_summary": telemetry})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checksums = {str(path.relative_to(recording)): sha256(path)
                 for path in sorted(recording.rglob("*"))
                 if path.is_file() and path.name != "sha256_manifest.json"}
    (recording / "sha256_manifest.json").write_text(json.dumps(checksums, indent=2) + "\n")
    archive = recording.parent / f"{recording.name}.tar.zst"
    run(["tar", "-I", "zstd -T2 -3", "-cf", str(archive), "-C", str(recording.parent), recording.name])
    print(json.dumps({"archive": str(archive), "frames": frame_count,
                      "armed_cycles": manifest["pid_armed_cycles"],
                      "raw_plot_gap_s": plot_gap}, ensure_ascii=False), flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    finish(parser.parse_args().recording)
