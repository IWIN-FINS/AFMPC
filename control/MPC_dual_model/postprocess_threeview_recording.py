"""Align, verify, and archive a stopped FineSUB three-view recording."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


FPS = 20
SOURCE_NAMES = (
    ("pool_top.mkv", "pool_top.log", "pool"),
    ("stereo_capture.mp4", "stereo_capture.log", "stereo"),
    ("plot_capture.mp4", "plot_capture.log", "plot"),
)


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
    else:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def input_start_and_capture_csv(log_path: Path, output: Path,
                                clock_offset_s: float) -> tuple[float, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Duration: N/A, start: ([0-9.]+)", text)
    if match is None:
        raise ValueError(f"missing input start timestamp: {log_path}")
    base_unix_s = float(match.group(1)) + clock_offset_s
    rows = re.findall(
        r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+)", text,
    )
    if not rows:
        raise ValueError(f"missing per-frame capture timestamps: {log_path}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capture_frame_index", "input_pts", "input_pts_time_s",
                         "host_unix_s"))
        for index, pts, pts_time in rows:
            writer.writerow((index, pts, pts_time,
                             f"{base_unix_s + float(pts_time):.9f}"))
    print(f"CAPTURE_TIMES {output.name}: {len(rows)}", flush=True)
    return base_unix_s, len(rows)


def encoded_frame_csv(video: Path, output: Path, base_unix_s: float) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0",
         str(video)],
        check=True, capture_output=True, text=True,
    )
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("encoded_frame_index", "video_pts_time_s", "host_unix_s"))
        for line in result.stdout.splitlines():
            candidate = line.split(",", 1)[0].strip()
            if not candidate:
                continue
            try:
                pts_time = float(candidate)
            except ValueError:
                continue
            writer.writerow((count, f"{pts_time:.9f}",
                             f"{base_unix_s + pts_time:.9f}"))
            count += 1
    if count == 0:
        raise ValueError(f"no encoded frames: {video}")
    print(f"ENCODED_TIMES {output.name}: {count}", flush=True)
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finish(recording: Path) -> Path:
    recording = recording.resolve()
    state = json.loads((recording / "run_state.json").read_text())
    telemetry = json.loads((recording / "postrun_telemetry_summary.json").read_text())
    if state.get("status") not in {
        "stopped_disarmed_pending_package", "stopped_disarmed_pending_postprocess",
    }:
        raise ValueError("recording is not marked as stopped and disarmed")
    if telemetry["samples"] < 20 or telemetry["armed_rows"] != 0:
        raise ValueError("post-run disarm telemetry check failed")
    if telemetry["max_abs_applied_throttle"] != 0:
        raise ValueError("post-run motor throttle is not zero")
    trace = [json.loads(line) for line in (recording / "mpc_trace.jsonl").read_text().splitlines()]
    starts = [row for row in trace if row.get("event") == "start"]
    if len(starts) != 1:
        raise ValueError("expected one controller start in trace")
    start = starts[0]
    start_unix_s = float(start["host_time_s"])
    clock_offset_s = start_unix_s - float(start["host_monotonic_s"])
    stop_unix_s = float(state["stop_requested_unix_s"])
    if stop_unix_s <= start_unix_s:
        raise ValueError("stop timestamp precedes controller start")
    frame_count = math.ceil((stop_unix_s - start_unix_s) * FPS)
    updates = [row for row in trace if row.get("event") == "control_update"]
    if not updates or not all(row.get("model1_weight") == [1.0] * 3 for row in updates):
        raise ValueError("fixed moving-model weight is absent from MPC trace")

    aligned = recording / "aligned"
    aligned.mkdir(exist_ok=True)
    source_info = {}
    for video_name, log_name, label in SOURCE_NAMES:
        video = recording / video_name
        offset = clock_offset_s if label == "pool" else 0.0
        capture_start_s, captured = input_start_and_capture_csv(
            recording / log_name, recording / f"{label}_capture_frame_times.csv", offset,
        )
        encoded = encoded_frame_csv(
            video, recording / f"{label}_encoded_frame_times.csv", capture_start_s,
        )
        input_probe = probe(video)
        if input_probe["avg_frame_rate"] != "20/1":
            raise ValueError(f"input is not 20 fps: {video}")
        seek_s = start_unix_s - capture_start_s
        if seek_s < 0:
            raise ValueError(f"capture started after controller: {video}")
        clip = aligned / f"{label}_20fps.mp4"
        filter_spec = (
            f"trim=start={seek_s:.9f},"
            f"setpts=PTS-{seek_s:.9f}/TB,"
            "fps=fps=20:start_time=0,"
            "scale=1280:720:flags=bicubic:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
        run_command(
            ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", str(video),
             "-vf", filter_spec, "-frames:v", str(frame_count), "-r", str(FPS),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-an", "-y", str(clip)],
            recording / f"align_{label}.log",
        )
        info = probe(clip)
        if int(info["nb_frames"]) != frame_count or info["avg_frame_rate"] != "20/1":
            raise ValueError(f"incorrect aligned clip length/fps: {clip}: {info}")
        source_info[label] = {
            "original_video": video_name,
            "capture_log": log_name,
            "capture_start_unix_s": capture_start_s,
            "capture_frame_count": captured,
            "encoded_frame_count": encoded,
            "aligned_video": str(clip.relative_to(recording)),
            "alignment_offset_s": seek_s,
        }
        print(f"ALIGNED {label}: {frame_count} frames", flush=True)

    composite = recording / "pool_stereo_3d_error_norm_vertical_20fps.mp4"
    run_command(
        ["ffmpeg", "-hide_banner", "-loglevel", "warning",
         "-i", str(aligned / "pool_20fps.mp4"),
         "-i", str(aligned / "stereo_20fps.mp4"),
         "-i", str(aligned / "plot_20fps.mp4"),
         "-filter_complex", "[0:v][1:v][2:v]vstack=inputs=3[v]",
         "-map", "[v]", "-frames:v", str(frame_count), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", "-y", str(composite)],
        recording / "composite.log",
    )
    info = probe(composite)
    if (int(info["nb_frames"]) != frame_count or
            info["avg_frame_rate"] != "20/1" or
            (info["width"], info["height"]) != (1280, 2160)):
        raise ValueError(f"invalid composite: {info}")
    with (aligned / "common_timeline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_index", "host_unix_s"))
        for index in range(frame_count):
            writer.writerow((index, f"{start_unix_s + index / FPS:.9f}"))

    manifest = {
        "status": "operator_stopped_and_disarmed",
        "time_basis": "host Unix UTC seconds",
        "frame_zero_unix_s": start_unix_s,
        "stop_requested_unix_s": stop_unix_s,
        "frame_k_unix_s_formula": "frame_zero_unix_s + k / 20",
        "fps": FPS,
        "frame_count": frame_count,
        "panel_order_top_to_bottom": ["pool_top", "stereo", "3d_error_norm_and_moving_model2_weight"],
        "composite_video": composite.name,
        "source_video_timestamps": source_info,
        "stereo_raw_frame_times": "stereo_frames.jsonl",
        "vision_results": "vision_results.jsonl",
        "mpc_trace": "mpc_trace.jsonl",
        "config_snapshots": ["mpc_config_snapshot.json", "vision_config_snapshot.yaml"],
        "postrun_disarmed_check": "postrun_disarmed.csv",
        "postrun_disarmed_summary": telemetry,
        "mpc_control_updates": len(updates),
        "fixed_article_moving_model_2_weight": 1.0,
    }
    (recording / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    checksums = {}
    for file in sorted(recording.rglob("*")):
        if file.is_file() and file.name != "sha256_manifest.json":
            checksums[str(file.relative_to(recording))] = sha256(file)
    (recording / "sha256_manifest.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
    )
    archive = recording.parent / f"{recording.name}.tar.zst"
    run_command(["tar", "-I", "zstd -T2 -3", "-cf", str(archive),
                 "-C", str(recording.parent), recording.name])
    run_command(["tar", "-I", "zstd", "-tf", str(archive)],
                recording / "archive_contents.log")
    print(f"COMPLETE {recording} archive={archive} frames={frame_count}", flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    args = parser.parse_args()
    for recording in args.recordings:
        finish(recording)


if __name__ == "__main__":
    main()
