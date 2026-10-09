
"""
Demo — stereo video with separate left/right streams + timestamp sync
======================================================================
Real-time fish 3D position estimation with proper dual-stream stereo.

Usage
-----
    # Two independent video files (recommended for real stereo):
    python demo_video.py --left left.mp4 --right right.mp4

    # Two webcams / capture devices:
    python demo_video.py --left 0 --right 1

    # Headless (no GUI, print positions to console):
    python demo_video.py --left left.mp4 --right right.mp4 --no-display

    # Live UDP stereo preview and save the annotated monitor video:
    python demo_video.py --udp-sbs-port 5600 --record-monitor records/

    # Save monitor video at a lower write rate to reduce overhead:
    python demo_video.py --udp-sbs-port 5600 --record-monitor records/ \
        --record-every 3 --record-fps 10

Controls
--------
    q / ESC  — quit
    p        — pause / resume
    s        — save current annotated frame
"""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import warnings

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from depth_estimation.fish_position import from_config_yaml
from depth_estimation.stereo_capture import (
    StereoCapture,
    StereoCaptureSideBySideGst,
)

CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


class DisplayWindow:
    """Realtime preview window with OpenCV-first, tkinter fallback."""

    def __init__(self, enabled: bool, display_scale: float = 1.4):
        self.enabled = enabled
        self.display_scale = max(float(display_scale), 0.1)
        self.backend = None
        self._root = None
        self._label = None
        self._photo = None
        self._quit_requested = False
        self._paused_requested = False
        self._save_requested = False

        if not enabled:
            return

        try:
            cv2.namedWindow("Fish Position Estimation", cv2.WINDOW_NORMAL)
            self.backend = "cv2"
        except cv2.error:
            self._root = tk.Tk()
            self._root.title("Fish Position Estimation")
            self._label = tk.Label(self._root)
            self._label.pack()
            self._root.resizable(True, True)
            self._root.bind("<KeyPress-q>", self._on_quit)
            self._root.bind("<Escape>", self._on_quit)
            self._root.bind("<KeyPress-p>", self._on_pause)
            self._root.bind("<KeyPress-s>", self._on_save)
            self._root.protocol("WM_DELETE_WINDOW", self._on_quit)
            self.backend = "tk"

    def show(self, frame):
        if not self.enabled:
            return
        display = self._scale_frame(frame)
        if self.backend == "cv2":
            cv2.imshow("Fish Position Estimation", display)
            return

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()

    def poll_key(self, paused: bool) -> int:
        if not self.enabled:
            return -1
        if self.backend == "cv2":
            return cv2.waitKey(1 if not paused else 0) & 0xFF

        self._root.update_idletasks()
        self._root.update()
        if self._quit_requested:
            self._quit_requested = False
            return ord("q")
        if self._paused_requested:
            self._paused_requested = False
            return ord("p")
        if self._save_requested:
            self._save_requested = False
            return ord("s")
        time.sleep(0.001 if not paused else 0.02)
        return -1

    def close(self):
        if not self.enabled:
            return
        if self.backend == "cv2":
            cv2.destroyAllWindows()
            return
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def _scale_frame(self, frame):
        if self.display_scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (int(w * self.display_scale), int(h * self.display_scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    def _on_quit(self, _event=None):
        self._quit_requested = True

    def _on_pause(self, _event=None):
        self._paused_requested = True

    def _on_save(self, _event=None):
        self._save_requested = True


def draw_overlay(img: cv2.typing.MatLike, fish_list: list,
                 fps: float, ts_diff_ms: float,
                 *,
                 display_frame_idx: int | None = None,
                 result_frame_idx: int | None = None,
                 result_age_ms: float | None = None,
                 show_timing: bool = True) -> cv2.typing.MatLike:
    """Draw detections, positions, FPS and sync info on the left frame."""
    out = img.copy()
    h, w = out.shape[:2]

    for f in fish_list:
        x1, y1, x2, y2 = [
            int(round(v)) for v in f["bbox"]
        ]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        px, py, pz = f["position"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lbl = f"ID:{f['id']} {f['confidence']:.2f}"
        source = f.get("source")
        if source:
            lbl += f" {source}"
        cv2.putText(out, lbl, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pos_lbl = f"X:{px:+.3f} Y:{py:+.3f} Z:{pz:.3f}m"
        cv2.putText(out, pos_lbl, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

    # HUD: FPS + sync status (top-left)
    sync_color = (0, 255, 0) if ts_diff_ms < 30 else (0, 165, 255)
    if show_timing:
        cv2.putText(out, f"FPS:{fps:.1f}  sync:{ts_diff_ms:.1f}ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # HUD: fish count (top-right)
    cv2.putText(out, f"fish:{len(fish_list)}",
                (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 2)

    if result_frame_idx is not None and result_frame_idx > 0:
        age_text = "n/a"
        if result_age_ms is not None and np.isfinite(result_age_ms):
            age_text = f"{max(result_age_ms, 0.0):.0f}ms"
        frame_text = (
            f"view_f:{display_frame_idx if display_frame_idx is not None else '-'}  "
            f"meas_f:{result_frame_idx}  age:{age_text}"
        )
    else:
        frame_text = (
            f"view_f:{display_frame_idx if display_frame_idx is not None else '-'}  "
            "meas_f:none"
        )
    cv2.putText(out, frame_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    return out


def compose_stereo_preview(left_annotated: cv2.typing.MatLike,
                           right_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Show annotated left view beside the synced right view."""
    right = right_img.copy()
    cv2.putText(right, "RIGHT", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.hconcat([left_annotated, right])


def print_fish_positions(frame_idx: int, fish_list: list[dict],
                         process_ms: float | None = None,
                         processed: bool = False,
                         sync_ms: float | None = None,
                         tracker_state: str | None = None):
    """Print compact 3D positions for the current tracked fish."""
    prefix = (
        f"frame {frame_idx:5d} | "
        f"{'processed' if processed else 'reused':9s}"
    )
    if process_ms is not None:
        prefix += f" | {process_ms:7.1f} ms"
    if sync_ms is not None:
        prefix += f" | sync {sync_ms:6.1f} ms"
    if tracker_state:
        prefix += f" | state {tracker_state}"
    if not fish_list:
        print(f"{prefix} | fish 0")
        return

    print(f"{prefix} | fish {len(fish_list)}")
    for f in fish_list:
        px, py, pz = f["position"]
        filter_mode = str(f.get("depth_filter_mode", ""))
        gain = f.get("depth_gain")
        quality = f.get("depth_quality")
        extra = ""
        if filter_mode:
            extra += f"  filt={filter_mode}"
        if gain is not None:
            extra += f"  K={float(gain):.2f}"
        if quality is not None:
            extra += f"  q={float(quality):.2f}"
        print(
            f"  ID {f['id']:02d} | "
            f"X={px:+.3f} m  Y={py:+.3f} m  Z={pz:+.3f} m  "
            f"conf={f['confidence']:.2f}  src={f.get('source', 'yolo')}"
            f"{extra}"
        )


def _serialize_result_payload(*,
                              frame_idx: int,
                              fish: list[dict],
                              process_ms: float | None,
                              tracker_state: str | None,
                              sframe,
                              result_time: float) -> dict:
    tracks = []
    for item in fish:
        tracks.append({
            "id": int(item["id"]),
            "position": [float(v) for v in item["position"]],
            "confidence": float(item["confidence"]),
            "source": str(item.get("source", "yolo")),
            "raw_center_uv": [
                float(v) for v in item.get("raw_center_uv", [float("nan"), float("nan")])
            ],
            "filtered_center_uv": [
                float(v) for v in item.get("filtered_center_uv", [float("nan"), float("nan")])
            ],
            "depth_valid": bool(item.get("depth_valid", False)),
            "depth_confidence": (
                None if item.get("depth_confidence") is None
                else float(item.get("depth_confidence"))
            ),
            "raw_depth": (
                None if item.get("raw_depth") is None
                else float(item.get("raw_depth"))
            ),
            "z_dot": float(item.get("z_dot", 0.0) or 0.0),
            "depth_gain": float(item.get("depth_gain", 0.0) or 0.0),
            "depth_quality": float(item.get("depth_quality", 0.0) or 0.0),
            "depth_filter_mode": str(item.get("depth_filter_mode", "")),
            "depth_r_t": float(item.get("depth_r_t", 0.0) or 0.0),
            "depth_R_t": float(item.get("depth_R_t", 0.0) or 0.0),
            "depth_nis": (
                None if item.get("depth_nis") is None
                else float(item.get("depth_nis"))
            ),
            "depth_innovation": (
                None if item.get("depth_innovation") is None
                else float(item.get("depth_innovation"))
            ),
            "depth_S_t": (
                None if item.get("depth_S_t") is None
                else float(item.get("depth_S_t"))
            ),
            "depth_P_zz_pred": (
                None if item.get("depth_P_zz_pred") is None
                else float(item.get("depth_P_zz_pred"))
            ),
            "depth_dt_s": float(item.get("depth_dt_s", 0.0) or 0.0),
            "core_px": int(item.get("core_px", 0) or 0),
            "z_iqr": (
                None if item.get("z_iqr") is None
                else float(item.get("z_iqr"))
            ),
            "sep_score": (
                None if item.get("sep_score") is None
                else float(item.get("sep_score"))
            ),
            "bbox": [float(v) for v in item.get("bbox", [])],
        })
    return {
        "frame_idx": int(frame_idx),
        "frame_ts_left": float(sframe.ts_left),
        "frame_ts_right": float(sframe.ts_right),
        "frame_ts_mean": 0.5 * (float(sframe.ts_left) + float(sframe.ts_right)),
        "sync_ms": float(sframe.ts_diff_ms),
        "process_ms": None if process_ms is None else float(process_ms),
        "result_time": float(result_time),
        "tracker_state": None if tracker_state is None else str(tracker_state),
        "tracks": tracks,
    }


def _append_result_jsonl(path: str | None, payload: dict):
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        fp.write("\n")


def _result_is_fresh(result: dict | None,
                     frame_idx: int,
                     *,
                     max_age_frames: int) -> bool:
    if not result:
        return False
    result_idx = int(result.get("frame_idx", 0) or 0)
    if result_idx <= 0 or frame_idx < result_idx:
        return False
    return (frame_idx - result_idx) <= max(int(max_age_frames), 0)


def _build_result_matched_preview(result_snapshot: dict,
                                  *,
                                  fps: float,
                                  show_timing: bool = True) -> cv2.typing.MatLike | None:
    sframe = result_snapshot.get("sframe")
    if sframe is None:
        return None
    result_frame_idx = int(result_snapshot.get("frame_idx", 0) or 0)
    annotated = draw_overlay(
        sframe.left,
        result_snapshot.get("fish", []),
        fps,
        float(result_snapshot.get("sync_ms", sframe.ts_diff_ms)),
        display_frame_idx=result_frame_idx,
        result_frame_idx=result_frame_idx,
        result_age_ms=0.0,
        show_timing=show_timing,
    )
    return compose_stereo_preview(annotated, sframe.right)


def _parse_src(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _display_env_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_demo(cap,
             args,
             *,
             source_label: str | None = None,
             throttle_fps: float | None = None,
             sync_inference: bool = False):
    if source_label:
        print(source_label)
    if not getattr(args, "no_display", False) and not _display_env_available():
        print("[demo] No DISPLAY/WAYLAND_DISPLAY detected; forcing --no-display.")
        args.no_display = True
    # ── Load estimator ─────────────────────────────────────────────
    print("Loading models …")
    estimator = from_config_yaml(
        args.config,
        pipeline_mode=getattr(args, "pipeline_mode", "full"),
        temporal_filter_enabled=(
            False if getattr(args, "no_temporal_filter", False) else None
        ),
        yolo_model=getattr(args, "yolo_model", None),
    )
    print(f"Camera: fx={estimator.camera_params['fx']:.1f}  "
          f"baseline={estimator.camera_params['baseline_m']:.3f}m")
    print(f"Pipeline mode: {getattr(args, 'pipeline_mode', 'full')}")
    print(
        "Temporal output filters: "
        f"{'disabled (raw u/v/z)' if getattr(args, 'no_temporal_filter', False) else 'config'}"
    )
    if getattr(args, "swap_lr", False):
        print("Stereo order: swapped (right -> left, left -> right)")
    print("Ready.\n")

    display = DisplayWindow(
        enabled=not args.no_display,
        display_scale=args.display_scale,
    )
    paused = False
    fps_smooth = 0.0
    t_start = time.time()
    annotated = None
    display_every = max(int(args.display_every), 1)
    process_every = max(int(args.process_every), 1)
    print_every = max(int(args.print_every), 1)
    record_every = max(int(args.record_every), 1)
    show_timing = not bool(getattr(args, "hide_fps_sync", False))
    record_writer = None
    record_path = None
    record_fps = max(float(args.record_fps), 1.0)
    raw_record_writer = None
    raw_record_path = None
    raw_record_fps = max(float(args.record_raw_fps), 1.0)
    result_jsonl_path = None
    frame_jsonl_path = None
    async_display_mode = str(getattr(args, "async_display_mode", "live")).strip().lower()
    overlay_max_age_frames = max(15, process_every * 6)
    state_lock = threading.Lock()
    candidate_event = threading.Event()
    stop_event = threading.Event()
    worker_error: list[BaseException | None] = [None]
    latest_candidate = {"frame": None, "frame_idx": 0}
    latest_result = {
        "frame_idx": 0,
        "fish": [],
        "process_ms": None,
        "sync_ms": None,
        "tracker_state": None,
        "result_time": 0.0,
        "sframe": None,
    }
    latest_rendered_preview = None
    latest_rendered_frame_idx = 0
    latest_displayed_frame_idx = 0
    latest_recorded_frame_idx = 0
    if args.result_jsonl is not None:
        result_jsonl_path = _resolve_result_jsonl_path(args.result_jsonl)
        print(f"Recording structured results: {result_jsonl_path}")
    if args.frame_jsonl is not None:
        frame_jsonl_path = _resolve_result_jsonl_path(args.frame_jsonl)
        print(f"Recording every stereo frame timestamp: {frame_jsonl_path}")

    def _inference_worker():
        last_processed_idx = 0
        try:
            while not stop_event.is_set():
                if not candidate_event.wait(0.1):
                    continue
                if stop_event.is_set():
                    break
                with state_lock:
                    sframe = latest_candidate["frame"]
                    frame_idx = int(latest_candidate["frame_idx"])
                    candidate_event.clear()
                if sframe is None or frame_idx <= last_processed_idx:
                    continue

                infer_t0 = time.time()
                fish = estimator.estimate_frame(sframe)
                process_ms = (time.time() - infer_t0) * 1000.0
                tracker_state = estimator.tracker_state
                result = {
                    "frame_idx": frame_idx,
                    "fish": fish,
                    "process_ms": process_ms,
                    "sync_ms": sframe.ts_diff_ms,
                    "tracker_state": tracker_state,
                    "result_time": time.time(),
                    "sframe": sframe,
                }
                with state_lock:
                    latest_result.update(result)
                _append_result_jsonl(
                    result_jsonl_path,
                    _serialize_result_payload(
                        frame_idx=frame_idx,
                        fish=fish,
                        process_ms=process_ms,
                        tracker_state=tracker_state,
                        sframe=sframe,
                        result_time=result["result_time"],
                    ),
                )
                print_fish_positions(
                    frame_idx,
                    fish,
                    process_ms=process_ms,
                    processed=True,
                    sync_ms=sframe.ts_diff_ms,
                    tracker_state=tracker_state,
                )
                last_processed_idx = frame_idx
        except BaseException as exc:
            worker_error[0] = exc
            stop_event.set()
            candidate_event.set()

    worker = None
    if not sync_inference:
        worker = threading.Thread(
            target=_inference_worker,
            name="depth-inference-worker",
            daemon=True,
        )
        worker.start()
    frame_interval_s = None
    if throttle_fps is not None and np.isfinite(throttle_fps) and throttle_fps > 0:
        frame_interval_s = 1.0 / float(throttle_fps)
    next_frame_deadline = time.time()

    try:
        while True:
            if not sync_inference and worker_error[0] is not None:
                raise RuntimeError("Inference worker failed") from worker_error[0]
            if not paused:
                if frame_interval_s is not None:
                    now = time.time()
                    if next_frame_deadline > now:
                        time.sleep(next_frame_deadline - now)
                    now = time.time()
                    next_frame_deadline = max(next_frame_deadline + frame_interval_s, now)
                t0 = time.time()
                sframe = cap.read()
                if sframe is None:
                    print("End of stream(s).")
                    break
                if getattr(args, "swap_lr", False):
                    sframe.left, sframe.right = sframe.right, sframe.left
                frame_idx = cap.frame_count

                if args.record_raw is not None:
                    if raw_record_writer is None:
                        raw_record_path = _resolve_record_path(args.record_raw)
                        raw_record_writer = _open_video_writer(
                            raw_record_path,
                            cv2.hconcat([sframe.left, sframe.right]),
                            raw_record_fps)
                        print(f"Recording raw stereo: {raw_record_path} "
                              f"(fps={raw_record_fps:.1f})")
                    raw_record_writer.write(cv2.hconcat([sframe.left, sframe.right]))
                if frame_jsonl_path is not None:
                    _append_result_jsonl(frame_jsonl_path, {
                        "frame_idx": int(frame_idx),
                        "frame_ts_left": float(sframe.ts_left),
                        "frame_ts_right": float(sframe.ts_right),
                        "frame_ts_mean": 0.5 * (
                            float(sframe.ts_left) + float(sframe.ts_right)
                        ),
                        "host_record_time_ns": time.time_ns(),
                        "host_record_monotonic_ns": time.monotonic_ns(),
                        "raw_recorded": raw_record_writer is not None,
                    })

                should_process = (
                    frame_idx == 1 or (frame_idx - 1) % process_every == 0
                )
                if should_process:
                    if sync_inference:
                        infer_t0 = time.time()
                        fish = estimator.estimate_frame(sframe)
                        process_ms = (time.time() - infer_t0) * 1000.0
                        tracker_state = estimator.tracker_state
                        result = {
                            "frame_idx": frame_idx,
                            "fish": fish,
                            "process_ms": process_ms,
                            "sync_ms": sframe.ts_diff_ms,
                            "tracker_state": tracker_state,
                            "result_time": time.time(),
                            "sframe": sframe,
                        }
                        with state_lock:
                            latest_result.update(result)
                        _append_result_jsonl(
                            result_jsonl_path,
                            _serialize_result_payload(
                                frame_idx=frame_idx,
                                fish=fish,
                                process_ms=process_ms,
                                tracker_state=tracker_state,
                                sframe=sframe,
                                result_time=result["result_time"],
                            ),
                        )
                        print_fish_positions(
                            frame_idx,
                            fish,
                            process_ms=process_ms,
                            processed=True,
                            sync_ms=sframe.ts_diff_ms,
                            tracker_state=tracker_state,
                        )
                    else:
                        with state_lock:
                            latest_candidate["frame"] = sframe
                            latest_candidate["frame_idx"] = frame_idx
                        candidate_event.set()

                dt = time.time() - t0
                alpha = 0.1
                fps_smooth = alpha * (1.0 / max(dt, 1e-6)) + (1 - alpha) * fps_smooth

                should_display = (
                    not args.no_display
                    and (frame_idx - 1) % display_every == 0
                )
                should_record = (
                    args.record_monitor is not None
                    and (frame_idx - 1) % record_every == 0
                )
                if should_display or should_record:
                    with state_lock:
                        result_snapshot = dict(latest_result)
                    if not sync_inference:
                        if async_display_mode == "matched":
                            result_frame_idx = int(result_snapshot.get("frame_idx", 0) or 0)
                            if result_frame_idx > latest_rendered_frame_idx:
                                preview = _build_result_matched_preview(
                                    result_snapshot,
                                    fps=fps_smooth,
                                    show_timing=show_timing,
                                )
                                if preview is not None:
                                    latest_rendered_preview = preview
                                    latest_rendered_frame_idx = result_frame_idx
                            if latest_rendered_preview is None:
                                annotated = draw_overlay(
                                    sframe.left,
                                    [],
                                    fps_smooth,
                                    sframe.ts_diff_ms,
                                    display_frame_idx=frame_idx,
                                    result_frame_idx=None,
                                    result_age_ms=None,
                                    show_timing=show_timing,
                                )
                                stereo_preview = compose_stereo_preview(annotated, sframe.right)
                                if should_display:
                                    display.show(stereo_preview)
                            else:
                                if should_record and latest_recorded_frame_idx != latest_rendered_frame_idx:
                                    if record_writer is None:
                                        record_path = _resolve_record_path(args.record_monitor)
                                        record_writer = _open_video_writer(
                                            record_path, latest_rendered_preview, record_fps)
                                        print(
                                            f"Recording monitor: {record_path} "
                                            f"(result-matched async preview, fps={record_fps:.1f})"
                                        )
                                    if (latest_rendered_frame_idx - 1) % record_every == 0:
                                        record_writer.write(latest_rendered_preview)
                                        latest_recorded_frame_idx = latest_rendered_frame_idx
                                if should_display and latest_displayed_frame_idx != latest_rendered_frame_idx:
                                    display.show(latest_rendered_preview)
                                    latest_displayed_frame_idx = latest_rendered_frame_idx
                        else:
                            # Low-latency live display: always show the newest
                            # camera frame immediately, while the latest
                            # asynchronously completed result is overlaid if it
                            # is still reasonably fresh.
                            result_frame_idx = result_snapshot.get("frame_idx")
                            result_age_ms = None
                            result_time = result_snapshot.get("result_time")
                            if result_time:
                                result_age_ms = (time.time() - float(result_time)) * 1000.0
                            if _result_is_fresh(
                                result_snapshot,
                                frame_idx,
                                max_age_frames=overlay_max_age_frames,
                            ):
                                overlay_fish = result_snapshot.get("fish", [])
                            else:
                                overlay_fish = []
                            annotated = draw_overlay(
                                sframe.left,
                                overlay_fish,
                                fps_smooth,
                                sframe.ts_diff_ms,
                                display_frame_idx=frame_idx,
                                result_frame_idx=(
                                    int(result_frame_idx)
                                    if result_frame_idx is not None
                                    else None
                                ),
                                result_age_ms=result_age_ms,
                                show_timing=show_timing,
                            )
                            stereo_preview = compose_stereo_preview(annotated, sframe.right)
                            if should_record:
                                if record_writer is None:
                                    record_path = _resolve_record_path(args.record_monitor)
                                    record_writer = _open_video_writer(
                                        record_path, stereo_preview, record_fps)
                                    print(
                                        f"Recording monitor: {record_path} "
                                        f"(low-latency live preview, fps={record_fps:.1f})"
                                    )
                                record_writer.write(stereo_preview)
                            if should_display:
                                display.show(stereo_preview)
                    else:
                        if _result_is_fresh(
                            result_snapshot,
                            frame_idx,
                            max_age_frames=overlay_max_age_frames,
                        ):
                            overlay_fish = result_snapshot.get("fish", [])
                        else:
                            overlay_fish = []
                        result_frame_idx = result_snapshot.get("frame_idx")
                        result_age_ms = None
                        result_time = result_snapshot.get("result_time")
                        if result_time:
                            result_age_ms = (time.time() - float(result_time)) * 1000.0
                        annotated = draw_overlay(
                            sframe.left,
                            overlay_fish,
                            fps_smooth,
                            sframe.ts_diff_ms,
                            display_frame_idx=frame_idx,
                            result_frame_idx=(
                                int(result_frame_idx)
                                if result_frame_idx is not None
                                else None
                            ),
                            result_age_ms=result_age_ms,
                            show_timing=show_timing,
                        )
                        stereo_preview = compose_stereo_preview(annotated, sframe.right)
                        if should_record:
                            if record_writer is None:
                                record_path = _resolve_record_path(args.record_monitor)
                                record_writer = _open_video_writer(
                                    record_path, stereo_preview, record_fps)
                                print(
                                    f"Recording monitor: {record_path} "
                                    f"(every {record_every} frame(s), fps={record_fps:.1f})"
                                )
                            record_writer.write(stereo_preview)
                        if should_display:
                            display.show(stereo_preview)

                if frame_idx % print_every == 0:
                    with state_lock:
                        result_snapshot = dict(latest_result)
                    if _result_is_fresh(
                        result_snapshot,
                        frame_idx,
                        max_age_frames=overlay_max_age_frames,
                    ):
                        fish_for_log = result_snapshot.get("fish", [])
                    else:
                        fish_for_log = []
                    print_fish_positions(
                        frame_idx,
                        fish_for_log,
                        process_ms=None, processed=False,
                        sync_ms=sframe.ts_diff_ms,
                        tracker_state=result_snapshot.get("tracker_state"),
                    )

                if frame_idx % 30 == 0:
                    with state_lock:
                        result_snapshot = dict(latest_result)
                    elapsed = time.time() - t_start
                    latest_result_idx = int(result_snapshot.get("frame_idx", 0) or 0)
                    async_gap = max(0, frame_idx - latest_result_idx) if latest_result_idx > 0 else -1
                    print(f"frame {frame_idx:5d}  |  "
                          f"{len(result_snapshot.get('fish', []))} fish  |  "
                          f"{fps_smooth:.1f} FPS  |  "
                          f"sync={sframe.ts_diff_ms:.1f}ms  |  "
                          f"async_gap={async_gap if async_gap >= 0 else 'n/a'}  |  "
                          f"elapsed {elapsed:.0f}s")

            if not args.no_display:
                key = display.poll_key(paused)
                if key in (ord("q"), 27):
                    break
                elif key == ord("p"):
                    paused = not paused
                    if not paused and frame_interval_s is not None:
                        next_frame_deadline = time.time()
                    print("[paused]" if paused else "[resumed]")
                elif key == ord("s"):
                    if annotated is None:
                        print("No frame available to save yet.")
                    else:
                        fname = f"snapshot_{cap.frame_count:04d}.png"
                        cv2.imwrite(fname, annotated)
                        print(f"Saved: {fname}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if worker is not None:
            stop_event.set()
            candidate_event.set()
            worker.join(timeout=1.0)
        elapsed = time.time() - t_start
        avg_fps = cap.frame_count / max(elapsed, 1e-6)
        print(f"\nDone. {cap.frame_count} frames in {elapsed:.1f}s  "
              f"({avg_fps:.1f} avg FPS)")
        if record_writer is not None:
            record_writer.release()
            _transcode_mp4_h264_inplace(record_path)
            print(f"Monitor video saved: {record_path}")
        if raw_record_writer is not None:
            raw_record_writer.release()
            _transcode_mp4_h264_inplace(raw_record_path)
            print(f"Raw stereo video saved: {raw_record_path}")
        display.close()
        cap.release()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time fish 3D position estimation from live stereo sources")
    parser.add_argument("--left", default=None,
                        help="Left camera: webcam index OR live capture path")
    parser.add_argument("--right", default=None,
                        help="Right camera: webcam index OR live capture path")
    parser.add_argument("--config", "-c", default=CONFIG,
                        help="Path to config.yaml")
    parser.add_argument("--yolo-model", default=None,
                        help="Override models.yolo_path without modifying the config file.")
    parser.add_argument("--pipeline-mode", default="full",
                        choices=["full", "yolo-only"],
                        help="Pipeline runtime mode: 'full' uses tracker/corrector; "
                             "'yolo-only' outputs raw YOLO detections with depth.")
    parser.add_argument(
        "--no-temporal-filter",
        action="store_true",
        help="Bypass Z Kalman and u/v center filtering while preserving the configured ROI mask.",
    )
    parser.add_argument("--no-display", action="store_true",
                        help="Headless — print positions to console, no GUI")
    parser.add_argument("--max-drift", type=float, default=50.0,
                        help="Max allowed left-right timestamp drift (ms)")
    parser.add_argument("--udp-sbs-port", type=int, default=None,
                        help="RTP/H264 UDP side-by-side stereo port")
    parser.add_argument("--udp-latency-ms", type=int, default=60,
                        help="GStreamer jitterbuffer latency for UDP stream")
    parser.add_argument("--swap-lr", action="store_true",
                        help="Swap left/right images before rectification and depth estimation.")
    parser.add_argument("--display-scale", type=float, default=1.4,
                        help="Preview window scale factor.")
    parser.add_argument("--display-every", type=int, default=1,
                        help="Refresh the GUI every N frames.")
    parser.add_argument("--process-every", type=int, default=1,
                        help="Run YOLO+stereo every N frames and reuse the last result in between.")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print tracked fish positions every N frames.")
    parser.add_argument("--record-monitor", default=None,
                        help="Save the same annotated stereo preview shown in the GUI.")
    parser.add_argument("--record-raw", default=None,
                        help="Save the unmodified side-by-side camera frames to an MP4 file.")
    parser.add_argument("--record-raw-fps", type=float, default=30.0,
                        help="FPS metadata for --record-raw output (camera nominal rate).")
    parser.add_argument("--record-every", type=int, default=1,
                        help="Write monitor video every N input frames.")
    parser.add_argument("--record-fps", type=float, default=20.0,
                        help="FPS metadata for --record-monitor output.")
    parser.add_argument("--hide-fps-sync", action="store_true",
                        help="Omit FPS and left/right sync text from the video overlay.")
    parser.add_argument("--result-jsonl", default=None,
                        help="Append processed 3D results with timestamps to a JSONL file.")
    parser.add_argument("--frame-jsonl", default=None,
                        help="Append one capture timestamp record for every stereo input frame.")
    parser.add_argument("--async-display-mode", default="live",
                        choices=["live", "matched"],
                        help="Async mode display policy: 'live' shows the newest camera frame immediately; "
                             "'matched' shows only frames that already have completed inference results.")
    parser.add_argument("--sync-inference", action="store_true",
                        help="Run inference synchronously on each replayed frame. "
                             "Recommended for offline export to keep overlays aligned.")
    parser.add_argument("--throttle-fps", type=float, default=None,
                        help="Throttle replay to this FPS. Useful for offline video export so "
                             "the reader does not outrun inference.")
    return parser


def main():
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = _build_parser().parse_args()

    if args.udp_sbs_port is not None:
        cap = StereoCaptureSideBySideGst(
            args.udp_sbs_port,
            latency_ms=args.udp_latency_ms,
        )
        source_label = f"UDP side-by-side stereo: port={args.udp_sbs_port}"
    elif args.left is not None and args.right is not None:
        cap = StereoCapture(_parse_src(args.left), _parse_src(args.right),
                            max_drift_ms=args.max_drift)
        fps_l, fps_r = cap.fps
        source_label = (
            f"Left : {args.left}  ({fps_l:.1f} fps)\n"
            f"Right: {args.right}  ({fps_r:.1f} fps)"
        )
    else:
        sys.exit(
            "ERROR: real-device mode requires --udp-sbs-port or --left + --right. "
            "Simulation/offline replay now uses demo_sim_video.py."
        )

    run_demo(
        cap,
        args,
        source_label=source_label,
        throttle_fps=args.throttle_fps,
        sync_inference=bool(args.sync_inference),
    )


def _resolve_record_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_dir() or str(path).endswith(os.sep):
        p = p / time.strftime("monitor_%Y%m%d_%H%M%S.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _resolve_result_jsonl_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_dir() or str(path).endswith(os.sep):
        p = p / time.strftime("depth_results_%Y%m%d_%H%M%S.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _open_video_writer(path: str, frame, fps: float):
    height, width = frame.shape[:2]
    suffix = Path(path).suffix.lower()
    codecs = ["mp4v", "avc1"] if suffix in (".mp4", ".m4v") else ["MJPG", "XVID"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to open monitor video writer: {path}")


def _transcode_mp4_h264_inplace(path: str | None):
    if not path:
        return
    p = Path(path)
    if p.suffix.lower() != ".mp4":
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not p.is_file():
        return

    temp_path = p.with_name(f"{p.stem}.h264_tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(p),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        str(temp_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        temp_path.replace(p)
    except Exception as exc:
        print(f"[monitor] H.264 transcode skipped for {p}: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
