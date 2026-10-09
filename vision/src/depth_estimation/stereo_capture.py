#!/usr/bin/env python3
"""
Stereo Video Capture
====================
Synchronised stereo capture from files, devices, and UDP side-by-side streams.
"""

import os
import shutil
import subprocess
import time
from collections import deque

import cv2
import numpy as np


class StereoCapture:
    """
    Synchronised stereo capture from two independent sources.

    Parameters
    ----------
    left_src : int or str
        Camera index (0, 1, …) or video file path for the left camera.
    right_src : int or str
        Camera index or video file path for the right camera.
    max_drift_ms : float
        Maximum allowed timestamp difference (ms) before logging a warning.
    buffer_size : int
        How many recent frames to buffer per camera for timestamp matching.

    Usage
    -----
        cap = StereoCapture(left_src=0, right_src=1)
        while True:
            frame = cap.read()
            if frame is None:
                break
            fish = estimator.estimate(frame.left, frame.right)
            print(f"ts_diff={frame.ts_diff_ms:.1f}ms")
    """

    def __init__(self,
                 left_src,
                 right_src,
                 max_drift_ms: float = 50.0,
                 buffer_size: int = 30):
        self._max_drift_ms = max_drift_ms
        self._buffer_size = buffer_size

        # Open both sources
        self._cap_left = cv2.VideoCapture(left_src)
        self._cap_right = cv2.VideoCapture(right_src)

        if not self._cap_left.isOpened():
            raise RuntimeError(f"Cannot open left source: {left_src}")
        if not self._cap_right.isOpened():
            raise RuntimeError(f"Cannot open right source: {right_src}")

        # Try to align fps — if both are files with matching fps this helps
        fps_l = self._cap_left.get(cv2.CAP_PROP_FPS)
        fps_r = self._cap_right.get(cv2.CAP_PROP_FPS)
        if fps_l > 0 and fps_r > 0 and abs(fps_l - fps_r) > 1:
            print(f"[StereoCapture] WARNING: fps mismatch — "
                  f"left={fps_l:.1f} right={fps_r:.1f}")

        # For offline file replay we want deterministic 1:1 pairing instead of
        # wall-clock matching; otherwise replay can duplicate/skip frames and
        # stretch the exported monitor video.
        self._direct_pair_mode = (
            self._is_file_source(left_src) and self._is_file_source(right_src)
        )

        self._buf_left: deque[tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._buf_right: deque[tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._drift_warned = False
        self._frame_count = 0

    # ── public ──────────────────────────────────────────────────────────

    def read(self) -> "StereoFrame | None":
        """
        Read the next synchronised stereo pair.

        Returns
        -------
        StereoFrame or None
            None when either stream is exhausted.
        """
        if self._direct_pair_mode:
            return self._read_direct_pair()

        # Fill buffers until we have at least one frame from each
        while len(self._buf_left) == 0:
            if not self._fill(self._cap_left, self._buf_left):
                return None
        while len(self._buf_right) == 0:
            if not self._fill(self._cap_right, self._buf_right):
                return None

        # Match: pop the older front, match against the other buffer
        ts_l, img_l = self._buf_left[0]
        ts_r, img_r = self._buf_right[0]

        if ts_l <= ts_r:
            # Left is older — use left, find closest right
            self._buf_left.popleft()
            idx_r = min(range(len(self._buf_right)),
                        key=lambda i: abs(self._buf_right[i][0] - ts_l))
            best_r = self._buf_right[idx_r]
            for _ in range(idx_r + 1):
                self._buf_right.popleft()
            ts_diff = abs(best_r[0] - ts_l)
            out = StereoFrame(img_l, best_r[1], ts_l, best_r[0], ts_diff)
        else:
            # Right is older — use right, find closest left
            self._buf_right.popleft()
            idx_l = min(range(len(self._buf_left)),
                        key=lambda i: abs(self._buf_left[i][0] - ts_r))
            best_l = self._buf_left[idx_l]
            for _ in range(idx_l + 1):
                self._buf_left.popleft()
            ts_diff = abs(best_l[0] - ts_r)
            out = StereoFrame(best_l[1], img_r, best_l[0], ts_r, ts_diff)

        self._frame_count += 1
        self._check_drift(out.ts_diff_ms)
        return out

    def release(self):
        self._cap_left.release()
        self._cap_right.release()

    @property
    def fps(self) -> tuple[float, float]:
        return (self._cap_left.get(cv2.CAP_PROP_FPS),
                self._cap_right.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── private ─────────────────────────────────────────────────────────

    def _fill(self, cap, buf):
        """Read one frame from *cap* and push (timestamp, img) into *buf*."""
        ok, frame = cap.read()
        if ok:
            buf.append((time.time(), frame))
            return True
        return False

    def _read_direct_pair(self) -> "StereoFrame | None":
        ok_l, frame_l = self._cap_left.read()
        ok_r, frame_r = self._cap_right.read()
        if not ok_l or not ok_r:
            return None

        ts_l = self._capture_stream_time_s(self._cap_left)
        ts_r = self._capture_stream_time_s(self._cap_right)
        out = StereoFrame(frame_l, frame_r, ts_l, ts_r, abs(ts_l - ts_r))
        self._frame_count += 1
        self._check_drift(out.ts_diff_ms)
        return out

    @staticmethod
    def _is_file_source(src) -> bool:
        return isinstance(src, str) and os.path.isfile(src)

    @staticmethod
    def _capture_stream_time_s(cap) -> float:
        pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_ms > 0:
            return pos_ms / 1000.0

        pos_frames = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if pos_frames > 0 and fps > 0:
            return pos_frames / fps

        return time.time()

    def _check_drift(self, ts_diff_ms: float):
        if ts_diff_ms > self._max_drift_ms and not self._drift_warned:
            print(f"[StereoCapture] WARNING: frame drift {ts_diff_ms:.0f}ms "
                  f"> {self._max_drift_ms:.0f}ms threshold. "
                  f"Check camera synchronisation!")
            self._drift_warned = True
        elif ts_diff_ms <= self._max_drift_ms:
            self._drift_warned = False


class StereoFrame:
    """One synchronised stereo pair."""
    __slots__ = ("left", "right", "ts_left", "ts_right", "ts_diff_ms")

    def __init__(self, left, right, ts_left, ts_right, ts_diff):
        self.left: np.ndarray = left
        self.right: np.ndarray = right
        self.ts_left: float = ts_left
        self.ts_right: float = ts_right
        self.ts_diff_ms: float = ts_diff * 1000.0


# ===================================================================
#  Convenience: side-by-side fallback (existing functionality)
# ===================================================================

class StereoCaptureSideBySide:
    """
    Fallback for single side-by-side video / webcam.
    Splits each frame into left and right halves.
    """

    def __init__(self, src):
        self._src = src
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        self._frame_count = 0
        self._is_file_source = isinstance(src, str) and os.path.isfile(src)

    def read(self) -> StereoFrame | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_count += 1
        h, w = frame.shape[:2]
        mid = w // 2
        ts = self._capture_stream_time_s(self._cap) if self._is_file_source else time.time()
        return StereoFrame(frame[:, :mid], frame[:, mid:], ts, ts, 0.0)

    def release(self):
        self._cap.release()

    @property
    def fps(self):
        f = self._cap.get(cv2.CAP_PROP_FPS)
        return (f, f)

    @property
    def frame_count(self):
        return self._frame_count

    @staticmethod
    def _capture_stream_time_s(cap) -> float:
        pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_ms > 0:
            return pos_ms / 1000.0

        pos_frames = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if pos_frames > 0 and fps > 0:
            return pos_frames / fps

        return time.time()


class StereoCaptureSideBySideGst:
    """
    Side-by-side stereo capture from RTP/H264 UDP using GStreamer.

    The stream is expected to carry a single BGR frame with left/right halves
    laid out horizontally.
    """

    def __init__(self,
                 port: int,
                 *,
                 latency_ms: int = 60,
                 caps: str | None = None,
                 width: int | None = None,
                 height: int | None = None):
        self._port = int(port)
        self._latency_ms = int(latency_ms)
        self._caps = caps or (
            "application/x-rtp, media=(string)video, clock-rate=(int)90000, "
            "encoding-name=(string)H264, payload=(int)96"
        )
        self._frame_count = 0
        self._width = width
        self._height = height
        self._backend = None
        self._pipeline = None
        self._appsink = None
        self._gst = None
        self._proc = None
        self._stdout = None

        try:
            self._init_gi_backend()
        except Exception:
            self._init_subprocess_backend()

    def _init_gi_backend(self):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._gst = Gst
        pipeline_desc = (
            f'udpsrc port={self._port} caps="{self._caps}" ! '
            f"rtpjitterbuffer latency={self._latency_ms} ! "
            "rtph264depay ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_desc)
        self._appsink = self._pipeline.get_by_name("appsink")
        if self._appsink is None:
            raise RuntimeError("Failed to create GStreamer appsink")
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")
        self._backend = "gi"

    def _init_subprocess_backend(self):
        gst_launch = shutil.which("gst-launch-1.0")
        if gst_launch is None:
            raise RuntimeError(
                "GStreamer is required: neither gi.repository.Gst nor gst-launch-1.0 is available"
            )
        if self._width is None or self._height is None:
            self._width = 1280
            self._height = 480

        cmd = [
            gst_launch,
            "-q",
            "udpsrc", f"port={self._port}", f"caps={self._caps}",
            "!", "rtpjitterbuffer", f"latency={self._latency_ms}",
            "!", "rtph264depay",
            "!", "avdec_h264",
            "!", "videoconvert",
            "!", f"video/x-raw,format=BGR,width={self._width},height={self._height}",
            "!", "fdsink", "fd=1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._stdout = self._proc.stdout
        self._backend = "subprocess"

    def read(self) -> StereoFrame | None:
        if self._backend == "gi":
            frame = self._read_gi()
        else:
            frame = self._read_subprocess()

        if frame is None:
            return None

        self._frame_count += 1
        mid = frame.shape[1] // 2
        now = time.time()
        return StereoFrame(frame[:, :mid], frame[:, mid:], now, now, 0.0)

    def _read_gi(self):
        sample = self._appsink.emit("try-pull-sample", 500_000_000)
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        self._width = width
        self._height = height

        ok, mapinfo = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return None
        try:
            return np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(mapinfo)

    def _read_subprocess(self):
        if self._proc is None or self._stdout is None:
            return None
        if self._proc.poll() is not None:
            return None

        frame_bytes = int(self._width) * int(self._height) * 3
        raw = self._read_exact(frame_bytes)
        if raw is None:
            return None
        return np.frombuffer(raw, np.uint8).reshape((self._height, self._width, 3)).copy()

    def _read_exact(self, size: int):
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self._stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def release(self):
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        if self._stdout is not None:
            self._stdout.close()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    @property
    def fps(self):
        return (0.0, 0.0)

    @property
    def frame_count(self):
        return self._frame_count
