from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    combined: np.ndarray
    width: int
    height: int


class StereoUdpSideBySideCapture:
    """Receive a side-by-side H264 RTP stereo stream over UDP using GStreamer."""

    def __init__(self, port: int = 5600, latency_ms: int = 60) -> None:
        self._port = int(port)
        self._latency_ms = int(latency_ms)
        self._pipeline = None
        self._appsink = None
        self._gst = None

    def open(self) -> None:
        try:
            import gi
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "PyGObject (gi) is required for UDP stereo capture. "
                "Install python3-gi / gir1.2-gstreamer-1.0 on the system."
            ) from exc

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        pipeline = Gst.parse_launch(
            " ".join(
                [
                    f"udpsrc port={self._port}",
                    'caps="application/x-rtp, media=video, encoding-name=H264, payload=96"',
                    f"! rtpjitterbuffer latency={self._latency_ms}",
                    "! rtph264depay",
                    "! h264parse",
                    "! avdec_h264",
                    "! videoconvert",
                    "! video/x-raw,format=BGR",
                    "! appsink name=stereo_sink emit-signals=true sync=false max-buffers=1 drop=true",
                ]
            )
        )
        appsink = pipeline.get_by_name("stereo_sink")
        if appsink is None:
            raise RuntimeError("failed to create GStreamer appsink")
        pipeline.set_state(Gst.State.PLAYING)
        self._pipeline = pipeline
        self._appsink = appsink
        self._gst = Gst

    def read(self) -> StereoFrame | None:
        if self._appsink is None or self._gst is None:
            raise RuntimeError("capture is not open")
        sample = self._appsink.emit("pull-sample")
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        ok, mapinfo = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return None
        try:
            frame = np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(mapinfo)
        if width % 2 != 0:
            raise RuntimeError(f"expected even side-by-side width, got {width}")
        eye_width = width // 2
        left = frame[:, :eye_width].copy()
        right = frame[:, eye_width:].copy()
        return StereoFrame(left=left, right=right, combined=frame, width=eye_width, height=height)

    def close(self) -> None:
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None
        self._appsink = None
        self._gst = None
