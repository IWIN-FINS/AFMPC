
"""
Fish Position Estimation Module
================================
Estimates 3D positions of fish relative to a stereo camera on an
underwater robot (ROV/AUV).

Pipeline
--------
  Left image ──┬──► YOLO ──────────► fish bounding boxes ──┐
               │                                            │
  Left+Right ──┴──► FoundationStereo ──► disparity map ────┼──► 3D positions
                                                            │
                                               Tracker ────┘   (ID + smoothing)

Coordinate frame (OpenCV / camera convention)
---------------------------------------------
  X → right   (metres)
  Y → down    (metres)
  Z → forward (metres, depth)

Usage
-----
    import yaml
    from fish_position import FishPositionEstimator

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    estimator = FishPositionEstimator(cfg)

    # On every stereo frame:
    left  = cv2.imread("left.png")
    right = cv2.imread("right.png")
    fish = estimator.estimate(left, right)

    for f in fish:
        print(f"Fish #{f['id']}: x={f['position'][0]:.2f}m, "
              f"y={f['position'][1]:.2f}m, z={f['position'][2]:.2f}m")
"""

import sys
import os
import time
import importlib
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from depth_estimation.corrector_smoother import (
    TemporalBBoxCorrectorSmootherConfig,
    TemporalBBoxCorrectorSmootherRuntime,
)
from depth_estimation.depth_temporal import (
    SequenceConfidenceKalmanDepthFilter,
    UVZJointKalmanFilter,
)
from depth_estimation.refiner import (
    TemporalBBoxRefinerConfig,
    TemporalBBoxRefinerRuntime,
)
from depth_estimation.rectifier import StereoRectifier


os.environ.setdefault("XFORMERS_DISABLED", "1")


# ── Path hacks: make FoundationStereo importable ──────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, "..", ".."))
for _FS_DIR in (
    os.path.join(_MODULE_DIR, "FoundationStereo"),
    os.path.join(_REPO_ROOT, "third_party", "FoundationStereo"),
    os.path.join(_REPO_ROOT, "third_party", "Fast-FoundationStereo"),
):
    _FS_DIR = os.path.normpath(_FS_DIR)
    if os.path.isfile(os.path.join(_FS_DIR, "core", "foundation_stereo.py")):
        if _FS_DIR not in sys.path:
            sys.path.insert(0, _FS_DIR)
        break


# ===================================================================
#  Confidence-aware Temporal Depth Filter
# ===================================================================

@dataclass
class DepthROIStats:
    """Robust per-detection depth statistics from a target ROI."""

    z_raw: float
    z_median: float
    z_iqr: float
    valid_ratio: float
    depth_histogram: list[float]
    depth_histogram_edges: list[float]
    center_uv: tuple[float, float] | None
    core_px: int
    sep_score: float
    valid: bool


@dataclass
class DepthROIDebug:
    """Intermediate masks and geometry used for one ROI depth estimate."""

    bbox_xyxy: tuple[int, int, int, int]
    roi_bbox_xyxy: tuple[int, int, int, int]
    center_fraction: float
    roi_disparity: np.ndarray
    roi_depth: np.ndarray
    valid_mask_initial: np.ndarray
    valid_mask_after_foreground: np.ndarray
    valid_mask_after_color: np.ndarray
    valid_mask_after_depth_prior: np.ndarray
    valid_mask_after_center_prior: np.ndarray
    valid_mask_final: np.ndarray
    valid_mask_depth_core: np.ndarray
    valid_mask_background_ring: np.ndarray
    selected_component_mask: np.ndarray | None
    center_prior_roi: tuple[float, float] | None
    depth_prior_m: float | None


def create_temporal_depth_filter(cfg: dict | None):
    cfg = dict(cfg or {})
    filter_type = str(cfg.get("filter_type", "legacy")).lower()
    if filter_type in {"uvz_joint_kalman", "uvz_joint", "joint_uvz_kalman"}:
        fusion_cfg = dict(cfg.get("uvz_joint", {}))
        fusion_cfg.setdefault("enabled", cfg.get("enabled", True))
        fusion_cfg.setdefault("fallback_dt_s", cfg.get("fallback_dt_s", cfg.get("dt_s", 0.1)))
        fusion_cfg.setdefault("confidence_decay", cfg.get("confidence_decay", 0.8))
        fusion_cfg.setdefault("camera", cfg.get("camera", {}))
        return UVZJointKalmanFilter(fusion_cfg)
    if filter_type in {"sequence_conf_kalman", "sequence", "buffer_kalman"}:
        fusion_cfg = dict(cfg.get("fusion", {}))
        fusion_cfg.setdefault("enabled", cfg.get("enabled", True))
        fusion_cfg.setdefault("fallback_dt_s", cfg.get("fallback_dt_s", cfg.get("dt_s", 0.1)))
        fusion_cfg.setdefault(
            "process_var_z",
            cfg.get("process_var_z", 0.02),
        )
        fusion_cfg.setdefault(
            "process_var_v",
            cfg.get("process_var_z_dot", 0.5),
        )
        fusion_cfg.setdefault("r_base", cfg.get("measurement_noise_m2", 0.04))
        fusion_cfg.setdefault("confidence_decay", cfg.get("confidence_decay", 0.8))
        fusion_cfg.setdefault("nis_gate_enabled", cfg.get("nis_gate_enabled", True))
        fusion_cfg.setdefault("nis_gate_threshold", cfg.get("nis_gate_threshold", 6.63))
        return SequenceConfidenceKalmanDepthFilter(fusion_cfg)
    return TemporalDepthFilter(cfg)


class TemporalDepthFilter:
    """Legacy 1D constant-velocity Kalman filter with physical jump gating."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.fallback_dt = float(cfg.get("fallback_dt_s", cfg.get("dt_s", 0.1)))
        self.v_max = float(cfg.get("v_max_mps", 1.5))
        self.margin = float(cfg.get("gate_margin_m", 0.25))
        self.soft_gate_scale = float(cfg.get("soft_gate_scale", 2.0))
        self.R0 = float(cfg.get("measurement_noise_m2", 0.04))
        self.sigma_jump = max(float(cfg.get("sigma_jump_m", 0.5)), 1e-6)
        self.process_var_z = float(cfg.get("process_var_z", 0.02))
        self.process_var_z_dot = float(cfg.get("process_var_z_dot", 0.5))
        self.min_confidence = float(cfg.get("min_confidence", 0.03))
        self.confidence_decay = float(cfg.get("confidence_decay", 0.8))

        self.initialized = False
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([1.0, 1.0]).astype(np.float64)
        self.confidence = 0.0
        self.last_valid = False
        self.last_rejected = False

    def predict_only(self, dt_s: float | None = None, **_: object):
        if not self.enabled or not self.initialized:
            self.confidence *= self.confidence_decay
            self.last_valid = False
            self.last_rejected = False
            return
        self._predict(dt_s)
        self.confidence *= self.confidence_decay
        self.last_valid = False
        self.last_rejected = False

    def update(self, z_raw: float | None, depth_stats=None,
               depth_confidence: float | None = None,
               dt_s: float | None = None, **_: object) -> dict:
        del depth_stats
        depth_confidence = float(np.clip(depth_confidence or 0.0, 0.0, 1.0))
        if not self.enabled:
            return {
                "z": z_raw,
                "z_dot": 0.0,
                "confidence": depth_confidence,
                "valid": z_raw is not None and np.isfinite(z_raw),
                "rejected": False,
            }

        if z_raw is None or not np.isfinite(z_raw):
            self.predict_only(dt_s)
            return self.state()

        if not self.initialized:
            self.x[:] = [float(z_raw), 0.0]
            self.P = np.diag([self.R0, 1.0]).astype(np.float64)
            self.initialized = True
            self.confidence = depth_confidence
            self.last_valid = True
            self.last_rejected = False
            return self.state()

        self._predict(dt_s)
        return self._correct(float(z_raw), depth_confidence)

    def correct_only(self, z_raw: float | None, depth_stats=None,
                     depth_confidence: float | None = None,
                     dt_s: float | None = None, **kwargs: object) -> dict:
        del dt_s
        if not self.initialized:
            return self.update(
                z_raw,
                depth_stats=depth_stats,
                depth_confidence=depth_confidence,
                dt_s=0.0,
                **kwargs,
            )
        if z_raw is None or not np.isfinite(z_raw):
            return self.state()
        confidence = float(np.clip(depth_confidence or 0.0, 0.0, 1.0))
        return self._correct(float(z_raw), confidence)

    def _correct(self, z_raw: float, depth_confidence: float) -> dict:
        z_pred = float(self.x[0])
        residual = float(z_raw - z_pred)
        dt = self.fallback_dt
        gate = self.v_max * dt + self.margin
        abs_residual = abs(residual)
        depth_confidence *= float(np.exp(-abs_residual / self.sigma_jump))

        if abs_residual > self.soft_gate_scale * gate:
            self.confidence *= self.confidence_decay
            self.last_valid = False
            self.last_rejected = True
            return self.state()
        if abs_residual > gate:
            depth_confidence *= 0.2

        effective_conf = max(depth_confidence, self.min_confidence)
        r_meas = self.R0 / effective_conf
        h = np.array([[1.0, 0.0]], dtype=np.float64)
        innovation_var = float((h @ self.P @ h.T + r_meas).item())
        gain = (self.P @ h.T / innovation_var).reshape(2)
        self.x += gain * residual
        self.P = (np.eye(2) - gain[:, None] @ h) @ self.P
        self.confidence = depth_confidence
        self.last_valid = True
        self.last_rejected = False
        return self.state()

    def state(self) -> dict:
        return {
            "z": float(self.x[0]) if self.initialized else None,
            "z_dot": float(self.x[1]) if self.initialized else 0.0,
            "confidence": float(self.confidence),
            "valid": bool(self.last_valid),
            "rejected": bool(self.last_rejected),
        }

    def _predict(self, dt_s: float | None = None):
        dt = self._resolve_dt(dt_s)
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = np.diag([self.process_var_z, self.process_var_z_dot]).astype(np.float64)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

    def _resolve_dt(self, dt_s: float | None) -> float:
        if dt_s is None or not np.isfinite(dt_s) or dt_s <= 0:
            return self.fallback_dt
        return float(dt_s)


# ===================================================================
#  Lightweight Output BBox Filter
# ===================================================================

@dataclass
class OutputBBoxFilterConfig:
    enabled: bool = False
    center_alpha_min: float = 0.18
    center_alpha_max: float = 0.88
    size_alpha_min: float = 0.14
    size_alpha_max: float = 0.72
    velocity_scale_px_per_s: float = 220.0
    size_velocity_scale_px_per_s: float = 180.0
    tracker_only_alpha_scale: float = 0.80
    reset_after_missed_frames: int = 3
    fallback_dt_s: float = 0.1

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "OutputBBoxFilterConfig":
        cfg = dict(cfg or {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


#  Lightweight Multi-Object Tracker
# ===================================================================

class FishTrack:
    """Internal state for one tracked fish."""

    __slots__ = ("id", "bbox", "output_bbox", "raw_bbox", "pos_3d", "confidence",
                 "age", "hits", "time_since_update",
                 "frame_id", "history", "depth_filter",
                 "raw_position", "raw_depth", "depth_confidence",
                 "raw_center_uv", "filtered_center_uv",
                 "camera_params", "center_ema_enabled", "center_ema_alpha",
                 "center_ema_tau_s", "center_ema_fallback_dt_s",
                 "center_ema_initialized", "center_ema_prior_uv",
                 "center_ema_step_alpha",
                 "center_filter_type", "center_ab_enabled",
                 "center_ab_alpha", "center_ab_beta",
                 "center_ab_velocity_uv", "center_ab_prior_uv",
                 "center_ab_prior_velocity_uv",
                 "center_ab_step_dt_s", "center_ab_step_alpha",
                 "center_ab_step_beta",
                 "depth_valid", "depth_rejected", "z_dot",
                 "depth_stats", "source",
                 "depth_gain", "depth_quality", "depth_filter_mode",
                 "depth_r_t", "depth_R_t", "depth_nis",
                 "depth_innovation", "depth_S_t", "depth_P_zz_pred",
                 "depth_dt_s")

    def __init__(self, track_id, bbox, pos_3d, confidence, frame_id,
                 temporal_depth_cfg=None, depth_stats=None,
                 depth_confidence=None, dt_s=None, source: str = "yolo",
                 seed_track=None):
        self.id = track_id
        self.bbox = list(bbox)                       # [x1, y1, x2, y2]
        self.output_bbox = list(bbox)
        self.raw_bbox = list(bbox)
        self.pos_3d = np.array(pos_3d, dtype=np.float32)  # (3,)
        self.raw_position = self.pos_3d.copy()
        self.confidence = confidence
        self.source = source
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.frame_id = frame_id
        self.history = deque(maxlen=30)
        self.depth_filter = create_temporal_depth_filter(temporal_depth_cfg)
        td_cfg = dict(temporal_depth_cfg or {})
        self.camera_params = dict(td_cfg.get("camera", {}))
        center_ema_cfg = dict(td_cfg.get("center_ema", {}))
        center_filter_cfg = dict(td_cfg.get("center_filter", {}))
        self.center_ema_enabled = bool(center_ema_cfg.get("enabled", False))
        self.center_ema_alpha = float(np.clip(center_ema_cfg.get("alpha", 0.15), 0.0, 1.0))
        self.center_ema_tau_s = float(center_ema_cfg.get("time_constant_s", 0.0))
        self.center_ema_fallback_dt_s = float(td_cfg.get("fallback_dt_s", 0.1))
        center_filter_type = str(center_filter_cfg.get("type", "") or "").strip().lower()
        if center_filter_type not in {"ema", "alpha_beta", "raw"}:
            center_filter_type = "ema" if self.center_ema_enabled else "raw"
        self.center_filter_type = center_filter_type
        self.center_ab_enabled = (
            self.center_filter_type == "alpha_beta"
            and bool(center_filter_cfg.get("enabled", True))
        )
        self.center_ab_alpha = float(
            np.clip(center_filter_cfg.get("alpha", 0.32), 0.0, 1.0)
        )
        self.center_ab_beta = float(
            np.clip(center_filter_cfg.get("beta", 0.04), 0.0, 1.0)
        )
        self.depth_stats = depth_stats
        self.depth_gain = 0.0
        self.depth_quality = 0.0
        self.depth_filter_mode = "unknown"
        self.depth_r_t = 0.0
        self.depth_R_t = 0.0
        self.depth_nis = None
        self.depth_innovation = None
        self.depth_S_t = None
        self.depth_P_zz_pred = None
        self.depth_dt_s = float(dt_s) if dt_s is not None and np.isfinite(dt_s) and dt_s > 0 else 0.0
        self.raw_center_uv = self._bbox_center_xyxy(self.raw_bbox)
        self.filtered_center_uv = self.raw_center_uv
        self.center_ema_initialized = False
        self.center_ema_prior_uv = None
        self.center_ema_step_alpha = 1.0
        self.center_ab_velocity_uv = (0.0, 0.0)
        self.center_ab_prior_uv = None
        self.center_ab_prior_velocity_uv = None
        self.center_ab_step_dt_s = float(self.center_ema_fallback_dt_s)
        self.center_ab_step_alpha = float(self.center_ab_alpha)
        self.center_ab_step_beta = float(self.center_ab_beta)
        self.raw_depth = self._extract_z(self.pos_3d)
        self.depth_confidence = (
            float(depth_confidence) if depth_confidence is not None else
            self._stats_confidence(depth_stats)
        )
        self._seed_depth_filter_from_track(seed_track)
        self._seed_center_filter_from_track(seed_track)
        state = self.depth_filter.update(
            self.raw_depth,
            depth_stats=depth_stats,
            depth_confidence=self.depth_confidence,
            dt_s=dt_s,
            center_uv=self.raw_center_uv,
        )
        self._apply_filter_state(state, dt_s=dt_s)
        self.history.append(self.pos_3d.copy())

    def predict(self):
        """No-motion model — just increment the miss counter."""
        self.time_since_update += 1

    def predict_depth_only(self, dt_s: float | None = None):
        """Advance depth state when the track receives no valid detection."""
        self.depth_filter.predict_only(dt_s, center_uv=self.filtered_center_uv)
        state = self.depth_filter.state()
        self._apply_filter_state(state, dt_s=dt_s, update_center=False)

    def predict_temporal_only(self, dt_s: float | None = None) -> None:
        """Advance depth and center states without publishing a detection."""
        self.predict_depth_only(dt_s)
        if self.center_filter_type != "alpha_beta" or not self.center_ab_enabled:
            return
        if not self.center_ema_initialized or self.filtered_center_uv is None:
            return
        dt = max(self._resolve_center_filter_dt(dt_s), 1e-9)
        prior = np.asarray(self.filtered_center_uv, dtype=np.float64)
        velocity = np.asarray(self.center_ab_velocity_uv, dtype=np.float64)
        predicted = prior + dt * velocity
        self.center_ab_prior_uv = (float(prior[0]), float(prior[1]))
        self.center_ab_prior_velocity_uv = (float(velocity[0]), float(velocity[1]))
        self.center_ab_step_dt_s = float(dt)
        self.filtered_center_uv = (float(predicted[0]), float(predicted[1]))

    def update(self, bbox, pos_3d, confidence, frame_id, alpha=0.7,
               depth_stats=None, depth_confidence=None, dt_s=None,
               source: str = "yolo",
               bbox_smoothing_enabled: bool = True,
               bbox_center_alpha: float = 0.35,
               bbox_size_alpha: float = 0.30):
        """Update with a matched detection (EMA smoothing on position)."""
        self.raw_bbox = list(bbox)
        if bbox_smoothing_enabled:
            self.bbox = self._smooth_bbox(
                self.bbox,
                bbox,
                center_alpha=bbox_center_alpha,
                size_alpha=bbox_size_alpha,
            )
        else:
            self.bbox = list(bbox)
        self.output_bbox = list(self.bbox)
        self.raw_position = np.array(pos_3d, dtype=np.float32)
        self.raw_center_uv = self._bbox_center_xyxy(self.raw_bbox)
        self.depth_stats = depth_stats
        self.source = source
        self.raw_depth = self._extract_z(self.raw_position)
        self.depth_confidence = (
            float(depth_confidence) if depth_confidence is not None else
            self._stats_confidence(depth_stats)
        )
        if self.raw_depth is None:
            prev_position = self.pos_3d.copy()
            prev_z = self._extract_z(prev_position)
            self.raw_position = prev_position.copy()
            self.raw_depth = prev_z
            state = self.depth_filter.update(
                None,
                depth_stats=depth_stats,
                depth_confidence=self.depth_confidence,
                dt_s=dt_s,
                center_uv=self.raw_center_uv,
            )
            self._apply_filter_state(state, dt_s=dt_s)
            self.confidence = confidence
            self.hits += 1
            self.age += 1
            self.time_since_update = 0
            self.frame_id = frame_id
            self.history.append(self.pos_3d.copy())
            return
        state = self.depth_filter.update(
            self.raw_depth,
            depth_stats=depth_stats,
            depth_confidence=self.depth_confidence,
            dt_s=dt_s,
            center_uv=self.raw_center_uv,
        )
        self._apply_filter_state(state, alpha=alpha, dt_s=dt_s)
        self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.frame_id = frame_id
        self.history.append(self.pos_3d.copy())

    @staticmethod
    def _smooth_bbox(prev_bbox, new_bbox, *, center_alpha: float, size_alpha: float):
        center_alpha = float(np.clip(center_alpha, 0.0, 1.0))
        size_alpha = float(np.clip(size_alpha, 0.0, 1.0))
        px1, py1, px2, py2 = [float(v) for v in prev_bbox]
        nx1, ny1, nx2, ny2 = [float(v) for v in new_bbox]

        pcx = 0.5 * (px1 + px2)
        pcy = 0.5 * (py1 + py2)
        pw = max(px2 - px1, 1.0)
        ph = max(py2 - py1, 1.0)

        ncx = 0.5 * (nx1 + nx2)
        ncy = 0.5 * (ny1 + ny2)
        nw = max(nx2 - nx1, 1.0)
        nh = max(ny2 - ny1, 1.0)

        cx = (1.0 - center_alpha) * pcx + center_alpha * ncx
        cy = (1.0 - center_alpha) * pcy + center_alpha * ncy
        w = (1.0 - size_alpha) * pw + size_alpha * nw
        h = (1.0 - size_alpha) * ph + size_alpha * nh

        return [
            float(cx - 0.5 * w),
            float(cy - 0.5 * h),
            float(cx + 0.5 * w),
            float(cy + 0.5 * h),
        ]

    @staticmethod
    def _bbox_center_xyxy(bbox) -> tuple[float, float]:
        return (
            0.5 * float(bbox[0] + bbox[2]),
            0.5 * float(bbox[1] + bbox[3]),
        )

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits

    @staticmethod
    def _extract_z(pos_3d):
        if pos_3d is None:
            return None
        z = float(np.asarray(pos_3d)[2])
        return z if np.isfinite(z) else None

    @staticmethod
    def _stats_confidence(depth_stats):
        if depth_stats is None:
            return 1.0
        return float(getattr(depth_stats, "valid_ratio", 0.0))

    def _seed_depth_filter_from_track(self, seed_track) -> None:
        if seed_track is None:
            return
        seed_filter = getattr(seed_track, "depth_filter", None)
        if seed_filter is None:
            return
        if not hasattr(self.depth_filter, "seed_state"):
            return
        state = seed_filter.state() if hasattr(seed_filter, "state") else None
        if not isinstance(state, dict):
            return
        z = state.get("z")
        if z is None or not np.isfinite(z):
            return
        z_dot = state.get("z_dot", 0.0)
        center_uv = state.get("center_uv", getattr(seed_track, "filtered_center_uv", None))
        u_dot = state.get("u_dot", 0.0)
        v_dot = state.get("v_dot", 0.0)
        confidence = state.get("confidence", getattr(seed_track, "depth_confidence", 0.0))
        p_zz = None
        p_vv = None
        p_diag = None
        cov = getattr(seed_filter, "P", None)
        if cov is not None:
            cov_arr = np.asarray(cov, dtype=np.float64)
            if cov_arr.ndim == 2 and cov_arr.shape[0] >= 6 and cov_arr.shape[1] >= 6:
                p_diag = [float(cov_arr[i, i]) for i in range(6)]
            elif cov_arr.ndim == 2 and cov_arr.shape[0] >= 2 and cov_arr.shape[1] >= 2:
                p_zz = float(cov_arr[0, 0])
                p_vv = float(cov_arr[1, 1])
        try:
            self.depth_filter.seed_state(
                z=float(z),
                center_uv=center_uv,
                z_dot=float(z_dot) if np.isfinite(z_dot) else 0.0,
                u_dot=float(u_dot) if np.isfinite(u_dot) else 0.0,
                v_dot=float(v_dot) if np.isfinite(v_dot) else 0.0,
                confidence=float(confidence) if np.isfinite(confidence) else 0.0,
                p_diag=p_diag,
                p_zz=p_zz,
                p_vv=p_vv,
            )
        except TypeError:
            self.depth_filter.seed_state(
                z=float(z),
                z_dot=float(z_dot) if np.isfinite(z_dot) else 0.0,
                confidence=float(confidence) if np.isfinite(confidence) else 0.0,
                p_zz=p_zz,
                p_vv=p_vv,
            )

    def _seed_center_filter_from_track(self, seed_track) -> None:
        """Carry the split center filter state across YOLO-only frame adapters."""
        if seed_track is None:
            return
        center = getattr(seed_track, "filtered_center_uv", None)
        if (
            center is None
            or not np.isfinite(center[0])
            or not np.isfinite(center[1])
        ):
            return
        self.filtered_center_uv = (float(center[0]), float(center[1]))
        self.center_ema_initialized = bool(
            getattr(seed_track, "center_ema_initialized", True)
        )
        velocity = getattr(seed_track, "center_ab_velocity_uv", (0.0, 0.0))
        if not (
            np.isfinite(velocity[0])
            and np.isfinite(velocity[1])
        ):
            velocity = (0.0, 0.0)
        self.center_ab_velocity_uv = (float(velocity[0]), float(velocity[1]))
        self.center_ema_prior_uv = self.filtered_center_uv
        self.center_ab_prior_uv = self.filtered_center_uv
        self.center_ab_prior_velocity_uv = self.center_ab_velocity_uv

    def _apply_filter_state(self,
                            state: dict,
                            alpha: float = 1.0,
                            dt_s: float | None = None,
                            *,
                            center_measurement_uv: tuple[float, float] | None = None,
                            replace_center_measurement: bool = False,
                            update_center: bool = True):
        self.depth_confidence = float(state.get("confidence", self.depth_confidence))
        self.depth_valid = bool(state.get("valid", False))
        self.depth_rejected = bool(state.get("rejected", False))
        self.z_dot = float(state.get("z_dot", 0.0))
        self.depth_gain = float(state.get("gain", 0.0))
        self.depth_quality = float(state.get("quality", state.get("confidence", 0.0)))
        self.depth_filter_mode = str(state.get("mode", "legacy"))
        self.depth_r_t = float(state.get("r_t", self.depth_quality))
        self.depth_R_t = float(state.get("R_t", getattr(self, "depth_R_t", 0.0)))
        depth_nis = state.get("nis")
        self.depth_nis = None if depth_nis is None or not np.isfinite(depth_nis) else float(depth_nis)
        depth_innovation = state.get("innovation")
        self.depth_innovation = (
            None
            if depth_innovation is None or not np.isfinite(depth_innovation)
            else float(depth_innovation)
        )
        depth_S_t = state.get("S_t")
        self.depth_S_t = None if depth_S_t is None or not np.isfinite(depth_S_t) else float(depth_S_t)
        depth_P_zz_pred = state.get("P_zz_pred")
        self.depth_P_zz_pred = (
            None
            if depth_P_zz_pred is None or not np.isfinite(depth_P_zz_pred)
            else float(depth_P_zz_pred)
        )
        dt_used = state.get("dt_s", dt_s)
        self.depth_dt_s = (
            float(dt_used)
            if dt_used is not None and np.isfinite(dt_used) and dt_used > 0
            else 0.0
        )

        center_uv = state.get("center_uv")
        if center_uv is not None and np.isfinite(center_uv[0]) and np.isfinite(center_uv[1]):
            self.filtered_center_uv = (float(center_uv[0]), float(center_uv[1]))
            self.center_ema_initialized = True
        elif update_center:
            center_measurement = (
                center_measurement_uv
                if center_measurement_uv is not None
                else self.raw_center_uv
            )
            self._update_center_filter(
                center_measurement,
                dt_s=dt_s,
                replace_current_step=replace_center_measurement,
            )

        if self.output_bbox is not None and self.filtered_center_uv is not None:
            self.output_bbox = self._recenter_bbox_keep_size(
                self.output_bbox,
                self.filtered_center_uv,
            )

        pos_state = state.get("position")
        if pos_state is not None:
            pos_state = np.asarray(pos_state, dtype=np.float32)
            if pos_state.shape[0] >= 3 and np.all(np.isfinite(pos_state[:3])):
                self.pos_3d = float(alpha) * pos_state + (1.0 - float(alpha)) * self.pos_3d
                return

        self._apply_filtered_depth(state.get("z"))

    def _apply_filtered_depth(self, z_filtered):
        if z_filtered is None or not np.isfinite(z_filtered):
            return
        xyz = self._uv_depth_to_xyz(self.filtered_center_uv, float(z_filtered))
        if xyz is not None:
            self.pos_3d = xyz
            return
        raw_z = self.raw_depth
        if raw_z is not None and np.isfinite(raw_z) and raw_z > 1e-6:
            scale = float(z_filtered) / raw_z
            self.pos_3d[:2] = self.raw_position[:2] * scale
        self.pos_3d[2] = z_filtered

    def _uv_depth_to_xyz(self,
                         center_uv: tuple[float, float] | None,
                         z_filtered: float) -> np.ndarray | None:
        if center_uv is None or not np.isfinite(z_filtered):
            return None
        fx = float(self.camera_params.get("fx", np.nan))
        fy = float(self.camera_params.get("fy", fx))
        cx = float(self.camera_params.get("cx", np.nan))
        cy = float(self.camera_params.get("cy", np.nan))
        if not all(np.isfinite(v) for v in (fx, fy, cx, cy, center_uv[0], center_uv[1])):
            return None
        u, v = float(center_uv[0]), float(center_uv[1])
        x = (u - cx) * z_filtered / fx
        y = (v - cy) * z_filtered / fy
        return np.array([x, y, float(z_filtered)], dtype=np.float32)

    def _resolve_center_ema_alpha(self, dt_s: float | None) -> float:
        tau = float(self.center_ema_tau_s)
        if np.isfinite(tau) and tau > 1e-9:
            dt = float(dt_s) if dt_s is not None and np.isfinite(dt_s) and dt_s > 0 else float(self.center_ema_fallback_dt_s)
            dt = max(dt, 1e-9)
            return float(np.clip(1.0 - np.exp(-dt / tau), 0.0, 1.0))
        return float(np.clip(self.center_ema_alpha, 0.0, 1.0))

    def _resolve_center_filter_dt(self, dt_s: float | None) -> float:
        if dt_s is not None and np.isfinite(dt_s) and dt_s > 0:
            return float(dt_s)
        return float(self.center_ema_fallback_dt_s)

    def _update_center_filter(self,
                              center_uv: tuple[float, float] | None,
                              *,
                              dt_s: float | None,
                              replace_current_step: bool) -> None:
        if self.center_filter_type == "alpha_beta" and self.center_ab_enabled:
            self._update_center_alpha_beta(
                center_uv,
                dt_s=dt_s,
                replace_current_step=replace_current_step,
            )
            return
        if self.center_filter_type == "raw":
            if (
                center_uv is None
                or not np.isfinite(center_uv[0])
                or not np.isfinite(center_uv[1])
            ):
                return
            measurement = (float(center_uv[0]), float(center_uv[1]))
            self.filtered_center_uv = measurement
            self.center_ema_initialized = True
            self.center_ab_velocity_uv = (0.0, 0.0)
            return
        self._update_center_ema(
            center_uv,
            dt_s=dt_s,
            replace_current_step=replace_current_step,
        )

    def _update_center_ema(self,
                           center_uv: tuple[float, float] | None,
                           *,
                           dt_s: float | None,
                           replace_current_step: bool) -> None:
        if (
            center_uv is None
            or not np.isfinite(center_uv[0])
            or not np.isfinite(center_uv[1])
        ):
            return
        measurement = (float(center_uv[0]), float(center_uv[1]))

        if replace_current_step:
            # Recompute this frame from the same pre-frame center. On the first
            # frame there is no prior, so the refined center initializes EMA.
            if self.center_ema_prior_uv is None:
                self.filtered_center_uv = measurement
                self.center_ema_initialized = True
                return
            prior = self.center_ema_prior_uv
            a = float(self.center_ema_step_alpha)
        else:
            if not self.center_ema_initialized or self.filtered_center_uv is None:
                self.center_ema_prior_uv = None
                self.center_ema_step_alpha = 1.0
                self.filtered_center_uv = measurement
                self.center_ema_initialized = True
                return
            prior = (
                float(self.filtered_center_uv[0]),
                float(self.filtered_center_uv[1]),
            )
            a = self._resolve_center_ema_alpha(dt_s) if self.center_ema_enabled else 1.0
            self.center_ema_prior_uv = prior
            self.center_ema_step_alpha = float(a)

        self.filtered_center_uv = (
            float((1.0 - a) * prior[0] + a * measurement[0]),
            float((1.0 - a) * prior[1] + a * measurement[1]),
        )
        self.center_ema_initialized = True

    def _update_center_alpha_beta(self,
                                  center_uv: tuple[float, float] | None,
                                  *,
                                  dt_s: float | None,
                                  replace_current_step: bool) -> None:
        if (
            center_uv is None
            or not np.isfinite(center_uv[0])
            or not np.isfinite(center_uv[1])
        ):
            return
        measurement = np.asarray(
            [float(center_uv[0]), float(center_uv[1])],
            dtype=np.float64,
        )

        if replace_current_step:
            if self.center_ab_prior_uv is None or self.center_ab_prior_velocity_uv is None:
                self.filtered_center_uv = (float(measurement[0]), float(measurement[1]))
                self.center_ab_velocity_uv = (0.0, 0.0)
                self.center_ema_initialized = True
                return
            prior = np.asarray(self.center_ab_prior_uv, dtype=np.float64)
            prior_vel = np.asarray(self.center_ab_prior_velocity_uv, dtype=np.float64)
            dt = max(float(self.center_ab_step_dt_s), 1e-9)
            alpha = float(self.center_ab_step_alpha)
            beta = float(self.center_ab_step_beta)
        else:
            if not self.center_ema_initialized or self.filtered_center_uv is None:
                self.center_ab_prior_uv = None
                self.center_ab_prior_velocity_uv = None
                self.center_ab_step_dt_s = self._resolve_center_filter_dt(dt_s)
                self.center_ab_step_alpha = float(self.center_ab_alpha)
                self.center_ab_step_beta = float(self.center_ab_beta)
                self.center_ab_velocity_uv = (0.0, 0.0)
                self.filtered_center_uv = (float(measurement[0]), float(measurement[1]))
                self.center_ema_initialized = True
                return
            prior = np.asarray(self.filtered_center_uv, dtype=np.float64)
            prior_vel = np.asarray(self.center_ab_velocity_uv, dtype=np.float64)
            dt = max(self._resolve_center_filter_dt(dt_s), 1e-9)
            alpha = float(self.center_ab_alpha)
            beta = float(self.center_ab_beta)
            self.center_ab_prior_uv = (float(prior[0]), float(prior[1]))
            self.center_ab_prior_velocity_uv = (
                float(prior_vel[0]),
                float(prior_vel[1]),
            )
            self.center_ab_step_dt_s = float(dt)
            self.center_ab_step_alpha = float(alpha)
            self.center_ab_step_beta = float(beta)

        pred = prior + dt * prior_vel
        residual = measurement - pred
        updated = pred + alpha * residual
        updated_vel = prior_vel + (beta / dt) * residual

        self.filtered_center_uv = (float(updated[0]), float(updated[1]))
        self.center_ab_velocity_uv = (
            float(updated_vel[0]),
            float(updated_vel[1]),
        )
        self.center_ema_initialized = True

    def replace_current_center_measurement(self,
                                           center_uv: tuple[float, float],
                                           *,
                                           dt_s: float | None = None) -> None:
        """Replace this frame's EMA input even when depth extraction failed."""
        self.raw_center_uv = (float(center_uv[0]), float(center_uv[1]))
        self._update_center_filter(
            self.raw_center_uv,
            dt_s=dt_s,
            replace_current_step=True,
        )
        if self.output_bbox is not None and self.filtered_center_uv is not None:
            self.output_bbox = self._recenter_bbox_keep_size(
                self.output_bbox,
                self.filtered_center_uv,
            )
        current_z = self._extract_z(self.pos_3d)
        if current_z is not None:
            xyz = self._uv_depth_to_xyz(self.filtered_center_uv, current_z)
            if xyz is not None:
                self.pos_3d = xyz

    @staticmethod
    def _recenter_bbox_keep_size(bbox, center_uv: tuple[float, float]):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cu, cv = float(center_uv[0]), float(center_uv[1])
        return [
            float(cu - 0.5 * w),
            float(cv - 0.5 * h),
            float(cu + 0.5 * w),
            float(cv + 0.5 * h),
        ]

    def correct_current_measurement(self, pos_3d, *,
                                    depth_stats=None,
                                    depth_confidence=None,
                                    alpha: float = 0.7,
                                    dt_s: float | None = None,
                                    center_uv: tuple[float, float] | None = None):
        """Refine the current frame's 3D estimate without advancing time twice."""
        self.raw_position = np.array(pos_3d, dtype=np.float32)
        self.raw_center_uv = (
            (float(center_uv[0]), float(center_uv[1]))
            if center_uv is not None
            else self._bbox_center_xyxy(self.raw_bbox)
        )
        self.depth_stats = depth_stats
        self.raw_depth = self._extract_z(self.raw_position)
        self.depth_confidence = (
            float(depth_confidence) if depth_confidence is not None else
            self._stats_confidence(depth_stats)
        )
        state = self.depth_filter.correct_only(
            self.raw_depth,
            depth_stats=depth_stats,
            depth_confidence=self.depth_confidence,
            dt_s=dt_s,
            center_uv=self.raw_center_uv,
        )
        self._apply_filter_state(
            state,
            alpha=alpha,
            dt_s=dt_s,
            center_measurement_uv=self.raw_center_uv,
            replace_center_measurement=True,
        )

    def depth_is_plausible(self, min_depth_m: float) -> bool:
        z = self._extract_z(self.pos_3d)
        if z is None or not np.isfinite(z):
            # Allow a just-detected bbox without depth to survive for the
            # current frame only; once it becomes memory-only, drop it.
            return int(getattr(self, "time_since_update", 1)) <= 0
        return z > float(min_depth_m)


class AdaptiveBBoxOutputFilter:
    """Lightweight adaptive filter for the final emitted bbox only."""

    def __init__(self, cfg: OutputBBoxFilterConfig):
        self.cfg = cfg
        self._states: dict[int, list[float]] = {}

    def reset(self) -> None:
        self._states.clear()

    def apply(self, tracks: list[FishTrack], dt_s: float | None = None) -> list[FishTrack]:
        if not tracks:
            self.reset()
            return tracks

        if not self.cfg.enabled:
            for track in tracks:
                track.output_bbox = list(track.bbox)
            self._cleanup({int(track.id) for track in tracks})
            return tracks

        dt = self._resolve_dt(dt_s)
        active_ids: set[int] = set()
        reset_after = max(int(self.cfg.reset_after_missed_frames), 0)

        for track in tracks:
            track_id = int(track.id)
            active_ids.add(track_id)
            new_bbox = [float(v) for v in track.bbox]
            prev_bbox = self._states.get(track_id)

            if prev_bbox is None or int(getattr(track, "time_since_update", 0)) > reset_after:
                filtered_bbox = new_bbox
            else:
                pcx, pcy, pw, ph = self._bbox_to_cxcywh(prev_bbox)
                ncx, ncy, nw, nh = self._bbox_to_cxcywh(new_bbox)
                center_speed = float(np.hypot(ncx - pcx, ncy - pcy)) / dt
                size_speed = float(max(abs(nw - pw), abs(nh - ph))) / dt
                center_alpha = self._adaptive_alpha(
                    center_speed,
                    min_alpha=self.cfg.center_alpha_min,
                    max_alpha=self.cfg.center_alpha_max,
                    scale=max(float(self.cfg.velocity_scale_px_per_s), 1e-6),
                )
                size_alpha = self._adaptive_alpha(
                    size_speed,
                    min_alpha=self.cfg.size_alpha_min,
                    max_alpha=self.cfg.size_alpha_max,
                    scale=max(float(self.cfg.size_velocity_scale_px_per_s), 1e-6),
                )
                if int(getattr(track, "time_since_update", 0)) > 0:
                    center_alpha *= float(np.clip(self.cfg.tracker_only_alpha_scale, 0.0, 1.0))
                    size_alpha *= float(np.clip(self.cfg.tracker_only_alpha_scale, 0.0, 1.0))
                filtered_bbox = FishTrack._smooth_bbox(
                    prev_bbox,
                    new_bbox,
                    center_alpha=center_alpha,
                    size_alpha=size_alpha,
                )

            track.output_bbox = list(filtered_bbox)
            self._states[track_id] = list(filtered_bbox)

        self._cleanup(active_ids)
        return tracks

    def _cleanup(self, active_ids: set[int]) -> None:
        stale_ids = [track_id for track_id in self._states if track_id not in active_ids]
        for track_id in stale_ids:
            self._states.pop(track_id, None)

    def _resolve_dt(self, dt_s: float | None) -> float:
        if dt_s is None or not np.isfinite(dt_s) or dt_s <= 1e-6:
            return max(float(self.cfg.fallback_dt_s), 1e-3)
        return float(dt_s)

    @staticmethod
    def _bbox_to_cxcywh(bbox: list[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return cx, cy, w, h

    @staticmethod
    def _adaptive_alpha(speed: float, *, min_alpha: float, max_alpha: float, scale: float) -> float:
        min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
        max_alpha = float(np.clip(max_alpha, min_alpha, 1.0))
        gain = 1.0 - np.exp(-max(float(speed), 0.0) / max(float(scale), 1e-6))
        return float(min_alpha + (max_alpha - min_alpha) * gain)


class FishTracker:
    """
    Simple online tracker.

    - IoU-based association between existing tracks and new detections.
    - Hungarian matching for optimal assignment.
    - Track birth after first detection, confirmation after *min_hits*.
    - Track death after *max_age* frames without a match.
    """

    def __init__(self,
                 max_age: int = 10,
                 min_hits: int = 2,
                 iou_threshold: float = 0.3,
                 smoothing_alpha: float = 0.7,
                 bbox_smoothing_enabled: bool = True,
                 bbox_center_alpha: float = 0.35,
                 bbox_size_alpha: float = 0.30,
                 temporal_depth: dict | None = None):
        self.tracks: list[FishTrack] = []
        self._next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.bbox_smoothing_enabled = bool(bbox_smoothing_enabled)
        self.bbox_center_alpha = float(bbox_center_alpha)
        self.bbox_size_alpha = float(bbox_size_alpha)
        self.temporal_depth_cfg = temporal_depth or {}
        self.memory_min_depth_m = float(
            self.temporal_depth_cfg.get("roi", {}).get("min_depth_m", 0.1)
        )
        self._frame_count = 0

    def update(self, detections: list[dict],
               frame: np.ndarray | None = None,
               dt_s: float | None = None) -> list[FishTrack]:
        """
        Parameters
        ----------
        detections : list of dict
            Each dict has keys ``bbox`` (x1,y1,x2,y2), ``pos_3d`` (x,y,z),
            and ``confidence``.

        Returns
        -------
        list[FishTrack]
            Confirmed tracks (each has ``id``, ``bbox``, ``pos_3d``, ``confidence``).
        """
        self._frame_count += 1

        for t in self.tracks:
            t.predict()

        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # Update matched pairs
        for t_idx, d_idx in matched:
            det = detections[d_idx]
            self.tracks[t_idx].update(
                det["bbox"], det["pos_3d"], det["confidence"],
                self._frame_count, alpha=self.smoothing_alpha,
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source=det.get("source", "yolo"),
                bbox_smoothing_enabled=self.bbox_smoothing_enabled,
                bbox_center_alpha=self.bbox_center_alpha,
                bbox_size_alpha=self.bbox_size_alpha,
            )

        # Prediction-only depth update for tracks that were not observed.
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].predict_depth_only(dt_s)
            if not self.tracks[t_idx].depth_is_plausible(self.memory_min_depth_m):
                self.tracks[t_idx].time_since_update = self.max_age + 1

        # Birth
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            self.tracks.append(FishTrack(
                self._next_id, det["bbox"], det["pos_3d"],
                det["confidence"], self._frame_count,
                temporal_depth_cfg=self.temporal_depth_cfg,
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source=det.get("source", "yolo"),
            ))
            self._next_id += 1

        # Death
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]

    # ── private ────────────────────────────────────────────────────────

    def _associate(self, detections):
        if not self.tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.tracks)))

        n_tracks = len(self.tracks)
        n_dets = len(detections)
        iou = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                iou[ti, di] = self._box_iou(t.bbox, d["bbox"])

        cost = 1.0 - iou
        row_idx, col_idx = linear_sum_assignment(cost)

        matched = []
        unmatched_dets = set(range(n_dets))
        unmatched_tracks = set(range(n_tracks))

        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.discard(c)
                unmatched_tracks.discard(r)

        return matched, list(unmatched_dets), list(unmatched_tracks)

    @staticmethod
    def _box_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 1e-12 else 0.0


class YOLOOnlyFishTracker:
    """Stateless per-frame output adapter for pure YOLO detections."""

    STATE_NO_TARGET = "NO_TARGET"
    STATE_YOLO_ONLY = "YOLO_ONLY"

    def __init__(self,
                 max_age: int = 0,
                 temporal_depth: dict | None = None):
        self.max_age = int(max_age)
        self.temporal_depth_cfg = temporal_depth or {}
        self.temporal_state_hold_max_s = max(
            float(self.temporal_depth_cfg.get("state_hold_max_s", 0.6)),
            0.0,
        )
        self.tracks: list[FishTrack] = []
        self.track: FishTrack | None = None
        self._temporal_track: FishTrack | None = None
        self._temporal_gap_s = 0.0
        self._frame_count = 0
        self._state = self.STATE_NO_TARGET

    @property
    def state(self) -> str:
        return str(self._state)

    def update(self, detections: list[dict],
               frame: np.ndarray | None = None,
               dt_s: float | None = None) -> list[FishTrack]:
        del frame
        self._frame_count += 1

        if not detections:
            if self._temporal_track is not None:
                dt = (
                    float(dt_s)
                    if dt_s is not None and np.isfinite(dt_s) and dt_s > 0
                    else float(self.temporal_depth_cfg.get("fallback_dt_s", 0.1))
                )
                self._temporal_gap_s += dt
                if self._temporal_gap_s <= self.temporal_state_hold_max_s:
                    self._temporal_track.predict_temporal_only(dt)
                else:
                    self._temporal_track = None
            self.tracks = []
            self.track = None
            self._state = self.STATE_NO_TARGET
            return []

        new_tracks: list[FishTrack] = []
        for idx, det in enumerate(detections):
            depth_stats = det.get("depth_stats")
            depth_confidence = det.get("depth_confidence")
            source = det.get("source", "yolo")
            pos_3d = np.asarray(det["pos_3d"], dtype=np.float32)
            prev_track = self._temporal_track if idx == 0 else None

            track = FishTrack(
                idx,
                det["bbox"],
                pos_3d,
                det["confidence"],
                self._frame_count,
                temporal_depth_cfg=self.temporal_depth_cfg,
                depth_stats=depth_stats,
                depth_confidence=depth_confidence,
                dt_s=dt_s,
                source=source,
                seed_track=prev_track,
            )
            track.output_bbox = list(det["bbox"])
            track.raw_bbox = list(det["bbox"])
            new_tracks.append(track)

        self.tracks = new_tracks
        self.track = new_tracks[0] if new_tracks else None
        self._temporal_track = self.track
        self._temporal_gap_s = 0.0
        self._state = self.STATE_YOLO_ONLY if new_tracks else self.STATE_NO_TARGET
        return list(new_tracks)


class _ByteTrackDetections:
    """Minimal Results-like wrapper for Ultralytics BYTETracker."""

    def __init__(self, xywh, conf, cls=None):
        xywh = np.asarray(xywh, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        if xywh.size == 0:
            xywh = xywh.reshape(0, 4)
        else:
            xywh = xywh.reshape(-1, 4)
        if conf.size == 0:
            conf = conf.reshape(0)
        else:
            conf = conf.reshape(-1)
        if cls is None:
            cls = np.zeros(len(conf), dtype=np.float32)
        else:
            cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _ByteTrackDetections(self.xywh[idx], self.conf[idx], self.cls[idx])


class ByteTrackFishTracker:
    """Ultralytics BYTETracker wrapper with FishTrack depth state."""

    STATE_NO_TARGET = "NO_TARGET"
    STATE_TRACKING_BYTETRACK = "TRACKING_BYTETRACK"
    STATE_MEMORY_ONLY = "MEMORY_ONLY"

    def __init__(self,
                 max_age: int = 20,
                 min_hits: int = 1,
                 iou_threshold: float = 0.3,
                 smoothing_alpha: float = 0.7,
                 bbox_smoothing_enabled: bool = True,
                 bbox_center_alpha: float = 0.35,
                 bbox_size_alpha: float = 0.30,
                 temporal_depth: dict | None = None,
                 track_high_thresh: float = 0.35,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.35,
                 track_buffer: int = 20,
                 match_thresh: float = 0.8,
                 fuse_score: bool = True,
                 single_target_force_association: bool = False,
                 single_target_output_only: bool = False,
                 sync_bbox_on_miss: bool = False,
                 external_bbox_writeback: bool = False,
                 force_assoc_min_confidence: float = 0.45,
                 force_assoc_edge_margin_px: int = 48,
                 force_assoc_min_iou: float = 0.02,
                 force_assoc_max_center_distance_ratio: float = 1.6,
                 force_assoc_min_area_ratio: float = 0.3,
                 force_assoc_max_area_ratio: float = 3.5):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.smoothing_alpha = float(smoothing_alpha)
        self.bbox_smoothing_enabled = bool(bbox_smoothing_enabled)
        self.bbox_center_alpha = float(bbox_center_alpha)
        self.bbox_size_alpha = float(bbox_size_alpha)
        self.temporal_depth_cfg = temporal_depth or {}
        self.memory_min_depth_m = float(
            self.temporal_depth_cfg.get("roi", {}).get("min_depth_m", 0.1)
        )
        self.track_buffer = max(int(track_buffer), self.max_age)
        self.single_target_force_association = bool(single_target_force_association)
        self.single_target_output_only = bool(single_target_output_only)
        self.sync_bbox_on_miss = bool(sync_bbox_on_miss)
        self.external_bbox_writeback = bool(external_bbox_writeback)
        self.force_assoc_min_confidence = float(force_assoc_min_confidence)
        self.force_assoc_edge_margin_px = int(force_assoc_edge_margin_px)
        self.force_assoc_min_iou = float(force_assoc_min_iou)
        self.force_assoc_max_center_distance_ratio = float(
            force_assoc_max_center_distance_ratio)
        self.force_assoc_min_area_ratio = float(force_assoc_min_area_ratio)
        self.force_assoc_max_area_ratio = float(force_assoc_max_area_ratio)
        self._frame_count = 0
        self._state = self.STATE_NO_TARGET
        self._tracks_by_id: dict[int, FishTrack] = {}

        args = SimpleNamespace(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=self.track_buffer,
            match_thresh=float(match_thresh),
            fuse_score=bool(fuse_score),
        )
        self._tracker = BYTETracker(args)

    def update(self, detections: list[dict],
               frame: np.ndarray | None = None,
               dt_s: float | None = None) -> list[FishTrack]:
        self._frame_count += 1

        if frame is None:
            frame = np.zeros((2, 2, 3), dtype=np.uint8)

        for track in self._tracks_by_id.values():
            track.predict()

        results = self._to_bytetrack_detections(detections)
        outputs = self._tracker.update(results, img=frame)
        output_rows = self._collect_output_rows(outputs, detections)
        active_ids: set[int] = set()

        for row in output_rows:
            x1, y1, x2, y2, track_id, score, _cls, det_idx = row
            track_id = int(track_id)
            det_idx = int(det_idx)
            if det_idx < 0 or det_idx >= len(detections):
                continue

            det = detections[det_idx]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            active_ids.add(track_id)

            if track_id not in self._tracks_by_id:
                seed_track = self._find_depth_seed_track(
                    bbox,
                    exclude_track_ids=active_ids | {track_id},
                )
                self._tracks_by_id[track_id] = FishTrack(
                    track_id,
                    bbox,
                    det["pos_3d"],
                    float(score),
                    self._frame_count,
                    temporal_depth_cfg=self.temporal_depth_cfg,
                    depth_stats=det.get("depth_stats"),
                    depth_confidence=det.get("depth_confidence"),
                    dt_s=dt_s,
                    source=det.get("source", "bytetrack"),
                    seed_track=seed_track,
                )
            else:
                self._tracks_by_id[track_id].update(
                    bbox,
                    det["pos_3d"],
                    float(score),
                    self._frame_count,
                    alpha=self.smoothing_alpha,
                    depth_stats=det.get("depth_stats"),
                    depth_confidence=det.get("depth_confidence"),
                    dt_s=dt_s,
                    source=det.get("source", "bytetrack"),
                    bbox_smoothing_enabled=self.bbox_smoothing_enabled,
                    bbox_center_alpha=self.bbox_center_alpha,
                    bbox_size_alpha=self.bbox_size_alpha,
                )

        rescued_track_id = self._maybe_force_single_track_association(
            detections,
            frame=frame,
            dt_s=dt_s,
        )
        if rescued_track_id is not None:
            active_ids.add(int(rescued_track_id))

        stale_ids = []
        for track_id, track in self._tracks_by_id.items():
            if track_id in active_ids:
                continue
            if self.sync_bbox_on_miss:
                self._sync_track_bbox_from_strack(
                    track_id, track, frame_shape=frame.shape)
            track.predict_depth_only(dt_s)
            if (
                track.time_since_update > self.max_age
                or not track.depth_is_plausible(self.memory_min_depth_m)
            ):
                stale_ids.append(track_id)

        removed_ids = {
            int(getattr(track, "track_id", -1))
            for track in getattr(self._tracker, "removed_stracks", [])
        }
        stale_ids.extend(
            track_id for track_id in removed_ids
            if track_id > 0 and track_id in self._tracks_by_id
        )

        for track_id in stale_ids:
            self._tracks_by_id.pop(track_id, None)

        confirmed = [
            track for track in self._tracks_by_id.values()
            if track.is_confirmed(self.min_hits)
            and track.time_since_update <= self.max_age
            and track.depth_is_plausible(self.memory_min_depth_m)
        ]

        if active_ids:
            self._set_state(self.STATE_TRACKING_BYTETRACK)
        elif confirmed:
            self._set_state(self.STATE_MEMORY_ONLY)
        else:
            self._set_state(self.STATE_NO_TARGET)

        confirmed.sort(key=lambda t: t.id)
        return confirmed

    def _find_depth_seed_track(self,
                               bbox,
                               *,
                               exclude_track_ids: set[int] | None = None) -> FishTrack | None:
        exclude_track_ids = {int(tid) for tid in (exclude_track_ids or set())}
        best_track = None
        best_score = None
        for track in self._tracks_by_id.values():
            if int(track.id) in exclude_track_ids:
                continue
            z = FishTrack._extract_z(track.pos_3d)
            if z is None or not np.isfinite(z):
                continue
            if int(getattr(track, "time_since_update", self.max_age + 1)) > max(self.max_age, 1):
                continue
            iou = FishTracker._box_iou(track.bbox, bbox)
            center_dist_ratio = self._bbox_center_distance_ratio(track.bbox, bbox)
            if iou <= 0.0 and center_dist_ratio > 1.25:
                continue
            score = (-float(iou), float(center_dist_ratio), int(track.time_since_update))
            if best_score is None or score < best_score:
                best_score = score
                best_track = track
        return best_track

    def predict_detections(self, frame: np.ndarray,
                           dt_s: float | None = None) -> list[dict]:
        # ByteTrack is only used for association / short-term memory in the
        # single-target pipeline. It should not synthesize a new visible
        # detection, otherwise the output semantics drift away from
        # YOLO/YOLO+corrector as the unique main box source.
        return []

    @property
    def tracks(self) -> list[FishTrack]:
        return sorted(self._tracks_by_id.values(), key=lambda t: t.id)

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str) -> None:
        self._state = str(state)

    def _maybe_force_single_track_association(self,
                                              detections: list[dict],
                                              *,
                                              frame: np.ndarray,
                                              dt_s: float | None) -> int | None:
        if not self.single_target_force_association:
            return None
        if len(detections) != 1 or len(self._tracks_by_id) != 1:
            return None

        track = next(iter(self._tracks_by_id.values()))
        if track.time_since_update <= 0 or track.time_since_update > self.max_age:
            return None

        det = detections[0]
        bbox = det.get("bbox")
        if bbox is None:
            return None

        edge_margin_px = self.force_assoc_edge_margin_px
        if not (
            self._bbox_near_edge(track.bbox, frame.shape, margin_px=edge_margin_px)
            or self._bbox_near_edge(bbox, frame.shape, margin_px=edge_margin_px)
        ):
            return None

        conf = float(det.get("confidence", 0.0))
        if conf < self.force_assoc_min_confidence:
            return None

        iou = FishTracker._box_iou(track.bbox, bbox)
        min_iou = self.force_assoc_min_iou
        area_ratio = self._bbox_area_ratio(track.bbox, bbox)
        min_area_ratio = self.force_assoc_min_area_ratio
        max_area_ratio = self.force_assoc_max_area_ratio
        center_distance_ratio = self._bbox_center_distance_ratio(track.bbox, bbox)
        max_center_distance_ratio = self.force_assoc_max_center_distance_ratio

        if iou < min_iou and center_distance_ratio > max_center_distance_ratio:
            return None
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return None

        track.update(
            bbox,
            det["pos_3d"],
            conf,
            self._frame_count,
            alpha=self.smoothing_alpha,
            depth_stats=det.get("depth_stats"),
            depth_confidence=det.get("depth_confidence"),
            dt_s=dt_s,
            source=f"{det.get('source', 'yolo')}+force",
            bbox_smoothing_enabled=self.bbox_smoothing_enabled,
            bbox_center_alpha=self.bbox_center_alpha,
            bbox_size_alpha=self.bbox_size_alpha,
        )
        if self.external_bbox_writeback:
            self.apply_external_bbox(track.id, bbox, score=conf)
        return track.id

    def _collect_output_rows(self,
                             outputs,
                             detections: list[dict]) -> list[list[float]]:
        rows: list[list[float]] = []
        seen_ids: set[int] = set()

        if outputs is not None:
            arr = np.asarray(outputs)
            if arr.size > 0:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                for row in arr.tolist():
                    if len(row) < 8:
                        continue
                    rows.append(row)
                    seen_ids.add(int(row[4]))

        # ByteTrack does not mark newborn tracks as activated unless they are
        # created on frame 1, so later re-acquired targets can exist internally
        # but still be absent from the formatted outputs for one frame.
        current_frame_id = int(getattr(self._tracker, "frame_id", self._frame_count))
        for strack in getattr(self._tracker, "tracked_stracks", []):
            track_id = int(getattr(strack, "track_id", -1))
            if track_id <= 0 or track_id in seen_ids:
                continue
            if bool(getattr(strack, "is_activated", False)):
                continue
            if int(getattr(strack, "start_frame", -1)) != current_frame_id:
                continue
            det_idx = int(getattr(strack, "idx", -1))
            if det_idx < 0 or det_idx >= len(detections):
                continue
            result = getattr(strack, "result", None)
            if result is None:
                continue
            row = np.asarray(result, dtype=np.float32).reshape(-1)
            if row.size < 8:
                continue
            # Promote this newborn track so ByteTrack keeps it in the normal
            # tracked/lost lifecycle instead of dropping it as "unconfirmed"
            # on the very next frame.
            strack.is_activated = True
            rows.append(row.tolist())
            seen_ids.add(track_id)

        return rows

    @staticmethod
    def _bbox_to_xywh_center(bbox) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return [
            x1 + 0.5 * w,
            y1 + 0.5 * h,
            w,
            h,
        ]

    @staticmethod
    def _bbox_area_ratio(prev_bbox, new_bbox) -> float:
        px1, py1, px2, py2 = [float(v) for v in prev_bbox]
        nx1, ny1, nx2, ny2 = [float(v) for v in new_bbox]
        prev_area = max((px2 - px1) * (py2 - py1), 1.0)
        new_area = max((nx2 - nx1) * (ny2 - ny1), 1.0)
        return float(new_area / prev_area)

    @staticmethod
    def _bbox_center_distance_ratio(prev_bbox, new_bbox) -> float:
        px1, py1, px2, py2 = [float(v) for v in prev_bbox]
        nx1, ny1, nx2, ny2 = [float(v) for v in new_bbox]
        pcx = 0.5 * (px1 + px2)
        pcy = 0.5 * (py1 + py2)
        ncx = 0.5 * (nx1 + nx2)
        ncy = 0.5 * (ny1 + ny2)
        prev_scale = max(px2 - px1, py2 - py1, 1.0)
        new_scale = max(nx2 - nx1, ny2 - ny1, 1.0)
        denom = max(prev_scale, new_scale, 1.0)
        return float(np.hypot(ncx - pcx, ncy - pcy) / denom)

    @staticmethod
    def _bbox_near_edge(bbox,
                        frame_shape,
                        *,
                        margin_px: int) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dist = min(x1, y1, max(0.0, width - x2), max(0.0, height - y2))
        return dist <= float(max(margin_px, 0))

    @staticmethod
    def _bbox_near_edge(bbox,
                        frame_shape,
                        *,
                        margin_px: int) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dist = min(x1, y1, max(0.0, width - x2), max(0.0, height - y2))
        return dist <= float(max(margin_px, 0))

    def _to_bytetrack_detections(self, detections: list[dict]) -> _ByteTrackDetections:
        if not detections:
            return _ByteTrackDetections([], [])
        xywh = [self._bbox_to_xywh_center(det["bbox"]) for det in detections]
        conf = [float(det.get("confidence", 0.0)) for det in detections]
        cls = np.zeros(len(detections), dtype=np.float32)
        return _ByteTrackDetections(xywh, conf, cls)

    @staticmethod
    def _bbox_to_tlwh(bbox) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return np.asarray(
            [
                x1,
                y1,
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
            ],
            dtype=np.float32,
        )

    def _find_strack_by_id(self, track_id: int):
        for pool in (
            self._tracker.tracked_stracks,
            self._tracker.lost_stracks,
        ):
            for strack in pool:
                if int(getattr(strack, "track_id", -1)) == int(track_id):
                    return strack
        return None

    def _predicted_bbox_for_track(self,
                                  track_id: int,
                                  *,
                                  frame_shape=None) -> list[float] | None:
        strack = self._find_strack_by_id(track_id)
        bbox = None
        if strack is not None:
            bbox = np.asarray(getattr(strack, "xyxy", None), dtype=np.float32).reshape(-1)
            if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
                bbox = bbox[:4].tolist()
            else:
                bbox = None
        if bbox is None:
            track = self._tracks_by_id.get(int(track_id))
            if track is None:
                return None
            bbox = [float(v) for v in track.bbox]
        return self._clamp_bbox_xyxy(bbox, frame_shape=frame_shape)

    def _sync_track_bbox_from_strack(self,
                                     track_id: int,
                                     track: FishTrack,
                                     *,
                                     frame_shape=None) -> None:
        bbox = self._predicted_bbox_for_track(track_id, frame_shape=frame_shape)
        if bbox is None:
            return
        track.raw_bbox = list(bbox)
        track.bbox = list(bbox)
        track.output_bbox = list(bbox)

    @staticmethod
    def _clamp_bbox_xyxy(bbox, *, frame_shape=None, min_size_px: float = 4.0) -> list[float] | None:
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if frame_shape is not None:
            height, width = frame_shape[:2]
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(1.0, min(float(width), x2))
            y2 = max(1.0, min(float(height), y2))
        if not all(np.isfinite([x1, y1, x2, y2])):
            return None
        if (x2 - x1) < float(min_size_px) or (y2 - y1) < float(min_size_px):
            return None
        return [x1, y1, x2, y2]

    def apply_external_bbox(self,
                            track_id: int,
                            bbox,
                            *,
                            score: float | None = None) -> bool:
        """Write an externally refined bbox back into ByteTrack's Kalman state."""
        strack = self._find_strack_by_id(track_id)
        if strack is None:
            return False

        tlwh = self._bbox_to_tlwh(bbox)
        if getattr(strack, "kalman_filter", None) is not None and \
                getattr(strack, "mean", None) is not None and \
                getattr(strack, "covariance", None) is not None:
            measurement = strack.convert_coords(tlwh)
            strack.mean, strack.covariance = strack.kalman_filter.update(
                strack.mean,
                strack.covariance,
                measurement,
            )
        strack._tlwh = tlwh
        strack.is_activated = True
        if score is not None:
            strack.score = float(score)
        return True


class CSRTFishTracker:
    """
    Single-target tracker using OpenCV CSRT between YOLO corrections.

    CSRT is useful for single-object range estimation because it can keep a box
    alive through short YOLO dropouts, letting the stereo/depth stage continue.
    """

    STATE_NO_TARGET = "NO_TARGET"
    STATE_TRACKING_CSRT = "TRACKING_CSRT"
    STATE_MEMORY_ONLY = "MEMORY_ONLY"

    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 1,
                 iou_threshold: float = 0.3,
                 smoothing_alpha: float = 0.7,
                 bbox_smoothing_enabled: bool = True,
                 bbox_center_alpha: float = 0.35,
                 bbox_size_alpha: float = 0.30,
                 temporal_depth: dict | None = None,
                 csrt_reinit_interval: int = 10,
                 csrt_iou_reset_threshold: float = 0.2,
                 csrt_init_min_confidence: float = 0.3,
                 csrt_init_min_hits: int = 2,
                 csrt_init_min_depth_confidence: float = 0.2,
                 init_min_bbox_area_ratio: float = 0.002,
                 init_max_bbox_area_ratio: float = 0.18,
                 init_min_bbox_aspect: float = 0.2,
                 init_max_bbox_aspect: float = 5.0,
                 csrt_max_tracker_only_frames: int = 15,
                 csrt_run_stereo_on_prediction: bool = False,
                 stereo_every_n_yolo_frames: int = 3,
                 stereo_every_n_tracker_frames: int = 5,
                 prefer_tracker_when_active: bool = True,
                 yolo_correction_min_confidence: float = 0.3,
                 yolo_correction_iou_threshold: float = 0.35,
                 yolo_correction_interval: int = 5,
                 max_bbox_area_growth: float = 2.2,
                 min_bbox_area_shrink: float = 0.35,
                 max_bbox_center_jump: float = 2.0):
        self.track: FishTrack | None = None
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.smoothing_alpha = float(smoothing_alpha)
        self.bbox_smoothing_enabled = bool(bbox_smoothing_enabled)
        self.bbox_center_alpha = float(bbox_center_alpha)
        self.bbox_size_alpha = float(bbox_size_alpha)
        self.temporal_depth_cfg = temporal_depth or {}
        self.csrt_reinit_interval = max(int(csrt_reinit_interval), 1)
        self.csrt_iou_reset_threshold = float(csrt_iou_reset_threshold)
        self.csrt_init_min_confidence = float(csrt_init_min_confidence)
        self.csrt_init_min_hits = max(int(csrt_init_min_hits), 1)
        self.csrt_init_min_depth_confidence = float(
            csrt_init_min_depth_confidence)
        self.init_min_bbox_area_ratio = float(init_min_bbox_area_ratio)
        self.init_max_bbox_area_ratio = float(init_max_bbox_area_ratio)
        self.init_min_bbox_aspect = float(init_min_bbox_aspect)
        self.init_max_bbox_aspect = float(init_max_bbox_aspect)
        self.csrt_max_tracker_only_frames = max(
            int(csrt_max_tracker_only_frames), 1)
        self.csrt_run_stereo_on_prediction = bool(csrt_run_stereo_on_prediction)
        self.stereo_every_n_yolo_frames = max(int(stereo_every_n_yolo_frames), 1)
        self.stereo_every_n_tracker_frames = max(
            int(stereo_every_n_tracker_frames), 1)
        self.prefer_tracker_when_active = bool(prefer_tracker_when_active)
        self.yolo_correction_min_confidence = float(
            yolo_correction_min_confidence)
        self.yolo_correction_iou_threshold = float(
            yolo_correction_iou_threshold)
        self.yolo_correction_interval = max(int(yolo_correction_interval), 1)
        self.max_bbox_area_growth = float(max_bbox_area_growth)
        self.min_bbox_area_shrink = float(min_bbox_area_shrink)
        self.max_bbox_center_jump = float(max_bbox_center_jump)
        self._frame_count = 0
        self._next_id = 0
        self._tracker = None
        self._tracker_ok = False
        self._last_init_frame = -10**9
        self._pending_init_bbox = None
        self._pending_init_hits = 0
        self._tracker_only_frames = 0
        self._state = self.STATE_NO_TARGET

    def prefer_active_detections(self, frame: np.ndarray,
                                 yolo_detections: list[dict]) -> list[dict]:
        """Use CSRT as the primary box once a single target is locked."""
        if not self.prefer_tracker_when_active or self.track is None:
            return yolo_detections

        tracker_detections = self.predict_detections(frame)
        if not tracker_detections:
            return yolo_detections

        tracker_det = tracker_detections[0]
        self._set_state(self.STATE_TRACKING_CSRT)
        correction = self._select_yolo_correction(
            yolo_detections, tracker_det["bbox"])
        if correction is not None:
            tracker_det["confidence"] = max(
                float(tracker_det.get("confidence", 0.0)),
                float(correction.get("confidence", 0.0)),
            )
            tracker_det["source"] = "csrt+yolo"
            tracker_det["yolo_verified"] = True
            if self._should_apply_yolo_correction():
                self._init_tracker(frame, correction["bbox"])

        return [tracker_det]

    def predict_detections(self, frame: np.ndarray,
                           dt_s: float | None = None) -> list[dict]:
        """Return a pseudo detection from CSRT when YOLO misses."""
        if self._tracker is None or self.track is None:
            return []
        ok, xywh = self._tracker.update(frame)
        if not ok:
            self._tracker_ok = False
            return []
        self._tracker_ok = True
        bbox = self._xywh_to_xyxy(xywh, frame.shape[1], frame.shape[0])
        if not self._bbox_is_plausible(bbox, self.track.bbox):
            self._reset()
            return []
        self._set_state(self.STATE_TRACKING_CSRT)
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        return [{
            "bbox": bbox,
            "center": (cx, cy),
            "pos_3d": self.track.pos_3d.copy(),
            "depth_stats": self.track.depth_stats,
            "depth_confidence": 0.0,
            "confidence": float(self.track.confidence) * 0.8,
            "tracker_predicted": True,
            "source": "csrt",
        }]

    def filter_detections_for_stereo(self, detections: list[dict],
                                     frame_shape=None) -> list[dict]:
        """Gate new YOLO detections before expensive stereo inference."""
        if not detections:
            return []
        if all(det.get("tracker_predicted", False) for det in detections):
            return detections
        if self.track is not None:
            plausible = [
                det for det in detections
                if self._bbox_is_plausible(det["bbox"], self.track.bbox)
            ]
            return [self._select_single_detection(plausible)] if plausible else []

        candidates = [
            det for det in detections
            if self._bbox_passes_init_geometry(det["bbox"], frame_shape)
        ]
        if not candidates:
            self._pending_init_bbox = None
            self._pending_init_hits = 0
            return []

        det = dict(self._select_single_detection(candidates))
        if not self._accept_initial_detection(det):
            return []
        det["_csrt_init_preaccepted"] = True
        return [det]

    def update(self, detections: list[dict],
               frame: np.ndarray | None = None,
               dt_s: float | None = None) -> list[FishTrack]:
        self._frame_count += 1

        if not detections:
            if self.track is not None:
                self.track.predict()
                self.track.predict_depth_only(dt_s)
                self._tracker_only_frames += 1
                if self._tracker_only_frames > self.csrt_max_tracker_only_frames:
                    self._reset()
                    return []
                self._set_state(self.STATE_MEMORY_ONLY)
            else:
                self._set_state(self.STATE_NO_TARGET)
            return self._confirmed_tracks()

        det = self._select_single_detection(detections)
        is_tracker_predicted = bool(det.get("tracker_predicted", False))
        previous_bbox = list(self.track.bbox) if self.track is not None else None

        if self.track is None:
            preaccepted = bool(det.get("_csrt_init_preaccepted", False))
            if is_tracker_predicted or not (preaccepted or self._accept_initial_detection(det)):
                return []
            if not self._depth_passes_initialization(det):
                return []
            self.track = FishTrack(
                self._next_id, det["bbox"], det["pos_3d"], det["confidence"],
                self._frame_count, temporal_depth_cfg=self.temporal_depth_cfg,
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source="yolo",
            )
            self._next_id += 1
            self._pending_init_bbox = None
            self._pending_init_hits = 0
            self._tracker_only_frames = 0
            self._set_state(self.STATE_TRACKING_CSRT)
        else:
            if is_tracker_predicted:
                if det.get("yolo_verified", False):
                    self._tracker_only_frames = 0
                else:
                    self._tracker_only_frames += 1
                if self._tracker_only_frames > self.csrt_max_tracker_only_frames:
                    self._reset()
                    return []
            else:
                self._tracker_only_frames = 0
            self.track.update(
                det["bbox"], det["pos_3d"], det["confidence"],
                self._frame_count, alpha=self.smoothing_alpha,
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source=det.get(
                    "source",
                    "csrt" if is_tracker_predicted else "yolo",
                ),
                bbox_smoothing_enabled=self.bbox_smoothing_enabled,
                bbox_center_alpha=self.bbox_center_alpha,
                bbox_size_alpha=self.bbox_size_alpha,
            )
            self._set_state(self.STATE_TRACKING_CSRT)

        if frame is not None and not is_tracker_predicted:
            should_reinit = (
                self._tracker is None
                or not self._tracker_ok
                or (self._frame_count - self._last_init_frame) >= self.csrt_reinit_interval
            )
            if previous_bbox is not None and self._tracker is not None and not should_reinit:
                iou = FishTracker._box_iou(previous_bbox, det["bbox"])
                should_reinit = iou < self.csrt_iou_reset_threshold
            if should_reinit:
                self._init_tracker(frame, det["bbox"])

        if self.track is not None and self.track.time_since_update > self.max_age:
            self._reset()

        return self._confirmed_tracks()

    def _confirmed_tracks(self) -> list[FishTrack]:
        if self.track is None:
            return []
        if self.track.time_since_update > self.max_age:
            return []
        return [self.track] if self.track.is_confirmed(self.min_hits) else []

    def _init_tracker(self, frame: np.ndarray, bbox) -> None:
        tracker = self._create_csrt_tracker()
        tracker.init(frame, self._xyxy_to_xywh(bbox))
        self._tracker = tracker
        self._tracker_ok = True
        self._last_init_frame = self._frame_count

    def _accept_initial_detection(self, det: dict) -> bool:
        if float(det.get("confidence", 0.0)) < self.csrt_init_min_confidence:
            self._pending_init_bbox = None
            self._pending_init_hits = 0
            return False

        if self._pending_init_bbox is None:
            self._pending_init_bbox = list(det["bbox"])
            self._pending_init_hits = 1
        else:
            iou = FishTracker._box_iou(self._pending_init_bbox, det["bbox"])
            if iou < self.iou_threshold:
                self._pending_init_bbox = list(det["bbox"])
                self._pending_init_hits = 1
            else:
                self._pending_init_bbox = list(det["bbox"])
                self._pending_init_hits += 1

        return self._pending_init_hits >= self.csrt_init_min_hits

    def _depth_passes_initialization(self, det: dict) -> bool:
        pos_3d = det.get("pos_3d")
        if pos_3d is None:
            return False
        if not np.all(np.isfinite(np.asarray(pos_3d, dtype=np.float32))):
            return False
        return float(det.get("depth_confidence", 0.0)) >= self.csrt_init_min_depth_confidence

    def _bbox_passes_init_geometry(self, bbox, frame_shape) -> bool:
        if frame_shape is None:
            return True
        height, width = frame_shape[:2]
        frame_area = max(float(width * height), 1.0)
        area_ratio = self._bbox_area(bbox) / frame_area
        if area_ratio < self.init_min_bbox_area_ratio:
            return False
        if area_ratio > self.init_max_bbox_area_ratio:
            return False

        box_w = max(float(bbox[2] - bbox[0]), 1.0)
        box_h = max(float(bbox[3] - bbox[1]), 1.0)
        aspect = box_w / box_h
        return self.init_min_bbox_aspect <= aspect <= self.init_max_bbox_aspect

    def _reset(self) -> None:
        self.track = None
        self._tracker = None
        self._tracker_ok = False
        self._pending_init_bbox = None
        self._pending_init_hits = 0
        self._tracker_only_frames = 0
        self._set_state(self.STATE_NO_TARGET)

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str) -> None:
        self._state = str(state)

    def _select_yolo_correction(self, detections: list[dict],
                                tracker_bbox) -> dict | None:
        candidates = []
        for det in detections:
            if bool(det.get("tracker_predicted", False)):
                continue
            if float(det.get("confidence", 0.0)) < self.yolo_correction_min_confidence:
                continue
            if not self._bbox_is_plausible(det["bbox"], tracker_bbox):
                continue
            if FishTracker._box_iou(det["bbox"], tracker_bbox) < self.yolo_correction_iou_threshold:
                continue
            candidates.append(det)
        if not candidates:
            return None
        return self._select_single_detection(candidates)

    def _should_apply_yolo_correction(self) -> bool:
        return (
            self._frame_count - self._last_init_frame
        ) >= self.yolo_correction_interval

    def _bbox_is_plausible(self, bbox, reference_bbox) -> bool:
        ref_area = max(self._bbox_area(reference_bbox), 1.0)
        area = max(self._bbox_area(bbox), 1.0)
        area_ratio = area / ref_area
        if area_ratio > self.max_bbox_area_growth:
            return False
        if area_ratio < self.min_bbox_area_shrink:
            return False

        ref_w = max(float(reference_bbox[2] - reference_bbox[0]), 1.0)
        ref_h = max(float(reference_bbox[3] - reference_bbox[1]), 1.0)
        max_ref_side = max(ref_w, ref_h)
        cx, cy = self._bbox_center(bbox)
        ref_cx, ref_cy = self._bbox_center(reference_bbox)
        center_jump = float(np.hypot(cx - ref_cx, cy - ref_cy)) / max_ref_side
        return center_jump <= self.max_bbox_center_jump

    @staticmethod
    def _bbox_area(bbox) -> float:
        return max(0.0, float(bbox[2] - bbox[0])) * max(
            0.0, float(bbox[3] - bbox[1]))

    @staticmethod
    def _bbox_center(bbox) -> tuple[float, float]:
        return (
            0.5 * float(bbox[0] + bbox[2]),
            0.5 * float(bbox[1] + bbox[3]),
        )

    @staticmethod
    def _create_csrt_tracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        raise RuntimeError(
            "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python "
            "or switch tracker.type back to 'simple'."
        )

    @staticmethod
    def _select_single_detection(detections: list[dict]) -> dict:
        return max(
            detections,
            key=lambda d: (
                float(d.get("confidence", 0.0)),
                max(0.0, d["bbox"][2] - d["bbox"][0]) *
                max(0.0, d["bbox"][3] - d["bbox"][1]),
            ),
        )

    @staticmethod
    def _xyxy_to_xywh(bbox) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (
            int(round(x1)),
            int(round(y1)),
            int(round(max(1.0, x2 - x1))),
            int(round(max(1.0, y2 - y1))),
        )

    @staticmethod
    def _xywh_to_xyxy(xywh, width: int, height: int) -> list[float]:
        x, y, w, h = [float(v) for v in xywh]
        x1 = max(0.0, min(float(width - 1), x))
        y1 = max(0.0, min(float(height - 1), y))
        x2 = max(0.0, min(float(width - 1), x + max(1.0, w)))
        y2 = max(0.0, min(float(height - 1), y + max(1.0, h)))
        return [x1, y1, x2, y2]


def create_fish_tracker(config: dict):
    tracker_cfg = dict(config.get("tracker", {}))
    tracker_type = str(tracker_cfg.pop("type", "simple")).lower()
    temporal_depth_cfg = dict(config.get("temporal_depth", {}))
    temporal_depth_cfg.setdefault("camera", dict(config.get("camera", {})))
    tracker_cfg["temporal_depth"] = temporal_depth_cfg
    if tracker_type in {"yolo_only", "yolo-only", "detector_only", "detector-only"}:
        for key in (
            "min_hits",
            "iou_threshold",
            "smoothing_alpha",
            "bbox_smoothing_enabled",
            "bbox_center_alpha",
            "bbox_size_alpha",
            "csrt_reinit_interval",
            "csrt_iou_reset_threshold",
            "csrt_init_min_confidence",
            "csrt_init_min_hits",
            "csrt_init_min_depth_confidence",
            "init_min_bbox_area_ratio",
            "init_max_bbox_area_ratio",
            "init_min_bbox_aspect",
            "init_max_bbox_aspect",
            "csrt_max_tracker_only_frames",
            "csrt_run_stereo_on_prediction",
            "stereo_every_n_yolo_frames",
            "stereo_every_n_tracker_frames",
            "prefer_tracker_when_active",
            "yolo_correction_min_confidence",
            "yolo_correction_iou_threshold",
            "yolo_correction_interval",
            "max_bbox_area_growth",
            "min_bbox_area_shrink",
            "max_bbox_center_jump",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "single_target_force_association",
            "force_assoc_min_confidence",
            "force_assoc_edge_margin_px",
            "force_assoc_min_iou",
            "force_assoc_max_center_distance_ratio",
            "force_assoc_min_area_ratio",
            "force_assoc_max_area_ratio",
            "run_stereo_on_memory_tracks",
            "single_target_output_only",
        ):
            tracker_cfg.pop(key, None)
        return YOLOOnlyFishTracker(**tracker_cfg)
    if tracker_type == "simple":
        for key in (
            "csrt_reinit_interval",
            "csrt_iou_reset_threshold",
            "csrt_init_min_confidence",
            "csrt_init_min_hits",
            "csrt_init_min_depth_confidence",
            "init_min_bbox_area_ratio",
            "init_max_bbox_area_ratio",
            "init_min_bbox_aspect",
            "init_max_bbox_aspect",
            "csrt_max_tracker_only_frames",
            "csrt_run_stereo_on_prediction",
            "stereo_every_n_yolo_frames",
            "stereo_every_n_tracker_frames",
            "prefer_tracker_when_active",
            "yolo_correction_min_confidence",
            "yolo_correction_iou_threshold",
            "yolo_correction_interval",
            "max_bbox_area_growth",
            "min_bbox_area_shrink",
            "max_bbox_center_jump",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "single_target_force_association",
            "force_assoc_min_confidence",
            "force_assoc_edge_margin_px",
            "force_assoc_min_iou",
            "force_assoc_max_center_distance_ratio",
            "force_assoc_min_area_ratio",
            "force_assoc_max_area_ratio",
        ):
            tracker_cfg.pop(key, None)
        return FishTracker(**tracker_cfg)
    if tracker_type == "bytetrack":
        for key in (
            "csrt_reinit_interval",
            "csrt_iou_reset_threshold",
            "csrt_init_min_confidence",
            "csrt_init_min_hits",
            "csrt_init_min_depth_confidence",
            "init_min_bbox_area_ratio",
            "init_max_bbox_area_ratio",
            "init_min_bbox_aspect",
            "init_max_bbox_aspect",
            "csrt_max_tracker_only_frames",
            "csrt_run_stereo_on_prediction",
            "stereo_every_n_yolo_frames",
            "stereo_every_n_tracker_frames",
            "prefer_tracker_when_active",
            "yolo_correction_min_confidence",
            "yolo_correction_iou_threshold",
            "yolo_correction_interval",
            "max_bbox_area_growth",
            "min_bbox_area_shrink",
            "max_bbox_center_jump",
            "run_stereo_on_memory_tracks",
        ):
            tracker_cfg.pop(key, None)
        return ByteTrackFishTracker(**tracker_cfg)
    if tracker_type == "csrt":
        for key in (
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "single_target_force_association",
            "force_assoc_min_confidence",
            "force_assoc_edge_margin_px",
            "force_assoc_min_iou",
            "force_assoc_max_center_distance_ratio",
            "force_assoc_min_area_ratio",
            "force_assoc_max_area_ratio",
        ):
            tracker_cfg.pop(key, None)
        return CSRTFishTracker(**tracker_cfg)
    raise ValueError(
        f"Unsupported tracker.type={tracker_type!r}. Use 'simple', 'csrt', 'bytetrack', or 'yolo_only'."
    )


def apply_pipeline_mode(config: dict, mode: str | None = None) -> dict:
    cfg = dict(config)
    selected_mode = str(mode or "full").strip().lower()
    if selected_mode in {"", "full", "default"}:
        cfg["runtime"] = dict(cfg.get("runtime", {}))
        cfg["runtime"]["pipeline_mode"] = "full"
        return cfg

    if selected_mode not in {"yolo-only", "yolo_only", "detector-only", "detector_only"}:
        raise ValueError(
            f"Unsupported pipeline mode {mode!r}. Use 'full' or 'yolo-only'."
        )

    cfg["runtime"] = dict(cfg.get("runtime", {}))
    cfg["runtime"]["pipeline_mode"] = "yolo-only"

    tracker_cfg = dict(cfg.get("tracker", {}))
    tracker_cfg["type"] = "yolo_only"
    tracker_cfg["max_age"] = 0
    cfg["tracker"] = tracker_cfg

    output_bbox_filter_cfg = dict(cfg.get("output_bbox_filter", {}))
    output_bbox_filter_cfg["enabled"] = False
    cfg["output_bbox_filter"] = output_bbox_filter_cfg

    refiner_cfg = dict(cfg.get("refiner", {}))
    refiner_cfg["enabled"] = False
    cfg["refiner"] = refiner_cfg

    corrector_cfg = dict(cfg.get("corrector_smoother", {}))
    corrector_cfg["enabled"] = False
    cfg["corrector_smoother"] = corrector_cfg

    detection_cfg = dict(cfg.get("detection", {}))
    roi_redetect_cfg = dict(detection_cfg.get("roi_redetect", {}))
    roi_redetect_cfg["enabled"] = False
    detection_cfg["roi_redetect"] = roi_redetect_cfg
    cfg["detection"] = detection_cfg

    return cfg


def apply_temporal_filter_override(config: dict,
                                   enabled: bool | None = None) -> dict:
    """Optionally bypass output temporal filters without changing ROI masks."""
    if enabled is not False:
        return config

    cfg = dict(config)
    runtime_cfg = dict(cfg.get("runtime", {}))
    runtime_cfg["temporal_filter_enabled"] = False
    cfg["runtime"] = runtime_cfg

    temporal_cfg = dict(cfg.get("temporal_depth", {}))
    temporal_cfg["enabled"] = False

    fusion_cfg = dict(temporal_cfg.get("fusion", {}))
    fusion_cfg["enabled"] = False
    temporal_cfg["fusion"] = fusion_cfg

    joint_cfg = dict(temporal_cfg.get("uvz_joint", {}))
    joint_cfg["enabled"] = False
    temporal_cfg["uvz_joint"] = joint_cfg

    center_filter_cfg = dict(temporal_cfg.get("center_filter", {}))
    center_filter_cfg["type"] = "raw"
    center_filter_cfg["enabled"] = False
    temporal_cfg["center_filter"] = center_filter_cfg

    center_ema_cfg = dict(temporal_cfg.get("center_ema", {}))
    center_ema_cfg["enabled"] = False
    temporal_cfg["center_ema"] = center_ema_cfg

    cfg["temporal_depth"] = temporal_cfg
    return cfg


# ===================================================================
#  Core estimator
# ===================================================================

class FishPositionEstimator:
    """
    End-to-end fish 3D position estimator.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see ``config.yaml``).
    """

    def __init__(self, config: dict):
        self._cfg = config
        self.yolo = None
        self.stereo = None
        self._input_padder = None
        self.runtime_cfg = dict(config.get("runtime", {}))
        self.rectifier = StereoRectifier(config.get("rectification", {}))
        self.tracker = create_fish_tracker(config)
        self.output_bbox_filter_cfg = OutputBBoxFilterConfig.from_dict(
            config.get("output_bbox_filter", {})
        )
        self.output_bbox_filter = AdaptiveBBoxOutputFilter(self.output_bbox_filter_cfg)
        self.corrector_cfg = TemporalBBoxCorrectorSmootherConfig.from_dict(
            config.get("corrector_smoother", {})
        )
        self.corrector = self._build_corrector_smoother(self.corrector_cfg)
        self.refiner_cfg = TemporalBBoxRefinerConfig.from_dict(
            config.get("refiner", {})
        )
        self.refiner = self._build_refiner(self.refiner_cfg)
        self._small_target_refine_last_frame_by_track: dict[int, int] = {}
        self._small_target_protection_state_by_track: dict[int, dict] = {}
        self._frame_idx = 0
        self._last_timestamp = None
        self._stereo_yolo_update_count = 0
        self._stereo_tracker_update_count = 0
        self._load_models()

    # ── Public API ─────────────────────────────────────────────────────

    def estimate(self,
                 left_img: np.ndarray,
                 right_img: np.ndarray,
                 *,
                 frame_ts_s: float | None = None) -> list[dict]:
        """
        Run the full pipeline on one stereo pair.

        Parameters
        ----------
        left_img  : np.ndarray  (H, W, 3)  uint8  BGR or RGB
        right_img : np.ndarray  (H, W, 3)  uint8  BGR or RGB

        Returns
        -------
        list[dict]
            Each dict::

                {
                    "id":         int,              # track ID (persistent across frames)
                    "bbox":       [x1, y1, x2, y2],# pixel bounding box in left image
                    "position":   [x,  y,  z],     # metres in camera frame
                    "confidence": float,            # YOLO detection confidence
                }

            Returns an empty list when no fish is tracked.
        """
        self._frame_idx += 1
        dt_s = self._measure_dt(frame_ts_s)
        left_img, right_img = self._rectify_pair(left_img, right_img)
        H0, W0 = left_img.shape[:2]

        # 1 ── YOLO detection ──────────────────────────────────────
        det_results = self._detect_fish(left_img)
        det_results = self._prefer_tracker_when_active(left_img, det_results)

        if not det_results:
            # Still age existing tracks
            tracks = self.tracker.update([], frame=left_img, dt_s=dt_s)
            self._maybe_empty_cuda_cache()
            tracks = self._apply_output_bbox_filter(tracks, dt_s=dt_s)
            if tracks and self._should_refresh_memory_track_depth():
                disparity_full = self._compute_disparity(left_img, right_img, H0, W0)
                tracks = self._refresh_tracks_from_final_bboxes(
                    tracks,
                    disparity_full,
                    left_img=left_img,
                    dt_s=dt_s,
                )
            return self._pack_results(tracks)

        det_results = self._filter_detections_for_stereo(
            det_results, frame_shape=left_img.shape)
        if not det_results:
            tracks = self.tracker.update([], frame=left_img, dt_s=dt_s)
            self._maybe_empty_cuda_cache()
            tracks = self._apply_output_bbox_filter(tracks, dt_s=dt_s)
            if tracks and self._should_refresh_memory_track_depth():
                disparity_full = self._compute_disparity(left_img, right_img, H0, W0)
                tracks = self._refresh_tracks_from_final_bboxes(
                    tracks,
                    disparity_full,
                    left_img=left_img,
                    dt_s=dt_s,
                )
            return self._pack_results(tracks)

        if self._skip_stereo_for_fast_yolo(det_results):
            return self._update_tracks_without_stereo(
                det_results, left_img, dt_s, source="yolo-fast")

        if self._skip_stereo_for_tracker_predictions(det_results):
            return self._update_tracks_without_stereo(
                det_results, left_img, dt_s, source="csrt")

        # Free YOLO GPU memory before running stereo
        self._maybe_empty_cuda_cache()

        # 2 ── FoundationStereo disparity ──────────────────────────
        disparity_full = self._compute_disparity(left_img, right_img, H0, W0)

        # 3 ── 3D position for each detection ──────────────────────
        detections = self._measure_detections_with_stereo(
            det_results,
            disparity_full,
            left_img,
            right_img,
            dt_s=dt_s,
        )

        # 4 ── Tracker update ──────────────────────────────────────
        tracks = self.tracker.update(detections, frame=left_img, dt_s=dt_s)
        if self.corrector is not None:
            tracks = self._maybe_correct_tracks_post(
                left_img,
                tracks,
            )
        elif self.refiner is not None:
            tracks = self._maybe_refine_tracks_post(
                left_img,
                disparity_full,
                tracks,
            )

        tracks = self._apply_output_bbox_filter(tracks, dt_s=dt_s)
        return self._pack_results(tracks)

    def _measure_detections_with_stereo(self,
                                        det_results: list[dict],
                                        disparity_full: np.ndarray,
                                        left_img: np.ndarray,
                                        right_img: np.ndarray,
                                        *,
                                        dt_s: float | None = None) -> list[dict]:
        detections = []
        for det in det_results:
            prior_track = self._lookup_roi_prior_track(det["bbox"])
            if prior_track is None:
                roi_prior = {"depth_m": None, "center_uv": None, "track_id": None}
            else:
                roi_prior = self._track_roi_prior(prior_track)
            depth_stats = self._extract_roi_depth(
                det["bbox"],
                disparity_full,
                color_image=left_img,
                depth_prior_m=roi_prior["depth_m"],
                center_prior_uv=roi_prior["center_uv"],
                dt_s=dt_s,
            )
            depth_stats, protected_pos_3d, depth_source = (
                self._maybe_refine_or_protect_small_target_depth(
                    det,
                    left_img,
                    right_img,
                    disparity_full,
                    depth_stats,
                    roi_prior,
                )
            )
            pos_3d = protected_pos_3d
            if pos_3d is None:
                pos_3d = self._bbox_depth_to_3d(det["center"], depth_stats)
            if pos_3d is not None:
                detections.append({
                    "bbox": det["bbox"],
                    "pos_3d": pos_3d,
                    "confidence": det["confidence"],
                    "depth_stats": depth_stats,
                    "depth_confidence": self._depth_confidence(depth_stats),
                    "tracker_predicted": det.get("tracker_predicted", False),
                    "source": depth_source or det.get("source", "yolo"),
                    "yolo_verified": det.get("yolo_verified", False),
                    "_csrt_init_preaccepted": det.get("_csrt_init_preaccepted", False),
                })
                continue

            # Detection/tracking must not depend on a valid depth solve.
            # Keep the bbox alive on this frame, but let temporal depth run in
            # predict-only mode instead of feeding the previous 3D state back
            # as a fake measurement.
            detections.append({
                "bbox": det["bbox"],
                "pos_3d": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                "confidence": det["confidence"],
                "depth_stats": depth_stats,
                "depth_confidence": 0.0,
                "tracker_predicted": det.get("tracker_predicted", False),
                "source": f"{depth_source or det.get('source', 'yolo')}+depth-missing",
                "yolo_verified": det.get("yolo_verified", False),
                "_csrt_init_preaccepted": det.get("_csrt_init_preaccepted", False),
            })
        return detections

    # ── Model loading ──────────────────────────────────────────────────

    def _load_models(self):
        """Load YOLO and FoundationStereo, move to GPU."""
        cfg = self._cfg

        # --- YOLOv8 ---
        yolo_path = self._resolve_model_path(
            cfg["models"]["yolo_path"], allow_ultralytics_alias=True)

        yolo_backend = cfg["models"].get("yolo_backend", "ultralytics")
        device = cfg["detection"].get("device", "cuda")
        if yolo_backend == "yolov5":
            repo = self._resolve_model_path(cfg["models"].get("yolov5_repo", ""))
            self.yolo = torch.hub.load(
                repo, "custom", path=yolo_path, source="local", verbose=False)
            self.yolo.to(device)
            self.yolo.eval()
            print(f"[FishPosition] YOLOv5 loaded from {yolo_path}")
        else:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_path)
            self.yolo.to(device)
            print(f"[FishPosition] YOLOv8 loaded from {yolo_path}")

        # --- Stereo backend ---
        stereo_backend = str(
            cfg["models"].get("stereo_backend", "foundationstereo")
        ).lower()
        if stereo_backend in {"fastfoundationstereo", "fast_foundationstereo", "ffs"}:
            self._load_fast_foundation_stereo(cfg)
        else:
            self._load_foundation_stereo(cfg)

    def _build_refiner(self, refiner_cfg: TemporalBBoxRefinerConfig):
        if not refiner_cfg.enabled:
            return None

        checkpoint_path = refiner_cfg.checkpoint_path
        if not checkpoint_path:
            print("[FishPosition] Refiner enabled but checkpoint_path is empty; disabling refiner.")
            return None

        resolved_ckpt = self._resolve_model_path(checkpoint_path)
        if not os.path.isfile(resolved_ckpt):
            print(
                "[FishPosition] Refiner checkpoint not found at "
                f"{resolved_ckpt}; disabling refiner."
            )
            return None

        runtime_cfg = TemporalBBoxRefinerConfig(**vars(refiner_cfg))
        runtime_cfg.checkpoint_path = resolved_ckpt
        try:
            refiner = TemporalBBoxRefinerRuntime(runtime_cfg)
        except Exception as exc:
            print(f"[FishPosition] Failed to load refiner from {resolved_ckpt}: {exc}")
            return None

        print(f"[FishPosition] Temporal bbox refiner loaded from {resolved_ckpt}")
        return refiner

    def _build_corrector_smoother(self,
                                  corrector_cfg: TemporalBBoxCorrectorSmootherConfig):
        if not corrector_cfg.enabled:
            return None

        checkpoint_path = corrector_cfg.checkpoint_path
        if not checkpoint_path:
            print(
                "[FishPosition] Corrector-smoother enabled but checkpoint_path is empty; "
                "disabling corrector-smoother."
            )
            return None

        resolved_ckpt = self._resolve_model_path(checkpoint_path)
        if not os.path.isfile(resolved_ckpt):
            print(
                "[FishPosition] Corrector-smoother checkpoint not found at "
                f"{resolved_ckpt}; disabling corrector-smoother."
            )
            return None

        runtime_cfg = TemporalBBoxCorrectorSmootherConfig(**vars(corrector_cfg))
        runtime_cfg.checkpoint_path = resolved_ckpt
        try:
            corrector = TemporalBBoxCorrectorSmootherRuntime(runtime_cfg)
        except Exception as exc:
            print(
                "[FishPosition] Failed to load corrector-smoother from "
                f"{resolved_ckpt}: {exc}"
            )
            return None

        print(f"[FishPosition] Temporal bbox corrector-smoother loaded from {resolved_ckpt}")
        return corrector

    def _load_foundation_stereo(self, cfg: dict) -> None:
        ckpt = self._resolve_model_path(cfg["models"]["stereo_ckpt"])
        cfg_yaml = self._resolve_model_path(cfg["models"]["stereo_cfg_yaml"])
        backend_root = self._resolve_stereo_backend_root("foundation")
        self._activate_stereo_backend_root(backend_root)

        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        self._InputPadder = InputPadder

        scfg = OmegaConf.create(yaml.safe_load(open(cfg_yaml)))
        s_opts = self._cfg.get("stereo", {})
        for k, v in dict(
            valid_iters=s_opts.get("valid_iters", 32),
            hiera=0,
            scale=s_opts.get("image_scale", 0.5),
            low_memory=s_opts.get("low_memory", True),
            get_pc=0,
            remove_invisible=1,
        ).items():
            scfg[k] = v

        self.stereo = FoundationStereo(scfg)
        ck = torch.load(ckpt, weights_only=False)
        self.stereo.load_state_dict(ck["model"])
        self.stereo.cuda().eval()
        self._maybe_empty_cuda_cache(force=True)
        self._stereo_backend = "foundationstereo"
        print(f"[FishPosition] FoundationStereo loaded from {ckpt}")

    def _load_fast_foundation_stereo(self, cfg: dict) -> None:
        ckpt = self._resolve_model_path(cfg["models"]["stereo_ckpt"])
        backend_root = self._resolve_stereo_backend_root("fast")
        self._activate_stereo_backend_root(backend_root)

        from core.utils.utils import InputPadder

        self._InputPadder = InputPadder
        model = torch.load(ckpt, map_location="cpu", weights_only=False)
        s_opts = self._cfg.get("stereo", {})
        if hasattr(model, "args"):
            defaults = dict(
                valid_iters=s_opts.get(
                    "valid_iters", self._cfg_get(model.args, "valid_iters", 8)
                ),
                scale=s_opts.get(
                    "image_scale", self._cfg_get(model.args, "scale", 1.0)
                ),
                low_memory=bool(
                    s_opts.get(
                        "low_memory", self._cfg_get(model.args, "low_memory", True)
                    )
                ),
                max_disp=int(
                    s_opts.get("max_disp", self._cfg_get(model.args, "max_disp", 192))
                ),
                normalize=bool(self._cfg_get(model.args, "normalize", True)),
                mixed_precision=bool(
                    self._cfg_get(model.args, "mixed_precision", True)
                ),
                corr_levels=int(self._cfg_get(model.args, "corr_levels", 2)),
                corr_radius=int(self._cfg_get(model.args, "corr_radius", 4)),
                n_gru_layers=int(self._cfg_get(model.args, "n_gru_layers", 1)),
                n_downsample=int(self._cfg_get(model.args, "n_downsample", 2)),
                vit_size=self._cfg_get(model.args, "vit_size", "vitl"),
                amp_dtype=self._cfg_get(model.args, "amp_dtype", "float16"),
            )
            for key, value in defaults.items():
                self._cfg_set(model.args, key, value)
        self.stereo = model.cuda().eval()
        self._maybe_empty_cuda_cache(force=True)
        self._stereo_backend = "fastfoundationstereo"
        print(f"[FishPosition] FastFoundationStereo loaded from {ckpt}")

    def _resolve_stereo_backend_root(self, backend: str) -> str:
        backend = str(backend).lower()
        if backend == "fast":
            candidates = [
                os.path.join(_REPO_ROOT, "third_party", "Fast-FoundationStereo"),
            ]
        else:
            candidates = [
                os.path.join(_MODULE_DIR, "FoundationStereo"),
                os.path.join(_REPO_ROOT, "third_party", "FoundationStereo"),
            ]
        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.isfile(os.path.join(candidate, "core", "foundation_stereo.py")):
                return candidate
        raise FileNotFoundError(f"Stereo backend root not found for backend={backend!r}")

    @staticmethod
    def _clear_core_modules() -> None:
        for name in list(sys.modules):
            if name == "core" or name.startswith("core."):
                sys.modules.pop(name, None)

    def _activate_stereo_backend_root(self, backend_root: str) -> None:
        backend_root = os.path.normpath(backend_root)
        self._clear_core_modules()
        if backend_root in sys.path:
            sys.path.remove(backend_root)
        sys.path.insert(0, backend_root)

    @staticmethod
    def _cfg_get(cfg_obj, key: str, default=None):
        try:
            value = cfg_obj.get(key, default)
        except Exception:
            value = getattr(cfg_obj, key, default)
        return default if value is None else value

    @staticmethod
    def _cfg_set(cfg_obj, key: str, value) -> None:
        try:
            cfg_obj[key] = value
            return
        except Exception:
            pass
        setattr(cfg_obj, key, value)
        importlib.invalidate_caches()

    # ── Detection ──────────────────────────────────────────────────────

    def _detect_fish(self, img: np.ndarray) -> list[dict]:
        """Run YOLO, return list of {bbox, center, confidence}."""
        det_cfg = self._cfg.get("detection", {})
        min_conf = float(det_cfg.get("min_confidence", 0.4))
        imgsz = det_cfg.get("detector_imgsz")
        dets = self._run_detector(img, min_conf=min_conf, imgsz=imgsz)
        dets = self._maybe_run_roi_redetect(img, dets)
        dets = self._maybe_run_edge_redetect(img, dets)
        dets = self._dedupe_detections(dets)
        single_cfg = det_cfg.get("single_target", {})
        if isinstance(single_cfg, dict) and bool(single_cfg.get("enabled", False)):
            if dets:
                strategy = str(single_cfg.get("strategy", "highest_confidence")).lower()
                if strategy in {"highest_confidence", "confidence", "conf"}:
                    dets = [max(dets, key=lambda det: float(det.get("confidence", 0.0)))]
                elif strategy in {"largest_area", "largest", "area"}:
                    dets = [max(
                        dets,
                        key=lambda det: max(0.0, float(det["bbox"][2]) - float(det["bbox"][0]))
                        * max(0.0, float(det["bbox"][3]) - float(det["bbox"][1])),
                    )]
                else:
                    raise ValueError(
                        "Unsupported detection.single_target.strategy: "
                        f"{strategy}. Use 'highest_confidence' or 'largest_area'."
                    )
        return dets

    def _run_detector(self, img: np.ndarray, *,
                      min_conf: float,
                      imgsz=None) -> list[dict]:
        if self._cfg["models"].get("yolo_backend") == "yolov5":
            return self._detect_fish_yolov5(img, min_conf, imgsz=imgsz)

        det_cfg = self._cfg.get("detection", {})
        infer_kwargs = {"verbose": False}
        if imgsz is not None:
            infer_kwargs["imgsz"] = int(imgsz)
        if "quantize" in det_cfg:
            quantize = det_cfg.get("quantize")
            if isinstance(quantize, bool):
                if quantize:
                    infer_kwargs["quantize"] = "fp16"
            elif quantize is not None:
                infer_kwargs["quantize"] = quantize
        elif "half" in det_cfg:
            # Backward-compatible fallback for older configs. Newer
            # Ultralytics versions renamed this inference flag to
            # "quantize" and emit a warning when "half" is used.
            if bool(det_cfg.get("half", False)):
                infer_kwargs["quantize"] = "fp16"
        results = self.yolo(img, **infer_kwargs)[0]

        dets = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if conf < min_conf:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dets.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": (cx, cy),
                    "confidence": float(conf),
                    "source": "yolo",
                })
        return dets

    def _detect_fish_yolov5(self, img: np.ndarray, min_conf: float,
                            imgsz=None) -> list[dict]:
        infer_kwargs = {}
        if imgsz is not None:
            infer_kwargs["size"] = int(imgsz)
        results = self.yolo(img, **infer_kwargs)
        pred = results.xyxy[0]
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()

        dets = []
        for row in pred:
            x1, y1, x2, y2, conf = row[:5]
            if conf < min_conf:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dets.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": (float(cx), float(cy)),
                "confidence": float(conf),
                "source": "yolo",
            })
        return dets

    def _maybe_run_roi_redetect(self, img: np.ndarray,
                                 base_dets: list[dict]) -> list[dict]:
        det_cfg = self._cfg.get("detection", {})
        roi_cfg = det_cfg.get("roi_redetect", {})
        if not bool(roi_cfg.get("enabled", False)):
            return base_dets

        prior_bbox = self._select_redetect_prior_bbox()
        if prior_bbox is None:
            return base_dets

        if not self._should_trigger_roi_redetect(
            base_dets,
            prior_bbox,
            img.shape,
            roi_cfg,
        ):
            return base_dets

        crop_box = self._expand_bbox_unclipped(
            prior_bbox,
            img.shape[1],
            img.shape[0],
            expand_ratio=float(roi_cfg.get("expand_ratio", 2.0)),
            min_size_px=int(roi_cfg.get("min_crop_size_px", 96)),
        )
        x1, y1, x2, y2 = crop_box
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return base_dets

        roi = self._extract_padded_crop(
            img,
            crop_box,
            border_mode=str(roi_cfg.get("border_mode", "replicate")),
        )
        if roi.size == 0:
            return base_dets

        roi_min_conf = float(roi_cfg.get(
            "min_confidence",
            det_cfg.get("min_confidence", 0.4),
        ))
        roi_imgsz = roi_cfg.get("detector_imgsz", det_cfg.get("detector_imgsz"))
        roi_dets = self._run_detector(roi, min_conf=roi_min_conf, imgsz=roi_imgsz)
        if not roi_dets:
            return base_dets

        remapped = []
        min_iou = float(roi_cfg.get("min_iou_with_prior", 0.05))
        for det in roi_dets:
            bx1, by1, bx2, by2 = det["bbox"]
            bbox = self._clip_bbox_to_image(
                [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                img.shape[1],
                img.shape[0],
            )
            if bbox is None:
                continue
            if FishTracker._box_iou(bbox, prior_bbox) < min_iou:
                continue
            remapped.append({
                "bbox": bbox,
                "center": (
                    float(np.clip(det["center"][0] + x1, 0, img.shape[1] - 1)),
                    float(np.clip(det["center"][1] + y1, 0, img.shape[0] - 1)),
                ),
                "confidence": float(det["confidence"]),
                "source": "roi-redetect",
            })

        return self._dedupe_detections(base_dets + remapped)

    def _maybe_run_edge_redetect(self, img: np.ndarray,
                                 base_dets: list[dict]) -> list[dict]:
        det_cfg = self._cfg.get("detection", {})
        edge_cfg = det_cfg.get("edge_redetect", {})
        if not bool(edge_cfg.get("enabled", False)):
            return base_dets
        if not self._should_trigger_edge_redetect(base_dets, img.shape, edge_cfg):
            return base_dets

        pad_px = max(int(edge_cfg.get("pad_px", 0)), 0)
        if pad_px < 4:
            return base_dets

        border_mode = self._cv_border_mode(str(edge_cfg.get("border_mode", "replicate")))
        padded = cv2.copyMakeBorder(
            img,
            pad_px,
            pad_px,
            pad_px,
            pad_px,
            borderType=border_mode,
        )
        edge_min_conf = float(edge_cfg.get(
            "min_confidence",
            det_cfg.get("min_confidence", 0.4),
        ))
        edge_imgsz = edge_cfg.get("detector_imgsz", det_cfg.get("detector_imgsz"))
        padded_dets = self._run_detector(
            padded,
            min_conf=edge_min_conf,
            imgsz=edge_imgsz,
        )
        if not padded_dets:
            return base_dets

        edge_margin_px = int(edge_cfg.get("edge_margin_px", 32))
        keep_near_edge_only = bool(edge_cfg.get("keep_near_edge_only", True))
        remapped = []
        for det in padded_dets:
            bx1, by1, bx2, by2 = det["bbox"]
            bbox = self._clip_bbox_to_image(
                [bx1 - pad_px, by1 - pad_px, bx2 - pad_px, by2 - pad_px],
                img.shape[1],
                img.shape[0],
            )
            if bbox is None:
                continue
            if keep_near_edge_only and not self._bbox_near_edge(
                bbox,
                img.shape,
                margin_px=edge_margin_px,
            ):
                continue
            remapped.append({
                "bbox": bbox,
                "center": (
                    float(np.clip(det["center"][0] - pad_px, 0, img.shape[1] - 1)),
                    float(np.clip(det["center"][1] - pad_px, 0, img.shape[0] - 1)),
                ),
                "confidence": float(det["confidence"]),
                "source": "edge-redetect",
            })

        return self._dedupe_detections(base_dets + remapped)

    def _select_redetect_prior_bbox(self) -> list[int] | None:
        single_track = getattr(self.tracker, "track", None)
        if single_track is not None and getattr(single_track, "time_since_update", 9999) <= self.tracker.max_age:
            return [int(round(v)) for v in single_track.bbox]

        tracks = getattr(self.tracker, "tracks", None)
        if not tracks:
            return None

        best_track = None
        best_key = None
        for track in tracks:
            if getattr(track, "time_since_update", 9999) > self.tracker.max_age:
                continue
            key = (
                -int(getattr(track, "hits", 0)),
                int(getattr(track, "time_since_update", 9999)),
                -float(getattr(track, "confidence", 0.0)),
            )
            if best_key is None or key < best_key:
                best_track = track
                best_key = key

        if best_track is None:
            return None
        return [int(round(v)) for v in best_track.bbox]

    @staticmethod
    def _expand_bbox(bbox, width: int, height: int, *,
                     expand_ratio: float,
                     min_size_px: int) -> list[int]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max((x2 - x1) * float(expand_ratio), float(min_size_px))
        bh = max((y2 - y1) * float(expand_ratio), float(min_size_px))
        nx1 = max(0, int(np.floor(cx - 0.5 * bw)))
        ny1 = max(0, int(np.floor(cy - 0.5 * bh)))
        nx2 = min(int(width), int(np.ceil(cx + 0.5 * bw)))
        ny2 = min(int(height), int(np.ceil(cy + 0.5 * bh)))
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _expand_bbox_unclipped(bbox, width: int, height: int, *,
                               expand_ratio: float,
                               min_size_px: int) -> list[int]:
        del width, height
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max((x2 - x1) * float(expand_ratio), float(min_size_px))
        bh = max((y2 - y1) * float(expand_ratio), float(min_size_px))
        nx1 = int(np.floor(cx - 0.5 * bw))
        ny1 = int(np.floor(cy - 0.5 * bh))
        nx2 = int(np.ceil(cx + 0.5 * bw))
        ny2 = int(np.ceil(cy + 0.5 * bh))
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _cv_border_mode(name: str) -> int:
        key = str(name).strip().lower()
        mapping = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect101": cv2.BORDER_REFLECT_101,
            "default": cv2.BORDER_DEFAULT,
        }
        return mapping.get(key, cv2.BORDER_REPLICATE)

    def _extract_padded_crop(self,
                             img: np.ndarray,
                             crop_box: list[int],
                             *,
                             border_mode: str) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        height, width = img.shape[:2]
        cx1 = max(0, min(width, x1))
        cy1 = max(0, min(height, y1))
        cx2 = max(0, min(width, x2))
        cy2 = max(0, min(height, y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return np.empty((0, 0, img.shape[2]), dtype=img.dtype)
        crop = img[cy1:cy2, cx1:cx2]
        left = max(0, cx1 - x1)
        top = max(0, cy1 - y1)
        right = max(0, x2 - cx2)
        bottom = max(0, y2 - cy2)
        if left == 0 and top == 0 and right == 0 and bottom == 0:
            return crop
        return cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            borderType=self._cv_border_mode(border_mode),
        )

    def _pad_crop_to_refine_bucket(self,
                                   img: np.ndarray,
                                   *,
                                   cfg: dict) -> tuple[np.ndarray, tuple[int, int]]:
        """Pad a local-refine crop into a small set of fixed bucket sizes."""
        if img.size == 0:
            return img, (0, 0)

        orig_h, orig_w = img.shape[:2]
        if not bool(cfg.get("bucket_enabled", True)):
            return img, (orig_h, orig_w)

        bucket_sizes = sorted({
            int(v) for v in cfg.get("bucket_sizes_px", [160, 192, 224, 256, 320])
            if int(v) > 0
        })
        if not bucket_sizes:
            return img, (orig_h, orig_w)

        pad_square = bool(cfg.get("bucket_pad_square", True))
        border_mode = self._cv_border_mode(str(cfg.get("border_mode", "replicate")))

        if pad_square:
            side = max(orig_h, orig_w)
            target_side = next((b for b in bucket_sizes if b >= side), None)
            if target_side is None:
                target_side = int(np.ceil(side / 32.0) * 32)
            target_h = target_side
            target_w = target_side
        else:
            target_h = next((b for b in bucket_sizes if b >= orig_h), None)
            target_w = next((b for b in bucket_sizes if b >= orig_w), None)
            if target_h is None:
                target_h = int(np.ceil(orig_h / 32.0) * 32)
            if target_w is None:
                target_w = int(np.ceil(orig_w / 32.0) * 32)

        if target_h == orig_h and target_w == orig_w:
            return img, (orig_h, orig_w)

        padded = cv2.copyMakeBorder(
            img,
            0,
            max(target_h - orig_h, 0),
            0,
            max(target_w - orig_w, 0),
            borderType=border_mode,
        )
        return padded, (orig_h, orig_w)

    @staticmethod
    def _clip_bbox_to_image(bbox, width: int, height: int,
                            min_size_px: int = 4) -> list[int] | None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(1.0, min(float(width), x2))
        y2 = max(1.0, min(float(height), y2))
        if (x2 - x1) < float(min_size_px) or (y2 - y1) < float(min_size_px):
            return None
        return [
            int(np.floor(x1)),
            int(np.floor(y1)),
            int(np.ceil(x2)),
            int(np.ceil(y2)),
        ]

    @staticmethod
    def _bbox_near_edge(bbox,
                        frame_shape,
                        *,
                        margin_px: int) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dist = min(x1, y1, max(0.0, width - x2), max(0.0, height - y2))
        return dist <= float(max(margin_px, 0))

    def _should_trigger_roi_redetect(self,
                                     base_dets: list[dict],
                                     prior_bbox,
                                     frame_shape,
                                     roi_cfg: dict) -> bool:
        if not base_dets:
            return True
        if bool(roi_cfg.get("trigger_on_empty_only", True)):
            return False

        edge_margin_px = int(roi_cfg.get("edge_margin_px", 32))
        if bool(roi_cfg.get("trigger_on_near_edge", False)):
            if self._bbox_near_edge(prior_bbox, frame_shape, margin_px=edge_margin_px):
                return True
            if any(self._bbox_near_edge(det["bbox"], frame_shape, margin_px=edge_margin_px)
                   for det in base_dets):
                return True

        if bool(roi_cfg.get("trigger_on_low_confidence", False)):
            threshold = float(roi_cfg.get("base_confidence_threshold", 0.45))
            best_conf = max(float(det.get("confidence", 0.0)) for det in base_dets)
            if best_conf <= threshold:
                return True

        if bool(roi_cfg.get("trigger_on_small_bbox", False)):
            if self._bbox_matches_roi_redetect_small_policy(prior_bbox, frame_shape, roi_cfg):
                return True
            if any(
                self._bbox_matches_roi_redetect_small_policy(det["bbox"], frame_shape, roi_cfg)
                for det in base_dets
            ):
                return True

        return bool(roi_cfg.get("run_when_base_exists", False))

    @staticmethod
    def _bbox_matches_roi_redetect_small_policy(bbox,
                                                frame_shape,
                                                roi_cfg: dict) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        area = bw * bh
        frame_area = max(float(width * height), 1.0)
        area_ratio = area / frame_area
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        max_area_ratio = float(roi_cfg.get("max_bbox_area_ratio", 0.008))
        max_long_side_px = float(roi_cfg.get("max_bbox_long_side_px", 96))
        max_short_side_px = float(roi_cfg.get("max_bbox_short_side_px", 72))
        return (
            area_ratio <= max_area_ratio
            or long_side <= max_long_side_px
            or short_side <= max_short_side_px
        )

    def _should_trigger_edge_redetect(self,
                                      base_dets: list[dict],
                                      frame_shape,
                                      edge_cfg: dict) -> bool:
        if not base_dets:
            return bool(edge_cfg.get("trigger_on_empty", True))
        if bool(edge_cfg.get("trigger_on_empty_only", True)):
            return False

        edge_margin_px = int(edge_cfg.get("edge_margin_px", 32))
        if bool(edge_cfg.get("trigger_on_near_edge", False)):
            if any(self._bbox_near_edge(det["bbox"], frame_shape, margin_px=edge_margin_px)
                   for det in base_dets):
                return True

        if bool(edge_cfg.get("trigger_on_low_confidence", False)):
            threshold = float(edge_cfg.get("base_confidence_threshold", 0.45))
            best_conf = max(float(det.get("confidence", 0.0)) for det in base_dets)
            if best_conf <= threshold:
                return True

        return bool(edge_cfg.get("run_when_base_exists", False))

    def _dedupe_detections(self, dets: list[dict]) -> list[dict]:
        if len(dets) <= 1:
            return dets
        merge_iou = float(
            self._cfg.get("detection", {})
            .get("roi_redetect", {})
            .get("merge_iou_threshold", 0.5)
        )
        ordered = sorted(
            dets,
            key=lambda det: float(det.get("confidence", 0.0)),
            reverse=True,
        )
        kept = []
        for det in ordered:
            if any(FishTracker._box_iou(det["bbox"], prev["bbox"]) >= merge_iou
                   for prev in kept):
                continue
            kept.append(det)
        return kept

    def _predict_tracker_detections(self, img: np.ndarray,
                                    dt_s: float | None = None) -> list[dict]:
        predict = getattr(self.tracker, "predict_detections", None)
        if predict is None:
            return []
        return predict(img, dt_s=dt_s)

    def _prefer_tracker_when_active(self, img: np.ndarray,
                                    det_results: list[dict]) -> list[dict]:
        prefer = getattr(self.tracker, "prefer_active_detections", None)
        if prefer is None:
            return det_results
        return prefer(img, det_results)

    def _filter_detections_for_stereo(self, det_results: list[dict],
                                      frame_shape=None) -> list[dict]:
        filter_fn = getattr(self.tracker, "filter_detections_for_stereo", None)
        if filter_fn is None:
            return det_results
        return filter_fn(det_results, frame_shape=frame_shape)

    def _skip_stereo_for_fast_yolo(self, det_results: list[dict]) -> bool:
        if not det_results:
            return False
        if any(det.get("tracker_predicted", False) for det in det_results):
            return False
        primary_track = getattr(self.tracker, "track", None)
        if primary_track is None:
            self._stereo_yolo_update_count = 0
            return False
        n = int(self._cfg.get("tracker", {}).get("stereo_every_n_yolo_frames", 1))
        if n <= 1:
            return False
        self._stereo_yolo_update_count += 1
        if self._stereo_yolo_update_count >= n:
            self._stereo_yolo_update_count = 0
            return False
        return True

    def _skip_stereo_for_tracker_predictions(self, det_results: list[dict]) -> bool:
        if not det_results:
            return False
        if not all(det.get("tracker_predicted", False) for det in det_results):
            return False
        tracker_cfg = self._cfg.get("tracker", {})
        if bool(tracker_cfg.get("csrt_run_stereo_on_prediction", False)):
            return False
        n = int(tracker_cfg.get("stereo_every_n_tracker_frames", 1))
        if n <= 1:
            return False
        self._stereo_tracker_update_count += 1
        if self._stereo_tracker_update_count >= n:
            self._stereo_tracker_update_count = 0
            return False
        return True

    def _maybe_correct_tracks_post(self,
                                   frame: np.ndarray,
                                   tracks: list[FishTrack]) -> list[FishTrack]:
        if self.corrector is None:
            return tracks
        if not tracks:
            self.corrector.reset()
            return tracks
        if len(tracks) != 1:
            self.corrector.reset()
            return tracks

        track = tracks[0]
        has_detection_support = int(getattr(track, "time_since_update", 0)) == 0
        if not has_detection_support and not self.corrector.cfg.apply_on_tracker_only:
            self.corrector.skip()
            return tracks

        bbox = self._sanitize_bbox_xyxy(
            getattr(track, "raw_bbox", track.bbox),
            frame.shape[1],
            frame.shape[0],
        )
        if bbox is None:
            self.corrector.reset()
            return tracks

        self.corrector.push(
            frame,
            bbox,
            confidence=float(getattr(track, "confidence", 0.0)),
            source=str(getattr(track, "source", "tracker")),
        )
        prediction = self.corrector.correct_current(
            has_detection_support=has_detection_support,
        )
        if prediction is None or not prediction.accepted:
            return tracks

        corrected_bbox = self._sanitize_bbox_xyxy(
            prediction.corrected_bbox,
            frame.shape[1],
            frame.shape[0],
        )
        final_bbox = self._sanitize_bbox_xyxy(
            prediction.final_bbox,
            frame.shape[1],
            frame.shape[0],
        )
        if corrected_bbox is None or final_bbox is None:
            return tracks

        track.raw_bbox = list(corrected_bbox)
        track.bbox = list(final_bbox)
        track.source = f"{getattr(track, 'source', 'tracker')}+corrector"
        if getattr(self.tracker, "external_bbox_writeback", False) and hasattr(self.tracker, "apply_external_bbox"):
            self.tracker.apply_external_bbox(
                track.id,
                final_bbox,
                score=float(getattr(track, "confidence", 0.0)),
            )
        return tracks

    def _maybe_refine_tracks_post(self,
                                  frame: np.ndarray,
                                  disparity_full: np.ndarray,
                                  tracks: list[FishTrack]) -> list[FishTrack]:
        if self.refiner is None:
            return tracks
        if not tracks:
            self.refiner.reset()
            return tracks
        if len(tracks) != 1:
            self.refiner.reset()
            return tracks

        track = tracks[0]
        apply_on_tracker_only = bool(
            getattr(self.refiner, "cfg", None) is not None
            and self.refiner.cfg.apply_on_tracker_only
        )
        has_detection_support = int(getattr(track, "time_since_update", 0)) == 0
        if not has_detection_support and not apply_on_tracker_only:
            self.refiner.reset()
            return tracks

        bbox = self._sanitize_bbox_xyxy(
            track.bbox,
            frame.shape[1],
            frame.shape[0],
        )
        if bbox is None:
            self.refiner.reset()
            return tracks

        self.refiner.push(
            frame,
            bbox,
            confidence=float(getattr(track, "confidence", 0.0)),
            source=str(getattr(track, "source", "tracker")),
        )
        prediction = self.refiner.refine_current()
        if prediction is None:
            return tracks

        if not prediction.accepted:
            return tracks

        refined_bbox = self._sanitize_bbox_xyxy(
            prediction.refined_bbox,
            frame.shape[1],
            frame.shape[0],
        )
        if refined_bbox is None:
            return tracks

        final_bbox = list(refined_bbox)
        if getattr(self.refiner, "cfg", None) is not None and \
                self.refiner.cfg.post_smooth_enabled:
            final_bbox = FishTrack._smooth_bbox(
                track.bbox,
                refined_bbox,
                center_alpha=float(self.refiner.cfg.post_smooth_center_alpha),
                size_alpha=float(self.refiner.cfg.post_smooth_size_alpha),
            )

        final_bbox = self._sanitize_bbox_xyxy(
            final_bbox,
            frame.shape[1],
            frame.shape[0],
        )
        if final_bbox is None:
            return tracks

        track.raw_bbox = list(refined_bbox)
        track.bbox = list(final_bbox)
        if getattr(self.tracker, "external_bbox_writeback", False) and hasattr(self.tracker, "apply_external_bbox"):
            self.tracker.apply_external_bbox(
                track.id,
                final_bbox,
                score=float(getattr(track, "confidence", 0.0)),
            )

        return tracks

    def _update_tracks_without_stereo(self, det_results: list[dict],
                                      left_img: np.ndarray,
                                      dt_s: float | None,
                                      source: str) -> list[dict]:
        primary_track = getattr(self.tracker, "track", None)
        detections = []
        for det in det_results:
            pos_3d = det.get("pos_3d")
            if pos_3d is None and primary_track is not None:
                pos_3d = primary_track.pos_3d.copy()
            if pos_3d is None:
                continue
            detections.append({
                "bbox": det["bbox"],
                "pos_3d": np.asarray(pos_3d, dtype=np.float32),
                "confidence": det["confidence"],
                "depth_stats": det.get(
                    "depth_stats",
                    getattr(primary_track, "depth_stats", None),
                ),
                "depth_confidence": det.get("depth_confidence", 0.0),
                "tracker_predicted": det.get("tracker_predicted", False),
                "source": det.get("source", source),
                "yolo_verified": det.get("yolo_verified", False),
            })

        tracks = self.tracker.update(detections, frame=left_img, dt_s=dt_s)
        self._maybe_empty_cuda_cache()
        if self.corrector is not None:
            tracks = self._maybe_correct_tracks_post(left_img, tracks)
        elif self.refiner is not None:
            tracks = self._maybe_refine_tracks_post(
                left_img,
                np.empty((0, 0), dtype=np.float32),
                tracks,
            )
        tracks = self._apply_output_bbox_filter(tracks, dt_s=dt_s)
        return self._pack_results(tracks)

    def _should_refresh_memory_track_depth(self) -> bool:
        return bool(
            self._cfg.get("tracker", {}).get("run_stereo_on_memory_tracks", False)
        )

    def _maybe_empty_cuda_cache(self, *, force: bool = False) -> None:
        if not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return

        if not bool(self.runtime_cfg.get("empty_cache_enabled", False)):
            return

        interval = int(self.runtime_cfg.get("empty_cache_interval_frames", 1))
        if interval <= 0:
            return
        if self._frame_idx <= 0 or (self._frame_idx % interval) != 0:
            return
        torch.cuda.empty_cache()

    def _apply_output_bbox_filter(self,
                                  tracks: list[FishTrack],
                                  *,
                                  dt_s: float | None) -> list[FishTrack]:
        if self.output_bbox_filter is None:
            for track in tracks:
                track.output_bbox = list(track.bbox)
            return tracks
        return self.output_bbox_filter.apply(tracks, dt_s=dt_s)

    def _refresh_tracks_from_final_bboxes(self,
                                          tracks: list[FishTrack],
                                          disparity_full: np.ndarray,
                                          left_img: np.ndarray | None = None,
                                          *,
                                          dt_s: float | None) -> list[FishTrack]:
        if disparity_full is None or disparity_full.size == 0:
            return tracks
        alpha = float(getattr(self.tracker, "smoothing_alpha", 0.7))
        for track in tracks:
            final_bbox = getattr(track, "output_bbox", None) or getattr(track, "bbox", None)
            if final_bbox is None:
                continue
            prior = self._track_roi_prior(track)
            depth_stats = self._extract_roi_depth(
                final_bbox,
                disparity_full,
                color_image=left_img,
                depth_prior_m=prior["depth_m"],
                center_prior_uv=prior["center_uv"],
                dt_s=dt_s,
            )
            center_uv = self._bbox_center_xyxy(final_bbox)
            pos_3d = self._bbox_depth_to_3d(center_uv, depth_stats)
            if pos_3d is None:
                track.replace_current_center_measurement(center_uv, dt_s=dt_s)
                continue
            track.correct_current_measurement(
                pos_3d,
                depth_stats=depth_stats,
                depth_confidence=self._depth_confidence(depth_stats),
                alpha=alpha,
                dt_s=dt_s,
                center_uv=center_uv,
            )
        return tracks

    def _resolve_model_path(self, path: str, *,
                            allow_ultralytics_alias: bool = False) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isabs(path) or os.path.exists(path):
            return os.path.normpath(path)

        cfg_dir = self._cfg.get("_config_dir")
        bases = [base for base in (cfg_dir, _MODULE_DIR, _REPO_ROOT) if base]
        for base in bases:
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        if allow_ultralytics_alias:
            return path
        return os.path.normpath(os.path.join(_REPO_ROOT, path))

    @staticmethod
    def _sanitize_bbox_xyxy(bbox, width: int, height: int,
                            min_size_px: int = 4) -> list[int] | None:
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(1.0, min(float(width), x2))
        y2 = max(1.0, min(float(height), y2))
        if (x2 - x1) < float(min_size_px) or (y2 - y1) < float(min_size_px):
            return None
        return [
            int(np.floor(x1)),
            int(np.floor(y1)),
            int(np.ceil(x2)),
            int(np.ceil(y2)),
        ]

    @staticmethod
    def _bbox_center_xyxy(bbox) -> tuple[float, float]:
        return (
            0.5 * float(bbox[0] + bbox[2]),
            0.5 * float(bbox[1] + bbox[3]),
        )

    # ── Stereo ─────────────────────────────────────────────────────────

    def _compute_disparity(self, left: np.ndarray, right: np.ndarray,
                           H0: int, W0: int) -> np.ndarray:
        """Run FoundationStereo, return disparity at original resolution."""
        return self._compute_disparity_with_options(left, right, H0, W0)

    def _compute_disparity_with_options(self,
                                        left: np.ndarray,
                                        right: np.ndarray,
                                        H0: int,
                                        W0: int,
                                        *,
                                        scale: float | None = None,
                                        low_mem: bool | None = None,
                                        valid_iters: int | None = None) -> np.ndarray:
        stereo_cfg = self._cfg["stereo"]
        scale = float(stereo_cfg.get("image_scale", 0.5) if scale is None else scale)
        low_mem = bool(stereo_cfg.get("low_memory", True) if low_mem is None else low_mem)
        valid_iters = int(stereo_cfg.get("valid_iters", 32) if valid_iters is None else valid_iters)

        # Resize
        if scale != 1.0:
            lr = cv2.resize(left, None, fx=scale, fy=scale)
            rr = cv2.resize(right, None, fx=scale, fy=scale)
        else:
            lr, rr = left, right

        # Ensure RGB
        for im in (lr, rr):
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            elif im.shape[-1] == 4:
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        H, W = lr.shape[:2]

        # To tensor (B, C, H, W) float, GPU
        t0 = torch.as_tensor(lr).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(rr).cuda().float()[None].permute(0, 3, 1, 2)

        padder = self._InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        with torch.cuda.amp.autocast(True):
            disp = self.stereo.forward(
                t0, t1, iters=valid_iters, test_mode=True, low_memory=low_mem,
            )

        disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(H, W)

        # Free stereo GPU memory
        del t0, t1, padder
        self._maybe_empty_cuda_cache()

        # Resize back to original
        if scale != 1.0:
            disp = cv2.resize(disp, (W0, H0), interpolation=cv2.INTER_LINEAR) / scale

        return disp

    def _maybe_refine_or_protect_small_target_depth(self,
                                                    det: dict,
                                                    left_img: np.ndarray,
                                                    right_img: np.ndarray,
                                                    disparity_full: np.ndarray,
                                                    depth_stats: DepthROIStats,
                                                    roi_prior: dict
                                                    ) -> tuple[DepthROIStats, np.ndarray | None, str | None]:
        cfg = (
            self._cfg.get("stereo", {})
            .get("small_target_refine", {})
        )
        if not bool(cfg.get("enabled", False)):
            return depth_stats, None, None

        bbox = det.get("bbox")
        prior_depth_m = roi_prior.get("depth_m")
        track_id = roi_prior.get("track_id")
        protected_state = self._get_active_small_target_protection_state(
            track_id,
            center_uv=det.get("center"),
            cfg=cfg,
        )
        prior_depth_confidence = float(roi_prior.get("depth_confidence", 0.0) or 0.0)
        prior_depth_valid = bool(roi_prior.get("depth_valid", False))
        prior_depth_rejected = bool(roi_prior.get("depth_rejected", False))
        prior_track_hits = int(roi_prior.get("hits", 0) or 0)
        prior_filter_mode = str(roi_prior.get("depth_filter_mode", "") or "").lower()
        trusted_prior = (
            prior_depth_m is not None
            and np.isfinite(prior_depth_m)
            and prior_depth_valid
            and (not prior_depth_rejected)
            and prior_depth_confidence >= float(cfg.get("prior_min_depth_confidence", 0.25))
            and prior_track_hits >= int(cfg.get("prior_min_track_hits", 2))
            and ("predict_only" not in prior_filter_mode)
            and ("depth_reject" not in prior_filter_mode)
        )
        trusted_prior_depth_m = (
            float(prior_depth_m)
            if trusted_prior and prior_depth_m is not None and np.isfinite(prior_depth_m)
            else None
        )
        if bool(cfg.get("require_depth_prior", True)):
            if trusted_prior_depth_m is None and protected_state is None:
                return depth_stats, None, None

        bbox_is_small_target = (
            bbox is not None
            and self._bbox_matches_small_target_policy(
                bbox,
                left_img.shape,
                cfg,
            )
        )
        if bbox is None or (protected_state is None and not bbox_is_small_target):
            return depth_stats, None, None
        force_local_refine = (
            bbox_is_small_target
            and self._bbox_matches_force_local_refine_policy(
                bbox,
                left_img.shape,
                cfg,
            )
        )
        protected_depth_m = (
            float(protected_state["depth_m"])
            if protected_state is not None
            else (
                float(trusted_prior_depth_m)
                if trusted_prior_depth_m is not None and np.isfinite(trusted_prior_depth_m)
                else None
            )
        )
        protected_center = (
            protected_state.get("center_uv")
            if protected_state is not None
            else det.get("center")
        )

        suspicious = self._is_small_target_depth_suspicious(
            depth_stats,
            prior_depth_m=protected_depth_m,
            cfg=cfg,
        )
        need_refine = (
            bool(cfg.get("local_refine_enabled", True))
            and (
                force_local_refine
                or suspicious
                or bool(cfg.get("always_refine_small_target", False))
            )
        )

        selected_stats = depth_stats
        selected_source = None
        refined_stats = None
        if need_refine and (
            force_local_refine
            or self._small_target_refine_cooldown_ready(track_id, cfg=cfg)
        ):
            refined_stats = self._try_local_stereo_refine(
                left_img,
                right_img,
                disparity_full,
                bbox,
                depth_prior_m=protected_depth_m if protected_state is not None else trusted_prior_depth_m,
                center_prior_uv=(
                    protected_center
                    if protected_state is not None
                    else roi_prior.get("center_uv")
                ),
                cfg=cfg,
            )
            if self._should_prefer_refined_depth_stats(
                original_stats=depth_stats,
                refined_stats=refined_stats,
                prior_depth_m=protected_depth_m if protected_state is not None else trusted_prior_depth_m,
                prefer_margin_m=float(cfg.get("prefer_if_closer_by_m", 0.05)),
            ):
                selected_stats = refined_stats
                selected_source = f"{det.get('source', 'yolo')}+local-stereo"
                self._mark_small_target_refine_trigger(track_id)
            elif (
                force_local_refine
                and self._force_refined_depth_is_consistent(
                    refined_stats,
                    prior_depth_m=protected_depth_m if protected_state is not None else trusted_prior_depth_m,
                    cfg=cfg,
                )
            ):
                selected_stats = refined_stats
                selected_source = f"{det.get('source', 'yolo')}+local-stereo-forced"
                self._mark_small_target_refine_trigger(track_id)

        if protected_state is not None:
            recovered = self._is_small_target_depth_recovered(
                selected_stats,
                reference_depth_m=protected_depth_m,
                cfg=cfg,
            )
            if recovered:
                recovery_count = int(protected_state.get("recovery_count", 0)) + 1
                protected_state["recovery_count"] = recovery_count
            else:
                protected_state["recovery_count"] = 0
            protected_state["last_active_frame"] = int(self._frame_idx)

            can_release = (
                int(self._frame_idx) >= int(protected_state.get("min_release_frame", self._frame_idx))
                and int(protected_state.get("recovery_count", 0))
                >= max(int(cfg.get("exit_consecutive_frames", 3)), 1)
            )
            if not can_release:
                protected_stats = self._make_protected_depth_stats(
                    protected_depth_m,
                    protected_center,
                )
                protected_pos_3d = self._depth_to_3d(
                    protected_center,
                    protected_depth_m,
                )
                suffix = "+depth-protected-hold"
                if recovered:
                    suffix = "+depth-protected-recovering"
                return (
                    protected_stats,
                    protected_pos_3d,
                    f"{det.get('source', 'yolo')}{suffix}",
                )
            self._clear_small_target_protection_state(track_id)
            return selected_stats, None, selected_source

        if not suspicious and not force_local_refine and not bool(cfg.get("always_refine_small_target", False)):
            return selected_stats, None, selected_source

        if (
            bool(cfg.get("protect_enabled", True))
            and trusted_prior_depth_m is not None
            and np.isfinite(trusted_prior_depth_m)
            and self._should_protect_small_target_depth(
                selected_stats,
                float(trusted_prior_depth_m),
                protect_depth_delta_m=float(cfg.get("protect_depth_delta_m", 0.45)),
                protect_sep_score_max=float(cfg.get("protect_sep_score_max", 0.0)),
            )
        ):
            self._mark_small_target_refine_trigger(track_id)
            self._set_small_target_protection_state(
                track_id,
                depth_m=float(trusted_prior_depth_m),
                center_uv=det.get("center"),
                cfg=cfg,
            )
            protected_center = det.get("center")
            protected_stats = self._make_protected_depth_stats(
                float(trusted_prior_depth_m),
                protected_center,
            )
            protected_pos_3d = self._depth_to_3d(protected_center, float(trusted_prior_depth_m))
            return (
                protected_stats,
                protected_pos_3d,
                f"{det.get('source', 'yolo')}+depth-protected",
            )

        return selected_stats, None, selected_source

    @staticmethod
    def _bbox_matches_small_target_policy(bbox,
                                          frame_shape,
                                          cfg: dict) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        area = bw * bh
        frame_area = max(float(width * height), 1.0)
        area_ratio = area / frame_area
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        max_area_ratio = float(cfg.get("max_bbox_area_ratio", 0.012))
        max_long_side_px = float(cfg.get("max_bbox_long_side_px", 96))
        max_short_side_px = float(cfg.get("max_bbox_short_side_px", 72))
        return (
            area_ratio <= max_area_ratio
            or long_side <= max_long_side_px
            or short_side <= max_short_side_px
        )

    @staticmethod
    def _bbox_matches_force_local_refine_policy(bbox,
                                                frame_shape,
                                                cfg: dict) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        area = bw * bh
        frame_area = max(float(width * height), 1.0)
        area_ratio = area / frame_area
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        max_area_ratio = float(
            cfg.get(
                "force_local_refine_max_bbox_area_ratio",
                cfg.get("max_bbox_area_ratio", 0.006) * 0.5,
            )
        )
        max_long_side_px = float(
            cfg.get(
                "force_local_refine_max_bbox_long_side_px",
                cfg.get("max_bbox_long_side_px", 72) * 0.8,
            )
        )
        max_short_side_px = float(
            cfg.get(
                "force_local_refine_max_bbox_short_side_px",
                cfg.get("max_bbox_short_side_px", 56) * 0.8,
            )
        )
        return (
            area_ratio <= max_area_ratio
            or long_side <= max_long_side_px
            or short_side <= max_short_side_px
        )

    @staticmethod
    def _is_small_target_depth_suspicious(depth_stats: DepthROIStats,
                                          *,
                                          prior_depth_m: float | None,
                                          cfg: dict) -> bool:
        if depth_stats is None or not depth_stats.valid:
            return bool(cfg.get("trigger_on_invalid", True))
        if prior_depth_m is None or not np.isfinite(prior_depth_m):
            return False

        z_raw = float(depth_stats.z_raw)
        if not np.isfinite(z_raw):
            return bool(cfg.get("trigger_on_invalid", True))
        sep_score = float(depth_stats.sep_score)
        trigger_sep_score_max = float(cfg.get("trigger_sep_score_max", 0.0))
        if np.isfinite(sep_score) and sep_score <= trigger_sep_score_max:
            return True

        delta = float(prior_depth_m - z_raw)
        abs_delta = abs(delta)
        min_delta = float(cfg.get("trigger_depth_delta_m", 0.35))
        max_valid_ratio = float(cfg.get("max_valid_ratio_for_trigger", 1.0))
        depth_ratio_max = float(cfg.get("trigger_depth_ratio_max", 1.0))
        near_only = bool(cfg.get("near_jump_only", False))

        if near_only and delta <= 0.0:
            return False
        if abs_delta < min_delta:
            return False
        if z_raw / max(float(prior_depth_m), 1e-6) > depth_ratio_max:
            return False
        if float(depth_stats.valid_ratio) > max_valid_ratio:
            return False
        return True

    def _small_target_refine_cooldown_ready(self,
                                            track_id: int | None,
                                            *,
                                            cfg: dict) -> bool:
        cooldown = max(int(cfg.get("cooldown_frames", 0)), 0)
        if cooldown <= 0 or track_id is None:
            return True
        last_frame = self._small_target_refine_last_frame_by_track.get(int(track_id))
        if last_frame is None:
            return True
        return (self._frame_idx - int(last_frame)) >= cooldown

    def _mark_small_target_refine_trigger(self, track_id: int | None) -> None:
        if track_id is None:
            return
        self._small_target_refine_last_frame_by_track[int(track_id)] = int(self._frame_idx)

    def _get_active_small_target_protection_state(self,
                                                  track_id: int | None,
                                                  *,
                                                  center_uv=None,
                                                  cfg: dict) -> dict | None:
        max_state_age = max(
            int(cfg.get("max_protection_state_age_frames", cfg.get("hold_frames", 0) + 24)),
            1,
        )
        track_key = None if track_id is None else int(track_id)
        if track_key is not None:
            state = self._small_target_protection_state_by_track.get(track_key)
            if state is not None:
                last_active = int(state.get("last_active_frame", state.get("min_release_frame", self._frame_idx)))
                if (int(self._frame_idx) - last_active) <= max_state_age:
                    return state
                self._small_target_protection_state_by_track.pop(track_key, None)

        if center_uv is None:
            if bool(self._cfg.get("tracker", {}).get("single_target_output_only", False)):
                active_states = list(self._small_target_protection_state_by_track.values())
                if len(active_states) == 1:
                    return active_states[0]
            return None
        fallback_max_center_distance_px = float(
            cfg.get("fallback_protection_max_center_distance_px", 96.0)
        )
        if bool(self._cfg.get("tracker", {}).get("single_target_output_only", False)):
            active_states = []
            for key, state in list(self._small_target_protection_state_by_track.items()):
                last_active = int(state.get("last_active_frame", state.get("min_release_frame", self._frame_idx)))
                if (int(self._frame_idx) - last_active) > max_state_age:
                    self._small_target_protection_state_by_track.pop(int(key), None)
                    continue
                active_states.append(state)
            if len(active_states) == 1:
                return active_states[0]
        cx, cy = float(center_uv[0]), float(center_uv[1])
        best_key = None
        best_dist = None
        for key, state in list(self._small_target_protection_state_by_track.items()):
            last_active = int(state.get("last_active_frame", state.get("min_release_frame", self._frame_idx)))
            if (int(self._frame_idx) - last_active) > max_state_age:
                self._small_target_protection_state_by_track.pop(int(key), None)
                continue
            state_center = state.get("center_uv")
            if state_center is None:
                continue
            dist = float(np.hypot(cx - float(state_center[0]), cy - float(state_center[1])))
            if dist > fallback_max_center_distance_px:
                continue
            if best_dist is None or dist < best_dist:
                best_key = int(key)
                best_dist = dist
        if best_key is None:
            return None
        return self._small_target_protection_state_by_track.get(best_key)

    def _set_small_target_protection_state(self,
                                           track_id: int | None,
                                           *,
                                           depth_m: float,
                                           center_uv,
                                           cfg: dict) -> None:
        hold_frames = max(int(cfg.get("hold_frames", 0)), 0)
        if hold_frames <= 0 or track_id is None:
            return
        if depth_m is None or not np.isfinite(depth_m):
            return
        self._small_target_protection_state_by_track[int(track_id)] = {
            "depth_m": float(depth_m),
            "center_uv": (
                None
                if center_uv is None
                else (float(center_uv[0]), float(center_uv[1]))
            ),
            "min_release_frame": int(self._frame_idx) + hold_frames,
            "recovery_count": 0,
            "last_active_frame": int(self._frame_idx),
        }

    def _clear_small_target_protection_state(self, track_id: int | None) -> None:
        if track_id is None:
            return
        self._small_target_protection_state_by_track.pop(int(track_id), None)

    def _try_local_stereo_refine(self,
                                 left_img: np.ndarray,
                                 right_img: np.ndarray,
                                 disparity_full: np.ndarray,
                                 bbox,
                                 *,
                                 depth_prior_m: float | None,
                                 center_prior_uv: tuple[float, float] | None,
                                 cfg: dict) -> DepthROIStats | None:
        height, width = left_img.shape[:2]
        crop_box = self._expand_bbox_unclipped(
            bbox,
            width,
            height,
            expand_ratio=float(cfg.get("expand_ratio", 2.6)),
            min_size_px=int(cfg.get("min_crop_size_px", 128)),
        )
        crop_left = self._extract_padded_crop(
            left_img,
            crop_box,
            border_mode=str(cfg.get("border_mode", "replicate")),
        )
        crop_right = self._extract_padded_crop(
            right_img,
            crop_box,
            border_mode=str(cfg.get("border_mode", "replicate")),
        )
        if crop_left.size == 0 or crop_right.size == 0:
            return None

        upsample_factor = max(float(cfg.get("upsample_factor", 2.0)), 1.0)
        if upsample_factor > 1.0:
            crop_left = cv2.resize(
                crop_left,
                None,
                fx=upsample_factor,
                fy=upsample_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            crop_right = cv2.resize(
                crop_right,
                None,
                fx=upsample_factor,
                fy=upsample_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        crop_left, refine_input_shape = self._pad_crop_to_refine_bucket(
            crop_left,
            cfg=cfg,
        )
        crop_right, _ = self._pad_crop_to_refine_bucket(
            crop_right,
            cfg=cfg,
        )

        disp_crop = self._compute_disparity_with_options(
            crop_left,
            crop_right,
            crop_left.shape[0],
            crop_left.shape[1],
            scale=float(cfg.get("image_scale", 1.0)),
            low_mem=bool(cfg.get("low_memory", True)),
            valid_iters=int(cfg.get("valid_iters", self._cfg["stereo"].get("valid_iters", 4))),
        )

        refine_input_h, refine_input_w = refine_input_shape
        disp_crop = disp_crop[:refine_input_h, :refine_input_w]

        raw_h = int(round(refine_input_h / upsample_factor))
        raw_w = int(round(refine_input_w / upsample_factor))
        if upsample_factor > 1.0:
            disp_crop = (
                cv2.resize(
                    disp_crop,
                    (raw_w, raw_h),
                    interpolation=cv2.INTER_LINEAR,
                ) / upsample_factor
            )

        x1, y1, x2, y2 = [int(v) for v in crop_box]
        cx1 = max(0, min(width, x1))
        cy1 = max(0, min(height, y1))
        cx2 = max(0, min(width, x2))
        cy2 = max(0, min(height, y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        pad_left = max(0, cx1 - x1)
        pad_top = max(0, cy1 - y1)
        inner_w = cx2 - cx1
        inner_h = cy2 - cy1
        disp_inner = disp_crop[
            pad_top: pad_top + inner_h,
            pad_left: pad_left + inner_w,
        ]
        if disp_inner.shape[:2] != (inner_h, inner_w):
            return None

        refined_map = disparity_full.copy()
        refined_map[cy1:cy2, cx1:cx2] = disp_inner
        return self._extract_roi_depth(
            bbox,
            refined_map,
            color_image=left_img,
            depth_prior_m=depth_prior_m,
            center_prior_uv=center_prior_uv,
        )

    @staticmethod
    def _should_prefer_refined_depth_stats(*,
                                           original_stats: DepthROIStats,
                                           refined_stats: DepthROIStats | None,
                                           prior_depth_m: float | None,
                                           prefer_margin_m: float) -> bool:
        if refined_stats is None:
            return False
        if not refined_stats.valid:
            return False
        if not original_stats.valid:
            return True

        if prior_depth_m is not None and np.isfinite(prior_depth_m):
            orig_err = abs(float(original_stats.z_raw) - float(prior_depth_m))
            ref_err = abs(float(refined_stats.z_raw) - float(prior_depth_m))
            if ref_err + max(float(prefer_margin_m), 0.0) < orig_err:
                return True

        if refined_stats.valid_ratio > (original_stats.valid_ratio + 0.08):
            return True
        if refined_stats.z_iqr < (original_stats.z_iqr * 0.7):
            return True
        return False

    @staticmethod
    def _force_refined_depth_is_consistent(refined_stats: DepthROIStats | None,
                                           *,
                                           prior_depth_m: float | None,
                                           cfg: dict) -> bool:
        if refined_stats is None or not refined_stats.valid:
            return False
        if prior_depth_m is None or not np.isfinite(prior_depth_m):
            return False
        z_raw = float(refined_stats.z_raw)
        if not np.isfinite(z_raw):
            return False
        abs_delta = abs(z_raw - float(prior_depth_m))
        abs_limit = float(
            cfg.get(
                "force_accept_max_depth_delta_m",
                max(
                    float(cfg.get("protect_depth_delta_m", 0.45)),
                    float(cfg.get("trigger_depth_delta_m", 0.45)),
                ),
            )
        )
        rel_limit = float(cfg.get("force_accept_max_depth_rel_ratio", 0.35))
        return abs_delta <= max(abs_limit, rel_limit * max(abs(float(prior_depth_m)), 1e-6))

    @staticmethod
    def _should_protect_small_target_depth(depth_stats: DepthROIStats,
                                           prior_depth_m: float,
                                           *,
                                           protect_depth_delta_m: float,
                                           protect_sep_score_max: float = 0.0) -> bool:
        if prior_depth_m is None or not np.isfinite(prior_depth_m):
            return False
        if depth_stats is None or not depth_stats.valid:
            return True
        sep_score = float(depth_stats.sep_score)
        if np.isfinite(sep_score) and sep_score <= float(protect_sep_score_max):
            return True
        return abs(float(depth_stats.z_raw) - float(prior_depth_m)) >= max(
            float(protect_depth_delta_m), 0.0
        )

    @staticmethod
    def _is_small_target_depth_recovered(depth_stats: DepthROIStats,
                                         *,
                                         reference_depth_m: float | None,
                                         cfg: dict) -> bool:
        if reference_depth_m is None or not np.isfinite(reference_depth_m):
            return False
        if depth_stats is None or not depth_stats.valid:
            return False
        z_raw = float(depth_stats.z_raw)
        if not np.isfinite(z_raw):
            return False
        exit_depth_delta_m = float(cfg.get("exit_depth_delta_m", 0.20))
        exit_min_valid_ratio = float(cfg.get("exit_min_valid_ratio", 0.18))
        exit_min_sep_score = float(cfg.get("exit_min_sep_score", 0.20))
        exit_min_core_px = int(cfg.get("exit_min_core_px", 120))
        if abs(z_raw - float(reference_depth_m)) > max(exit_depth_delta_m, 0.0):
            return False
        if float(depth_stats.valid_ratio) < exit_min_valid_ratio:
            return False
        if int(depth_stats.core_px) < exit_min_core_px:
            return False
        sep_score = float(depth_stats.sep_score)
        if np.isfinite(sep_score) and sep_score < exit_min_sep_score:
            return False
        return True

    @staticmethod
    def _make_protected_depth_stats(prior_depth_m: float,
                                    center_uv: tuple[float, float] | None) -> DepthROIStats:
        return DepthROIStats(
            z_raw=float(prior_depth_m),
            z_median=float(prior_depth_m),
            z_iqr=float("inf"),
            valid_ratio=0.0,
            depth_histogram=[],
            depth_histogram_edges=[],
            center_uv=(
                (float(center_uv[0]), float(center_uv[1]))
                if center_uv is not None
                else None
            ),
            core_px=0,
            sep_score=float("nan"),
            valid=False,
        )

    # ── 3D reconstruction ──────────────────────────────────────────────

    def _extract_roi_depth_impl(self, bbox,
                                disparity_map: np.ndarray,
                                color_image: np.ndarray | None = None,
                                depth_prior_m: float | None = None,
                                center_prior_uv: tuple[float, float] | None = None,
                                dt_s: float | None = None,
                                *,
                                return_debug: bool = False
                                ) -> DepthROIStats | tuple[DepthROIStats, DepthROIDebug]:
        """Aggregate target depth from a detection bbox, optionally with debug masks."""
        cfg = self._cfg.get("temporal_depth", {}).get("roi", {})
        center_fraction = float(cfg.get("center_fraction", 0.7))
        min_disp = float(cfg.get("min_disparity_px", 0.5))
        min_depth = float(cfg.get("min_depth_m", 0.2))
        max_depth = float(cfg.get("max_depth_m", 20.0))
        trim = float(cfg.get("trim_fraction", 0.1))
        min_valid_ratio = float(cfg.get("min_valid_ratio", 0.15))
        hist_bins = int(cfg.get("histogram_bins", 16))
        max_valid_iqr = float(cfg.get("max_valid_iqr_m", float("inf")))
        use_depth_prior = bool(cfg.get("use_depth_prior", False))
        depth_prior_margin_m = float(cfg.get("depth_prior_margin_m", 0.35))
        depth_prior_min_pixels = int(cfg.get("depth_prior_min_pixels", 12))
        depth_prior_min_ratio = float(cfg.get("depth_prior_min_ratio", 0.03))
        depth_prior_sigma_m = max(
            float(cfg.get("depth_prior_sigma_m", 0.25)), 1e-6)
        use_center_prior = bool(cfg.get("use_center_prior", False))
        center_prior_radius_px = float(cfg.get("center_prior_radius_px", 45.0))
        center_prior_min_pixels = int(cfg.get("center_prior_min_pixels", 12))
        center_prior_min_ratio = float(cfg.get("center_prior_min_ratio", 0.03))
        center_prior_sigma_px = max(
            float(cfg.get("center_prior_sigma_px", 24.0)), 1e-6)
        use_connected_component = bool(cfg.get("use_connected_component", False))
        min_component_area = int(cfg.get("min_component_area_px", 20))

        H, W = disparity_map.shape
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            empty = self._empty_depth_stats()
            if not return_debug:
                return empty
            return empty, DepthROIDebug(
                bbox_xyxy=(x1, y1, x2, y2),
                roi_bbox_xyxy=(x1, y1, x2, y2),
                center_fraction=float(np.clip(center_fraction, 0.1, 1.0)),
                roi_disparity=np.empty((0, 0), dtype=np.float32),
                roi_depth=np.empty((0, 0), dtype=np.float32),
                valid_mask_initial=np.zeros((0, 0), dtype=bool),
                valid_mask_after_foreground=np.zeros((0, 0), dtype=bool),
                valid_mask_after_color=np.zeros((0, 0), dtype=bool),
                valid_mask_after_depth_prior=np.zeros((0, 0), dtype=bool),
                valid_mask_after_center_prior=np.zeros((0, 0), dtype=bool),
                valid_mask_final=np.zeros((0, 0), dtype=bool),
                valid_mask_depth_core=np.zeros((0, 0), dtype=bool),
                valid_mask_background_ring=np.zeros((0, 0), dtype=bool),
                selected_component_mask=None,
                center_prior_roi=None,
                depth_prior_m=(
                    float(depth_prior_m)
                    if depth_prior_m is not None and np.isfinite(depth_prior_m)
                    else None
                ),
            )

        frac = float(np.clip(center_fraction, 0.1, 1.0))
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        roi_disp = disparity_map[ry1:ry2, rx1:rx2]
        roi_pixels = int(roi_disp.size)
        if roi_pixels == 0:
            empty = self._empty_depth_stats()
            if not return_debug:
                return empty
            return empty, DepthROIDebug(
                bbox_xyxy=(x1, y1, x2, y2),
                roi_bbox_xyxy=(rx1, ry1, rx2, ry2),
                center_fraction=frac,
                roi_disparity=np.empty((0, 0), dtype=np.float32),
                roi_depth=np.empty((0, 0), dtype=np.float32),
                valid_mask_initial=np.zeros((0, 0), dtype=bool),
                valid_mask_after_foreground=np.zeros((0, 0), dtype=bool),
                valid_mask_after_color=np.zeros((0, 0), dtype=bool),
                valid_mask_after_depth_prior=np.zeros((0, 0), dtype=bool),
                valid_mask_after_center_prior=np.zeros((0, 0), dtype=bool),
                valid_mask_final=np.zeros((0, 0), dtype=bool),
                valid_mask_depth_core=np.zeros((0, 0), dtype=bool),
                valid_mask_background_ring=np.zeros((0, 0), dtype=bool),
                selected_component_mask=None,
                center_prior_roi=None,
                depth_prior_m=(
                    float(depth_prior_m)
                    if depth_prior_m is not None and np.isfinite(depth_prior_m)
                    else None
                ),
            )

        cam = self._cfg["camera"]
        fx = float(cam["fx"])
        B = float(cam["baseline_m"])
        finite = np.isfinite(roi_disp)
        disp_ok = roi_disp > min_disp
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = fx * B / roi_disp
        depth_ok = (depth >= min_depth) & (depth <= max_depth)
        valid_mask = finite & disp_ok & np.isfinite(depth) & depth_ok
        valid_mask_initial = valid_mask.copy()
        mask_version = str(cfg.get("mask_version", "legacy")).lower()
        if mask_version in {
            "external_ring_mahalanobis_otsu",
            "mahalanobis_otsu_external_ring",
            "color_disp_intersection",
            "external_ring",
        }:
            valid_mask = self._apply_foreground_disparity_mask_otsu(
                valid_mask,
                roi_disp,
                cfg=cfg,
            )
        else:
            valid_mask = self._apply_foreground_depth_mask(
                valid_mask,
                depth,
                depth_prior_m=depth_prior_m,
                cfg=cfg,
            )
        valid_mask_after_foreground = valid_mask.copy()
        color_mask = self._apply_color_foreground_mask(
            valid_mask_initial,
            color_image=color_image,
            roi_bbox_xyxy=(rx1, ry1, rx2, ry2),
            cfg=cfg,
        )
        valid_mask_after_color = color_mask.copy()
        valid_mask = self._fuse_foreground_masks(
            valid_mask_initial,
            depth_mask=valid_mask_after_foreground,
            color_mask=valid_mask_after_color,
            cfg=cfg,
        )
        center_prior_roi = self._project_prior_center_to_roi(
            center_prior_uv, rx1, ry1, rx2, ry2)
        temporal_prior_recovery_enabled = bool(
            cfg.get("temporal_prior_recovery_enabled", False)
        )
        if temporal_prior_recovery_enabled:
            prior_support_mask = self._build_temporal_prior_support_mask(
                valid_mask_initial,
                roi_disp,
                center_prior_uv=center_prior_roi if use_center_prior else None,
                depth_prior_m=float(depth_prior_m) if use_depth_prior and depth_prior_m is not None and np.isfinite(depth_prior_m) else None,
                fx=fx,
                baseline_m=B,
                cfg=cfg,
                dt_s=dt_s,
                reference_dt_s=float(
                    self._cfg.get("temporal_depth", {}).get(
                        "fallback_dt_s",
                        self._cfg.get("dt_s", 0.1),
                    )
                ),
            )
            weak_mask = valid_mask_after_foreground | valid_mask_after_color
            valid_mask = self._recover_mask_with_temporal_prior(
                valid_mask,
                weak_mask=weak_mask,
                prior_support_mask=prior_support_mask,
                bbox_area_px=max(int((x2 - x1) * (y2 - y1)), 1),
                cfg=cfg,
            )
            valid_mask_after_depth_prior = prior_support_mask.copy()
            valid_mask_after_center_prior = valid_mask.copy()
        else:
            if (
                use_depth_prior
                and depth_prior_m is not None
                and np.isfinite(depth_prior_m)
            ):
                valid_mask = self._apply_depth_prior_mask(
                    valid_mask,
                    depth,
                    float(depth_prior_m),
                    margin_m=depth_prior_margin_m,
                    min_pixels=depth_prior_min_pixels,
                    min_ratio=depth_prior_min_ratio,
                )
            valid_mask_after_depth_prior = valid_mask.copy()
            if use_center_prior and center_prior_roi is not None:
                valid_mask = self._apply_center_prior_mask(
                    valid_mask,
                    center_prior_roi,
                    radius_px=center_prior_radius_px,
                    min_pixels=center_prior_min_pixels,
                    min_ratio=center_prior_min_ratio,
                )
            valid_mask_after_center_prior = valid_mask.copy()
        selected_component_mask = None
        if use_connected_component:
            component_mask = self._select_target_component(
                valid_mask,
                min_component_area,
                depth_map=depth,
                center_prior_uv=center_prior_roi,
                center_prior_sigma_px=center_prior_sigma_px,
                depth_prior_m=(
                    float(depth_prior_m)
                    if use_depth_prior
                    and depth_prior_m is not None
                    and np.isfinite(depth_prior_m)
                    else None
                ),
                depth_prior_sigma_m=depth_prior_sigma_m,
            )
            selected_component_mask = component_mask.copy()
            valid_mask = component_mask
        valid_mask_full = valid_mask.copy()

        center_mask = self._select_center_mask(
            color_mask=valid_mask_after_color,
            final_mask=valid_mask_full,
            center_prior_uv=center_prior_roi,
            center_prior_sigma_px=center_prior_sigma_px,
            min_component_area=min_component_area,
            cfg=cfg,
        )
        center_roi = self._estimate_mask_geometric_center(
            center_mask,
            method=cfg.get("center_geometry", "ellipse_or_min_rect"),
        )
        center_uv = None
        if center_roi is not None:
            center_uv = (float(rx1 + center_roi[0]), float(ry1 + center_roi[1]))

        depth_mask = self._extract_depth_core_mask(valid_mask_full, cfg=cfg)
        background_ring_mask = self._extract_background_ring_mask(
            valid_mask_initial,
            valid_mask_full,
            depth_mask,
            cfg=cfg,
        )
        valid_depth = depth[depth_mask].astype(np.float64)
        valid_ratio = float(valid_depth.size / roi_pixels)
        core_px = int(np.count_nonzero(depth_mask))
        sep_score = self._compute_depth_separation_score(
            roi_disp,
            depth_mask,
            background_ring_mask,
        )

        if valid_depth.size == 0:
            stats = DepthROIStats(
                z_raw=np.nan,
                z_median=np.nan,
                z_iqr=np.inf,
                valid_ratio=valid_ratio,
                depth_histogram=[],
                depth_histogram_edges=[],
                center_uv=center_uv,
                core_px=0,
                sep_score=np.nan,
                valid=False,
            )
            if not return_debug:
                return stats
            return stats, DepthROIDebug(
                bbox_xyxy=(x1, y1, x2, y2),
                roi_bbox_xyxy=(rx1, ry1, rx2, ry2),
                center_fraction=frac,
                roi_disparity=roi_disp.copy(),
                roi_depth=depth.astype(np.float32, copy=True),
                valid_mask_initial=valid_mask_initial,
                valid_mask_after_foreground=valid_mask_after_foreground,
                valid_mask_after_color=valid_mask_after_color,
                valid_mask_after_depth_prior=valid_mask_after_depth_prior,
                valid_mask_after_center_prior=valid_mask_after_center_prior,
                valid_mask_final=valid_mask_full,
                valid_mask_depth_core=depth_mask,
                valid_mask_background_ring=background_ring_mask,
                selected_component_mask=selected_component_mask,
                center_prior_roi=center_prior_roi,
                depth_prior_m=(
                    float(depth_prior_m)
                    if depth_prior_m is not None and np.isfinite(depth_prior_m)
                    else None
                ),
            )

        valid_depth.sort()
        trim = float(np.clip(trim, 0.0, 0.45))
        cut = int(round(valid_depth.size * trim))
        trimmed = valid_depth[cut: valid_depth.size - cut] if cut > 0 else valid_depth
        if trimmed.size == 0:
            trimmed = valid_depth

        z_median = float(np.median(trimmed))
        q25, q75 = np.percentile(trimmed, [25, 75])
        z_iqr = float(q75 - q25)
        hist, edges = np.histogram(trimmed, bins=hist_bins)
        hist_norm = hist.astype(np.float64)
        if hist_norm.sum() > 0:
            hist_norm /= hist_norm.sum()

        stats = DepthROIStats(
            z_raw=z_median,
            z_median=z_median,
            z_iqr=z_iqr,
            valid_ratio=valid_ratio,
            depth_histogram=hist_norm.tolist(),
            depth_histogram_edges=edges.astype(np.float64).tolist(),
            center_uv=center_uv,
            core_px=core_px,
            sep_score=sep_score,
            valid=(
                valid_ratio >= min_valid_ratio
                and z_iqr <= max_valid_iqr
            ),
        )
        if not return_debug:
            return stats
        return stats, DepthROIDebug(
            bbox_xyxy=(x1, y1, x2, y2),
            roi_bbox_xyxy=(rx1, ry1, rx2, ry2),
            center_fraction=frac,
            roi_disparity=roi_disp.copy(),
            roi_depth=depth.astype(np.float32, copy=True),
            valid_mask_initial=valid_mask_initial,
            valid_mask_after_foreground=valid_mask_after_foreground,
            valid_mask_after_color=valid_mask_after_color,
            valid_mask_after_depth_prior=valid_mask_after_depth_prior,
            valid_mask_after_center_prior=valid_mask_after_center_prior,
            valid_mask_final=valid_mask_full,
            valid_mask_depth_core=depth_mask,
            valid_mask_background_ring=background_ring_mask,
            selected_component_mask=selected_component_mask,
            center_prior_roi=center_prior_roi,
            depth_prior_m=(
                float(depth_prior_m)
                if depth_prior_m is not None and np.isfinite(depth_prior_m)
                else None
            ),
        )

    def _extract_roi_depth(self, bbox,
                           disparity_map: np.ndarray,
                           color_image: np.ndarray | None = None,
                           depth_prior_m: float | None = None,
                           center_prior_uv: tuple[float, float] | None = None,
                           dt_s: float | None = None) -> DepthROIStats:
        return self._extract_roi_depth_impl(
            bbox,
            disparity_map,
            color_image=color_image,
            depth_prior_m=depth_prior_m,
            center_prior_uv=center_prior_uv,
            dt_s=dt_s,
            return_debug=False,
        )

    def extract_roi_depth_debug(self,
                                bbox,
                                disparity_map: np.ndarray,
                                color_image: np.ndarray | None = None,
                                depth_prior_m: float | None = None,
                                center_prior_uv: tuple[float, float] | None = None,
                                dt_s: float | None = None,
                                ) -> tuple[DepthROIStats, DepthROIDebug]:
        """Public debug helper: return ROI depth stats plus intermediate masks."""
        stats, debug = self._extract_roi_depth_impl(
            bbox,
            disparity_map,
            color_image=color_image,
            depth_prior_m=depth_prior_m,
            center_prior_uv=center_prior_uv,
            dt_s=dt_s,
            return_debug=True,
        )
        return stats, debug

    @staticmethod
    def _empty_depth_stats() -> DepthROIStats:
        return DepthROIStats(
            z_raw=np.nan,
            z_median=np.nan,
            z_iqr=np.inf,
            valid_ratio=0.0,
            depth_histogram=[],
            depth_histogram_edges=[],
            center_uv=None,
            core_px=0,
            sep_score=np.nan,
            valid=False,
        )

    def _depth_confidence(self, depth_stats: DepthROIStats) -> float:
        """Depth-only confidence; deliberately excludes YOLO confidence."""
        if depth_stats is None or not depth_stats.valid:
            return 0.0
        cfg = self._cfg.get("temporal_depth", {}).get("confidence", {})
        sigma_iqr = max(float(cfg.get("sigma_iqr_m", 0.4)), 1e-6)
        c_valid = float(np.clip(depth_stats.valid_ratio, 0.0, 1.0))
        c_spread = float(np.exp(-max(depth_stats.z_iqr, 0.0) / sigma_iqr))
        return float(np.clip(c_valid * c_spread, 0.0, 1.0))

    def _bbox_depth_to_3d(self, center_uv, depth_stats: DepthROIStats):
        if depth_stats is None or not depth_stats.valid:
            return None
        # Final LOS/output center now follows the detector bbox center
        # directly. The ROI mask is used only to select stable depth pixels.
        return self._depth_to_3d(center_uv, depth_stats.z_raw)

    @staticmethod
    def _estimate_mask_geometric_center(mask: np.ndarray,
                                        *,
                                        method: str = "ellipse_or_min_rect") -> tuple[float, float] | None:
        if mask.size == 0 or not np.any(mask):
            return None

        yy, xx = np.nonzero(mask)
        if xx.size == 0 or yy.size == 0:
            return None

        pts = np.column_stack([xx.astype(np.float32), yy.astype(np.float32)])
        method = str(method or "ellipse_or_min_rect").lower()

        if method in {"ellipse", "ellipse_or_min_rect", "ellipse_or_minarea"} and pts.shape[0] >= 5:
            try:
                (cx, cy), _, _ = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                if np.isfinite(cx) and np.isfinite(cy):
                    return (float(cx), float(cy))
            except cv2.error:
                pass

        if method in {"min_rect", "minarea", "min_area_rect", "ellipse_or_min_rect", "ellipse_or_minarea"} and pts.shape[0] >= 3:
            try:
                (cx, cy), _, _ = cv2.minAreaRect(pts.reshape(-1, 1, 2))
                if np.isfinite(cx) and np.isfinite(cy):
                    return (float(cx), float(cy))
            except cv2.error:
                pass

        return (float(np.mean(xx)), float(np.mean(yy)))

    @staticmethod
    def _select_center_mask(*,
                            color_mask: np.ndarray,
                            final_mask: np.ndarray,
                            center_prior_uv: tuple[float, float] | None,
                            center_prior_sigma_px: float,
                            min_component_area: int,
                            cfg: dict) -> np.ndarray:
        if color_mask.size == 0 and final_mask.size == 0:
            return np.zeros((0, 0), dtype=bool)

        use_color_as_primary = bool(cfg.get("center_use_color_mask", True))
        support_dilate_iters = max(int(cfg.get("center_support_dilate_iterations", 1)), 0)

        primary_mask = color_mask.copy() if use_color_as_primary else final_mask.copy()
        fallback_mask = final_mask.copy() if use_color_as_primary else color_mask.copy()
        if primary_mask.size == 0 or not np.any(primary_mask):
            primary_mask = fallback_mask.copy()

        if (
            primary_mask.size > 0
            and np.any(primary_mask)
            and final_mask.size == primary_mask.size
            and np.any(final_mask)
        ):
            support_mask = final_mask
            if support_dilate_iters > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                support_mask = cv2.dilate(
                    final_mask.astype(np.uint8),
                    kernel,
                    iterations=support_dilate_iters,
                ).astype(bool)
            overlap = primary_mask & support_mask
            if np.any(overlap):
                primary_mask = overlap

        if primary_mask.size == 0 or not np.any(primary_mask):
            return final_mask

        if not bool(cfg.get("center_select_main_component", True)):
            return primary_mask

        component_min_area = max(
            int(cfg.get("center_component_min_area_px", min_component_area)),
            1,
        )
        return FishPositionEstimator._select_target_component(
            primary_mask,
            component_min_area,
            center_prior_uv=center_prior_uv,
            center_prior_sigma_px=center_prior_sigma_px,
        )

    @staticmethod
    def _select_target_component(
        valid_mask: np.ndarray,
        min_component_area: int,
        *,
        depth_map: np.ndarray | None = None,
        center_prior_uv: tuple[float, float] | None = None,
        center_prior_sigma_px: float = 24.0,
        depth_prior_m: float | None = None,
        depth_prior_sigma_m: float = 0.25,
    ) -> np.ndarray:
        """Keep only the connected component most likely to be the target body."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask

        mask_u8 = valid_mask.astype(np.uint8, copy=False)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return valid_mask

        h, w = mask_u8.shape
        diag = max(float(np.hypot(h, w)), 1.0)
        roi_area = max(float(h * w), 1.0)
        roi_center = np.array([w * 0.5, h * 0.5], dtype=np.float64)
        best_label = None
        best_score = None

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(1, min_component_area):
                continue
            centroid = centroids[label]
            dist = float(np.linalg.norm(centroid - roi_center))
            # Prefer central, sufficiently large components. When we have a
            # prior depth from the previous frame, also bias toward components
            # whose median depth stays close to that prior.
            score = dist / diag - 0.2 * np.sqrt(area / roi_area)
            if center_prior_uv is not None:
                center_prior_dist = float(
                    np.linalg.norm(centroid - np.asarray(center_prior_uv, dtype=np.float64))
                )
                score += 0.5 * (center_prior_dist / max(center_prior_sigma_px, 1e-6))
            if (
                depth_map is not None
                and depth_prior_m is not None
                and np.isfinite(depth_prior_m)
            ):
                component_depth = depth_map[labels == label]
                component_depth = component_depth[np.isfinite(component_depth)]
                if component_depth.size > 0:
                    depth_delta = abs(float(np.median(component_depth)) - depth_prior_m)
                    score += depth_delta / max(depth_prior_sigma_m, 1e-6)
            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return valid_mask
        return labels == best_label

    @staticmethod
    def _apply_depth_prior_mask(valid_mask: np.ndarray,
                                depth_map: np.ndarray,
                                depth_prior_m: float,
                                *,
                                margin_m: float,
                                min_pixels: int,
                                min_ratio: float) -> np.ndarray:
        """Keep only depths close to the previous-frame target depth when possible."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask

        prior_mask = (
            valid_mask
            & np.isfinite(depth_map)
            & (np.abs(depth_map - depth_prior_m) <= max(float(margin_m), 1e-6))
        )
        prior_pixels = int(np.count_nonzero(prior_mask))
        required_pixels = max(int(min_pixels), int(round(valid_mask.size * max(min_ratio, 0.0))))
        if prior_pixels >= max(required_pixels, 1):
            return prior_mask
        return valid_mask

    @staticmethod
    def _extract_depth_core_mask(valid_mask: np.ndarray, *, cfg: dict) -> np.ndarray:
        """Keep only the stable inner core of the selected foreground for Z."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask

        if not bool(cfg.get("use_foreground_core_for_depth", False)):
            return valid_mask

        frac = float(np.clip(cfg.get("center_fraction", 0.7), 0.1, 1.0))
        keep_fraction = float(np.clip(cfg.get("foreground_core_keep_fraction", 0.45), 0.05, 1.0))
        min_pixels = max(int(cfg.get("foreground_core_min_pixels", 12)), 1)
        min_ratio = max(float(cfg.get("foreground_core_min_ratio_of_foreground", 0.20)), 0.0)
        fg_pixels = int(np.count_nonzero(valid_mask))
        required_pixels = max(min_pixels, int(round(fg_pixels * min_ratio)))

        if frac >= 0.999 and keep_fraction >= 0.999:
            return valid_mask

        yy, xx = np.nonzero(valid_mask)
        if xx.size > 0 and yy.size > 0:
            fx1, fx2 = int(xx.min()), int(xx.max()) + 1
            fy1, fy2 = int(yy.min()), int(yy.max()) + 1
            fw = fx2 - fx1
            fh = fy2 - fy1
            pad_x = int(round((1.0 - frac) * fw * 0.5))
            pad_y = int(round((1.0 - frac) * fh * 0.5))
            cx1 = max(0, fx1 + pad_x)
            cy1 = max(0, fy1 + pad_y)
            cx2 = min(valid_mask.shape[1], fx2 - pad_x)
            cy2 = min(valid_mask.shape[0], fy2 - pad_y)
            if cx2 > cx1 and cy2 > cy1:
                center_box_mask = np.zeros_like(valid_mask, dtype=bool)
                center_box_mask[cy1:cy2, cx1:cx2] = True
                center_core = valid_mask & center_box_mask
                if int(np.count_nonzero(center_core)) >= required_pixels:
                    return center_core

        mask_u8 = valid_mask.astype(np.uint8, copy=False)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        dist = dist.astype(np.float32, copy=False)
        positive = dist[dist > 0]
        if positive.size == 0:
            return valid_mask

        threshold_q = float(np.clip(1.0 - keep_fraction, 0.0, 0.95))
        threshold = float(np.quantile(positive, threshold_q))
        core_mask = valid_mask & (dist >= threshold)
        if int(np.count_nonzero(core_mask)) >= required_pixels:
            return core_mask

        # Fallback: binary erosion keeps the center-most support without
        # over-pruning thin targets.
        erode_iters = max(int(cfg.get("foreground_core_erode_iterations", 1)), 1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=erode_iters).astype(bool)
        eroded &= valid_mask
        if int(np.count_nonzero(eroded)) >= required_pixels:
            return eroded

        return valid_mask

    @staticmethod
    def _extract_background_ring_mask(valid_mask_initial: np.ndarray,
                                      foreground_mask: np.ndarray,
                                      core_mask: np.ndarray,
                                      *,
                                      cfg: dict) -> np.ndarray:
        """Build a local background ring around the foreground target."""
        if valid_mask_initial.size == 0 or not np.any(valid_mask_initial):
            return np.zeros_like(valid_mask_initial, dtype=bool)
        seed_mask = foreground_mask if np.any(foreground_mask) else core_mask
        if seed_mask.size == 0 or not np.any(seed_mask):
            return np.zeros_like(valid_mask_initial, dtype=bool)

        kernel = np.ones((3, 3), dtype=np.uint8)
        ring_iters = max(int(cfg.get("background_ring_dilate_iterations", 3)), 1)
        outer = cv2.dilate(seed_mask.astype(np.uint8), kernel, iterations=ring_iters).astype(bool)
        ring = outer & valid_mask_initial & ~seed_mask

        min_bg_px = max(int(cfg.get("background_ring_min_pixels", 20)), 1)
        if int(np.count_nonzero(ring)) >= min_bg_px:
            return ring

        fallback = valid_mask_initial & ~core_mask
        if int(np.count_nonzero(fallback)) >= min_bg_px:
            return fallback
        return ring

    @staticmethod
    def _compute_depth_separation_score(roi_disp: np.ndarray,
                                        core_mask: np.ndarray,
                                        background_ring_mask: np.ndarray) -> float:
        """Estimate how separable the target disparity is from local background."""
        if roi_disp.size == 0 or core_mask.size == 0 or background_ring_mask.size == 0:
            return float("nan")

        fg_disp = roi_disp[core_mask]
        bg_disp = roi_disp[background_ring_mask]
        fg_disp = fg_disp[np.isfinite(fg_disp) & (fg_disp > 0.5)]
        bg_disp = bg_disp[np.isfinite(bg_disp) & (bg_disp > 0.5)]
        if fg_disp.size == 0 or bg_disp.size == 0:
            return float("nan")

        disp_fg = float(np.median(fg_disp))
        disp_bg = float(np.median(bg_disp))
        mad_fg = float(np.median(np.abs(fg_disp - disp_fg)))
        mad_bg = float(np.median(np.abs(bg_disp - disp_bg)))
        return float((disp_fg - disp_bg) / max(mad_fg + mad_bg, 1e-6))

    @staticmethod
    def _otsu_threshold_1d(values: np.ndarray,
                           *,
                           bins: int = 128) -> float | None:
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi <= lo + 1e-9:
            return lo

        hist, edges = np.histogram(vals, bins=max(int(bins), 8), range=(lo, hi))
        if hist.size == 0 or int(hist.sum()) <= 0:
            return float(np.median(vals))

        prob = hist.astype(np.float64)
        prob /= max(float(prob.sum()), 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        omega = np.cumsum(prob)
        mu = np.cumsum(prob * centers)
        mu_t = float(mu[-1])

        sigma_b2 = np.full_like(omega, -np.inf, dtype=np.float64)
        denom = omega * (1.0 - omega)
        valid = denom > 1e-12
        sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
        return float(centers[int(np.argmax(sigma_b2))])

    @staticmethod
    def _expand_bbox_xyxy(bbox_xyxy: tuple[int, int, int, int],
                          *,
                          scale: float,
                          image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = image_shape
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(float(x2 - x1), 1.0)
        bh = max(float(y2 - y1), 1.0)
        scale = max(float(scale), 1.0)
        half_w = 0.5 * bw * scale
        half_h = 0.5 * bh * scale
        ex1 = max(0, int(np.floor(cx - half_w)))
        ey1 = max(0, int(np.floor(cy - half_h)))
        ex2 = min(w, int(np.ceil(cx + half_w)))
        ey2 = min(h, int(np.ceil(cy + half_h)))
        return ex1, ey1, ex2, ey2

    @staticmethod
    def _circular_hsv_features(hsv_image: np.ndarray) -> np.ndarray:
        hsv = hsv_image.astype(np.float32, copy=False)
        hue = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        return np.stack(
            [sat * np.cos(hue), sat * np.sin(hue), val],
            axis=-1,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _apply_color_foreground_mask(valid_mask: np.ndarray,
                                     *,
                                     color_image: np.ndarray | None,
                                     roi_bbox_xyxy: tuple[int, int, int, int],
                                     cfg: dict) -> np.ndarray:
        method = str(cfg.get("color_fg_method", "legacy")).lower()
        if method in {
            "external_ring_mahalanobis_otsu",
            "mahalanobis_otsu_external_ring",
            "external_ring",
            "mahalanobis_otsu",
        }:
            return FishPositionEstimator._apply_color_foreground_mask_external_ring(
                valid_mask,
                color_image=color_image,
                roi_bbox_xyxy=roi_bbox_xyxy,
                cfg=cfg,
            )
        return FishPositionEstimator._apply_color_foreground_mask_legacy(
            valid_mask,
            color_image=color_image,
            roi_bbox_xyxy=roi_bbox_xyxy,
            cfg=cfg,
        )

    @staticmethod
    def _apply_color_foreground_mask_legacy(valid_mask: np.ndarray,
                                            *,
                                            color_image: np.ndarray | None,
                                            roi_bbox_xyxy: tuple[int, int, int, int],
                                            cfg: dict) -> np.ndarray:
        """Select foreground pixels by local color contrast to the ROI border."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask
        if color_image is None or color_image.size == 0:
            return valid_mask
        if not bool(cfg.get("use_color_foreground_mask", False)):
            return valid_mask

        x1, y1, x2, y2 = roi_bbox_xyxy
        h, w = color_image.shape[:2]
        x1 = max(0, min(w, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return valid_mask

        roi_bgr = color_image[y1:y2, x1:x2]
        if roi_bgr.shape[:2] != valid_mask.shape:
            return valid_mask

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32, copy=False)
        hue = roi_hsv[:, :, 0]
        sat = roi_hsv[:, :, 1]
        val = roi_hsv[:, :, 2]

        border_fraction = float(np.clip(cfg.get("color_fg_border_fraction", 0.18), 0.05, 0.45))
        margin_x = max(int(round(roi_hsv.shape[1] * border_fraction)), 1)
        margin_y = max(int(round(roi_hsv.shape[0] * border_fraction)), 1)
        border_mask = np.zeros(valid_mask.shape, dtype=bool)
        border_mask[:margin_y, :] = True
        border_mask[-margin_y:, :] = True
        border_mask[:, :margin_x] = True
        border_mask[:, -margin_x:] = True

        bg_seed = border_mask
        bg_hsv_center = None
        if np.any(bg_seed):
            bg_hsv = roi_hsv[bg_seed]
            if bg_hsv.size > 0:
                bg_hsv_center = np.median(bg_hsv, axis=0)
        if bg_hsv_center is None:
            bg_hsv_center = cfg.get("background_hsv_center")
            if bg_hsv_center is None or len(bg_hsv_center) != 3:
                return valid_mask
        bg_h, bg_s, bg_v = [float(v) for v in bg_hsv_center]

        hue_tol = max(float(cfg.get("color_fg_hue_tolerance", 14.0)), 1e-6)
        sat_tol = max(float(cfg.get("color_fg_sat_tolerance", 52.0)), 1e-6)
        val_tol = max(float(cfg.get("color_fg_val_tolerance", 52.0)), 1e-6)
        min_score = float(cfg.get("color_fg_score_min", 1.35))
        quantile = float(np.clip(cfg.get("color_fg_quantile", 0.78), 0.5, 0.98))
        keep_min_pixels = max(int(cfg.get("color_fg_min_pixels", 24)), 1)
        keep_min_ratio = max(float(cfg.get("color_fg_min_ratio", 0.05)), 0.0)

        hue_delta = np.abs(hue - bg_h)
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        sat_delta = np.abs(sat - bg_s)
        val_delta = np.abs(val - bg_v)

        color_score = (
            (hue_delta / hue_tol) ** 2
            + (sat_delta / sat_tol) ** 2
            + (val_delta / val_tol) ** 2
        )
        valid_scores = color_score[valid_mask]
        if valid_scores.size == 0:
            return valid_mask
        bg_scores = color_score[bg_seed] if np.any(bg_seed) else valid_scores
        fg_presence_q = float(np.clip(cfg.get("color_fg_presence_quantile", 0.90), 0.5, 0.99))
        fg_presence_score = float(np.quantile(valid_scores, fg_presence_q))
        bg_presence_score = float(np.median(bg_scores)) if bg_scores.size > 0 else 0.0
        presence_margin = fg_presence_score - bg_presence_score
        reject_weak_presence = bool(cfg.get("color_fg_reject_weak_presence", True))
        presence_min_score = float(cfg.get("color_fg_presence_min_score", 1.60))
        presence_min_margin = float(cfg.get("color_fg_presence_min_margin", 0.45))
        if reject_weak_presence and (
            fg_presence_score < presence_min_score
            or presence_margin < presence_min_margin
        ):
            return np.zeros_like(valid_mask, dtype=bool)
        score_threshold = max(float(np.quantile(valid_scores, quantile)), min_score)
        filtered_mask = valid_mask & (color_score >= score_threshold)

        required_pixels = max(
            keep_min_pixels,
            int(round(int(np.count_nonzero(valid_mask)) * keep_min_ratio)),
        )
        if int(np.count_nonzero(filtered_mask)) >= required_pixels:
            dilate_iters = max(int(cfg.get("color_fg_dilate_iterations", 1)), 0)
            if dilate_iters > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                filtered_mask = cv2.dilate(
                    filtered_mask.astype(np.uint8), kernel, iterations=dilate_iters
                ).astype(bool) & valid_mask
            return filtered_mask
        if bool(cfg.get("color_fg_fallback_to_valid_mask", False)):
            return valid_mask
        return np.zeros_like(valid_mask, dtype=bool)

    @staticmethod
    def _apply_color_foreground_mask_external_ring(valid_mask: np.ndarray,
                                                   *,
                                                   color_image: np.ndarray | None,
                                                   roi_bbox_xyxy: tuple[int, int, int, int],
                                                   cfg: dict) -> np.ndarray:
        """Select foreground pixels using a local background ring outside the ROI."""
        failure_mask = (
            valid_mask
            if str(cfg.get("color_fg_external_ring_failure", "valid_mask")).lower()
            == "valid_mask"
            else np.zeros_like(valid_mask, dtype=bool)
        )
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask
        if color_image is None or color_image.size == 0:
            return failure_mask
        if not bool(cfg.get("use_color_foreground_mask", False)):
            return valid_mask

        x1, y1, x2, y2 = roi_bbox_xyxy
        h, w = color_image.shape[:2]
        x1 = max(0, min(w, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return failure_mask

        roi_bgr = color_image[y1:y2, x1:x2]
        if roi_bgr.shape[:2] != valid_mask.shape:
            return failure_mask

        expand_ratio = max(float(cfg.get("color_fg_bg_expand_ratio", 1.35)), 1.01)
        ex1, ey1, ex2, ey2 = FishPositionEstimator._expand_bbox_xyxy(
            (x1, y1, x2, y2),
            scale=expand_ratio,
            image_shape=(h, w),
        )
        if ex2 <= ex1 or ey2 <= ey1:
            return failure_mask

        expanded_bgr = color_image[ey1:ey2, ex1:ex2]
        if expanded_bgr.size == 0:
            return failure_mask

        local_x1 = x1 - ex1
        local_x2 = x2 - ex1
        local_y1 = y1 - ey1
        local_y2 = y2 - ey1

        bg_ring_mask = np.ones(expanded_bgr.shape[:2], dtype=bool)
        bg_ring_mask[local_y1:local_y2, local_x1:local_x2] = False
        bg_min_pixels = max(int(cfg.get("color_fg_bg_min_pixels", 64)), 4)
        if int(np.count_nonzero(bg_ring_mask)) < bg_min_pixels:
            return failure_mask

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        expanded_hsv = cv2.cvtColor(expanded_bgr, cv2.COLOR_BGR2HSV)

        roi_feat = FishPositionEstimator._circular_hsv_features(roi_hsv)
        bg_feat = FishPositionEstimator._circular_hsv_features(expanded_hsv)[bg_ring_mask]
        if bg_feat.ndim != 2 or bg_feat.shape[0] < bg_min_pixels:
            return failure_mask

        bg_feat = bg_feat.astype(np.float64, copy=False)
        mu_bg = np.mean(bg_feat, axis=0)
        if bg_feat.shape[0] <= 1:
            return failure_mask

        sigma_bg = np.cov(bg_feat, rowvar=False, bias=False)
        if sigma_bg.ndim != 2 or sigma_bg.shape != (3, 3):
            return failure_mask

        shrink = float(np.clip(cfg.get("color_fg_cov_shrinkage", 0.08), 0.0, 1.0))
        d = float(sigma_bg.shape[0])
        trace_term = float(np.trace(sigma_bg)) / max(d, 1.0)
        sigma_bg_tilde = (
            (1.0 - shrink) * sigma_bg
            + shrink * trace_term * np.eye(sigma_bg.shape[0], dtype=np.float64)
        )
        sigma_bg_tilde += 1e-6 * np.eye(sigma_bg.shape[0], dtype=np.float64)
        sigma_inv = np.linalg.pinv(sigma_bg_tilde)

        delta = roi_feat.astype(np.float64, copy=False) - mu_bg.reshape(1, 1, -1)
        color_score = np.einsum("...i,ij,...j->...", delta, sigma_inv, delta)
        valid_scores = color_score[valid_mask]
        if valid_scores.size == 0:
            return failure_mask

        tau = FishPositionEstimator._otsu_threshold_1d(
            valid_scores,
            bins=int(cfg.get("color_fg_otsu_bins", 128)),
        )
        if tau is None or not np.isfinite(tau):
            return failure_mask

        filtered_mask = valid_mask & np.isfinite(color_score) & (color_score >= float(tau))
        keep_min_pixels = max(int(cfg.get("color_fg_min_pixels", 24)), 1)
        keep_min_ratio = max(float(cfg.get("color_fg_min_ratio", 0.05)), 0.0)
        required_pixels = max(
            keep_min_pixels,
            int(round(int(np.count_nonzero(valid_mask)) * keep_min_ratio)),
        )
        if int(np.count_nonzero(filtered_mask)) >= required_pixels:
            dilate_iters = max(int(cfg.get("color_fg_dilate_iterations", 0)), 0)
            if dilate_iters > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                filtered_mask = cv2.dilate(
                    filtered_mask.astype(np.uint8),
                    kernel,
                    iterations=dilate_iters,
                ).astype(bool) & valid_mask
            return filtered_mask
        if bool(cfg.get("color_fg_fallback_to_valid_mask", False)):
            return valid_mask
        return np.zeros_like(valid_mask, dtype=bool)

    @staticmethod
    def _fuse_foreground_masks(valid_mask_initial: np.ndarray,
                               *,
                               depth_mask: np.ndarray,
                               color_mask: np.ndarray,
                               cfg: dict) -> np.ndarray:
        """Fuse color and depth cues, using color as the primary foreground cue."""
        if valid_mask_initial.size == 0 or not np.any(valid_mask_initial):
            return valid_mask_initial

        use_color = bool(cfg.get("use_color_foreground_mask", False))
        use_depth = bool(cfg.get("use_foreground_depth_mask", False))
        if not use_color:
            return depth_mask if use_depth else valid_mask_initial
        if not use_depth:
            return color_mask

        color_pixels = int(np.count_nonzero(color_mask))
        depth_pixels = int(np.count_nonzero(depth_mask))
        strict_intersection = bool(cfg.get("intersection_invalid_if_sparse", False))
        if color_pixels == 0:
            if bool(cfg.get("color_fg_allow_depth_fallback", False)) and depth_pixels > 0:
                return depth_mask
            return np.zeros_like(valid_mask_initial, dtype=bool)
        if depth_pixels == 0:
            return (
                np.zeros_like(valid_mask_initial, dtype=bool)
                if strict_intersection
                else color_mask
            )

        kernel = np.ones((3, 3), dtype=np.uint8)
        support_iters = max(int(cfg.get("color_depth_support_dilate_iterations", 1)), 0)
        depth_support = depth_mask
        if support_iters > 0:
            depth_support = cv2.dilate(
                depth_mask.astype(np.uint8), kernel, iterations=support_iters
            ).astype(bool)

        fused = color_mask & depth_support
        min_pixels = max(int(cfg.get("color_depth_consensus_min_pixels", 16)), 1)
        min_ratio = max(float(cfg.get("color_depth_consensus_min_ratio", 0.30)), 0.0)
        required_pixels = max(min_pixels, int(round(color_pixels * min_ratio)))
        if int(np.count_nonzero(fused)) >= required_pixels:
            return fused
        if strict_intersection:
            return np.zeros_like(valid_mask_initial, dtype=bool)
        return color_mask

    @staticmethod
    def _apply_foreground_disparity_mask_otsu(valid_mask: np.ndarray,
                                              disparity_map: np.ndarray,
                                              *,
                                              cfg: dict) -> np.ndarray:
        """Select near-foreground directly from disparity values using Otsu."""
        failure_mask = (
            valid_mask
            if str(cfg.get("foreground_disp_otsu_failure", "valid_mask")).lower()
            == "valid_mask"
            else np.zeros_like(valid_mask, dtype=bool)
        )
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask
        if not bool(cfg.get("use_foreground_depth_mask", False)):
            return valid_mask

        valid_disp = disparity_map[valid_mask]
        valid_disp = valid_disp[np.isfinite(valid_disp) & (valid_disp > 0.5)]
        if valid_disp.size == 0:
            return failure_mask

        min_pixels = max(int(cfg.get("foreground_min_pixels", 20)), 1)
        min_ratio = max(float(cfg.get("foreground_min_ratio", 0.08)), 0.0)
        required_pixels = max(
            min_pixels,
            int(round(valid_disp.size * min_ratio)),
        )
        if valid_disp.size < required_pixels:
            return failure_mask

        tau = FishPositionEstimator._otsu_threshold_1d(
            valid_disp,
            bins=int(cfg.get("foreground_disp_otsu_bins", 128)),
        )
        if tau is None or not np.isfinite(tau):
            return failure_mask

        fg_mask = valid_mask & np.isfinite(disparity_map) & (disparity_map >= float(tau))
        if int(np.count_nonzero(fg_mask)) >= required_pixels:
            return fg_mask
        return failure_mask

    @staticmethod
    def _apply_foreground_depth_mask(valid_mask: np.ndarray,
                                     depth_map: np.ndarray,
                                     *,
                                     depth_prior_m: float | None,
                                     cfg: dict) -> np.ndarray:
        """Prefer the closest substantial depth cluster inside the bbox ROI."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask

        if not bool(cfg.get("use_foreground_depth_mask", False)):
            return valid_mask

        valid_depth = depth_map[valid_mask]
        valid_depth = valid_depth[np.isfinite(valid_depth)]
        if valid_depth.size == 0:
            return valid_mask

        min_pixels = max(int(cfg.get("foreground_min_pixels", 20)), 1)
        min_ratio = max(float(cfg.get("foreground_min_ratio", 0.08)), 0.0)
        required_pixels = max(
            min_pixels,
            int(round(valid_depth.size * min_ratio)),
        )
        if valid_depth.size < required_pixels:
            return valid_mask

        bins = max(int(cfg.get("foreground_histogram_bins", 24)), 4)
        depth_min = float(np.min(valid_depth))
        depth_max = float(np.max(valid_depth))
        if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
            return valid_mask

        hist, edges = np.histogram(valid_depth, bins=bins, range=(depth_min, depth_max))
        if hist.size == 0 or int(np.max(hist)) <= 0:
            return valid_mask

        smooth = np.convolve(hist.astype(np.float64), np.array([1.0, 2.0, 1.0]), mode="same")
        peak_rel_height = float(cfg.get("foreground_peak_rel_height", 0.35))
        prefer_prior = bool(cfg.get("foreground_prefer_depth_prior", False))

        candidate_bins = []
        for idx, count in enumerate(hist):
            if count < required_pixels:
                continue
            left = smooth[idx - 1] if idx > 0 else -np.inf
            right = smooth[idx + 1] if idx + 1 < smooth.size else -np.inf
            if smooth[idx] >= left and smooth[idx] >= right:
                candidate_bins.append(idx)
        if not candidate_bins:
            candidate_bins = [int(np.argmax(smooth))]

        if prefer_prior and depth_prior_m is not None and np.isfinite(depth_prior_m):
            peak_idx = min(
                candidate_bins,
                key=lambda idx: (
                    abs(0.5 * (edges[idx] + edges[idx + 1]) - float(depth_prior_m)),
                    0.5 * (edges[idx] + edges[idx + 1]),
                ),
            )
        else:
            peak_idx = min(
                candidate_bins,
                key=lambda idx: (
                    0.5 * (edges[idx] + edges[idx + 1]),
                    -smooth[idx],
                ),
            )

        peak_value = max(float(smooth[peak_idx]), 1.0)
        threshold = peak_value * np.clip(peak_rel_height, 0.05, 0.95)
        lo_idx = peak_idx
        hi_idx = peak_idx
        while lo_idx > 0 and smooth[lo_idx - 1] >= threshold:
            lo_idx -= 1
        while hi_idx + 1 < smooth.size and smooth[hi_idx + 1] >= threshold:
            hi_idx += 1

        band_margin_m = max(float(cfg.get("foreground_band_margin_m", 0.06)), 0.0)
        depth_lo = max(float(edges[lo_idx]) - band_margin_m, 0.0)
        depth_hi = float(edges[hi_idx + 1]) + band_margin_m
        fg_mask = valid_mask & np.isfinite(depth_map) & (depth_map >= depth_lo) & (depth_map <= depth_hi)
        fg_mask = FishPositionEstimator._refine_foreground_from_near_seed(
            fg_mask,
            depth_map,
            valid_mask=valid_mask,
            band_margin_m=band_margin_m,
            depth_prior_m=depth_prior_m,
            cfg=cfg,
        )

        fg_pixels = int(np.count_nonzero(fg_mask))
        if fg_pixels >= required_pixels:
            return fg_mask

        fallback_quantile = float(cfg.get("foreground_fallback_near_quantile", 0.35))
        fallback_quantile = float(np.clip(fallback_quantile, 0.05, 0.95))
        depth_cut = float(np.quantile(valid_depth, fallback_quantile))
        fg_mask = valid_mask & np.isfinite(depth_map) & (depth_map <= (depth_cut + band_margin_m))
        if int(np.count_nonzero(fg_mask)) >= required_pixels:
            return fg_mask

        return valid_mask

    @staticmethod
    def _refine_foreground_from_near_seed(fg_mask: np.ndarray,
                                          depth_map: np.ndarray,
                                          *,
                                          valid_mask: np.ndarray,
                                          band_margin_m: float,
                                          depth_prior_m: float | None,
                                          cfg: dict) -> np.ndarray:
        """Shrink a broad foreground band toward the nearest compact target blob."""
        if fg_mask.size == 0 or not np.any(fg_mask):
            return fg_mask

        if not bool(cfg.get("foreground_use_near_seed_refine", True)):
            return fg_mask

        fg_depth = depth_map[fg_mask]
        fg_depth = fg_depth[np.isfinite(fg_depth)]
        if fg_depth.size == 0:
            return fg_mask

        seed_quantile = float(np.clip(cfg.get("foreground_seed_near_quantile", 0.12), 0.01, 0.5))
        seed_min_pixels = max(int(cfg.get("foreground_seed_min_pixels", 12)), 1)
        target_min_pixels = max(int(cfg.get("foreground_target_min_pixels", 24)), 1)
        target_min_ratio_roi = max(float(cfg.get("foreground_target_min_ratio_roi", 0.01)), 0.0)
        target_required_pixels = max(
            target_min_pixels,
            int(round(valid_mask.size * target_min_ratio_roi)),
        )

        seed_depth_cut = float(np.quantile(fg_depth, seed_quantile))
        seed_mask = fg_mask & np.isfinite(depth_map) & (depth_map <= seed_depth_cut)
        if int(np.count_nonzero(seed_mask)) < seed_min_pixels:
            return fg_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            seed_mask.astype(np.uint8, copy=False), connectivity=8
        )
        if num_labels <= 1:
            return fg_mask

        best_label = None
        best_score = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < seed_min_pixels:
                continue
            component = labels == label
            comp_depth = depth_map[component]
            comp_depth = comp_depth[np.isfinite(comp_depth)]
            if comp_depth.size == 0:
                continue
            comp_depth_median = float(np.median(comp_depth))
            if depth_prior_m is not None and np.isfinite(depth_prior_m):
                depth_score = abs(comp_depth_median - float(depth_prior_m))
            else:
                depth_score = comp_depth_median
            score = (depth_score, -area)
            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return fg_mask

        seed_component = labels == best_label
        seed_depth = depth_map[seed_component]
        seed_depth = seed_depth[np.isfinite(seed_depth)]
        if seed_depth.size == 0:
            return fg_mask

        seed_depth_median = float(np.median(seed_depth))
        refine_margin_m = max(
            float(cfg.get("foreground_seed_band_margin_m", min(max(band_margin_m * 0.5, 0.02), 0.08))),
            1e-3,
        )
        refined = fg_mask & np.isfinite(depth_map) & (
            np.abs(depth_map - seed_depth_median) <= refine_margin_m
        )
        if not np.any(refined):
            return fg_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            refined.astype(np.uint8, copy=False), connectivity=8
        )
        if num_labels <= 1:
            return fg_mask if int(np.count_nonzero(refined)) < target_required_pixels else refined

        kernel = np.ones((3, 3), dtype=np.uint8)
        dilate_iters = max(int(cfg.get("foreground_seed_dilate_iterations", 2)), 1)
        seed_support = cv2.dilate(seed_component.astype(np.uint8), kernel, iterations=dilate_iters).astype(bool)

        best_label = None
        best_overlap = -1
        best_area = -1
        for label in range(1, num_labels):
            component = labels == label
            overlap = int(np.count_nonzero(component & seed_support))
            area = int(stats[label, cv2.CC_STAT_AREA])
            if overlap > best_overlap or (overlap == best_overlap and area > best_area):
                best_overlap = overlap
                best_area = area
                best_label = label

        if best_label is None:
            return fg_mask

        target_mask = labels == best_label
        if int(np.count_nonzero(target_mask)) >= target_required_pixels:
            return target_mask
        if int(np.count_nonzero(seed_component)) >= target_min_pixels:
            return seed_component
        return fg_mask

    @staticmethod
    def _apply_center_prior_mask(valid_mask: np.ndarray,
                                 center_prior_uv: tuple[float, float],
                                 *,
                                 radius_px: float,
                                 min_pixels: int,
                                 min_ratio: float) -> np.ndarray:
        """Prefer valid pixels near the previous-frame image location when possible."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return valid_mask

        h, w = valid_mask.shape
        cx, cy = center_prior_uv
        yy, xx = np.ogrid[:h, :w]
        radius_sq = max(float(radius_px), 1.0) ** 2
        center_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_sq
        prior_mask = valid_mask & center_mask
        prior_pixels = int(np.count_nonzero(prior_mask))
        required_pixels = max(int(min_pixels), int(round(valid_mask.size * max(min_ratio, 0.0))))
        if prior_pixels >= max(required_pixels, 1):
            return prior_mask
        return valid_mask

    @staticmethod
    def _build_temporal_prior_support_mask(valid_mask: np.ndarray,
                                           disparity_map: np.ndarray,
                                           *,
                                           center_prior_uv: tuple[float, float] | None,
                                           depth_prior_m: float | None,
                                           fx: float,
                                           baseline_m: float,
                                           cfg: dict,
                                           dt_s: float | None = None,
                                           reference_dt_s: float = 0.1) -> np.ndarray:
        """Build a soft temporal support region in (u, v, disparity) space."""
        if valid_mask.size == 0 or not np.any(valid_mask):
            return np.zeros_like(valid_mask, dtype=bool)

        support = valid_mask.copy()
        h, w = valid_mask.shape
        yy, xx = np.ogrid[:h, :w]
        m2 = np.zeros(valid_mask.shape, dtype=np.float64)
        dof = 0
        reference_dt_s = max(float(reference_dt_s), 1e-6)
        if dt_s is not None and np.isfinite(dt_s) and dt_s > 0:
            temporal_scale = max(float(dt_s) / reference_dt_s, 1e-6)
        else:
            temporal_scale = 1.0

        if center_prior_uv is not None:
            sigma_uv = max(float(cfg.get("center_prior_sigma_px", 24.0)), 1e-6)
            sigma_uv *= temporal_scale
            cx, cy = center_prior_uv
            m2 += ((xx - cx) / sigma_uv) ** 2 + ((yy - cy) / sigma_uv) ** 2
            dof += 2

        if depth_prior_m is not None and np.isfinite(depth_prior_m) and depth_prior_m > 1e-6:
            disp_prior = float(fx * baseline_m / depth_prior_m)
            sigma_z = max(float(cfg.get("depth_prior_sigma_m", 0.25)), 1e-6)
            sigma_z *= temporal_scale
            sigma_disp = abs(float(fx * baseline_m) * sigma_z / max(depth_prior_m ** 2, 1e-6))
            sigma_disp = max(
                sigma_disp,
                float(cfg.get("temporal_prior_min_sigma_disp_px", 0.75)),
            )
            disp_ok = np.isfinite(disparity_map)
            support &= disp_ok
            if np.any(disp_ok):
                dd = (disparity_map[disp_ok] - disp_prior) / sigma_disp
                m2[disp_ok] += dd * dd
            dof += 1

        if dof <= 0:
            return np.zeros_like(valid_mask, dtype=bool)

        default_threshold = {
            1: 6.63,
            2: 9.21,
            3: 11.34,
        }.get(dof, 11.34)
        chi2_threshold = float(cfg.get("temporal_prior_chi2_threshold", default_threshold))
        return support & np.isfinite(m2) & (m2 <= chi2_threshold)

    @staticmethod
    def _recover_mask_with_temporal_prior(single_frame_mask: np.ndarray,
                                          *,
                                          weak_mask: np.ndarray,
                                          prior_support_mask: np.ndarray,
                                          bbox_area_px: int,
                                          cfg: dict) -> np.ndarray:
        """Use temporal support to recover a weak single-frame mask."""
        if single_frame_mask.size == 0:
            return single_frame_mask

        strong_ratio = max(float(cfg.get("temporal_recover_strong_ratio_bbox", 0.08)), 0.0)
        strong_min = max(int(cfg.get("temporal_recover_strong_min_pixels", 20)), 1)
        recover_ratio = max(float(cfg.get("temporal_recover_min_ratio_bbox", 0.02)), 0.0)
        recover_min = max(int(cfg.get("temporal_recover_min_pixels", 8)), 1)

        strong_required = max(strong_min, int(round(max(bbox_area_px, 1) * strong_ratio)))
        recover_required = max(recover_min, int(round(max(bbox_area_px, 1) * recover_ratio)))

        strong_pixels = int(np.count_nonzero(single_frame_mask))
        if strong_pixels >= strong_required:
            return single_frame_mask
        if prior_support_mask.size == 0 or not np.any(prior_support_mask):
            return single_frame_mask

        recovered = weak_mask & prior_support_mask
        if int(np.count_nonzero(recovered)) >= recover_required:
            return single_frame_mask | recovered
        return single_frame_mask

    @staticmethod
    def _project_prior_center_to_roi(center_prior_uv: tuple[float, float] | None,
                                     rx1: int,
                                     ry1: int,
                                     rx2: int,
                                     ry2: int) -> tuple[float, float] | None:
        """Map a global image-space prior center into ROI-local coordinates."""
        if center_prior_uv is None:
            return None
        u, v = center_prior_uv
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        if u < rx1 or u >= rx2 or v < ry1 or v >= ry2:
            return None
        return (float(u - rx1), float(v - ry1))

    def _lookup_roi_prior(self, det_bbox) -> dict:
        """Find previous-frame image/depth priors for the current detection."""
        best_track = self._lookup_roi_prior_track(det_bbox)
        if best_track is None:
            return {"depth_m": None, "center_uv": None}
        return self._track_roi_prior(best_track)

    def _lookup_roi_prior_track(self, det_bbox):
        single_track = getattr(self.tracker, "track", None)
        if single_track is not None:
            return single_track

        tracks = getattr(self.tracker, "tracks", None)
        if not tracks:
            return None

        best_track = None
        best_iou = 0.0
        for track in tracks:
            iou = FishTracker._box_iou(track.bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None or best_iou <= 0.0:
            return None
        return best_track

    @staticmethod
    def _track_roi_prior(track) -> dict:
        center_uv = None
        bbox = getattr(track, "bbox", None)
        if bbox is not None:
            center_uv = (
                0.5 * float(bbox[0] + bbox[2]),
                0.5 * float(bbox[1] + bbox[3]),
            )
        if center_uv is None:
            depth_stats = getattr(track, "depth_stats", None)
            if depth_stats is not None:
                center_uv = getattr(depth_stats, "center_uv", None)

        pos_3d = getattr(track, "pos_3d", None)
        meta = {
            "center_uv": center_uv,
            "track_id": int(getattr(track, "id", -1)),
            "depth_valid": bool(getattr(track, "depth_valid", False)),
            "depth_rejected": bool(getattr(track, "depth_rejected", False)),
            "depth_confidence": float(getattr(track, "depth_confidence", 0.0) or 0.0),
            "depth_filter_mode": str(getattr(track, "depth_filter_mode", "") or ""),
            "hits": int(getattr(track, "hits", 0) or 0),
        }
        if pos_3d is None:
            return {"depth_m": None, **meta}

        # Do not recycle an already rejected / predicted-only depth as a new
        # ROI prior. Otherwise one bad far-depth solve can get latched by the
        # small-target protection logic and keep being held for many frames.
        depth_valid = bool(meta["depth_valid"])
        depth_rejected = bool(meta["depth_rejected"])
        depth_confidence = float(meta["depth_confidence"])
        depth_filter_mode = str(meta["depth_filter_mode"]).lower()
        if (
            (not depth_valid)
            or depth_rejected
            or depth_confidence < 0.15
            or ("predict_only" in depth_filter_mode)
            or ("depth_reject" in depth_filter_mode)
        ):
            return {"depth_m": None, **meta}

        z = float(np.asarray(pos_3d, dtype=np.float32)[2])
        if not np.isfinite(z):
            z = None
        return {"depth_m": z, **meta}

    def _depth_to_3d(self, center_uv, depth_m: float):
        if depth_m is None or not np.isfinite(depth_m):
            return None
        u, v = center_uv
        cam = self._cfg["camera"]
        fx = cam["fx"]
        fy = cam.get("fy", fx)
        cx = cam["cx"]
        cy = cam["cy"]

        Z = float(depth_m)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return np.array([X, Y, Z], dtype=np.float32)

    def _pixel_to_3d(self, center_uv, disparity_map: np.ndarray):
        """
        Convert a pixel coordinate + disparity map → 3D camera coordinates.

        Parameters
        ----------
        center_uv : tuple (u, v)
            Pixel coordinate (column, row) in the left image.
        disparity_map : np.ndarray (H, W)
            Dense disparity map at the original image resolution.

        Returns
        -------
        np.ndarray shape (3,)  [x, y, z] in metres, or None on failure.
        """
        u, v = center_uv
        u_i, v_i = int(round(u)), int(round(v))
        H, W = disparity_map.shape

        # Clamp to image bounds
        u_i = max(0, min(W - 1, u_i))
        v_i = max(0, min(H - 1, v_i))

        # Extract a robust disparity from a small patch
        r = 6  # half-window size
        patch = disparity_map[
            max(0, v_i - r): min(H, v_i + r + 1),
            max(0, u_i - r): min(W, u_i + r + 1),
        ]
        valid = patch[patch > 0.5]  # ignore near-zero disparity
        if len(valid) == 0:
            return None
        disp_val = float(np.median(valid))

        cam = self._cfg["camera"]
        Z = cam["fx"] * cam["baseline_m"] / disp_val
        return self._depth_to_3d(center_uv, Z)

    # ── Helpers ────────────────────────────────────────────────────────

    def _pack_results(self, tracks: list[FishTrack]) -> list[dict]:
        tracks = self._select_output_tracks(tracks)
        out = []
        tracker_state = self.tracker_state
        for t in tracks:
            position = self._resolve_track_output_position(t)
            out.append({
                "id": t.id,
                "bbox": getattr(t, "output_bbox", t.bbox),
                "position": position.tolist(),
                "confidence": t.confidence,
                "raw_depth": t.raw_depth,
                "raw_center_uv": list(getattr(t, "raw_center_uv", (np.nan, np.nan))),
                "filtered_center_uv": list(getattr(t, "filtered_center_uv", (np.nan, np.nan))),
                "depth_confidence": t.depth_confidence,
                "depth_valid": t.depth_valid,
                "depth_rejected": t.depth_rejected,
                "z_dot": t.z_dot,
                "depth_gain": getattr(t, "depth_gain", 0.0),
                "depth_quality": getattr(t, "depth_quality", 0.0),
                "depth_filter_mode": getattr(t, "depth_filter_mode", "unknown"),
                "depth_r_t": getattr(t, "depth_r_t", 0.0),
                "depth_R_t": getattr(t, "depth_R_t", 0.0),
                "depth_nis": getattr(t, "depth_nis", None),
                "depth_innovation": getattr(t, "depth_innovation", None),
                "depth_S_t": getattr(t, "depth_S_t", None),
                "depth_P_zz_pred": getattr(t, "depth_P_zz_pred", None),
                "depth_dt_s": getattr(t, "depth_dt_s", 0.0),
                "core_px": int(getattr(getattr(t, "depth_stats", None), "core_px", 0) or 0),
                "z_iqr": float(getattr(getattr(t, "depth_stats", None), "z_iqr", np.nan)),
                "sep_score": float(getattr(getattr(t, "depth_stats", None), "sep_score", np.nan)),
                "source": getattr(t, "source", "yolo"),
                "track_state": tracker_state,
            })
        return out

    def _resolve_track_output_position(self, track: FishTrack) -> np.ndarray:
        pos = np.asarray(track.pos_3d, dtype=np.float32).copy()
        if pos.shape[0] < 3:
            return pos
        if np.all(np.isfinite(pos)):
            return pos

        z = float(pos[2]) if np.isfinite(pos[2]) else None
        if z is None:
            return pos

        center_uv = None
        bbox = getattr(track, "output_bbox", None) or getattr(track, "bbox", None)
        if bbox is not None:
            center_uv = self._bbox_center_xyxy(bbox)
        if center_uv is None:
            depth_stats = getattr(track, "depth_stats", None)
            if depth_stats is not None:
                center_uv = getattr(depth_stats, "center_uv", None)
        if center_uv is None:
            return pos

        recovered = self._depth_to_3d(center_uv, z)
        if recovered is None:
            return pos

        recovered = np.asarray(recovered, dtype=np.float32)
        if recovered.shape[0] >= 3 and np.all(np.isfinite(recovered)):
            track.pos_3d = recovered.copy()
            return recovered
        return pos

    def _select_output_tracks(self, tracks: list[FishTrack]) -> list[FishTrack]:
        if len(tracks) <= 1:
            return tracks
        tracker_cfg = self._cfg.get("tracker", {})
        if not bool(tracker_cfg.get("single_target_output_only", False)):
            return tracks
        best = min(tracks, key=self._primary_output_track_key)
        return [best]

    @staticmethod
    def _primary_output_track_key(track: FishTrack):
        z = float(np.asarray(track.pos_3d)[2]) if track.pos_3d is not None else float("inf")
        if not np.isfinite(z):
            z = float("inf")
        source = str(getattr(track, "source", "") or "")
        source_rank = {
            "yolo+corrector": 0,
            "yolo": 1,
            "yolo-fast": 2,
            "csrt+yolo": 2,
            "roi-redetect": 3,
            "edge-redetect": 4,
            "csrt": 5,
            "bytetrack": 5,
        }.get(source, 6)
        return (
            int(getattr(track, "time_since_update", 9999)),
            source_rank,
            -int(getattr(track, "hits", 0)),
            -float(getattr(track, "confidence", 0.0)),
            -float(getattr(track, "depth_confidence", 0.0)),
            int(not bool(getattr(track, "depth_valid", False))),
            abs(z),
            int(getattr(track, "id", 0)),
        )

    @property
    def camera_params(self) -> dict:
        """Return a copy of the camera intrinsics being used."""
        return dict(self._cfg["camera"])

    @property
    def tracker_state(self) -> str:
        state = getattr(self.tracker, "state", None)
        if state is not None:
            return str(state)
        tracks = getattr(self.tracker, "tracks", None)
        if tracks:
            return "TRACKING_SIMPLE"
        return "NO_TARGET"

    def estimate_frame(self, stereo_frame) -> list[dict]:
        """Convenience: accept a ``StereoFrame`` from ``stereo_capture.py``."""
        frame_ts_s = 0.5 * (
            float(stereo_frame.ts_left) + float(stereo_frame.ts_right)
        )
        return self.estimate(
            stereo_frame.left,
            stereo_frame.right,
            frame_ts_s=frame_ts_s,
        )

    def _measure_dt(self, frame_ts_s: float | None = None) -> float:
        cfg = self._cfg.get("temporal_depth", {})
        fallback = float(cfg.get("fallback_dt_s", cfg.get("dt_s", 0.1)))
        min_dt = float(cfg.get("min_dt_s", 0.005))
        max_dt = float(cfg.get("max_dt_s", 2.0))
        if frame_ts_s is not None and np.isfinite(frame_ts_s):
            now = float(frame_ts_s)
        else:
            now = time.monotonic()
        if self._last_timestamp is None:
            self._last_timestamp = now
            return fallback
        dt = now - self._last_timestamp
        self._last_timestamp = now
        if not np.isfinite(dt) or dt <= 0:
            return fallback
        return float(np.clip(dt, min_dt, max_dt))

    def _rectify_pair(self, left_img: np.ndarray,
                      right_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_rect, right_rect = self.rectifier.rectify(left_img, right_img)
        rectified_camera = self.rectifier.camera_params
        if rectified_camera is not None:
            self._cfg["camera"] = rectified_camera
            tracker_td = getattr(self.tracker, "temporal_depth_cfg", None)
            if isinstance(tracker_td, dict):
                tracker_td["camera"] = dict(rectified_camera)
            tracks = getattr(self.tracker, "tracks", None)
            if tracks:
                for track in tracks:
                    depth_filter = getattr(track, "depth_filter", None)
                    if hasattr(depth_filter, "camera"):
                        depth_filter.camera = dict(rectified_camera)
        return left_rect, right_rect


# ===================================================================
#  Convenience: build from YAML path
# ===================================================================

def from_config_yaml(path: str,
                     *,
                     pipeline_mode: str | None = None,
                     temporal_filter_enabled: bool | None = None,
                     yolo_model: str | None = None) -> FishPositionEstimator:
    """Load configuration from a YAML file and return a ready estimator."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    if yolo_model:
        cfg.setdefault("models", {})["yolo_path"] = str(yolo_model)
    cfg = apply_pipeline_mode(cfg, pipeline_mode)
    cfg = apply_temporal_filter_override(cfg, temporal_filter_enabled)
    return FishPositionEstimator(cfg)
