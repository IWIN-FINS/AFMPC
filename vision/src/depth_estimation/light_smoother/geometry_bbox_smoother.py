from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
from torch import nn


def bbox_xyxy_to_cxcywh(bbox_xyxy: list[float] | np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return np.asarray(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 1.0), max(y2 - y1, 1.0)],
        dtype=np.float32,
    )


def bbox_cxcywh_to_xyxy(bbox_cxcywh: list[float] | np.ndarray) -> list[float]:
    cx, cy, w, h = [float(v) for v in bbox_cxcywh]
    hw = max(w, 1.0) * 0.5
    hh = max(h, 1.0) * 0.5
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def apply_bbox_delta(coarse_bbox_xyxy: list[float], delta: list[float] | np.ndarray) -> list[float]:
    ccx, ccy, cw, ch = bbox_xyxy_to_cxcywh(coarse_bbox_xyxy)
    dx, dy, dw, dh = [float(v) for v in delta]
    cx = ccx + dx * max(cw, 1.0)
    cy = ccy + dy * max(ch, 1.0)
    w = max(cw * float(np.exp(dw)), 1.0)
    h = max(ch * float(np.exp(dh)), 1.0)
    return bbox_cxcywh_to_xyxy([cx, cy, w, h])


def sanitize_bbox_xyxy(bbox_xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return [x1, y1, x2, y2]


def compute_prior_bbox(
    prev_bbox_xyxy: list[float] | None,
    prev_prev_bbox_xyxy: list[float] | None,
) -> list[float] | None:
    if prev_bbox_xyxy is None:
        return None
    prev_cs = bbox_xyxy_to_cxcywh(prev_bbox_xyxy)
    if prev_prev_bbox_xyxy is None:
        return bbox_cxcywh_to_xyxy(prev_cs)
    prev_prev_cs = bbox_xyxy_to_cxcywh(prev_prev_bbox_xyxy)
    velocity = prev_cs - prev_prev_cs
    prior_cs = prev_cs + velocity
    prior_cs[2:4] = np.maximum(prior_cs[2:4], 1.0)
    return bbox_cxcywh_to_xyxy(prior_cs)


def build_step_feature(
    coarse_bbox_xyxy: list[float],
    *,
    frame_w: int,
    frame_h: int,
    confidence: float,
    source_flag: float,
    prev_coarse_bbox_xyxy: list[float] | None,
    prev_smooth_bbox_xyxy: list[float] | None,
    prev_prev_smooth_bbox_xyxy: list[float] | None,
) -> np.ndarray:
    coarse_cs = bbox_xyxy_to_cxcywh(coarse_bbox_xyxy)
    cx, cy, w, h = [float(v) for v in coarse_cs]
    dist_left = cx - 0.5 * w
    dist_right = float(frame_w) - (cx + 0.5 * w)
    dist_top = cy - 0.5 * h
    dist_bottom = float(frame_h) - (cy + 0.5 * h)

    if prev_coarse_bbox_xyxy is None:
        coarse_delta = np.zeros(4, dtype=np.float32)
    else:
        prev_coarse_cs = bbox_xyxy_to_cxcywh(prev_coarse_bbox_xyxy)
        coarse_delta = np.asarray(
            [
                (cx - float(prev_coarse_cs[0])) / max(float(prev_coarse_cs[2]), 1.0),
                (cy - float(prev_coarse_cs[1])) / max(float(prev_coarse_cs[3]), 1.0),
                np.log(max(w, 1.0) / max(float(prev_coarse_cs[2]), 1.0)),
                np.log(max(h, 1.0) / max(float(prev_coarse_cs[3]), 1.0)),
            ],
            dtype=np.float32,
        )

    prior_bbox = compute_prior_bbox(prev_smooth_bbox_xyxy, prev_prev_smooth_bbox_xyxy)
    if prior_bbox is None:
        prior_residual = np.zeros(4, dtype=np.float32)
        smooth_vel = np.zeros(4, dtype=np.float32)
    else:
        prior_cs = bbox_xyxy_to_cxcywh(prior_bbox)
        prior_residual = np.asarray(
            [
                (cx - float(prior_cs[0])) / max(float(prior_cs[2]), 1.0),
                (cy - float(prior_cs[1])) / max(float(prior_cs[3]), 1.0),
                np.log(max(w, 1.0) / max(float(prior_cs[2]), 1.0)),
                np.log(max(h, 1.0) / max(float(prior_cs[3]), 1.0)),
            ],
            dtype=np.float32,
        )
        if prev_smooth_bbox_xyxy is not None and prev_prev_smooth_bbox_xyxy is not None:
            prev_smooth_cs = bbox_xyxy_to_cxcywh(prev_smooth_bbox_xyxy)
            prev_prev_smooth_cs = bbox_xyxy_to_cxcywh(prev_prev_smooth_bbox_xyxy)
            smooth_vel = np.asarray(
                [
                    (float(prev_smooth_cs[0]) - float(prev_prev_smooth_cs[0])) / max(float(prev_prev_smooth_cs[2]), 1.0),
                    (float(prev_smooth_cs[1]) - float(prev_prev_smooth_cs[1])) / max(float(prev_prev_smooth_cs[3]), 1.0),
                    np.log(max(float(prev_smooth_cs[2]), 1.0) / max(float(prev_prev_smooth_cs[2]), 1.0)),
                    np.log(max(float(prev_smooth_cs[3]), 1.0) / max(float(prev_prev_smooth_cs[3]), 1.0)),
                ],
                dtype=np.float32,
            )
        else:
            smooth_vel = np.zeros(4, dtype=np.float32)

    return np.asarray(
        [
            cx / max(float(frame_w), 1.0),
            cy / max(float(frame_h), 1.0),
            w / max(float(frame_w), 1.0),
            h / max(float(frame_h), 1.0),
            float(confidence),
            dist_left / max(float(frame_w), 1.0),
            dist_right / max(float(frame_w), 1.0),
            dist_top / max(float(frame_h), 1.0),
            dist_bottom / max(float(frame_h), 1.0),
            *coarse_delta.tolist(),
            *prior_residual.tolist(),
            *smooth_vel.tolist(),
            float(source_flag),
        ],
        dtype=np.float32,
    )


@dataclass
class GeometryBBoxSmootherConfig:
    enabled: bool = False
    device: str = "cuda"
    checkpoint_path: str | None = None
    hidden_dim: int = 64
    feature_dim: int = 22
    max_idle_frames: int = 2
    min_gain: float = 0.05
    max_gain: float = 0.98
    max_delta_log_scale: float = 0.40
    max_center_shift_ratio: float = 0.35
    max_size_change_ratio: float = 0.40

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "GeometryBBoxSmootherConfig":
        cfg = dict(cfg or {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class GeometryBBoxSmootherState:
    coarse_bbox_xyxy: list[float]
    smooth_bbox_xyxy: list[float]
    confidence: float
    source_flag: float


class GeometryBBoxSmoother(nn.Module):
    def __init__(self, feature_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def step(self, feature: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(feature)
        hidden = self.temporal(x, hidden)
        gain_logits = self.gain_head(hidden)
        delta_raw = self.delta_head(hidden)
        return hidden, gain_logits, delta_raw


class GeometryBBoxSmootherRuntime:
    def __init__(self, config: GeometryBBoxSmootherConfig):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = GeometryBBoxSmoother(
            feature_dim=int(config.feature_dim),
            hidden_dim=int(config.hidden_dim),
        ).to(self.device).eval()
        if config.checkpoint_path:
            state = torch.load(config.checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
        self.hidden: torch.Tensor | None = None
        self.history: Deque[GeometryBBoxSmootherState] = deque(maxlen=3)
        self.idle_frames = 0

    def reset(self) -> None:
        self.hidden = None
        self.history.clear()
        self.idle_frames = 0

    def skip(self) -> None:
        self.idle_frames += 1
        if self.idle_frames > max(int(self.cfg.max_idle_frames), 0):
            self.reset()

    def update(
        self,
        bbox_xyxy: list[float],
        *,
        frame_w: int,
        frame_h: int,
        confidence: float,
        source_flag: float,
    ) -> list[float]:
        prev_coarse = self.history[-1].coarse_bbox_xyxy if self.history else None
        prev_smooth = self.history[-1].smooth_bbox_xyxy if self.history else None
        prev_prev_smooth = self.history[-2].smooth_bbox_xyxy if len(self.history) >= 2 else None

        feature = build_step_feature(
            bbox_xyxy,
            frame_w=frame_w,
            frame_h=frame_h,
            confidence=float(confidence),
            source_flag=float(source_flag),
            prev_coarse_bbox_xyxy=prev_coarse,
            prev_smooth_bbox_xyxy=prev_smooth,
            prev_prev_smooth_bbox_xyxy=prev_prev_smooth,
        )
        feature_t = torch.from_numpy(feature).to(self.device).unsqueeze(0)
        if self.hidden is None:
            self.hidden = torch.zeros((1, int(self.cfg.hidden_dim)), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            hidden, gain_logits, delta_raw = self.model.step(feature_t, self.hidden)
        self.hidden = hidden

        gain = torch.sigmoid(gain_logits)[0].detach().cpu().numpy().astype(np.float32)
        gain = np.clip(gain, float(self.cfg.min_gain), float(self.cfg.max_gain))
        delta = torch.tanh(delta_raw)[0].detach().cpu().numpy().astype(np.float32)
        delta[2:4] *= float(self.cfg.max_delta_log_scale)

        corrected_bbox = apply_bbox_delta(bbox_xyxy, delta.tolist())
        prior_bbox = compute_prior_bbox(prev_smooth, prev_prev_smooth)
        if prior_bbox is None:
            final_cs = bbox_xyxy_to_cxcywh(corrected_bbox)
        else:
            corrected_cs = bbox_xyxy_to_cxcywh(corrected_bbox)
            prior_cs = bbox_xyxy_to_cxcywh(prior_bbox)
            final_cs = prior_cs + gain * (corrected_cs - prior_cs)
            final_cs[2:4] = np.maximum(final_cs[2:4], 1.0)

        final_bbox = sanitize_bbox_xyxy(
            bbox_cxcywh_to_xyxy(final_cs),
            frame_w,
            frame_h,
        )
        self.history.append(
            GeometryBBoxSmootherState(
                coarse_bbox_xyxy=[float(v) for v in bbox_xyxy],
                smooth_bbox_xyxy=[float(v) for v in final_bbox],
                confidence=float(confidence),
                source_flag=float(source_flag),
            )
        )
        self.idle_frames = 0
        return final_bbox
