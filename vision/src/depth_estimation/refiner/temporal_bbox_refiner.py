from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import cv2
import numpy as np
import torch
from torch import nn


@dataclass
class TemporalBBoxRefinerConfig:
    enabled: bool = False
    device: str = "cuda"
    checkpoint_path: str | None = None
    buffer_size: int = 5
    roi_size: int = 128
    crop_expand_ratio: float = 1.8
    apply_on_tracker_only: bool = False
    post_smooth_enabled: bool = True
    post_smooth_center_alpha: float = 0.65
    post_smooth_size_alpha: float = 0.55
    visible_accept_threshold: float = 0.35
    truncation_threshold: float = 0.50
    max_center_shift_ratio: float = 0.25
    max_size_change_ratio: float = 0.35

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "TemporalBBoxRefinerConfig":
        cfg = dict(cfg or {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TemporalBBoxRefinerPrediction:
    gain_score: float
    refine_alpha: float
    visible_score: float
    truncation_score: float
    bbox_delta: tuple[float, float, float, float]
    refined_bbox: list[float]
    accepted: bool


@dataclass
class _FrameState:
    frame: np.ndarray
    bbox_xyxy: list[float]
    confidence: float
    source_flag: float


class _TinyRoiEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True),
            nn.Conv2d(40, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)


class TemporalBBoxRefinerV1(nn.Module):
    """Unified temporal bbox refinement model.

    Inputs
    ------
    roi_seq: [B, K, 3, H, W]
    geo_seq: [B, K, G]

    Outputs
    -------
    dict with:
      - gain_value: [B, 1]
      - bbox_delta: [B, 4]
      - visible_logit: [B, 1]
      - truncation_logit: [B, 1]
    """

    def __init__(
        self,
        *,
        roi_feature_dim: int = 128,
        geo_dim: int = 16,
        geo_hidden_dim: int = 32,
        temporal_hidden_dim: int = 128,
    ):
        super().__init__()
        self.roi_encoder = _TinyRoiEncoder(out_dim=roi_feature_dim)
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(geo_hidden_dim, geo_hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.GRU(
            input_size=roi_feature_dim + geo_hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.gain_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.visible_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.truncation_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, roi_seq: torch.Tensor, geo_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = roi_seq.shape
        roi_flat = roi_seq.reshape(batch * steps, channels, height, width)
        roi_feat = self.roi_encoder(roi_flat).reshape(batch, steps, -1)
        geo_feat = self.geo_encoder(geo_seq.reshape(batch * steps, -1)).reshape(batch, steps, -1)
        fused = torch.cat([roi_feat, geo_feat], dim=-1)
        _, hidden = self.temporal(fused)
        last = hidden[-1]
        return {
            "gain_value": self.gain_head(last),
            "bbox_delta": self.bbox_head(last),
            "visible_logit": self.visible_head(last),
            "truncation_logit": self.truncation_head(last),
        }


class TemporalBBoxRefinerRuntime:
    """Single-target runtime buffer and inference wrapper.

    This class is intentionally narrow:
      - one active target
      - causal buffer only
      - safe acceptance gate
    """

    def __init__(self, config: TemporalBBoxRefinerConfig, model: TemporalBBoxRefinerV1 | None = None):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model or TemporalBBoxRefinerV1()
        self.model.to(self.device).eval()
        self.buffer: Deque[_FrameState] = deque(maxlen=max(int(config.buffer_size), 1))
        if config.checkpoint_path:
            state = torch.load(config.checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)

    def reset(self) -> None:
        self.buffer.clear()

    def push(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: list[float],
        *,
        confidence: float,
        source: str,
    ) -> None:
        source_flag = 1.0 if str(source).lower().startswith("yolo") else 0.0
        self.buffer.append(
            _FrameState(
                frame=frame_bgr,
                bbox_xyxy=[float(v) for v in bbox_xyxy],
                confidence=float(confidence),
                source_flag=source_flag,
            )
        )

    def ready(self) -> bool:
        return len(self.buffer) >= self.cfg.buffer_size

    def refine_current(self) -> TemporalBBoxRefinerPrediction | None:
        if not self.ready():
            return None

        roi_seq = []
        geo_seq = []
        prev_bbox = None
        frame_h, frame_w = self.buffer[-1].frame.shape[:2]

        for state in self.buffer:
            roi = _extract_roi(
                state.frame,
                state.bbox_xyxy,
                roi_size=int(self.cfg.roi_size),
                expand_ratio=float(self.cfg.crop_expand_ratio),
            )
            roi_seq.append(roi)
            geo_seq.append(
                _bbox_geometry_features(
                    state.bbox_xyxy,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    confidence=state.confidence,
                    source_flag=state.source_flag,
                    prev_bbox=prev_bbox,
                )
            )
            prev_bbox = state.bbox_xyxy

        roi_tensor = torch.from_numpy(np.stack(roi_seq)).permute(0, 3, 1, 2).float() / 255.0
        geo_tensor = torch.from_numpy(np.stack(geo_seq)).float()
        roi_tensor = roi_tensor.unsqueeze(0).to(self.device)
        geo_tensor = geo_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(roi_tensor, geo_tensor)

        gain_score = outputs["gain_value"][0, 0].item()
        refine_alpha = torch.sigmoid(outputs["gain_value"])[0, 0].item()
        visible_score = torch.sigmoid(outputs["visible_logit"])[0, 0].item()
        truncation_score = torch.sigmoid(outputs["truncation_logit"])[0, 0].item()
        dx, dy, dw, dh = outputs["bbox_delta"][0].detach().cpu().tolist()
        coarse_bbox = self.buffer[-1].bbox_xyxy
        refined_bbox_raw = _apply_bbox_delta(coarse_bbox, (dx, dy, dw, dh))
        refined_bbox = _blend_bbox(coarse_bbox, refined_bbox_raw, refine_alpha)
        accepted = self._accept_prediction(
            coarse_bbox,
            refined_bbox,
            visible_score=visible_score,
        )
        return TemporalBBoxRefinerPrediction(
            gain_score=gain_score,
            refine_alpha=refine_alpha,
            visible_score=visible_score,
            truncation_score=truncation_score,
            bbox_delta=(float(dx), float(dy), float(dw), float(dh)),
            refined_bbox=refined_bbox,
            accepted=accepted,
        )

    def _accept_prediction(
        self,
        coarse_bbox: list[float],
        refined_bbox: list[float],
        *,
        visible_score: float,
    ) -> bool:
        if visible_score < self.cfg.visible_accept_threshold:
            return False

        ccx, ccy, cw, ch = _bbox_xyxy_to_cxcywh(coarse_bbox)
        rcx, rcy, rw, rh = _bbox_xyxy_to_cxcywh(refined_bbox)
        max_side = max(cw, ch, 1.0)
        center_shift = float(np.hypot(rcx - ccx, rcy - ccy)) / max_side
        if center_shift > self.cfg.max_center_shift_ratio:
            return False

        size_change = max(abs(rw - cw) / max(cw, 1.0), abs(rh - ch) / max(ch, 1.0))
        if size_change > self.cfg.max_size_change_ratio:
            return False
        return True


def _extract_roi(
    frame_bgr: np.ndarray,
    bbox_xyxy: list[float],
    *,
    roi_size: int,
    expand_ratio: float,
) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    cx, cy, bw, bh = _bbox_xyxy_to_cxcywh(bbox_xyxy)
    crop_w = max(8.0, bw * expand_ratio)
    crop_h = max(8.0, bh * expand_ratio)
    x1 = int(round(max(0.0, cx - 0.5 * crop_w)))
    y1 = int(round(max(0.0, cy - 0.5 * crop_h)))
    x2 = int(round(min(float(frame_w), cx + 0.5 * crop_w)))
    y2 = int(round(min(float(frame_h), cy + 0.5 * crop_h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


def _bbox_xyxy_to_cxcywh(bbox_xyxy: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy, w, h


def _bbox_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [
        float(cx - 0.5 * w),
        float(cy - 0.5 * h),
        float(cx + 0.5 * w),
        float(cy + 0.5 * h),
    ]


def _apply_bbox_delta(bbox_xyxy: list[float], delta_xywh: tuple[float, float, float, float]) -> list[float]:
    cx, cy, w, h = _bbox_xyxy_to_cxcywh(bbox_xyxy)
    dx, dy, dw, dh = [float(v) for v in delta_xywh]
    refined_cx = cx + dx * w
    refined_cy = cy + dy * h
    refined_w = w * float(np.exp(dw))
    refined_h = h * float(np.exp(dh))
    return _bbox_cxcywh_to_xyxy(refined_cx, refined_cy, refined_w, refined_h)


def _blend_bbox(coarse_bbox: list[float], refined_bbox: list[float], alpha: float) -> list[float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    ccx, ccy, cw, ch = _bbox_xyxy_to_cxcywh(coarse_bbox)
    rcx, rcy, rw, rh = _bbox_xyxy_to_cxcywh(refined_bbox)
    cx = (1.0 - alpha) * ccx + alpha * rcx
    cy = (1.0 - alpha) * ccy + alpha * rcy
    w = (1.0 - alpha) * cw + alpha * rw
    h = (1.0 - alpha) * ch + alpha * rh
    return _bbox_cxcywh_to_xyxy(cx, cy, max(w, 1.0), max(h, 1.0))


def _bbox_geometry_features(
    bbox_xyxy: list[float],
    *,
    frame_w: int,
    frame_h: int,
    confidence: float,
    source_flag: float,
    prev_bbox: list[float] | None,
) -> np.ndarray:
    cx, cy, w, h = _bbox_xyxy_to_cxcywh(bbox_xyxy)
    dist_left = cx - 0.5 * w
    dist_right = frame_w - (cx + 0.5 * w)
    dist_top = cy - 0.5 * h
    dist_bottom = frame_h - (cy + 0.5 * h)

    if prev_bbox is None:
        delta_cx = delta_cy = delta_w = delta_h = 0.0
        area_ratio = 1.0
    else:
        pcx, pcy, pw, ph = _bbox_xyxy_to_cxcywh(prev_bbox)
        delta_cx = (cx - pcx) / max(pw, 1.0)
        delta_cy = (cy - pcy) / max(ph, 1.0)
        delta_w = np.log(max(w, 1.0) / max(pw, 1.0))
        delta_h = np.log(max(h, 1.0) / max(ph, 1.0))
        area_ratio = (w * h) / max(pw * ph, 1.0)

    features = np.array(
        [
            cx / max(frame_w, 1.0),
            cy / max(frame_h, 1.0),
            w / max(frame_w, 1.0),
            h / max(frame_h, 1.0),
            float(confidence),
            dist_left / max(frame_w, 1.0),
            dist_right / max(frame_w, 1.0),
            dist_top / max(frame_h, 1.0),
            dist_bottom / max(frame_h, 1.0),
            float(delta_cx),
            float(delta_cy),
            float(delta_w),
            float(delta_h),
            float(area_ratio),
            float(w / max(h, 1.0)),
            float(source_flag),
        ],
        dtype=np.float32,
    )
    return features
