from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import cv2
import numpy as np
import torch
from torch import nn

from depth_estimation.refiner.temporal_bbox_refiner import (
    _apply_bbox_delta,
    _bbox_cxcywh_to_xyxy,
    _bbox_geometry_features,
    _bbox_xyxy_to_cxcywh,
    _extract_roi,
)


@dataclass
class TemporalBBoxCorrectorSmootherConfig:
    enabled: bool = False
    device: str = "cuda"
    checkpoint_path: str | None = None
    buffer_size: int = 5
    roi_size: int = 128
    crop_expand_ratio: float = 1.8
    quality_threshold: float = 0.45
    visible_accept_threshold: float = 0.35
    max_center_shift_ratio: float = 0.35
    max_size_change_ratio: float = 0.45
    output_smooth_enabled: bool = True
    output_smooth_center_alpha: float = 0.70
    output_smooth_size_alpha: float = 0.60
    apply_on_tracker_only: bool = False
    max_idle_frames: int = 3
    future_prior_enabled: bool = True
    future_prior_tracker_only: bool = True
    future_quality_threshold: float = 0.40
    future_visible_threshold: float = 0.30
    future_prior_center_alpha: float = 0.45
    future_prior_size_alpha: float = 0.35
    future_max_center_shift_ratio: float = 0.60
    future_max_size_change_ratio: float = 0.60

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "TemporalBBoxCorrectorSmootherConfig":
        cfg = dict(cfg or {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TemporalBBoxCorrectorSmootherPrediction:
    quality_score: float
    visible_score: float
    bbox_delta: tuple[float, float, float, float]
    future_quality_score: float
    future_visible_score: float
    future_bbox_delta: tuple[float, float, float, float]
    corrected_bbox: list[float]
    predicted_next_bbox: list[float] | None
    final_bbox: list[float]
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


class TemporalBBoxCorrectorSmootherV1(nn.Module):
    """Temporal bbox corrector that directly predicts a GT-facing current box."""

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
        self.bbox_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.visible_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.future_bbox_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.future_quality_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.future_visible_head = nn.Sequential(
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
            "bbox_delta": self.bbox_head(last),
            "quality_logit": self.quality_head(last),
            "visible_logit": self.visible_head(last),
            "future_bbox_delta": self.future_bbox_head(last),
            "future_quality_logit": self.future_quality_head(last),
            "future_visible_logit": self.future_visible_head(last),
        }


class TemporalBBoxCorrectorSmootherRuntime:
    """Single-target causal runtime for the corrector-smoother network."""

    def __init__(
        self,
        config: TemporalBBoxCorrectorSmootherConfig,
        model: TemporalBBoxCorrectorSmootherV1 | None = None,
    ):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model or TemporalBBoxCorrectorSmootherV1()
        self.model.to(self.device).eval()
        self.buffer: Deque[_FrameState] = deque(maxlen=max(int(config.buffer_size), 1))
        self.last_output_bbox: list[float] | None = None
        self.predicted_next_bbox: list[float] | None = None
        self.predicted_next_quality: float = 0.0
        self.predicted_next_visible: float = 0.0
        self.idle_frames = 0
        if config.checkpoint_path:
            state = torch.load(config.checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)

    def reset(self) -> None:
        self.buffer.clear()
        self.last_output_bbox = None
        self.predicted_next_bbox = None
        self.predicted_next_quality = 0.0
        self.predicted_next_visible = 0.0
        self.idle_frames = 0

    def skip(self) -> None:
        self.idle_frames += 1
        if self.idle_frames > max(int(self.cfg.max_idle_frames), 0):
            self.buffer.clear()
            self.last_output_bbox = None
            self.predicted_next_bbox = None
            self.predicted_next_quality = 0.0
            self.predicted_next_visible = 0.0

    def push(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: list[float],
        *,
        confidence: float,
        source: str,
    ) -> None:
        source_flag = 1.0 if str(source).lower().startswith(("yolo", "roi")) else 0.0
        self.buffer.append(
            _FrameState(
                frame=frame_bgr,
                bbox_xyxy=[float(v) for v in bbox_xyxy],
                confidence=float(confidence),
                source_flag=source_flag,
            )
        )
        self.idle_frames = 0

    def ready(self) -> bool:
        return len(self.buffer) >= self.cfg.buffer_size

    def correct_current(
        self,
        *,
        has_detection_support: bool,
    ) -> TemporalBBoxCorrectorSmootherPrediction | None:
        if not self.ready():
            return None
        if not has_detection_support and not self.cfg.apply_on_tracker_only:
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

        dx, dy, dw, dh = outputs["bbox_delta"][0].detach().cpu().tolist()
        quality_score = torch.sigmoid(outputs["quality_logit"])[0, 0].item()
        visible_score = torch.sigmoid(outputs["visible_logit"])[0, 0].item()
        future_dx, future_dy, future_dw, future_dh = (
            outputs["future_bbox_delta"][0].detach().cpu().tolist()
        )
        future_quality_score = torch.sigmoid(outputs["future_quality_logit"])[0, 0].item()
        future_visible_score = torch.sigmoid(outputs["future_visible_logit"])[0, 0].item()
        coarse_bbox = self.buffer[-1].bbox_xyxy
        corrected_bbox = _apply_bbox_delta(coarse_bbox, (dx, dy, dw, dh))
        accepted = self._accept_prediction(
            coarse_bbox,
            corrected_bbox,
            quality_score=quality_score,
            visible_score=visible_score,
        )

        final_bbox = list(coarse_bbox)
        if accepted:
            final_bbox = list(corrected_bbox)
            final_bbox = self._apply_future_prior(
                coarse_bbox,
                final_bbox,
                has_detection_support=has_detection_support,
            )
            if self.cfg.output_smooth_enabled and self.last_output_bbox is not None:
                final_bbox = _smooth_bbox(
                    self.last_output_bbox,
                    final_bbox,
                    center_alpha=float(self.cfg.output_smooth_center_alpha),
                    size_alpha=float(self.cfg.output_smooth_size_alpha),
                )
            self.last_output_bbox = list(final_bbox)
        elif self._should_use_future_prior(has_detection_support=has_detection_support):
            prior_bbox = self._build_future_prior_bbox(coarse_bbox)
            if prior_bbox is not None:
                final_bbox = prior_bbox
                accepted = True
                self.last_output_bbox = list(final_bbox)
        elif has_detection_support:
            self.last_output_bbox = list(coarse_bbox)

        predicted_next_bbox = None
        if accepted:
            predicted_next_bbox = _apply_bbox_delta(
                final_bbox,
                (future_dx, future_dy, future_dw, future_dh),
            )
            if self._accept_future_bbox(
                final_bbox,
                predicted_next_bbox,
                quality_score=future_quality_score,
                visible_score=future_visible_score,
            ):
                self.predicted_next_bbox = list(predicted_next_bbox)
                self.predicted_next_quality = float(future_quality_score)
                self.predicted_next_visible = float(future_visible_score)
            else:
                predicted_next_bbox = None
                self.predicted_next_bbox = None
                self.predicted_next_quality = 0.0
                self.predicted_next_visible = 0.0
        elif not has_detection_support:
            predicted_next_bbox = self.predicted_next_bbox

        return TemporalBBoxCorrectorSmootherPrediction(
            quality_score=float(quality_score),
            visible_score=float(visible_score),
            bbox_delta=(float(dx), float(dy), float(dw), float(dh)),
            future_quality_score=float(future_quality_score),
            future_visible_score=float(future_visible_score),
            future_bbox_delta=(
                float(future_dx),
                float(future_dy),
                float(future_dw),
                float(future_dh),
            ),
            corrected_bbox=list(corrected_bbox),
            predicted_next_bbox=list(predicted_next_bbox) if predicted_next_bbox is not None else None,
            final_bbox=list(final_bbox),
            accepted=accepted,
        )

    def _accept_prediction(
        self,
        coarse_bbox: list[float],
        corrected_bbox: list[float],
        *,
        quality_score: float,
        visible_score: float,
    ) -> bool:
        if quality_score < self.cfg.quality_threshold:
            return False
        if visible_score < self.cfg.visible_accept_threshold:
            return False

        ccx, ccy, cw, ch = _bbox_xyxy_to_cxcywh(coarse_bbox)
        rcx, rcy, rw, rh = _bbox_xyxy_to_cxcywh(corrected_bbox)
        max_side = max(cw, ch, 1.0)
        center_shift = float(np.hypot(rcx - ccx, rcy - ccy)) / max_side
        if center_shift > self.cfg.max_center_shift_ratio:
            return False

        size_change = max(abs(rw - cw) / max(cw, 1.0), abs(rh - ch) / max(ch, 1.0))
        if size_change > self.cfg.max_size_change_ratio:
            return False
        return True

    def _should_use_future_prior(self, *, has_detection_support: bool) -> bool:
        if not self.cfg.future_prior_enabled:
            return False
        if self.predicted_next_bbox is None:
            return False
        if self.predicted_next_quality < self.cfg.future_quality_threshold:
            return False
        if self.predicted_next_visible < self.cfg.future_visible_threshold:
            return False
        if self.cfg.future_prior_tracker_only and has_detection_support:
            return False
        return True

    def _build_future_prior_bbox(self, coarse_bbox: list[float]) -> list[float] | None:
        if self.predicted_next_bbox is None:
            return None
        if not self._accept_future_bbox(
            coarse_bbox,
            self.predicted_next_bbox,
            quality_score=self.predicted_next_quality,
            visible_score=self.predicted_next_visible,
        ):
            return None
        return _smooth_bbox(
            coarse_bbox,
            self.predicted_next_bbox,
            center_alpha=float(self.cfg.future_prior_center_alpha),
            size_alpha=float(self.cfg.future_prior_size_alpha),
        )

    def _apply_future_prior(
        self,
        coarse_bbox: list[float],
        final_bbox: list[float],
        *,
        has_detection_support: bool,
    ) -> list[float]:
        if not self._should_use_future_prior(has_detection_support=has_detection_support):
            return final_bbox
        prior_bbox = self._build_future_prior_bbox(coarse_bbox)
        if prior_bbox is None:
            return final_bbox
        return _smooth_bbox(
            final_bbox,
            prior_bbox,
            center_alpha=float(self.cfg.future_prior_center_alpha),
            size_alpha=float(self.cfg.future_prior_size_alpha),
        )

    def _accept_future_bbox(
        self,
        anchor_bbox: list[float],
        predicted_bbox: list[float],
        *,
        quality_score: float,
        visible_score: float,
    ) -> bool:
        if quality_score < self.cfg.future_quality_threshold:
            return False
        if visible_score < self.cfg.future_visible_threshold:
            return False

        acx, acy, aw, ah = _bbox_xyxy_to_cxcywh(anchor_bbox)
        pcx, pcy, pw, ph = _bbox_xyxy_to_cxcywh(predicted_bbox)
        max_side = max(aw, ah, 1.0)
        center_shift = float(np.hypot(pcx - acx, pcy - acy)) / max_side
        if center_shift > self.cfg.future_max_center_shift_ratio:
            return False

        size_change = max(abs(pw - aw) / max(aw, 1.0), abs(ph - ah) / max(ah, 1.0))
        if size_change > self.cfg.future_max_size_change_ratio:
            return False
        return True


def _smooth_bbox(
    prev_bbox: list[float],
    new_bbox: list[float],
    *,
    center_alpha: float,
    size_alpha: float,
) -> list[float]:
    center_alpha = float(np.clip(center_alpha, 0.0, 1.0))
    size_alpha = float(np.clip(size_alpha, 0.0, 1.0))
    pcx, pcy, pw, ph = _bbox_xyxy_to_cxcywh(prev_bbox)
    ncx, ncy, nw, nh = _bbox_xyxy_to_cxcywh(new_bbox)
    cx = (1.0 - center_alpha) * pcx + center_alpha * ncx
    cy = (1.0 - center_alpha) * pcy + center_alpha * ncy
    w = (1.0 - size_alpha) * pw + size_alpha * nw
    h = (1.0 - size_alpha) * ph + size_alpha * nh
    return _bbox_cxcywh_to_xyxy(cx, cy, max(w, 1.0), max(h, 1.0))
