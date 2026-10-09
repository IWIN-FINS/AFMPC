"""
Generic Target Position Estimation Module
=========================================
Separate generic-target version of the fish pipeline.

This keeps the original fish estimator intact while allowing quick smoke tests
with any object detected by a generic COCO YOLO model.
"""

import os
import sys

import numpy as np
import torch
import yaml

from depth_estimation.fish_position import FishPositionEstimator, FishTrack, FishTracker
from depth_estimation.rectifier import StereoRectifier


class GenericTargetTrack(FishTrack):
    """Track state with original YOLO class metadata kept for debugging."""

    __slots__ = ("class_id", "class_name", "label")

    def __init__(self, track_id, bbox, pos_3d, confidence, frame_id,
                 class_id=None, class_name=None, label=None,
                 temporal_depth_cfg=None, depth_stats=None,
                 depth_confidence=None, dt_s=None):
        super().__init__(
            track_id, bbox, pos_3d, confidence, frame_id,
            temporal_depth_cfg=temporal_depth_cfg,
            depth_stats=depth_stats,
            depth_confidence=depth_confidence,
            dt_s=dt_s,
            source="yolo",
        )
        self.class_id = class_id
        self.class_name = class_name
        self.label = label

    def update(self, bbox, pos_3d, confidence, frame_id, alpha=0.7,
               class_id=None, class_name=None, label=None,
               depth_stats=None, depth_confidence=None, dt_s=None):
        super().update(
            bbox, pos_3d, confidence, frame_id, alpha=alpha,
            depth_stats=depth_stats,
            depth_confidence=depth_confidence,
            dt_s=dt_s,
        )
        self.class_id = class_id
        self.class_name = class_name
        self.label = label


class GenericTargetTracker(FishTracker):
    """FishTracker variant that keeps generic target metadata."""

    def update(self, detections: list[dict],
               dt_s: float | None = None) -> list[GenericTargetTrack]:
        self._frame_count += 1

        for track in self.tracks:
            track.predict()

        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        for track_idx, det_idx in matched:
            det = detections[det_idx]
            self.tracks[track_idx].update(
                det["bbox"], det["pos_3d"], det["confidence"],
                self._frame_count, alpha=self.smoothing_alpha,
                class_id=det.get("class_id"),
                class_name=det.get("class_name"),
                label=det.get("label"),
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source="yolo",
            )

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].predict_depth_only(dt_s)

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks.append(GenericTargetTrack(
                self._next_id, det["bbox"], det["pos_3d"],
                det["confidence"], self._frame_count,
                class_id=det.get("class_id"),
                class_name=det.get("class_name"),
                label=det.get("label"),
                temporal_depth_cfg=self.temporal_depth_cfg,
                depth_stats=det.get("depth_stats"),
                depth_confidence=det.get("depth_confidence"),
                dt_s=dt_s,
                source="yolo",
            ))
            self._next_id += 1

        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        return [track for track in self.tracks if track.is_confirmed(self.min_hits)]


class GenericTargetPositionEstimator(FishPositionEstimator):
    """
    End-to-end generic target 3D position estimator.

    It uses YOLO detections as class-agnostic targets by default:
    any YOLO box above the confidence threshold enters the stereo-depth and
    tracking pipeline. Set detection.target_classes to narrow this later.
    """

    def __init__(self, config: dict):
        self._cfg = config
        self.yolo = None
        self.stereo = None
        self._input_padder = None
        self.rectifier = StereoRectifier(config.get("rectification", {}))
        tracker_cfg = dict(config.get("tracker", {}))
        tracker_type = str(tracker_cfg.pop("type", "simple")).lower()
        if tracker_type != "simple":
            raise ValueError(
                "GenericTargetPositionEstimator currently supports "
                "tracker.type='simple' only."
            )
        tracker_cfg["temporal_depth"] = config.get("temporal_depth", {})
        self.tracker = GenericTargetTracker(**tracker_cfg)
        self._frame_idx = 0
        self._last_timestamp = None
        self._load_models()

    def estimate(self,
                 left_img: np.ndarray,
                 right_img: np.ndarray) -> list[dict]:
        """Run detection, stereo depth, 3D projection, and tracking."""
        self._frame_idx += 1
        dt_s = self._measure_dt()
        left_img, right_img = self._rectify_pair(left_img, right_img)
        height, width = left_img.shape[:2]

        det_results = self._detect_targets(left_img)

        if not det_results:
            self.tracker.update([], dt_s=dt_s)
            torch.cuda.empty_cache()
            return self._pack_results([])

        torch.cuda.empty_cache()

        disparity_full = self._compute_disparity(left_img, right_img, height, width)

        detections = []
        has_valid_disp = (disparity_full > 0.5).sum() > 100
        for det in det_results:
            depth_stats = self._extract_roi_depth(det["bbox"], disparity_full)
            pos_3d = self._bbox_depth_to_3d(det["center"], depth_stats)
            if pos_3d is not None:
                detections.append(self._build_tracker_detection(
                    det, pos_3d, depth_stats, self._depth_confidence(depth_stats)))
            elif not has_valid_disp:
                nan_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                detections.append(self._build_tracker_detection(
                    det, nan_pos, depth_stats, 0.0))

        tracks = self.tracker.update(detections, dt_s=dt_s)
        return self._pack_results(tracks)

    def _load_models(self):
        """Load YOLO and FoundationStereo, preserving Ultralytics aliases."""
        cfg = self._cfg

        from ultralytics import YOLO

        yolo_path = self._resolve_yolo_path(cfg["models"]["yolo_path"])
        self.yolo = YOLO(yolo_path)
        self.yolo.to(cfg["detection"].get("device", "cuda"))
        print(f"[GenericTarget] YOLOv8 loaded from {yolo_path}")

        ckpt = self._resolve_local_path(cfg["models"]["stereo_ckpt"])
        cfg_yaml = self._resolve_local_path(cfg["models"]["stereo_cfg_yaml"])
        self._ensure_foundation_stereo_import_path()

        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        self._InputPadder = InputPadder

        if not os.path.isfile(cfg_yaml):
            raise FileNotFoundError(
                "FoundationStereo config not found: "
                f"{cfg_yaml}. Download/copy the whole FoundationStereo checkpoint "
                "folder so cfg.yaml and model_best_bp2.pth sit side by side, "
                "then update models.stereo_cfg_yaml in generic_config.yaml if needed."
            )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                "FoundationStereo checkpoint not found: "
                f"{ckpt}. Download/copy the whole FoundationStereo checkpoint "
                "folder so cfg.yaml and model_best_bp2.pth sit side by side, "
                "then update models.stereo_ckpt in generic_config.yaml if needed."
            )

        with open(cfg_yaml, "r", encoding="utf-8") as f:
            scfg = OmegaConf.create(yaml.safe_load(f))
        stereo_opts = self._cfg.get("stereo", {})
        for key, value in dict(
            valid_iters=stereo_opts.get("valid_iters", 32),
            hiera=0,
            scale=stereo_opts.get("image_scale", 0.5),
            low_memory=stereo_opts.get("low_memory", True),
            get_pc=0,
            remove_invisible=1,
        ).items():
            scfg[key] = value

        self.stereo = FoundationStereo(scfg)
        ck = torch.load(ckpt, weights_only=False)
        self.stereo.load_state_dict(ck["model"])
        self.stereo.cuda().eval()
        torch.cuda.empty_cache()
        print(f"[GenericTarget] FoundationStereo loaded from {ckpt}")

    def _detect_targets(self, img: np.ndarray) -> list[dict]:
        """Run YOLOv8 and return generic target detections."""
        det_cfg = self._cfg["detection"]
        min_conf = det_cfg.get("min_confidence", 0.4)
        target_label = det_cfg.get("target_label", "target")
        max_detections = det_cfg.get("max_detections")

        results = self.yolo(img, verbose=False)[0]
        allowed_classes = self._resolve_target_classes(
            det_cfg.get("target_classes"),
            getattr(results, "names", None),
        )

        dets = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            names = getattr(results, "names", {}) or {}

            for box, conf, cls_id in zip(boxes, confs, classes):
                if conf < min_conf:
                    continue
                if allowed_classes is not None and cls_id not in allowed_classes:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dets.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": (float(cx), float(cy)),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": str(names.get(int(cls_id), int(cls_id))),
                    "label": target_label,
                })

        dets.sort(key=lambda det: det["confidence"], reverse=True)
        dets = self._select_single_target(dets, det_cfg.get("single_target", {}))
        if max_detections:
            dets = dets[:int(max_detections)]
        return dets

    @staticmethod
    def _select_single_target(dets: list[dict], cfg) -> list[dict]:
        """Optionally keep one target for single-object tracking scenarios."""
        if not dets:
            return dets
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return dets

        strategy = str(cfg.get("strategy", "largest_area")).lower()

        def area(det):
            x1, y1, x2, y2 = det["bbox"]
            return max(0, x2 - x1) * max(0, y2 - y1)

        if strategy in ("largest_area", "largest", "area"):
            selected = max(dets, key=area)
        elif strategy in ("highest_confidence", "confidence", "conf"):
            selected = max(dets, key=lambda det: det["confidence"])
        else:
            raise ValueError(
                "Unsupported detection.single_target.strategy: "
                f"{strategy}. Use 'largest_area' or 'highest_confidence'."
            )
        return [selected]

    @staticmethod
    def _build_tracker_detection(det: dict, pos_3d: np.ndarray,
                                 depth_stats=None,
                                 depth_confidence: float = 0.0) -> dict:
        return {
            "bbox": det["bbox"],
            "pos_3d": pos_3d,
            "confidence": det["confidence"],
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "label": det.get("label"),
            "depth_stats": depth_stats,
            "depth_confidence": depth_confidence,
        }

    def _pack_results(self, tracks: list[GenericTargetTrack]) -> list[dict]:
        out = []
        for track in tracks:
            out.append({
                "id": track.id,
                "bbox": track.bbox,
                "position": track.pos_3d.tolist(),
                "confidence": track.confidence,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "label": track.label,
                "raw_depth": track.raw_depth,
                "depth_confidence": track.depth_confidence,
                "depth_valid": track.depth_valid,
                "depth_rejected": track.depth_rejected,
                "z_dot": track.z_dot,
            })
        return out

    def _resolve_yolo_path(self, yolo_path: str) -> str:
        """Resolve local weights while still allowing Ultralytics model aliases."""
        yolo_path = os.path.expanduser(str(yolo_path))
        if os.path.isabs(yolo_path) or os.path.exists(yolo_path):
            return os.path.normpath(yolo_path)

        for base in self._candidate_bases():
            candidate = os.path.join(base, yolo_path)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        return yolo_path

    def _resolve_local_path(self, path: str) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return os.path.normpath(path)

        for base in self._candidate_bases():
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        return os.path.normpath(os.path.join(os.path.dirname(__file__), path))

    def _candidate_bases(self) -> list[str]:
        bases = []
        cfg_dir = self._cfg.get("_config_dir")
        if cfg_dir:
            bases.append(cfg_dir)
        module_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        bases.extend([module_dir, repo_root])
        return bases

    def _ensure_foundation_stereo_import_path(self):
        module_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        candidates = [
            os.path.join(module_dir, "FoundationStereo"),
            os.path.join(repo_root, "third_party", "FoundationStereo"),
            os.path.join(repo_root, "third_party", "Fast-FoundationStereo"),
        ]

        for candidate in candidates:
            core_path = os.path.join(candidate, "core", "foundation_stereo.py")
            if os.path.isfile(core_path):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                return

        raise FileNotFoundError(
            "FoundationStereo source not found. Expected a directory containing "
            "core/foundation_stereo.py under src/depth_estimation/FoundationStereo "
            "or third_party/FoundationStereo."
        )

    @staticmethod
    def _resolve_target_classes(target_classes, names) -> set[int] | None:
        """Return class ids to keep; None means class-agnostic/all classes."""
        if target_classes in (None, "", []):
            return None
        if not isinstance(target_classes, (list, tuple, set)):
            target_classes = [target_classes]

        name_to_id = {}
        if isinstance(names, dict):
            name_to_id = {str(name).lower(): int(idx) for idx, name in names.items()}

        resolved = set()
        unknown = []
        for item in target_classes:
            if isinstance(item, int):
                resolved.add(item)
                continue

            text = str(item).strip()
            if text.isdigit():
                resolved.add(int(text))
            elif text.lower() in name_to_id:
                resolved.add(name_to_id[text.lower()])
            else:
                unknown.append(text)

        if unknown:
            print(f"[GenericTarget] WARNING: unknown YOLO target_classes: {unknown}")
        return resolved


def from_config_yaml(path: str) -> GenericTargetPositionEstimator:
    """Load configuration from a YAML file and return a ready estimator."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return GenericTargetPositionEstimator(cfg)
