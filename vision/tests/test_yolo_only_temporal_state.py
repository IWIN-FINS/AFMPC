from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from depth_estimation.fish_position import YOLOOnlyFishTracker


def _stats(z: float = 1.2):
    del z
    return SimpleNamespace(
        core_px=600,
        z_iqr=0.02,
        sep_score=2.0,
        valid_ratio=0.15,
    )


def _detection(center_u: float, z: float = 1.2) -> dict:
    return {
        "bbox": [center_u - 20.0, 100.0, center_u + 20.0, 140.0],
        "pos_3d": np.array([0.0, 0.0, z], dtype=np.float32),
        "confidence": 0.8,
        "depth_stats": _stats(z),
        "depth_confidence": 0.8,
        "source": "yolo",
    }


def _tracker() -> YOLOOnlyFishTracker:
    return YOLOOnlyFishTracker(
        temporal_depth={
            "filter_type": "sequence_conf_kalman",
            "enabled": True,
            "fallback_dt_s": 0.1,
            "state_hold_max_s": 0.6,
            "center_filter": {
                "type": "alpha_beta",
                "enabled": True,
                "alpha": 0.32,
                "beta": 0.04,
            },
            "fusion": {
                "fallback_dt_s": 0.1,
                "effective_sample_divisor": 500.0,
                "measurement_var_min": 1.0e-6,
                "measurement_var_max": 1.0e-3,
                "process_var_z": 0.003,
                "process_var_v": 0.2,
                "nis_gate_enabled": True,
                "nis_gate_threshold": 6.63,
            },
            "camera": {"fx": 715.0, "fy": 715.0, "cx": 320.0, "cy": 240.0},
        }
    )


def test_yolo_only_carries_alpha_beta_center_state_between_frames():
    tracker = _tracker()
    first = tracker.update([_detection(100.0)], dt_s=0.1)[0]
    second = tracker.update([_detection(110.0)], dt_s=0.1)[0]

    assert first.filtered_center_uv == (100.0, 120.0)
    np.testing.assert_allclose(second.filtered_center_uv, (103.2, 120.0), atol=1e-9)
    assert second.center_ab_velocity_uv[0] > 0.0
    assert second.output_bbox == _detection(110.0)["bbox"]


def test_yolo_only_keeps_internal_state_but_emits_no_box_during_short_miss():
    tracker = _tracker()
    tracker.update([_detection(100.0)], dt_s=0.1)
    tracker.update([_detection(110.0)], dt_s=0.1)

    assert tracker.update([], dt_s=0.1) == []
    assert tracker.track is None
    assert tracker.tracks == []
    assert tracker._temporal_track is not None
    predicted_u = tracker._temporal_track.filtered_center_uv[0]

    recovered = tracker.update([_detection(120.0)], dt_s=0.1)[0]
    assert predicted_u < recovered.filtered_center_uv[0] < 120.0


def test_yolo_only_discards_temporal_state_after_hold_timeout():
    tracker = _tracker()
    tracker.update([_detection(100.0)], dt_s=0.1)
    tracker.update([], dt_s=0.7)

    assert tracker._temporal_track is None
    restarted = tracker.update([_detection(130.0)], dt_s=0.1)[0]
    assert restarted.filtered_center_uv == (130.0, 120.0)
