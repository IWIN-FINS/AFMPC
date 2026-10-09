import cv2
import numpy as np

from depth_estimation.fish_position import FishPositionEstimator


def test_external_ring_color_model_selects_roi_color_contrast():
    image = np.full((40, 40, 3), (120, 120, 120), dtype=np.uint8)
    image[12:28, 12:28] = (20, 20, 220)
    valid = np.ones((16, 16), dtype=bool)
    cfg = {
        "use_color_foreground_mask": True,
        "color_fg_bg_expand_ratio": 1.5,
        "color_fg_bg_min_pixels": 32,
        "color_fg_cov_shrinkage": 0.08,
        "color_fg_otsu_bins": 64,
        "color_fg_min_pixels": 8,
        "color_fg_min_ratio": 0.01,
        "color_fg_dilate_iterations": 0,
        "color_fg_external_ring_failure": "empty",
    }

    mask = FishPositionEstimator._apply_color_foreground_mask_external_ring(
        valid,
        color_image=image,
        roi_bbox_xyxy=(12, 12, 28, 28),
        cfg=cfg,
    )

    assert mask.shape == valid.shape
    assert np.count_nonzero(mask) > 0


def test_disparity_otsu_failure_does_not_fall_back_to_full_roi():
    valid = np.ones((4, 4), dtype=bool)
    disparity = np.full((4, 4), 8.0, dtype=np.float32)
    cfg = {
        "use_foreground_depth_mask": True,
        "foreground_min_pixels": 20,
        "foreground_min_ratio": 0.08,
        "foreground_disp_otsu_failure": "empty",
    }

    mask = FishPositionEstimator._apply_foreground_disparity_mask_otsu(
        valid,
        disparity,
        cfg=cfg,
    )

    assert not np.any(mask)


def test_strict_color_disparity_fusion_never_falls_back_to_color_only():
    valid = np.ones((10, 10), dtype=bool)
    color = np.zeros_like(valid)
    color[2:8, 2:8] = True
    depth = np.zeros_like(valid)
    depth[2:3, 2:3] = True
    cfg = {
        "use_color_foreground_mask": True,
        "use_foreground_depth_mask": True,
        "color_depth_support_dilate_iterations": 0,
        "color_depth_consensus_min_pixels": 16,
        "color_depth_consensus_min_ratio": 0.30,
        "intersection_invalid_if_sparse": True,
        "color_fg_allow_depth_fallback": False,
    }

    fused = FishPositionEstimator._fuse_foreground_masks(
        valid,
        depth_mask=depth,
        color_mask=color,
        cfg=cfg,
    )

    assert not np.any(fused)


def test_temporal_support_expands_with_measurement_interval():
    valid = np.ones((41, 41), dtype=bool)
    disparity = np.full((41, 41), 10.0, dtype=np.float32)
    cfg = {
        "center_prior_sigma_px": 3.0,
        "temporal_prior_chi2_threshold": 5.99,
    }

    short_dt = FishPositionEstimator._build_temporal_prior_support_mask(
        valid,
        disparity,
        center_prior_uv=(20.0, 20.0),
        depth_prior_m=None,
        fx=300.0,
        baseline_m=0.04,
        cfg=cfg,
        dt_s=0.1,
        reference_dt_s=0.1,
    )
    long_dt = FishPositionEstimator._build_temporal_prior_support_mask(
        valid,
        disparity,
        center_prior_uv=(20.0, 20.0),
        depth_prior_m=None,
        fx=300.0,
        baseline_m=0.04,
        cfg=cfg,
        dt_s=0.2,
        reference_dt_s=0.1,
    )

    assert np.count_nonzero(long_dt) > np.count_nonzero(short_dt)
