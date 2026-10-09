from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from depth_estimation.depth_temporal import SequenceConfidenceKalmanDepthFilter
from depth_estimation.fish_position import FishPositionEstimator, YOLOOnlyFishTracker


def stats(*, core_px=48, z_iqr=0.0, sep_score=2.0):
    return SimpleNamespace(
        core_px=core_px,
        z_iqr=z_iqr,
        sep_score=sep_score,
    )


class SequenceDepthFilterAugust5RollbackTests(unittest.TestCase):
    @staticmethod
    def make_filter(**overrides):
        cfg = {
            "buffer_size": 6,
            "fallback_dt_s": 0.1,
            "N_ref": 48,
            "sigma_iqr": 0.04,
            "sep_ref": 2.0,
            "sigma_res": 0.25,
            "R_base": 0.04,
            "measurement_variance_mode": "quality_scaled",
            "process_var_z": 0.02,
            "process_var_v": 0.5,
            "init_var_z": 0.04,
            "init_var_v": 1.0,
            "r_min": 0.05,
            "nis_gate_enabled": False,
            "hard_gate_enabled": True,
            "hard_gate_history_size": 4,
            "hard_gate_min_history": 3,
            "hard_gate_pred_margin_m": 0.28,
            "hard_gate_hist_margin_m": 0.28,
            "hard_gate_rel_ratio": 0.35,
        }
        cfg.update(overrides)
        return SequenceConfidenceKalmanDepthFilter(cfg)

    def test_august5_quality_scaled_measurement_variance(self):
        filt = self.make_filter()

        good = filt._measurement_variance_from_stats(
            stats(), fallback_confidence=0.9, quality=0.5
        )
        weak = filt._measurement_variance_from_stats(
            stats(), fallback_confidence=0.9, quality=0.01
        )

        self.assertAlmostEqual(good, 0.08)
        self.assertAlmostEqual(weak, 0.8)

    def test_state_is_scalar_depth_and_depth_velocity(self):
        filt = self.make_filter()

        self.assertEqual(filt.x.shape, (2,))
        self.assertEqual(filt.P.shape, (2, 2))

    def test_residual_is_part_of_august5_quality_score(self):
        filt = self.make_filter()
        roi_stats = stats()

        quality_near, _ = filt._quality_from_stats(
            1.0,
            depth_stats=roi_stats,
            z_pred=1.0,
            depth_confidence=1.0,
        )
        quality_far, _ = filt._quality_from_stats(
            1.5,
            depth_stats=roi_stats,
            z_pred=1.0,
            depth_confidence=1.0,
        )

        self.assertGreater(quality_near, quality_far)

    def test_august5_history_gate_rejects_large_outlier(self):
        filt = self.make_filter()
        for _ in range(4):
            filt.update(1.0, depth_stats=stats(), dt_s=0.1)

        state = filt.update(3.0, depth_stats=stats(), dt_s=0.1)

        self.assertTrue(state["rejected"])
        self.assertEqual(state["mode"], "depth_reject_hard")

    def test_august5_process_covariance_and_parameter_names(self):
        filt = self.make_filter()
        filt.initialized = True
        filt.x[:] = [2.0, -0.25]
        filt.P[:] = 0.0
        dt = 0.2

        _, p_pred = filt._predict_state(dt)

        expected = np.array([
            [0.02 * max(dt, 0.1) + 0.25 * 0.5 * dt**4, 0.5 * 0.5 * dt**3],
            [0.5 * 0.5 * dt**3, 0.5 * dt**2 + 1e-9],
        ])
        np.testing.assert_allclose(p_pred, expected)

    def test_august5_state_keeps_measurement_diagnostics(self):
        filt = self.make_filter()

        state = filt.update(1.0, depth_stats=stats(), dt_s=0.1)

        self.assertIn("R_t", state)
        self.assertIn("nis", state)
        self.assertIn("S_t", state)

    def test_disabled_depth_core_uses_complete_final_mask(self):
        final_mask = np.array([[True, False], [True, True]])

        depth_mask = FishPositionEstimator._extract_depth_core_mask(
            final_mask,
            cfg={"use_foreground_core_for_depth": False},
        )

        np.testing.assert_array_equal(depth_mask, final_mask)

    def test_yolo_only_path_predicts_before_each_new_measurement(self):
        tracker = YOLOOnlyFishTracker(
            max_age=0,
            temporal_depth={
                "filter_type": "sequence_conf_kalman",
                "enabled": True,
                "fallback_dt_s": 0.1,
                "fusion": self.make_filter().cfg.__dict__,
                "camera": {"fx": 715.0, "fy": 715.0, "cx": 320.0, "cy": 240.0},
                "center_ema": {"enabled": False},
            },
        )
        modes = []
        gains = []
        for _ in range(8):
            track = tracker.update([{
                "bbox": [100, 100, 140, 140],
                "pos_3d": np.array([0.0, 0.0, 1.2], dtype=np.float32),
                "confidence": 0.8,
                "depth_stats": stats(core_px=40, z_iqr=0.02, sep_score=1.5),
                "depth_confidence": 0.7,
                "source": "yolo",
            }], dt_s=0.1)[0]
            modes.append(track.depth_filter_mode)
            gains.append(track.depth_gain)

        self.assertEqual(modes[0], "update")
        self.assertTrue(all(mode == "weak_update" for mode in modes[1:]))
        self.assertGreater(min(gains[1:]), 0.15)


if __name__ == "__main__":
    unittest.main()
