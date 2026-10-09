from __future__ import annotations

import unittest

import numpy as np

from depth_estimation.fish_position import FishTrack


class CenterEMAFormulaTests(unittest.TestCase):
    @staticmethod
    def make_track():
        track = FishTrack.__new__(FishTrack)
        track.center_ema_tau_s = 0.615
        track.center_ema_fallback_dt_s = 0.1
        track.center_ema_alpha = 0.15
        track.center_ema_enabled = True
        track.center_ema_initialized = False
        track.center_ema_prior_uv = None
        track.center_ema_step_alpha = 1.0
        track.filtered_center_uv = None
        return track

    def test_variable_dt_alpha_matches_time_constant_solution(self):
        track = self.make_track()

        alpha = track._resolve_center_ema_alpha(0.2)

        self.assertAlmostEqual(alpha, 1.0 - np.exp(-0.2 / 0.615), places=15)

    def test_fixed_alpha_is_only_fallback_when_time_constant_disabled(self):
        track = self.make_track()
        track.center_ema_tau_s = 0.0

        self.assertEqual(track._resolve_center_ema_alpha(0.2), 0.15)

    def test_same_frame_refinement_replaces_instead_of_reapplying_ema(self):
        track = self.make_track()
        track._update_center_ema((0.0, 0.0), dt_s=0.1, replace_current_step=False)
        track._update_center_ema((10.0, 4.0), dt_s=0.2, replace_current_step=False)
        alpha = 1.0 - np.exp(-0.2 / 0.615)
        np.testing.assert_allclose(track.filtered_center_uv, (10.0 * alpha, 4.0 * alpha))

        track._update_center_ema((20.0, 8.0), dt_s=0.2, replace_current_step=True)

        np.testing.assert_allclose(track.filtered_center_uv, (20.0 * alpha, 8.0 * alpha))

    def test_first_frame_refinement_replaces_ema_initialization(self):
        track = self.make_track()
        track._update_center_ema((1.0, 2.0), dt_s=0.1, replace_current_step=False)

        track._update_center_ema((3.0, 4.0), dt_s=0.1, replace_current_step=True)

        self.assertEqual(track.filtered_center_uv, (3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
