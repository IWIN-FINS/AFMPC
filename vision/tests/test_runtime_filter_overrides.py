from __future__ import annotations

import unittest

from depth_estimation.fish_position import apply_temporal_filter_override


class RuntimeFilterOverrideTests(unittest.TestCase):
    def test_disables_output_filters_but_preserves_mask_settings(self):
        config = {
            "runtime": {},
            "temporal_depth": {
                "enabled": True,
                "center_filter": {"type": "alpha_beta", "enabled": True},
                "center_ema": {"enabled": False},
                "uvz_joint": {"enabled": True},
                "fusion": {"enabled": True},
                "roi": {
                    "mask_version": "external_ring_mahalanobis_otsu",
                    "temporal_prior_recovery_enabled": True,
                },
            },
        }

        overridden = apply_temporal_filter_override(config, False)
        temporal = overridden["temporal_depth"]

        self.assertFalse(temporal["enabled"])
        self.assertFalse(temporal["fusion"]["enabled"])
        self.assertFalse(temporal["uvz_joint"]["enabled"])
        self.assertEqual(temporal["center_filter"]["type"], "raw")
        self.assertFalse(temporal["center_filter"]["enabled"])
        self.assertEqual(
            temporal["roi"]["mask_version"],
            "external_ring_mahalanobis_otsu",
        )
        self.assertTrue(temporal["roi"]["temporal_prior_recovery_enabled"])

    def test_none_leaves_config_unchanged(self):
        config = {"temporal_depth": {"enabled": True}}

        self.assertIs(apply_temporal_filter_override(config, None), config)


if __name__ == "__main__":
    unittest.main()
