import unittest

import numpy as np

from MPC_dual_model.rigid_target_extrinsic_analysis import (
    quaternion_xyzw_to_rotation,
    rigid_fit,
    robust_fit,
    rotation_angle_deg,
)


class RigidTargetExtrinsicAnalysisTests(unittest.TestCase):
    def test_quaternion_rotation(self):
        rotation = quaternion_xyzw_to_rotation((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
        np.testing.assert_allclose(
            rotation @ np.asarray([1.0, 0.0, 0.0]),
            (0.0, 1.0, 0.0),
            atol=1e-12,
        )
        self.assertAlmostEqual(rotation_angle_deg(rotation), 90.0)

    def test_rigid_fit_recovers_known_transform(self):
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.3, -0.2, 0.8],
            ]
        )
        rotation = quaternion_xyzw_to_rotation((0.1, -0.2, 0.05, 0.97))
        translation = np.asarray((0.25, -0.08, 0.13))
        target = source @ rotation.T + translation
        fitted_rotation, fitted_translation, _ = rigid_fit(source, target)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-12)

    def test_robust_fit_rejects_large_outlier(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(30, 3)) * 0.2
        rotation = quaternion_xyzw_to_rotation((0.02, -0.04, 0.01, 0.998))
        translation = np.asarray((0.2, -0.03, 0.05))
        target = source @ rotation.T + translation
        target[-1] += (1.0, -1.0, 0.8)
        fitted_rotation, fitted_translation, _, keep, _, _ = robust_fit(source, target)
        self.assertFalse(keep[-1])
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
