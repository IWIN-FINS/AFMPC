import numpy as np
import unittest

from MPC_dual_model.channel_dynamics_analysis import _quaternion_matrix_xyzw


class ChannelDynamicsAnalysisTest(unittest.TestCase):
    def test_quaternion_axes_preserve_body_right_with_world_z_up(self) -> None:
        # 180 degrees about world/body X maps body FRD z-down into world z-up.
        rotation = _quaternion_matrix_xyzw(np.asarray([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotation[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)

    def test_quaternion_is_normalized(self) -> None:
        rotation = _quaternion_matrix_xyzw(np.asarray([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
