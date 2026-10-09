import unittest

import numpy as np

from fossen_fixed_dl_model import FixedLinearDampingRelativeModel


class ModelTest(unittest.TestCase):
    def test_diagonal_closed_form_and_sign(self) -> None:
        mass = np.array([20.0, 25.0, 30.0])
        damping = np.array([8.0, 10.0, 12.0])
        dt = 0.1
        model = FixedLinearDampingRelativeModel(
            np.diag(mass), np.diag(damping), dt
        )
        expected_F = np.exp(-damping * dt / mass)
        expected_G = -(1.0 - expected_F) / damping
        np.testing.assert_allclose(np.diag(model.F), expected_F, rtol=1e-12)
        np.testing.assert_allclose(np.diag(model.G), expected_G, rtol=1e-12)

        _, velocity = model.predict_delta([1, 0, 0], [0, 0, 0], [2, 0, 0])
        self.assertLess(velocity[0], 0.0)

    def test_pdf_position_uses_end_of_step_relative_velocity(self) -> None:
        model = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.1,
        )
        position = np.array((1.0, -0.2, 0.3))
        velocity = np.array((0.4, -0.1, 0.2))
        force = np.array((2.0, -1.0, 0.5))
        position_next, velocity_next = model.predict_delta(
            position, velocity, force
        )
        np.testing.assert_allclose(
            position_next,
            position + model.dt * velocity_next,
            atol=1e-12,
        )

    def test_stationary_and_matched_equilibria_are_separate(self) -> None:
        restoring = np.array((0.0, 0.0, 0.80729))
        model = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.1,
            restoring_force=restoring,
        )
        position = np.array((0.6, 0.1, -0.2))
        zero_velocity = np.zeros(3)
        p_stationary, v_stationary = model.predict_stationary(
            position, zero_velocity, restoring
        )
        np.testing.assert_allclose(p_stationary, position, atol=1e-12)
        np.testing.assert_allclose(v_stationary, zero_velocity, atol=1e-12)

        matched_force = np.array((1.4, -0.5, 1.1))
        p_matched, v_matched = model.predict_matched(
            position, zero_velocity, matched_force, matched_force
        )
        np.testing.assert_allclose(p_matched, position, atol=1e-12)
        np.testing.assert_allclose(v_matched, zero_velocity, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
