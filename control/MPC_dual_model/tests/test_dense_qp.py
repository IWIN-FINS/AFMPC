import unittest

import numpy as np

from dense_qp import DenseADMMQPSolver, QPSolverSettings


class DenseQPTest(unittest.TestCase):
    def test_bound_constrained_scalar(self) -> None:
        # min (x-2)^2 subject to 0 <= x <= 1; optimum is x=1.
        solver = DenseADMMQPSolver(
            QPSolverSettings(
                max_iterations=2000,
                absolute_tolerance=1e-8,
                relative_tolerance=1e-8,
            )
        )
        result = solver.solve(
            P=np.array([[2.0]]),
            q=np.array([-4.0]),
            A=np.array([[1.0]]),
            lower=np.array([0.0]),
            upper=np.array([1.0]),
        )
        self.assertTrue(result.solved, result.status)
        self.assertAlmostEqual(result.x[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
