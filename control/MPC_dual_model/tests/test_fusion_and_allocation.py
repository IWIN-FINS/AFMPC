import unittest

import numpy as np

from device_adapter import FineSUBThrusterAllocator
from fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from model_fusion import FusionConfig, OnlineModelFusion
from mpc_controller import MPCConfig, RelativeMPCController


class FusionAndAllocationTest(unittest.TestCase):
    def test_fusion_prefers_better_model_and_stays_bounded(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(minimum_weight=0.05, initial_model1_weight=(0.8, 0.8, 0.8))
        )
        weight = fusion.observe_position(
            actual_position=(0.0, 0.0, 0.0),
            prediction1=(0.0, 0.0, 0.0),
            prediction2=(0.5, -0.4, 0.3),
            horizon_step=1,
        )
        np.testing.assert_allclose(weight, np.full(3, 0.95))
        self.assertTrue(np.allclose(fusion.model1_weight + fusion.model2_weight, 1.0))

    def test_small_steady_residuals_are_not_hidden_by_epsilon(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                epsilon=1.0e-10,
                minimum_weight=0.05,
                initial_model1_weight=(0.8, 0.8, 0.8),
            )
        )
        weight = fusion.observe_position(
            actual_position=(0.0, 0.0, 0.0),
            prediction1=(0.0, 0.0, 0.0),
            prediction2=(0.001, 0.0, 0.0),
            horizon_step=1,
        )
        # A 1e-6 m2 candidate difference must still select model 1. A 1e-5
        # regularizer would incorrectly pull this result toward 0.5.
        self.assertAlmostEqual(weight[0], 0.95)

    def test_noise_scale_difference_keeps_previous_model_preference(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                initial_model1_weight=(0.8, 0.8, 0.8),
                minimum_weight=0.01,
                indistinguishable_score_threshold=(1.0e-7, 1.0e-7, 1.0e-7),
            )
        )
        weight = fusion.observe_position(
            actual_position=(0.0, 0.0, 0.0),
            prediction1=(0.0, 0.0, 0.0),
            prediction2=(1.0e-4, 1.0e-4, 1.0e-4),
            horizon_step=1,
        )
        # A 0.1 mm candidate difference is below the configured score floor;
        # neutral evidence leaves the preceding model preference unchanged.
        np.testing.assert_allclose(weight, (0.8, 0.8, 0.8))

    def test_fixed_model2_override_holds_zero_model1_weight(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                fixed_model1_weight=(0.0, 0.0, 0.0),
                initial_model1_weight=(0.8, 0.8, 0.8),
                minimum_weight=0.01,
            )
        )
        self.assertEqual(fusion.model1_weight.tolist(), [0.0, 0.0, 0.0])
        weight = fusion.observe_position(
            actual_position=(0.0, 0.0, 0.0),
            prediction1=(0.0, 0.0, 0.0),
            prediction2=(0.5, -0.4, 0.3),
            horizon_step=1,
        )
        np.testing.assert_allclose(weight, (0.0, 0.0, 0.0))

    def test_position_history_keeps_completed_multi_step_scores(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                window=2,
                prediction_horizon=3,
                horizon_weight_decay=0.8,
            )
        )
        for step in range(8):
            fusion.observe_position(
                actual_position=(0.0, 0.0, 0.0),
                prediction1=(0.0, 0.0, 0.0),
                prediction2=(0.2, 0.0, 0.0),
                horizon_step=step % 3 + 1,
            )
        # Two historical start times, each retaining three horizons.
        self.assertEqual(fusion.sample_count, 6)
        # Only x differs between the two candidate predictions.  Identical
        # y/z candidates are intentionally neutral rather than favouring one.
        self.assertGreater(fusion.model1_weight[0], fusion.model2_weight[0])
        # Identical candidates preserve the prior rather than inventing a
        # model-1 observation or drifting toward an arbitrary 50/50 split.
        np.testing.assert_allclose(fusion.model1_weight[1:], (0.8, 0.8))

    def test_default_staircase_can_reach_ninety_nine_percent_model1(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                minimum_weight=0.01,
                weight_update_rate=1.0,
                initial_model1_weight=(0.8, 0.8, 0.8),
            )
        )
        weight = fusion.observe_position(
            actual_position=(0.0, 0.0, 0.0),
            prediction1=(0.0, 0.0, 0.0),
            prediction2=(0.5, -0.4, 0.3),
            horizon_step=1,
        )
        np.testing.assert_allclose(weight, (0.99, 0.99, 0.99))

    def test_staircase_history_masks_old_origin_long_prediction(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                window=6,
                prediction_horizon=3,
                forgetting_factor=0.8,
                prediction_horizon_weights=(0.5, 0.3, 0.2),
                staircase_horizon_caps=(3, 3, 2, 2, 1, 1),
            )
        )
        for target in (1, 2, 3):
            fusion.observe_position(
                actual_position=(0.0, 0.0, 0.0),
                prediction1=(0.0, 0.0, 0.0),
                prediction2=(0.2, 0.0, 0.0),
                horizon_step=target,
                origin_index=0,
                target_index=target,
            )
        fusion.advance_time(3)
        self.assertIn((0, 1), fusion.active_pairs)
        self.assertIn((0, 2), fusion.active_pairs)
        self.assertNotIn((0, 3), fusion.active_pairs)  # k|k-3 is masked.
        self.assertEqual(fusion.sample_count, 2)
        self.assertGreater(fusion.model1_weight[0], fusion.model2_weight[0])

    def test_staircase_rejects_only_ambiguous_prediction_window(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                window=3,
                prediction_horizon=3,
                staircase_horizon_caps=(3, 3, 3),
                indistinguishable_score_threshold=(1.0e-4, 1.0e-4, 1.0e-4),
                initial_model1_weight=(0.5, 0.5, 0.5),
                minimum_weight=0.01,
            )
        )
        # Origin 0 has only a 1 mm candidate gap and is rejected as one
        # complete window. Origin 1 has a 100 mm gap and remains usable.
        for target in (1, 2, 3):
            fusion.observe_position(
                actual_position=(0.0, 0.0, 0.0),
                prediction1=(0.0, 0.0, 0.0),
                prediction2=(0.001, 0.001, 0.001),
                horizon_step=target,
                origin_index=0,
                target_index=target,
            )
        for target in (2, 3):
            fusion.observe_position(
                actual_position=(0.0, 0.0, 0.0),
                prediction1=(0.0, 0.0, 0.0),
                prediction2=(0.1, 0.1, 0.1),
                horizon_step=target - 1,
                origin_index=1,
                target_index=target,
            )
        fusion.advance_time(3)
        np.testing.assert_array_equal(fusion.window_valid_count, (1, 1, 1))
        np.testing.assert_array_equal(fusion.window_rejected_count, (1, 1, 1))
        self.assertGreater(fusion.model1_weight[0], fusion.model2_weight[0])

    def test_six_windows_one_rejected_uses_other_five(self) -> None:
        fusion = OnlineModelFusion(
            FusionConfig(
                window=6,
                prediction_horizon=3,
                prediction_horizon_weights=(1.0, 1.0, 1.0),
                staircase_horizon_caps=(3, 3, 3, 3, 3, 3),
                forgetting_factor=1.0,
                indistinguishable_score_threshold=(1.0e-4,) * 3,
                initial_model1_weight=(0.5,) * 3,
                minimum_weight=0.01,
            )
        )
        # Feed a rolling stream. At frame 7 the six retained origins are
        # 1..6. Origin 6 has only a one-step ambiguous candidate gap; the
        # other five origins have three clear steps each.
        for target in range(8):
            for origin in range(max(0, target - 3), target):
                horizon = target - origin
                candidate2 = 0.001 if origin == 6 else 0.1
                fusion.observe_position(
                    actual_position=(0.0, 0.0, 0.0),
                    prediction1=(0.0, 0.0, 0.0),
                    prediction2=(candidate2, candidate2, candidate2),
                    horizon_step=horizon,
                    origin_index=origin,
                    target_index=target,
                )
        np.testing.assert_array_equal(fusion.window_valid_count, (5, 5, 5))
        np.testing.assert_array_equal(fusion.window_rejected_count, (1, 1, 1))
        # Only the five clear windows enter the aggregate: M1=0 and M2=0.01.
        np.testing.assert_allclose(fusion.M1, (0.0, 0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(fusion.M2, (0.01, 0.01, 0.01), atol=1.0e-12)

    def test_model1_keeps_same_base_through_complete_horizon(self) -> None:
        model = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.1,
        )
        controller = RelativeMPCController(model, MPCConfig(horizon=2))
        z0 = np.concatenate((np.zeros(6), np.array([2.0, 0.0, 0.0])))
        baseline = np.array([2.0, 0.0, 0.0])
        increment = np.array([1.0, 0.0, 0.0])
        commands = np.tile(baseline + increment, 2)
        controller._build_prediction_matrices(np.ones(3))
        predicted = (
            controller.Sx @ z0 + controller.baseline_effect_matrix @ baseline
            + controller.restoring_effect_matrix @ model.restoring_force
            + controller.Su @ commands
        ).reshape(2, 6)
        first = model.B_d @ increment
        second = model.A_d @ first + model.B_d @ increment
        np.testing.assert_allclose(predicted[0], first, atol=1e-12)
        np.testing.assert_allclose(predicted[1], second, atol=1e-12)

    def test_model2_subtracts_tau_h_and_ignores_model1_base(self) -> None:
        restoring = np.array([0.0, 0.0, 0.80729])
        model = FixedLinearDampingRelativeModel(
            np.diag([26.0, 27.0, 26.0]),
            np.diag([94.0, 144.0, 281.0]),
            0.05,
            restoring_force=restoring,
        )
        controller = RelativeMPCController(model, MPCConfig(horizon=3))
        baseline = np.array([11.25, -2.0, 1.9])
        augmented = np.concatenate((np.zeros(6), baseline))
        commands = np.tile(restoring, controller.config.horizon)

        controller._build_prediction_matrices((0.0, 0.0, 0.0))
        predicted = (
            controller.Sx @ augmented
            + controller.baseline_effect_matrix @ baseline
            + controller.restoring_effect_matrix @ restoring
            + controller.Su @ commands
        ).reshape(controller.config.horizon, 6)

        np.testing.assert_allclose(predicted, np.zeros((3, 6)), atol=1e-12)

    def test_diagonal_horizontal_force_hits_joint_thruster_limit(self) -> None:
        from device_adapter import (
            FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
            FINESUB_V4_PRO1_FORCE_POSITIVE_N,
            finesub_translation_thruster_force_matrix,
        )

        matrix = finesub_translation_thruster_force_matrix()
        positive = np.asarray(FINESUB_V4_PRO1_FORCE_POSITIVE_N)
        negative = np.asarray(FINESUB_V4_PRO1_FORCE_NEGATIVE_N)
        single_axis = matrix @ np.array([16.2968314073626, 0.0, 0.0])
        diagonal = matrix @ np.array([16.2968314073626, 16.2968314073626, 0.0])
        self.assertTrue(np.all(single_axis <= positive + 1e-12))
        self.assertTrue(np.all(single_axis >= -negative - 1e-12))
        self.assertTrue(
            np.any((diagonal > positive + 1e-12) | (diagonal < -negative - 1e-12))
        )

    def test_canonical_translation_force_matrix_order_and_signs(self) -> None:
        from device_adapter import finesub_translation_thruster_force_matrix

        matrix = finesub_translation_thruster_force_matrix()
        np.testing.assert_allclose(matrix[:4, :2], np.zeros((4, 2)))
        np.testing.assert_allclose(matrix[:4, 2], np.full(4, 0.25))
        expected_signs = np.array(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
        )
        np.testing.assert_allclose(
            np.sign(matrix[4:, :2]),
            expected_signs,
        )

    def test_force_cost_penalizes_absolute_total_force(self) -> None:
        model = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.1,
        )
        controller = RelativeMPCController(model, MPCConfig(horizon=3))

        free = np.zeros(18)
        reference = np.zeros(3)
        previous = np.array([2.0, -1.0, 0.5])
        equilibrium = np.array([1.5, -0.25, 0.2])
        zero = np.zeros(3)
        P_with_reference, q_with_reference = controller._cost(
            free, reference, previous, equilibrium
        )
        P_without_reference, q_without_reference = controller._cost(
            free, reference, previous, zero
        )
        # The fourth argument is retained only for compatibility with older
        # callers; it no longer shifts the absolute-force penalty.
        np.testing.assert_allclose(P_with_reference, P_without_reference, atol=1e-12)
        np.testing.assert_allclose(q_with_reference, q_without_reference, atol=1e-12)

    def test_pure_model1_zero_error_still_penalizes_nonzero_total_force(self) -> None:
        restoring = np.array([0.0, 0.0, 0.80729])
        model = FixedLinearDampingRelativeModel(
            np.diag([24.82, 26.26, 26.26]),
            np.diag([9.589, 14.96, 11.29]),
            0.1,
            restoring_force=restoring,
        )
        reference = np.array([0.857634, -0.055545, -0.120815])
        controller = RelativeMPCController(
            model,
            MPCConfig(horizon=5, reference_position=reference),
        )
        baseline = np.array([2.0, -0.5, 0.80729])
        state = np.concatenate((reference, np.zeros(3)))

        result = controller.solve(
            state,
            baseline,
            model1_weight=np.ones(3),
        )

        np.testing.assert_allclose(result.force_reference, np.zeros(3), atol=1e-12)
        self.assertLess(np.linalg.norm(result.force), np.linalg.norm(baseline))
        self.assertGreater(np.linalg.norm(result.force - baseline), 1.0e-2)
        # The position remains close to the target, but the optimizer is now
        # allowed to trade a small model-1 velocity error for lower total force.
        self.assertLess(np.max(np.abs(result.predicted_states[:, :3] - reference)), 5.0e-3)

    def test_finesub_forward_mixer_and_motor_order(self) -> None:
        allocator = FineSUBThrusterAllocator(deadband=0.0)
        allocation = allocator.allocate((20.0, 0.0, 0.0))
        expected = np.array([-0.35, 0.0, 0.0, -0.35, 0.35, 0.0, 0.0, 0.35])
        np.testing.assert_allclose(allocation.throttles, expected)

    def test_finesub_attitude_and_depth_use_upper_group(self) -> None:
        allocator = FineSUBThrusterAllocator(deadband=0.0)
        allocation = allocator.allocate(
            (0.0, 0.0, 15.0), attitude_control=(0.1, -0.1, 0.0)
        )
        self.assertTrue(np.allclose(allocation.throttles[[0, 3, 4, 7]], 0.0))
        self.assertTrue(np.any(np.abs(allocation.throttles[[1, 2, 5, 6]]) > 0.0))


if __name__ == "__main__":
    unittest.main()
