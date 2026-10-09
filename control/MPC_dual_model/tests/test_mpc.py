import unittest

import numpy as np

from device_adapter import FineSUBThrusterAllocator, ForceCommandAdapter
from dense_qp import QPSolution
from fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from model_fusion import FusionConfig, OnlineModelFusion
from live_integration_example import build_tracker as build_live_tracker
from mpc_controller import MPCConfig, RelativeMPCController
from mpc_tracker import (
    BaselineAdaptationConfig,
    DEFAULT_PREDICTION_HORIZON_WEIGHTS,
    DEFAULT_STAIRCASE_HORIZON_CAPS,
    MPCTracker,
)
from relative_kalman import RelativePositionKalmanFilter


class MPCTest(unittest.TestCase):
    def build(self, restoring_force=(0.0, 0.0, 0.0)):
        model = FixedLinearDampingRelativeModel(
            M_t=np.diag([20.0, 25.0, 30.0]),
            D_L=np.diag([8.0, 10.0, 12.0]),
            dt=0.1,
            restoring_force=restoring_force,
        )
        config = MPCConfig(
            horizon=5,
            reference_position=(0.6, 0.0, 0.0),
            force_min=(-12.0, -10.0, -10.0),
            force_max=(12.0, 10.0, 10.0),
            delta_force_min=(-3.0, -2.0, -2.0),
            delta_force_max=(3.0, 2.0, 2.0),
        )
        return model, RelativeMPCController(model, config)

    def test_force_and_rate_limits(self) -> None:
        model, controller = self.build()
        result = controller.solve(
            state=np.array([1.2, 0.15, -0.10, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertFalse(result.used_fallback, result.status)
        self.assertTrue(np.all(result.force <= [3.0, 2.0, 2.0] + np.ones(3) * 1e-9))
        self.assertTrue(np.all(result.force >= [-3.0, -2.0, -2.0] - np.ones(3) * 1e-9))
        self.assertEqual(result.predicted_states.shape, (5, 6))
        self.assertEqual(result.delta_force_sequence.shape, (5, 3))
        self.assertEqual(result.slacks.shape, (5, 6))
        self.assertGreaterEqual(float(np.min(result.slacks)), -2e-5)

    def test_asymmetric_thruster_bounds_apply_to_fallback_and_utilization(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.diag([20.0, 25.0, 30.0]),
            D_L=np.diag([8.0, 10.0, 12.0]),
            dt=0.1,
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=1,
                force_min=(-100.0, -100.0, -100.0),
                force_max=(100.0, 100.0, 100.0),
                delta_force_min=(-100.0, -100.0, -100.0),
                delta_force_max=(100.0, 100.0, 100.0),
                thruster_command_matrix=np.eye(3),
                thruster_command_min=(-1.0, -2.0, -3.0),
                thruster_command_max=(4.0, 5.0, 6.0),
            ),
        )

        safe = controller.safe_force(np.zeros(3), target_force=(10.0, -10.0, 1.0))
        np.testing.assert_allclose(safe, (4.0, -2.0, 1.0))
        self.assertAlmostEqual(
            controller.thruster_utilization((2.0, -1.0, 3.0)),
            0.5,
        )

    def test_high_level_tracker(self) -> None:
        model, controller = self.build()
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
            thruster_allocator=FineSUBThrusterAllocator(
                positive_force_at_limit=(12.0, 10.0, 10.0)
            ),
        )
        output = tracker.update(
            position_body=(1.0, 0.1, -0.05),
            tau_achieved_previous=(0.0, 0.0, 0.0),
        )
        self.assertEqual(output.estimated_state.shape, (6,))
        self.assertLessEqual(abs(output.device_command.planar_forward), 99)
        self.assertLessEqual(abs(output.device_command.planar_right), 99)
        self.assertLessEqual(abs(output.device_command.depth_force), 45)
        self.assertIsNotNone(output.thruster_allocation)
        self.assertEqual(output.thruster_allocation.throttles.shape, (8,))
        self.assertEqual(output.estimated_state.shape, (6,))
        self.assertFalse(hasattr(output.mpc, "yaw_moment"))

    def test_history_reset_preserves_fixed_fossen_force(self) -> None:
        model, controller = self.build((0.0, 0.0, 0.80729))
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
        )
        tracker.update((1.0, 0.1, -0.05), (0.0, 0.0, 0.0))
        tracker.reset_history()
        np.testing.assert_allclose(
            tracker.model.restoring_force, (0.0, 0.0, 0.80729)
        )
        self.assertEqual(tracker._frame_index, -1)
        self.assertEqual(len(tracker._pending_position_predictions), 0)

    def test_model2_estimator_subtracts_fixed_fossen_force(self) -> None:
        restoring = np.array((0.0, 0.0, 0.80729))
        model, controller = self.build(restoring)

        class RecordingEstimator:
            initialized = True

            def __init__(self, relative_model):
                self.model = relative_model
                self.x = np.zeros(6)
                self.prediction = None

            def predict_mean(self, prediction, yaw_delta_rad=0.0):
                self.prediction = np.asarray(prediction, dtype=float)
                self.x = self.prediction.copy()

            def update(self, _position):
                return self.x.copy()

        estimator = RecordingEstimator(model)
        tracker = MPCTracker(
            model=model,
            estimator=estimator,
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
        )
        tracker.fusion.model1_weight.fill(0.0)
        tracker.update(
            position_body=(0.6, 0.0, 0.0),
            tau_achieved_previous=restoring,
        )
        np.testing.assert_allclose(
            estimator.prediction,
            model.A_d @ np.zeros(6),
            atol=1e-12,
        )

    def test_model1_estimator_uses_consecutive_actual_force_difference(self) -> None:
        model, controller = self.build()

        class RecordingEstimator:
            initialized = True

            def __init__(self, relative_model):
                self.model = relative_model
                self.x = np.zeros(6)
                self.prediction = None

            def predict_mean(self, prediction, yaw_delta_rad=0.0):
                self.prediction = np.asarray(prediction, dtype=float)
                self.x = self.prediction.copy()

            def update(self, _position):
                return self.x.copy()

        estimator = RecordingEstimator(model)
        tracker = MPCTracker(
            model=model,
            estimator=estimator,
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
        )
        tracker.latch_baseline((2.0, -1.0, 0.5))
        tracker.fusion.model1_weight.fill(1.0)
        current = np.array((3.0, -0.5, 0.25))
        tracker.update((0.6, 0.0, 0.0), current)
        np.testing.assert_allclose(
            estimator.prediction,
            model.B_d @ (current - np.array((2.0, -1.0, 0.5))),
            atol=1e-12,
        )

    def test_model1_gated_ema_is_separate_from_actual_force_rate_reference(self) -> None:
        model, controller = self.build()
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
            baseline_adaptation=BaselineAdaptationConfig(
                enabled=True,
                adaptation_rate=0.02,
                transient_adaptation_rate=0.08,
                position_error_tolerance=0.20,
                velocity_tolerance=0.20,
            ),
        )
        tracker.latch_baseline((0.0, 0.0, 0.0))
        achieved = np.array((1.0, 2.0, -2.0))
        first = tracker.update((0.6, 0.0, 0.0), achieved)
        np.testing.assert_allclose(
            first.mpc.model1_base_force,
            (0.02, 0.04, -0.04),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            first.mpc.delta_force_sequence[0],
            first.mpc.force - achieved,
            atol=1e-12,
        )
        first_base = first.mpc.model1_base_force.copy()
        state = np.concatenate(
            (
                controller.config.reference_position + (0.04, 0.25, 0.0),
                (0.03, 0.0, 0.09),
            )
        )
        second_base = tracker._adapt_model1_base(
            state,
            achieved,
            controller.config.reference_position,
        )
        np.testing.assert_allclose(
            second_base,
            (
                0.92 * first_base[0] + 0.08 * achieved[0],
                first_base[1],
                0.92 * first_base[2] + 0.08 * achieved[2],
            ),
            atol=1e-12,
        )

    def test_controller_accepts_base_distinct_from_actual_previous_force(self) -> None:
        _, controller = self.build()
        actual = np.array((3.0, -2.0, 1.0))
        baseline = np.array((0.5, -0.25, 0.75))
        result = controller.solve(
            state=np.array((0.6, 0.0, 0.0, 0.0, 0.0, 0.0)),
            tau_previous=actual,
            tau_base=baseline,
            model1_weight=np.ones(3),
        )
        np.testing.assert_allclose(result.model1_base_force, baseline)
        np.testing.assert_allclose(result.force_reference, np.zeros(3))
        np.testing.assert_allclose(
            result.delta_force_sequence[0],
            result.force - actual,
            atol=1e-12,
        )

    def test_forward_matched_motion_gate_requires_persistent_low_acceleration(self) -> None:
        model, controller = self.build()
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
            baseline_adaptation=BaselineAdaptationConfig(
                enabled=True,
                matched_motion_axis_enabled=(True, False, False),
                matched_motion_confirmation_updates=3,
                velocity_tolerance=0.08,
            ),
        )
        tracker.latch_baseline((0.0, 0.0, 0.0))
        moving_state = np.array((0.60, 0.0, 0.0, 0.12, 0.12, 0.0))
        achieved = np.array((1.0, 0.0, 0.0))

        # First sample establishes the velocity history.  The next two only
        # confirm the direction; no baseline learning is allowed yet.
        for _ in range(3):
            base = tracker._adapt_model1_base(
                moving_state,
                achieved,
                controller.config.reference_position,
            )
            np.testing.assert_allclose(base, (0.0, 0.0, 0.0), atol=1e-12)

        # The fourth same-sign, low-acceleration sample enables the transient
        # EMA on forward only.  Right remains on the ordinary static gate.
        base = tracker._adapt_model1_base(
            moving_state,
            achieved,
            controller.config.reference_position,
        )
        np.testing.assert_allclose(base, (0.08, 0.0, 0.0), atol=1e-12)

        # A reversal breaks the streak and cannot immediately rewrite the
        # learned operating point.
        reversed_state = moving_state.copy()
        reversed_state[3] = -0.12
        base = tracker._adapt_model1_base(
            reversed_state,
            achieved,
            controller.config.reference_position,
        )
        np.testing.assert_allclose(base, (0.08, 0.0, 0.0), atol=1e-12)

    def test_gated_ema_never_changes_independent_fossen_tau_h(self) -> None:
        restoring = np.array((0.0, 0.0, 0.80729))
        model, controller = self.build(restoring)
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
            baseline_adaptation=BaselineAdaptationConfig(enabled=True),
        )
        tracker.latch_baseline((0.0, 0.0, 0.0))
        output = tracker.update(
            controller.config.reference_position,
            (1.0, -2.0, 3.0),
        )
        np.testing.assert_allclose(model.restoring_force, restoring)
        np.testing.assert_allclose(
            output.mpc.model1_base_force,
            (0.02, -0.04, 0.06),
            atol=1e-12,
        )

    def test_default_tracker_uses_staircase_translation_history(self) -> None:
        model, controller = self.build()
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
        )
        for _ in range(4):
            tracker.update((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # At k=3, the three-step cell 3|0 is outside cap(age=3)=2.
        self.assertNotIn((0, 3), tracker.fusion.active_pairs)
        self.assertIn((0, 2), tracker.fusion.active_pairs)
        self.assertEqual(
            tracker.fusion.config.staircase_horizon_caps,
            DEFAULT_STAIRCASE_HORIZON_CAPS,
        )
        self.assertEqual(tracker.estimator.x.shape, (6,))

    def test_live_builder_uses_same_six_level_staircase(self) -> None:
        tracker = build_live_tracker()
        self.assertEqual(
            tracker.fusion.config.staircase_horizon_caps,
            DEFAULT_STAIRCASE_HORIZON_CAPS,
        )
        self.assertEqual(
            tracker.fusion.config.prediction_horizon_weights,
            DEFAULT_PREDICTION_HORIZON_WEIGHTS,
        )
        self.assertEqual(tracker.fusion.config.window, 6)
        self.assertEqual(tracker.fusion.config.prediction_horizon, 3)
        self.assertAlmostEqual(tracker.fusion.config.weight_update_rate, 0.35)
        self.assertAlmostEqual(tracker.fusion.config.minimum_weight, 0.01)

    def test_live_builder_respects_v4_pro1_thruster_envelope(self) -> None:
        tracker = build_live_tracker()
        np.testing.assert_allclose(
            tracker.controller.config.force_max,
            (16.2968314073626, 16.2968314073626, 29.5236),
        )
        np.testing.assert_allclose(
            tracker.controller.config.force_min,
            (-16.2968314073626, -16.2968314073626, -23.0472),
        )
        np.testing.assert_allclose(
            tracker.adapter.positive_force_at_limit,
            tracker.controller.config.force_max,
        )
        np.testing.assert_allclose(
            tracker.controller.config.thruster_command_max,
            (8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749),
        )
        np.testing.assert_allclose(
            tracker.controller.config.thruster_command_min,
            -np.array(
                [7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750]
            ),
        )

    def test_live_solver_deadline_fits_control_period(self) -> None:
        tracker = build_live_tracker()
        self.assertLess(
            tracker.controller.config.solver_settings.time_limit_seconds,
            tracker.model.dt,
        )

    def test_live_builder_uses_fixed_restoring_force_and_gated_ema_base(self) -> None:
        tracker = build_live_tracker()
        np.testing.assert_allclose(tracker.model.restoring_force, np.zeros(3))
        self.assertTrue(tracker.baseline_adaptation.enabled)
        self.assertEqual(tracker.baseline_adaptation.update_mode, "gated_ema")
        np.testing.assert_array_equal(
            tracker.baseline_adaptation.axis_enabled,
            (True, True, True),
        )
        np.testing.assert_allclose(
            tracker.controller.config.position_weights,
            (10000.0, 14000.0, 25000.0),
        )
        np.testing.assert_allclose(
            tracker.controller.config.velocity_weights,
            (2.0, 20.0, 12.0),
        )
        np.testing.assert_allclose(
            tracker.controller.config.force_weights,
            (0.003, 0.002, 0.04),
        )
        np.testing.assert_allclose(
            tracker.controller.config.delta_force_weights,
            (0.01, 0.01, 0.5),
        )
        np.testing.assert_allclose(
            np.diag(tracker.model.M_t),
            (26.07276, 26.79684, 26.07276),
        )
        np.testing.assert_allclose(
            np.diag(tracker.model.D_L),
            (93.88006, 143.69195, 280.86849),
        )

        tracker.update(
            position_body=tracker.controller.config.reference_position,
            tau_achieved_previous=(5.0, -2.0, 1.0),
        )
        np.testing.assert_allclose(
            tracker._previous_achieved_force, (5.0, -2.0, 1.0)
        )

    def test_solver_failure_returns_to_fossen_balance_force(self) -> None:
        model, controller = self.build((5.0, 1.0, -1.0))

        class AlwaysFailSolver:
            backend_name = "test_failure"

            def solve(self, P, q, A, lower, upper, warm_start=None):
                return QPSolution(
                    x=np.zeros_like(q),
                    status="time_limit",
                    iterations=0,
                    objective=np.inf,
                    primal_residual=np.inf,
                    dual_residual=np.inf,
                )

        controller.solver = AlwaysFailSolver()
        result = controller.solve(
            state=np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.status, "fallback:restoring_force:time_limit")
        # Fixed Fossen force is rate-limited from zero on this first step.
        np.testing.assert_allclose(result.force, (3.0, 1.0, -1.0))

    def test_solver_failure_holds_last_feasible_force(self) -> None:
        _, controller = self.build()
        solved = controller.solve(
            state=np.array([1.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        last_feasible = solved.force.copy()

        class AlwaysFailSolver:
            backend_name = "test_failure"

            def solve(self, P, q, A, lower, upper, warm_start=None):
                return QPSolution(
                    x=np.zeros_like(q),
                    status="maximum_iterations",
                    iterations=0,
                    objective=np.inf,
                    primal_residual=np.inf,
                    dual_residual=np.inf,
                )

        controller.solver = AlwaysFailSolver()
        failed = controller.solve(
            state=np.array([1.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
            tau_previous=last_feasible,
        )
        self.assertEqual(
            failed.status,
            "fallback:last_feasible:maximum_iterations",
        )
        np.testing.assert_allclose(failed.force, last_feasible)

    def test_tracker_scores_triangular_multi_step_position_history(self) -> None:
        model, controller = self.build()
        fusion = OnlineModelFusion(
            FusionConfig(window=3, prediction_horizon=2)
        )
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
            fusion=fusion,
        )

        tracker.update((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(fusion.sample_count, 0)
        tracker.update((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(fusion.sample_count, 1)  # start i=0, h=1
        tracker.update((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        # start i=0, h=2 and start i=1, h=1 both complete now.
        self.assertEqual(fusion.sample_count, 3)

    def test_prediction_rotates_body_coordinates_over_yaw_horizon(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.eye(3),
            D_L=np.zeros((3, 3)),
            dt=0.1,
        )
        controller = RelativeMPCController(model, MPCConfig(horizon=1))
        yaw_rate = 0.5 * np.pi / model.dt
        controller._build_prediction_matrices(np.ones(3), yaw_rate)
        augmented = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        predicted = controller.Sx @ augmented
        np.testing.assert_allclose(predicted[:3], (0.0, -1.0, 0.0), atol=1e-12)

    def test_tracker_estimator_rotates_across_measured_yaw_delta(self) -> None:
        model, controller = self.build()

        class RecordingEstimator:
            initialized = True

            def __init__(self, relative_model):
                self.model = relative_model
                self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.prediction = None

            def predict_mean(self, prediction, yaw_delta_rad=0.0):
                self.prediction = np.asarray(prediction, dtype=float)
                self.x = self.prediction.copy()

            def update(self, _position):
                return self.x.copy()

        estimator = RecordingEstimator(model)
        tracker = MPCTracker(
            model=model,
            estimator=estimator,
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                command_limits=(99.0, 99.0, 45.0),
            ),
        )
        tracker.fusion.model1_weight.fill(1.0)
        tracker.update(
            position_body=(0.0, -1.0, 0.0),
            tau_achieved_previous=(0.0, 0.0, 0.0),
            yaw_delta_rad=0.5 * np.pi,
        )
        np.testing.assert_allclose(
            estimator.prediction[:3],
            (0.0, -1.0, 0.0),
            atol=1e-12,
        )

    def test_visibility_constraints_use_camera_rotation_and_origin(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.eye(3),
            D_L=np.zeros((3, 3)),
            dt=0.1,
        )
        visibility_from_body = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=1,
                forward_distance_min=0.7,
                rotation_visibility_from_body=visibility_from_body,
                camera_origin_in_body=(0.3, 0.2, 0.0),
            ),
        )
        state = np.array([0.3, 0.8, 0.0, 0.0, 0.0, 0.0])
        augmented = np.concatenate((state, np.zeros(3)))
        free_prediction = controller.Sx @ augmented
        _, _, upper = controller._constraints(free_prediction, np.zeros(3))
        # 3 force + 3 rate + 6 slack rows precede the six visibility rows.
        near_distance_row = 12 + 4
        # Camera ray is [forward,right,down]=[0.6,0,0], so the 0.7 m
        # minimum is violated by 0.1 m. Body x/y alone would not give this.
        self.assertAlmostEqual(upper[near_distance_row], -0.1, places=12)

    def test_actuator_delay_diagnostic_does_not_apply_new_command_immediately(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.eye(3),
            D_L=np.zeros((3, 3)),
            dt=0.1,
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=2,
                actuator_model_enabled=True,
                actuator_pure_delay_s=0.1,
                actuator_time_constant_s=0.2,
            ),
        )
        commands = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        predicted = controller._predict_actuator_force_sequence(
            np.zeros(3), commands
        )
        alpha = np.exp(-0.1 / 0.2)
        np.testing.assert_allclose(predicted[0], np.zeros(3), atol=1.0e-12)
        np.testing.assert_allclose(
            predicted[1], ((1.0 - alpha), 0.0, 0.0), atol=1.0e-12
        )

    def test_actuator_delay_queue_persists_between_solves(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.eye(3),
            D_L=np.zeros((3, 3)),
            dt=0.1,
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=2,
                actuator_model_enabled=True,
                actuator_pure_delay_s=0.1,
                actuator_time_constant_s=0.2,
            ),
        )
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        first = controller.solve(state, np.zeros(3))
        self.assertFalse(first.used_fallback, first.status)
        np.testing.assert_allclose(
            first.predicted_actuator_force_sequence[0],
            np.zeros(3),
            atol=1.0e-12,
        )

        second = controller.solve(state, np.zeros(3))
        self.assertFalse(second.used_fallback, second.status)
        alpha = np.exp(-0.1 / 0.2)
        np.testing.assert_allclose(
            second.predicted_actuator_force_sequence[0],
            (1.0 - alpha) * first.force,
            atol=1.0e-10,
        )

        controller.reset()
        reset = controller.solve(state, np.zeros(3))
        np.testing.assert_allclose(
            reset.predicted_actuator_force_sequence[0],
            np.zeros(3),
            atol=1.0e-12,
        )

    def test_actuator_model_disabled_preserves_command_diagnostic(self) -> None:
        model = FixedLinearDampingRelativeModel(
            M_t=np.eye(3),
            D_L=np.zeros((3, 3)),
            dt=0.1,
        )
        controller = RelativeMPCController(model, MPCConfig(horizon=2))
        commands = np.array([[1.0, -2.0, 0.5], [0.2, 0.3, -0.4]])
        predicted = controller._predict_actuator_force_sequence(
            np.array([9.0, 9.0, 9.0]), commands
        )
        np.testing.assert_allclose(predicted, commands, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
