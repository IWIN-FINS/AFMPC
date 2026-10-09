import math
import unittest

import numpy as np

from MPC_dual_model.auto_only_runtime import _build_runtime_tracker
from MPC_dual_model.auto_readiness import evaluate_auto_readiness
from MPC_dual_model.smc_config import (
    DEFAULT_SMC_PROFILE_PATH,
    load_smc_runtime_config,
)
from MPC_dual_model.smc_controller import (
    AxisSMCConfig,
    FullVehicleSMCController,
    RelativeStateEstimator,
    RelativePositionKalmanFilter,
    SaturatedSMCAxis,
    build_smc_tracker,
)
from MPC_dual_model.fossen_fixed_dl_model import FixedLinearDampingRelativeModel


def _controller() -> FullVehicleSMCController:
    axis = AxisSMCConfig(
        outer_gain=2.0,
        rate_limit=0.4,
        rate_filter_tau=0.1,
        reaching_gain=2.0,
        robust_gain=0.4,
        boundary_layer=0.05,
        delta_input_limit=0.5,
    )
    return FullVehicleSMCController(
        mass_matrix=np.diag([10.0, 11.0, 12.0]),
        linear_damping=np.diag([5.0, 6.0, 7.0]),
        restoring_force=[0.0, 0.0, 0.8],
        yaw_inertia=0.35,
        yaw_linear_damping=0.3,
        translation_config=(axis, axis, axis),
        yaw_config=axis,
        positive_force_limit=[2.0, 2.0, 2.0],
        negative_force_limit=[2.0, 2.0, 2.0],
        positive_yaw_limit=1.0,
        negative_yaw_limit=1.0,
        period_s=0.1,
    )


class SMCControllerTests(unittest.TestCase):
    def test_relative_forward_and_right_errors_have_vehicle_force_sign(self):
        controller = _controller()
        forward = controller.compute(
            position_relative=[1.0, 0.0, 0.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.8, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        controller.reset()
        right = controller.compute(
            position_relative=[0.8, 0.2, 0.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.8, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        # The target-relative coordinate is too far forward/right; the vehicle
        # must move forward/right to reduce that relative distance.
        self.assertGreater(forward.force[0], 0.0)
        self.assertGreater(right.force[1], 0.0)

    def test_restoring_force_is_used_at_depth_equilibrium(self):
        output = _controller().compute(
            position_relative=[0.8, 0.0, 0.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.8, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        np.testing.assert_allclose(output.force, [0.0, 0.0, 0.8])

    def test_relative_velocity_damping_matches_relative_mpc_model(self):
        axis = AxisSMCConfig(
            outer_gain=1.0,
            rate_limit=1.0,
            rate_filter_tau=0.1,
            reaching_gain=1.0e-9,
            robust_gain=1.0e-9,
            boundary_layer=1.0,
            delta_input_limit=100.0,
        )
        controller = FullVehicleSMCController(
            mass_matrix=np.diag([10.0, 10.0, 10.0]),
            linear_damping=np.diag([5.0, 5.0, 5.0]),
            restoring_force=[0.0, 0.0, 0.0],
            yaw_inertia=1.0,
            yaw_linear_damping=0.0,
            translation_config=(axis, axis, axis),
            yaw_config=axis,
            positive_force_limit=[100.0, 100.0, 100.0],
            negative_force_limit=[100.0, 100.0, 100.0],
            positive_yaw_limit=100.0,
            negative_yaw_limit=100.0,
            period_s=0.1,
        )
        positive_relative_velocity = controller.compute(
            position_relative=[0.8, 0.0, 0.0],
            velocity_relative=[0.1, 0.0, 0.0],
            position_reference=[0.8, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        controller.reset()
        negative_relative_velocity = controller.compute(
            position_relative=[0.8, 0.0, 0.0],
            velocity_relative=[-0.1, 0.0, 0.0],
            position_reference=[0.8, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        # The active MPC relative model is
        #   v_rel_dot = -M^-1 D v_rel - M^-1 (tau - tau_h).
        # With zero requested relative acceleration, the inverse dynamics
        # therefore applies -D*v_rel, not +D*v_rel.
        self.assertLess(positive_relative_velocity.force[0], 0.0)
        self.assertGreater(negative_relative_velocity.force[0], 0.0)

    def test_force_and_yaw_limits_and_slew_are_hard(self):
        controller = _controller()
        first = controller.compute(
            position_relative=[-100.0, -100.0, -100.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.0, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=-100.0,
        )
        second = controller.compute(
            position_relative=[100.0, 100.0, 100.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.0, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=100.0,
        )
        self.assertTrue(np.all(first.force <= [2.0, 2.0, 2.0]))
        self.assertTrue(np.all(first.force >= [-2.0, -2.0, -2.0]))
        self.assertLessEqual(abs(first.yaw_moment), 1.0)
        self.assertTrue(np.all(np.abs(second.force - first.force) <= 0.5 + 1e-12))
        self.assertLessEqual(abs(second.yaw_moment - first.yaw_moment), 0.5 + 1e-12)

    def test_force_reduction_uses_the_faster_brake_slew_limit(self):
        axis = AxisSMCConfig(
            outer_gain=1.0,
            rate_limit=1.0,
            rate_filter_tau=0.1,
            reaching_gain=1.0,
            robust_gain=1.0e-9,
            boundary_layer=0.1,
            delta_input_limit=0.1,
            brake_input_limit=0.8,
        )
        controller = FullVehicleSMCController(
            mass_matrix=np.diag([10.0, 10.0, 10.0]),
            linear_damping=np.zeros((3, 3)),
            restoring_force=[0.0, 0.0, 0.0],
            yaw_inertia=1.0,
            yaw_linear_damping=0.0,
            translation_config=(axis, axis, axis),
            yaw_config=axis,
            positive_force_limit=[10.0, 10.0, 10.0],
            negative_force_limit=[10.0, 10.0, 10.0],
            positive_yaw_limit=10.0,
            negative_yaw_limit=10.0,
            period_s=0.1,
        )
        controller.reset(previous_force=[1.0, 0.0, 0.0])
        output = controller.compute(
            position_relative=[-100.0, 0.0, 0.0],
            velocity_relative=[0.0, 0.0, 0.0],
            position_reference=[0.0, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        self.assertAlmostEqual(output.force[0], 0.2, places=12)

    def test_yaw_error_wraps_across_pi(self):
        axis = SaturatedSMCAxis(
            AxisSMCConfig(2.0, 1.0, 0.1, 1.0, 0.1, 0.05, 1.0),
            angular_position=True,
        )
        output = axis.compute(math.pi - 0.01, 0.0, -math.pi + 0.01, 0.1)
        self.assertAlmostEqual(output.position_error, 0.02, places=12)

    def test_estimator_exposes_six_state_covariance_and_resets(self):
        estimator = RelativeStateEstimator()
        estimator.update([1.0, 0.0, 0.0], 0.0, timestamp_s=1.0)
        estimator.update([0.99, 0.0, 0.0], 0.0, timestamp_s=1.1)
        self.assertEqual(estimator.P.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(estimator.P)))
        estimator.reset()
        self.assertEqual(estimator.P.shape, (6, 6))

    def test_kalman_predict_ahead_uses_fractional_measurement_delay(self):
        model = FixedLinearDampingRelativeModel(
            M_t=np.diag([10.0, 10.0, 10.0]),
            D_L=np.zeros((3, 3)),
            dt=0.1,
            restoring_force=[0.0, 0.0, 0.0],
        )
        estimator = RelativePositionKalmanFilter(model)
        estimator.initialize([0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0])
        state = estimator.predict_ahead(
            [0.0, 0.0, 0.0],
            duration_s=0.05,
        )
        self.assertAlmostEqual(state[0], 0.05, places=12)
        self.assertAlmostEqual(state[3], 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(estimator.P)))

    def test_target_loss_returns_fossen_balance(self):
        tracker = build_smc_tracker(
            load_smc_runtime_config(DEFAULT_SMC_PROFILE_PATH, experimental=True)
        )
        output = tracker.target_lost([1.0, 1.0, 1.0], 0.5)
        np.testing.assert_allclose(output.force, tracker.controller.restoring_force)
        self.assertEqual(output.yaw_moment, tracker.controller.yaw_moment_base)

    def test_forward_guard_uses_mpc_reference_and_rejects_approach_force(self):
        tracker = build_smc_tracker(
            load_smc_runtime_config(DEFAULT_SMC_PROFILE_PATH, experimental=True)
        )
        self.assertAlmostEqual(
            tracker.forward_distance_guard.hard_minimum_m, 0.30, places=12
        )
        # Construct target-relative body positions from camera coordinates so
        # the test exercises the calibrated camera axis, not an axis-aligned
        # approximation.
        position_body = (
            tracker.camera_origin_in_body
            + tracker.rotation_body_from_camera[:, 2] * 0.58
        )
        output = tracker.update(
            position_body=position_body,
            force_achieved_previous=[0.0, 0.0, 0.8],
            yaw_moment_achieved_previous=0.0,
            reference_position=tracker.reference_position,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        self.assertAlmostEqual(tracker.last_forward_distance_m, 0.58, places=6)
        self.assertLessEqual(
            float(np.dot(tracker.camera_forward_body, output.mpc.force)),
            1.0e-10,
        )
        # The post-guard force, not the stale positive SMC force, is the
        # starting point for the next slew-limited update.
        self.assertAlmostEqual(
            float(
                np.dot(
                    tracker.camera_forward_body,
                    tracker.controller.last_force,
                )
            ),
            float(np.dot(tracker.camera_forward_body, output.mpc.force)),
            places=10,
        )

    def test_tracker_uses_achieved_wrench_as_slew_reference(self):
        tracker = build_smc_tracker(
            load_smc_runtime_config(DEFAULT_SMC_PROFILE_PATH, experimental=True)
        )
        achieved = np.asarray([0.8, -0.7, 0.9])
        output = tracker.update(
            position_body=[1.0, 0.0, 0.0],
            force_achieved_previous=achieved,
            yaw_moment_achieved_previous=0.12,
            reference_position=tracker.reference_position,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        np.testing.assert_allclose(
            output.mpc.delta_force_sequence[0],
            output.mpc.force - achieved,
        )

    def test_forward_guard_requires_the_mpc_reference_to_equal_060_m(self):
        runtime = load_smc_runtime_config(
            DEFAULT_SMC_PROFILE_PATH,
            experimental=True,
        )
        runtime["smc_parameters"]["reference_position"] = [0.90, 0.0, -0.12]
        with self.assertRaisesRegex(ValueError, "must match the active MPC"):
            build_smc_tracker(runtime)

    def test_forward_guard_target_cannot_diverge_from_mpc_reference_projection(self):
        runtime = load_smc_runtime_config(
            DEFAULT_SMC_PROFILE_PATH,
            experimental=True,
        )
        runtime["smc_parameters"]["forward_distance_guard"][
            "target_distance_m"
        ] = 0.65
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_smc_tracker(runtime)

    def test_profile_selects_full_smc_in_experiment_and_skips_osqp(self):
        runtime = load_smc_runtime_config(DEFAULT_SMC_PROFILE_PATH, experimental=True)
        report = evaluate_auto_readiness(
            runtime,
            config_path=DEFAULT_SMC_PROFILE_PATH,
            selected_model="smc-full",
            expected_mode="EXPERIMENTAL_AUTO",
            allow_unaccepted_calibration_candidates=True,
            allow_zero_detector_confidence=True,
            maximum_authorized_channel_abs=0.20,
            expected_model_sample_period_s=0.10,
            minimum_startup_confirmation_samples=1,
            minimum_reacquire_confirmation_samples=1,
            maximum_depth_nis=100.0,
            minimum_forward_m=0.10,
            maximum_forward_m=2.80,
            maximum_jump_margin_m=0.20,
        )
        self.assertTrue(report.ready, report.blockers)
        tracker, yaw_direct = _build_runtime_tracker(runtime)
        self.assertFalse(yaw_direct)
        self.assertEqual(tracker.kind, "smc-full")
        self.assertIsInstance(tracker.estimator, RelativePositionKalmanFilter)
        self.assertAlmostEqual(tracker.model.dt, 0.10, places=12)
        output = tracker.update(
            position_body=[1.0, 0.0, 0.0],
            force_achieved_previous=[0.0, 0.0, 0.8],
            yaw_moment_achieved_previous=0.0,
            reference_position=[0.857634, -0.055545, -0.120815],
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
        )
        self.assertEqual(output.mpc.status, "smc")
        self.assertFalse(output.mpc.used_fallback)
        self.assertEqual(output.mpc.yaw_moment, 0.0)
        self.assertEqual(output.mpc.predicted_states.shape, (1, 6))

    def test_formal_smc_profile_remains_blocked(self):
        runtime = load_smc_runtime_config(DEFAULT_SMC_PROFILE_PATH)
        report = evaluate_auto_readiness(
            runtime,
            config_path=DEFAULT_SMC_PROFILE_PATH,
            selected_model="smc-full",
        )
        self.assertFalse(report.ready)
        self.assertTrue(any("active_mpc_parameters" in item for item in report.blockers))


if __name__ == "__main__":
    unittest.main()
