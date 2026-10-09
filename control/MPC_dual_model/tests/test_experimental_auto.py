import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from MPC_dual_model.experimental_auto import (
    MAX_VALIDATED_CHANNEL_ABS,
    build_experimental_runtime_config,
    evaluate_experimental_readiness,
    main as experimental_main,
)
from MPC_dual_model.auto_tracker import build_auto_tracker
from MPC_dual_model.mpc_tracker import MPCTracker
from MPC_dual_model.finesub_transport import load_runtime_config


CONFIG_PATH = Path(__file__).parents[1] / "finesub_v4pro1_mpc.json"


class ExperimentalAutoTests(unittest.TestCase):
    def test_repository_experiment_uses_selected_translation_limit(self) -> None:
        source = load_runtime_config(CONFIG_PATH)
        runtime, report = evaluate_experimental_readiness(
            source, config_path=CONFIG_PATH
        )
        self.assertTrue(report.ready, report.blockers)
        self.assertIsNotNone(runtime)
        self.assertEqual(
            runtime["hardware_adapter"]["translation_channel_limits"],
            [source["experimental_auto"]["max_channel_abs"]] * 3,
        )
        self.assertEqual(runtime["hardware_adapter"]["yaw_channel_limit"], 0.20)
        controller = runtime["auto_runtime"]["active_mpc_parameters"]["controller"]
        self.assertEqual(controller["delta_force_min"], [-1.20, -0.80, -1.00])
        self.assertEqual(controller["delta_force_max"], [1.20, 0.80, 1.00])
        tracker = build_auto_tracker(runtime)
        self.assertIsInstance(tracker, MPCTracker)
        self.assertEqual(runtime["auto_runtime"]["required_model"], "dual")
        self.assertEqual(tracker.model.dt, 0.10)
        self.assertEqual(tracker.controller.config.horizon, 15)
        self.assertEqual(tracker.controller.config.terminal_weight_scale, 2.0)
        self.assertTrue(tracker.baseline_adaptation.enabled)
        self.assertEqual(tracker.baseline_adaptation.update_mode, "gated_ema")
        self.assertEqual(tracker.baseline_adaptation.adaptation_rate, 0.02)
        self.assertEqual(
            tracker.baseline_adaptation.transient_adaptation_rate,
            0.08,
        )
        self.assertEqual(
            tracker.baseline_adaptation.axis_enabled.tolist(),
            [True, True, True],
        )
        self.assertEqual(
            tracker.baseline_adaptation.matched_motion_axis_enabled.tolist(),
            [False, False, False],
        )
        self.assertEqual(
            tracker.baseline_adaptation.matched_motion_confirmation_updates,
            3,
        )
        self.assertEqual(
            tracker.model.restoring_force.tolist(),
            [0.0, 0.0, 0.807290],
        )
        # The configured preset is zero, then normalized to the 0.01 safety
        # floor so fusion can still adapt without an exact model-2 lock.
        self.assertEqual(tracker.fusion.model1_weight.tolist(), [0.01, 0.01, 0.01])
        self.assertEqual(tracker.fusion.config.window, 8)
        self.assertEqual(tracker.fusion.config.prediction_horizon, 5)
        predicted_position, predicted_velocity = tracker.model.predict(
            [0.80, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            tracker.model.restoring_force,
        )
        self.assertAlmostEqual(predicted_position[2], 0.0, places=12)
        self.assertAlmostEqual(predicted_velocity[2], 0.0, places=12)
        self.assertEqual(report.vision_gate_config.min_confidence, 0.0)
        self.assertEqual(
            report.vision_gate_config.accepted_depth_filter_modes,
            ("update", "weak_update"),
        )
        self.assertEqual(report.vision_gate_config.startup_confirmation_samples, 1)
        self.assertEqual(report.vision_gate_config.reacquire_confirmation_samples, 1)
        self.assertFalse(report.vision_gate_config.clamp_implausible_steps)
        self.assertEqual(report.vision_gate_config.max_depth_nis, 25.0)
        self.assertEqual(report.vision_gate_config.jump_margin_m, 0.20)
        self.assertEqual(report.vision_gate_config.max_step_m, 0.30)
        self.assertEqual(report.vision_gate_config.max_inter_sample_gap_s, 1.00)
        configured_forward_range = source["experimental_auto"][
            "vision_gate_overrides"
        ]["forward_range_m"]
        self.assertEqual(
            report.vision_gate_config.min_forward_m,
            configured_forward_range[0],
        )
        self.assertEqual(
            report.vision_gate_config.max_forward_m,
            configured_forward_range[1],
        )
        self.assertFalse(source["auto_runtime"]["enabled"])
        self.assertFalse(
            source["experimental_auto"]["active_mpc_parameters"][
                "enabled_for_control"
            ]
        )
        self.assertFalse(
            source["experimental_auto"]["active_yaw_parameters"][
                "enabled_for_control"
            ]
        )
        self.assertEqual(
            tracker.controller.config.thruster_command_matrix.shape, (3, 3)
        )
        self.assertFalse(
            runtime["auto_runtime"]["active_yaw_parameters"]["enabled_for_control"]
        )

    def test_limit_above_validated_envelope_is_rejected(self) -> None:
        source = load_runtime_config(CONFIG_PATH)
        source = copy.deepcopy(source)
        source["experimental_auto"]["max_channel_abs"] = 0.5001
        runtime, report = evaluate_experimental_readiness(
            source, config_path=CONFIG_PATH
        )
        self.assertIsNone(runtime)
        self.assertFalse(report.ready)
        self.assertTrue(any("0.50" in item for item in report.blockers))

    def test_preflight_never_creates_transport(self) -> None:
        with patch(
            "MPC_dual_model.auto_only_runtime.make_transport",
            side_effect=AssertionError("preflight must not create transport"),
        ):
            self.assertEqual(
                experimental_main(["--config", str(CONFIG_PATH)]),
                0,
            )

    def test_runtime_copy_does_not_authorize_formal_auto(self) -> None:
        source = load_runtime_config(CONFIG_PATH)
        runtime = build_experimental_runtime_config(source)
        self.assertEqual(runtime["auto_runtime"]["mode"], "EXPERIMENTAL_AUTO")
        self.assertEqual(source["auto_runtime"]["mode"], "AUTO_ONLY")
        self.assertFalse(source["auto_runtime"]["enabled"])
        self.assertEqual(
            source["camera_calibration_prior"]
            ["onboard_stereo_udp5600_historical_real_calibration"]
            ["external_red_fish_pipeline_20260812"]
            ["mpc_input_gate"]["min_confidence"],
            0.50,
        )

    def test_rotation_experiment_enables_only_the_in_memory_yaw_wrapper(self) -> None:
        source = load_runtime_config(CONFIG_PATH)
        runtime, report = evaluate_experimental_readiness(
            source, config_path=CONFIG_PATH, rotation=True
        )
        self.assertTrue(report.ready, report.blockers)
        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime["auto_runtime"]["required_model"], "dual-yaw")
        self.assertTrue(
            runtime["auto_runtime"]["active_yaw_parameters"][
                "enabled_for_control"
            ]
        )
        self.assertFalse(source["auto_runtime"]["enabled"])
        self.assertFalse(
            source["experimental_auto"]["active_yaw_parameters"][
                "enabled_for_control"
            ]
        )

    def test_unbounded_zero_runtime_is_forwarded_as_none(self) -> None:
        with patch(
            "MPC_dual_model.experimental_auto.run_auto_only", return_value=0
        ) as run:
            self.assertEqual(
                experimental_main(["--config", str(CONFIG_PATH)]),
                0,
            )
        self.assertIsNone(run.call_args.kwargs["max_runtime_s"])
        self.assertFalse(run.call_args.kwargs["accept_any_vision_track"])
        self.assertTrue(run.call_args.kwargs["hold_armed_on_vision_loss"])
        self.assertFalse(
            run.call_args.kwargs["lock_reference_to_first_measurement"]
        )


if __name__ == "__main__":
    unittest.main()
