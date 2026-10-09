import unittest

import numpy as np

from MPC_dual_model.finesub_protocol import (
    FineSUBTelemetry,
    pack_command,
    unpack_command,
)
from MPC_model1 import live_integration_example as model1_live
from MPC_model2 import live_integration_example as model2_live
from MPC_dual_model import live_integration_example as dual_live
from MPC_dual_model_yaw import live_integration_example as dual_yaw_live
from rov_track_control3 import LiveModelBackend


class AllHardwareAdaptersTest(unittest.TestCase):
    def test_all_four_models_emit_finesub_v4_commands(self) -> None:
        cases = (
            (model1_live, False),
            (model2_live, False),
            (dual_live, False),
            (dual_yaw_live, True),
        )
        for sequence, (module, yaw_direct) in enumerate(cases):
            with self.subTest(module=module.__package__):
                tracker = module.build_tracker()
                adapter = module.build_hardware_adapter()
                if yaw_direct:
                    output = module.one_control_update(
                        tracker=tracker,
                        position_camera_xyz=(0.0, 0.0, 1.0),
                        imu_yaw_rad=0.0,
                        imu_yaw_rate_rad_s=0.0,
                        last_achieved_force_body=np.zeros(3),
                        last_achieved_yaw_moment=0.0,
                    )
                else:
                    output = module.one_control_update(
                        tracker,
                        (0.0, 0.0, 1.0),
                        np.zeros(3),
                    )
                command = module.to_finesub_command(
                    output,
                    armed=True,
                    adapter=adapter,
                )
                decoded, decoded_sequence = unpack_command(
                    pack_command(command, sequence)
                )
                self.assertEqual(decoded_sequence, sequence)
                self.assertTrue(decoded.armed)
                self.assertEqual(decoded.yaw_direct, yaw_direct)
                self.assertLessEqual(abs(decoded.forward), 0.35 + 1.0e-6)
                self.assertLessEqual(abs(decoded.right), 0.35 + 1.0e-6)
                self.assertLessEqual(abs(decoded.down), 0.50 + 1.0e-6)
                self.assertLessEqual(abs(decoded.yaw), 0.20 + 1.0e-6)

    def test_live_backend_selects_each_model_and_matching_yaw_source(self) -> None:
        adapter = dual_live.build_hardware_adapter()
        for name, yaw_direct in (
            ("model1", False),
            ("model2", False),
            ("dual", False),
            ("dual-yaw", True),
        ):
            with self.subTest(model=name):
                backend = LiveModelBackend(name)
                backend.latch_baseline(np.zeros(3), 0.0, 0.0)
                command, detail = backend.update(
                    position_camera=np.array([0.0, 0.0, 1.0]),
                    force=np.zeros(3),
                    yaw_moment=0.0,
                    yaw_rad=0.0,
                    yaw_rate_rad_s=0.0,
                    hardware_adapter=adapter,
                )
                self.assertTrue(detail)
                self.assertTrue(command.armed)
                self.assertEqual(command.yaw_direct, yaw_direct)
                self.assertTrue(
                    np.all(
                        np.isfinite(
                            [
                                command.forward,
                                command.right,
                                command.down,
                                command.yaw,
                            ]
                        )
                    )
                )
                safe_command = backend.target_lost(
                    np.zeros(3),
                    0.0,
                    adapter,
                )
                self.assertTrue(safe_command.armed)
                self.assertEqual(safe_command.yaw_direct, yaw_direct)
                expected_safe = (
                    adapter.convert(
                        np.array([0.0, 0.0, 0.80729]),
                        0.0,
                        armed=True,
                        yaw_direct=True,
                    )
                    if name == "dual-yaw"
                    else adapter.convert(
                        np.zeros(3),
                        0.0,
                        armed=True,
                        yaw_direct=yaw_direct,
                    )
                )
                np.testing.assert_allclose(
                    [
                        safe_command.forward,
                        safe_command.right,
                        safe_command.down,
                        safe_command.yaw,
                    ],
                    [
                        expected_safe.forward,
                        expected_safe.right,
                        expected_safe.down,
                        expected_safe.yaw,
                    ],
                    atol=1.0e-6,
                )

    def test_all_backends_use_applied_motor_execution_feedback(self) -> None:
        adapter = dual_live.build_hardware_adapter()
        # Pure forward channel through the exact physical M1..M8 output order.
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
            applied_motor_throttle=(-0.1, -0.1, 0.0, 0.0, 0.0, -0.1, 0.1, 0.0),
            execution_feedback_valid=True,
        )
        for name in ("model1", "model2", "dual", "dual-yaw"):
            with self.subTest(model=name):
                backend = LiveModelBackend(name)
                force, moment = backend.execution_feedback(telemetry, adapter)
                np.testing.assert_allclose(force, (5.714285714, 0.0, 0.0))
                self.assertAlmostEqual(moment, 0.0)


if __name__ == "__main__":
    unittest.main()
