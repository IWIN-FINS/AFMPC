import csv
from pathlib import Path
import tempfile
import unittest

from MPC_dual_model.calibration_tool import (
    CSV_FIELDS,
    CalibrationCsvRecorder,
    ChannelStep,
    MAX_CALIBRATION_CHANNEL_ABS,
    MotorStep,
    _parser,
)
from MPC_dual_model.finesub_protocol import FineSUBControlCommand, FineSUBTelemetry
from MPC_dual_model.finesub_transport import DryRunTransport, FineSUBConnection


class CalibrationToolTest(unittest.TestCase):
    def test_channel_step_is_hard_limited(self) -> None:
        self.assertEqual(ChannelStep("forward", 0.10, 1.0, 1).amplitude, 0.10)
        self.assertEqual(
            ChannelStep("forward", 0.10, 3.0, 1, 12.0).resolved_neutral_hold_s,
            12.0,
        )
        self.assertEqual(
            ChannelStep("forward", 0.10, 3.0, 1).resolved_neutral_hold_s,
            3.0,
        )
        with self.assertRaises(ValueError):
            ChannelStep("forward", MAX_CALIBRATION_CHANNEL_ABS + 1e-6, 1.0, 1)
        with self.assertRaises(ValueError):
            ChannelStep("forward", 0.0, 1.0, 1)
        with self.assertRaises(ValueError):
            ChannelStep("forward", 0.10, 1.0, 1, 0.0)
        with self.assertRaises(ValueError):
            ChannelStep("forward", 0.10, 1.0, 1, None, "invalid")

    def test_motor_step_is_hard_limited(self) -> None:
        self.assertEqual(MotorStep(8, -0.10, 5.0, 1).motor_index, 8)
        with self.assertRaises(ValueError):
            MotorStep(0, 0.05, 1.0, 1)
        with self.assertRaises(ValueError):
            MotorStep(1, 0.1001, 1.0, 1)

    def test_csv_contains_commands_and_telemetry_vectors(self) -> None:
        telemetry = FineSUBTelemetry(
            sequence=3,
            tick_ms=10,
            state=1,
            armed=False,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.1,
            yaw_rate_rad_s=0.2,
            depth_m=0.3,
            forward=0.4,
            right=0.5,
            down=0.6,
            yaw=0.7,
            applied_motor_throttle=tuple(float(i) for i in range(8)),
            motor_rpm=tuple(float(10 + i) for i in range(8)),
            quat_wxyz=(0.9, 0.1, 0.2, 0.3),
            angular_velocity_xyz=(0.01, 0.02, 0.03),
            linear_acceleration_xyz=(0.1, 0.2, 9.7),
        )
        connection = FineSUBConnection(DryRunTransport(), logger=None)
        command = FineSUBControlCommand(0.05, 0.0, 0.0, 0.0, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with CalibrationCsvRecorder(path) as recorder:
                recorder.record(
                    phase="test",
                    command=command,
                    sent=True,
                    connection=connection,
                    telemetry=telemetry,
                    telemetry_fresh=True,
                    now=100.0,
                )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), list(CSV_FIELDS))
            self.assertEqual(rows[0]["requested_forward"], "0.0500000")
            self.assertIn("0.0,1.0,2.0", rows[0]["applied_motor_throttle_m1_m8"])
            self.assertIn("10.0,11.0,12.0", rows[0]["motor_rpm_m1_m8"])
            self.assertEqual(rows[0]["imu_quat_wxyz"], "[0.9,0.1,0.2,0.3]")
            self.assertEqual(
                rows[0]["imu_angular_velocity_xyz_rad_s"],
                "[0.01,0.02,0.03]",
            )

    def test_failsafe_parser_requires_explicit_confirmation_flag(self) -> None:
        parsed = _parser().parse_args(
            ["failsafe-zero", "--csv", "result.csv"]
        )
        self.assertFalse(parsed.confirm_propulsion_disconnected)
        self.assertEqual(parsed.armed_zero, 1.0)

    def test_channel_step_parser_accepts_long_neutral_hold(self) -> None:
        parsed = _parser().parse_args(
            [
                "channel-step",
                "--channel",
                "forward",
                "--hold",
                "3",
                "--neutral-hold",
                "12",
                "--direction",
                "positive",
                "--max-depth-excursion",
                "0.25",
                "--csv",
                "result.csv",
            ]
        )
        self.assertEqual(parsed.hold, 3.0)
        self.assertEqual(parsed.neutral_hold, 12.0)
        self.assertEqual(parsed.direction, "positive")
        self.assertEqual(parsed.max_depth_excursion, 0.25)


if __name__ == "__main__":
    unittest.main()
