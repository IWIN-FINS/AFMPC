import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MPC_dual_model.calibration_analysis import (
    analyze_synchronized_csv,
    best_signed_axis_mapping,
    visual_body_rates,
)
from MPC_dual_model.ros_sensor_calibration import CSV_FIELDS


class CalibrationAnalysisTest(unittest.TestCase):
    def test_visual_body_rates_for_positive_yaw(self) -> None:
        times = np.linspace(0.0, 1.0, 21)
        angle = 0.4 * times
        quaternion = np.column_stack(
            (
                np.zeros_like(angle),
                np.zeros_like(angle),
                np.sin(angle / 2.0),
                np.cos(angle / 2.0),
            )
        )
        _, rates = visual_body_rates(times, quaternion)
        np.testing.assert_allclose(np.mean(rates, axis=0), [0.0, 0.0, 0.4], atol=1e-5)

    def test_signed_axis_mapping_recovers_permutation(self) -> None:
        rng = np.random.default_rng(4)
        measured = rng.normal(size=(300, 3))
        reference = np.column_stack((-measured[:, 2], measured[:, 0], -measured[:, 1]))
        result = best_signed_axis_mapping(reference, measured, max_lag_rows=0)
        self.assertIsNotNone(result)
        self.assertEqual(result["permutation_reference_axes_from_measured"], [2, 0, 1])
        self.assertEqual(result["signs"], [-1.0, 1.0, -1.0])
        self.assertGreater(result["score_mean_correlation"], 0.999)

    def test_static_csv_summary_and_safety_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for index in range(80):
                    row = {field: "" for field in CSV_FIELDS}
                    row.update(
                        host_monotonic_s=f"{index * 0.05:.3f}",
                        phase="static_disarmed",
                        requested_armed="0",
                        requested_channels_forward_right_down_yaw="[0.0,0.0,0.0,0.0]",
                        telemetry_fresh="1",
                        telemetry_sequence=str(index),
                        imu_quat_wxyz="[1.0,0.0,0.0,0.0]",
                        imu_angular_velocity_xyz_rad_s="[0.01,-0.02,0.03]",
                        imu_linear_acceleration_xyz_m_s2="[0.0,0.0,9.80665]",
                        depth_m="0.2",
                        pressure_pa="101325",
                        vision_fresh="1",
                        vision_count=str(index),
                        vision_position_xyz_m=json.dumps([1.0, 2.0, 3.0]),
                        vision_quat_xyzw="[0.0,0.0,0.0,1.0]",
                    )
                    writer.writerow(row)
            result = analyze_synchronized_csv(path)
            self.assertEqual(result["safety"]["armed_rows"], 0)
            self.assertEqual(result["availability"]["telemetry_unique_samples"], 80)
            self.assertEqual(result["availability"]["vision_unique_samples"], 80)
            self.assertTrue(result["gate"]["static_bias_estimation_possible"])
            self.assertTrue(result["gate"]["vision_noise_estimation_possible"])

    def test_empty_sensor_streams_produce_unavailable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unavailable.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for index in range(3):
                    row = {field: "" for field in CSV_FIELDS}
                    row.update(
                        host_monotonic_s=str(index * 0.05),
                        phase="readiness_disarmed",
                        requested_armed="0",
                        requested_channels_forward_right_down_yaw="[0.0,0.0,0.0,0.0]",
                        telemetry_fresh="0",
                        vision_fresh="0",
                    )
                    writer.writerow(row)
            result = analyze_synchronized_csv(path)
            self.assertEqual(result["availability"]["telemetry_unique_samples"], 0)
            self.assertEqual(result["availability"]["vision_unique_samples"], 0)
            self.assertFalse(result["gate"]["static_bias_estimation_possible"])

    def test_motion_phase_cannot_be_used_as_static_bias_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for index in range(80):
                    row = {field: "" for field in CSV_FIELDS}
                    row.update(
                        host_monotonic_s=f"{index * 0.01:.3f}",
                        phase="camera_experiment_passive_observation",
                        requested_armed="0",
                        requested_channels_forward_right_down_yaw=(
                            "[0.0,0.0,0.0,0.0]"
                        ),
                        telemetry_fresh="1",
                        telemetry_sequence=str(index),
                        imu_quat_wxyz="[1.0,0.0,0.0,0.0]",
                        imu_angular_velocity_xyz_rad_s="[0.0,0.0,0.0]",
                        imu_linear_acceleration_xyz_m_s2="[0.0,0.0,9.8]",
                        depth_m="0.2",
                        pressure_pa="2000",
                        vision_fresh="0",
                    )
                    writer.writerow(row)
            result = analyze_synchronized_csv(path)
            self.assertFalse(result["gate"]["static_phase_declared"])
            self.assertFalse(result["gate"]["static_bias_estimation_possible"])


if __name__ == "__main__":
    unittest.main()
