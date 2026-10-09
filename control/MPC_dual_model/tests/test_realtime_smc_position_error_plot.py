import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from MPC_dual_model.realtime_position_error_plot import (
    JsonlTraceFollower,
    extract_position_error_sample,
)
from MPC_dual_model.realtime_smc_position_error_plot import (
    LiveSMCPositionErrorPlot,
    load_smc_reference,
)


def test_extracts_smc_diagnostics() -> None:
    record = {
        "event": "control_update",
        "host_monotonic_s": 12.5,
        "host_time_s": 100.0,
        "position_body_frd_m": [0.90, -0.02, -0.10],
        "estimated_state": [0.88, -0.03, -0.11, 0.01, -0.02, 0.03],
        "reference_position_body_frd_m": [0.85, -0.05, -0.12],
        "smc_desired_acceleration_frd_m_s2": [0.1, 0.2, 0.3],
        "smc_sliding_variable": [0.01, 0.02, 0.03],
        "requested_force_frd_n": [1.0, -0.5, 0.25],
        "achieved_force_previous_frd_n": [0.9, -0.4, 0.2],
        "camera_forward_distance_m": 0.60,
        "camera_forward_force_before_guard_n": 1.2,
        "camera_forward_force_after_guard_n": 0.8,
        "vision_measurement_age_s": 0.08,
    }
    sample = extract_position_error_sample(record, np.zeros(3))
    assert sample is not None
    np.testing.assert_allclose(sample.smc_desired_acceleration_m_s2, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(sample.smc_sliding_variable, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(sample.requested_force_frd_n, [1.0, -0.5, 0.25])
    np.testing.assert_allclose(sample.achieved_force_frd_n, [0.9, -0.4, 0.2])
    assert np.isclose(sample.camera_forward_distance_m, 0.60)
    assert np.isclose(sample.camera_forward_force_after_guard_n, 0.8)


def test_loads_reference_from_smc_profile() -> None:
    reference = load_smc_reference("finesub_v4pro1_smc.json")
    np.testing.assert_allclose(reference, [0.857634, -0.055545, -0.120815])


def test_smc_plot_refreshes_jsonl_trace(tmp_path) -> None:
    trace = tmp_path / "smc_test.jsonl"
    records = []
    for index in range(3):
        records.append(
            {
                "event": "control_update",
                "host_monotonic_s": 10.0 + index * 0.1,
                "host_time_s": 100.0 + index * 0.1,
                "position_body_frd_m": [0.86 + 0.01 * index, -0.05, -0.12],
                "estimated_state": [0.86 + 0.01 * index, -0.05, -0.12, 0.01, 0.0, 0.0],
                "reference_position_body_frd_m": [0.86, -0.05, -0.12],
                "requested_force_frd_n": [0.1, 0.0, 0.0],
                "achieved_force_previous_frd_n": [0.1, 0.0, 0.0],
                "smc_desired_acceleration_frd_m_s2": [0.0, 0.0, 0.0],
                "camera_forward_distance_m": 0.60,
                "vision_measurement_age_s": 0.05,
            }
        )
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    follower = JsonlTraceFollower(
        trace,
        tmp_path,
        np.asarray([0.86, -0.05, -0.12]),
        trace_pattern="smc_*.jsonl",
    )
    plot = LiveSMCPositionErrorPlot(follower, window_s=10.0)
    try:
        plot.refresh()
        assert len(plot.samples) == 3
        assert len(plot.measured_lines[0].get_xdata()) == 3
        assert len(plot.measured_lines[0].get_ydata()) == 3
        assert len(plot.estimated_lines[0].get_xdata()) == 3
        assert len(plot.norm_line.get_ydata()) == 3
    finally:
        plot.figure.clf()
