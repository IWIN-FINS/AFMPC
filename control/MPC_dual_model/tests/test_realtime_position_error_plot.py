import os

import numpy as np

from MPC_dual_model.realtime_position_error_plot import (
    OverheadTargetVelocitySource,
    extract_position_error_sample,
)


def test_extracts_measured_and_estimated_position_error() -> None:
    record = {
        "event": "control_update",
        "host_monotonic_s": 12.5,
        "position_body_frd_m": [0.90, -0.02, -0.10],
        "estimated_state": [0.88, -0.03, -0.11, 1.0, 2.0, 3.0],
        "model1_weight": [0.9, 0.8, 0.7],
        "reference_position_body_frd_m": None,
    }
    sample = extract_position_error_sample(
        record,
        np.asarray([0.85, -0.05, -0.12]),
    )
    assert sample is not None
    np.testing.assert_allclose(sample.measured_error_m, [0.05, 0.03, 0.02])
    np.testing.assert_allclose(sample.estimated_error_m, [0.03, 0.02, 0.01])
    np.testing.assert_allclose(sample.estimated_velocity_m_s, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sample.model1_weight, [0.9, 0.8, 0.7])
    assert np.isnan(sample.target_absolute_speed_m_s)


def test_uses_per_trace_locked_reference() -> None:
    record = {
        "event": "control_update",
        "host_time_s": 20.0,
        "position_body_frd_m": [1.0, 2.0, 3.0],
        "estimated_state": [1.1, 2.2, 3.3],
        "reference_position_body_frd_m": [1.0, 2.0, 3.0],
    }
    sample = extract_position_error_sample(record, np.zeros(3))
    assert sample is not None
    np.testing.assert_allclose(sample.measured_error_m, np.zeros(3))
    np.testing.assert_allclose(sample.estimated_error_m, [0.1, 0.2, 0.3])


def test_ignores_non_control_events() -> None:
    assert extract_position_error_sample({"event": "start"}, np.zeros(3)) is None


def test_overhead_source_interpolates_target_absolute_speed() -> None:
    source = OverheadTargetVelocitySource(None)
    source._times_s = np.asarray([10.0, 11.0])
    source._velocity_xy_m_s = np.asarray([[0.0, 0.0], [0.3, 0.4]])
    assert np.isclose(source.speed_at(10.5), 0.25)
    assert np.isnan(source.speed_at(9.0))


def test_overhead_source_uses_newest_database_in_directory(tmp_path) -> None:
    older = tmp_path / "older.db3"
    newer = tmp_path / "newer.db3"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    source = OverheadTargetVelocitySource(tmp_path)
    assert source._database_path() == newer
