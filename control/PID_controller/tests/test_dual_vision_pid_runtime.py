from __future__ import annotations

import json

import numpy as np
import pytest

from dual_vision_pid_runtime import (
    OverheadMotionMonitor,
    StereoJsonlTail,
    StereoQualityConfig,
    _parse_stereo_record,
    is_new_stereo_sample,
    retain_initial_arm_request,
    validate_stereo_sample,
)


def _record(frame_idx: int = 1) -> dict:
    return {
        "frame_idx": frame_idx,
        "frame_ts_mean": 100.0 + frame_idx * 0.05,
        "result_time": 100.01 + frame_idx * 0.05,
        "tracks": [
            {
                "position": [-0.1, 0.02, 0.9],
                "confidence": 0.8,
                "depth_valid": True,
                "depth_confidence": 0.7,
                "depth_nis": 1.0,
                "depth_filter_mode": "update",
            }
        ],
    }


def test_stereo_tail_reads_only_completed_json_lines(tmp_path) -> None:
    path = tmp_path / "pipeline_results.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    tail = StereoJsonlTail(path, start_at_end=False)
    records = tail.poll()
    assert records[0]["frame_idx"] == 1
    path.write_text(json.dumps(_record(2)) + "\n", encoding="utf-8")
    assert tail.poll()[0]["frame_idx"] == 2


def test_stereo_quality_rejects_stale_sample() -> None:
    sample, reason = _parse_stereo_record(_record())
    assert reason == "parsed"
    assert sample is not None
    assert validate_stereo_sample(sample, StereoQualityConfig(max_result_age_s=0.1), now_s=100.5) == "stereo_stale"


def test_weak_depth_update_is_usable_when_quality_passes() -> None:
    record = _record()
    record["tracks"][0]["depth_filter_mode"] = "weak_update"
    sample, reason = _parse_stereo_record(record)
    assert reason == "parsed"
    assert sample is not None
    assert sample.depth_filter_mode == "weak_update"
    assert validate_stereo_sample(sample, StereoQualityConfig(), now_s=100.06) is None

    record["tracks"][0]["depth_confidence"] = 0.1
    poor_sample, reason = _parse_stereo_record(record)
    assert reason == "parsed"
    assert poor_sample is not None
    assert validate_stereo_sample(poor_sample, StereoQualityConfig(), now_s=100.06) == "stereo_low_depth_confidence"


def test_predicted_depth_is_not_accepted_as_measurement() -> None:
    record = _record()
    record["tracks"][0]["depth_filter_mode"] = "predict_only"
    sample, reason = _parse_stereo_record(record)
    assert sample is None
    assert reason == "depth_not_updated"


def test_same_camera_frame_can_only_be_consumed_once() -> None:
    sample, _ = _parse_stereo_record(_record())
    assert sample is not None
    assert is_new_stereo_sample(sample, None)
    assert not is_new_stereo_sample(sample, sample.acquisition_time_s)
    assert not is_new_stereo_sample(sample, sample.acquisition_time_s + 0.01)
    assert is_new_stereo_sample(sample, sample.acquisition_time_s - 0.05)


def test_initial_manual_arm_request_waits_for_vision_but_not_safety_fault() -> None:
    assert retain_initial_arm_request(True, False, has_armed=False, safety_latched=False)
    assert retain_initial_arm_request(False, True, has_armed=False, safety_latched=False)
    assert not retain_initial_arm_request(False, True, has_armed=True, safety_latched=False)
    assert not retain_initial_arm_request(True, False, has_armed=False, safety_latched=True)
    assert not retain_initial_arm_request(False, False, has_armed=False, safety_latched=False)


def test_pid_uses_mpc_experiment_quality_limits_without_accepting_bad_depth() -> None:
    record = _record()
    record["tracks"][0]["confidence"] = 0.1
    sample, reason = _parse_stereo_record(record)
    assert reason == "parsed"
    assert sample is not None
    assert validate_stereo_sample(sample, StereoQualityConfig(), now_s=100.06) is None

    record["tracks"][0]["position"][2] = 1.5
    distant, _ = _parse_stereo_record(record)
    assert distant is not None
    assert validate_stereo_sample(distant, StereoQualityConfig(), now_s=100.06) == "stereo_forward_range"

    record["tracks"][0]["position"][2] = 0.9
    record["result_time"] = record["frame_ts_mean"] + 0.16
    delayed, _ = _parse_stereo_record(record)
    assert delayed is not None
    assert validate_stereo_sample(delayed, StereoQualityConfig(), now_s=delayed.result_time_s) == "stereo_pipeline_delay"


def test_overhead_monitor_derives_speed_without_becoming_pid_input() -> None:
    monitor = OverheadMotionMonitor(max_position_jump_m=0.5)
    monitor.update((1.0, 2.0, 3.0), stamp_s=10.0)
    snapshot = monitor.update((1.2, 2.0, 3.0), stamp_s=10.1)
    np.testing.assert_allclose(snapshot.position_xyz, (1.2, 2.0, 3.0))
    np.testing.assert_allclose(snapshot.velocity_xyz, (2.0, 0.0, 0.0))
    assert snapshot.speed_m_s == pytest.approx(2.0)
    assert snapshot.jump is False
