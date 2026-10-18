"""Camera-frame velocity output follows the paper's pinhole Jacobian."""

from types import SimpleNamespace

import numpy as np

from depth_estimation.demo_video import _serialize_result_payload
from depth_estimation.fish_position import FishPositionEstimator, FishTrack


def _track_for_velocity() -> FishTrack:
    track = FishTrack.__new__(FishTrack)
    track.center_ab_enabled = True
    track.center_ema_initialized = True
    track.center_ab_prior_uv = (330.0, 260.0)
    track.filtered_center_uv = (340.0, 260.0)
    track.center_ab_velocity_uv = (20.0, -10.0)
    track.z_dot = 0.4
    track.camera_params = {"fx": 400.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    return track


def test_camera_velocity_uses_filtered_uv_and_depth_rates():
    track = _track_for_velocity()

    velocity = track.camera_velocity(np.array([0.1, 0.08, 2.0]))

    np.testing.assert_allclose(velocity, [0.12, -0.024, 0.4], atol=1e-12)


def test_camera_velocity_is_unavailable_until_center_rate_is_established():
    track = _track_for_velocity()
    track.center_ab_prior_uv = None
    assert track.camera_velocity(np.array([0.1, 0.08, 2.0])) is None

    track.center_ab_prior_uv = (330.0, 260.0)
    track.camera_params["fx"] = 0.0
    assert track.camera_velocity(np.array([0.1, 0.08, 2.0])) is None


def test_packed_result_and_jsonl_keep_camera_velocity():
    estimator = FishPositionEstimator.__new__(FishPositionEstimator)
    estimator.tracker = SimpleNamespace(state="TRACKING")
    track = _track_for_velocity()
    track.id = 1
    track.bbox = [320.0, 240.0, 360.0, 280.0]
    track.output_bbox = list(track.bbox)
    track.pos_3d = np.array([0.1, 0.08, 2.0], dtype=np.float32)
    track.confidence = 0.9
    track.raw_depth = 2.0
    track.raw_center_uv = (340.0, 260.0)
    track.depth_confidence = 0.8
    track.depth_valid = True
    track.depth_rejected = False
    track.depth_stats = None
    track.source = "yolo"

    result = estimator._pack_results([track])
    np.testing.assert_allclose(result[0]["velocity"], [0.12, -0.024, 0.4])

    sframe = SimpleNamespace(ts_left=1.0, ts_right=1.0, ts_diff_ms=0.0)
    payload = _serialize_result_payload(
        frame_idx=1,
        fish=result,
        process_ms=5.0,
        tracker_state="TRACKING",
        sframe=sframe,
        result_time=1.01,
    )
    np.testing.assert_allclose(payload["tracks"][0]["velocity"], [0.12, -0.024, 0.4])


def test_jsonl_retains_null_when_velocity_is_unavailable():
    sframe = SimpleNamespace(ts_left=1.0, ts_right=1.0, ts_diff_ms=0.0)
    fish = [{"id": 1, "position": [0.0, 0.0, 2.0], "velocity": None, "confidence": 0.9}]

    payload = _serialize_result_payload(
        frame_idx=1,
        fish=fish,
        process_ms=5.0,
        tracker_state="TRACKING",
        sframe=sframe,
        result_time=1.01,
    )

    assert payload["tracks"][0]["velocity"] is None
