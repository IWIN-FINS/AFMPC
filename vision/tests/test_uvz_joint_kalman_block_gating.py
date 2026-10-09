from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from depth_estimation.depth_temporal.uvz_joint_kalman import UVZJointKalmanFilter


def _filter(*, nis_gate_enabled: bool = True,
            hard_gate_enabled: bool = False,
            partial_update_enabled: bool = True) -> UVZJointKalmanFilter:
    return UVZJointKalmanFilter({
        "enabled": True,
        "nis_gate_enabled": nis_gate_enabled,
        "hard_gate_enabled": hard_gate_enabled,
        "partial_update_enabled": partial_update_enabled,
        "fallback_dt_s": 0.1,
        "sigma_a_u_px": 20.0,
        "sigma_a_v_px": 20.0,
        "sigma_a_z_m": 0.5,
        "sigma_u_meas_px": 1.0,
        "sigma_v_meas_px": 1.0,
        "effective_sample_divisor": 100.0,
        "r_z_min_m2": 1.0e-5,
        "r_z_max_m2": 1.0e-3,
        "gate_chi2_df1": 6.63,
        "gate_chi2_df2": 9.21,
    })


def _stats():
    return SimpleNamespace(core_px=600, z_iqr=0.02, valid_ratio=0.2)


def _initialized_filter() -> UVZJointKalmanFilter:
    filt = _filter()
    state = filt.update(
        1.2, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(100.0, 120.0),
    )
    assert state["mode"] == "init_uvz"
    return filt


def test_normal_uvz_measurement_updates_both_blocks():
    filt = _initialized_filter()
    state = filt.update(
        1.21, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(101.0, 120.5),
    )

    assert state["mode"] == "update_uvz"
    assert state["nis"] < filt.cfg.gate_chi2_df1
    assert state["uv_nis"] < filt.cfg.gate_chi2_df2


def test_depth_outlier_does_not_block_valid_center_update():
    filt = _initialized_filter()
    state = filt.update(
        5.0, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(102.0, 121.0),
    )

    assert state["mode"] == "depth_reject_uv_only"
    assert state["center_uv"][0] > 100.0
    assert state["z"] < 2.0
    assert state["nis"] > filt.cfg.gate_chi2_df1


def test_center_outlier_does_not_block_valid_depth_update():
    filt = _initialized_filter()
    state = filt.update(
        1.23, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(1000.0, 900.0),
    )

    assert state["mode"] == "center_reject_z_only"
    np.testing.assert_allclose(state["center_uv"], (100.0, 120.0), atol=1.0)
    assert state["z"] > 1.2
    assert state["uv_nis"] > filt.cfg.gate_chi2_df2


def test_nis_is_diagnostic_only_when_gate_is_disabled():
    filt = _filter(nis_gate_enabled=False)
    filt.update(
        1.2, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(100.0, 120.0),
    )

    state = filt.update(
        5.0, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(1000.0, 900.0),
    )

    assert state["mode"] == "update_uvz"
    assert state["nis"] > filt.cfg.gate_chi2_df1
    assert state["uv_nis"] > filt.cfg.gate_chi2_df2
    assert state["center_uv"][0] > 500.0
    assert state["z"] > 2.0


def test_history_hard_gate_rejects_only_depth_block():
    filt = _filter(nis_gate_enabled=False, hard_gate_enabled=True)
    for index, z in enumerate((1.20, 1.21, 1.22)):
        filt.update(
            z, depth_stats=_stats(), depth_confidence=0.8,
            dt_s=0.1, center_uv=(100.0 + index, 120.0),
        )

    state = filt.update(
        5.0, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(104.0, 121.0),
    )

    assert state["mode"] == "depth_reject_hard_uv_only"
    assert state["center_uv"][0] > 102.0
    assert state["z"] < 2.0
    assert state["rejected"] is True

    recovered = filt.update(
        1.23, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(105.0, 121.5),
    )
    assert recovered["mode"] == "update_uvz"


def test_missing_depth_still_updates_center_without_rejection():
    filt = _filter(nis_gate_enabled=False, hard_gate_enabled=True)
    filt.update(
        1.2, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(100.0, 120.0),
    )

    state = filt.update(
        None, depth_stats=_stats(), depth_confidence=0.0,
        dt_s=0.1, center_uv=(104.0, 121.0),
    )

    assert state["mode"] == "update_uv_only"
    assert state["center_uv"][0] > 100.0
    assert state["z"] == 1.2
    assert state["rejected"] is False


def test_complete_measurement_policy_skips_center_when_depth_is_missing():
    filt = _filter(
        nis_gate_enabled=False,
        hard_gate_enabled=True,
        partial_update_enabled=False,
    )
    filt.update(
        1.2, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(100.0, 120.0),
    )

    state = filt.update(
        None, depth_stats=_stats(), depth_confidence=0.0,
        dt_s=0.1, center_uv=(110.0, 121.0),
    )

    assert state["mode"] == "depth_missing_predict_all"
    np.testing.assert_allclose(state["center_uv"], (100.0, 120.0), atol=1e-9)
    assert state["rejected"] is False


def test_complete_measurement_policy_rolls_back_uv_on_hard_depth_reject():
    partial = _filter(
        nis_gate_enabled=False,
        hard_gate_enabled=True,
        partial_update_enabled=True,
    )
    complete = _filter(
        nis_gate_enabled=False,
        hard_gate_enabled=True,
        partial_update_enabled=False,
    )
    for index, z in enumerate((1.20, 1.21, 1.22)):
        for filt in (partial, complete):
            filt.update(
                z, depth_stats=_stats(), depth_confidence=0.8,
                dt_s=0.1, center_uv=(100.0 + index, 120.0),
            )

    partial_state = partial.update(
        5.0, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(110.0, 121.0),
    )
    complete_state = complete.update(
        5.0, depth_stats=_stats(), depth_confidence=0.8,
        dt_s=0.1, center_uv=(110.0, 121.0),
    )

    assert partial_state["mode"] == "depth_reject_hard_uv_only"
    assert complete_state["mode"] == "depth_reject_hard_predict_all"
    assert partial_state["center_uv"][0] > complete_state["center_uv"][0]
    assert partial_state["z"] == complete_state["z"]
