"""Validate forward model-fusion identifiability against overhead-camera motion.

This is an offline-only tool.  It replays recorded onboard position/force
measurements without constructing the MPC QP or opening a hardware transport.
Overhead AprilTag motion supplies independent static and matched-cruise labels:
tag 17 is the hand-held target and tags 15/16 are combined into the vehicle
centre.  The output compares the maintained short staircase with genuinely
completed five- and eight-step prediction histories.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import numpy as np

from .auto_tracker import build_auto_tracker
from .camera_transform import rotation_state_body_from_previous
from .experimental_auto import build_experimental_runtime_config
from .finesub_transport import load_runtime_config
from .model_fusion import FusionConfig, OnlineModelFusion


@dataclass(frozen=True)
class Pair:
    trace: Path
    bag: Path


@dataclass
class PendingPrediction:
    origin_index: int
    state1: np.ndarray
    state2: np.ndarray
    tau_base: np.ndarray
    steps: int = 0


class DefaultOnAmbiguityFusion(OnlineModelFusion):
    """Offline candidate that decays to model 2 when evidence is too small."""

    def __init__(self, config: FusionConfig, default_model1_weight: float) -> None:
        self.default_model1_weight = float(default_model1_weight)
        super().__init__(config)

    def _set_scores(
        self,
        errors1: np.ndarray,
        errors2: np.ndarray,
        weights: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size == 0 or np.sum(weights) <= 0.0:
            return self.model1_weight.copy()
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != errors1.shape:
                raise ValueError("valid_mask must match fusion error shape")
        weights = weights / np.sum(weights)
        self.M1 = np.zeros(3)
        self.M2 = np.zeros(3)
        self.C12 = np.zeros(3)
        for axis in range(3):
            mask = (
                np.ones(errors1.shape[0], dtype=bool)
                if valid_mask is None
                else valid_mask[:, axis]
            )
            if not np.any(mask):
                continue
            axis_weights = weights[mask]
            axis_weights /= np.sum(axis_weights)
            self.M1[axis] = np.sum(axis_weights * errors1[mask, axis] ** 2)
            self.M2[axis] = np.sum(axis_weights * errors2[mask, axis] ** 2)
            self.C12[axis] = np.sum(
                axis_weights * errors1[mask, axis] * errors2[mask, axis]
            )
        denominator = np.maximum(self.M1 + self.M2 - 2.0 * self.C12, 0.0)
        raw = (self.M2 - self.C12 + self.config.epsilon) / (
            denominator + 2.0 * self.config.epsilon
        )
        threshold = np.asarray(
            self.config.indistinguishable_score_threshold, dtype=float
        )
        default = np.full(3, self.default_model1_weight, dtype=float)
        raw = np.where(denominator <= threshold, default, raw)
        lower = self.config.minimum_weight
        raw = np.clip(raw, lower, 1.0 - lower)
        rate = self.config.weight_update_rate
        self.model1_weight = np.clip(
            (1.0 - rate) * self.model1_weight + rate * raw,
            lower,
            1.0 - lower,
        )
        return self.model1_weight.copy()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _decode_std_msgs_string(data: bytes) -> str:
    """Decode the CDR representation used by std_msgs/msg/String."""
    if len(data) < 9:
        raise ValueError("short CDR string")
    little_endian = data[1] == 1
    length = int.from_bytes(data[4:8], "little" if little_endian else "big")
    payload = data[8 : 8 + max(0, length - 1)]
    return payload.decode("utf-8")


def _circular_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(math.atan2(np.mean(np.sin(array)), np.mean(np.cos(array))))


def _load_overhead(
    path: Path,
    *,
    read_only_snapshot: bool = False,
) -> dict[str, np.ndarray]:
    """Load the overhead-camera status stream from a ROS2 SQLite bag.

    The live error plot may read a bag while ``ros2 bag record`` is writing it.
    In that case use an immutable read-only connection so the reader never
    takes a SQLite lock that can terminate the recorder.  A partially written
    snapshot is allowed to fail and will be retried by the live source.
    """
    if read_only_snapshot:
        uri = f"file:{Path(path).resolve()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=0.0)
    else:
        connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT id FROM topics WHERE name='/finsrov/vision/status'"
        ).fetchone()
        if row is None:
            raise ValueError(f"missing /finsrov/vision/status in {path}")
        topic_id = int(row[0])
        raw: list[tuple[float, dict[int, tuple[np.ndarray, float]]]] = []
        for timestamp_ns, data in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,),
        ):
            try:
                message = json.loads(_decode_std_msgs_string(data))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                continue
            detections: dict[int, tuple[np.ndarray, float]] = {}
            for detection in message.get("detections_detail", ()):  # type: ignore[union-attr]
                try:
                    tag_id = int(detection["tag_id"])
                    pose = np.asarray(detection["world_xy_yaw"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    continue
                if pose.shape == (3,) and np.all(np.isfinite(pose)):
                    detections[tag_id] = (pose[:2], float(pose[2]))
            if 17 in detections and (15 in detections or 16 in detections):
                raw.append((float(timestamp_ns) * 1.0e-9, detections))
    finally:
        connection.close()
    if len(raw) < 20:
        raise ValueError(f"too few simultaneous target/vehicle detections in {path}")

    tag_offsets = [
        detections[16][0] - detections[15][0]
        for _, detections in raw
        if 15 in detections and 16 in detections
    ]
    offset_15_to_16 = (
        np.median(np.asarray(tag_offsets), axis=0)
        if tag_offsets
        else np.zeros(2)
    )
    times: list[float] = []
    target: list[np.ndarray] = []
    vehicle: list[np.ndarray] = []
    vehicle_yaw: list[float] = []
    for timestamp, detections in raw:
        if 15 in detections and 16 in detections:
            centre = 0.5 * (detections[15][0] + detections[16][0])
            yaw = _circular_mean((detections[15][1], detections[16][1]))
        elif 15 in detections:
            centre = detections[15][0] + 0.5 * offset_15_to_16
            yaw = detections[15][1]
        else:
            centre = detections[16][0] - 0.5 * offset_15_to_16
            yaw = detections[16][1]
        times.append(timestamp)
        target.append(detections[17][0])
        vehicle.append(centre)
        vehicle_yaw.append(yaw)
    time_array = np.asarray(times)
    return {
        "time": time_array,
        "target": np.asarray(target),
        "vehicle": np.asarray(vehicle),
        "vehicle_yaw": np.unwrap(np.asarray(vehicle_yaw)),
        "target_velocity": _local_linear_velocity(time_array, np.asarray(target)),
        "vehicle_velocity": _local_linear_velocity(time_array, np.asarray(vehicle)),
    }


def _local_linear_velocity(
    times: np.ndarray,
    positions: np.ndarray,
    half_window_s: float = 0.5,
) -> np.ndarray:
    velocity = np.full_like(positions, np.nan, dtype=float)
    left = 0
    right = 0
    for index, centre in enumerate(times):
        while left < len(times) and times[left] < centre - half_window_s:
            left += 1
        while right < len(times) and times[right] <= centre + half_window_s:
            right += 1
        if right - left < 5:
            continue
        local_time = times[left:right] - centre
        denominator = float(local_time @ local_time)
        if denominator > 1.0e-9:
            centred_position = positions[left:right] - np.mean(
                positions[left:right], axis=0
            )
            velocity[index] = local_time @ centred_position / denominator
    return velocity


def _interp_columns(
    target_times: np.ndarray,
    source_times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_times, source_times, values[:, axis]) for axis in range(values.shape[1])]
    )


def _overhead_labels(
    control_times: np.ndarray,
    overhead: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target_velocity = _interp_columns(
        control_times, overhead["time"], overhead["target_velocity"]
    )
    vehicle_velocity = _interp_columns(
        control_times, overhead["time"], overhead["vehicle_velocity"]
    )
    target_acceleration_source = _local_linear_velocity(
        overhead["time"], overhead["target_velocity"], half_window_s=0.75
    )
    vehicle_acceleration_source = _local_linear_velocity(
        overhead["time"], overhead["vehicle_velocity"], half_window_s=0.75
    )
    target_acceleration = _interp_columns(
        control_times, overhead["time"], target_acceleration_source
    )
    vehicle_acceleration = _interp_columns(
        control_times, overhead["time"], vehicle_acceleration_source
    )
    target_speed = np.linalg.norm(target_velocity, axis=1)
    vehicle_speed = np.linalg.norm(vehicle_velocity, axis=1)
    denominator = np.maximum(target_speed * vehicle_speed, 1.0e-9)
    cosine = np.sum(target_velocity * vehicle_velocity, axis=1) / denominator
    valid = (
        (control_times >= overhead["time"][0])
        & (control_times <= overhead["time"][-1])
        & np.all(np.isfinite(target_velocity), axis=1)
        & np.all(np.isfinite(vehicle_velocity), axis=1)
    )
    labels = np.full(len(control_times), "other", dtype=object)
    labels[
        valid & (target_speed < 0.015) & (vehicle_speed < 0.015)
    ] = "static"
    labels[
        valid
        & (target_speed > 0.040)
        & (vehicle_speed > 0.025)
        & (cosine > 0.70)
        & (np.abs(target_speed - vehicle_speed) < 0.050)
    ] = "cruise"
    steady_cruise = (
        (labels == "cruise")
        & (np.linalg.norm(target_acceleration, axis=1) < 0.040)
        & (np.linalg.norm(vehicle_acceleration, axis=1) < 0.040)
    )
    labels[steady_cruise] = "steady_cruise"
    return labels, {
        "target_speed": target_speed,
        "vehicle_speed": vehicle_speed,
        "direction_cosine": cosine,
        "target_acceleration": np.linalg.norm(target_acceleration, axis=1),
        "vehicle_acceleration": np.linalg.norm(vehicle_acceleration, axis=1),
    }


def _variant_config(base: dict[str, Any], horizon: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    if horizon == 2:
        return config
    if horizon == 5:
        config.update(
            window=8,
            prediction_horizon=5,
            prediction_horizon_weights=[1.0] * 5,
            staircase_horizon_caps=[5, 5, 5, 5, 5, 4, 3, 1],
        )
    elif horizon == 8:
        config.update(
            window=12,
            prediction_horizon=8,
            prediction_horizon_weights=[1.0] * 8,
            staircase_horizon_caps=[8, 8, 8, 8, 8, 8, 8, 8, 6, 4, 2, 1],
        )
    else:
        raise ValueError(f"unsupported horizon {horizon}")
    return config


def _make_fusion(
    config_data: dict[str, Any],
    ambiguity_threshold: float | None,
) -> OnlineModelFusion:
    candidate = copy.deepcopy(config_data)
    if ambiguity_threshold is not None:
        thresholds = np.asarray(
            candidate["indistinguishable_score_threshold"], dtype=float
        )
        thresholds[0] = ambiguity_threshold
        candidate["indistinguishable_score_threshold"] = thresholds.tolist()
    config = FusionConfig(**candidate)
    if ambiguity_threshold is None:
        return OnlineModelFusion(config)
    return DefaultOnAmbiguityFusion(config, default_model1_weight=0.01)


def _replay(
    records: list[dict[str, Any]],
    tracker_template: Any,
    fusion_data: dict[str, Any],
    ambiguity_threshold: float | None,
) -> dict[str, np.ndarray]:
    model = tracker_template.model
    estimator_type = type(tracker_template.estimator)
    estimator_config = copy.deepcopy(tracker_template.estimator.config)
    estimator = estimator_type(model, copy.deepcopy(estimator_config))
    fusion = _make_fusion(fusion_data, ambiguity_threshold)
    pending: deque[PendingPrediction] = deque()
    frame_index = -1
    previous_base: np.ndarray | None = None
    times: list[float] = []
    weights: list[float] = []
    separations: list[float] = []
    ambiguous: list[bool] = []
    model1_mse: list[float] = []
    model2_mse: list[float] = []
    raw_optimal_weight: list[float] = []
    base_steps: list[float] = []
    base_lags: list[float] = []
    last_logged_base: np.ndarray | None = None

    for record in records:
        if record.get("event") == "vision_hold_started":
            estimator = estimator_type(model, copy.deepcopy(estimator_config))
            fusion.reset()
            pending.clear()
            frame_index = -1
            previous_base = None
            last_logged_base = None
            continue
        if record.get("event") != "control_update":
            continue
        position = np.asarray(record["position_body_frd_m"], dtype=float)
        achieved = np.asarray(
            record["achieved_force_previous_frd_n"], dtype=float
        )
        tau_base = np.asarray(
            record.get("model1_fixed_base_force_frd_n", achieved), dtype=float
        )
        yaw_delta = float(record.get("yaw_delta_body_frd_rad", 0.0))
        rotation = rotation_state_body_from_previous(yaw_delta)
        frame_index += 1
        if not estimator.initialized:
            state = estimator.initialize(position)
            fusion.advance_time(frame_index)
        else:
            old_state = estimator.x.copy()
            base_for_interval = achieved if previous_base is None else previous_base
            prediction1 = rotation @ (
                model.A_d @ old_state
                + model.B_d @ (achieved - base_for_interval)
            )
            prediction2 = rotation @ (
                model.A_d @ old_state
                + model.B_d @ (achieved - model.restoring_force)
            )
            weight6 = np.diag(
                np.concatenate((fusion.model1_weight, fusion.model1_weight))
            )
            fused_prediction = (
                weight6 @ prediction1 + (np.eye(6) - weight6) @ prediction2
            )
            estimator.predict_mean(fused_prediction, yaw_delta_rad=yaw_delta)
            state = estimator.update(position)
            retained: deque[PendingPrediction] = deque()
            for item in pending:
                item.state1 = rotation @ (
                    model.A_d @ item.state1
                    + model.B_d @ (achieved - item.tau_base)
                )
                item.state2 = rotation @ (
                    model.A_d @ item.state2
                    + model.B_d @ (achieved - model.restoring_force)
                )
                item.steps += 1
                fusion.observe_position(
                    actual_position=state[:3],
                    prediction1=item.state1[:3],
                    prediction2=item.state2[:3],
                    horizon_step=item.steps,
                    origin_index=item.origin_index,
                    target_index=frame_index,
                )
                if item.steps < fusion.config.prediction_horizon:
                    retained.append(item)
            pending = retained
        previous_base = tau_base.copy()
        pending.append(
            PendingPrediction(
                origin_index=frame_index,
                state1=state.copy(),
                state2=state.copy(),
                tau_base=tau_base.copy(),
            )
        )
        denominator = max(
            0.0, float(fusion.M1[0] + fusion.M2[0] - 2.0 * fusion.C12[0])
        )
        raw = float(
            (fusion.M2[0] - fusion.C12[0] + fusion.config.epsilon)
            / (denominator + 2.0 * fusion.config.epsilon)
        )
        times.append(float(record["host_time_s"]))
        weights.append(float(fusion.model1_weight[0]))
        separations.append(math.sqrt(denominator))
        ambiguous.append(
            denominator
            <= float(fusion.config.indistinguishable_score_threshold[0])
        )
        model1_mse.append(float(fusion.M1[0]))
        model2_mse.append(float(fusion.M2[0]))
        raw_optimal_weight.append(raw)
        base_steps.append(
            math.nan
            if last_logged_base is None
            else abs(float(tau_base[0] - last_logged_base[0]))
        )
        base_lags.append(abs(float(tau_base[0] - achieved[0])))
        last_logged_base = tau_base.copy()
    return {
        "time": np.asarray(times),
        "weight": np.asarray(weights),
        "separation": np.asarray(separations),
        "ambiguous": np.asarray(ambiguous),
        "model1_mse": np.asarray(model1_mse),
        "model2_mse": np.asarray(model2_mse),
        "raw_optimal_weight": np.asarray(raw_optimal_weight),
        "base_step": np.asarray(base_steps),
        "base_lag": np.asarray(base_lags),
    }


def _auc(static: np.ndarray, cruise: np.ndarray) -> float | None:
    if len(static) == 0 or len(cruise) == 0:
        return None
    ordered = np.sort(static)
    lower = np.searchsorted(ordered, cruise, side="left")
    upper = np.searchsorted(ordered, cruise, side="right")
    return float(np.mean((lower + upper) / (2.0 * len(ordered))))


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    p10, p50, p90, p95 = np.percentile(values, (10, 50, 90, 95))
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
    }


def validate(config_path: Path, pairs: list[Pair]) -> dict[str, Any]:
    source = load_runtime_config(config_path)
    runtime = build_experimental_runtime_config(source)
    tracker_template = build_auto_tracker(runtime)
    base_fusion = runtime["auto_runtime"]["active_mpc_parameters"]["fusion"]
    variants: list[tuple[str, int, float | None]] = [
        ("current_effective_h2_hold", 2, None),
        ("h5_hold", 5, None),
        ("h8_hold", 8, None),
    ]
    for horizon in (5, 8):
        for threshold in (1.0e-6, 1.0e-5, 1.0e-4):
            variants.append(
                (f"h{horizon}_default_model2_threshold_{threshold:.0e}", horizon, threshold)
            )

    replayed: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name, _, _ in variants
    }
    run_reports: list[dict[str, Any]] = []
    for pair in pairs:
        records = _jsonl(pair.trace)
        overhead = _load_overhead(pair.bag)
        run_replays: dict[str, dict[str, np.ndarray]] = {}
        for name, horizon, threshold in variants:
            result = _replay(
                records,
                tracker_template,
                _variant_config(base_fusion, horizon),
                threshold,
            )
            labels, kinematics = _overhead_labels(result["time"], overhead)
            result["labels"] = labels
            result.update(kinematics)
            replayed[name].append(result)
            run_replays[name] = result
        first = run_replays[variants[0][0]]
        run_reports.append(
            {
                "trace": str(pair.trace),
                "bag": str(pair.bag),
                "updates": int(len(first["time"])),
                "static_samples": int(np.sum(first["labels"] == "static")),
                "cruise_samples": int(np.sum(first["labels"] == "cruise")),
                "steady_cruise_samples": int(
                    np.sum(first["labels"] == "steady_cruise")
                ),
            }
        )

    variant_reports: list[dict[str, Any]] = []
    for name, horizon, threshold in variants:
        candidate_fusion_config = _variant_config(base_fusion, horizon)
        joined = {
            key: np.concatenate([item[key] for item in replayed[name]])
            for key in (
                "weight",
                "separation",
                "ambiguous",
                "labels",
                "model1_mse",
                "model2_mse",
                "raw_optimal_weight",
                "base_step",
                "base_lag",
            )
        }
        static_mask = joined["labels"] == "static"
        cruise_mask = np.isin(joined["labels"], ("cruise", "steady_cruise"))
        steady_cruise_mask = joined["labels"] == "steady_cruise"
        credible_base_mask = (
            steady_cruise_mask
            & np.isfinite(joined["base_step"])
            & (joined["base_step"] < 0.10)
            & (joined["base_lag"] < 0.30)
        )
        static_weight = joined["weight"][static_mask]
        cruise_weight = joined["weight"][cruise_mask]
        steady_cruise_weight = joined["weight"][steady_cruise_mask]
        variant_reports.append(
            {
                "name": name,
                "configured_prediction_horizon": int(
                    candidate_fusion_config["prediction_horizon"]
                ),
                "maximum_effectively_scored_horizon": horizon,
                "ambiguity_policy": (
                    "hold_previous_weight"
                    if threshold is None
                    else "decay_toward_model2"
                ),
                "forward_ambiguity_threshold_m2": (
                    float(base_fusion["indistinguishable_score_threshold"][0])
                    if threshold is None
                    else threshold
                ),
                "static_weight": _percentiles(static_weight),
                "cruise_weight": _percentiles(cruise_weight),
                "steady_cruise_weight": _percentiles(steady_cruise_weight),
                "auc_probability_cruise_weight_exceeds_static": _auc(
                    static_weight, cruise_weight
                ),
                "balanced_accuracy_at_weight_0_5": (
                    float(
                        0.5
                        * (
                            np.mean(static_weight < 0.5)
                            + np.mean(cruise_weight >= 0.5)
                        )
                    )
                    if len(static_weight) and len(cruise_weight)
                    else None
                ),
                "static_model1_above_0_5_fraction": (
                    float(np.mean(static_weight >= 0.5)) if len(static_weight) else None
                ),
                "cruise_model1_below_0_5_fraction": (
                    float(np.mean(cruise_weight < 0.5)) if len(cruise_weight) else None
                ),
                "static_model1_candidate_lower_mse_fraction": float(
                    np.mean(
                        joined["model1_mse"][static_mask]
                        < joined["model2_mse"][static_mask]
                    )
                ),
                "cruise_model1_candidate_lower_mse_fraction": float(
                    np.mean(
                        joined["model1_mse"][cruise_mask]
                        < joined["model2_mse"][cruise_mask]
                    )
                ),
                "steady_cruise_model1_candidate_lower_mse_fraction": (
                    float(
                        np.mean(
                            joined["model1_mse"][steady_cruise_mask]
                            < joined["model2_mse"][steady_cruise_mask]
                        )
                    )
                    if np.any(steady_cruise_mask)
                    else None
                ),
                "steady_cruise_credible_base_samples": int(
                    np.sum(credible_base_mask)
                ),
                "steady_cruise_credible_base_model1_lower_mse_fraction": (
                    float(
                        np.mean(
                            joined["model1_mse"][credible_base_mask]
                            < joined["model2_mse"][credible_base_mask]
                        )
                    )
                    if np.any(credible_base_mask)
                    else None
                ),
                "steady_cruise_credible_base_weight": _percentiles(
                    joined["weight"][credible_base_mask]
                ),
                "static_raw_optimal_weight": _percentiles(
                    joined["raw_optimal_weight"][static_mask]
                ),
                "steady_cruise_raw_optimal_weight": _percentiles(
                    joined["raw_optimal_weight"][steady_cruise_mask]
                ),
                "static_candidate_separation_m": _percentiles(
                    joined["separation"][static_mask]
                ),
                "cruise_candidate_separation_m": _percentiles(
                    joined["separation"][cruise_mask]
                ),
                "steady_cruise_candidate_separation_m": _percentiles(
                    joined["separation"][steady_cruise_mask]
                ),
                "static_ambiguous_fraction": float(
                    np.mean(joined["ambiguous"][static_mask])
                ),
                "cruise_ambiguous_fraction": float(
                    np.mean(joined["ambiguous"][cruise_mask])
                ),
                "steady_cruise_ambiguous_fraction": (
                    float(np.mean(joined["ambiguous"][steady_cruise_mask]))
                    if np.any(steady_cruise_mask)
                    else None
                ),
            }
        )
    return {
        "analysis": "offline forward fusion identifiability validation",
        "hardware_transport_opened": False,
        "controller_config_modified": False,
        "overhead_labels": {
            "static": "target_speed<0.015 m/s and vehicle_speed<0.015 m/s",
            "cruise": (
                "target_speed>0.040 m/s, vehicle_speed>0.025 m/s, "
                "direction cosine>0.70, and speed mismatch<0.050 m/s"
            ),
            "steady_cruise": (
                "cruise plus target and vehicle acceleration each <0.040 m/s^2"
            ),
        },
        "runs": run_reports,
        "variants": variant_reports,
    }


def _default_pairs(root: Path) -> list[Pair]:
    names = (
        ("experimental_auto_20260814_214726.jsonl", "top_camera_20260814_214614"),
        ("experimental_auto_20260814_215909.jsonl", "top_camera_20260814_215826"),
        ("experimental_auto_20260814_224318.jsonl", "top_camera_20260814_224228"),
        ("experimental_auto_20260814_225507.jsonl", "top_camera_20260814_225441"),
        ("experimental_auto_20260814_230123.jsonl", "top_camera_20260814_230109"),
    )
    pairs: list[Pair] = []
    for trace_name, bag_name in names:
        trace = root / trace_name
        bag_dir = root / bag_name
        databases = sorted(bag_dir.glob("*.db3"))
        if trace.exists() and databases:
            pairs.append(Pair(trace=trace, bag=databases[0]))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    module_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--config",
        type=Path,
        default=module_dir / "finesub_v4pro1_mpc.json",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=module_dir.parent / "calibration_logs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.config, _default_pairs(args.logs_dir))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
