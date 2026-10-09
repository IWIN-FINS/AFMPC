"""Fit the onboard red-fish camera extrinsic from raw overhead AprilTags.

This module is deliberately offline and read-only with respect to both camera
pipelines.  It freezes external JSONL records into the MPC calibration log,
reconstructs the fish reference point from AprilTag 17 and the measured rigid
offset, applies the same MPC input gate used by the live reader, and reports a
candidate transform with cross-validation and sensitivity checks.

The fitted transform is::

    target_body_frd = R_body_from_camera @ target_camera_opencv + translation

The translation is first reported relative to the midpoint of vehicle tags
15/16.  A body-origin value is also reported using the existing, unverified
tag-midpoint height candidate; it must not be treated as a new measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .camera_transform import ALIGNED_OPENCV_TO_BODY
from .vision_measurement import VisionMeasurementGate


# Fixed overhead-camera calibration used by the already-running raw AprilTag
# node.  This is copied into the MPC-side report so rerunning the analysis does
# not depend on, or write into, FinsSim.
OVERHEAD_WORLD_FROM_CAMERA_QUAT_XYZW = np.asarray(
    [
        0.9989334485930018,
        -0.0019376405095682715,
        0.00007356191223612514,
        -0.04613247684650684,
    ],
    dtype=float,
)

# Operator estimate on 2026-08-13: the tag-15/16 midpoint is approximately
# 0.15 m directly above the body control origin, hence -0.15 m in MPC FRD.
TAG_MIDPOINT_IN_BODY_FRD_CANDIDATE_M = np.asarray([0.0, 0.0, -0.15])
CAD_CAMERA_ORIGIN_IN_BODY_FRD_M = np.asarray(
    [0.2609931, 0.0077881, -0.1236508]
)

PRIOR_DYNAMIC_ROTATION = np.asarray(
    [
        [0.105483644, -0.006124566, 0.994402178],
        [0.994329497, -0.012918533, -0.105555500],
        [0.013492699, 0.999897795, 0.004727143],
    ]
)
PRIOR_RAW_TAG_ROTATION = np.asarray(
    [
        [0.03973590, -0.10832709, 0.99332084],
        [0.99810721, -0.04239394, -0.04455066],
        [0.04693682, 0.99321095, 0.10643749],
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quaternion_xyzw_to_rotation(quaternion: Iterable[float]) -> np.ndarray:
    """Return a 3x3 active rotation matrix for an xyzw quaternion."""

    x, y, z, w = np.asarray(tuple(quaternion), dtype=float).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must have finite nonzero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _pool_down_in_overhead_camera() -> np.ndarray:
    world_from_camera = quaternion_xyzw_to_rotation(
        OVERHEAD_WORLD_FROM_CAMERA_QUAT_XYZW
    )
    down = world_from_camera.T @ np.asarray([0.0, 0.0, -1.0])
    return down / np.linalg.norm(down)


def _parse_label_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path.strip())


def _best_detection_by_id(raw: str) -> dict[int, dict[str, Any]]:
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(values, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            tag_id = int(value["tag_id"])
        except (KeyError, TypeError, ValueError):
            continue
        margin = _finite_float(value.get("decision_margin"))
        previous = result.get(tag_id)
        previous_margin = (
            _finite_float(previous.get("decision_margin"))
            if previous is not None
            else None
        )
        if previous is None or (margin or -math.inf) > (previous_margin or -math.inf):
            result[tag_id] = value
    return result


def _position(detection: dict[str, Any]) -> np.ndarray | None:
    try:
        value = np.asarray(detection["pnp_camera_xyz"], dtype=float).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return None
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    return value


def _tag_normal(
    detection: dict[str, Any], pool_down: np.ndarray
) -> np.ndarray | None:
    try:
        rotation = quaternion_xyzw_to_rotation(
            detection["pnp_camera_quat_xyzw"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    normal = rotation[:, 2]
    if float(normal @ pool_down) < 0.0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _row_source_time(row: dict[str, str]) -> float | None:
    try:
        receive = datetime.fromisoformat(row["host_time_utc"]).timestamp()
    except (KeyError, ValueError):
        return None
    age = _finite_float(row.get("vision_status_age_s"))
    return receive - (0.0 if age is None else age)


def load_overhead_run(
    label: str,
    path: Path,
    *,
    target_offset_m: float,
) -> dict[str, Any]:
    """Load raw tag 15/16/17 poses and construct two target references."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty overhead CSV: {path}")
    pool_down = _pool_down_in_overhead_camera()
    samples_fixed: list[list[float]] = []
    samples_tag_normal: list[list[float]] = []
    separations: list[float] = []
    normal_tilts: list[float] = []
    reprojection_errors: list[float] = []
    seen_status: set[str] = set()
    detected_id_counts: Counter[str] = Counter()
    for row in rows:
        status_count = row.get("vision_status_count", "")
        if not status_count or status_count in seen_status:
            continue
        seen_status.add(status_count)
        detections = _best_detection_by_id(row.get("vision_detections_detail", ""))
        detected_id_counts[str(tuple(sorted(detections)))] += 1
        if not {15, 16, 17}.issubset(detections):
            continue
        source_time = _row_source_time(row)
        positions = {tag_id: _position(detections[tag_id]) for tag_id in (15, 16, 17)}
        if source_time is None or any(value is None for value in positions.values()):
            continue
        p15, p16, p17 = (positions[tag_id] for tag_id in (15, 16, 17))
        assert p15 is not None and p16 is not None and p17 is not None
        forward = p15 - p16
        forward = forward - pool_down * float(forward @ pool_down)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 0.05:
            continue
        forward /= forward_norm
        right = np.cross(pool_down, forward)
        right /= np.linalg.norm(right)
        body_down = np.cross(forward, right)
        body_down /= np.linalg.norm(body_down)
        camera_from_body = np.column_stack((forward, right, body_down))
        midpoint = 0.5 * (p15 + p16)

        fixed_target = p17 + target_offset_m * pool_down
        fixed_body = camera_from_body.T @ (fixed_target - midpoint)
        samples_fixed.append([source_time, *fixed_body.tolist()])

        normal17 = _tag_normal(detections[17], pool_down)
        if normal17 is not None:
            normal_target = p17 + target_offset_m * normal17
            normal_body = camera_from_body.T @ (normal_target - midpoint)
            samples_tag_normal.append([source_time, *normal_body.tolist()])
            normal_tilts.append(
                math.degrees(
                    math.acos(float(np.clip(normal17 @ pool_down, -1.0, 1.0)))
                )
            )
        separations.append(float(np.linalg.norm(p15 - p16)))
        for tag_id in (15, 16, 17):
            error = _finite_float(
                detections[tag_id].get("pnp_reprojection_error_px")
            )
            if error is not None:
                reprojection_errors.append(error)

    fixed = np.asarray(samples_fixed, dtype=float)
    tag_normal = np.asarray(samples_tag_normal, dtype=float)
    if len(fixed) < 3:
        raise RuntimeError(f"too few simultaneous tag 15/16/17 samples in {path}")
    start_s = datetime.fromisoformat(rows[0]["host_time_utc"]).timestamp()
    end_s = datetime.fromisoformat(rows[-1]["host_time_utc"]).timestamp()
    return {
        "label": label,
        "path": path,
        "sha256": _sha256(path),
        "rows": len(rows),
        "start_s": start_s,
        "end_s": end_s,
        "fixed": fixed,
        "tag_normal": tag_normal,
        "tag_separation_m": np.asarray(separations),
        "tag_normal_tilt_deg": np.asarray(normal_tilts),
        "reprojection_error_px": np.asarray(reprojection_errors),
        "detected_id_counts": dict(sorted(detected_id_counts.items())),
    }


def snapshot_front_jsonl(
    source: Path,
    destination: Path,
    runs: list[dict[str, Any]],
    *,
    margin_s: float = 0.60,
) -> int:
    """Freeze complete records around all capture windows into MPC storage."""

    selected: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                timestamp = float(json.loads(line)["frame_ts_mean"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if any(
                run["start_s"] - margin_s
                <= timestamp
                <= run["end_s"] + margin_s
                for run in runs
            ):
                selected.append(line if line.endswith("\n") else line + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(selected), encoding="utf-8")
    return len(selected)


def load_gated_front(
    path: Path,
    runs: list[dict[str, Any]],
    *,
    margin_s: float = 0.60,
) -> None:
    records_by_label: dict[str, list[dict[str, Any]]] = {
        run["label"]: [] for run in runs
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                timestamp = float(record["frame_ts_mean"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for run in runs:
                if (
                    run["start_s"] - margin_s
                    <= timestamp
                    <= run["end_s"] + margin_s
                ):
                    records_by_label[run["label"]].append(record)
                    break

    for run in runs:
        gate = VisionMeasurementGate()
        samples: list[list[float]] = []
        reasons: Counter[str] = Counter()
        for record in records_by_label[run["label"]]:
            result_time = _finite_float(record.get("result_time"))
            if result_time is None:
                continue
            decision = gate.evaluate(record, now_s=result_time + 0.01)
            reasons[decision.reason] += 1
            if decision.control_ready and decision.measurement is not None:
                measurement = decision.measurement
                samples.append(
                    [
                        measurement.acquisition_time_s,
                        *measurement.position_camera_xyz_m.tolist(),
                    ]
                )
        run["front"] = np.asarray(samples, dtype=float)
        run["front_gate_reasons"] = dict(sorted(reasons.items()))


def replay_front_gate(path: Path) -> dict[str, Any]:
    """Replay one frozen JSONL continuously and audit out-of-range readiness."""

    gate = VisionMeasurementGate()
    reasons: Counter[str] = Counter()
    records = 0
    control_ready = 0
    above_control_max = 0
    above_control_max_ready = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                result_time = float(record["result_time"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            decision = gate.evaluate(record, now_s=result_time + 0.01)
            records += 1
            control_ready += int(decision.control_ready)
            reasons[decision.reason] += 1
            tracks = [
                track
                for track in (record.get("tracks") or [])
                if isinstance(track, dict)
            ]
            if tracks:
                track = max(
                    tracks,
                    key=lambda item: _finite_float(item.get("confidence")) or -1.0,
                )
                try:
                    position = np.asarray(
                        track.get("position", track.get("position_xyz")), dtype=float
                    ).reshape(-1)
                except (TypeError, ValueError):
                    position = np.empty(0)
                if position.shape == (3,) and position[2] > 1.50:
                    above_control_max += 1
                    above_control_max_ready += int(decision.control_ready)
    return {
        "records": records,
        "control_ready_records": control_ready,
        "rejected_or_pending_records": records - control_ready,
        "decision_reasons": dict(sorted(reasons.items())),
        "records_above_1p50m": above_control_max,
        "records_above_1p50m_control_ready": above_control_max_ready,
    }


def rigid_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) < 3 or source.shape != target.shape or source.shape[1:] != (3,):
        raise ValueError("rigid fit requires matching Nx3 arrays with N >= 3")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    left, singular_values, right_t = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation, singular_values


def robust_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    keep = np.ones(len(source), dtype=bool)
    threshold = 0.025
    for _ in range(20):
        rotation, translation, singular_values = rigid_fit(source[keep], target[keep])
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        values = residual[keep]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = max(0.025, median + 3.0 * 1.4826 * mad)
        updated = residual < threshold
        if np.array_equal(updated, keep):
            break
        if np.count_nonzero(updated) < 3:
            break
        keep = updated
    return rotation, translation, singular_values, keep, residual, threshold


def _paired(
    runs: list[dict[str, Any]], lag_s: float, reference_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera: list[np.ndarray] = []
    body: list[np.ndarray] = []
    run_indices: list[int] = []
    for run_index, run in enumerate(runs):
        front = run["front"]
        reference = run[reference_key]
        if len(front) == 0 or len(reference) == 0:
            continue
        shifted = front[:, 0] + lag_s
        valid = (shifted >= reference[0, 0]) & (shifted <= reference[-1, 0])
        selected = front[valid]
        shifted = shifted[valid]
        if len(selected) == 0:
            continue
        interpolated = np.column_stack(
            [
                np.interp(shifted, reference[:, 0], reference[:, axis])
                for axis in (1, 2, 3)
            ]
        )
        camera.append(selected[:, 1:4])
        body.append(interpolated)
        run_indices.extend([run_index] * len(selected))
    if not camera:
        raise RuntimeError("no paired front/overhead samples")
    return np.vstack(camera), np.vstack(body), np.asarray(run_indices, dtype=int)


def _fit_for_reference(
    runs: list[dict[str, Any]], reference_key: str
) -> dict[str, Any]:
    best: tuple[Any, ...] | None = None
    lag_scores: list[tuple[float, float]] = []
    for lag_s in np.arange(-0.35, 0.3501, 0.005):
        camera, body, run_indices = _paired(runs, float(lag_s), reference_key)
        fit = robust_fit(camera, body)
        rmse = float(np.sqrt(np.mean(fit[4][fit[3]] ** 2)))
        lag_scores.append((float(lag_s), rmse))
        candidate = (rmse, float(lag_s), fit, camera, body, run_indices)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("lag scan produced no result")
    rmse, lag_s, fit, camera, body, run_indices = best
    rotation, translation, singular_values, keep, residual, threshold = fit

    per_run = []
    leave_one_out = []
    for run_index, run in enumerate(runs):
        selected = keep & (run_indices == run_index)
        values = residual[selected]
        per_run.append(
            {
                "label": run["label"],
                "paired_samples": int(np.count_nonzero(run_indices == run_index)),
                "inlier_samples": int(np.count_nonzero(selected)),
                "rmse_m": float(np.sqrt(np.mean(values**2))),
                "median_m": float(np.median(values)),
                "p95_m": float(np.quantile(values, 0.95)),
            }
        )
        training = keep & (run_indices != run_index)
        held_out = run_indices == run_index
        if np.count_nonzero(training) < 3 or not np.any(held_out):
            continue
        loo_rotation, loo_translation, _ = rigid_fit(
            camera[training], body[training]
        )
        loo_residual = np.linalg.norm(
            camera[held_out] @ loo_rotation.T + loo_translation - body[held_out],
            axis=1,
        )
        leave_one_out.append(
            {
                "held_out_label": run["label"],
                "held_out_samples": int(np.count_nonzero(held_out)),
                "rmse_m": float(np.sqrt(np.mean(loo_residual**2))),
                "median_m": float(np.median(loo_residual)),
                "p95_m": float(np.quantile(loo_residual, 0.95)),
                "rotation_difference_from_all_deg": rotation_angle_deg(
                    loo_rotation @ rotation.T
                ),
                "translation_difference_from_all_m": float(
                    np.linalg.norm(loo_translation - translation)
                ),
            }
        )

    source_center = np.mean(camera[keep], axis=0)
    target_center = np.mean(body[keep], axis=0)
    rotated_centered = (camera[keep] - source_center) @ rotation.T
    target_centered = body[keep] - target_center
    similarity_scale = float(
        np.sum(target_centered * rotated_centered)
        / np.sum((camera[keep] - source_center) ** 2)
    )
    similarity_translation = target_center - similarity_scale * rotation @ source_center
    similarity_residual = np.linalg.norm(
        similarity_scale * (camera[keep] @ rotation.T)
        + similarity_translation
        - body[keep],
        axis=1,
    )

    centroid_camera: list[np.ndarray] = []
    centroid_body: list[np.ndarray] = []
    centroid_labels: list[str] = []
    for run_index, run in enumerate(runs):
        selected = keep & (run_indices == run_index)
        if not np.any(selected):
            continue
        centroid_camera.append(np.median(camera[selected], axis=0))
        centroid_body.append(np.median(body[selected], axis=0))
        centroid_labels.append(run["label"])
    centroid_camera_array = np.asarray(centroid_camera)
    centroid_body_array = np.asarray(centroid_body)
    centroid_rotation, centroid_translation, centroid_singular_values = rigid_fit(
        centroid_camera_array, centroid_body_array
    )
    centroid_residual = np.linalg.norm(
        centroid_camera_array @ centroid_rotation.T
        + centroid_translation
        - centroid_body_array,
        axis=1,
    )

    comparisons = []
    for name, candidate_rotation in (
        ("axis_aligned_nominal", ALIGNED_OPENCV_TO_BODY),
        ("three_dynamic_runs_20260812", PRIOR_DYNAMIC_ROTATION),
        ("earlier_raw_tag_recalculation", PRIOR_RAW_TAG_ROTATION),
        ("new_rigid_target_fit", rotation),
    ):
        candidate_translation = np.mean(
            body[keep] - camera[keep] @ candidate_rotation.T, axis=0
        )
        candidate_residual = np.linalg.norm(
            camera[keep] @ candidate_rotation.T + candidate_translation - body[keep],
            axis=1,
        )
        comparisons.append(
            {
                "name": name,
                "rotation_deviation_from_axis_aligned_deg": rotation_angle_deg(
                    candidate_rotation @ ALIGNED_OPENCV_TO_BODY.T
                ),
                "rotation_difference_from_new_fit_deg": rotation_angle_deg(
                    candidate_rotation @ rotation.T
                ),
                "best_translation_relative_tag_midpoint_m": candidate_translation.tolist(),
                "rmse_m": float(np.sqrt(np.mean(candidate_residual**2))),
                "median_m": float(np.median(candidate_residual)),
                "p95_m": float(np.quantile(candidate_residual, 0.95)),
            }
        )

    return {
        "reference_key": reference_key,
        "paired_samples": int(len(camera)),
        "inlier_samples": int(np.count_nonzero(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "best_lag_s": lag_s,
        "effective_front_delay_vs_overhead_s": float(-lag_s),
        "near_optimal_lag_within_1mm_rmse_s": [
            min(score[0] for score in lag_scores if score[1] <= rmse + 0.001),
            max(score[0] for score in lag_scores if score[1] <= rmse + 0.001),
        ],
        "lag_identifiability_note": (
            "short mostly-static placements give a broader lag minimum than the "
            "earlier continuous-motion experiment"
        ),
        "outlier_threshold_m": threshold,
        "rmse_m": rmse,
        "median_m": float(np.median(residual[keep])),
        "p95_m": float(np.quantile(residual[keep], 0.95)),
        "singular_values": singular_values.tolist(),
        "rotation_body_from_camera": rotation.tolist(),
        "rotation_deviation_from_axis_aligned_deg": rotation_angle_deg(
            rotation @ ALIGNED_OPENCV_TO_BODY.T
        ),
        "translation_relative_tag_midpoint_frd_m": translation.tolist(),
        "per_run": per_run,
        "leave_one_run_out": leave_one_out,
        "similarity_diagnostic": {
            "uniform_scale": similarity_scale,
            "translation_relative_tag_midpoint_frd_m": similarity_translation.tolist(),
            "rmse_m": float(np.sqrt(np.mean(similarity_residual**2))),
            "median_m": float(np.median(similarity_residual)),
            "p95_m": float(np.quantile(similarity_residual, 0.95)),
            "control_use": False,
            "reason": "diagnostic only; MPC camera transform must remain rigid",
        },
        "equal_weight_placement_centroid_fit": {
            "labels": centroid_labels,
            "rotation_body_from_camera": centroid_rotation.tolist(),
            "translation_relative_tag_midpoint_frd_m": centroid_translation.tolist(),
            "rotation_difference_from_all_samples_deg": rotation_angle_deg(
                centroid_rotation @ rotation.T
            ),
            "singular_values": centroid_singular_values.tolist(),
            "rmse_m": float(np.sqrt(np.mean(centroid_residual**2))),
            "residual_by_label_m": {
                label: float(value)
                for label, value in zip(centroid_labels, centroid_residual)
            },
            "control_use": False,
            "reason": "five placement medians are a weighting sensitivity check",
        },
        "candidate_rotation_comparison": comparisons,
    }


def _evaluate_fixed_transform(
    runs: list[dict[str, Any]],
    *,
    reference_key: str,
    lag_s: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate excluded captures without allowing them to alter the fit."""

    results: list[dict[str, Any]] = []
    for run in runs:
        try:
            camera, body, _ = _paired([run], lag_s, reference_key)
        except RuntimeError:
            continue
        residual = np.linalg.norm(
            camera @ rotation.T + translation - body, axis=1
        )
        results.append(
            {
                "label": run["label"],
                "paired_samples": int(len(residual)),
                "rmse_m": float(np.sqrt(np.mean(residual**2))),
                "median_m": float(np.median(residual)),
                "p95_m": float(np.quantile(residual, 0.95)),
                "used_for_fit": False,
                "used_for_acceptance": False,
            }
        )
    return results


def _run_source_summary(run: dict[str, Any]) -> dict[str, Any]:
    separation = run["tag_separation_m"]
    tilts = run["tag_normal_tilt_deg"]
    reprojection = run["reprojection_error_px"]
    front = run.get("front", np.empty((0, 4)))
    return {
        "label": run["label"],
        "overhead_csv": str(run["path"]),
        "overhead_csv_sha256": run["sha256"],
        "csv_rows": int(run["rows"]),
        "simultaneous_three_tag_samples": int(len(run["fixed"])),
        "mpc_gate_approved_front_samples": int(len(front)),
        "front_gate_reasons": run.get("front_gate_reasons", {}),
        "tag_15_16_separation_mean_m": float(np.mean(separation)),
        "tag_15_16_separation_std_m": float(np.std(separation)),
        "tag17_normal_tilt_median_deg": float(np.median(tilts)),
        "tag17_normal_tilt_p95_deg": float(np.quantile(tilts, 0.95)),
        "pnp_reprojection_error_p95_px": float(np.quantile(reprojection, 0.95)),
        "detected_id_counts": run["detected_id_counts"],
    }


def analyze(
    front_jsonl: Path,
    primary_csvs: list[tuple[str, Path]],
    *,
    excluded_csvs: list[tuple[str, Path]] | None = None,
    target_offset_m: float = 0.23,
    front_snapshot: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(target_offset_m) or target_offset_m <= 0.0:
        raise ValueError("target offset must be finite and positive")
    excluded_csvs = excluded_csvs or []
    primary = [
        load_overhead_run(label, path, target_offset_m=target_offset_m)
        for label, path in primary_csvs
    ]
    excluded = [
        load_overhead_run(label, path, target_offset_m=target_offset_m)
        for label, path in excluded_csvs
        if path.exists()
    ]
    all_runs = primary + excluded
    external_source = front_jsonl
    snapshot_records = None
    if front_snapshot is not None:
        snapshot_records = snapshot_front_jsonl(
            front_jsonl, front_snapshot, all_runs
        )
        front_jsonl = front_snapshot
    load_gated_front(front_jsonl, all_runs)
    for run in primary:
        if len(run["front"]) < 10:
            raise RuntimeError(
                f"too few MPC-gate-approved front samples in {run['label']}: "
                f"{len(run['front'])}"
            )

    primary_fit = _fit_for_reference(primary, "fixed")
    normal_sensitivity = _fit_for_reference(primary, "tag_normal")
    excluded_diagnostics = _evaluate_fixed_transform(
        excluded,
        reference_key="fixed",
        lag_s=float(primary_fit["best_lag_s"]),
        rotation=np.asarray(primary_fit["rotation_body_from_camera"]),
        translation=np.asarray(
            primary_fit["translation_relative_tag_midpoint_frd_m"]
        ),
    )
    translation_midpoint = np.asarray(
        primary_fit["translation_relative_tag_midpoint_frd_m"]
    )
    camera_origin_candidate = (
        translation_midpoint + TAG_MIDPOINT_IN_BODY_FRD_CANDIDATE_M
    )
    origin_difference = camera_origin_candidate - CAD_CAMERA_ORIGIN_IN_BODY_FRD_M
    rotation_difference = rotation_angle_deg(
        np.asarray(primary_fit["rotation_body_from_camera"])
        @ PRIOR_DYNAMIC_ROTATION.T
    )
    maximum_loo_rotation = max(
        item["rotation_difference_from_all_deg"]
        for item in primary_fit["leave_one_run_out"]
    )

    gates = {
        "rigid_rmse_at_most_0p03": primary_fit["rmse_m"] <= 0.03,
        "rigid_p95_at_most_0p05": primary_fit["p95_m"] <= 0.05,
        "max_leave_one_run_rotation_at_most_3deg": maximum_loo_rotation <= 3.0,
        "agreement_with_prior_dynamic_candidate_at_most_3deg": rotation_difference <= 3.0,
        "camera_origin_difference_from_cad_at_most_0p03m": float(
            np.linalg.norm(origin_difference)
        )
        <= 0.03,
    }
    enabled = all(gates.values())
    return {
        "analysis": "rigid tag-17/red-fish onboard-camera extrinsic cross-check",
        "coordinate_conventions": {
            "front_input": "OpenCV [right, down, forward] metres",
            "fit_output": "ROV body FRD [forward, right, down] metres",
            "vehicle_forward": "tag15 center minus tag16 center, projected onto pool horizontal",
            "vehicle_right": "pool_down cross vehicle_forward",
            "vehicle_reference_origin": "midpoint of tag15/tag16 centers",
            "pool_down_in_overhead_camera": _pool_down_in_overhead_camera().tolist(),
        },
        "target_fixture": {
            "target_tag_id": 17,
            "fish_visual_reference_below_tag_center_m": target_offset_m,
            "horizontal_offset_m": [0.0, 0.0],
            "operator_precision": "approximately 23 cm",
            "primary_interpretation": "fixed pool-vertical down direction",
            "sensitivity_interpretation": "tag-17 PnP normal, oriented toward pool down",
            "translation_uncertainty_note": (
                "any error in the approximately 0.23 m fixture length transfers "
                "directly into the fitted down translation"
            ),
        },
        "sources": {
            "front_jsonl": str(front_jsonl),
            "front_jsonl_sha256": _sha256(front_jsonl),
            "snapshot_records": snapshot_records,
            "continuous_snapshot_gate_replay": replay_front_gate(front_jsonl),
            "external_read_only_source": (
                str(external_source) if front_snapshot is not None else None
            ),
            "primary_runs": [_run_source_summary(run) for run in primary],
            "excluded_runs_not_used_for_fit": [
                _run_source_summary(run) for run in excluded
            ],
        },
        "method": {
            "front_filter": "production MPC VisionMeasurementGate, reset per capture",
            "pairing": "linear interpolation of raw-tag reference at front frame timestamp plus scanned lag",
            "lag_scan_s": [-0.35, 0.35, 0.005],
            "fit": "Kabsch rigid transform with iterative median/MAD rejection",
            "cross_validation": "leave one physical placement out",
            "external_repositories_modified": False,
        },
        "primary_fixed_vertical_fit": primary_fit,
        "tag_normal_sensitivity_fit": normal_sensitivity,
        "excluded_capture_diagnostics": excluded_diagnostics,
        "body_origin_translation_cross_check": {
            "tag_midpoint_in_body_frd_candidate_m": TAG_MIDPOINT_IN_BODY_FRD_CANDIDATE_M.tolist(),
            "tag_midpoint_source": (
                "operator estimate on 2026-08-13: approximately 0.15 m "
                "directly above the body control origin, with approximately "
                "zero horizontal offset"
            ),
            "camera_origin_from_new_fit_candidate_frd_m": camera_origin_candidate.tolist(),
            "cad_camera_origin_candidate_frd_m": CAD_CAMERA_ORIGIN_IN_BODY_FRD_M.tolist(),
            "new_minus_cad_frd_m": origin_difference.tolist(),
            "difference_norm_m": float(np.linalg.norm(origin_difference)),
        },
        "control_acceptance": {
            "gates": gates,
            "all_gates_pass": enabled,
            "enabled_for_control": False,
            "auto_authorized": False,
            "rotation_difference_from_prior_dynamic_candidate_deg": rotation_difference,
            "maximum_leave_one_run_rotation_difference_deg": maximum_loo_rotation,
            "reason_disabled": (
                "candidate disagreement, held-out stability, residual and/or body-origin "
                "cross-check gates remain unresolved; retain results as offline evidence only"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-jsonl", type=Path, required=True)
    parser.add_argument(
        "--primary-csv",
        type=_parse_label_path,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--excluded-csv",
        type=_parse_label_path,
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--target-offset-m", type=float, default=0.23)
    parser.add_argument("--front-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze(
        args.front_jsonl,
        args.primary_csv,
        excluded_csvs=args.excluded_csv,
        target_offset_m=args.target_offset_m,
        front_snapshot=args.front_snapshot,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
