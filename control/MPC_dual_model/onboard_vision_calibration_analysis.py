"""Fit onboard-stereo motion against synchronized overhead-camera truth.

This is an offline MPC-side analysis.  It reads, but never modifies, the
external vision JSONL and the synchronized CSV records.
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
from typing import Any

import numpy as np

from .camera_transform import ALIGNED_OPENCV_TO_BODY
from .vision_measurement import VisionMeasurementGate


OVERHEAD_OUTPUT_AXIS_CORRECTION = np.diag([-1.0, -1.0, 1.0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _load_overhead(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty overhead CSV: {path}")
    start = datetime.fromisoformat(rows[0]["host_time_utc"]).timestamp()
    end = datetime.fromisoformat(rows[-1]["host_time_utc"]).timestamp()
    seen: set[str] = set()
    samples: list[list[float]] = []
    for row in rows:
        count = row.get("vision_count", "")
        if row.get("vision_fresh") != "1" or not count or count in seen:
            continue
        try:
            position = np.asarray(json.loads(row["vision_position_xyz_m"]), dtype=float)
            stamp = float(row["vision_stamp_s"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        seen.add(count)
        samples.append([stamp, *position.tolist()])
    array = np.asarray(samples, dtype=float)
    if len(array) < 20:
        raise RuntimeError(f"too few valid overhead samples in {path}: {len(array)}")
    return {
        "path": path,
        "start_s": start,
        "end_s": end,
        "rows": len(rows),
        "samples": array,
    }


def _load_front(path: Path, runs: list[dict[str, Any]]) -> None:
    for run in runs:
        run["front"] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                timestamp = float(record["frame_ts_mean"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            tracks = record.get("tracks")
            if not isinstance(tracks, list) or not tracks:
                continue
            tracks = [track for track in tracks if isinstance(track, dict)]
            if not tracks:
                continue
            track = max(tracks, key=lambda item: float(item.get("confidence", -1.0)))
            try:
                position = np.asarray(
                    track.get("position", track.get("position_xyz")), dtype=float
                ).reshape(-1)
            except (TypeError, ValueError):
                continue
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            for run in runs:
                if run["start_s"] <= timestamp <= run["end_s"]:
                    run["front"].append(
                        [
                            timestamp,
                            *position.tolist(),
                            float(track.get("confidence", float("nan"))),
                            float(track.get("depth_nis", float("nan"))),
                        ]
                    )
                    break
    for run in runs:
        run["front"] = np.asarray(run["front"], dtype=float)
        if len(run["front"]) < 20:
            raise RuntimeError(
                f"too few onboard samples during {run['path']}: {len(run['front'])}"
            )


def _paired(
    runs: list[dict[str, Any]],
    lag_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    references: list[np.ndarray] = []
    onboard: list[np.ndarray] = []
    run_indices: list[int] = []
    diagnostics: list[np.ndarray] = []
    for run_index, run in enumerate(runs, start=1):
        truth = run["samples"]
        front = run["front"]
        shifted = front[:, 0] + lag_s
        valid = (shifted >= truth[0, 0]) & (shifted <= truth[-1, 0])
        selected = front[valid]
        shifted = shifted[valid]
        reference = np.column_stack(
            [np.interp(shifted, truth[:, 0], truth[:, axis]) for axis in (1, 2, 3)]
        )
        references.append(reference)
        onboard.append(selected[:, 1:4])
        diagnostics.append(selected[:, 4:6])
        run_indices.extend([run_index] * len(selected))
    return (
        np.vstack(references),
        np.vstack(onboard),
        np.asarray(run_indices, dtype=int),
        np.vstack(diagnostics),
    )


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _robust_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones(len(source), dtype=bool)
    for _ in range(10):
        rotation, translation, singular_values = _rigid_fit(source[keep], target[keep])
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        values = residual[keep]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = max(0.035, median + 3.0 * 1.4826 * mad)
        updated = residual < threshold
        if np.array_equal(updated, keep):
            break
        keep = updated
    return rotation, translation, singular_values, keep, residual


def _snapshot_front_jsonl(
    source: Path,
    destination: Path,
    runs: list[dict[str, Any]],
) -> int:
    """Freeze only records inside the synchronized calibration intervals."""

    selected: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                timestamp = float(json.loads(line)["frame_ts_mean"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if any(run["start_s"] <= timestamp <= run["end_s"] for run in runs):
                selected.append(line if line.endswith("\n") else line + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(selected), encoding="utf-8")
    return len(selected)


def _replay_mpc_gate(path: Path) -> dict[str, Any]:
    gate = VisionMeasurementGate()
    reasons: Counter[str] = Counter()
    total = 0
    control_ready = 0
    excursion_samples = 0
    excursion_control_ready = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                result_time = float(record["result_time"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            decision = gate.evaluate(record, now_s=result_time + 0.01)
            total += 1
            control_ready += int(decision.control_ready)
            reasons[decision.reason] += 1
            tracks = record.get("tracks") or []
            if tracks:
                position = tracks[0].get("position", tracks[0].get("position_xyz"))
                if position and float(position[2]) > 1.5:
                    excursion_samples += 1
                    excursion_control_ready += int(decision.control_ready)
    return {
        "records": total,
        "control_ready_records": control_ready,
        "rejected_or_pending_records": total - control_ready,
        "decision_reasons": dict(sorted(reasons.items())),
        "false_depth_excursion_records": excursion_samples,
        "false_depth_excursion_control_ready_records": excursion_control_ready,
    }


def analyze(
    front_jsonl: Path,
    overhead_csvs: list[Path],
    *,
    front_snapshot: Path | None = None,
) -> dict[str, Any]:
    runs = [_load_overhead(path) for path in overhead_csvs]
    original_front_jsonl = front_jsonl
    snapshot_records = None
    if front_snapshot is not None:
        snapshot_records = _snapshot_front_jsonl(front_jsonl, front_snapshot, runs)
        front_jsonl = front_snapshot
    _load_front(front_jsonl, runs)

    best = None
    for lag_s in np.arange(-0.35, 0.3501, 0.005):
        reference, onboard, run_index, diagnostics = _paired(runs, float(lag_s))
        fit = _robust_fit(reference, onboard)
        rmse = float(np.sqrt(np.mean(fit[4][fit[3]] ** 2)))
        candidate = (rmse, float(lag_s), fit, reference, onboard, run_index, diagnostics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("lag scan produced no fit")

    rmse, lag_s, fit, reference, onboard, run_index, diagnostics = best
    raw_rotation, raw_translation, singular_values, keep, residual = fit
    corrected_reference_to_onboard = raw_rotation @ OVERHEAD_OUTPUT_AXIS_CORRECTION
    rotation_body_from_onboard = (
        ALIGNED_OPENCV_TO_BODY @ corrected_reference_to_onboard.T
    )
    nominal_error_rotation = rotation_body_from_onboard @ ALIGNED_OPENCV_TO_BODY.T

    run_results = []
    for index, run in enumerate(runs, start=1):
        selected = keep & (run_index == index)
        values = residual[selected]
        run_results.append(
            {
                "overhead_csv": str(run["path"]),
                "overhead_csv_sha256": _sha256(run["path"]),
                "csv_rows": int(run["rows"]),
                "overhead_unique_samples": int(len(run["samples"])),
                "onboard_detected_samples": int(len(run["front"])),
                "fit_inliers": int(np.count_nonzero(selected)),
                "fit_rmse_m": float(np.sqrt(np.mean(values**2))),
                "fit_median_m": float(np.median(values)),
                "fit_p95_m": float(np.quantile(values, 0.95)),
                "overhead_motion_range_xyz_m": np.ptp(
                    run["samples"][:, 1:4], axis=0
                ).tolist(),
                "onboard_motion_range_xyz_m": np.ptp(
                    run["front"][:, 1:4], axis=0
                ).tolist(),
            }
        )

    finite_nis = diagnostics[:, 1][np.isfinite(diagnostics[:, 1])]
    return {
        "method": {
            "pairing": "linear interpolation of overhead truth at onboard frame timestamps",
            "lag_convention": "truth_time = onboard_frame_time + lag; negative means onboard is delayed relative to overhead",
            "fit": "Kabsch rigid transform with iterative median/MAD outlier rejection",
            "overhead_output_axis_correction_diag": [-1.0, -1.0, 1.0],
            "axis_correction_reason": (
                "raw fitted motion showed x/y sign inversion (approximately 180 deg about "
                "optical z) between the existing overhead-derived left-camera topic and "
                "the onboard OpenCV output; correction is applied only in this MPC-side analysis"
            ),
        },
        "sources": {
            "front_jsonl": str(front_jsonl),
            "front_jsonl_sha256": _sha256(front_jsonl),
            "front_snapshot_records": snapshot_records,
            "front_external_source": (
                str(original_front_jsonl) if front_snapshot is not None else None
            ),
            "runs": run_results,
        },
        "fit": {
            "paired_samples": int(len(reference)),
            "inlier_samples": int(np.count_nonzero(keep)),
            "inlier_fraction": float(np.mean(keep)),
            "onboard_effective_delay_vs_overhead_s": float(-lag_s),
            "best_lag_s": lag_s,
            "rmse_m": rmse,
            "median_m": float(np.median(residual[keep])),
            "p95_m": float(np.quantile(residual[keep], 0.95)),
            "singular_values": singular_values.tolist(),
            "raw_rotation_onboard_from_overhead_topic": raw_rotation.tolist(),
            "raw_rotation_angle_deg": _rotation_angle_deg(raw_rotation),
            "raw_translation_onboard_from_overhead_topic_m": raw_translation.tolist(),
            "corrected_rotation_onboard_from_reference_optical": (
                corrected_reference_to_onboard.tolist()
            ),
            "corrected_rotation_angle_deg": _rotation_angle_deg(
                corrected_reference_to_onboard
            ),
        },
        "mpc_camera_candidate": {
            "coordinate_input": "OpenCV [right, down, forward] m",
            "coordinate_output": "body FRD [forward, right, down] m",
            "rotation_body_from_camera": rotation_body_from_onboard.tolist(),
            "rotation_deviation_from_axis_aligned_deg": _rotation_angle_deg(
                nominal_error_rotation
            ),
            "camera_origin_in_body_frd_m": [0.2609931, 0.0077881, -0.1236508],
            "camera_origin_source": (
                "CAD candidate with independent two-distance forward-origin cross-check; "
                "not refit here because overhead target-center offset is approximate"
            ),
            "enabled_for_control": False,
            "reason_disabled": (
                "rotation is a useful candidate, but the overhead-derived reference topic "
                "required an inferred x/y axis correction and the run-specific transform "
                "is not yet an independent as-built absolute reference"
            ),
        },
        "mpc_input_gate_evidence": {
            "finite_depth_nis_samples": int(len(finite_nis)),
            "depth_nis_over_25_samples": int(np.count_nonzero(finite_nis > 25.0)),
            "max_depth_nis": float(np.max(finite_nis)),
            "observed_false_depth_excursion_m": [1.8408148, 1.8528523],
            "normal_depth_cluster_near_excursion_m": [0.336, 0.483],
            "recommended_max_depth_nis": 25.0,
            "recommended_reacquire_samples": 5,
            "frozen_log_replay": _replay_mpc_gate(front_jsonl),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-jsonl", type=Path, required=True)
    parser.add_argument("--overhead-csv", type=Path, action="append", required=True)
    parser.add_argument("--front-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze(
        args.front_jsonl,
        args.overhead_csv,
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
