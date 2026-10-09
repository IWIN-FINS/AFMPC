"""Offline quality checks for synchronized FineSUB/camera calibration CSVs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _vector(row: dict[str, str], key: str, length: int) -> np.ndarray:
    try:
        value = np.asarray(json.loads(row.get(key, "")), dtype=float).reshape(-1)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.full(length, np.nan)
    if value.size != length or not np.all(np.isfinite(value)):
        return np.full(length, np.nan)
    return value


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"sample_count": 0}
    return {
        "sample_count": int(len(values)),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=1 if len(values) > 1 else 0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "peak_to_peak": np.ptp(values, axis=0).tolist(),
    }


def _quat_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=float,
    )
    if angle < 1e-7:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) < 1e-7:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= max(np.linalg.norm(axis), np.finfo(float).eps)
        return angle * axis
    return angle * skew / (2.0 * sine)


def visual_body_rates(times: np.ndarray, quaternions_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).reshape(-1)
    quaternions = np.asarray(quaternions_xyzw, dtype=float).reshape(-1, 4)
    output_time: list[float] = []
    rates: list[np.ndarray] = []
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if not 0.005 <= dt <= 0.5:
            continue
        previous = _quat_xyzw_to_rotation(quaternions[index - 1])
        current = _quat_xyzw_to_rotation(quaternions[index])
        rates.append(_rotation_vector(previous.T @ current) / dt)
        output_time.append(0.5 * (times[index] + times[index - 1]))
    return np.asarray(output_time, dtype=float), np.asarray(rates, dtype=float).reshape(-1, 3)


def best_signed_axis_mapping(
    reference: np.ndarray,
    measured: np.ndarray,
    *,
    max_lag_rows: int = 20,
) -> dict[str, Any] | None:
    reference = np.asarray(reference, dtype=float).reshape(-1, 3)
    measured = np.asarray(measured, dtype=float).reshape(-1, 3)
    count = min(len(reference), len(measured))
    if count < 20:
        return None
    reference = reference[:count]
    measured = measured[:count]
    best: dict[str, Any] | None = None
    for lag in range(-max_lag_rows, max_lag_rows + 1):
        if lag >= 0:
            ref = reference[lag:]
            obs = measured[: count - lag]
        else:
            ref = reference[: count + lag]
            obs = measured[-lag:]
        if len(ref) < 20:
            continue
        for permutation in itertools.permutations(range(3)):
            permuted = obs[:, permutation]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                predicted = permuted * np.asarray(signs)
                active = np.std(ref, axis=0) > 0.015
                if not np.any(active):
                    continue
                correlations = []
                for axis in range(3):
                    if not active[axis] or np.std(predicted[:, axis]) <= 1e-9:
                        correlations.append(float("nan"))
                    else:
                        correlations.append(float(np.corrcoef(ref[:, axis], predicted[:, axis])[0, 1]))
                score = float(np.nanmean(np.asarray(correlations)[active]))
                rmse = float(np.sqrt(np.mean((ref[:, active] - predicted[:, active]) ** 2)))
                candidate = {
                    "score_mean_correlation": score,
                    "rmse_rad_s": rmse,
                    "lag_rows_positive_means_camera_delayed": lag,
                    "permutation_reference_axes_from_measured": list(permutation),
                    "signs": list(signs),
                    "correlation_per_reference_axis": correlations,
                    "active_reference_axes": active.astype(int).tolist(),
                    "sample_count": int(len(ref)),
                }
                if best is None or (score, -rmse) > (
                    best["score_mean_correlation"],
                    -best["rmse_rad_s"],
                ):
                    best = candidate
    return best


def _unique_rows(rows: list[dict[str, str]], key: str, fresh_key: str) -> list[dict[str, str]]:
    output = []
    previous: str | None = None
    for row in rows:
        if row.get(fresh_key) != "1":
            continue
        current = row.get(key, "")
        if not current or current == previous:
            continue
        previous = current
        output.append(row)
    return output


def analyze_synchronized_csv(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("calibration CSV contains no rows")

    times = np.asarray([_float(row, "host_monotonic_s") for row in rows], dtype=float)
    telemetry_rows = _unique_rows(rows, "telemetry_sequence", "telemetry_fresh")
    vision_rows = _unique_rows(rows, "vision_count", "vision_fresh")

    gyro = np.asarray(
        [_vector(row, "imu_angular_velocity_xyz_rad_s", 3) for row in telemetry_rows],
        dtype=float,
    ).reshape(-1, 3)
    acceleration = np.asarray(
        [_vector(row, "imu_linear_acceleration_xyz_m_s2", 3) for row in telemetry_rows],
        dtype=float,
    ).reshape(-1, 3)
    imu_quaternion = np.asarray(
        [_vector(row, "imu_quat_wxyz", 4) for row in telemetry_rows],
        dtype=float,
    ).reshape(-1, 4)
    depth = np.asarray([_float(row, "depth_m") for row in telemetry_rows], dtype=float)
    pressure = np.asarray([_float(row, "pressure_pa") for row in telemetry_rows], dtype=float)
    valid_imu = np.all(np.isfinite(gyro), axis=1) & np.all(np.isfinite(acceleration), axis=1)
    valid_depth = np.isfinite(depth)

    vision_position = np.asarray(
        [_vector(row, "vision_position_xyz_m", 3) for row in vision_rows], dtype=float
    ).reshape(-1, 3)
    vision_quaternion = np.asarray(
        [_vector(row, "vision_quat_xyzw", 4) for row in vision_rows], dtype=float
    ).reshape(-1, 4)
    vision_times = np.asarray(
        [_float(row, "host_monotonic_s") for row in vision_rows], dtype=float
    )
    valid_vision = np.all(np.isfinite(vision_position), axis=1) & np.all(
        np.isfinite(vision_quaternion), axis=1
    )

    phase_values = sorted({row.get("phase", "") for row in rows})
    static_phase_declared = any(
        "static" in phase.strip().lower() for phase in phase_values
    )
    result: dict[str, Any] = {
        "source_csv": str(source),
        "row_count": len(rows),
        "duration_s": float(np.nanmax(times) - np.nanmin(times)),
        "phase_values": phase_values,
        "safety": {
            "armed_rows": sum(row.get("requested_armed") not in ("", "0") for row in rows),
            "nonzero_requested_channel_rows": sum(
                row.get("requested_channels_forward_right_down_yaw", "")
                not in ("", "[0.0,0.0,0.0,0.0]")
                for row in rows
            ),
        },
        "availability": {
            "telemetry_unique_samples": len(telemetry_rows),
            "vision_unique_samples": len(vision_rows),
            "telemetry_fresh_fraction": float(np.mean([row.get("telemetry_fresh") == "1" for row in rows])),
            "vision_fresh_fraction": float(np.mean([row.get("vision_fresh") == "1" for row in rows])),
        },
        "imu": {
            "angular_velocity_rad_s": _stats(gyro[valid_imu]),
            "linear_acceleration_m_s2": _stats(acceleration[valid_imu]),
            "acceleration_norm_m_s2": _stats(np.linalg.norm(acceleration[valid_imu], axis=1)),
            "quaternion_norm": _stats(
                np.linalg.norm(imu_quaternion[np.all(np.isfinite(imu_quaternion), axis=1)], axis=1)
            ),
        },
        "depth": {
            "depth_m": _stats(depth[valid_depth]),
            "pressure_pa": _stats(pressure[np.isfinite(pressure)]),
        },
        "vision": {
            "position_m": _stats(vision_position[valid_vision]),
            "quaternion_norm": _stats(np.linalg.norm(vision_quaternion[valid_vision], axis=1)),
        },
    }

    mapping = None
    if np.count_nonzero(valid_vision) >= 20 and len(telemetry_rows) >= 20:
        rate_times, rates = visual_body_rates(
            vision_times[valid_vision], vision_quaternion[valid_vision]
        )
        if len(rates) >= 20:
            telemetry_times = np.asarray(
                [_float(row, "host_monotonic_s") for row in telemetry_rows], dtype=float
            )
            telemetry_gyro = gyro
            nearest = np.searchsorted(telemetry_times, rate_times)
            nearest = np.clip(nearest, 1, len(telemetry_times) - 1)
            left = nearest - 1
            choose_left = np.abs(telemetry_times[left] - rate_times) <= np.abs(
                telemetry_times[nearest] - rate_times
            )
            indices = np.where(choose_left, left, nearest)
            measured = telemetry_gyro[indices]
            finite = np.all(np.isfinite(measured), axis=1) & np.all(np.isfinite(rates), axis=1)
            mapping = best_signed_axis_mapping(rates[finite], measured[finite])
            if mapping is not None and len(rate_times) > 1:
                mapping["lag_seconds_estimate"] = float(
                    mapping["lag_rows_positive_means_camera_delayed"]
                    * np.median(np.diff(rate_times))
                )
                mapping["note"] = (
                    "Use only after deliberate roll/pitch/yaw manual excitation; "
                    "a static or single-axis recording cannot identify all axes."
                )
    result["camera_imu_axis_and_delay_candidate"] = mapping

    depth_fit = None
    if np.count_nonzero(valid_vision) >= 20 and len(telemetry_rows) >= 20:
        telemetry_times = np.asarray(
            [_float(row, "host_monotonic_s") for row in telemetry_rows], dtype=float
        )
        nearest = np.searchsorted(telemetry_times, vision_times[valid_vision])
        nearest = np.clip(nearest, 0, len(telemetry_times) - 1)
        paired_depth = depth[nearest]
        paired_z = vision_position[valid_vision, 2]
        finite = np.isfinite(paired_depth) & np.isfinite(paired_z)
        # A static overhead-camera record has nearly constant world z.  Do not
        # accept a numerically perfect but physically meaningless fit in that
        # case; both signals must contain an observable excursion.
        if (
            np.count_nonzero(finite) >= 20
            and np.ptp(paired_depth[finite]) >= 0.05
            and np.ptp(paired_z[finite]) >= 0.05
        ):
            design = np.column_stack((paired_depth[finite], np.ones(np.count_nonzero(finite))))
            slope, intercept = np.linalg.lstsq(design, paired_z[finite], rcond=None)[0]
            predicted = design @ np.asarray([slope, intercept])
            residual = paired_z[finite] - predicted
            total = paired_z[finite] - np.mean(paired_z[finite])
            r2 = 1.0 - float(residual @ residual) / max(float(total @ total), 1e-12)
            depth_fit = {
                "model": "vision_world_z_m = slope * pressure_depth_m + intercept",
                "slope": float(slope),
                "intercept_m": float(intercept),
                "r2": r2,
                "sample_count": int(np.count_nonzero(finite)),
                "expected_pool_world_z_up_slope": -1.0,
            }
    result["camera_depth_sign_scale_candidate"] = depth_fit
    result["gate"] = {
        "static_phase_declared": static_phase_declared,
        "static_bias_estimation_possible": (
            static_phase_declared and len(telemetry_rows) >= 50
        ),
        "vision_noise_estimation_possible": len(vision_rows) >= 20,
        "axis_mapping_possible": mapping is not None
        and sum(mapping.get("active_reference_axes", [])) == 3
        and mapping.get("score_mean_correlation", 0.0) >= 0.8,
        "depth_sign_scale_possible": depth_fit is not None and depth_fit["r2"] >= 0.8,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze synchronized disarmed calibration CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_synchronized_csv(args.csv)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
