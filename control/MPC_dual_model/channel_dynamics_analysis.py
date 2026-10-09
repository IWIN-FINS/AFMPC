"""Fit a preliminary horizontal axis model from guarded channel-step logs.

This analyzer deliberately keeps the result out of the control configuration.
The current-vehicle RPM is converted to force with a disabled historical
load-cell prior, and the vehicle is tethered, so the fit is only a candidate
for planning a later untethered/force-instrumented identification.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _motor_force_samples(
    path: Path,
    config: dict[str, Any],
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    prior = config["thruster_feedback"]["rpm_force_prior"]
    c_positive = np.asarray(
        prior["c1_positive_n_per_rad_s_sq_m1_m8"], dtype=float
    )
    c_negative = np.asarray(
        prior["c1_negative_abs_n_per_rad_s_sq_m1_m8"], dtype=float
    )
    directions = np.asarray(
        config["thruster_geometry"][
            "positive_throttle_force_directions_frd_m1_m8"
        ],
        dtype=float,
    )
    times: list[float] = []
    phases: list[str] = []
    rpm_samples: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("telemetry_fresh") != "1" or not row.get(
                "motor_rpm_m1_m8"
            ):
                continue
            rpm = np.asarray(ast.literal_eval(row["motor_rpm_m1_m8"]), dtype=float)
            times.append(float(row["host_monotonic_s"]))
            phases.append(row["phase"])
            rpm_samples.append(rpm)
    if len(times) < 20:
        raise RuntimeError("motor CSV has too few fresh RPM telemetry samples")
    raw_rpm = np.asarray(rpm_samples, dtype=float)
    filtered_rpm = raw_rpm.copy()
    replaced_count = 0
    # DSHOT telemetry occasionally reports an isolated 1200+ RPM point among
    # otherwise stable 700/900 RPM samples. Replace only local outliers; the
    # original CSV remains untouched and transitions with a local majority are
    # retained.
    for row_index in range(len(raw_rpm)):
        start = max(0, row_index - 2)
        end = min(len(raw_rpm), row_index + 3)
        local_median = np.median(raw_rpm[start:end], axis=0)
        threshold = np.maximum(150.0, 0.30 * np.abs(local_median))
        outlier = np.abs(raw_rpm[row_index] - local_median) > threshold
        filtered_rpm[row_index, outlier] = local_median[outlier]
        replaced_count += int(np.count_nonzero(outlier))
    omega = filtered_rpm * (2.0 * math.pi / 60.0)
    coefficient = np.where(filtered_rpm >= 0.0, c_positive, c_negative)
    signed_thrust = np.sign(filtered_rpm) * coefficient * omega * omega
    forces = signed_thrust @ directions[:, axis_index]
    return (
        np.asarray(times, dtype=float),
        np.asarray(forces, dtype=float),
        np.asarray(phases, dtype=object),
        replaced_count,
    )


def _camera_samples(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen_status: set[str] = set()
    times: list[float] = []
    centers: list[np.ndarray] = []
    forwards: list[np.ndarray] = []
    separations: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status_count = row.get("vision_status_count", "")
            if (
                not status_count
                or status_count in seen_status
                or not row.get("vision_stamp_s")
            ):
                continue
            seen_status.add(status_count)
            details = json.loads(row.get("vision_detections_detail") or "[]")
            tags = {
                int(item["tag_id"]): item
                for item in details
                if item.get("pnp_valid") and item.get("pnp_camera_xyz")
            }
            if 15 not in tags or 16 not in tags:
                continue
            p15 = np.asarray(tags[15]["pnp_camera_xyz"], dtype=float)
            p16 = np.asarray(tags[16]["pnp_camera_xyz"], dtype=float)
            forward = p15[:2] - p16[:2]
            norm = float(np.linalg.norm(forward))
            if norm < 1.0e-6:
                continue
            host_epoch = datetime.fromisoformat(row["host_time_utc"]).timestamp()
            host_monotonic = float(row["host_monotonic_s"])
            acquisition_monotonic = float(row["vision_stamp_s"]) - (
                host_epoch - host_monotonic
            )
            times.append(acquisition_monotonic)
            centers.append((p15 + p16) * 0.5)
            forwards.append(forward / norm)
            separations.append(float(np.linalg.norm(p15 - p16)))
    if len(times) < 40:
        raise RuntimeError("camera CSV has too few valid simultaneous tag 15/16 samples")
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(centers, dtype=float)[order],
        np.asarray(forwards, dtype=float)[order],
        np.asarray(separations, dtype=float)[order],
    )


def _quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid pose quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _refracted_pose_samples(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen_pose_counts: set[str] = set()
    times: list[float] = []
    centers: list[np.ndarray] = []
    forwards: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pose_count = row.get("vision_count", "")
            if (
                row.get("vision_fresh") != "1"
                or row.get("refracted_valid") != "1"
                or not pose_count
                or pose_count in seen_pose_counts
                or not row.get("vision_stamp_s")
                or row.get("vision_frame_id") != "pool_world"
            ):
                continue
            seen_pose_counts.add(pose_count)
            position = np.asarray(
                ast.literal_eval(row["vision_position_xyz_m"]), dtype=float
            )
            quaternion = np.asarray(
                ast.literal_eval(row["vision_quat_xyzw"]), dtype=float
            )
            if position.shape != (3,) or quaternion.shape != (4,):
                continue
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
                continue
            try:
                rotation_world_from_body = _quaternion_matrix_xyzw(quaternion)
            except ValueError:
                continue
            forward = rotation_world_from_body[:2, 0]
            right = rotation_world_from_body[:2, 1]
            if np.linalg.norm(forward) < 1.0e-6 or np.linalg.norm(right) < 1.0e-6:
                continue
            host_epoch = datetime.fromisoformat(row["host_time_utc"]).timestamp()
            host_monotonic = float(row["host_monotonic_s"])
            acquisition_monotonic = float(row["vision_stamp_s"]) - (
                host_epoch - host_monotonic
            )
            times.append(acquisition_monotonic)
            centers.append(position)
            forwards.append(forward / np.linalg.norm(forward))
            rights.append(right / np.linalg.norm(right))
    if len(times) < 40:
        raise RuntimeError("camera CSV has too few valid refracted pool-world poses")
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(centers, dtype=float)[order],
        np.asarray(forwards, dtype=float)[order],
        np.asarray(rights, dtype=float)[order],
    )


def _basis_response(
    mass: float,
    damping: float,
    observation_times: np.ndarray,
    start: float,
    end: float,
    motor_times: np.ndarray,
    motor_force: np.ndarray,
    integration_dt: float = 0.01,
) -> np.ndarray:
    """Return forced, initial-velocity, and unit-bias position responses."""

    count = int(math.ceil((end - start) / integration_dt)) + 1
    grid = start + np.arange(count, dtype=float) * integration_dt
    force = np.interp(grid, motor_times, motor_force)
    forced_s = forced_v = 0.0
    velocity_s = 0.0
    velocity_v = 1.0
    bias_s = bias_v = 0.0
    output: list[tuple[float, float, float]] = []
    observation_index = 0
    decay = math.exp(-damping * integration_dt / mass)
    velocity_gain = (1.0 - decay) / damping
    position_velocity_gain = mass * (1.0 - decay) / damping
    position_force_gain = (
        integration_dt / damping
        - mass * (1.0 - decay) / (damping * damping)
    )
    for index, grid_time in enumerate(grid):
        while (
            observation_index < len(observation_times)
            and observation_times[observation_index] <= grid_time + integration_dt * 0.5
        ):
            output.append((forced_s, velocity_s, bias_s))
            observation_index += 1
        if index == len(grid) - 1:
            break
        current_force = float(force[index])
        forced_s += position_velocity_gain * forced_v + position_force_gain * current_force
        forced_v = decay * forced_v + velocity_gain * current_force
        velocity_s += position_velocity_gain * velocity_v
        velocity_v = decay * velocity_v
        bias_s += position_velocity_gain * bias_v + position_force_gain
        bias_v = decay * bias_v + velocity_gain
    if len(output) != len(observation_times):
        raise RuntimeError("failed to align all camera observations to the integration grid")
    return np.asarray(output, dtype=float)


def _fit_for_mass_damping(
    mass: float,
    damping: float,
    segments: list[tuple[np.ndarray, np.ndarray, float, float]],
    motor_times: np.ndarray,
    motor_force: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    parameter_count = 2 * len(segments) + 1
    for segment_index, (times, positions, start, end) in enumerate(segments):
        response = _basis_response(
            mass,
            damping,
            times,
            start,
            end,
            motor_times,
            motor_force,
        )
        target = positions - response[:, 0]
        matrix = np.zeros((len(times), parameter_count), dtype=float)
        matrix[:, segment_index] = 1.0
        matrix[:, len(segments) + segment_index] = response[:, 1]
        matrix[:, -1] = response[:, 2]
        targets.append(target)
        matrices.append(matrix)
    target = np.concatenate(targets)
    matrix = np.vstack(matrices)
    coefficient, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficient
    rmse = math.sqrt(float(np.mean(residual * residual)))
    return rmse, coefficient, residual


def _grid_fit(
    segments: list[tuple[np.ndarray, np.ndarray, float, float]],
    motor_times: np.ndarray,
    motor_force: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    best = (float("inf"), 0.0, 0.0)
    for mass in np.geomspace(0.5, 80.0, 42):
        for damping in np.geomspace(0.2, 150.0, 48):
            rmse, _, _ = _fit_for_mass_damping(
                float(mass),
                float(damping),
                segments,
                motor_times,
                motor_force,
            )
            if rmse < best[0]:
                best = (rmse, float(mass), float(damping))
    for _ in range(2):
        _, best_mass, best_damping = best
        for mass in np.linspace(best_mass * 0.70, best_mass * 1.30, 25):
            for damping in np.linspace(best_damping * 0.70, best_damping * 1.30, 25):
                if mass <= 0.0 or damping <= 0.0:
                    continue
                rmse, _, _ = _fit_for_mass_damping(
                    float(mass),
                    float(damping),
                    segments,
                    motor_times,
                    motor_force,
                )
                if rmse < best[0]:
                    best = (rmse, float(mass), float(damping))
    rmse, mass, damping = best
    _, coefficient, residual = _fit_for_mass_damping(
        mass,
        damping,
        segments,
        motor_times,
        motor_force,
    )
    return rmse, mass, damping, coefficient, residual


def _fit_for_mass_damping_zero_initial_velocity(
    mass: float,
    damping: float,
    segments: list[tuple[np.ndarray, np.ndarray, float, float]],
    motor_times: np.ndarray,
    motor_force: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    parameter_count = len(segments) + 1
    for segment_index, (times, positions, start, end) in enumerate(segments):
        response = _basis_response(
            mass,
            damping,
            times,
            start,
            end,
            motor_times,
            motor_force,
        )
        target = positions - response[:, 0]
        matrix = np.zeros((len(times), parameter_count), dtype=float)
        matrix[:, segment_index] = 1.0
        matrix[:, -1] = response[:, 2]
        targets.append(target)
        matrices.append(matrix)
    target = np.concatenate(targets)
    matrix = np.vstack(matrices)
    coefficient, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficient
    rmse = math.sqrt(float(np.mean(residual * residual)))
    return rmse, coefficient, residual


def _grid_fit_zero_initial_velocity(
    segments: list[tuple[np.ndarray, np.ndarray, float, float]],
    motor_times: np.ndarray,
    motor_force: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    best = (float("inf"), 0.0, 0.0)
    for mass in np.geomspace(0.5, 80.0, 42):
        for damping in np.geomspace(0.2, 150.0, 48):
            rmse, _, _ = _fit_for_mass_damping_zero_initial_velocity(
                float(mass), float(damping), segments, motor_times, motor_force
            )
            if rmse < best[0]:
                best = (rmse, float(mass), float(damping))
    for _ in range(2):
        _, best_mass, best_damping = best
        for mass in np.linspace(best_mass * 0.70, best_mass * 1.30, 25):
            for damping in np.linspace(best_damping * 0.70, best_damping * 1.30, 25):
                if mass <= 0.0 or damping <= 0.0:
                    continue
                rmse, _, _ = _fit_for_mass_damping_zero_initial_velocity(
                    float(mass),
                    float(damping),
                    segments,
                    motor_times,
                    motor_force,
                )
                if rmse < best[0]:
                    best = (rmse, float(mass), float(damping))
    rmse, mass, damping = best
    _, coefficient, residual = _fit_for_mass_damping_zero_initial_velocity(
        mass, damping, segments, motor_times, motor_force
    )
    return rmse, mass, damping, coefficient, residual


def analyze(
    camera_csv: Path,
    motor_csv: Path,
    config_path: Path,
    axis: str,
    position_source: str = "raw_tags",
    include_neutral_coast: bool = False,
) -> dict[str, Any]:
    axis_index = {"forward": 0, "right": 1}[axis]
    motor_times, motor_force, motor_phases, rpm_outlier_count = _motor_force_samples(
        motor_csv, _load_config(config_path), axis_index
    )
    if position_source == "raw_tags":
        camera_times, centers, forward_vectors, separations = _camera_samples(
            camera_csv
        )
        reference_forward = np.mean(
            forward_vectors[: min(20, len(forward_vectors))], axis=0
        )
        reference_forward /= np.linalg.norm(reference_forward)
        reference_axis = (
            reference_forward
            if axis == "forward"
            # The OpenCV camera frame is x-right, y-down, z-forward/down. Its
            # horizontal x/y basis has the same handedness as body FRD.
            else np.asarray([-reference_forward[1], reference_forward[0]], dtype=float)
        )
        camera_description = (
            "midpoint of tag 15/16 raw pinhole PnP positions in fixed camera "
            "frame, projected on initial body axis"
        )
        refraction_corrected = False
        separation_mean = float(np.mean(separations))
        separation_std = float(np.std(separations))
    elif position_source == "refracted_pose":
        camera_times, centers, forward_vectors, right_vectors = (
            _refracted_pose_samples(camera_csv)
        )
        axis_vectors = forward_vectors if axis == "forward" else right_vectors
        reference_axis = np.mean(axis_vectors[: min(20, len(axis_vectors))], axis=0)
        reference_axis /= np.linalg.norm(reference_axis)
        camera_description = (
            "refracted_pose_6d pool_world position projected on the initial body "
            "forward/right axis taken directly from its pose quaternion"
        )
        refraction_corrected = True
        separation_mean = None
        separation_std = None
    else:
        raise ValueError("position_source must be raw_tags or refracted_pose")
    positions = centers[:, :2] @ reference_axis

    segments: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    segment_summary: list[dict[str, Any]] = []
    for phase_suffix in ("positive", "negative"):
        matching = np.asarray(
            [phase.endswith(phase_suffix) for phase in motor_phases], dtype=bool
        )
        if not np.any(matching):
            raise RuntimeError(f"motor CSV has no {phase_suffix} excitation phase")
        phase_times = motor_times[matching]
        start = float(np.min(phase_times) + 0.10)
        if include_neutral_coast:
            active_indices = np.flatnonzero(matching)
            end_index = int(active_indices[-1])
            while end_index + 1 < len(motor_phases):
                next_phase = str(motor_phases[end_index + 1])
                if next_phase.endswith("positive") or next_phase.endswith("negative"):
                    break
                end_index += 1
            end = float(motor_times[end_index] - 0.10)
        else:
            end = float(np.max(phase_times) - 0.10)
        camera_mask = (camera_times >= start) & (camera_times <= end)
        if int(np.count_nonzero(camera_mask)) < 20:
            raise RuntimeError(f"too few camera samples in {phase_suffix} phase")
        times = camera_times[camera_mask]
        phase_positions = positions[camera_mask]
        segments.append((times, phase_positions, start, end))
        segment_summary.append(
            {
                "phase": phase_suffix,
                "sample_count": int(len(times)),
                "duration_s": float(times[-1] - times[0]),
                "net_axis_displacement_m": float(
                    np.mean(phase_positions[-min(10, len(times)) :])
                    - np.mean(phase_positions[: min(10, len(times))])
                ),
            }
        )

    rmse, mass, damping, coefficient, residual = _grid_fit(
        segments, motor_times, motor_force
    )
    force_active = np.asarray(
        [
            value
            for value, phase in zip(motor_force, motor_phases)
            if phase.endswith("positive") or phase.endswith("negative")
        ],
        dtype=float,
    )
    result = {
        "status": "candidate_only_not_enabled_for_control",
        "axis_frd": axis,
        "model": "F_axis = M_eff * dv/dt + D_linear * v + bias",
        "method": (
            "Exact position-domain integration at 0.01 s; grid search over positive "
            "M_eff/D_linear; independent initial position/velocity per positive and "
            "negative phase; shared force bias. "
            + (
                "Each excitation includes its following zero-command coast. "
                if include_neutral_coast
                else "Only the active excitation windows are fitted. "
            )
            + "Camera and RPM are aligned by the "
            "camera header timestamp on the host monotonic clock."
        ),
        "source": {
            "camera_csv": str(camera_csv),
            "camera_csv_sha256": _sha256(camera_csv),
            "motor_csv": str(motor_csv),
            "motor_csv_sha256": _sha256(motor_csv),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "camera": {
            "position_source": position_source,
            "coordinate": camera_description,
            "valid_sample_count": int(len(camera_times)),
            "tag_15_16_separation_mean_m": separation_mean,
            "tag_15_16_separation_std_m": separation_std,
            "underwater_refraction_corrected": refraction_corrected,
        },
        "force": {
            "source": "current-vehicle measured RPM converted by disabled historical FinsSim load-cell c1 prior",
            "rpm_isolated_outlier_filter": "5-sample local median; replace deviations greater than max(150 RPM, 30% of local median)",
            "rpm_values_replaced": rpm_outlier_count,
            "active_min_n": float(np.min(force_active)),
            "active_max_n": float(np.max(force_active)),
            "current_vehicle_load_cell_measured": False,
        },
        "segments": segment_summary,
        "fit": {
            "effective_mass_kg": mass,
            "linear_damping_n_per_m_s": damping,
            "bias_n": float(coefficient[-1]),
            "initial_velocity_m_s_positive_negative": [
                float(value) for value in coefficient[len(segments) : -1]
            ],
            "position_rmse_m": rmse,
            "position_abs_error_p95_m": float(np.percentile(np.abs(residual), 95.0)),
            "quadratic_damping_identified": False,
        },
        "confidence": {
            "level": "low",
            "enabled_for_control": False,
            "reasons": [
                "force scale is a historical curve prior, not a current-vehicle load-cell measurement",
                "the ROV is tethered, so external tether force is not separated from hydrodynamic damping",
                *(
                    ["raw underwater pinhole PnP is not refraction corrected"]
                    if position_source == "raw_tags"
                    else []
                ),
                "one command amplitude cannot independently identify linear and quadratic damping",
            ],
        },
    }
    return result


def analyze_split_refracted_runs(
    positive_camera_csv: Path,
    positive_motor_csv: Path,
    negative_camera_csv: Path,
    negative_motor_csv: Path,
    config_path: Path,
    axis: str,
    *,
    include_neutral_coast: bool = True,
    constrain_initial_velocity_zero: bool = False,
) -> dict[str, Any]:
    """Fit signed segments recorded in separate synchronized ROS/MPC runs."""

    axis_index = {"forward": 0, "right": 1}[axis]
    config = _load_config(config_path)
    segments: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    segment_summary: list[dict[str, Any]] = []
    motor_time_parts: list[np.ndarray] = []
    motor_force_parts: list[np.ndarray] = []
    active_force_parts: list[np.ndarray] = []
    rpm_outlier_count = 0
    camera_sample_count = 0
    yaw_ranges_deg: list[float] = []

    for phase_suffix, camera_csv, motor_csv in (
        ("positive", positive_camera_csv, positive_motor_csv),
        ("negative", negative_camera_csv, negative_motor_csv),
    ):
        motor_times, motor_force, motor_phases, replaced = _motor_force_samples(
            motor_csv, config, axis_index
        )
        camera_times, centers, forward_vectors, right_vectors = (
            _refracted_pose_samples(camera_csv)
        )
        axis_vectors = forward_vectors if axis == "forward" else right_vectors
        reference_axis = np.mean(axis_vectors[: min(20, len(axis_vectors))], axis=0)
        reference_axis /= np.linalg.norm(reference_axis)
        positions = centers[:, :2] @ reference_axis
        matching = np.asarray(
            [str(phase).endswith(phase_suffix) for phase in motor_phases],
            dtype=bool,
        )
        if not np.any(matching):
            raise RuntimeError(
                f"{phase_suffix} motor CSV has no {phase_suffix} excitation phase"
            )
        active_indices = np.flatnonzero(matching)
        end_index = int(active_indices[-1])
        if include_neutral_coast:
            while end_index + 1 < len(motor_phases):
                next_phase = str(motor_phases[end_index + 1])
                if next_phase.endswith("positive") or next_phase.endswith("negative"):
                    break
                end_index += 1
        integration_start = float(motor_times[active_indices[0]])
        observation_start = integration_start + 0.10
        end = float(motor_times[end_index] - 0.10)
        camera_mask = (camera_times >= observation_start) & (camera_times <= end)
        if int(np.count_nonzero(camera_mask)) < 20:
            raise RuntimeError(
                f"too few refracted pose samples in split {phase_suffix} phase"
            )
        times = camera_times[camera_mask]
        phase_positions = positions[camera_mask]
        segment_axis_vectors = axis_vectors[camera_mask]
        axis_angles = np.unwrap(
            np.arctan2(segment_axis_vectors[:, 1], segment_axis_vectors[:, 0])
        )
        yaw_range_deg = float(np.ptp(axis_angles) * 180.0 / math.pi)
        yaw_ranges_deg.append(yaw_range_deg)
        segments.append((times, phase_positions, integration_start, end))
        segment_summary.append(
            {
                "phase": phase_suffix,
                "camera_csv": str(camera_csv),
                "motor_csv": str(motor_csv),
                "sample_count": int(len(times)),
                "duration_s": float(times[-1] - times[0]),
                "net_axis_displacement_m": float(
                    phase_positions[-1] - phase_positions[0]
                ),
                "axis_position_range_m": float(np.ptp(phase_positions)),
                "body_axis_angle_range_deg": yaw_range_deg,
            }
        )
        motor_time_parts.append(motor_times)
        motor_force_parts.append(motor_force)
        active_force_parts.append(motor_force[matching])
        rpm_outlier_count += replaced
        camera_sample_count += len(camera_times)

    motor_times = np.concatenate(motor_time_parts)
    motor_force = np.concatenate(motor_force_parts)
    order = np.argsort(motor_times)
    motor_times = motor_times[order]
    motor_force = motor_force[order]
    active_force = np.concatenate(active_force_parts)
    if constrain_initial_velocity_zero:
        rmse, mass, damping, coefficient, residual = _grid_fit_zero_initial_velocity(
            segments, motor_times, motor_force
        )
    else:
        rmse, mass, damping, coefficient, residual = _grid_fit(
            segments, motor_times, motor_force
        )
    initial_velocity_method = (
        "independent initial position and zero initial velocity, constrained by "
        "each preceding 10 s neutral baseline"
        if constrain_initial_velocity_zero
        else "independent initial position and velocity per segment"
    )
    window_method = (
        "Each excitation includes its following zero-command coast."
        if include_neutral_coast
        else "Only active excitation windows are included."
    )
    return {
        "status": "candidate_only_not_enabled_for_control",
        "axis_frd": axis,
        "model": "F_axis = M_eff * dv/dt + D_linear * v + bias",
        "method": (
            "Positive and negative excitations were captured in separate synchronized "
            "runs. Refracted pool-world pose is projected on each run's initial body "
            "axis. Exact 0.01 s position-domain integration uses shared M_eff, "
            f"D_linear and force bias with {initial_velocity_method}. "
            f"{window_method}"
        ),
        "source": {
            "positive_camera_csv": str(positive_camera_csv),
            "positive_camera_csv_sha256": _sha256(positive_camera_csv),
            "positive_motor_csv": str(positive_motor_csv),
            "positive_motor_csv_sha256": _sha256(positive_motor_csv),
            "negative_camera_csv": str(negative_camera_csv),
            "negative_camera_csv_sha256": _sha256(negative_camera_csv),
            "negative_motor_csv": str(negative_motor_csv),
            "negative_motor_csv_sha256": _sha256(negative_motor_csv),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "camera": {
            "position_source": "refracted_pose",
            "coordinate": "pool_world position projected on each run's initial body axis from pose quaternion",
            "valid_sample_count_all_runs": int(camera_sample_count),
            "underwater_refraction_corrected": True,
        },
        "force": {
            "source": "current-vehicle RPM converted by accepted same-vehicle low-limit RPM-force prior",
            "rpm_values_replaced": int(rpm_outlier_count),
            "active_min_n": float(np.min(active_force)),
            "active_max_n": float(np.max(active_force)),
            "current_vehicle_load_cell_measured": True,
            "vertical_curve_transfer_used": False,
        },
        "segments": segment_summary,
        "fit": {
            "effective_mass_kg": float(mass),
            "linear_damping_n_per_m_s": float(damping),
            "bias_n": float(coefficient[-1]),
            "initial_velocity_constraint": (
                "zero_from_preceding_10_s_neutral_baseline"
                if constrain_initial_velocity_zero
                else "free_per_segment"
            ),
            "initial_velocity_m_s_positive_negative": (
                [0.0] * len(segments)
                if constrain_initial_velocity_zero
                else [
                    float(value)
                    for value in coefficient[len(segments) : -1]
                ]
            ),
            "position_rmse_m": float(rmse),
            "position_abs_error_p95_m": float(
                np.percentile(np.abs(residual), 95.0)
            ),
            "quadratic_damping_identified": False,
        },
        "confidence": {
            "level": "medium candidate",
            "enabled_for_control": False,
            "reasons": [
                "the ROV is tethered, so external tether force is not separated from hydrodynamic damping",
                f"body-axis angle varied by up to {max(yaw_ranges_deg):.2f} deg during a fitted segment",
                "one command amplitude cannot independently identify linear and quadratic damping",
            ],
        },
    }


def analyze_split_window_sensitivity(
    positive_camera_csv: Path,
    positive_motor_csv: Path,
    negative_camera_csv: Path,
    negative_motor_csv: Path,
    config_path: Path,
    axis: str,
    durations_s: list[float],
) -> dict[str, Any]:
    """Show whether a zero-initial-velocity fit is stable across window lengths."""

    if not durations_s or any(duration <= 0.2 for duration in durations_s):
        raise ValueError("window sensitivity durations must all exceed 0.2 s")
    axis_index = {"forward": 0, "right": 1}[axis]
    config = _load_config(config_path)
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    motor_time_parts: list[np.ndarray] = []
    motor_force_parts: list[np.ndarray] = []
    for phase_suffix, camera_csv, motor_csv in (
        ("positive", positive_camera_csv, positive_motor_csv),
        ("negative", negative_camera_csv, negative_motor_csv),
    ):
        camera_times, centers, forward_vectors, right_vectors = (
            _refracted_pose_samples(camera_csv)
        )
        motor_times, motor_force, motor_phases, _ = _motor_force_samples(
            motor_csv, config, axis_index
        )
        axis_vectors = forward_vectors if axis == "forward" else right_vectors
        reference_axis = np.mean(axis_vectors[: min(20, len(axis_vectors))], axis=0)
        reference_axis /= np.linalg.norm(reference_axis)
        positions = centers[:, :2] @ reference_axis
        matching = np.asarray(
            [str(phase).endswith(phase_suffix) for phase in motor_phases],
            dtype=bool,
        )
        active_indices = np.flatnonzero(matching)
        if len(active_indices) == 0:
            raise RuntimeError(f"missing {phase_suffix} excitation")
        start = float(motor_times[active_indices[0]])
        runs.append((camera_times, positions, motor_times, motor_force, start))
        motor_time_parts.append(motor_times)
        motor_force_parts.append(motor_force)

    all_motor_times = np.concatenate(motor_time_parts)
    all_motor_force = np.concatenate(motor_force_parts)
    order = np.argsort(all_motor_times)
    all_motor_times = all_motor_times[order]
    all_motor_force = all_motor_force[order]
    fits: list[dict[str, Any]] = []
    for duration_s in sorted(set(float(value) for value in durations_s)):
        segments: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for camera_times, positions, _, _, start in runs:
            end = start + duration_s
            mask = (camera_times >= start + 0.10) & (camera_times <= end)
            if int(np.count_nonzero(mask)) < 20:
                raise RuntimeError(
                    f"too few camera samples for {duration_s:.3f} s window"
                )
            segments.append((camera_times[mask], positions[mask], start, end))
        rmse, mass, damping, coefficient, residual = (
            _grid_fit_zero_initial_velocity(
                segments, all_motor_times, all_motor_force
            )
        )
        fits.append(
            {
                "window_duration_s": duration_s,
                "effective_mass_kg": float(mass),
                "linear_damping_n_per_m_s": float(damping),
                "time_constant_s": float(mass / damping),
                "bias_n": float(coefficient[-1]),
                "position_rmse_m": float(rmse),
                "position_abs_error_p95_m": float(
                    np.percentile(np.abs(residual), 95.0)
                ),
                "sample_count": int(sum(len(segment[0]) for segment in segments)),
            }
        )
    masses = [item["effective_mass_kg"] for item in fits]
    dampings = [item["linear_damping_n_per_m_s"] for item in fits]
    time_constants = [item["time_constant_s"] for item in fits]
    return {
        "status": "window_sensitive_rejected_as_control_parameter",
        "axis_frd": axis,
        "model": "F_axis = M_eff * dv/dt + D_linear * v + bias",
        "method": (
            "Bidirectional split runs, refracted pool-world pose, zero velocity at "
            "pulse onset after each 10 s neutral baseline. The same shared-parameter "
            "fit is repeated with progressively longer pulse-plus-coast windows."
        ),
        "source": {
            "positive_camera_csv": str(positive_camera_csv),
            "positive_camera_csv_sha256": _sha256(positive_camera_csv),
            "positive_motor_csv": str(positive_motor_csv),
            "positive_motor_csv_sha256": _sha256(positive_motor_csv),
            "negative_camera_csv": str(negative_camera_csv),
            "negative_camera_csv_sha256": _sha256(negative_camera_csv),
            "negative_motor_csv": str(negative_motor_csv),
            "negative_motor_csv_sha256": _sha256(negative_motor_csv),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "fits": fits,
        "stability": {
            "effective_mass_range_kg": [float(min(masses)), float(max(masses))],
            "linear_damping_range_n_per_m_s": [
                float(min(dampings)),
                float(max(dampings)),
            ],
            "time_constant_range_s": [
                float(min(time_constants)),
                float(max(time_constants)),
            ],
            "stable_enough_for_control": False,
            "reason": (
                "M_eff rises and D_linear falls monotonically as late slow drift is "
                "included; a single linear free-decay model is not window-invariant."
            ),
        },
        "enabled_for_control": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a preliminary horizontal dynamic model from channel-step CSVs"
    )
    parser.add_argument("--camera-csv", required=True)
    parser.add_argument("--motor-csv", required=True)
    parser.add_argument("--negative-camera-csv")
    parser.add_argument("--negative-motor-csv")
    parser.add_argument(
        "--config", default="MPC_dual_model/finesub_v4pro1_mpc.json"
    )
    parser.add_argument("--axis", choices=("forward", "right"), required=True)
    parser.add_argument(
        "--position-source",
        choices=("raw_tags", "refracted_pose"),
        default="raw_tags",
    )
    parser.add_argument(
        "--include-neutral-coast",
        action="store_true",
        help="include the zero-command phase after each signed excitation",
    )
    parser.add_argument(
        "--constrain-initial-velocity-zero",
        action="store_true",
        help="for split runs, impose zero velocity at pulse onset after a long neutral baseline",
    )
    parser.add_argument(
        "--window-sensitivity-durations",
        help="comma-separated split-run window lengths in seconds",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split_arguments = (args.negative_camera_csv, args.negative_motor_csv)
    if any(split_arguments) and not all(split_arguments):
        parser.error(
            "--negative-camera-csv and --negative-motor-csv must be supplied together"
        )
    result = (
        analyze_split_window_sensitivity(
            Path(args.camera_csv),
            Path(args.motor_csv),
            Path(args.negative_camera_csv),
            Path(args.negative_motor_csv),
            Path(args.config),
            args.axis,
            [
                float(value)
                for value in args.window_sensitivity_durations.split(",")
                if value.strip()
            ],
        )
        if all(split_arguments) and args.window_sensitivity_durations
        else analyze_split_refracted_runs(
            Path(args.camera_csv),
            Path(args.motor_csv),
            Path(args.negative_camera_csv),
            Path(args.negative_motor_csv),
            Path(args.config),
            args.axis,
            include_neutral_coast=args.include_neutral_coast,
            constrain_initial_velocity_zero=args.constrain_initial_velocity_zero,
        )
        if all(split_arguments)
        else analyze(
            Path(args.camera_csv),
            Path(args.motor_csv),
            Path(args.config),
            args.axis,
            args.position_source,
            args.include_neutral_coast,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
