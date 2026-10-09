"""Fit a pressure-depth heave candidate from a guarded ``down`` channel step.

The pressure sensor and motor RPM are recorded by the same lower-controller
telemetry packet, so this analysis does not depend on either camera system.
Results remain candidates until repeated amplitudes agree and the tether is
shown not to load the vertical motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .channel_dynamics_analysis import (
    _grid_fit,
    _load_config,
    _motor_force_samples,
    _sha256,
)


_ACTIVE_PHASE = re.compile(r"^cycle_(\d+)_(positive|negative)$")


def _depth_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seen_sequences: set[str] = set()
    times: list[float] = []
    depths: list[float] = []
    phases: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("telemetry_sequence", "")
            if (
                row.get("telemetry_fresh") != "1"
                or not sequence
                or sequence in seen_sequences
                or not row.get("depth_m")
            ):
                continue
            seen_sequences.add(sequence)
            time_s = float(row["host_monotonic_s"])
            depth_m = float(row["depth_m"])
            if not math.isfinite(time_s) or not math.isfinite(depth_m):
                continue
            times.append(time_s)
            depths.append(depth_m)
            phases.append(row.get("phase", ""))
    if len(times) < 40:
        raise RuntimeError("motor CSV has too few unique pressure-depth samples")
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(depths, dtype=float)[order],
        np.asarray(phases, dtype=object)[order],
    )


def _segments_from_phases(
    times: np.ndarray,
    depths: np.ndarray,
    phases: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, float]], list[dict[str, Any]]]:
    segments: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    summaries: list[dict[str, Any]] = []
    phase_names = [str(value) for value in phases]
    active_names = list(dict.fromkeys(name for name in phase_names if _ACTIVE_PHASE.match(name)))
    if not active_names:
        raise RuntimeError("motor CSV has no down excitation phase")

    for name in active_names:
        active_indices = [index for index, phase in enumerate(phase_names) if phase == name]
        first = active_indices[0]
        last = active_indices[-1]
        end_index = last
        while end_index + 1 < len(phase_names):
            next_name = phase_names[end_index + 1]
            if _ACTIVE_PHASE.match(next_name):
                break
            end_index += 1
        # A positively buoyant vehicle eventually returns to the surface and
        # is then mechanically constrained by it.  That flat portion is not a
        # free heave response.  Retain the commanded motion and coast only up
        # to the first return through the pre-pulse depth after the excursion
        # extremum.
        active_depths = depths[first : last + 1]
        edge_count = min(5, len(active_depths))
        baseline_depth = float(np.mean(active_depths[:edge_count]))
        direction = _ACTIVE_PHASE.match(name).group(2)  # type: ignore[union-attr]
        search_depths = depths[first : end_index + 1]
        extreme_offset = int(
            np.argmax(search_depths)
            if direction == "positive"
            else np.argmin(search_depths)
        )
        extreme_index = first + extreme_offset
        returned_index: int | None = None
        for candidate in range(extreme_index + 1, end_index + 1):
            returned = (
                depths[candidate] <= baseline_depth
                if direction == "positive"
                else depths[candidate] >= baseline_depth
            )
            if returned:
                returned_index = candidate
                end_index = candidate
                break
        start = float(times[first] + 0.10)
        end = float(times[end_index] - 0.10)
        mask = (times >= start) & (times <= end)
        if int(np.count_nonzero(mask)) < 20 or end <= start:
            raise RuntimeError(f"too few samples in {name} plus its neutral coast")
        segment_times = times[mask]
        segment_depths = depths[mask]
        active_mask = np.asarray([phase == name for phase in phase_names], dtype=bool)
        active_depths = depths[active_mask]
        displacement_edge_count = min(10, len(active_depths))
        segments.append((segment_times, segment_depths, start, end))
        summaries.append(
            {
                "phase": name,
                "sample_count_with_neutral_coast": int(len(segment_times)),
                "duration_with_neutral_coast_s": float(
                    segment_times[-1] - segment_times[0]
                ),
                "active_depth_displacement_m": float(
                    np.mean(active_depths[-displacement_edge_count:])
                    - np.mean(active_depths[:displacement_edge_count])
                ),
                "pre_pulse_depth_m": baseline_depth,
                "coast_truncated_at_first_baseline_return": returned_index
                is not None,
                "segment_depth_min_m": float(np.min(segment_depths)),
                "segment_depth_max_m": float(np.max(segment_depths)),
            }
        )
    return segments, summaries


def analyze(motor_csv: Path, config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    motor_times, motor_force, motor_phases, rpm_outlier_count = _motor_force_samples(
        motor_csv, config, axis_index=2
    )
    depth_times, depths, depth_phases = _depth_samples(motor_csv)
    segments, segment_summary = _segments_from_phases(
        depth_times, depths, depth_phases
    )
    rmse, mass, damping, coefficient, residual = _grid_fit(
        segments, motor_times, motor_force
    )
    active_mask = np.asarray(
        [_ACTIVE_PHASE.match(str(phase)) is not None for phase in motor_phases],
        dtype=bool,
    )
    active_force = motor_force[active_mask]
    direction_coverage = sorted(
        {match.group(2) for phase in motor_phases if (match := _ACTIVE_PHASE.match(str(phase)))}
    )
    requested_amplitudes: set[float] = set()
    with motor_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _ACTIVE_PHASE.match(row.get("phase", "")):
                requested_amplitudes.add(abs(float(row["requested_down"])))

    return {
        "status": "candidate_only_not_enabled_for_control",
        "axis_frd": "down",
        "model": "F_down = M_eff * d(depth_velocity)/dt + D_linear * depth_velocity + bias",
        "method": (
            "Exact position-domain integration at 0.01 s; pressure depth and RPM "
            "from the same telemetry stream; each signed excitation is fitted with "
            "its following zero-command coast; independent initial position and "
            "velocity per segment; shared constant force bias."
        ),
        "source": {
            "motor_csv": str(motor_csv),
            "motor_csv_sha256": _sha256(motor_csv),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "depth": {
            "coordinate": "firmware pressure depth, FRD positive down",
            "unique_sample_count": int(len(depth_times)),
            "range_m": [float(np.min(depths)), float(np.max(depths))],
            "total_excursion_m": float(np.max(depths) - np.min(depths)),
            "absolute_pool_surface_reference": False,
        },
        "force": {
            "source": "current-vehicle measured RPM converted by accepted same-vehicle low-limit RPM-force prior",
            "rpm_values_replaced_as_isolated_outliers": int(rpm_outlier_count),
            "active_min_n": float(np.min(active_force)),
            "active_max_n": float(np.max(active_force)),
            "requested_absolute_channel_amplitudes": sorted(requested_amplitudes),
            "excitation_directions": direction_coverage,
            "vertical_motor_force_curves_individually_load_cell_measured": False,
        },
        "segments": segment_summary,
        "fit": {
            "effective_mass_kg": float(mass),
            "linear_damping_n_per_m_s": float(damping),
            "constant_force_bias_n": float(coefficient[-1]),
            "initial_velocity_m_s_by_segment": [
                float(value) for value in coefficient[len(segments) : -1]
            ],
            "position_rmse_m": float(rmse),
            "position_abs_error_p95_m": float(
                np.percentile(np.abs(residual), 95.0)
            ),
            "quadratic_damping_identified": False,
        },
        "confidence": {
            "level": "medium-low candidate",
            "enabled_for_control": False,
            "reasons": [
                "a repeated second run and a second command amplitude are required for cross-validation",
                *(
                    [
                        "only one commanded direction is present; the zero-command natural-buoyancy coast constrains the opposite motion, but a signed-thrust asymmetry check is unavailable"
                    ]
                    if len(direction_coverage) == 1
                    else []
                ),
                "the vertical motor curves are transferred by installed propulsion-unit rotation group rather than individually load-cell measured",
                "any vertical tether loading is inseparable from hydrodynamic damping",
                "the pressure zero is relative to firmware initialization, not an absolute pool surface datum",
                "linear and quadratic damping are not independently identified from one low-amplitude sweep",
            ],
        },
    }


def analyze_joint(motor_csvs: list[Path], config_path: Path) -> dict[str, Any]:
    if len(motor_csvs) < 2:
        raise ValueError("joint heave analysis requires at least two motor CSVs")
    config = _load_config(config_path)
    all_motor_times: list[np.ndarray] = []
    all_motor_force: list[np.ndarray] = []
    all_motor_phases: list[np.ndarray] = []
    all_depths: list[np.ndarray] = []
    all_segments: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    all_segment_summaries: list[dict[str, Any]] = []
    requested_amplitudes: set[float] = set()
    rpm_outlier_count = 0
    independent_results: list[dict[str, Any]] = []

    for motor_csv in motor_csvs:
        times, force, phases, replaced = _motor_force_samples(
            motor_csv, config, axis_index=2
        )
        depth_times, depths, depth_phases = _depth_samples(motor_csv)
        segments, summaries = _segments_from_phases(
            depth_times, depths, depth_phases
        )
        for summary in summaries:
            summary["source_motor_csv"] = str(motor_csv)
        with motor_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _ACTIVE_PHASE.match(row.get("phase", "")):
                    requested_amplitudes.add(abs(float(row["requested_down"])))
        all_motor_times.append(times)
        all_motor_force.append(force)
        all_motor_phases.append(phases)
        all_depths.append(depths)
        all_segments.extend(segments)
        all_segment_summaries.extend(summaries)
        rpm_outlier_count += replaced
        independent_results.append(analyze(motor_csv, config_path))

    motor_times = np.concatenate(all_motor_times)
    motor_force = np.concatenate(all_motor_force)
    motor_phases = np.concatenate(all_motor_phases)
    order = np.argsort(motor_times)
    motor_times = motor_times[order]
    motor_force = motor_force[order]
    motor_phases = motor_phases[order]
    depths = np.concatenate(all_depths)
    rmse, mass, damping, coefficient, residual = _grid_fit(
        all_segments, motor_times, motor_force
    )
    active_mask = np.asarray(
        [_ACTIVE_PHASE.match(str(phase)) is not None for phase in motor_phases],
        dtype=bool,
    )
    active_force = motor_force[active_mask]
    direction_coverage = sorted(
        {
            match.group(2)
            for phase in motor_phases
            if (match := _ACTIVE_PHASE.match(str(phase)))
        }
    )
    masses = [float(result["fit"]["effective_mass_kg"]) for result in independent_results]
    dampings = [
        float(result["fit"]["linear_damping_n_per_m_s"])
        for result in independent_results
    ]
    biases = [
        float(result["fit"]["constant_force_bias_n"])
        for result in independent_results
    ]

    def relative_span(values: list[float]) -> float:
        mean = abs(float(np.mean(values)))
        return float((max(values) - min(values)) / mean) if mean > 1.0e-9 else math.inf

    mass_relative_span = relative_span(masses)
    damping_relative_span = relative_span(dampings)
    bias_relative_span = relative_span(biases)
    return {
        "status": "cross_validated_candidate_only_not_enabled_for_control",
        "axis_frd": "down",
        "model": "F_down = M_eff * d(depth_velocity)/dt + D_linear * depth_velocity + bias",
        "method": (
            "Joint exact position-domain fit of all pulse-and-natural-buoyancy-coast "
            "segments with shared M_eff, D_linear and force bias; independent initial "
            "position and velocity per segment. Surface-constrained data after the "
            "first baseline return are excluded."
        ),
        "source": {
            "motor_csvs": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in motor_csvs
            ],
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
        },
        "depth": {
            "coordinate": "firmware pressure depth, FRD positive down",
            "combined_unique_sample_count": int(sum(len(values) for values in all_depths)),
            "range_m": [float(np.min(depths)), float(np.max(depths))],
            "absolute_pool_surface_reference": False,
        },
        "force": {
            "source": "current-vehicle measured RPM converted by accepted same-vehicle low-limit RPM-force prior",
            "rpm_values_replaced_as_isolated_outliers": int(rpm_outlier_count),
            "active_min_n": float(np.min(active_force)),
            "active_max_n": float(np.max(active_force)),
            "requested_absolute_channel_amplitudes": sorted(requested_amplitudes),
            "excitation_directions": direction_coverage,
            "vertical_motor_force_curves_individually_load_cell_measured": False,
        },
        "segments": all_segment_summaries,
        "independent_fits": [
            {
                "motor_csv": result["source"]["motor_csv"],
                "requested_absolute_channel_amplitudes": result["force"][
                    "requested_absolute_channel_amplitudes"
                ],
                **result["fit"],
            }
            for result in independent_results
        ],
        "cross_amplitude_consistency": {
            "effective_mass_relative_span": mass_relative_span,
            "linear_damping_relative_span": damping_relative_span,
            "force_bias_relative_span": bias_relative_span,
            "candidate_thresholds": {
                "effective_mass_relative_span_max": 0.30,
                "linear_damping_relative_span_max": 0.40,
                "force_bias_relative_span_max": 0.30,
            },
            "passes_candidate_thresholds": bool(
                len(requested_amplitudes) >= 2
                and mass_relative_span <= 0.30
                and damping_relative_span <= 0.40
                and bias_relative_span <= 0.30
            ),
        },
        "joint_fit": {
            "effective_mass_kg": float(mass),
            "linear_damping_n_per_m_s": float(damping),
            "constant_force_bias_n": float(coefficient[-1]),
            "initial_velocity_m_s_by_segment": [
                float(value)
                for value in coefficient[len(all_segments) : -1]
            ],
            "position_rmse_m": float(rmse),
            "position_abs_error_p95_m": float(
                np.percentile(np.abs(residual), 95.0)
            ),
            "quadratic_damping_identified": False,
        },
        "confidence": {
            "level": "medium-low cross-validated candidate",
            "enabled_for_control": False,
            "reasons": [
                "only positive commanded heave is present; natural buoyancy supplies the upward coast but not a negative-thrust asymmetry check",
                "vertical motor force curves are transferred by propulsion-unit rotation group rather than individually load-cell measured",
                "vertical tether loading is inseparable from hydrodynamic damping",
                "quadratic damping is not independently identified",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a pressure-depth heave candidate from a guarded down step"
    )
    parser.add_argument("--motor-csv", required=True)
    parser.add_argument(
        "--additional-motor-csv",
        action="append",
        default=[],
        help="additional amplitude/run CSV; repeat for a joint cross-validation fit",
    )
    parser.add_argument(
        "--config", default="MPC_dual_model/finesub_v4pro1_mpc.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    motor_csvs = [Path(args.motor_csv), *map(Path, args.additional_motor_csv)]
    result = (
        analyze_joint(motor_csvs, Path(args.config))
        if len(motor_csvs) > 1
        else analyze(motor_csvs[0], Path(args.config))
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
