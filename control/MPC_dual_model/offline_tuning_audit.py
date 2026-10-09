"""Read-only audit of FineSUB real-device MPC tuning traces.

The tool never opens a hardware transport.  It summarizes timing, force
feedback, lower-controller mixing, fixed-per-solve model-1 operating force,
and deterministic closed-loop horizon candidates from existing JSONL traces.
"""

from __future__ import annotations

import argparse
import copy
from datetime import date
import glob
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .auto_tracker import build_auto_tracker
from .experimental_auto import build_experimental_runtime_config
from .finesub_transport import load_runtime_config


LOWER_MIXER = np.asarray(
    [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
     [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
    dtype=float,
)
UPPER_MIXER = np.asarray(
    [[-1.0, 1.0, 1.0], [1.0, 1.0, -1.0],
     [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=float,
)


def _percentiles(values: np.ndarray, points=(5, 50, 95, 100)) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    result = np.percentile(array, points, axis=0)
    return {
        f"p{point}": np.asarray(value).round(6).tolist()
        for point, value in zip(points, result)
    }


def _load_trace(path: Path) -> list[dict[str, Any]]:
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


def _vision_timestamp_lookup(path: Path | None) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    result: dict[int, tuple[float, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
                frame = int(value["frame_idx"])
                acquisition = float(value["frame_ts_mean"])
                completed = float(value["result_time"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(item) for item in (acquisition, completed)):
                result[frame] = (acquisition, completed)
    return result


def _rpm_force(
    rpm: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    prior = config["thruster_feedback"]["rpm_force_prior"]
    positive = np.asarray(
        prior["c1_positive_n_per_rad_s_sq_m1_m8"], dtype=float
    )
    negative = np.asarray(
        prior["c1_negative_abs_n_per_rad_s_sq_m1_m8"], dtype=float
    )
    geometry = config["thruster_geometry"]
    directions = np.asarray(
        geometry["positive_throttle_force_directions_frd_m1_m8"], dtype=float
    )
    yaw_arms = np.asarray(
        geometry["yaw_moment_arm_about_cad_origin_m_per_positive_force_m1_m8"],
        dtype=float,
    )
    filtered = rpm.copy()
    for index in range(len(rpm)):
        start = max(0, index - 2)
        end = min(len(rpm), index + 3)
        median = np.median(rpm[start:end], axis=0)
        outlier = np.abs(rpm[index] - median) > np.maximum(
            150.0, 0.30 * np.abs(median)
        )
        filtered[index, outlier] = median[outlier]
    omega = filtered * (2.0 * math.pi / 60.0)
    coefficient = np.where(filtered >= 0.0, positive, negative)
    thrust = np.sign(filtered) * coefficient * omega * omega
    return thrust @ directions, thrust @ yaw_arms


def _trace_summary(
    path: Path,
    source_config: dict[str, Any],
    runtime_config: dict[str, Any],
    vision_timestamps: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    records = _load_trace(path)
    updates = [item for item in records if item.get("event") == "control_update"]
    if not updates:
        return {"path": str(path), "control_updates": 0}

    host_time = np.asarray([item["host_monotonic_s"] for item in updates])
    force = np.asarray([item["requested_force_frd_n"] for item in updates])
    channels = np.asarray([item["requested_channels"] for item in updates])
    motors = np.asarray([item["applied_motor_throttle"] for item in updates])
    achieved = np.asarray(
        [item["achieved_force_previous_frd_n"] for item in updates]
    )
    rpm = np.asarray([item["motor_rpm"] for item in updates])

    lower = motors[:, [0, 1, 5, 6]]
    upper = motors[:, [2, 3, 4, 7]]
    yaw_forward_right = lower @ LOWER_MIXER / 4.0
    roll_pitch_down = upper @ UPPER_MIXER / 4.0
    applied_channels = np.column_stack(
        (yaw_forward_right[:, 1], yaw_forward_right[:, 2], roll_pitch_down[:, 2])
    )
    channel_error = applied_channels - channels

    controller = runtime_config["auto_runtime"]["active_mpc_parameters"][
        "controller"
    ]
    positive_limit = np.asarray(controller["force_max"], dtype=float)
    negative_limit = -np.asarray(controller["force_min"], dtype=float)
    force_scale = np.where(force >= 0.0, positive_limit, negative_limit)
    force_utilization = np.max(np.abs(force) / force_scale, axis=1)

    rpm_force, rpm_yaw = _rpm_force(rpm, source_config)
    active = (np.max(np.abs(motors), axis=1) > 0.01) | (
        np.max(np.abs(rpm), axis=1) > 200.0
    )
    rpm_error = rpm_force[active] - achieved[active]

    tracker = build_auto_tracker(runtime_config)
    fixed_model1_base: list[np.ndarray] = []
    replay_state_error: list[np.ndarray] = []
    for record in records:
        if record.get("event") == "vision_hold_started":
            tracker = build_auto_tracker(runtime_config)
            continue
        if record.get("event") != "control_update":
            continue
        achieved_for_interval = np.asarray(
            record["achieved_force_previous_frd_n"], dtype=float
        )
        output = tracker.update(
            record["position_body_frd_m"],
            achieved_for_interval,
        )
        # Record the persistent, slew-limited model-1 operating force used by
        # this solve.  It remains fixed for the complete QP horizon, while the
        # unfiltered achieved force remains the first force-rate reference.
        fixed_model1_base.append(output.mpc.model1_base_force.copy())
        replay_state_error.append(
            output.estimated_state
            - np.asarray(record["estimated_state"], dtype=float)
        )

    acquisition = [
        vision_timestamps.get(int(item["frame_index"])) for item in updates
    ]
    acquisition = [item for item in acquisition if item is not None]
    acquisition_time = np.asarray([item[0] for item in acquisition], dtype=float)
    pipeline_delay = np.asarray(
        [item[1] - item[0] for item in acquisition], dtype=float
    )

    return {
        "path": str(path),
        "control_updates": len(updates),
        "host_update_interval_s": _percentiles(np.diff(host_time)),
        "vision_acquisition_interval_s": _percentiles(np.diff(acquisition_time)),
        "vision_interval_over_0_15_fraction": (
            float(np.mean(np.diff(acquisition_time) > 0.15))
            if len(acquisition_time) > 1
            else None
        ),
        "pipeline_delay_s": _percentiles(pipeline_delay),
        "requested_force_utilization": _percentiles(force_utilization),
        "final_motor_cap_fraction": float(
            np.mean(np.max(np.abs(motors), axis=1) >= 0.495)
        ),
        "lower_mixer_channel_distortion_over_0_005_fraction": float(
            np.mean(np.max(np.abs(channel_error), axis=1) > 0.005)
        ),
        "applied_minus_requested_channel_abs": _percentiles(
            np.abs(channel_error)
        ),
        "applied_local_yaw_channel_abs": _percentiles(
            np.abs(yaw_forward_right[:, 0])
        ),
        "rpm_force_minus_command_force_abs_n": _percentiles(
            np.abs(rpm_error)
        ),
        "rpm_reconstructed_yaw_moment_abs_n_m": _percentiles(
            np.abs(rpm_yaw[active])
        ),
        "model1_fixed_base_force_frd_n": _percentiles(
            np.asarray(fixed_model1_base)
        ),
        "fossen_restoring_force_frd_n": (
            tracker.model.restoring_force.round(6).tolist()
        ),
        "replayed_state_max_abs_error": np.max(
            np.abs(np.asarray(replay_state_error)), axis=0
        ).round(9).tolist(),
    }


def _horizon_benchmark(
    runtime_config: dict[str, Any],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    reference = np.asarray(
        runtime_config["auto_runtime"]["active_mpc_parameters"]["controller"][
            "reference_position"
        ],
        dtype=float,
    )
    weight = np.asarray(
        runtime_config["auto_runtime"]["active_mpc_parameters"]["fusion"][
            "initial_model1_weight"
        ],
        dtype=float,
    )
    scenarios = {
        "surge_15cm": np.asarray((0.15, 0.0, 0.0)),
        "sway_15cm": np.asarray((0.0, 0.15, 0.0)),
        "diagonal_15cm": np.asarray((0.15, 0.15, 0.0)),
    }
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        for scenario, offset in scenarios.items():
            candidate = copy.deepcopy(runtime_config)
            candidate["auto_runtime"]["active_mpc_parameters"]["controller"][
                "horizon"
            ] = horizon
            controller = build_auto_tracker(candidate).controller
            model = controller.model
            state = np.concatenate((reference + offset, np.zeros(3)))
            force = model.restoring_force.copy()
            errors: list[np.ndarray] = []
            forces: list[np.ndarray] = []
            elapsed: list[float] = []
            fallback = 0
            for _ in range(120):
                started = time.perf_counter()
                output = controller.solve(
                    state,
                    force,
                    reference_position=reference,
                    model1_weight=weight,
                )
                elapsed.append(time.perf_counter() - started)
                fallback += int(output.used_fallback)
                force = output.force
                state = (
                    model.A_d @ state
                    + model.B_d @ (force - model.restoring_force)
                )
                errors.append(state[:3] - reference)
                forces.append(output.delta_force_sequence[0])
            error = np.asarray(errors)
            force_history = np.asarray(forces)
            active_axes = np.flatnonzero(offset)
            opposite = np.maximum(
                0.0,
                -np.sign(offset[active_axes]) * error[:, active_axes],
            )
            results.append(
                {
                    "horizon": horizon,
                    "prediction_time_s": horizon * model.dt,
                    "scenario": scenario,
                    "overshoot_percent": float(
                        100.0 * np.max(opposite / np.abs(offset[active_axes]))
                    ),
                    "final_error_cm": float(
                        100.0 * np.linalg.norm(error[-1, active_axes])
                    ),
                    "maximum_incremental_force_n": float(
                        np.max(np.abs(force_history[:, active_axes]))
                    ),
                    "solve_time_ms": _percentiles(1000.0 * np.asarray(elapsed)),
                    "fallback_count": fallback,
                }
            )
    return results


def audit(
    config_path: Path,
    trace_paths: list[Path],
    vision_jsonl: Path | None,
) -> dict[str, Any]:
    source = load_runtime_config(config_path)
    runtime = build_experimental_runtime_config(source)
    model = runtime["auto_runtime"]["active_mpc_parameters"]["model"]
    mass = np.diag(np.asarray(model["effective_mass_matrix_kg"], dtype=float))
    damping = np.diag(
        np.asarray(model["linear_damping_matrix_n_s_per_m"], dtype=float)
    )
    vision_timestamps = _vision_timestamp_lookup(vision_jsonl)
    control = source["control"]
    deadband = 0.01
    attitude_deadband = {}
    for axis in ("roll", "pitch", "yaw"):
        outer = float(control[f"local_{axis}_attitude_outer_gain"])
        rate_kp = float(control[f"local_{axis}_rate_pid"]["kp"])
        attitude_deadband[axis] = math.degrees(deadband / (outer * rate_kp))
    return {
        "analysis": "FineSUB V4 Pro1 read-only MPC tuning audit",
        "date": date.today().isoformat(),
        "config": str(config_path),
        "hardware_transport_opened": False,
        "model1_base_policy": (
            "per-axis gated EMA of tau_achieved,k-1 inside the configured "
            "position/velocity gates; optional forward matched-motion gate; "
            "fixed for the complete solve horizon"
        ),
        "model2_input_policy": "tau[j]-tau_h with fixed Fossen tau_h",
        "cost_policy": (
            "absolute total tau[j] plus consecutive Delta tau[j]"
        ),
        "model_time_constant_s_mass_over_linear_damping": (
            mass / damping
        ).round(6).tolist(),
        "model_sample_period_s": float(model["sample_period_s"]),
        "configured_horizon": int(
            runtime["auto_runtime"]["active_mpc_parameters"]["controller"][
                "horizon"
            ]
        ),
        "configured_prediction_time_s": float(model["sample_period_s"])
        * int(
            runtime["auto_runtime"]["active_mpc_parameters"]["controller"][
                "horizon"
            ]
        ),
        "software_motor_deadband": deadband,
        "p_only_attitude_error_needed_to_cross_deadband_deg": attitude_deadband,
        "trace_summaries": [
            _trace_summary(path, source, runtime, vision_timestamps)
            for path in trace_paths
        ],
        "deterministic_horizon_benchmark": _horizon_benchmark(
            runtime, (5, 10, 15, 20, 25)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("finesub_v4pro1_mpc.json"),
    )
    parser.add_argument(
        "--trace-glob",
        default="../calibration_logs/mpc_weight_tuning*.jsonl",
    )
    parser.add_argument("--vision-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    trace_paths = [Path(value) for value in sorted(glob.glob(args.trace_glob))]
    report = audit(args.config.resolve(), trace_paths, args.vision_jsonl)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
