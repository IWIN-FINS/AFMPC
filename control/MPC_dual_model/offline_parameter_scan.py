"""Read-only frozen-state forward-parameter sensitivity scan.

This tool never opens hardware and never edits the active JSON.  It re-solves
the current MPC from representative states recorded in existing AUTO traces,
holding the recorded model-1 base, fusion weight, previous achieved force and
yaw rate fixed.  Each state is solved independently so the result is a
first-command/terminal-prediction sensitivity report, not a closed-loop claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .auto_tracker import build_auto_tracker
from .experimental_auto import build_experimental_runtime_config
from .finesub_transport import load_runtime_config


REFERENCE = np.asarray([0.857634, -0.055545, -0.120815], dtype=float)


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    p = np.percentile(values, [50.0, 90.0, 95.0, 100.0])
    return {
        "p50": float(p[0]),
        "p90": float(p[1]),
        "p95": float(p[2]),
        "p100": float(p[3]),
    }


def _clean_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect accepted control rows and classify clean dynamic/static samples."""
    dynamic: list[dict[str, Any]] = []
    steady: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    reference = REFERENCE
    for path in paths:
        previous_position: np.ndarray | None = None
        previous_time: float | None = None
        for item in _load(path):
            if item.get("event") != "control_update":
                continue
            try:
                state = np.asarray(item["estimated_state"], dtype=float)
                position = np.asarray(item["position_body_frd_m"], dtype=float)
                host_time = float(item["host_monotonic_s"])
                interval = float(item.get("vision_acquisition_interval_s", 0.10))
                if state.shape != (6,) or position.shape != (3,):
                    raise ValueError
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(position)):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                rejected += 1
                continue
            # Exclude the first sample after a long hold/reacquisition and
            # implausible accepted jumps.  This is intentionally conservative:
            # it removes visual discontinuities, not ordinary tracking error.
            jump = (
                float(np.linalg.norm(position - previous_position))
                if previous_position is not None
                else 0.0
            )
            time_gap = (
                host_time - previous_time if previous_time is not None else interval
            )
            previous_position = position
            previous_time = host_time
            if interval > 0.25 or time_gap > 0.40 or jump > 0.12:
                rejected += 1
                continue
            error = state[:3] - reference
            row = dict(item)
            row["_forward_error"] = float(error[0])
            row["_forward_speed"] = float(state[3])
            row["_clean_jump_m"] = jump
            row["_clean_interval_s"] = interval
            accepted += 1
            if abs(state[3]) >= 0.08:
                dynamic.append(row)
            if abs(state[3]) <= 0.03 and abs(error[0]) <= 0.05:
                steady.append(row)
    # Keep chronological coverage while bounding the computational scan.
    def thin(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(rows) <= limit:
            return rows
        indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
        return [rows[int(i)] for i in indices]

    dynamic = thin(dynamic, 180)
    steady = thin(steady, 180)
    return dynamic + steady, {
        "accepted_clean_updates": accepted,
        "rejected_visual_or_gap_updates": rejected,
        "dynamic_samples": len(dynamic),
        "steady_samples": len(steady),
    }


def _candidate_runtime(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(base)
    controller = runtime["auto_runtime"]["active_mpc_parameters"]["controller"]
    if "q_forward" in spec:
        controller["position_weights"][0] = float(spec["q_forward"])
    if "r_forward" in spec:
        controller["force_weights"][0] = float(spec["r_forward"])
    if "s_forward" in spec:
        controller["delta_force_weights"][0] = float(spec["s_forward"])
    if "delta_limit_forward" in spec:
        limit = float(spec["delta_limit_forward"])
        controller["delta_force_min"][0] = -limit
        controller["delta_force_max"][0] = limit
    if "horizon" in spec:
        controller["horizon"] = int(spec["horizon"])
    return runtime


def _channel(force: float, axis: int, source: dict[str, Any]) -> float:
    adapter = source["hardware_adapter"]
    positive = float(adapter["positive_force_at_limit"][axis])
    negative = float(adapter["negative_force_at_limit"][axis])
    limit = float(adapter["translation_channel_limits"][axis])
    return force / (positive if force >= 0.0 else negative) * limit


def _evaluate(
    runtime: dict[str, Any],
    source: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tracker = build_auto_tracker(runtime)
    controller = tracker.controller
    reference = np.asarray(controller.config.reference_position, dtype=float)
    groups: dict[str, list[dict[str, Any]]] = {"dynamic": [], "steady": []}
    for row in rows:
        groups["dynamic" if abs(float(row["_forward_speed"])) >= 0.08 else "steady"].append(row)

    result: dict[str, Any] = {}
    for name, group in groups.items():
        first_force: list[float] = []
        first_delta: list[float] = []
        terminal_error: list[float] = []
        effort: list[float] = []
        channel: list[float] = []
        rate_touch = 0
        channel_touch = 0
        solved = 0
        toward = 0
        solve_elapsed_ms: list[float] = []
        for row in group:
            state = np.asarray(row["estimated_state"], dtype=float)
            tau_previous = np.asarray(row["achieved_force_previous_frd_n"], dtype=float)
            tau_base = np.asarray(row["model1_fixed_base_force_frd_n"], dtype=float)
            weight = np.asarray(row["model1_weight"], dtype=float)
            yaw_rate = float(row.get("yaw_rate_body_frd_rad_s", 0.0))
            controller.reset()
            started = time.perf_counter()
            output = controller.solve(
                state,
                tau_previous,
                tau_base=tau_base,
                reference_position=reference,
                model1_weight=weight,
                yaw_rate_rad_s=yaw_rate,
            )
            solve_elapsed_ms.append(1000.0 * (time.perf_counter() - started))
            solved += int(not output.used_fallback)
            f = float(output.force[0])
            df = float(output.delta_force_sequence[0, 0])
            first_force.append(f)
            first_delta.append(df)
            terminal_error.append(abs(float(output.predicted_states[-1, 0] - reference[0])) * 100.0)
            effort.append(float(np.mean(np.linalg.norm(output.force_sequence, axis=1))))
            c = abs(_channel(f, 0, source))
            channel.append(c)
            rate_limit = float(max(abs(controller.config.delta_force_min[0]), abs(controller.config.delta_force_max[0])))
            rate_touch += int(abs(df) >= 0.98 * rate_limit)
            channel_touch += int(c >= 0.195)
            error = float(row["_forward_error"])
            toward += int(error * f > 0.0)
        count = len(group)
        result[name] = {
            "samples": count,
            "solved_fraction": float(solved / count) if count else None,
            "first_force_forward_n": _percentiles(np.abs(first_force)),
            "first_delta_force_forward_n": _percentiles(np.abs(first_delta)),
            "predicted_terminal_forward_abs_error_cm": _percentiles(terminal_error),
            "mean_predicted_force_norm_n": _percentiles(effort),
            "first_forward_channel_abs": _percentiles(channel),
            "delta_force_limit_touch_fraction": float(rate_touch / count) if count else None,
            "forward_channel_20pct_touch_fraction": float(channel_touch / count) if count else None,
            "force_toward_current_position_error_fraction": float(toward / count) if count else None,
            "solve_time_ms": _percentiles(np.asarray(solve_elapsed_ms)),
        }
    return result


def scan(config_path: Path, trace_paths: list[Path]) -> dict[str, Any]:
    source = load_runtime_config(config_path)
    base = build_experimental_runtime_config(source)
    rows, sample_counts = _clean_rows(trace_paths)
    specs: list[tuple[str, dict[str, Any]]] = [("current", {})]
    # Full Q/R grid isolates the interaction that determines whether more
    # position urgency actually produces a larger command.
    for q in (800.0, 1000.0, 1200.0, 1500.0):
        for r in (0.4, 0.6, 0.8):
            specs.append((f"q{int(q)}_r{r:.1f}", {"q_forward": q, "r_forward": r}))
    for s in (3.0, 5.0, 7.0):
        specs.append((f"s{int(s)}", {"s_forward": s}))
    for limit in (1.2, 1.4):
        specs.append((f"dlimit{limit:.1f}", {"delta_limit_forward": limit}))
    for horizon in (5, 10, 15, 20):
        specs.append((f"h{horizon}", {"horizon": horizon}))
    results = []
    for name, spec in specs:
        runtime = _candidate_runtime(base, spec)
        metrics = _evaluate(runtime, source, rows)
        results.append({"name": name, "changes": spec, "metrics": metrics})
    return {
        "analysis": "FineSUB frozen-state forward MPC parameter sensitivity scan",
        "hardware_transport_opened": False,
        "active_config": str(config_path.resolve()),
        "traces": [str(path.resolve()) for path in trace_paths],
        "method": "Each recorded state is solved independently with recorded tau_previous, tau_base, fusion weight and yaw rate; this is not a closed-loop prediction.",
        "sample_counts": sample_counts,
        "reference_position_body_frd_m": REFERENCE.tolist(),
        "candidates": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = scan(args.config.resolve(), [path.resolve() for path in args.trace])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
