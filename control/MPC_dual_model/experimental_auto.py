"""Explicitly risk-accepted real-vehicle AUTO experiment configuration.

Formal AUTO readiness remains untouched and fail-closed.  This module maps the
separate ``experimental_auto`` block into the existing guarded runtime only in
memory, after checking that its authority, candidate provenance, and explicit
operator-authorized command envelope are present.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .auto_only_runtime import run_auto_only
from .auto_readiness import (
    DEFAULT_CONFIG_PATH,
    AutoReadinessReport,
    evaluate_auto_readiness,
)
from .finesub_transport import load_runtime_config


EXPERIMENTAL_MODE = "EXPERIMENTAL_AUTO"
# The vehicle was physically mapped only through 0.10.  This is the absolute
# operator-authorized ceiling for experimental configuration, not a claim that
# the range above 0.10 has been calibrated.  The active translation limit is
# selected separately in ``experimental_auto.max_channel_abs``.
MAX_VALIDATED_CHANNEL_ABS = 0.50
DEFAULT_EXPERIMENT_RUNTIME_S = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_evidence(config_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    beside = config_path.parent / candidate
    repository = config_path.parent.parent / candidate
    return beside if beside.exists() or not repository.exists() else repository


def build_experimental_runtime_config(
    config: dict[str, Any], *, rotation: bool = False
) -> dict[str, Any]:
    """Return a runtime copy; never mutate or authorize the formal block."""

    experimental = config.get("experimental_auto")
    if not isinstance(experimental, dict):
        raise ValueError("experimental_auto must be an object")
    if experimental.get("mode") != EXPERIMENTAL_MODE:
        raise ValueError(f"experimental_auto.mode must be {EXPERIMENTAL_MODE}")
    if experimental.get("enabled_for_experiment") is not True:
        raise ValueError("experimental_auto is not enabled for experiment")
    if experimental.get("known_calibration_risks_accepted_by_operator") is not True:
        raise ValueError("known calibration risks are not explicitly accepted")
    if experimental.get("disable_detector_confidence_threshold") is not True:
        raise ValueError(
            "experimental detector-confidence threshold removal is not authorized"
        )
    if experimental.get("accept_any_finite_track_output") is not False:
        raise ValueError("experimental finite-track quality bypass must remain disabled")
    if experimental.get("vision_quality_gate_enabled") is not True:
        raise ValueError("experimental vision quality gate is not enabled")
    if experimental.get("hold_armed_during_vision_gaps") is not True:
        raise ValueError("armed attitude hold during vision gaps is not authorized")
    if experimental.get("lock_reference_to_first_valid_position") is not False:
        raise ValueError(
            "experimental reference must remain fixed at the configured camera standoff"
        )
    maximum = float(experimental.get("max_channel_abs", 0.0))
    if not 0.0 < maximum <= MAX_VALIDATED_CHANNEL_ABS:
        raise ValueError("experimental max_channel_abs must be in (0, 0.50]")
    expected_vision_period = float(
        experimental.get("expected_vision_update_period_s", 0.0)
    )
    if expected_vision_period <= 0.0:
        raise ValueError("experimental expected vision update period must be positive")

    runtime = copy.deepcopy(config)
    runtime_auto = copy.deepcopy(experimental)
    runtime_auto["enabled"] = True
    # Translation remains the maintained source of the shared MPC parameters.
    # The optional rotation experiment adds the yaw-aware wrapper in memory;
    # formal AUTO stays untouched and still fails closed on the unaccepted yaw
    # candidate.
    runtime_auto["required_model"] = "dual-yaw" if rotation else "dual"
    runtime_auto["require_execute_flag"] = True
    runtime_auto["legacy_joystick_csrt_entry_allowed_for_auto"] = False
    transform = runtime_auto.get("active_camera_transform")
    if not isinstance(transform, dict) or transform.get("enabled_for_experiment") is not True:
        raise ValueError("experimental camera transform is not enabled")
    transform["enabled_for_control"] = True
    active = runtime_auto.get("active_mpc_parameters")
    if not isinstance(active, dict) or active.get("enabled_for_experiment") is not True:
        raise ValueError("experimental MPC parameters are not enabled")
    active["enabled_for_control"] = True
    active_yaw = runtime_auto.get("active_yaw_parameters")
    if (
        not isinstance(active_yaw, dict)
        or active_yaw.get("enabled_for_experiment") is not True
    ):
        raise ValueError("experimental yaw parameters are not enabled")
    active_yaw["enabled_for_control"] = bool(rotation)
    runtime["auto_runtime"] = runtime_auto

    vision_gate = runtime[
        "camera_calibration_prior"
    ]["onboard_stereo_udp5600_historical_real_calibration"][
        "external_red_fish_pipeline_20260812"
    ]["mpc_input_gate"]
    vision_gate["min_confidence"] = 0.0
    gate_overrides = experimental.get("vision_gate_overrides")
    if not isinstance(gate_overrides, dict):
        raise ValueError("experimental_auto.vision_gate_overrides must be an object")
    allowed_gate_overrides = {
        "accepted_depth_filter_modes",
        "clamp_implausible_steps",
        "forward_range_m",
        "jump_margin_m",
        "max_depth_nis",
        "max_inter_sample_gap_s",
        "max_step_m",
        "startup_confirmation_samples",
        "reacquire_confirmation_samples",
    }
    unknown_gate_overrides = sorted(set(gate_overrides) - allowed_gate_overrides)
    if unknown_gate_overrides:
        raise ValueError(
            "unsupported experimental vision gate overrides: "
            + ", ".join(unknown_gate_overrides)
        )
    vision_gate.update(copy.deepcopy(gate_overrides))

    adapter = runtime.get("hardware_adapter")
    if not isinstance(adapter, dict):
        raise ValueError("hardware_adapter must be an object")
    configured_limits = np.asarray(
        adapter.get("translation_channel_limits"), dtype=float
    ).reshape(-1)
    if configured_limits.shape != (3,) or np.any(configured_limits <= 0.0):
        raise ValueError("hardware translation channel limits are invalid")
    if np.any(configured_limits + 1.0e-12 < maximum):
        raise ValueError("experimental limit exceeds a hardware adapter channel limit")
    adapter["translation_channel_limits"] = [maximum] * 3
    if float(adapter.get("yaw_channel_limit", 0.0)) > maximum:
        adapter["yaw_channel_limit"] = maximum
    return runtime


def evaluate_experimental_readiness(
    config: dict[str, Any],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    require_vision_source: bool = True,
    rotation: bool = False,
) -> tuple[dict[str, Any] | None, AutoReadinessReport]:
    try:
        runtime = build_experimental_runtime_config(config, rotation=rotation)
    except (TypeError, ValueError) as error:
        report = AutoReadinessReport(
            ready=False,
            blockers=(f"experimental configuration is invalid: {error}",),
            warnings=(),
            vision_jsonl=None,
            rotation_body_from_camera=None,
            camera_origin_in_body_frd_m=None,
            vision_gate_config=None,
        )
        return None, report
    report = evaluate_auto_readiness(
        runtime,
        config_path=config_path,
        selected_model="dual-yaw" if rotation else "dual",
        require_vision_source=require_vision_source,
        expected_mode=EXPERIMENTAL_MODE,
        allow_unaccepted_calibration_candidates=True,
        allow_zero_detector_confidence=True,
        maximum_authorized_channel_abs=float(
            config["experimental_auto"]["max_channel_abs"]
        ),
        expected_model_sample_period_s=float(
            config["experimental_auto"]["expected_vision_update_period_s"]
        ),
        minimum_startup_confirmation_samples=1,
        minimum_reacquire_confirmation_samples=1,
        maximum_depth_nis=100.0,
        minimum_forward_m=0.10,
        maximum_forward_m=2.80,
        maximum_jump_margin_m=0.20,
        maximum_step_m=0.30,
    )
    evidence_blockers: list[str] = []
    experimental = config.get("experimental_auto", {})
    evidence = experimental.get("candidate_evidence")
    if not isinstance(evidence, list) or not evidence:
        evidence_blockers.append("experimental candidate evidence is missing")
    else:
        resolved_config_path = Path(config_path).resolve()
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                evidence_blockers.append(
                    f"experimental candidate evidence {index} is not an object"
                )
                continue
            candidate = _resolve_evidence(resolved_config_path, item.get("path"))
            expected = item.get("sha256")
            if candidate is None or not isinstance(expected, str) or len(expected) != 64:
                evidence_blockers.append(
                    f"experimental candidate evidence {index} path/SHA-256 is incomplete"
                )
            elif not candidate.is_file():
                evidence_blockers.append(
                    f"experimental candidate evidence does not exist: {candidate}"
                )
            elif _sha256(candidate) != expected.lower():
                evidence_blockers.append(
                    f"experimental candidate evidence SHA-256 mismatch: {candidate}"
                )
    if evidence_blockers:
        report = AutoReadinessReport(
            ready=False,
            blockers=tuple((*report.blockers, *evidence_blockers)),
            warnings=report.warnings,
            vision_jsonl=report.vision_jsonl,
            rotation_body_from_camera=report.rotation_body_from_camera,
            camera_origin_in_body_frd_m=report.camera_origin_in_body_frd_m,
            vision_gate_config=report.vision_gate_config,
        )
    return runtime, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="connect and run the explicitly risk-accepted experiment",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=DEFAULT_EXPERIMENT_RUNTIME_S,
        help="optional bounded duration; 0 (default) runs until operator stop or a safety fault",
    )
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="per-control-update experiment trace; defaults to a timestamped calibration_logs file",
    )
    parser.add_argument(
        "--vision-jsonl",
        default=None,
        help="read-only override for the external vision result JSONL",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="enable the explicitly risk-accepted direct-yaw experiment in memory",
    )
    args = parser.parse_args(argv)
    if args.max_runtime_sec < 0.0:
        parser.error("--max-runtime-sec must be non-negative")
    max_runtime_s = None if args.max_runtime_sec == 0.0 else args.max_runtime_sec
    config_path = Path(args.config).resolve()
    source_config = load_runtime_config(config_path)
    if args.vision_jsonl:
        source_config = copy.deepcopy(source_config)
        source_config["experimental_auto"]["vision_jsonl"] = str(
            Path(args.vision_jsonl).resolve()
        )
    runtime, report = evaluate_experimental_readiness(
        source_config,
        config_path=config_path,
        rotation=args.rotation,
    )
    if runtime is None:
        for blocker in report.blockers:
            print(f"EXPERIMENTAL AUTO BLOCKED: {blocker}")
        return 2
    for warning in report.warnings:
        print(f"[EXPERIMENTAL RISK] {warning}")
    trace_path = None
    if args.execute:
        trace_path = Path(args.trace_jsonl) if args.trace_jsonl else Path(
            "calibration_logs"
        ) / f"experimental_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"[EXPERIMENTAL AUTO] trace={trace_path}")
    return run_auto_only(
        runtime,
        report,
        execute=args.execute,
        max_runtime_s=max_runtime_s,
        runtime_label=(
            "EXPERIMENTAL ROTATION AUTO" if args.rotation else "EXPERIMENTAL AUTO"
        ),
        trace_jsonl_path=trace_path,
        reacquire_on_vision_loss=True,
        accept_any_vision_track=False,
        hold_armed_on_vision_loss=True,
        lock_reference_to_first_measurement=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
