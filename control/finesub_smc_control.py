"""Guarded FineSUB full-vehicle SMC entry point.

The default command is formal preflight only.  ``--experimental`` selects the
operator-accepted candidate dynamics and camera transform; it still opens no
hardware link unless ``--execute`` is also supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from MPC_dual_model.auto_only_runtime import run_auto_only
from MPC_dual_model.auto_readiness import evaluate_auto_readiness
from MPC_dual_model.smc_config import (
    DEFAULT_SMC_PROFILE_PATH,
    load_smc_runtime_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_SMC_PROFILE_PATH))
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="select the explicit experimental_auto candidate source",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open FineSUB and execute only after every readiness gate passes",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="optional bounded run duration; shutdown always sends disarmed zero",
    )
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="optional guarded control trace path",
    )
    parser.add_argument(
        "--vision-jsonl",
        default=None,
        help="read-only override for the live vision producer JSONL path",
    )
    parser.add_argument(
        "--direct-yaw",
        action="store_true",
        help="request upper-level direct yaw; requires matching flashed firmware",
    )
    args = parser.parse_args(argv)
    if args.max_runtime_sec is not None and args.max_runtime_sec <= 0.0:
        parser.error("--max-runtime-sec must be positive")

    profile_path = Path(args.config).expanduser().resolve()
    config = load_smc_runtime_config(profile_path, experimental=args.experimental)
    smc_safety = config.get("smc_parameters", {}).get("runtime_safety", {})
    if not isinstance(smc_safety, dict):
        raise ValueError("smc_parameters.runtime_safety must be an object")
    if args.vision_jsonl is not None:
        vision_path = Path(args.vision_jsonl).expanduser().resolve()
        config["auto_runtime"]["vision_jsonl"] = str(vision_path)
    if args.direct_yaw:
        config["smc_parameters"]["yaw_authority"] = "direct"
    report = evaluate_auto_readiness(
        config,
        config_path=profile_path,
        selected_model="smc-full",
        expected_mode=("EXPERIMENTAL_AUTO" if args.experimental else "AUTO_ONLY"),
        allow_unaccepted_calibration_candidates=args.experimental,
        allow_zero_detector_confidence=args.experimental,
        maximum_authorized_channel_abs=(
            float(config["experimental_auto"]["max_channel_abs"])
            if args.experimental
            else 0.10
        ),
        expected_model_sample_period_s=(
            float(config["experimental_auto"]["expected_vision_update_period_s"])
            if args.experimental
            else None
        ),
        minimum_startup_confirmation_samples=int(
            smc_safety.get("startup_confirmation_samples", 3)
            if args.experimental
            else 3
        ),
        minimum_reacquire_confirmation_samples=int(
            smc_safety.get("reacquire_confirmation_samples", 5)
            if args.experimental
            else 5
        ),
        maximum_depth_nis=float(
            smc_safety.get("max_depth_nis", 25.0)
            if args.experimental
            else 25.0
        ),
        # The experimental vision acceptance range and the SMC hard minimum
        # are both 0.30 m.  Samples in 0.30--1.40 m reach SMC; only samples
        # below 0.30 m or above 1.40 m are rejected and reacquired.
        minimum_forward_m=(0.30 if args.experimental else 0.15),
        maximum_forward_m=(1.40 if args.experimental else 1.50),
        maximum_jump_margin_m=(0.20 if args.experimental else 0.10),
        maximum_step_m=0.30,
    )
    for warning in report.warnings:
        print(f"[SMC RISK] {warning}")
    runtime_label = "SMC EXPERIMENTAL" if args.experimental else "SMC"
    return run_auto_only(
        config,
        report,
        execute=args.execute,
        max_runtime_s=args.max_runtime_sec,
        runtime_label=runtime_label,
        trace_jsonl_path=(None if args.trace_jsonl is None else Path(args.trace_jsonl)),
        reacquire_on_vision_loss=args.experimental,
        accept_any_vision_track=False,
        hold_armed_on_vision_loss=False,
        lock_reference_to_first_measurement=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
