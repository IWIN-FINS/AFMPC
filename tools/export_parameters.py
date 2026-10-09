#!/usr/bin/env python3
"""Export compact review snapshots from authoritative runtime parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control"
PARAMETERS = ROOT / "parameters"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _controller_snapshot(source: Path, label: str) -> dict[str, Any]:
    config = _read_json(source)
    experimental = config["experimental_auto"]
    return {
        "schema_version": 1,
        "profile": label,
        "authoritative_source": str(source.relative_to(ROOT)),
        "source_sha256": _digest(source),
        "reference_position_body_frd_m": experimental[
            "reference_position_body_frd_m"
        ],
        "expected_vision_update_period_s": experimental[
            "expected_vision_update_period_s"
        ],
        "command_channel_limit_abs": experimental["max_channel_abs"],
        "camera_transform": experimental["active_camera_transform"],
        "mpc": experimental["active_mpc_parameters"],
        "yaw": experimental["active_yaw_parameters"],
        "vision_gate_overrides": experimental["vision_gate_overrides"],
    }


def main() -> int:
    control_out = PARAMETERS / "control"
    profiles = {
        "mpc_fusion.json": "finesub_v4pro1_mpc.json",
        "mpc_fixed_model1.json": "finesub_v4pro1_mpc_fixed_model1_20260919.json",
        "mpc_motion_model2.json": "finesub_v4pro1_mpc_motion_model2_20260919.json",
    }
    base_dir = CONTROL / "MPC_dual_model"
    for output_name, source_name in profiles.items():
        source = base_dir / source_name
        _write_json(control_out / output_name, _controller_snapshot(source, output_name))

    base_source = base_dir / "finesub_v4pro1_mpc.json"
    base = _read_json(base_source)
    _write_json(
        control_out / "hardware_and_protocol.json",
        {
            "schema_version": 1,
            "authoritative_source": str(base_source.relative_to(ROOT)),
            "source_sha256": _digest(base_source),
            "control": base["control"],
            "hardware_adapter": base["hardware_adapter"],
            "thruster_feedback": base["thruster_feedback"],
            "thruster_geometry": base["thruster_geometry"],
        },
    )

    smc_source = base_dir / "finesub_v4pro1_smc.json"
    smc = _read_json(smc_source)
    _write_json(
        control_out / "smc.json",
        {
            "schema_version": 1,
            "authoritative_source": str(smc_source.relative_to(ROOT)),
            "source_sha256": _digest(smc_source),
            **smc,
        },
    )

    sys.path.insert(0, str(CONTROL))
    from PID_controller.live_integration_example import build_tracker

    tracker = build_tracker(calibrated_reference=True)
    _write_json(
        control_out / "pid.json",
        {
            "schema_version": 1,
            "authoritative_source": (
                "control/PID_controller/live_integration_example.py"
            ),
            "source_sha256": _digest(
                CONTROL / "PID_controller" / "live_integration_example.py"
            ),
            "translation": tracker.controller.config,
            "yaw": tracker.yaw_controller.config,
            "camera_calibrated": tracker.camera_calibrated,
            "track_target_bearing": tracker.track_target_bearing,
        },
    )

    vision_source = ROOT / "vision" / "src" / "depth_estimation" / "config.yaml"
    vision_out = PARAMETERS / "vision" / "stereo_pipeline.yaml"
    vision_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vision_source, vision_out)
    print(f"Exported parameter snapshots to {PARAMETERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

