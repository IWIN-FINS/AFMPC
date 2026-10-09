"""Fail-closed readiness checks for the formal FineSUB AUTO-only entry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .auto_tracker import (
    AutoParameterError,
    build_auto_tracker as build_translation_auto_tracker,
)
from MPC_dual_model_yaw.auto_tracker import (
    build_auto_tracker as build_rotation_auto_tracker,
)
from .finesub_protocol import PROTOCOL_VERSION, build_runtime_hardware_adapter
from .finesub_transport import load_runtime_config
from SMC_controller.smc_controller import build_smc_tracker
from .vision_measurement import VisionGateConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("finesub_v4pro1_mpc.json")


@dataclass(frozen=True)
class AutoReadinessReport:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    vision_jsonl: Path | None
    rotation_body_from_camera: np.ndarray | None
    camera_origin_in_body_frd_m: np.ndarray | None
    vision_gate_config: VisionGateConfig | None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = mapping
    for key in keys:
        value = _mapping(value).get(key)
    return _mapping(value)


def _resolve(config_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # The deployed config lives in MPC_dual_model while its recorded evidence
    # lives in the repository-level calibration_logs directory.  A config
    # beside its evidence remains supported for isolated tests/deployments.
    beside_config = config_path.parent / path
    repository_relative = config_path.parent.parent / path
    if beside_config.exists() or not repository_relative.exists():
        return beside_config
    return repository_relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vision_gate_config_from_runtime(config: dict[str, Any]) -> VisionGateConfig:
    gate = _nested(
        config,
        "camera_calibration_prior",
        "onboard_stereo_udp5600_historical_real_calibration",
        "external_red_fish_pipeline_20260812",
        "mpc_input_gate",
    )
    forward = gate.get("forward_range_m")
    if not isinstance(forward, list) or len(forward) != 2:
        raise ValueError("mpc_input_gate.forward_range_m must contain two values")
    return VisionGateConfig(
        max_result_age_s=float(gate["max_result_age_s"]),
        max_pipeline_delay_s=float(gate["max_pipeline_delay_s"]),
        min_confidence=float(gate["min_confidence"]),
        min_depth_confidence=float(gate["min_depth_confidence"]),
        max_depth_nis=float(gate["max_depth_nis"]),
        min_forward_m=float(forward[0]),
        max_forward_m=float(forward[1]),
        max_speed_m_s=float(gate["max_speed_m_s"]),
        jump_margin_m=float(gate["jump_margin_m"]),
        max_step_m=float(gate.get("max_step_m", 0.30)),
        max_inter_sample_gap_s=float(gate.get("max_inter_sample_gap_s", 0.50)),
        startup_confirmation_samples=int(gate["startup_confirmation_samples"]),
        reacquire_confirmation_samples=int(gate["reacquire_confirmation_samples"]),
        accepted_depth_filter_modes=tuple(
            gate.get("accepted_depth_filter_modes", ["update"])
        ),
        clamp_implausible_steps=bool(gate.get("clamp_implausible_steps", False)),
    )


def _check_evidence(
    blockers: list[str],
    config_path: Path,
    label: str,
    section: dict[str, Any],
    path_key: str,
    sha_key: str,
) -> None:
    path = _resolve(config_path, section.get(path_key))
    expected = section.get(sha_key)
    if path is None or not isinstance(expected, str) or len(expected) != 64:
        blockers.append(f"{label}: evidence path/SHA-256 is incomplete")
        return
    if not path.is_file():
        blockers.append(f"{label}: evidence file does not exist: {path}")
        return
    if _sha256(path) != expected.lower():
        blockers.append(f"{label}: evidence SHA-256 mismatch: {path}")


def evaluate_auto_readiness(
    config: dict[str, Any],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    selected_model: str = "dual-yaw",
    require_vision_source: bool = True,
    expected_mode: str = "AUTO_ONLY",
    allow_unaccepted_calibration_candidates: bool = False,
    allow_zero_detector_confidence: bool = False,
    maximum_authorized_channel_abs: float = 0.10,
    expected_model_sample_period_s: float | None = None,
    minimum_startup_confirmation_samples: int = 3,
    minimum_reacquire_confirmation_samples: int = 5,
    maximum_depth_nis: float = 25.0,
    minimum_forward_m: float = 0.15,
    maximum_forward_m: float = 1.50,
    maximum_jump_margin_m: float = 0.10,
    maximum_step_m: float = 0.30,
) -> AutoReadinessReport:
    """Return all blockers without opening a camera, socket, or serial port."""

    path = Path(config_path).resolve()
    blockers: list[str] = []
    warnings: list[str] = []
    auto = _mapping(config.get("auto_runtime"))

    if auto.get("mode") != expected_mode:
        blockers.append(f"auto_runtime.mode must be {expected_mode}")
    if auto.get("enabled") is not True:
        blockers.append("auto_runtime.enabled is not true")
    if auto.get("require_execute_flag") is not True:
        blockers.append("AUTO must require an explicit --execute flag")
    if auto.get("legacy_joystick_csrt_entry_allowed_for_auto") is not False:
        blockers.append("legacy joystick/CSRT AUTO entry is not explicitly disabled")
    required_model = auto.get("required_model")
    if required_model not in {"dual", "dual-yaw", "smc-full"} or selected_model != required_model:
        blockers.append(
            "AUTO selected model must exactly match required_model "
            "(dual, dual-yaw or smc-full)"
        )

    vision_jsonl = _resolve(path, auto.get("vision_jsonl"))
    if vision_jsonl is None:
        blockers.append("auto_runtime.vision_jsonl is missing")
    elif require_vision_source and not vision_jsonl.is_file():
        blockers.append(f"vision JSONL does not exist: {vision_jsonl}")

    camera = _nested(config, "camera_calibration_prior")
    stereo = _nested(
        config,
        "camera_calibration_prior",
        "onboard_stereo_udp5600_historical_real_calibration",
    )
    pipeline = _nested(
        config,
        "camera_calibration_prior",
        "onboard_stereo_udp5600_historical_real_calibration",
        "external_red_fish_pipeline_20260812",
    )
    rigid = _nested(
        config,
        "camera_calibration_prior",
        "onboard_stereo_udp5600_historical_real_calibration",
        "external_red_fish_pipeline_20260812",
        "rigid_target_cross_check_20260813",
    )
    if allow_unaccepted_calibration_candidates:
        warnings.append(
            "experimental authority accepts the recorded camera/body candidate "
            "although formal acceptance gates did not pass"
        )
    else:
        if camera.get("enabled_for_mpc_state_correction") is not True:
            blockers.append("camera calibration is not enabled for MPC state correction")
        if stereo.get("enabled_for_control") is not True:
            blockers.append("onboard stereo calibration is not enabled for control")
        if pipeline.get("enabled_for_control") is not True:
            blockers.append("external red-fish pipeline is not enabled for control")
        if rigid.get("acceptance_gates_passed") is not True:
            blockers.append("rigid-target camera/body acceptance gates have not passed")
        if rigid.get("enabled_for_control") is not True:
            blockers.append("rigid-target camera/body result is not enabled for control")

    real = _nested(config, "real_vehicle_calibration_candidates")
    if allow_unaccepted_calibration_candidates:
        warnings.append(
            "experimental authority accepts the recorded three-axis dynamic "
            "candidates although the formal real-vehicle set remains unresolved"
        )
    else:
        if real.get("enabled_for_control") is not True:
            blockers.append("real-vehicle calibration set is not enabled for control")
        unresolved = real.get("unresolved_gates")
        if not isinstance(unresolved, list) or unresolved:
            blockers.append("real-vehicle unresolved_gates is not an empty list")

    transform = _mapping(auto.get("active_camera_transform"))
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    if transform.get("enabled_for_control") is not True:
        blockers.append("active camera/body transform is not enabled for control")
    try:
        rotation = np.asarray(transform.get("rotation_body_from_camera"), dtype=float)
        translation = np.asarray(
            transform.get("camera_origin_in_body_frd_m"), dtype=float
        ).reshape(-1)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
            raise ValueError
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("active camera/body transform is missing or invalid")
        rotation = None
        translation = None

    gate_config: VisionGateConfig | None = None
    try:
        gate_config = vision_gate_config_from_runtime(config)
        if gate_config.max_result_age_s > 0.25:
            blockers.append("vision max_result_age_s is weaker than 0.25 s")
        if gate_config.max_pipeline_delay_s > 0.15:
            blockers.append("vision max_pipeline_delay_s is weaker than 0.15 s")
        minimum_detector_confidence = (
            0.0 if allow_zero_detector_confidence else 0.50
        )
        if gate_config.min_confidence < minimum_detector_confidence:
            blockers.append("vision min_confidence is weaker than 0.50")
        if allow_zero_detector_confidence and gate_config.min_confidence == 0.0:
            warnings.append(
                "experimental authority disables the detector-confidence threshold"
            )
        if gate_config.min_depth_confidence < 0.20:
            blockers.append("vision min_depth_confidence is weaker than 0.20")
        if gate_config.max_depth_nis > float(maximum_depth_nis):
            blockers.append("vision max_depth_nis exceeds the authorized value")
        if (
            gate_config.min_forward_m < float(minimum_forward_m)
            or gate_config.max_forward_m > float(maximum_forward_m)
        ):
            blockers.append("vision forward range exceeds the authorized envelope")
        if (
            gate_config.max_speed_m_s > 1.0
            or gate_config.jump_margin_m > float(maximum_jump_margin_m)
            or gate_config.max_step_m > float(maximum_step_m)
        ):
            blockers.append("vision motion gate is weaker than the replayed limit")
        if (
            gate_config.startup_confirmation_samples
            < int(minimum_startup_confirmation_samples)
        ):
            blockers.append(
                "vision startup requires fewer than the authorized confirmations"
            )
        if (
            gate_config.reacquire_confirmation_samples
            < int(minimum_reacquire_confirmation_samples)
        ):
            blockers.append(
                "vision reacquisition requires fewer than the authorized confirmations"
            )
    except (KeyError, TypeError, ValueError) as error:
        blockers.append(f"vision input gate is incomplete or invalid: {error}")

    control = _mapping(config.get("control"))
    if control.get("protocol_version") != PROTOCOL_VERSION:
        blockers.append(f"control protocol must be version {PROTOCOL_VERSION}")
    try:
        period = float(control["period_sec"])
        telemetry_age = float(control["telemetry_max_age_sec"])
        confirmation_age = float(control["confirmation_max_age_sec"])
        calibration_maximum = float(control["calibration_max_channel_abs"])
        authorized_maximum = float(maximum_authorized_channel_abs)
        if period <= 0.0 or period > 0.05:
            blockers.append("control period must be in (0, 0.05] s")
        if telemetry_age <= 0.0 or telemetry_age > 0.25:
            blockers.append("telemetry maximum age must be in (0, 0.25] s")
        if confirmation_age <= 0.0 or confirmation_age > 0.25:
            blockers.append("command-confirmation maximum age must be in (0, 0.25] s")
        adapter = build_runtime_hardware_adapter(config)
        channel_limits = np.concatenate(
            (adapter.translation_channel_limits, [adapter.yaw_channel_limit])
        )
        if calibration_maximum > 0.10:
            blockers.append("direct calibration channel limit exceeds 0.10")
        if (
            authorized_maximum <= 0.0
            or authorized_maximum > 1.0
            or np.any(channel_limits > authorized_maximum + 1.0e-12)
        ):
            blockers.append(
                "hardware channel limits exceed the authorized "
                f"<={authorized_maximum:.2f} envelope"
            )
    except (KeyError, TypeError, ValueError) as error:
        blockers.append(f"hardware adapter/control limits are invalid: {error}")

    transport = _mapping(config.get("transport"))
    if transport.get("type") != "udp":
        blockers.append("formal real-vehicle AUTO requires the approved UDP transport")
    try:
        if not str(transport["bind_host"]).strip():
            raise ValueError("empty bind_host")
        if not str(transport["remote_host"]).strip():
            raise ValueError("empty remote_host")
        for key in ("bind_port", "remote_port"):
            port = int(transport[key])
            if not 1 <= port <= 65535:
                raise ValueError(f"invalid {key}")
        if float(transport["reconnect_interval_sec"]) < 0.0:
            raise ValueError("negative reconnect_interval_sec")
    except (KeyError, TypeError, ValueError) as error:
        blockers.append(f"formal UDP transport is incomplete or invalid: {error}")

    thruster_feedback = _nested(config, "thruster_feedback")
    rpm_force = _nested(config, "thruster_feedback", "rpm_force_prior")
    if thruster_feedback.get("use_rpm_for_force_estimate") is not False:
        blockers.append(
            "RPM-based achieved force must remain diagnostic-only until its "
            "dynamic water response is validated"
        )
    if thruster_feedback.get("log_rpm_force_estimate") is not True:
        blockers.append("RPM-force diagnostic logging is not enabled")
    if rpm_force.get("enabled_for_diagnostics") is not True:
        blockers.append("same-vehicle RPM/force prior is not enabled for diagnostics")
    if rpm_force.get("same_physical_vehicle_confirmed_by_operator_20260812") is not True:
        blockers.append("RPM/force prior is not confirmed as the same physical vehicle")

    try:
        if selected_model == "dual":
            tracker = build_translation_auto_tracker(config)
        elif selected_model == "dual-yaw":
            tracker = build_rotation_auto_tracker(config)
        else:
            tracker = build_smc_tracker(config)
        if selected_model != "smc-full":
            solver_settings = _nested(
                config,
                "auto_runtime",
                "active_mpc_parameters",
                "controller",
                "solver_settings",
            )
            if solver_settings.get("backend") != "osqp":
                blockers.append("formal AUTO requires the explicit OSQP backend")
            if float(solver_settings["time_limit_seconds"]) > 0.035:
                blockers.append("OSQP time limit must not exceed 0.035 s")
        model_dt = tracker.model.dt
        expected_model_dt = (
            float(control["period_sec"])
            if expected_model_sample_period_s is None
            else float(expected_model_sample_period_s)
        )
        if expected_model_dt <= 0.0 or not np.isclose(
            model_dt, expected_model_dt, atol=1.0e-12
        ):
            blockers.append("active model sample period differs from its measurement period")
        adapter_cfg = _mapping(config.get("hardware_adapter"))
        positive = np.asarray(adapter_cfg["positive_force_at_limit"], dtype=float)
        negative = np.asarray(adapter_cfg["negative_force_at_limit"], dtype=float)
        hardware = build_runtime_hardware_adapter(config)
        if selected_model == "smc-full":
            controller = tracker.controller
            force_max = controller.positive_force_limit
            force_min = -controller.negative_force_limit
            yaw_max = controller.positive_yaw_limit
            yaw_min = -controller.negative_yaw_limit
        else:
            controller = tracker.controller.config
            force_max = controller.force_max
            force_min = controller.force_min
            yaw_max = yaw_min = None
            if selected_model == "dual-yaw":
                yaw_controller = tracker.yaw_controller.config
                yaw_max = yaw_controller.yaw_moment_max
                yaw_min = yaw_controller.yaw_moment_min
        if np.any(force_max > positive + 1.0e-9) or np.any(
            force_min < -negative - 1.0e-9
        ):
            blockers.append(
                f"{selected_model} force bounds exceed the hardware conversion envelope"
            )
        if yaw_max is not None and (
            yaw_max > hardware.positive_yaw_moment_at_limit + 1.0e-9
            or yaw_min < -hardware.negative_yaw_moment_at_limit - 1.0e-9
        ):
            blockers.append(
                f"{selected_model} yaw moment bounds exceed the hardware conversion envelope"
            )
    except (
        AutoParameterError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        blockers.append(f"active MPC parameters are not usable: {error}")

    if rigid:
        _check_evidence(
            blockers,
            path,
            "rigid-target frozen input",
            rigid,
            "frozen_front_jsonl",
            "frozen_front_jsonl_sha256",
        )
        _check_evidence(
            blockers,
            path,
            "rigid-target analysis",
            rigid,
            "analysis_file",
            "analysis_file_sha256",
        )

    # Keep order deterministic while avoiding repeated messages from related
    # malformed fields.
    blockers = list(dict.fromkeys(blockers))
    return AutoReadinessReport(
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        vision_jsonl=vision_jsonl,
        rotation_body_from_camera=rotation,
        camera_origin_in_body_frd_m=translation,
        vision_gate_config=gate_config,
    )


def format_report(report: AutoReadinessReport) -> str:
    if report.ready:
        return f"AUTO READY\nvision={report.vision_jsonl}"
    lines = ["AUTO BLOCKED (no hardware connection was opened)"]
    lines.extend(f"- {item}" for item in report.blockers)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--model", choices=("dual", "dual-yaw", "smc-full"), default="dual-yaw"
    )
    args = parser.parse_args(argv)
    config = load_runtime_config(args.config)
    report = evaluate_auto_readiness(
        config,
        config_path=args.config,
        selected_model=args.model,
    )
    print(format_report(report))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
