"""Load the SMC profile without duplicating the FineSUB runtime contract.

The profile only owns SMC gains and the base-config reference.  Transport,
protocol, vision gates, telemetry policy, hardware conversion and calibration
evidence remain sourced from the maintained MPC runtime configuration.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from MPC_dual_model.experimental_auto import build_experimental_runtime_config
from MPC_dual_model.finesub_transport import load_runtime_config


DEFAULT_SMC_PROFILE_PATH = Path(__file__).with_name("finesub_v4pro1_smc.json")


def _relative_to(path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else path.parent / candidate


def load_smc_runtime_config(
    profile_path: str | Path = DEFAULT_SMC_PROFILE_PATH,
    *,
    experimental: bool = False,
) -> dict[str, Any]:
    """Resolve a profile into the standard guarded runtime config.

    Formal mode deliberately keeps the base config's disabled/null dynamics,
    so readiness cannot accidentally turn an unapproved model into formal
    AUTO.  Experimental mode reuses the existing explicit risk-acceptance
    mapping and changes only the selected controller to ``smc-full``.
    """

    profile_file = Path(profile_path).expanduser().resolve()
    profile = load_runtime_config(profile_file)
    base_file = _relative_to(profile_file, profile.get("base_config"), "base_config")
    base = load_runtime_config(base_file.resolve())
    source_field = "experimental_source" if experimental else "formal_source"
    expected_source = "experimental_auto" if experimental else "auto_runtime"
    if profile.get(source_field) != expected_source:
        raise ValueError(f"{source_field} must select {expected_source}")

    if experimental:
        runtime = build_experimental_runtime_config(base)
        runtime_auto = runtime["auto_runtime"]
        active_yaw = runtime_auto.get("active_yaw_parameters")
        if not isinstance(active_yaw, dict):
            raise ValueError("experimental SMC profile has no yaw dynamics")
        # The legacy experiment intentionally kept yaw under the lower local
        # hold loop.  Full SMC owns all four high-level axes, so it explicitly
        # enables the already risk-accepted yaw candidate for this profile.
        active_yaw["enabled_for_control"] = True
    else:
        runtime = copy.deepcopy(base)
        runtime_auto = runtime.get("auto_runtime")
        if not isinstance(runtime_auto, dict):
            raise ValueError("base config has no auto_runtime object")

    runtime_auto["required_model"] = "smc-full"
    runtime_auto["require_execute_flag"] = True
    runtime_auto["legacy_joystick_csrt_entry_allowed_for_auto"] = False
    smc_parameters = profile.get("smc_parameters")
    if not isinstance(smc_parameters, dict):
        raise ValueError("smc_parameters must be an object")
    runtime["smc_parameters"] = copy.deepcopy(smc_parameters)
    if experimental:
        safety = runtime["smc_parameters"].get("runtime_safety")
        if not isinstance(safety, dict):
            raise ValueError("smc_parameters.runtime_safety must be an object")
        try:
            maximum_channel = float(safety["max_channel_abs"])
            startup_samples = int(safety["startup_confirmation_samples"])
            reacquire_samples = int(safety["reacquire_confirmation_samples"])
            maximum_depth_nis = float(safety["max_depth_nis"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid smc_parameters.runtime_safety: {error}"
            ) from error
        if not 0.0 < maximum_channel <= 0.20:
            raise ValueError(
                "SMC experimental max_channel_abs must be in (0, 0.20]"
            )
        if startup_samples < 1 or reacquire_samples < 1:
            raise ValueError("SMC vision confirmation samples must be positive")
        if maximum_depth_nis <= 0.0:
            raise ValueError("SMC max_depth_nis must be positive")

        # Keep the SMC profile's authority explicit and separate from the
        # generic experimental MPC profile.  This must match the authorized
        # experiment envelope; model errors are handled in the controller,
        # not hidden by silently reducing actuator authority.
        experimental_auto = runtime.get("experimental_auto")
        if not isinstance(experimental_auto, dict):
            raise ValueError("experimental_auto must be an object")
        experimental_auto["max_channel_abs"] = maximum_channel
        gate_overrides = experimental_auto.get("vision_gate_overrides")
        if not isinstance(gate_overrides, dict):
            raise ValueError("experimental_auto.vision_gate_overrides must be an object")
        gate_overrides.update(
            {
                "startup_confirmation_samples": startup_samples,
                "reacquire_confirmation_samples": reacquire_samples,
                "max_depth_nis": maximum_depth_nis,
            }
        )
        adapter = runtime.get("hardware_adapter")
        if not isinstance(adapter, dict):
            raise ValueError("hardware_adapter must be an object")
        configured_limits = [
            float(value)
            for value in adapter.get("translation_channel_limits", [])
        ]
        if len(configured_limits) != 3 or any(
            value < maximum_channel for value in configured_limits
        ):
            raise ValueError("SMC channel limit exceeds hardware adapter limits")
        adapter["translation_channel_limits"] = [maximum_channel] * 3
        if float(adapter.get("yaw_channel_limit", 0.0)) > maximum_channel:
            adapter["yaw_channel_limit"] = maximum_channel
        gate = runtime["camera_calibration_prior"][
            "onboard_stereo_udp5600_historical_real_calibration"
        ]["external_red_fish_pipeline_20260812"]["mpc_input_gate"]
        gate["startup_confirmation_samples"] = startup_samples
        gate["reacquire_confirmation_samples"] = reacquire_samples
        gate["max_depth_nis"] = maximum_depth_nis
    runtime["smc_profile_path"] = str(profile_file)
    runtime["smc_source"] = "experimental_auto" if experimental else "auto_runtime"
    return runtime
