"""Formal no-joystick AUTO-only runtime for the FineSUB real vehicle.

The default invocation is a read-only preflight.  Hardware communication is
possible only with ``--execute`` and only after every readiness gate passes.
Once armed, any vision, telemetry, command-confirmation, or solver fault is a
latched stop that requires restarting this process.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .auto_readiness import (
    DEFAULT_CONFIG_PATH,
    AutoReadinessReport,
    evaluate_auto_readiness,
    format_report,
)
from .auto_tracker import build_auto_tracker as build_translation_auto_tracker
from MPC_dual_model_yaw.auto_tracker import (
    build_auto_tracker as build_rotation_auto_tracker,
)
from .camera_transform import camera_to_body_position, wrap_angle
from SMC_controller.smc_controller import build_smc_tracker
from .finesub_protocol import (
    FineSUBControlCommand,
    build_runtime_hardware_adapter,
)
from .finesub_transport import (
    FineSUBConnection,
    load_runtime_config,
    make_transport,
)
from .vision_measurement import PipelineJsonlTail, VisionMeasurementGate


class _TranslationRuntimeTracker:
    """Expose the translation tracker through the guarded runtime interface."""

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self.model = SimpleNamespace(translation=tracker.model)
        self.estimator = tracker.estimator
        self.controller = tracker.controller
        self.fusion = tracker.fusion

    def latch_baseline(self, force, _yaw_moment, _yaw_rad) -> None:
        self._tracker.latch_baseline(force)

    def target_lost(self, force, _yaw_moment):
        output = self._tracker.target_lost(force)
        return SimpleNamespace(force=output.force, yaw_moment=0.0)

    def update(
        self,
        *,
        position_body,
        force_achieved_previous,
        yaw_moment_achieved_previous,
        reference_position,
        yaw_rad,
        yaw_rate_rad_s,
        yaw_delta_rad=0.0,
    ):
        del yaw_moment_achieved_previous
        output = self._tracker.update(
            position_body=position_body,
            tau_achieved_previous=force_achieved_previous,
            reference_position=reference_position,
            yaw_delta_rad=yaw_delta_rad,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )
        mpc = SimpleNamespace(**vars(output.mpc))
        mpc.yaw_moment = 0.0
        mpc.frozen_delta_yaw = output.mpc.predicted_delta_yaw_rad
        mpc.frozen_yaw_moments = np.zeros(len(output.mpc.force_sequence))
        return SimpleNamespace(
            estimated_state=output.estimated_state,
            mpc=mpc,
            yaw_control=SimpleNamespace(
                mode=SimpleNamespace(value="lower_local_hold"),
                goal_angle=float(yaw_rad),
            ),
            line_of_sight_angle=0.0,
        )


def _build_runtime_tracker(config: dict[str, Any]):
    required_model = config.get("auto_runtime", {}).get("required_model")
    if required_model == "dual":
        return _TranslationRuntimeTracker(build_translation_auto_tracker(config)), False
    if required_model == "dual-yaw":
        return build_rotation_auto_tracker(config), True
    if required_model == "smc-full":
        tracker = build_smc_tracker(config)
        return tracker, tracker.yaw_direct
    raise ValueError(f"unsupported AUTO model: {required_model!r}")


def zero_command(*, armed: bool, yaw_direct: bool) -> FineSUBControlCommand:
    """Zero translation/yaw while retaining MCU roll/pitch stabilization."""

    return FineSUBControlCommand(
        forward=0.0,
        right=0.0,
        down=0.0,
        yaw=0.0,
        armed=armed,
        yaw_direct=yaw_direct,
    )


def _assert_command_yaw_mode(
    command: FineSUBControlCommand, *, yaw_direct: bool
) -> None:
    """Fail closed if a command contradicts the selected yaw authority."""

    if command.yaw_direct is not yaw_direct:
        raise RuntimeError("command yaw_direct flag contradicts selected model")
    if not yaw_direct and abs(float(command.yaw)) > 1.0e-12:
        raise RuntimeError(
            "translation controller attempted a nonzero direct yaw command"
        )


def _telemetry_fault(
    telemetry,
    connection: FineSUBConnection,
    *,
    require_armed: bool = False,
    yaw_direct: bool,
) -> str | None:
    if telemetry is None:
        return "telemetry stale"
    if telemetry.failsafe:
        return "lower-controller failsafe"
    if (
        telemetry.last_command_rejected
        and telemetry.last_command_session == connection.session_id
    ):
        return f"lower controller rejected command flags=0x{telemetry.reject_flags:08x}"
    if not telemetry.execution_feedback_valid:
        return "execution feedback unavailable"
    if not telemetry.mpc_direct:
        return "lower controller is not in MPC-direct mode"
    if telemetry.yaw_direct != yaw_direct:
        return (
            "lower controller yaw mode mismatch: expected "
            + ("direct yaw" if yaw_direct else "local yaw hold")
        )
    if require_armed and not telemetry.armed:
        return "lower controller is no longer armed"
    return None


def _shutdown_disarmed(
    connection: FineSUBConnection, *, yaw_direct: bool
) -> None:
    """Best-effort disarm; the firmware watchdog remains the final backstop."""

    try:
        for _ in range(3):
            connection.send(zero_command(armed=False, yaw_direct=yaw_direct))
            connection.poll_telemetry()
    except Exception:
        pass
    finally:
        connection.close()


def run_auto_only(
    config: dict[str, Any],
    report: AutoReadinessReport,
    *,
    execute: bool,
    max_runtime_s: float | None = None,
    logger: Callable[[str], None] = print,
    runtime_label: str = "AUTO",
    trace_jsonl_path: str | Path | None = None,
    reacquire_on_vision_loss: bool = False,
    accept_any_vision_track: bool = False,
    hold_armed_on_vision_loss: bool = False,
    lock_reference_to_first_measurement: bool = False,
) -> int:
    """Run preflight or the guarded control loop.

    Returns 0 for a successful preflight/clean operator stop, 2 for readiness
    refusal, and 3 for a latched runtime safety stop.
    """

    report_text = format_report(report)
    if runtime_label != "AUTO":
        report_text = report_text.replace("AUTO", runtime_label, 1)
    logger(report_text)
    if not report.ready:
        return 2
    if not execute:
        logger("PREFLIGHT ONLY: no hardware connection was opened; add --execute to run AUTO")
        return 0

    # Construction still occurs before transport creation.  This ensures an
    # unexpected parameter incompatibility cannot open a hardware link.
    tracker, yaw_direct = _build_runtime_tracker(config)
    required_model = config.get("auto_runtime", {}).get("required_model")
    if required_model == "dual" and yaw_direct:
        raise RuntimeError("translation MPC cannot request direct yaw authority")
    if required_model == "dual-yaw" and not yaw_direct:
        raise RuntimeError("rotation MPC requires direct yaw authority")
    hardware_adapter = build_runtime_hardware_adapter(config)
    assert report.vision_jsonl is not None
    assert report.vision_gate_config is not None
    assert report.rotation_body_from_camera is not None
    assert report.camera_origin_in_body_frd_m is not None
    tail = PipelineJsonlTail(report.vision_jsonl, start_at_end=True)
    gate = VisionMeasurementGate(
        report.vision_gate_config,
        accept_any_finite_track_output=accept_any_vision_track,
    )
    tail.poll()  # Establish EOF before any command can be sent.

    control = config["control"]
    transport = config["transport"]
    connection = FineSUBConnection(
        make_transport(transport),
        telemetry_max_age_s=float(control["telemetry_max_age_sec"]),
        confirmation_max_age_s=float(control["confirmation_max_age_sec"]),
        reconnect_interval_s=float(transport["reconnect_interval_sec"]),
        logger=logger,
    )
    period_s = float(control["period_sec"])
    state = "DISARMED_HANDSHAKE"
    latest_measurement = None
    latest_measurement_seen_monotonic: float | None = None
    locked_reference_position = None
    vision_hold_active = False
    previous_control_acquisition_time_s: float | None = None
    previous_control_yaw_rad: float | None = None
    last_command = zero_command(armed=False, yaw_direct=yaw_direct)
    started = time.monotonic()
    next_cycle = started
    last_status = ""
    fatal_reason: str | None = None
    if trace_jsonl_path is not None:
        Path(trace_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    trace_context = (
        Path(trace_jsonl_path).open("x", encoding="utf-8")
        if trace_jsonl_path is not None
        else nullcontext(None)
    )

    def trace(handle, event: str, **fields: Any) -> None:
        if handle is None:
            return
        record = {
            "event": event,
            "host_time_s": time.time(),
            "host_monotonic_s": time.monotonic(),
            "runtime_state": state,
            **fields,
        }
        handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
        handle.write("\n")
        handle.flush()

    yaw_authority = "DIRECT_YAW" if yaw_direct else "LOWER_LOCAL_HOLD"
    logger(
        f"[{runtime_label}] Starting guarded camera control session; "
        f"model={required_model} yaw={yaw_authority}; "
        "no joystick/manual mode exists; FineSUB transport and MCU mixer retained"
    )
    try:
        with trace_context as trace_handle:
            trace(
                trace_handle,
                "start",
                runtime_label=runtime_label,
                required_model=required_model,
                yaw_direct=yaw_direct,
                yaw_authority=yaw_authority,
            )
            while True:
                now_monotonic = time.monotonic()
                if max_runtime_s is not None and now_monotonic - started >= max_runtime_s:
                    logger(f"[{runtime_label}] Maximum runtime reached; disarming")
                    trace(trace_handle, "bounded_stop")
                    break
                if now_monotonic < next_cycle:
                    time.sleep(min(next_cycle - now_monotonic, 0.01))
                    continue
                next_cycle = max(next_cycle + period_s, now_monotonic)

                new_measurement = None
                vision_reacquire_reason: str | None = None
                for record in tail.poll():
                    decision = gate.evaluate(record, now_s=time.time())
                    if decision.control_ready and decision.measurement is not None:
                        latest_measurement = decision.measurement
                        latest_measurement_seen_monotonic = now_monotonic
                        new_measurement = decision.measurement
                    elif accept_any_vision_track and decision.reason == "no_target":
                        # An empty processed frame is absence of an update, not
                        # a rejection of the most recent finite track.  Keep
                        # waiting for the next coordinate; the existing age
                        # check below still disarms if coordinates truly stop.
                        trace(trace_handle, "vision_empty_ignored")
                        continue
                    elif state in {"ARMING", "ACTIVE"}:
                        if hold_armed_on_vision_loss:
                            # Invalid/absent coordinates are simply no update
                            # in the operator-authorized experiment.  The
                            # ACTIVE branch below replaces any prior motion
                            # command with armed zero while local attitude hold
                            # remains enabled in the lower controller.
                            latest_measurement = None
                            latest_measurement_seen_monotonic = None
                            new_measurement = None
                            trace(
                                trace_handle,
                                "vision_gap_held_armed",
                                reason=decision.reason,
                            )
                        elif reacquire_on_vision_loss:
                            vision_reacquire_reason = decision.reason
                        else:
                            fatal_reason = (
                                "vision gate rejected a new result: "
                                f"{decision.reason}"
                            )
                            trace(
                                trace_handle,
                                "vision_reject",
                                reason=decision.reason,
                            )
                        break
                if fatal_reason is not None:
                    break
                if vision_reacquire_reason is not None:
                    # Experimental reacquisition remains fail-closed at the
                    # actuator boundary: disarm immediately, discard the old
                    # state estimate, then require the full fresh-vision and
                    # armed-zero handshake again.  Formal AUTO never enables
                    # this branch and retains its latched-stop policy.
                    last_command = zero_command(armed=False, yaw_direct=yaw_direct)
                    if not connection.send(last_command):
                        fatal_reason = "failed to send vision-loss disarm"
                        break
                    trace(
                        trace_handle,
                        "vision_reacquire",
                        reason=vision_reacquire_reason,
                    )
                    state = "DISARMED_HANDSHAKE"
                    tracker = _build_runtime_tracker(config)[0]
                    previous_control_yaw_rad = None
                    gate.reset()
                    latest_measurement = None
                    latest_measurement_seen_monotonic = None
                    status = (
                        "WAIT: vision rejected ("
                        f"{vision_reacquire_reason}); disarmed for reacquisition"
                    )
                    if status != last_status:
                        logger(f"[{runtime_label}] {status}")
                        last_status = status
                    continue

                telemetry = connection.fresh_telemetry()
                measurement_fresh = (
                    latest_measurement is not None
                    and latest_measurement_seen_monotonic is not None
                    and now_monotonic - latest_measurement_seen_monotonic
                    <= report.vision_gate_config.max_result_age_s
                    and time.time() - latest_measurement.result_time_s
                    <= report.vision_gate_config.max_result_age_s
                )

                if state == "DISARMED_HANDSHAKE":
                    last_command = zero_command(armed=False, yaw_direct=yaw_direct)
                    if not connection.send(last_command):
                        status = "WAIT TRANSPORT"
                    else:
                        fault = _telemetry_fault(
                            telemetry, connection, yaw_direct=yaw_direct
                        )
                        if fault is None and telemetry.armed:
                            fault = "lower controller has not confirmed disarmed state"
                        ready_to_arm = (
                            fault is None
                            and connection.session_confirmed
                            and connection.confirmation_fresh()
                            and measurement_fresh
                        )
                        if ready_to_arm:
                            state = "ARMING"
                            last_command = zero_command(
                                armed=True, yaw_direct=yaw_direct
                            )
                            if not connection.send(last_command):
                                fatal_reason = "failed to send armed-zero command"
                                break
                            status = "ARMING: WAIT ARMED-ZERO ACK"
                            trace(trace_handle, "arming")
                        else:
                            reasons = []
                            if fault is not None:
                                reasons.append(fault)
                            if not connection.session_confirmed:
                                reasons.append("disarmed session not confirmed")
                            if not measurement_fresh:
                                reasons.append("waiting for confirmed fresh vision")
                            status = "WAIT: " + ", ".join(dict.fromkeys(reasons))

                elif state == "ARMING":
                    fault = _telemetry_fault(
                        telemetry, connection, yaw_direct=yaw_direct
                    )
                    if fault is not None:
                        fatal_reason = fault
                        break
                    if not measurement_fresh:
                        if hold_armed_on_vision_loss:
                            last_command = zero_command(
                                armed=True, yaw_direct=yaw_direct
                            )
                            if not connection.send(last_command):
                                fatal_reason = "failed to send armed vision-wait zero"
                                break
                            status = "ARMING: HOLD ZERO, WAIT VISION"
                            trace(trace_handle, "vision_wait_armed_zero")
                            if status != last_status:
                                logger(f"[{runtime_label}] {status}")
                                last_status = status
                            continue
                        if reacquire_on_vision_loss:
                            last_command = zero_command(
                                armed=False, yaw_direct=yaw_direct
                            )
                            if not connection.send(last_command):
                                fatal_reason = "failed to send stale-vision disarm"
                                break
                            trace(
                                trace_handle,
                                "vision_reacquire",
                                reason="stale_while_arming",
                            )
                            state = "DISARMED_HANDSHAKE"
                            tracker = _build_runtime_tracker(config)[0]
                            previous_control_yaw_rad = None
                            gate.reset()
                            latest_measurement = None
                            latest_measurement_seen_monotonic = None
                            status = (
                                "WAIT: vision stale while arming; "
                                "disarmed for reacquisition"
                            )
                            if status != last_status:
                                logger(f"[{runtime_label}] {status}")
                                last_status = status
                            continue
                        fatal_reason = "vision became stale while arming"
                        break
                    last_command = zero_command(armed=True, yaw_direct=yaw_direct)
                    if not connection.send(last_command):
                        fatal_reason = "failed to send armed-zero command"
                        break
                    if connection.armed_confirmation_fresh() and telemetry.armed:
                        achieved_force, achieved_yaw_moment = (
                            hardware_adapter.achieved_wrench(telemetry)
                        )
                        tracker.latch_baseline(
                            achieved_force,
                            achieved_yaw_moment,
                            telemetry.yaw_rad,
                        )
                        previous_control_yaw_rad = None
                        state = "ACTIVE"
                        # Never use a measurement obtained before armed-zero was
                        # acknowledged for the first nonzero command.
                        status = "ACTIVE: WAIT NEW VISION"
                        trace(trace_handle, "active")
                    else:
                        status = (
                            "ARMING: WAIT ARMED-ZERO ACK "
                            f"telemetry_state={telemetry.state} "
                            f"telemetry_armed={telemetry.armed} "
                            f"pressure_pa={telemetry.pressure_pa:.1f} "
                            f"quat_w={telemetry.quat_wxyz[0]:.3f} "
                            f"armed_echo={connection.last_confirmed_armed} "
                            f"confirmation_fresh={connection.confirmation_fresh()} "
                            f"command_status={telemetry.command_status} "
                            f"reject_flags=0x{telemetry.reject_flags:08x}"
                        )

                else:  # ACTIVE
                    fault = _telemetry_fault(
                        telemetry,
                        connection,
                        require_armed=True,
                        yaw_direct=yaw_direct,
                    )
                    if fault is not None:
                        fatal_reason = fault
                        break
                    if not connection.armed_confirmation_fresh():
                        fatal_reason = "armed command confirmation stale"
                        break
                    if not measurement_fresh:
                        if hold_armed_on_vision_loss:
                            if not vision_hold_active:
                                trace(trace_handle, "vision_hold_started")
                            vision_hold_active = True
                            achieved_force, achieved_yaw_moment = (
                                hardware_adapter.achieved_wrench(telemetry)
                            )
                            safe_output = tracker.target_lost(
                                achieved_force,
                                achieved_yaw_moment,
                            )
                            last_command = hardware_adapter.convert(
                                safe_output.force,
                                safe_output.yaw_moment,
                                armed=True,
                                yaw_direct=yaw_direct,
                            )
                            _assert_command_yaw_mode(
                                last_command, yaw_direct=yaw_direct
                            )
                            status = (
                                "ACTIVE: RETURN TO FOSSEN BALANCE, WAIT VISION"
                            )
                        elif reacquire_on_vision_loss:
                            last_command = zero_command(
                                armed=False, yaw_direct=yaw_direct
                            )
                            if not connection.send(last_command):
                                fatal_reason = "failed to send stale-vision disarm"
                                break
                            trace(
                                trace_handle,
                                "vision_reacquire",
                                reason="stale_while_active",
                            )
                            state = "DISARMED_HANDSHAKE"
                            tracker = _build_runtime_tracker(config)[0]
                            previous_control_yaw_rad = None
                            gate.reset()
                            latest_measurement = None
                            latest_measurement_seen_monotonic = None
                            status = (
                                "WAIT: vision stale while active; "
                                "disarmed for reacquisition"
                            )
                            if status != last_status:
                                logger(f"[{runtime_label}] {status}")
                                last_status = status
                            continue
                        else:
                            fatal_reason = "vision result stale or absent after arming"
                            break
                    if measurement_fresh and new_measurement is not None:
                        achieved_force, achieved_yaw_moment = (
                            hardware_adapter.achieved_wrench(telemetry)
                        )
                        position_body = camera_to_body_position(
                            new_measurement.position_camera_xyz_m,
                            rotation_body_from_camera=report.rotation_body_from_camera,
                            camera_origin_in_body=report.camera_origin_in_body_frd_m,
                        )
                        if (
                            lock_reference_to_first_measurement
                            and locked_reference_position is None
                        ):
                            locked_reference_position = position_body.copy()
                            trace(
                                trace_handle,
                                "reference_locked",
                                reference_position_body_frd_m=list(
                                    map(float, locked_reference_position)
                                ),
                            )
                        if vision_hold_active:
                            # Discard the stale velocity/covariance accumulated
                            # before the visual gap.  The fixed reference is
                            # retained separately; the first reacquired sample
                            # initializes a fresh relative state.
                            tracker = _build_runtime_tracker(config)[0]
                            vision_hold_active = False
                            previous_control_yaw_rad = None
                            tracker.latch_baseline(
                                achieved_force,
                                achieved_yaw_moment,
                                telemetry.yaw_rad,
                            )
                        yaw_delta_rad = (
                            0.0
                            if previous_control_yaw_rad is None
                            else wrap_angle(
                                float(telemetry.yaw_rad)
                                - previous_control_yaw_rad
                            )
                        )
                        # TaskSUB v5 currently streams the raw H30 IMU rate,
                        # while V5_SUB negates it into body-FRD for control.
                        # Apply that same sign conversion before yaw-aware MPC.
                        yaw_rate_body_frd_rad_s = (
                            telemetry.body_frd_yaw_rate_rad_s
                        )
                        measurement_delay_s = max(
                            0.0,
                            time.time() - new_measurement.acquisition_time_s,
                        )
                        tracker_update_started = time.perf_counter()
                        update_kwargs = dict(
                            position_body=position_body,
                            force_achieved_previous=achieved_force,
                            yaw_moment_achieved_previous=achieved_yaw_moment,
                            reference_position=locked_reference_position,
                            yaw_rad=telemetry.yaw_rad,
                            yaw_rate_rad_s=yaw_rate_body_frd_rad_s,
                        )
                        if required_model == "smc-full":
                            # SMC's relative Kalman state is timestamped at
                            # image acquisition.  Lead it to the current
                            # control time using the same yaw-frame
                            # compensation that is used between visual
                            # updates.  The established yaw MPC path keeps
                            # its existing interface unchanged.
                            update_kwargs["yaw_delta_rad"] = yaw_delta_rad
                            update_kwargs["measurement_delay_s"] = (
                                measurement_delay_s
                            )
                        elif not yaw_direct:
                            update_kwargs["yaw_delta_rad"] = yaw_delta_rad
                        output = tracker.update(**update_kwargs)
                        previous_control_yaw_rad = float(telemetry.yaw_rad)
                        tracker_update_elapsed_ms = (
                            time.perf_counter() - tracker_update_started
                        ) * 1000.0
                        if output.mpc.used_fallback:
                            fatal_reason = (
                                f"{required_model} controller fallback: "
                                f"{output.mpc.status}"
                            )
                            break
                        last_command = hardware_adapter.convert(
                            output.mpc.force,
                            output.mpc.yaw_moment,
                            armed=True,
                            yaw_direct=yaw_direct,
                        )
                        _assert_command_yaw_mode(
                            last_command, yaw_direct=yaw_direct
                        )
                        acquisition_interval_s = (
                            None
                            if previous_control_acquisition_time_s is None
                            else new_measurement.acquisition_time_s
                            - previous_control_acquisition_time_s
                        )
                        previous_control_acquisition_time_s = (
                            new_measurement.acquisition_time_s
                        )
                        planned_step_count = min(3, len(output.mpc.force_sequence))
                        smc_output = getattr(output, "smc", None)
                        smc_axes = (
                            ()
                            if smc_output is None
                            else tuple(smc_output.translation_axes)
                        )
                        fusion_difference = (
                            tracker.fusion.M1
                            + tracker.fusion.M2
                            - 2.0 * tracker.fusion.C12
                        )
                        trace(
                            trace_handle,
                            "control_update",
                            frame_index=new_measurement.frame_index,
                            vision_acquisition_time_s=float(
                                new_measurement.acquisition_time_s
                            ),
                            vision_result_time_s=float(new_measurement.result_time_s),
                            vision_pipeline_delay_s=float(
                                new_measurement.result_time_s
                                - new_measurement.acquisition_time_s
                            ),
                            vision_measurement_age_s=float(measurement_delay_s),
                            vision_acquisition_interval_s=(
                                None
                                if acquisition_interval_s is None
                                else float(acquisition_interval_s)
                            ),
                            position_camera_xyz_m=list(
                                map(float, new_measurement.position_camera_xyz_m)
                            ),
                            position_body_frd_m=list(map(float, position_body)),
                            reference_position_body_frd_m=(
                                None
                                if locked_reference_position is None
                                else list(map(float, locked_reference_position))
                            ),
                            estimated_state=list(map(float, output.estimated_state)),
                            requested_force_frd_n=list(map(float, output.mpc.force)),
                            # SMC-only diagnostics make model/force-box
                            # mismatch visible in the trace.  The requested
                            # channel authority is intentionally not changed
                            # by this instrumentation.
                            smc_unsaturated_force_frd_n=(
                                None
                                if smc_output is None
                                else list(map(float, smc_output.unsaturated_force))
                            ),
                            smc_desired_acceleration_frd_m_s2=(
                                None
                                if not smc_axes
                                else [
                                    float(axis.desired_acceleration)
                                    for axis in smc_axes
                                ]
                            ),
                            smc_position_error=(
                                None
                                if not smc_axes
                                else [float(axis.position_error) for axis in smc_axes]
                            ),
                            smc_rate_reference=(
                                None
                                if not smc_axes
                                else [float(axis.rate_reference) for axis in smc_axes]
                            ),
                            smc_sliding_variable=(
                                None
                                if not smc_axes
                                else [
                                    float(axis.sliding_variable)
                                    for axis in smc_axes
                                ]
                            ),
                            camera_forward_distance_m=(
                                None
                                if getattr(tracker, "last_forward_distance_m", None)
                                is None
                                else float(tracker.last_forward_distance_m)
                            ),
                            camera_forward_force_before_guard_n=(
                                None
                                if getattr(
                                    tracker,
                                    "last_forward_force_before_guard_n",
                                    None,
                                )
                                is None
                                else float(
                                    tracker.last_forward_force_before_guard_n
                                )
                            ),
                            camera_forward_force_after_guard_n=(
                                None
                                if getattr(
                                    tracker,
                                    "last_forward_force_after_guard_n",
                                    None,
                                )
                                is None
                                else float(
                                    tracker.last_forward_force_after_guard_n
                                )
                            ),
                            requested_channels=[
                                last_command.forward,
                                last_command.right,
                                last_command.down,
                            ],
                            requested_yaw_moment_n_m=float(output.mpc.yaw_moment),
                            requested_yaw_channel=float(last_command.yaw),
                            yaw_control_mode=output.yaw_control.mode.value,
                            yaw_goal_rad=float(output.yaw_control.goal_angle),
                            line_of_sight_angle_rad=float(
                                output.line_of_sight_angle
                            ),
                            mpc_status=output.mpc.status,
                            mpc_iterations=output.mpc.iterations,
                            mpc_objective=float(output.mpc.objective),
                            tracker_update_elapsed_ms=float(
                                tracker_update_elapsed_ms
                            ),
                            mpc_planned_force_first3_frd_n=[
                                list(map(float, force))
                                for force in output.mpc.force_sequence[
                                    :planned_step_count
                                ]
                            ],
                            mpc_planned_delta_force_first3_frd_n=[
                                list(map(float, delta_force))
                                for delta_force in output.mpc.delta_force_sequence[
                                    :planned_step_count
                                ]
                            ],
                            mpc_predicted_actuator_force_first3_frd_n=(
                                None
                                if getattr(
                                    output.mpc,
                                    "predicted_actuator_force_sequence",
                                    None,
                                )
                                is None
                                else [
                                    list(map(float, force))
                                    for force in output.mpc.predicted_actuator_force_sequence[
                                        :planned_step_count
                                    ]
                                ]
                            ),
                            mpc_force_reference_frd_n=(
                                None
                                if not hasattr(output.mpc, "force_reference")
                                else list(map(float, output.mpc.force_reference))
                            ),
                            mpc_predicted_terminal_state=list(
                                map(float, output.mpc.predicted_states[-1])
                            ),
                            mpc_slack_max=float(np.max(output.mpc.slacks)),
                            model1_weight=list(map(float, output.mpc.model1_weight)),
                            fusion_sample_count=int(tracker.fusion.sample_count),
                            fusion_model1_mse=list(map(float, tracker.fusion.M1)),
                            fusion_model2_mse=list(map(float, tracker.fusion.M2)),
                            fusion_cross_error=list(map(float, tracker.fusion.C12)),
                            fusion_candidate_difference=list(
                                map(float, fusion_difference)
                            ),
                            fusion_window_count=int(tracker.fusion.window_count),
                            fusion_window_valid_count=list(
                                map(int, tracker.fusion.window_valid_count)
                            ),
                            fusion_window_rejected_count=list(
                                map(int, tracker.fusion.window_rejected_count)
                            ),
                            fusion_indistinguishable=list(
                                map(bool, tracker.fusion.indistinguishable)
                            ),
                            model1_fixed_base_force_frd_n=list(
                                map(
                                    float,
                                    getattr(
                                        output.mpc,
                                        "model1_base_force",
                                        achieved_force,
                                    ),
                                )
                            ),
                            model1_base_minus_achieved_force_frd_n=list(
                                map(
                                    float,
                                    np.asarray(
                                        getattr(
                                            output.mpc,
                                            "model1_base_force",
                                            achieved_force,
                                        ),
                                        dtype=float,
                                    )
                                    - achieved_force,
                                )
                            ),
                            fossen_restoring_force_frd_n=list(
                                map(float, tracker.model.translation.restoring_force)
                            ),
                            estimator_covariance_diagonal=list(
                                map(float, np.diag(tracker.estimator.P))
                            ),
                            achieved_force_previous_frd_n=list(
                                map(float, achieved_force)
                            ),
                            achieved_yaw_moment_previous_n_m=float(
                                achieved_yaw_moment
                            ),
                            achieved_force_source_axes=list(
                                hardware_adapter.last_achieved_wrench_diagnostics[
                                    "force_axis_sources"
                                ]
                            ),
                            achieved_yaw_source=(
                                hardware_adapter.last_achieved_wrench_diagnostics[
                                    "yaw_source"
                                ]
                            ),
                            rpm_valid_mask=int(telemetry.rpm_valid_mask),
                            rpm_thruster_force_n=list(
                                map(
                                    float,
                                    hardware_adapter.last_achieved_wrench_diagnostics[
                                        "rpm_thruster_force_n"
                                    ],
                                )
                            ),
                            rpm_force_diagnostic_frd_n=list(
                                map(
                                    float,
                                    hardware_adapter.last_achieved_wrench_diagnostics[
                                        "rpm_force_frd_n"
                                    ],
                                )
                            ),
                            rpm_yaw_moment_diagnostic_n_m=float(
                                hardware_adapter.last_achieved_wrench_diagnostics[
                                    "rpm_yaw_moment_n_m"
                                ]
                            ),
                            yaw_delta_body_frd_rad=float(yaw_delta_rad),
                            yaw_rate_body_frd_rad_s=float(
                                yaw_rate_body_frd_rad_s
                            ),
                            mpc_predicted_delta_yaw_rad=list(
                                map(float, output.mpc.frozen_delta_yaw)
                            ),
                            mpc_predicted_yaw_moment_n_m=list(
                                map(float, output.mpc.frozen_yaw_moments)
                            ),
                            telemetry_depth_m=float(telemetry.depth_m),
                            telemetry_yaw_rad=float(telemetry.yaw_rad),
                            applied_motor_throttle=list(
                                map(float, telemetry.applied_motor_throttle)
                            ),
                            motor_rpm=list(map(float, telemetry.motor_rpm)),
                        )
                    if not connection.send(last_command):
                        fatal_reason = "failed to send AUTO command"
                        break
                    if measurement_fresh and latest_measurement is not None:
                        yaw_status = (
                            f"yaw={last_command.yaw:+.3f}"
                            if yaw_direct
                            else "yaw=LOCAL_HOLD"
                        )
                        status = (
                            f"ACTIVE frame={latest_measurement.frame_index} "
                            f"cmd=({last_command.forward:+.3f},"
                            f"{last_command.right:+.3f},{last_command.down:+.3f}) "
                            f"{yaw_status}"
                        )

                if status != last_status:
                    logger(f"[{runtime_label}] {status}")
                    last_status = status
            trace(
                trace_handle,
                "stop",
                fatal_reason=fatal_reason,
                clean_stop=fatal_reason is None,
            )
    except KeyboardInterrupt:
        logger(f"[{runtime_label}] Operator stop; disarming")
    except Exception as error:
        fatal_reason = f"unexpected runtime error: {error}"
    finally:
        _shutdown_disarmed(connection, yaw_direct=yaw_direct)

    if fatal_reason is not None:
        logger(
            f"[{runtime_label} SAFETY STOP] {fatal_reason}; process restart required"
        )
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="connect and run AUTO after all gates pass (default: preflight only)",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="optional bounded run duration; shutdown is always disarmed",
    )
    args = parser.parse_args(argv)
    if args.max_runtime_sec is not None and args.max_runtime_sec <= 0.0:
        parser.error("--max-runtime-sec must be positive")
    config_path = Path(args.config).resolve()
    config = load_runtime_config(config_path)
    report = evaluate_auto_readiness(
        config,
        config_path=config_path,
        selected_model="dual-yaw",
    )
    return run_auto_only(
        config,
        report,
        execute=args.execute,
        max_runtime_s=args.max_runtime_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
