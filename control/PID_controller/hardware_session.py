"""End-to-end PID-to-V4pro1_MPC hardware control session.

The caller remains responsible for supplying timestamp-current stereo target
positions at 20 Hz. This module owns the command-session safety handshake, uses
MCU IMU telemetry for yaw PID, and feeds measured applied wrench back to the PID
limiter. Transport may be direct serial or the MPC-style TCP/UDP bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

try:
    from .camera_transform import camera_to_pid_body_position
    from .finesub_protocol import FineSUBHardwareAdapter, FineSUBTelemetry
    from .finesub_transport import (
        FineSUBConnection,
        SerialTransport,
        load_runtime_config,
        make_transport,
    )
    from .live_integration_example import build_tracker
    from .pid_tracker import PIDTracker, PIDTrackerOutput
    from .vision_gate import PIDVisionGate, VisionGateConfig, VisionGateResult
except ImportError:
    from camera_transform import camera_to_pid_body_position
    from finesub_protocol import FineSUBHardwareAdapter, FineSUBTelemetry
    from finesub_transport import (
        FineSUBConnection,
        SerialTransport,
        load_runtime_config,
        make_transport,
    )
    from live_integration_example import build_tracker
    from pid_tracker import PIDTracker, PIDTrackerOutput
    from vision_gate import PIDVisionGate, VisionGateConfig, VisionGateResult


@dataclass(frozen=True)
class HardwareStepResult:
    status: str
    telemetry: FineSUBTelemetry | None
    controller_output: PIDTrackerOutput | None
    achieved_force: np.ndarray
    achieved_yaw_moment: float
    requested_arm: bool
    transmitted_arm: bool
    command_sent: bool


class PIDHardwareSession:
    """Safety-gated bridge from one camera measurement to one v5 command."""

    def __init__(
        self,
        tracker: PIDTracker,
        connection: FineSUBConnection,
        hardware_adapter: FineSUBHardwareAdapter,
        vision_gate: PIDVisionGate | None = None,
    ) -> None:
        self.tracker = tracker
        self.connection = connection
        self.hardware_adapter = hardware_adapter
        self.vision_gate = vision_gate or PIDVisionGate()
        self.yaw_direct = bool(hardware_adapter.yaw_direct)
        # Keep yaw active at the hardware boundary.  The tracker latches the
        # current telemetry yaw when the first valid armed-session sample is
        # accepted, then holds that angle with its scalar yaw PID.
        self.yaw_output_frozen = False
        self.tracker.enable_yaw()
        self.tracker.track_target_bearing = False
        self._controller_initialized = False
        self._safety_latched = False
        self._armed_session_active = False
        self._last_arm_requested = False
        self._last_fresh_vision_monotonic: float | None = None
        self.vision_max_age_s = 0.25

    @property
    def safety_latched(self) -> bool:
        """Whether an armed safety fault requires a fresh explicit arm cycle."""
        return self._safety_latched

    @property
    def vision_ready(self) -> bool:
        return self.vision_gate.locked

    def connect(self) -> bool:
        """Open the port and send a zero/disarmed session-establishment frame."""
        if not self.connection.connect():
            return False
        self.vision_gate.reset()
        self._safety_latched = False
        self._armed_session_active = False
        self._last_arm_requested = False
        self._controller_initialized = False
        self._last_fresh_vision_monotonic = None
        return self.connection.send_disarm()

    def close(self) -> None:
        """Best-effort disarm followed by port close."""
        try:
            self.connection.send_disarm()
        finally:
            self.connection.close()
            self._controller_initialized = False
            self._armed_session_active = False
            self._last_arm_requested = False
            self._last_fresh_vision_monotonic = None

    def _latch_safety_stop(self) -> None:
        self._safety_latched = True
        self._armed_session_active = False
        self._controller_initialized = False
        self.vision_gate.reset(preserve_lock_history=True)
        self._last_fresh_vision_monotonic = None

    def _clear_explicit_disarm(self) -> None:
        """Clear a latched stop only after the caller has requested disarm."""
        self._safety_latched = False
        self._armed_session_active = False
        self._controller_initialized = False
        self.vision_gate.reset(preserve_lock_history=True)
        self._last_fresh_vision_monotonic = None

    def step(
        self,
        position_camera_xyz: object,
        *,
        arm_requested: bool,
        reference_position: object | None = None,
        reference_yaw_rad: float | None = None,
        fresh_vision_sample: bool = True,
    ) -> HardwareStepResult:
        """Run exactly one 20 Hz control update and transmit one command.

        Arming is suppressed until fresh v5 telemetry confirms this process's
        disarmed session. Invalid/stale telemetry and invalid/range-violating
        vision cause a zero disarm. An isolated implausible vision jump is
        held at the last valid position and does not interrupt an armed loop.
        Once an armed session hits a real safety fault, the stop is latched
        until the caller explicitly requests disarm and then arms again.
        """
        arm_requested = bool(arm_requested)
        if not arm_requested and self._last_arm_requested:
            self._clear_explicit_disarm()
        self._last_arm_requested = arm_requested

        if self._safety_latched:
            sent = self.connection.send_disarm()
            return HardwareStepResult(
                status="disarmed:latched_safety_stop",
                telemetry=self.connection.fresh_telemetry(),
                controller_output=None,
                achieved_force=np.zeros(3),
                achieved_yaw_moment=0.0,
                requested_arm=False,
                transmitted_arm=False,
                command_sent=sent,
            )

        telemetry = self.connection.fresh_telemetry()
        if telemetry is None:
            sent = self.connection.send_disarm()
            if self._armed_session_active or self._controller_initialized:
                self._latch_safety_stop()
            return HardwareStepResult(
                status="disarmed:no_fresh_telemetry",
                telemetry=None,
                controller_output=None,
                achieved_force=np.zeros(3),
                achieved_yaw_moment=0.0,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=sent,
            )

        achieved_force, achieved_yaw_moment = self.hardware_adapter.achieved_wrench(
            telemetry
        )
        telemetry_safe = (
            not telemetry.failsafe
            and not telemetry.last_command_rejected
            and telemetry.pid_direct
            and telemetry.yaw_direct == self.yaw_direct
            and telemetry.execution_feedback_valid
        )
        if not telemetry_safe:
            sent = self.connection.send_disarm()
            self._controller_initialized = False
            if self._armed_session_active or arm_requested:
                self._latch_safety_stop()
            return HardwareStepResult(
                status="disarmed:unsafe_telemetry",
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=sent,
            )

        if fresh_vision_sample:
            gate_result: VisionGateResult = self.vision_gate.update(position_camera_xyz)
            self._last_fresh_vision_monotonic = time.monotonic()
        elif (
            self._last_fresh_vision_monotonic is None
            or time.monotonic() - self._last_fresh_vision_monotonic > self.vision_max_age_s
        ):
            if telemetry.armed and self.connection.armed_confirmation_fresh():
                holding = self.target_lost(keep_armed=True)
                return HardwareStepResult(
                    status=(
                        "active:vision_gap_baseline"
                        if holding else "disarmed:vision_gap_hardware_fault"
                    ),
                    telemetry=telemetry,
                    controller_output=None,
                    achieved_force=achieved_force,
                    achieved_yaw_moment=achieved_yaw_moment,
                    requested_arm=arm_requested,
                    transmitted_arm=holding,
                    command_sent=holding,
                )
            sent = self.connection.send_disarm()
            if self._armed_session_active or self._controller_initialized:
                self._latch_safety_stop()
            return HardwareStepResult(
                status="disarmed:vision_stale",
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=sent,
            )
        else:
            gate_result = self.vision_gate.hold_without_sample()
        if not gate_result.ready:
            sent = self.connection.send_disarm()
            self._controller_initialized = False
            if gate_result.lost and (self._armed_session_active or arm_requested):
                self._latch_safety_stop()
            status = (
                "disarmed:vision_lost"
                if gate_result.lost
                else f"disarmed:{gate_result.reason}"
            )
            return HardwareStepResult(
                status=status,
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=sent,
            )

        if not arm_requested:
            sent = self.connection.send_disarm()
            return HardwareStepResult(
                status="disarmed:vision_ready",
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=False,
                transmitted_arm=False,
                command_sent=sent,
            )

        if not self._controller_initialized:
            self.tracker.latch_baseline(
                achieved_force,
                achieved_yaw_moment,
                telemetry.yaw_rad,
            )
            self._controller_initialized = True

        # The live sample is in OpenCV camera coordinates.  The gate may have
        # rejected an isolated jump and returned the previous valid sample;
        # always use that accepted/held position rather than the raw outlier.
        accepted_position_camera = gate_result.position_camera_xyz
        if accepted_position_camera is None:
            # ``ready`` implies a position by construction; keep this branch
            # fail-closed if a custom gate violates that contract.
            sent = self.connection.send_disarm()
            self._latch_safety_stop()
            return HardwareStepResult(
                status="disarmed:vision_missing_after_gate",
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=sent,
            )
        # Use the calibrated PID transform (including the real camera mount
        # offset), not the historical axis-aligned zero-origin fallback.
        position_body = camera_to_pid_body_position(accepted_position_camera)
        output = self.tracker.update(
            position_body,
            achieved_force,
            reference_position=reference_position,
            yaw_rad=telemetry.yaw_rad,
            yaw_rate_rad_s=telemetry.body_frd_yaw_rate_rad_s,
            achieved_yaw_moment_previous=achieved_yaw_moment,
            reference_yaw_rad=reference_yaw_rad,
        )
        yaw_moment = (
            0.0
            if self.yaw_output_frozen or output.yaw_pid is None
            else output.yaw_pid.yaw_moment
        )
        command = self.hardware_adapter.convert(
            output.pid.force,
            yaw_moment,
            armed=arm_requested,
        )
        sent = self.connection.send(command)
        if not sent:
            self._latch_safety_stop()
            return HardwareStepResult(
                status="disarmed:send_failed",
                telemetry=telemetry,
                controller_output=None,
                achieved_force=achieved_force,
                achieved_yaw_moment=achieved_yaw_moment,
                requested_arm=arm_requested,
                transmitted_arm=False,
                command_sent=False,
            )
        effective = self.connection.last_effective_command
        transmitted_arm = bool(effective is not None and effective.armed and sent)
        if not transmitted_arm:
            status = "disarmed:awaiting_session_confirmation"
        elif telemetry.armed and self.connection.armed_confirmation_fresh():
            status = "active:confirmed"
        else:
            status = "arming:command_sent"
        if telemetry.armed and self.connection.armed_confirmation_fresh():
            self._armed_session_active = True
        return HardwareStepResult(
            status=status,
            telemetry=telemetry,
            controller_output=output,
            achieved_force=achieved_force,
            achieved_yaw_moment=achieved_yaw_moment,
            requested_arm=arm_requested,
            transmitted_arm=transmitted_arm,
            command_sent=sent,
        )

    def target_lost(self, *, keep_armed: bool = False) -> bool:
        """On vision loss, disarm or return to baseline while already armed.

        The armed-baseline path never initiates arming. It requires fresh,
        accepted, armed telemetry and sends no stale PID position output.
        """
        if not keep_armed:
            if self._armed_session_active or self._last_arm_requested:
                self._latch_safety_stop()
            else:
                self._controller_initialized = False
                self.vision_gate.reset()
            return self.connection.send_disarm()
        telemetry = self.connection.fresh_telemetry()
        if (
            telemetry is None
            or not telemetry.armed
            or telemetry.failsafe
            or telemetry.last_command_rejected
            or not telemetry.pid_direct
            or telemetry.yaw_direct != self.yaw_direct
            or not telemetry.execution_feedback_valid
            or not self.connection.armed_confirmation_fresh()
        ):
            self.connection.send_disarm()
            self._latch_safety_stop()
            return False
        force, moment = self.hardware_adapter.achieved_wrench(telemetry)
        safe = self.tracker.target_lost(force, moment)
        safe_yaw_moment = 0.0 if self.yaw_output_frozen else safe.yaw_moment
        command = self.hardware_adapter.convert(
            safe.force,
            safe_yaw_moment,
            armed=True,
        )
        sent = self.connection.send(command)
        effective = self.connection.last_effective_command
        if not sent or effective is None or not effective.armed:
            self.connection.send_disarm()
            self._latch_safety_stop()
            return False
        return True


def build_serial_hardware_session(
    serial_port: str,
    *,
    logger=print,
) -> PIDHardwareSession:
    """Build the PID stack for V4pro1_MPC USART3 at 115200 8N1."""
    # Real hardware uses the calibrated camera/body frame and its matching
    # image-centre standoff reference.  The plain ``build_tracker()`` API
    # remains available for legacy aligned-frame simulations/tests.
    tracker = build_tracker(calibrated_reference=True)
    config = tracker.controller.config
    yaw_config = tracker.yaw_controller.config
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=config.force_max,
        negative_force_at_limit=-config.force_min,
        # Keep the serial path at the same active MPC translation authority.
        translation_channel_limits=(0.20, 0.20, 0.20),
        translation_signs=(1.0, 1.0, 1.0),
        positive_yaw_moment_at_limit=yaw_config.yaw_moment_max,
        negative_yaw_moment_at_limit=-yaw_config.yaw_moment_min,
        yaw_channel_limit=0.20,
        yaw_sign=1.0,
        yaw_direct=False,
    )
    connection = FineSUBConnection(
        SerialTransport(serial_port, baudrate=115200),
        telemetry_max_age_s=0.20,
        confirmation_max_age_s=0.30,
        logger=logger,
        yaw_direct=adapter.yaw_direct,
    )
    return PIDHardwareSession(tracker, connection, adapter, PIDVisionGate())


def _vision_gate_from_runtime_config(config: dict) -> PIDVisionGate:
    """Read the experiment's accepted vision-gate limits for PID.

    This does not import or execute an MPC controller.  Existing MPC runtime
    files put the accepted forward range and jump margin under
    ``experimental_auto.vision_gate_overrides``; a PID-specific
    ``pid_vision_gate`` object may override the same small set of fields.
    A PID-specific ``pid_vision_gate`` object may override the experiment's
    values without editing the global MPC runtime configuration.
    """

    values: dict[str, object] = {}
    explicit_values: dict[str, object] = {}
    experimental = config.get("experimental_auto")
    if isinstance(experimental, dict):
        overrides = experimental.get("vision_gate_overrides")
        if isinstance(overrides, dict):
            values.update(overrides)
    explicit = config.get("pid_vision_gate")
    if isinstance(explicit, dict):
        explicit_values.update(explicit)
        values.update(explicit)

    forward_range = values.get("forward_range_m")
    minimum = 0.15
    maximum = 2.50
    if isinstance(forward_range, (list, tuple)) and len(forward_range) == 2:
        minimum = float(forward_range[0])
        maximum = float(forward_range[1])
    return PIDVisionGate(
        VisionGateConfig(
            min_forward_m=minimum,
            max_forward_m=maximum,
            max_speed_m_s=float(values.get("max_speed_m_s", 1.0)),
            jump_margin_m=float(values.get("jump_margin_m", 0.10)),
            max_step_m=float(values.get("max_step_m", 0.30)),
            max_inter_sample_gap_s=float(
                values.get("max_inter_sample_gap_s", 0.50)
            ),
            startup_confirmation_samples=int(
                values.get("startup_confirmation_samples", 3)
            ),
            reacquire_confirmation_samples=int(
                values.get("reacquire_confirmation_samples", 5)
            ),
        )
    )


def build_runtime_hardware_session(
    runtime_config_path: str,
    *,
    tracker: PIDTracker | None = None,
    logger=print,
) -> PIDHardwareSession:
    """Build a PID session from the MPC-style transport JSON.

    Only the transport, timeout, and actuator-channel calibration are read
    from the file. PID gains and PID state always come from this directory's
    ``build_tracker`` (or the supplied tracker), so no MPC controller/model is
    imported or executed. Construction is side-effect free; the caller must
    explicitly invoke ``connect()`` before any disarm handshake is sent.
    """
    config = load_runtime_config(runtime_config_path)
    transport_config = config.get("transport")
    if not isinstance(transport_config, dict):
        raise ValueError("runtime config must contain a transport object")
    control_config = config.get("control")
    if not isinstance(control_config, dict):
        control_config = {}
    adapter_config = config.get("hardware_adapter")
    if not isinstance(adapter_config, dict):
        adapter_config = {}

    active_tracker = (
        build_tracker(calibrated_reference=True) if tracker is None else tracker
    )
    tracker_config = active_tracker.controller.config
    yaw_config = active_tracker.yaw_controller.config
    positive_force = adapter_config.get(
        "positive_force_at_limit", tracker_config.force_max
    )
    negative_force = adapter_config.get(
        "negative_force_at_limit", -tracker_config.force_min
    )
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=positive_force,
        negative_force_at_limit=negative_force,
        # Use the active MPC adapter envelope.  This is read from the MPC
        # runtime JSON when present, while the fallback remains the same
        # operator-authorized 0.20 cap.
        translation_channel_limits=adapter_config.get(
            "translation_channel_limits", (0.20, 0.20, 0.20)
        ),
        translation_signs=adapter_config.get("translation_signs", (1.0, 1.0, 1.0)),
        positive_yaw_moment_at_limit=adapter_config.get(
            "positive_yaw_moment_at_limit", yaw_config.yaw_moment_max
        ),
        negative_yaw_moment_at_limit=adapter_config.get(
            "negative_yaw_moment_at_limit", -yaw_config.yaw_moment_min
        ),
        yaw_channel_limit=adapter_config.get("yaw_channel_limit", 0.20),
        yaw_sign=adapter_config.get("yaw_sign", 1.0),
        yaw_direct=adapter_config.get("yaw_direct", False),
    )
    connection = FineSUBConnection(
        make_transport(transport_config),
        telemetry_max_age_s=float(control_config.get("telemetry_max_age_sec", 0.20)),
        confirmation_max_age_s=float(
            control_config.get("confirmation_max_age_sec", 0.30)
        ),
        reconnect_interval_s=float(
            transport_config.get("reconnect_interval_sec", 1.0)
        ),
        logger=logger,
        yaw_direct=adapter.yaw_direct,
    )
    return PIDHardwareSession(
        active_tracker,
        connection,
        adapter,
        _vision_gate_from_runtime_config(config),
    )
