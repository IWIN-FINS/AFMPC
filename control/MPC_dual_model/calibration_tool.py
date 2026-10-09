"""Safety-gated FineSUB communication and low-limit calibration runner.

The v4 wire protocol provides both four normalized mixer channels and a
dedicated physical single-motor calibration command.  ``motor-step`` is hard
limited to 10 percent raw DSHOT throttle in both host and firmware; it bypasses
the attitude/depth mixer and never uses the low-RPM closed loop.

``link`` is disarmed-only and is the first real-vehicle test.  ``channel-step``
requires explicit operator confirmations and is hard limited to 10 percent of
the normalized mixer channel range.  Both modes always send a disarmed zero
burst on exit and write a CSV row even when telemetry is missing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Iterable

from .finesub_protocol import FineSUBControlCommand, FineSUBTelemetry
from .finesub_transport import FineSUBConnection, load_runtime_config, make_transport


DEFAULT_CONFIG_PATH = Path(__file__).with_name("finesub_v4pro1_mpc.json")
CHANNEL_NAMES = ("forward", "right", "down", "yaw")
MAX_CALIBRATION_CHANNEL_ABS = 0.10


def _json_vector(values: Iterable[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _telemetry_age(telemetry: FineSUBTelemetry | None, now: float) -> float:
    if telemetry is None:
        return float("nan")
    return max(0.0, now - telemetry.received_monotonic)


CSV_FIELDS = (
    "host_time_utc",
    "host_monotonic_s",
    "phase",
    "requested_forward",
    "requested_right",
    "requested_down",
    "requested_yaw",
    "requested_motor_index",
    "requested_motor_throttle",
    "requested_armed",
    "sent",
    "transport_open",
    "session_confirmed",
    "armed_confirmation_fresh",
    "telemetry_fresh",
    "telemetry_age_s",
    "telemetry_sequence",
    "mcu_tick_ms",
    "state",
    "telemetry_armed",
    "telemetry_mpc_direct",
    "telemetry_yaw_direct",
    "telemetry_failsafe",
    "command_status",
    "reject_flags",
    "last_command_session",
    "last_command_sequence",
    "last_command_crc",
    "last_command_sender_time_ms",
    "command_count",
    "rejected_command_count",
    "yaw_rad",
    "yaw_rate_rad_s",
    "imu_quat_wxyz",
    "imu_angular_velocity_xyz_rad_s",
    "imu_linear_acceleration_xyz_m_s2",
    "depth_m",
    "pressure_pa",
    "received_channels",
    "applied_motor_throttle_m1_m8",
    "motor_rpm_m1_m8",
    "rpm_valid_mask",
    "execution_feedback_valid",
    "rpm_available",
)


class CalibrationCsvRecorder:
    """Write a self-contained, row-oriented command/telemetry log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite a previous raw recording.
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CalibrationCsvRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def record(
        self,
        *,
        phase: str,
        command: FineSUBControlCommand,
        sent: bool,
        connection: FineSUBConnection,
        telemetry: FineSUBTelemetry | None,
        telemetry_fresh: bool,
        now: float | None = None,
    ) -> None:
        monotonic = time.monotonic() if now is None else float(now)
        wall_time = datetime.now(timezone.utc).isoformat()
        row = {field: "" for field in CSV_FIELDS}
        row.update(
            host_time_utc=wall_time,
            host_monotonic_s=f"{monotonic:.6f}",
            phase=phase,
            requested_forward=f"{command.forward:.7f}",
            requested_right=f"{command.right:.7f}",
            requested_down=f"{command.down:.7f}",
            requested_yaw=f"{command.yaw:.7f}",
            requested_motor_index=(
                ""
                if command.calibration_motor_index is None
                else command.calibration_motor_index
            ),
            requested_motor_throttle=(
                ""
                if command.calibration_motor_index is None
                else f"{command.calibration_motor_throttle:.7f}"
            ),
            requested_armed=int(command.armed),
            sent=int(sent),
            transport_open=int(bool(getattr(connection, "_opened", False))),
            session_confirmed=int(connection.session_confirmed),
            armed_confirmation_fresh=int(connection.armed_confirmation_fresh()),
            telemetry_fresh=int(telemetry_fresh),
            telemetry_age_s=f"{_telemetry_age(telemetry, monotonic):.6f}",
        )
        if telemetry is not None:
            row.update(
                telemetry_sequence=telemetry.sequence,
                mcu_tick_ms=telemetry.tick_ms,
                state=telemetry.state,
                telemetry_armed=int(telemetry.armed),
                telemetry_mpc_direct=int(telemetry.mpc_direct),
                telemetry_yaw_direct=int(telemetry.yaw_direct),
                telemetry_failsafe=int(telemetry.failsafe),
                command_status=telemetry.command_status,
                reject_flags=telemetry.reject_flags,
                last_command_session=telemetry.last_command_session,
                last_command_sequence=telemetry.last_command_sequence,
                last_command_crc=telemetry.last_command_crc,
                last_command_sender_time_ms=telemetry.last_command_sender_time_ms,
                command_count=telemetry.command_count,
                rejected_command_count=telemetry.rejected_command_count,
                yaw_rad=f"{telemetry.yaw_rad:.7f}",
                yaw_rate_rad_s=f"{telemetry.yaw_rate_rad_s:.7f}",
                imu_quat_wxyz=_json_vector(telemetry.quat_wxyz),
                imu_angular_velocity_xyz_rad_s=_json_vector(
                    telemetry.angular_velocity_xyz
                ),
                imu_linear_acceleration_xyz_m_s2=_json_vector(
                    telemetry.linear_acceleration_xyz
                ),
                depth_m=f"{telemetry.depth_m:.7f}",
                pressure_pa=f"{telemetry.pressure_pa:.3f}",
                received_channels=_json_vector(
                    (telemetry.forward, telemetry.right, telemetry.down, telemetry.yaw)
                ),
                applied_motor_throttle_m1_m8=_json_vector(
                    telemetry.applied_motor_throttle
                ),
                motor_rpm_m1_m8=_json_vector(telemetry.motor_rpm),
                rpm_valid_mask=telemetry.rpm_valid_mask,
                execution_feedback_valid=int(telemetry.execution_feedback_valid),
                rpm_available=int(telemetry.rpm_available),
            )
        self._writer.writerow(row)
        self._handle.flush()


@dataclass(frozen=True)
class ChannelStep:
    channel: str
    amplitude: float
    hold_s: float
    cycles: int
    neutral_hold_s: float | None = None
    direction: str = "both"

    def __post_init__(self) -> None:
        if self.channel not in CHANNEL_NAMES:
            raise ValueError(f"channel must be one of {CHANNEL_NAMES}")
        if not 0.0 < float(self.amplitude) <= MAX_CALIBRATION_CHANNEL_ABS:
            raise ValueError(
                f"amplitude must be in (0, {MAX_CALIBRATION_CHANNEL_ABS}]"
            )
        if float(self.hold_s) <= 0.0:
            raise ValueError("hold_s must be positive")
        if self.neutral_hold_s is not None and float(self.neutral_hold_s) <= 0.0:
            raise ValueError("neutral_hold_s must be positive when provided")
        if self.direction not in ("positive", "negative", "both"):
            raise ValueError("direction must be positive, negative, or both")
        if int(self.cycles) <= 0:
            raise ValueError("cycles must be positive")

    @property
    def resolved_neutral_hold_s(self) -> float:
        return (
            float(self.hold_s)
            if self.neutral_hold_s is None
            else float(self.neutral_hold_s)
        )


@dataclass(frozen=True)
class MotorStep:
    motor_index: int
    amplitude: float
    hold_s: float
    cycles: int

    def __post_init__(self) -> None:
        if not 1 <= int(self.motor_index) <= 8:
            raise ValueError("motor_index must be in [1, 8]")
        if int(self.motor_index) != self.motor_index:
            raise ValueError("motor_index must be an integer")
        if not 0.0 < abs(float(self.amplitude)) <= MAX_CALIBRATION_CHANNEL_ABS:
            raise ValueError(
                f"absolute amplitude must be in (0, {MAX_CALIBRATION_CHANNEL_ABS}]"
            )
        if float(self.hold_s) <= 0.0:
            raise ValueError("hold_s must be positive")
        if int(self.cycles) <= 0:
            raise ValueError("cycles must be positive")


def _command(values: tuple[float, float, float, float], armed: bool) -> FineSUBControlCommand:
    return FineSUBControlCommand(*values, armed=armed, yaw_direct=True)


def _channel_command(channel: str, value: float, *, armed: bool) -> FineSUBControlCommand:
    values = [0.0] * 4
    values[CHANNEL_NAMES.index(channel)] = float(value)
    return FineSUBControlCommand(
        *values,
        armed=armed,
        yaw_direct=True,
        calibration_channel=True,
    )


def _motor_command(motor_index: int, value: float, *, armed: bool) -> FineSUBControlCommand:
    return FineSUBControlCommand(
        0.0,
        0.0,
        0.0,
        0.0,
        armed=armed,
        yaw_direct=True,
        calibration_motor_index=int(motor_index),
        calibration_motor_throttle=float(value),
    )


def _send_disarmed_burst(connection: FineSUBConnection, period_s: float) -> None:
    zero = _command((0.0, 0.0, 0.0, 0.0), armed=False)
    deadline = time.monotonic() + max(0.20, 4.0 * period_s)
    while time.monotonic() < deadline:
        connection.poll_telemetry()
        connection.send(zero)
        time.sleep(max(0.01, min(period_s, 0.05)))
    connection.send(zero)


def _make_connection(config_path: str | Path) -> FineSUBConnection:
    runtime_config = load_runtime_config(config_path)
    control = runtime_config.get("control", {})
    transport = runtime_config.get("transport", {})
    return FineSUBConnection(
        make_transport(transport),
        telemetry_max_age_s=float(control.get("telemetry_max_age_sec", 0.25)),
        confirmation_max_age_s=float(control.get("confirmation_max_age_sec", 0.25)),
        reconnect_interval_s=float(transport.get("reconnect_interval_sec", 1.0)),
    )


def run_link_test(
    config_path: str | Path,
    csv_path: str | Path,
    *,
    duration_s: float,
    period_s: float,
) -> None:
    """Run a disarmed-only link test and record all available telemetry."""

    if duration_s <= 0.0 or period_s <= 0.0:
        raise ValueError("duration_s and period_s must be positive")
    connection = _make_connection(config_path)
    zero = _command((0.0, 0.0, 0.0, 0.0), armed=False)
    start = time.monotonic()
    next_tick = start
    print(f"[CAL] disarmed link test for {duration_s:.1f}s; CSV={Path(csv_path)}")
    try:
        with CalibrationCsvRecorder(csv_path) as recorder:
            while time.monotonic() - start < duration_s:
                now = time.monotonic()
                connection.poll_telemetry()
                telemetry = connection.fresh_telemetry()
                sent = connection.send(zero)
                recorder.record(
                    phase="disarmed_link",
                    command=zero,
                    sent=sent,
                    connection=connection,
                    telemetry=telemetry,
                    telemetry_fresh=telemetry is not None,
                    now=now,
                )
                next_tick += period_s
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        _send_disarmed_burst(connection, period_s)
        connection.close()
        print(f"[CAL] disarmed link test stopped; CSV={Path(csv_path)}")


def run_channel_step_test(
    config_path: str | Path,
    csv_path: str | Path,
    step: ChannelStep,
    *,
    period_s: float,
    confirm_water_tethered: bool,
    confirm_propeller_guards: bool,
    confirm_physical_estop: bool,
    max_depth_excursion_m: float | None = None,
) -> None:
    """Run a guarded low-limit mixer-channel step test."""

    runtime_config = load_runtime_config(config_path)
    control_config = runtime_config.get("control", {})
    if not bool(control_config.get("calibration_channel_step_enabled", False)):
        reason = control_config.get(
            "calibration_channel_step_disabled_reason",
            "the physical mixer has not been verified",
        )
        raise RuntimeError(f"channel-step is disabled: {reason}")

    if not (
        confirm_water_tethered
        and confirm_propeller_guards
        and confirm_physical_estop
    ):
        raise RuntimeError(
            "channel-step requires --confirm-water-tethered, "
            "--confirm-propeller-guards, and --confirm-physical-estop"
        )
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if max_depth_excursion_m is not None and max_depth_excursion_m <= 0.0:
        raise ValueError("max_depth_excursion_m must be positive when provided")
    if step.channel == "down" and max_depth_excursion_m is None:
        raise RuntimeError(
            "down channel-step requires --max-depth-excursion as an automatic "
            "vertical travel limit"
        )

    connection = _make_connection(config_path)
    zero_disarmed = _command((0.0, 0.0, 0.0, 0.0), armed=False)
    print(
        f"[CAL] guarded {step.channel} mixer step at +/-{step.amplitude:.3f}; "
        f"CSV={Path(csv_path)}"
    )
    start = time.monotonic()
    initial_depth_m: float | None = None
    phases = []
    neutral_hold_s = step.resolved_neutral_hold_s
    for cycle in range(step.cycles):
        phases.append(
            (f"cycle_{cycle + 1}_neutral_before", 0.0, neutral_hold_s)
        )
        if step.direction in ("positive", "both"):
            phases.append(
                (f"cycle_{cycle + 1}_positive", step.amplitude, step.hold_s)
            )
        if step.direction == "both":
            phases.append((f"cycle_{cycle + 1}_neutral", 0.0, neutral_hold_s))
        if step.direction in ("negative", "both"):
            phases.append(
                (f"cycle_{cycle + 1}_negative", -step.amplitude, step.hold_s)
            )
        phases.append(
            (f"cycle_{cycle + 1}_neutral_after", 0.0, neutral_hold_s)
        )
    try:
        with CalibrationCsvRecorder(csv_path) as recorder:
            for phase, value, hold_s in phases:
                phase_start = time.monotonic()
                while time.monotonic() - phase_start < hold_s:
                    now = time.monotonic()
                    connection.poll_telemetry()
                    telemetry = connection.fresh_telemetry()
                    if telemetry is not None and initial_depth_m is None:
                        initial_depth_m = float(telemetry.depth_m)
                    if (
                        telemetry is not None
                        and initial_depth_m is not None
                        and max_depth_excursion_m is not None
                        and abs(float(telemetry.depth_m) - initial_depth_m)
                        > max_depth_excursion_m
                    ):
                        raise RuntimeError(
                            "depth excursion exceeded automatic limit: "
                            f"start={initial_depth_m:.3f} m, "
                            f"current={telemetry.depth_m:.3f} m, "
                            f"limit={max_depth_excursion_m:.3f} m"
                        )
                    ready = (
                        telemetry is not None
                        and not telemetry.failsafe
                        and connection.session_confirmed
                    )
                    requested = (
                        _channel_command(step.channel, value, armed=True)
                        if ready and value != 0.0
                        else zero_disarmed
                    )
                    sent = connection.send(requested)
                    recorder.record(
                        phase=phase if ready and value != 0.0 else f"{phase}_blocked",
                        command=requested,
                        sent=sent,
                        connection=connection,
                        telemetry=telemetry,
                        telemetry_fresh=telemetry is not None,
                        now=now,
                    )
                    if telemetry is not None and telemetry.failsafe:
                        raise RuntimeError("lower controller reports failsafe")
                    time.sleep(period_s)
    finally:
        _send_disarmed_burst(connection, period_s)
        connection.close()
        print(f"[CAL] mixer step stopped; CSV={Path(csv_path)}")


def run_motor_step_test(
    config_path: str | Path,
    csv_path: str | Path,
    step: MotorStep,
    *,
    period_s: float,
    confirm_water_tethered: bool,
    confirm_propeller_guards: bool,
    confirm_physical_estop: bool,
) -> None:
    """Run a guarded, hard-limited physical single-motor step."""

    if not (
        confirm_water_tethered
        and confirm_propeller_guards
        and confirm_physical_estop
    ):
        raise RuntimeError(
            "motor-step requires --confirm-water-tethered, "
            "--confirm-propeller-guards, and --confirm-physical-estop"
        )
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")

    connection = _make_connection(config_path)
    zero_disarmed = _command((0.0, 0.0, 0.0, 0.0), armed=False)
    active = _motor_command(step.motor_index, step.amplitude, armed=True)
    print(
        f"[CAL] guarded M{step.motor_index} physical motor step at "
        f"{step.amplitude:+.3f}; CSV={Path(csv_path)}"
    )
    phases = []
    for cycle in range(step.cycles):
        phases.extend(
            [
                (f"cycle_{cycle + 1}_neutral_before", zero_disarmed, 0.5),
                (f"cycle_{cycle + 1}_active", active, step.hold_s),
                (f"cycle_{cycle + 1}_neutral_after", zero_disarmed, 1.0),
            ]
        )

    try:
        with CalibrationCsvRecorder(csv_path) as recorder:
            for phase, requested_active, hold_s in phases:
                phase_start = time.monotonic()
                while time.monotonic() - phase_start < hold_s:
                    now = time.monotonic()
                    connection.poll_telemetry()
                    telemetry = connection.fresh_telemetry()
                    ready = (
                        telemetry is not None
                        and not telemetry.failsafe
                        and connection.session_confirmed
                    )
                    requested = (
                        requested_active
                        if ready and requested_active.armed
                        else zero_disarmed
                    )
                    sent = connection.send(requested)
                    recorder.record(
                        phase=phase if requested is requested_active else f"{phase}_blocked",
                        command=requested,
                        sent=sent,
                        connection=connection,
                        telemetry=telemetry,
                        telemetry_fresh=telemetry is not None,
                        now=now,
                    )
                    if telemetry is not None and telemetry.failsafe:
                        raise RuntimeError("lower controller reports failsafe")
                    if telemetry is not None and telemetry.armed:
                        applied = telemetry.applied_motor_throttle
                        selected = step.motor_index - 1
                        if abs(applied[selected]) > MAX_CALIBRATION_CHANNEL_ABS + 1.0e-5:
                            raise RuntimeError("selected motor exceeded calibration limit")
                        if any(
                            abs(value) > 1.0e-5
                            for index, value in enumerate(applied)
                            if index != selected
                        ):
                            raise RuntimeError("non-selected motor received non-zero throttle")
                    time.sleep(period_s)
    finally:
        _send_disarmed_burst(connection, period_s)
        connection.close()
        print(f"[CAL] physical motor step stopped; CSV={Path(csv_path)}")


def run_zero_failsafe_test(
    config_path: str | Path,
    csv_path: str | Path,
    *,
    period_s: float,
    establish_s: float,
    armed_zero_s: float,
    silence_s: float,
    recovery_s: float,
    confirm_propulsion_disconnected: bool,
) -> dict[str, bool]:
    """Verify command-timeout failsafe without ever requesting non-zero output."""

    if not confirm_propulsion_disconnected:
        raise RuntimeError(
            "failsafe-zero requires --confirm-propulsion-disconnected "
            "(or all propellers removed)"
        )
    durations = (establish_s, armed_zero_s, silence_s, recovery_s)
    if period_s <= 0.0 or any(value <= 0.0 for value in durations):
        raise ValueError("period and all failsafe phase durations must be positive")

    connection = _make_connection(config_path)
    disarmed_zero = _command((0.0, 0.0, 0.0, 0.0), armed=False)
    armed_zero = _command((0.0, 0.0, 0.0, 0.0), armed=True)
    established = False
    armed_confirmed = False
    failsafe_seen = False
    recovered_disarmed = False

    def record_phase(
        recorder: CalibrationCsvRecorder,
        *,
        phase: str,
        duration: float,
        command: FineSUBControlCommand,
        send: bool,
    ) -> None:
        nonlocal established, armed_confirmed, failsafe_seen, recovered_disarmed
        deadline = time.monotonic() + duration
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            connection.poll_telemetry()
            telemetry = connection.fresh_telemetry()
            sent = connection.send(command) if send else False
            if telemetry is not None:
                established = established or (
                    connection.session_confirmed and not telemetry.armed
                )
                armed_confirmed = armed_confirmed or (
                    telemetry.armed and connection.armed_confirmation_fresh()
                )
                failsafe_seen = failsafe_seen or (
                    phase == "silence_no_write" and telemetry.failsafe
                )
                recovered_disarmed = recovered_disarmed or (
                    phase == "recovery_disarmed"
                    and not telemetry.armed
                    and not telemetry.failsafe
                    and connection.confirmation_fresh()
                    and connection.last_confirmed_armed is False
                )
                if phase == "silence_no_write" and telemetry.failsafe:
                    if any(abs(value) > 1e-6 for value in telemetry.applied_motor_throttle):
                        raise RuntimeError(
                            "failsafe reported but applied motor throttle is non-zero"
                        )
            recorder.record(
                phase=phase,
                command=command,
                sent=sent,
                connection=connection,
                telemetry=telemetry,
                telemetry_fresh=telemetry is not None,
                now=now,
            )
            next_tick += period_s
            time.sleep(max(0.0, next_tick - time.monotonic()))

    print(f"[CAL] zero-only failsafe test; CSV={Path(csv_path)}")
    try:
        with CalibrationCsvRecorder(csv_path) as recorder:
            record_phase(
                recorder,
                phase="establish_disarmed",
                duration=establish_s,
                command=disarmed_zero,
                send=True,
            )
            if not established:
                raise RuntimeError("disarmed session/telemetry was not confirmed")
            record_phase(
                recorder,
                phase="armed_zero",
                duration=armed_zero_s,
                command=armed_zero,
                send=True,
            )
            if not armed_confirmed:
                raise RuntimeError("armed-zero command was not positively confirmed")
            record_phase(
                recorder,
                phase="silence_no_write",
                duration=silence_s,
                command=armed_zero,
                send=False,
            )
            record_phase(
                recorder,
                phase="recovery_disarmed",
                duration=recovery_s,
                command=disarmed_zero,
                send=True,
            )
    finally:
        _send_disarmed_burst(connection, period_s)
        connection.close()

    result = {
        "disarmed_session_confirmed": established,
        "armed_zero_confirmed": armed_confirmed,
        "failsafe_seen_after_silence": failsafe_seen,
        "recovered_disarmed": recovered_disarmed,
    }
    print(f"[CAL] failsafe result: {json.dumps(result, sort_keys=True)}")
    if not all(result.values()):
        raise RuntimeError(f"zero-only failsafe test failed: {result}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FineSUB disarmed link recorder and guarded low-limit calibration"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    link = subparsers.add_parser("link", help="disarmed-only communication test")
    link.add_argument("--duration", type=float, default=10.0)
    link.add_argument("--period", type=float, default=0.05)
    link.add_argument("--csv", type=Path, required=True)

    failsafe = subparsers.add_parser(
        "failsafe-zero",
        help="armed-zero command timeout test; propulsion must be disconnected",
    )
    failsafe.add_argument("--period", type=float, default=0.05)
    failsafe.add_argument("--establish", type=float, default=2.0)
    failsafe.add_argument("--armed-zero", type=float, default=1.0)
    failsafe.add_argument("--silence", type=float, default=1.5)
    failsafe.add_argument("--recovery", type=float, default=1.0)
    failsafe.add_argument("--csv", type=Path, required=True)
    failsafe.add_argument("--confirm-propulsion-disconnected", action="store_true")

    step = subparsers.add_parser(
        "channel-step",
        help="guarded four-channel mixer step; not a single-motor test",
    )
    step.add_argument("--channel", choices=CHANNEL_NAMES, required=True)
    step.add_argument("--amplitude", type=float, default=0.05)
    step.add_argument("--hold", type=float, default=1.0)
    step.add_argument(
        "--neutral-hold",
        type=float,
        default=None,
        help="zero-output duration for each neutral phase; defaults to --hold",
    )
    step.add_argument(
        "--direction",
        choices=("positive", "negative", "both"),
        default="both",
        help="run one sign only or the historical positive-and-negative sequence",
    )
    step.add_argument("--cycles", type=int, default=1)
    step.add_argument("--period", type=float, default=0.05)
    step.add_argument("--csv", type=Path, required=True)
    step.add_argument("--confirm-water-tethered", action="store_true")
    step.add_argument("--confirm-propeller-guards", action="store_true")
    step.add_argument("--confirm-physical-estop", action="store_true")
    step.add_argument(
        "--max-depth-excursion",
        type=float,
        default=None,
        help=(
            "abort and disarm if pressure depth changes by more than this many "
            "metres from the first fresh sample; required for --channel down"
        ),
    )

    motor = subparsers.add_parser(
        "motor-step",
        help="guarded physical single-motor step, hard limited to +/-0.10",
    )
    motor.add_argument("--motor", type=int, choices=range(1, 9), required=True)
    motor.add_argument("--amplitude", type=float, default=0.05)
    motor.add_argument("--hold", type=float, default=1.0)
    motor.add_argument("--cycles", type=int, default=1)
    motor.add_argument("--period", type=float, default=0.05)
    motor.add_argument("--csv", type=Path, required=True)
    motor.add_argument("--confirm-water-tethered", action="store_true")
    motor.add_argument("--confirm-propeller-guards", action="store_true")
    motor.add_argument("--confirm-physical-estop", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "link":
        run_link_test(
            args.config,
            args.csv,
            duration_s=args.duration,
            period_s=args.period,
        )
        return 0
    if args.mode == "channel-step":
        run_channel_step_test(
            args.config,
            args.csv,
            ChannelStep(
                args.channel,
                args.amplitude,
                args.hold,
                args.cycles,
                args.neutral_hold,
                args.direction,
            ),
            period_s=args.period,
            confirm_water_tethered=args.confirm_water_tethered,
            confirm_propeller_guards=args.confirm_propeller_guards,
            confirm_physical_estop=args.confirm_physical_estop,
            max_depth_excursion_m=args.max_depth_excursion,
        )
        return 0
    if args.mode == "motor-step":
        run_motor_step_test(
            args.config,
            args.csv,
            MotorStep(args.motor, args.amplitude, args.hold, args.cycles),
            period_s=args.period,
            confirm_water_tethered=args.confirm_water_tethered,
            confirm_propeller_guards=args.confirm_propeller_guards,
            confirm_physical_estop=args.confirm_physical_estop,
        )
        return 0
    if args.mode == "failsafe-zero":
        run_zero_failsafe_test(
            args.config,
            args.csv,
            period_s=args.period,
            establish_s=args.establish,
            armed_zero_s=args.armed_zero,
            silence_s=args.silence,
            recovery_s=args.recovery,
            confirm_propulsion_disconnected=args.confirm_propulsion_disconnected,
        )
        return 0
    raise AssertionError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
