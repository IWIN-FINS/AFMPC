"""FineSUB v5 serial protocol used by the PID controller.

This mirrors ``V4pro1_MPC/Services/V5Streamer/Streamer.hpp`` without importing
any MPC Python package.  Body axes are FRD and positive yaw turns the bow right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
import time

import numpy as np

try:
    from .pid_controller import vector3
except ImportError:
    from pid_controller import vector3


PROTOCOL_VERSION = 5
COMMAND_SYNC = b"\xAA\x00"
COMMAND_TAIL = 0xBB
TELEMETRY_SYNC = b"\x55\x54"
TELEMETRY_MESSAGE_STATE_EXECUTION = 0x01

COMMAND_FLAG_ARMED = 0x01
COMMAND_FLAG_MPC_DIRECT = 0x02  # Firmware transport flag used by the PID bridge.
COMMAND_FLAG_YAW_DIRECT = 0x04

TELEMETRY_FLAG_ARMED = 0x01
TELEMETRY_FLAG_MPC_DIRECT = 0x02
TELEMETRY_FLAG_FAILSAFE = 0x04
TELEMETRY_FLAG_YAW_DIRECT = 0x08
TELEMETRY_FLAG_EXECUTION_FEEDBACK = 0x10
TELEMETRY_FLAG_RPM_AVAILABLE = 0x20

COMMAND_STATUS_NONE = 0
COMMAND_STATUS_ACCEPTED = 1
COMMAND_STATUS_REJECTED = 2

COMMAND_REJECT_CRC = 1 << 0
COMMAND_REJECT_VERSION = 1 << 1
COMMAND_REJECT_FLAGS = 1 << 2
COMMAND_REJECT_NONFINITE = 1 << 3
COMMAND_REJECT_STALE_SEQUENCE = 1 << 4
COMMAND_REJECT_SESSION_REQUIRES_DISARM = 1 << 5
COMMAND_REJECT_FORMAT = 1 << 6
COMMAND_REJECT_CALIBRATION = 1 << 7

_COMMAND_BODY = struct.Struct("<2sBBHIIffff4x")
_TELEMETRY_HEADER = struct.Struct("<2sBBHHI")
_TELEMETRY_PAYLOAD = struct.Struct("<BBBBIIHHIII33f")
_CRC = struct.Struct("<H")

COMMAND_FRAME_SIZE = _COMMAND_BODY.size + _CRC.size + 1
TELEMETRY_PAYLOAD_SIZE = _TELEMETRY_PAYLOAD.size
TELEMETRY_FRAME_SIZE = _TELEMETRY_HEADER.size + TELEMETRY_PAYLOAD_SIZE + _CRC.size
MOTOR_COUNT = 8

# Exact current matrices and physical indices from V4pro1_MPC/V5_SUB.hpp.
_LOWER_MIXER = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)  # columns: yaw, forward, right; physical motors M1,M2,M6,M7
_UPPER_MIXER = np.asarray(
    [
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=float,
)  # columns: roll, pitch, down; physical motors M3,M4,M5,M8


def _finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(item) for item in array)


def crc16_modbus(data: bytes | bytearray | memoryview) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def is_newer_u16_sequence(current: int, previous: int) -> bool:
    delta = (int(current) - int(previous)) & 0xFFFF
    return 0 < delta < 0x8000


@dataclass(frozen=True)
class FineSUBControlCommand:
    """Normalized lower-controller channels, not SI force/moment."""

    forward: float
    right: float
    down: float
    yaw: float
    armed: bool
    yaw_direct: bool = True

    def __post_init__(self) -> None:
        values = _finite_tuple(
            (self.forward, self.right, self.down, self.yaw), 4, "control channels"
        )
        limits = (0.35, 0.35, 0.50, 0.20)
        if any(abs(value) > limit + 1.0e-6 for value, limit in zip(values, limits)):
            raise ValueError("control channel exceeds V4pro1_MPC firmware limit")


@dataclass(frozen=True)
class FineSUBCommandEnvelope:
    command: FineSUBControlCommand
    sequence: int
    session_id: int
    sender_time_ms: int
    crc: int


@dataclass(frozen=True)
class FineSUBTelemetry:
    sequence: int
    tick_ms: int
    state: int
    armed: bool
    pid_direct: bool
    yaw_direct: bool
    failsafe: bool
    yaw_rad: float
    yaw_rate_rad_s: float
    depth_m: float
    forward: float
    right: float
    down: float
    yaw: float
    command_status: int = COMMAND_STATUS_NONE
    reject_flags: int = 0
    last_command_session: int = 0
    last_command_sequence: int = 0
    last_command_crc: int = 0
    last_command_sender_time_ms: int = 0
    command_count: int = 0
    rejected_command_count: int = 0
    quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_xyz: tuple[float, ...] = (0.0, 0.0, 0.0)
    linear_acceleration_xyz: tuple[float, ...] = (0.0, 0.0, 0.0)
    pressure_pa: float = 0.0
    applied_motor_throttle: tuple[float, ...] = (0.0,) * MOTOR_COUNT
    motor_rpm: tuple[float, ...] = (0.0,) * MOTOR_COUNT
    execution_feedback_valid: bool = False
    rpm_available: bool = False
    rpm_valid_mask: int = 0
    received_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        _finite_tuple(
            (
                self.yaw_rad,
                self.yaw_rate_rad_s,
                self.depth_m,
                self.forward,
                self.right,
                self.down,
                self.yaw,
                self.pressure_pa,
                self.received_monotonic,
            ),
            9,
            "telemetry scalars",
        )
        object.__setattr__(self, "quat_wxyz", _finite_tuple(self.quat_wxyz, 4, "quat"))
        object.__setattr__(
            self,
            "angular_velocity_xyz",
            _finite_tuple(self.angular_velocity_xyz, 3, "angular velocity"),
        )
        object.__setattr__(
            self,
            "linear_acceleration_xyz",
            _finite_tuple(self.linear_acceleration_xyz, 3, "linear acceleration"),
        )
        object.__setattr__(
            self,
            "applied_motor_throttle",
            _finite_tuple(self.applied_motor_throttle, MOTOR_COUNT, "motor throttle"),
        )
        object.__setattr__(
            self,
            "motor_rpm",
            _finite_tuple(self.motor_rpm, MOTOR_COUNT, "motor RPM"),
        )

    @property
    def mpc_direct(self) -> bool:
        """Firmware compatibility alias; this channel is driven by PID here."""
        return self.pid_direct

    @property
    def last_command_accepted(self) -> bool:
        return self.command_status == COMMAND_STATUS_ACCEPTED

    @property
    def last_command_rejected(self) -> bool:
        return self.command_status == COMMAND_STATUS_REJECTED

    @property
    def body_frd_yaw_rate_rad_s(self) -> float:
        """Return raw H30 yaw rate in the body-FRD sign convention."""
        # V5_SUB negates the raw IMU angular-rate vector before feeding its
        # attitude controllers. Keep the wire value unchanged and expose the
        # converted value to the host-side PID controller.
        return -float(self.angular_velocity_xyz[2])


def is_newer_telemetry(current: FineSUBTelemetry, previous: FineSUBTelemetry) -> bool:
    if is_newer_u16_sequence(current.sequence, previous.sequence):
        return True
    return not current.armed and current.sequence < 8 and current.tick_ms < previous.tick_ms


def motor_throttles_to_channels(
    motor_throttle: object,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Invert the current V4pro1_MPC physical M1..M8 mixer."""
    physical = np.asarray(
        _finite_tuple(motor_throttle, MOTOR_COUNT, "motor_throttle"), dtype=float
    )
    lower = physical[[0, 1, 5, 6]]
    upper = physical[[2, 3, 4, 7]]
    yaw_forward_right = np.linalg.lstsq(_LOWER_MIXER, lower, rcond=None)[0]
    roll_pitch_down = np.linalg.lstsq(_UPPER_MIXER, upper, rcond=None)[0]
    translation = np.asarray(
        [yaw_forward_right[1], yaw_forward_right[2], roll_pitch_down[2]],
        dtype=float,
    )
    return translation, float(yaw_forward_right[0]), roll_pitch_down[:2]


@dataclass
class FineSUBHardwareAdapter:
    """Convert PID SI outputs to firmware channels and telemetry back to SI."""

    positive_force_at_limit: object
    negative_force_at_limit: object
    # Active MPC experimental_auto.max_channel_abs.  The protocol validator
    # keeps the firmware hard ceiling separately; this adapter is the PID
    # runtime authority cap.
    translation_channel_limits: object = (0.20, 0.20, 0.20)
    translation_signs: object = (1.0, 1.0, 1.0)
    positive_yaw_moment_at_limit: float = 2.0
    negative_yaw_moment_at_limit: float = 2.0
    yaw_channel_limit: float = 0.20
    yaw_sign: float = 1.0
    # ``False`` selects the firmware's validated local yaw-hold loop.  The
    # host still computes the PID yaw diagnostic, but does not inject an
    # unvalidated direct-yaw channel into the shared lower mixer.
    yaw_direct: bool = True

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.negative_force_at_limit = vector3(
            self.negative_force_at_limit, "negative_force_at_limit"
        )
        self.translation_channel_limits = vector3(
            self.translation_channel_limits, "translation_channel_limits"
        )
        self.translation_signs = vector3(self.translation_signs, "translation_signs")
        if np.any(self.positive_force_at_limit <= 0.0) or np.any(
            self.negative_force_at_limit <= 0.0
        ):
            raise ValueError("force-at-limit calibration must be positive")
        if np.any(self.translation_channel_limits <= 0.0):
            raise ValueError("translation_channel_limits must be positive")
        if np.any(np.abs(self.translation_signs) != 1.0):
            raise ValueError("translation_signs must contain only +1 or -1")
        for name in (
            "positive_yaw_moment_at_limit",
            "negative_yaw_moment_at_limit",
            "yaw_channel_limit",
        ):
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if self.yaw_sign not in (-1.0, 1.0):
            raise ValueError("yaw_sign must be +1 or -1")
        self.yaw_direct = bool(self.yaw_direct)

    def convert(
        self,
        force_body: object,
        yaw_moment: float,
        *,
        armed: bool,
    ) -> FineSUBControlCommand:
        signed_force = self.translation_signs * vector3(force_body, "force_body")
        force_scale = np.where(
            signed_force >= 0.0,
            self.positive_force_at_limit,
            self.negative_force_at_limit,
        )
        translation = np.clip(
            signed_force / force_scale * self.translation_channel_limits,
            -self.translation_channel_limits,
            self.translation_channel_limits,
        )
        if self.yaw_direct:
            signed_moment = self.yaw_sign * _finite_scalar(yaw_moment, "yaw_moment")
            moment_scale = (
                self.positive_yaw_moment_at_limit
                if signed_moment >= 0.0
                else self.negative_yaw_moment_at_limit
            )
            yaw = float(
                np.clip(
                    signed_moment / moment_scale * self.yaw_channel_limit,
                    -self.yaw_channel_limit,
                    self.yaw_channel_limit,
                )
            )
        else:
            # Firmware local yaw hold captures current attitude on the first
            # armed command.  The channel must be exactly zero in this mode.
            _finite_scalar(yaw_moment, "yaw_moment")
            yaw = 0.0
        return FineSUBControlCommand(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            yaw,
            armed=bool(armed),
            yaw_direct=self.yaw_direct,
        )

    def _channels_to_wrench(
        self, translation_channels: object, yaw_channel: float
    ) -> tuple[np.ndarray, float]:
        channels = vector3(translation_channels, "translation_channels")
        force_scale = np.where(
            channels >= 0.0,
            self.positive_force_at_limit,
            self.negative_force_at_limit,
        )
        force = (
            channels / self.translation_channel_limits * force_scale
            / self.translation_signs
        )
        moment_scale = (
            self.positive_yaw_moment_at_limit
            if yaw_channel >= 0.0
            else self.negative_yaw_moment_at_limit
        )
        moment = yaw_channel / self.yaw_channel_limit * moment_scale / self.yaw_sign
        return force, float(moment)

    def achieved_wrench(self, telemetry: FineSUBTelemetry) -> tuple[np.ndarray, float]:
        if telemetry.failsafe or not telemetry.armed or not telemetry.pid_direct:
            return np.zeros(3), 0.0
        if telemetry.execution_feedback_valid:
            translation, yaw, _ = motor_throttles_to_channels(
                telemetry.applied_motor_throttle
            )
        else:
            translation = np.asarray(
                [telemetry.forward, telemetry.right, telemetry.down], dtype=float
            )
            yaw = telemetry.yaw
        return self._channels_to_wrench(translation, yaw)


def pack_command(
    command: FineSUBControlCommand,
    sequence: int,
    *,
    session_id: int,
    sender_time_ms: int,
) -> bytes:
    flags = COMMAND_FLAG_MPC_DIRECT
    if command.yaw_direct:
        flags |= COMMAND_FLAG_YAW_DIRECT
    if command.armed:
        flags |= COMMAND_FLAG_ARMED
    body = _COMMAND_BODY.pack(
        COMMAND_SYNC,
        PROTOCOL_VERSION,
        flags,
        int(sequence) & 0xFFFF,
        int(session_id) & 0xFFFFFFFF,
        int(sender_time_ms) & 0xFFFFFFFF,
        command.forward,
        command.right,
        command.down,
        command.yaw,
    )
    return body + _CRC.pack(crc16_modbus(body)) + bytes((COMMAND_TAIL,))


def unpack_command_frame(frame: bytes) -> FineSUBCommandEnvelope:
    if len(frame) != COMMAND_FRAME_SIZE:
        raise ValueError(f"command frame must be {COMMAND_FRAME_SIZE} bytes")
    if frame[-1] != COMMAND_TAIL:
        raise ValueError("command tail mismatch")
    body = frame[: _COMMAND_BODY.size]
    received_crc = _CRC.unpack_from(frame, len(body))[0]
    if crc16_modbus(body) != received_crc:
        raise ValueError("command CRC mismatch")
    sync, version, flags, sequence, session, sender_ms, fwd, right, down, yaw = (
        _COMMAND_BODY.unpack(body)
    )
    if sync != COMMAND_SYNC or version != PROTOCOL_VERSION:
        raise ValueError("unsupported command frame")
    if not flags & COMMAND_FLAG_MPC_DIRECT:
        raise ValueError("direct-control flag is absent")
    command = FineSUBControlCommand(
        fwd,
        right,
        down,
        yaw,
        armed=bool(flags & COMMAND_FLAG_ARMED),
        yaw_direct=bool(flags & COMMAND_FLAG_YAW_DIRECT),
    )
    return FineSUBCommandEnvelope(command, sequence, session, sender_ms, received_crc)


def pack_telemetry(telemetry: FineSUBTelemetry) -> bytes:
    """Reference encoder used by protocol/connection tests."""
    flags = 0
    if telemetry.armed:
        flags |= TELEMETRY_FLAG_ARMED
    if telemetry.pid_direct:
        flags |= TELEMETRY_FLAG_MPC_DIRECT
    if telemetry.failsafe:
        flags |= TELEMETRY_FLAG_FAILSAFE
    if telemetry.yaw_direct:
        flags |= TELEMETRY_FLAG_YAW_DIRECT
    if telemetry.execution_feedback_valid:
        flags |= TELEMETRY_FLAG_EXECUTION_FEEDBACK
    if telemetry.rpm_available:
        flags |= TELEMETRY_FLAG_RPM_AVAILABLE
    angular = list(telemetry.angular_velocity_xyz)
    angular[2] = telemetry.yaw_rate_rad_s
    floats = (
        *telemetry.quat_wxyz,
        *angular,
        *telemetry.linear_acceleration_xyz,
        telemetry.yaw_rad,
        telemetry.depth_m,
        telemetry.pressure_pa,
        telemetry.forward,
        telemetry.right,
        telemetry.down,
        telemetry.yaw,
        *telemetry.applied_motor_throttle,
        *telemetry.motor_rpm,
    )
    payload = _TELEMETRY_PAYLOAD.pack(
        flags,
        telemetry.state & 0xFF,
        telemetry.command_status & 0xFF,
        telemetry.rpm_valid_mask & 0xFF,
        telemetry.reject_flags & 0xFFFFFFFF,
        telemetry.last_command_session & 0xFFFFFFFF,
        telemetry.last_command_sequence & 0xFFFF,
        telemetry.last_command_crc & 0xFFFF,
        telemetry.last_command_sender_time_ms & 0xFFFFFFFF,
        telemetry.command_count & 0xFFFFFFFF,
        telemetry.rejected_command_count & 0xFFFFFFFF,
        *floats,
    )
    header = _TELEMETRY_HEADER.pack(
        TELEMETRY_SYNC,
        TELEMETRY_MESSAGE_STATE_EXECUTION,
        PROTOCOL_VERSION,
        len(payload),
        telemetry.sequence & 0xFFFF,
        telemetry.tick_ms & 0xFFFFFFFF,
    )
    body = header + payload
    return body + _CRC.pack(crc16_modbus(body))


def unpack_telemetry(frame: bytes) -> FineSUBTelemetry:
    if len(frame) != TELEMETRY_FRAME_SIZE:
        raise ValueError(f"telemetry frame must be {TELEMETRY_FRAME_SIZE} bytes")
    sync, message_type, version, payload_size, sequence, tick_ms = (
        _TELEMETRY_HEADER.unpack_from(frame)
    )
    if (
        sync != TELEMETRY_SYNC
        or message_type != TELEMETRY_MESSAGE_STATE_EXECUTION
        or version != PROTOCOL_VERSION
        or payload_size != TELEMETRY_PAYLOAD_SIZE
    ):
        raise ValueError("unsupported telemetry frame")
    if crc16_modbus(frame[:-2]) != _CRC.unpack_from(frame, len(frame) - 2)[0]:
        raise ValueError("telemetry CRC mismatch")
    values = _TELEMETRY_PAYLOAD.unpack_from(frame, _TELEMETRY_HEADER.size)
    flags = int(values[0])
    floats = values[11:]
    quat = tuple(float(value) for value in floats[0:4])
    angular = tuple(float(value) for value in floats[4:7])
    acceleration = tuple(float(value) for value in floats[7:10])
    received = floats[13:17]
    applied = tuple(float(value) for value in floats[17:25])
    rpm = tuple(float(value) for value in floats[25:33])
    return FineSUBTelemetry(
        sequence=sequence,
        tick_ms=tick_ms,
        state=int(values[1]),
        armed=bool(flags & TELEMETRY_FLAG_ARMED),
        pid_direct=bool(flags & TELEMETRY_FLAG_MPC_DIRECT),
        yaw_direct=bool(flags & TELEMETRY_FLAG_YAW_DIRECT),
        failsafe=bool(flags & TELEMETRY_FLAG_FAILSAFE),
        yaw_rad=float(floats[10]),
        yaw_rate_rad_s=float(angular[2]),
        depth_m=float(floats[11]),
        forward=float(received[0]),
        right=float(received[1]),
        down=float(received[2]),
        yaw=float(received[3]),
        command_status=int(values[2]),
        reject_flags=int(values[4]),
        last_command_session=int(values[5]),
        last_command_sequence=int(values[6]),
        last_command_crc=int(values[7]),
        last_command_sender_time_ms=int(values[8]),
        command_count=int(values[9]),
        rejected_command_count=int(values[10]),
        quat_wxyz=quat,
        angular_velocity_xyz=angular,
        linear_acceleration_xyz=acceleration,
        pressure_pa=float(floats[12]),
        applied_motor_throttle=applied,
        motor_rpm=rpm,
        execution_feedback_valid=bool(flags & TELEMETRY_FLAG_EXECUTION_FEEDBACK),
        rpm_available=bool(flags & TELEMETRY_FLAG_RPM_AVAILABLE),
        rpm_valid_mask=int(values[3]),
    )


class TelemetryStreamDecoder:
    """Recover fixed v5 telemetry frames from arbitrary UART chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.dropped_bytes = 0
        self.crc_errors = 0

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[FineSUBTelemetry]:
        self._buffer.extend(data)
        output: list[FineSUBTelemetry] = []
        while True:
            start = self._buffer.find(TELEMETRY_SYNC)
            if start < 0:
                keep = 1 if self._buffer.endswith(TELEMETRY_SYNC[:1]) else 0
                self.dropped_bytes += len(self._buffer) - keep
                if keep:
                    del self._buffer[:-1]
                else:
                    self._buffer.clear()
                break
            if start:
                self.dropped_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < TELEMETRY_FRAME_SIZE:
                break
            frame = bytes(self._buffer[:TELEMETRY_FRAME_SIZE])
            try:
                output.append(unpack_telemetry(frame))
            except ValueError:
                self.crc_errors += 1
                del self._buffer[0]
                continue
            del self._buffer[:TELEMETRY_FRAME_SIZE]
        return output
