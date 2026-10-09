from __future__ import annotations

import numpy as np
import pytest

from PID_controller.finesub_protocol import (
    COMMAND_FRAME_SIZE,
    COMMAND_STATUS_ACCEPTED,
    TELEMETRY_FRAME_SIZE,
    FineSUBControlCommand,
    FineSUBHardwareAdapter,
    FineSUBTelemetry,
    TelemetryStreamDecoder,
    crc16_modbus,
    motor_throttles_to_channels,
    pack_command,
    pack_telemetry,
    unpack_command_frame,
    unpack_telemetry,
)
from PID_controller.device_adapter import FineSUBThrusterAllocator


def make_telemetry(**overrides) -> FineSUBTelemetry:
    values = dict(
        sequence=4,
        tick_ms=123,
        state=1,
        armed=True,
        pid_direct=True,
        yaw_direct=True,
        failsafe=False,
        yaw_rad=0.25,
        yaw_rate_rad_s=-0.1,
        depth_m=0.8,
        forward=0.1,
        right=-0.05,
        down=0.2,
        yaw=0.03,
        command_status=COMMAND_STATUS_ACCEPTED,
        execution_feedback_valid=True,
        rpm_available=True,
    )
    values.update(overrides)
    return FineSUBTelemetry(**values)


def test_protocol_sizes_crc_and_command_round_trip() -> None:
    assert COMMAND_FRAME_SIZE == 37
    assert TELEMETRY_FRAME_SIZE == 174
    assert crc16_modbus(b"123456789") == 0x4B37
    command = FineSUBControlCommand(0.1, -0.2, 0.3, -0.1, armed=True)
    frame = pack_command(command, 7, session_id=9, sender_time_ms=11)
    envelope = unpack_command_frame(frame)
    assert envelope.sequence == 7
    assert envelope.session_id == 9
    assert envelope.sender_time_ms == 11
    assert envelope.command.armed
    np.testing.assert_allclose(
        (
            envelope.command.forward,
            envelope.command.right,
            envelope.command.down,
            envelope.command.yaw,
        ),
        (0.1, -0.2, 0.3, -0.1),
        atol=1e-7,
    )


def test_v5_telemetry_round_trip_and_fragment_decoder() -> None:
    telemetry = make_telemetry(
        applied_motor_throttle=tuple(np.linspace(-0.3, 0.4, 8)),
        motor_rpm=tuple(np.linspace(100.0, 800.0, 8)),
    )
    frame = pack_telemetry(telemetry)
    decoded = unpack_telemetry(frame)
    assert decoded.pid_direct
    assert decoded.yaw_direct
    assert decoded.execution_feedback_valid
    assert decoded.yaw_rad == np.float32(0.25)
    decoder = TelemetryStreamDecoder()
    assert decoder.feed(b"noise" + frame[:20]) == []
    output = decoder.feed(frame[20:])
    assert len(output) == 1
    assert output[0].sequence == 4


def test_body_frd_yaw_rate_matches_v5_sign_conversion() -> None:
    telemetry = make_telemetry(angular_velocity_xyz=(0.0, 0.0, -0.2))
    assert telemetry.body_frd_yaw_rate_rad_s == pytest.approx(0.2)


def test_current_firmware_mixer_is_inverted_from_actual_motor_feedback() -> None:
    lower_matrix = np.array(
        [[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1]],
        dtype=float,
    )
    upper_matrix = np.array(
        [[-1, 1, 1], [1, 1, -1], [1, -1, 1], [1, 1, 1]],
        dtype=float,
    )
    yaw_forward_right = np.array([0.04, 0.08, -0.03])
    roll_pitch_down = np.array([0.01, -0.02, 0.12])
    physical = np.zeros(8)
    physical[[0, 1, 5, 6]] = lower_matrix @ yaw_forward_right
    physical[[2, 3, 4, 7]] = upper_matrix @ roll_pitch_down
    translation, yaw, roll_pitch = motor_throttles_to_channels(physical)
    np.testing.assert_allclose(translation, (0.08, -0.03, 0.12), atol=1e-12)
    assert yaw == pytest.approx(0.04)
    np.testing.assert_allclose(roll_pitch, (0.01, -0.02), atol=1e-12)


def test_python_allocator_matches_firmware_attitude_mixer() -> None:
    allocator = FineSUBThrusterAllocator(
        positive_force_at_limit=(10.0, 10.0, 10.0),
        translation_channel_limits=(0.10, 0.10, 0.10),
        attitude_channel_limits=(0.20, 0.20, 0.20),
        deadband=0.0,
    )
    allocation = allocator.allocate(
        (0.0, 0.0, 0.0), attitude_control=(0.03, -0.04, 0.05)
    )
    translation, yaw, roll_pitch = motor_throttles_to_channels(
        allocation.throttles
    )
    np.testing.assert_allclose(translation, 0.0, atol=1e-12)
    np.testing.assert_allclose(roll_pitch, (0.03, -0.04), atol=1e-12)
    assert yaw == pytest.approx(0.05)


def test_hardware_adapter_uses_asymmetric_force_scale() -> None:
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=(10.0, 20.0, 30.0),
        negative_force_at_limit=(8.0, 16.0, 24.0),
    )
    command = adapter.convert((5.0, -8.0, 15.0), 1.0, armed=True)
    np.testing.assert_allclose(
        (command.forward, command.right, command.down, command.yaw),
        (0.10, -0.10, 0.10, 0.1),
    )


def test_local_yaw_hold_clears_direct_flag_and_channel() -> None:
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=(10.0, 20.0, 30.0),
        negative_force_at_limit=(8.0, 16.0, 24.0),
        yaw_direct=False,
    )
    command = adapter.convert((0.0, 0.0, 0.0), 1.0, armed=True)
    assert command.yaw == 0.0
    assert command.yaw_direct is False
    decoded = unpack_command_frame(
        pack_command(command, 2, session_id=9, sender_time_ms=11)
    ).command
    assert decoded.yaw == 0.0
    assert decoded.yaw_direct is False


def test_hardware_adapter_preserves_positive_frd_axis_directions() -> None:
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=(10.0, 20.0, 30.0),
        negative_force_at_limit=(8.0, 16.0, 24.0),
        translation_channel_limits=(0.20, 0.20, 0.20),
    )
    basis = np.eye(3)
    for axis in range(3):
        positive = adapter.convert(basis[axis], 0.0, armed=True)
        negative = adapter.convert(-basis[axis], 0.0, armed=True)
        assert np.asarray(
            (positive.forward, positive.right, positive.down), dtype=float
        )[axis] > 0.0
        assert np.asarray(
            (negative.forward, negative.right, negative.down), dtype=float
        )[axis] < 0.0
