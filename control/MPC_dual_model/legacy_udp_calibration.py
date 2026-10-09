"""Passive logger for the currently deployed FinsSim v1 MCU telemetry.

The NX bridge continuously forwards MCU UART bytes to the host UDP telemetry
port.  This recorder only binds that receive port: it never creates a command
frame and never sends a UDP datagram.  It exists so the deployed v1 firmware
can be characterized before the reviewed FineSUB v4 firmware is flashed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import struct
import time
from typing import Any

from .finesub_protocol import crc16_modbus
from .finesub_transport import load_runtime_config
from .ros_sensor_calibration import CSV_FIELDS as SYNCHRONIZED_CSV_FIELDS


DEFAULT_CONFIG_PATH = Path(__file__).with_name("finesub_v4pro1_mpc.json")
TELEMETRY_MAGIC = b"\xA5\x5A"
TELEMETRY_TYPE_IMU_DEPTH_V1 = 0x01
TELEMETRY_VERSION = 0x01
TELEMETRY_HEADER = struct.Struct("<2sBBHHI")
TELEMETRY_PAYLOAD = struct.Struct("<12fI8f")
CRC = struct.Struct("<H")
MAX_PAYLOAD_SIZE = 256

PASSIVE_FIELDS = (
    "telemetry_protocol",
    "udp_source",
    "telemetry_status_flags",
    "raw_frame_hex",
    "udp_datagram_count",
    "udp_byte_count",
    "parser_dropped_bytes",
    "parser_crc_errors",
    "parser_decode_errors",
)
CSV_FIELDS = (*SYNCHRONIZED_CSV_FIELDS, *PASSIVE_FIELDS)


def _json_vector(values: tuple[float, ...]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


@dataclass(frozen=True)
class LegacyTelemetry:
    sequence: int
    mcu_time_ms: int
    quat_wxyz: tuple[float, float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    linear_acceleration_xyz: tuple[float, float, float]
    depth_m: float
    pressure_pa: float
    status_flags: int
    motor_rpm: tuple[float, float, float, float, float, float, float, float]
    raw_frame: bytes


def parse_legacy_telemetry_frame(frame: bytes) -> LegacyTelemetry:
    if len(frame) < TELEMETRY_HEADER.size + CRC.size:
        raise ValueError("telemetry frame is too short")
    magic, message_type, version, payload_length, sequence, mcu_time_ms = (
        TELEMETRY_HEADER.unpack_from(frame)
    )
    if magic != TELEMETRY_MAGIC:
        raise ValueError("invalid telemetry magic")
    if message_type != TELEMETRY_TYPE_IMU_DEPTH_V1:
        raise ValueError(f"unsupported telemetry type {message_type}")
    if version != TELEMETRY_VERSION:
        raise ValueError(f"unsupported telemetry version {version}")
    if payload_length != TELEMETRY_PAYLOAD.size:
        raise ValueError(f"unexpected telemetry payload size {payload_length}")
    expected_size = TELEMETRY_HEADER.size + payload_length + CRC.size
    if len(frame) != expected_size:
        raise ValueError(f"unexpected telemetry frame size {len(frame)}")
    expected_crc = CRC.unpack_from(frame, len(frame) - CRC.size)[0]
    actual_crc = crc16_modbus(frame[2:-CRC.size])
    if actual_crc != expected_crc:
        raise ValueError(
            f"telemetry CRC mismatch: expected 0x{expected_crc:04x}, "
            f"got 0x{actual_crc:04x}"
        )

    unpacked = TELEMETRY_PAYLOAD.unpack(
        frame[TELEMETRY_HEADER.size : -CRC.size]
    )
    return LegacyTelemetry(
        sequence=int(sequence),
        mcu_time_ms=int(mcu_time_ms),
        quat_wxyz=tuple(float(value) for value in unpacked[0:4]),
        angular_velocity_xyz=tuple(float(value) for value in unpacked[4:7]),
        linear_acceleration_xyz=tuple(float(value) for value in unpacked[7:10]),
        depth_m=float(unpacked[10]),
        pressure_pa=float(unpacked[11]),
        status_flags=int(unpacked[12]),
        motor_rpm=tuple(float(value) for value in unpacked[13:21]),
        raw_frame=bytes(frame),
    )


class LegacyTelemetryStreamParser:
    """Recover complete v1 frames from arbitrarily split UDP/UART chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.dropped_bytes = 0
        self.crc_errors = 0
        self.decode_errors = 0

    def feed(self, data: bytes) -> list[LegacyTelemetry]:
        self._buffer.extend(data)
        packets: list[LegacyTelemetry] = []
        while True:
            magic_index = self._buffer.find(TELEMETRY_MAGIC)
            if magic_index < 0:
                if self._buffer.endswith(TELEMETRY_MAGIC[:1]):
                    self.dropped_bytes += max(0, len(self._buffer) - 1)
                    del self._buffer[:-1]
                else:
                    self.dropped_bytes += len(self._buffer)
                    self._buffer.clear()
                break
            if magic_index:
                self.dropped_bytes += magic_index
                del self._buffer[:magic_index]
            if len(self._buffer) < TELEMETRY_HEADER.size:
                break
            payload_length = struct.unpack_from("<H", self._buffer, 4)[0]
            if payload_length > MAX_PAYLOAD_SIZE:
                self.decode_errors += 1
                del self._buffer[0]
                continue
            frame_size = TELEMETRY_HEADER.size + payload_length + CRC.size
            if len(self._buffer) < frame_size:
                break
            frame = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            try:
                packets.append(parse_legacy_telemetry_frame(frame))
            except ValueError as error:
                if "CRC mismatch" in str(error):
                    self.crc_errors += 1
                else:
                    self.decode_errors += 1
        return packets


def _row(
    packet: LegacyTelemetry,
    *,
    phase: str,
    host_monotonic: float,
    source: tuple[str, int],
    datagram_count: int,
    byte_count: int,
    parser: LegacyTelemetryStreamParser,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        host_time_utc=datetime.now(timezone.utc).isoformat(),
        host_monotonic_s=f"{host_monotonic:.9f}",
        phase=phase,
        requested_armed=0,
        requested_channels_forward_right_down_yaw="[0.0,0.0,0.0,0.0]",
        sent=0,
        transport_open=1,
        session_confirmed=0,
        telemetry_fresh=1,
        telemetry_age_s="0.000000",
        telemetry_sequence=packet.sequence,
        mcu_tick_ms=packet.mcu_time_ms,
        imu_quat_wxyz=_json_vector(packet.quat_wxyz),
        imu_angular_velocity_xyz_rad_s=_json_vector(
            packet.angular_velocity_xyz
        ),
        imu_linear_acceleration_xyz_m_s2=_json_vector(
            packet.linear_acceleration_xyz
        ),
        depth_m=f"{packet.depth_m:.9g}",
        pressure_pa=f"{packet.pressure_pa:.9g}",
        motor_rpm_m1_m8=_json_vector(packet.motor_rpm),
        execution_feedback_valid=0,
        rpm_available=1,
        vision_fresh=0,
        telemetry_protocol="finsim_v1_passive",
        udp_source=f"{source[0]}:{source[1]}",
        telemetry_status_flags=packet.status_flags,
        raw_frame_hex=packet.raw_frame.hex(),
        udp_datagram_count=datagram_count,
        udp_byte_count=byte_count,
        parser_dropped_bytes=parser.dropped_bytes,
        parser_crc_errors=parser.crc_errors,
        parser_decode_errors=parser.decode_errors,
    )
    return row


def run_passive_recording(
    config_path: str | Path,
    csv_path: str | Path,
    *,
    duration_s: float,
    phase: str,
    bind_host: str | None = None,
    bind_port: int | None = None,
    expected_source_host: str | None = None,
) -> dict[str, int | str]:
    """Receive v1 telemetry for ``duration_s`` without transmitting anything."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    runtime = load_runtime_config(config_path)
    transport = runtime.get("transport", {})
    local_host = str(bind_host or transport.get("bind_host", "0.0.0.0"))
    local_port = int(bind_port or transport.get("bind_port", 54321))
    expected_host = str(
        expected_source_host or transport.get("remote_host", "192.168.0.2")
    )

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parser = LegacyTelemetryStreamParser()
    datagram_count = 0
    byte_count = 0
    foreign_datagrams = 0
    packet_count = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_host, local_port))
    sock.settimeout(0.2)
    print(
        f"[CAL] passive FinsSim v1 telemetry {local_host}:{local_port} for "
        f"{duration_s:.1f}s; expected_source={expected_host}; CSV={destination}"
    )
    try:
        with destination.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                try:
                    data, source = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if source[0] != expected_host:
                    foreign_datagrams += 1
                    continue
                datagram_count += 1
                byte_count += len(data)
                received_monotonic = time.monotonic()
                for packet in parser.feed(data):
                    packet_count += 1
                    writer.writerow(
                        _row(
                            packet,
                            phase=phase,
                            host_monotonic=received_monotonic,
                            source=source,
                            datagram_count=datagram_count,
                            byte_count=byte_count,
                            parser=parser,
                        )
                    )
            handle.flush()
    finally:
        sock.close()

    result: dict[str, int | str] = {
        "csv": str(destination),
        "packets": packet_count,
        "datagrams": datagram_count,
        "bytes": byte_count,
        "foreign_datagrams": foreign_datagrams,
        "dropped_bytes": parser.dropped_bytes,
        "crc_errors": parser.crc_errors,
        "decode_errors": parser.decode_errors,
    }
    print(f"[CAL] passive result: {json.dumps(result, sort_keys=True)}")
    if packet_count == 0:
        raise RuntimeError("no valid passive FinsSim v1 telemetry received")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive-only FinsSim v1 IMU/depth/RPM calibration logger"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--phase", default="static_passive_legacy_v1")
    parser.add_argument("--bind-host")
    parser.add_argument("--bind-port", type=int)
    parser.add_argument("--expected-source-host")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_passive_recording(
        args.config,
        args.csv,
        duration_s=args.duration,
        phase=args.phase,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        expected_source_host=args.expected_source_host,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
