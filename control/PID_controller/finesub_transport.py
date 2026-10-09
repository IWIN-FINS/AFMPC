"""Safe serial transport and confirmed-session link for PID hardware control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import secrets
import socket
import time
from typing import Callable

try:
    from .finesub_protocol import (
        COMMAND_STATUS_ACCEPTED,
        COMMAND_STATUS_REJECTED,
        FineSUBControlCommand,
        FineSUBTelemetry,
        TelemetryStreamDecoder,
        is_newer_telemetry,
        pack_command,
        unpack_command_frame,
    )
except ImportError:
    from finesub_protocol import (
        COMMAND_STATUS_ACCEPTED,
        COMMAND_STATUS_REJECTED,
        FineSUBControlCommand,
        FineSUBTelemetry,
        TelemetryStreamDecoder,
        is_newer_telemetry,
        pack_command,
        unpack_command_frame,
    )


class TransportError(RuntimeError):
    pass


class FullDuplexTransport(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write(self, data: bytes) -> None: ...

    @abstractmethod
    def read(self, max_bytes: int) -> bytes: ...


class DryRunTransport(FullDuplexTransport):
    """Memory transport for protocol tests; it can never drive hardware."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self._reads: deque[bytes] = deque()

    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, max_bytes: int) -> bytes:
        if not self._reads:
            return b""
        data = self._reads.popleft()
        if len(data) <= max_bytes:
            return data
        self._reads.appendleft(data[max_bytes:])
        return data[:max_bytes]

    def inject_read(self, data: bytes) -> None:
        self._reads.append(bytes(data))


class SerialTransport(FullDuplexTransport):
    """Non-blocking 115200 8N1 UART transport used by V4pro1_MPC USART3."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        write_timeout_s: float = 0.1,
    ) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.write_timeout_s = float(write_timeout_s)
        self._serial = None

    def open(self) -> None:
        try:
            import serial
        except ImportError as error:
            raise TransportError("pyserial is required for real hardware") from error
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.0,
                write_timeout=self.write_timeout_s,
            )
            self._serial.reset_input_buffer()
        except Exception as error:
            self._serial = None
            raise TransportError(str(error)) from error

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def write(self, data: bytes) -> None:
        if self._serial is None:
            raise TransportError("serial transport is closed")
        try:
            written = self._serial.write(data)
        except Exception as error:
            raise TransportError(str(error)) from error
        if written != len(data):
            raise TransportError(f"short serial write: {written}/{len(data)}")

    def read(self, max_bytes: int) -> bytes:
        if self._serial is None:
            raise TransportError("serial transport is closed")
        try:
            waiting = int(self._serial.in_waiting)
            return b"" if waiting <= 0 else bytes(self._serial.read(min(waiting, max_bytes)))
        except Exception as error:
            raise TransportError(str(error)) from error


class TcpTransport(FullDuplexTransport):
    """Non-blocking TCP transport compatible with the MPC runtime bridge."""

    def __init__(self, host: str, port: int, timeout_sec: float = 0.2) -> None:
        self._address = (str(host), int(port))
        self._timeout_sec = float(timeout_sec)
        if self._timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        self._socket: socket.socket | None = None

    def open(self) -> None:
        try:
            sock = socket.create_connection(self._address, timeout=self._timeout_sec)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
            self._socket = sock
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def write(self, data: bytes) -> None:
        if self._socket is None:
            raise TransportError("TCP transport is closed")
        try:
            self._socket.settimeout(self._timeout_sec)
            self._socket.sendall(data)
            self._socket.setblocking(False)
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error

    def read(self, max_bytes: int) -> bytes:
        if self._socket is None:
            raise TransportError("TCP transport is closed")
        try:
            data = self._socket.recv(max_bytes)
        except BlockingIOError:
            return b""
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error
        if not data:
            self.close()
            raise TransportError("TCP peer closed the connection")
        return data


class UdpTransport(FullDuplexTransport):
    """MPC-compatible UDP transport with source filtering for telemetry."""

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        remote_host: str,
        remote_port: int,
        command_datagram_size: int = 0,
        telemetry_source_port: int = 0,
    ) -> None:
        self._bind_address = (str(bind_host), int(bind_port))
        self._remote_address = (str(remote_host), int(remote_port))
        self._command_datagram_size = int(command_datagram_size)
        self._telemetry_source_port = int(telemetry_source_port)
        if self._command_datagram_size < 0 or self._telemetry_source_port < 0:
            raise ValueError("UDP datagram and source-port limits must be nonnegative")
        self._socket: socket.socket | None = None

    def open(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self._bind_address)
            sock.setblocking(False)
            self._socket = sock
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def write(self, data: bytes) -> None:
        if self._socket is None:
            raise TransportError("UDP transport is closed")
        wire_data = bytes(data)
        if self._command_datagram_size:
            if len(wire_data) + 1 > self._command_datagram_size:
                raise TransportError("command frame does not fit UDP datagram size")
            wire_data = (
                wire_data
                + bytes(self._command_datagram_size - len(wire_data) - 1)
                + b"\xBB"
            )
        try:
            sent = self._socket.sendto(wire_data, self._remote_address)
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error
        if sent != len(wire_data):
            raise TransportError(f"short UDP write: {sent}/{len(wire_data)}")

    def read(self, max_bytes: int) -> bytes:
        if self._socket is None:
            raise TransportError("UDP transport is closed")
        try:
            data, sender = self._socket.recvfrom(max_bytes)
        except BlockingIOError:
            return b""
        except OSError as error:
            self.close()
            raise TransportError(str(error)) from error
        # Ignore unsolicited LAN datagrams; only the configured bridge may
        # establish command confirmation or provide telemetry.
        if sender[0] != self._remote_address[0]:
            return b""
        if self._telemetry_source_port and sender[1] != self._telemetry_source_port:
            return b""
        return data


def make_transport(config: dict) -> FullDuplexTransport:
    """Build the same transport variants used by the MPC runtime config."""
    transport_type = str(config.get("type", "serial")).strip().lower()
    timeout_sec = float(config.get("timeout_sec", 0.2))
    if transport_type == "dry_run":
        return DryRunTransport()
    if transport_type == "serial":
        return SerialTransport(
            str(config.get("serial_port", "/dev/ttyUSB0")),
            int(config.get("baudrate", 115200)),
            timeout_sec,
        )
    if transport_type == "tcp":
        return TcpTransport(
            str(config.get("host", "127.0.0.1")),
            int(config.get("port", 5000)),
            timeout_sec,
        )
    if transport_type == "udp":
        return UdpTransport(
            str(config.get("bind_host", "0.0.0.0")),
            int(config.get("bind_port", 54321)),
            str(config.get("remote_host", "192.0.2.2")),
            int(config.get("remote_port", 58766)),
            int(config.get("command_datagram_size", 0)),
            int(config.get("telemetry_source_port", 0)),
        )
    raise ValueError("transport type must be serial, tcp, udp, or dry_run")


def load_runtime_config(path: str | Path) -> dict:
    """Load a JSON runtime config without opening any transport."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("FineSUB runtime config must be a JSON object")
    return config


@dataclass(frozen=True)
class _SentCommand:
    sequence: int
    crc: int
    sender_time_ms: int
    sent_monotonic: float
    armed: bool


class FineSUBConnection:
    """Session-aware connection with lower-controller positive confirmation."""

    def __init__(
        self,
        transport: FullDuplexTransport,
        *,
        telemetry_max_age_s: float = 0.20,
        confirmation_max_age_s: float = 0.30,
        reconnect_interval_s: float = 1.0,
        logger: Callable[[str], None] | None = print,
        session_id: int | None = None,
        yaw_direct: bool = True,
    ) -> None:
        self.transport = transport
        self.telemetry_max_age_s = float(telemetry_max_age_s)
        self.confirmation_max_age_s = float(confirmation_max_age_s)
        self.reconnect_interval_s = float(reconnect_interval_s)
        self.logger = logger
        self.yaw_direct = bool(yaw_direct)
        self.session_id = (
            int(session_id) & 0xFFFFFFFF
            if session_id is not None
            else secrets.randbits(32) or 1
        )
        self.sequence = 0
        self.decoder = TelemetryStreamDecoder()
        self.latest_telemetry: FineSUBTelemetry | None = None
        self.session_confirmed = False
        self.last_confirmed_monotonic: float | None = None
        self.last_confirmed_armed: bool | None = None
        self.last_confirmed_sequence: int | None = None
        self.last_round_trip_ms: int | None = None
        self.last_effective_command: FineSUBControlCommand | None = None
        self._opened = False
        self._last_open_attempt = float("-inf")
        self._sent: dict[int, _SentCommand] = {}
        self.send_errors = 0
        self.read_errors = 0

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def connect(self) -> bool:
        if self._opened:
            return True
        now = time.monotonic()
        if now - self._last_open_attempt < self.reconnect_interval_s:
            return False
        self._last_open_attempt = now
        try:
            self.transport.open()
        except TransportError as error:
            self._log(f"[PID-HW] connect failed: {error}")
            return False
        self._opened = True
        self.decoder.reset()
        self.latest_telemetry = None
        self.session_confirmed = False
        self.last_confirmed_monotonic = None
        self.last_confirmed_armed = None
        self.last_effective_command = None
        self._sent.clear()
        self._log("[PID-HW] FineSUB serial connected")
        return True

    def close(self) -> None:
        self.transport.close()
        self._opened = False
        self.latest_telemetry = None
        self.session_confirmed = False
        self.last_confirmed_monotonic = None
        self.last_confirmed_armed = None
        self.last_effective_command = None
        self._sent.clear()

    def _fail(self, message: str) -> None:
        self._log(message)
        self.close()

    def poll_telemetry(self) -> None:
        if not self._opened and not self.connect():
            return
        for _ in range(16):
            try:
                data = self.transport.read(4096)
            except TransportError as error:
                self.read_errors += 1
                self._fail(f"[PID-HW] receive failed: {error}")
                return
            if not data:
                break
            for telemetry in self.decoder.feed(data):
                if self.latest_telemetry is None or is_newer_telemetry(
                    telemetry, self.latest_telemetry
                ):
                    self.latest_telemetry = telemetry
                    self._process_echo(telemetry)

    def _process_echo(self, telemetry: FineSUBTelemetry) -> None:
        if telemetry.last_command_session != self.session_id:
            return
        if telemetry.command_status == COMMAND_STATUS_REJECTED:
            self.session_confirmed = False
            self.last_confirmed_monotonic = None
            return
        if telemetry.command_status != COMMAND_STATUS_ACCEPTED:
            return
        sent = self._sent.get(telemetry.last_command_sequence)
        if sent is None or sent.crc != telemetry.last_command_crc:
            return
        now = time.monotonic()
        if now - sent.sent_monotonic > self.confirmation_max_age_s:
            del self._sent[sent.sequence]
            return
        self.session_confirmed = True
        self.last_confirmed_monotonic = now
        self.last_confirmed_armed = sent.armed
        self.last_confirmed_sequence = sent.sequence
        now_ms = int(now * 1000.0) & 0xFFFFFFFF
        self.last_round_trip_ms = (
            now_ms - telemetry.last_command_sender_time_ms
        ) & 0xFFFFFFFF
        for sequence in list(self._sent):
            if sequence == sent.sequence or now - self._sent[sequence].sent_monotonic > 2.0:
                del self._sent[sequence]

    def fresh_telemetry(self) -> FineSUBTelemetry | None:
        self.poll_telemetry()
        if self.latest_telemetry is None:
            return None
        if time.monotonic() - self.latest_telemetry.received_monotonic > self.telemetry_max_age_s:
            return None
        return self.latest_telemetry

    def confirmation_fresh(self) -> bool:
        return bool(
            self.session_confirmed
            and self.last_confirmed_monotonic is not None
            and time.monotonic() - self.last_confirmed_monotonic
            <= self.confirmation_max_age_s
        )

    def armed_confirmation_fresh(self) -> bool:
        return self.last_confirmed_armed is True and self.confirmation_fresh()

    def send(self, command: FineSUBControlCommand) -> bool:
        if not self._opened and not self.connect():
            return False
        effective = command
        if command.armed and not self.confirmation_fresh():
            # Firmware requires a confirmed disarmed frame for every new session.
            effective = FineSUBControlCommand(
                0.0,
                0.0,
                0.0,
                0.0,
                armed=False,
                yaw_direct=command.yaw_direct,
            )
        sender_ms = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        sequence = self.sequence
        frame = pack_command(
            effective,
            sequence,
            session_id=self.session_id,
            sender_time_ms=sender_ms,
        )
        envelope = unpack_command_frame(frame)
        try:
            self.transport.write(frame)
        except TransportError as error:
            self.send_errors += 1
            self._fail(f"[PID-HW] send failed: {error}")
            return False
        self._sent[sequence] = _SentCommand(
            sequence,
            envelope.crc,
            sender_ms,
            time.monotonic(),
            effective.armed,
        )
        self.last_effective_command = effective
        self.sequence = (sequence + 1) & 0xFFFF
        if len(self._sent) > 256:
            oldest = min(self._sent, key=lambda item: self._sent[item].sent_monotonic)
            del self._sent[oldest]
        return True

    def send_disarm(self) -> bool:
        return self.send(
            FineSUBControlCommand(
                0.0,
                0.0,
                0.0,
                0.0,
                armed=False,
                yaw_direct=self.yaw_direct,
            )
        )
