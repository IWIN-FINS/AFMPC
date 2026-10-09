import struct
import unittest

from MPC_dual_model.finesub_protocol import crc16_modbus
from MPC_dual_model.legacy_udp_calibration import (
    TELEMETRY_HEADER,
    TELEMETRY_MAGIC,
    TELEMETRY_PAYLOAD,
    LegacyTelemetryStreamParser,
    parse_legacy_telemetry_frame,
)


def _frame() -> bytes:
    payload = TELEMETRY_PAYLOAD.pack(
        1.0,
        0.0,
        0.0,
        0.0,
        0.01,
        -0.02,
        0.03,
        0.1,
        0.2,
        9.7,
        0.4,
        105000.0,
        7,
        *tuple(float(index) for index in range(8)),
    )
    header = TELEMETRY_HEADER.pack(
        TELEMETRY_MAGIC,
        1,
        1,
        len(payload),
        123,
        4567,
    )
    body = header[2:] + payload
    return header + payload + struct.pack("<H", crc16_modbus(body))


class LegacyUdpCalibrationTest(unittest.TestCase):
    def test_parse_v1_telemetry(self) -> None:
        packet = parse_legacy_telemetry_frame(_frame())
        self.assertEqual(packet.sequence, 123)
        self.assertEqual(packet.mcu_time_ms, 4567)
        self.assertEqual(packet.status_flags, 7)
        self.assertAlmostEqual(packet.angular_velocity_xyz[1], -0.02, places=6)
        self.assertEqual(packet.motor_rpm, tuple(float(index) for index in range(8)))

    def test_stream_parser_handles_split_uart_chunks(self) -> None:
        frame = _frame()
        parser = LegacyTelemetryStreamParser()
        self.assertEqual(parser.feed(b"noise" + frame[:1]), [])
        packets = parser.feed(frame[1:])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].raw_frame, frame)
        self.assertEqual(parser.dropped_bytes, 5)

    def test_stream_parser_counts_bad_crc_and_recovers(self) -> None:
        bad = bytearray(_frame())
        bad[-1] ^= 0x01
        parser = LegacyTelemetryStreamParser()
        packets = parser.feed(bytes(bad) + _frame())
        self.assertEqual(len(packets), 1)
        self.assertEqual(parser.crc_errors, 1)


if __name__ == "__main__":
    unittest.main()
