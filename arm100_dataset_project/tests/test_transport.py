"""Wire-level checks against the active collection transport."""
import struct
import unittest
from arm100_collect import transport as app

def feedback_bytes(position=1500, raw_current=100, moving=1):
    data = bytearray(15)
    struct.pack_into("<HHH", data, 0, position, 200, 0)
    data[6:8] = bytes((120, 35))
    data[10] = moving
    struct.pack_into("<H", data, 13, raw_current)
    return bytes(data)


class FakeSerial:
    """Wire-level emulator: echoes, fragmented reads, noise and real checksums."""
    def __init__(self, fault=False, bad_initial=False):
        self.buffer = bytearray()
        self.writes = []
        self.positions = {i: 1500 for i in range(1, 7)}
        if bad_initial:
            self.positions[6] = 3000
        self.fault = fault
        self.enabled = False
        self.read_counts = {i: 0 for i in self.positions}
        self.inject_noise = False

    @property
    def in_waiting(self):
        return len(self.buffer)

    def reset_input_buffer(self):
        self.buffer.clear()

    def flush(self):
        pass

    def write(self, request):
        self.writes.append(request)
        assert request[:2] == b"\xff\xff"
        assert len(request) == request[3] + 4
        assert sum(request[2:]) & 255 == 255
        servo_id, instruction = request[2], request[4]
        if instruction == 2:
            address, size = request[5:7]
            if address == 33:
                data, status = b"\x00", 0
            else:
                self.read_counts[servo_id] += 1
                data = feedback_bytes(self.positions[servo_id], 0x8000 | (100 + servo_id))
                status = 0x20 if self.fault and self.enabled else 0
            assert len(data) == size
            response = app.packet(servo_id, status, data)
            self.buffer.extend(request)  # Actual TX echo must never be treated as status.
            if self.inject_noise:
                damaged = bytearray(response)
                damaged[-1] ^= 1
                self.buffer.extend(b"noise" + damaged + app.packet(99, 0, data))
            self.buffer.extend(response)
        elif instruction == 0x83:
            address, size = request[5:7]
            for offset in range(7, len(request) - 1, size + 1):
                i = request[offset]
                value = request[offset + 1:offset + 1 + size]
                if address == 41:
                    self.positions[i] = int.from_bytes(value[1:3], "little")
                elif address == 40:
                    self.enabled = bool(value[0])
        else:
            raise AssertionError(f"Unexpected instruction {instruction}")
        return len(request)

    def read(self, count):
        count = min(count, 3)  # Exercise incomplete headers and payloads.
        value = bytes(self.buffer[:count])
        del self.buffer[:count]
        return value


class ProtocolTests(unittest.TestCase):
    def test_known_read_packet(self):
        self.assertEqual(app.packet(1, 2, b"\x45\x02"), bytes.fromhex("FF FF 01 04 02 45 02 B1"))

    def test_signed_current_and_offsets(self):
        f = app.decode_feedback(3, feedback_bytes(raw_current=0x800A), 0)
        self.assertEqual(f.current_ma, -65)
        self.assertEqual(f.position, 1500)
        self.assertEqual(f.voltage_v, 12)
        self.assertEqual(f.temperature_c, 35)

    def test_echo_noise_checksum_wrong_id_and_fragments(self):
        serial = FakeSerial()
        serial.inject_noise = True
        bus = app.ServoBus(serial)
        self.assertEqual(bus.read(1, 33, 1), (b"\x00", 0))
        self.assertEqual(bus.feedback(1).current_ma, -656.5)
        self.assertEqual(bus.bad_checksums, 2)
        self.assertEqual(bus.ignored_packets, 2)

    def test_timeout(self):
        serial = FakeSerial()
        serial.read = lambda count: b""
        with self.assertRaises(app.ProtocolError):
            app.ServoBus(serial, timeout=0.001).feedback(1)


    def test_sync_write_preserves_id_positions_and_checksum(self):
        serial = FakeSerial()
        app.ServoBus(serial).move({1: 2048, 6: 1600}, 600, 20)
        self.assertEqual(serial.positions[1], 2048)
        self.assertEqual(serial.positions[6], 1600)
        self.assertEqual(serial.writes[0][4:7], bytes((0x83, 41, 7)))

    def test_overload_status_reaches_control_layer(self):
        serial = FakeSerial(fault=True)
        serial.enabled = True
        self.assertEqual(app.ServoBus(serial).feedback(1).status, 0x20)
