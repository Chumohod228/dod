import contextlib
import csv
import io
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest

import servo_current_test as app


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

    def test_random_targets_remain_within_limits_and_step(self):
        rng = random.Random(42)
        positions = {i: limits[0] for i, limits in app.DEFAULT_LIMITS.items()}
        for _ in range(10000):
            new = app.next_targets(rng, positions, app.DEFAULT_LIMITS, 256)
            for i, value in new.items():
                lo, hi = app.DEFAULT_LIMITS[i]
                self.assertTrue(lo <= value <= hi)
                self.assertTrue(0 < abs(value - positions[i]) <= 256)
            positions = new

    def execute(self, serial, stop=False):
        args = app.parse_args(["--port", "FAKE", "--duration", "0.03", "--hz", "500",
                               "--move-interval", "0.005", "--seed", "42"])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            recorder = app.Recorder(folder, args.ids, vars(args))
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = app.run(app.ServoBus(serial), recorder, args, app.DEFAULT_LIMITS, [stop])
                summary = json.loads((folder / "summary.json").read_text())
                with (folder / "samples.csv").open() as stream:
                    samples = list(csv.DictReader(stream))
                with (folder / "peaks.csv").open() as stream:
                    peaks = list(csv.DictReader(stream))
            finally:
                recorder.close()
        return result, summary, samples, peaks

    def test_wide_targets_have_minimum_separation(self):
        import math
        rng = random.Random(81)
        limits = {1: (1070, 3025), 5: (1995, 2099), 6: (1791, 2304)}
        positions = {i: (lo + hi) // 2 for i, (lo, hi) in limits.items()}
        for _ in range(2000):
            new = app.next_targets(rng, positions, limits, 4095, .35)
            for i, value in new.items():
                lo, hi = limits[i]
                self.assertTrue(lo <= value <= hi)
                self.assertGreaterEqual(abs(value - positions[i]), math.ceil((hi - lo) * .35))
            positions = new

    def test_step_limit_takes_priority_over_separation(self):
        new = app.next_targets(random.Random(1), {1: 2000}, {1: (1000, 3000)}, 100, .35)
        self.assertEqual(abs(new[1] - 2000), 100)

    def test_full_run_tracks_all_six_peaks_and_holds(self):
        serial = FakeSerial()
        result, summary, samples, peaks = self.execute(serial)
        self.assertEqual(result, ("completed", 0))
        self.assertEqual(len(peaks), 6)
        self.assertEqual(summary["servos"]["6"]["max_abs_current_ma"], 689)
        self.assertTrue(any(row["phase"] == "active" for row in samples))
        self.assertEqual(serial.writes[-1][4:7], bytes((0x83, 41, 7)))
        self.assertTrue(serial.enabled)  # Do not release an arm under gravity.

    def test_invalid_initial_pose_sends_no_motion(self):
        serial = FakeSerial(bad_initial=True)
        result, summary, samples, peaks = self.execute(serial)
        self.assertEqual(result[1], 1)
        self.assertTrue(all(frame[4] == 2 for frame in serial.writes))

    def test_fault_sample_saved_before_stop(self):
        result, summary, samples, peaks = self.execute(FakeSerial(fault=True))
        self.assertEqual(result[1], 1)
        self.assertTrue(any(int(row["status"]) == 0x20 for row in samples))
        self.assertGreater(summary["servos"]["1"]["fault_samples"], 0)

    def test_stop_before_motion_has_no_write_and_null_peaks(self):
        serial = FakeSerial()
        result, summary, samples, peaks = self.execute(serial, stop=True)
        self.assertEqual(result, ("interrupted", 0))
        self.assertEqual(summary["outcome"], "interrupted")
        self.assertEqual(serial.writes, [])
        self.assertIsNone(summary["servos"]["1"]["max_abs_current_ma"])
        self.assertIsNone(summary["servos"]["1"]["min_voltage_v"])
        self.assertEqual(peaks[0]["min_voltage_v"], "")

    def test_voltage_dip_is_preserved_after_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = app.Recorder(Path(tmp) / "run", [1, 2], {})
            try:
                for voltage, status in [(5.2, 0), (4.6, 0), (3.5, 1), (5.1, 0)]:
                    f = app.decode_feedback(1, feedback_bytes(), status)
                    f.voltage_v = voltage
                    recorder.sample(f, "active")
                recorder.save("error")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    recorder.display()
                self.assertIn("ID 1: 5.1 V (min 3.5, max 5.2)", output.getvalue())
                self.assertIn("ID 2: нет данных", output.getvalue())
                saved = json.loads((recorder.folder / "summary.json").read_text())["servos"]["1"]
                self.assertEqual(saved["min_voltage_v"], 3.5)
                self.assertEqual(saved["max_voltage_v"], 5.2)
                self.assertEqual(saved["voltage_v"], 5.1)
                self.assertEqual(saved["min_voltage_sample"]["status"], 1)
                with (recorder.folder / "peaks.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(rows[0]["min_voltage_v"], "3.5")
                self.assertEqual(rows[0]["min_voltage_utc"], saved["min_voltage_sample"]["utc"])
                self.assertEqual(rows[1]["voltage_v"], "")
            finally:
                recorder.close()

    def test_voltage_fault_message_includes_measured_values(self):
        f = app.decode_feedback(6, feedback_bytes(), 1)
        f.voltage_v = 3.5
        message = app.feedback_error(f)
        self.assertIn("ошибка напряжения питания", message)
        self.assertIn("U=3.5 V", message)
        self.assertIn("I=0.650 A", message)


if __name__ == "__main__":
    unittest.main()
