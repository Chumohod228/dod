import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import leader_follower as app
from servo_current_test import Feedback, ProtocolError, Recorder


class FakeBus:
    bad_checksums = ignored_packets = 0

    def __init__(self, missing=None, fault=False):
        self.missing = missing
        self.fault = fault
        self.positions = {i: 2000 for i in range(1, 19)}
        self.torque = {}
        self.moves = []
        self.writes = []
        self.active = False

    def read(self, i, address, size):
        if i == self.missing:
            raise ProtocolError(f"ID {i}: timeout")
        values = {3: (777).to_bytes(2, "little"), 33: b"\0", 18: b"\x0c",
                  40: bytes([self.torque.get(i, 0)]), 42: self.positions[i].to_bytes(2, "little"), 9: b"\0\0\xff\x0f"}
        return values[address], 0

    def feedback(self, i):
        status = 1 if self.active and self.fault and i == 7 else 0
        position = 3000 if self.active and i >= 7 else self.positions[i]
        return Feedback(i, position, 0, 3.5 if status else 7.4, 25, 0, 0, 0, status)

    def move(self, positions, speed, acceleration):
        self.moves.append(dict(positions))
        self.positions.update(positions)
        self.active = True

    def send(self, payload):
        i, address, value = payload[2], payload[5], payload[6]
        self.writes.append((address, {i: bytes([value])}))
        self.torque[i] = value

    def sync_write(self, address, values):
        self.writes.append((address, values))


class TeleopTests(unittest.TestCase):
    def test_custom_ids_and_collision_rejected(self):
        args = app.parse_args(["--follower-ids", *map(str, range(13, 19))])
        self.assertEqual(args.pairs, dict(zip(range(7, 13), range(13, 19))))
        bus = FakeBus()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp) / "run", range(7, 19), {})
            try:
                args.duration = .03
                result = app.run(bus, recorder, args, {i: (1000, 3000) for i in args.pairs},
                                 {i: (1000, 3000) for i in args.pairs.values()}, [False])
                self.assertEqual(result, ("completed", 0))
                self.assertTrue(all(set(move) <= set(range(13, 19)) for move in bus.moves))
                self.assertTrue(all(set(values) <= set(range(13, 19)) for _, values in bus.writes))
            finally:
                recorder.close()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            app.parse_args(["--follower-ids", "7", "2", "3", "4", "5", "6"])

    def test_transport_failure_then_recovery_starts_from_present_pose(self):
        class DropOnce(FakeBus):
            dropped = False
            def feedback(self, i):
                if self.active and i == 3 and not self.dropped:
                    self.dropped = True
                    raise ProtocolError("ID 3: timeout reading register 56")
                return super().feedback(i)
        bus = DropOnce()
        result, _ = self.execute(bus)
        self.assertEqual(result, ("reconnect", 1))
        bus.torque.clear()  # Fake leader torque remains off.
        result, _ = self.execute(bus)
        self.assertEqual(result, ("completed", 0))

    def test_only_transport_errors_are_recoverable(self):
        self.assertTrue(app.recoverable_link_error(OSError("disconnected")))
        self.assertTrue(app.recoverable_link_error(ProtocolError("ID 3: timeout reading register 56")))
        for message in ["ID 3: status 0x05", "Follower position outside recorded range",
                        "ID 3: torque enable failed"]:
            self.assertFalse(app.recoverable_link_error(ProtocolError(message)))

    def test_voltage_and_leader_range_wait_for_recovery(self):
        for message in ["ID 3: status 0x01 reading register 3", "ID 3: status=0x21 U=4.0 V",
                        "Leader position 3118 outside recorded range 896..3098"]:
            self.assertTrue(app.recoverable_link_error(ProtocolError(message)))
        self.assertFalse(app.recoverable_link_error(ProtocolError("ID 3: status 0x25")))

    def test_recovery_checks_do_not_write_while_voltage_fault_persists(self):
        from unittest.mock import patch
        bus = FakeBus()
        args = app.parse_args([])
        args.recovering = True
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp) / "run", range(1, 13), {})
            try:
                original = app.verify_configuration
                count = [0]
                def check(*a, **kw):
                    count[0] += 1
                    if count[0] == 2:
                        raise ProtocolError("ID 3: status 0x01 reading register 3")
                    return original(*a, **kw)
                with patch.object(app, "verify_configuration", side_effect=check), patch.object(app, "reconnect_pause"), contextlib.redirect_stderr(io.StringIO()):
                    result = app.run(bus, recorder, args, {i: (1000,3000) for i in app.PAIRS},
                                     {i: (1000,3000) for i in app.PAIRS.values()}, [False])
                self.assertEqual(result, ("reconnect", 1))
                self.assertFalse(bus.moves or bus.writes)
            finally:
                recorder.close()

    def test_main_reopens_port_after_transport_failure(self):
        from unittest.mock import patch, MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            leader = Path(tmp) / "leader.json"
            follower = Path(tmp) / "follower.json"
            leader.write_text(json.dumps({str(i): [1000, 3000] for i in app.PAIRS}))
            follower.write_text(json.dumps({str(i): [1000, 3000] for i in app.PAIRS.values()}))
            with patch("serial.Serial") as port, patch.object(app, "run", side_effect=[("reconnect", 1), ("completed", 0)]) as run, patch.object(app, "reconnect_pause"), contextlib.redirect_stdout(io.StringIO()):
                result = app.main(["--leader-ranges", str(leader), "--follower-limits", str(follower),
                                   "--output", str(Path(tmp) / "out")])
            self.assertEqual(result, 0)
            self.assertEqual(port.call_count, 2)
            self.assertEqual(run.call_count, 2)

    def test_pairing_and_normalization(self):
        self.assertEqual(app.PAIRS, {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6})
        for raw, expected in [(1000, 1400), (2000, 1800), (3000, 2200)]:
            self.assertEqual(app.map_position(raw, (1000, 3000), (1400, 2200)), expected)
        self.assertEqual(app.map_position(1000, (1000, 3000), (1400, 2200), True), 2200)

    def test_zero_crossing_both_branches(self):
        self.assertEqual(app.map_position(0, (3900, 4300), (1000, 3000)), 1980)
        self.assertEqual(app.unwrap_in_range(4095, (-300, 1000)), -1)
        with self.assertRaises(ProtocolError):
            app.map_position(2000, (3900, 4300), (1000, 3000))

    def test_boundary_noise_clamped_and_larger_excursion_rejected(self):
        self.assertEqual(app.map_position(990, (1000, 3000), (1400, 2200)), 1400)
        with self.assertRaises(ProtocolError):
            app.map_position(900, (1000, 3000), (1400, 2200))

    def test_slew_rate(self):
        targets = app.approach_targets({1: 2000, 2: 2000}, {1: 1000, 2: 3000}, 600, .02)
        self.assertEqual(targets, {1: 1988, 2: 2012})

    def test_ranges_loader_rejects_incomplete_or_ambiguous_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "ranges.json"
            file.write_text(json.dumps({str(i): [1000, 3000] for i in app.PAIRS}))
            self.assertEqual(app.load_leader_ranges(file)[7], (1000, 3000))
            file.write_text(json.dumps({"servos": {str(i): dict(minimum=-100, maximum=3000,
                                                            ambiguous_steps=int(i == 7)) for i in app.PAIRS}}))
            with self.assertRaises(ValueError):
                app.load_leader_ranges(file)

    def execute(self, bus, extra=()):
        args = app.parse_args(["--leader-ranges", "unused", "--duration", "0.03", "--hz", "500", *extra])
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp) / "run", range(1, 13), {})
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = app.run(bus, recorder, args, {i: (1000, 3000) for i in app.PAIRS},
                                     {i: (1000, 3000) for i in app.PAIRS.values()}, [False])
                summary = json.loads((recorder.folder / "summary.json").read_text())
                return result, summary
            finally:
                recorder.close()

    def test_following_never_writes_leader_ids_and_starts_at_present_pose(self):
        bus = FakeBus()
        result, summary = self.execute(bus)
        self.assertEqual(result, ("completed", 0))
        self.assertEqual(bus.moves[0], {i: 2000 for i in range(1, 7)})
        self.assertTrue(any(p[1] > 2000 for p in bus.moves if 1 in p))
        for move in bus.moves:
            self.assertTrue(set(move) <= set(range(1, 7)))
        for address, values in bus.writes:
            self.assertEqual(address, 40)
            self.assertTrue(set(values) <= set(range(1, 7)))

    def test_check_only_sends_no_writes(self):
        bus = FakeBus()
        result, summary = self.execute(bus, ["--check-only"])
        self.assertEqual(result, ("checked", 0))
        self.assertEqual(summary["outcome"], "checked")
        self.assertFalse(bus.moves or bus.writes)

    def test_missing_leader_prevents_activation(self):
        bus = FakeBus(missing=12)
        result, summary = self.execute(bus)
        self.assertEqual(result[1], 1)
        self.assertFalse(bus.moves or bus.writes)

    def test_leader_voltage_fault_stops_following_and_is_logged(self):
        bus = FakeBus(fault=True)
        result, summary = self.execute(bus)
        self.assertEqual(result[1], 1)
        self.assertIn("U=3.5 V", summary["errors"][0]["message"])
        self.assertEqual(summary["servos"]["7"]["min_voltage_v"], 3.5)
        self.assertEqual(len(bus.moves), 2)  # Initial hold, then stop hold; no follow command.

    def test_unchanged_unreached_target_is_retried(self):
        f = Feedback(6, 2000, 0, 7.4, 25, 0, 0, 0, 0)
        args = ({6: 2200}, {6: 2200}, {6: f}, {6: 1.0})
        self.assertEqual(app.select_commands(*args, 1.4, 2, .5), {})
        self.assertEqual(app.select_commands(*args, 1.5, 2, .5), {6: 2200})
        f.position = 2200
        self.assertEqual(app.select_commands(*args, 1.5, 2, .5), {})

    def test_overload_retries_are_rate_limited_even_when_target_changes(self):
        f = Feedback(6, 2000, 0, 7.4, 25, 0, 0, 0, 0x20)
        args = ({6: 2300}, {6: 2200}, {6: f}, {6: 1.0})
        self.assertEqual(app.select_commands(*args, 1.4, 2, .5), {})
        self.assertEqual(app.select_commands(*args, 1.5, 2, .5), {6: 2300})

    def test_follower_overload_continues_and_allows_restart(self):
        class OverloadedBus(FakeBus):
            def read(self, i, address, size):
                data, status = super().read(i, address, size)
                return data, 0x20 if i == 6 else status

            def feedback(self, i):
                f = super().feedback(i)
                if i == 6:
                    f.status = 0x20
                    f.position = 2000  # Physical obstruction.
                return f

        bus = OverloadedBus()
        result, summary = self.execute(bus, ["--duration", "0.15", "--retry-interval", "0.1"])
        self.assertEqual(result, ("completed", 0))
        self.assertTrue(any(move.get(6, 0) > 2000 for move in bus.moves))
        self.assertGreater(summary["servos"]["6"]["fault_samples"], 1)
        self.assertEqual(summary["errors"], [])

    def test_combined_overload_and_voltage_fault_still_stops(self):
        class CriticalBus(FakeBus):
            def feedback(self, i):
                f = super().feedback(i)
                if self.active and i == 6:
                    f.status = 0x21
                    f.voltage_v = 3.5
                return f

        bus = CriticalBus()
        result, summary = self.execute(bus)
        self.assertEqual(result[1], 1)
        self.assertIn("0x21", summary["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
