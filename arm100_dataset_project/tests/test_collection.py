import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from arm100_collect.config import ARM_NAMES, PROJECT, load_config, normalized, raw_segments, unwrap
from arm100_collect.dataset import open_dataset, validate_dataset
from arm100_collect.hardware import Rig, snapshot
from arm100_collect.transport import Feedback, ProtocolError


class FakeBus:
    def __init__(self, ids, position=2000):
        self.values = {i: Feedback(i, position, 0, 7.5, 28, 0, 0, 0, 0) for i in ids}
        self.moves = []
        self.writes = []
        self.fail_read = False

    def read(self, i, address, size):
        values = {3: b"\x09\x03", 9: b"\x00\x00\xff\x0f", 18: b"\x0c", 31: b"\x00\x00", 33: b"\x00", 40: b"\x00"}
        return values[address], 0

    def feedback(self, i):
        if self.fail_read:
            raise ProtocolError("mock feedback lost")
        return self.values[i]

    def move(self, values, speed, acceleration):
        self.moves.append(dict(values))

    def sync_write(self, address, values):
        self.writes.append((address, values))


def fake_rig():
    cfg = load_config(PROJECT / "configs/rig.json")
    rig = Rig(cfg)
    for name, arm in cfg["arms"].items():
        rig.buses[name] = FakeBus(arm["ids"])
        rig.ranges[name] = {i: (1000, 3000) for i in arm["ids"]}
    rig.feedback = rig.read_all()
    rig.start()
    for bus in rig.buses.values():
        bus.moves.clear()
        bus.writes.clear()
    return rig


class CoordinateTests(unittest.TestCase):
    def test_zero_crossing_both_directions(self):
        self.assertEqual(unwrap(4000, (-200, 500)), -96)
        self.assertEqual(unwrap(100, (3500, 4700)), 4196)
        self.assertEqual(raw_segments((-200, 500)), [(3896, 4095), (0, 500)])
        self.assertEqual(raw_segments((4100, 4500)), [(4, 404)])

    def test_outside_arc_is_rejected(self):
        with self.assertRaises(ValueError):
            unwrap(2000, (-200, 500))

    def test_normalization_gripper_and_joint(self):
        self.assertEqual(normalized(1000, (1000, 3000)), -100)
        self.assertEqual(normalized(3000, (1000, 3000)), 100)
        self.assertEqual(normalized(2000, (1000, 3000), True), 50)

    def test_port_alias_cannot_own_same_adapter_twice(self):
        cfg = load_config(PROJECT / "configs/rig.json")
        cfg.pop("_root")
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "adapter"
            real.touch()
            alias = Path(tmp) / "alias"
            alias.symlink_to(real)
            cfg["arms"]["left_leader"]["port"] = str(real)
            cfg["arms"]["right_leader"]["port"] = str(alias)
            path = Path(tmp) / "rig.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "aliases"):
                load_config(path)


class HardwareTests(unittest.TestCase):
    def test_start_addresses_only_followers(self):
        rig = fake_rig()
        rig.start()
        for name, bus in rig.buses.items():
            if name.endswith("leader"):
                self.assertEqual(bus.moves, [])
                self.assertEqual(bus.writes, [])
            else:
                self.assertEqual(bus.moves[-1], {i: 2000 for i in rig.cfg["arms"][name]["ids"]})

    def test_actions_are_sent_commands_after_rate_limit(self):
        rig = fake_rig()
        for name in ("left_leader", "right_leader"):
            for f in rig.buses[name].values.values():
                f.position = 2700
        rig.previous_time = time.monotonic() - 0.05
        state, action, _, _ = rig.step(lambda: {})
        np.testing.assert_allclose(state[[0, 6]], [0, 0])
        self.assertGreater(action[0], 0)
        self.assertLessEqual(action[0], 4.01)  # At most 40 ticks per step.
        for offset, side in ((0, "left"), (6, "right")):
            name = f"{side}_follower"
            for j, i in enumerate(rig.cfg["arms"][name]["ids"]):
                self.assertAlmostEqual(action[offset + j], normalized(rig.sent[name][i], (1000, 3000), j == 5), places=4)

    def test_overload_does_not_stop_and_labels_previous_goal(self):
        rig = fake_rig()
        f = rig.buses["left_follower"].values[1]
        f.status = 0x20
        rig.buses["left_leader"].values[7].position = 2700
        rig.previous_time = time.monotonic() - 0.05
        with self.assertWarns(UserWarning):
            _, action, _, _ = rig.step(lambda: {})
        self.assertEqual(action[0], 0)  # No retransmission yet, even though desired target changed.
        self.assertNotIn(1, rig.buses["left_follower"].moves[-1] if rig.buses["left_follower"].moves else {})
        rig.last_sent["left_follower"][1] = time.monotonic() - 1
        rig.previous_time = time.monotonic() - 0.05
        _, action, _, _ = rig.step(lambda: {})
        self.assertIn(1, rig.buses["left_follower"].moves[-1])
        self.assertGreater(action[0], 0)

    def test_voltage_error_stops_before_new_commands(self):
        rig = fake_rig()
        rig.buses["right_follower"].values[1].status = 0x01
        with self.assertRaises(ProtocolError):
            rig.step(lambda: {})
        self.assertTrue(all(bus.moves == [] for bus in rig.buses.values()))

    def test_lost_camera_stops_before_commands(self):
        rig = fake_rig()
        def bad_camera():
            raise TimeoutError("camera lost")
        with self.assertRaises(TimeoutError):
            rig.step(bad_camera)
        self.assertTrue(all(bus.moves == [] for bus in rig.buses.values()))

    def test_invalid_right_range_blocks_both_followers(self):
        rig = fake_rig()
        rig.buses["left_leader"].values[7].position = 2700
        rig.buses["right_follower"].values[1].position = 3500
        rig.previous_time = time.monotonic() - 0.05
        with self.assertRaises(ValueError):
            rig.step(lambda: {})
        self.assertTrue(all(bus.moves == [] for bus in rig.buses.values()))

    def test_stop_signal_stops_before_commands(self):
        rig = fake_rig()
        with self.assertRaises(InterruptedError):
            rig.step(lambda: {}, lambda: True)
        self.assertTrue(all(bus.moves == [] for bus in rig.buses.values()))

    def test_shutdown_holds_and_never_writes_leaders(self):
        rig = fake_rig()
        rig.close()
        for name, bus in rig.buses.items():
            self.assertEqual(bool(bus.moves), name.endswith("follower"))
            self.assertEqual(bus.writes, [])

    def test_shutdown_fallback_torque_off_after_lost_feedback(self):
        rig = fake_rig()
        rig.buses["left_follower"].fail_read = True
        with self.assertRaisesRegex(ProtocolError, "Shutdown"):
            rig.close()
        self.assertEqual(rig.buses["left_follower"].writes[-1], (40, {i: b"\x00" for i in range(1, 7)}))
        self.assertEqual(rig.buses["left_leader"].writes, [])
        self.assertEqual(rig.buses["right_leader"].writes, [])

    def test_eeprom_change_rejected_without_motion(self):
        rig = fake_rig()
        calibrations = {}
        for name in ARM_NAMES:
            calibrations[name] = {"hardware_snapshot": snapshot(rig.buses[name], rig.cfg["arms"][name]["ids"])}
        calibrations["right_follower"]["hardware_snapshot"]["1"]["offset"] = "0100"
        with patch("arm100_collect.hardware.load_calibration", side_effect=lambda cfg, name: (rig.ranges[name], calibrations[name])), \
             patch("arm100_collect.hardware.open_bus", side_effect=lambda stack, port: next(rig.buses[name] for name, arm in rig.cfg["arms"].items() if arm["port"] == port)):
            # connect performs reads only, active set must correspond to a fresh connection.
            rig.active.clear()
            with self.assertRaisesRegex(ValueError, "EEPROM"):
                rig.connect()
        self.assertTrue(all(bus.moves == [] for bus in rig.buses.values()))


class DatasetIntegrationTests(unittest.TestCase):
    def test_resume_video_and_reject_changed_rig(self):
        cfg = copy.deepcopy(load_config(PROJECT / "configs/rig.json"))
        cfg["fps"] = 10
        cfg["cameras"] = {"overhead": {"serial": "synthetic", "width": 64, "height": 64, "fps": 10}}
        frame = {"observation.state": np.zeros(12, np.float32), "action": np.zeros(12, np.float32),
                 "task": "test", "observation.images.overhead": np.zeros((64, 64, 3), np.uint8)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            for resume in (False, True):
                ds = open_dataset(cfg, "local/test", root, resume=resume, simulated=True)
                try:
                    for _ in range(4):
                        ds.add_frame(dict(frame))
                    ds.save_episode(parallel_encoding=False)
                finally:
                    ds.finalize()
            report = validate_dataset(root)
            self.assertEqual((report["episodes"], report["frames"]), (2, 8))
            self.assertEqual(report["checked_episode_boundary_frames"], 4)
            cfg["invert"]["left"][0] = True
            with self.assertRaisesRegex(ValueError, "Cannot append"):
                open_dataset(cfg, "local/test", root, resume=True, simulated=True)


if __name__ == "__main__":
    unittest.main()
