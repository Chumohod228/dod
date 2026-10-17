"""Regression cases found during the full collection-project review."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from arm100_collect.cli import cleanup_recording, record
from arm100_collect.config import PROJECT, load_config, validate_follower_arc
from arm100_collect.dataset import open_dataset, validate_dataset
from arm100_collect.hardware import Cameras
from test_collection import fake_rig


class StartTests(unittest.TestCase):
    def test_invalid_right_arm_prevents_left_start(self):
        rig = fake_rig()
        rig.buses["right_follower"].values[1].position = 3500
        with self.assertRaises(ValueError):
            rig.start()
        self.assertTrue(all(not b.moves and not b.writes for b in rig.buses.values()))

    def test_invalid_leader_prevents_start(self):
        rig = fake_rig()
        rig.buses["right_leader"].values[1].position = 3500
        with self.assertRaises(ValueError):
            rig.start()
        self.assertTrue(all(not b.moves for b in rig.buses.values()))

    def test_signal_before_start_prevents_torque_enable(self):
        rig = fake_rig()
        with self.assertRaises(InterruptedError):
            rig.start(lambda: True)
        self.assertTrue(all(not b.moves and not b.writes for b in rig.buses.values()))

    def test_fractional_motion_accumulates_at_low_speed(self):
        rig = fake_rig()
        rig.cfg["speed_ticks_s"] = 1
        rig.cfg["deadband_ticks"] = 1
        rig.buses["left_leader"].values[7].position = 2700
        for _ in range(25):
            rig.previous_time = time.monotonic() - 0.05
            rig.step(lambda: {})
        self.assertGreater(rig.sent["left_follower"][1], 2000)
        self.assertLessEqual(rig.sent["left_follower"][1], 2002)

    def test_close_is_idempotent(self):
        rig = fake_rig()
        rig.close()
        counts = [len(b.moves) for b in rig.buses.values()]
        rig.close()
        self.assertEqual(counts, [len(b.moves) for b in rig.buses.values()])


class ConfigTests(unittest.TestCase):
    def test_invalid_numeric_values_fail_before_hardware(self):
        original = load_config(PROJECT / "configs/rig.json")
        for key, value in (("acceleration", 20.5), ("speed_ticks_s", True), ("retry_interval_s", float("nan")),
                           ("max_telemetry_age_s", float("inf")), ("fps", "30")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                cfg = copy.deepcopy(original)
                cfg[key] = value
                path = Path(tmp) / "rig.json"
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_follower_wrap_requires_hardware_offset(self):
        for limits in ((-100, 800), (3500, 4600)):
            with self.assertRaisesRegex(ValueError, "hardware zero"):
                validate_follower_arc(limits)
        validate_follower_arc((0, 4095))

    def test_camera_dimension_and_depth_conflicts(self):
        original = load_config(PROJECT / "configs/rig.json")
        for key, value in (("width", 641), ("use_depth", True), ("fps", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                cfg = copy.deepcopy(original)
                cfg["cameras"]["overhead"][key] = value
                path = Path(tmp) / "rig.json"
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_too_short_episode_rejected_without_connecting(self):
        args = Mock(config=PROJECT / "configs/rig.json", episode_seconds=0.001, reset_seconds=0,
                    task="test", episodes=1, auto_start=True, dataset="local/test")
        with patch("arm100_collect.hardware.Rig") as rig:
            with self.assertRaisesRegex(ValueError, "episode_seconds"):
                record(args)
            rig.assert_not_called()

    def test_calibration_files_cannot_alias(self):
        cfg = load_config(PROJECT / "configs/rig.json")
        cfg["arms"]["right_leader"]["calibration"] = cfg["arms"]["left_leader"]["calibration"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "separate calibration"):
                load_config(path)


class CameraCleanupTests(unittest.TestCase):
    def test_stale_buffered_image_is_rejected(self):
        cfg = load_config(PROJECT / "configs/rig.json")
        cameras = Cameras(cfg)
        device = Mock()
        device.async_read.return_value = np.zeros((480, 640, 3), np.uint8)
        device.read_latest.side_effect = TimeoutError("stale frame")
        cameras.devices["overhead"] = device
        with self.assertRaises(TimeoutError):
            cameras.read()

    def test_cleanup_finalizes_despite_clear_failure(self):
        rig, cameras, ds = Mock(), Mock(), Mock()
        rig.close.side_effect = RuntimeError("disconnected bus")
        ds.has_pending_frames.side_effect = KeyError("size")
        errors = cleanup_recording(rig, cameras, ds)
        cameras.close.assert_called_once()
        ds.finalize.assert_called_once()
        self.assertEqual(len(errors), 2)

    def test_all_cameras_disconnect_after_first_failure(self):
        cameras = Cameras({})
        first, second = Mock(is_connected=True), Mock(is_connected=True)
        first.disconnect.side_effect = OSError("USB lost")
        cameras.devices = {"first": first, "second": second}
        with self.assertRaises(RuntimeError):
            cameras.close()
        second.disconnect.assert_called_once()


class DatasetFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.cfg = load_config(PROJECT / "configs/rig.json")
        self.cfg["fps"] = 10
        self.cfg["cameras"] = {"overhead": {"serial": "test", "width": 64, "height": 64, "fps": 10}}

    def create(self):
        ds = open_dataset(self.cfg, "local/test", self.root, simulated=True)
        self.addCleanup(ds.finalize)
        return ds

    def frames(self, ds, invalid_middle=False):
        for k in range(4):
            state = np.zeros(12, np.float32)
            if invalid_middle and k == 1:
                state[0] = 999
            ds.add_frame({"observation.state": state, "action": np.zeros(12, np.float32),
                          "observation.images.overhead": np.zeros((64, 64, 3), np.uint8), "task": "test"})

    def test_two_writers_cannot_open_same_dataset(self):
        ds = self.create()
        with self.assertRaisesRegex(RuntimeError, "already open"):
            open_dataset(self.cfg, "local/test", self.root, resume=True, simulated=True)
        self.frames(ds)
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        resumed = open_dataset(self.cfg, "local/test", self.root, resume=True, simulated=True)
        resumed.finalize()

    def test_encoder_failure_leaves_marker_and_does_not_mask_error(self):
        ds = self.create()
        self.frames(ds)
        with patch.object(ds.dataset.writer, "_save_episode_video", side_effect=OSError("encoder failed")):
            with self.assertRaisesRegex(OSError, "encoder failed"):
                ds.save_episode(parallel_encoding=False)
        # Native LeRobot has already removed the buffer's 'size' key at this point.
        self.assertFalse(ds.has_pending_frames())
        ds.finalize()
        self.assertTrue((self.root / "capture_incomplete.json").exists())
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            open_dataset(self.cfg, "local/test", self.root, resume=True, simulated=True)

    def test_normal_finalize_removes_marker(self):
        ds = self.create()
        self.assertTrue((self.root / "capture_incomplete.json").exists())
        self.frames(ds)
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        self.assertFalse((self.root / "capture_incomplete.json").exists())

    def test_full_validation_finds_corrupt_middle_frame(self):
        ds = self.create()
        self.frames(ds, invalid_middle=True)
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        validate_dataset(self.root, quick=True)
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_dataset(self.root)

    def test_resume_rejects_changed_joint_order_in_metadata(self):
        ds = self.create()
        self.frames(ds)
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        path = self.root / "meta/info.json"
        info = json.loads(path.read_text())
        names = info["features"]["action"]["names"]
        names[0], names[1] = names[1], names[0]
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "joint order"):
            open_dataset(self.cfg, "local/test", self.root, resume=True, simulated=True)

    def test_record_summary_preserves_encoder_error_and_cleanup_failure(self):
        base = Path(self.temp.name)
        path = base / "configs/rig.json"
        path.parent.mkdir()
        path.write_text(json.dumps(self.cfg))
        rig = fake_rig()
        rig.cfg = self.cfg
        rig.connect = Mock(return_value=rig)
        rig.close = Mock(side_effect=OSError("hold failed"))
        camera = Mock()
        camera.read.return_value = {"observation.images.overhead": np.zeros((64, 64, 3), np.uint8)}
        args = Mock(config=path, dataset="local/test", root=self.root, episode_seconds=0.3,
                    reset_seconds=0, task="test", episodes=1, auto_start=True, resume=False)
        with patch("arm100_collect.cli.PROJECT", base), patch("arm100_collect.cli.stop_flag", return_value=[False]), \
             patch("arm100_collect.hardware.Rig", return_value=rig), \
             patch("arm100_collect.hardware.Cameras", return_value=camera), \
             patch("lerobot.datasets.dataset_writer.DatasetWriter._save_episode_video", side_effect=OSError("encoder failed")):
            with self.assertRaisesRegex(OSError, "encoder failed"):
                record(args)
        summaries = list((base / "logs").glob("record_*/summary.json"))
        self.assertEqual(len(summaries), 1)
        summary = json.loads(summaries[0].read_text())
        self.assertEqual(summary["outcome"], "error")
        self.assertIn("encoder failed", summary["error"])
        self.assertIn("hold failed", summary["cleanup_errors"][0])
        self.assertEqual(summary["attempts"][0]["outcome"], "error")
        camera.close.assert_called_once()
