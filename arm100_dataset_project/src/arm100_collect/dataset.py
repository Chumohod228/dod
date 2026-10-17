"""LeRobot v3 recordings and reproducible capture metadata."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import fcntl
import os

from .config import NAMES


class RecordingDataset:
    """Own a single writer; retain an incomplete-session marker on failed writes."""
    def __init__(self, dataset, lock):
        self.dataset, self.lock = dataset, lock
        self.failed_write = False
        self.closed = False
        self.marker = dataset.root / "capture_incomplete.json"
        self.marker.write_text(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}))

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def add_frame(self, frame):
        try:
            self.dataset.add_frame(frame)
        except BaseException:
            self.failed_write = True
            raise

    def save_episode(self, **kwargs):
        try:
            self.dataset.save_episode(**kwargs)
        except BaseException:
            # LeRobot removes 'size' from the live buffer before encoding video.
            self.failed_write = True
            raise

    def has_pending_frames(self):
        buffer = self.dataset.writer.episode_buffer
        return bool(buffer and buffer.get("size", 0))

    def clear_episode_buffer(self):
        if not self.failed_write:
            try:
                self.dataset.clear_episode_buffer()
            except BaseException:
                self.failed_write = True
                raise

    def finalize(self):
        if self.closed:
            return
        try:
            self.dataset.finalize()
            if not self.failed_write:
                self.marker.unlink(missing_ok=True)
        finally:
            self.closed = True
            self.lock.close()


def features(cfg):
    result = {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (12,), "names": NAMES},
    }
    for name, cam in cfg["cameras"].items():
        result[f"observation.images.{name}"] = {"dtype": "video", "shape": (cam["height"], cam["width"], 3), "names": ["height", "width", "channels"]}
    return result


def recording_signature(cfg, calibrations, simulated):
    relevant = {"fps": cfg["fps"], "arms": cfg["arms"], "cameras": cfg["cameras"], "invert": cfg["invert"],
                "speed_ticks_s": cfg["speed_ticks_s"], "acceleration": cfg["acceleration"],
                "deadband_ticks": cfg["deadband_ticks"], "retry_interval_s": cfg["retry_interval_s"],
                "calibrations": calibrations, "simulated": simulated}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def open_dataset(cfg, repo_id, root, calibrations=None, resume=False, simulated=False):
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Do not unlink this file: all processes must lock the same inode.
    lock = (root.parent / f".{root.name}.capture.lock").open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Dataset is already open for recording: {root}") from exc
        if (root / "capture_incomplete.json").exists():
            raise RuntimeError(f"Previous write did not complete: {root}. Preserve a backup and review/repair it before resuming; use a new dataset for further recordings.")
        ds = _open_dataset(cfg, repo_id, root, calibrations, resume, simulated)
        try:
            return RecordingDataset(ds, lock)
        except BaseException:
            ds.finalize()
            raise
    except BaseException:
        lock.close()
        raise


def _open_dataset(cfg, repo_id, root, calibrations, resume, simulated):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root).resolve()
    signature = recording_signature(cfg, calibrations or {}, simulated)
    encoder = RGBEncoderConfig(vcodec="h264", preset="fast", crf=18, g=2)
    if resume:
        metadata = json.loads((root / "capture_config.json").read_text())
        if metadata["signature"] != signature or metadata["repo_id"] != repo_id:
            raise ValueError("Cannot append: rig, calibration, camera settings, simulation flag or dataset ID changed")
        check_schema(cfg, json.loads((root / "meta/info.json").read_text()))
        ds = LeRobotDataset.resume(repo_id, root=root, rgb_encoder=encoder, image_writer_threads=2,
                                  video_backend="pyav", encoder_threads=2)
    else:
        if root.exists():
            raise FileExistsError(f"Dataset already exists: {root}; use --resume or a new --root")
        ds = LeRobotDataset.create(repo_id, fps=cfg["fps"], features=features(cfg), root=root,
                                   robot_type="bi_so100_follower_sts3215", use_videos=True,
                                   rgb_encoder=encoder, image_writer_threads=2, video_backend="pyav",
                                   encoder_threads=2, metadata_buffer_size=1)
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "repo_id": repo_id,
                    "signature": signature, "simulated": simulated, "config": cfg,
                    "calibrations": calibrations or {}, "lerobot_version": "0.6.1",
                    "coordinates": "joints -100..100, grippers 0..100; left six then right six",
                    "action_semantics": "last command actually transmitted, after mapping, rate limit and deadband",
                    "video": "RGB H264 MP4; depth disabled; software synchronization per dataset frame"}
        try:
            (root / "capture_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        except BaseException:
            ds.finalize()
            raise
    return ds


def check_schema(cfg, info):
    if info["fps"] != cfg["fps"]:
        raise ValueError("Dataset FPS differs from capture configuration")
    expected = features(cfg)
    for key, spec in expected.items():
        actual = info["features"].get(key, {})
        if (actual.get("dtype") != spec["dtype"] or tuple(actual.get("shape", [])) != spec["shape"]
                or actual.get("names") != spec["names"]):
            raise ValueError(f"Feature schema or joint order mismatch: {key}")
    actual_videos = {k for k, v in info["features"].items() if v["dtype"] == "video"}
    if actual_videos != {k for k, v in expected.items() if v["dtype"] == "video"}:
        raise ValueError("Dataset camera set differs from capture configuration")


def validate_dataset(root, quick=False):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root).resolve()
    capture = json.loads((root / "capture_config.json").read_text())
    if (root / "capture_incomplete.json").exists():
        raise ValueError("Dataset contains an unfinished/failed recording session")
    ds = LeRobotDataset(capture["repo_id"], root=root, video_backend="pyav")
    if ds.num_episodes < 1 or ds.num_frames < 1:
        raise ValueError("Dataset has no saved episodes")
    check_schema(capture["config"], json.loads((root / "meta/info.json").read_text()))
    # Every episode's first and last frame: exercises parquet/video indexing and boundaries.
    indices = set()
    end = 0
    for episode in ds.meta.episodes:
        if int(episode["dataset_from_index"]) != end:
            raise ValueError("Dataset episode intervals are not contiguous")
        end = int(episode["dataset_to_index"])
        indices.add(int(episode["dataset_from_index"]))
        indices.add(int(episode["dataset_to_index"]) - 1)
    if end != ds.num_frames:
        raise ValueError("Episode lengths do not match dataset frame count")
    checked = sorted(indices) if quick else range(ds.num_frames)
    for index in checked:
        frame = ds[index]
        for key in ("observation.state", "action"):
            array = frame[key].numpy()
            if array.shape != (12,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key} at frame {index}")
            for j, value in enumerate(array):
                low = 0 if j % 6 == 5 else -100
                if not low - 0.01 <= value <= 100.01:
                    raise ValueError(f"Coordinate out of range: {key}[{j}]={value}")
        for key in ds.meta.video_keys:
            cam_shape = ds.meta.features[key]["shape"]
            if tuple(frame[key].shape) != (3, cam_shape[0], cam_shape[1]):
                raise ValueError(f"Invalid decoded image {key}")
    return {"root": str(root), "episodes": ds.num_episodes, "frames": ds.num_frames,
            "video_streams": ds.meta.video_keys, "joints": 12, "simulated": capture["simulated"],
            "checked_episode_boundary_frames": len(indices), "checked_frames": len(checked), "quick": quick}
