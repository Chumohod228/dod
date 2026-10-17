"""Portable rig configuration and continuous single-turn coordinates."""
import json
import math
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_NAMES = tuple(f"{side}_{role}" for side in ("left", "right") for role in ("leader", "follower"))
NAMES = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINTS]
PROJECT = Path(__file__).resolve().parents[2]


def number(value, name, low, high, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}: expected {'integer' if integer else 'finite number'} in {low}..{high}")
    return value


def validate_follower_arc(limits):
    if not 0 <= limits[0] < limits[1] <= 4095:
        raise ValueError("Follower range must lie within 0..4095 in position mode; calibrate the hardware zero offset first. Leader ranges may cross zero.")


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("arms"), dict):
        raise ValueError("Rig configuration must be an object with an 'arms' object")
    cfg["_root"] = str(path.parent.parent)
    if set(cfg["arms"]) != set(ARM_NAMES):
        raise ValueError(f"Expected four arms: {ARM_NAMES}")
    ports = []
    calibration_paths = []
    for name, arm in cfg["arms"].items():
        if not isinstance(arm, dict):
            raise ValueError(f"{name}: arm configuration must be an object")
        ids = arm["ids"]
        if not isinstance(ids, list) or len(ids) != 6 or any(type(i) is not int or not 1 <= i <= 253 for i in ids) or len(set(ids)) != 6:
            raise ValueError(f"{name}: six unique IDs in joint order required")
        if not isinstance(arm["port"], str) or not Path(arm["port"]).is_absolute():
            raise ValueError(f"{name}: port must be an absolute device path")
        if not isinstance(arm["calibration"], str) or not arm["calibration"].strip():
            raise ValueError(f"{name}: calibration file path required")
        ports.append(str(Path(arm["port"]).resolve()))
        calibration_paths.append(str(resolve_path(cfg, arm["calibration"]).resolve()))
    if len(set(ports)) != 4:
        raise ValueError("Four separate serial adapters required; port aliases must not overlap")
    if len(set(calibration_paths)) != 4:
        raise ValueError("Each arm must have a separate calibration file")
    for key, low, high in (("fps", 1, 60), ("speed_ticks_s", 1, 2000), ("acceleration", 1, 254),
                           ("deadband_ticks", 0, 100), ("max_leader_step_ticks", 1, 2047)):
        number(cfg[key], key, low, high, integer=True)
    number(cfg["retry_interval_s"], "retry_interval_s", 0.1, 60)
    number(cfg["max_telemetry_age_s"], "max_telemetry_age_s", 0.001, 0.5)
    if cfg["shutdown"] not in ("hold", "torque_off"):
        raise ValueError("shutdown must be hold or torque_off")
    for side in ("left", "right"):
        if len(cfg["invert"][side]) != 6 or any(type(v) is not bool for v in cfg["invert"][side]):
            raise ValueError(f"{side}: invert must contain six booleans")
    if not cfg["cameras"]:
        raise ValueError("At least one RGB camera required for visual training")
    serials = []
    for name, cam in cfg["cameras"].items():
        if not name.isidentifier() or not isinstance(cam["serial"], str) or not cam["serial"].strip():
            raise ValueError(f"Invalid camera {name}")
        for key in ("width", "height"):
            number(cam[key], f"{name}.{key}", 32, 8192, integer=True)
            if cam[key] % 2:
                raise ValueError(f"{name}.{key}: H264 yuv420p needs an even dimension")
        number(cam["fps"], f"{name}.fps", 1, 60, integer=True)
        if cam.get("use_depth", False) is not False:
            raise ValueError("This project records RGB only; depth is disabled")
        if cam["fps"] != cfg["fps"]:
            raise ValueError("Camera and dataset FPS must match")
        serials.append(cam["serial"])
    if len(set(serials)) != len(serials):
        raise ValueError("Camera serial numbers must be unique")
    if len({(c["width"], c["height"]) for c in cfg["cameras"].values()}) != 1:
        raise ValueError("ACT requires all RGB cameras to have the same resolution")
    return cfg


def resolve_path(cfg, path):
    return Path(cfg["_root"]) / path


def load_calibration(cfg, name):
    arm = cfg["arms"][name]
    path = resolve_path(cfg, arm["calibration"])
    data = json.loads(path.read_text())
    if data.get("arm") != name or data.get("ids") != arm["ids"]:
        raise ValueError(f"{path}: arm name or IDs differ from rig.json")
    ranges = {}
    for i in arm["ids"]:
        pair = data["ranges"][str(i)]
        if len(pair) != 2 or any(type(v) is not int for v in pair) or not 32 <= pair[1] - pair[0] < 4096:
            raise ValueError(f"{name} ID {i}: invalid range (span must be 32..4095 ticks)")
        ranges[i] = tuple(pair)
        if name.endswith("follower"):
            validate_follower_arc(pair)
    return ranges, data


def unwrap(raw, limits, tolerance=16):
    if not 0 <= raw < 4096:
        raise ValueError(f"Invalid single-turn feedback: {raw}")
    lo, hi = limits
    value = raw + 4096 * math.floor(((lo + hi) / 2 - raw + 2048) / 4096)
    if not lo - tolerance <= value <= hi + tolerance:
        raise ValueError(f"Position {value} outside calibrated range {lo}..{hi}")
    return value


def fraction(value, limits):
    lo, hi = limits
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def normalized(value, limits, gripper=False):
    ratio = fraction(value, limits)
    return 100 * ratio if gripper else 200 * ratio - 100


def raw_segments(limits):
    """Raw intervals occupied by a continuous arc, including arcs across zero."""
    lo, hi = limits
    low, high = lo % 4096, hi % 4096
    return [(low, high)] if lo // 4096 == hi // 4096 else [(low, 4095), (0, high)]
