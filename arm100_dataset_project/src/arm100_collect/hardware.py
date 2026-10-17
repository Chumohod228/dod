"""Four-port STS3215 teleoperation, without calibration or PID writes."""
from contextlib import ExitStack
import time
import warnings

import numpy as np
import serial

from .config import ARM_NAMES, fraction, load_calibration, normalized, raw_segments, unwrap
from .transport import ProtocolError, ServoBus, feedback_error

OVERLOAD = 0x20
REGISTERS = {"model": (3, 2), "limits": (9, 4), "phase": (18, 1), "offset": (31, 2), "mode": (33, 1)}


def open_bus(stack, port):
    handle = stack.enter_context(serial.Serial(port, 1000000, timeout=0.002, write_timeout=0.2, exclusive=True))
    return ServoBus(handle)


def checked_read(bus, servo_id, address, size, allowed=0):
    data, status = bus.read(servo_id, address, size)
    if status & ~allowed:
        raise ProtocolError(f"ID {servo_id}: status 0x{status:02X} reading {address}")
    return data


def snapshot(bus, ids, allowed=0):
    result = {}
    for i in ids:
        values = {key: checked_read(bus, i, *reg, allowed).hex() for key, reg in REGISTERS.items()}
        if int.from_bytes(bytes.fromhex(values["model"]), "little") != 777:
            raise ProtocolError(f"ID {i}: expected STS3215 model 777")
        if values["mode"] != "00" or bytes.fromhex(values["phase"])[0] & 0x10:
            raise ProtocolError(f"ID {i}: expected single-turn position mode")
        result[str(i)] = values
    return result


class Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stack = ExitStack()
        self.buses = {}
        self.ranges = {}
        self.calibrations = {}
        self.feedback = {}
        self.targets = {}
        self.sent = {}
        self.last_sent = {}
        self.previous_leader = {}
        self.active = set()
        self.overloaded = set()
        self.previous_time = None

    def connect(self):
        try:
            # Validate every calibration before opening any port.
            for name in ARM_NAMES:
                self.ranges[name], self.calibrations[name] = load_calibration(self.cfg, name)
            for name in ARM_NAMES:
                self.buses[name] = open_bus(self.stack, self.cfg["arms"][name]["port"])
            for name, bus in self.buses.items():
                ids = self.cfg["arms"][name]["ids"]
                allowed = OVERLOAD if name.endswith("follower") else 0
                actual = snapshot(bus, ids, allowed)
                expected = self.calibrations[name].get("hardware_snapshot")
                if not expected or actual != expected:
                    raise ValueError(f"{name}: EEPROM configuration changed or calibration lacks snapshot; recalibrate")
                if name.endswith("leader"):
                    for i in ids:
                        if checked_read(bus, i, 40, 1) != b"\x00":
                            raise ProtocolError(f"{name} ID {i}: leader torque must be off")
                else:
                    for i in ids:
                        bounds = bytes.fromhex(actual[str(i)]["limits"])
                        hwlo, hwhi = int.from_bytes(bounds[:2], "little"), int.from_bytes(bounds[2:], "little")
                        if any(not hwlo <= lo <= hi <= hwhi for lo, hi in raw_segments(self.ranges[name][i])):
                            raise ValueError(f"{name} ID {i}: software arc exceeds EEPROM limits {hwlo}..{hwhi}")
            self.feedback = self.read_all()
            for name, feedback in self.feedback.items():
                for i, f in feedback.items():
                    unwrap(f.position, self.ranges[name][i])
            return self
        except BaseException:
            self.stack.close()
            raise

    def read_all(self):
        result = {}
        for name, bus in self.buses.items():
            values = {}
            for i in self.cfg["arms"][name]["ids"]:
                f = bus.feedback(i)
                allowed = OVERLOAD if name.endswith("follower") else 0
                if f.status & ~allowed:
                    raise ProtocolError(f"{name}: {feedback_error(f)}")
                values[i] = f
            result[name] = values
        return result

    def start(self, stop=lambda: False):
        # Hold each follower's own position before enabling following.
        started = time.monotonic()
        self.feedback = self.read_all()
        for name, feedback in self.feedback.items():
            for i, f in feedback.items():
                unwrap(f.position, self.ranges[name][i], tolerance=0 if name.endswith("follower") else 16)
        if time.monotonic() - started > self.cfg["max_telemetry_age_s"]:
            raise ProtocolError("Initial telemetry too old; following not started")
        for side in ("left", "right"):
            if stop():
                raise InterruptedError("Stop received before enabling following")
            name = f"{side}_follower"
            self.targets[name] = {i: float(unwrap(f.position, self.ranges[name][i])) for i, f in self.feedback[name].items()}
            self.sent[name] = {i: round(value) for i, value in self.targets[name].items()}
            self.active.add(name)  # Goal writes can themselves enable torque.
            self.buses[name].move({i: v % 4096 for i, v in self.sent[name].items()}, self.cfg["speed_ticks_s"], self.cfg["acceleration"])
            self.buses[name].sync_write(40, {i: b"\x01" for i in self.sent[name]})
            self.last_sent[name] = {i: time.monotonic() for i in self.sent[name]}
            leader = f"{side}_leader"
            self.previous_leader[leader] = {i: f.position for i, f in self.feedback[leader].items()}
        self.previous_time = time.monotonic()

    def pause(self):
        """Cancel motion while the episode is encoded or the scene is reset."""
        for name in self.active:
            positions = {}
            for i in self.cfg["arms"][name]["ids"]:
                f = self.buses[name].feedback(i)
                if f.status & ~OVERLOAD:
                    raise ProtocolError(feedback_error(f))
                unwrap(f.position, self.ranges[name][i], tolerance=8)
                positions[i] = f.position
            self.buses[name].move(positions, self.cfg["speed_ticks_s"], self.cfg["acceleration"])

    def step(self, images, stop=lambda: False):
        started = time.monotonic()
        self.feedback = self.read_all()
        # Read fresh images before issuing a new movement command.
        camera_frames = images()
        now = time.monotonic()
        if now - started > self.cfg["max_telemetry_age_s"] or now - self.previous_time > self.cfg["max_telemetry_age_s"]:
            raise ProtocolError("Telemetry/camera stalled; following stopped")
        dt = min(now - self.previous_time, 2 / self.cfg["fps"])
        self.previous_time = now
        states, actions, telemetry = [], [], []
        current_overload = set()
        pending = {}
        for side in ("left", "right"):
            leader, follower = f"{side}_leader", f"{side}_follower"
            commands = {}
            for j, (lid, fid) in enumerate(zip(self.cfg["arms"][leader]["ids"], self.cfg["arms"][follower]["ids"], strict=True)):
                lf, ff = self.feedback[leader][lid], self.feedback[follower][fid]
                delta = (lf.position - self.previous_leader[leader][lid] + 2048) % 4096 - 2048
                if abs(delta) > self.cfg["max_leader_step_ticks"]:
                    raise ProtocolError(f"{leader} ID {lid}: implausible jump {delta}")
                self.previous_leader[leader][lid] = lf.position
                ratio = fraction(unwrap(lf.position, self.ranges[leader][lid]), self.ranges[leader][lid])
                if self.cfg["invert"][side][j]:
                    ratio = 1 - ratio
                lo, hi = self.ranges[follower][fid]
                desired = lo + ratio * (hi - lo)
                old = self.targets[follower][fid]
                change = self.cfg["speed_ticks_s"] * dt
                self.targets[follower][fid] = old + max(-change, min(change, desired - old))
                target = round(self.targets[follower][fid])
                position = unwrap(ff.position, (lo, hi), tolerance=8)
                due = now - self.last_sent[follower][fid] >= self.cfg["retry_interval_s"]
                overloaded = bool(ff.status & OVERLOAD)
                if overloaded:
                    current_overload.add((follower, fid))
                if (overloaded and due) or (not overloaded and (
                    abs(target - self.sent[follower][fid]) >= self.cfg["deadband_ticks"]
                    or (due and abs(target - position) >= self.cfg["deadband_ticks"]))):
                    commands[fid] = target
                states.append(normalized(position, (lo, hi), j == 5))
                telemetry.append((follower, fid, ff.position, ff.voltage_v, ff.current_ma / 1000, ff.temperature_c, ff.status))
            pending[follower] = commands
        # Both arms must pass all checks before either receives a new goal.
        for follower, commands in pending.items():
            if stop():
                raise InterruptedError("Stop received before sending motion commands")
            if commands:
                self.buses[follower].move({i: v % 4096 for i, v in commands.items()}, self.cfg["speed_ticks_s"], self.cfg["acceleration"])
                self.sent[follower].update(commands)
                self.last_sent[follower].update({i: now for i in commands})
        # Labels are the actual last sent commands, including deadband/retry.
        for side in ("left", "right"):
            follower = f"{side}_follower"
            actions.extend(normalized(self.sent[follower][fid], self.ranges[follower][fid], j == 5)
                           for j, fid in enumerate(self.cfg["arms"][follower]["ids"]))
        for name, i in current_overload - self.overloaded:
            warnings.warn(f"{name} ID {i}: overload 0x20; retry every {self.cfg['retry_interval_s']} s", stacklevel=2)
        self.overloaded = current_overload
        return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), camera_frames, telemetry

    def close(self):
        errors = []
        for name in self.active:
            try:
                bus = self.buses[name]
                ids = self.cfg["arms"][name]["ids"]
                if self.cfg["shutdown"] == "torque_off":
                    bus.sync_write(40, {i: b"\x00" for i in ids})
                else:
                    positions = {}
                    for i in ids:
                        f = bus.feedback(i)
                        if f.status & ~OVERLOAD:
                            raise ProtocolError(feedback_error(f))
                        unwrap(f.position, self.ranges[name][i], tolerance=8)
                        positions[i] = f.position
                    bus.move(positions, self.cfg["speed_ticks_s"], self.cfg["acceleration"])
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                # If holding is impossible, cancel the previous movement goal.
                try:
                    self.buses[name].sync_write(40, {i: b"\x00" for i in self.cfg["arms"][name]["ids"]})
                except Exception as off_error:
                    errors.append(f"{name}: torque-off failed: {off_error}")
        self.stack.close()
        self.active.clear()
        if errors:
            raise ProtocolError("Shutdown: " + "; ".join(errors))


class Cameras:
    def __init__(self, cfg):
        self.cfg = cfg
        self.devices = {}

    def connect(self):
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        try:
            for name, cam in self.cfg["cameras"].items():
                device = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=cam["serial"],
                                          fps=cam["fps"], width=cam["width"], height=cam["height"], use_depth=False))
                self.devices[name] = device
                device.connect()
            self.read()
            return self
        except BaseException:
            self.close()
            raise

    def read(self):
        frames = {}
        for name, device in self.devices.items():
            device.async_read(timeout_ms=100)
            # A buffered event alone does not prove the sensor is still producing frames.
            frame = device.read_latest(max_age_ms=self.cfg["max_telemetry_age_s"] * 1000)
            cam = self.cfg["cameras"][name]
            if frame.shape != (cam["height"], cam["width"], 3) or frame.dtype != np.uint8:
                raise ValueError(f"{name}: unexpected RGB format {frame.shape}/{frame.dtype}")
            frames[f"observation.images.{name}"] = frame.copy()
        return frames

    def close(self):
        errors = []
        for device in self.devices.values():
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Camera shutdown: " + "; ".join(errors))
