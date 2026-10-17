#!/usr/bin/env python3
"""Random STS3215 motion and per-servo sampled current peaks via a USB TTL adapter."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import signal
import struct
import sys
import time


CURRENT_MA_PER_TICK = 6.5
DEFAULT_LIMITS = {i: (1023, 2047 if i == 6 else 3071) for i in range(1, 7)}


class ProtocolError(Exception):
    pass


def packet(servo_id: int, instruction: int, parameters: bytes = b"") -> bytes:
    body = bytes((servo_id, len(parameters) + 2, instruction)) + parameters
    return b"\xff\xff" + body + bytes(((~sum(body)) & 0xFF,))


def signed_magnitude(word: int) -> int:
    magnitude = word & 0x7FFF
    return -magnitude if word & 0x8000 else magnitude


@dataclass
class Feedback:
    servo_id: int
    position: int
    speed: int
    voltage_v: float
    temperature_c: int
    moving: int
    current_raw: int
    current_ma: float
    status: int


def decode_feedback(servo_id: int, data: bytes, status: int) -> Feedback:
    if len(data) != 15:
        raise ProtocolError(f"ID {servo_id}: expected 15 feedback bytes, got {len(data)}")
    raw = int.from_bytes(data[13:15], "little")
    return Feedback(
        servo_id, signed_magnitude(int.from_bytes(data[0:2], "little")),
        signed_magnitude(int.from_bytes(data[2:4], "little")),
        data[6] / 10, data[7], data[10], raw,
        signed_magnitude(raw) * CURRENT_MA_PER_TICK, status,
    )


def feedback_error(f: Feedback) -> str:
    reason = "; ошибка напряжения питания" if f.status & 0x01 else ""
    return (f"ID {f.servo_id}: servo reports status 0x{f.status:02X}{reason}; "
            f"U={f.voltage_v:.1f} V, I={f.current_ma / 1000:.3f} A")


class ServoBus:
    """One owner, one outstanding request. Never share this port with another master."""

    def __init__(self, port, timeout: float = 0.03):
        self.port = port
        self.timeout = timeout
        self.bad_checksums = 0
        self.ignored_packets = 0

    def send(self, data: bytes):
        if self.port.write(data) != len(data):
            raise ProtocolError("Incomplete serial write")
        self.port.flush()

    def read(self, servo_id: int, address: int, size: int) -> tuple[bytes, int]:
        request = packet(servo_id, 0x02, bytes((address, size)))
        self.port.reset_input_buffer()
        self.send(request)
        deadline = time.monotonic() + self.timeout
        buf = bytearray()
        while time.monotonic() < deadline:
            buf.extend(self.port.read(max(1, min(self.port.in_waiting, 256))))
            while len(buf) >= 4:
                start = buf.find(b"\xff\xff")
                if start < 0:
                    buf[:] = buf[-1:] if buf[-1:] == b"\xff" else b""
                    break
                del buf[:start]
                if len(buf) < 4:
                    break
                # Accept only known response lengths or our own adapter echo.
                if buf[2] > 253 or buf[3] not in (2, 4, size + 2):
                    del buf[0]
                    continue
                length = buf[3] + 4
                if len(buf) < length:
                    break
                candidate = bytes(buf[:length])
                if sum(candidate[2:]) & 0xFF != 0xFF:
                    self.bad_checksums += 1
                    del buf[0]
                    continue
                del buf[:length]
                if candidate == request:  # Some USB/TTL adapters echo TX.
                    continue
                if candidate[2] != servo_id:
                    self.ignored_packets += 1
                    continue
                status = candidate[4]
                data = candidate[5:-1]
                if len(data) == size:
                    return data, status
                if len(data) == 0 and status:
                    raise ProtocolError(f"ID {servo_id}: status 0x{status:02X}, no data")
                self.ignored_packets += 1
        raise ProtocolError(f"ID {servo_id}: timeout reading register {address}")

    def feedback(self, servo_id: int) -> Feedback:
        data, status = self.read(servo_id, 56, 15)
        return decode_feedback(servo_id, data, status)

    def sync_write(self, address: int, values: dict[int, bytes]):
        if not values:
            return
        size = len(next(iter(values.values())))
        if any(len(value) != size for value in values.values()):
            raise ValueError("Inconsistent sync-write sizes")
        params = bytes((address, size))
        params += b"".join(bytes((i,)) + value for i, value in values.items())
        self.send(packet(0xFE, 0x83, params))

    def move(self, positions: dict[int, int], speed: int, acceleration: int):
        # STS registers 41..47: ACC, position LE, time=0, speed LE.
        self.sync_write(41, {
            i: struct.pack("<BHHH", acceleration, position, 0, speed)
            for i, position in positions.items()
        })


def next_targets(rng, positions, limits, max_step, min_separation=0):
    result = {}
    for servo_id, position in positions.items():
        lo, hi = limits[servo_id]
        span = hi - lo
        lo, hi = max(lo, position - max_step), min(hi, position + max_step)
        distance = min(max(1, math.ceil(span * min_separation)),
                       max(position - lo, hi - position))
        intervals = [(lo, min(hi, position - distance)),
                     (max(lo, position + distance), hi)] if distance else []
        intervals = [(a, b) for a, b in intervals if a <= b]
        choices = sum(b - a + 1 for a, b in intervals)
        if not choices:
            result[servo_id] = position
        else:
            value = rng.randrange(choices)
            for a, b in intervals:
                if value < b - a + 1:
                    result[servo_id] = a + value
                    break
                value -= b - a + 1
    return result


class Recorder:
    fields = ["utc", "elapsed_s", "phase", "servo_id", "target_position",
              "position", "speed", "voltage_v", "temperature_c", "moving",
              "current_raw", "current_ma", "abs_current_ma", "status"]

    def __init__(self, folder: Path, ids, config):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder
        self.start = time.monotonic()
        self.config = config
        self.rows = {i: {"samples": 0, "fault_samples": 0, "max_abs_current_ma": None,
                         "peak": None, "first_elapsed_s": None, "last_elapsed_s": None,
                         "voltage_v": None, "min_voltage_v": None, "max_voltage_v": None,
                         "min_voltage_sample": None, "max_voltage_sample": None}
                     for i in ids}
        self.file = (folder / "samples.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        self.errors = []
        self.commands_file = (folder / "commands.csv").open("w", newline="", encoding="utf-8")
        self.commands = csv.writer(self.commands_file)
        self.commands.writerow(["elapsed_s", "phase", "servo_id", "target_position"])
        self.save("starting")

    def sample(self, f: Feedback, phase: str, target=None):
        row = dict(utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=round(time.monotonic() - self.start, 6),
                   phase=phase, target_position=target, **asdict(f),
                   abs_current_ma=abs(f.current_ma))
        self.writer.writerow(row)
        stats = self.rows[f.servo_id]
        stats["samples"] += 1
        stats["fault_samples"] += bool(f.status)
        if stats["first_elapsed_s"] is None:
            stats["first_elapsed_s"] = row["elapsed_s"]
        stats["last_elapsed_s"] = row["elapsed_s"]
        stats["voltage_v"] = f.voltage_v
        for kind, compare in (("min", lambda a, b: a < b), ("max", lambda a, b: a > b)):
            key = f"{kind}_voltage_v"
            if stats[key] is None or compare(f.voltage_v, stats[key]):
                stats[key] = f.voltage_v
                stats[f"{kind}_voltage_sample"] = row
        if stats["peak"] is None or abs(f.current_ma) > stats["max_abs_current_ma"]:
            stats["max_abs_current_ma"] = abs(f.current_ma)
            stats["peak"] = row

    def command(self, positions, phase):
        for i, position in positions.items():
            self.commands.writerow([round(time.monotonic() - self.start, 6), phase, i, position])

    def error(self, message):
        self.errors.append({"elapsed_s": time.monotonic() - self.start, "message": message})
        print(message, file=sys.stderr, flush=True)

    def save(self, outcome, bus=None):
        self.file.flush()
        self.commands_file.flush()
        document = {"outcome": outcome, "config": self.config,
                    "duration_s": time.monotonic() - self.start,
                    "peak_definition": "max(abs(signed current)) over valid sampled telemetry",
                    "current_ma_per_tick": CURRENT_MA_PER_TICK,
                    "bad_checksums": bus.bad_checksums if bus else 0,
                    "ignored_packets": bus.ignored_packets if bus else 0,
                    "servos": self.rows, "errors": self.errors}
        temp = self.folder / "summary.json.tmp"
        temp.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.folder / "summary.json")
        temp = self.folder / "peaks.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["servo_id", "max_abs_current_ma", "max_abs_current_a",
                             "peak_utc", "peak_position", "peak_phase", "samples", "fault_samples",
                             "voltage_v", "min_voltage_v", "max_voltage_v",
                             "min_voltage_utc", "max_voltage_utc", "voltage_at_current_peak_v"])
            for i, stats in self.rows.items():
                peak = stats["peak"] or {}
                value = stats["max_abs_current_ma"]
                writer.writerow([i, value, None if value is None else value / 1000,
                                 peak.get("utc"), peak.get("position"), peak.get("phase"),
                                 stats["samples"], stats["fault_samples"],
                                 stats["voltage_v"], stats["min_voltage_v"], stats["max_voltage_v"],
                                 (stats["min_voltage_sample"] or {}).get("utc"),
                                 (stats["max_voltage_sample"] or {}).get("utc"), peak.get("voltage_v")])
        temp.replace(self.folder / "peaks.csv")

    def display(self):
        parts = []
        for i, stats in self.rows.items():
            value = stats["max_abs_current_ma"]
            parts.append(f"ID {i}: {'нет данных' if value is None else f'{value / 1000:.3f} A'}")
        print("Максимумы | " + " | ".join(parts), flush=True)
        parts = []
        for i, stats in self.rows.items():
            if stats["voltage_v"] is None:
                parts.append(f"ID {i}: нет данных")
            else:
                parts.append(f"ID {i}: {stats['voltage_v']:.1f} V "
                             f"(min {stats['min_voltage_v']:.1f}, max {stats['max_voltage_v']:.1f})")
        print("Напряжение | " + " | ".join(parts), flush=True)

    def close(self):
        self.file.close()
        self.commands_file.close()


def load_limits(path, ids):
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        limits = {int(i): tuple(pair) for i, pair in data.items()}
    else:
        limits = DEFAULT_LIMITS
    for i in ids:
        pair = limits.get(i, ())
        if (len(pair) != 2 or any(type(n) is not int for n in pair)
                or not 0 <= pair[0] < pair[1] <= 4095):
            raise ValueError(f"ID {i}: limits must be [min, max] within 0..4095")
    return {i: limits[i] for i in ids}


def hold_current_positions(bus, recorder, limits, speed, acceleration, allowed_status=0):
    positions = {}
    for servo_id in limits:
        try:
            f = bus.feedback(servo_id)
            recorder.sample(f, "stopping")
            lo, hi = limits[servo_id]
            if not (f.status & ~allowed_status) and lo <= f.position <= hi:
                positions[servo_id] = f.position
            else:
                recorder.error(f"ID {servo_id}: cannot set hold target (fault or outside limits)")
        except (ProtocolError, OSError) as exc:
            recorder.error(f"ID {servo_id}: hold unavailable: {exc}")
    if positions:
        bus.move(positions, speed, acceleration)
        recorder.command(positions, "hold")
    return len(positions) == len(limits)


def run(bus, recorder, args, limits, stop):
    rng = random.Random(args.seed)
    positions = {}
    movement_started = False
    outcome = "completed"
    exit_code = 0
    try:
        # Validate all six before sending ANY command that can move a motor.
        for i, (lo, hi) in limits.items():
            if stop[0]:
                outcome = "interrupted"
                return "interrupted", 0
            mode, status = bus.read(i, 33, 1)
            if status or mode != b"\x00":
                raise ProtocolError(f"ID {i}: expected position mode 0, got {list(mode)}, status={status}")
            f = bus.feedback(i)
            recorder.sample(f, "preflight")
            if f.status:
                raise ProtocolError(feedback_error(f))
            if not lo <= f.position <= hi:
                raise ProtocolError(f"ID {i}: initial position {f.position} outside {lo}..{hi}; "
                                    "set the initial pose or correct the limits first")
            positions[i] = f.position
        if stop[0]:
            outcome = "interrupted"
            return "interrupted", 0
        # Overwrite stale targets with measured positions before enabling torque.
        movement_started = True
        bus.move(positions, args.speed, args.acceleration)
        recorder.command(positions, "initial_hold")
        bus.sync_write(40, {i: b"\x01" for i in limits})
        active_start = time.monotonic()
        next_motion = active_start
        next_report = active_start
        while not stop[0] and time.monotonic() - active_start < args.duration:
            cycle_start = time.monotonic()
            if cycle_start >= next_motion:
                previous = positions
                positions = next_targets(rng, previous, limits, args.max_step, args.min_separation)
                bus.move(positions, args.speed, args.acceleration)
                recorder.command(positions, "random_move")
                travel_estimate = max(abs(positions[i] - previous[i]) for i in positions) / args.speed
                next_motion = cycle_start + max(args.move_interval, travel_estimate + .25)
            for i in limits:
                f = bus.feedback(i)
                recorder.sample(f, "active", positions[i])
                # Keep a valid overcurrent/fault sample BEFORE ending motion.
                if f.status:
                    raise ProtocolError(feedback_error(f) + "; test stopped")
                if not limits[i][0] <= f.position <= limits[i][1]:
                    raise ProtocolError(f"ID {i}: position {f.position} outside configured range")
            recorder.file.flush()
            if time.monotonic() >= next_report:
                recorder.save("running", bus)
                recorder.display()
                next_report = time.monotonic() + 1
            pause = 1 / args.hz - (time.monotonic() - cycle_start)
            if pause > 0:
                time.sleep(min(pause, max(0, args.duration - (time.monotonic() - active_start))))
        if stop[0]:
            outcome = "interrupted"
    except (ProtocolError, OSError) as exc:
        recorder.error(str(exc))
        outcome, exit_code = "error", 1
    finally:
        if movement_started:
            try:
                if not hold_current_positions(bus, recorder, limits, args.speed, args.acceleration):
                    outcome, exit_code = "error", 1
            except (ProtocolError, OSError) as exc:
                recorder.error(f"Could not send hold command: {exc}")
                outcome, exit_code = "error", 1
        recorder.save(outcome, bus)
    return outcome, exit_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="STS3215: случайные движения и максимумы тока по ID")
    parser.add_argument("--port", required=True, help="USB-порт Waveshare: /dev/ttyUSB0 или COM3")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--ids", type=int, nargs="+", default=list(DEFAULT_LIMITS))
    parser.add_argument("--duration", type=float, default=60, help="Длительность движения, с (60)")
    parser.add_argument("--hz", type=float, default=50, help="Целевая частота опроса каждого ID (50)")
    parser.add_argument("--timeout", type=float, default=0.03, help="Таймаут ответа одного привода, с")
    parser.add_argument("--move-interval", type=float, default=1, help="Период новых случайных целей, с")
    parser.add_argument("--max-step", type=int, default=4095, help="Максимальный шаг цели, отсчёты (4095)")
    parser.add_argument("--min-separation", type=float, default=.35,
                        help="Минимальное расстояние между целями как доля диапазона, 0..0.5 (0.35)")
    parser.add_argument("--speed", type=int, default=1000, help="Скорость привода, отсчёты/с")
    parser.add_argument("--acceleration", type=int, default=50, help="Значение регистра ускорения 1..254")
    parser.add_argument("--limits", default=str(Path(__file__).with_name("limits.json")),
                        help="JSON с диапазонами (по умолчанию limits.json рядом со скриптом)")
    parser.add_argument("--seed", type=int, help="Начальное число генератора для повторяемого теста")
    parser.add_argument("--output", help="Новая папка результатов (не должна существовать)")
    args = parser.parse_args(argv)
    if len(set(args.ids)) != len(args.ids) or any(not 0 <= i <= 253 for i in args.ids):
        parser.error("IDs must be unique and within 0..253")
    if len(args.ids) > 29:
        parser.error("At most 29 IDs fit in one sync-write packet")
    for key in ("duration", "hz", "timeout", "move_interval"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and > 0")
    if not 1 <= args.speed <= 4095 or not 1 <= args.acceleration <= 254:
        parser.error("speed must be 1..4095, acceleration 1..254")
    if not 1 <= args.max_step <= 4095 or args.baud <= 0:
        parser.error("max-step must be 1..4095; baud must be > 0")
    if not math.isfinite(args.min_separation) or not 0 <= args.min_separation <= .5:
        parser.error("min-separation must be within 0..0.5")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        limits = load_limits(args.limits, args.ids)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Ошибка диапазонов: {exc}", file=sys.stderr)
        return 2
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    folder = Path(args.output) if args.output else Path("results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        import serial
    except ImportError:
        print("Установите pyserial: python3 -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    try:
        recorder = Recorder(folder, args.ids, dict(vars(args), resolved_limits=limits))
    except OSError as exc:
        print(f"Не удалось создать папку результатов: {exc}", file=sys.stderr)
        return 2
    stop = [False]
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    try:
        print(f"Результаты: {folder.resolve()}\nДиапазоны: {limits}\n"
              f"Случайные движения: {args.duration:g} с; остановка — Ctrl+C", flush=True)
        kwargs = dict(port=args.port, baudrate=args.baud, timeout=min(0.002, args.timeout),
                      write_timeout=0.2)
        if sys.platform != "win32":
            kwargs["exclusive"] = True
        with serial.Serial(**kwargs) as port:
            outcome, code = run(ServoBus(port, args.timeout), recorder, args, limits, stop)
        print(f"Завершение: {outcome}")
        recorder.display()
        for i, stats in recorder.rows.items():
            span = (stats["last_elapsed_s"] or 0) - (stats["first_elapsed_s"] or 0)
            hz = (stats["samples"] - 1) / span if span > 0 else 0
            print(f"ID {i}: {stats['samples']} измерений, средняя частота {hz:.1f} Гц")
        return code
    except (OSError, ProtocolError) as exc:
        recorder.error(str(exc))
        recorder.save("error")
        return 1
    finally:
        recorder.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
