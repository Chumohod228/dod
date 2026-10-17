#!/usr/bin/env python3
"""STS3215 teleoperation on one bus: leader 7..12 -> follower 1..6."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

from servo_current_test import (ServoBus, ProtocolError, Recorder, feedback_error,
                               hold_current_positions, load_limits)

PAIRS = {leader: leader - 6 for leader in range(7, 13)}
OVERLOAD = 0x20


def load_leader_ranges(path):
    """Accept analyze_ranges.py output, or explicit unwrapped {ID: [min, max]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "servos" in data:
        for i in PAIRS:
            s = data["servos"].get(str(i), {})
            if s.get("ambiguous_steps", 0):
                raise ValueError(f"ID {i}: ambiguous zero crossing; repeat manual recording")
        data = {i: [s["minimum"], s["maximum"]] for i, s in data["servos"].items()}
    ranges = {}
    for i in PAIRS:
        pair = data.get(str(i), ())
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) is not int for v in pair) or not 1 <= pair[1] - pair[0] < 4096):
            raise ValueError(f"ID {i}: expected continuous [min, max], span 1..4095")
        ranges[i] = tuple(pair)
    return ranges


def unwrap_in_range(raw, limits, tolerance=16):
    """Select the branch nearest the middle of a recorded interval (< one turn)."""
    if not 0 <= raw <= 4095:
        raise ProtocolError(f"Unexpected single-turn position: {raw}")
    lo, hi = limits
    center = (lo + hi) / 2
    position = raw + 4096 * math.floor((center - raw + 2048) / 4096)
    if not lo - tolerance <= position <= hi + tolerance:
        raise ProtocolError(f"Leader position {position} outside recorded range {lo}..{hi}")
    return position


def map_position(raw, leader_limits, follower_limits, inverted=False):
    position = unwrap_in_range(raw, leader_limits)
    lo, hi = leader_limits
    fraction = max(0.0, min(1.0, (position - lo) / (hi - lo)))
    if inverted:
        fraction = 1 - fraction
    low, high = follower_limits
    return low + fraction * (high - low)


def approach_targets(current, desired, speed, dt):
    limit = speed * dt
    return {i: value + max(-limit, min(limit, desired[i] - value)) for i, value in current.items()}


def read_register(bus, i, address, size, allowed_status=0):
    data, status = bus.read(i, address, size)
    if status & ~allowed_status:
        raise ProtocolError(f"ID {i}: status 0x{status:02X} reading register {address}")
    return data


def read_feedback(bus, recorder, ids, phase, allowed_status=0):
    result = {}
    for i in ids:
        f = bus.feedback(i)
        recorder.sample(f, phase)
        if f.status & ~allowed_status:
            raise ProtocolError(feedback_error(f))
        result[i] = f
    return result


def verify_configuration(bus, recorder, leader_ranges, follower_ranges):
    """No writes until every motor and range has been checked."""
    for i in (*PAIRS.values(), *PAIRS):
        allowed = OVERLOAD if i in follower_ranges else 0
        if int.from_bytes(read_register(bus, i, 3, 2, allowed), "little") != 777:
            raise ProtocolError(f"ID {i}: expected STS3215 model 777")
        if read_register(bus, i, 33, 1, allowed) != b"\x00":
            raise ProtocolError(f"ID {i}: expected position mode 0")
        if read_register(bus, i, 18, 1, allowed)[0] & 0x10:
            raise ProtocolError(f"ID {i}: expected position feedback in 0..4095")
        if i in PAIRS and read_register(bus, i, 40, 1) != b"\x00":
            raise ProtocolError(f"Leader ID {i}: torque enabled; switch it off before manual teleoperation")
        if i in follower_ranges:
            data = read_register(bus, i, 9, 4, allowed)
            hw_low, hw_high = int.from_bytes(data[:2], "little"), int.from_bytes(data[2:], "little")
            lo, hi = follower_ranges[i]
            if not hw_low <= lo < hi <= hw_high:
                raise ProtocolError(f"ID {i}: configured range {lo}..{hi} outside hardware limits {hw_low}..{hw_high}")
    follower = read_feedback(bus, recorder, PAIRS.values(), "follower_preflight", OVERLOAD)
    leader = read_feedback(bus, recorder, PAIRS, "leader_preflight")
    for i, f in follower.items():
        lo, hi = follower_ranges[i]
        if not lo <= f.position <= hi:
            raise ProtocolError(f"Follower ID {i}: initial position {f.position} outside {lo}..{hi}")
    for i, f in leader.items():
        unwrap_in_range(f.position, leader_ranges[i])
    return follower, leader


def select_commands(targets, previous_sent, feedback, last_sent, now, deadband, retry_interval):
    commands = {}
    for i, target in targets.items():
        retry_due = now - last_sent[i] >= retry_interval
        overloaded = bool(feedback[i].status & OVERLOAD)
        if overloaded:
            # Do not hammer a protected motor at the main loop's 50 Hz.
            if retry_due:
                commands[i] = target
        elif (abs(target - previous_sent[i]) >= deadband
              or (retry_due and abs(target - feedback[i].position) >= deadband)):
            commands[i] = target
    return commands


def run(bus, recorder, args, leader_ranges, follower_ranges, stop):
    active = False
    outcome, code = "completed", 0
    try:
        follower, leader = verify_configuration(bus, recorder, leader_ranges, follower_ranges)
        if args.check_only:
            print("Проверка пройдена: все 12 приводов отвечают, диапазоны согласованы.", flush=True)
            return "checked", 0
        targets = {i: float(f.position) for i, f in follower.items()}
        previous_leader = {i: f.position for i, f in leader.items()}
        previous_sent = {i: round(v) for i, v in targets.items()}
        if stop[0]:
            outcome = "interrupted"
            return outcome, 0
        # Goal writes can enable torque on this firmware. Address ONLY follower IDs.
        active = True
        bus.move(previous_sent, args.speed, args.acceleration)
        bus.sync_write(40, {i: b"\x01" for i in PAIRS.values()})
        recorder.command(previous_sent, "initial_hold")
        print("Следование включено: 7→1, 8→2, 9→3, 10→4, 11→5, 12→6. Ctrl+C — остановка.", flush=True)
        started = previous_time = time.monotonic()
        last_sent = {i: started for i in targets}
        overloaded_ids = set()
        next_report = started
        while not stop[0] and (args.duration == 0 or time.monotonic() - started < args.duration):
            cycle = time.monotonic()
            leader = read_feedback(bus, recorder, PAIRS, "leader")
            follower = read_feedback(bus, recorder, PAIRS.values(), "follower", OVERLOAD)
            # One serial owner; never use old leader data after a stalled cycle.
            now = time.monotonic()
            if now - cycle > args.max_read_age or now - previous_time > args.max_read_age:
                raise ProtocolError("Telemetry too old; following stopped")
            overloaded = {i for i, f in follower.items() if f.status & OVERLOAD}
            for i in overloaded - overloaded_ids:
                print(f"ID {i}: перегрузка 0x20; повтор цели каждые {args.retry_interval:g} с", flush=True)
            for i in overloaded_ids - overloaded:
                print(f"ID {i}: статус перегрузки снят", flush=True)
            overloaded_ids = overloaded
            for i, f in follower.items():
                lo, hi = follower_ranges[i]
                if not lo - 8 <= f.position <= hi + 8:
                    raise ProtocolError(f"Follower ID {i}: position {f.position} outside {lo}..{hi}")
            desired = {}
            for leader_id, follower_id in PAIRS.items():
                raw = leader[leader_id].position
                delta = (raw - previous_leader[leader_id] + 2048) % 4096 - 2048
                if abs(delta) > args.max_leader_step:
                    raise ProtocolError(f"Leader ID {leader_id}: implausible position jump {delta}")
                desired[follower_id] = map_position(raw, leader_ranges[leader_id],
                                                    follower_ranges[follower_id], follower_id in args.invert)
                previous_leader[leader_id] = raw
            dt = min(now - previous_time, 2 / args.hz)
            targets = approach_targets(targets, desired, args.speed, dt)
            previous_time = now
            rounded = {i: round(v) for i, v in targets.items()}
            changed = select_commands(rounded, previous_sent, follower, last_sent, now,
                                      args.deadband, args.retry_interval)
            # A signal arriving during serial reads must prevent the next movement command.
            if stop[0]:
                break
            if changed:
                bus.move(changed, args.speed, args.acceleration)
                recorder.command(changed, "follow")
                previous_sent.update(changed)
                last_sent.update({i: now for i in changed})
            recorder.file.flush()
            if now >= next_report:
                recorder.save("following", bus)
                print(" | ".join(f"{l}→{f}: {leader[l].position}→{rounded[f]} "
                                 f"U={follower[f].voltage_v:.1f} V I={follower[f].current_ma / 1000:.3f} A "
                                 f"err={rounded[f] - follower[f].position:+d} status=0x{follower[f].status:02X}"
                                 for l, f in PAIRS.items()), flush=True)
                next_report = now + 1
            time.sleep(max(0, 1 / args.hz - (time.monotonic() - cycle)))
        if stop[0]:
            outcome = "interrupted"
    except (ProtocolError, OSError) as exc:
        recorder.error(str(exc))
        outcome, code = "error", 1
    finally:
        if active:
            try:
                if not hold_current_positions(bus, recorder, follower_ranges, args.speed, args.acceleration,
                                              allowed_status=OVERLOAD):
                    outcome, code = "error", 1
            except (ProtocolError, OSError) as exc:
                recorder.error(f"Follower hold unavailable: {exc}")
                outcome, code = "error", 1
        recorder.save("checked" if args.check_only and outcome == "completed" else outcome, bus)
    return outcome, code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Рука-лидер 7–12 → фолловер 1–6, общая шина STS3215")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--leader-ranges", default=str(Path(__file__).with_name("leader_ranges.json")),
                        help="Диапазоны лидера (по умолчанию leader_ranges.json рядом со скриптом)")
    parser.add_argument("--follower-limits", default=str(Path(__file__).with_name("limits.json")))
    parser.add_argument("--invert", type=int, nargs="*", default=[], help="ID фолловера с обратным направлением")
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--speed", type=int, default=600, help="Максимальная скорость цели и привода, отсчёты/с")
    parser.add_argument("--acceleration", type=int, default=50)
    parser.add_argument("--deadband", type=int, default=2, help="Минимальное изменение отправляемой цели, отсчёты")
    parser.add_argument("--retry-interval", type=float, default=.5,
                        help="Период повторения недостигнутой цели/перегрузки, с (не менее 0.1)")
    parser.add_argument("--timeout", type=float, default=.03)
    parser.add_argument("--max-read-age", type=float, default=.2, help="Максимальный возраст данных, с")
    parser.add_argument("--max-leader-step", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=0, help="Секунды, 0 — до Ctrl+C")
    parser.add_argument("--check-only", action="store_true", help="Только чтение и проверка, без команд движения")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    for key in ("hz", "timeout", "max_read_age"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and > 0")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and >= 0")
    if not math.isfinite(args.retry_interval) or args.retry_interval < .1:
        parser.error("retry-interval must be finite and >= 0.1 seconds")
    if not 1 <= args.speed <= 4095 or not 1 <= args.acceleration <= 254:
        parser.error("speed: 1..4095; acceleration: 1..254")
    if not 1 <= args.deadband <= 100 or not 1 <= args.max_leader_step < 2048:
        parser.error("deadband: 1..100; max-leader-step: 1..2047")
    if any(i not in PAIRS.values() for i in args.invert):
        parser.error("invert must contain follower IDs 1..6")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        leader_ranges = load_leader_ranges(args.leader_ranges)
        follower_ranges = load_limits(args.follower_limits, PAIRS.values())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Ошибка калибровки: {exc}", file=sys.stderr)
        return 2
    folder = args.output or Path("results") / ("teleop_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(leader_ranges=leader_ranges, follower_ranges=follower_ranges, pairs=PAIRS)
    try:
        recorder = Recorder(folder, [*PAIRS.values(), *PAIRS], config)
    except OSError as exc:
        print(f"Ошибка записи: {exc}", file=sys.stderr)
        return 2
    stop = [False]
    handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        import serial
        kwargs = dict(port=args.port, baudrate=1_000_000, timeout=min(.002, args.timeout), write_timeout=.2)
        if sys.platform != "win32":
            kwargs["exclusive"] = True
        print(f"Журнал: {folder.resolve()}", flush=True)
        with serial.Serial(**kwargs) as port:
            outcome, code = run(ServoBus(port, args.timeout), recorder, args, leader_ranges, follower_ranges, stop)
        print(f"Завершение: {outcome}")
        recorder.display()
        return code
    except (OSError, ImportError) as exc:
        recorder.error(str(exc))
        recorder.save("error")
        return 1
    finally:
        recorder.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
