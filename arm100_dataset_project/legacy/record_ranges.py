#!/usr/bin/env python3
"""Record observed joint extrema during manual movement; sends READ packets only."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import serial
from servo_current_test import ServoBus, ProtocolError


def main():
    parser = argparse.ArgumentParser(description="Запись границ при ручном движении суставов")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--ids", type=int, nargs="+", default=list(range(1, 7)),
                        help="ID для записи; для лидера: 7 8 9 10 11 12")
    parser.add_argument("--output", type=Path, help="Новая папка результатов")
    args = parser.parse_args()
    if len(set(args.ids)) != len(args.ids) or any(not 0 <= i <= 253 for i in args.ids):
        parser.error("IDs must be unique and within 0..253")
    folder = args.output or Path("results") / ("manual_ranges_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True)
    state = {i: {"min": None, "max": None, "current": None, "samples": 0, "errors": 0} for i in args.ids}
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    start = time.monotonic()

    def save(outcome):
        record = {"outcome": outcome, "elapsed_s": time.monotonic() - start,
                  "note": "Observed manual extrema, not verified collision-free limits", "servos": state}
        for name, data in (("ranges.json", record),
                           ("limits_observed.json", {i: [s["min"], s["max"]] for i, s in state.items()})):
            temp = folder / (name + ".tmp")
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(folder / name)

    outcome = "error"
    try:
        with serial.Serial(args.port, 1000000, timeout=.002, write_timeout=.2, exclusive=True) as port, \
                (folder / "positions.csv").open("w", newline="", encoding="utf-8") as stream:
            bus = ServoBus(port)
            for i in state:
                torque, status = bus.read(i, 40, 1)
                if status or torque != b"\x00":
                    raise ProtocolError(f"ID {i}: torque enabled or error; manual recording not started")
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "servo_id", "position", "status", "error"])
            print(f"RECORDING {folder.resolve()}", flush=True)
            report = 0
            while not stop[0]:
                cycle = time.monotonic()
                for i, s in state.items():
                    try:
                        raw, status = bus.read(i, 56, 2)
                        position = int.from_bytes(raw, "little")
                        if status or not 0 <= position <= 4095:
                            raise ProtocolError(f"status={status}, position={position}")
                        s["current"] = position
                        s["min"] = position if s["min"] is None else min(s["min"], position)
                        s["max"] = position if s["max"] is None else max(s["max"], position)
                        s["samples"] += 1
                        writer.writerow([time.monotonic() - start, i, position, status, ""])
                    except ProtocolError as exc:
                        s["errors"] += 1
                        writer.writerow([time.monotonic() - start, i, "", "", str(exc)])
                if time.monotonic() >= report:
                    stream.flush()
                    save("recording")
                    print(" | ".join(f"ID {i}: {s['min']}..{s['max']} now={s['current']} err={s['errors']}" for i, s in state.items()), flush=True)
                    report = time.monotonic() + 2
                time.sleep(max(0, .02 - (time.monotonic() - cycle)))
            outcome = "stopped"
    finally:
        save(outcome)
        print(f"SAVED {folder.resolve()}", flush=True)


if __name__ == "__main__":
    main()
