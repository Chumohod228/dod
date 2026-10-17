#!/usr/bin/env python3
"""Unwrap recorded 12-bit joint positions across zero; no hardware writes."""
import argparse
import csv
import json
from pathlib import Path


def analyze(rows):
    states = {}
    for row in rows:
        if row.get("error") or not row.get("position") or int(row.get("status") or 0):
            continue
        servo_id = int(row["servo_id"])
        raw = int(row["position"])
        if not 0 <= raw <= 4095:
            continue
        if servo_id not in states:
            states[servo_id] = dict(previous_raw=raw, continuous=raw, minimum=raw,
                                    maximum=raw, wraps=0, samples=1, ambiguous_steps=0)
            continue
        s = states[servo_id]
        delta = raw - s["previous_raw"]
        if abs(delta) == 2048:
            s["ambiguous_steps"] += 1
        if delta > 2048:
            delta -= 4096
            s["wraps"] += 1
        elif delta < -2048:
            delta += 4096
            s["wraps"] += 1
        s["continuous"] += delta
        s["previous_raw"] = raw
        s["minimum"] = min(s["minimum"], s["continuous"])
        s["maximum"] = max(s["maximum"], s["continuous"])
        s["samples"] += 1
    for s in states.values():
        width = s["maximum"] - s["minimum"]
        s["span_ticks"] = width
        s["span_degrees"] = width * 360 / 4096
        # Proposed coordinate transformation, NOT a write to the servo offset register.
        if 0 < width <= 4095 and not s["ambiguous_steps"]:
            low = (4095 - width) // 2
            s["proposed_centered_limits"] = [low, low + width]
            s["proposed_coordinate_shift"] = low - s["minimum"]
        else:
            s["proposed_centered_limits"] = None
            s["proposed_coordinate_shift"] = None
    return states


def main():
    parser = argparse.ArgumentParser(description="Восстановить диапазоны суставов с учётом перехода через 0")
    parser.add_argument("folder", type=Path, help="Папка записи record_ranges.py")
    args = parser.parse_args()
    with (args.folder / "positions.csv").open(newline="", encoding="utf-8") as stream:
        states = analyze(csv.DictReader(stream))
    output = args.folder / "continuous_ranges.json"
    result = {"note": "Observed manual ranges. Coordinate shifts are proposals, not applied to hardware.",
              "assumption": "Each joint moves less than 2048 ticks between successive valid samples.",
              "servos": states}
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output)
    for i, s in states.items():
        print(f"ID {i}: {s['minimum']}..{s['maximum']}, span={s['span_degrees']:.1f}°, "
              f"zero crossings={s['wraps']}, proposed limits={s['proposed_centered_limits']}")
    print(output.resolve())


if __name__ == "__main__":
    main()
