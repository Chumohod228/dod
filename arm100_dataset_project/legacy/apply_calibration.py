#!/usr/bin/env python3
"""Apply one recorded calibration with backup, readback and rollback; torque stays off."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import time

import serial
from servo_current_test import ServoBus, ProtocolError, packet

REGISTERS = {"model": (3, 2), "limits": (9, 4), "phase": (18, 1),
             "offset": (31, 2), "mode": (33, 1), "torque": (40, 1),
             "goal": (42, 2), "lock": (55, 1), "position": (56, 2)}


def decode_offset(raw):
    if raw & ~0xFFF:
        raise ValueError("Unexpected offset encoding")
    return -(raw & 0x7FF) if raw & 0x800 else raw


def encode_offset(value):
    if not -2047 <= value <= 2047:
        raise ValueError(f"Offset cannot be encoded: {value}")
    return abs(value) | (0x800 if value < 0 else 0)


def read(bus, i, address, size):
    for attempt in range(3):
        try:
            data, status = bus.read(i, address, size)
            if status:
                raise ProtocolError(f"ID {i}: status={status}")
            return data
        except ProtocolError:
            if attempt == 2:
                raise
            time.sleep(.05)


def write_checked(bus, i, address, data):
    bus.send(packet(i, 3, bytes((address,)) + data))
    time.sleep(.05)
    actual = read(bus, i, address, len(data))
    if actual != data:
        raise ProtocolError(f"ID {i}, register {address}: wrote {data.hex()}, read {actual.hex()}")
    if address != 40 and read(bus, i, 40, 1) != b"\0":
        bus.send(packet(i, 3, b"\x28\x00"))
        time.sleep(.05)
        raise ProtocolError(f"ID {i}: torque unexpectedly enabled after register {address}; disabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = json.loads((args.recording / "continuous_ranges.json").read_text())
    receipt = args.recording / "applied_calibration.json"
    if receipt.exists():
        raise SystemExit("Calibration already applied; refusing to shift twice")
    folder = args.recording / ("calibration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir()
    report = {"outcome": "planning", "before": {}, "plan": {}, "verified": {}}

    def save():
        temp = folder / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp.replace(folder / "report.json")

    with serial.Serial(args.port, 1000000, timeout=.002, write_timeout=.2, exclusive=True) as port:
        bus = ServoBus(port, .05)
        for key, observed in source["servos"].items():
            i = int(key)
            before = {name: read(bus, i, address, size).hex()
                      for name, (address, size) in REGISTERS.items()}
            report["before"][key] = before
            if before["torque"] != "00" or before["mode"] != "00" or before["model"] != "0903":
                raise ProtocolError(f"ID {i}: requires STS3215, position mode, torque off")
            if int(before["phase"], 16) & 0x10:
                raise ProtocolError(f"ID {i}: position feedback is not modulo 4096")
            old = decode_offset(int.from_bytes(bytes.fromhex(before["offset"]), "little"))
            shift = observed["proposed_coordinate_shift"]
            if shift is None:
                raise ValueError(f"ID {i}: no valid recorded range")
            new = (old - shift + 2048) % 4096 - 2048
            raw = encode_offset(new)
            pos = int.from_bytes(bytes.fromhex(before["position"]), "little")
            expected = (pos + shift) % 4096
            limits = observed["proposed_centered_limits"]
            report["plan"][key] = dict(old_offset=old, new_offset=new, offset_raw=raw,
                                       shift=shift, limits=limits, expected_position=expected,
                                       current_inside_limits=limits[0] <= expected <= limits[1])
        (folder / "limits.before.json").write_bytes(Path("limits.json").read_bytes())
        save()  # All settings backed up BEFORE the first write.
        print(json.dumps(report["plan"], indent=2), flush=True)
        if not args.apply:
            print(f"Plan only: {folder}")
            return
        touched = []
        try:
            for key, plan in report["plan"].items():
                i = int(key)
                if read(bus, i, 40, 1) != b"\0":
                    raise ProtocolError(f"ID {i}: torque was enabled externally")
                touched.append(key)
                write_checked(bus, i, 55, b"\0")
                write_checked(bus, i, 31, struct.pack("<H", plan["offset_raw"]))
                pos = int.from_bytes(read(bus, i, 56, 2), "little")
                error = (pos - plan["expected_position"] + 2048) % 4096 - 2048
                if abs(error) > 8:
                    raise ProtocolError(f"ID {i}: offset verification failed, expected {plan['expected_position']}, got {pos}")
                write_checked(bus, i, 9, struct.pack("<HH", *plan["limits"]))
                write_checked(bus, i, 55, b"\1")
                if read(bus, i, 40, 1) != b"\0":
                    raise ProtocolError(f"ID {i}: torque unexpectedly enabled")
                report["verified"][key] = dict(position=pos, offset_error=error)
                save()
                print(f"ID {i}: verified position={pos}, limits={plan['limits']}, torque=0", flush=True)
            limits = {i: p["limits"] for i, p in report["plan"].items()}
            Path("limits.json.tmp").write_text(json.dumps(limits, indent=2) + "\n")
            Path("limits.json.tmp").replace("limits.json")
            report["outcome"] = "applied"
            save()
            receipt.write_text(json.dumps({"report": str((folder / 'report.json').resolve()),
                                           "limits": limits}, indent=2))
        except BaseException as exc:
            report["outcome"] = "rolling_back"
            report["error"] = repr(exc)
            report["rollback_errors"] = []
            save()
            for key in reversed(touched):
                i = int(key)
                try:
                    write_checked(bus, i, 40, b"\0")
                    write_checked(bus, i, 55, b"\0")
                    for name in ("offset", "limits", "lock"):
                        write_checked(bus, i, REGISTERS[name][0], bytes.fromhex(report["before"][key][name]))
                except Exception as rollback_error:
                    report["rollback_errors"].append(f"ID {i}: {rollback_error}")
            report["outcome"] = "rollback_failed" if report["rollback_errors"] else "rolled_back"
            save()
            raise
    print(f"Saved: {folder / 'report.json'}")


if __name__ == "__main__":
    main()
