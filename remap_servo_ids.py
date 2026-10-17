#!/usr/bin/env python3
"""Rename unique STS3215 IDs on an assembled bus, without movement commands."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial

from servo_current_test import ServoBus, ProtocolError, packet
from apply_calibration import read, write_checked


def scan(bus):
    found = {}
    for i in range(254):
        try:
            data, status = bus.read(i, 3, 4)
        except ProtocolError as exc:
            if "timeout" not in str(exc):
                raise
            continue
        if status or int.from_bytes(data[:2], "little") != 777 or data[2] != i:
            raise ProtocolError(f"ID {i}: unexpected identity/status")
        found[i] = data.hex()
    return found


def snapshot(bus, servo_id):
    return {"eeprom": read(bus, servo_id, 0, 40).hex(),
            "control": read(bus, servo_id, 40, 16).hex()}


def main():
    parser = argparse.ArgumentParser(description="Смена непересекающихся ID без движения")
    parser.add_argument("--port", required=True)
    parser.add_argument("--old-ids", type=int, nargs="+", required=True)
    parser.add_argument("--new-ids", type=int, nargs="+", required=True)
    parser.add_argument("--apply", action="store_true", help="Разрешить запись ID; по умолчанию только проверка")
    args = parser.parse_args()
    old, new = args.old_ids, args.new_ids
    if (len(old) != len(new) or len(set(old)) != len(old) or len(set(new)) != len(new)
            or set(old) & set(new) or any(not 1 <= i <= 253 for i in old + new)):
        parser.error("Source/destination IDs must be unique, disjoint, and in 1..253")
    folder = Path("results") / ("remap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True)
    report = {"port": args.port, "mapping": dict(zip(old, new)), "outcome": "preflight", "changes": []}

    def save():
        temp = folder / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(folder / "report.json")

    with serial.Serial(args.port, 1000000, timeout=.002, write_timeout=.2, exclusive=True) as port:
        bus = ServoBus(port, .02)
        try:
            found = scan(bus)
            if not set(old) <= found.keys() or set(new) & found.keys():
                raise ProtocolError("Source IDs missing or destination IDs occupied")
            before = {i: snapshot(bus, i) for i in found}
            report["before"] = before
            save()
            for i in old:
                if bytes.fromhex(before[i]["control"])[0] != 0:
                    raise ProtocolError(f"ID {i}: torque must be off; no writes sent")
            if not args.apply:
                report["outcome"] = "checked"
                print(f"Read-only check passed: {report['mapping']}. {folder / 'report.json'}")
                return
            for src, dst in zip(old, new):
                change = {"old": src, "new": dst, "outcome": "changing"}
                report["changes"].append(change)
                save()
                try:
                    if snapshot(bus, src) != before[src]:
                        raise ProtocolError(f"ID {src}: configuration changed since preflight")
                    write_checked(bus, src, 55, b"\x00")
                    bus.send(packet(src, 3, bytes((5, dst))))
                    time.sleep(.1)
                    identity = read(bus, dst, 3, 4)
                    if int.from_bytes(identity[:2], "little") != 777 or identity[2] != dst:
                        raise ProtocolError("New ID verification failed")
                    write_checked(bus, dst, 55, b"\x01")
                    actual = snapshot(bus, dst)
                    eeprom = bytearray.fromhex(before[src]["eeprom"])
                    eeprom[5] = dst
                    control = bytearray.fromhex(before[src]["control"])
                    control[15] = 1
                    expected = {"eeprom": eeprom.hex(), "control": control.hex()}
                    if actual != expected:
                        raise ProtocolError(f"ID {dst}: unexpected register changes")
                    change["after"] = actual
                    change["outcome"] = "verified"
                    save()
                    print(f"ID {src} -> {dst}: verified, torque off, EEPROM locked", flush=True)
                except BaseException:
                    for i in (dst, src):
                        try:
                            write_checked(bus, i, 55, b"\x01")
                        except Exception:
                            pass
                    change["outcome"] = "error"
                    raise
            after_ids = scan(bus)
            if set(after_ids) != (set(found) - set(old)) | set(new):
                raise ProtocolError("Unexpected IDs after remapping")
            for i in set(found) - set(old):
                if snapshot(bus, i) != before[i]:
                    raise ProtocolError(f"Unrelated ID {i}: configuration mismatch")
            report["after_ids"] = sorted(after_ids)
            report["outcome"] = "verified"
        except BaseException as exc:
            report["outcome"], report["error"] = "error", repr(exc)
            raise
        finally:
            save()
    print(f"Remapping complete; no position commands sent. {folder / 'report.json'}")


if __name__ == "__main__":
    main()
