#!/usr/bin/env python3
"""Rename a physically isolated STS3215 and verify its new ID. No motion commands."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
from servo_current_test import ServoBus, ProtocolError, packet
from apply_calibration import read, write_checked


def main():
    parser = argparse.ArgumentParser(description="Смена ID одного физически изолированного STS3215")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--old", type=int, default=1)
    parser.add_argument("--new", type=int, required=True)
    parser.add_argument("--isolated", action="store_true", required=True,
                        help="Подтверждение: подключён только один физический привод")
    args = parser.parse_args()
    if not 0 <= args.old <= 253 or not 0 <= args.new <= 253 or args.old == args.new:
        parser.error("IDs must differ and be within 0..253")
    folder = Path("results") / ("rename_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True)
    report = dict(old_id=args.old, new_id=args.new, outcome="preflight")

    def save():
        temp = folder / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(folder / "report.json")

    with serial.Serial(args.port, 1000000, timeout=.002, write_timeout=.2, exclusive=True) as port:
        bus = ServoBus(port, .1)
        try:
            identity = read(bus, args.old, 3, 4)
            if int.from_bytes(identity[:2], "little") != 777 or identity[2] != args.old:
                raise ProtocolError("Unexpected model or stored ID")
            if read(bus, args.old, 40, 1) != b"\0":
                raise ProtocolError("Torque must be off before changing ID")
            try:
                bus.read(args.new, 5, 1)
            except ProtocolError as exc:
                if "timeout" not in str(exc):
                    raise
            else:
                raise ProtocolError("Destination ID already responds")
            report["before"] = {str(addr): read(bus, args.old, addr, size).hex()
                                for addr, size in ((3, 4), (9, 4), (31, 2), (33, 1), (40, 1), (55, 1))}
            save()
            try:
                write_checked(bus, args.old, 55, b"\0")
                report["outcome"] = "changing_id"
                save()
                bus.send(packet(args.old, 3, bytes((5, args.new))))
                time.sleep(.1)
                identity = read(bus, args.new, 3, 4)
                if identity[2] != args.new or int.from_bytes(identity[:2], "little") != 777:
                    raise ProtocolError("New ID verification failed")
                write_checked(bus, args.new, 55, b"\1")
                time.sleep(.1)
                if read(bus, args.new, 5, 1) != bytes((args.new,)):
                    raise ProtocolError("ID did not persist after locking")
                if read(bus, args.new, 40, 1) != b"\0":
                    raise ProtocolError("Unexpected torque state")
                try:
                    bus.read(args.old, 5, 1)
                except ProtocolError as exc:
                    if "timeout" not in str(exc):
                        raise
                else:
                    raise ProtocolError("Old ID still responds; more than one servo may be connected")
                report["outcome"] = "verified"
            except BaseException:
                # A failed response can leave the ID changed. Try to relock both addresses.
                for address in (args.new, args.old):
                    try:
                        write_checked(bus, address, 55, b"\1")
                    except Exception:
                        pass
                raise
        except BaseException as exc:
            report["error"] = repr(exc)
            report["outcome"] = "error"
            raise
        finally:
            save()
    print(f"ID {args.old} -> {args.new}: verified; EEPROM locked, torque off. {folder / 'report.json'}")


if __name__ == "__main__":
    main()
