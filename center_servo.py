#!/usr/bin/env python3
"""Physically center one isolated STS3215, then release torque. No EEPROM writes."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import serial

from servo_current_test import ServoBus, ProtocolError, packet


def main():
    parser = argparse.ArgumentParser(description="Физически повернуть единственный STS3215 в 2048")
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--isolated", action="store_true", required=True)
    args = parser.parse_args()
    if not 1 <= args.id <= 253:
        parser.error("ID must be in 1..253")
    folder = Path("results") / ("center_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True)
    report = {"id": args.id, "port": args.port, "target": 2048, "speed": 150,
              "acceleration": 10, "outcome": "preflight", "samples": []}
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def save():
        temp = folder / "report.json.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(folder / "report.json")

    with serial.Serial(args.port, 1000000, timeout=.002, write_timeout=.2, exclusive=True) as port:
        bus = ServoBus(port, .05)

        def read(address, size):
            data, status = bus.read(args.id, address, size)
            if status:
                raise ProtocolError(f"ID {args.id}: status 0x{status:02X} at register {address}")
            return data

        def feedback():
            f = bus.feedback(args.id)
            report["samples"].append({"monotonic_s": time.monotonic(), **asdict(f)})
            if f.status or not 0 <= f.position <= 4095:
                raise ProtocolError(f"ID {args.id}: invalid feedback {asdict(f)}")
            return f

        active = False
        try:
            registers = ((3, 4), (9, 4), (18, 1), (31, 2), (33, 1), (40, 1), (55, 1))
            before = {str(a): read(a, size).hex() for a, size in registers}
            report["before"] = before
            identity = bytes.fromhex(before["3"])
            if int.from_bytes(identity[:2], "little") != 777 or identity[2] != args.id:
                raise ProtocolError("Expected STS3215 with matching stored ID")
            if before["33"] != "00" or bytes.fromhex(before["18"])[0] & 0x10:
                raise ProtocolError("Expected single-turn position mode")
            if before["40"] != "00":
                raise ProtocolError("Initial torque must be off")
            limits = bytes.fromhex(before["9"])
            if not int.from_bytes(limits[:2], "little") <= 2048 <= int.from_bytes(limits[2:], "little"):
                raise ProtocolError("2048 is outside hardware limits")
            first = feedback()
            report["initial_position"] = first.position
            save()
            if stop[0]:
                raise InterruptedError("Stopped before motion")
            active = True  # Goal writes can enable torque on this firmware.
            bus.move({args.id: 2048}, 150, 10)
            time.sleep(.02)
            if read(42, 2) != (2048).to_bytes(2, "little"):
                raise ProtocolError("Goal position readback failed")
            if stop[0]:
                raise InterruptedError("Stopped during activation")
            bus.send(packet(args.id, 3, bytes((40, 1))))
            time.sleep(.02)
            if read(40, 1) != b"\x01":
                raise ProtocolError("Torque-enable readback failed")
            report["torque_on_verified"] = True
            bus.move({args.id: 2048}, 150, 10)
            report["outcome"] = "moving"
            report["timeout_s"] = max(12, abs(first.position - 2048) / 150 + 5)
            deadline = time.monotonic() + report["timeout_s"]
            stable = 0
            while time.monotonic() < deadline:
                if stop[0]:
                    raise InterruptedError("Centering interrupted")
                f = feedback()
                stable = stable + 1 if abs(f.position - 2048) <= 8 and not f.moving else 0
                if stable >= 3:
                    break
                time.sleep(.05)
            else:
                raise ProtocolError("Centering timed out; torque will be disabled")
            report["outcome"] = "reached"
        except BaseException as exc:
            report["outcome"], report["error"] = "error", repr(exc)
            raise
        finally:
            try:
                if active:
                    bus.sync_write(40, {args.id: b"\x00"})
                    time.sleep(.1)
                    if read(40, 1) != b"\x00":
                        raise ProtocolError("Torque-off verification failed")
                    report["torque_off_verified"] = True
            except BaseException as exc:
                report["outcome"], report["shutdown_error"] = "error", repr(exc)
                raise
            finally:
                save()
        final = feedback()
        after = {str(a): read(a, size).hex() for a, size in registers}
        report["after"] = after
        if after != before:
            report["outcome"] = "error"
            save()
            raise ProtocolError("Configuration changed during centering")
        if abs(final.position - 2048) > 12:
            report["outcome"] = "error"
            save()
            raise ProtocolError(f"Position shifted after torque-off: {final.position}")
        report["outcome"] = "verified"
        report["final_position"] = final.position
        save()
    print(f"ID {args.id}: position={final.position}, torque off, EEPROM unchanged. {folder / 'report.json'}")


if __name__ == "__main__":
    main()
