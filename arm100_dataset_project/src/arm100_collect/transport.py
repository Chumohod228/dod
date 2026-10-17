"""STS3215 packet transport and feedback decoding; no standalone motion program."""
from __future__ import annotations
from dataclasses import dataclass
import struct
import time

CURRENT_MA_PER_TICK = 6.5


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

