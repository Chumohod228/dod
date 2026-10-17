"""Local collection tools; no network uploads and no automatic motor renumbering."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import select
import signal
import sys
import time

from .config import ARM_NAMES, PROJECT, load_config, resolve_path, number, validate_follower_arc
from .transport import ProtocolError


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def stop_flag():
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    return stop


def doctor(args):
    from serial.tools import list_ports
    import pyrealsense2 as rs
    import torch
    result = {"python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "packages": {}, "serial_ports": [], "realsense": []}
    try:
        import torchcodec
        result["torchcodec"] = torchcodec.__version__
    except (ImportError, RuntimeError, OSError) as exc:
        result["torchcodec"] = {"optional_decoder_error": str(exc), "recording_decoder": "pyav"}
    for name in ("lerobot", "pyrealsense2", "pyserial", "av", "datasets", "accelerate", "feetech-servo-sdk"):
        result["packages"][name] = importlib.metadata.version(name)
    for port in list_ports.comports():
        if port.vid is None and not port.device.startswith(("/dev/ttyACM", "/dev/ttyUSB")):
            continue
        aliases = [str(p) for p in Path("/dev/serial/by-id").glob("*") if str(p.resolve()) == port.device]
        result["serial_ports"].append({"device": port.device, "description": port.description, "serial": port.serial_number, "persistent_paths": aliases})
    for dev in rs.context().query_devices():
        result["realsense"].append({"name": dev.get_info(rs.camera_info.name), "serial": dev.get_info(rs.camera_info.serial_number)})
    cfg = load_config(args.config)
    result["calibration_files"] = {name: resolve_path(cfg, arm["calibration"]).exists() for name, arm in cfg["arms"].items()}
    result["configured_ports_present"] = {name: Path(arm["port"]).exists() for name, arm in cfg["arms"].items()}
    result["recording_prerequisites_present"] = (all(result["calibration_files"].values()) and all(result["configured_ports_present"].values())
                                and all(cam["serial"] in {d["serial"] for d in result["realsense"]} for cam in cfg["cameras"].values()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_json(PROJECT / "logs/doctor.json", result)


def calibrate(args):
    from contextlib import ExitStack
    from .hardware import checked_read, open_bus, snapshot
    cfg = load_config(args.config)
    arm = cfg["arms"][args.arm]
    output = resolve_path(cfg, arm["calibration"])
    stop = stop_flag()
    states = {i: {"previous": None, "continuous": None, "minimum": None, "maximum": None, "samples": 0} for i in arm["ids"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = PROJECT / "logs" / f"calibration_{args.arm}_{stamp}"
    folder.mkdir(parents=True)
    started = time.monotonic()
    with ExitStack() as stack:
        bus = open_bus(stack, arm["port"])
        hardware = snapshot(bus, arm["ids"])
        for i in arm["ids"]:
            if checked_read(bus, i, 40, 1) != b"\x00":
                raise ProtocolError(f"ID {i}: torque is on; manual capture sends READ packets only")
        write_json(folder / "hardware_snapshot.json", hardware)
        with (folder / "positions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "id", "raw", "continuous"])
            next_report = started
            print(f"{args.arm}: двигайте все шесть суставов по разрешённым границам. Ctrl+C — сохранить.", flush=True)
            try:
                while not stop[0]:
                    cycle = time.monotonic()
                    for i, state in states.items():
                        raw = int.from_bytes(checked_read(bus, i, 56, 2), "little")
                        if not 0 <= raw < 4096:
                            raise ProtocolError(f"ID {i}: invalid position {raw}")
                        if state["previous"] is None:
                            continuous = raw
                        else:
                            delta = (raw - state["previous"] + 2048) % 4096 - 2048
                            if abs(delta) > cfg["max_leader_step_ticks"]:
                                raise ProtocolError(f"ID {i}: ambiguous/large step {delta}; repeat calibration slowly")
                            continuous = state["continuous"] + delta
                        state["previous"], state["continuous"] = raw, continuous
                        state["minimum"] = continuous if state["minimum"] is None else min(state["minimum"], continuous)
                        state["maximum"] = continuous if state["maximum"] is None else max(state["maximum"], continuous)
                        state["samples"] += 1
                        writer.writerow([time.monotonic() - started, i, raw, continuous])
                    if time.monotonic() >= next_report:
                        stream.flush()
                        write_json(folder / "observed.json", states)
                        print(" | ".join(f"ID {i}: {s['minimum']}..{s['maximum']}" for i, s in states.items()), flush=True)
                        next_report = time.monotonic() + 1
                    time.sleep(max(0, 0.02 - (time.monotonic() - cycle)))
            finally:
                write_json(folder / "observed.json", states)
        for i, state in states.items():
            if state["samples"] < 30 or not 32 <= state["maximum"] - state["minimum"] < 4096:
                raise ValueError(f"ID {i}: insufficient movement or ambiguous full-turn range; previous calibration retained")
            if args.arm.endswith("follower"):
                validate_follower_arc((state["minimum"], state["maximum"]))
        if hardware != snapshot(bus, arm["ids"]):
            raise ValueError("Motor configuration changed during capture; previous calibration retained")
    data = {"schema": 1, "arm": args.arm, "ids": arm["ids"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "ranges": {i: [s["minimum"], s["maximum"]] for i, s in states.items()}, "hardware_snapshot": hardware,
            "source": str(folder.relative_to(PROJECT)), "note": "Manual continuous arcs; EEPROM unchanged"}
    if output.exists():
        write_json(folder / "previous_calibration.json", json.loads(output.read_text()))
    write_json(output, data)
    print(f"Сохранено: {output}")


def inspect_port(args):
    from contextlib import ExitStack
    from .hardware import open_bus
    if not args.ids or len(set(args.ids)) != len(args.ids) or any(not 1 <= i <= 253 for i in args.ids):
        raise ValueError("IDs must be unique and in 1..253")
    with ExitStack() as stack:
        bus = open_bus(stack, args.port)
        for i in args.ids:
            try:
                data, status = bus.read(i, 3, 2)
                print(f"ID {i}: model={int.from_bytes(data, 'little')}, status=0x{status:02X}")
            except ProtocolError as exc:
                print(f"ID {i}: {exc}")


def torque_off(args):
    from contextlib import ExitStack
    from .hardware import checked_read, open_bus, snapshot, OVERLOAD
    cfg = load_config(args.config)
    arm = cfg["arms"][args.arm]
    allowed = OVERLOAD if args.arm.endswith("follower") else 0
    with ExitStack() as stack:
        bus = open_bus(stack, arm["port"])
        snapshot(bus, arm["ids"], allowed)
        bus.sync_write(40, {i: b"\x00" for i in arm["ids"]})
        for i in arm["ids"]:
            if checked_read(bus, i, 40, 1, allowed) != b"\x00":
                raise ProtocolError(f"ID {i}: torque-off readback failed")
    print(f"{args.arm}: torque выключен для ID {arm['ids']}")


def import_ranges(args):
    from contextlib import ExitStack
    from .config import raw_segments, unwrap
    from .hardware import open_bus, snapshot
    cfg = load_config(args.config)
    arm = cfg["arms"][args.arm]
    source = json.loads(args.file.read_text())
    if "servos" in source:
        if any(s.get("ambiguous_steps", 0) for s in source["servos"].values()):
            raise ValueError("Source has ambiguous zero crossings")
        source = {i: [s["minimum"], s["maximum"]] for i, s in source["servos"].items()}
    elif "ranges" in source:
        source = source["ranges"]
    ranges = {str(i): source[str(i)] for i in arm["ids"]}
    for i, pair in ranges.items():
        if len(pair) != 2 or any(type(v) is not int for v in pair) or not 32 <= pair[1] - pair[0] < 4096:
            raise ValueError(f"ID {i}: invalid continuous range")
        if args.arm.endswith("follower"):
            validate_follower_arc(pair)
    with ExitStack() as stack:
        bus = open_bus(stack, arm["port"])
        hardware = snapshot(bus, arm["ids"])
        for i in arm["ids"]:
            f = bus.feedback(i)
            if f.status:
                raise ProtocolError(f"ID {i}: status 0x{f.status:02X}")
            unwrap(f.position, ranges[str(i)])
            if args.arm.endswith("follower"):
                bounds = bytes.fromhex(hardware[str(i)]["limits"])
                hwlo, hwhi = int.from_bytes(bounds[:2], "little"), int.from_bytes(bounds[2:], "little")
                if any(not hwlo <= lo <= hi <= hwhi for lo, hi in raw_segments(ranges[str(i)])):
                    raise ValueError(f"ID {i}: imported arc outside EEPROM limits")
    output = resolve_path(cfg, arm["calibration"])
    if output.exists():
        backup = PROJECT / "logs" / ("import_backup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ.json"))
        write_json(backup, json.loads(output.read_text()))
    write_json(output, {"schema": 1, "arm": args.arm, "ids": arm["ids"], "ranges": ranges,
                        "hardware_snapshot": hardware, "source": str(args.file.resolve()),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "note": "Explicitly imported manual ranges; EEPROM unchanged"})
    print(f"Диапазоны импортированы в {output}")


def check(args):
    from .hardware import Cameras, Rig
    cfg = load_config(args.config)
    rig, cameras = Rig(cfg), Cameras(cfg)
    try:
        rig.connect()
        cameras.connect()
        print("Проверка пройдена: 24 привода, четыре адаптера, RGB RealSense. Команды движения не отправлялись.")
    finally:
        try:
            rig.close()
        finally:
            cameras.close()


def simulate(args):
    import numpy as np
    from .dataset import open_dataset, validate_dataset
    cfg = copy.deepcopy(load_config(args.config))
    cfg["fps"] = args.fps
    for cam in cfg["cameras"].values():
        cam.update(width=160, height=120, fps=args.fps)
    if args.frames < 4 or args.episodes < 1 or not 1 <= args.fps <= 60:
        raise ValueError("Simulation requires >=4 frames, >=1 episode and fps in 1..60")
    root = args.root or PROJECT / "datasets" / ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    ds = open_dataset(cfg, "local/synthetic_bimanual", root, simulated=True)
    try:
        for episode in range(args.episodes):
            for k in range(args.frames):
                state = (60 * np.sin(np.arange(12) * 0.4 + k * 0.1 + episode)).astype(np.float32)
                action = (60 * np.sin(np.arange(12) * 0.4 + (k + 1) * 0.1 + episode)).astype(np.float32)
                state[[5, 11]] = state[[5, 11]] / 2 + 50
                action[[5, 11]] = action[[5, 11]] / 2 + 50
                frame = {"observation.state": state, "action": action, "task": "SYNTHETIC TEST ONLY: move both arms"}
                for name in cfg["cameras"]:
                    image = np.zeros((120, 160, 3), dtype=np.uint8)
                    image[:, :, 1] = np.arange(160, dtype=np.uint8)
                    image[k % 100:k % 100 + 20, :, 0] = 200
                    image[:, :, 2] = (episode * 40 + k * 8) % 256
                    frame[f"observation.images.{name}"] = image
                ds.add_frame(frame)
            ds.save_episode(parallel_encoding=False)
    finally:
        ds.finalize()
    print(json.dumps(validate_dataset(root), indent=2))


def input_line():
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return "q" if line == "" else line.strip().lower()
    return None


def record(args):
    from huggingface_hub.utils import validate_repo_id
    from .dataset import open_dataset
    from .hardware import Cameras, Rig
    cfg = load_config(args.config)
    validate_repo_id(args.dataset)
    number(args.episode_seconds, "episode_seconds", 2 / cfg["fps"], 86400)
    number(args.reset_seconds, "reset_seconds", 0, 86400)
    if not args.task.strip() or args.episodes < 1:
        raise ValueError("Task, positive episode count/duration, and nonnegative reset time required")
    if not args.auto_start and not sys.stdin.isatty():
        raise ValueError("Interactive recording needs a terminal; timed recording uses --auto-start")
    root = args.root or PROJECT / "datasets" / args.dataset.split("/")[-1]
    if len(args.dataset.split("/")) != 2 or any(not p or p in (".", "..") for p in args.dataset.split("/")):
        raise ValueError("Dataset ID must have form owner/name (local/name is fine)")
    # Fail before movement if the output already exists or cannot be resumed.
    if not args.resume and Path(root).exists():
        raise FileExistsError(f"{root}: use --resume or a new output directory")
    stop = stop_flag()
    rig, cameras, ds = Rig(cfg), Cameras(cfg), None
    session = PROJECT / "logs" / ("record_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    session.mkdir(parents=True)
    completed, outcome = 0, "error"
    attempts = []
    error = None
    session_start = time.monotonic()
    try:
        rig.connect()
        cameras.connect()
        ds = open_dataset(cfg, args.dataset, root, rig.calibrations, args.resume)
        write_json(session / "config.json", cfg)
        with (session / "telemetry.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["attempt", "episode", "frame", "elapsed_s", "cycle_s", "arm", "id", "raw_position", "voltage_v", "current_a", "temperature_c", "status"])
            while completed < args.episodes and not stop[0]:
                if not args.auto_start:
                    print("r + Enter — начать эпизод; s — сохранить; d — отбросить; q — выход. Ctrl+C — остановка.", flush=True)
                    while not stop[0]:
                        command = input_line()
                        if command == "q":
                            stop[0] = True
                        if command == "r":
                            break
                        time.sleep(0.05)
                    if stop[0]:
                        break
                rig.start(lambda: stop[0])
                episode_index = ds.num_episodes
                attempt = {"attempt": len(attempts), "episode": episode_index, "outcome": "incomplete"}
                attempts.append(attempt)
                frames, late_cycles = 0, 0
                started = deadline = time.monotonic()
                save, quit_requested = False, False
                print(f"Эпизод {episode_index}: запись RGB + 12 суставов, {cfg['fps']} Гц", flush=True)
                while not stop[0]:
                    if time.monotonic() - started >= args.episode_seconds:
                        save = True
                        break
                    if not args.auto_start:
                        command = input_line()
                        if command in ("s", "d", "q"):
                            save = command == "s"
                            quit_requested = command == "q"
                            break
                    cycle = time.monotonic()
                    state, action, images, telemetry = rig.step(cameras.read, lambda: stop[0])
                    if stop[0]:
                        break
                    ds.add_frame({"observation.state": state, "action": action, "task": args.task, **images})
                    elapsed = time.monotonic() - cycle
                    for row in telemetry:
                        writer.writerow([attempt["attempt"], episode_index, frames, cycle - session_start, elapsed, *row])
                    frames += 1
                    attempt["frames"] = frames
                    late_cycles = late_cycles + 1 if elapsed > 1.5 / cfg["fps"] else 0
                    if late_cycles >= 5:
                        raise RuntimeError("Five slow capture cycles; episode discarded. Lower fps or image size in rig.json")
                    deadline += 1 / cfg["fps"]
                    now = time.monotonic()
                    if now - deadline > 2 / cfg["fps"]:
                        raise RuntimeError("Capture schedule stalled; episode discarded. Lower fps/image size")
                    time.sleep(max(0, deadline - now))
                rig.pause()
                stream.flush()
                if save and frames < 2 and not stop[0]:
                    raise RuntimeError("Episode contains fewer than two frames; increase episode-seconds or lower FPS")
                if save and frames >= 2 and not stop[0]:
                    capture_duration = time.monotonic() - started
                    ds.save_episode(parallel_encoding=False)
                    completed += 1
                    attempt["outcome"] = "saved"
                    print(f"Сохранён эпизод {episode_index}: {frames} кадров, средняя частота {frames / max(0.001, capture_duration):.1f} Гц", flush=True)
                else:
                    ds.clear_episode_buffer()
                    attempt["outcome"] = "discarded"
                    print("Текущий эпизод отброшен; ранее сохранённые эпизоды остаются.", flush=True)
                if quit_requested:
                    stop[0] = True
                if args.auto_start and completed < args.episodes:
                    reset_deadline = time.monotonic() + args.reset_seconds
                    print(f"Пауза {args.reset_seconds:g} с для возврата сцены. Фолловеры удерживают позиции.", flush=True)
                    while not stop[0] and time.monotonic() < reset_deadline:
                        time.sleep(0.05)
            outcome = "interrupted" if stop[0] else "completed"
    except InterruptedError:
        outcome = "interrupted"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Run every cleanup even if a previous cleanup fails; preserve the original error.
        cleanup_errors = cleanup_recording(rig, cameras, ds)
        if cleanup_errors:
            outcome = "error"
        for attempt in attempts:
            if attempt["outcome"] == "incomplete":
                attempt["outcome"] = outcome
        write_json(session / "summary.json", {"outcome": outcome, "saved_episodes_this_session": completed,
                   "dataset_root": str(Path(root).resolve()), "elapsed_s": time.monotonic() - session_start,
                   "error": error, "cleanup_errors": cleanup_errors, "attempts": attempts})
        if cleanup_errors:
            if error is None:
                raise RuntimeError("; ".join(cleanup_errors))
            print("Cleanup errors: " + "; ".join(cleanup_errors), file=sys.stderr)
    print(f"{outcome}: {completed} эпизодов сохранено в {root}")


def cleanup_recording(rig, cameras, ds):
    errors = []
    operations = [("rig", rig.close), ("cameras", cameras.close)]
    if ds is not None:
        def discard():
            if ds.has_pending_frames():
                ds.clear_episode_buffer()
        operations.extend([("discard", discard), ("finalize", ds.finalize)])
    for name, operation in operations:
        try:
            operation()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Две руки SO-ARM100, два лидера, RGB RealSense → LeRobot")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/rig.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Пакеты, USB-порты, RealSense; без команд сервоприводам")
    commands.add_parser("check", help="Проверить 24 привода, калибровки и камеры; без движения")
    cal = commands.add_parser("calibrate", help="Ручная запись границ одной руки; READ only")
    cal.add_argument("--arm", choices=ARM_NAMES, required=True)
    off = commands.add_parser("torque-off", help="Выключить torque выбранной руки перед ручной калибровкой")
    off.add_argument("--arm", choices=ARM_NAMES, required=True)
    imp = commands.add_parser("import-ranges", help="Применить существующий файл диапазонов к выбранной руке без EEPROM writes")
    imp.add_argument("--arm", choices=ARM_NAMES, required=True)
    imp.add_argument("--file", type=Path, required=True)
    inspect = commands.add_parser("inspect-port", help="READ-only поиск ID/model на одном адаптере")
    inspect.add_argument("--port", required=True)
    inspect.add_argument("--ids", type=int, nargs="+", default=list(range(1, 13)))
    sim = commands.add_parser("simulate", help="Синтетический датасет: MP4 + parquet, без оборудования")
    sim.add_argument("--root", type=Path)
    sim.add_argument("--frames", type=int, default=16)
    sim.add_argument("--episodes", type=int, default=2)
    sim.add_argument("--fps", type=int, default=10)
    val = commands.add_parser("validate", help="Проверить датасет и декодировать границы всех эпизодов")
    val.add_argument("root", type=Path)
    val.add_argument("--quick", action="store_true", help="Проверять только граничные кадры эпизодов")
    rec = commands.add_parser("record", help="Движение двух фолловеров и запись локального RGB-датасета")
    rec.add_argument("--task", required=True)
    rec.add_argument("--dataset", default="local/bimanual_arm100")
    rec.add_argument("--root", type=Path)
    rec.add_argument("--episodes", type=int, default=50, help="Количество новых сохранённых эпизодов")
    rec.add_argument("--episode-seconds", type=float, default=60)
    rec.add_argument("--reset-seconds", type=float, default=10)
    rec.add_argument("--auto-start", action="store_true")
    rec.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            from .dataset import validate_dataset
            print(json.dumps(validate_dataset(args.root, quick=args.quick), indent=2))
        else:
            {"doctor": doctor, "check": check, "calibrate": calibrate, "inspect-port": inspect_port,
             "simulate": simulate, "record": record, "torque-off": torque_off, "import-ranges": import_ranges}[args.command](args)
    except (ValueError, KeyError, OSError, ProtocolError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
