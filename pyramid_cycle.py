#!/usr/bin/env python3
"""Teach static follower poses and execute a sequential, two-arm pyramid cycle."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import time

from servo_current_test import ServoBus, ProtocolError, feedback_error, packet
from leader_follower import approach_targets, map_position, read_register, unwrap_in_range

ROOT = Path(__file__).resolve().parent
STAGES = ('pick_approach', 'pick', 'grasp', 'lift', 'place_approach', 'place', 'release', 'retreat')
LABELS = {
    'park': 'парковка вне общей зоны; обе руки должны начинать здесь',
    'pick_approach': 'подход над кубиком; захват открыт',
    'pick': 'захват на высоте кубика; захват ещё открыт',
    'grasp': 'сжать кубик, сохранив положение первых пяти суставов',
    'lift': 'поднять захваченный кубик над окружающими предметами',
    'place_approach': 'подход сверху к месту установки, над готовыми слоями',
    'place': 'положение установки на указанной высоте основания',
    'release': 'разжать захват, сохранив положение первых пяти суставов',
    'retreat': 'вертикальный отвод от установленного кубика',
}


class ServoProtection(ProtocolError):
    """A reported voltage, temperature or overload protection condition."""


class Stopped(Exception):
    pass


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def finite(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name}: требуется число {low}..{high}')


def pose_valid(pose, arm):
    if not isinstance(pose, list) or len(pose) != 6:
        raise ValueError('Поза должна содержать 6 целых положений в порядке ids')
    for i, value in zip(arm['ids'], pose):
        lo, hi = arm['limits'][str(i)]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'ID {i}: положение {value} вне {lo}..{hi}')
    return pose


def load_scene(path, profile=None):
    scene = json.loads(Path(path).read_text())
    if profile is not None:
        if profile not in ('smooth', 'drifty'):
            raise ValueError('Профиль: smooth или drifty')
        profiles = json.loads((ROOT/'pyramid/motion_profiles.json').read_text())
        scene['motion'] = profiles[profile]
        scene['motion_profile'] = profile
    if scene['version'] != 1 or set(scene['arms']) != {'left', 'right'}:
        raise ValueError('Нужен формат version=1, arms.left и arms.right')
    m = scene['motion']
    m.setdefault('transit_settle_s', m['settle_s'])
    m.setdefault('contact_settle_s', m['settle_s'])
    m.setdefault('grip_settle_s', m['settle_s'])
    m.setdefault('continuous_transit', False)
    if type(m['continuous_transit']) is not bool:
        raise ValueError('continuous_transit: требуется true/false')
    m.setdefault('trajectory_profile', 'quintic')
    if m['trajectory_profile'] not in ('cubic', 'quintic'):
        raise ValueError('trajectory_profile: cubic или quintic')
    m.setdefault('transit_tolerance_ticks', 32)
    m.setdefault('transit_speed_ticks_s', m['speed_ticks_s'])
    m.setdefault('gripper_speed_ticks_s', m['speed_ticks_s'])
    scales = m.setdefault('cube_approach_speed_scales', {})
    if not isinstance(scales, dict) or set(scales) - {'cube_5', 'cube_4', 'cube_3', 'cube_2'}:
        raise ValueError('cube_approach_speed_scales: ожидаются имена кубиков')
    for name, value in scales.items():
        finite(value, .1, 1, name + ' approach speed scale')
    for key, bounds in {'hz': (5, 50), 'speed_ticks_s': (1, 1500), 'gripper_speed_ticks_s': (1, 1500), 'transit_speed_ticks_s': (1, 4095), 'acceleration': (1, 254),
                        'tolerance_ticks': (1, 40), 'transit_tolerance_ticks': (1, 50), 'park_tolerance_ticks': (1, 100),
                        'settle_s': (.05, 3), 'transit_settle_s': (0, 3),
                        'contact_settle_s': (0, 3), 'grip_settle_s': (0, 3), 'waypoint_timeout_s': (1, 120),
                        'max_read_age_s': (.03, .5), 'grip_dwell_s': (0, 5),
                        'release_dwell_s': (0, 5)}.items():
        finite(m[key], *bounds, key)
    for key in ('speed_ticks_s', 'gripper_speed_ticks_s', 'transit_speed_ticks_s', 'acceleration', 'tolerance_ticks', 'transit_tolerance_ticks', 'park_tolerance_ticks'):
        if type(m[key]) is not int:
            raise ValueError(f'{key}: требуется целое число')
    ports = []
    for arm in scene['arms'].values():
        ids = arm['ids'] + arm['leader_ids']
        if len(arm['ids']) != 6 or len(arm['leader_ids']) != 6 or len(set(ids)) != 12:
            raise ValueError('На каждой шине нужны 12 разных ID: 6 follower и 6 leader')
        if any(type(i) is not int or not 1 <= i <= 253 for i in ids):
            raise ValueError('ID: 1..253')
        ignored = arm.setdefault('position_error_ignored_ids', [])
        if not isinstance(ignored, list) or len(set(ignored)) != len(ignored) or any(type(i) is not int or i not in arm['ids'] for i in ignored):
            raise ValueError('position_error_ignored_ids: список ID фолловера')
        ports.append(str(Path(arm['port']).resolve()))
        for i in arm['ids']:
            limits = arm['limits'][str(i)]
            if len(limits) != 2 or any(type(v) is not int for v in limits) or not 0 <= limits[0] < limits[1] <= 4095:
                raise ValueError(f'Некорректные границы ID {i}')
            offset = arm['calibration'][str(i)]['offset_raw']
            if type(offset) is not int or not 0 <= offset <= 4095:
                raise ValueError('Некорректное смещение калибровки')
        if any(i not in arm['ids'] for i in arm['invert']):
            raise ValueError('invert должен содержать ID фолловера')
        for i in arm['leader_ids']:
            r = arm['leader_ranges'][str(i)]
            if r.get('ambiguous_steps') or not 0 < r['maximum'] - r['minimum'] < 4096:
                raise ValueError(f'Некорректный диапазон лидера {i}')
    if len(set(ports)) != 2:
        raise ValueError('Для рук нужны два разных USB-порта')
    finite(scene['geometry']['table_length_cm'], 1, 1000, 'table_length_cm')
    finite(scene['geometry']['stack_x_from_left_edge_cm'], 0, scene['geometry']['table_length_cm'], 'stack_x')
    height = 0
    if len(scene['cubes']) != 4:
        raise ValueError('Ожидается 4 кубика')
    for cube, size, arm in zip(scene['cubes'], [5, 4, 3, 2], ['left', 'right', 'left', 'right']):
        if (cube['name'] != f'cube_{size}' or cube['side_cm'] != size or cube['arm'] != arm
                or cube['support_height_cm'] != height or cube['top_height_cm'] != height + size):
            raise ValueError('Порядок: Л5, П4, Л3, П2; высоты основания: 0, 5, 9, 12 см')
        if set(cube['poses']) != set(STAGES):
            raise ValueError(f"{cube['name']}: неполный список именованных поз")
        height += size
    for _, arm_name, _, _, pose in slots(scene):
        if pose is not None:
            pose_valid(pose, scene['arms'][arm_name])
    return scene


def slots(scene):
    for name, arm in scene['arms'].items():
        yield f'{name}.park', name, arm, 'park', arm['park']
    for cube in scene['cubes']:
        for name in ('pick_approach', 'pick', 'grasp', 'lift', 'outbound',
                     'place_approach', 'place', 'release', 'retreat', 'inbound'):
            if name in ('outbound', 'inbound'):
                if not isinstance(cube[name], list):
                    raise ValueError('Маршрут должен быть списком поз')
                for index, pose in enumerate(cube[name]):
                    yield f"{cube['name']}.{name}.{index}", cube['arm'], cube[name], index, pose
            else:
                yield f"{cube['name']}.{name}", cube['arm'], cube['poses'], name, cube['poses'][name]


@dataclass
class Step:
    cube: str
    arm: str
    name: str
    target: list[int]
    dwell: float = 0


def build_plan(scene):
    missing = [name for name, _, _, _, pose in slots(scene) if pose is None]
    if missing:
        raise ValueError('Не записаны позы: ' + ', '.join(missing))
    result = []
    for c in scene['cubes']:
        p, arm = c['poses'], scene['arms'][c['arm']]
        tolerance = scene['motion']['tolerance_ticks']
        for grip, stationary in [('grasp', 'pick'), ('release', 'place')]:
            if any(abs(x-y) > tolerance for x,y in zip(p[grip][:5], p[stationary][:5])):
                raise ValueError(f"{c['name']}.{grip}: первые пять суставов должны совпадать с {stationary}")
        open_pick, closed, open_release = p['pick'][5], p['grasp'][5], p['release'][5]
        if abs(closed-open_pick) <= 2*tolerance or abs(closed-open_release) <= 2*tolerance:
            raise ValueError(f"{c['name']}: открытый и закрытый захват слишком близки")
        def add(name, pose, grip, dwell=0):
            target = pose[:5] + [grip]
            result.append(Step(c['name'], c['arm'], name, pose_valid(target, arm), dwell))
        add('pick_approach', p['pick_approach'], open_pick)
        add('pick', p['pick'], open_pick)
        add('grasp', p['pick'], closed, scene['motion']['grip_dwell_s'])
        add('lift', p['lift'], closed)
        for i, pose in enumerate(c['outbound']):
            add(f'outbound.{i}', pose, closed)
        add('place_approach', p['place_approach'], closed)
        add('place', p['place'], closed)
        add('release', p['place'], open_release, scene['motion']['release_dwell_s'])
        add('retreat', p['retreat'], open_release)
        for i, pose in enumerate(c['inbound']):
            add(f'inbound.{i}', pose, open_release)
        result.append(Step(c['name'], c['arm'], 'park', list(arm['park'])))
    return result


def build_disassembly_plan(scene):
    """Reverse the taught paths, explicitly swapping pick/place gripper actions."""
    build_plan(scene)  # Validate the complete source scene before opening USB.
    result = []
    for c in reversed(scene['cubes']):
        p, arm = c['poses'], scene['arms'][c['arm']]
        closed, tower_open, mat_open = p['grasp'][5], p['release'][5], p['pick'][5]
        def add(name, pose, grip, dwell=0):
            result.append(Step(c['name'], c['arm'], name,
                               pose_valid(pose[:5]+[grip], arm), dwell))
        # Reverse the original retreat-to-park route while empty.
        for i, pose in enumerate(reversed(c['inbound'])):
            add(f'tower_approach.{i}', pose, tower_open)
        add('pick_approach', p['retreat'], tower_open)
        add('pick', p['place'], tower_open)
        add('grasp', p['place'], closed, scene['motion']['grip_dwell_s'])
        add('lift', p['place_approach'], closed)
        for i, pose in enumerate(reversed(c['outbound'])):
            add(f'outbound.{i}', pose, closed)
        add('place_approach', p['lift'], closed)
        add('place', p['pick'], closed)
        add('release', p['pick'], mat_open, scene['motion']['release_dwell_s'])
        add('retreat', p['pick_approach'], mat_open)
        result.append(Step(c['name'], c['arm'], 'park', list(arm['park'])))
    return result


class Rig:
    """Own both USB ports; every packet targets follower IDs except leader reads."""
    def __init__(self, scene, stop, event=lambda *a, **kw: None, teach_gripper=False):
        self.scene, self.stop, self.event = scene, stop, event
        self.buses = {}
        self.stack = ExitStack()
        self.enabled = set()
        self.teach_gripper = teach_gripper
        self.grip_status = {}
        self.last_grip_command = {}

    def __enter__(self):
        import serial
        try:
            for name, arm in self.scene['arms'].items():
                port = self.stack.enter_context(serial.Serial(arm['port'], 1_000_000,
                        timeout=.002, write_timeout=.2, exclusive=True))
                self.buses[name] = ServoBus(port, .03)
            self.preflight()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        # Goal writes can enable torque: hold only arms explicitly activated here.
        for name in self.enabled:
            try:
                self.hold(name)
            except (ProtocolError, OSError) as err:
                self.event('hold_failed', arm=name, error=str(err))
                print(f'Не удалось удержать {name}: {err}', file=sys.stderr)
        self.stack.close()

    def check_stop(self):
        if self.stop[0]:
            raise Stopped('Остановка пользователем')

    def preflight(self):
        for name, arm in self.scene['arms'].items():
            bus = self.buses[name]
            for i in arm['ids']:
                allowed = 0x20 if self.teach_gripper and i == arm['ids'][-1] else 0
                expected = arm['calibration'][str(i)]
                checks = [(3, 2, 777), (33, 1, 0), (18, 1, expected['phase_raw']),
                          (31, 2, expected['offset_raw'])]
                for address, size, value in checks:
                    if int.from_bytes(read_register(bus, i, address, size, allowed), 'little') != value:
                        raise ProtocolError(f'{name}, ID {i}: изменилась модель/калибровка, регистр {address}; покажите позы заново')
                hw = read_register(bus, i, 9, 4, allowed)
                limits = [int.from_bytes(hw[:2], 'little'), int.from_bytes(hw[2:], 'little')]
                if limits != arm['limits'][str(i)]:
                    raise ProtocolError(f'{name}, ID {i}: изменились аппаратные границы')
            if self.teach_gripper:
                for i in arm['leader_ids']:
                    if int.from_bytes(read_register(bus, i, 3, 2), 'little') != 777 or read_register(bus, i, 33, 1) != b'\0':
                        raise ProtocolError(f'{name}: неверная модель/режим лидера ID {i}')
                    if read_register(bus, i, 18, 1)[0] & 0x10:
                        raise ProtocolError(f'{name}: неверный формат положения лидера ID {i}')
                    if read_register(bus, i, 40, 1) != b'\0':
                        raise ProtocolError(f'{name}: момент лидера ID {i} должен быть выключен')
        return self.sample()

    def sample(self):
        self.check_stop()
        start = time.monotonic()
        feedback = {}
        for name, arm in self.scene['arms'].items():
            feedback[name] = []
            for i in arm['ids']:
                f = self.buses[name].feedback(i)
                allowed = 0x20 if self.teach_gripper and i == arm['ids'][-1] else 0
                if f.status & ~allowed:
                    error = ServoProtection if f.status & ~0x25 == 0 else ProtocolError
                    raise error(feedback_error(f))
                if i == arm['ids'][-1]:
                    self.grip_status[name] = f.status
                lo, hi = arm['limits'][str(i)]
                if not lo - 8 <= f.position <= hi + 8:
                    raise ProtocolError(f'{name}, ID {i}: положение вне диапазона')
                feedback[name].append(f)
        if time.monotonic() - start > self.scene['motion']['max_read_age_s']:
            raise ProtocolError('Устаревшая телеметрия; движение остановлено')
        self.event('telemetry', arms={n: [vars(f) for f in fs] for n, fs in feedback.items()})
        return feedback

    def positions(self, feedback, name):
        return [f.position for f in feedback[name]]

    def send(self, name, positions):
        self.check_stop()
        arm = self.scene['arms'][name]
        pose_valid(positions, arm)
        m = self.scene['motion']
        targets = dict(zip(arm['ids'], positions))
        now = time.monotonic()
        if self.teach_gripper and self.grip_status.get(name, 0) & 0x20:
            if now-self.last_grip_command.get(name, -math.inf) < .5:
                targets.pop(arm['ids'][-1])
        if arm['ids'][-1] in targets:
            self.last_grip_command[name] = now
        self.buses[name].move(targets, getattr(self, 'command_speed', m['speed_ticks_s']), m['acceleration'])
        self.event('command', arm=name, positions=targets)

    def enable(self, name, positions):
        self.check_stop()
        self.enabled.add(name)
        self.send(name, positions)
        time.sleep(.03)
        bus = self.buses[name]
        for i, value in zip(self.scene['arms'][name]['ids'], positions):
            self.check_stop()
            allowed = 0x20 if self.teach_gripper and i == self.scene['arms'][name]['ids'][-1] else 0
            if int.from_bytes(read_register(bus, i, 42, 2, allowed), 'little') != value:
                raise ProtocolError(f'ID {i}: начальная цель не подтверждена')
            bus.send(packet(i, 3, b'\x28\x01'))
            time.sleep(.02)
            if read_register(bus, i, 40, 1, allowed) != b'\1':
                raise ProtocolError(f'ID {i}: момент не включился')

    def hold(self, name):
        bus = self.buses[name]
        targets, errors = {}, []
        for i in self.scene['arms'][name]['ids']:
            try:
                f = bus.feedback(i)
                allowed = 0x20 if self.teach_gripper and i == self.scene['arms'][name]['ids'][-1] else 0
                if f.status & ~allowed:
                    raise ProtocolError(feedback_error(f))
                lo, hi = self.scene['arms'][name]['limits'][str(i)]
                if not lo <= f.position <= hi:
                    raise ProtocolError(f'ID {i}: положение удержания вне диапазона')
                targets[i] = f.position
            except (ProtocolError, OSError) as exc:
                errors.append(str(exc))
        # Hold healthy servos even if another one has lost power or communication.
        # Shutdown hold must still work after Ctrl+C.
        if targets:
            m = self.scene['motion']
            bus.move(targets, m['speed_ticks_s'], m['acceleration'])
        if errors:
            raise ProtocolError('; '.join(errors))


def parked(scene, feedback, except_arm=None, reference=None):
    for name, arm in scene['arms'].items():
        if name == except_arm:
            continue
        pose = reference[name] if reference is not None else arm['park']
        if pose is None or any(abs(f.position - v) > scene['motion']['park_tolerance_ticks']
                               or f.moving != 0
                               for f, v in zip(feedback[name], pose)):
            raise ProtocolError(f'{name}: рука должна быть в показанной парковочной позе вне общей зоны')


def trajectory_blend(u, profile):
    """Zero endpoint velocity; quintic also has zero endpoint acceleration."""
    return u*u*u*(10 + u*(-15 + 6*u)) if profile == 'quintic' else u*u*(3-2*u)


def step_settle(m, step):
    if step.name == 'grasp':
        return m.get('grip_settle_s', m['settle_s'])
    if step.name in ('park', 'release'):
        return m['settle_s']
    key = 'contact_settle_s' if step.name in ('pick', 'place') else 'transit_settle_s'
    return m.get(key, m['settle_s'])


def step_speed(m, step):
    if step.name in ('grasp', 'release'):
        return m.get('gripper_speed_ticks_s', m['speed_ticks_s'])
    speed = (m['speed_ticks_s'] if step.name in ('pick', 'place')
             else m.get('transit_speed_ticks_s', m['speed_ticks_s']))
    if step.name in ('pick_approach', 'pick'):
        speed = max(1, round(speed*m.get('cube_approach_speed_scales', {}).get(step.cube, 1)))
    return speed


def transit_duration(start, target, speed):
    return max(.1, 1.5 * max(abs(b-a) for a, b in zip(start, target)) / speed)


def junction_velocity(start, target, following, duration, next_duration):
    # Monotone cubic Hermite: each normalized tangent is in [0, 1].
    # No joint overshoot; peak derivative <= 1.5 times segment mean speed.
    result = []
    for a, b, c in zip(start, target, following):
        incoming, outgoing = (b-a)/duration, (c-b)/next_duration
        result.append(math.copysign(min(abs(incoming), abs(outgoing)), incoming)
                      if incoming*outgoing > 0 else 0.)
    return result


def hermite_position(u, start, target, v0, v1, duration):
    h = u*u*(3-2*u)
    return [round(a+(b-a)*h + duration*(x*u*(1-u)**2-y*u*u*(1-u)))
            for a,b,x,y in zip(start, target, v0, v1)]


def execute_step(rig, step, waiting_positions=None, next_step=None):
    while True:
        try:
            return execute_step_once(rig, step, waiting_positions, next_step)
        except ServoProtection as exc:
            rig._trajectory_handoff = None
            rig.event('protection_pause', arm=step.arm, step=step.name, error=str(exc))
            print(f'Защита: {exc}. Проверка восстановления через 0,5 с.', flush=True)
            # Hold healthy servos; do not reset or command a protected motor.
            for name in rig.enabled:
                try:
                    rig.hold(name)
                except ProtocolError as hold_error:
                    rig.event('recovery_hold_failed', arm=name, error=str(hold_error))
            while True:
                deadline = time.monotonic() + .5
                while time.monotonic() < deadline:
                    rig.check_stop()
                    time.sleep(min(.05, max(0, deadline-time.monotonic())))
                rig.check_stop()
                try:
                    feedback = rig.sample()
                except ServoProtection:
                    continue
                # Re-enable only arms already activated, at their present position.
                # Hardware protection must have cleared before any goal is sent.
                for name in tuple(rig.enabled):
                    rig.enable(name, rig.positions(feedback, name))
                rig.event('protection_recovered', arm=step.arm, step=step.name)
                print(f'Защита снята: продолжаю {step.cube}/{step.name}.', flush=True)
                break


def execute_step_once(rig, step, waiting_positions=None, next_step=None):
    scene, m = rig.scene, rig.scene['motion']
    tolerance = (m['tolerance_ticks'] if step.name in ('pick', 'grasp', 'place', 'release')
                 else m['park_tolerance_ticks'] if step.name == 'park'
                 else m.get('transit_tolerance_ticks', 32))
    speed = step_speed(m, step)
    rig.command_speed = speed
    ignored = scene['arms'][step.arm].get('position_error_ignored_ids', [])
    feedback = rig.sample()
    parked(scene, feedback, step.arm, waiting_positions)
    start = rig.positions(feedback, step.arm)
    profile = m.get('trajectory_profile', 'quintic')
    peak_slope = 1.875 if profile == 'quintic' else 1.5
    settle = step_settle(m, step)
    duration = max(.1, peak_slope * max(abs(a-b) for a,b in zip(start, step.target)) / speed)
    continuous = m.get('continuous_transit', False)
    passthrough = (continuous and next_step is not None and step.dwell == 0
                   and step.name not in ('pick', 'place', 'grasp', 'release', 'park')
                   and next_step.arm == step.arm and next_step.cube == step.cube
                   and next_step.name not in ('grasp', 'release'))
    handoff = getattr(rig, '_trajectory_handoff', None)
    rig._trajectory_handoff = None
    v0, v1 = [0.]*6, [0.]*6
    if continuous and handoff is not None and handoff['step'] == step:
        if time.monotonic()-handoff['time'] > m['max_read_age_s']:
            raise ProtocolError('Пауза между переходами; циклограмма остановлена')
        start, v0 = handoff['start'], handoff['velocity']
    use_hermite = passthrough or any(v0)
    if use_hermite:
        duration = transit_duration(start, step.target, speed)
        if passthrough:
            next_duration = transit_duration(step.target, next_step.target, step_speed(m, next_step))
            v1 = junction_velocity(start, step.target, next_step.target, duration, next_duration)
    if duration + settle + step.dwell >= m['waypoint_timeout_s']:
        raise ValueError('Время этапа превышает timeout; добавьте промежуточные позы или увеличьте timeout')
    begun = previous = time.monotonic()
    stable_since = None
    stable_anchor = None
    previous_command = None
    last_command_time = -math.inf
    while True:
        rig.check_stop()
        feedback = rig.sample()
        parked(scene, feedback, step.arm, waiting_positions)
        now = time.monotonic()
        if now-previous > m['max_read_age_s']:
            raise ProtocolError('Пауза в управлении; циклограмма остановлена')
        previous = now
        if now-begun > m['waypoint_timeout_s']:
            raise ProtocolError(f'{step.cube}/{step.name}: цель не достигнута за отведённое время')
        u = min(1., (now-begun)/duration)
        blend = trajectory_blend(u, profile)
        command = (hermite_position(u, start, step.target, v0, v1, duration) if use_hermite
                   else [round(a+(b-a)*blend) for a,b in zip(start, step.target)])
        fs = feedback[step.arm]
        lagging = any(f.servo_id not in ignored and abs(f.position-v) > tolerance for f,v in zip(fs, step.target))
        if command != previous_command or (u == 1 and lagging and now-last_command_time >= .5):
            rig.send(step.arm, command)
            previous_command = command
            last_command_time = now
        if u == 1 and passthrough:
            if not lagging:
                rig._trajectory_handoff = dict(step=next_step, start=list(step.target),
                                               velocity=v1, time=now)
                rig.event('waypoint_passed', arm=step.arm, step=step.name,
                          errors={f.servo_id: v-f.position for f,v in zip(fs,step.target)})
                return
            # A required joint fell behind: finish this point at rest instead.
            passthrough = False
        if stable_anchor is None or any(abs(f.position-p)>6 for f,p in zip(fs,stable_anchor)):
            stable_anchor = [f.position for f in fs]
            stable_since = None
        arrived = u == 1 and all((f.servo_id in ignored or abs(f.position-v) <= tolerance)
                                 and abs(f.speed) <= 20
                                 for f,v in zip(fs, step.target))
        if arrived:
            if stable_since is None:
                stable_since = now
            if now-stable_since >= settle + step.dwell:
                offsets = {f.servo_id: v-f.position for f,v in zip(fs,step.target)
                           if f.servo_id in ignored and abs(v-f.position)>tolerance}
                if offsets:
                    rig.event('position_offset_accepted', arm=step.arm, step=step.name, errors=offsets)
                return
        else:
            stable_since = None
        time.sleep(max(0, 1/m['hz']-(time.monotonic()-now)))


def activate_at_current_pose(rig):
    """Give both arms their own present position before enabling torque."""
    for name in rig.scene['arms']:
        feedback = rig.sample()
        rig.enable(name, rig.positions(feedback, name))


def park_hands(rig, on_start=lambda step: None, on_done=lambda step: None):
    """Park sequentially while the other arm holds its current position."""
    feedback = rig.sample()
    waiting = {name: rig.positions(feedback, name) for name in rig.scene['arms']}
    for index, (name, arm) in enumerate(rig.scene['arms'].items(), 1):
        step = Step('startup', name, 'park', list(arm['park']))
        print(f'Парковка {index}/2: {name}', flush=True)
        on_start(step)
        execute_step(rig, step, waiting_positions=waiting)
        waiting[name] = list(arm['park'])
        on_done(step)
    parked(rig.scene, rig.sample())


def resume_index(scene, plan, folder, disassemble=False):
    state = json.loads((folder/'state.json').read_text())
    old = json.loads((folder/'poses.snapshot.json').read_text())
    if state.get('operation', 'assemble') != ('disassemble' if disassemble else 'assemble'):
        raise ValueError('Нельзя менять сборку на разборку при продолжении')
    if state['outcome'] not in ('error', 'interrupted'):
        raise ValueError('Продолжать можно только остановленный цикл')
    index = state['completed_steps']
    if type(index) is not int or not 0 <= index < len(plan) or state['current'] != vars(plan[index]):
        raise ValueError('Текущий этап не совпадает с сохранённым планом')
    if old['cubes'] != scene['cubes'] or old['geometry'] != scene['geometry']:
        raise ValueError('Позы или геометрия изменились после остановки')
    for name, arm in scene['arms'].items():
        for key in ('port', 'ids', 'limits', 'calibration', 'park'):
            if old['arms'][name][key] != arm[key]:
                raise ValueError('Подключение или калибровка изменились после остановки')
    return index


def run_cycle(scene, output, stop, resume=None, disassemble=False):
    plan = build_disassembly_plan(scene) if disassemble else build_plan(scene)  # Refuse incomplete scenes before opening ports.
    start_index = resume_index(scene, plan, resume, disassemble) if resume is not None else 0
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'poses.snapshot.json', scene)
    state = {'operation': 'disassemble' if disassemble else 'assemble', 'outcome': 'starting', 'completed_steps': start_index, 'current': None,
             'resumed_from': str(resume) if resume is not None else None,
             'scene_sha256': hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()}
    with (output/'events.jsonl').open('a', buffering=1) as stream:
        def event(kind, **data):
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                         monotonic_s=time.monotonic(), event=kind, **data))+'\n')
        try:
            with Rig(scene, stop, event) as rig:
                if resume is not None:
                    parked(scene, rig.sample(), plan[start_index].arm)
                activate_at_current_pose(rig)
                if resume is None:
                    def parking_start(step):
                        state.update(outcome='parking', current=vars(step))
                        atomic_json(output/'state.json', state)
                        event('parking_start', **vars(step))
                    def parking_done(step):
                        state.setdefault('startup_parking_completed', []).append(step.arm)
                        atomic_json(output/'state.json', state)
                        event('parking_done', **vars(step))
                    park_hands(rig, parking_start, parking_done)
                for number, step in enumerate(plan[start_index:], start_index+1):
                    state.update(outcome='running', current=vars(step))
                    atomic_json(output/'state.json', state)
                    print(f'{number}/{len(plan)}: {step.arm} {step.cube} / {step.name}', flush=True)
                    event('step_start', **vars(step))
                    execute_step(rig, step, next_step=plan[number] if number < len(plan) else None)
                    state['completed_steps'] = number
                    event('step_done', **vars(step))
                parked(scene, rig.sample())
                state['outcome'] = 'completed'
        except BaseException as exc:
            state.update(outcome='interrupted' if isinstance(exc, Stopped) else 'error', error=str(exc),
                         restart_note='Проверить кубики и восстановить исходную сцену; автоматического повтора нет.')
            event('stopped', error=str(exc))
            raise
        finally:
            atomic_json(output/'state.json', state)
    return output


def teach(scene, path, selected_arm, selected_pose, mode, stop, start_at=None):
    selected = [s for s in slots(scene) if (selected_arm is None or s[1] == selected_arm)
                and (selected_pose is None or s[0] == selected_pose)]
    if start_at is not None:
        names = [s[0] for s in selected]
        if start_at not in names:
            raise ValueError('Нет такой начальной позы')
        selected = selected[names.index(start_at):]
    if not selected:
        raise ValueError('Нет такой позы/руки. Список: команда plan')
    if not sys.stdin.isatty():
        raise ValueError('Показ поз требует интерактивного терминала')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = Path(path).with_name(Path(path).stem+'.before_teach_'+stamp+'.json')
    backup.write_bytes(Path(path).read_bytes())
    active_arm = None
    with Rig(scene, stop, teach_gripper=True) as rig:
        for slot, name, container, key, old in selected:
            arm = scene['arms'][name]
            if active_arm is not None and active_arm != name:
                rig.hold(active_arm)
            active_arm = name
            feedback = rig.sample()
            current = rig.positions(feedback, name)
            if mode == 'teleop' and name not in rig.enabled:
                rig.enable(name, current)
            if mode == 'passive':
                for i in arm['ids']:
                    if read_register(rig.buses[name], i, 40, 1) != b'\0':
                        raise ProtocolError('Для passive момент выбранного фолловера должен быть выключен; поддерживайте руку вручную')
            label = str(key) if isinstance(key, str) else slot.split('.')[-2]
            print(f'\n{slot}: {LABELS.get(label, "промежуточная поза маршрута")}.', flush=True)
            if slot.startswith('cube_') and label in ('place_approach', 'place', 'release'):
                cube = next(c for c in scene['cubes'] if c['name'] == slot.split('.')[0])
                print(f"Основание кубика: {cube['support_height_cm']} см; верх: {cube['top_height_cm']} см.")
            print('Двигайте лидер.' if mode == 'teleop' else 'Перемещайте фолловер вручную.')
            print('Enter — сохранить; s + Enter — пропустить; q + Enter — закончить.', flush=True)
            targets = dict(zip(arm['ids'], map(float, current)))
            previous_time = time.monotonic()
            previous_raw = None
            stable_since = None
            stable_anchor = None
            while True:
                tick = time.monotonic()
                feedback = rig.sample()
                values = rig.positions(feedback, name)
                settled = all(abs(f.speed) <= 20 for f in feedback[name])
                if stable_anchor is None or max(abs(a-b) for a,b in zip(values, stable_anchor)) > 6 or not settled:
                    stable_anchor = list(values)
                    stable_since = tick
                if mode == 'teleop':
                    raw = [rig.buses[name].feedback(i) for i in arm['leader_ids']]
                    if time.monotonic()-tick > scene['motion']['max_read_age_s'] or tick-previous_time > scene['motion']['max_read_age_s']:
                        raise ProtocolError('Устаревшие данные при показе поз')
                    desired = {}
                    for index, (li, fi) in enumerate(zip(arm['leader_ids'], arm['ids'])):
                        f = raw[index]
                        if f.status:
                            raise ProtocolError(feedback_error(f))
                        if previous_raw is not None and abs((f.position-previous_raw[index]+2048)%4096-2048)>1024:
                            raise ProtocolError('Скачок положения лидера')
                        r = arm['leader_ranges'][str(li)]
                        desired[fi] = map_position(f.position, (r['minimum'], r['maximum']),
                                                   arm['limits'][str(fi)], fi in arm['invert'])
                    previous_raw = [f.position for f in raw]
                    targets = approach_targets(targets, desired, scene['motion']['speed_ticks_s'], min(tick-previous_time, 2/scene['motion']['hz']))
                    rig.send(name, [round(targets[i]) for i in arm['ids']])
                    previous_time = tick
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        return
                    action = line.strip().lower()
                    if action == 'q':
                        return
                    if action == 's':
                        break
                    if action:
                        print('Enter, s или q', flush=True)
                        continue
                    if tick-stable_since < .3:
                        print('Дождитесь остановки фолловера и нажмите Enter.', flush=True)
                        continue
                    pose_valid(values, arm)
                    if label in ('grasp', 'release'):
                        basekey = 'pick' if label == 'grasp' else 'place'
                        reference = container[basekey]
                        if reference is None or any(i not in arm.get('position_error_ignored_ids', []) and abs(x-y)>scene['motion']['tolerance_ticks'] for i,x,y in zip(arm['ids'][:5], reference[:5], values[:5])):
                            print(f'Сначала запишите {basekey}; при {label} изменяется только захват.', flush=True)
                            continue
                        values = reference[:5]+[values[5]]
                    container[key] = list(values)
                    scene['taught'][slot] = {'utc': datetime.now(timezone.utc).isoformat(), 'mode': mode}
                    atomic_json(path, scene)
                    print(f'Сохранено {slot}: {values}', flush=True)
                    break
                time.sleep(max(0, 1/scene['motion']['hz']-(time.monotonic()-tick)))


def stop_teleop(scene):
    """Stop only our recorded teleop owners, with command/port identity checks."""
    ports = {arm['port'] for arm in scene['arms'].values()}
    for name in ('first', 'second'):
        statefile = ROOT/'results'/f'{name}_pair_setup.json'
        if not statefile.exists():
            continue
        state = json.loads(statefile.read_text())
        if state.get('port') not in ports:
            continue
        pid = state['pid']
        proc = Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or not proc.read_bytes():
            continue
        command = proc.read_bytes().split(b'\0')
        if str(ROOT/'leader_follower.py').encode() not in command or state['port'].encode() not in command:
            raise ValueError(f'PID {pid} принадлежит другой программе; процесс не остановлен')
        os.kill(pid, signal.SIGINT)
        deadline = time.monotonic()+5
        while proc.exists() and proc.read_bytes() and time.monotonic()<deadline:
            time.sleep(.05)
        if proc.exists() and proc.read_bytes():
            raise ValueError(f'PID {pid} ещё завершает работу; повторите команду')
        state['phase'] = 'stopped_for_pyramid'
        atomic_json(statefile, state)
        print(f'Телеоп {name} остановлен; фолловер удерживает положение.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Показ статичных поз и сборка пирамиды Л5–П4–Л3–П2')
    parser.add_argument('--scene', type=Path, default=ROOT/'pyramid/poses.json')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan', help='Офлайн: позы, порядок, высоты; без открытия USB')
    commands.add_parser('stop-teleop', help='Остановить наши два процесса телеопа и освободить USB')
    commands.add_parser('park', help='Плавно перевести обе руки в исходные парковки по очереди')
    commands.add_parser('check', help='Только чтение: приводы, калибровка, готовность сцены и парковок')
    teaching = commands.add_parser('teach', help='Показ поз в интерактивном терминале')
    teaching.add_argument('--arm', choices=['left', 'right'])
    teaching.add_argument('--pose', help='Например left.park или cube_5.pick')
    teaching.add_argument('--start-at', help='Продолжить мастер с указанной позы, например right.park')
    teaching.add_argument('--mode', choices=['teleop', 'passive'], default='teleop')
    running = commands.add_parser('run', help='Сначала план; реальные движения только с --execute')
    running.add_argument('--execute', action='store_true')
    running.add_argument('--profile', choices=['smooth', 'drifty'], help='smooth: исходный режим; drifty: текущий проходной')
    running.add_argument('--disassemble', action='store_true', help='Разобрать башню сверху вниз: П2–Л3–П4–Л5')
    running.add_argument('--output', type=Path)
    running.add_argument('--resume', type=Path, help='Продолжить текущий этап остановленного цикла из его папки результатов')
    args = parser.parse_args(argv)
    stop = [False]
    handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        scene = load_scene(args.scene, getattr(args, 'profile', None))
        if getattr(args, 'profile', None):
            print(f'Профиль движения: {args.profile}', flush=True)
        if args.command == 'plan' or args.command == 'run' and not args.execute:
            for cube in scene['cubes']:
                print(f"{cube['name']}: {cube['arm']}, {cube['side_cm']} см, основание {cube['support_height_cm']} см, верх {cube['top_height_cm']} см")
            missing = [name for name, _, _, _, pose in slots(scene) if pose is None]
            for name in missing:
                print(f'  НЕ ЗАПИСАНО: {name}')
            if missing:
                return 2
            for index, step in enumerate(build_disassembly_plan(scene) if getattr(args, 'disassemble', False) else build_plan(scene), 1):
                print(f'{index}: {step.arm} {step.cube}.{step.name} {step.target}')
            return 0
        if args.command == 'stop-teleop':
            stop_teleop(scene)
        elif args.command == 'teach':
            teach(scene, args.scene, args.arm, args.pose, args.mode, stop, args.start_at)
        elif args.command == 'park':
            for arm in scene['arms'].values():
                pose_valid(arm['park'], arm)
            with Rig(scene, stop) as rig:
                activate_at_current_pose(rig)
                park_hands(rig)
            print('Обе руки переведены в исходные парковки.')
        elif args.command == 'check':
            build_plan(scene)
            with Rig(scene, stop) as rig:
                parked(scene, rig.sample())
            print('Проверка пройдена; команды движения не отправлялись.')
        else:
            output = args.output or ROOT/'results'/('pyramid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
            if args.resume:
                print(f'Продолжение текущего этапа из {args.resume}; выполненные этапы не повторяются.', flush=True)
            elif args.disassemble:
                print('Разборка: парковка, затем П2–Л3–П4–Л5. Башня собрана, исходные метки свободны.', flush=True)
            else:
                print('Один цикл: парковка обеих рук, затем сборка. Кубики на исходных метках, место сборки пустое.', flush=True)
            run_cycle(scene, output, stop, args.resume, args.disassemble)
            print(f'Готово. Журнал: {output}')
        return 0
    except (ValueError, KeyError, TypeError, OSError, ProtocolError, Stopped) as exc:
        print(f'Остановлено: {exc}', file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
