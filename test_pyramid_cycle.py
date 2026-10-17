import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyramid_cycle as app
from servo_current_test import Feedback, ProtocolError


def fixture():
    s = app.load_scene(Path(__file__).with_name('pyramid')/'poses.json')
    s['motion']['grip_settle_s']=.25
    s['motion']['grip_dwell_s']=.5
    s['motion']['continuous_transit']=False
    s['motion']['gripper_speed_ticks_s']=s['motion']['speed_ticks_s']
    s['motion']['cube_approach_speed_scales']={}
    s['motion']['transit_speed_ticks_s']=s['motion']['speed_ticks_s']
    s['motion']['transit_settle_s']=s['motion']['settle_s']
    s['motion']['contact_settle_s']=s['motion']['settle_s']
    for arm in s['arms'].values():
        arm['position_error_ignored_ids']=[]
        arm['limits'] = {str(i): [1000, 3000] for i in arm['ids']}
        arm['park'] = [2000]*5+[1200]
    for c in s['cubes']:
        for index, name in enumerate(app.STAGES):
            c['poses'][name] = [1800+20*index]*5+[1200]
        c['poses']['grasp'] = c['poses']['pick'][:5]+[2200]
        c['poses']['release'] = c['poses']['place'][:5]+[1200]
    return s


class Clock:
    def __init__(self):
        self.now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += max(0, seconds)


class SimRig:
    def __init__(self, scene, stop, event=lambda *a, **kw: None):
        self.scene, self.stop, self.event = scene, stop, event
        self.pos = {name: list(arm['park']) for name, arm in scene['arms'].items()}
        self.commands = []
        self.enabled = set()
        self.stuck = None
        self.fault = False
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
    def check_stop(self):
        if self.stop[0]:
            raise app.Stopped()
    def sample(self):
        self.check_stop()
        if self.fault:
            raise ProtocolError('ID 2: status 0x01')
        return {name: [Feedback(i, p, 0, 8, 25, 0, 0, 0, 0)
                       for i,p in zip(self.scene['arms'][name]['ids'], pos)]
                for name, pos in self.pos.items()}
    positions = app.Rig.positions
    def send(self, name, values):
        self.check_stop()
        self.commands.append((name, list(values)))
        if self.stuck != name:
            self.pos[name] = list(values)
    def enable(self, name, values):
        self.enabled.add(name)
        self.send(name, values)


class PyramidTests(unittest.TestCase):
    def setUp(self):
        self.scene = fixture()
        self.clock = Clock()
        self.monotonic = patch.object(app.time, 'monotonic', self.clock.monotonic)
        self.sleep = patch.object(app.time, 'sleep', self.clock.sleep)
        self.monotonic.start()
        self.sleep.start()
    def tearDown(self):
        self.sleep.stop()
        self.monotonic.stop()

    def test_current_scene_has_correct_identity_and_valid_recorded_poses(self):
        scene = app.load_scene(Path(__file__).with_name('pyramid')/'poses.json')
        self.assertEqual(scene['arms']['left']['ids'], list(range(1,7)))
        self.assertEqual(scene['arms']['right']['ids'], list(range(13,19)))
        for _,name,_,_,pose in app.slots(scene):
            if pose is not None:app.pose_valid(pose,scene['arms'][name])

    def test_left_right_order_and_stack_heights(self):
        p = app.build_plan(self.scene)
        self.assertEqual(len(p), 36)
        self.assertEqual([(s.cube,s.arm) for s in p if s.name=='grasp'],
                         [('cube_5','left'),('cube_4','right'),('cube_3','left'),('cube_2','right')])
        self.assertEqual([c['support_height_cm'] for c in self.scene['cubes']], [0,5,9,12])
        for i in range(0,36,9):
            self.assertEqual(p[i+8].name, 'park')

    def test_grip_is_preserved_during_transport_and_only_gripper_changes_at_contact(self):
        p = app.build_plan(self.scene)[:9]
        self.assertEqual(p[1].target[:5], p[2].target[:5])
        self.assertEqual(p[5].target[:5], p[6].target[:5])
        self.assertTrue(all(s.target[5] == 2200 for s in p[2:6]))
        self.assertEqual(p[6].target[5], 1200)
        self.scene['cubes'][0]['poses']['grasp'][0] += 100
        with self.assertRaisesRegex(ValueError, 'первые пять'):
            app.build_plan(self.scene)

    def test_extra_waypoints_preserve_grip(self):
        c = self.scene['cubes'][0]
        c['outbound'] = [[1900]*6, [1950]*6]
        c['inbound'] = [[1800]*6]
        p = app.build_plan(self.scene)
        self.assertEqual([s.target[5] for s in p if s.name.startswith('outbound')], [2200,2200])
        self.assertEqual([s.target[5] for s in p if s.name.startswith('inbound')], [1200])

    def test_pose_outside_calibration_is_rejected(self):
        self.scene['cubes'][0]['poses']['pick'][0] = 999
        with self.assertRaises(ValueError):
            app.build_plan(self.scene)

    def test_actual_pose_settling_before_next_step_and_velocity_limit(self):
        rig = SimRig(self.scene, [False])
        step = app.build_plan(self.scene)[0]
        app.execute_step(rig, step)
        self.assertEqual(rig.pos['left'], step.target)
        self.assertGreaterEqual(self.clock.now, self.scene['motion']['settle_s'])
        for (_,a),(_,b) in zip(rig.commands,rig.commands[1:]):
            self.assertLessEqual(max(abs(x-y) for x,y in zip(a,b)),
                                 self.scene['motion']['speed_ticks_s']/self.scene['motion']['hz']+2)

    def test_quintic_is_monotone_bounded_and_has_smooth_endpoints(self):
        values = [app.trajectory_blend(i/10000, 'quintic') for i in range(10001)]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 1)
        differences = [b-a for a,b in zip(values, values[1:])]
        self.assertTrue(all(d >= 0 for d in differences))
        self.assertLessEqual(max(differences)*10000, 1.875)
        self.assertLess(differences[0], 1e-10)
        self.assertLess(differences[-1], 1e-10)

    def test_short_transit_settle_preserves_grasp_release_and_park_waits(self):
        m = self.scene['motion']
        m.update(transit_settle_s=.04, contact_settle_s=.08)
        for name, expected in [('lift', .04), ('outbound.0', .04),
                               ('pick', .08), ('place', .08),
                               ('grasp', .25), ('release', .25), ('park', .25)]:
            self.assertEqual(app.step_settle(m, app.Step('cube_5', 'left', name, [])), expected)
        rig = SimRig(self.scene, [False])
        step = app.build_plan(self.scene)[0]
        app.execute_step(rig, step)
        quick_time = self.clock.now
        self.clock.now = 0
        m['transit_settle_s'] = .25
        app.execute_step(SimRig(self.scene, [False]), step)
        self.assertGreater(self.clock.now-quick_time, .15)

    def test_continuous_transit_does_not_wait_for_zero_speed(self):
        self.scene['motion']['continuous_transit'] = True
        rig = SimRig(self.scene, [False])
        step = app.Step('cube_5', 'left', 'lift', [2100]*5+[1200])
        following = app.Step('cube_5', 'left', 'place_approach', [2200]*5+[1200])
        original = rig.sample
        def sample():
            fs = original()
            for f in fs['left']:
                f.speed = 100
            return fs
        rig.sample = sample
        app.execute_step(rig, step, next_step=following)
        self.assertEqual(rig.pos['left'], step.target)
        self.assertTrue(any(rig._trajectory_handoff['velocity']))
        self.assertLess(self.clock.now, .6)
        rig.sample = original
        app.execute_step(rig, following)
        self.assertIsNone(rig._trajectory_handoff)
        self.assertEqual(rig.pos['left'], following.target)

    def test_contact_still_waits_for_stop_and_dwell_in_continuous_mode(self):
        self.scene['motion']['continuous_transit'] = True
        rig = SimRig(self.scene, [False])
        original = rig.sample
        def sample():
            fs = original()
            if self.clock.now < 2:
                for f in fs['left']: f.speed = 100
            return fs
        rig.sample = sample
        plan = app.build_plan(self.scene)
        app.execute_step(rig, plan[2], next_step=plan[3])
        self.assertGreaterEqual(self.clock.now, 2 + plan[2].dwell + self.scene['motion']['settle_s'])
        self.assertIsNone(rig._trajectory_handoff)

    def test_junction_reversals_stop_and_curves_stay_bounded_at_speed(self):
        start, target, following = [1000,2000,1500], [2000,1000,1500], [2500,2000,1700]
        v = app.junction_velocity(start, target, following, 3, 4)
        self.assertGreater(v[0], 0)
        self.assertEqual(v[1:], [0,0])
        # Worst monotone tangent combinations, including zero and full mean speed.
        for t0 in (0, .5, 1):
            for t1 in (0, .5, 1):
                duration = app.transit_duration(start, target, 350)
                v0 = [(b-a)/duration*t0 for a,b in zip(start,target)]
                v1 = [(b-a)/duration*t1 for a,b in zip(start,target)]
                values = [app.hermite_position(i/100, start,target,v0,v1,duration) for i in range(101)]
                self.assertEqual(values[0], start)
                self.assertEqual(values[-1], target)
                for row in values:
                    self.assertTrue(all(min(a,b)<=p<=max(a,b) for a,b,p in zip(start,target,row)))
                for a,b in zip(values,values[1:]):
                    self.assertLessEqual(max(abs(x-y) for x,y in zip(a,b)),350*duration/100+1)

    def test_gripper_speed_and_cube_approach_scale_are_isolated(self):
        m = self.scene['motion']
        m.update(gripper_speed_ticks_s=700, transit_speed_ticks_s=2048,
                 cube_approach_speed_scales={'cube_4': .8})
        for cube in ('cube_5', 'cube_4'):
            for stage in app.STAGES:
                step = app.Step(cube, 'right', stage, [])
                expected = (700 if stage in ('grasp', 'release') else
                            350 if stage in ('pick', 'place') else 2048)
                if cube == 'cube_4' and stage in ('pick_approach', 'pick'):
                    expected = round(expected*.8)
                self.assertEqual(app.step_speed(m, step), expected)

    def test_grasp_without_dwell_advances_as_soon_as_gripper_stops(self):
        m = self.scene['motion']
        m.update(grip_settle_s=0, grip_dwell_s=0)
        rig = SimRig(self.scene, [False])
        original = rig.sample
        def sample():
            fs = original()
            if self.clock.now < 10:
                fs['left'][-1].speed = 100
            return fs
        rig.sample = sample
        step = app.build_plan(self.scene)[2]
        app.execute_step(rig, step)
        self.assertGreaterEqual(self.clock.now, 10)
        self.assertLess(self.clock.now, 10.05)
        self.assertEqual(app.step_settle(m, step), 0)
        release = app.build_plan(self.scene)[6]
        self.assertEqual(release.dwell, .5)
        self.assertEqual(app.step_settle(m, release), .25)

    def test_disassembly_reverses_cube_order_paths_and_gripper_actions(self):
        c = self.scene['cubes'][-1]
        c['outbound'] = [[1900]*6, [1950]*6]
        c['inbound'] = [[1850]*6, [1800]*6]
        plan = app.build_disassembly_plan(self.scene)
        self.assertEqual([s.cube for s in plan if s.name == 'grasp'],
                         ['cube_2', 'cube_3', 'cube_4', 'cube_5'])
        self.assertEqual([s.arm for s in plan if s.name == 'grasp'],
                         ['right', 'left', 'right', 'left'])
        self.assertEqual([s.target[0] for s in plan[:2]], [1800,1850])
        for c in self.scene['cubes']:
            steps = {s.name:s for s in plan if s.cube == c['name']}
            self.assertEqual(steps['pick'].target[:5], c['poses']['place'][:5])
            self.assertEqual(steps['grasp'].target[:5], steps['pick'].target[:5])
            self.assertEqual(steps['place'].target[:5], c['poses']['pick'][:5])
            self.assertEqual(steps['release'].target[:5], steps['place'].target[:5])
            for stage in ('grasp', 'lift', 'place_approach', 'place'):
                self.assertEqual(steps[stage].target[5], c['poses']['grasp'][5])
            self.assertEqual(steps['release'].target[5], c['poses']['pick'][5])
        self.assertEqual([s.target[0] for s in plan if s.cube=='cube_2' and s.name.startswith('outbound')], [1950,1900])

    def test_protection_retries_same_step_only_after_half_second_and_clear_status(self):
        rig = SimRig(self.scene, [False])
        rig.enabled = {'left'}
        rig.hold = lambda name: None
        original = rig.sample
        polls = []
        def sample():
            polls.append(self.clock.now)
            if len(polls) == 1:
                raise app.ServoProtection('still overloaded')
            return original()
        rig.sample = sample
        step = app.build_plan(self.scene)[2]
        calls = []
        def execute(*args):
            calls.append(args[1])
            if len(calls) == 1:
                raise app.ServoProtection('overload')
        with patch.object(app, 'execute_step_once', side_effect=execute), patch('sys.stdout',new=io.StringIO()):
            app.execute_step(rig, step)
        self.assertEqual(calls, [step, step])
        self.assertGreaterEqual(polls[0], .5)
        self.assertGreaterEqual(polls[1]-polls[0], .5)
        self.assertIsNone(rig._trajectory_handoff)

    def test_non_protection_fault_is_not_retried(self):
        rig = SimRig(self.scene, [False])
        with patch.object(app, 'execute_step_once', side_effect=ProtocolError('bounds')) as execute:
            with self.assertRaises(ProtocolError):
                app.execute_step(rig, app.build_plan(self.scene)[0])
            self.assertEqual(execute.call_count, 1)

    def test_stop_interrupts_protection_wait(self):
        rig = SimRig(self.scene, [True])
        with patch.object(app, 'execute_step_once', side_effect=app.ServoProtection('overload')), patch('sys.stdout',new=io.StringIO()):
            with self.assertRaises(app.Stopped):
                app.execute_step(rig, app.build_plan(self.scene)[0])
        self.assertEqual(self.clock.now, 0)

    def test_named_profiles_use_same_current_poses_and_different_motion(self):
        path = Path(__file__).with_name('pyramid')/'poses.json'
        before = path.read_bytes()
        smooth = app.load_scene(path, 'smooth')
        drifty = app.load_scene(path, 'drifty')
        current = app.load_scene(path)
        self.assertEqual(smooth['cubes'], current['cubes'])
        self.assertEqual(drifty['cubes'], current['cubes'])
        self.assertEqual(smooth['arms'], drifty['arms'])
        self.assertFalse(smooth['motion']['continuous_transit'])
        self.assertEqual(smooth['motion']['trajectory_profile'], 'cubic')
        self.assertEqual(smooth['motion']['grip_dwell_s'], .5)
        self.assertTrue(drifty['motion']['continuous_transit'])
        self.assertEqual(drifty['motion']['grip_dwell_s'], 0)
        self.assertEqual(drifty['motion']['gripper_speed_ticks_s'], 700)
        for builder in (app.build_plan, app.build_disassembly_plan):
            a, b = builder(smooth), builder(drifty)
            self.assertEqual([(x.cube,x.name,x.target) for x in a],
                             [(x.cube,x.name,x.target) for x in b])
        self.assertEqual(path.read_bytes(), before)

    def test_profile_dry_run_does_not_open_hardware(self):
        for profile in ('smooth', 'drifty'):
            with patch.object(app, 'Rig') as rig, patch('sys.stdout',new=io.StringIO()):
                self.assertEqual(app.main(['run', '--profile', profile, '--disassemble']), 0)
                rig.assert_not_called()

    def test_lost_final_goal_is_retried_before_advancing(self):
        rig=SimRig(self.scene,[False]);step=app.build_plan(self.scene)[0]
        original=rig.send;dropped=[False];retry=[False]
        def send(name,values):
            if values==step.target:
                if not dropped[0]:
                    dropped[0]=True
                    rig.pos[name]=[v+100 for v in values]
                    return
                retry[0]=True
            original(name,values)
        rig.send=send
        app.execute_step(rig,step)
        self.assertTrue(dropped[0])
        self.assertTrue(retry[0])
        self.assertEqual(rig.pos[step.arm],step.target)

    def test_full_transit_speed_never_leaks_into_precision_steps(self):
        self.scene['motion']['transit_speed_ticks_s']=4095
        rig=SimRig(self.scene,[False]);plan=app.build_plan(self.scene)
        for step in plan[:3]:
            app.execute_step(rig,step)
            expected=4095 if step.name=='pick_approach' else self.scene['motion']['speed_ticks_s']
            self.assertEqual(rig.command_speed,expected)

    def test_loaded_shoulder_offset_is_accepted_only_after_it_stops(self):
        rig=SimRig(self.scene,[False]);original=rig.sample
        self.scene['arms']['left']['position_error_ignored_ids']=[2]
        def sample():
            fs=original();f=fs['left'][1];f.position+=100
            if self.clock.now<2:
                f.speed=100;f.moving=1
            return fs
        rig.sample=sample
        app.execute_step(rig,app.build_plan(self.scene)[0])
        self.assertGreaterEqual(self.clock.now,2+self.scene['motion']['settle_s'])

    def test_loaded_shoulder_does_not_bypass_other_joint_position_checks(self):
        self.scene['arms']['left']['position_error_ignored_ids']=[2]
        rig=SimRig(self.scene,[False]);original=rig.sample
        def sample():
            fs=original();fs['left'][2].position+=100;return fs
        rig.sample=sample
        with self.assertRaises(ProtocolError):app.execute_step(rig,app.build_plan(self.scene)[0])

    def test_contact_blocked_gripper_can_complete_after_settling(self):
        self.scene['arms']['left']['position_error_ignored_ids']=[2,6]
        rig=SimRig(self.scene,[False]);original=rig.sample
        def sample():
            fs=original();fs['left'][5].position+=120;return fs
        rig.sample=sample
        app.execute_step(rig,app.build_plan(self.scene)[2])

    def test_resume_starts_at_failed_phase_and_rejects_changed_poses(self):
        plan=app.build_plan(self.scene)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            app.atomic_json(p/'state.json',{'outcome':'error','completed_steps':2,'current':vars(plan[2])})
            app.atomic_json(p/'poses.snapshot.json',self.scene)
            self.scene['arms']['left']['position_error_ignored_ids']=[2,6]
            self.assertEqual(app.resume_index(self.scene,plan,p),2)
            self.scene['cubes'][0]['poses']['pick'][0]+=1
            with self.assertRaises(ValueError):app.resume_index(self.scene,plan,p)

    def test_all_position_offsets_and_stale_moving_flag_are_ignored_on_request(self):
        self.scene['arms']['left']['position_error_ignored_ids']=list(self.scene['arms']['left']['ids'])
        rig=SimRig(self.scene,[False]);original=rig.sample
        def sample():
            fs=original()
            for f in fs['left']:
                f.position+=100
                f.moving=1
            return fs
        rig.sample=sample
        app.execute_step(rig,app.build_plan(self.scene)[2])
        self.assertLess(self.clock.now,self.scene['motion']['waypoint_timeout_s'])

    def test_stuck_motor_prevents_progress(self):
        rig = SimRig(self.scene, [False]); rig.stuck = 'left'
        with self.assertRaisesRegex(ProtocolError, 'цель не достигнута'):
            app.execute_step(rig, app.build_plan(self.scene)[0])
        self.assertEqual(rig.pos['right'], self.scene['arms']['right']['park'])

    def test_other_arm_must_be_parked_before_any_command(self):
        rig = SimRig(self.scene, [False]); rig.pos['right'][0] += 200
        with self.assertRaisesRegex(ProtocolError, 'парковочной'):
            app.execute_step(rig, app.build_plan(self.scene)[0])
        self.assertEqual(rig.commands, [])

    def test_parking_interlock_uses_position_and_moving_flag_not_speed_reading(self):
        rig = SimRig(self.scene, [False])
        rig.pos['right'][0] += 100
        self.scene['arms']['right']['position_error_ignored_ids'] = list(self.scene['arms']['right']['ids'])
        feedback = rig.sample()
        feedback['right'][0].speed = 150
        app.parked(self.scene, feedback)
        rig.pos['right'][0] += 1
        with self.assertRaisesRegex(ProtocolError, 'парковочной'):
            app.parked(self.scene, rig.sample())
        rig.pos['right'][0] -= 1
        feedback = rig.sample()
        feedback['right'][0].moving = 1
        feedback['right'][0].speed = 0
        with self.assertRaisesRegex(ProtocolError, 'парковочной'):
            app.parked(self.scene, feedback)

    def test_voltage_fault_and_stop_prevent_any_command(self):
        for fault, stop in [(True,False), (False,True)]:
            rig = SimRig(self.scene, [stop]); rig.fault=fault
            with self.assertRaises((ProtocolError, app.Stopped)):
                app.execute_step(rig, app.build_plan(self.scene)[0])
            self.assertFalse(rig.commands)

    def test_startup_parks_both_arms_sequentially_from_different_positions(self):
        rig=SimRig(self.scene,[False]);rig.pos['left'][0]=1800;rig.pos['right'][0]=1700
        initial=copy.deepcopy(rig.pos)
        observed=[]
        with patch('sys.stdout',new=io.StringIO()):
            app.activate_at_current_pose(rig)
            app.park_hands(rig,on_done=lambda step:observed.append((step.arm,copy.deepcopy(rig.pos))))
        self.assertEqual(rig.commands[:2],[('left',initial['left']),('right',initial['right'])])
        self.assertEqual(observed[0][0],'left')
        self.assertEqual(observed[0][1]['left'],self.scene['arms']['left']['park'])
        self.assertEqual(observed[0][1]['right'],initial['right'])
        self.assertEqual(observed[1][1]['right'],self.scene['arms']['right']['park'])

    def test_resume_does_not_park_the_arm_holding_a_cube(self):
        plan=app.build_plan(self.scene);rig=SimRig(self.scene,[False])
        rig.pos['left']=list(plan[1].target)
        with tempfile.TemporaryDirectory() as d,patch.object(app,'Rig',return_value=rig),patch('sys.stdout',new=io.StringIO()):
            source=Path(d)/'old';source.mkdir()
            app.atomic_json(source/'state.json',{'outcome':'error','completed_steps':2,'current':vars(plan[2])})
            app.atomic_json(source/'poses.snapshot.json',self.scene)
            output=Path(d)/'resume'
            app.run_cycle(self.scene,output,[False],source)
            events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e['event']=='parking_start' for e in events))
            self.assertEqual(next(e for e in events if e['event']=='step_start')['name'],'grasp')
            self.assertEqual(json.loads((output/'state.json').read_text())['outcome'],'completed')

    def test_stop_during_startup_parking_never_starts_picking(self):
        stop=[False];rig=SimRig(self.scene,stop);rig.pos['left'][0]=1800
        original=rig.send
        def send(name,values):
            original(name,values)
            if name=='left' and values[0]!=1800:stop[0]=True
        rig.send=send
        with tempfile.TemporaryDirectory() as d,patch.object(app,'Rig',return_value=rig),patch('sys.stdout',new=io.StringIO()):
            output=Path(d)/'cycle'
            with self.assertRaises(app.Stopped):app.run_cycle(self.scene,output,stop)
            state=json.loads((output/'state.json').read_text())
            self.assertEqual(state['current']['cube'],'startup')
            self.assertEqual(state['completed_steps'],0)
            events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e['event']=='step_start' for e in events))

    def test_full_cycle_has_park_between_arms_and_records_state(self):
        rig = SimRig(self.scene, [False])
        with tempfile.TemporaryDirectory() as d, patch.object(app,'Rig',return_value=rig), patch('sys.stdout',new=io.StringIO()):
            output = Path(d)/'cycle'
            app.run_cycle(self.scene,output,[False])
            state=json.loads((output/'state.json').read_text())
            self.assertEqual(state['outcome'], 'completed')
            self.assertEqual(state['completed_steps'], 36)
            self.assertTrue((output/'poses.snapshot.json').exists())
        for (a,pa),(b,pb) in zip(rig.commands[2:],rig.commands[3:]):
            if a!=b:
                self.assertEqual(pa, self.scene['arms'][a]['park'])
        for name in ['left','right']:
            self.assertEqual(rig.pos[name],self.scene['arms'][name]['park'])

    def test_incomplete_scene_never_opens_ports(self):
        self.scene['arms']['left']['park']=None
        with tempfile.TemporaryDirectory() as d, patch.object(app, 'Rig') as rig:
            with self.assertRaises(ValueError):
                app.run_cycle(self.scene,Path(d)/'cycle',[False])
            rig.assert_not_called()

    def test_failure_checkpoint_does_not_claim_completion(self):
        rig = SimRig(self.scene, [False]); rig.stuck='left'
        with tempfile.TemporaryDirectory() as d, patch.object(app,'Rig',return_value=rig), patch('sys.stdout',new=io.StringIO()):
            output=Path(d)/'cycle'
            with self.assertRaises(ProtocolError):
                app.run_cycle(self.scene,output,[False])
            state=json.loads((output/'state.json').read_text())
            self.assertEqual(state['outcome'],'error')
            self.assertEqual(state['completed_steps'],0)
            self.assertEqual(state['current']['cube'],'cube_5')

    def test_teaching_saves_actual_follower_pose_and_backup(self):
        rig=SimRig(self.scene,[False])
        rig.buses={name:MemoryBus(arm) for name,arm in self.scene['arms'].items()}
        rig.pos['left']=[2100]*5+[1300]
        terminal=io.StringIO('\n')
        terminal.isatty=lambda:True
        def ready(*a):
            return ([terminal],[],[]) if self.clock.now>.35 else ([],[],[])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'poses.json';app.atomic_json(path,self.scene)
            with patch.object(app,'Rig',return_value=rig), patch.object(app.sys,'stdin',terminal), patch.object(app.select,'select',side_effect=ready), patch('sys.stdout',new=io.StringIO()):
                app.teach(self.scene,path,'left','left.park','passive',[False])
            saved=json.loads(path.read_text())
            self.assertEqual(saved['arms']['left']['park'],[2100]*5+[1300])
            self.assertEqual(saved['taught']['left.park']['mode'],'passive')
            self.assertEqual(len(list(Path(d).glob('poses.before_teach_*.json'))),1)
            self.assertFalse(rig.commands)

    def test_teaching_arm_does_not_insert_return_to_park_step(self):
        rig=SimRig(self.scene,[False])
        rig.buses={name:MemoryBus(arm) for name,arm in self.scene['arms'].items()}
        terminal=io.StringIO('\n'*8+'q\n')
        terminal.isatty=lambda:True
        def ready(*a):
            return ([terminal],[],[]) if self.clock.now>.35 else ([],[],[])
        output=io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'poses.json';app.atomic_json(path,self.scene)
            with patch.object(app,'Rig',return_value=rig), patch.object(app.sys,'stdin',terminal), patch.object(app.select,'select',side_effect=ready), patch('sys.stdout',output):
                app.teach(self.scene,path,'right',None,'passive',[False],'cube_4.pick_approach')
            saved=json.loads(path.read_text())
            self.assertFalse(any(key.endswith('.return_park') for key in saved['taught']))
            self.assertEqual(sum(key.startswith('cube_4.') for key in saved['taught']),8)
            self.assertNotIn('return_park',output.getvalue())

    def test_no_hardware_for_dry_run(self):
        with patch.object(app,'Rig') as rig, patch('sys.stdout',new=io.StringIO()):
            self.assertIn(app.main(['run']),(0,2))
            rig.assert_not_called()

class MemoryBus:
    def __init__(self, arm):
        self.arm = arm
        self.position = {i: 2000 for i in arm['ids']}
        self.goals = dict(self.position)
        self.torque = {i: 0 for i in arm['ids']+arm['leader_ids']}
        self.writes = []
        self.bad_offset = False
    def read(self, i, address, size):
        c = self.arm['calibration'].get(str(i), {})
        data = {3: (777).to_bytes(2,'little'), 33: b'\0', 18: bytes([12]),
                40: bytes([self.torque[i]])}
        if i in self.position:
            data.update({31: (c['offset_raw']+int(self.bad_offset)).to_bytes(2,'little'),
                         42: self.goals[i].to_bytes(2,'little'),
                         9: b''.join(v.to_bytes(2,'little') for v in self.arm['limits'][str(i)])})
        return data[address],0
    def feedback(self,i):
        return Feedback(i,self.position[i],0,8,25,0,0,0,0)
    def move(self,values,speed,acceleration):
        self.writes.append(dict(values));self.goals.update(values);self.position.update(values)
    def send(self, payload):
        i=payload[2]
        self.writes.append({i:payload[5:]})
        self.torque[i]=payload[6]


class HardwareBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.scene=fixture()
        self.rig=app.Rig(self.scene,[False])
        self.rig.buses={name:MemoryBus(arm) for name,arm in self.scene['arms'].items()}
    def test_preflight_reads_only_and_never_writes_a_leader(self):
        with patch.object(app.time,'sleep'):
            self.rig.preflight()
            self.assertFalse(any(b.writes for b in self.rig.buses.values()))
            self.rig.enable('left',[2000]*6)
            for name,bus in self.rig.buses.items():
                self.assertTrue(all(set(w)<=set(self.scene['arms'][name]['ids']) for w in bus.writes))
            self.assertFalse(self.rig.buses['right'].writes)
    def test_teaching_gripper_overload_is_rate_limited_and_voltage_still_fails(self):
        bus=self.rig.buses['left'];original=bus.feedback
        def feedback(i):
            f=original(i)
            if i==6:f.status=0x20
            return f
        bus.feedback=feedback
        with self.assertRaises(ProtocolError):self.rig.sample()
        self.rig.teach_gripper=True
        self.rig.sample()
        with patch.object(app.time,'monotonic',return_value=10):
            self.rig.send('left',[2000]*6)
            self.rig.send('left',[2000]*6)
        self.assertIn(6,bus.writes[-2])
        self.assertNotIn(6,bus.writes[-1])
        with patch.object(app.time,'monotonic',return_value=10.5):
            self.rig.send('left',[2000]*6)
        self.assertIn(6,bus.writes[-1])
        def voltage(i):
            f=original(i)
            if i==6:f.status=0x21
            return f
        bus.feedback=voltage
        with self.assertRaises(ProtocolError):self.rig.sample()

    def test_unused_leader_fault_does_not_block_autonomous_cycle(self):
        bus=self.rig.buses['left'];original=bus.read
        def read(i,address,size):
            data,status=original(i,address,size)
            return data,1 if i==19 else status
        bus.read=read
        self.rig.preflight()
        self.assertFalse(any(b.writes for b in self.rig.buses.values()))
        self.rig.teach_gripper=True
        with self.assertRaises(ProtocolError):self.rig.preflight()

    def test_changed_offset_blocks_preflight_without_writes(self):
        self.rig.buses['left'].bad_offset=True
        with self.assertRaises(ProtocolError):self.rig.preflight()
        self.assertFalse(any(b.writes for b in self.rig.buses.values()))
    def test_faulty_motor_does_not_prevent_holding_the_remaining_motors(self):
        bus=self.rig.buses['left']
        original=bus.feedback
        def feedback(i):
            if i==2:
                raise ProtocolError('ID 2: timeout reading register 56')
            return original(i)
        bus.feedback=feedback
        with self.assertRaises(ProtocolError):
            self.rig.hold('left')
        self.assertEqual(set(bus.writes[-1]),{1,3,4,5,6})

    def test_shutdown_does_not_enable_an_arm_that_was_only_read(self):
        self.rig.__exit__(None,None,None)
        self.assertFalse(any(b.writes for b in self.rig.buses.values()))


if __name__ == '__main__':
    unittest.main()
