import logging
from types import SimpleNamespace
import unittest

import numpy as np

from common import SOURCE, load
from manual_gripper import ManualGripper, install
from recording import install_handover, rx


class FakeHAL:
    def __init__(self):
        self.tracking = False
        self.opening = .09
        self.sent = []
        self.fail_send = False
        self.client = SimpleNamespace(send_cmd=self.send)
        self._snap_phase = 'holding'
        self._snap_release_count = 2

    def send(self, cmd):
        if self.fail_send:
            raise RuntimeError('send failed')
        self.sent.append(cmd)

    def feedback_with_metadata(self):
        return None, None, self.opening, .03, 0, 0, 0

    def _validate_grippers(self, target):
        if not np.isfinite(target) or not 0 <= target <= .09:
            raise ValueError('invalid gripper feedback/target')
        return target, .03


class GripperTests(unittest.TestCase):
    def setUp(self):
        self.time = 10.
        self.manual = ManualGripper(lambda: self.time)
        self.hal = FakeHAL()

    def press(self):
        self.manual.press()
        self.manual.poll(self.hal)

    def test_toggle_holds_and_only_commands_left(self):
        self.manual.enter('READY', self.hal)
        self.assertEqual(self.hal.sent, [])
        self.press()
        self.assertEqual(set(self.hal.sent[0]), {'leftarm_gripper_cmd_pos'})
        self.assertAlmostEqual(float(self.hal.sent[0]['leftarm_gripper_cmd_pos'][0]), 0.)
        self.manual.poll(self.hal)
        self.assertEqual(len(self.hal.sent), 1)
        self.time += .3
        self.press()  # feedback can lag; toggle the acknowledged target
        self.assertAlmostEqual(float(self.hal.sent[1]['leftarm_gripper_cmd_pos'][0]), .09)
        self.assertEqual(self.hal._snap_phase, 'closing')

    def test_pause_preserves_grasp_until_x(self):
        self.hal.opening = .02
        self.manual.enter('HUMAN', self.hal)
        self.assertEqual(self.hal.sent, [])
        self.press()
        self.assertAlmostEqual(float(self.hal.sent[0]['leftarm_gripper_cmd_pos'][0]), .09)

    def test_disabled_debounce_and_expiry(self):
        self.press()
        self.manual.enter('READY', self.hal)
        self.manual.poll(self.hal)
        self.assertEqual(self.hal.sent, [])
        self.press(); self.press()
        self.assertEqual(len(self.hal.sent), 1)
        self.time += .3
        self.manual.press(); self.time += .4; self.manual.poll(self.hal)
        self.assertEqual(len(self.hal.sent), 1)
        self.time += .3
        self.manual.press(); self.manual.leave(); self.manual.enter('HUMAN', self.hal)
        self.manual.poll(self.hal)
        self.assertEqual(len(self.hal.sent), 1)

    def test_no_state_commit_on_failed_send(self):
        self.manual.enter('HUMAN', self.hal)
        self.hal.fail_send = True
        with self.assertRaises(RuntimeError): self.press()
        self.assertEqual(self.manual.target, .09)
        self.assertEqual(self.hal._snap_phase, 'holding')

    def test_tracking_and_bad_feedback_fail_closed(self):
        self.hal.tracking = True
        with self.assertRaises(RuntimeError):self.manual.enter('HUMAN', self.hal)
        self.hal.tracking = False; self.hal.opening = float('nan')
        with self.assertRaises(ValueError):self.manual.enter('HUMAN', self.hal)
        self.assertEqual(self.hal.sent, [])

    def test_recording_keeps_gripper_only_intervention(self):
        states = np.zeros((20, 30), np.float32)
        states[:, 14] = np.r_[np.full(5, .09), np.linspace(.09, 0, 10), np.zeros(5)]
        frames = dict(states=states, sources=np.full(20, rx.SOURCE_HUMAN),
                      observation_sequence=np.arange(20), hal_time_ns=np.arange(20),
                      jpeg_frames={'test': [b'image'] * 20})
        trimmed, _ = rx.trim_idle_runs(frames, rx.SOURCE_HUMAN)
        self.assertGreater(len(trimmed['states']), 2)
        self.assertGreater(np.ptp(trimmed['states'][:, 14]), .08)
        self.assertTrue(np.all(trimmed['sources'] == rx.SOURCE_HUMAN))

    def test_real_vr_edge_long_hold(self):
        import sys
        sys.path.insert(0, str(SOURCE.parent / 'vr'))
        from deeptouch_vr_controls import VrManualActionMapper
        mapper = VrManualActionMapper()
        events = []
        for i in range(100):
            events.extend(mapper.on_buttons(t=1+i*.02,left_x=1,left_y=0,thumb_x=0,thumb_y=0))
        self.assertEqual([name for name, _ in events], ['critical'])

    def test_wait_layers_disable_after_verdict_and_during_policy(self):
        runner = load('manual_gripper_test_runner', SOURCE)
        runner.install_gripper_snap = lambda stable, snap: None
        manual = install(runner)
        stable = SimpleNamespace(CartesianHAL=FakeHAL)
        runner.install_gripper_snap(stable, .012)
        control = stable.CartesianHAL()
        signals = runner.VrSignals()
        signals.set_left_index_trigger(1.)
        self.assertIsNone(signals.trigger)
        # Real READY wait polls X on the control thread, then clears on Y.
        calls = []
        def idle():
            calls.append(True)
            if len(calls) == 1:signals.request_critical_start()
            else:signals.events.put('start')
        self.assertEqual(signals.wait(('start','home'),idle,.001),'start')
        self.assertEqual(len(control.sent),1)
        signals.request_critical_start();manual.poll(control)
        self.assertEqual(len(control.sent),1)
        # Install the handover wrapper in the same order as production.
        runner.run_policy_segment = lambda *a: ('pause',None)
        install_handover(runner,None,lambda c:None)
        runner.run_policy_segment(stable,None,None,control,signals)
        signals.events.put('success')
        self.assertEqual(signals.wait(('pause','success','failure','home')),'success')
        self.assertIsNone(manual.mode)
        self.assertEqual(len(control.sent),1)


if __name__ == '__main__':unittest.main()
