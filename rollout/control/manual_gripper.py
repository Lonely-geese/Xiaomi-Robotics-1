"""X toggles only the left gripper in READY and paused human-recording waits."""
import logging
import threading
import time
import sys

import numpy as np

LOG = logging.getLogger('recap.manual_gripper')


class MappingLogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.replace('index trigger=left gripper', 'X=left gripper toggle (READY/HUMAN only); trigger disabled')
            record.msg = record.msg.replace('index trigger = gripper while waiting', 'X = gripper toggle while waiting/paused')
        return True


class ManualGripper:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.mode = None
        self.pending = None
        self.last_press = float('-inf')
        self.target = None

    def enter(self, mode, control):
        with self.lock:
            self.mode = None
            self.pending = None
            if mode is None:
                return
            if control.tracking:
                raise RuntimeError('Manual gripper requires stopped policy tracking')
            _, _, opening, *_ = control.feedback_with_metadata()
            control._validate_grippers(opening)
            # An intermediate grasp counts as closed; the next press releases it.
            self.target = 0.09 if opening > .045 else 0.0
            self.mode = mode
        LOG.warning('X gripper enabled in %s; trigger disabled; current opening %.4f m', mode, opening)

    def leave(self):
        with self.lock:
            self.mode = None
            self.pending = None

    def press(self):
        with self.lock:
            now = self.clock()
            if self.mode is None:
                LOG.info('X ignored: gripper is available only in READY or HUMAN pause')
                return
            if now - self.last_press < .25 or self.pending is not None:
                return
            self.last_press = now
            self.pending = now

    def poll(self, control):
        with self.lock:
            if self.pending is None or self.mode is None:
                return
            pressed, self.pending = self.pending, None
            if self.clock() - pressed > .3:
                LOG.warning('Expired X event discarded')
                return
            if control.tracking:
                raise RuntimeError('Refusing manual gripper command during policy tracking')
            # Require current, valid feedback. Never replay stale button commands.
            control.feedback_with_metadata()
            target = 0.0 if self.target == 0.09 else 0.09
            target, _ = control._validate_grippers(target)
            control.client.send_cmd({'leftarm_gripper_cmd_pos': np.array([target], np.float32)})
            # Same latch reset as the existing manual set_grippers(), after ACK.
            control._snap_phase = 'closing'
            control._snap_release_count = 0
            self.target = target
            LOG.warning('X gripper %s: left target %.3f m (%s)',
                        'CLOSE' if target == 0 else 'OPEN', target, self.mode)


def install(runner):
    """Install before recording so verdict and handback waits disable X immediately."""
    manual = ManualGripper()
    runner.LOG.addFilter(MappingLogFilter())
    controls = sys.modules.get('deeptouch_vr_controls')
    if controls is not None:
        controls.LOG.addFilter(MappingLogFilter())
    state = {'control': None}
    original_snap = runner.install_gripper_snap

    def install_snap(stable, snap_m):
        original_snap(stable, snap_m)
        base = stable.CartesianHAL

        class HAL(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                state['control'] = self

        stable.CartesianHAL = HAL

    runner.install_gripper_snap = install_snap
    original_wait = runner.VrSignals.wait

    def wait(signals, accepted, on_idle=None, poll_s=.02):
        mode = ('READY' if set(accepted) == {'start', 'home'} else
                'HUMAN' if set(accepted) == {'pause', 'success', 'failure', 'home'} else None)
        control = state['control']
        manual.enter(mode, control)

        def idle():
            # Trigger is never populated, so legacy follow_trigger is inert.
            if on_idle is not None:
                on_idle()
            if mode is not None:
                manual.poll(control)

        try:
            return original_wait(signals, accepted, idle, poll_s)
        finally:
            manual.leave()

    runner.VrSignals.wait = wait
    runner.VrSignals.request_critical_start = lambda signals: manual.press()
    runner.VrSignals.set_left_index_trigger = lambda signals, value: None
    return manual
