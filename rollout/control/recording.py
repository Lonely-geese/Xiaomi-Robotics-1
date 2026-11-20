"""LeRobot recording on the pinned RECAP runner; physical drag is operator controlled."""
import json
import fcntl
import logging
from pathlib import Path
import sys
import time

from common import ROOT, load, verify_dependencies

VENDOR = ROOT / 'recording_vendor'
sys.path.insert(0, str(VENDOR))
import record_xr1_async_rollout as rx

LOG = logging.getLogger(__name__)


def wait_handover(control, helper):
    """Read-only confirmation of rest after the operator exits physical drag."""
    if control.tracking:
        raise RuntimeError('Cannot hand over while policy tracking is active')
    anchor, *_ = control.feedback_with_metadata()
    started = time.monotonic()
    checker = helper.ArmStopped(anchor, started, control.stop_limits)
    while time.monotonic() - started < control.stop_limits.timeout:
        pose, joints, _, _, stamp, sequence, _ = control.feedback_with_metadata()
        if checker.update(pose, joints, stamp, sequence, time.monotonic())['confirmed']:
            control.assert_mode(control.client.get_motion_mode())
            LOG.warning('Physical drag handover: fresh measured rest verified')
            return
        time.sleep(1 / control.feedback_hz)
    raise RuntimeError('Physical drag handover did not settle; no resume/reset allowed')


def install_handover(runner, rec, verify_rest):
    """Keep all motion stopped until a deliberate handback after manual intervention."""
    original_segment = runner.run_policy_segment
    original_wait = runner.VrSignals.wait
    state = {'control': None, 'paused': False}

    def segment(stable, args, observations, control, signals, options=None, synced=False):
        state['control'] = control
        result = original_segment(stable, args, observations, control, signals, options, synced)
        if result[0] == 'pause':
            state['paused'] = True
            LOG.warning('HUMAN RECORDING READY: arm stopped. Enable physical drag manually. '
                        'Exit physical drag and release the arm before stick UP to resume or '
                        'LEFT/RIGHT to save and reset.')
        return result

    def wait(signals, accepted, on_idle=None, poll_s=.02):
        event = original_wait(signals, accepted, on_idle, poll_s)
        if state['paused'] and 'pause' in accepted:
            # The recording hook has already saved/discarded a verdict here.
            if event == 'home':
                LOG.warning('Episode discarded. Exit physical drag, release the arm, '
                            'then stick UP to allow reset.')
                original_wait(signals, ('pause',), None, poll_s)
            verify_rest(state['control'])
            state['paused'] = False
        return event

    runner.run_policy_segment = segment
    runner.VrSignals.wait = wait


class Session:
    def __init__(self, rec, writer, lock):
        self.rec, self.writer, self.lock = rec, writer, lock

    def close(self):
        self.rec.abort('controller exited; incomplete episode discarded')
        try:
            if self.rec.recorder is not None:
                self.rec.recorder.close()
        finally:
            try:
                self.writer.close()
                LOG.warning('Recording closed: %s', self.writer.summary())
            finally:
                self.lock.close()


def enable_recording(runner, output, identity):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.recording.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError('Another recorder is already writing this dataset')
    contract = {'model': identity, 'action_shift': 3, 'pause_mode': 'human',
                'physical_drag': 'operator_enabled; exit before stick up',
                'task': 'task_16', 'async_hz': 7.5,
                'stored_cameras': ['head', 'left_wrist', 'physical_right_wrist'],
                'model_cameras': ['head', 'left_wrist', 'left_wrist_center_crop_256']}
    provenance = output / 'recap_recording.json'
    if provenance.exists() and json.loads(provenance.read_text()) != contract:
        raise RuntimeError('Recording dataset belongs to a different model/configuration')
    provenance.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    opts, _ = rx._parse(['--steam-out', str(output)])
    writer = rx.LeRobotV21RolloutWriter(output, action_shift=3, crf=opts.steam_crf, gop=opts.steam_gop)
    rec = rx.Recording(runner, opts, writer)
    rx.install(runner, rec)
    profile = verify_dependencies()
    helper = load('recap_recording_stop', profile.STOP_HELPER)
    install_handover(runner, rec, lambda control: wait_handover(control, helper))
    print(f'RECAP RECORDING ENABLED: {output}; LEFT=save success, RIGHT=save failure, '
          'DOWN=discard; minimum 30 frames; after manual drag, release before LEFT/RIGHT to save and reset', flush=True)
    return Session(rec, writer, lock)
