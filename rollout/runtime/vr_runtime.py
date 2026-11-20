#!/usr/bin/env python3
"""Official XR1 asynchronous robot client for the crop server (xr1_crop_server, port 10092).

Same structure as the USB bundle's ``entry.py client`` (``start_robot.sh --server-port 10091``): it runs
the pinned stable runner ``tianji_adapter/experiments/pre_7dof_interpolation_1257_smooth_async/run_robot.py``
of ``$XR1_USB_BUNDLE`` (official ``action_prefix`` async inference, one continuous online-tracking
stream, latency checks) with the bundle's ``hal_adapter`` stop compatibility. Nothing in the bundle is
modified; only this process patches the loaded module. Additions:

* model pin: ``XR1_MODEL_MANIFEST`` = the crop server's ``profiles/*/model_manifest.json`` (newest by
  default, ``--manifest`` to choose), and the handshake must also report the same ``left_wrist_crop`` /
  ``zero_state``;
* gripper-driven execution window: each response executes at most ``--open-execute-steps`` new targets
  while the measured gripper opening of its request observation is above ``--closed-gripper-m``, else
  ``--closed-execute-steps`` (the bundle's ``--execute-steps`` with two values). The committed prefix is
  always kept, so new steps per inference = replanning period;
* optional ``--gripper-snap-close-m``: gripper targets <= this are sent to HAL as 0 (full grip). Only the
  HAL command changes; the model, prefix and handover checks see the original targets;
* Quest VR left-controller episodes (same mapping as the RL rollout, ``deeptouch_vr_controls``): reset to
  the start pose -> Y starts (index trigger drives the gripper while waiting) -> stick left = success,
  right = failure, down = home (abort), up = pause / resume. Each stop drains the HAL command, verifies
  the soft stop and drops the inference connection, so no in-flight chunk survives. Outcomes go to
  ``runs/episodes.jsonl``. ``--no-vr`` keeps the official single continuous run (Ctrl-C to stop);
* async -> sync switch (VR episodes): ``--sync-when-closed`` (measured gripper <= ``--closed-gripper-m``)
  and/or ``--sync-near-xyz X Y Z`` (measured left EEF within ``--sync-near-radius-m`` of that HAL world
  point). Once either holds, the arm stops (verified) and the rest of the episode runs the official sync
  mode: observe at rest, execute ``--sync-execute-steps`` targets, settle, repeat. Async prefix
  conditioning delays a correction by the prefix (0.4 s at 6 steps / 15 Hz) and favours continuing the
  committed motion; sync keeps the fine adjustments for insertion at the cost of a stop between chunks.

Crop and zero-state are applied by the server, so the request is the official one (full images).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = Path(__file__).resolve().parent
USB_TASK = 'Pick up the USB cable and insert it into the USB port.'
DEFAULT_PORT = 10092
HORIZON = 30
VR_CONTROLS = BUNDLE / 'vr'
STOP_EVENTS = ('success', 'failure', 'home', 'pause')
LOG = logging.getLogger('xr1.usb.leftcrop.client')


def newest_manifest() -> Path:
    manifests = sorted((HERE / 'profiles').glob('*/model_manifest.json'), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise SystemExit(f'no crop profile manifest under {HERE / "profiles"}; start the crop server first')
    return manifests[-1]


class GripperExecutionWindow:
    """Keep the committed prefix, then at most N new targets; N depends on the measured gripper."""

    def __init__(self, predictor, *, open_steps: int, closed_steps: int, closed_gripper_m: float):
        self.predictor = predictor
        self.open_steps = open_steps
        self.closed_steps = closed_steps
        self.closed_gripper_m = closed_gripper_m
        self.force_steps: int | None = None  # sync segments: fixed count regardless of the gripper

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def receive(self, timeout):
        return self.select(self.predictor.receive(timeout))

    def select(self, result):
        request, chunk, metadata, latency = result
        prefix_length = 0 if request.prefix is None else len(request.prefix)
        gripper = float(np.asarray(request.raw['state']['left_gripper'], dtype=np.float64).reshape(-1)[0])
        closed = gripper <= self.closed_gripper_m
        steps = self.closed_steps if closed else self.open_steps
        if self.force_steps is not None:
            steps = self.force_steps
        end = min(prefix_length + steps, len(chunk['left_pose']))
        LOG.info('request %s: gripper %.4f m -> %s, prefix %d + %d new targets (latency %.0f ms)',
                 request.serial, gripper, 'closed' if closed else 'open', prefix_length, end - prefix_length,
                 1e3 * float(latency))
        # Transport validation and server checks already saw the full prediction; never mutate it.
        selected = dict(chunk)
        for key in ('left_pose', 'left_gripper'):
            selected[key] = chunk[key][:end]
        return request, selected, metadata, latency


def install_window(stable, options) -> None:
    original_runner = stable.Runner

    class Runner(original_runner):
        def __init__(self, args, observations, hal, predictor):
            super().__init__(args, observations, hal, GripperExecutionWindow(
                predictor, open_steps=options.open_execute_steps, closed_steps=options.closed_execute_steps,
                closed_gripper_m=options.closed_gripper_m))
            self.stop_check = None  # set while a VR episode may interrupt execution
            self.switch_check = None  # async segment ends once this returns a reason for the feedback

        def execute(self, first_result):
            # Also cover callers that supply the first prediction directly.
            return super().execute(self.predictor.select(first_result))

        def feedback(self):
            # Called every servo frame of the execution loop: the earliest safe point to leave it.
            stop_check = getattr(self, 'stop_check', None)
            if stop_check is not None:
                reason = stop_check()
                if reason is not None:
                    raise EpisodeStop(reason)
            sample = super().feedback()
            switch_check = getattr(self, 'switch_check', None)
            if switch_check is not None:
                reason = switch_check(sample[0], sample[2])
                if reason is not None:
                    raise EpisodeStop('switch', reason)
            return sample

    stable.Runner = Runner


class EpisodeStop(BaseException):
    """Operator ended or paused the running policy (raised inside Runner.execute)."""

    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class VrSignals:
    """Runtime target of deeptouch_vr_controls.start_vr_manual_controls (called from its thread)."""

    def __init__(self):
        self.events: queue.Queue[str] = queue.Queue()
        self.home_key = SimpleNamespace(requested=threading.Event())
        self.trigger: float | None = None
        self.shutdown = threading.Event()

    # --- API used by deeptouch_vr_controls -------------------------------------------------------
    def shutdown_requested(self) -> bool:
        return self.shutdown.is_set()

    def set_left_index_trigger(self, value: float) -> None:
        self.trigger = float(value)

    def request_full_task_start(self) -> None:
        self.events.put('start')

    def request_critical_start(self) -> None:
        LOG.info('VR X (critical) has no meaning here; press Y to start')

    def toggle_drag_teach(self) -> None:
        self.events.put('pause')

    def mark_manual_success(self) -> None:
        self.events.put('success')

    def mark_manual_failure(self) -> None:
        self.events.put('failure')

    # --- consumer side --------------------------------------------------------------------------
    def pop(self) -> str | None:
        if self.home_key.requested.is_set():
            self.home_key.requested.clear()
            return 'home'
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def pop_stop(self) -> str | None:
        """Next stop event; start presses during a run are ignored."""
        while True:
            event = self.pop()
            if event is None or event in STOP_EVENTS:
                return event

    def wait(self, accepted: tuple[str, ...], on_idle=None, poll_s: float = 0.02) -> str:
        while True:
            event = self.pop()
            if event in accepted:
                return event
            if event is not None:
                LOG.info('ignoring VR %s here (waiting for %s)', event, '/'.join(accepted))
            if on_idle is not None:
                on_idle()
            time.sleep(poll_s)

    def clear(self) -> None:
        while self.pop() is not None:
            pass


def sync_switch_reason(options, pose, gripper) -> str | None:
    """Why this measured state should run sync (None = stay async)."""
    if options is None:
        return None
    if getattr(options, 'sync_when_closed', False) and float(gripper) <= options.closed_gripper_m:
        return f'gripper {float(gripper):.4f} m <= {options.closed_gripper_m:.4f} m'
    target = getattr(options, 'sync_near_xyz', None)
    if target is not None:
        distance = float(np.linalg.norm(np.asarray(pose, dtype=np.float64)[:3] - np.asarray(target, dtype=np.float64)))
        if distance <= options.sync_near_radius_m:
            return f'EEF {distance * 100:.1f} cm from {list(target)} (<= {options.sync_near_radius_m * 100:.1f} cm)'
    return None


def run_policy_segment(stable, args, observations, control, signals: VrSignals, options=None,
                       synced: bool = False) -> tuple[str, str | None]:
    """Official execution from the current pose until a VR stop event; arm verified at rest after.

    Returns (reason, detail). Runs sync when ``synced`` or the switch condition already holds;
    otherwise async, ending with ('switch', why) as soon as the condition holds.
    """
    switching = (options is not None and args.inference_mode == 'async'
                 and (getattr(options, 'sync_when_closed', False) or getattr(options, 'sync_near_xyz', None) is not None))
    switch_check = None
    why = None
    if switching:
        pose, _, gripper, *_ = control.feedback_with_metadata()
        why = 'already switched this episode' if synced else sync_switch_reason(options, pose, gripper)
        if why is not None:
            args = copy.copy(args)
            args.inference_mode = 'sync'
        else:
            switch_check = lambda p, g: sync_switch_reason(options, p, g)  # noqa: E731
    if getattr(options, 'async_control_hz', None) is not None and args.inference_mode == 'async':
        # Slower playback of the same 30 Hz model targets while async (sync keeps --control-hz).
        args = copy.copy(args)
        args.control_hz = options.async_control_hz
    if options is not None:
        LOG.warning('policy segment: %s at %g Hz%s', args.inference_mode, args.control_hz,
                    '' if not switching or why is None else f' ({why})')
    observations.latest(min_stamp=time.monotonic())  # observe after the arm is still
    predictor = stable.Predictor(stable.PolicyClient(args.server_host, args.server_port, args.request_timeout))
    runner = stable.Runner(args, observations, control, predictor)
    if switching and args.inference_mode == 'sync':
        runner.predictor.force_steps = options.sync_execute_steps
    reason, detail = 'chunks_done', None
    try:
        first = runner.initial_for_execution()
        early = signals.pop_stop()
        if early is not None:
            return early, None
        runner.stop_check = signals.pop_stop
        runner.switch_check = switch_check
        runner.execute(first)
    except EpisodeStop as stop:
        reason, detail = stop.reason, stop.detail
    finally:
        runner.stop_check = None
        runner.switch_check = None
        try:
            runner.close()  # drain the pipelined HAL command
            if control.tracking:
                control.stop_tracking()  # hal_adapter: soft stop + measured rest
        finally:
            runner.save_trace()
            predictor.close()  # a still running request dies with its connection
    return reason, detail


def run_vr_episodes(stable, args, options) -> None:
    """stable.main's connection setup, then VR-driven episodes instead of one endless run."""
    sys.path.insert(0, str(VR_CONTROLS))
    from deeptouch_vr_controls import gripper_opening_from_index_trigger, start_vr_manual_controls

    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signals = VrSignals()
    if start_vr_manual_controls(signals, lambda: signals.home_key) is None:
        raise SystemExit('VR controller unavailable (see log); use --no-vr for the official continuous run')
    record_path = HERE / 'runs' / 'episodes.jsonl'
    record_path.parent.mkdir(parents=True, exist_ok=True)
    observations = control = None
    try:
        observations = stable.ObservationSource(args.hal_host, args.hal_port, args.max_observation_age)
        observations.start()
        control = stable.CartesianHAL(args.hal_host, args.hal_port, args.feedback_poll_hz, args.max_feedback_age)
        control.connect()
        state, _, _ = observations.latest()
        warm = stable.PolicyClient(args.server_host, args.server_port, args.request_timeout)
        try:
            if not warm.metadata.get('model_warmed', False):
                started = time.monotonic()
                warm.request(stable.build_raw(state, args.task), 0)  # never dispatched
                LOG.info('stationary model warm-up %.3fs before control acquisition', time.monotonic() - started)
        finally:
            warm.close()
        control.acquire()
        episode = 0
        while True:
            control.reset_for_inference(args.home_velocity, args.reset_timeout,
                                        args.reset_joint_tolerance_deg, args.reset_gripper_tolerance_m)
            signals.clear()
            LOG.warning('READY episode %d: Y = start | stick left = success, right = failure, '
                        'down = home, up = pause/resume | index trigger = gripper while waiting', episode + 1)
            gripper = [float(stable.INITIAL_GRIPPER_M)]

            def follow_trigger():
                if signals.trigger is None:
                    return
                opening = gripper_opening_from_index_trigger(signals.trigger, stable.INITIAL_GRIPPER_M)
                if abs(opening - gripper[0]) > 0.002:
                    control.set_grippers(opening)
                    gripper[0] = opening

            if signals.wait(('start', 'home'), on_idle=follow_trigger) == 'home':
                continue  # re-run the reset
            episode += 1
            started = time.time()
            pauses = 0
            synced = False  # latched: once sync, the rest of the episode stays sync
            switched_after_s = None
            LOG.warning('episode %d started', episode)
            while True:
                reason, detail = run_policy_segment(stable, args, observations, control, signals, options, synced)
                if reason == 'switch':
                    synced = True
                    switched_after_s = round(time.time() - started, 2)
                    LOG.warning('episode %d: %s -> continuing in sync mode', episode, detail)
                    continue
                if reason != 'pause':
                    break
                pauses += 1
                LOG.warning('episode %d paused (arm at rest): stick up = resume, left = success, '
                            'right = failure, down = home', episode)
                reason = signals.wait(('pause', 'success', 'failure', 'home'))
                if reason != 'pause':
                    break
                LOG.warning('episode %d resumed from the current pose', episode)
            row = {'episode': episode, 'outcome': reason, 'duration_s': round(time.time() - started, 2),
                   'pauses': pauses, 'sync_after_s': switched_after_s, 'wall_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'profile': os.environ.get('XR1_MODEL_MANIFEST')}
            with record_path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            LOG.warning('episode %d: %s after %.1fs (recorded in %s)', episode, reason, row['duration_s'], record_path)
    except KeyboardInterrupt:
        LOG.info('interrupted; soft-stopping Cartesian tracking')
    finally:
        signals.shutdown.set()
        if control is not None:
            control.close()
        if observations is not None:
            observations.close()


def install_crop_pin(stable, manifest: dict) -> None:
    original_client = stable.PolicyClient

    class PolicyClient(original_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # official identity/protocol checks (XR1_MODEL_MANIFEST)
            for key, default in (('left_wrist_crop', None), ('zero_state', 'off')):
                if self.metadata.get(key, default) != manifest.get(key, default):
                    raise RuntimeError(f'crop server handshake {key}={self.metadata.get(key, default)!r}, '
                                       f'manifest expects {manifest.get(key, default)!r}')

    stable.PolicyClient = PolicyClient


def install_gripper_snap(stable, snap_m: float) -> None:
    base = stable.CartesianHAL

    class HAL(base):
        def send_command(self, pose, left_gripper=None):
            if left_gripper is not None and float(left_gripper) <= snap_m:
                left_gripper = 0.0
            return super().send_command(pose, left_gripper)

    stable.CartesianHAL = HAL


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, description=__doc__)
    parser.add_argument('--manifest', type=Path, default=None,
                        help='crop profile model_manifest.json (default: newest under profiles/)')
    parser.add_argument('--open-execute-steps', type=int, default=None,
                        help='new targets per inference while the gripper is open (default: 30 - prefix)')
    parser.add_argument('--closed-execute-steps', type=int, default=10,
                        help='new targets per inference once the gripper is closed (default: 10)')
    parser.add_argument('--closed-gripper-m', type=float, default=0.02,
                        help='measured left gripper opening (m) at or below which it counts as closed')
    parser.add_argument('--gripper-snap-close-m', type=float, default=None,
                        help='optional: send gripper 0 for targets <= this opening (m), e.g. 0.008')
    parser.add_argument('--sync-when-closed', action='store_true',
                        help='VR episodes: switch async -> official sync (observe at rest) once the gripper closes')
    parser.add_argument('--sync-near-xyz', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                        help='VR episodes: switch to sync once the left EEF is within --sync-near-radius-m of '
                             'this HAL world point (m), e.g. the port: 0.47375 0.20553 -0.21435')
    parser.add_argument('--sync-near-radius-m', type=float, default=0.10)
    parser.add_argument('--async-control-hz', type=float, default=None,
                        help='VR episodes: keyframe playback Hz for async segments (e.g. 7.5 = half speed); '
                             'sync segments keep --control-hz; must divide --servo-hz')
    parser.add_argument('--sync-execute-steps', type=int, default=10,
                        help='new targets per inference after switching to sync (default: 10)')
    parser.add_argument('--no-vr', action='store_true',
                        help='official single continuous run without VR episodes (Ctrl-C to stop)')
    options, remaining = parser.parse_known_args(argv)
    if '-h' in remaining or '--help' in remaining:
        parser.print_help()

    manifest_path = (options.manifest or newest_manifest()).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    os.environ['XR1_MODEL_MANIFEST'] = str(manifest_path)  # read by the stable runner's identity pin

    sys.path.insert(0, str(BUNDLE))
    import model_profile as profile  # read-only: STABLE / STOP_HELPER paths, digests and loader
    for name, expected in profile.DEPENDENCIES.items():
        if profile.digest(profile.STABLE / name) != expected:
            raise RuntimeError(f'Stable dependency changed; review before use: {name}')
    stable = profile.load('xr1_usb_leftcrop_async_client', profile.STABLE / 'run_robot.py')
    stable.ROOT = HERE  # only this process's trace directory (xr1_crop_server/runs)

    forwarded = ['--server-port', str(DEFAULT_PORT), '--inference-mode', 'async',
                 '--async-latency-margin-s', '0.01', '--task', USB_TASK, *remaining]
    args = stable.parse_args(forwarded)
    if options.open_execute_steps is None:
        options.open_execute_steps = HORIZON - args.prefix_length
    if options.async_control_hz is not None:
        ratio = args.servo_hz / options.async_control_hz
        if options.async_control_hz <= 0 or abs(ratio - round(ratio)) > 1e-9:
            parser.error(f'--async-control-hz {options.async_control_hz:g} must divide --servo-hz {args.servo_hz:g}')
    if not 1 <= options.sync_execute_steps <= HORIZON:
        parser.error(f'--sync-execute-steps must be in [1, {HORIZON}]')
    for name in ('open_execute_steps', 'closed_execute_steps'):
        value = getattr(options, name)
        # Same bound as the bundle's --execute-steps: the next prefix is taken from these targets.
        if not args.prefix_length <= value <= HORIZON - args.prefix_length:
            parser.error(f'--{name.replace("_", "-")} {value} must be in '
                         f'[{args.prefix_length}, {HORIZON - args.prefix_length}] for prefix {args.prefix_length}')

    install_crop_pin(stable, manifest)
    install_window(stable, options)
    helper = profile.load('xr1_usb_leftcrop_stop_helper', profile.STOP_HELPER)
    sys.path.insert(0, str(BUNDLE))
    from hal_adapter import install as install_hal
    install_hal(stable, helper, args)
    if options.gripper_snap_close_m is not None:
        install_gripper_snap(stable, options.gripper_snap_close_m)
    print(json.dumps({'manifest': str(manifest_path), 'profile': manifest.get('profile'),
                      'zero_state': manifest.get('zero_state'), 'left_wrist_crop': manifest.get('left_wrist_crop'),
                      'server_port': args.server_port, 'inference_mode': args.inference_mode,
                      'prefix_length': args.prefix_length, 'open_execute_steps': options.open_execute_steps,
                      'closed_execute_steps': options.closed_execute_steps,
                      'closed_gripper_m': options.closed_gripper_m,
                      'gripper_snap_close_m': options.gripper_snap_close_m,
                      'vr_episodes': not options.no_vr, 'sync_when_closed': options.sync_when_closed,
                      'sync_near_xyz': options.sync_near_xyz, 'sync_near_radius_m': options.sync_near_radius_m,
                      'sync_execute_steps': options.sync_execute_steps,
                      'control_hz': args.control_hz, 'async_control_hz': options.async_control_hz,
                      'reset_config': str(stable.RESET_CONFIG), 'log_root': str(HERE / 'runs')},
                     ensure_ascii=False, indent=2), flush=True)
    offline = args.dry_run or args.probe or args.observe_infer_test or args.live_infer_test
    if options.no_vr or offline:
        stable.main(forwarded)
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.chunks != 0:
        parser.error('VR episodes run until a stick ending; drop --chunks or use --no-vr')
    run_vr_episodes(stable, args, options)


if __name__ == '__main__':
    main()
