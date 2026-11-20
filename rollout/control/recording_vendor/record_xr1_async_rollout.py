#!/usr/bin/env python3
"""XR1 (Xiaomi) official async client + LeRobot v2.1 recording for RLinf STEAM.

Runs ``rlt_online_rl_usb/xr1_crop_server/run_robot_async.py`` (crop server on 10092, official
action_prefix async inference, VR episodes) unchanged, loaded as a module and patched only in this
process, and records every VR episode to a LeRobot v2.1 "rollout" dataset (see lerobot_v21_writer).

Time axis: one dataset frame = one policy step. While the policy plays at ``--async-control-hz``
(7.5 Hz -> every 4th 30 Hz HAL frame) and after the switch to sync at ``--control-hz`` (15 Hz ->
every 2nd), the stride follows the segment, so ``action = state[t+3]`` and STEAM's ``data.k`` mean the
same as on the 30 Hz SFT data. Frames end at the verdict.

Human takeover = stick up (pause): once the runner reports the arm at rest, the following frames are
the operator moving the arm by hand. They are recorded every 30 Hz HAL frame (like the SFT teleop),
with ``intervene_flag`` / source HUMAN, until the policy resumes or the verdict; the still frames at
both ends of that stretch are trimmed. ``--steam-pause-mode skip`` restores "do not record pauses".

Options of this script start with --steam-; everything else goes to run_robot_async.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RRA_PATH = REPO / "rlt_online_rl_usb" / "xr1_crop_server" / "run_robot_async.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hal_frame_recorder import HalFrameRecorder, trim_idle_runs  # noqa: E402
from lerobot_v21_writer import SOURCE_HUMAN, SOURCE_POLICY, EpisodeRecord, LeRobotV21RolloutWriter  # noqa: E402

logger = logging.getLogger("steam_rollout.xr1_async")
HAL_FPS = 30
SFT_TASK = "task_16"
# build_raw camera key -> recorder image key (same mapping as scripts.deeptouch.policy_client)
IMAGE_KEYS = {"head_rgb": "base_0_rgb", "left_wrist_rgb": "left_wrist_0_rgb", "right_wrist_rgb": "right_wrist_0_rgb"}


def stride_for(hz: float) -> int:
    stride = HAL_FPS / float(hz)
    if round(stride) < 1 or abs(stride - round(stride)) > 1e-6:
        raise SystemExit(f"policy rate {hz:g} Hz does not divide the {HAL_FPS} Hz HAL stream")
    return int(round(stride))


def make_observation_fn(stable: Any):
    """HAL state dict -> {state30, RGB HWC images} with the runner's own build_raw (same validation)."""
    def make(state: dict, task: str) -> dict:
        raw = stable.build_raw(state, task)
        values = raw["state"]
        state30 = np.concatenate([values["left_arm"], values["left_cart"], values["left_gripper"],
                                  values["right_arm"], values["right_cart"], values["right_gripper"]])
        images = {target: np.ascontiguousarray(raw["images"][source].transpose(1, 2, 0))
                  for source, target in IMAGE_KEYS.items()}
        return {"state": state30.astype(np.float32), "image": images}
    return make


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Recording:
    """Episode bookkeeping driven by the runner's VR events and policy segments."""

    def __init__(self, rra: Any, opts: argparse.Namespace, writer: LeRobotV21RolloutWriter):
        self.rra, self.opts, self.writer = rra, opts, writer
        self.recorder: HalFrameRecorder | None = None
        self.signals: Any = None
        self.hz = 15.0
        self.human = False  # operator has the arm (paused, arm released by the runner)
        self.pending_start = False
        self._episode: dict[str, Any] | None = None

    # ---- hooks ----
    def on_segment_start(self, stable, args, observations, control, signals, options, synced) -> None:
        self.signals = signals
        if self.recorder is None:
            self.recorder = HalFrameRecorder(
                observations, make_observation_fn(stable), self.opts.steam_task,
                stride=lambda: 1 if self.human else stride_for(self.hz),
                label_fn=lambda: SOURCE_HUMAN if self.human else SOURCE_POLICY,
                freeze_fn=lambda: bool(self.signals is not None and self.signals.home_key.requested.is_set()),
                jpeg_quality=self.opts.steam_jpeg_quality, max_frames=self.opts.steam_max_frames)
        sync = args.inference_mode == "sync" or bool(synced)
        if not sync and options is not None and (getattr(options, "sync_when_closed", False)
                                                 or getattr(options, "sync_near_xyz", None) is not None):
            pose, _, gripper, *_ = control.feedback_with_metadata()
            sync = self.rra.sync_switch_reason(options, pose, gripper) is not None
        async_hz = getattr(options, "async_control_hz", None)
        self.hz = float(args.control_hz if sync or async_hz is None else async_hz)
        self.human = False
        if self.pending_start:
            self.pending_start = False
            self.recorder.start_episode()
            self._episode = {"started_at_unix": time.time(), "segments": 0, "pauses": 0, "sync_start_frame": None,
                             "control_hz": float(args.control_hz), "async_control_hz": async_hz,
                             "model_manifest": os.environ.get("XR1_MODEL_MANIFEST")}
        if self._episode is None:
            return
        self._episode["segments"] += 1
        if sync and self._episode["sync_start_frame"] is None:
            self._episode["sync_start_frame"] = self.recorder.frame_count()
        logger.warning("STEAM: segment %s %s at %g Hz -> record every %s HAL frames",
                       self._episode["segments"], "sync" if sync else "async", self.hz, stride_for(self.hz))
        self.recorder.resume()

    def on_segment_end(self, reason: str) -> None:
        if self._episode is None or self.recorder is None:
            return
        if reason == "switch":
            return
        if reason == "pause":
            self._episode["pauses"] += 1
            if self.opts.steam_pause_mode == "human":
                self.human = True  # arm verified at rest: from here on the operator moves it
                logger.warning("STEAM: paused -> recording operator takeover (HUMAN, every HAL frame)")
            else:
                self.recorder.pause()
            return
        self.finish(reason)

    def abort(self, why: str) -> None:
        if self.recorder is not None:
            self.recorder.abort_episode(why)
        self._episode = None

    def finish(self, reason: str) -> None:
        if self._episode is None or self.recorder is None:
            return
        episode, self._episode = self._episode, None
        self.human = False
        frames = self.recorder.finish_episode()
        frames, trimmed = trim_idle_runs(frames, SOURCE_HUMAN)
        length = int(frames["states"].shape[0])
        human_frames = int(np.sum(frames["sources"] == SOURCE_HUMAN))
        if reason == "home" and not self.opts.steam_save_home:
            logger.warning("STEAM: home-aborted episode not saved (%s frames); --steam-save-home keeps it", length)
            return
        if length < self.opts.steam_min_frames:
            logger.warning("STEAM: episode with %s frames < --steam-min-frames, not saved", length)
            return
        success = reason == "success"
        # is_success = task completed. policy_success also requires no operator takeover.
        episode.update(human_assisted=human_frames > 0, policy_success=success and human_frames == 0,
                       human_idle_frames_trimmed=trimmed)
        record = EpisodeRecord(
            task=self.opts.steam_task, states=frames["states"], jpeg_frames=frames["jpeg_frames"],
            sources=frames["sources"], observation_sequence=frames["observation_sequence"],
            hal_time_ns=frames["hal_time_ns"], success=success,
            extra={"termination": reason, "policy": "usb_recap_epoch10_step15490", **episode,
                   "sequence_gaps": frames["sequence_gaps"], "bad_frames": frames["bad_frames"]})
        index = self.writer.submit(record)
        logger.warning("STEAM: queued episode %s: %s, %s frames (%s human, %s idle trimmed, sync from frame %s), "
                       "capture %.1f ms/frame. %s", index, reason, length, human_frames, trimmed,
                       episode["sync_start_frame"], frames["mean_capture_ms"], self.writer.summary())


def install(rra: Any, rec: Recording) -> None:
    original_segment = rra.run_policy_segment

    def run_policy_segment(stable, args, observations, control, signals, options=None, synced=False):
        rec.on_segment_start(stable, args, observations, control, signals, options, synced)
        try:
            reason, detail = original_segment(stable, args, observations, control, signals, options, synced)
        except BaseException:
            rec.abort("exception during the policy segment")
            raise
        rec.on_segment_end(reason)
        return reason, detail

    rra.run_policy_segment = run_policy_segment
    signals_cls = rra.VrSignals
    original_wait = signals_cls.wait
    original_success, original_failure = signals_cls.mark_manual_success, signals_cls.mark_manual_failure
    original_pause = signals_cls.toggle_drag_teach

    def wait(self, accepted, on_idle=None, poll_s: float = 0.02):
        event = original_wait(self, accepted, on_idle, poll_s)
        if "start" in accepted and event == "start":
            rec.pending_start = True
        elif "pause" in accepted and event in ("success", "failure", "home"):
            rec.finish(event)  # verdict given while paused
        return event

    def freeze_then(original):
        def handler(self):
            if rec.recorder is not None:
                rec.recorder.freeze()  # VR thread: cut the frames at the press, not when the stop lands
            return original(self)
        return handler

    def pause(self):
        # skip mode: stop at the press. human mode: keep recording; the takeover starts once the
        # runner has stopped the arm (on_segment_end "pause"), so the braking stays POLICY.
        if rec.recorder is not None and rec.opts.steam_pause_mode == "skip" and not rec.human:
            rec.recorder.pause()
        return original_pause(self)

    signals_cls.wait = wait
    signals_cls.mark_manual_success = freeze_then(original_success)
    signals_cls.mark_manual_failure = freeze_then(original_failure)
    signals_cls.toggle_drag_teach = pause


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steam-out", required=True, type=Path,
                        help="LeRobot v2.1 dataset directory (created, or appended to if it exists)")
    parser.add_argument("--steam-task", default=SFT_TASK,
                        help="task string in tasks.jsonl; use the one of your SFT dataset (default task_16)")
    parser.add_argument("--steam-action-shift", type=int, default=3, help="action[t] = state[t+shift]")
    parser.add_argument("--steam-save-home", action="store_true",
                        help="also save episodes ended with home (stick down) as failures (default: drop)")
    parser.add_argument("--steam-pause-mode", choices=("human", "skip"), default="human",
                        help="human (default): record the operator's hand takeover during a pause as HUMAN frames; "
                             "skip: do not record paused stretches")
    parser.add_argument("--steam-min-frames", type=int, default=30)
    parser.add_argument("--steam-max-frames", type=int, default=6000)
    parser.add_argument("--steam-jpeg-quality", type=int, default=95)
    parser.add_argument("--steam-crf", type=int, default=18)
    parser.add_argument("--steam-gop", type=int, default=10)
    opts, rest = parser.parse_known_args(argv)
    if "--no-vr" in rest:
        raise SystemExit("recording needs VR episodes (Y / stick verdicts); drop --no-vr")
    return opts, rest


def main(argv: list[str] | None = None) -> None:
    opts, rest = _parse(sys.argv[1:] if argv is None else argv)
    for flag, value in (("--async-control-hz", "7.5"), ("--control-hz", "15")):
        if not any(a == flag or a.startswith(flag + "=") for a in rest):
            rest += [flag, value]
    rates = {flag: float(rest[rest.index(flag) + 1]) for flag in ("--async-control-hz", "--control-hz") if flag in rest}
    for hz in rates.values():
        stride_for(hz)  # 7.5 -> 4, 15 -> 2; fail before the robot moves
    rra = load_module("run_robot_async", RRA_PATH)
    writer = LeRobotV21RolloutWriter(opts.steam_out, action_shift=opts.steam_action_shift,
                                     crf=opts.steam_crf, gop=opts.steam_gop)
    rec = Recording(rra, opts, writer)
    install(rra, rec)
    print(f"[steam] XR1 async rollout -> {writer.root} (task {opts.steam_task!r}, action shift "
          f"{opts.steam_action_shift}, rates {rates})", flush=True)
    try:
        rra.main(rest)
    finally:
        rec.abort("runner exited")
        if rec.recorder is not None:
            rec.recorder.close()
        print("[steam] waiting for video encoding to finish ...", flush=True)
        writer.close()
        print(f"[steam] {writer.summary()} -> {writer.root}", flush=True)


if __name__ == "__main__":
    main()
