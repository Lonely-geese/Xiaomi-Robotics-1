"""Offline test of record_xr1_async_rollout (no HAL / VR / server): a fake run_robot_async module.

    python steam_rollout/tests/test_xr1_async_rollout.py <out_dir>
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_xr1_async_rollout as rx  # noqa: E402
from lerobot_v21_writer import LeRobotV21RolloutWriter, shifted_actions  # noqa: E402
from recording_fixture import FakeObservationSource  # noqa: E402


HAND = {"x": 0.0}  # operator pushing the arm (EEF x) while paused


def fake_build_raw(state, task):
    seq = state["seq"]
    left = np.zeros(7, np.float32)
    left[0] = seq
    images = {}
    for i, key in enumerate(("head_rgb", "left_wrist_rgb", "right_wrist_rgb")):
        frame = np.zeros((3, 480, 848), np.uint8)
        frame[i] = (seq * 7) % 256
        images[key] = frame
    zeros7 = np.zeros(7, np.float32)
    cart = np.r_[HAND["x"], 0, 0, 0, 0, 0, 1].astype(np.float32)
    return {"images": images, "task": task,
            "state": {"left_arm": left, "left_cart": cart, "left_gripper": np.array([0.01], np.float32),
                      "right_arm": zeros7, "right_cart": cart, "right_gripper": np.array([0.05], np.float32)}}


class FakeVrSignals:
    def __init__(self):
        self.events = queue.Queue()
        self.home_key = SimpleNamespace(requested=threading.Event())

    def mark_manual_success(self):
        self.events.put("success")

    def mark_manual_failure(self):
        self.events.put("failure")

    def toggle_drag_teach(self):
        self.events.put("pause")

    def wait(self, accepted, on_idle=None, poll_s=0.02):
        while True:
            event = self.events.get()
            if event in accepted:
                return event


def make_fake_rra(script):
    """script: per episode, list of (segment duration s, closed gripper?, VR action, reason)."""
    module = SimpleNamespace()
    module.VrSignals = FakeVrSignals
    module.sync_switch_reason = lambda options, pose, gripper: "closed" if gripper <= options.closed_gripper_m else None

    def run_policy_segment(stable, args, observations, control, signals, options=None, synced=False):
        duration, action, reason = control.plan.pop(0)
        if action is not None:
            threading.Timer(duration * 0.8, getattr(signals, action)).start()
        time.sleep(duration)
        if action in ("mark_manual_success", "mark_manual_failure", "toggle_drag_teach"):
            reason = signals.events.get()
            if reason == "pause":
                pass
        return reason, None

    module.run_policy_segment = run_policy_segment

    def main(rest):
        source = FakeObservationSource()
        stable = SimpleNamespace(build_raw=fake_build_raw)
        options = SimpleNamespace(sync_when_closed=True, sync_near_xyz=None, closed_gripper_m=0.02,
                                  async_control_hz=7.5)
        args = SimpleNamespace(inference_mode="async", control_hz=15.0)
        signals = module.VrSignals()
        for episode in script:
            control = SimpleNamespace(plan=list(episode["segments"]), gripper=0.05)
            control.feedback_with_metadata = lambda c=control: (np.zeros(7), None, c.gripper)
            signals.events.put("start")
            assert signals.wait(("start", "home")) == "start"
            synced = False
            while True:
                control.gripper = 0.0 if synced else 0.05
                reason, _ = module.run_policy_segment(stable, args, source, control, signals, options, synced)
                if reason == "switch":
                    synced = True
                    continue
                if reason != "pause":
                    break
                time.sleep(0.4)  # operator getting ready: still frames, trimmed
                if episode.get("push"):
                    for _ in range(10):  # hand-insert: 10 mm over ~0.3 s
                        HAND["x"] += 0.001
                        time.sleep(0.03)
                    time.sleep(0.4)  # still again before the verdict: trimmed
                getattr(signals, episode["after_pause"])()
                reason = signals.wait(("pause", "success", "failure", "home"))
                if reason != "pause":
                    break
        source.close()

    module.main = main
    return module


def run(out: Path) -> None:
    script = [
        {"segments": [(1.6, None, "switch"), (1.0, "mark_manual_success", None)]},           # async -> sync -> success
        {"segments": [(1.0, "toggle_drag_teach", None), (1.0, "mark_manual_failure", None)],
         "after_pause": "toggle_drag_teach"},                                                # pause (no push), resume, failure
        {"segments": [(1.0, "toggle_drag_teach", None)], "push": True,
         "after_pause": "mark_manual_success"},                                              # jam, hand insert, success
        {"segments": [(0.8, None, "home")]},                                                 # dropped
    ]
    rra = make_fake_rra(script)
    opts = argparse.Namespace(steam_task="task_16", steam_save_home=False, steam_min_frames=5, steam_pause_mode="human",
                              steam_max_frames=6000, steam_jpeg_quality=95)
    writer = LeRobotV21RolloutWriter(out, action_shift=3)
    rec = rx.Recording(rra, opts, writer)
    rx.install(rra, rec)
    rra.main([])
    rec.recorder.close()
    writer.close()

    rows = [json.loads(line) for line in (out / "meta/episodes.jsonl").read_text().splitlines()]
    assert [r["termination"] for r in rows[-3:]] == ["success", "failure", "success"], rows
    for row in rows[-3:]:
        ep = row["episode_index"]
        table = pq.read_table(out / f"data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
        states = np.asarray(table["observation.state"], np.float32)
        seq = np.asarray(table["observation_sequence"])
        assert np.array_equal(states[:, 0], seq.astype(np.float32))
        assert np.array_equal(np.asarray(table["action"], np.float32), shifted_actions(states, 3))
        steps = np.diff(seq)
        print(f"episode {ep}: {row['termination']} len={len(seq)} sync_from={row['sync_start_frame']} "
              f"pauses={row['pauses']} steps={np.unique(steps, return_counts=True)}")
        human = np.asarray(table["intervene_flag"])
        if row["human_assisted"]:
            run = np.flatnonzero(human)
            assert np.all(np.diff(run) == 1) and np.all(np.diff(seq[run]) <= 2), "human run must be every HAL frame (fake source drops 1 in 50)"
            x = states[run, 7]
            assert x[-1] - x[0] > 0.008 and len(run) < 20, (len(run), x)  # push kept, still frames trimmed
            assert row["policy_success"] is False and set(table["is_success"]) == {True}
            assert row["human_idle_frames_trimmed"] > 10
            print("  human run", len(run), "frames, trimmed", row["human_idle_frames_trimmed"])
        elif row["termination"] == "success":
            assert row["policy_success"] is True and not human.any()
            k = row["sync_start_frame"]
            assert k and np.median(steps[: k - 1]) == 4 and np.median(steps[k:]) == 2, steps
        else:
            assert row["pauses"] == 1 and steps.max() > 10 and not human.any()  # no push: whole still run dropped
            assert np.median(steps) == 4
    print("OK", out)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
