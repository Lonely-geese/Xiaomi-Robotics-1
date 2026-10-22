"""Selection and conversion invariants for the LFD / RECAP cleaning path."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

tools = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(tools))
spec = importlib.util.spec_from_file_location("prepare_lwd_pools", tools / "prepare_lwd_pools.py")
pools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pools)


def test_source_dedup_quarantine_precedence_and_split_segments():
    def record(name, session, index):
        return Path(name + ".json"), {"source": {"session": session, "episode_index": index}}
    entries = [record("bad", "old", 1), record("duplicate", "old", 2),
               record("fresh", "new", 1), record("part_a", "split", 1), record("part_b", "split", 1)]
    selected = pools.choose_added(entries, {("old", 1), ("old", 2)}, {"bad"})
    assert {p.stem: bucket for p, _, _, bucket, _ in selected} == {
        "bad": "quarantine_gripper", "duplicate": "duplicate", "fresh": "ready",
        "part_a": "pending_segment_boundary", "part_b": "pending_segment_boundary"}
    assert pools.source_key({"original_source_session": "old", "original_source_episode_index": 1}) == ("old", 1)


def test_conversion_keeps_frame_alignment_and_units():
    state = np.zeros((8, 30), dtype=np.float32)
    state[:, :7] = np.arange(8)[:, None] * 10
    state[:, 7] = np.arange(8) * .01
    state[:, 13] = state[:, 28] = 1  # XYZW identity quaternions
    action = state[np.minimum(np.arange(8) + 3, 7)].copy()
    arrays = pools.trajectory_arrays(state, action)
    np.testing.assert_array_equal(arrays["actions.left_ee_pos"], action[:, 7:10])
    np.testing.assert_allclose(arrays["proprios.left_arm_joint"], np.deg2rad(state[:, :7]))
    assert arrays["actions.left_ee_pos"][0, 0] == state[3, 7]
    assert len(arrays["actions.left_ee_pos"]) == len(state)


def test_split_keeps_entire_source_session_together():
    records = [dict(pool="lfd", status="success", split_group=f"session_{s}")
               for s in range(10) for _ in range(3)]
    validation = pools.validation_groups(records, .2)
    assert len(validation) == 2
    assert sum(row["split_group"] in validation for row in records) == 6
    assert pools.validation_groups(list(reversed(records)), .2) == validation
