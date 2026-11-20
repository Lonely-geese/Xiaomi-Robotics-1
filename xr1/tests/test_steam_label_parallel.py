"""Restarted parallel labeling must cover each unfinished episode exactly once."""

import pytest

from tools.label_steam_advantages import assign_remaining


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_remaining_episodes_are_disjoint_and_complete(tmp_path, world_size):
    episodes = [
        {"episode_id": str(i), "usable_frames": size}
        for i, size in enumerate([100, 4, 700, 35, 200, 18, 65, 40, 300, 9, 20])
    ]
    for i in [0, 3, 8]:
        (tmp_path / f"{i}.parquet").touch()
    # A partial write is not a completed episode and must be recomputed.
    (tmp_path / "2.parquet.tmp").touch()
    groups = assign_remaining(episodes, tmp_path, world_size)
    flat = [i for group in groups for i in group]
    assert len(groups) == world_size
    assert len(flat) == len(set(flat))
    assert sorted(flat) == [1, 2, 4, 5, 6, 7, 9, 10]
    if world_size == 8:
        assert all(len(group) == 1 for group in groups)


def test_balances_frames_and_handles_completed_run(tmp_path):
    episodes = [{"episode_id": str(i), "usable_frames": size}
                for i, size in enumerate([80, 70, 60, 50, 40, 30, 20, 10])]
    groups = assign_remaining(episodes, tmp_path, 4)
    assert [sum(episodes[i]["usable_frames"] for i in group) for group in groups] == [90] * 4
    for record in episodes:
        (tmp_path / (record["episode_id"] + ".parquet")).touch()
    assert assign_remaining(episodes, tmp_path, 8) == [[] for _ in range(8)]
