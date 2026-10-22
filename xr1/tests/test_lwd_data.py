"""Exercise the real JSON adapter with explicit success/failure/truncation labels."""

import json

import numpy as np
import pytest
import torch

from mibot.rl.data import EpisodeTransitions, TransitionCollate


def episode(tmp_path):
    count = 5
    arrays = {}
    for section in ("proprios", "actions"):
        for arm in ("left", "right"):
            arrays[f"{section}.{arm}_ee_pos"] = np.zeros((count, 3), np.float32)
            arrays[f"{section}.{arm}_ee_rotm"] = np.tile(np.eye(3).reshape(1, 9), (count, 1))
            arrays[f"{section}.{arm}_gripper_pos"] = np.zeros((count, 1), np.float32)
            arrays[f"{section}.{arm}_arm_joint"] = np.arange(count, dtype=np.float32)[:, None].repeat(7, 1)
        arrays[f"{section}.waist_pos"] = np.zeros((count, 1), np.float32)
    arrays["actions.base_vel"] = np.zeros((count, 3), np.float32)
    np.savez(tmp_path / "arrays.npz", **arrays)
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(dict(num_frames=count, arrays_path="arrays.npz", trajectory_type="failure",
        instruction={"general": [{"images": [], "conversations": [
            {"from": "human", "value": "insert the plug"}, {"from": "gpt", "value": ""}]}]})))
    config = dict(paths=[str(path)], action_length=3, sample_stride=1,
                  mean=np.zeros((3, 60)), std=np.ones((3, 60)),
                  q01=np.zeros((1, 60)), q99=np.ones((1, 60)) * 10,
                  active_action_dims=[0, 1, 2, 3, 4, 5, 6])
    return path, config


def test_success_n_chunks_and_short_terminal_action_mask(tmp_path):
    path, config = episode(tmp_path)
    dataset = EpisodeTransitions(config, {str(path): {"terminated": True, "success": True}}, n_chunks=2, gamma=.9)
    first = dataset[0]
    assert first["reward"] == pytest.approx(.9 ** 4)
    assert first["steps"] == 5 and first["terminated"]
    last = dataset[-1]
    assert last["reward"] == 1 and last["steps"] == 1
    assert last["action_mask"][0].sum() == 7
    assert last["action_mask"][1:].sum() == 0
    assert len(first["observation"]["messages"]) == 2
    assert "<state>" not in first["observation"]["text"]
    assert "<a_" not in first["observation"]["text"]


def test_failure_actions_are_kept_and_timeout_has_real_next_state(tmp_path):
    path, config = episode(tmp_path)
    failure = EpisodeTransitions(config, {str(path): {"terminated": True, "success": False}})
    assert len(failure) == 5
    assert failure[-1]["terminated"] and failure[-1]["reward"] == 0
    truncated = EpisodeTransitions(config, {str(path): {"terminated": False, "success": False}}, n_chunks=10)
    assert len(truncated) == 4
    transition = truncated[-1]
    assert not transition["terminated"] and transition["steps"] == 1
    assert transition["action_mask"][1:].sum() == 0
    assert not torch.equal(transition["observation"]["state"], transition["next_observation"]["state"])
    torch.testing.assert_close(transition["next_observation"]["state"], failure[-1]["observation"]["state"])


def test_replay_refuses_missing_or_inconsistent_outcomes(tmp_path):
    path, config = episode(tmp_path)
    with pytest.raises(ValueError, match="missing explicit"):
        EpisodeTransitions(config, {})
    with pytest.raises(ValueError, match="true episode"):
        EpisodeTransitions(config, {str(path): {"terminated": False, "success": True}})


def test_unknown_ending_uses_only_full_windows_before_final_state(tmp_path):
    path, config = episode(tmp_path)
    label = {str(path): {"terminated": False, "success": False, "end_status": "unknown"}}
    dataset = EpisodeTransitions(config, label, n_chunks=1)
    # Five states and a three-action chunk: only frame 0 bootstraps strictly
    # before the ambiguous final state. Frame 1 would reach that state.
    assert len(dataset) == 1
    item = dataset[0]
    assert item["steps"] == 3 and not item["terminated"] and item["reward"] == 0
    assert item["action_mask"].sum() == 21
    with pytest.raises(ValueError, match="no usable"):
        EpisodeTransitions(config, label, n_chunks=2)
    label[str(path)]["terminated"] = True
    with pytest.raises(ValueError, match="inconsistent end_status"):
        EpisodeTransitions(config, label)


def test_unknown_reset_tail_is_censored_before_action_windows(tmp_path):
    path, config = episode(tmp_path)
    config.update(action_length=1, mean=np.zeros((1, 60)), std=np.ones((1, 60)))
    label = {str(path): {"terminated": False, "success": False,
                         "end_status": "unknown", "usable_frames": 3}}
    dataset = EpisodeTransitions(config, label)
    assert len(dataset) == 1
    assert dataset[0]["steps"] == 1
    assert dataset.indices[0][3] == 3
    label[str(path)]["usable_frames"] = 6
    with pytest.raises(ValueError, match="invalid usable_frames"):
        EpisodeTransitions(config, label)


def test_collate_pads_cameras_and_keeps_observations_separate():
    from PIL import Image

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), 3, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 3, dtype=torch.long)}

    class Images:
        def __call__(self, images, **kwargs):
            return {"pixel_values": torch.ones(len(images), 3, 8, 8)}

    class ActorProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.ones(len(messages), 3, dtype=torch.long),
                    "attention_mask": torch.ones(len(messages), 3, dtype=torch.long)}

    collate = TransitionCollate.__new__(TransitionCollate)
    collate.tokenizer, collate.image_processor, collate.actor_processor = Tokenizer(), Images(), ActorProcessor()
    items = []
    for views in (1, 2):
        observation = {"state": torch.zeros(1, 60), "messages": [], "text": "insert plug",
                       "images": [Image.new("RGB", (8, 8)) for _ in range(views)]}
        items.append(dict(observation=observation, next_observation=observation,
                          action=torch.zeros(30, 60), action_mask=torch.ones(30, 60),
                          reward=0., steps=30, terminated=False))
    batch = collate(items)
    assert batch["critic_observation"]["pixel_values"].shape == (2, 2, 3, 8, 8)
    assert batch["critic_observation"]["view_mask"].tolist() == [[True, False], [True, True]]
    assert batch["actor_observation"]["state"].shape == (2, 1, 60)
    assert batch["action"].shape == (2, 30, 60)
