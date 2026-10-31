"""BEE objective, data integrity, action semantics, and restart regression tests."""

import importlib.util
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from mibot.bee.core import BeeConfig, BeeLearner, BeePolicy, correction_nll, td_target
from mibot.bee.data import OfflineReplay, validate_data, validate_payload
from mibot.bee.runtime import USBBEEPolicy
from mibot.rlt.runtime import encode_executed_left, recover_left_targets


@pytest.fixture
def cfg():
    return BeeConfig(z_dim=8, hidden=16, chunk_len=3)


def batch(cfg, n=6):
    def obs():
        return {
            "z_rl": torch.randn(n, cfg.z_dim),
            "proprio": torch.randn(n, 16),
            "ref_chunk": 3 * torch.randn(n, cfg.chunk_len, 7),
        }

    data = {
        "curr_obs": obs(),
        "next_obs": obs(),
        "actions": 4 * torch.randn(n, cfg.chunk_len, 7),
        "rewards": torch.randn(n, cfg.chunk_len),
        "terminations": torch.zeros(n, cfg.chunk_len, dtype=torch.bool),
        "truncations": torch.zeros(n, cfg.chunk_len, dtype=torch.bool),
        "intervene_flags": torch.zeros(n, cfg.chunk_len, 7, dtype=torch.bool),
    }
    data["intervene_flags"][0, 0, :6] = True
    data["intervene_flags"][1, -1, :6] = True
    data["terminations"][0, -1] = True
    data["truncations"][1, -1] = True
    return data


def test_initial_residual_is_exact_proposal_and_gripper_never_changes(cfg):
    model = BeePolicy(cfg)
    obs = batch(cfg)["curr_obs"]
    torch.testing.assert_close(
        model.action(obs, dropout=True), obs["ref_chunk"], rtol=0, atol=0
    )
    with torch.no_grad():
        model.actor[-1].bias.fill_(2)
    result = model.action(obs)
    torch.testing.assert_close(result[..., 6], obs["ref_chunk"][..., 6], rtol=0, atol=0)
    torch.testing.assert_close(result[..., :6], obs["ref_chunk"][..., :6] + 2)
    assert result.abs().max() > 1  # No inherited LIBERO clamp or RLT tanh.


def test_nll_uses_only_actual_correction_dimensions_and_residual():
    action = torch.tensor([[[5.0, 999.0]]])
    ref = torch.tensor([[[2.0, 0.0]]])
    mean = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    var = torch.tensor([[[4.0, 0.1]]], requires_grad=True)
    mask = torch.tensor([[[True, False]]])
    loss = correction_nll(action, ref, mean, var, mask)
    torch.testing.assert_close(
        loss, torch.tensor(0.5 * (1 + np.log(4)), dtype=torch.float32)
    )
    loss.backward()
    assert mean.grad[0, 0, 1] == var.grad[0, 0, 1] == 0
    with pytest.raises(ValueError, match="no editable"):
        correction_nll(action, ref, mean, var, torch.zeros_like(mask))


def test_mahalanobis_weight_and_full_chunk_normalizer(cfg):
    model = BeePolicy(cfg)
    action = torch.zeros(1, cfg.chunk_len, 7)
    ref, mean, variance = (
        torch.zeros_like(action),
        torch.zeros_like(action),
        torch.ones_like(action),
    )
    action[0, 0, :2] = 2
    action[..., 6] = 1000  # Gripper contributes zero.
    variance[0, 0, 1] = 4
    torch.testing.assert_close(
        model.rho(action, ref, mean, variance), torch.tensor([5 / 21])
    )


def test_bellman_target_terminates_but_truncation_bootstraps():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    ends = torch.tensor([[False, False, True], [False, False, False]])
    torch.testing.assert_close(
        td_target(rewards, ends, torch.tensor([8.0, 8.0]), 0.5),
        torch.tensor([2.75, 3.75]),
    )


def test_actor_objective_equation_and_gradient_isolation(cfg):
    cfg = BeeConfig(**{**asdict(cfg), "reference_dropout": 0.0})
    model = BeePolicy(cfg)
    obs = batch(cfg)["curr_obs"]
    with torch.no_grad():
        model.actor[-1].bias.fill_(0.1)
    model.critics.requires_grad_(False)
    action = model.action(obs)
    with torch.no_grad():
        mean, var = model.distribution(obs)
        weight = model.dual(obs)
    q = model.q(obs, action).min(-1).values
    ref = obs["ref_chunk"]
    rho = (((action - ref - mean).square() / var) * model.editable).sum((1, 2)) / (
        cfg.chunk_len * 7
    )
    anchor = ((action - ref).square() * model.editable).sum((1, 2))
    expected = ((-q + cfg.eta * anchor + weight * rho) / (1 + weight)).mean()
    actual = model.actor_objective(obs)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.actor[-1].weight.grad.abs().sum() > 0
    assert all(
        p.grad is None
        for m in (model.correction, model.multiplier, model.critics)
        for p in m.parameters()
    )


def test_dual_ascent_direction():
    for violation in (-0.2, 0.2):
        raw = torch.tensor(0.0, requires_grad=True)
        before = torch.nn.functional.softplus(raw).item()
        loss = -torch.nn.functional.softplus(raw) * violation
        loss.backward()
        after = torch.nn.functional.softplus(raw - 0.1 * raw.grad).item()
        assert (after > before) == (violation > 0)


def test_strict_replay_rejects_missing_rewards_bad_dtype_and_early_end(cfg):
    data = batch(cfg)
    validate_data(data, cfg)
    bad = deepcopy(data)
    del bad["rewards"]
    with pytest.raises(ValueError, match="Missing"):
        validate_data(bad, cfg)
    bad = deepcopy(data)
    bad["intervene_flags"] = bad["intervene_flags"].float()
    with pytest.raises(ValueError, match="bool"):
        validate_data(bad, cfg)
    bad = deepcopy(data)
    bad["terminations"][2, 0] = True
    with pytest.raises(ValueError, match="complete K-step"):
        validate_data(bad, cfg)
    bad = deepcopy(data)
    bad["curr_obs"]["ref_chunk"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_data(bad, cfg)


def test_demo_seed_and_bounded_replay_cannot_be_used_as_rl(cfg):
    for fmt, source in (
        ("xr1-bee-corrections-v1", "demonstration_pairs"),
        ("xr1-rlt-replay-v1", "collected_transitions"),
    ):
        with pytest.raises(ValueError, match="format/assets"):
            validate_payload(
                {
                    "format": fmt,
                    "assets": {},
                    "source": source,
                    "transitions": batch(cfg),
                },
                cfg,
                {},
            )
    with pytest.raises(ValueError, match="data source"):
        validate_payload(
            {
                "format": "xr1-bee-replay-v1",
                "assets": {},
                "source": "demonstration_pairs",
                "transitions": batch(cfg),
            },
            cfg,
            {},
        )


def test_replay_half_human_and_correction_mask_preserve_proposals(cfg):
    data = batch(cfg)
    original = deepcopy(data)
    replay = OfflineReplay(data, cfg)
    sampled = replay.sample(20)
    assert sampled["intervene_flags"][10:].flatten(1).any(-1).all()
    corrected = replay.sample(20, correction_only=True)
    assert corrected["correction_mask"].flatten(1).any(-1).all()
    learner = BeeLearner(cfg)
    learner.update_correction(corrected)
    learner.update(sampled)
    learner.update(sampled)
    for k in data["curr_obs"]:
        torch.testing.assert_close(
            data["curr_obs"][k], original["curr_obs"][k], atol=0, rtol=0
        )
    no_human = deepcopy(data)
    no_human["intervene_flags"].zero_()
    with pytest.raises(ValueError, match="explicit human"):
        OfflineReplay(no_human, cfg)


def test_resume_next_sample_update_and_optimizer_exact(cfg, tmp_path):
    torch.manual_seed(12)
    replay = OfflineReplay(batch(cfg), cfg)
    learner = BeeLearner(cfg)
    learner.update_correction(replay.sample(6, correction_only=True))
    learner.update(replay.sample(6))
    assets, training = {"hash": "test"}, {"data_sha256": "test", "mode": "offline-rl"}
    path = tmp_path / "checkpoint.pt"
    learner.save(path, assets, training)
    expected_corr = learner.update_correction(replay.sample(6, correction_only=True))
    expected = learner.update(replay.sample(6))
    restored = BeeLearner(cfg)
    restored.restore(path, assets, training)
    actual_corr = restored.update_correction(replay.sample(6, correction_only=True))
    actual = restored.update(replay.sample(6))
    assert expected_corr == actual_corr and expected == actual
    for k, v in learner.model.state_dict().items():
        torch.testing.assert_close(v, restored.model.state_dict()[k], atol=0, rtol=0)
    for name, opt in learner.optimizers.items():
        for index, state in opt.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value,
                    restored.optimizers[name].state_dict()["state"][index][key],
                    atol=0,
                    rtol=0,
                )
    with pytest.raises(ValueError, match="assets"):
        restored.restore(path, {"hash": "wrong"}, training)
    with pytest.raises(ValueError, match="data/schedule"):
        restored.restore(path, assets, {"data_sha256": "changed"})


def test_correction_only_checkpoint_cannot_be_deployed(cfg, tmp_path, monkeypatch):
    from mibot.bee import runtime

    monkeypatch.setattr(runtime, "feature_assets", lambda *args: {})
    learner = BeeLearner(cfg)
    learner.save(tmp_path / "correction.pt", {}, {"mode": "correction"})
    with pytest.raises(ValueError, match="actor has not been trained"):
        USBBEEPolicy({}, "unused", cfg, tmp_path / "correction.pt", device="cpu")


def test_zscore_action_roundtrip_keeps_chunk_start_frame():
    raw = np.zeros(30, dtype=np.float32)
    raw[7:10] = [1, 2, 3]
    raw[13] = raw[28] = 1
    raw[14] = 0.5
    norm = {"mean": np.zeros((30, 60)).tolist(), "std": np.ones((30, 60)).tolist()}
    z = np.zeros((3, 7), dtype=np.float32)
    z[:, 0] = [2.0, 3.0, 4.0]
    z[:, 3] = [0.1, 0.2, 0.3]
    z[:, 6] = 0.2
    targets = recover_left_targets(z, raw, norm)
    np.testing.assert_allclose(targets["left_ee_pos"][:, 0], [3, 4, 5], atol=5e-6)
    recovered = encode_executed_left(targets, raw, norm, action_coordinates="zscore")
    torch.testing.assert_close(recovered, torch.from_numpy(z), atol=1e-5, rtol=1e-5)


def tool_module(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_episode_split_is_disjoint_and_reproducible():
    module = tool_module("prewarm_usb_bee")
    records = [{"episode_index": i} for i in range(1041)]
    train, val = module.episode_split(records, 42, 0.1)
    assert len(train) == 937 and len(val) == 104
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(1041))
    assert (train, val) == module.episode_split(records, 42, 0.1)


def test_offline_cli_and_resume_keep_correct_schedule(cfg, tmp_path, monkeypatch):
    module = tool_module("train_usb_bee")
    monkeypatch.setattr(
        module,
        "load_settings",
        lambda path: ({"seed": 4, "stage1_checkpoint": "unused"}, {}, cfg),
    )
    monkeypatch.setattr(module, "feature_assets", lambda *args: {})
    replay_path = tmp_path / "synthetic_test_only.pt"
    torch.save(
        {
            "format": "xr1-bee-replay-v1",
            "source": "collected_transitions",
            "assets": {},
            "transitions": batch(cfg),
        },
        replay_path,
    )
    output = tmp_path / "test_run"
    args = [
        "train_usb_bee.py",
        "--data",
        str(replay_path),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--updates",
        "2",
        "--batch-size",
        "4",
        "--correction-pretrain",
        "2",
        "--correction-every",
        "2",
        "--correction-updates",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    saved = torch.load(output / "latest.pt", weights_only=True)
    assert saved["step"] == 2 and saved["correction_steps"] == 3
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(output / "latest.pt")])
    module.main()
    saved = torch.load(output / "latest.pt", weights_only=True)
    assert saved["step"] == 4 and saved["correction_steps"] == 4
