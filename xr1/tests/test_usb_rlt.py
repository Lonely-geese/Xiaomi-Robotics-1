"""Action semantics, upstream loss parity, masking and resume contract tests."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from mibot.rlt.data import USBDataset, quaternion_matrix
from mibot.rlt.learner import RLTLearner, bc_loss, td_target, validate_batch
from mibot.rlt.model import ActionCoordinates, unpack_prefix
from mibot.rlt.runtime import encode_executed_left, recover_left_targets
from mibot.rlt.upstream.base_policy import ForwardType
from mibot.rlt.upstream.modules.rlt_token_transformer import RLTTokenTransformer
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = yaml.safe_load((ROOT / "configs/usb_rlt.yaml").read_text())
    cfg["rlt"].update(
        input_dim=16, embed_dim=16, num_heads=4, num_layers=1, prefix_seq_len=12
    )
    cfg["stage2"]["chunk_len"] = 3
    return cfg


def batch(config, b=4):
    k = config["stage2"]["chunk_len"]

    def obs():
        return {
            "z_rl": torch.randn(b, config["rlt"]["embed_dim"]),
            "proprio": torch.randn(b, 16),
            "ref_chunk": torch.tanh(torch.randn(b, 30, 7)),
        }

    result = {
        "curr_obs": obs(),
        "next_obs": obs(),
        "actions": torch.tanh(torch.randn(b, k, 7)),
        "rewards": torch.randn(b, k),
        "terminations": torch.zeros(b, k, dtype=torch.bool),
        "truncations": torch.zeros(b, k, dtype=torch.bool),
        "intervene_flags": torch.zeros(b, k, 7, dtype=torch.bool),
    }
    result["terminations"][0, -1] = True
    result["truncations"][1, -1] = True
    result["intervene_flags"][2, 1, 0] = True
    return result


def test_bounded_coordinates_preserve_values_outside_unit_interval():
    c = ActionCoordinates(5)
    z = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0])
    torch.testing.assert_close(c.decode(c.encode(z)), z)
    with pytest.raises(ValueError, match="saturated"):
        c.decode(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="saturated"):
        c.encode(torch.tensor([1000.0]))


def test_chunk_td_discount_and_termination_not_truncation():
    reward = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    terminated = torch.tensor([[False, False, True], [False, False, False]])
    target = td_target(reward, terminated, torch.tensor([[10.0], [10.0]]), 0.5)
    torch.testing.assert_close(target, torch.tensor([[2.75], [4.0]]))


def test_intervention_replaces_whole_step_bc_target():
    pi = torch.zeros(1, 2, 7)
    action = torch.full_like(pi, 0.8)
    reference = torch.full_like(pi, 0.2)
    flags = torch.zeros_like(pi, dtype=torch.bool)
    flags[0, 0, 2] = True
    torch.testing.assert_close(
        bc_loss(pi, action, reference, flags), torch.tensor(0.34)
    )


def test_packed_prefix_excludes_sft_labels_and_other_samples():
    x = torch.arange(20.0).reshape(1, 10, 2).requires_grad_()
    prefix, mask = unpack_prefix(x, torch.tensor([[0, 2], [6, 9]]))
    torch.testing.assert_close(prefix[0, :2], x[0, :2])
    torch.testing.assert_close(prefix[1], x[0, 6:9])
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert not prefix.requires_grad


def test_rlt_padding_mask_and_detached_target(config):
    torch.manual_seed(1)
    model = RLTTokenTransformer(**config["rlt"])
    x = torch.randn(2, 7, 16, requires_grad=True)
    mask = torch.tensor([[True] * 4 + [False] * 3, [True] * 7])
    changed = x.detach().clone()
    changed[0, 4:] = 1000
    loss, info = model(x, mask)
    loss2, info2 = model(changed, mask)
    torch.testing.assert_close(loss, loss2)
    torch.testing.assert_close(info["z_rl"], info2["z_rl"])
    loss.backward()
    assert x.grad is None
    assert model.encoder.rl_token_embed.grad.abs().sum() > 0


def test_replay_rejects_zscores_and_cross_episode_chunks(config):
    learner = RLTLearner(config)
    data = batch(config)
    validate_batch(data, learner.policy)
    data["actions"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="bounded"):
        validate_batch(data, learner.policy)
    data["actions"][0, 0, 0] = 0.2
    data["terminations"][0, 0] = True
    with pytest.raises(ValueError, match="complete K-step"):
        validate_batch(data, learner.policy)


def test_checkpoint_resume_exact_next_actor_and_critic_update(config, tmp_path):
    torch.manual_seed(7)
    learner = RLTLearner(config)
    data = batch(config)
    for _ in range(3):
        learner.update(data)
    learner.save(tmp_path / "heads.pt", {"norm": "test"})
    expected = learner.update(data)
    other = RLTLearner(config)
    other.restore(tmp_path / "heads.pt", {"norm": "test"})
    actual = other.update(data)
    assert actual == expected
    for key, value in learner.policy.state_dict().items():
        torch.testing.assert_close(
            value, other.policy.state_dict()[key], atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="assets"):
        other.restore(tmp_path / "heads.pt", {"norm": "wrong"})


def test_loss_parity_with_actual_rlinf_worker(config):
    source = Path(
        "/data/wz/references/RLinf/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"
    )
    if not source.exists():
        pytest.skip("RLinf checkout required for source-level parity")
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RLTACLossMixin"
    )
    names = {
        "_flatten_chunk",
        "_chunk_shape",
        "_ref_chunk",
        "_require_twin_q",
        "_min_twin_q",
        "_q1",
        "_discounted_chunk_rewards",
        "_bc_metrics",
        "_actor_objective_weights",
        "_next_actions_for_critic_target",
        "forward_critic",
        "forward_actor",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for n in cls.body:
        n.decorator_list = [
            d
            for d in n.decorator_list
            if isinstance(d, ast.Name) and d.id == "staticmethod"
        ]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {
        "torch": torch,
        "F": F,
        "ForwardType": ForwardType,
        "use_simulator_transition_replay": lambda cfg: False,
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - execute trusted pinned upstream methods for parity
    original = namespace["RLTACLossMixin"]()
    learner = RLTLearner(config)
    original.model, original.target_model = learner.policy, learner.target
    original.torch_dtype = torch.float32
    original.cfg = OmegaConf.create(
        {
            "algorithm": config["stage2"],
            "actor": {"model": {"num_action_chunks": 3, "action_dim": 7}},
        }
    )
    data = batch(config)
    torch.manual_seed(100)
    critic1, actor1 = learner.losses(data)
    torch.manual_seed(100)
    critic2, _ = original.forward_critic(data)
    actor2, _, _ = original.forward_actor(data)
    torch.testing.assert_close(critic1, critic2, atol=0, rtol=0)
    torch.testing.assert_close(actor1, actor2, atol=0, rtol=0)


def test_vendored_algorithms_unchanged():
    root = ROOT / "mibot/rlt/upstream"
    manifest = json.loads((root / "provenance.json").read_text())
    for rel, hashes in manifest["files"].items():
        text = (root / rel).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == hashes["vendored_sha256"]
        original = text.replace(
            "from mibot.rlt.upstream.", "from rlinf.models.embodiment."
        )
        assert hashlib.sha256(original.encode()).hexdigest() == hashes["source_sha256"]


def test_real_usb_action_recovery_matches_supplied_future_targets():
    cfg = yaml.safe_load((ROOT / "configs/usb_rlt.yaml").read_text())
    if not Path(cfg["dataset"]).exists():
        pytest.skip("USB dataset is not installed")
    import pyarrow.parquet as pq

    ds = USBDataset(cfg["dataset"], cfg["normalization"])
    record = ds.records[0]
    table = pq.read_table(Path(cfg["dataset"]) / record["data_path"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    # This chunk crosses a supplied static-cut boundary. No extra shift is valid.
    frame = 2
    traj = ds._read_json(str(Path(cfg["dataset"]) / record["data_path"]))
    parts = ds._arm_action(traj, "left", frame, 30)
    physical = np.concatenate(parts, -1)
    normalized = (physical - ds.mean[:, :7]) / (ds.std[:, :7] + 1e-6)
    norm = json.loads(Path(cfg["normalization"]).read_text())
    recovered = recover_left_targets(normalized, states[frame], norm)
    np.testing.assert_allclose(
        recovered["left_ee_pos"], actions[frame : frame + 30, 7:10], atol=1e-6
    )
    # XR1's float32 acos-based rotation conversion loses precision near zero.
    np.testing.assert_allclose(
        recovered["left_ee_rotm"],
        quaternion_matrix(actions[frame : frame + 30, 10:14]),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        recovered["left_gripper_pos"], actions[frame : frame + 30, 14:15], atol=1e-6
    )
    encoded = encode_executed_left(recovered, states[frame], norm)
    torch.testing.assert_close(
        encoded,
        ActionCoordinates().encode(torch.from_numpy(normalized)),
        atol=3e-4,
        rtol=3e-4,
    )


def test_live_observation_matches_dataset_without_action_labels():
    from mibot.rlt.data import CAMERAS, ObservationCollator

    cfg = yaml.safe_load((ROOT / "configs/usb_rlt.yaml").read_text())
    if not Path(cfg["dataset"]).exists():
        pytest.skip("USB dataset is not installed")
    import pyarrow.parquet as pq

    ds = USBDataset(cfg["dataset"], cfg["normalization"])
    record = ds.records[0]
    path = ds.root / record["data_path"]
    trajectory = ds._read_json(str(path))
    views = ds._images(
        trajectory,
        ["observations.ego", "observations.wrist_left", "observations.wrist_right"],
        0,
    )
    raw = np.asarray(
        pq.read_table(path, columns=["observation.state"])["observation.state"][
            0
        ].as_py(),
        dtype=np.float32,
    )
    collator = ObservationCollator(cfg["vlm_path"])
    from_dataset = collator([ds.item_at(0, 0)])
    live = collator.from_raw(
        dict(zip(CAMERAS, views, strict=True)),
        raw,
        json.loads(Path(cfg["normalization"]).read_text()),
        record["prompt"],
    )
    assert live.keys() == from_dataset.keys()
    for key in live:
        torch.testing.assert_close(live[key], from_dataset[key], atol=0, rtol=0)
