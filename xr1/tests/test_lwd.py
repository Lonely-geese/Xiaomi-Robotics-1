"""CPU numerical tests; no pretrained weights or robot connection required."""

import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
import torch.utils.checkpoint

from mibot.rl import DIVL, LWDConfig, LWDLearner
from mibot.rl.actor import XR1FlowActor
from mibot.rl.backbone import ValueBackbone
from mibot.rl.divl import adaptive_quantile, categorical_projection, update_ema
from mibot.rl.qam import build_adjoint_targets, local_loss
from mibot.rl.replay import chunk_return, validate_batch


class ToyXR1(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(3, 4)
        self.dit = nn.Linear(7, 2)
        self.sink = nn.Embedding(1, 4)
        for name in ("state_projector_choice", "action_projector_choice", "score_projector_choice"):
            setattr(self, name, nn.Linear(3, 2))

    def prepare_flow_condition(self, observation, mask):
        return {"features": self.vlm(observation["state"]), "mask": mask}

    def dit_forward(self, action, time, features, mask):
        features = features[:, None].expand(-1, action.shape[1], -1)
        return self.dit(torch.cat([action * mask, features, time.expand(-1, action.shape[1], -1)], -1))


class ToyEncoder(nn.Linear):
    def forward(self, observation):
        return super().forward(observation["state"])


def make_batch():
    torch.manual_seed(23)
    state = {"state": torch.randn(3, 3)}
    mask = torch.ones(3, 2, 2)
    mask[:, 1, 1] = 0
    return dict(actor_observation=state, critic_observation=state,
                next_critic_observation={"state": torch.randn(3, 3)},
                action=torch.randn(3, 2, 2), action_mask=mask,
                reward=torch.tensor([0., 1., 0.]), steps=torch.tensor([2, 1, 2]),
                terminated=torch.tensor([False, True, False]))


def make_learner():
    config = LWDConfig(atoms=11, flow_steps=3, total_steps=20, actor_lr=0.01)
    return LWDLearner(XR1FlowActor(ToyXR1()), DIVL(ToyEncoder(3, 4), 4, 2, 2, config), config)


def test_projection_preserves_mass_at_atoms_and_boundaries():
    support = torch.linspace(-.1, 1.1, 201)
    values = torch.tensor([-10., -.1, .203, 1.1, 10.])
    probabilities = categorical_projection(values, support)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(5))
    torch.testing.assert_close(probabilities @ support, values.clamp(-.1, 1.1))
    assert (probabilities >= 0).all()
    assert probabilities[0, 0] == 1 and probabilities[-1, -1] == 1


def test_entropy_adapts_tau_and_quantile_selects_first_cdf_crossing():
    config = LWDConfig(atoms=5, v_min=0, v_max=4)
    support = torch.arange(5.)
    logits = torch.tensor([[0., 0., 0., 0., 0.], [-100., -100., 100., -100., -100.]])
    value, tau, entropy = adaptive_quantile(logits, support, config)
    torch.testing.assert_close(tau, torch.tensor([.3, .6]))
    torch.testing.assert_close(value, torch.tensor([1., 2.]))
    torch.testing.assert_close(entropy, torch.tensor([1., 0.]))
    _, online_tau, _ = adaptive_quantile(logits, support, replace(config, stage="online"))
    torch.testing.assert_close(online_tau, torch.tensor([.6, .9]))


def test_terminal_td_ignores_next_value_and_scalar_shapes_do_not_broadcast():
    learner = make_learner()
    batch = make_batch()
    batch["terminated"][:] = True
    batch["reward"] = torch.tensor([0., 1., 0.])
    _, flat = learner.critic.loss(learner.target, batch, learner.config)
    for key in ("reward", "steps", "terminated"):
        batch[key] = batch[key][:, None]
    batch["next_critic_observation"]["state"] *= 1000
    _, column = learner.critic.loss(learner.target, batch, learner.config)
    torch.testing.assert_close(flat["q_loss"], column["q_loss"])
    torch.testing.assert_close(flat["td_mean"], torch.tensor(1 / 3))


def test_action_pool_mask_removes_invalid_coordinates_and_time_steps():
    critic = make_learner().critic
    batch = make_batch()
    batch["action_mask"][:, -1] = 0
    state = critic.encode(batch["critic_observation"])
    action = batch["action"].requires_grad_()
    q = critic.q(state, action, batch["action_mask"])
    changed = action.detach() + (1 - batch["action_mask"]) * 10000
    torch.testing.assert_close(q, critic.q(state, changed, batch["action_mask"]))
    gradient = torch.autograd.grad(q.sum(), action)[0]
    assert (gradient[~batch["action_mask"].bool()] == 0).all()


def test_qam_adjoint_matches_analytic_linear_reference():
    coefficient = .3
    steps, temperature = 4, 2.
    mask = torch.tensor([[[1., 0.], [1., 1.]]])
    targets = build_adjoint_targets(lambda x, t: coefficient * x,
                                    lambda x: (x * 2).flatten(1).sum(-1), mask,
                                    steps=steps, temperature=temperature)
    # Terminal g=-2/lambda. Backward drift Jacobian=2*c-1/(t+h).
    expected = -2 / temperature * mask
    for i in reversed(range(steps)):
        expected = expected * (1 + 1 / steps * (2 * coefficient - 1 / ((i + 1) / steps)))
        torch.testing.assert_close(targets.adjoints[i], expected)
    for tensors in (targets.states, targets.adjoints, targets.reference_velocities):
        assert all(not item.requires_grad for item in tensors)
        assert all((item[~mask.bool()] == 0).all() for item in tensors)


def test_qam_local_update_increases_linear_q_and_leaves_reference_frozen():
    actor = nn.Parameter(torch.tensor(0.))
    reference = nn.Parameter(torch.tensor(.2), requires_grad=False)
    mask = torch.ones(2, 2, 2)
    targets = build_adjoint_targets(lambda x, t: reference * x,
                                    lambda x: x.flatten(1).sum(-1), mask, steps=3)
    loss = local_loss(lambda x, t: reference * x + actor, targets, mask)
    loss.backward()
    assert torch.isfinite(actor.grad) and actor.grad < 0  # descent raises action/Q
    assert reference.grad is None


def test_learner_updates_correct_branches_and_roundtrips(tmp_path):
    learner = make_learner()
    batch = make_batch()
    actor_before = copy.deepcopy(learner.actor.state_dict())
    reference_before = copy.deepcopy(learner.reference.state_dict())
    target_before = copy.deepcopy(learner.target.state_dict())
    metrics = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(actor_before["model.dit.weight"], learner.actor.model.dit.weight)
    assert not torch.equal(actor_before["model.vlm.weight"], learner.actor.model.vlm.weight)
    for key, value in learner.reference.state_dict().items():
        torch.testing.assert_close(value, reference_before[key], rtol=0, atol=0)
    for key, value in learner.target.named_parameters():
        torch.testing.assert_close(value, target_before[key].lerp(learner.critic.state_dict()[key], learner.config.ema_rate))
    assert all(p.grad is None for p in learner.reference.parameters())
    assert all(p.grad is None for p in learner.target.parameters())
    assert all(p.grad is None for p in learner.critic.parameters())
    path = tmp_path / "checkpoint.pt"
    torch.save(learner.checkpoint(), path)
    expected_metrics = learner.update(batch)
    restored = make_learner()
    restored.restore(torch.load(path, weights_only=True))
    actual_metrics = restored.update(batch)
    assert expected_metrics == actual_metrics
    for a, b in zip(learner.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_online_freezes_only_actor_vlm_and_retains_original_reference():
    learner = make_learner()
    batch = make_batch()
    learner.update(batch)
    reference = copy.deepcopy(learner.reference.state_dict())
    learner.start_online(total_steps=10)
    vlm = learner.actor.model.vlm.weight.detach().clone()
    dit = learner.actor.model.dit.weight.detach().clone()
    critic = learner.critic.encoder.weight.detach().clone()
    learner.update(batch)
    torch.testing.assert_close(vlm, learner.actor.model.vlm.weight, rtol=0, atol=0)
    assert not torch.equal(dit, learner.actor.model.dit.weight)
    assert not torch.equal(critic, learner.critic.encoder.weight)
    assert not learner.actor.model.vlm.training
    for key, value in learner.reference.state_dict().items():
        torch.testing.assert_close(value, reference[key], rtol=0, atol=0)


def test_n_chunk_return_stops_at_terminal_and_discounts_each_frame():
    result = chunk_return([0, 0, 1, 999], [False, False, True, False], 0, 2, 10, .9)
    assert result == pytest.approx((.81, 3, True, 3))
    assert chunk_return([0, 1, 0], [False] * 3, 0, 2, 1, .9) == pytest.approx((.9, 2, False, 2))


@pytest.mark.parametrize("field,value", [("steps", torch.tensor([0, 1, 2])),
    ("reward", torch.zeros(3, 3)), ("terminated", torch.tensor([0, 2, 0]))])
def test_transition_validation_rejects_ambiguous_scalars(field, value):
    batch = make_batch()
    batch[field] = value
    with pytest.raises(ValueError):
        validate_batch(batch)


def load_xr1_function(name, method=False):
    # Execute the actual source in isolation: the full model imports CUDA-only
    # Liger kernels and constructs billions of parameters, irrelevant here.
    path = Path(__file__).parents[1] / "mibot/models/VLA/XR1.py"
    tree = ast.parse(path.read_text())
    nodes = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "xr1").body if method else tree.body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "ACTION_START_ID": 151671, "ACTION_END_ID": 151731,
                 "STATE_ID": 151670, "SCORE_ID": 151669}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_checkpointed_mlp_deepcopy_is_independent_and_differentiable():
    mlp = nn.Linear(2, 2)
    mlp.forward = MethodType(load_xr1_function("checkpoint_mlp_forward"), mlp)
    reference = copy.deepcopy(mlp).requires_grad_(False)
    x = torch.ones(1, 2, requires_grad=True)
    before = reference(x).detach().clone()
    with torch.no_grad():
        mlp.weight.add_(10)
    torch.testing.assert_close(reference(x), before)
    reference(x).sum().backward()
    assert x.grad is not None and mlp.weight.grad is None


def test_real_gemma_siglip_modules_receive_shared_value_and_q_gradients():
    from transformers import Gemma3TextModel, Gemma3TextConfig, SiglipVisionModel, SiglipVisionConfig
    language = Gemma3TextModel(Gemma3TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, sliding_window=16))
    vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=4))
    encoder = ValueBackbone(language, vision, state_dim=3)
    critic = DIVL(encoder, 32, action_dim=2, horizon=2)
    observation = dict(pixel_values=torch.randn(2, 2, 3, 8, 8), input_ids=torch.ones(2, 3, dtype=torch.long),
                       attention_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]), state=torch.randn(2, 1, 3),
                       view_mask=torch.tensor([[1, 1], [1, 0]]))
    features = critic.encode(observation)
    loss = critic.value_head(features).square().mean() + critic.q(features, torch.randn(2, 2, 2), torch.ones(2, 2, 2)).square().mean()
    loss.backward()
    for module in (encoder.language, encoder.vision, encoder.visual_projector, encoder.state_projector):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_xr1_condition_preserves_batch_and_backpropagates_vlm_cache():
    class FakeVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, input_ids, attention_mask, **kwargs):
            batch, length = input_ids.shape
            value = self.weight.expand(batch, 1, length, 4)
            return SimpleNamespace(past_key_values=[(value, value)],
                position_ids=torch.arange(length).view(1, 1, -1).expand(3, batch, -1), attention_mask=attention_mask)

    model = SimpleNamespace(sink=nn.Embedding(1, 4), active_action_dims=[0], vlm=FakeVLM(),
                            state_projector=nn.Linear(3, 4), rotary_emb=lambda x, positions: positions)
    prepare = MethodType(load_xr1_function("prepare_flow_condition", method=True), model)
    observation = dict(input_ids=torch.ones(2, 3, dtype=torch.long),
                       attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]), state=torch.ones(2, 1, 3))
    mask = torch.zeros(2, 2, 2)
    mask[:, :, 0] = 1
    result = prepare(observation, mask)
    assert set(observation) == {"input_ids", "attention_mask", "state"}
    assert result["state_embed"].shape == (2, 1, 4)
    assert result["attn_mask"].shape == (2, 1, 4, 7)
    assert not result["attn_mask"][1, :, :, 2].any()
    result["past_key_values"][0][0].sum().backward()
    assert model.vlm.weight.grad is not None
    with pytest.raises(ValueError, match="SFT target"):
        prepare({**observation, "input_ids": torch.full((2, 3), 151671)}, mask)
    with pytest.raises(ValueError, match="active_action_dims"):
        prepare(observation, torch.ones_like(mask))
