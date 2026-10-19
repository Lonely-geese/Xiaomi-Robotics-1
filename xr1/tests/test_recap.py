import sys
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compute_recap_advantages import label_episode
from mibot.data.recap import (
    BalancedPoolSampler, action_intervention_mask, conditional_flags,
    episode_returns, nstep_advantages,
)
from mibot.models.recap_value import RecapValueCritic
from mibot.models.VLA.XR1 import xr1
from mibot.data.datasets.recap_dataset import RecapJsonDataset


def test_returns_and_nstep_terminal_bootstrap():
    rewards, returns = episode_returns(4, True)
    np.testing.assert_array_equal(returns, [-3, -2, -1, 0])
    np.testing.assert_allclose(nstep_advantages(rewards, returns/10, 10, horizon=2), 0, atol=1e-7)
    rewards, returns = episode_returns(4, False)
    np.testing.assert_array_equal(returns, [-303, -302, -301, -300])
    np.testing.assert_allclose(nstep_advantages(rewards, returns/1000, 1000, horizon=30), 0, atol=1e-7)


def test_discounted_returns_and_advantages():
    rewards, returns = episode_returns(5, True, gamma=.9)
    np.testing.assert_allclose(nstep_advantages(rewards, returns/10, 10, horizon=2, gamma=.9), 0, atol=1e-7)


@pytest.mark.parametrize('sizes', [(1280,300), (300,1280), (1280,1280)])
def test_all_eight_ranks_have_exact_pool_ratio_and_cover_larger_pool(sizes):
    all_sft, all_rollout = [], []
    streams = []
    for rank in range(8):
        sampler = BalancedPoolSampler(*sizes, batch_size=32, num_replicas=8, rank=rank)
        batches = np.array(list(sampler)).reshape(-1, 32)
        assert len(batches) == 10
        assert np.all((batches < sizes[0]).sum(axis=1) == 16)
        all_sft.extend(batches[batches < sizes[0]].tolist())
        all_rollout.extend((batches[batches >= sizes[0]]-sizes[0]).tolist())
        first = batches.copy()
        sampler.set_epoch(1)
        assert not np.array_equal(first.flatten(), np.array(list(sampler)))
        streams.append(first)
    large = all_sft if sizes[0] >= sizes[1] else all_rollout
    assert len(set(large)) == 1280
    assert min(all_sft) >= 0 and max(all_sft) < sizes[0]
    assert min(all_rollout) >= 0 and max(all_rollout) < sizes[1]


def test_intervention_uses_action_time_and_preserves_unknown():
    row = dict(length=12, human_intervened=True,
        human_intervals=[dict(start_frame_index=8,end_frame_index_exclusive=12)],
        action_human_intervals=[dict(start_frame_index=5,end_frame_index_exclusive=12)])
    np.testing.assert_array_equal(action_intervention_mask(row), [0]*5+[1]*7)
    assert np.all(action_intervention_mask(dict(length=8,human_intervened=None)) == -1)


def test_human_corrections_override_outcomes_and_pre_takeover_is_negative():
    human = np.array([0,0,0,1,1,0,0,-1])
    scores = np.ones(8)
    positive = label_episode('rollout', scores, human, .5, horizon=3)
    np.testing.assert_array_equal(positive, [True,False,False,True,True,True,True,True])
    unknown = label_episode('rollout', np.full(8,np.nan), human, .5, horizon=3)
    np.testing.assert_array_equal(unknown, human == 1)
    assert label_episode('sft', scores, human, .5).all()


def test_cfg_dropout_never_conditions_negative_samples():
    labels = torch.tensor([True,False,True,False])
    assert torch.equal(conditional_flags(labels, 0), labels)
    assert not conditional_flags(labels, 1).any()
    assert torch.equal(conditional_flags(labels, 1, training=False), labels)


def test_action_chunk_stops_before_human_takeover(tmp_path):
    path = tmp_path/'sidecar.npz'
    np.savez(path, human=np.array([0]*12+[1]*28, dtype=np.int8))
    dataset = RecapJsonDataset.__new__(RecapJsonDataset)
    dataset.action_length = 30
    traj = dict(num_frames=40, recap=dict(pool='rollout', offset=0, sidecar=str(path)))
    assert dataset._action_steps(traj, 10) == 2
    assert dataset._action_steps(traj, 12) == 28
    assert dataset._action_steps(traj, 39) == 1
    traj['recap']['pool'] = 'sft'
    assert dataset._action_steps(traj, 10) == 30


def test_cfg_inference_combines_two_velocities():
    model = xr1.__new__(xr1)
    torch.nn.Module.__init__(model)
    model.recap_enabled, model.recap_guidance_scale, model.num_steps = True, 2.0, 5
    model.eval()
    def velocity(self, sample, timestep, recap_condition):
        return torch.ones_like(sample)*(1+recap_condition[:,None,None])
    model.dit_forward = MethodType(velocity, model)
    result = model._generate(torch.zeros(2,30,60), dict(recap_condition=torch.ones(2)))
    torch.testing.assert_close(result, torch.full_like(result,3.0))


def test_categorical_critic_has_finite_gradients_and_bounded_values():
    model = RecapValueCritic()
    images = torch.randint(0,256,(2,3,3,64,64),dtype=torch.uint8)
    state = torch.randn(2,30)
    logits = model(images,state)
    loss = model.loss(logits, torch.tensor([-1.0,0.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.vision[0].weight.grad.norm() > 0
    values = model.value(images,state)
    assert ((values >= -1) & (values <= 0)).all()
