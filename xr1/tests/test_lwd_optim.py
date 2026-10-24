"""Precision, migration, distributed equivalence and checkpoint regression tests."""

import copy
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mibot.rl.optim import ShardedMasterAdamW, clip_grad_norm_fp32
from mibot.rl.qam import build_adjoint_targets, local_loss
from test_lwd import make_batch, make_learner


def test_bfloat16_clipping_uses_fp32_norm_and_rejects_nonfinite():
    p = torch.nn.Parameter(torch.ones(1000, dtype=torch.bfloat16))
    p.grad = torch.linspace(-3., 4., 1000).bfloat16()
    initial = p.grad.float().clone()
    expected = torch.linalg.vector_norm(initial)
    actual = clip_grad_norm_fp32([p], 1.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, (initial / (expected + 1e-6)).bfloat16(), rtol=0, atol=0)
    p.grad[0] = torch.inf
    with pytest.raises(FloatingPointError):
        clip_grad_norm_fp32([p], 1.)


def test_master_accumulates_updates_below_bfloat16_resolution():
    low = torch.nn.Parameter(torch.tensor([.02, 1.], dtype=torch.bfloat16))
    initial = low.detach().clone()
    reference = torch.nn.Parameter(low.detach().float())
    opt = ShardedMasterAdamW([low], lr=2e-5)
    expected = torch.optim.AdamW([reference], lr=2e-5, foreach=False)
    for _ in range(100):
        low.grad = torch.full_like(low, 1e-4)
        reference.grad = low.grad.float()
        opt.step()
        expected.step()
    torch.testing.assert_close(opt.masters[0], reference, rtol=0, atol=0)
    torch.testing.assert_close(low, reference.to(low.dtype), rtol=0, atol=0)
    assert torch.all(low != initial)
    assert all(s['exp_avg'].dtype == torch.float32 for s in opt.inner.state.values())


def test_legacy_migration_preserves_steps_moments_and_resume(tmp_path):
    learner = make_learner()
    batch = make_batch()
    learner.update(batch)
    old = copy.deepcopy(learner.checkpoint())
    restored = make_learner()
    restored.restore(old, actor_optimizer='fp32_master_sharded')
    assert restored.config.actor_optimizer == 'fp32_master_sharded'
    assert restored.updates == learner.updates
    assert restored.schedulers[0].state_dict() == learner.schedulers[0].state_dict()
    for index, master in zip(restored.actor_optimizer.local_indices, restored.actor_optimizer.masters):
        torch.testing.assert_close(master, restored.actor_optimizer.parameters[index].float(), rtol=0, atol=0)
    states = list(restored.actor_optimizer.inner.state.values())
    for actual, expected in zip(states, old['actor_optimizer']['state'].values(), strict=True):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(actual[key], expected[key].float(), rtol=0, atol=0)
    restored.update(batch)
    path = tmp_path / 'state.pt'
    torch.save(restored.checkpoint(), path)
    expected = restored.update(batch)
    resumed = make_learner()
    resumed.restore(torch.load(path, weights_only=True))
    assert resumed.update(batch) == expected
    for actual, expected in zip(resumed.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qam_diagnostics_preserve_loss_and_gradients():
    mask = torch.ones(2, 3, 2)
    targets = build_adjoint_targets(lambda x, t: -x, lambda a: a.square().flatten(1).sum(-1), mask)
    p = torch.tensor(.3, requires_grad=True)
    basic = local_loss(lambda x, t: p*x, targets, mask)
    detailed, metrics = local_loss(lambda x, t: p*x, targets, mask, diagnostics=True)
    torch.testing.assert_close(basic, detailed, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(basic, p)[0], torch.autograd.grad(detailed, p)[0], rtol=0, atol=0)
    torch.testing.assert_close(detailed.detach(), sum(metrics[k] for k in
                               ['qam_policy_term', 'qam_adjoint_term', 'qam_cross_term']))


def test_lr_override_preserves_schedule_progress_and_adam_state():
    learner = make_learner()
    batch = make_batch()
    learner.update(batch)
    checkpoint = copy.deepcopy(learner.checkpoint())
    resumed = make_learner()
    resumed.restore(checkpoint, actor_optimizer='fp32_master_sharded', actor_lr=learner.config.actor_lr/10)
    assert resumed.updates == learner.updates
    assert resumed.schedulers[0].last_epoch == learner.schedulers[0].last_epoch
    assert resumed.actor_optimizer.param_groups[0]['lr'] == pytest.approx(learner.actor_optimizer.param_groups[0]['lr']/10)
    assert resumed.schedulers[0].base_lrs[0] == pytest.approx(learner.schedulers[0].base_lrs[0]/10)
    assert resumed.critic_optimizer.param_groups[0]['lr'] == learner.critic_optimizer.param_groups[0]['lr']
    for state in resumed.actor_optimizer.inner.state.values():
        assert state['step'] == 1
    resumed.update(batch)
    learner.update(batch)
    assert resumed.actor_optimizer.param_groups[0]['lr'] == pytest.approx(learner.actor_optimizer.param_groups[0]['lr']/10)
    changed = copy.deepcopy(resumed.checkpoint())
    expected = resumed.update(batch)
    continued = make_learner()
    continued.restore(changed)
    assert continued.update(batch) == expected


def distributed_master_worker(rank, rendezvous, directory):
    from pathlib import Path
    from tools.train_lwd import save_checkpoint
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        # All ranks see the averaged gradient. Ownership must not change AdamW.
        params = [torch.nn.Parameter(torch.ones(n, dtype=torch.bfloat16)) for n in [2, 3, 5]]
        unused = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        params.append(unused)
        reference = [torch.nn.Parameter(p.detach().float()) for p in params]
        opt = ShardedMasterAdamW(params, lr=0.001)
        baseline = torch.optim.AdamW(reference, lr=0.001, foreach=False)
        for step in range(4):
            for p, ref in zip(params[:-1], reference[:-1], strict=True):
                p.grad = torch.full_like(p, .1*(step+1)); ref.grad = p.grad.float()
            opt.step(); baseline.step()
            for p, ref in zip(params, reference, strict=True):
                torch.testing.assert_close(p, ref.bfloat16(), rtol=0, atol=0)
        assert torch.equal(unused, torch.ones_like(unused))
        for i, master in zip(opt.local_indices, opt.masters, strict=True):
            torch.testing.assert_close(master, reference[i], rtol=0, atol=0)

        torch.manual_seed(7)
        learner = make_learner()
        learner.config = replace(learner.config, actor_optimizer='fp32_master_sharded')
        learner._configure_optimizers()
        batch = make_batch()
        torch.manual_seed(100+rank)
        learner.update(batch)
        save_checkpoint(learner, Path(directory))
        expected = learner.update(batch)
        resumed = make_learner()
        resumed.restore(torch.load(Path(directory)/'latest.pt', weights_only=True))
        assert resumed.update(batch) == expected
        for actual, expected in zip(resumed.parameters(), learner.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_distributed_master_equivalence_and_sharded_checkpoint(tmp_path):
    mp.spawn(distributed_master_worker,
             args=((tmp_path/'rendezvous').as_uri(), str(tmp_path/'output')), nprocs=2, join=True)
