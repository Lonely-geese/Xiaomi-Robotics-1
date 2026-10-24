"""Real multi-process checks for gradient averaging and exact LWD resume."""

import copy
import random
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mibot.rl.distributed import broadcast_model, mean_metrics, synchronize_gradients
from test_lwd import make_batch, make_learner


def assert_replicas_equal(model):
    values = torch.cat([p.detach().flatten() for p in model.parameters()])
    replicas = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, values)
    for replica in replicas:
        torch.testing.assert_close(replica, values, rtol=0, atol=0)


def distributed_worker(rank, rendezvous, checkpoint_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        # A parameter used on one rank only still gets the global-batch gradient.
        parameter = torch.nn.Parameter(torch.zeros(3))
        unused = torch.nn.Parameter(torch.zeros(1))
        if rank == 0:
            parameter.grad = torch.tensor([2., 4., 6.])
        synchronize_gradients([parameter, unused], bucket_numel=2)
        torch.testing.assert_close(parameter.grad, torch.tensor([1., 2., 3.]))
        assert unused.grad is None

        # Compare to a single-process global-batch derivative, across bucket boundaries.
        weight = torch.nn.Parameter(torch.tensor([.2, -.3, .7]))
        x = torch.tensor([[1., 2., 3.], [4., -1., .5]])
        (weight * x[rank]).sum().square().backward()
        synchronize_gradients([weight], bucket_numel=2)
        expected = torch.autograd.grad(((weight * x).sum(-1).square()).mean(), weight)[0]
        torch.testing.assert_close(weight.grad, expected)
        assert mean_metrics({'value': float(rank)}, torch.device('cpu')) == {'value': .5}

        torch.manual_seed(7)
        learner = make_learner()
        broadcast_model(learner)
        reference = copy.deepcopy(learner.reference.state_dict())
        batch = make_batch()
        batch['action'] += rank * .2
        torch.manual_seed(100 + rank)
        random.seed(100 + rank)
        for _ in range(2):
            learner.update(batch)
        assert_replicas_equal(learner)
        for key, tensor in reference.items():
            torch.testing.assert_close(learner.reference.state_dict()[key], tensor, rtol=0, atol=0)

        checkpoint = learner.checkpoint()
        states = checkpoint['distributed']['rng_by_rank']
        assert not torch.equal(states[0]['torch'], states[1]['torch'])
        if rank == 0:
            torch.save(checkpoint, checkpoint_path)
        dist.barrier()
        expected_metrics = learner.update(batch)
        restored = make_learner()
        restored.restore(torch.load(checkpoint_path, weights_only=True))
        assert restored.update(batch) == expected_metrics
        for actual, expected in zip(restored.parameters(), learner.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_replicas_equal(restored)

        # The online optimizer changes its parameter set; frozen VLM stays fixed.
        restored.start_online(10)
        vlm = copy.deepcopy(restored.actor.model.vlm.state_dict())
        restored.update(batch)
        assert_replicas_equal(restored)
        for key, tensor in vlm.items():
            torch.testing.assert_close(restored.actor.model.vlm.state_dict()[key], tensor, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_distributed_gradients_and_resume(tmp_path):
    mp.spawn(distributed_worker, args=((tmp_path / 'rendezvous').as_uri(), str(tmp_path / 'state.pt')),
             nprocs=2, join=True)
