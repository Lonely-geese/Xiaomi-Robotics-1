"""Checkpoint boundaries, input parity and safe batch-size migration."""

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from mibot.steam import policy_prefetch
from mibot.steam.policy_prefetch import PolicyPrefetcher, prepare_batch
from mibot.steam.runtime import validate_resume_config


def test_resume_allows_only_explicit_batch_and_output_changes():
    saved = {"micro_batch_size": 16, "global_batch_size": 128, "output": "old", "lr": 2e-6}
    current = dict(saved, micro_batch_size=32, global_batch_size=256, output="new")
    allowed = ("micro_batch_size", "global_batch_size", "output")
    validate_resume_config(saved, current, allowed)
    with pytest.raises(ValueError):
        validate_resume_config(saved, current)
    with pytest.raises(ValueError, match="lr"):
        validate_resume_config(saved, dict(current, lr=1e-4), allowed)


def test_ordered_prefetch_preserves_consumed_rng_and_resume(monkeypatch):
    def pool(**kwargs):
        return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

    monkeypatch.setattr(policy_prefetch, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(policy_prefetch, "_process_batch", lambda indices, seed: (indices.tolist(), seed))
    rng = np.random.default_rng(123)
    reference = np.random.default_rng(123)
    stream = PolicyPrefetcher({}, 100, rng, 8, pin_memory=False)
    try:
        for _ in range(5):
            indices = reference.integers(100, size=8)
            seed = int(reference.integers(0, 2**63 - 8))
            assert next(stream) == (indices.tolist(), seed)
            assert rng.bit_generator.state == reference.bit_generator.state
        saved = copy.deepcopy(rng.bit_generator.state)
        expected = next(stream)
    finally:
        stream.close()
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    resumed = PolicyPrefetcher({}, 100, restored, 8, pin_memory=False)
    try:
        assert next(resumed) == expected
    finally:
        resumed.close()


def test_parallel_items_and_cfg_draws_match_serial():
    class Dataset:
        def load_seeded(self, request):
            index, seed = request
            return (index, float(np.random.default_rng(seed).random()))

    def collate(items, random_values):
        return torch.tensor(items), random_values

    dataset = Dataset()
    indices = np.array([3, 2, 9, 3, 1])
    expected = prepare_batch(dataset, collate, indices, 789)
    cpu_rng = torch.get_rng_state()
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = prepare_batch(dataset, collate, indices, 789, pool)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    assert torch.equal(cpu_rng, torch.get_rng_state())


def test_prefetch_surfaces_worker_errors(monkeypatch):
    def pool(**kwargs):
        return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

    def fail(indices, seed):
        raise ValueError("bad frame")

    monkeypatch.setattr(policy_prefetch, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(policy_prefetch, "_process_batch", fail)
    stream = PolicyPrefetcher({}, 100, np.random.default_rng(0), 8, pin_memory=False)
    try:
        with pytest.raises(ValueError, match="bad frame"):
            next(stream)
    finally:
        stream.close()
