"""RECAP math and exact distributed demonstration/rollout sampling.

Algorithm reference: RLinf/examples/offline_rl/advantage_labeling/recap.
The critic architecture is independent of these return/TD/CFG definitions.
"""
import math

import numpy as np
import torch
from torch.utils.data import Sampler


def episode_returns(length, success, gamma=1.0, failure_reward=-300.0):
    if length <= 0 or not 0 < gamma <= 1:
        raise ValueError("Expected a positive episode length and gamma in (0, 1]")
    rewards = np.full(length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if success else failure_reward
    returns = np.empty_like(rewards)
    running = 0.0
    for i in range(length - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return rewards, returns


def nstep_advantages(rewards, values, scale, horizon=30, gamma=1.0):
    """Zero bootstrap only at the true episode end; values use [-1, 0].

    Returns have maximum zero, so RLinf's affine normalization reduces to
    division by the global return range, including for the n-step reward sum.
    """
    rewards, values = np.asarray(rewards), np.asarray(values)
    if rewards.shape != values.shape or scale <= 0 or horizon <= 0:
        raise ValueError("Invalid rewards, values, scale or horizon")
    result = np.empty(len(values), dtype=np.float32)
    for t in range(len(values)):
        k = min(horizon, len(values) - t)
        reward_sum = np.dot(gamma ** np.arange(k), rewards[t:t+k])
        bootstrap = values[t+k] if t+k < len(values) else 0.0
        result[t] = reward_sum / scale + gamma**k * bootstrap - values[t]
    return result


def action_intervention_mask(row):
    """1=known human action, 0=known autonomous, -1=unknown coverage."""
    n = int(row['length'])
    known = row.get('human_intervened') is not None
    mask = np.full(n, 0 if known else -1, dtype=np.int8)
    for interval in row.get('action_human_intervals', []):
        a, b = int(interval['start_frame_index']), int(interval['end_frame_index_exclusive'])
        if not 0 <= a < b <= n:
            raise ValueError(f"Action intervention interval outside output: {interval}")
        mask[a:b] = 1
    if row.get('human_intervened') is True and not row.get('action_human_intervals'):
        # An episode-level flag cannot locate the human actions.
        mask[:] = -1
    return mask


def conditional_flags(positive, unconditional_prob, training=True):
    positive = positive.bool()
    if not 0 <= unconditional_prob <= 1:
        raise ValueError("unconditional_prob must be in [0, 1]")
    if training:
        return positive & (torch.rand(positive.shape, device=positive.device) >= unconditional_prob)
    return positive


def rlinf_positive_labels(pool, advantage, threshold):
    """Match RLinf's dataset-type override after global quantile selection.

    Demonstrations are always positive. Rollouts use the inclusive threshold;
    unknown-outcome rollout frames remain unconditional. Human corrections in
    rollouts receive no additional override in RLinf's offline implementation.
    """
    advantage = np.asarray(advantage)
    if pool == 'sft':
        return np.ones(advantage.shape, dtype=bool)
    if pool != 'rollout':
        raise ValueError(f'Unknown RECAP pool: {pool}')
    return np.isfinite(advantage) & (advantage >= threshold)


class BalancedPoolSampler(Sampler):
    """Every local batch has exactly half of each pool, on every rank.

    An epoch covers the larger pool once, dropping only its incomplete half
    global batch. The smaller pool is reshuffled and cycled as necessary.
    All ranks construct the same global stream before taking their slices.
    """
    def __init__(self, sft_size, rollout_size, batch_size=32, num_replicas=1, rank=0, seed=42):
        if min(sft_size, rollout_size, batch_size, num_replicas) <= 0 or batch_size % 2:
            raise ValueError("Nonempty pools and an even positive local batch are required")
        if not 0 <= rank < num_replicas:
            raise ValueError("Invalid sampler rank")
        self.sft_size, self.rollout_size = int(sft_size), int(rollout_size)
        self.batch_size, self.num_replicas, self.rank = batch_size, num_replicas, rank
        self.seed, self.epoch = seed, 0
        self.start_batch = 0
        self.half_global = batch_size // 2 * num_replicas
        self.steps = max(sft_size, rollout_size) // self.half_global
        if self.steps == 0:
            raise ValueError("Pools too small for a balanced global batch")

    def __len__(self):
        return self.steps * self.batch_size

    def set_epoch(self, epoch):
        if int(epoch) != self.epoch:
            self.start_batch = 0
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        count = self.steps * self.half_global

        def stream(size):
            return np.concatenate([rng.permutation(size) for _ in range(math.ceil(count / size))])[:count]

        sft = stream(self.sft_size).reshape(self.steps, self.num_replicas, -1)[:, self.rank]
        rollout = stream(self.rollout_size).reshape(self.steps, self.num_replicas, -1)[:, self.rank]
        # Interleave the two pools: local microbatch boundaries remain exact.
        indices = np.stack([sft, rollout + self.sft_size], axis=-1).reshape(-1)
        return iter(indices[self.start_batch*self.batch_size:].tolist())
