"""Load every BEE tensor strictly; refuse to use the untrained actor.

The first-half residual interpretation of the correction head is an explicit
experimental convention. The second half is exported raw, not claimed to be
calibrated variance (the original parameterization is unavailable).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim, hidden):
    layers = []
    for i in range(3):
        layers.extend([nn.Linear(input_dim if i == 0 else hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()])
    return nn.Sequential(*layers, nn.Linear(hidden, output_dim))


class BEE(nn.Module):
    def __init__(self, config, trained_steps=0):
        super().__init__()
        self.config = dict(config)
        self.trained_steps = int(trained_steps)
        self.chunk_len = int(config['chunk_len'])
        self.action_dim = int(config['action_dim'])
        self.z_dim = int(config['z_dim'])
        self.proprio_dim = int(config['proprio_dim'])
        n = self.chunk_len * self.action_dim
        obs = self.z_dim + self.proprio_dim
        hidden = int(config['hidden'])
        editable = torch.zeros(self.chunk_len, self.action_dim, dtype=torch.bool)
        editable[:, list(config['editable_dims'])] = True
        self.register_buffer('editable', editable)
        self.actor = mlp(obs + n, n, hidden)
        self.correction = mlp(obs + n, 2 * n, hidden)
        self.multiplier = mlp(obs, 1, hidden)
        self.critics = nn.ModuleList([mlp(obs + n, 1, hidden) for _ in range(2)])
        self.target_critics = nn.ModuleList([mlp(obs + n, 1, hidden) for _ in range(2)])
        if config.get('algorithm') == 'bee-offline-iql-v1':
            self.value = mlp(obs, 1, hidden)

    def inputs(self, z, proprio, reference):
        b = z.shape[0]
        if z.shape != (b, self.z_dim) or proprio.shape != (b, self.proprio_dim):
            raise ValueError('BEE feature dimensions do not match the checkpoint')
        if reference.shape != (b, self.chunk_len, self.action_dim):
            raise ValueError('BEE action window does not match the checkpoint')
        if not all(torch.isfinite(x).all() for x in (z, proprio, reference)):
            raise ValueError('Nonfinite BEE inputs')
        return torch.cat([z, proprio, reference.flatten(1)], dim=-1)

    def correct(self, z, proprio, reference):
        raw = self.correction(self.inputs(z, proprio, reference))
        delta, spread_raw = raw.chunk(2, dim=-1)
        delta = delta.reshape_as(reference).masked_fill(~self.editable, 0)
        return {
            'corrected': reference + delta,
            'delta': delta,
            'spread_raw': spread_raw.reshape_as(reference),
        }

    def actor_action(self, z, proprio, reference):
        if self.trained_steps == 0:
            raise RuntimeError('The checkpoint actor is untrained (step=0); use correction experiments only')
        if self.config.get('algorithm') not in ('bee-paper-v1', 'bee-offline-iql-v1'):
            raise RuntimeError('Legacy actor parameterization is unknown')
        return self.policy(z, proprio, reference)

    def policy(self, z, proprio, reference, reference_dropout=0.):
        """Training primitive; deployment must use guarded actor_action().

        Drop whole conditioning proposals, never the residual skip connection.
        The paper specifies dropout rate but not granularity; this is our choice.
        """
        conditioned = reference
        if reference_dropout:
            keep = torch.rand((len(z), 1, 1), device=z.device) >= reference_dropout
            conditioned = reference * keep
        delta = self.actor(self.inputs(z, proprio, conditioned)).reshape_as(reference)
        return reference + delta.masked_fill(~self.editable, 0)

    def distribution(self, z, proprio, reference):
        if self.config.get('variance_parameterization') != 'softplus-plus-floor-v1':
            raise RuntimeError('Legacy correction variance semantics are unknown; migrate explicitly')
        result = self.correct(z, proprio, reference)
        result['variance'] = F.softplus(result['spread_raw']) + self.config.get('variance_floor', .02)
        return result

    def lagrange(self, z, proprio):
        return F.softplus(self.multiplier(torch.cat([z, proprio], -1))).squeeze(-1)

    def q_values(self, z, proprio, action, target=False, valid_mask=None):
        if self.config.get('algorithm') == 'bee-offline-iql-v1':
            # Offline action space is pose6. Slot 7 is a horizon mask, NOT an
            # invented executed gripper command. Preserve the actor's 7D ABI.
            if valid_mask is None:
                valid_mask = torch.ones_like(action[..., 0], dtype=torch.bool)
            pose = action[..., :6].masked_fill(~valid_mask.bool()[..., None], 0)
            action = torch.cat([pose, valid_mask.to(action.dtype)[..., None]], -1)
        x = self.inputs(z, proprio, action)
        return torch.cat([q(x) for q in (self.target_critics if target else self.critics)], -1)

    def state_value(self, z, proprio):
        return self.value(torch.cat([z, proprio], -1)).squeeze(-1)


def load_bee(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if ckpt.get('format') != 'xr1-bee-v1':
        raise ValueError('Unsupported BEE checkpoint format')
    model = BEE(ckpt['config'], ckpt.get('step', 0))
    model.load_state_dict(ckpt['model'], strict=True)
    expected = torch.zeros_like(model.editable)
    expected[:, list(ckpt['config']['editable_dims'])] = True
    if not torch.equal(model.editable, expected):
        raise ValueError('Checkpoint editable mask contradicts config')
    return model.to(device).eval().requires_grad_(False), {k: v for k, v in ckpt.items() if k != 'model'}
