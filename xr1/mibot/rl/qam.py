"""QAM local regression; no gradient through an unrolled action sampler.

The stabilized SDE and backward VJP discretization are adapted from
ColinQiyangLi/qam/agents/qam.py (MIT; see THIRD_PARTY_NOTICES.md).
LWD uses a fixed SFT reference for both rollout and adjoints, and min twin Q.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class AdjointTargets:
    states: list
    times: list
    sigmas: list
    adjoints: list
    reference_velocities: list


def build_adjoint_targets(reference_velocity, q_value, mask, steps=5, temperature=2., generator=None):
    """All returned targets are detached; VJPs differentiate only w.r.t. actions.

    Velocity closures accept x:[B,H,A], t:[B,1,1]. q_value returns [B].
    Callers must provide a frozen reference and detached conditioning features.
    """
    if steps < 1 or temperature <= 0:
        raise ValueError("steps and temperature must be positive")
    mask = mask.float()
    if mask.ndim != 3 or not mask.bool().flatten(1).any(-1).all():
        raise ValueError("mask must be [B,H,A] with a valid action in each sample")
    h = 1. / steps
    states, times, sigmas, reference_velocities = [], [], [], []

    def noise():
        return torch.randn(mask.shape, device=mask.device, dtype=torch.float32, generator=generator) * mask

    with torch.no_grad():
        x = noise()
        for i in range(steps):
            t = x.new_full((x.shape[0], 1, 1), i * h)
            sigma = math.sqrt(2 * (1 - i * h + h) / (i * h + h))
            v = reference_velocity(x, t).float() * mask
            states.append(x.detach())
            times.append(t)
            sigmas.append(sigma)
            reference_velocities.append(v.detach())
            if i == steps - 1:
                x = (x + h * v) * mask
            else:
                x = (x + h * (2 * v - x / (t + h)) + math.sqrt(h) * sigma * noise()) * mask

    with torch.enable_grad():
        endpoint = x.detach().requires_grad_(True)
        q = q_value(endpoint * mask)
        if q.shape != (mask.shape[0],):
            raise ValueError("q_value must return [B]")
        adjoint = -torch.autograd.grad(q.sum(), endpoint)[0].detach() * mask / temperature
        adjoints = []
        for i in reversed(range(steps)):
            xi = states[i].detach().requires_grad_(True)
            # Follow the authors' shifted-time backward discretization, including
            # the last step; this is not backpropagation through Euler sampling.
            drift = (2 * reference_velocity(xi, times[i] + h).float()
                     - xi / (times[i] + h)) * mask
            vjp = torch.autograd.grad(drift, xi, grad_outputs=adjoint)[0]
            adjoint = (adjoint + h * vjp).detach() * mask
            adjoints.append(adjoint)
    return AdjointTargets(states, times, sigmas, list(reversed(adjoints)), reference_velocities)


def local_loss(velocity, targets, mask):
    """Sum action dimensions and time, mean batch, as in the QAM code."""
    loss = 0.
    for x, t, sigma, adjoint, base in zip(
        targets.states, targets.times, targets.sigmas,
        targets.adjoints, targets.reference_velocities, strict=True
    ):
        residual = (2 * (velocity(x, t).float() - base) / sigma + sigma * adjoint) * mask
        loss = loss + residual.square().flatten(1).sum(-1).mean()
    return loss
