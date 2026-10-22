"""Distribution over replay Q values, adaptive quantile TD, and twin scalar Q."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def categorical_projection(values, support):
    """Project scalar Q targets onto the adjacent uniformly spaced atoms."""
    values = values.detach().float().reshape(-1).clamp(support[0], support[-1])
    location = (values - support[0]) / (support[-1] - support[0]) * (len(support) - 1)
    lower = location.floor().long().clamp(0, len(support) - 1)
    upper = (lower + 1).clamp(max=len(support) - 1)
    upper_weight = location - lower
    result = values.new_zeros(len(values), len(support))
    result.scatter_add_(1, lower[:, None], (1 - upper_weight)[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


@torch.no_grad()
def adaptive_quantile(logits, support, config):
    log_probs = logits.float().log_softmax(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1) / math.log(logits.shape[-1])
    tau = (config.tau_base - config.entropy_coefficient * entropy).clamp(
        config.tau_min, config.tau_max
    )
    index = (probs.cumsum(-1) < tau[:, None]).sum(-1).clamp(max=len(support) - 1)
    return support[index], tau, entropy


class TemporalActionPool(nn.Module):
    def __init__(self, action_dim, hidden_dim, horizon):
        super().__init__()
        self.project = nn.Linear(action_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, action, mask):
        if action.shape != mask.shape or action.ndim != 3:
            raise ValueError("action and mask must both have shape [B,H,A]")
        valid = mask.bool().any(-1)
        if not valid.any(-1).all():
            raise ValueError("every action chunk needs at least one valid element")
        if action.shape[1] > len(self.position):
            raise ValueError("action chunk exceeds configured horizon")
        hidden = torch.tanh(self.project(action * mask) + self.position[:action.shape[1]])
        scores = self.attention(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        return (hidden * scores.softmax(-1)[..., None]).sum(1)


class DIVL(nn.Module):
    """Encoder is shared by V and Q, independent of the XR-1 actor encoder."""

    def __init__(self, encoder, feature_dim, action_dim=60, horizon=30, config=None):
        super().__init__()
        from .config import LWDConfig
        config = config or LWDConfig()
        self.encoder = encoder
        self.register_buffer("support", torch.linspace(config.v_min, config.v_max, config.atoms))
        self.value_head = nn.Linear(feature_dim, config.atoms)
        self.action_pool = TemporalActionPool(action_dim, feature_dim, horizon)
        self.q_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.SiLU(), nn.Linear(feature_dim, 1))
            for _ in range(2)
        ])

    def encode(self, observation):
        return self.encoder(observation)

    def q(self, features, action, mask):
        joint = torch.cat([features, self.action_pool(action, mask)], -1)
        return torch.cat([head(joint) for head in self.q_heads], -1)

    def loss(self, target, batch, config):
        features = self.encode(batch["critic_observation"])
        logits = self.value_head(features)
        qs = self.q(features, batch["action"], batch["action_mask"])
        with torch.no_grad():
            replay_q = target.q(target.encode(batch["critic_observation"]),
                                batch["action"], batch["action_mask"]).amin(-1)
            probabilities = categorical_projection(replay_q, self.support)
            next_logits = target.value_head(target.encode(batch["next_critic_observation"]))
            next_v, tau, entropy = adaptive_quantile(next_logits, self.support, config)
            # Scalars are [B], never [B,1]: avoid accidental [B,B] broadcasting.
            reward = batch["reward"].reshape(-1).float()
            discount = config.gamma ** batch["steps"].reshape(-1).float()
            bootstrap = (~batch["terminated"].reshape(-1).bool()).float()
            td = reward + discount * bootstrap * next_v
        v_loss = -(probabilities * logits.float().log_softmax(-1)).sum(-1).mean()
        q_loss = F.mse_loss(qs.float(), td[:, None].expand_as(qs))
        return v_loss + q_loss, {
            "value_loss": v_loss.detach(), "q_loss": q_loss.detach(),
            "q_mean": qs.detach().mean(), "td_mean": td.mean(),
            "tau": tau.mean(), "entropy": entropy.mean(),
        }


@torch.no_grad()
def update_ema(target, source, rate):
    for target_parameter, parameter in zip(target.parameters(), source.parameters(), strict=True):
        target_parameter.lerp_(parameter, rate)
    for target_buffer, buffer in zip(target.buffers(), source.buffers(), strict=True):
        target_buffer.copy_(buffer)
