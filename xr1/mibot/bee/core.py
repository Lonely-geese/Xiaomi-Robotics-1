"""BEE equations (1)-(9); frozen XR1 features and native z-score actions.

No simulator, teacher, physical action clipping, or inferred reward is included.
Paper: https://arxiv.org/pdf/2609.27450, Table III. USB defaults edit pose only.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BeeConfig:
    z_dim: int = 2048
    proprio_dim: int = 16
    chunk_len: int = 10
    action_dim: int = 7
    editable_dims: tuple = (0, 1, 2, 3, 4, 5)
    hidden: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    dual_lr: float = 3e-6
    eta: float = 1.0
    epsilon: float = 0.3
    variance_floor: float = 0.02
    reference_dropout: float = 0.5
    policy_delay: int = 2
    grad_clip: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "editable_dims", tuple(self.editable_dims))
        if self.action_dim != 7 or self.proprio_dim != 16:
            raise ValueError("USB requires 7 action and 16 proprio dimensions")
        if not 1 <= self.chunk_len <= 30 or self.z_dim < 1 or self.hidden < 1:
            raise ValueError("Invalid network/chunk dimensions")
        if not self.editable_dims or len(set(self.editable_dims)) != len(
            self.editable_dims
        ):
            raise ValueError("editable_dims must be nonempty and unique")
        if any(i not in range(7) for i in self.editable_dims):
            raise ValueError("Invalid editable dimension")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid discount/target rate")
        if not 0 <= self.reference_dropout < 1 or self.policy_delay < 1:
            raise ValueError("Invalid dropout/policy delay")
        if (
            any(
                v <= 0
                for v in (
                    self.lr,
                    self.dual_lr,
                    self.epsilon,
                    self.variance_floor,
                    self.grad_clip,
                )
            )
            or self.eta < 0
        ):
            raise ValueError("Invalid optimization parameters")
        if not all(
            torch.isfinite(torch.tensor(v))
            for v in asdict(self).values()
            if isinstance(v, float)
        ):
            raise ValueError("Nonfinite configuration")


def mlp(inputs, outputs, hidden):
    layers = []
    for _ in range(3):
        layers.extend([nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.ReLU()])
        inputs = hidden
    return nn.Sequential(*layers, nn.Linear(hidden, outputs))


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


def td_target(rewards, terminations, next_q, gamma):
    """Full K-step chunks; time-limit truncation still bootstraps."""
    k = rewards.shape[1]
    discount = gamma ** torch.arange(k, device=rewards.device, dtype=rewards.dtype)
    return (rewards * discount).sum(-1) + gamma**k * (~terminations.any(-1)) * next_q


def correction_nll(action, reference, mean, variance, mask):
    # Only explicitly supervised dimensions/timesteps contribute to correction fitting.
    if not mask.any():
        raise ValueError("Correction batch has no editable supervised actions")
    terms = 0.5 * ((action - reference - mean).square() / variance + variance.log())
    return terms.masked_select(mask).mean()  # constant log(2*pi)/2 omitted


class BeePolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        s, d = cfg.z_dim + cfg.proprio_dim, cfg.chunk_len * cfg.action_dim
        self.actor = mlp(s + d, d, cfg.hidden)
        self.correction = mlp(s + d, 2 * d, cfg.hidden)
        self.multiplier = mlp(s, 1, cfg.hidden)
        self.critics = nn.ModuleList([mlp(s + d, 1, cfg.hidden) for _ in range(2)])
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        mask = torch.zeros(cfg.chunk_len, cfg.action_dim, dtype=torch.bool)
        mask[:, list(cfg.editable_dims)] = True
        self.register_buffer("editable", mask)

    def inputs(self, obs):
        return torch.cat([obs["z_rl"], obs["proprio"]], -1), obs["ref_chunk"]

    def action(self, obs, *, dropout=False):
        state, ref = self.inputs(obs)
        conditioned = ref.flatten(1)
        if dropout and self.cfg.reference_dropout:
            keep = (
                torch.rand(len(ref), 1, device=ref.device) >= self.cfg.reference_dropout
            )
            conditioned = conditioned * keep
        residual = self.actor(torch.cat([state, conditioned], -1)).reshape_as(ref)
        # Keep the original proposal as the residual anchor even under input dropout.
        return ref + residual * self.editable

    def distribution(self, obs):
        state, ref = self.inputs(obs)
        mean, raw = self.correction(torch.cat([state, ref.flatten(1)], -1)).chunk(2, -1)
        return mean.reshape_as(ref), F.softplus(raw).reshape_as(
            ref
        ) + self.cfg.variance_floor

    def dual(self, obs):
        return F.softplus(self.multiplier(self.inputs(obs)[0])).squeeze(-1)

    def q(self, obs, actions, *, target=False):
        x = torch.cat([self.inputs(obs)[0], actions.flatten(1)], -1)
        heads = self.target_critics if target else self.critics
        return torch.cat([head(x) for head in heads], -1)

    def rho(self, action, ref, mean, variance):
        # Normalize by full D=K*7, including the noneditable zero contributions.
        return (((action - ref - mean).square() / variance) * self.editable).mean(
            (1, 2)
        )

    def actor_objective(self, obs):
        with torch.no_grad():
            mean, variance = self.distribution(obs)
            weight = self.dual(obs)
        action = self.action(obs, dropout=True)
        ref = obs["ref_chunk"]
        rho = self.rho(action, ref, mean, variance)
        anchor = ((action - ref).square() * self.editable).sum((1, 2))
        q = self.q(obs, action).min(-1).values
        return ((-q + self.cfg.eta * anchor + weight * rho) / (1 + weight)).mean()


class BeeLearner:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.model = BeePolicy(cfg).to(device)
        self.optimizers = {
            name: torch.optim.Adam(
                getattr(self.model, name).parameters(),
                lr=cfg.dual_lr if name == "multiplier" else cfg.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            for name in ("actor", "critics", "correction", "multiplier")
        }
        self.step = 0
        self.correction_steps = 0

    def optimize(self, name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {name} loss")
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            getattr(self.model, name).parameters(),
            self.cfg.grad_clip,
            error_if_nonfinite=True,
        )
        self.optimizers[name].step()

    def update_correction(self, batch):
        m = self.model
        obs = batch["curr_obs"]
        mean, variance = m.distribution(obs)
        mask = batch["correction_mask"] & m.editable
        loss = correction_nll(batch["actions"], obs["ref_chunk"], mean, variance, mask)
        self.optimize("correction", loss)
        self.correction_steps += 1
        return {
            "correction_step": self.correction_steps,
            "correction_nll": float(loss.detach()),
            "variance_mean": float(variance.detach().masked_select(mask).mean()),
        }

    def update(self, batch):
        m, cfg = self.model, self.cfg
        with torch.no_grad():
            action = m.action(batch["next_obs"])
            q = m.q(batch["next_obs"], action, target=True).min(-1).values
            target = td_target(batch["rewards"], batch["terminations"], q, cfg.gamma)
        q = m.q(batch["curr_obs"], batch["actions"])
        critic = F.mse_loss(q, target[:, None].expand_as(q))
        self.optimize("critics", critic)
        with torch.no_grad():
            for old, new in zip(
                m.target_critics.parameters(), m.critics.parameters(), strict=True
            ):
                old.lerp_(new, cfg.tau)
        self.step += 1
        metrics = {
            "step": self.step,
            "critic_loss": float(critic.detach()),
            "q_mean": float(q.detach().mean()),
        }
        if self.step % cfg.policy_delay == 0:
            with frozen(m.critics):
                actor = m.actor_objective(batch["curr_obs"])
                self.optimize("actor", actor)
            with torch.no_grad():
                obs = batch["curr_obs"]
                mean, variance = m.distribution(obs)
                rho = m.rho(m.action(obs), obs["ref_chunk"], mean, variance)
            weight = m.dual(obs)
            dual = -(weight * (rho - cfg.epsilon)).mean()
            self.optimize("multiplier", dual)
            metrics.update(
                actor_loss=float(actor.detach()),
                dual_loss=float(dual.detach()),
                rho=float(rho.mean()),
                multiplier=float(weight.detach().mean()),
                violation_rate=float((rho > cfg.epsilon).float().mean()),
            )
        return metrics

    def save(self, path, assets, training):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "xr1-bee-v1",
            "config": asdict(self.cfg),
            "assets": assets,
            "model": self.model.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "step": self.step,
            "correction_steps": self.correction_steps,
            "training": training,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def restore(self, path, assets, training):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["format"] != "xr1-bee-v1"
            or saved["assets"] != assets
            or saved["config"] != asdict(self.cfg)
        ):
            raise ValueError("BEE checkpoint config/assets mismatch")
        if saved["training"] != training:
            raise ValueError("BEE training data/schedule mismatch")
        self.model.load_state_dict(saved["model"], strict=True)
        for k, opt in self.optimizers.items():
            opt.load_state_dict(saved["optimizers"][k])
        self.step, self.correction_steps = saved["step"], saved["correction_steps"]
        torch.set_rng_state(saved["rng"])
        if saved["cuda_rng"] is not None:
            if not next(self.model.parameters()).is_cuda:
                raise ValueError("Exact resume requires the original CUDA device type")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
