"""Single-process RLT actor-critic learner matching RLinf's default Q-head path.

The upstream transformer and MLP are unmodified. This loop replaces Ray/FSDP
plumbing only; no reward model or robot transport is inferred from demonstrations.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from .upstream.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def build_policy(config):
    return RLTMLPPolicy(
        z_dim=config["rlt"]["embed_dim"],
        proprio_dim=16,
        action_dim=7,
        num_action_chunks=config["stage2"]["chunk_len"],
        ref_num_action_chunks=30,
        fixed_std=config["stage2"]["fixed_std"],
    )


def validate_batch(batch, policy):
    required = {
        "curr_obs",
        "next_obs",
        "actions",
        "rewards",
        "terminations",
        "truncations",
        "intervene_flags",
    }
    if not required.issubset(batch):
        raise ValueError(f"Missing transition fields: {sorted(required - set(batch))}")
    b = batch["actions"].shape[0]
    k, a = policy.chunk_len, policy.step_action_dim
    expected = {
        "actions": (b, k, a),
        "rewards": (b, k),
        "terminations": (b, k),
        "truncations": (b, k),
        "intervene_flags": (b, k, a),
    }
    for key, shape in expected.items():
        value = batch[key]
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
    if (batch["actions"].abs() >= 1).any():
        raise ValueError(
            "Replay actions must use bounded RLT coordinates, not raw XR1 z-scores"
        )
    for key in ("terminations", "truncations", "intervene_flags"):
        if batch[key].dtype != torch.bool:
            raise ValueError(f"{key} must have bool dtype")
    # A fixed K-step transition cannot silently span resets or include actions
    # that were never executed. A partial final chunk needs its own collector.
    if (batch["terminations"][:, :-1] | batch["truncations"][:, :-1]).any():
        raise ValueError(
            "Only complete K-step chunks are accepted; do not pad actions across an early episode end"
        )
    for key in ("curr_obs", "next_obs"):
        for field, shape in {
            "z_rl": (b, policy.z_dim),
            "proprio": (b, policy.proprio_dim),
            "ref_chunk": (b, policy.ref_chunk_len, a),
        }.items():
            value = batch[key][field]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"{key}.{field} must be finite with shape {shape}")
        if (batch[key]["ref_chunk"].abs() >= 1).any():
            raise ValueError("VLA reference must use bounded RLT coordinates")


def td_target(rewards, terminations, next_q, gamma):
    k = rewards.shape[-1]
    discount = gamma ** torch.arange(k, device=rewards.device, dtype=rewards.dtype)
    reward = (rewards * discount).sum(-1, keepdim=True)
    not_done = ~terminations.any(-1, keepdim=True)
    return reward + not_done * (gamma**k) * next_q


def bc_loss(pi, actions, ref_chunk, intervene_flags):
    """Per timestep intervention replaces the entire 7-D BC target, as RLinf."""
    pi = pi.reshape_as(actions)
    human = intervene_flags.bool().any(-1)
    reference = ref_chunk[:, : actions.shape[1]]
    target = torch.where(human[..., None], actions, reference)
    return F.mse_loss(pi, target)


class RLTLearner:
    def __init__(self, config, device="cpu"):
        self.config = copy.deepcopy(config)
        self.cfg = config["stage2"]
        self.policy = build_policy(config).to(device=device, dtype=torch.float32)
        self.target = copy.deepcopy(self.policy).requires_grad_(False).eval()
        self.actor_parameters = [
            p
            for name, p in self.policy.named_parameters()
            if not name.startswith("q_head.") and not name.startswith("actor_logstd.")
        ]
        self.actor_optim = torch.optim.Adam(self.actor_parameters, lr=self.cfg["lr"])
        self.critic_optim = torch.optim.Adam(
            self.policy.q_head.parameters(), lr=self.cfg["lr"]
        )
        self.step = 0

    def losses(self, batch):
        """Expose nonmutating losses for parity tests; no optimizer step."""
        validate_batch(batch, self.policy)
        with torch.no_grad():
            # RLinf uses the current actor, target twin-Q, and no entropy backup.
            next_actions, _, _ = self.policy.sac_forward(batch["next_obs"])
            q_next = (
                self.target.sac_q_forward(batch["next_obs"], next_actions)
                .min(-1, keepdim=True)
                .values
            )
            target = td_target(
                batch["rewards"], batch["terminations"], q_next, self.cfg["gamma"]
            )
        q = self.policy.sac_q_forward(batch["curr_obs"], batch["actions"])
        critic = F.mse_loss(q, target.expand_as(q))
        pi, _, _ = self.policy.sac_forward(
            batch["curr_obs"],
            apply_reference_dropout=True,
            reference_dropout_prob=self.cfg["reference_dropout_prob"],
        )
        q1 = self.policy.sac_q_forward(batch["curr_obs"], pi)[..., :1]
        bc = bc_loss(
            pi,
            batch["actions"],
            batch["curr_obs"]["ref_chunk"],
            batch["intervene_flags"],
        )
        actor = -self.cfg["q_weight"] * q1.mean() + self.cfg["bc_weight"] * bc
        return critic, actor

    def update(self, batch):
        validate_batch(batch, self.policy)
        with torch.no_grad():
            action, _, _ = self.policy.sac_forward(batch["next_obs"])
            q = (
                self.target.sac_q_forward(batch["next_obs"], action)
                .min(-1, keepdim=True)
                .values
            )
            target = td_target(
                batch["rewards"], batch["terminations"], q, self.cfg["gamma"]
            )
        q = self.policy.sac_q_forward(batch["curr_obs"], batch["actions"])
        critic = F.mse_loss(q, target.expand_as(q))
        self.critic_optim.zero_grad(set_to_none=True)
        critic.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.q_head.parameters(),
            self.cfg["clip_grad"],
            error_if_nonfinite=True,
        )
        self.critic_optim.step()
        self.step += 1
        metrics = {"step": self.step, "critic_loss": float(critic.detach())}
        if self.step % self.cfg["critic_actor_ratio"] == 0:
            self.policy.q_head.requires_grad_(False)
            try:
                pi, _, _ = self.policy.sac_forward(
                    batch["curr_obs"],
                    apply_reference_dropout=True,
                    reference_dropout_prob=self.cfg["reference_dropout_prob"],
                )
                q1 = self.policy.sac_q_forward(batch["curr_obs"], pi)[..., :1]
                bc = bc_loss(
                    pi,
                    batch["actions"],
                    batch["curr_obs"]["ref_chunk"],
                    batch["intervene_flags"],
                )
                actor = -self.cfg["q_weight"] * q1.mean() + self.cfg["bc_weight"] * bc
                self.actor_optim.zero_grad(set_to_none=True)
                actor.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    self.cfg["clip_grad"],
                    error_if_nonfinite=True,
                )
                self.actor_optim.step()
                metrics.update(
                    actor_loss=float(actor.detach()),
                    bc_loss=float(bc.detach()),
                    q_pi=float(q1.mean().detach()),
                )
            finally:
                self.policy.q_head.requires_grad_(True)
        with torch.no_grad():
            for target, current in zip(
                self.target.parameters(), self.policy.parameters(), strict=True
            ):
                target.lerp_(current, self.cfg["tau"])
        return metrics

    def save(self, path, assets):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "xr1-rlt-stage2-v1",
            "config": self.config,
            "assets": assets,
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "step": self.step,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def restore(self, path, assets):
        saved = torch.load(
            path, map_location=next(self.policy.parameters()).device, weights_only=False
        )
        if saved["config"] != self.config or saved["assets"] != assets:
            raise ValueError("Stage 2 config/assets mismatch")
        for key in ("policy", "target", "actor_optim", "critic_optim"):
            getattr(self, key).load_state_dict(saved[key])
        self.step = saved["step"]
        torch.set_rng_state(saved["rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
