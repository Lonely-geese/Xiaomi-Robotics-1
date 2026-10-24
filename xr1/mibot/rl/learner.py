"""Separate DIVL and QAM optimizer paths, fixed SFT reference, EMA and resume."""

import copy
import random
from pathlib import Path
from dataclasses import asdict, replace

import torch
from torch import nn

from .config import LWDConfig
from .divl import update_ema
from .distributed import gather_rng, rank, restore_rng, synchronize_gradients, world_size
from .optim import ShardedMasterAdamW, clip_grad_norm_fp32
from .qam import build_adjoint_targets, local_loss
from .replay import validate_batch


class LWDLearner(nn.Module):
    def __init__(self, actor, critic, config=None, reference=None):
        super().__init__()
        self.config = config or LWDConfig()
        self.actor = actor
        # Always clone BEFORE any policy improvement. Resume also restores this
        # original SFT snapshot; never recreate it from the improved actor.
        self.reference = copy.deepcopy(actor) if reference is None else reference
        if self.reference is actor:
            raise ValueError("reference must be a distinct, fixed SFT model")
        self.reference.requires_grad_(False).eval()
        self.critic = critic
        self.target = copy.deepcopy(critic).requires_grad_(False).eval()
        self.updates = 0
        self._configure_optimizers()

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        self.target.eval()
        return self

    def _configure_optimizers(self):
        self.actor.set_stage(self.config.stage)
        optimizer = ShardedMasterAdamW if self.config.actor_optimizer == "fp32_master_sharded" else torch.optim.AdamW
        self.actor_optimizer = optimizer(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=self.config.actor_lr, weight_decay=self.config.actor_weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config.total_steps)
                           for opt in (self.actor_optimizer, self.critic_optimizer)]

    def start_online(self, total_steps=None):
        """Keep offline weights/EMA/SFT reference; start the online LR schedule."""
        self.config = replace(self.config, stage="online",
                              total_steps=total_steps or self.config.total_steps)
        self.updates = 0
        self._configure_optimizers()

    def update(self, batch, generator=None):
        validate_batch(batch)
        self.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, metrics = self.critic.loss(self.target, batch, self.config)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite DIVL loss")
        critic_loss.backward()
        synchronize_gradients(self.critic.parameters())
        critic_norm = clip_grad_norm_fp32(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        update_ema(self.target, self.critic, self.config.ema_rate)

        mask = batch["action_mask"].float()
        # Evaluating the critic still permits gradients w.r.t. generated actions.
        self.critic.eval()
        with torch.no_grad():
            features = self.critic.encode(batch["critic_observation"]).detach()
            reference_condition = self.reference.prepare(batch["actor_observation"], mask)
        reference_velocity = lambda x, t: self.reference.velocity(reference_condition, x, t)
        targets = build_adjoint_targets(
            reference_velocity, lambda a: self.critic.q(features, a, mask).amin(-1), mask,
            steps=self.config.flow_steps, temperature=self.config.temperature, generator=generator)
        self.actor_optimizer.zero_grad(set_to_none=True)
        condition = self.actor.prepare(batch["actor_observation"], mask)
        actor_loss, diagnostics = local_loss(lambda x, t: self.actor.velocity(condition, x, t),
                                            targets, mask, diagnostics=True)
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("non-finite QAM loss")
        actor_loss.backward()
        synchronize_gradients(self.actor.parameters())
        actor_norm = clip_grad_norm_fp32(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        metrics.update(diagnostics)
        metrics.update(actor_lr=self.actor_optimizer.param_groups[0]["lr"],
                       critic_lr=self.critic_optimizer.param_groups[0]["lr"],
                       actor_clip_scale=(self.config.grad_clip / (actor_norm.float() + 1e-6)).clamp(max=1),
                       critic_clip_scale=(self.config.grad_clip / (critic_norm.float() + 1e-6)).clamp(max=1))
        for scheduler in self.schedulers:
            scheduler.step()
        self.updates += 1
        metrics.update(actor_loss=actor_loss.detach(), actor_grad_norm=actor_norm.detach(),
                       critic_grad_norm=critic_norm.detach())
        return {key: float(value) for key, value in metrics.items()}

    def checkpoint(self):
        result = {"format": "xr1-lwd-v1", "config": asdict(self.config), "updates": self.updates,
                "model": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "schedulers": [s.state_dict() for s in self.schedulers],
                "rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() and world_size() == 1 else []}
        if world_size() > 1:
            result["distributed"] = gather_rng()
        return result

    def restore(self, checkpoint, *, actor_optimizer=None, actor_lr=None):
        if checkpoint.get("format") != "xr1-lwd-v1":
            raise ValueError("not an LWD checkpoint (must include the original SFT reference)")
        if checkpoint.get("distributed", {}).get("world_size", 1) != world_size():
            raise ValueError("exact LWD resume requires the same distributed world size")
        self.config = LWDConfig(**checkpoint["config"])
        old_optimizer = self.config.actor_optimizer
        if actor_optimizer is not None:
            if old_optimizer != "adamw" or actor_optimizer != "fp32_master_sharded":
                raise ValueError("only legacy AdamW to FP32 master migration is supported")
            self.config = replace(self.config, actor_optimizer=actor_optimizer)
        self.load_state_dict(checkpoint["model"], strict=True)
        # Masters must be initialized from the restored actor, not fresh SFT.
        self._configure_optimizers()
        optimizer_state = checkpoint["actor_optimizer"]
        if optimizer_state.get("format") == "external-master-shards-v1":
            shard = Path(optimizer_state["directory"]) / f"rank_{rank()}.pt"
            optimizer_state = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
        if actor_optimizer is not None:
            self.actor_optimizer.load_legacy_adamw(optimizer_state)
        else:
            self.actor_optimizer.load_state_dict(optimizer_state)
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        for scheduler, state in zip(self.schedulers, checkpoint["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        if actor_lr is not None:
            if actor_lr <= 0:
                raise ValueError("actor learning rate must be positive")
            ratio = actor_lr / self.config.actor_lr
            group = self.actor_optimizer.param_groups[0]
            group["lr"] *= ratio
            group["initial_lr"] *= ratio
            scheduler = self.schedulers[0]
            scheduler.base_lrs = [lr * ratio for lr in scheduler.base_lrs]
            scheduler._last_lr = [lr * ratio for lr in scheduler._last_lr]
            self.config = replace(self.config, actor_lr=actor_lr)
        self.updates = checkpoint["updates"]
        if not restore_rng(checkpoint):
            torch.set_rng_state(checkpoint["rng"].cpu())
            random.setstate(checkpoint["python_rng"])
            if torch.cuda.is_available() and checkpoint["cuda_rng"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        self.reference.requires_grad_(False).eval()
        self.target.requires_grad_(False).eval()
