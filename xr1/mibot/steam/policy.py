"""STEAM CFG flow matching with XR1's noise-to-action time convention."""

import math
import random

import torch
from torch import nn


class SteamPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # These SFT-only heads are never used by observation-only flow matching.
        for name in (
            "state_projector_choice",
            "action_projector_choice",
            "score_projector_choice",
        ):
            for p in getattr(model, name).parameters():
                p.requires_grad = False
        if hasattr(model.vlm, "lm_head"):
            for p in model.vlm.lm_head.parameters():
                p.requires_grad = False

    def forward(self, observation, action, action_mask, noise=None, time=None, prefix_length=None):
        if not action_mask.bool().any():
            raise ValueError("empty action mask")
        if prefix_length is None:
            use_prefix = self.training and self.model.async_train and random.random() < 0.5
            prefix_length = random.randint(1, 6) if use_prefix else 0
            if prefix_length and not action_mask[:, prefix_length:].any():
                prefix_length = 0
        if not 0 <= prefix_length <= min(6, action.shape[1] - 1):
            raise ValueError("unsupported prefix length")
        self.last_prefix_length = prefix_length
        condition = self.model.prepare_flow_condition(observation, action_mask, prefix_length)
        action = action.float()
        mask = action_mask.float()
        noise = torch.randn_like(action) if noise is None else noise.float()
        if time is None:
            # XR1 uses the reverse of OpenPI's flow direction: noise at 0, action at 1.
            time = (1.0 - self.model.beta.sample((len(action),))).to(
                action.device
            ) * 0.999
        x = ((1 - time[:, None, None]) * noise + time[:, None, None] * action) * mask
        x[:, :prefix_length] = action[:, :prefix_length] * mask[:, :prefix_length]
        velocity = self.model.dit_forward(
            x.to(self.model.sink.weight.dtype), time[:, None, None], **condition
        )
        error = (velocity.float() - (action - noise)).square() * mask
        # Average valid dimensions within each sample, then samples equally.
        counts = mask[:, prefix_length:].sum((-1, -2))
        sample_loss = error[:, prefix_length:].sum((-1, -2)) / counts.clamp_min(1)
        return sample_loss.sum() / (counts > 0).sum().clamp_min(1)

    @torch.no_grad()
    def generate(
        self,
        unconditional_observation,
        positive_observation,
        action_mask,
        guidance_scale=1.0,
        num_steps=5,
        noise=None,
        prefix=None,
    ):
        if self.training:
            raise ValueError("call eval() before CFG inference")
        if num_steps < 1 or not math.isfinite(guidance_scale):
            raise ValueError("invalid CFG inference settings")
        mask = action_mask.to(self.model.sink.weight.dtype)
        x = torch.randn_like(mask) if noise is None else noise.to(mask)
        x = x * mask
        length = 0
        if prefix is not None:
            if prefix.ndim != 3 or prefix.shape[0] != x.shape[0] or prefix.shape[2] != x.shape[2]:
                raise ValueError("expected normalized prefix [batch, steps, action_dim]")
            length = prefix.shape[1]
            if length > min(6, x.shape[1] - 1) or not torch.isfinite(prefix).all():
                raise ValueError("invalid action prefix")
            if length and not self.model.async_train:
                raise ValueError("checkpoint configured without prefix training")
            x[:, :length] = prefix.to(x) * mask[:, :length]
        # Both branches retain the same image/state/task; only Advantage: positive differs.
        unconditional = (
            self.model.prepare_flow_condition(unconditional_observation, mask, length)
            if guidance_scale != 1
            else None
        )
        positive = (
            self.model.prepare_flow_condition(positive_observation, mask, length)
            if guidance_scale != 0
            else None
        )
        for i in range(num_steps):
            time = torch.full((len(x),), i / num_steps, device=x.device, dtype=x.dtype)
            v0 = (
                self.model.dit_forward(x, time[:, None, None], **unconditional)
                if unconditional is not None
                else 0
            )
            v1 = (
                self.model.dit_forward(x, time[:, None, None], **positive)
                if positive is not None
                else 0
            )
            x = (
                x + ((1 - guidance_scale) * v0 + guidance_scale * v1) / num_steps
            ) * mask
        return x


def build_policy(config, *, initialize=True):
    import os

    os.environ["XR1_VLM_PATH"] = config["actor_vlm_path"]
    from mibot.models.VLA.XR1 import xr1

    model = xr1(
        full_finetune=True,
        async_train=config.get("async_train", False),
        active_action_dims=config.get("active_action_dims", list(range(7))),
    )
    if initialize:
        checkpoint = torch.load(
            config["sft_checkpoint"], map_location="cpu", weights_only=True, mmap=True
        )
        weights = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
        if all(k.startswith("model.") for k in weights):
            weights = {k[6:]: v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
    return SteamPolicy(model)
