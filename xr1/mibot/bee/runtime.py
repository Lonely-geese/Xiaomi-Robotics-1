"""Frozen USB backbone + BEE residual inference; no robot transport."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from mibot.rlt.model import build_model, move
from mibot.rlt.runtime import recover_left_targets

from .core import BeePolicy
from .data import feature_assets, validate_observation


class USBBEEPolicy:
    def __init__(self, base, stage1, cfg, checkpoint=None, device="cuda:0"):
        self.cfg, self.device = cfg, device
        self.assets = feature_assets(base, stage1, cfg)
        self.actor = None
        if checkpoint:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                saved["format"] != "xr1-bee-v1"
                or saved["assets"] != self.assets
                or saved["config"] != asdict(cfg)
            ):
                raise ValueError("BEE checkpoint assets/config mismatch")
            if (
                saved["step"] < cfg.policy_delay
                or saved["training"].get("mode") != "offline-rl"
            ):
                raise ValueError(
                    "BEE actor has not been trained; correction-only checkpoints cannot drive inference"
                )
            self.actor = BeePolicy(cfg).to(device).eval().requires_grad_(False)
            self.actor.load_state_dict(saved["model"], strict=True)
        self.features = build_model(base, device, stage1).eval().requires_grad_(False)
        self.features.base.to(dtype=torch.bfloat16)
        self.norm = json.loads(Path(base["normalization"]).read_text())

    @torch.inference_mode()
    def extract(self, observation, *, noise=None):
        obs, _ = self.features.extract(
            move(observation, self.device),
            chunk_len=self.cfg.chunk_len,
            noise=noise,
            action_coordinates="zscore",
        )
        obs["ref_chunk"] = obs["ref_chunk"][:, : self.cfg.chunk_len].contiguous()
        validate_observation(obs, len(obs["z_rl"]), self.cfg)
        return obs

    @torch.inference_mode()
    def predict(self, observation, raw_state, *, use_actor=False):
        if observation["state"].shape[0] != 1:
            raise ValueError("USB target recovery expects one observation")
        if use_actor and self.actor is None:
            raise ValueError("A trained BEE checkpoint is required")
        obs = self.extract(observation)
        action = self.actor.action(obs) if use_actor else obs["ref_chunk"]
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite BEE action")
        return {
            "targets": recover_left_targets(
                action[0].cpu().numpy(), raw_state, self.norm
            ),
            "normalized_action": action[0].cpu(),
            "reference_action": obs["ref_chunk"][0].cpu(),
            "bee_observation": move(obs, "cpu"),
            "assets": self.assets,
            "action_source": "bee" if use_actor else "vla",
        }
