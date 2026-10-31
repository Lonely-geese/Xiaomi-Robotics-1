"""Strict offline feature contracts; demonstrations and RL replay stay distinct."""

from pathlib import Path

import torch
import yaml

from .core import BeeConfig


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text())
    base = yaml.safe_load(Path(settings["stage1_config"]).read_text())
    cfg = BeeConfig(**settings["bee"])
    if base["rlt"]["embed_dim"] != cfg.z_dim:
        raise ValueError("BEE feature dimension disagrees with Stage 1")
    return settings, base, cfg


def feature_assets(base, stage1, cfg):
    from mibot.rlt.data import sha256

    return {
        "stage1_sha256": sha256(stage1),
        "base_sha256": sha256(base["base_checkpoint"]),
        "normalization_sha256": sha256(base["normalization"]),
        "z_dim": cfg.z_dim,
        "proprio_dim": cfg.proprio_dim,
        "action_dim": cfg.action_dim,
        "chunk_len": cfg.chunk_len,
        "action_coordinates": "xr1-zscore-per-offset-v1",
        "features": "xr1-observation-only-rlt-v1",
    }


def tensor(value, shape, dtype, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != dtype
    ):
        raise ValueError(f"{name} must be a {dtype} tensor of shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def validate_observation(obs, n, cfg, name="obs"):
    for key, shape in {
        "z_rl": (n, cfg.z_dim),
        "proprio": (n, cfg.proprio_dim),
        "ref_chunk": (n, cfg.chunk_len, cfg.action_dim),
    }.items():
        if key not in obs:
            raise ValueError(f"Missing {name}.{key}")
        tensor(obs[key], shape, torch.float32, f"{name}.{key}")


def validate_data(data, cfg, *, corrections=False):
    required = (
        {"curr_obs", "actions", "correction_mask"}
        if corrections
        else {
            "curr_obs",
            "next_obs",
            "actions",
            "rewards",
            "terminations",
            "truncations",
            "intervene_flags",
        }
    )
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Missing transition fields: {sorted(required - set(data))}")
    if not isinstance(data["actions"], torch.Tensor) or data["actions"].ndim != 3:
        raise ValueError("Actions must be [N,K,7]")
    n, k, d = len(data["actions"]), cfg.chunk_len, cfg.action_dim
    if n < 1:
        raise ValueError("Empty dataset")
    tensor(data["actions"], (n, k, d), torch.float32, "actions")
    validate_observation(data["curr_obs"], n, cfg, "curr_obs")
    if corrections:
        tensor(data["correction_mask"], (n, k, d), torch.bool, "correction_mask")
        if not data["correction_mask"][..., list(cfg.editable_dims)].any():
            raise ValueError("No editable correction labels")
        return
    validate_observation(data["next_obs"], n, cfg, "next_obs")
    tensor(data["rewards"], (n, k), torch.float32, "rewards")
    for key in ("terminations", "truncations"):
        tensor(data[key], (n, k), torch.bool, key)
    tensor(data["intervene_flags"], (n, k, d), torch.bool, "intervene_flags")
    if (data["terminations"] & data["truncations"]).any():
        raise ValueError("Termination and truncation must be distinguished")
    if (data["terminations"][:, :-1] | data["truncations"][:, :-1]).any():
        raise ValueError(
            "Only complete K-step chunks supported; reject early-end padding/cross-reset chunks"
        )


def validate_payload(payload, cfg, assets, *, corrections=False):
    expected = "xr1-bee-corrections-v1" if corrections else "xr1-bee-replay-v1"
    if payload.get("format") != expected or payload.get("assets") != assets:
        raise ValueError(
            "BEE data format/assets mismatch (RLT bounded coordinates are not BEE z-scores)"
        )
    allowed = (
        ("demonstration_pairs", "human_corrections")
        if corrections
        else ("collected_transitions",)
    )
    if payload.get("source") not in allowed:
        raise ValueError("Missing/invalid explicit data source")
    validate_data(payload["transitions"], cfg, corrections=corrections)


def take(data, indices, device="cpu"):
    return {
        k: take(v, indices, device) if isinstance(v, dict) else v[indices].to(device)
        for k, v in data.items()
    }


class OfflineReplay:
    """BR includes all transitions; BH is its explicitly intervened subset."""

    def __init__(self, data, cfg):
        validate_data(data, cfg)
        self.data = data
        self.human = torch.where(data["intervene_flags"].flatten(1).any(-1))[0]
        self.editable_human = torch.where(
            data["intervene_flags"][..., list(cfg.editable_dims)].flatten(1).any(-1)
        )[0]
        if not len(self.editable_human):
            raise ValueError(
                "Offline BEE needs explicit human corrections on editable dimensions"
            )

    def sample(self, size, device="cpu", *, correction_only=False):
        if size < 2 or size % 2:
            raise ValueError(
                "BEE batch size must be positive and even for 1:1 sampling"
            )
        if correction_only:
            indices = self.editable_human[
                torch.randint(len(self.editable_human), (size,))
            ]
            data = take(self.data, indices, device)
            return {
                "curr_obs": data["curr_obs"],
                "actions": data["actions"],
                "correction_mask": data["intervene_flags"],
            }
        indices = torch.cat(
            [
                torch.randint(len(self.data["actions"]), (size // 2,)),
                self.human[torch.randint(len(self.human), (size // 2,))],
            ]
        )
        return take(self.data, indices, device)
