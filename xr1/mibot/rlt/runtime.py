"""Frozen Stage 1 + Stage 2 inference, with explicit physical action recovery."""

import json
from pathlib import Path

import numpy as np
import torch

from mibot.utils.io import aa2rotm, rotm2aa_batch

from .data import quaternion_matrix, sha256
from .learner import build_policy
from .model import ActionCoordinates, build_model, move


def recover_left_targets(normalized_action, raw_state, normalization):
    """Recover absolute left targets in the same base frame as the dataset poses.

    Every chunk offset is relative to raw_state at the chunk's START, never
    integrated as a succession of per-step deltas. Right arm is not commanded.
    """
    z = np.asarray(normalized_action, dtype=np.float32)
    raw = np.asarray(raw_state, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 7 or not 1 <= len(z) <= 30 or raw.shape != (30,):
        raise ValueError("Expected [K,7] normalized actions and one raw [30] USB state")
    if not np.isfinite(z).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite action/state")
    mean = np.asarray(normalization["mean"], dtype=np.float32)[: len(z), :7]
    std = np.asarray(normalization["std"], dtype=np.float32)[: len(z), :7]
    delta = z * (std + 1e-6) + mean
    rotation = quaternion_matrix(raw[10:14])
    return {
        "left_ee_pos": raw[7:10] + delta[:, :3] @ rotation.T,
        "left_ee_rotm": np.stack([rotation @ aa2rotm(x) for x in delta[:, 3:6]]),
        "left_gripper_pos": raw[14:15] + delta[:, 6:7],
    }


def encode_executed_left(
    targets, raw_start_state, normalization, scale=5.0, *, action_coordinates="bounded"
):
    """Convert ACTUALLY EXECUTED absolute targets to replay coordinates."""
    raw = np.asarray(raw_start_state, dtype=np.float32)
    pos = np.asarray(targets["left_ee_pos"], dtype=np.float32)
    rotation = np.asarray(targets["left_ee_rotm"], dtype=np.float32)
    gripper = np.asarray(targets["left_gripper_pos"], dtype=np.float32)
    k = len(pos)
    if (
        raw.shape != (30,)
        or pos.shape != (k, 3)
        or rotation.shape != (k, 3, 3)
        or gripper.shape != (k, 1)
        or not 1 <= k <= 30
    ):
        raise ValueError(
            "Executed targets require [K,3], [K,3,3], [K,1] and raw start state [30]"
        )
    r0 = quaternion_matrix(raw[10:14])
    physical = np.concatenate(
        ((pos - raw[7:10]) @ r0, rotm2aa_batch(r0.T @ rotation), gripper - raw[14:15]),
        -1,
    )
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:k, :7]
    std = np.asarray(normalization["std"], dtype=np.float32)[:k, :7]
    z = (physical - mean) / (std + 1e-6)
    z = torch.from_numpy(z.astype(np.float32))
    if action_coordinates == "zscore":
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite executed action")
        return z
    if action_coordinates != "bounded":
        raise ValueError("Unknown action coordinates")
    return ActionCoordinates(scale).encode(z)


def feature_assets(config, stage1_checkpoint):
    return {
        "stage1_sha256": sha256(stage1_checkpoint),
        "normalization_sha256": sha256(config["normalization"]),
        "action_scale": config["action_scale"],
        "chunk_len": config["stage2"]["chunk_len"],
        "z_dim": config["rlt"]["embed_dim"],
        "proprio_dim": 16,
        "action_dim": 7,
        "ref_num_action_chunks": 30,
    }


class USBRLTPolicy:
    """Offline callable; no socket, robot connection, or actuator side effects."""

    def __init__(
        self, config, stage1_checkpoint, stage2_checkpoint=None, device="cuda:0"
    ):
        self.config, self.device = config, device
        self.features = (
            build_model(config, device, stage1_checkpoint).eval().requires_grad_(False)
        )
        self.features.base.to(dtype=torch.bfloat16)
        self.norm = json.loads(Path(config["normalization"]).read_text())
        self.policy = None
        if stage2_checkpoint:
            saved = torch.load(
                stage2_checkpoint, map_location="cpu", weights_only=False
            )
            if saved["format"] != "xr1-rlt-stage2-v1" or saved[
                "assets"
            ] != feature_assets(config, stage1_checkpoint):
                raise ValueError(
                    "Stage 2 feature/action/normalization provenance mismatch"
                )
            if saved["config"]["stage2"] != config["stage2"]:
                raise ValueError("Stage 2 policy configuration mismatch")
            self.policy = build_policy(config).to(device).eval().requires_grad_(False)
            self.policy.load_state_dict(saved["policy"], strict=True)

    @torch.inference_mode()
    def predict(self, observation, raw_state, *, use_actor=False):
        if use_actor and self.policy is None:
            raise ValueError(
                "An explicitly trained Stage 2 checkpoint is required to use the actor"
            )
        if observation["state"].shape[0] != 1:
            raise ValueError(
                "Physical target recovery currently accepts one robot observation"
            )
        obs, reference = self.features.extract(move(observation, self.device))
        k = self.config["stage2"]["chunk_len"]
        if use_actor:
            bounded, _ = self.policy.predict_action_batch(obs, mode="eval")
            z = self.features.coordinates.decode(bounded)[0]
        else:
            z = reference[0, :k, :7]
        result = recover_left_targets(z.cpu().numpy(), raw_state, self.norm)
        return {
            "targets": result,
            "normalized_action": z.cpu(),
            "rlt_observation": {k: v.cpu() for k, v in obs.items()},
            "action_source": "actor" if use_actor else "vla",
        }
