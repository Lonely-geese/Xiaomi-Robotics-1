"""Offline CFG inference on an XR1 dataset observation (does not control a robot)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mibot.steam.data import PolicyCollator, digest, read_manifest
from mibot.steam.policy import build_policy
from mibot.steam.runtime import move, resolve_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    saved = torch.load(
        resolve_checkpoint(a.checkpoint),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    cfg = saved["config"]
    manifest = read_manifest(a.manifest)
    if digest(manifest["normalization"]) != manifest["normalization_sha256"]:
        raise ValueError("normalization hash mismatch")
    if cfg["manifest_sha256"] != digest(a.manifest):
        raise ValueError("inference manifest differs from training")
    record = manifest["episodes"][a.episode]
    if not 0 <= a.frame < record["usable_frames"]:
        raise ValueError("frame outside usable interval")
    from mibot.data.datasets.json_dataset import JsonDataset

    dataset = JsonDataset(
        {
            "train_datasets": {
                "paths": [record["path"]],
                "normalization": manifest["normalization"],
                "action_length": 30,
                "active_action_dims": manifest["active_action_dims"],
                "repeat_to_max_steps": False,
                "augment": False,
            }
        }
    )
    dataset.samples = [{"file": record["path"], "frame_index": a.frame}]
    item = dataset[0]
    item["messages"] = item["messages"][:-2]
    collate = PolicyCollator(cfg["actor_vlm_path"])
    uncond = move(collate.observations([item], [False]), "cuda")
    positive = move(collate.observations([item], [True]), "cuda")
    mask = torch.zeros(1, 30, 60, device="cuda")
    mask[..., manifest["active_action_dims"]] = 1
    model = build_policy(cfg, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    del saved
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        action = model.generate(uncond, positive, mask, a.guidance_scale, a.num_steps)
    if not torch.isfinite(action).all():
        raise FloatingPointError("nonfinite generated action")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.output,
        normalized_action=action.float().cpu().numpy(),
        active_action_dims=manifest["active_action_dims"],
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "shape": list(action.shape),
                "guidance_scale": a.guidance_scale,
                "normalization": manifest["normalization"],
                "units": "normalized XR1 action chunks",
            }
        )
    )


if __name__ == "__main__":
    main()
