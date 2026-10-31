"""Export demo/VLA pairs for correction prewarm; never fabricate RL transitions."""

import argparse
import os
from pathlib import Path

import torch
from mibot.bee.data import load_settings, validate_payload
from mibot.bee.runtime import USBBEEPolicy
from mibot.rlt.data import ObservationCollator, USBDataset
from mibot.rlt.model import move


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_bee.yaml"),
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--episodes", type=int, nargs="+", help="Manifest episode IDs")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Bound memory/GPU work; use multiple output shards for larger exports",
    )
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if min(a.stride, a.max_samples) < 1:
        p.error("stride and max-samples must be positive")
    if a.output.exists():
        raise FileExistsError(a.output)
    settings, base, cfg = load_settings(a.config)
    torch.set_num_threads(8)
    torch.manual_seed(settings["seed"])
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    if base.get("image_cache"):
        os.environ["XR1_USB_IMAGE_CACHE"] = base["image_cache"]
    ds = USBDataset(
        base["dataset"], base["normalization"], stride=a.stride, episode_ids=a.episodes
    )
    collator = ObservationCollator(base["vlm_path"])
    policy = USBBEEPolicy(base, settings["stage1_checkpoint"], cfg, device=a.device)
    rows, sources = [], []
    for sample in ds.samples:
        record = ds.record_by_path[sample["file"]]
        frame = sample["frame_index"]
        if frame + cfg.chunk_len > record["length"]:
            continue
        item = ds.item_at(record["episode_index"], frame)
        obs = move(policy.extract(collator([item])), "cpu")
        rows.append(
            {
                "curr_obs": obs,
                "actions": item["action"][: cfg.chunk_len, :7].float()[None],
                "correction_mask": item["action_mask"][: cfg.chunk_len, :7].bool()[
                    None
                ],
            }
        )
        sources.append({"episode": record["episode_index"], "frame": frame})
        print(f"Exported {len(rows)}/{a.max_samples} demonstration pairs", flush=True)
        if len(rows) >= a.max_samples:
            break
    if not rows:
        raise ValueError("No complete demonstration chunks selected")
    data = {
        "curr_obs": {
            k: torch.cat([r["curr_obs"][k] for r in rows]) for k in rows[0]["curr_obs"]
        },
        "actions": torch.cat([r["actions"] for r in rows]),
        "correction_mask": torch.cat([r["correction_mask"] for r in rows]),
    }
    payload = {
        "format": "xr1-bee-corrections-v1",
        "assets": policy.assets,
        "source": "demonstration_pairs",
        "transitions": data,
        "samples": sources,
        "extraction_seed": settings["seed"],
        "note": "Offline demonstration/VLA pairs; not human takeovers or reward-bearing RL replay",
    }
    validate_payload(payload, cfg, policy.assets, corrections=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix(a.output.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(a.output)
    print(f"Saved {a.output}; correction prewarm only, no rewards inferred", flush=True)


if __name__ == "__main__":
    main()
