"""Offline USB BEE inference: save physical targets, send no robot commands."""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from mibot.bee.data import load_settings
from mibot.bee.runtime import USBBEEPolicy
from mibot.rlt.data import ObservationCollator, USBDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_bee.yaml"),
    )
    p.add_argument(
        "--bee", help="Trained offline-RL BEE checkpoint; omission uses frozen VLA only"
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    settings, base, cfg = load_settings(a.config)
    torch.set_num_threads(8)
    torch.manual_seed(settings["seed"])
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    if base.get("image_cache"):
        os.environ["XR1_USB_IMAGE_CACHE"] = base["image_cache"]
    ds = USBDataset(base["dataset"], base["normalization"])
    item = ds.item_at(a.episode, a.frame)
    observation = ObservationCollator(base["vlm_path"])([item])
    record = next(r for r in ds.records if r["episode_index"] == a.episode)
    raw = pq.read_table(ds.root / record["data_path"], columns=["observation.state"])[
        "observation.state"
    ][a.frame].as_py()
    policy = USBBEEPolicy(base, settings["stage1_checkpoint"], cfg, a.bee, a.device)
    result = policy.predict(
        observation, np.asarray(raw, dtype=np.float32), use_actor=bool(a.bee)
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, a.output)
    print(
        f"Saved {result['action_source']} physical targets to {a.output}; no robot command sent"
    )


if __name__ == "__main__":
    main()
