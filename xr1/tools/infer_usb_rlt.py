"""Offline USB inference. Saves targets/features and never connects to a robot."""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from mibot.rlt.data import ObservationCollator, USBDataset
from mibot.rlt.runtime import USBRLTPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_rlt.yaml"),
    )
    p.add_argument("--stage1", required=True)
    p.add_argument("--stage2")
    p.add_argument("--actor", action="store_true")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    ds = USBDataset(cfg["dataset"], cfg["normalization"])
    item = ds.item_at(a.episode, a.frame)
    obs = ObservationCollator(cfg["vlm_path"])([item])
    record = next(r for r in ds.records if r["episode_index"] == a.episode)
    states = pq.read_table(
        ds.root / record["data_path"], columns=["observation.state"]
    )["observation.state"]
    raw_state = np.asarray(states[a.frame].as_py(), dtype=np.float32)
    policy = USBRLTPolicy(cfg, a.stage1, a.stage2, a.device)
    result = policy.predict(obs, raw_state, use_actor=a.actor)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.save(result, a.output)
    print(
        f"Saved {result['action_source']} targets and features to {a.output}; no robot command sent"
    )


if __name__ == "__main__":
    main()
