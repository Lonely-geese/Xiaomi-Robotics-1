"""Train RLT heads on explicitly collected feature transitions, not bare demos."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from mibot.rlt.learner import RLTLearner, validate_batch
from mibot.rlt.model import move
from mibot.rlt.runtime import feature_assets


def take(batch, indices):
    return {
        k: take(v, indices) if isinstance(v, dict) else v[indices]
        for k, v in batch.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_rlt.yaml"),
    )
    p.add_argument("--stage1", required=True)
    p.add_argument(
        "--replay",
        required=True,
        help="torch file with format, assets and transitions; see docs/usb_rlt.rst",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume")
    a = p.parse_args()
    if a.updates < 1 or a.batch_size < 1:
        raise ValueError("updates and batch-size must be positive")
    cfg = yaml.safe_load(Path(a.config).read_text())
    assets = feature_assets(cfg, a.stage1)
    replay = torch.load(a.replay, map_location="cpu", weights_only=True)
    if replay.get("format") != "xr1-rlt-replay-v1" or replay.get("assets") != assets:
        raise ValueError("Replay feature model, norm or action coordinates mismatch")
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])
    learner = RLTLearner(cfg, a.device)
    batch = replay["transitions"]
    validate_batch(batch, learner.policy)
    count = len(batch["actions"])
    if count < 2:
        raise ValueError("Replay needs at least 2 real transitions")
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise ValueError(
            "Output directory is nonempty; choose a new run or explicitly resume"
        )
    a.output.mkdir(parents=True, exist_ok=True)
    if a.resume:
        learner.restore(a.resume, assets)
    for _ in range(a.updates):
        indices = torch.randint(count, (a.batch_size,))
        metrics = learner.update(move(take(batch, indices), a.device))
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if learner.step % 100 == 0:
            learner.save(a.output / "latest.pt", assets)
    learner.save(a.output / "latest.pt", assets)


if __name__ == "__main__":
    main()
