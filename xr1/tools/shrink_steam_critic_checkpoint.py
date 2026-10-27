"""Explicitly shrink an immutable critic checkpoint to fewer data-parallel ranks.

Critic optimizer states are replicated, unlike the policy's sharded optimizer.
Keep each retained rank's own RNG, and never use this tool for policy states.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    saved = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    old = saved["config"]
    if "ensemble_size" not in old or not 1 <= args.world_size < saved["world_size"]:
        raise ValueError("Only shrinking a replicated critic checkpoint is supported")
    cfg = json.loads(args.config.read_text())
    cfg["manifest_sha256"] = hashlib.sha256(
        Path(cfg["manifest"]).read_bytes()
    ).hexdigest()
    changed = {k for k in old.keys() | cfg.keys() if old.get(k) != cfg.get(k)}
    if changed != {"output", "global_batch_size"}:
        raise ValueError(f"Unexpected config changes: {changed}")
    if cfg["global_batch_size"] != cfg["micro_batch_size"] * args.world_size:
        raise ValueError("Migration requires no gradient accumulation")
    for rank in range(args.world_size):
        if not (source.parent / f"rank_{rank}.pt").is_file():
            raise ValueError(f"Missing rank {rank} state")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    target = output / f"step_{saved['step']:08d}_{time.time_ns()}"
    target.mkdir()
    # These files are immutable checkpoint generations. Hard links preserve the
    # complete optimizer/RNG bytes without duplicating >100 GB of state.
    for rank in range(args.world_size):
        os.link(source.parent / f"rank_{rank}.pt", target / f"rank_{rank}.pt")
    shutil.copyfile(source.parent / "normalize.json", target / "normalize.json")
    saved["config"], saved["world_size"] = cfg, args.world_size
    temp = target / "model.pt.tmp"
    torch.save(saved, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(target / "model.pt")
    (target / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    # Verify all model tensors after serialization before publishing the pointer.
    check = torch.load(
        target / "model.pt", map_location="cpu", weights_only=True, mmap=True
    )
    assert check["step"] == saved["step"] and check["config"] == cfg
    assert check["world_size"] == args.world_size
    assert all(torch.equal(v, check["model"][k]) for k, v in saved["model"].items())
    record = {
        "source_checkpoint": str(source),
        "step": saved["step"],
        "world_size_before": old["global_batch_size"] // old["micro_batch_size"],
        "world_size_after": args.world_size,
        "global_batch_before": old["global_batch_size"],
        "global_batch_after": cfg["global_batch_size"],
        "retained_source_ranks": list(range(args.world_size)),
        "model_tensors_equal": True,
        "optimizer_rng_files": "identical immutable hard links",
        "trajectory_note": "Batch and participating ranks changed; not bitwise continuation",
    }
    (output / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
    pointer = output / "latest.json.tmp"
    pointer.write_text(
        json.dumps({"step": saved["step"], "checkpoint": str(target / "model.pt")})
        + "\n"
    )
    pointer.replace(output / "latest.json")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
