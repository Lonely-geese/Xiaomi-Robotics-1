"""Stage 1: independent temporal-pair ensemble training, faithful to RLinf STEAM."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mibot.steam.critic import build_critic
from mibot.steam.data import digest, pair_dataset, read_manifest
from mibot.steam.distributed import broadcast_model, mean_metrics, rank, world_size
from mibot.steam.runtime import log, move, restore, save, setup, stopping, update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    device = setup(cfg.get("seed", 0))
    if a.stop_after is not None and a.stop_after < 1:
        raise ValueError("stop-after must be positive")
    cfg["manifest_sha256"] = digest(cfg["manifest"])
    m = read_manifest(cfg["manifest"])
    model, collate = build_critic(cfg)
    model = model.to(device).train()
    broadcast_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-5,
        foreach=False,
    )
    rng = np.random.default_rng(cfg.get("seed", 0) + rank())
    torch.manual_seed(cfg.get("seed", 0) + rank())
    # RLinf balance_weights=true multiplies raw source weights by dataset lengths.
    groups = []
    for kind in ("sft", "rollout"):
        records = [
            r for r in m["episodes"] if r["split"] == "train" and r["type"] == kind
        ]
        if records:
            groups.append(pair_dataset(records, cfg))
    probabilities = np.array([len(ds) for ds in groups], dtype=float)
    probabilities /= probabilities.sum()
    for ds in groups:
        ds._rng = rng
    batch = cfg["micro_batch_size"]
    global_batch = cfg["global_batch_size"]
    if global_batch % (batch * world_size()):
        raise ValueError("global batch must divide micro batch * world size")
    accumulation = global_batch // (batch * world_size())
    step = restore(a.resume, model, optimizer, cfg, rng) if a.resume else 0
    end = (
        min(cfg["max_steps"], step + a.stop_after) if a.stop_after else cfg["max_steps"]
    )
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if rank() == 0:
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log(
        {
            "event": "ready",
            "members": len(model.members),
            "world_size": world_size(),
            "global_batch_per_member": global_batch,
        }
    )
    while step < end:
        started = time.monotonic()
        losses = []
        for member in range(len(model.members)):
            total = 0.0
            for _ in range(accumulation):
                items = []
                for _ in range(batch):
                    ds = groups[int(rng.choice(len(groups), p=probabilities))]
                    items.append(ds[int(rng.integers(len(ds)))])
                data = move(collate(items), device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        data["observation"], data["labels"], member_idx=member
                    ).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite critic loss")
                (loss / accumulation).backward()
                total += float(loss.detach()) / accumulation
            losses.append(total)
            # Update and clear one member before the next: no sibling weight decay/moments.
            parameters = list(model.members[member].parameters())
            lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg.get("warmup_steps", 500)))
            update(parameters, optimizer, lr, 10.0)
        step += 1
        metrics = mean_metrics({f"loss_{i}": v for i, v in enumerate(losses)}, device)
        log({"step": step, "seconds": time.monotonic() - started, **metrics})
        stop = stopping(device)
        if step % cfg["save_every"] == 0 or step == end or stop:
            save(output, model, optimizer, step, cfg, rng)
        if stop:
            break
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
