"""Stage 3: XR1 CFG flow matching with FP32 master AdamW, supports torchrun."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mibot.steam.data import PolicyCollator, PolicyDataset, digest
from mibot.steam.distributed import broadcast_model, mean_metrics, rank, world_size
from mibot.steam.optim import ShardedMasterAdamW
from mibot.steam.policy import build_policy
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
    cfg["advantages_sha256"] = digest(cfg["advantages"])
    if cfg.get("smoke_only") and cfg["max_steps"] > 2:
        raise ValueError("smoke configurations are limited to two steps")
    dataset = PolicyDataset(
        cfg["manifest"],
        cfg["advantages"],
        stride=cfg.get("sample_stride", 1),
        allow_smoke=cfg.get("smoke_only", False),
    )
    collate = PolicyCollator(cfg["actor_vlm_path"], cfg.get("unconditional_prob", 0.1))
    model = (
        build_policy(cfg, initialize=not bool(a.resume))
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    broadcast_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = ShardedMasterAdamW(
        params, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-5
    )
    rng = np.random.default_rng(cfg.get("seed", 0) + rank())
    torch.manual_seed(cfg.get("seed", 0) + rank())
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
            "samples": len(dataset),
            "world_size": world_size(),
            "global_batch": global_batch,
            "micro_batch_size": batch,
            "gradient_accumulation_steps": accumulation,
        }
    )
    while step < end:
        started = time.monotonic()
        total = 0.0
        conditional = 0.0
        for _ in range(accumulation):
            data = move(
                collate(
                    [dataset[int(i)] for i in rng.integers(len(dataset), size=batch)]
                ),
                device,
            )
            conditional += float(data.pop("conditional_fraction")) / accumulation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**data)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite policy loss")
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg.get("warmup_steps", 500)))
        norm = update(params, optimizer, lr, cfg.get("clip_grad", 1.0))
        step += 1
        metrics = mean_metrics(
            {"loss": total, "conditional_fraction": conditional, "grad_norm": norm},
            device,
        )
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
