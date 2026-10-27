"""Stage 3: XR1 CFG flow matching with FP32 master AdamW, supports torchrun."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from mibot.steam.data import PolicyCollator, PolicyDataset, digest
from mibot.steam.distributed import broadcast_model, fp32_average_hook, mean_metrics, rank, world_size
from mibot.steam.optim import ShardedMasterAdamW, clip_grad_norm_fp32
from mibot.steam.policy import build_policy
from mibot.steam.policy_prefetch import PolicyPrefetcher
from mibot.steam.runtime import log, move, restore, save, setup, stopping, update
from torch.nn.parallel import DistributedDataParallel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--loader-processes", type=int, default=0)
    p.add_argument("--decode-workers", type=int, default=4)
    p.add_argument("--prefetch-batches", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--fused-optimizer", action="store_true")
    p.add_argument("--coalesced-broadcast", action="store_true")
    p.add_argument("--allow-batch-change", action="store_true",
                   help="Preserve weights/optimizer/RNG while allowing batch sizes and output to change")
    a = p.parse_args()
    if min(a.benchmark_steps, a.loader_processes) < 0 or min(a.decode_workers, a.prefetch_batches, a.cpu_threads) < 1:
        p.error("invalid worker, prefetch, or benchmark counts")
    torch.set_num_threads(a.cpu_threads)
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
    allowed = ("micro_batch_size", "global_batch_size", "output") if a.allow_batch_change else ()
    step = restore(a.resume, model, optimizer, cfg, rng, allow_config_changes=allowed) if a.resume else 0
    optimizer.configure_execution(fused=a.fused_optimizer, coalesced_broadcast=a.coalesced_broadcast)
    learner = model
    if a.ddp and world_size() > 1:
        learner = DistributedDataParallel(
            model, device_ids=[device.index], static_graph=True,
            gradient_as_bucket_view=True, bucket_cap_mb=64, broadcast_buffers=False,
        )
        learner.register_comm_hook(None, fp32_average_hook)
    prefetch = PolicyPrefetcher(
        cfg, len(dataset), rng, batch, processes=a.loader_processes,
        decode_workers=a.decode_workers, depth=a.prefetch_batches, cpu_threads=a.cpu_threads,
    ) if a.loader_processes else None
    end = (
        min(cfg["max_steps"], step + a.stop_after) if a.stop_after else cfg["max_steps"]
    )
    if a.benchmark_steps:
        end = step + a.benchmark_steps
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if rank() == 0 and not a.benchmark_steps:
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log(
        {
            "event": "ready",
            "samples": len(dataset),
            "world_size": world_size(),
            "global_batch": global_batch,
            "micro_batch_size": batch,
            "gradient_accumulation_steps": accumulation,
            "resume_step": step,
            "execution": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        }
    )
    while step < end:
        started = time.monotonic()
        total = 0.0
        conditional = 0.0
        timings = {"data_seconds": 0.0, "compute_seconds": 0.0, "update_seconds": 0.0}
        for micro in range(accumulation):
            stamp = time.monotonic()
            data = move(
                next(prefetch) if prefetch else collate(
                    [dataset[int(i)] for i in rng.integers(len(dataset), size=batch)]
                ),
                device,
                non_blocking=bool(prefetch),
            )
            conditional += float(data.pop("conditional_fraction")) / accumulation
            if a.profile:
                torch.cuda.synchronize()
                timings["data_seconds"] += time.monotonic() - stamp
                stamp = time.monotonic()
            context = learner.no_sync() if learner is not model and micro < accumulation - 1 else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = learner(**data)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite policy loss")
                (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
            if a.profile:
                torch.cuda.synchronize()
                timings["compute_seconds"] += time.monotonic() - stamp
        stamp = time.monotonic()
        lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg.get("warmup_steps", 500)))
        if learner is not model:
            optimizer.param_groups[0]["lr"] = lr
            norm = float(clip_grad_norm_fp32(params, cfg.get("clip_grad", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            norm = update(params, optimizer, lr, cfg.get("clip_grad", 1.0))
        if a.profile:
            torch.cuda.synchronize()
            timings["update_seconds"] += time.monotonic() - stamp
        step += 1
        metrics = mean_metrics(
            {"loss": total, "conditional_fraction": conditional, "grad_norm": norm},
            device,
        )
        seconds = time.monotonic() - started
        duration = torch.tensor(seconds, device=device)
        if world_size() > 1:
            torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX)
        log({"step": step, "seconds": float(duration), **metrics,
             **(mean_metrics(timings, device) if a.profile else {})})
        stop = stopping(device)
        if not a.benchmark_steps and (step % cfg["save_every"] == 0 or step == end or stop):
            save(output, model, optimizer, step, cfg, rng)
        if stop:
            break
    if prefetch:
        prefetch.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
