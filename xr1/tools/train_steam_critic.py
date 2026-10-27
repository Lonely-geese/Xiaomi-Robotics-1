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
from mibot.steam.optim import clip_grad_norm_fp32
from mibot.steam.prefetch import DeferredPairDataset, PairPrefetcher
from mibot.steam.runtime import log, move, restore, save, setup, stopping, update
from mibot.steam.upstream.data.pair_dataset import PairDataset
from torch.nn.parallel import DistributedDataParallel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run real updates from the resume state without writing checkpoints",
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument("--decode-workers", type=int, default=0)
    p.add_argument("--prefetch-batches", type=int, default=2)
    p.add_argument("--loader-processes", type=int, default=0)
    p.add_argument("--fused-optimizer", action="store_true")
    p.add_argument("--ddp", action="store_true")
    p.add_argument(
        "--benchmark-static-data",
        action="store_true",
        help="Diagnostic compute ceiling only; never use as throughput acceptance",
    )
    a = p.parse_args()
    if (
        min(a.benchmark_steps, a.decode_workers, a.loader_processes) < 0
        or a.prefetch_batches < 1
    ):
        p.error(
            "worker/benchmark counts must be nonnegative and prefetch-batches positive"
        )
    if a.benchmark_static_data and (not a.benchmark_steps or not a.decode_workers):
        p.error("static-data diagnosis requires benchmark-steps and decode-workers")
    if a.loader_processes and not a.decode_workers:
        p.error("loader-processes requires decode-workers")
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
        fused=a.fused_optimizer,
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
            groups.append(
                pair_dataset(
                    records,
                    cfg,
                    dataset_class=(
                        DeferredPairDataset if a.decode_workers else PairDataset
                    ),
                )
            )
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
    # Optimizer loading restores the checkpoint's execution flags as well.
    for group in optimizer.param_groups:
        group["fused"] = a.fused_optimizer
    if a.fused_optimizer:
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].to(device)
    members = (
        [
            DistributedDataParallel(
                member,
                device_ids=[device.index],
                static_graph=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=64,
                broadcast_buffers=False,
            )
            for member in model.members
        ]
        if a.ddp and world_size() > 1
        else None
    )
    prefetch = (
        PairPrefetcher(
            groups,
            probabilities,
            rng,
            batch,
            collate,
            workers=a.decode_workers,
            depth=a.prefetch_batches,
            processes=a.loader_processes,
            language_path=cfg["language_path"],
            num_bins=cfg["num_bins"],
        )
        if a.decode_workers
        else None
    )
    static_data = None
    if a.benchmark_static_data:
        static_data = [move(next(prefetch), device) for _ in model.members]
        prefetch.close()
        prefetch = None
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
            "members": len(model.members),
            "world_size": world_size(),
            "global_batch_per_member": global_batch,
            "accumulation": accumulation,
            "execution": {k: v for k, v in vars(a).items() if k != "config"},
        }
    )
    while step < end:
        started = time.monotonic()
        losses = []
        timings = {"data_seconds": 0.0, "compute_seconds": 0.0, "update_seconds": 0.0}
        for member in range(len(model.members)):
            total = 0.0
            for _ in range(accumulation):
                stamp = time.monotonic()
                if static_data is not None:
                    data = static_data[member]
                elif prefetch is not None:
                    data = move(next(prefetch), device)
                else:
                    items = []
                    for _ in range(batch):
                        ds = groups[int(rng.choice(len(groups), p=probabilities))]
                        items.append(ds[int(rng.integers(len(ds)))])
                    data = move(collate(items), device)
                if a.profile:
                    torch.cuda.synchronize()
                    timings["data_seconds"] += time.monotonic() - stamp
                    stamp = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = (
                        members[member](data["observation"], data["labels"]).loss
                        if members is not None
                        else model(
                            data["observation"], data["labels"], member_idx=member
                        ).loss
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite critic loss")
                (loss / accumulation).backward()
                total += float(loss.detach()) / accumulation
                if a.profile:
                    torch.cuda.synchronize()
                    timings["compute_seconds"] += time.monotonic() - stamp
            losses.append(total)
            # Update and clear one member before the next: no sibling weight decay/moments.
            parameters = list(model.members[member].parameters())
            lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg.get("warmup_steps", 500)))
            stamp = time.monotonic()
            if members is not None:
                optimizer.param_groups[0]["lr"] = lr
                clip_grad_norm_fp32(parameters, 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                update(parameters, optimizer, lr, 10.0)
            if a.profile:
                torch.cuda.synchronize()
                timings["update_seconds"] += time.monotonic() - stamp
        step += 1
        metrics = mean_metrics({f"loss_{i}": v for i, v in enumerate(losses)}, device)
        log(
            {
                "step": step,
                "seconds": time.monotonic() - started,
                **metrics,
                **(mean_metrics(timings, device) if a.profile else {}),
            }
        )
        stop = stopping(device)
        if not a.benchmark_steps and (
            step % cfg["save_every"] == 0 or step == end or stop
        ):
            save(output, model, optimizer, step, cfg, rng)
        if stop:
            break
    if prefetch is not None:
        prefetch.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
