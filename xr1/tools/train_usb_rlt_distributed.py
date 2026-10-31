"""Eight-card USB RLToken training with per-rank batches and sharded Adam state."""

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from mibot.rlt.data import USBDataset
from mibot.rlt.fast import (
    DenseUSBCollator,
    configure_fast,
    configure_fast_zero_checkpoint,
)
from mibot.rlt.model import build_model, move, save_stage1
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = cfg["stage1"]
    speed = cfg["speed"]
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if (
        world != 8
        or s["micro_batch_size"] != 32
        or s["accumulate"] != 1
        or cfg["global_batch_size"] != 256
    ):
        raise ValueError(
            "This run requires 8 ranks x 32 actual samples, one optimizer update per batch"
        )
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=device)
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    os.environ.update(
        XR1_VLM_PATH=cfg["vlm_path"],
        XR1_STRICT_BATCH="1",
        MAX_LENGTH="65536",
        XR1_USB_IMAGE_CACHE=cfg["image_cache"],
        TOKENIZERS_PARALLELISM="false",
    )
    if rank == 0:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Choose a fresh output directory")
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dist.barrier()
    dataset = USBDataset(
        cfg["dataset"], cfg["normalization"], augment=s["augment"], stride=s["stride"]
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=cfg["seed"],
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        drop_last=True,
        num_workers=s["workers"],
        collate_fn=DenseUSBCollator(),
        pin_memory=True,
        persistent_workers=s["workers"] > 0,
        multiprocessing_context="spawn" if s["workers"] else None,
        prefetch_factor=2 if s["workers"] else None,
    )
    model = build_model(cfg, device, a.resume).train()
    configure_fast(
        model,
        rlt_amp=speed["rlt_bf16"],
        checkpoint_rlt=speed["checkpoint_rlt"],
        checkpoint_dit=speed["checkpoint_dit"],
    )
    params = [p for p in model.parameters() if p.requires_grad]
    ddp = DDP(
        model,
        device_ids=[local],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=64,
        find_unused_parameters=False,
    )
    configure_fast_zero_checkpoint()
    optimizer = ZeroRedundancyOptimizer(
        params,
        optimizer_class=torch.optim.AdamW,
        lr=s["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=s["weight_decay"],
        fused=True,
    )
    saved = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
    for k in (
        "mode",
        "alpha",
        "lr",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "clip_grad",
    ):
        if saved["config"]["stage1"][k] != s[k]:
            raise ValueError(
                f"Resume changes training objective or optimizer field {k}"
            )
    optimizer.load_state_dict(saved["optimizer"])
    step = int(saved["step"])
    assets = saved["assets"]
    samples_seen = int(
        saved.get(
            "samples_seen",
            step
            * saved["config"]["stage1"]["micro_batch_size"]
            * saved["config"]["stage1"]["accumulate"],
        )
    )
    start_step = step
    del saved
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "initialized",
                    "world_size": world,
                    "per_card_batch": 32,
                    "global_batch": 256,
                    "resume_step": step,
                    "samples_seen": samples_seen,
                }
            ),
            flush=True,
        )
    stop_requested = [False]

    def stop(signum, frame):
        stop_requested[0] = True

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    while step < s["max_steps"]:
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        if batch["action"].shape[0] != 32:
            raise ValueError("Actual per-rank batch size changed")
        batch = move(batch, device)
        data_seconds = time.monotonic() - start
        optimizer.zero_grad(set_to_none=True)
        events[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = ddp(batch)
        events[1].record()
        result["loss"].backward()
        events[2].record()
        torch.nn.utils.clip_grad_norm_(params, s["clip_grad"], error_if_nonfinite=True)
        warmup = s["warmup_steps"]
        if step < warmup:
            lr = s["lr"] * (step + 1) / max(warmup, 1)
        else:
            progress = min(1.0, (step - warmup) / max(s["max_steps"] - warmup, 1))
            lr = s["min_lr"] + (s["lr"] - s["min_lr"]) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        events[3].record()
        torch.cuda.synchronize()
        step += 1
        samples_seen += 256
        values = torch.stack(
            [result[k].detach().float() for k in ("loss", "rlt_loss", "vla_loss")]
        )
        dist.all_reduce(values)
        values /= world
        elapsed = torch.tensor(time.monotonic() - start, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        control = torch.tensor(
            [
                int(stop_requested[0] or (rank == 0 and (a.output / "STOP").exists())),
                int(rank == 0 and (a.output / "SAVE").exists()),
            ],
            device=device,
        )
        dist.all_reduce(control, op=dist.ReduceOp.MAX)
        do_stop, do_save = control.tolist()
        if rank == 0:
            row = {
                "step": step,
                "loss": float(values[0]),
                "rlt_loss": float(values[1]),
                "vla_loss": float(values[2]),
                "lr": lr,
                "step_seconds": float(elapsed),
                "data_seconds": data_seconds,
                "forward_seconds": events[0].elapsed_time(events[1]) / 1000,
                "backward_seconds": events[1].elapsed_time(events[2]) / 1000,
                "optim_seconds": events[2].elapsed_time(events[3]) / 1000,
                "samples_seen": samples_seen,
                "epoch_equivalent": samples_seen / len(dataset),
                "per_card_batch": 32,
                "global_batch_size": 256,
                "world_size": 8,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with (a.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (
            do_stop
            or do_save
            or step % s["save_every"] == 0
            or step == s["max_steps"]
            or step == start_step + 10
        ):
            checkpoint_start = time.monotonic()
            optimizer.consolidate_state_dict(to=0)
            if rank == 0:
                save_stage1(
                    a.output / "latest.pt",
                    model,
                    cfg,
                    optimizer,
                    step,
                    assets,
                    samples_seen=samples_seen,
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                )
                (a.output / "SAVE").unlink(missing_ok=True)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "seconds": time.monotonic() - checkpoint_start,
                        }
                    ),
                    flush=True,
                )
            dist.barrier()
        if do_stop:
            break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
