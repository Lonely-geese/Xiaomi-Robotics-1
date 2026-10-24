"""Shared deterministic sampling, distributed training and atomic checkpoints."""

import json
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .distributed import rank, synchronize_gradients, world_size
from .optim import clip_grad_norm_fp32

STOP = False


def setup(seed):
    global STOP
    STOP = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def stop(signum, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return device


def stopping(device):
    flag = torch.tensor(int(STOP), device=device)
    if world_size() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def move(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move(v, device) for v in x)
    return x


def update(parameters, optimizer, lr, clip):
    optimizer.param_groups[0]["lr"] = lr
    synchronize_gradients(parameters)
    norm = clip_grad_norm_fp32(parameters, clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def save(directory, model, optimizer, step, config, rng=None):
    """Publish latest.json only after every rank's FP32 optimizer and RNG are saved."""
    directory = Path(directory)
    generation = [f"step_{step:08d}_{time.time_ns()}" if rank() == 0 else None]
    if world_size() > 1:
        dist.broadcast_object_list(generation, src=0)
    target = directory / generation[0]
    target.mkdir(parents=True, exist_ok=True)
    local = {
        "optimizer": optimizer.state_dict(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "python": random.getstate(),
        "sampler": rng.bit_generator.state if rng is not None else None,
    }
    atomic_torch(local, target / f"rank_{rank()}.pt")
    if world_size() > 1:
        dist.barrier()
    if rank() == 0:
        manifest = json.loads(Path(config["manifest"]).read_text())
        shutil.copyfile(manifest["normalization"], target / "normalize.json")
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        atomic_torch(
            {
                "format": "xr1-steam-checkpoint-v1",
                "step": step,
                "config": config,
                "world_size": world_size(),
                "model": model.state_dict(),
            },
            target / "model.pt",
        )
        pointer = directory / "latest.json.tmp"
        pointer.write_text(
            json.dumps({"checkpoint": str((target / "model.pt").resolve())}) + "\n"
        )
        pointer.replace(directory / "latest.json")
        print(
            json.dumps({"event": "saved", "step": step, "path": str(target)}),
            flush=True,
        )
    if world_size() > 1:
        dist.barrier()
    return target


def atomic_torch(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as f:
        os.fsync(f.fileno())
    temporary.replace(path)


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / "latest.json"
    if path.suffix == ".json":
        path = Path(json.loads(path.read_text())["checkpoint"])
    return path


def restore(path, model, optimizer, config, rng=None):
    path = resolve_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["config"] != config or saved["world_size"] != world_size():
        raise ValueError("resume requires the saved configuration and world size")
    model.load_state_dict(saved["model"], strict=True)
    local = torch.load(
        path.parent / f"rank_{rank()}.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    optimizer.load_state_dict(local["optimizer"])
    torch.set_rng_state(local["torch"])
    torch.cuda.set_rng_state(local["cuda"])
    random.setstate(local["python"])
    if rng is not None:
        rng.bit_generator.state = local["sampler"]
    return saved["step"]


def log(value):
    if rank() == 0:
        print(json.dumps(value), flush=True)
