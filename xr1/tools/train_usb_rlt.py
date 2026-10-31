"""Explicit single-GPU Stage 1 launcher; importing this file never starts training."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from mibot.rlt.data import ObservationCollator, USBDataset
from mibot.rlt.model import asset_metadata, build_model, move, save_stage1
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_rlt.yaml"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=["joint", "rlt_only"])
    p.add_argument("--max-steps", type=int)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This adapter launcher is single-GPU; it does not launch RLinf Ray/FSDP workers"
        )
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.mode:
        cfg["stage1"]["mode"] = a.mode
    if a.max_steps is not None:
        cfg["stage1"]["max_steps"] = a.max_steps
    s = cfg["stage1"]
    if s["max_steps"] < 1 or s["accumulate"] < 1:
        raise ValueError("max_steps and accumulate must be positive")
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise ValueError(
            "Output directory is nonempty; choose a new run or explicitly resume"
        )
    a.output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(8)
    os.environ["XR1_VLM_PATH"] = cfg["vlm_path"]
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    os.environ["XR1_STRICT_BATCH"] = "1"
    # Packed samples retain their full batch; do not silently discard long samples.
    os.environ["MAX_LENGTH"] = str(2048 * s["micro_batch_size"])
    dataset = USBDataset(
        cfg["dataset"], cfg["normalization"], stride=s["stride"], augment=s["augment"]
    )
    if s["mode"] == "joint":
        from mibot.data.collate.custom_collate import CustomCollate

        collator = CustomCollate()
    else:
        collator = ObservationCollator(cfg["vlm_path"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        dataset,
        batch_size=s["micro_batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=s["workers"],
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=s["workers"] > 0,
        multiprocessing_context="spawn" if s["workers"] > 0 else None,
    )
    model = build_model(cfg, a.device, a.resume)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=s["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=s["weight_decay"]
    )
    assets = asset_metadata(cfg)
    step = 0
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)
        if saved["config"] != cfg:
            raise ValueError("Resume config differs from checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        # Resume at a fresh shuffled pass. Optimizer/scheduler/model are restored;
        # exact mid-epoch sample order is intentionally not claimed.
    (a.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    iterator = iter(loader)
    with (a.output / "metrics.jsonl").open("a") as log:
        while step < s["max_steps"]:
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            values = {"loss": 0.0, "vla_loss": 0.0, "rlt_loss": 0.0}
            for _ in range(s["accumulate"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = move(batch, a.device)
                if s["mode"] == "rlt_only":
                    batch = {"observation": batch}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(batch)
                (result["loss"] / s["accumulate"]).backward()
                for key in values:
                    values[key] += float(result[key].detach()) / s["accumulate"]
            torch.nn.utils.clip_grad_norm_(
                params, s["clip_grad"], error_if_nonfinite=True
            )
            warmup = s["warmup_steps"]
            if step < warmup:
                lr = s["lr"] * (step + 1) / max(warmup, 1)
            else:
                t = min(1.0, (step - warmup) / max(s["max_steps"] - warmup, 1))
                lr = s["min_lr"] + (s["lr"] - s["min_lr"]) * 0.5 * (
                    1 + math.cos(math.pi * t)
                )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            values.update(
                step=step,
                lr=lr,
                step_seconds=time.monotonic() - step_start,
                samples_seen=step * s["micro_batch_size"] * s["accumulate"],
                epoch_equivalent=step
                * s["micro_batch_size"]
                * s["accumulate"]
                / len(dataset),
            )
            line = json.dumps(values)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
            if step % s["save_every"] == 0 or step == s["max_steps"]:
                save_stage1(
                    a.output / "latest.pt",
                    model,
                    cfg,
                    optimizer,
                    step,
                    assets,
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                )


if __name__ == "__main__":
    main()
