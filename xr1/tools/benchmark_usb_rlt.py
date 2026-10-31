"""Check frozen-prefix equivalence and time actual per-card 32-sample batches."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml
from mibot.data.collate.custom_collate import CustomCollate
from mibot.rlt.data import USBDataset
from mibot.rlt.fast import DenseUSBCollator, configure_fast
from mibot.rlt.model import build_model, move


def seed():
    random.seed(123)
    torch.manual_seed(123)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--parity", action="store_true")
    p.add_argument("--no-checkpoint-rlt", action="store_true")
    p.add_argument("--no-checkpoint-dit", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/usb_rlt.yaml").read_text()
    )
    torch.set_num_threads(4)
    os.environ.update(
        XR1_VLM_PATH=cfg["vlm_path"],
        XR1_STRICT_BATCH="1",
        MAX_LENGTH="65536",
        TOKENIZERS_PARALLELISM="false",
    )
    ds = USBDataset(cfg["dataset"], cfg["normalization"], augment=False)
    items = [
        ds.item_at(
            i % len(ds.records),
            min(i * 3, ds.records[i % len(ds.records)]["length"] - 1),
        )
        for i in range(a.batch)
    ]
    dense = DenseUSBCollator()
    model = build_model(cfg, "cuda", a.resume).train()
    report = {}
    if a.parity:
        ordinary = move(CustomCollate()(items[:2]), "cuda")
        seed()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ordinary)
        out["loss"].backward()
        baseline = {k: float(out[k].detach()) for k in ("loss", "vla_loss", "rlt_loss")}
        grad = model.base.state_projector_choice.layers[0].weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        del out, ordinary
        configure_fast(model, rlt_amp=False, checkpoint_rlt=False, checkpoint_dit=False)
        seed()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(move(dense(items[:2]), "cuda"))
        out["loss"].backward()
        fast = {k: float(out[k].detach()) for k in baseline}
        other = model.base.state_projector_choice.layers[0].weight.grad
        relative = float((grad - other).norm() / grad.norm().clamp_min(1e-12))
        cosine = float(
            torch.nn.functional.cosine_similarity(
                grad.flatten(), other.flatten(), dim=0
            )
        )
        for k, expected_value in baseline.items():
            torch.testing.assert_close(
                torch.tensor(fast[k]),
                torch.tensor(expected_value),
                rtol=0.02,
                atol=0.005,
            )
        print(
            json.dumps(
                {
                    "baseline": baseline,
                    "fast": fast,
                    "grad_relative": relative,
                    "grad_cosine": cosine,
                }
            ),
            flush=True,
        )
        if cosine < 0.995 or relative > 0.25:
            raise AssertionError(f"State gradient mismatch {relative=} {cosine=}")
        report.update(
            parity={
                "packed": baseline,
                "split": fast,
                "state_grad_relative_error": relative,
                "state_grad_cosine": cosine,
            }
        )
        print(json.dumps(report), flush=True)
        model.zero_grad(set_to_none=True)
        del out, other, grad
        del model
        torch.cuda.empty_cache()
        model = build_model(cfg, "cuda", a.resume).train()
    configure_fast(
        model,
        rlt_amp=True,
        checkpoint_rlt=not a.no_checkpoint_rlt,
        checkpoint_dit=not a.no_checkpoint_dit,
    )
    batch = move(dense(items), "cuda")
    print("batch_ready", batch["input_ids"].shape, flush=True)
    times = []
    for step in range(a.steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        out["loss"].backward()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        times.append(elapsed)
        row = {
            "step": step,
            "seconds": elapsed,
            "loss": float(out["loss"].detach()),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(json.dumps(row), flush=True)
    report.update(
        batch=a.batch,
        forward_backward_seconds=times,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    Path(a.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
