"""Resumable USB demonstration-pair export and held-out correction prewarming.

Freezes Stage 1; trains ONLY q_phi. This is not online BEE or an RL actor run.
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from mibot.bee.core import BeeLearner, correction_nll
from mibot.bee.data import load_settings, take, validate_payload
from mibot.bee.runtime import USBBEEPolicy
from mibot.rlt.data import ObservationCollator, USBDataset, sha256
from mibot.rlt.model import move


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def episode_split(records, seed, fraction):
    ids = sorted(r["episode_index"] for r in records)
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two episodes and a proper validation fraction")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    n = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
    validation = {ids[i] for i in order[:n].tolist()}
    return sorted(set(ids) - validation), sorted(validation)


def concatenate(rows):
    return {
        k: concatenate([r[k] for r in rows])
        if isinstance(rows[0][k], dict)
        else torch.cat([r[k] for r in rows])
        for k in rows[0]
    }


@torch.no_grad()
def evaluate(learner, data, batch_size, device):
    totals = {
        "nll": 0.0,
        "residual_mse": 0.0,
        "zero_residual_mse": 0.0,
        "variance": 0.0,
    }
    count = 0
    for start in range(0, len(data["actions"]), batch_size):
        batch = take(
            data,
            torch.arange(start, min(start + batch_size, len(data["actions"]))),
            device,
        )
        obs = batch["curr_obs"]
        mean, var = learner.model.distribution(obs)
        mask = batch["correction_mask"] & learner.model.editable
        n = int(mask.sum())
        residual = batch["actions"] - obs["ref_chunk"]
        totals["nll"] += (
            float(correction_nll(batch["actions"], obs["ref_chunk"], mean, var, mask))
            * n
        )
        totals["residual_mse"] += float(
            (residual - mean).square().masked_select(mask).sum()
        )
        totals["zero_residual_mse"] += float(
            residual.square().masked_select(mask).sum()
        )
        totals["variance"] += float(var.masked_select(mask).sum())
        count += n
    return {k: v / count for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_bee.yaml"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--feature-batch", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--max-samples",
        type=int,
        help="Diagnostic subset only; omission exports all selected chunks",
    )
    a = p.parse_args()
    if (
        min(a.feature_batch, a.shard_size, a.stride, a.batch_size, a.epochs, a.patience)
        < 1
    ):
        p.error("All sizes and counts must be positive")
    if a.max_samples is not None and a.max_samples < 2:
        p.error("max-samples must be at least 2")
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise ValueError("Nonempty output; use --resume or a new directory")
    a.output.mkdir(parents=True, exist_ok=True)
    cache = a.output / "features"
    cache.mkdir(exist_ok=True)
    settings, base, cfg = load_settings(a.config)
    torch.set_num_threads(8)
    torch.manual_seed(settings["seed"])
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    if base.get("image_cache"):
        os.environ["XR1_USB_IMAGE_CACHE"] = base["image_cache"]
    ds = USBDataset(base["dataset"], base["normalization"], stride=a.stride)
    train_episodes, validation_episodes = episode_split(
        ds.records, settings["seed"], a.validation_fraction
    )
    plan_samples = []
    for sample in ds.samples:
        record = ds.record_by_path[sample["file"]]
        if sample["frame_index"] + cfg.chunk_len <= record["length"]:
            plan_samples.append(
                {"episode": record["episode_index"], "frame": sample["frame_index"]}
            )
    if a.max_samples:
        # Diagnostic mode still includes examples from both split groups.
        val = [s for s in plan_samples if s["episode"] in validation_episodes]
        train = [s for s in plan_samples if s["episode"] in train_episodes]
        plan_samples = (
            train[: a.max_samples // 2] + val[: a.max_samples - a.max_samples // 2]
        )
    policy = USBBEEPolicy(base, settings["stage1_checkpoint"], cfg, device=a.device)
    assert not any(p.requires_grad for p in policy.features.parameters())
    assets = policy.assets
    plan = {
        "assets": assets,
        "manifest_sha256": sha256(Path(base["dataset"]) / "meta/manifest.jsonl"),
        "seed": settings["seed"],
        "stride": a.stride,
        "feature_batch": a.feature_batch,
        "shard_size": a.shard_size,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "samples": plan_samples,
        "source": "demonstration_pairs",
        "max_samples": a.max_samples,
    }
    plan_path = a.output / "feature_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Cached feature plan/assets mismatch")
    write_json(plan_path, plan)
    collator = ObservationCollator(base["vlm_path"])
    start_time, newly_exported = time.monotonic(), 0
    shard_paths = []
    rows = obs = items = None
    for start in range(0, len(plan_samples), a.shard_size):
        shard_id = start // a.shard_size
        path = cache / f"pairs_{shard_id:05d}.pt"
        selection = plan_samples[start : start + a.shard_size]
        shard_paths.append(path)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            validate_payload(payload, cfg, assets, corrections=True)
            if payload["samples"] != selection:
                raise ValueError("Cached shard sample order mismatch")
            continue
        # A per-shard RNG seed makes resumed exports reproducible independently of skipped shards.
        torch.manual_seed(settings["seed"] + shard_id)
        rows = []
        for offset in range(0, len(selection), a.feature_batch):
            selected = selection[offset : offset + a.feature_batch]
            items = [ds.item_at(s["episode"], s["frame"]) for s in selected]
            obs = move(policy.extract(collator(items)), "cpu")
            rows.append(
                {
                    "curr_obs": obs,
                    "actions": torch.stack(
                        [item["action"][: cfg.chunk_len, :7].float() for item in items]
                    ),
                    "correction_mask": torch.stack(
                        [
                            item["action_mask"][: cfg.chunk_len, :7].bool()
                            for item in items
                        ]
                    ),
                }
            )
            newly_exported += len(selected)
            status = {
                "phase": "exporting_demonstration_pairs",
                "completed": start + offset + len(selected),
                "total": len(plan_samples),
                "seconds_per_sample": (time.monotonic() - start_time) / newly_exported,
                "pid": os.getpid(),
                "device": a.device,
                "actor_training": False,
            }
            write_json(a.output / "status.json", status)
        payload = {
            "format": "xr1-bee-corrections-v1",
            "assets": assets,
            "source": "demonstration_pairs",
            "samples": selection,
            "transitions": concatenate(rows),
        }
        validate_payload(payload, cfg, assets, corrections=True)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        print(
            json.dumps(dict(event="feature_shard_saved", shard=shard_id, **status)),
            flush=True,
        )
    del policy, rows, obs, items
    gc.collect()
    if torch.device(a.device).type == "cuda":
        torch.cuda.empty_cache()
    train_rows, validation_rows = [], []
    fingerprint = hashlib.sha256()
    for path in shard_paths:
        fingerprint.update(sha256(path).encode())
        payload = torch.load(path, map_location="cpu", weights_only=True)
        flags = torch.tensor(
            [s["episode"] in validation_episodes for s in payload["samples"]]
        )
        if (~flags).any():
            train_rows.append(take(payload["transitions"], torch.where(~flags)[0]))
        if flags.any():
            validation_rows.append(take(payload["transitions"], torch.where(flags)[0]))
    if not train_rows or not validation_rows:
        raise ValueError("Empty train or validation split")
    train_data, validation_data = concatenate(train_rows), concatenate(validation_rows)
    torch.manual_seed(settings["seed"])
    learner = BeeLearner(cfg, a.device)
    training = {
        "mode": "correction",
        "source": "demonstration_pairs",
        "data_sha256": fingerprint.hexdigest(),
        "feature_plan_sha256": sha256(plan_path),
        "batch_size": a.batch_size,
        "epochs": a.epochs,
        "patience": a.patience,
        "seed": settings["seed"],
        "device_type": torch.device(a.device).type,
        "train_samples": len(train_data["actions"]),
        "validation_samples": len(validation_data["actions"]),
    }
    last = a.output / "latest.pt"
    epoch, best, bad, best_epoch = 0, float("inf"), 0, 0
    if a.resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        previous = dict(saved["training"])
        progress = previous.pop("progress")
        if previous != training:
            raise ValueError("Prewarm training settings/data changed")
        learner.restore(last, assets, saved["training"])
        epoch, best, bad, best_epoch = (
            progress[k] for k in ("epoch", "best_nll", "bad_epochs", "best_epoch")
        )
    else:
        baseline = evaluate(learner, validation_data, a.batch_size, a.device)
        write_json(a.output / "validation_before.json", baseline)
    while epoch < a.epochs and bad < a.patience:
        order = torch.randperm(len(train_data["actions"]))
        loss_total, examples = 0.0, 0
        for start in range(0, len(order), a.batch_size):
            indices = order[start : start + a.batch_size]
            metrics = learner.update_correction(take(train_data, indices, a.device))
            loss_total += metrics["correction_nll"] * len(indices)
            examples += len(indices)
        epoch += 1
        validation = evaluate(learner, validation_data, a.batch_size, a.device)
        improved = validation["nll"] < best
        if improved:
            best, bad, best_epoch = validation["nll"], 0, epoch
        else:
            bad += 1
        record = {
            "epoch": epoch,
            "train_nll": loss_total / examples,
            "validation": validation,
            "correction_steps": learner.correction_steps,
            "best_epoch": best_epoch,
            "actor_training": False,
        }
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        training["progress"] = {
            "epoch": epoch,
            "best_nll": best,
            "bad_epochs": bad,
            "best_epoch": best_epoch,
        }
        learner.save(last, assets, training)
        training.pop("progress")
        if improved:
            tmp = a.output / "best.pt.tmp"
            shutil.copyfile(last, tmp)
            tmp.replace(a.output / "best.pt")
        write_json(
            a.output / "status.json", dict(phase="correction_pretraining", **record)
        )
        print(json.dumps(record), flush=True)
    write_json(
        a.output / "status.json",
        {
            "phase": "complete",
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_validation_nll": best,
            "actor_training": False,
            "correction_steps": learner.correction_steps,
            "checkpoint": str(a.output / "best.pt"),
            "train_samples": len(train_data["actions"]),
            "validation_samples": len(validation_data["actions"]),
        },
    )


if __name__ == "__main__":
    main()
