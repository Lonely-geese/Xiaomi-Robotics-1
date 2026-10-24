"""Stage 2: conservative ensemble advantages and strict RLinf quantile labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mibot.steam.critic import build_critic
from mibot.steam.data import digest, pair_dataset, read_manifest
from mibot.steam.runtime import move, resolve_checkpoint
from mibot.steam.upstream.advantage import apply_boolean_label, quantile_threshold
from torch.utils.data import DataLoader


def label_scores(frame, rollout_fraction=0.3, expert_fraction=0.8):
    """Fit thresholds on training episodes only; use the same thresholds for validation."""
    thresholds = {}
    if not np.isfinite(frame.advantage_continuous).all():
        raise ValueError("nonfinite scores")
    for kind, fraction in [("rollout", rollout_fraction), ("sft", expert_fraction)]:
        selected = frame.type == kind
        if not selected.any():
            continue
        train = frame.loc[selected & (frame.split == "train"), "advantage_continuous"]
        if len(train) == 0:
            raise ValueError(f"no training scores for {kind} threshold")
        if fraction is not None and not 0 < fraction < 1:
            raise ValueError("quantile fractions must be in (0,1)")
        threshold = (
            quantile_threshold(train, fraction) if fraction is not None else None
        )
        thresholds[kind] = threshold
        frame.loc[selected, "advantage_boolean"] = (
            True
            if threshold is None
            else apply_boolean_label(
                frame.loc[selected, "advantage_continuous"], threshold, inclusive=False
            )
        )
    frame["advantage_boolean"] = frame.advantage_boolean.astype(bool)
    return frame, thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rollout-fraction", type=float, default=0.3)
    p.add_argument("--expert-fraction", type=float, default=0.8)
    p.add_argument("--all-experts-positive", action="store_true")
    p.add_argument("--allow-smoke", action="store_true")
    a = p.parse_args()
    path = resolve_checkpoint(a.checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = saved["config"]
    if saved["step"] < 1:
        raise ValueError("advantages require a trained critic")
    if cfg.get("smoke_only") and not a.allow_smoke:
        raise ValueError("refusing smoke-only checkpoint")
    manifest = read_manifest(a.manifest)
    if cfg["manifest_sha256"] != digest(a.manifest):
        raise ValueError("critic and labeling manifests differ")
    model, collate = build_critic(cfg)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to("cuda").eval()
    del saved
    # Per-episode shards make a long labeling run restartable. A checkpoint hash prevents stale reuse.
    identity = {
        "manifest_sha256": digest(a.manifest),
        "checkpoint_sha256": digest(path),
    }
    cache = Path(str(a.output) + ".parts")
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / "identity.json"
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError("stale labeling cache")
    metadata.write_text(json.dumps(identity) + "\n")
    for i, record in enumerate(manifest["episodes"]):
        shard = cache / (record["episode_id"] + ".parquet")
        if shard.exists():
            continue
        dataset = pair_dataset([record], cfg, inference=True)
        rows = []
        loader = DataLoader(
            dataset, batch_size=a.batch_size, num_workers=a.workers, collate_fn=collate
        )
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.predict(move(batch["observation"], "cuda"))
                scores = out.predicted_values.float().cpu().tolist()
                members = out.member_predicted_values.float().T.cpu().tolist()
                for frame, score, values in zip(
                    batch["frame_idx_t"].tolist(), scores, members
                ):
                    rows.append(
                        {
                            "episode_id": record["episode_id"],
                            "frame_index": frame,
                            "advantage_continuous": score,
                            "ensemble_signed_score": score,
                            "member_values": values,
                            "split": record["split"],
                            "type": record["type"],
                        }
                    )
        # Exact RLinf terminal-row convention; this is a score placeholder, not a success reward.
        rows.append(
            {
                "episode_id": record["episode_id"],
                "frame_index": record["usable_frames"] - 1,
                "advantage_continuous": 0.0,
                "ensemble_signed_score": 0.0,
                "member_values": [0.0] * len(model.members),
                "split": record["split"],
                "type": record["type"],
            }
        )
        if sorted(r["frame_index"] for r in rows) != list(
            range(record["usable_frames"])
        ):
            raise ValueError("incomplete episode scoring")
        temporary = Path(str(shard) + ".tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(shard)
        print(
            json.dumps({"episodes_done": i + 1, "total": len(manifest["episodes"])}),
            flush=True,
        )
    result = pd.concat(
        [
            pd.read_parquet(cache / (r["episode_id"] + ".parquet"))
            for r in manifest["episodes"]
        ],
        ignore_index=True,
    )
    expected = sum(r["usable_frames"] for r in manifest["episodes"])
    if (
        len(result) != expected
        or result.duplicated(["episode_id", "frame_index"]).any()
    ):
        raise ValueError("invalid coverage")
    result, thresholds = label_scores(
        result,
        a.rollout_fraction,
        None if a.all_experts_positive else a.expert_fraction,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.output) + ".tmp")
    result.to_parquet(tmp, index=False)
    tmp.replace(a.output)
    sidecar = dict(
        **identity,
        complete=True,
        rows=len(result),
        thresholds=thresholds,
        strict_comparison=">",
        threshold_fit_split="train",
        smoke_only=cfg.get("smoke_only", False),
        advantages_sha256=digest(a.output),
    )
    Path(str(a.output) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar), flush=True)


if __name__ == "__main__":
    main()
