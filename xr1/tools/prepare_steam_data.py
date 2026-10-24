"""Audit LWD assets and generate an episode-isolated STEAM dataset manifest."""

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
from mibot.steam.data import CAMERAS, digest


def prepare(source, output):
    meta = source / "metadata"
    outcomes = json.loads((meta / "outcomes.json").read_text())
    train = json.loads((meta / "train_paths.json").read_text())
    val = json.loads((meta / "validation_paths.json").read_text())
    if set(train) & set(val) or len(set(train + val)) != len(train + val):
        raise ValueError("duplicate or overlapping train/validation episodes")

    def audit(entry):
        split, path = entry
        ep = json.loads(Path(path).read_text())
        end = outcomes[path]
        n = end.get("usable_frames", ep["num_frames"])
        if not 2 <= n <= ep["num_frames"]:
            raise ValueError(f"invalid usable frame bound: {path}")
        with np.load(ep["arrays_path"], allow_pickle=False) as arrays:
            for key in arrays.files:
                x = arrays[key]
                if x.shape[0] != ep["num_frames"] or not np.isfinite(x).all():
                    raise ValueError(f"invalid action/state arrays: {path}:{key}")
        for camera in CAMERAS:
            infos = ep["observations"][camera]
            if len(infos) != 1:
                raise ValueError("exactly one video per camera required")
            info = infos[0]
            with av.open(info["path"]) as container:
                stream = container.streams.video[0]
                if not stream.frames or stream.frames < n + int(info.get("start", 0)):
                    raise ValueError(f"insufficient video frame count: {info['path']}")
        prompt = ep["instruction"]["general"][0]["conversations"][0]["value"]
        task = prompt.split("Generate robot actions for the task:\n")[-1].strip()
        category = ep["lwd_package"]["category"]
        return {
            "episode_id": digest(path)[:24],
            "path": str(Path(path).resolve()),
            "split": split,
            "type": "sft" if category == "LFD" else "rollout",
            "category": category,
            "usable_frames": n,
            "original_frames": ep["num_frames"],
            "end_status": end["end_status"],
            "task": task,
            "source": ep.get("source", {}),
            "json_sha256": digest(path),
            "arrays_sha256": digest(ep["arrays_path"]),
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(
            pool.map(
                audit, [("train", p) for p in train] + [("validation", p) for p in val]
            )
        )
    # Also reject duplicated source trajectories or arrays across the split.
    source_splits = {}
    for record in records:
        src = record["source"]
        key = (
            src.get("dataset"),
            src.get("source_session"),
            src.get("source_episode_index", src.get("episode_index")),
        )
        if key in source_splits and source_splits[key] != record["split"]:
            raise ValueError("same source episode across train/validation")
        source_splits[key] = record["split"]
    by_hash = {}
    for r in records:
        h = r["arrays_sha256"]
        if h in by_hash and by_hash[h] != r["split"]:
            raise ValueError(
                "identical action/state trajectory across train/validation"
            )
        by_hash[h] = r["split"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(meta / "normalize.json", output / "normalize.json")
    manifest = {
        "format": "xr1-steam-v1",
        "source_root": str(source.resolve()),
        "normalization": str((output / "normalize.json").resolve()),
        "normalization_sha256": digest(output / "normalize.json"),
        "active_action_dims": list(range(7)),
        "action_horizon": 30,
        "additional_action_shift": 0,
        "cameras": CAMERAS,
        "episodes": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "episodes": len(records),
        "splits": dict(Counter(r["split"] for r in records)),
        "pools": dict(Counter(r["type"] for r in records)),
        "categories": dict(Counter(r["category"] for r in records)),
        "frames": sum(r["usable_frames"] for r in records),
        "censored_frames": sum(
            r["original_frames"] - r["usable_frames"] for r in records
        ),
        "critic_pair_anchors": sum(r["usable_frames"] - 1 for r in records),
        "videos_checked": 3 * len(records),
        "advantage_labels": "not fabricated: generate with a trained STEAM ensemble",
        "assets": "existing videos and aligned action arrays referenced without modification",
        "manifest_sha256": digest(output / "manifest.json"),
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.output)
