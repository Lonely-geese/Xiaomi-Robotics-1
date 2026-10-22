"""Deduplicate LFD, isolate gripper segments and adapt RECAP without a second shift.

Source files remain intact. Videos are referenced at their original absolute
paths. Numeric sidecars are independent copies/conversions in the output pool.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_ethernet_dataset import CAMERAS, NAMES, inspect_episode, trajectory_arrays


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_key(episode, converted=False):
    if converted:
        source = episode["source"]
        return source["session"], int(source["episode_index"])
    session = episode.get("original_source_session", episode.get("source_session"))
    index = episode.get("original_source_episode_index", episode.get("source_episode_index"))
    if session is None or index is None:
        raise ValueError("missing source identity; cannot safely deduplicate")
    return session, int(index)


def outcome(end_status, review=None):
    if end_status not in {"success", "failure", "truncated", "unknown"}:
        raise ValueError(f"invalid end_status: {end_status}")
    result = {"terminated": end_status in {"success", "failure"},
              "success": end_status == "success", "end_status": end_status}
    if end_status == "unknown" and review and "usable_frames" in review:
        result["usable_frames"] = review["usable_frames"]
    return result


def choose_added(episodes, canonical_keys, flagged):
    """Quarantine has precedence; canonical LFD wins source-identity duplicates."""
    groups = defaultdict(list)
    for path, episode in episodes:
        groups[source_key(episode, True)].append((path, episode))
    rows = []
    for key, members in sorted(groups.items()):
        for path, episode in sorted(members):
            if path.stem in flagged:
                bucket = "quarantine_gripper"
            elif key in canonical_keys:
                bucket = "duplicate"
            elif len(members) > 1:
                # A source-level success verdict cannot label every segment end.
                bucket = "pending_segment_boundary"
            else:
                bucket = "ready"
            rows.append((path, episode, key, bucket, key in canonical_keys))
    return rows


def copy_converted(path, episode, destination):
    copied = copy.deepcopy(episode)
    source_array = (path.parent / copied["arrays_path"]).resolve()
    array_path = destination / "arrays" / f"{path.stem}.npz"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_array, array_path)
    copied["arrays_path"] = str(array_path.resolve())
    for views in copied["observations"].values():
        for view in views:
            view["path"] = str((path.parent / view["path"]).resolve())
            if not Path(view["path"]).is_file():
                raise FileNotFoundError(view["path"])
    if "gripper_preprocessing" in copied:
        copied["gripper_preprocessing"]["script"] = str((path.parent / copied["gripper_preprocessing"]["script"]).resolve())
    copied["lwd_preparation"] = {"original_json": str(path), "additional_action_shift": 0}
    output = destination / "episodes" / path.name
    write_json(output, copied)
    return output.resolve()


def convert_parquet(job):
    root, info, episode, destination, status = job
    audit, state, action, videos = inspect_episode((root, info, episode))
    if audit["errors"]:
        raise ValueError(f"{root.name}/{episode['episode_index']}: {audit['errors']}")
    if not audit["action_equals_state_plus3"]:
        raise ValueError("expected already aligned state[t+3] targets; no automatic repair allowed")
    stem = f"{root.name}_episode_{episode['episode_index']:06d}"
    arrays = destination / "arrays" / f"{stem}.npz"
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arrays, **trajectory_arrays(state, action))
    prompt = episode.get("prompt") or info.get("prompt") or episode["tasks"][0]
    source = {"dataset": str(root), "episode_index": episode["episode_index"],
              "source_episode_index": episode.get("source_episode_index"),
              "source_session": episode.get("original_source_session", episode.get("source_session")),
              "outcome": episode.get("outcome", episode.get("operator_verdict"))}
    record = {"trajectory_type": "success" if status == "success" else "ongoing",
              "time": stem, "num_frames": len(state), "arrays_path": str(arrays.resolve()),
              "observations": videos, "source": source,
              "arm_action_definition": "source measured state[t+3], already aligned; additional shift=0",
              "lwd_preparation": {"additional_action_shift": 0, "original_metadata": episode,
                                  "end_status": status, "audit": audit},
              "instruction": {"general": [{"images": ["observations." + v for v in CAMERAS],
                  "conversations": [{"from": "human", "value":
                      "The following observations are captured from multiple views.\n# Ego View\n<image>\n"
                      "# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
                      "Generate robot actions for the task:\n" + prompt}, {"from": "gpt", "value": ""}]}]}}
    output = destination / "episodes" / f"{stem}.json"
    write_json(output, record)
    return output.resolve(), audit


def validation_groups(records, fraction):
    """Split entire LFD sessions; RECAP lacks session IDs, so split whole episodes."""
    result = set()
    strata = defaultdict(set)
    for row in records:
        strata[(row["pool"], row["status"])].add(row["split_group"])
    for groups in strata.values():
        ordered = sorted(groups, key=lambda key: hashlib.sha256(("lwd-v1:" + key).encode()).hexdigest())
        count = min(len(ordered) - 1, max(1, round(len(ordered) * fraction))) if fraction and len(ordered) > 1 else 0
        result.update(ordered[:count])
    return result


def prepare(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; use a new path for another cleaning version")
    output.mkdir(parents=True, exist_ok=True)
    canonical, added, recap = [p.resolve() for p in (args.lfd, args.added, args.recap)]
    flagged = set(json.loads(args.gripper_flags.read_text()))
    canonical_episodes = read_jsonl(canonical / "meta/episodes.jsonl")
    keys = {source_key(e) for e in canonical_episodes}
    if len(keys) != len(canonical_episodes):
        raise ValueError("canonical LFD itself contains duplicate source episodes")
    added_episodes = [(p, json.loads(p.read_text())) for p in sorted((added / "episodes").glob("*.json"))]
    if flagged - {p.stem for p, _ in added_episodes}:
        raise ValueError("gripper quarantine list contains unknown segments")
    # Quarantine selection is the reviewed list, never re-inferred from filenames.
    choices = choose_added(added_episodes, keys, flagged)
    reviews = json.loads(args.ending_review.read_text()) if args.ending_review else []
    review_map = {int(row["episode_index"]): row for row in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("duplicate episode IDs in ending review")
    recap_episodes = read_jsonl(recap / "meta/episodes.jsonl")
    if set(review_map) - {e['episode_index'] for e in recap_episodes}:
        raise ValueError("ending review refers to removed or absent episodes")
    jobs, job_metadata, records, manifests = [], [], [], []
    for root, eps in [(canonical, canonical_episodes), (recap, recap_episodes)]:
        info = json.loads((root / "meta/info.json").read_text())
        for field in ("observation.state", "action"):
            if info["features"][field]["names"] != NAMES:
                raise ValueError(f"unexpected {field} layout in {root}")
        for ep in eps:
            is_lfd = root == canonical
            label = ep.get("operator_verdict") if is_lfd else ep.get("outcome")
            review = review_map.get(ep["episode_index"], {}) if not is_lfd else {}
            if label == "success":
                status = "success"
            elif label == "failure":
                status = review.get("end_status", "unknown")
                if status not in {"failure", "truncated", "unknown"}:
                    raise ValueError("failure reviews may not relabel task outcome as success")
            else:
                status = "unlabeled"
            bucket = output / "ready" / ("lfd" if is_lfd else "recap") / status
            if status == "unknown" and review.get("visual_category"):
                category = review["visual_category"]
                if category not in {"release_withdraw", "human_reset", "contact_at_cut",
                                    "withdraw_holding", "fixture_at_cut", "connector_on_table"}:
                    raise ValueError(f"unsupported visual category: {category}")
                bucket /= category
            if status == "unlabeled":
                bucket = output / "quarantine" / "unlabeled"
            jobs.append((root, info, ep, bucket, status))
            group = "lfd:" + source_key(ep)[0] if is_lfd else f"recap:{ep['source_episode_index']}"
            job_metadata.append(dict(pool="lfd" if is_lfd else "recap", status=status,
                                     split_group=group, source_episode=ep["episode_index"], review=review))
    with ThreadPoolExecutor(args.workers) as executor:
        for number, ((path, audit), meta) in enumerate(zip(executor.map(convert_parquet, jobs), job_metadata), 1):
            meta.update(path=str(path), frames=audit["length"])
            manifests.append(meta)
            if meta["status"] != "unlabeled":
                records.append(meta)
            if number % 300 == 0:
                print(f"Converted {number}/{len(jobs)} parquet episodes", flush=True)
    for number, (path, ep, key, bucket, duplicate) in enumerate(choices, 1):
        if bucket == "duplicate":
            destination = None
        else:
            folder = output / "ready/lfd/success" if bucket == "ready" else output / "quarantine" / bucket.removeprefix("quarantine_")
            destination = copy_converted(path, ep, folder)
        row = dict(pool="lfd_added", original_path=str(path), path=str(destination) if destination else None,
                   status=bucket, source_session=key[0], source_episode=key[1],
                   duplicate_of_canonical=duplicate, frames=ep["num_frames"])
        manifests.append(row)
        if bucket == "ready":
            records.append(dict(pool="lfd", status="success", path=str(destination),
                                split_group="lfd:"+key[0], frames=ep["num_frames"], source_episode=key[1]))
        if number % 500 == 0:
            print(f"Sorted {number}/{len(choices)} added segments", flush=True)
    normalization = output / "normalize.json"
    shutil.copy2(args.normalization or added / "normalize.json", normalization)
    write_json(output / "outcomes.json", {r["path"]: outcome(r["status"], r.get("review")) for r in records})
    validation = validation_groups(records, args.validation_fraction)
    train = [r for r in records if r["split_group"] not in validation]
    val = [r for r in records if r["split_group"] in validation]
    for name, subset in [("train", train), ("validation", val)]:
        write_json(output / f"{name}_paths.json", [r["path"] for r in subset])
    write_json(output / "manifest.json", manifests)
    write_json(output / "ending_review.json", reviews)
    pending = [r for r in records if r["status"] == "unknown"]
    with (output / "ending_labels_to_review.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_index", "end_status", "basis", "path"])
        writer.writeheader()
        for r in pending:
            writer.writerow(dict(episode_index=r["source_episode"], end_status="unknown",
                                 basis=r.get("review", {}).get("basis", "missing ending reason"), path=r["path"]))
    config = json.loads((Path(__file__).parents[1] / "configs/lwd.json").read_text())
    config["offline_outcomes"] = str(output / "outcomes.json")
    config["offline_data"].update(paths=[r["path"] for r in train], normalization=str(normalization),
                                   active_action_dims=list(range(7)), sample_stride=30)
    config["output_dir"] = str(output / "checkpoints")
    write_json(output / "lwd_offline.json", config)
    summary = dict(status="complete", source_roots=[str(p) for p in (canonical, added, recap)],
        additional_action_shift=0, normalization=str(normalization),
        normalization_sha256=hashlib.sha256(normalization.read_bytes()).hexdigest(),
        ready_by_pool_status=dict(Counter(r['pool']+":"+r['status'] for r in records)),
        added_selection=dict(Counter(row[3] for row in choices)),
        source_overlap=len(keys & {source_key(e,True) for _,e in added_episodes}),
        duplicate_segments=sum(row[4] for row in choices), ready_episodes=len(records),
        ready_frames=sum(r['frames'] for r in records), train_episodes=len(train), validation_episodes=len(val),
        validation_fraction_requested=args.validation_fraction,
        split_policy="LFD source-session groups; RECAP whole source episodes stratified by end status",
        unknown_boundary_policy="censor reviewed ambiguous tails using usable_frames, then exclude n-chunk windows reaching the retained boundary",
        ending_tail_frames_excluded=sum(r['frames'] - r.get('review', {}).get('usable_frames', r['frames'])
                                       for r in records if r['status'] == 'unknown'),
        videos="Referenced from original absolute paths; source pools must remain available",
        deleted_source_episodes_restored=0, source_files_modified=0,
        ending_labels_are="Review-derived with explicit evidence/confidence; unknown remains unknown")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfd",type=Path,default=Path('/data/wz/data/LFD/20261016_20261019_1563'))
    parser.add_argument("--added",type=Path,default=Path('/data/wz/data/LFD/xr1_added_oct16_oct19_oct20_20261021'))
    parser.add_argument("--recap",type=Path,default=Path('/data/wz/data/RECAP/recap_755'))
    parser.add_argument("--gripper-flags",type=Path,required=True)
    parser.add_argument("--ending-review",type=Path)
    parser.add_argument("--normalization",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--validation-fraction",type=float,default=.1)
    args=parser.parse_args()
    if not 0 <= args.validation_fraction < 1 or args.workers < 1:
        parser.error("invalid validation fraction or worker count")
    prepare(args)


if __name__=="__main__":main()
