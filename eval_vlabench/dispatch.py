# Copyright (C) 2026 Xiaomi Corporation.
"""Parallel VLABench dispatcher with one persistent XR-1 Client per worker."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tyro

from main import VLABenchPolicy, VLABenchPolicyArgs, _find_track_config, _load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    model_path: str
    eval_track: str
    save_dir: str
    host: str = "127.0.0.1"
    base_port: int = 10086
    num_workers: int = 8
    robot_type: str = "vlabench_choice"
    tasks: Optional[str] = None
    n_episode: int = 50
    metrics: str = "success_rate intention_score progress_score"
    visualization: bool = True
    state_dim: int = 60
    action_dim: int = 7
    action_chunk_size: int = 10
    replan_steps: int = 5
    image_size: int = 480
    cot: bool = False
    position_offset_x: float = 0.0
    position_offset_y: float = -0.4
    position_offset_z: float = 0.78
    gripper_threshold: float = 0.2
    gripper_open_value: float = 0.04
    request_seed: int = 42


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if value is None:
        return default
    try:
        normalized = str(value).strip().lower()
        if normalized in {"true", "false"}:
            return float(int(normalized == "true"))
        return float(normalized)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _get_first_existing(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _parse_video_metrics(filename: str) -> Optional[Tuple[bool, float]]:
    """Parse VLABench video names such as success_True_progress_1.00.mp4."""
    match = re.search(
        r"(?:^|_)success_(?P<success>True|False)_progress_(?P<progress>[-+0-9.eE]+)\.mp4$",
        filename,
    )
    if match is None:
        return None
    return match.group("success") == "True", float(match.group("progress"))


def _safe_load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _select_tasks(track_config: Dict[str, Any], tasks_text: Optional[str]) -> List[str]:
    if not tasks_text:
        return list(track_config.keys())
    requested = tasks_text.split()
    selected = [task for task in requested if task in track_config]
    missing = [task for task in requested if task not in track_config]
    if missing:
        logger.warning("Tasks absent from the selected track: %s", missing)
    return selected


def _build_jobs(track_config: Dict[str, Any], tasks: Iterable[str], n_episode: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for task in tasks:
        episodes = track_config[task]
        if not isinstance(episodes, list):
            raise ValueError(f"Track entry for {task} must be a list, got {type(episodes).__name__}")
        for episode_index, episode_config in enumerate(episodes[:n_episode]):
            jobs.append(
                {
                    "task": task,
                    "episode_index": episode_index,
                    "episode_config": episode_config,
                }
            )
    return jobs


def _normalize_detail_records(value: Any, task: str, episode_index: int) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["episode_id"] = episode_index
        record["task"] = str(record.get("task", task))
        record["success"] = _coerce_bool(
            _get_first_existing(
                record,
                ("success", "success_rate", "is_success", "task_success"),
                False,
            )
        )
        record["consumed_step"] = int(record.get("consumed_step", 0) or 0)
        record["intention_score"] = _coerce_float(
            _get_first_existing(
                record,
                ("intention_score", "intention", "task_intention"),
                0.0,
            )
        )
        record["progress_score"] = _coerce_float(
            _get_first_existing(
                record,
                ("progress_score", "progress", "task_progress"),
                0.0,
            )
        )
        records.append(record)
    return records


def _merge_video_metrics(
    records: List[Dict[str, Any]],
    *,
    task: str,
    episode_index: int,
    video_metric: Optional[Tuple[bool, float]],
) -> List[Dict[str, Any]]:
    """Use the final video filename as a fallback/override for SR and PS."""
    if video_metric is None:
        return records

    success, progress = video_metric
    if records:
        record = records[0]
    else:
        record = {
            "episode_id": episode_index,
            "task": task,
            "consumed_step": 0,
            "intention_score": 0.0,
        }
        records = [record]

    record["episode_id"] = episode_index
    record["task"] = str(record.get("task", task))
    record["success"] = bool(success)
    record["progress_score"] = float(progress)
    record["intention_score"] = _coerce_float(record.get("intention_score"))
    record["consumed_step"] = int(record.get("consumed_step", 0) or 0)
    return records


def _worker(worker_id: int, port: int, args: Args, jobs: List[Dict[str, Any]], result_queue: mp.Queue) -> None:
    """Run multiple episodes while keeping one processor and one socket connection alive."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        from VLABench.evaluation.evaluator import Evaluator

        policy = VLABenchPolicy(
            VLABenchPolicyArgs(
                model_path=args.model_path,
                host=args.host,
                port=port,
                robot_type=args.robot_type,
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                action_chunk_size=args.action_chunk_size,
                replan_steps=args.replan_steps,
                image_size=args.image_size,
                cot=args.cot,
                position_offset_x=args.position_offset_x,
                position_offset_y=args.position_offset_y,
                position_offset_z=args.position_offset_z,
                gripper_threshold=args.gripper_threshold,
                gripper_open_value=args.gripper_open_value,
                request_seed=args.request_seed,
            )
        )
    except Exception:
        error = traceback.format_exc()
        for job in jobs:
            result_queue.put(
                {
                    "task": job["task"],
                    "episode_index": job["episode_index"],
                    "worker_id": worker_id,
                    "port": port,
                    "ok": False,
                    "error": error,
                }
            )
        return

    try:
        for job in jobs:
            task = str(job["task"])
            episode_index = int(job["episode_index"])
            run_dir = Path(args.save_dir) / "_worker_runs" / f"worker_{worker_id:02d}" / task / f"ep_{episode_index:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                policy.reset()
                evaluator = Evaluator(
                    tasks=[task],
                    n_episodes=1,
                    episode_config={task: [job["episode_config"]]},
                    max_substeps=1,
                    save_dir=str(run_dir),
                    visulization=args.visualization,
                    metrics=args.metrics.split(),
                )
                evaluator.evaluate(policy)
                result_queue.put(
                    {
                        "task": task,
                        "episode_index": episode_index,
                        "worker_id": worker_id,
                        "port": port,
                        "ok": True,
                        "run_dir": str(run_dir),
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "task": task,
                        "episode_index": episode_index,
                        "worker_id": worker_id,
                        "port": port,
                        "ok": False,
                        "run_dir": str(run_dir),
                        "error": traceback.format_exc(),
                    }
                )
    finally:
        policy.close()


def _merge_worker_result(save_root: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    task = str(result["task"])
    episode_index = int(result["episode_index"])
    task_dir = save_root / task
    videos_dir = task_dir / "videos"
    logs_dir = task_dir / "worker_logs"
    task_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not result.get("ok"):
        error_file = logs_dir / f"ep_{episode_index:04d}.error.log"
        error_file.write_text(str(result.get("error", "Unknown worker error")), encoding="utf-8")
        return result

    run_dir = Path(result["run_dir"])
    detail_records = _normalize_detail_records(
        _safe_load_json(run_dir / task / "detail_info.json"), task, episode_index
    )

    source_videos = run_dir / task / "videos"
    moved_videos: List[str] = []
    video_metric: Optional[Tuple[bool, float]] = None
    if source_videos.is_dir():
        for source in sorted(source_videos.iterdir()):
            if not source.is_file():
                continue
            suffix = source.name.split("_", 1)[-1]
            destination = videos_dir / f"{episode_index}_{suffix}"
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
            moved_videos.append(str(destination))
            parsed_metric = _parse_video_metrics(destination.name)
            if parsed_metric is not None:
                video_metric = parsed_metric

    detail_records = _merge_video_metrics(
        detail_records,
        task=task,
        episode_index=episode_index,
        video_metric=video_metric,
    )
    detail_path = task_dir / "detail_info.json"
    existing = _safe_load_json(detail_path)
    if not isinstance(existing, list):
        existing = []
    existing = [
        record
        for record in existing
        if isinstance(record, dict) and int(record.get("episode_id", -1)) != episode_index
    ]
    existing.extend(detail_records)
    existing.sort(key=lambda record: int(record.get("episode_id", -1)))
    detail_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    result["detail_records"] = len(detail_records)
    result["videos"] = moved_videos
    shutil.rmtree(run_dir, ignore_errors=True)
    return result


def _compute_metrics(detail_records: Any) -> Dict[str, float]:
    if not isinstance(detail_records, list) or not detail_records:
        return {"success_rate": 0.0, "intention_score": 0.0, "progress_score": 0.0}
    count = len(detail_records)
    return {
        "success_rate": sum(_coerce_bool(record.get("success")) for record in detail_records) / count,
        "intention_score": sum(_coerce_float(record.get("intention_score")) for record in detail_records) / count,
        "progress_score": sum(_coerce_float(record.get("progress_score")) for record in detail_records) / count,
    }


def run(args: Args) -> None:
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if args.n_episode <= 0:
        raise ValueError("n_episode must be positive")

    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    track_config = _load_json(_find_track_config(args.eval_track))
    tasks = _select_tasks(track_config, args.tasks)
    jobs = _build_jobs(track_config, tasks, args.n_episode)
    if not jobs:
        logger.warning("No VLABench jobs were resolved.")
        return

    worker_count = min(args.num_workers, len(jobs))
    assignments: List[List[Dict[str, Any]]] = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        assignments[index % worker_count].append(job)

    logger.info(
        "Dispatching track=%s tasks=%d episodes=%d workers=%d ports=%d-%d",
        args.eval_track,
        len(tasks),
        len(jobs),
        worker_count,
        args.base_port,
        args.base_port + worker_count - 1,
    )

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    processes: List[mp.Process] = []
    for worker_id, worker_jobs in enumerate(assignments):
        process = context.Process(
            target=_worker,
            args=(worker_id, args.base_port + worker_id, args, worker_jobs, result_queue),
            daemon=False,
        )
        process.start()
        processes.append(process)

    raw_results: List[Dict[str, Any]] = []
    received_jobs = set()
    while len(raw_results) < len(jobs):
        try:
            result = result_queue.get(timeout=5.0)
            raw_results.append(result)
            received_jobs.add((str(result["task"]), int(result["episode_index"])))
        except queue.Empty:
            if any(process.is_alive() for process in processes):
                continue
            break

    for process in processes:
        process.join()

    for job in jobs:
        job_key = (str(job["task"]), int(job["episode_index"]))
        if job_key not in received_jobs:
            raw_results.append(
                {
                    "task": job["task"],
                    "episode_index": job["episode_index"],
                    "worker_id": None,
                    "port": None,
                    "ok": False,
                    "error": "Worker exited before returning a result for this episode.",
                }
            )

    merged_results = [_merge_worker_result(save_root, result) for result in raw_results]
    merged_results.sort(key=lambda result: (str(result["task"]), int(result["episode_index"])))

    track_metrics: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        track_metrics[task] = _compute_metrics(_safe_load_json(save_root / task / "detail_info.json"))
    (save_root / "metrics.json").write_text(json.dumps(track_metrics, indent=2), encoding="utf-8")

    manifest = {
        "eval_track": args.eval_track,
        "model_path": args.model_path,
        "robot_type": args.robot_type,
        "num_workers": worker_count,
        "base_port": args.base_port,
        "num_jobs": len(jobs),
        "metrics": track_metrics,
        "results": merged_results,
    }
    (save_root / "dispatch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(save_root / "_worker_runs", ignore_errors=True)

    failed = sum(not result.get("ok", False) for result in merged_results)
    logger.info("Track complete: %d succeeded, %d failed. Metrics: %s", len(jobs) - failed, failed, save_root / "metrics.json")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run(tyro.cli(Args))
