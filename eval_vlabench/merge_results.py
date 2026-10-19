# Copyright (C) 2026 Xiaomi Corporation.
"""Merge per-track VLABench metrics into JSON and Markdown summaries."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

import tyro


@dataclasses.dataclass
class Args:
    eval_log_dir: str = "./eval_vlabench/eval_logs"
    output_name: str = "merged_results.json"
    markdown_name: str = "results.md"


def _load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run(args: Args) -> None:
    root = Path(args.eval_log_dir).resolve()
    track_results: Dict[str, Any] = {}
    all_task_metrics: Dict[str, list[Dict[str, float]]] = {}

    for metrics_path in sorted(root.glob("track_*/metrics.json")):
        track = metrics_path.parent.name
        task_metrics = _load_metrics(metrics_path)
        track_results[track] = {
            "tasks": task_metrics,
            "average": {
                metric: _average([float(scores.get(metric, 0.0)) for scores in task_metrics.values()])
                for metric in ("success_rate", "intention_score", "progress_score")
            },
        }
        for task, scores in task_metrics.items():
            all_task_metrics.setdefault(task, []).append(scores)

    if not track_results:
        raise FileNotFoundError(f"No track_*/metrics.json files found under {root}")

    overall = {
        metric: _average(
            [
                float(scores.get(metric, 0.0))
                for track in track_results.values()
                for scores in track["tasks"].values()
            ]
        )
        for metric in ("success_rate", "intention_score", "progress_score")
    }
    merged = {"tracks": track_results, "overall": overall}
    output_path = root / args.output_name
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    rows = [
        "| Track | Tasks | SR | IS | PS |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for track, result in track_results.items():
        average = result["average"]
        rows.append(
            f"| {track} | {len(result['tasks'])} | {average['success_rate']:.3f} | "
            f"{average['intention_score']:.3f} | {average['progress_score']:.3f} |"
        )
    rows.append(
        f"| **Overall** | **{sum(len(result['tasks']) for result in track_results.values())}** | "
        f"**{overall['success_rate']:.3f}** | **{overall['intention_score']:.3f}** | "
        f"**{overall['progress_score']:.3f}** |"
    )
    markdown_path = root / args.markdown_name
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Merged JSON: {output_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    run(tyro.cli(Args))
