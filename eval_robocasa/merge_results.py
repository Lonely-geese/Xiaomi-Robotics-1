# Copyright (C) 2026 Xiaomi Corporation.
import json
import os
import glob
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    eval_log_dir: str = "./eval_robocasa/eval_logs"


def merge_results(args: Args):
    result_dir = args.eval_log_dir
    json_pattern = os.path.join(result_dir, "eval_results_*.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print("No JSON files found matching pattern:", json_pattern)
        return

    print(f"Found {len(json_files)} result files in: {result_dir}\n")

    all_results = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        for task_name, task_data in data.items():
            all_results.append({
                "task_name": task_name,
                "num_success_rollouts": task_data["num_success_rollouts"],
                "num_rollouts": task_data["num_rollouts"],
                "success_rate": task_data["success_rate"],
            })

    print("Individual Task Results:")
    print("=" * 70)
    print(f"{'Task Name':<30} {'Success Rate':<15} {'Detail'}")
    print("-" * 70)

    total_success = 0
    total_rollouts = 0
    for result in all_results:
        print(f"{result['task_name']:<30} {result['success_rate']:<15.1%} "
              f"({result['num_success_rollouts']}/{result['num_rollouts']})")
        total_success += result["num_success_rollouts"]
        total_rollouts += result["num_rollouts"]

    overall_rate = total_success / total_rollouts if total_rollouts > 0 else 0
    print("-" * 70)
    print(f"{'Overall':<30} {overall_rate:<15.1%} ({total_success}/{total_rollouts})")
    print()

    print("Summary Statistics:")
    print(f"  Total Tasks: {len(all_results)}")
    print(f"  Overall Success Rate: {overall_rate:.1%}")

    merged_data = {
        "individual_results": all_results,
        "summary": {
            "total_tasks": len(all_results),
            "total_success_rollouts": total_success,
            "total_rollouts": total_rollouts,
            "overall_success_rate": overall_rate,
        },
    }

    output_file = os.path.join(result_dir, "merged_results.json")
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")
    return merged_data


if __name__ == "__main__":
    merge_results(tyro.cli(Args))
