#!/usr/bin/env python3
"""Convert the Data_0904_clean network datasets from LeRobot parquet to XR-1 JSON.

Only action[0:30] is used. In particular, action[28:30] contains the original
gripper targets. Legacy 32-D datasets are also accepted; their appended smoothed
gripper commands at action[30:32] are discarded completely.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ACTION_LENGTH = 30
ACTION_DIM = 60
REQUIRED_COLUMNS = (
    "observation.state.left_arm_joint",
    "observation.state.left_arm_cart",
    "observation.state.right_arm_joint",
    "observation.state.right_arm_cart",
    "observation.state.left_gripper_pos",
    "observation.state.right_gripper_pos",
    "action",
    "frame_index",
    "episode_index",
)
CAMERAS = {
    "ego": "observation.images.cam_high",
    "wrist_left": "observation.images.cam_left_wrist",
    "wrist_right": "observation.images.cam_right_wrist",
}


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("encountered a zero-length end-effector quaternion")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def relative_axis_angle(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return axis-angle for inv(current_xyzw) * target_xyzw."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    current /= np.linalg.norm(current, axis=-1, keepdims=True)
    target /= np.linalg.norm(target, axis=-1, keepdims=True)

    ax, ay, az, aw = np.moveaxis(current, -1, 0)
    bx, by, bz, bw = np.moveaxis(target, -1, 0)
    # Hamilton product conjugate(current) * target.
    xyz = np.stack(
        (
            aw * bx - ax * bw - ay * bz + az * by,
            aw * by + ax * bz - ay * bw - az * bx,
            aw * bz - ax * by + ay * bx - az * bw,
        ),
        axis=-1,
    )
    w = aw * bw + ax * bx + ay * by + az * bz
    # q and -q encode the same rotation; choose the shortest rotation.
    sign = np.where(w < 0.0, -1.0, 1.0)
    xyz *= sign[..., None]
    w *= sign
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(w, 0.0, 1.0))
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1e-12)
    return xyz * scale[..., None]


def packed_state(data: dict[str, np.ndarray], joints_in_degrees: bool) -> np.ndarray:
    count = len(data["frame_index"])
    state = np.zeros((count, ACTION_DIM), dtype=np.float32)
    joint_scale = math.pi / 180.0 if joints_in_degrees else 1.0
    state[:, 0:7] = data["observation.state.left_arm_joint"] * joint_scale
    state[:, 7:8] = data["observation.state.left_gripper_pos"]
    state[:, 8:15] = data["observation.state.right_arm_joint"] * joint_scale
    state[:, 15:16] = data["observation.state.right_gripper_pos"]
    return state


def iter_packed_action_steps(data: dict[str, np.ndarray], length: int):
    """Yield each temporal offset over all complete action windows."""
    count = len(data["frame_index"])
    starts = count - length + 1
    if starts <= 0:
        return

    # Drop the appended command dimensions before any training transformation.
    action = data["action"][:, :30].astype(np.float64, copy=False)
    left_cart = data["observation.state.left_arm_cart"].astype(np.float64, copy=False)
    right_cart = data["observation.state.right_arm_cart"].astype(np.float64, copy=False)
    left_rotation_t = np.swapaxes(quaternion_xyzw_to_matrix(left_cart[:starts, 3:7]), 1, 2)
    right_rotation_t = np.swapaxes(quaternion_xyzw_to_matrix(right_cart[:starts, 3:7]), 1, 2)
    left_gripper = data["observation.state.left_gripper_pos"][:starts]
    right_gripper = data["observation.state.right_gripper_pos"][:starts]

    for offset in range(length):
        future = action[offset : offset + starts]
        packed = np.zeros((starts, ACTION_DIM), dtype=np.float64)
        packed[:, 0:3] = np.einsum(
            "nij,nj->ni", left_rotation_t, future[:, 7:10] - left_cart[:starts, 0:3]
        )
        packed[:, 3:6] = relative_axis_angle(left_cart[:starts, 3:7], future[:, 10:14])
        packed[:, 6:7] = future[:, 28:29] - left_gripper
        packed[:, 8:11] = np.einsum(
            "nij,nj->ni", right_rotation_t, future[:, 21:24] - right_cart[:starts, 0:3]
        )
        packed[:, 11:14] = relative_axis_angle(right_cart[:starts, 3:7], future[:, 24:28])
        packed[:, 14:15] = future[:, 29:30] - right_gripper
        yield offset, packed


class Statistics:
    def __init__(self, length: int):
        self.length = length
        self.count = np.zeros(length, dtype=np.int64)
        self.total = np.zeros((length, ACTION_DIM), dtype=np.float64)
        self.square_total = np.zeros((length, ACTION_DIM), dtype=np.float64)
        self.states: list[np.ndarray] = []

    def add(self, data: dict[str, np.ndarray], joints_in_degrees: bool) -> None:
        self.states.append(packed_state(data, joints_in_degrees))
        for offset, values in iter_packed_action_steps(data, self.length):
            self.count[offset] += len(values)
            self.total[offset] += values.sum(axis=0)
            self.square_total[offset] += np.square(values).sum(axis=0)

    def result(self) -> dict[str, list]:
        if np.any(self.count == 0):
            raise ValueError(f"no complete {self.length}-frame action windows were found")
        mean = self.total / self.count[:, None]
        variance = self.square_total / self.count[:, None] - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        states = np.concatenate(self.states, axis=0)
        return {
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "q01": np.quantile(states, 0.01, axis=0, keepdims=True).astype(np.float32).tolist(),
            "q99": np.quantile(states, 0.99, axis=0, keepdims=True).astype(np.float32).tolist(),
        }


def array_column(batch, name: str, width: int | None = None) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    values = column.to_numpy(zero_copy_only=False) if width is None else column.to_pylist()
    array = np.asarray(values)
    if width is not None:
        array = array.reshape(len(column), width)
    return array


def read_episode_chunks(parquet_files: list[Path], source_action_dim: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; run `pip install -e .` from xr1/") from exc

    widths = {
        "observation.state.left_arm_joint": 7,
        "observation.state.left_arm_cart": 7,
        "observation.state.right_arm_joint": 7,
        "observation.state.right_arm_cart": 7,
        "observation.state.left_gripper_pos": 1,
        "observation.state.right_gripper_pos": 1,
        "action": source_action_dim,
    }
    active_episode = None
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in REQUIRED_COLUMNS if key != "episode_index"}

    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=8192, columns=list(REQUIRED_COLUMNS)):
            episode_ids = array_column(batch, "episode_index").astype(np.int64)
            columns = {
                key: array_column(batch, key, widths.get(key))
                for key in chunks
            }
            boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
            for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(episode_ids)]):
                episode = int(episode_ids[start])
                if active_episode is not None and episode != active_episode:
                    yield active_episode, {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
                    chunks = {key: [] for key in chunks}
                active_episode = episode
                for key, values in columns.items():
                    chunks[key].append(values[start:end])

    if active_episode is not None:
        yield active_episode, {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def trajectory_json(
    episode: int,
    data: dict[str, np.ndarray],
    dataset_dir: Path,
    prompt: str,
    joints_in_degrees: bool,
) -> dict:
    frames = data["frame_index"].astype(np.int64)
    expected = np.arange(len(frames), dtype=np.int64)
    if not np.array_equal(frames, expected):
        raise ValueError(f"episode {episode} has non-contiguous frame_index values")

    # Keep the canonical 30-D action; this also drops action[30:32] for legacy data.
    action = data["action"][:, :30].astype(np.float64, copy=False)
    left_cart = data["observation.state.left_arm_cart"].astype(np.float64, copy=False)
    right_cart = data["observation.state.right_arm_cart"].astype(np.float64, copy=False)
    joint_scale = math.pi / 180.0 if joints_in_degrees else 1.0
    video_number = episode + 1
    observations = {}
    for output_key, source_key in CAMERAS.items():
        path = (dataset_dir / "videos" / source_key / f"episode_{video_number:04d}.mp4").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing video for episode {episode}: {path}")
        observations[output_key] = [{"path": str(path), "start": 0, "crop_bbox": None}]

    zeros1 = np.zeros((len(frames), 1), dtype=np.float32).tolist()
    return {
        "trajectory_type": "success",
        "time": f"network_episode_{episode:04d}",
        "num_frames": len(frames),
        "instruction": {
            "general": [
                {
                    "images": ["observations.ego", "observations.wrist_left", "observations.wrist_right"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "The following observations are captured from multiple views.\n"
                                "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
                                "# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n" + prompt
                            ),
                        },
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        },
        "observations": observations,
        "proprios": {
            "left_ee_pos": left_cart[:, 0:3].tolist(),
            "left_ee_rotm": quaternion_xyzw_to_matrix(left_cart[:, 3:7]).reshape(-1, 9).tolist(),
            "left_arm_joint": (data["observation.state.left_arm_joint"] * joint_scale).tolist(),
            "left_gripper_pos": data["observation.state.left_gripper_pos"].tolist(),
            "right_ee_pos": right_cart[:, 0:3].tolist(),
            "right_ee_rotm": quaternion_xyzw_to_matrix(right_cart[:, 3:7]).reshape(-1, 9).tolist(),
            "right_arm_joint": (data["observation.state.right_arm_joint"] * joint_scale).tolist(),
            "right_gripper_pos": data["observation.state.right_gripper_pos"].tolist(),
            "waist_pos": zeros1,
        },
        "actions": {
            "left_ee_pos": action[:, 7:10].tolist(),
            "left_ee_rotm": quaternion_xyzw_to_matrix(action[:, 10:14]).reshape(-1, 9).tolist(),
            # Use the original gripper targets. Legacy action[30:32] is discarded.
            "left_gripper_pos": action[:, 28:29].tolist(),
            "right_ee_pos": action[:, 21:24].tolist(),
            "right_ee_rotm": quaternion_xyzw_to_matrix(action[:, 24:28]).reshape(-1, 9).tolist(),
            "right_gripper_pos": action[:, 29:30].tolist(),
            "waist_pos": zeros1,
            "base_vel": np.zeros((len(frames), 3), dtype=np.float32).tolist(),
        },
    }


def load_metadata(dataset_dir: Path) -> tuple[dict, str]:
    with (dataset_dir / "meta" / "info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    prompt_path = dataset_dir / "meta" / "prompt.json"
    with prompt_path.open(encoding="utf-8") as file:
        prompt = json.load(file)["prompt"]
    action = info.get("features", {}).get("action", {})
    shape = action.get("shape")
    if shape not in ([30], [32]):
        raise ValueError(f"expected a 30-D or legacy 32-D action, got {shape}")
    names = action.get("names") or []
    if shape == [32] and (
        len(names) != 32 or names[30:32] != ["left_gripper_cmd", "right_gripper_cmd"]
    ):
        raise ValueError("action[30:32] must be left_gripper_cmd/right_gripper_cmd")
    return info, prompt


def prepare(args) -> None:
    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    episodes_dir = output_dir / "episodes"
    normalization_path = output_dir / "normalize.json"
    marker_path = output_dir / "prepared.json"
    info, prompt = load_metadata(dataset_dir)
    source_action_dim = int(info["features"]["action"]["shape"][0])
    marker = {
        "format_version": 2,
        "source": str(dataset_dir),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "action_length": args.action_length,
        "source_action_dim": source_action_dim,
        "joints_in_degrees": not args.joints_in_radians,
        "gripper_targets": [28, 29],
        "discarded_action_dimensions": list(range(30, source_action_dim)),
    }
    if marker_path.is_file() and not args.force:
        with marker_path.open(encoding="utf-8") as file:
            old_marker = json.load(file)
        if old_marker == marker and normalization_path.is_file():
            count = sum(1 for _ in episodes_dir.glob("episode_*.json"))
            if count == marker["total_episodes"]:
                print(f"Prepared data is current: {output_dir}")
                return
        raise RuntimeError(f"incomplete or stale output at {output_dir}; rerun with --force")

    if args.force and episodes_dir.is_dir():
        for path in episodes_dir.glob("episode_*.json"):
            path.unlink()
    episodes_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found below {dataset_dir / 'data'}")

    statistics = Statistics(args.action_length)
    episode_count = frame_count = 0
    expected_episode = 0
    for episode, data in read_episode_chunks(parquet_files, source_action_dim):
        if episode != expected_episode:
            raise ValueError(f"expected episode_index {expected_episode}, got {episode}")
        trajectory = trajectory_json(
            episode, data, dataset_dir, args.prompt or prompt, not args.joints_in_radians
        )
        output_path = episodes_dir / f"episode_{episode:04d}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(trajectory, file, separators=(",", ":"), allow_nan=False)
        statistics.add(data, not args.joints_in_radians)
        episode_count += 1
        frame_count += len(data["frame_index"])
        expected_episode += 1
        if episode_count % 50 == 0:
            print(f"Prepared {episode_count}/{info['total_episodes']} episodes", flush=True)

    if episode_count != marker["total_episodes"] or frame_count != marker["total_frames"]:
        raise ValueError(
            f"source counts disagree: got {episode_count} episodes/{frame_count} frames, "
            f"expected {marker['total_episodes']}/{marker['total_frames']}"
        )
    with normalization_path.open("w", encoding="utf-8") as file:
        json.dump(statistics.result(), file, separators=(",", ":"), allow_nan=False)
    with marker_path.open("w", encoding="utf-8") as file:
        json.dump(marker, file, indent=2)
    print(f"Prepared {episode_count} episodes and {frame_count} frames in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Data_0904_clean network dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="output directory for XR-1 JSON")
    parser.add_argument("--action-length", type=int, default=ACTION_LENGTH)
    parser.add_argument("--prompt", help="override meta/prompt.json")
    parser.add_argument(
        "--joints-in-radians",
        action="store_true",
        help="do not convert joint state from degrees to radians",
    )
    parser.add_argument("--force", action="store_true", help="replace generated episode JSON files")
    args = parser.parse_args()
    if args.action_length <= 0:
        parser.error("--action-length must be positive")
    return args


if __name__ == "__main__":
    try:
        prepare(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
