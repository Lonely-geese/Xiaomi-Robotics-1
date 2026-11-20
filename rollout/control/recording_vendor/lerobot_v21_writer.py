"""Append-only LeRobot v2.1 writer for XR1 USB rollouts (RLinf STEAM / RECAP "rollout" datasets).

Layout and the shared columns match the deeptouch recorder's teleop sessions (the pi05 USB SFT data):
30-dim ``observation.state`` / ``action``, three 480x848 h264 videos named ``observation.images.
{headDepthCamera,leftCamera,rightCamera}`` (left wrist NOT cropped; training crops it), fps 30, and
``action[t] = state[min(t + action_shift, T - 1)]`` (the SFT conversion uses shift 3).

Extra per-frame columns read by RLinf:
    is_success      bool  episode outcome on every frame (PairDataset scans one frame per episode)
    intervene_flag  bool  frame recorded while the operator had drag-teach control
    done            bool  True on the last frame
    reward          f32   1.0 on the last frame of a successful episode, else 0
    source          i64   0 TRANSITION, 1 POLICY, 2 HUMAN (same codes as the RECAP conversion)
    observation_sequence / hal_time_ns   HAL provenance of each kept frame

No robot or lerobot import: the heavy part (video encoding) runs in a worker process.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FPS = 30
CHUNKS_SIZE = 1000
STATE_DIM = 30
IMAGE_SHAPE = (480, 848, 3)
# observation["image"] key (scripts.deeptouch.policy_client) -> LeRobot video key of the SFT sessions.
VIDEO_KEYS = {
    "base_0_rgb": "observation.images.headDepthCamera",
    "left_wrist_0_rgb": "observation.images.leftCamera",
    "right_wrist_0_rgb": "observation.images.rightCamera",
}
SOURCE_TRANSITION, SOURCE_POLICY, SOURCE_HUMAN = 0, 1, 2
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

_SIDES = ("leftarm", "rightarm")
STATE_NAMES = [f"{side}_{name}" for side in _SIDES for name in
               [f"state_joint_pos_{i}" for i in range(7)] + [f"state_cart_pos_{i}" for i in range(7)]
               + ["gripper_state_pos"]]
# The rollout "action" is the shifted measured state (same as SFT), not the commanded pose.
ACTION_NAMES = [f"shifted_{name}" for name in STATE_NAMES]


@dataclass
class EpisodeRecord:
    """One finished episode, everything the worker process needs (picklable)."""

    task: str
    states: np.ndarray                 # [T, 30] float32
    jpeg_frames: dict[str, list[bytes]]  # VIDEO_KEYS value -> T JPEG-encoded RGB frames
    sources: np.ndarray                # [T] int64
    observation_sequence: np.ndarray   # [T] int64
    hal_time_ns: np.ndarray            # [T] int64
    success: bool
    extra: dict[str, Any] = field(default_factory=dict)  # stored in meta/episodes.jsonl

    def __len__(self) -> int:
        return int(self.states.shape[0])


def _features(video_codec: str) -> dict[str, Any]:
    def scalar(dtype: str) -> dict[str, Any]:
        return {"dtype": dtype, "shape": [1], "names": None}

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": ACTION_NAMES},
        "timestamp": scalar("float32"),
        "frame_index": scalar("int64"),
        "episode_index": scalar("int64"),
        "index": scalar("int64"),
        "task_index": scalar("int64"),
        "is_success": scalar("bool"),
        "intervene_flag": scalar("bool"),
        "done": scalar("bool"),
        "reward": scalar("float32"),
        "source": scalar("int64"),
        "observation_sequence": scalar("int64"),
        "hal_time_ns": scalar("int64"),
    }
    for key in VIDEO_KEYS.values():
        features[key] = {
            "dtype": "video", "shape": list(IMAGE_SHAPE), "names": ["height", "width", "channels"],
            "info": {"video.height": IMAGE_SHAPE[0], "video.width": IMAGE_SHAPE[1], "video.channels": 3,
                     "video.fps": FPS, "video.codec": video_codec, "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    return features


def shifted_actions(states: np.ndarray, shift: int) -> np.ndarray:
    """action[t] = state[min(t + shift, T - 1)], exactly the SFT target alignment."""
    index = np.minimum(np.arange(len(states)) + int(shift), len(states) - 1)
    return np.asarray(states, dtype=np.float32)[index]


def _numeric_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    as_float = values.astype(np.float64)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": as_float.mean(axis=0).tolist(), "std": as_float.std(axis=0).tolist(),
            "count": [int(values.shape[0])]}


def _image_stats(frames: list[np.ndarray]) -> dict[str, list]:
    """LeRobot v2.1 image stats: per channel over sampled frames, in [0, 1], shape (3, 1, 1)."""
    stacked = np.stack(frames).astype(np.float64) / 255.0  # [N, H, W, 3]
    per_channel = stacked.reshape(-1, 3)
    shape = lambda v: np.asarray(v).reshape(3, 1, 1).tolist()  # noqa: E731
    return {"min": shape(per_channel.min(0)), "max": shape(per_channel.max(0)),
            "mean": shape(per_channel.mean(0)), "std": shape(per_channel.std(0)),
            "count": [len(frames)]}


def _encode_video(jpegs: list[bytes], path: Path, *, codec: str, crf: int, gop: int,
                  stats_every: int) -> list[np.ndarray]:
    """JPEG frames -> mp4 at FPS. Returns a frame subsample for the episode stats."""
    import av
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.mp4")
    sampled: list[np.ndarray] = []
    with av.open(str(tmp), mode="w") as container:
        stream = container.add_stream(codec, rate=FPS)
        stream.width, stream.height = IMAGE_SHAPE[1], IMAGE_SHAPE[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "g": str(gop), "preset": "veryfast"}
        for index, data in enumerate(jpegs):
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != IMAGE_SHAPE:
                raise ValueError(f"bad frame {index} for {path.name}: {None if bgr is None else bgr.shape}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index % stats_every == 0:
                sampled.append(rgb[::8, ::8])
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(tmp, path)
    return sampled


def write_episode_files(root: str, episode_index: int, global_start: int, task_index: int,
                        record: EpisodeRecord, action_shift: int, video_codec: str, crf: int,
                        gop: int) -> dict[str, Any]:
    """Worker-process entry: write parquet + videos of one episode, return its stats line."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root_path = Path(root)
    length = len(record)
    chunk = episode_index // CHUNKS_SIZE
    states = np.asarray(record.states, dtype=np.float32)
    actions = shifted_actions(states, action_shift)
    frame_index = np.arange(length, dtype=np.int64)
    columns = {
        "observation.state": states,
        "action": actions,
        "timestamp": (frame_index / FPS).astype(np.float32),
        "frame_index": frame_index,
        "episode_index": np.full(length, episode_index, np.int64),
        "index": global_start + frame_index,
        "task_index": np.full(length, task_index, np.int64),
        "is_success": np.full(length, bool(record.success)),
        "intervene_flag": np.asarray(record.sources) == SOURCE_HUMAN,
        "done": frame_index == length - 1,
        "reward": ((frame_index == length - 1) & bool(record.success)).astype(np.float32),
        "source": np.asarray(record.sources, dtype=np.int64),
        "observation_sequence": np.asarray(record.observation_sequence, dtype=np.int64),
        "hal_time_ns": np.asarray(record.hal_time_ns, dtype=np.int64),
    }
    stats_every = max(1, length // 100)
    stats: dict[str, Any] = {}
    for key, frames in record.jpeg_frames.items():
        if len(frames) != length:
            raise ValueError(f"{key}: {len(frames)} frames for {length} states")
        path = root_path / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=episode_index)
        stats[key] = _image_stats(_encode_video(frames, path, codec=video_codec, crf=crf, gop=gop,
                                                stats_every=stats_every))
    arrays = {}
    for key, value in columns.items():
        if value.ndim == 2:
            arrays[key] = pa.array(list(value), type=pa.list_(pa.float32()))
        else:
            arrays[key] = pa.array(value)
        stats[key] = _numeric_stats(value)
    data_path = root_path / DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_name(data_path.name + ".tmp")
    pq.write_table(pa.table(arrays), tmp)
    os.replace(tmp, data_path)
    return {"episode_index": episode_index, "stats": stats}


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LeRobotV21RolloutWriter:
    """Appends episodes to ``root`` (created if missing, resumed if it exists).

    ``submit`` returns immediately; parquet/video encoding runs in one worker process and the meta
    files are updated in submission order once each episode is on disk. ``close`` waits for all.
    """

    def __init__(self, root: str | Path, *, robot_type: str = "deeptouch_dualarm", action_shift: int = 3,
                 video_codec: str = "libx264", crf: int = 18, gop: int = 10, workers: int = 1):
        self.root = Path(root).resolve()
        self.action_shift = int(action_shift)
        self.video_codec, self.crf, self.gop = video_codec, int(crf), int(gop)
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        info_path = meta / "info.json"
        self._episodes = _read_jsonl(meta / "episodes.jsonl")
        self._tasks = {row["task"]: int(row["task_index"]) for row in _read_jsonl(meta / "tasks.jsonl")}
        if info_path.exists():
            self.info = json.loads(info_path.read_text())
            if self.info.get("codebase_version") != "v2.1" or self.info.get("fps") != FPS:
                raise ValueError(f"{self.root} is not a {FPS} fps LeRobot v2.1 dataset")
            if set(self.info["features"]) != set(_features(video_codec)):
                raise ValueError(f"{self.root} has a different feature schema; use a new output directory")
            if self.info.get("rollout_action_shift", self.action_shift) != self.action_shift:
                raise ValueError(f"{self.root} was written with a different action shift")
        else:
            self.info = {
                "codebase_version": "v2.1", "robot_type": robot_type, "total_episodes": 0, "total_frames": 0,
                "total_tasks": 0, "total_videos": 0, "total_chunks": 0, "chunks_size": CHUNKS_SIZE, "fps": FPS,
                "splits": {"train": "0:0"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
                "features": _features(video_codec), "rollout_action_shift": self.action_shift,
            }
        if len(self._episodes) != self.info["total_episodes"]:
            raise ValueError(f"{self.root}: meta/episodes.jsonl has {len(self._episodes)} rows but info.json "
                             f"total_episodes={self.info['total_episodes']}; repair before appending")
        self._next_episode = len(self._episodes)
        self._next_index = int(self.info["total_frames"])
        orphans = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))[self._next_episode:]
        if orphans:
            raise ValueError(f"{self.root}: {len(orphans)} episode file(s) on disk are not in meta (e.g. {orphans[0].name}); "
                             f"register them first: python steam_rollout/repair_meta.py {self.root}")
        self._lock = threading.RLock()
        self._pending: list[tuple[Future, dict[str, Any], int]] = []
        # spawn: the parent holds HAL sockets / threads; a forked child must not inherit them.
        self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        logger.info("LeRobot rollout dataset %s: %d episodes / %d frames so far", self.root,
                    self._next_episode, self._next_index)

    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
            lines = [json.dumps({"task_index": index, "task": name}, ensure_ascii=False)
                     for name, index in sorted(self._tasks.items(), key=lambda item: item[1])]
            _atomic_write_text(self.root / "meta" / "tasks.jsonl", "\n".join(lines) + "\n")
        return self._tasks[task]

    def submit(self, record: EpisodeRecord) -> int:
        """Queue one episode; returns its episode_index."""
        if len(record) < 2:
            raise ValueError("episode needs at least 2 frames")
        episode_index, start = self._next_episode, self._next_index
        self._next_episode += 1
        self._next_index += len(record)
        task_index = self._task_index(record.task)
        future = self._pool.submit(write_episode_files, str(self.root), episode_index, start, task_index, record,
                                   self.action_shift, self.video_codec, self.crf, self.gop)
        human = int(np.sum(np.asarray(record.sources) == SOURCE_HUMAN))
        row = {"episode_index": episode_index, "tasks": [record.task], "length": len(record),
               "success": bool(record.success), "outcome": "success" if record.success else "failure",
               "human_intervened": human > 0, "human_frames": human, **record.extra}
        with self._lock:
            self._pending.append((future, row, episode_index))
        # Commit to meta as soon as this episode is on disk (not only at the next submit / close).
        future.add_done_callback(self._on_done)
        return episode_index

    def _on_done(self, _future: Future) -> None:
        try:
            self.flush(block=False)
        except Exception:
            pass  # logged in flush; close() raises it

    def flush(self, *, block: bool) -> None:
        """Commit finished episodes to meta, in order."""
        with self._lock:
            self._flush_locked(block)

    def _flush_locked(self, block: bool) -> None:
        while self._pending and (block or self._pending[0][0].done()):
            future, row, episode_index = self._pending[0]
            try:
                stats_line = future.result()
            except Exception:
                # Later episodes already have indices; stop rather than leave a hole in the dataset.
                logger.exception("writing episode %s failed; dataset meta left at %s episodes",
                                 episode_index, len(self._episodes))
                raise
            self._pending.pop(0)
            self._commit(row, stats_line)

    def _commit(self, row: dict[str, Any], stats_line: dict[str, Any]) -> None:
        meta = self.root / "meta"
        with (meta / "episodes_stats.jsonl").open("a") as f:
            f.write(json.dumps(stats_line) + "\n")
        with (meta / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._episodes.append(row)
        info = self.info
        info["total_episodes"] = len(self._episodes)
        info["total_frames"] = int(sum(int(r["length"]) for r in self._episodes))
        info["total_tasks"] = len(self._tasks)
        info["total_videos"] = info["total_episodes"] * len(VIDEO_KEYS)
        info["total_chunks"] = (info["total_episodes"] - 1) // CHUNKS_SIZE + 1
        info["splits"] = {"train": f"0:{info['total_episodes']}"}
        _atomic_write_text(meta / "info.json", json.dumps(info, indent=2, ensure_ascii=False) + "\n")
        logger.warning("Saved rollout episode %s (%s, %s frames, %s human) -> %s", row["episode_index"],
                       row["outcome"], row["length"], row["human_frames"], self.root)

    def summary(self) -> str:
        with self._lock:
            wins = sum(bool(r["success"]) for r in self._episodes)
            return f"{len(self._episodes)} episodes saved ({wins} success), {len(self._pending)} still encoding"

    def close(self) -> None:
        try:
            self.flush(block=True)
        finally:
            self._pool.shutdown(wait=True)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """RGB uint8 -> JPEG bytes (cv2 releases the GIL while encoding)."""
    import cv2

    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


__all__ = ["EpisodeRecord", "LeRobotV21RolloutWriter", "VIDEO_KEYS", "SOURCE_HUMAN", "SOURCE_POLICY",
           "SOURCE_TRANSITION", "encode_jpeg", "shifted_actions"]
