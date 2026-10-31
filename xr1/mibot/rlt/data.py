"""Read the USB manifest using the pretrained XR1 coordinate and norm contract."""

import copy
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import (
    compose_state,
    get_value,
    normalize_quantile,
    resize_image,
    validate_quantiles,
    validate_stats,
)

CAMERAS = ("headDepthCamera", "leftCamera", "rightCamera")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12) or not np.isfinite(q).all():
        raise ValueError("USB poses contain invalid XYZW quaternions")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return (
        np.stack(
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
            -1,
        )
        .reshape(q.shape[:-1] + (3, 3))
        .astype(np.float32)
    )


def trajectory_arrays(state, action):
    result = {}
    for section, values in (("proprios", state), ("actions", action)):
        result[section] = {}
        for side, offset in (("left", 0), ("right", 15)):
            fields = result[section]
            fields[f"{side}_ee_pos"] = values[:, offset + 7 : offset + 10]
            fields[f"{side}_ee_rotm"] = quaternion_matrix(
                values[:, offset + 10 : offset + 14]
            ).reshape(-1, 9)
            fields[f"{side}_gripper_pos"] = values[:, offset + 14 : offset + 15]
            if section == "proprios":
                fields[f"{side}_arm_joint"] = np.deg2rad(
                    values[:, offset : offset + 7]
                ).astype(np.float32)
        result[section]["waist_pos"] = np.zeros((len(values), 1), dtype=np.float32)
    result["actions"]["base_vel"] = np.zeros((len(action), 3), dtype=np.float32)
    return result


class USBDataset(JsonDataset):
    """No second time shift, no norm recomputation, and no episode-crossing windows."""

    def __init__(
        self, root, normalization, *, stride=1, augment=False, episode_ids=None
    ):
        self.root = Path(root)
        self._episode_cache = OrderedDict()
        self.normalization = str(Path(normalization).resolve())
        self.records = [
            json.loads(line)
            for line in (self.root / "meta/manifest.jsonl").read_text().splitlines()
            if line
        ]
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if len(self.records) != self.info["total_episodes"]:
            raise ValueError("USB manifest episode count mismatch")
        if sum(r["length"] for r in self.records) != self.info["total_frames"]:
            raise ValueError("USB manifest frame count mismatch")
        if any(r.get("action_shift_frames") != 3 for r in self.records):
            raise ValueError(
                "Expected supplied USB dataset with actions already shifted by 3 frames"
            )
        self.action_length, self.batch_size = 30, 1
        self.active_action_dims = list(range(7))
        self.augment = augment
        self.repeat_to_max_steps = False
        self.sample_stride = int(stride)
        if self.sample_stride < 1:
            raise ValueError("stride must be positive")
        stats = json.loads(Path(normalization).read_text())
        self.mean, self.std = validate_stats(stats["mean"], stats["std"], 30)
        self.q01, self.q99 = validate_quantiles(stats["q01"], stats["q99"])
        self.record_by_path = {str(self.root / r["data_path"]): r for r in self.records}
        selected = (
            set(episode_ids)
            if episode_ids is not None
            else {r["episode_index"] for r in self.records}
        )
        if not selected.issubset({r["episode_index"] for r in self.records}):
            raise ValueError("Unknown episode ID")
        self.files = [
            p for p, r in self.record_by_path.items() if r["episode_index"] in selected
        ]
        self.samples = [
            {"file": p, "frame_index": i}
            for p in self.files
            for i in range(0, self.record_by_path[p]["length"], self.sample_stride)
        ]
        if not self.samples:
            raise ValueError("No USB samples selected")

    def _read_json(self, path):
        if path in self._episode_cache:
            self._episode_cache.move_to_end(path)
            return self._episode_cache[path]
        record = self.record_by_path[path]
        table = pq.read_table(
            path, columns=["observation.state", "action", "frame_index"]
        )
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        n = record["length"]
        if state.shape != (n, 30) or action.shape != (n, 30):
            raise ValueError(f"USB numeric shape mismatch in {path}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Nonfinite USB numbers in {path}")
        if not np.array_equal(np.asarray(table["frame_index"]), np.arange(n)):
            raise ValueError(f"Unexpected frame order in {path}")
        trajectory = trajectory_arrays(state, action)
        trajectory.update(
            time=record["episode_name"], num_frames=n, trajectory_type="success"
        )
        trajectory["observations"] = {
            view: [{"path": str(self.root / record["video_paths"][camera]), "start": 0}]
            for view, camera in zip(("ego", "wrist_left", "wrist_right"), CAMERAS)
        }
        text = (
            "The following observations are captured from multiple views.\n"
            "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
            "Generate robot actions for the task:\n" + record["prompt"]
        )
        trajectory["instruction"] = {
            "general": [
                {
                    "images": [
                        "observations.ego",
                        "observations.wrist_left",
                        "observations.wrist_right",
                    ],
                    "conversations": [
                        {"from": "human", "value": text},
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        }
        self._episode_cache[path] = trajectory
        if len(self._episode_cache) > 8:
            self._episode_cache.popitem(last=False)
        return trajectory

    @staticmethod
    def _images(traj, keys, frame):
        # This dataset uses HEVC; use the decoder configured below directly.
        from PIL import Image

        images = []
        for key in keys:
            infos = get_value(traj, key)
            if infos is None:
                raise ValueError(f"Missing video key: {key}")
            for info in infos:
                array = USBDataset._video_frame(
                    info["path"], frame + int(info.get("start", 0))
                )
                cropped = JsonDataset._crop_bbox(array[None], info.get("crop_bbox"))
                images.extend(Image.fromarray(x) for x in cropped)
        return images

    @staticmethod
    def _video_frame(path, index):
        import av

        cache = os.environ.get("XR1_USB_IMAGE_CACHE")
        if cache:
            cached = Path(cache) / (Path(path).stem + ".npy")
            if cached.exists() and cached.with_suffix(".json").exists():
                frames = np.load(cached, mmap_mode="r")
                return np.array(frames[index], copy=True)

        # FFmpeg decoder threads can wait for a Python logging callback while
        # codec destruction holds the GIL. Disable that callback and frame threading.
        av.logging.set_level(None)
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            stream.thread_type = "SLICE"
            rate = float(stream.average_rate or stream.guessed_rate)
            target = int(stream.start_time or 0) + round(
                index / rate / float(stream.time_base)
            )
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for decoded in container.decode(stream):
                if decoded.pts is not None and decoded.pts >= target:
                    if decoded.pts != target:
                        raise ValueError(
                            f"Missing exact USB video frame {path}:{index}"
                        )
                    return decoded.to_ndarray(format="rgb24")
        raise IndexError(f"Video has no frame {path}:{index}")

    def item_at(self, episode, frame):
        record = next(r for r in self.records if r["episode_index"] == episode)
        if not 0 <= frame < record["length"]:
            raise IndexError("Frame outside episode")
        # Use the inherited conversion without mutating this dataset's sample index.
        view = copy.copy(self)
        view.samples = [
            {"file": str(self.root / record["data_path"]), "frame_index": frame}
        ]
        return view[0]


class ObservationCollator:
    def __init__(self, vlm_path):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)

    def __call__(self, items):
        # The final two SFT turns contain state/action/score targets. Exclude them
        # completely: Stage 2 must use only information available at rollout.
        prompts = [item["messages"][:-2] for item in items]
        return self._process(prompts, [item["state"] for item in items])

    def _process(self, prompts, states):
        obs = dict(
            self.processor.apply_chat_template(
                prompts,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
                do_resize=False,
                add_generation_prompt=False,
            )
        )
        obs["state"] = torch.stack(states)
        return obs

    def from_raw(self, images, raw_state, normalization, prompt):
        """Build a single live observation without any action/score labels.

        images is a dict of RGB arrays/PIL images keyed by CAMERAS; raw_state
        follows the supplied USB 30D schema (joint angles in degrees).
        """
        from PIL import Image

        raw = np.asarray(raw_state, dtype=np.float32)
        if raw.shape != (30,) or not np.isfinite(raw).all():
            raise ValueError("Expected finite raw USB state [30]")
        state = compose_state(
            raw[14:15], np.deg2rad(raw[:7]), raw[29:30], np.deg2rad(raw[15:22])
        )
        q01, q99 = validate_quantiles(normalization["q01"], normalization["q99"])
        state = torch.from_numpy(normalize_quantile(state, q01, q99))
        views = []
        for key in CAMERAS:
            view = images[key]
            if not isinstance(view, Image.Image):
                view = Image.fromarray(np.asarray(view))
            views.append(
                resize_image(view.convert("RGB"), factor=32, max_pixels=160000)
            )
        text = (
            "The following observations are captured from multiple views.\n"
            "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
            "Generate robot actions for the task:\n" + prompt + " /no_cot"
        )
        messages = JsonDataset._messages(
            [{"from": "human", "value": text}, {"from": "gpt", "value": "<cot></cot>"}],
            views,
        )
        return self._process([messages], [state])
