"""XR1 trajectories adapted to RLinf's unchanged STEAM pair contract."""

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .upstream.data.pair_dataset import (
    BinaryPairInferenceDataset,
    PairDataset,
    TrajectorySource,
)

CAMERAS = ("ego", "wrist_left", "wrist_right")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["format"] != "xr1-steam-v1":
        raise ValueError("unsupported STEAM dataset format")
    return manifest


@lru_cache(maxsize=64)
def episode_json(path):
    return json.loads(Path(path).read_text())


class XR1TrajectorySource(TrajectorySource):
    base = None  # Disable the upstream optional LeRobot batched-video fast path.

    def __init__(self, records):
        self.records = records

    def num_episodes(self):
        return len(self.records)

    def episode_length(self, episode):
        return self.records[episode]["usable_frames"]

    def episode_is_success(self, episode):
        return self.records[episode]["end_status"] == "success"

    def get_prompt(self, episode, frame):
        return self.records[episode]["task"]

    def get_raw_sample(self, episode, frame):
        if not 0 <= frame < self.episode_length(episode):
            raise IndexError("frame outside censored episode")
        return {"episode": episode, "frame": frame}

    def get_view_from_sample(self, sample, camera_key):
        return self.get_view(sample["episode"], sample["frame"], camera_key)

    def get_view(self, episode, frame, camera_key):
        import av

        record = self.records[episode]
        if not 0 <= frame < record["usable_frames"]:
            raise IndexError("frame outside censored episode")
        info = episode_json(record["path"])["observations"][camera_key][0]
        index = frame + int(info.get("start", 0))
        with av.open(info["path"]) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            rate = float(stream.average_rate or stream.guessed_rate)
            target = int(stream.start_time or 0) + round(
                index / rate / float(stream.time_base)
            )
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for decoded in container.decode(stream):
                if decoded.pts is not None and decoded.pts >= target:
                    image = decoded.to_ndarray(format="rgb24")
                    if info.get("crop_bbox") is not None:
                        top, left, height, width = info["crop_bbox"]
                        image = image[top : top + height, left : left + width]
                    return image
        raise IndexError(f"video has no requested frame: {info['path']}:{index}")


def pair_dataset(records, config, *, inference=False, seed=0):
    source = XR1TrajectorySource(records)
    common = {
        "dataset_path": "xr1-json",
        "source": source,
        "camera_keys": list(CAMERAS),
        "k": config.get("k", 32),
        "dataset_type": records[0]["type"],
    }
    if inference:
        return BinaryPairInferenceDataset(**common, prompt=None)
    dataset = PairDataset(
        **common,
        only_success=False,
        num_bins=config.get("num_bins", 32),
        length_scale_enabled=config.get("length_scale_enabled", False),
        length_scale_reference=config.get("length_scale_reference"),
    )
    dataset._rng = np.random.default_rng(seed)
    return dataset


class PolicyDataset(Dataset):
    """Normalized XR1 action chunks with no action/score tokens in the prompt."""

    def __init__(
        self,
        manifest_path,
        advantages_path,
        split="train",
        stride=1,
        *,
        allow_smoke=False,
    ):
        import pandas as pd

        from mibot.data.datasets.json_dataset import JsonDataset

        self.manifest = read_manifest(manifest_path)
        records = [r for r in self.manifest["episodes"] if r["split"] == split]
        labels = pd.read_parquet(advantages_path)
        sidecar = json.loads(Path(str(advantages_path) + ".json").read_text())
        if (
            sidecar["manifest_sha256"] != digest(manifest_path)
            or not sidecar["complete"]
        ):
            raise ValueError("advantages do not cover this exact manifest")
        if sidecar.get("smoke_only") and not allow_smoke:
            raise ValueError("smoke labels cannot be used for policy training")
        if sidecar.get("advantages_sha256") != digest(advantages_path):
            raise ValueError("advantage file hash differs from metadata")
        if (
            digest(self.manifest["normalization"])
            != self.manifest["normalization_sha256"]
        ):
            raise ValueError("normalization changed since data preparation")
        if labels.duplicated(["episode_id", "frame_index"]).any():
            raise ValueError("duplicate advantage keys")
        if not pd.api.types.is_bool_dtype(labels.advantage_boolean):
            raise ValueError("advantage_boolean must contain booleans")
        lookup = dict(
            zip(zip(labels.episode_id, labels.frame_index), labels.advantage_boolean)
        )

        class CensoredJsonDataset(JsonDataset):
            @staticmethod
            def _read_json(path):
                result = copy.deepcopy(JsonDataset._read_json(path))
                result["num_frames"] = lengths[path]
                return result

        lengths = {r["path"]: r["usable_frames"] for r in records}
        self.dataset = CensoredJsonDataset(
            {
                "train_datasets": {
                    "paths": list(lengths),
                    "normalization": self.manifest["normalization"],
                    "action_length": 30,
                    "active_action_dims": self.manifest["active_action_dims"],
                    "sample_stride": stride,
                    "repeat_to_max_steps": False,
                    "augment": False,
                }
            }
        )
        self.dataset.samples = []
        self.advantages = []
        self.types = []
        for record in records:
            for frame in range(0, record["usable_frames"], stride):
                # Unknown endings must have a fully observed chunk strictly inside the boundary.
                if (
                    record["end_status"] == "unknown"
                    and frame + 30 >= record["usable_frames"] - 1
                ):
                    continue
                key = (record["episode_id"], frame)
                if key not in lookup:
                    raise ValueError(f"missing advantage for {key}")
                self.dataset.samples.append(
                    {"file": record["path"], "frame_index": frame}
                )
                self.advantages.append(bool(lookup[key]))
                self.types.append(record["type"])
        if not self.advantages:
            raise ValueError("empty policy dataset")

    def __len__(self):
        return len(self.advantages)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["messages"] = item["messages"][:-2]
        item["advantage"] = self.advantages[index]
        return item


class PolicyCollator:
    def __init__(self, actor_path, unconditional_prob=0.1):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(actor_path)
        self.unconditional_prob = unconditional_prob

    def observations(self, items, conditional):
        messages = []
        for item, positive in zip(items, conditional):
            prompt = copy.deepcopy(item["messages"])
            if positive:
                # Match RLinf's exact positive CFG instruction; retain task, images and state.
                human = next(m for m in reversed(prompt) if m["role"] == "user")
                human["content"].append(
                    {"type": "text", "text": "\nAdvantage: positive"}
                )
            messages.append(prompt)
        observation = dict(
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
                do_resize=False,
                add_generation_prompt=False,
            )
        )
        observation["state"] = torch.stack([item["state"] for item in items])
        return observation

    def __call__(self, items):
        from .upstream.routing import compute_cfg_routing_masks

        advantage = torch.tensor([i["advantage"] for i in items], dtype=torch.bool)
        routing = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=True,
            unconditional_prob=self.unconditional_prob,
        )
        return {
            "observation": self.observations(
                items, routing["conditional_mask"].tolist()
            ),
            "action": torch.stack([i["action"] for i in items]),
            "action_mask": torch.stack([i["action_mask"] for i in items]),
            "conditional_fraction": routing["conditional_mask"].float().mean(),
        }
