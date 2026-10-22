"""XR-1 JSON episodes -> LWD transitions, with explicitly labelled outcomes."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .replay import chunk_return


class EpisodeTransitions(Dataset):
    def __init__(self, data_config, outcomes, n_chunks=1, gamma=0.9999):
        # Heavy video/vision imports are needed only for real data training.
        from mibot.data.datasets.json_dataset import JsonDataset

        class ReplayDataset(JsonDataset):
            def _mask(self, traj, steps):
                # Success/failure are rewards, not reasons to drop actions.
                mask = np.zeros(self.action_length, dtype=np.int32)
                mask[:steps] = 1
                return mask

        config = copy.deepcopy(data_config)
        config.update(repeat_to_max_steps=False, augment=False)
        self.dataset = ReplayDataset({"train_datasets": config})
        self.n_chunks, self.gamma = n_chunks, gamma
        self.outcomes = {str(Path(key).resolve()): value for key, value in outcomes.items()}
        self.indices = []
        self.lookup = {}
        # Include every frame for next-state lookup, regardless of sample stride.
        self.dataset.samples = []
        stride = int(config.get("sample_stride", 1))
        for file in self.dataset.files:
            path = str(Path(file).resolve())
            if path not in self.outcomes:
                raise ValueError(f"missing explicit terminal/success label for {path}")
            label = self.outcomes[path]
            if (set(label) - {"terminated", "success", "end_status", "usable_frames"}
                    or not {"terminated", "success"} <= set(label)
                    or any(type(label[key]) is not bool for key in ("terminated", "success"))):
                raise ValueError(f"outcome for {path} must contain boolean terminated and success")
            if label["success"] and not label["terminated"]:
                raise ValueError("success must be a true episode termination")
            end_status = label.get("end_status")
            expected = {"success": (True, True), "failure": (True, False),
                        "truncated": (False, False), "unknown": (False, False)}
            if end_status is not None and (end_status not in expected or
                    expected[end_status] != (label["terminated"], label["success"])):
                raise ValueError(f"inconsistent end_status for {path}")
            with open(file) as stream:
                count = int(json.load(stream)["num_frames"])
            if "usable_frames" in label:
                limit = label["usable_frames"]
                if end_status != "unknown" or type(limit) is not int or not 2 <= limit <= count:
                    raise ValueError(f"invalid usable_frames for {path}")
                # Keep the full recording on disk, but censor visually ambiguous
                # reset/withdrawal tails before constructing any training window.
                count = limit
            # For truncated recordings the last frame is the bootstrap state;
            # its action is not used because its next observation is unavailable.
            usable = count if label["terminated"] else count - 1
            for frame in range(count):
                index = len(self.dataset.samples)
                self.dataset.samples.append({"file": file, "frame_index": frame})
                self.lookup[(path, frame)] = index
                # Unknown ending is NOT labelled as a timeout or terminal failure.
                # Use only full n-chunk windows with a recorded next state before
                # the ambiguous final frame. No loss ever targets that boundary.
                interior = end_status != "unknown" or frame + n_chunks * self.dataset.action_length < count - 1
                if frame < usable and frame % stride == 0 and interior:
                    self.indices.append((index, path, frame, count, usable))
        if not self.indices:
            raise ValueError("no usable replay transitions")

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def observation(item):
        # JsonDataset appends exactly these two SFT-only turns. Removing them
        # also prevents demonstration action tokens leaking into the RL state.
        messages = item["messages"][:-2]
        images, texts = [], []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    images.append(content["image"])
                elif content["type"] == "text":
                    texts.append(content["text"])
        return {"messages": messages, "images": images, "text": "\n".join(texts), "state": item["state"]}

    def __getitem__(self, index):
        source_index, path, frame, count, usable = self.indices[index]
        current = self.dataset[source_index]
        label = self.outcomes[path]
        rewards, terminals = np.zeros(usable), np.zeros(usable, dtype=bool)
        if label["terminated"]:
            terminals[-1] = True
            rewards[-1] = float(label["success"])
        reward, steps, done, next_frame = chunk_return(
            rewards, terminals, frame, self.dataset.action_length, self.n_chunks, self.gamma)
        # The terminal state is immaterial because its bootstrap multiplier is 0.
        next_item = self.dataset[self.lookup[(path, min(next_frame, count - 1))]]
        action_mask = current["action_mask"].clone()
        action_mask[min(self.dataset.action_length, usable - frame):] = 0
        return {"observation": self.observation(current), "next_observation": self.observation(next_item),
                "action": current["action"], "action_mask": action_mask,
                "reward": reward, "steps": steps, "terminated": done}


class TransitionCollate:
    def __init__(self, actor_path, language_path, vision_path):
        from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
        self.actor_processor = AutoProcessor.from_pretrained(actor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(language_path)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_path)

    def critic_observation(self, observations):
        text = self.tokenizer([item["text"] for item in observations], padding=True,
                              truncation=False, return_tensors="pt")
        counts = [len(item["images"]) for item in observations]
        if not all(counts):
            raise ValueError("DIVL requires at least one camera per observation")
        pixels = self.image_processor(images=[image for item in observations for image in item["images"]],
                                      return_tensors="pt")["pixel_values"]
        padded = pixels.new_zeros(len(observations), max(counts), *pixels.shape[1:])
        view_mask = torch.zeros(len(observations), max(counts), dtype=torch.bool)
        offset = 0
        for i, count in enumerate(counts):
            padded[i, :count] = pixels[offset:offset + count]
            view_mask[i, :count] = True
            offset += count
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"],
                "pixel_values": padded, "view_mask": view_mask,
                "state": torch.stack([item["state"] for item in observations])}

    def __call__(self, items):
        observations = [item["observation"] for item in items]
        actor = dict(self.actor_processor.apply_chat_template(
            [item["messages"] for item in observations], tokenize=True, padding=True,
            return_dict=True, return_tensors="pt", do_resize=False, add_generation_prompt=False))
        actor["state"] = torch.stack([item["state"] for item in observations])
        result = {"actor_observation": actor, "critic_observation": self.critic_observation(observations),
                  "next_critic_observation": self.critic_observation([item["next_observation"] for item in items])}
        for key in ("action", "action_mask"):
            result[key] = torch.stack([item[key] for item in items])
        for key in ("reward", "steps", "terminated"):
            result[key] = torch.tensor([item[key] for item in items])
        return result
