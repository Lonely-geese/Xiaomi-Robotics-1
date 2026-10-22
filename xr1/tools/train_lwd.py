"""Single-device XR-1 LWD learner; see docs/lwd.md for labels and checkpoints."""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from mibot.rl import DIVL, LWDConfig, LWDLearner
from mibot.rl.actor import XR1FlowActor
from mibot.rl.backbone import ValueBackbone
from mibot.rl.data import EpisodeTransitions, TransitionCollate
from mibot.rl.replay import to_device


def load_actor_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weights = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    if weights and all(key.startswith("model.") for key in weights):
        weights = {key[len("model."):]: value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)


def save_checkpoint(learner, directory):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "latest.tmp"
    torch.save(learner.checkpoint(), temporary)
    temporary.replace(directory / "latest.pt")
    # Compatible with BaseRunner's model-prefixed checkpoint format.
    actor_path = directory / "actor.tmp"
    torch.save({"module": {"model." + key: value for key, value in learner.actor.model.state_dict().items()}}, actor_path)
    actor_path.replace(directory / "actor.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start-online", action="store_true", help="transition an offline checkpoint to online")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    lwd = LWDConfig(**config.get("lwd", {}))
    if args.start_online and (not args.resume or lwd.stage != "online"):
        parser.error("--start-online requires --resume and lwd.stage=online")
    if lwd.stage == "online" and not args.resume:
        parser.error("online training must resume an offline/online LWD checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("full XR-1 training requires the CUDA environment from scripts/setup_env.sh")
    torch.manual_seed(config.get("seed", 0))
    random.seed(config.get("seed", 0))
    device = torch.device(config.get("device", "cuda:0"))
    os.environ["XR1_VLM_PATH"] = config["actor_vlm_path"]
    from mibot.models.VLA.XR1 import xr1

    model = xr1(full_finetune=True, async_train=False,
                active_action_dims=config["offline_data"].get("active_action_dims"))
    load_actor_weights(model, config["sft_checkpoint"])
    actor = XR1FlowActor(model).to(device)
    encoder = ValueBackbone.from_pretrained(config["critic_language_path"], config["critic_vision_path"])
    critic = DIVL(encoder, encoder.feature_dim, config=lwd).to(device)
    learner = LWDLearner(actor, critic, lwd)
    if args.resume:
        learner.restore(torch.load(args.resume, map_location="cpu", weights_only=True))
        if args.start_online:
            learner.start_online(lwd.total_steps)
        elif learner.config != lwd:
            raise ValueError("resume configuration differs from checkpoint; use the saved LWD settings")
    lwd = learner.config

    def dataset(data, outcomes_path, n_chunks):
        outcomes = json.loads(Path(outcomes_path).read_text())
        if data.get("action_length", 30) != 30:
            raise ValueError("XR-1 action_length must be 30")
        return EpisodeTransitions(data, outcomes, n_chunks=n_chunks, gamma=lwd.gamma)

    # Both offline and online replay use one-step chunk backups in online mode.
    offline = dataset(config["offline_data"], config["offline_outcomes"],
                      config.get("offline_n_chunks", 1) if lwd.stage == "offline" else 1)
    online = None
    if lwd.stage == "online":
        for key in ("normalization", "active_action_dims", "action_length"):
            if config["online_data"].get(key) != config["offline_data"].get(key):
                raise ValueError(f"online/offline {key} must match")
        online = dataset(config["online_data"], config["online_outcomes"], 1)
    collate = TransitionCollate(config["actor_vlm_path"], config["critic_language_path"], config["critic_vision_path"])
    batch_size = int(config.get("batch_size", 2))
    if batch_size < 1 or (online is not None and batch_size % 2):
        raise ValueError("online batches must be positive and even for 1:1 replay")
    save_every = int(config.get("save_every", 1000))
    if save_every < 1:
        raise ValueError("save_every must be positive")
    while learner.updates < lwd.total_steps:
        off_count = batch_size if online is None else batch_size // 2
        items = [offline[i] for i in torch.randint(len(offline), (off_count,)).tolist()]
        if online is not None:
            items += [online[i] for i in torch.randint(len(online), (batch_size - off_count,)).tolist()]
        batch = to_device(collate(items), device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            metrics = learner.update(batch)
        print(json.dumps({"step": learner.updates, **metrics}), flush=True)
        if learner.updates % save_every == 0:
            save_checkpoint(learner, Path(config["output_dir"]))
    save_checkpoint(learner, Path(config["output_dir"]))


if __name__ == "__main__":
    main()
