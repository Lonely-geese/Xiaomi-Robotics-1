"""Offline BEE head training on explicit replay, or correction-only demo prewarm."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mibot.bee.core import BeeLearner
from mibot.bee.data import (
    OfflineReplay,
    feature_assets,
    load_settings,
    take,
    validate_payload,
)
from mibot.rlt.data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_bee.yaml"),
    )
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--mode", choices=("offline-rl", "correction"), default="offline-rl")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--updates",
        required=True,
        type=int,
        help="Additional updates in the selected mode",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--correction-pretrain", type=int, default=500)
    p.add_argument(
        "--correction-every",
        type=int,
        default=100,
        help="Offline critic steps, NOT newly collected samples",
    )
    p.add_argument("--correction-updates", type=int, default=12)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--initialize-correction",
        type=Path,
        help="Load only the correction model from a prewarm checkpoint",
    )
    a = p.parse_args()
    if (
        min(a.updates, a.correction_every, a.save_every) < 1
        or a.batch_size < 2
        or a.batch_size % 2
    ):
        p.error(
            "updates/intervals must be positive; batch size must be positive and even"
        )
    if min(a.correction_pretrain, a.correction_updates) < 0:
        p.error("correction update counts must be nonnegative")
    if a.resume and a.initialize_correction:
        p.error("Use resume or initialize-correction, not both")
    if a.initialize_correction and a.mode != "offline-rl":
        p.error("initialize-correction is only used to initialize offline RL")
    settings, base, cfg = load_settings(a.config)
    assets = feature_assets(base, settings["stage1_checkpoint"], cfg)
    payload = torch.load(a.data, map_location="cpu", weights_only=True)
    validate_payload(payload, cfg, assets, corrections=a.mode == "correction")
    data = payload["transitions"]
    replay = OfflineReplay(data, cfg) if a.mode == "offline-rl" else None
    if replay is None:
        eligible = torch.where(
            data["correction_mask"][..., list(cfg.editable_dims)].flatten(1).any(-1)
        )[0]
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise ValueError(
            "Output directory is nonempty; use a new directory or explicit resume"
        )
    torch.set_num_threads(8)
    torch.manual_seed(settings["seed"])
    learner = BeeLearner(cfg, a.device)
    training = {
        "mode": a.mode,
        "data_sha256": sha256(a.data),
        "source": payload["source"],
        "batch_size": a.batch_size,
        "seed": settings["seed"],
        "correction_pretrain": a.correction_pretrain,
        "correction_every": a.correction_every,
        "correction_updates": a.correction_updates,
        "device_type": torch.device(a.device).type,
        "schedule": "fixed-offline-replay; correction interval counts critic updates",
    }
    if a.initialize_correction:
        init = torch.load(
            a.initialize_correction, map_location="cpu", weights_only=True
        )
        if (
            init["format"] != "xr1-bee-v1"
            or init["assets"] != assets
            or init["config"] != asdict(cfg)
            or init["correction_steps"] < 1
        ):
            raise ValueError(
                "Correction initialization mismatch or untrained correction model"
            )
        state = {
            k.removeprefix("correction."): v
            for k, v in init["model"].items()
            if k.startswith("correction.")
        }
        learner.model.correction.load_state_dict(state, strict=True)
        training["initial_correction_sha256"] = sha256(a.initialize_correction)
    if a.resume:
        previous = torch.load(a.resume, map_location="cpu", weights_only=True)
        if "initial_correction_sha256" in previous["training"]:
            training["initial_correction_sha256"] = previous["training"][
                "initial_correction_sha256"
            ]
        learner.restore(a.resume, assets, training)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "run.json").write_text(
        json.dumps(
            {"assets": assets, "config": asdict(cfg), "training": training}, indent=2
        )
    )

    def report(metrics, phase):
        metrics = dict(phase=phase, **metrics)
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)

    if replay is not None:
        # Resume unfinished pretraining after interruption; never pretrain again after RL begins.
        if learner.step == 0:
            for _ in range(max(0, a.correction_pretrain - learner.correction_steps)):
                report(
                    learner.update_correction(
                        replay.sample(a.batch_size, a.device, correction_only=True)
                    ),
                    "pretrain",
                )
                if learner.correction_steps % a.save_every == 0:
                    learner.save(a.output / "latest.pt", assets, training)
        if learner.correction_steps == 0 and not training.get(
            "initial_correction_sha256"
        ):
            raise ValueError(
                "Fit or explicitly initialize the correction model before offline RL"
            )
    for i in range(a.updates):
        if replay is None:
            indices = eligible[torch.randint(len(eligible), (a.batch_size,))]
            report(
                learner.update_correction(take(data, indices, a.device)), "correction"
            )
        else:
            report(learner.update(replay.sample(a.batch_size, a.device)), "offline-rl")
            if learner.step % a.correction_every == 0:
                for _ in range(a.correction_updates):
                    report(
                        learner.update_correction(
                            replay.sample(a.batch_size, a.device, correction_only=True)
                        ),
                        "correction",
                    )
        if (i + 1) % a.save_every == 0:
            learner.save(a.output / "latest.pt", assets, training)
    learner.save(a.output / "latest.pt", assets, training)
    print(
        json.dumps(
            {
                "event": "saved",
                "path": str(a.output / "latest.pt"),
                "step": learner.step,
                "correction_steps": learner.correction_steps,
                "actor_updated": learner.step >= cfg.policy_delay,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
