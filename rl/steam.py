import json
from pathlib import Path


def settings(root, cfg, run, stage):
    native = "critic" if stage == "value" else "policy"
    template = root / "xr1/configs" / f"steam_{native}.json"
    params = json.loads(template.read_text())
    params.update(cfg[stage])
    params.update(manifest=str((root / cfg["manifest"]).resolve()), output=str(run / stage))
    for key in (("vision_path", "language_path") if stage == "value" else ("actor_vlm_path", "sft_checkpoint")):
        params[key] = str((root / cfg[stage][key]).resolve())
    if stage == "policy":
        params["advantages"] = str(run / "advantages.parquet")
    return params


def verify(cfg, run, stage, root):
    if stage == "label":
        meta = json.loads((run / "advantages.parquet.json").read_text())
        if not meta["complete"] or not (run / "advantages.parquet").is_file():
            raise RuntimeError("Incomplete STEAM labels")
    else:
        latest = json.loads((run / stage / "latest.json").read_text())
        if latest["step"] < settings(root, cfg, run, stage)["max_steps"] or not Path(latest["checkpoint"]).is_file():
            raise RuntimeError(f"STEAM {stage} stopped before its requested final checkpoint")


def command(root, cfg, run, stage, launcher, dry):
    if stage == "label":
        if not dry:
            verify(cfg, run, "value", root)
        return [*launcher, "tools/label_steam_advantages.py", "--checkpoint", run / "value",
                "--manifest", (root / cfg["manifest"]).resolve(), "--output", run / "advantages.parquet",
                "--batch-size", cfg["label"]["batch_size"], "--rollout-fraction", cfg["label"]["rollout_fraction"],
                "--expert-fraction", cfg["label"]["expert_fraction"]], {}
    params = settings(root, cfg, run, stage)
    config_file = run / f"{stage}.json"
    if not dry:
        if stage == "policy":
            verify(cfg, run, "label", root)
        config_file.write_text(json.dumps(params, indent=2))
    native = "critic" if stage == "value" else "policy"
    cmd = [*launcher, f"tools/train_steam_{native}.py", "--config", config_file, "--loader-processes", "4",
           "--decode-workers", "4", "--prefetch-batches", "3", "--fused-optimizer", "--ddp"]
    if stage == "policy":
        cmd.append("--coalesced-broadcast")
    if (run / stage / "latest.json").exists():
        cmd.extend(["--resume", run / stage])
    return cmd, {}
