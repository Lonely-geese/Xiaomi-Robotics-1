import json
from pathlib import Path


def verify(cfg, run, stage, root):
    if stage == "value":
        meta = json.loads((run / "value/completed.json").read_text())
        if meta["steps"] < cfg["value"]["steps"] or not (run / "value/best.pt").is_file():
            raise RuntimeError("Incomplete RECAP value checkpoint")
    elif stage == "label":
        meta = json.loads((root / cfg["prepared_data"] / "advantages_ready.json").read_text())
        if not meta["ready"] or Path(meta["value_checkpoint"]).resolve() != run / "value/best.pt":
            raise RuntimeError("RECAP labels reference another value checkpoint")
    else:
        out = root / "xr1/outputs/project_xiaomi-robotics-1" / cfg["policy"]["experiment"]
        if not (out / "completed.json").is_file() or not (out / "last.ckpt/checkpoint/mp_rank_00_model_states.pt").is_file():
            raise RuntimeError("Incomplete RECAP policy checkpoint")


def command(root, cfg, run, stage, launcher, dry):
    data = (root / cfg["prepared_data"]).resolve()
    if stage == "value":
        value = cfg["value"]
        return [*launcher, "tools/train_recap_rlinf_value.py", "--data", data, "--output", run / "value",
                "--siglip", root / value["vision_path"], "--gemma", root / value["language_path"],
                "--steps", value["steps"], "--workers", value["workers"], "--resume"], {}
    if stage == "label":
        if not dry:
            verify(cfg, run, "value", root)
        return [*launcher, "tools/compute_recap_advantages.py", "--data", data, "--checkpoint",
                run / "value/best.pt", "--batch-size", cfg["label"]["batch_size"]], {}
    if not dry:
        verify(cfg, run, "label", root)
    policy = cfg["policy"]
    return ["bash", "scripts/train_recap_8gpu.sh", "train"], dict(
        RECAP_CONFIG="recap_rlinf", RECAP_DATA_DIR=str(data), EXP_NAME=policy["experiment"],
        XR1_VLM_PATH=str((root / policy["actor_vlm_path"]).resolve()),
        PRETRAINED_CKPT=str((root / policy["sft_checkpoint"]).resolve()),
        MASTER_PORT=str(cfg.get("port", 29557)))
