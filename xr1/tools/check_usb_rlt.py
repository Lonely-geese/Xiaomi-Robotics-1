"""Read-only dataset/weight checks and optional one-batch GPU integration test."""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from mibot.rlt.data import ObservationCollator, USBDataset, sha256
from mibot.rlt.learner import build_policy
from mibot.rlt.model import (
    asset_metadata,
    build_model,
    move,
    save_stage1,
    unpack_prefix,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/usb_rlt.yaml"),
    )
    p.add_argument("--gpu-smoke", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint-roundtrip", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.set_num_threads(8)
    os.environ["XR1_VLM_PATH"] = cfg["vlm_path"]
    os.environ["XR1_VIDEO_BACKEND"] = "pyav"
    ds = USBDataset(cfg["dataset"], cfg["normalization"])
    report = {
        "episodes": len(ds.records),
        "frames": len(ds),
        "videos": 0,
        "action": "left local EE delta xyz, axis-angle, delta gripper",
        "proprio": "16 normalized joint/gripper values for both arms",
        "normalization_sha256": sha256(cfg["normalization"]),
        "training_started": False,
    }
    prepared = json.loads(
        (Path(cfg["normalization"]).parent / "prepared.json").read_text()
    )
    if report["normalization_sha256"] != prepared["normalization_sha256"]:
        raise ValueError("Checkpoint normalization hash mismatch")
    for r in ds.records:
        if pq.ParquetFile(ds.root / r["data_path"]).metadata.num_rows != r["length"]:
            raise ValueError("Parquet length mismatch")
        for video in r["video_paths"].values():
            if not (ds.root / video).is_file():
                raise FileNotFoundError(video)
            report["videos"] += 1
    samples = [ds.item_at(0, 0), ds.item_at(0, ds.records[0]["length"] - 1)]
    report["sample_action_shapes"] = [list(s["action"].shape) for s in samples]
    report["tail_valid_steps"] = int(samples[-1]["action_mask"][:, 0].sum())
    if a.gpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU integration requires CUDA")
        model = build_model(cfg, a.device).eval()
        # Compare against the original all-bf16 XR1 inference representation.
        model.base.to(dtype=torch.bfloat16)
        observation = move(ObservationCollator(cfg["vlm_path"])(samples[:1]), a.device)
        torch.manual_seed(cfg["seed"])
        with torch.inference_mode():
            obs, reference = model.extract(observation)
        # Check equivalence to the original XR1 inference path with identical noise.
        torch.manual_seed(cfg["seed"])
        mask = torch.zeros_like(reference)
        mask[..., :7] = 1
        baseline_batch = dict(
            observation,
            action=torch.zeros_like(reference, dtype=torch.bfloat16),
            action_mask=mask.to(torch.bfloat16),
        )
        with torch.inference_mode():
            baseline = model.base.generate(baseline_batch)
        torch.testing.assert_close(reference, baseline.float(), atol=0, rtol=0)
        report["reference_exactly_matches_xr1"] = True
        report["feature_shapes"] = {k: list(v.shape) for k, v in obs.items()}
        report["prefix_tokens"] = observation["input_ids"].shape[1]
        policy = build_policy(cfg).to(a.device).eval()
        with torch.inference_mode():
            predicted, _ = policy.predict_action_batch(obs, mode="eval")
            decoded = model.coordinates.decode(predicted)
        report["stage2_shape"] = list(decoded.shape)
        del policy, reference, baseline, obs, decoded
        # Match packed SFT prefix to the observation-only feature path.
        from mibot.data.collate.custom_collate import CustomCollate

        batch = move(CustomCollate()(samples[:1]), a.device)
        prefixes = []
        hook = model.base.vlm.register_forward_hook(
            lambda m, args, out: prefixes.append(out.hidden_states.detach())
        )
        model.set_stage1_mode(cfg["stage1"]["mode"])
        model.train()
        if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
            raise AssertionError("Trainable master parameters must be fp32")
        report["trainable_master_weights_fp32"] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        hook.remove()
        packed, packed_mask = unpack_prefix(
            prefixes.pop(), batch["action_vlm_condition_segments"]
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.base.vlm(
                **{k: v for k, v in observation.items() if k != "state"},
                use_cache=False,
                skip_logits=True,
            )
        torch.testing.assert_close(packed, output.hidden_states, rtol=0.02, atol=0.05)
        report["sft_prefix_matches_observation_only"] = True
        report["losses"] = {
            k: float(loss[k].detach()) for k in ("loss", "rlt_loss", "vla_loss")
        }
        loss["loss"].backward()
        if any(p.grad is not None for p in model.base.vlm.parameters()):
            raise AssertionError("Frozen VLM received gradients")
        for name, module in [
            ("rlt", model.rlt_module),
            ("action_expert", model.base.dit),
        ]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if (
                not gradients
                or not all(torch.isfinite(g).all() for g in gradients)
                or not any(torch.count_nonzero(g) for g in gradients)
            ):
                raise AssertionError(f"Invalid {name} gradients")
            report[name + "_backward_ok"] = True
        report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["checkpoint_strict_load"] = True
        if a.checkpoint_roundtrip:
            model.zero_grad(set_to_none=True)
            del loss, output, packed, packed_mask, prefixes, baseline_batch
            model.eval()
            torch.manual_seed(123)
            expected_obs, expected_ref = model.extract(observation)
            expected_obs = {k: v.cpu() for k, v in expected_obs.items()}
            expected_ref = expected_ref.cpu()
            assets = asset_metadata(cfg)
            optimizer = torch.optim.AdamW(model.rlt_module.parameters(), lr=1e-5)
            with tempfile.TemporaryDirectory(
                prefix="xr1_rlt_roundtrip_", dir=a.output.parent
            ) as temporary:
                path = Path(temporary) / "stage1.pt"
                save_stage1(path, model, cfg, optimizer, 0, assets)
                del optimizer, model
                gc.collect()
                torch.cuda.empty_cache()
                restored = build_model(cfg, a.device, path).eval()
                torch.manual_seed(123)
                actual_obs, actual_ref = restored.extract(observation)
                torch.testing.assert_close(
                    actual_ref.cpu(), expected_ref, atol=0, rtol=0
                )
                for key in expected_obs:
                    torch.testing.assert_close(
                        actual_obs[key].cpu(), expected_obs[key], atol=0, rtol=0
                    )
            report["stage1_checkpoint_exact_roundtrip"] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
