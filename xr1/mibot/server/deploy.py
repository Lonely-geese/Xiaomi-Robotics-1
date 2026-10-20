# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from os.path import join as osp
from pathlib import Path

import torch
from mmengine import Config

from mibot.models import MIMODEL
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.server.runtime.server import Server
from mibot.utils.io import build_action_mask, validate_quantiles, validate_stats


def strip_prefix(state_dict, prefix):
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_model(model_dir, device):
    cfg = Config.fromfile(osp(model_dir, "config.py"))
    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)
    ckpt = torch.load(
        osp(model_dir, "last.ckpt/checkpoint", "mp_rank_00_model_states.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["module"]
    print(model.load_state_dict(strip_prefix(ckpt, "model."), strict=True))
    return cfg, model.eval().to(device)


def load_stats(cfg, device, model_dir=None):
    data = cfg.data.params.train_datasets.copy()
    if model_dir is not None and Path(model_dir, 'normalize.json').exists():
        path = Path(model_dir, 'normalize.json')
        record = Path(model_dir, 'run_record.json')
        if record.exists():
            expected = json.loads(record.read_text()).get('normalization_sha256')
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Deployment normalization does not match the training run')
        data['normalization'] = str(path)
    action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    stats = JsonDataset._normalization(data)
    mean, std = validate_stats(stats['mean'], stats['std'], action_length)
    q01, q99 = validate_quantiles(stats['q01'], stats['q99'])
    return (
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.tensor(q01, device=device),
        torch.tensor(q99, device=device),
        torch.from_numpy(build_action_mask(action_length)).to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument('--recap-guidance-scale', type=float, default=None,
        help='Override the RECAP inference guidance strength (0=unconditional, 1=positive policy).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0"
    cfg, model = load_model(args.model, device)
    if args.recap_guidance_scale is not None:
        if not model.recap_enabled or args.recap_guidance_scale < 0:
            raise ValueError('A nonnegative guidance scale requires a RECAP model')
        model.recap_guidance_scale = args.recap_guidance_scale
    mean, std, q01, q99, action_mask = load_stats(cfg, device, args.model)
    print(f"RECAP enabled={model.recap_enabled}, conditioning={model.recap_conditioning}, guidance_scale={model.recap_guidance_scale}")

    try:
        server = Server(args.host, args.port, model, mean, std, q01, q99, action_mask, device)
        print(f"Starting server on {args.host}:{args.port}")
        server.run()
    except OSError as error:
        if error.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
