"""Portable strict checkpoint loading and two explicitly named inference profiles."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
import yaml

from mibot.rlt.modules import RLToken

PROFILES = ('direct-cache', 'reencoded-prefix')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('base_checkpoint', 'normalization', 'vlm_path'):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def verify_package(root, cache_path=None):
    """Verify actual bytes, caching only while every file's size and mtime match."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = manifest['files_sha256']
    stats = {name: [str((root/name).resolve()), (root/name).stat().st_size,
                    (root/name).stat().st_mtime_ns] for name in expected}
    if cache_path is not None and Path(cache_path).exists():
        old = json.loads(Path(cache_path).read_text())
        if old.get('files') == stats and old.get('sha256') == expected:
            return old
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise ValueError(f'Package SHA256 mismatch: {name}')
    report = {'files': stats, 'sha256': expected}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(cache_path) + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(cache_path)
    return report


def validate_assets(config, stage, bee_meta, package_hashes):
    if stage.get('format') != 'xr1-rlt-stage1-v1':
        raise ValueError('Unsupported Stage 1 checkpoint format')
    if stage['config']['rlt'] != config['rlt']:
        raise ValueError('RLToken config differs from checkpoint')
    for asset, filename in [('base_checkpoint', 'base/mp_rank_00_model_states.pt'),
                            ('normalization', 'normalize.json')]:
        if stage['assets'][asset] != package_hashes[filename]:
            raise ValueError(f'Stage 1 asset mismatch: {asset}')
    if bee_meta is not None:
        for key, filename in [('stage1_sha256','stage1_step20000.pt'),
                              ('base_sha256','base/mp_rank_00_model_states.pt'),
                              ('normalization_sha256','normalize.json')]:
            if bee_meta['assets'][key] != package_hashes[filename]:
                raise ValueError(f'BEE asset mismatch: {key}')
        if bee_meta['assets']['action_coordinates'] != 'xr1-zscore-per-offset-v1':
            raise ValueError('Unknown BEE action coordinates')
        if bee_meta['assets']['features'] != 'xr1-observation-only-rlt-v1':
            raise ValueError('Unknown BEE feature coordinates')


class ReconstructedPolicy(nn.Module):
    def __init__(self, base, rlt, profile):
        super().__init__()
        if profile not in PROFILES:
            raise ValueError(f'Unknown reconstruction profile: {profile}')
        self.base = base
        self.rlt_module = rlt
        self.profile = profile

    @torch.inference_mode()
    def forward(self, batch):
        # Hook only this observation-only forward. No action/teacher targets can
        # enter the RLT encoder. The direct profile leaves the XR-1 cache intact.
        features = {}

        def observation_hook(module, args, out):
            mask = out.attention_mask.bool()
            z = self.rlt_module.encoder(out.hidden_states.float(), mask)
            features['z'] = z
            if self.profile == 'direct-cache':
                return out
            reconstructed = self.rlt_module.decoder(z, mask.shape[1], mask)
            lm_out = self.base.vlm.model.language_model(
                inputs_embeds=reconstructed.to(out.hidden_states.dtype),
                attention_mask=mask, position_ids=out.position_ids, use_cache=True,
            )
            return type(out)(hidden_states=lm_out.last_hidden_state,
                             past_key_values=lm_out.past_key_values,
                             position_ids=out.position_ids, attention_mask=mask,
                             rope_deltas=out.rope_deltas)

        # Whitelist prevents evaluation labels or caller-supplied actions leaking
        # into the conditioning computation.
        allowed = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'state')
        request = {key: batch[key] for key in allowed if key in batch}
        state = request['state']
        action = torch.zeros(state.shape[0], 30, 60, device=state.device, dtype=state.dtype)
        mask = torch.zeros_like(action)
        mask[..., :7] = 1
        request.update(action=action, action_mask=mask)
        handle = self.base.vlm.register_forward_hook(observation_hook)
        try:
            action = self.base.generate(request).float()
        finally:
            handle.remove()
        if not torch.isfinite(action).all() or not torch.isfinite(features['z']).all():
            raise RuntimeError('Nonfinite model outputs')
        return {'normalized_action': action, 'z': features['z']}


def build_model(config, device, checkpoint, *, profile='direct-cache'):
    """Compatible entry point; profile is reconstruction metadata, not original code."""
    from mibot.models.VLA.XR1 import xr1

    config = dict(config)
    stage = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if stage.get('format') != 'xr1-rlt-stage1-v1' or stage['config']['rlt'] != config['rlt']:
        raise ValueError('Stage 1 format/config mismatch')
    base_checkpoint = torch.load(config['base_checkpoint'], map_location='cpu', mmap=True, weights_only=True)
    state = {key.removeprefix('model.'): value for key, value in base_checkpoint['module'].items()}
    non_vlm = {key for key in state if not key.startswith('vlm.')}
    if non_vlm != set(stage['base_non_vlm']):
        raise ValueError('Stage 1 must replace exactly all non-VLM base parameters')
    state.update(stage['base_non_vlm'])
    previous = os.environ.get('XR1_VLM_PATH')
    os.environ['XR1_VLM_PATH'] = config['vlm_path']
    try:
        # Construct on the target GPU to avoid random-initialized 5B CPU tensors.
        with torch.device(device):
            base = xr1(ffn_gradient_checkpointing=False, async_train=False,
                       active_action_dims=list(range(7)))
    finally:
        if previous is None:
            os.environ.pop('XR1_VLM_PATH', None)
        else:
            os.environ['XR1_VLM_PATH'] = previous
    base.load_state_dict(state, strict=True)
    rlt = RLToken(config['rlt'])
    rlt.load_state_dict(stage['rlt_module'], strict=True)
    rlt = rlt.to(device)
    model = ReconstructedPolicy(base, rlt, profile).eval().requires_grad_(False)
    model.load_report = {
        'base_keys': len(state), 'stage1_non_vlm_keys': len(stage['base_non_vlm']),
        'rlt_keys': len(stage['rlt_module']), 'strict': True,
        'base_compute_dtype': 'bfloat16', 'rlt_compute_dtype': 'float32',
        'profile': profile, 'original_implementation_equivalence': 'unverified',
    }
    return model
