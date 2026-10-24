#!/usr/bin/env python3
"""Infer per-frame values and label positive advantages/corrections."""
import argparse
from collections import Counter
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from mibot.data.recap import nstep_advantages, rlinf_positive_labels
from mibot.data.recap_events import REWARD_VERSION, event_positive_labels, require_matching_data
from mibot.models.recap_value import RecapValueCritic
from prepare_recap_dataset import read_jsonl


def label_episode(pool, advantage, human, threshold, horizon=30):
    if pool == 'sft':
        return np.ones(len(human), dtype=bool)
    positive = np.isfinite(advantage) & (advantage >= threshold)
    # A successful repair must not credit the preceding autonomous failure.
    upcoming_human = np.array([np.any(human[t:t+horizon] == 1) for t in range(len(human))])
    positive[upcoming_human & (human != 1)] = False
    positive[human == 1] = True
    return positive


def finalize_advantages(root, checkpoint_path, metadata, records, native, digest, metrics):
    """Publish labels only after validating cached inference provenance."""
    scores = []
    event_mode = metadata.get('reward_version') == REWARD_VERSION
    for record in records:
        with np.load(root/'advantages'/f"{record['name']}.npz", allow_pickle=False) as data:
            if str(data['value_checkpoint_digest']) != digest:
                raise ValueError(f"Cached critic fingerprint mismatch: {record['name']}")
            if any(len(data[k]) != record['length'] for k in ('values','advantage','human')):
                raise ValueError(f"Cached frame count mismatch: {record['name']}")
            if not np.isfinite(data['values']).all() or (record['labeled'] and not np.isfinite(data['advantage']).all()):
                raise ValueError(f"Nonfinite critic outputs: {record['name']}")
            eligible = (record['labeled'] and record['pool'] == 'rollout' and record['split'] == 'train') if event_mode else (
                record['labeled'] and (native or record['pool'] == 'rollout'))
            if eligible:
                selected = data['advantage'] if native else data['advantage'][data['human'] != 1]
                if not np.isfinite(selected).all():
                    raise ValueError(f"Nonfinite known-outcome advantage: {record['name']}")
                scores.append(selected.copy())
    if not scores or not sum(len(s) for s in scores):
        raise ValueError('No eligible training advantages for threshold calibration')
    threshold = float(np.quantile(np.concatenate(scores), 1-metadata['positive_fraction']))
    if event_mode:
        threshold = max(0.0, threshold)
    counts = Counter()
    for record in records:
        path = root/'advantages'/f"{record['name']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
        if event_mode:
            positive = event_positive_labels(record['event'], data['advantage'], threshold)
            counts[record['event']+'_positive'] += int(positive.sum())
        else:
            positive = rlinf_positive_labels(record['pool'],data['advantage'],threshold) if native else label_episode(
                record['pool'],data['advantage'],data['human'],threshold,metadata['horizon'])
        counts[record['pool']+'_positive'] += int(positive.sum())
        counts[record['pool']+'_frames'] += len(positive)
        counts['human_correction_frames'] += int((data['human'] == 1).sum()) if record['pool'] == 'rollout' else 0
        temporary = path.with_suffix('.partial.npz')
        np.savez(temporary, **{k:v for k,v in data.items() if k!='positive'}, positive=positive)
        temporary.replace(path)
    if counts['sft_positive'] != counts['sft_frames']:
        raise ValueError('Every SFT demonstration frame must have a positive advantage label')
    report = dict(ready=True,value_checkpoint=str(checkpoint_path.resolve()),value_checkpoint_sha256=digest,
        value_metrics=metrics,threshold=threshold,
        threshold_population='all known-outcome SFT and rollout frames; SFT labels then forced positive as in RLinf' if native else 'non-correction frames from known-outcome rollouts; unknown intervention coverage retained',
        positive_fraction=metadata['positive_fraction'],gamma=metadata['gamma'],horizon=metadata['horizon'],
        label_rule='rlinf_sft_all_positive_v1' if native else 'legacy_human_overrides',
        correction_rule='SFT always positive; rollout quantile labels without intervention overrides' if native else 'human action positive; preceding 30-frame autonomous horizon negative',
        backend='native_rlinf' if native else 'cnn',**dict(counts))
    if event_mode:
        report.update(label_rule=REWARD_VERSION, experiment_fingerprint=metadata['experiment_fingerprint'],
            threshold_population='training autonomous segments only; strictly positive advantage',
            correction_rule='all retained demonstrations and successful human correction segments positive; no cross-event bootstrap')
    temporary = root/'advantages_ready.partial.json'
    temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'advantages_ready.json')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main(args):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    checksum = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):checksum.update(block)
    digest = checksum.hexdigest()
    metadata = json.loads((args.data/'prepared.json').read_text())
    require_matching_data(checkpoint['metadata'], metadata)
    native = checkpoint.get('backend') == 'native_rlinf'
    records = read_jsonl(args.data/'episodes.jsonl')
    if args.labels_only:
        if world != 1:
            raise ValueError('--labels-only runs on one CPU process')
        finalize_advantages(args.data,args.checkpoint,metadata,records,native,digest,checkpoint['metrics'])
        return
    if world > 1:
        dist.init_process_group('nccl')
    rank = int(os.environ.get('RANK', '0'))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    if native:
        from mibot.models.recap_rlinf import build_critic, build_processor, observation
        model = build_critic(checkpoint['config']['siglip_path'], checkpoint['config']['gemma3_path']).float().to(device).eval()
        processor = build_processor(checkpoint['config']['gemma3_path'])
        texts = {}
    else:
        model = RecapValueCritic().to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    for position, record in enumerate(records[rank::world]):
        destination = args.data/'advantages'/f"{record['name']}.npz"
        if destination.exists():
            try:
                with np.load(destination, allow_pickle=False) as cached:
                    if (str(cached['value_checkpoint_digest']) == digest
                            and len(cached['values']) == record['length']
                            and len(cached['advantage']) == record['length']):
                        continue
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass
        with np.load(record['numeric'], allow_pickle=False) as archive:
            state, rewards, human = archive['state'], archive['rewards'], archive['human']
        image_start = int(record.get('image_start', 0))
        images = np.load(record['images'], mmap_mode='r')[image_start:image_start+record['length']]
        if len(images) != record['length'] or len(rewards) != record['length']:
            raise ValueError(f"Image/numeric segment length mismatch: {record['name']}")
        values = np.empty(len(images), dtype=np.float32)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for a in range(0, len(images), args.batch_size):
                b = min(a+args.batch_size, len(images))
                image = torch.from_numpy(images[a:b].copy()).to(device)
                if native:
                    if record['prompt'] not in texts:
                        texts[record['prompt']] = processor.process_text([record['prompt']])
                    text = texts[record['prompt']]
                    tokens = text['input_ids'].to(device).expand(b-a,-1)
                    mask = text['attention_mask'].to(device).expand(b-a,-1)
                    prediction = model(observation(image,tokens,mask,processor)).predicted_values
                else:
                    proprio = torch.from_numpy(state[a:b]).to(device)
                    prediction = model.value(image, proprio)
                values[a:b] = prediction.float().cpu().numpy()
        advantage = nstep_advantages(rewards, values, metadata['return_scale'], metadata['horizon']) if record['labeled'] else np.full(len(images), np.nan, np.float32)
        temporary = destination.with_suffix('.inference.partial.npz')
        np.savez(temporary, values=values, advantage=advantage, human=human,
            value_checkpoint_digest=digest)
        temporary.replace(destination)
        if position % 100 == 0:
            print(json.dumps(dict(stage='value_inference', rank=rank, episodes=position+1,
                total_local=len(records[rank::world]))), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        finalize_advantages(args.data,args.checkpoint,metadata,records,native,digest,checkpoint['metrics'])
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=base/'data/recap_20261020')
    parser.add_argument('--checkpoint', type=Path, default=base/'outputs/recap_value_20261020/best.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--labels-only',action='store_true',help='Validate cached critic outputs and relabel on CPU')
    main(parser.parse_args())
