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

from mibot.data.recap import nstep_advantages
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


def main(args):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        dist.init_process_group('nccl')
    rank = int(os.environ.get('RANK', '0'))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    metadata = json.loads((args.data/'prepared.json').read_text())
    if checkpoint['metadata']['source_metadata_sha256'] != metadata['source_metadata_sha256']:
        raise ValueError('Value checkpoint was trained on different data')
    native = checkpoint.get('backend') == 'native_rlinf'
    if native:
        from mibot.models.recap_rlinf import build_critic, build_processor, observation
        model = build_critic(checkpoint['config']['siglip_path'], checkpoint['config']['gemma3_path']).float().to(device).eval()
        processor = build_processor(checkpoint['config']['gemma3_path'])
        texts = {}
    else:
        model = RecapValueCritic().to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    records = read_jsonl(args.data/'episodes.jsonl')
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
        images = np.load(record['images'], mmap_mode='r')
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
        # SFT and known corrections are already positive. Define the top-30%
        # threshold on labeled autonomous rollout frames, not the larger SFT pool.
        scores = []
        for record in records:
            if record['labeled'] and (native or record['pool'] == 'rollout'):
                with np.load(args.data/'advantages'/f"{record['name']}.npz") as data:
                    scores.append(data['advantage'] if native else data['advantage'][data['human'] != 1])
        threshold = float(np.quantile(np.concatenate(scores), 1-metadata['positive_fraction']))
        counts = Counter()
        for record in records:
            path = args.data/'advantages'/f"{record['name']}.npz"
            with np.load(path, allow_pickle=False) as archive:
                data = {key:archive[key] for key in archive.files}
            positive = (np.isfinite(data['advantage']) & (data['advantage'] >= threshold)) if native else label_episode(
                record['pool'], data['advantage'], data['human'], threshold, metadata['horizon'])
            counts[record['pool']+'_positive'] += int(positive.sum())
            counts[record['pool']+'_frames'] += len(positive)
            counts['human_correction_frames'] += int((data['human'] == 1).sum()) if record['pool'] == 'rollout' else 0
            temporary = path.with_suffix('.partial.npz')
            np.savez(temporary, **data, positive=positive)
            temporary.replace(path)
        report = dict(ready=True, value_checkpoint=str(args.checkpoint.resolve()),
            value_metrics=checkpoint['metrics'], threshold=threshold,
            threshold_population='all known-outcome SFT and rollout frames, native RLinf global quantile' if native else 'non-correction frames from known-outcome rollouts; unknown intervention coverage retained',
            positive_fraction=metadata['positive_fraction'], gamma=metadata['gamma'], horizon=metadata['horizon'],
            correction_rule='native RLinf labels; no manual outcome/intervention overrides' if native else 'human action positive; preceding 30-frame autonomous horizon negative',
            backend='native_rlinf' if native else 'cnn',
            **dict(counts))
        (args.data/'advantages_ready.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=base/'data/recap_20261020')
    parser.add_argument('--checkpoint', type=Path, default=base/'outputs/recap_value_20261020/best.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    main(parser.parse_args())
