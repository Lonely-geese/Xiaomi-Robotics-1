#!/usr/bin/env python3
"""CPU audit of every HG-DAgger target and its original action/video alignment."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from mibot.data.dagger import VERSION, fingerprint, sha256, is_validation
from prepare_dagger_dataset import read_jsonl


def verify(root):
    root = Path(root).resolve()
    meta = json.loads((root/'prepared.json').read_text())
    payload = {k:v for k,v in meta.items() if k != 'experiment_fingerprint'}
    if meta['version'] != VERSION or not meta['ready'] or fingerprint(payload) != meta['experiment_fingerprint']:
        raise ValueError('Invalid HG-DAgger preparation fingerprint')
    if sha256(root/'normalize.json') != meta['normalization_sha256']:
        raise ValueError('Normalization changed')
    for path, expected in meta['manifests'].items():
        if sha256(root/path) != expected:
            raise ValueError(f'Manifest changed: {path}')
    for dataset, expected in meta['source_metadata_sha256'].items():
        if sha256(Path(dataset)/'meta/episodes.jsonl') != expected:
            raise ValueError(f'Source metadata changed: {dataset}')
    parents = {(r['dataset'], r['episode_index']): r for r in read_jsonl(root/'episodes.jsonl')}
    masks = {}
    for key, record in parents.items():
        if sha256(record['numeric']) != record['numeric_sha256']:
            raise ValueError(f'Action-aligned source changed: {key}')
        with np.load(record['numeric'], allow_pickle=False) as arrays:
            masks[key] = arrays['human']
    counts, splits, frame_ids = Counter(), {}, set()
    checked_hashes = {}

    def check_hash(path, expected):
        path = str(path)
        if path not in checked_hashes:
            checked_hashes[path] = sha256(path)
        if checked_hashes[path] != expected:
            raise ValueError(f'Data changed: {path}')

    listed = read_jsonl(root/'files.jsonl')
    actual = {str(p.relative_to(root)) for pool in ('demo', 'correction')
              for prefix in ('', 'validation') for p in (root/prefix/pool/'episodes').glob('*.json')}
    if actual != {r['path'] for r in listed}:
        raise ValueError('Unexpected or missing policy files')
    for item in listed:
        for path, expected in ((root/item['path'], item['sha256']),
                               (root/item['arrays_path'], item['arrays_sha256']),
                               (item['source_policy'], item['source_policy_sha256']),
                               (item['source_arrays'], item['source_arrays_sha256'])):
            check_hash(path, expected)
        traj = json.loads((root/item['path']).read_text())
        original = json.loads(Path(item['source_policy']).read_text())
        source, dagger = traj['source'], traj['dagger']
        key = (source['dataset'], source['episode_index'])
        record, human = parents[key], masks[key]
        a, b = source['start'], source['end_exclusive']
        n, samples, horizon = traj['num_frames'], dagger['sample_frames'], dagger['horizon']
        pool, split = dagger['pool'], dagger['split']
        expected_split = 'validation' if is_validation(*key) else 'train'
        if split != expected_split or record['split'] != split or splits.setdefault(key, split) != split:
            raise ValueError('Parent episode leaks across train/validation')
        if not (0 <= a < b <= record['length'] and n == b-a and 0 < samples <= n):
            raise ValueError('Invalid target range')
        if horizon != meta['horizon'] or not dagger['expert_only'] or 'recap' in traj:
            raise ValueError('Invalid policy supervision metadata')
        if record['outcome'] != 'success':
            raise ValueError('Unsuccessful trajectory used as expert data')
        start = int(original['recap']['offset'])
        end = start + original['num_frames']
        if not start <= a < b <= end:
            raise ValueError('Action range crosses a temporal integrity boundary')
        if pool == 'correction':
            if record['pool'] != 'rollout' or not record['human_intervened'] or not np.all(human[a:b] == 1):
                raise ValueError('Autonomous/unknown action in correction targets')
            # Independent literal RLinf test, including the ORIGINAL end, so
            # cutting at handback cannot manufacture terminal padding.
            for t in range(a, a+samples):
                if not np.all(human[t:min(t+horizon, end)] == 1):
                    raise ValueError('Action chunk crosses expert/policy boundary')
        elif pool != 'demo' or record['pool'] != 'sft' or samples != n:
            raise ValueError('Invalid demonstration target')
        for t in range(a, a+samples):
            identity = (*key, t)
            if identity in frame_ids:
                raise ValueError('Duplicate expert frame in aggregation')
            frame_ids.add(identity)
        with np.load(root/item['arrays_path'], allow_pickle=False) as arrays, np.load(item['source_arrays'], allow_pickle=False) as src:
            if set(arrays.files) != set(src.files):
                raise ValueError('Action/state fields changed')
            for field in arrays.files:
                expected = src[field][a-start:b-start]
                if not np.array_equal(arrays[field], expected) or not np.isfinite(arrays[field]).all():
                    raise ValueError(f'Action/state alignment changed: {field}')
        for view, videos in traj['observations'].items():
            before = original['observations'][view][0]
            expected = dict(before, start=before['start'] + a-start)
            if videos != [expected] or not Path(before['path']).is_file():
                raise ValueError('Video alignment or path changed')
        counts[f'{split}_{pool}_frames'] += samples
        counts[f'{split}_{pool}_segments'] += 1
    if dict(counts) != meta['counts']:
        raise ValueError('Pool sizes disagree with manifest')
    report = dict(verified=True, experiment_fingerprint=meta['experiment_fingerprint'],
                  counts=dict(counts), policy_segments=len(listed), parent_overlap=0,
                  expert_only=True, action_video_alignment=True, normalization_sha256=meta['normalization_sha256'])
    (root/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    verify(parser.parse_args().data)
