#!/usr/bin/env python3
"""Aggregate immutable, temporally aligned XR1 caches into HG-DAgger pools.

Pass --source repeatedly for later collection rounds. Raw data is never
deleted. Cache input is the output of prepare_recap_dataset/prepare_recap_rlinf,
before event splitting; no critic weights or advantage files are read.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from mibot.data.dagger import VERSION, classify_segments, expert_chunk_starts, fingerprint, is_validation, sha256


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(sources, output, horizon=30):
    sources = sorted({Path(s).resolve() for s in sources})
    output = Path(output).resolve()
    if not sources or horizon <= 0:
        raise ValueError('Nonempty source list and positive action horizon required')
    if output.exists():
        raise ValueError(f'Refusing to overwrite dataset: {output}')
    metadata = [json.loads((s/'prepared.json').read_text()) for s in sources]
    norms = {sha256(s/'normalize.json') for s in sources}
    if len(norms) != 1:
        raise ValueError('Aggregation requires identical frozen normalization')
    norm_hash = next(iter(norms))
    parents, policies, source_rows = {}, defaultdict(list), {}
    for source, meta in zip(sources, metadata):
        if not meta.get('ready') or meta['normalization_sha256'] != norm_hash:
            raise ValueError('Source cache is incomplete or normalization changed')
        for dataset, expected in meta['source_metadata_sha256'].items():
            path = Path(dataset)/'meta/episodes.jsonl'
            if sha256(path) != expected:
                raise ValueError(f'Source metadata changed: {path}')
            source_rows[dataset] = {r['episode_index']: r for r in read_jsonl(path)}
        for record in read_jsonl(source/'episodes.jsonl'):
            if 'event' in record or int(record.get('image_start', 0)):
                raise ValueError('Use the unsplit action-aligned cache, not a RECAP event cache')
            key = (record['dataset'], record['episode_index'])
            if key in parents:
                raise ValueError(f'Duplicate parent episode across aggregation sources: {key}')
            parents[key] = record
        for pool in ('sft', 'rollout'):
            for path in sorted((source/pool/'episodes').glob('*.json')):
                traj = json.loads(path.read_text())
                key = (traj['source']['dataset'], traj['source']['episode_index'])
                policies[key].append((path, traj))
    for sub in ('demo/episodes', 'correction/episodes', 'validation/demo/episodes',
                'validation/correction/episodes', 'arrays'):
        (output/sub).mkdir(parents=True, exist_ok=True)
    records, segments, files = [], [], []
    counts, events = Counter(), defaultdict(Counter)
    for number, (key, original) in enumerate(parents.items()):
        dataset, episode = key
        row = source_rows[dataset][episode]
        with np.load(original['numeric'], allow_pickle=False) as archive:
            human = archive['human']
        if len(human) != original['length']:
            raise ValueError(f'Invalid action-aligned mask: {key}')
        split = 'validation' if is_validation(dataset, episode) else 'train'
        spans = classify_segments(original['pool'], original['outcome'], row.get('human_intervened'), human)
        for span in spans:
            events[span['event']]['segments'] += 1
            events[span['event']]['frames'] += span['end_exclusive'] - span['start']
            segments.append(dict(dataset=dataset, episode_index=episode, split=split, **span))
        record = dict(dataset=dataset, episode_index=episode, split=split, outcome=original['outcome'],
                      numeric=original['numeric'], numeric_sha256=sha256(original['numeric']),
                      pool=original['pool'], length=len(human), human_intervened=row.get('human_intervened'))
        records.append(record)
        previous_end = 0
        for path, traj in sorted(policies[key], key=lambda p: p[1]['recap']['offset']):
            start, n = int(traj['recap']['offset']), int(traj['num_frames'])
            end = start + n
            if start < previous_end or end > len(human) or n <= 0:
                raise ValueError(f'Overlapping or invalid temporal segments: {path}')
            previous_end = end
            # Use the complete temporal segment for RLinf's all-expert chunk
            # criterion. A later handback cannot be reinterpreted as padding.
            valid = expert_chunk_starts(human[start:end], horizon) + start
            for span in spans:
                pool = span['target_pool']
                if pool is None:
                    continue
                left, right = max(start, span['start']), min(end, span['end_exclusive'])
                if left >= right:
                    continue
                starts = np.arange(left, right) if pool == 'demo' else valid[(valid >= left) & (valid < right)]
                if not len(starts):
                    continue
                if not np.array_equal(starts, np.arange(left, left + len(starts))):
                    raise ValueError('Expert starts must be a contiguous prefix of the human segment')
                name = f'{fingerprint([dataset, episode])[:20]}_{left:06d}_{right:06d}'
                arrays_path = Path(traj['arrays_path'])
                if not arrays_path.is_absolute():
                    arrays_path = (path.parent/arrays_path).resolve()
                target_arrays = output/'arrays'/f'{name}.npz'
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    if any(len(arrays[k]) != n for k in arrays.files):
                        raise ValueError(f'Array length mismatch: {arrays_path}')
                    np.savez(target_arrays, **{k: arrays[k][left-start:right-start] for k in arrays.files})
                adapted = deepcopy(traj)
                adapted.pop('recap', None)
                adapted.update(time=name, num_frames=right-left, arrays_path=str(target_arrays))
                for videos in adapted['observations'].values():
                    if len(videos) != 1:
                        raise ValueError('Expected one video per view')
                    videos[0]['start'] += left-start
                adapted['source'].update(start=left, end_exclusive=right, event=span['event'])
                adapted['dagger'] = dict(pool=pool, expert_only=True, horizon=horizon,
                    sample_frames=len(starts), split=split, terminal_label=0)
                relative = Path('validation')/pool/'episodes' if split == 'validation' else Path(pool)/'episodes'
                target = output/relative/f'{name}.json'
                target.write_text(json.dumps(adapted))
                files.append(dict(path=str(target.relative_to(output)), sha256=sha256(target),
                    arrays_path=str(target_arrays.relative_to(output)), arrays_sha256=sha256(target_arrays),
                    source_policy=str(path), source_policy_sha256=sha256(path),
                    source_arrays=str(arrays_path), source_arrays_sha256=sha256(arrays_path)))
                counts[f'{split}_{pool}_frames'] += len(starts)
                counts[f'{split}_{pool}_segments'] += 1
        if number % 400 == 0:
            print(json.dumps(dict(processed_parents=number+1, total=len(parents))), flush=True)
    if not counts['train_demo_frames'] or not counts['train_correction_frames']:
        raise ValueError('Both expert pools must have training data')
    shutil.copyfile(sources[0]/'normalize.json', output/'normalize.json')
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'segments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in segments))
    (output/'files.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in files))
    report = dict(version=VERSION, ready=True, sources=list(map(str, sources)), horizon=horizon,
        normalization_sha256=norm_hash, counts=dict(counts), events={k:dict(v) for k,v in events.items()},
        source_metadata_sha256={k:v for m in metadata for k,v in m['source_metadata_sha256'].items()},
        manifests={p:sha256(output/p) for p in ('episodes.jsonl', 'segments.jsonl', 'files.jsonl')},
        epoch_definition='larger expert pool once; smaller pool reshuffled and cycled; drop incomplete global half batch',
        policy_validation_excluded=True, independent_robot_evaluation=False,
        only_success=True, only_save_expert=True, critic_required=False,
        action_alignment='reuse existing aligned actions/masks; do not shift again',
        rlinf_commit='50f22ad9a7825eeaaf4180dca5300f2fa70758be')
    report['experiment_fingerprint'] = fingerprint(report)
    (output/'prepared.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=30)
    args = parser.parse_args()
    prepare(args.source, args.output, args.horizon)
