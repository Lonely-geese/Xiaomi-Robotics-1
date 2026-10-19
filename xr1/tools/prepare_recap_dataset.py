#!/usr/bin/env python3
"""Build RECAP pools and return sidecars, preserving source data and labels.

Reuse the audited continuous SFT segments. Source videos are referenced, never
copied/re-encoded for policy training. Critic thumbnails use every source frame.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow.parquet as pq

from mibot.data.recap import action_intervention_mask, episode_returns
from prepare_ethernet_dataset import CAMERAS, NAMES, trajectory_arrays


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_episode(job):
    root, info, row, pool, out, image_size = job
    n, index = int(row['length']), int(row['episode_index'])
    fmt = dict(episode_index=index, episode_chunk=index // info['chunks_size'])
    name = f'{root.name}_ep{index:06d}'
    numeric = out/'critic'/f'{name}.npz'
    images = out/'images'/f'{name}.npy'
    table = pq.read_table(root/info['data_path'].format(**fmt))
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if state.shape != (n, 30) or action.shape != (n, 30):
        raise ValueError(f'{name}: state/action shape mismatch')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f'{name}: nonfinite numeric inputs')
    if not np.array_equal(table['frame_index'].to_numpy(), np.arange(n)):
        raise ValueError(f'{name}: noncontiguous frames')
    outcome = row.get('outcome', row.get('operator_verdict'))
    # The user explicitly identifies both historical datasets as demonstrations.
    # All demo actions are positive for CFG, but unknown outcomes stay unknown
    # and do not acquire invented value targets.
    labeled = outcome in ('success', 'failure')
    rewards, returns = episode_returns(n, outcome == 'success') if labeled else (
        np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32))
    human = action_intervention_mask(row) if pool == 'rollout' else np.ones(n, np.int8)
    if pool == 'rollout':
        audit = pq.read_table(root/info['audit']['path'].format(**fmt))
        if len(audit) != n or not np.array_equal(audit['frame_index'].to_numpy(), np.arange(n)):
            raise ValueError(f'{name}: audit frame mismatch')
        recorded = np.array([-1 if x is None else int(x) for x in audit['action_human_intervened'].to_pylist()], dtype=np.int8)
        known = human >= 0
        if not np.array_equal(recorded[known], human[known]):
            raise ValueError(f'{name}: action intervention metadata disagrees with per-frame audit')
        # An episode-level True with no interval cannot locate a takeover.
        # Preserve that flag, and use the explicit per-frame audit coverage.
        human = recorded
    np.savez(numeric, state=state, rewards=rewards, returns=returns, human=human)
    videos = {view: [dict(path=str(root/info['video_path'].format(**fmt, video_key='observation.images.'+camera)),
                          start=0, crop_bbox=None)] for view, camera in CAMERAS.items()}
    if pool == 'rollout':
        arrays = out/'arrays'/f'{name}.npz'
        np.savez(arrays, **trajectory_arrays(state, action))
        trajectory = dict(trajectory_type='ongoing', time=name, num_frames=n,
            arrays_path=str(arrays), observations=videos,
            source=dict(dataset=str(root), episode_index=index, start=0, end_exclusive=n),
            recap=dict(pool=pool, sidecar=str(out/'advantages'/f'{name}.npz'), offset=0),
            instruction={'general':[{'images':['observations.'+v for v in CAMERAS],
                'conversations':[{'from':'human','value':'The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n'+row['tasks'][0]}, {'from':'gpt','value':''}]}]})
        (out/pool/'episodes'/f'{name}.json').write_text(json.dumps(trajectory))
    # Atomic image pack, resumable independently after an interrupted decode.
    if not images.exists():
        temporary = images.with_suffix('.partial.npy')
        # Assemble in RAM and issue a sequential file write. Per-frame mmap
        # writes are very slow on CFS and interleave the three camera streams.
        packed = np.empty((n, 3, 3, image_size, image_size), dtype=np.uint8)
        for view_index, view in enumerate(CAMERAS):
            with av.open(videos[view][0]['path']) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                count = 0
                for frame in container.decode(stream):
                    if count >= n:
                        raise ValueError(f'{name}: extra video frames')
                    rgb = frame.reformat(width=image_size, height=image_size, format='rgb24').to_ndarray()
                    packed[count, view_index] = rgb.transpose(2, 0, 1)
                    count += 1
                if count != n:
                    raise ValueError(f'{name}: decoded {count} != {n}')
        np.save(temporary, packed)
        del packed
        temporary.replace(images)
    return dict(name=name, dataset=str(root), episode_index=index, pool=pool,
        length=n, outcome=outcome, labeled=labeled, numeric=str(numeric), images=str(images),
        human_frames=int((human == 1).sum()), unknown_intervention_frames=int((human < 0).sum()),
        human_intervened=row.get('human_intervened'),
        human_intervals=row.get('human_intervals', []), action_human_intervals=row.get('action_human_intervals', []),
        source_human_intervals=row.get('source_human_intervals', []),
        minimum_return=float(returns.min()) if labeled else None)


def prepare(args):
    out = args.output.resolve()
    if (out/'prepared.json').exists():
        print('Preparation already complete:', out, flush=True)
        return
    for sub in ('sft/episodes','rollout/episodes','arrays','critic','images','advantages'):
        (out/sub).mkdir(parents=True, exist_ok=True)
    sources = [p.resolve() for p in args.sft] + [args.rollout.resolve()]
    rows_by_source, jobs, excluded = {}, [], []
    for root in sources:
        info = json.loads((root/'meta/info.json').read_text())
        if info['fps'] != 30 or info['features']['observation.state']['names'] != NAMES:
            raise ValueError(f'{root}: incompatible FPS/state layout')
        if info['cleaning']['action_shift_frames'] != 3:
            raise ValueError(f'{root}: expected already shifted action labels')
        rows = read_jsonl(root/'meta/episodes.jsonl')
        if len(rows) != info['total_episodes'] or sum(r['length'] for r in rows) != info['total_frames']:
            raise ValueError(f'{root}: metadata totals mismatch')
        rows_by_source[str(root)] = {int(r['episode_index']):r for r in rows}
        pool = 'rollout' if root == args.rollout.resolve() else 'sft'
        for row in rows:
            if any(row.get(k) is False for k in ('data_valid','structurally_valid')) or row.get('integrity_error_code'):
                excluded.append(dict(dataset=str(root), episode_index=row['episode_index'], reason='explicit_invalid'))
                continue
            jobs.append((root, info, row, pool, out, args.image_size))
    valid = {(str(j[0]), int(j[2]['episode_index'])) for j in jobs}
    counts = Counter()
    for record in read_jsonl(args.sft_prepared/'segment_manifest.jsonl'):
        if (record['dataset'], int(record['episode_index'])) not in valid:
            continue
        if record['dataset'] not in {str(p) for p in sources[:-1]}:
            raise ValueError('Unexpected dataset in reused temporal preparation')
        old = args.sft_prepared/record['split']/'episodes'/f"{record['name']}.json"
        traj = json.loads(old.read_text())
        traj['arrays_path'] = str((old.parent/traj['arrays_path']).resolve())
        name = f"{Path(record['dataset']).name}_ep{record['episode_index']:06d}"
        traj['recap'] = dict(pool='sft', sidecar=str(out/'advantages'/f'{name}.npz'), offset=record['start'])
        destination = out/'sft/episodes'/old.name
        if not destination.exists():
            destination.write_text(json.dumps(traj))
        counts['sft_frames'] += traj['num_frames']
        counts['sft_segments'] += 1
    shutil.copyfile(args.normalization, out/'normalize.json')
    jobs.sort(key=lambda job: job[3] != 'rollout')
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor, (out/'episodes.jsonl').open('w') as manifest:
        for start in range(0, len(jobs), args.workers):
            for rec in executor.map(prepare_episode, jobs[start:start+args.workers]):
                records.append(rec)
                manifest.write(json.dumps(rec)+'\n')
                counts[rec['pool']+'_episodes'] += 1
                if rec['pool'] == 'rollout':
                    counts['rollout_frames'] += rec['length']
            manifest.flush()
            print(json.dumps(dict(processed=len(records), total=len(jobs), counts=dict(counts))), flush=True)
    report = dict(ready=True, **dict(counts), excluded=excluded,
        sources=[str(p) for p in sources], source_metadata_sha256={str(p):sha256(p/'meta/episodes.jsonl') for p in sources},
        normalization_source=str(args.normalization.resolve()), normalization_sha256=sha256(out/'normalize.json'),
        return_scale=-min(r['minimum_return'] for r in records if r['labeled']),
        gamma=1.0, failure_reward=-300.0, horizon=30, positive_fraction=0.3,
        image_size=args.image_size, critic='three-view CNN + measured state, 201-bin categorical value head',
        unknown_outcome_policy='excluded from critic targets; rollout unconditional unless action is known human correction',
        sft_policy='all identified demonstration actions positive; original unknown outcomes preserved',
        epoch_definition='larger pool once, smaller pool cycled; complete global half batches',
        former_holdouts_consumed=True, independent_policy_evaluation=False,
        action_shift_applied=0, intervention_timebase='action_human_intervals verified against action_human_intervened audit')
    (out/'prepared.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    data = Path('/mnt/cfs/7rnh3z/kele/data/00_traindata')
    parser.add_argument('--sft', type=Path, nargs='+', default=[data/'0923_8072', data/'20260928_935'])
    parser.add_argument('--rollout', type=Path, default=data/'recap_755')
    parser.add_argument('--sft-prepared', type=Path, default=base/'data/ethernet_temporal_left_v1')
    parser.add_argument('--normalization', type=Path, default=base/'data/ethernet_middle_mixed_old2000_20261013/normalize.json')
    parser.add_argument('--output', type=Path, default=base/'data/recap_20261020')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=64)
    prepare(parser.parse_args())
