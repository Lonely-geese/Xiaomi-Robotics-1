#!/usr/bin/env python3
"""Build isolated, filtered event-return pools from immutable XR1 caches.

No original data is deleted. Excluded samples are absent from every training
manifest. RGB arrays are reused read-only; rewards, policies and labels live in
the new experiment directory. No model inference or training is started here.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from mibot.data.recap_events import (
    REWARD_VERSION, TERMINAL_REWARDS, event_returns, event_segments,
    experiment_fingerprint, validation_episode,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f'Refusing to overwrite dataset: {output}')
    metadata = json.loads((source/'prepared.json').read_text())
    records = read_jsonl(source/'episodes.jsonl')
    rows = {}
    for dataset, expected in metadata['source_metadata_sha256'].items():
        path = Path(dataset)/'meta/episodes.jsonl'
        if sha256(path) != expected:
            raise ValueError(f'Source changed: {path}')
        rows[dataset] = {r['episode_index']: r for r in read_jsonl(path)}
    policies = defaultdict(list)
    for pool in ('sft', 'rollout'):
        for path in sorted((source/pool/'episodes').glob('*.json')):
            traj = json.loads(path.read_text())
            key = (traj['source']['dataset'], traj['source']['episode_index'])
            policies[key].append((path, traj))
    selected, exclusions = [], []
    whole_excluded = Counter()
    retained_episodes = Counter()
    for record in records:
        row = rows[record['dataset']][record['episode_index']]
        with np.load(record['numeric'], allow_pickle=False) as data:
            human = data['human']
        if len(human) != record['length']:
            raise ValueError(f"Cache length mismatch: {record['name']}")
        spans, rejected = event_segments(record['pool'], record['outcome'], row.get('human_intervened'), human)
        if spans:
            retained_episodes[record['pool']] += 1
        else:
            whole_excluded[rejected[0][2]] += 1
        for a, b, reason in rejected:
            exclusions.append(dict(source_name=record['name'], dataset=record['dataset'],
                episode_index=record['episode_index'], start=a, end_exclusive=b, reason=reason,
                frames=b-a, whole_episode=not spans))
        for a, b, event in spans:
            selected.append((record, a, b, event))
    # A frame-count bound for a small shared time cost, not an inferred timeout.
    time_reference = max(r['length'] for r, _, _, _ in selected)
    def identity(record):
        return f"{record['dataset']}:{record['episode_index']}"
    validation_ids = {identity(r) for r, _, _, _ in selected if validation_episode(r)}
    # Filtering invalid data can empty a rare validation class. Deterministically
    # hold out at least two distinct parent episodes, never individual segments.
    for event in ('autonomous_success', 'takeover', 'failure'):
        parents = {identity(r) for r, _, _, e in selected if e == event}
        if len(parents) >= 4:
            candidates = sorted(parents-validation_ids, key=lambda s:hashlib.sha256(s.encode()).hexdigest())
            needed = max(0, 2-len(parents & validation_ids))
            validation_ids.update(candidates[:needed])
    payload = dict(version=REWARD_VERSION, terminal_rewards=TERMINAL_REWARDS,
        time_reference_frames=time_reference, total_time_cost_bound=0.1, gamma=1.0,
        return_scale=1.1, source_metadata_sha256=metadata['source_metadata_sha256'],
        source_manifest_sha256=sha256(source/'episodes.jsonl'),
        normalization_sha256=sha256(source/'normalize.json'),
        policy_excludes_validation=True,
        validation_parent_ids=sorted(validation_ids),
        selection=[(r['name'], a, b, event) for r, a, b, event in selected])
    fingerprint = experiment_fingerprint(payload)
    for sub in ('sft/episodes', 'rollout/episodes', 'critic', 'arrays', 'advantages'):
        (output/sub).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    event_counts = defaultdict(Counter)
    split_counts = defaultdict(Counter)
    new_records = []
    for i, (original, a, b, event) in enumerate(selected):
        name = f"{original['name']}_event{a:06d}_{b:06d}"
        with np.load(original['numeric'], allow_pickle=False) as archive:
            state, human = archive['state'][a:b], archive['human'][a:b]
        labeled = event != 'correction'
        rewards, returns = event_returns(b-a, event, time_reference) if labeled else (
            np.full(b-a, np.nan, np.float32), np.full(b-a, np.nan, np.float32))
        numeric = output/'critic'/f'{name}.npz'
        np.savez(numeric, state=state, human=human, rewards=rewards, returns=returns)
        validation = identity(original) in validation_ids
        record = dict(name=name, dataset=original['dataset'], episode_index=original['episode_index'],
            source_name=original['name'], pool=original['pool'], length=b-a,
            event=event, outcome=original['outcome'], labeled=labeled,
            split='validation' if validation else 'train',
            source_start=a, source_end_exclusive=b, images=original['images'], image_start=a,
            numeric=str(numeric), prompt=original['prompt'], policy_frames=0, policy_segments=0)
        # Intersect with the previous temporal-integrity segments. Physical JSON
        # boundaries also stop action chunks before takeovers and handbacks.
        if not validation:
            key = (original['dataset'], original['episode_index'])
            for path, traj in policies[key]:
                start = int(traj['recap']['offset'])
                left, right = max(a, start), min(b, start+int(traj['num_frames']))
                if left >= right:
                    continue
                adapted = deepcopy(traj)
                part = f'{name}_policy{left:06d}_{right:06d}'
                arrays_path = Path(traj['arrays_path'])
                if not arrays_path.is_absolute():
                    arrays_path = (path.parent/arrays_path).resolve()
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    if any(len(arrays[k]) != traj['num_frames'] for k in arrays.files):
                        raise ValueError(f'Array frame count mismatch: {arrays_path}')
                    np.savez(output/'arrays'/f'{part}.npz', **{
                        k: arrays[k][left-start:right-start] for k in arrays.files})
                adapted.update(time=part, num_frames=right-left,
                    arrays_path=str(output/'arrays'/f'{part}.npz'))
                for videos in adapted['observations'].values():
                    if len(videos) != 1:
                        raise ValueError('Expected one video per view')
                    videos[0]['start'] += left-start
                adapted['source'].update(start=left, end_exclusive=right, event=event)
                adapted['recap'].update(sidecar=str(output/'advantages'/f'{name}.npz'),
                    offset=left-a, event=event, experiment_fingerprint=fingerprint)
                (output/original['pool']/'episodes'/f'{part}.json').write_text(json.dumps(adapted))
                record['policy_frames'] += right-left
                record['policy_segments'] += 1
        counts[original['pool']+'_frames'] += record['policy_frames']
        counts[original['pool']+'_segments'] += record['policy_segments']
        event_counts[event]['segments'] += 1
        event_counts[event]['frames'] += b-a
        event_counts[event]['policy_frames'] += record['policy_frames']
        split_key = f"{record['split']}_{original['pool']}"
        split_counts[split_key]['policy_frames'] += record['policy_frames']
        if labeled:
            split_counts[split_key]['critic_frames'] += b-a
            split_counts[split_key]['critic_segments'] += 1
        new_records.append(record)
        if i % 200 == 0:
            print(json.dumps(dict(prepared_segments=i+1, total_segments=len(selected))), flush=True)
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new_records))
    (output/'excluded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exclusions))
    shutil.copyfile(source/'normalize.json', output/'normalize.json')
    audit = dict(retained_source_episodes=dict(retained_episodes),
        excluded_whole_episodes=dict(whole_excluded), excluded_frames=sum(r['frames'] for r in exclusions),
        events={k:dict(v) for k,v in event_counts.items()}, splits={k:dict(v) for k,v in split_counts.items()},
        stop_reason_available=False, stop_policy='Use explicit outcome only; never infer timeout/danger from end of recording',
        takeover_evidence='Known human_intervened plus action-aligned 0->1 transition; human intervention treated as task correction',
        correction_policy='Successful rollout corrections are policy-only positives; failed/unknown corrections excluded')
    audit['validation_rule'] = 'Original episode hash split; top up rare autonomous events to at least two parent episodes when at least four exist; all sibling segments held out together'
    (output/'selection_report.json').write_text(json.dumps(audit, indent=2))
    (output/'reward_spec.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='selection'}, indent=2))
    report = dict(metadata, **dict(counts))
    report.update(ready=True, derived_from=str(source), reward_version=REWARD_VERSION,
        experiment_fingerprint=fingerprint, return_scale=1.1,
        time_reference_frames=time_reference, terminal_rewards=TERMINAL_REWARDS,
        step_reward=-0.1/time_reference, gamma=1.0, label_mode=REWARD_VERSION,
        human_label_override=True, independent_policy_evaluation=False, policy_validation_excluded=True,
        unknown_outcome_policy='excluded from critic and policy',
        threshold_population='training autonomous segments only; positive advantage and top 30 percent',
        sft_episodes=retained_episodes['sft'], rollout_episodes=retained_episodes['rollout'],
        selection_report='selection_report.json')
    report.pop('failure_reward', None)
    temporary = output/'prepared.partial.json'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output/'prepared.json')
    print(json.dumps(audit, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--source', type=Path, default=base/'data/recap_rlinf_1563_20261020')
    parser.add_argument('--output', type=Path, default=base/'data/recap_events_1563_20261021')
    args = parser.parse_args()
    prepare(args.source, args.output)
