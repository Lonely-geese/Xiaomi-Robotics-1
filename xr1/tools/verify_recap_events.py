#!/usr/bin/env python3
"""CPU audit of event boundaries, rewards, source alignment and held-out groups."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from mibot.data.recap import nstep_advantages
from mibot.data.recap_events import REWARD_VERSION, event_returns, experiment_fingerprint
from prepare_recap_events import read_jsonl, sha256


def verify(root):
    root = root.resolve()
    meta = json.loads((root/'prepared.json').read_text())
    assert meta['reward_version'] == REWARD_VERSION
    source = Path(meta['derived_from'])
    originals = {r['name']: r for r in read_jsonl(source/'episodes.jsonl')}
    records = read_jsonl(root/'episodes.jsonl')
    by_name = {r['name']: r for r in records}
    assert len(by_name) == len(records)
    spec = json.loads((root/'reward_spec.json').read_text())
    spec['selection'] = [(r['source_name'],r['source_start'],r['source_end_exclusive'],r['event']) for r in records]
    assert experiment_fingerprint(spec) == meta['experiment_fingerprint']
    assert sha256(source/'episodes.jsonl') == spec['source_manifest_sha256']
    assert sha256(root/'normalize.json') == meta['normalization_sha256'] == spec['normalization_sha256']
    for dataset, expected in meta['source_metadata_sha256'].items():
        assert sha256(Path(dataset)/'meta/episodes.jsonl') == expected
    parent_splits = defaultdict(set)
    spans = defaultdict(list)
    stats = Counter()
    td_error = 0.0
    event_validation = Counter()
    for r in records:
        parent_splits[r['source_name']].add(r['split'])
        a,b = r['source_start'],r['source_end_exclusive']
        original = originals[r['source_name']]
        spans[r['source_name']].append((a,b))
        assert b-a == r['length'] and 0 <= a < b <= original['length']
        assert r['image_start'] == a and r['images'] == original['images']
        images = np.load(r['images'],mmap_mode='r')
        assert len(images) == original['length']
        with np.load(r['numeric']) as data, np.load(original['numeric']) as old:
            np.testing.assert_array_equal(data['state'],old['state'][a:b])
            np.testing.assert_array_equal(data['human'],old['human'][a:b])
            assert not np.any(data['human'] < 0)
            if r['labeled']:
                rewards,returns = event_returns(b-a,r['event'],meta['time_reference_frames'])
                np.testing.assert_array_equal(data['rewards'],rewards)
                np.testing.assert_array_equal(data['returns'],returns)
                assert np.isfinite(returns).all() and np.all((returns/meta['return_scale']>=-1)&(returns<=0))
                td = nstep_advantages(rewards,returns/meta['return_scale'],meta['return_scale'],meta['horizon'])
                td_error = max(td_error,float(np.abs(td).max()))
                if r['split']=='validation' and r['pool']=='rollout':
                    event_validation[r['event']] += 1
            else:
                assert r['event']=='correction' and r['outcome']=='success'
                assert np.all(data['human']==1) and np.isnan(data['returns']).all()
            if r['event']=='takeover':
                assert b < original['length'] and old['human'][b-1]==0 and old['human'][b]==1
            if r['event'] in ('autonomous_success','failure'):
                assert b==original['length'] and np.all(data['human']==0)
        stats['event_segments'] += 1
        stats['critic_frames'] += r['length'] if r['labeled'] else 0
    assert all(len(s)==1 for s in parent_splits.values())
    assert all(event_validation[e]>=2 for e in ('autonomous_success','takeover','failure'))
    excluded = read_jsonl(root/'excluded.jsonl')
    for r in excluded:
        spans[r['source_name']].append((r['start'],r['end_exclusive']))
        if r['whole_episode']:
            assert r['source_name'] not in parent_splits
    # Every original frame is either retained once or explicitly excluded once.
    for name, original in originals.items():
        cursor = 0
        for a,b in sorted(spans[name]):
            assert a==cursor
            cursor=b
        assert cursor==original['length']
    policy_counts = Counter()
    record_policy = Counter()
    original_policies = defaultdict(list)
    for pool in ('sft','rollout'):
        for path in (source/pool/'episodes').glob('*.json'):
            traj=json.loads(path.read_text())
            original_policies[(traj['source']['dataset'],traj['source']['episode_index'])].append((path,traj))
        for path in (root/pool/'episodes').glob('*.json'):
            traj=json.loads(path.read_text())
            r=by_name[Path(traj['recap']['sidecar']).stem]
            assert r['split']=='train' and r['pool']==pool
            assert traj['recap']['experiment_fingerprint']==meta['experiment_fingerprint']
            a,b=traj['source']['start'],traj['source']['end_exclusive']
            assert r['source_start']<=a<b<=r['source_end_exclusive']
            assert traj['recap']['offset']==a-r['source_start']
            assert traj['num_frames']==b-a
            candidates=[(p,t) for p,t in original_policies[(r['dataset'],r['episode_index'])]
                        if t['recap']['offset']<=a and b<=t['recap']['offset']+t['num_frames']]
            assert len(candidates)==1
            old_path,old=candidates[0]
            offset=a-old['recap']['offset']
            old_arrays=Path(old['arrays_path'])
            if not old_arrays.is_absolute():old_arrays=old_path.parent/old_arrays
            with np.load(traj['arrays_path']) as data,np.load(old_arrays) as original:
                assert set(data.files)==set(original.files)
                for key in data.files:
                    np.testing.assert_array_equal(data[key],original[key][offset:offset+b-a])
            for view, videos in traj['observations'].items():
                assert videos[0]['path']==old['observations'][view][0]['path']
                assert videos[0]['start']==old['observations'][view][0]['start']+offset
            policy_counts[pool] += b-a
            record_policy[r['name']] += b-a
    assert all(record_policy[r['name']]==r['policy_frames'] for r in records)
    assert all(policy_counts[p]==meta[p+'_frames'] for p in ('sft','rollout'))
    assert td_error < 1e-6
    result=dict(passed=True,experiment_fingerprint=meta['experiment_fingerprint'],
        checked_segments=len(records),checked_source_episodes=len(originals),
        policy_frames=dict(policy_counts),validation_events=dict(event_validation),
        max_exact_return_td_error=td_error,source_alignment_verified=True,
        excluded_frames_absent=True,train_validation_parent_overlap=0,
        original_files_preserved=True,critic_or_policy_training_started=False)
    temporary=root/'verification.partial.json'
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(root/'verification.json')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/recap_events_1563_20261021')
    verify(parser.parse_args().data)
