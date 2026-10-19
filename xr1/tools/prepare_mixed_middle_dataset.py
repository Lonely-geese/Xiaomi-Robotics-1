#!/usr/bin/env python3
"""Prepare an immutable, success-only natural mixture; never modify source data.

Old action labels are preserved. New raw labels are state[t+3] within verified
continuous acquisition intervals. Original validation/test are intentionally
consumed as training data, as authorized; this output has no holdout split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from prepare_ethernet_dataset import CAMERAS, NAMES, inspect_episode, trajectory_arrays
from prepare_temporal_dataset import audit_one, session_key, temporal_segments


PROMPT = 'Pick up the Ethernet cable and insert it into the RJ45 network port.'
POLICY = dict(window_frames=10, joint_range_degrees=0.05,
              position_displacement_m=0.0005, rotation_displacement_degrees=0.15,
              gripper_range_m=0.0002, min_static_frames=10,
              prefix_context_frames=10, suffix_hold_frames=15,
              min_segment_frames=30, action_shift_frames=3)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_rejection(row):
    if row.get('operator_verdict') != 'success':
        return 'explicit_failure' if row.get('operator_verdict') == 'failure' else 'unknown_verdict'
    if any(row.get(key) is False for key in ('data_valid', 'structurally_valid')):
        return 'explicit_invalid'
    if row.get('integrity_error_code') or row.get('integrity_error_reason'):
        return 'integrity_error_metadata'
    if row.get('quality_accounting_complete') is False:
        return 'incomplete_quality_accounting'
    return None


def stratified_selection(records, count, seed=42):
    """One episode minimum per source/session, then proportional capacities."""
    groups = defaultdict(list)
    for record in records:
        groups[(record['dataset'], record['session'])].append(record)
    if not len(groups) <= count <= len(records):
        raise ValueError(f'cannot cover {len(groups)} strata with {count} of {len(records)} episodes')
    quotas = {key: 1 for key in groups}
    remaining = count - len(groups)
    capacity = {key: len(value)-1 for key, value in groups.items()}
    total_capacity = sum(capacity.values())
    if remaining:
        exact = {key: remaining * value / total_capacity for key, value in capacity.items()}
        for key in groups:
            quotas[key] += int(exact[key])
        leftover = count - sum(quotas.values())
        order = sorted(groups, key=lambda key: (-(exact[key]-int(exact[key])), key))
        for key in order[:leftover]:
            quotas[key] += 1
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda r: hashlib.sha256(
            f"{seed}:{r['dataset']}:{r['session']}:{r['episode_index']}".encode()).hexdigest())
        selected.extend(ranked[:quotas[key]])
    assert len(selected) == count
    return selected, {f'{key[0]}::{key[1]}': value for key, value in quotas.items()}


def edge_crop(state, policy=POLICY):
    """Use window motion, retain internal waits, and err toward retaining data.

    Sliding windows mark their entire support as potentially moving. Both arms
    and both grippers participate, so small joint movement can preserve a slow
    insertion even when Cartesian displacement is below its threshold.
    """
    n = len(state)
    width = int(policy['window_frames'])
    if n < width:
        return 0, n, {'status': 'too_short_to_classify', 'removed_prefix': 0, 'removed_suffix': 0}
    moving_windows = np.zeros(n-width+1, dtype=bool)
    for offset in (0, 15):
        for start in range(n-width+1):
            window = state[start:start+width, offset:offset+15]
            joint = float(np.max(np.ptp(window[:, :7], axis=0)))
            xyz = float(np.max(np.linalg.norm(window[:, 7:10]-window[0, 7:10], axis=1)))
            q = window[:, 10:14].astype(np.float64)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            angle = float(np.rad2deg(2*np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))).max())
            gripper = float(np.ptp(window[:, 14]))
            moving_windows[start] |= (joint > policy['joint_range_degrees'] or
                xyz > policy['position_displacement_m'] or
                angle > policy['rotation_displacement_degrees'] or
                gripper > policy['gripper_range_m'])
    starts = np.flatnonzero(moving_windows)
    if len(starts) == 0:
        # No confident movement identification: preserve instead of deleting
        # potentially subtle successful manipulation. Explicitly flag review.
        return 0, n, {'status': 'no_confident_motion_preserved', 'removed_prefix': 0, 'removed_suffix': 0}
    first, last = int(starts[0]), int(starts[-1])+width
    a = max(0, first-policy['prefix_context_frames']) if first >= policy['min_static_frames'] else 0
    b = min(n, last+policy['suffix_hold_frames']) if n-last >= policy['min_static_frames'] else n
    return a, b, {'status': 'window_motion', 'first_motion_window': first,
                  'last_motion_window_end': last, 'removed_prefix': a, 'removed_suffix': n-b}


def new_temporal_segments(audit, timestamps, n, fps=30):
    """Guard shifted targets on the RAW timeline, including final 3 frames."""
    columns = {'source_timestamp': timestamps}
    for key in audit.column_names:
        if key == 'observation_sequence' or key in {
                'camera_sample_sequence.'+name for name in CAMERAS.values()}:
            columns[key] = audit[key].to_numpy(zero_copy_only=False)
    if 'observation_sequence' not in columns:
        raise ValueError('new source lacks actual observation_sequence')
    segments, details = temporal_segments(columns, n, fps, POLICY['min_segment_frames'])
    safe = []
    seq = np.asarray(columns['observation_sequence'])
    if not np.isfinite(seq).all():
        raise ValueError('nonfinite actual observation sequences')
    for a, b in segments:
        b = min(b, n-3)
        if b-a < POLICY['min_segment_frames']:
            continue
        if not np.all(seq[a+3:b+3]-seq[a:b] == 3):
            raise ValueError('shift3 target crosses acquisition gap')
        lag = timestamps[a+3:b+3]-timestamps[a:b]
        if not np.all(np.isfinite(lag) & (lag >= 0.05-1e-6) & (lag <= 0.15+1e-6)):
            raise ValueError('shift3 target time outside permitted range')
        safe.append((a,b))
    details['retained_frames'] = sum(b-a for a,b in safe)
    details['final_unlabelable_observations'] = 3
    return safe, details


def inspect_selected(job):
    root, info, row, record = job
    inspection, state, source_action, videos = inspect_episode((root, info, row))
    if inspection['errors']:
        raise ValueError(f'{root}/{row["episode_index"]}: {inspection["errors"]}')
    if inspection['quaternion_max_norm_error'] > 0.01:
        raise ValueError(f'{root}/{row["episode_index"]}: malformed quaternion')
    idx = row['episode_index']
    fmt = {'episode_index': idx, 'episode_chunk': idx//info['chunks_size']}
    data_path = root/info['data_path'].format(**fmt)
    audit_path = root/info['audit']['path'].format(**fmt)
    if record['origin'] == 'new_middle':
        if not np.array_equal(state, source_action):
            raise ValueError(f'{root}/{idx}: raw action != same-frame state; stop for review')
        audit = pq.read_table(audit_path)
        if len(audit) != len(state) or not np.array_equal(audit['frame_index'].to_numpy(), np.arange(len(state))):
            raise ValueError('new audit index/length mismatch')
        if not np.all(audit['episode_index'].to_numpy() == idx):
            raise ValueError('new audit episode mismatch')
        timestamps = pq.read_table(data_path, columns=['timestamp'])['timestamp'].to_numpy()
        segments, temporal = new_temporal_segments(audit, timestamps, len(state), info['fps'])
        inspection['raw_action_equals_same_frame_state'] = True
        action = np.concatenate((state[3:], state[-1:].repeat(3,axis=0)))
        # Padded values are never exported; b <= n-3 is asserted below.
        additional_shift = 3
    else:
        audited = audit_one((root, info, row))
        segments, temporal = audited['segments'], audited['temporal']
        if segments != record['segments']:
            raise ValueError(f'{root}/{idx}: old temporal manifest changed')
        action = source_action
        additional_shift = 0
    a, b, crop = edge_crop(state)
    retained = [(max(a,x), min(b,y)) for x,y in segments
                if min(b,y)-max(a,x) >= POLICY['min_segment_frames']]
    if not retained:
        raise ValueError(f'{root}/{idx}: no usable segment after conservative crop; stop for review')
    if record['origin'] == 'new_middle':
        for x,y in retained:
            assert y <= len(state)-3 and np.array_equal(action[x:y], state[x+3:y+3])
    inspection.update(temporal=temporal, edge_crop=crop, segments=retained,
                      additional_action_shift_frames=additional_shift,
                      raw_numeric_sha256=inspection.pop('numeric_sha256'),
                      data_parquet_sha256=digest(data_path), audit_parquet_sha256=digest(audit_path))
    return inspection, state, action, videos


def prepare(args):
    out, old = args.output.resolve(), args.old_prepared.resolve()
    if (out/'prepared.json').exists() or (out/'episodes').exists():
        raise ValueError('output already contains a preparation; choose a new output directory')
    old_report = json.loads((old/'prepared.json').read_text())
    manifest = read_jsonl(old/'segment_manifest.jsonl')
    old_records = {}
    for item in manifest:
        key = (item['dataset'], int(item['episode_index']))
        record = old_records.setdefault(key, dict(dataset=key[0], episode_index=key[1],
            session=item['session'], original_split=item['split'], origin='old', segments=[]))
        if record['original_split'] != item['split']:
            raise ValueError('original episode appears in multiple splits')
        record['segments'].append((item['start'], item['end_exclusive']))
    jobs_by_key, all_records, candidate_train = {}, [], []
    source_hashes = {}
    roots = [(Path(path), 'old') for path in old_report['sources']]
    roots += [(p, 'new_middle') for p in sorted(args.new_root.resolve().iterdir()) if p.is_dir()]
    for root, origin in roots:
        info = json.loads((root/'meta/info.json').read_text())
        rows = read_jsonl(root/'meta/episodes.jsonl')
        if info['fps'] != 30 or len(rows) != info['total_episodes'] or sum(r['length'] for r in rows) != info['total_frames']:
            raise ValueError(f'{root}: metadata total/FPS mismatch')
        if len({r['episode_index'] for r in rows}) != len(rows):
            raise ValueError(f'{root}: duplicate episode indices')
        for name in ('observation.state', 'action'):
            if info['features'][name]['names'] != NAMES:
                raise ValueError(f'{root}: incompatible {name} layout')
        if origin == 'old' and info.get('cleaning',{}).get('action_shift_frames') != 3:
            raise ValueError(f'{root}: missing existing shift3 declaration')
        source_hashes[str(root)] = {str(path.relative_to(root)):digest(path)
            for path in (root/'meta/info.json',root/'meta/episodes.jsonl')}
        for row in rows:
            key = (str(root), row['episode_index'])
            if origin == 'old':
                if key not in old_records:
                    raise ValueError(f'{root}/{key[1]}: absent from original temporal split')
                record = dict(old_records[key])
            else:
                record = dict(dataset=str(root), episode_index=key[1], session=root.name,
                              original_split='new', origin=origin)
            record.update(verdict=row.get('operator_verdict'), source_frames=row['length'],
                          exclusion_reason=metadata_rejection(row))
            if origin == 'old' and session_key(row, str(root)) != record['session']:
                raise ValueError('session identity changed from original manifest')
            jobs_by_key[key] = (root, info, row, record)
            all_records.append(record)
            if record['original_split'] == 'train' and not record['exclusion_reason']:
                candidate_train.append(record)
    retained_keys = set()
    if args.retain_old_selection:
        retained_keys = {(r['dataset'], r['episode_index'])
                         for r in read_jsonl(args.retain_old_selection/'segment_manifest.jsonl')
                         if r['origin'] == 'old' and r['original_split'] == 'train'}
        eligible_keys = {(r['dataset'], r['episode_index']) for r in candidate_train}
        if not retained_keys <= eligible_keys or len(retained_keys) >= args.old_train_episodes:
            raise ValueError('retained selection must be eligible and smaller than the requested total')
        remaining = [r for r in candidate_train if (r['dataset'], r['episode_index']) not in retained_keys]
        added, _ = stratified_selection(remaining, args.old_train_episodes-len(retained_keys), args.seed)
        selected_train = [r for r in candidate_train if (r['dataset'], r['episode_index']) in retained_keys] + added
        quotas = dict(Counter(f"{r['dataset']}::{r['session']}" for r in selected_train))
    else:
        selected_train, quotas = stratified_selection(candidate_train, args.old_train_episodes, args.seed)
    selected_keys = {(r['dataset'],r['episode_index']) for r in selected_train}
    selected = []
    for record in all_records:
        key = record['dataset'], record['episode_index']
        if record['exclusion_reason']:
            continue
        if record['original_split'] == 'train' and key not in selected_keys:
            record['exclusion_reason'] = 'not_selected_old_train'
        else:
            selected.append(record)
    selected.sort(key=lambda r:(r['origin'],r['dataset'],r['episode_index']))
    assert len({(r['dataset'],r['episode_index']) for r in selected}) == len(selected)
    out.mkdir(parents=True,exist_ok=True)
    (out/'episodes').mkdir()
    (out/'arrays').mkdir()
    (out/'selection.json').write_text(json.dumps(dict(seed=args.seed, strata_quotas=quotas,
        selected_original_episode_count=len(selected), eligible_old_train_count=len(candidate_train),
        retained_old_train_episodes=len(retained_keys),
        retained_selection=str(args.retain_old_selection) if args.retain_old_selection else None), indent=2))
    with (out/'exclusions.jsonl').open('w') as file:
        for record in all_records:
            if record['exclusion_reason']:
                file.write(json.dumps(record)+'\n')
    counts, by_source, by_origin_split = Counter(), defaultdict(Counter), defaultdict(Counter)
    crop_status, crop_totals, quality = Counter(), Counter(), Counter()
    selected_lengths, hashes = [], []
    selected_jobs = [jobs_by_key[(r['dataset'],r['episode_index'])] for r in selected]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            (out/'episode_audit.jsonl').open('w') as auditfile, \
            (out/'segment_manifest.jsonl').open('w') as manifestfile:
        for start in range(0,len(selected_jobs),32):
            batch = selected_jobs[start:start+32]
            for job, result in zip(batch,pool.map(inspect_selected,batch)):
                root, info, row, record = job
                inspection, state, action, videos = result
                source = str(root)
                bucket = record['origin']+':'+record['original_split']
                counts['source_episodes'] += 1
                by_source[source]['source_episodes'] += 1
                by_origin_split[bucket]['source_episodes'] += 1
                crop_status[inspection['edge_crop']['status']] += 1
                crop_totals.update({k:inspection['edge_crop'][k] for k in ('removed_prefix','removed_suffix')})
                quality[row.get('quality_status','missing')] += 1
                episode_frames = 0
                auditfile.write(json.dumps({**record,**inspection})+'\n')
                for j,(a,b) in enumerate(inspection['segments']):
                    name = f'{root.name}_ep{row["episode_index"]:06d}_seg{j:03d}'
                    n = b-a
                    arrays = trajectory_arrays(state[a:b],action[a:b])
                    for key in list(arrays):
                        if key.startswith('actions.') and not key.startswith('actions.left_'):
                            arrays[key] = np.zeros_like(arrays[key])
                    array_path = out/'arrays'/f'{name}.npz'
                    np.savez(array_path,**arrays)
                    source_record = {key:record[key] for key in ('dataset','episode_index','session','origin','original_split','verdict')}
                    source_record.update(start=a, end_exclusive=b, action_shift_frames=3,
                        additional_action_shift_frames=inspection['additional_action_shift_frames'],
                        action_definition='raw_state[t+3]' if record['origin']=='new_middle' else 'preserved_already_shifted_source_action[t]',
                        edge_crop=inspection['edge_crop'])
                    views = {view:[dict(videos[view][0],start=a)] for view in CAMERAS}
                    trajectory = dict(trajectory_type='success', time=name, num_frames=n,
                        arrays_path=f'../arrays/{name}.npz', observations=views,source=source_record,
                        instruction={'general':[{'images':['observations.'+v for v in CAMERAS],
                            'conversations':[{'from':'human','value':'The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n'+PROMPT},{'from':'gpt','value':''}]}]})
                    (out/'episodes'/f'{name}.json').write_text(json.dumps(trajectory))
                    manifestfile.write(json.dumps(dict(name=name,split='train',frames=n,**source_record))+'\n')
                    hashes.append((name,digest(array_path)))
                    for target in (counts,by_source[source],by_origin_split[bucket]):
                        target['segments'] += 1
                        target['frames'] += n
                    episode_frames += n
                selected_lengths.append(episode_frames)
            auditfile.flush(); manifestfile.flush()
            print(json.dumps({'processed':min(start+32,len(selected_jobs)),'total':len(selected_jobs),
                              'counts':dict(counts)}),flush=True)
    # Check exported manifest independently before publishing the ready marker.
    exported = read_jsonl(out/'segment_manifest.jsonl')
    if any(r['verdict'] != 'success' for r in exported):
        raise ValueError('non-success episode reached output')
    if len(list((out/'episodes').glob('*.json'))) != counts['segments'] or sum(r['frames'] for r in exported) != counts['frames']:
        raise ValueError('output totals mismatch')
    shutil.copyfile(old/'normalize.json',out/'normalize.json')
    if digest(old/'normalize.json') != digest(out/'normalize.json'):
        raise ValueError('normalization changed')
    (out/'arrays_sha256.json').write_text(json.dumps(dict(hashes),indent=2))
    excluded = Counter(r['exclusion_reason'] for r in all_records if r['exclusion_reason'])
    excluded_by_split = Counter(f"{r['origin']}:{r['original_split']}:{r['exclusion_reason']}"
                               for r in all_records if r['exclusion_reason'])
    report = dict(ready=True, sources=[str(root) for root,_ in roots],
        old_prepared=str(old), new_root=str(args.new_root.resolve()), seed=args.seed,
        requested_old_train_episodes=args.old_train_episodes, raw_episodes=len(all_records),
        train_frames=counts['frames'], splits={'train':dict(counts)},
        counts_by_source={k:dict(v) for k,v in by_source.items()},
        counts_by_origin_split={k:dict(v) for k,v in by_origin_split.items()},
        excluded=dict(excluded), excluded_by_origin_split=dict(excluded_by_split),
        active_action_dims=list(range(7)), sampling='natural one sample per retained frame; no source balancing',
        quality_filtering='explicit invalid/integrity errors excluded; degraded is not a task verdict',
        verdict_filtering='only explicit success; unknown/failure always excluded',
        edge_crop_policy=POLICY, edge_crop_status=dict(crop_status), raw_edge_crop_totals=dict(crop_totals),
        selected_quality_status=dict(quality),
        retained_frames_per_episode_quantiles={str(q):float(np.quantile(selected_lengths,q)) for q in (0,.1,.5,.9,1)},
        temporal_policy=old_report['temporal_policy'],
        action_definition='old source actions preserved without shift; new raw verified action==state, then state[t+3] within continuous raw segments',
        normalization_source=str(old/'normalize.json'), normalization_sha256=digest(out/'normalize.json'),
        source_metadata_sha256=source_hashes, original_manifest_sha256=digest(old/'segment_manifest.jsonl'),
        preparation_script_sha256=digest(Path(__file__)),
        manifest_sha256=digest(out/'segment_manifest.jsonl'),
        former_holdouts_consumed=True, heldout_evaluation_available=False,
        limits=['No new held-out split: original validation/test intentionally included in training.',
                'New timestamps are nominal frame timestamps; observation/camera counters verify adjacency, not hardware exposure timestamps.',
                'All three training video headers checked; deterministic distributed first-frame decode samples, not full video decoding.',
                'Window motion is conservative and preserves ambiguous/all-static successes for inspection.'])
    (out/'prepared.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--old-prepared',type=Path,default=base/'data/ethernet_temporal_left_v1')
    parser.add_argument('--new-root',type=Path,default=Path('/mnt/cfs/7rnh3z/kele/1012/middle'))
    parser.add_argument('--output',type=Path,default=base/'data/ethernet_middle_mixed_20261013')
    parser.add_argument('--old-train-episodes',type=int,default=1500)
    parser.add_argument('--retain-old-selection',type=Path,
                        help='Preserve every old-training episode in an earlier mixed preparation and add more')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=8)
    prepare(parser.parse_args())
