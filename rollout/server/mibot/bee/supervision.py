"""Verified USB correction pairs in the frozen XR1 action coordinates."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from mibot.bee.data import robot_state
from mibot.utils.io import (ACTION_EPS, aa2rotm, compose_state, normalize_quantile,
                            rotm2aa_batch)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def human_to_normalized(observation, human, reference, norm):
    """Absolute base-frame pose -> current-tool delta -> per-offset z-score.

    The gripper is not a human label. Its value is copied only as a placeholder
    and MUST be excluded by the correction likelihood's editable mask.
    """
    state = robot_state(observation)
    rotation = state['left_ee_rotm']
    relative = np.empty_like(reference)
    relative[:, :3] = (human[:, :3]-state['left_ee_pos']) @ rotation
    relative[:, 3:6] = rotm2aa_batch(np.stack([rotation.T @ aa2rotm(a) for a in human[:, 3:6]]))
    target = reference.copy()
    target[:, :6] = (relative[:, :6]-norm['mean'][:len(human), :6])/(norm['std'][:len(human), :6]+ACTION_EPS)
    packed = compose_state(state['left_gripper_pos'], state['left_arm_joint'],
                           state['right_gripper_pos'], state['right_arm_joint'])
    proprio = normalize_quantile(packed, norm['q01'], norm['q99'])[0, :16]
    return target, proprio


def prepare(cache_root, dataset, weights, seed=42):
    cache_root, dataset, weights = map(Path, (cache_root, dataset, weights))
    norm = {k: np.array(v, dtype=np.float32) for k,v in json.loads((weights/'normalize.json').read_text()).items()}
    manifest = json.loads((weights/'manifest.json').read_text())['files_sha256']
    if sha256(weights/'normalize.json') != manifest['normalize.json']:
        raise ValueError('Normalization hash mismatch')
    rows, inventory, counts, seen = [], [], Counter(), set()
    for report_path in sorted(cache_root.glob('shard_*/report.json')):
        report = json.loads(report_path.read_text())
        if report['status'] != 'complete' or report['profile'] != 'direct-cache' or report['selection']['kind'] != 'human-chunks':
            raise ValueError('Training requires completed observation-only direct-cache human chunks')
        if Path(report['dataset']).resolve() != dataset.resolve() or report['checkpoint_sha256'] != manifest:
            raise ValueError('Feature provenance mismatch')
        for filename in report['episodes_written']:
            path = report_path.parent/filename
            inventory.append({'path':str(path.resolve()), 'sha256':sha256(path)})
            with np.load(path) as a:
                eid = int(a['episode_index'][0])
                source_path = dataset/f'data/chunk-000/episode_{eid:06d}.parquet'
                table = pq.read_table(source_path).to_pydict()
                inventory.append({'path':str(source_path.resolve()), 'sha256':sha256(source_path)})
                for i in range(len(a['z'])):
                    counts['all_windows'] += 1
                    frame = int(a['frame_index'][i]); n = int(a['valid_horizon'][i])
                    if (eid, frame) in seen:
                        raise ValueError('Duplicate episode/frame pair')
                    seen.add((eid, frame))
                    assert np.array_equal(np.array(table['action'][frame:frame+n],np.float32), a['human_action6'][i,:n])
                    assert np.array_equal(np.array(table['observation.state'][frame],np.float32), a['observation'][i])
                    assert all(x == 2 for x in table['annotation.source'][frame:frame+n])
                    assert all(table['annotation.intervention'][frame:frame+n])
                    assert np.array_equal(table['annotation.drop_transition'][frame:frame+n],a['drop_transition'][i,:n])
                    if a['drop_transition'][i,:n].any():
                        counts['excluded_dropped_windows'] += 1
                        continue
                    if n != 10:
                        counts['excluded_partial_windows'] += 1
                        continue
                    camera_fields = [k for k in table if k.startswith('observation.camera_valid.')]
                    if not all(table[k][frame] for k in camera_fields):
                        counts['excluded_invalid_camera_windows'] += 1
                        continue
                    ref = a['normalized_action'][i,:10].copy()
                    human, proprio = human_to_normalized(a['observation'][i], a['human_action6'][i,:10],ref,norm)
                    rows.append(dict(z=a['z'][i].copy(), proprio=proprio, reference=ref, human_action=human,
                                     observation=a['observation'][i].copy(), human_action6=a['human_action6'][i,:10].copy(),
                                     legacy_corrected=a['bee_normalized_action'][i].copy(), episode=eid, frame=frame))
    if not rows:
        raise ValueError('No complete verified human chunks')
    data = {k:torch.as_tensor(np.stack([r[k] for r in rows])) for k in rows[0]}
    for k,v in data.items():
        if not torch.isfinite(v).all():
            raise ValueError(f'Nonfinite training feature: {k}')
    episodes = np.unique(data['episode'].numpy())
    np.random.default_rng(seed).shuffle(episodes)
    nval = max(1, len(episodes)//10)
    groups = dict(train=sorted(episodes[2*nval:].tolist()), validation=sorted(episodes[nval:2*nval].tolist()), test=sorted(episodes[:nval].tolist()))
    splits = {k:torch.where(torch.isin(data['episode'],torch.tensor(ids)))[0] for k,ids in groups.items()}
    counts['usable_windows'] = len(rows)
    return data, splits, dict(counts=dict(counts), episodes=groups, split_windows={k:len(v) for k,v in splits.items()},
                             split_seed=seed, inputs=inventory, feature_equivalence='unverified-reconstruction',
                             normalization_sha256=manifest['normalize.json'],
                             note='Full 10-step non-dropped human chunks only. No measured gripper supervision. Episode split; original prewarm overlap unknown.'), norm
