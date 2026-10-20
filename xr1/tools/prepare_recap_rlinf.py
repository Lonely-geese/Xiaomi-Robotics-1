#!/usr/bin/env python3
"""Prepare new demonstration/rollout pools and native RLinf RGB inputs."""
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
import torch

from mibot.data.recap import action_intervention_mask, episode_returns
from mibot.models.rlinf_recap.value_common.image_text_processing import resize_with_pad
from prepare_ethernet_dataset import CAMERAS, NAMES, trajectory_arrays
from prepare_temporal_dataset import temporal_segments


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_one(job):
    root, info, row, pool, out = job
    index, n = int(row['episode_index']), int(row['length'])
    name = f'{root.name}_ep{index:06d}'
    done = out/'records'/f'{name}.json'
    if done.exists():
        return json.loads(done.read_text())
    fmt = dict(episode_index=index, episode_chunk=index//info['chunks_size'])
    table = pq.read_table(root/info['data_path'].format(**fmt))
    audit = pq.read_table(root/info['audit']['path'].format(**fmt))
    for t in (table, audit):
        if len(t) != n or not np.array_equal(t['frame_index'].to_numpy(), np.arange(n)):
            raise ValueError(f'{name}: noncontiguous source/audit frames')
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if state.shape != (n, 30) or action.shape != state.shape or not np.isfinite([state, action]).all():
        raise ValueError(f'{name}: invalid state/action')
    outcome = row.get('outcome', row.get('operator_verdict'))
    labeled = outcome in ('success', 'failure')
    if pool == 'sft' and outcome != 'success':
        raise ValueError(f'{name}: RLinf demonstration returns require known success')
    rewards, returns = episode_returns(n, outcome == 'success') if labeled else (
        np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32))
    human = np.ones(n, np.int8)
    if pool == 'rollout':
        expected = action_intervention_mask(row)
        human = np.array([-1 if x is None else int(x) for x in audit['action_human_intervened'].to_pylist()], np.int8)
        known = expected >= 0
        if not np.array_equal(human[known], expected[known]):
            raise ValueError(f'{name}: intervention intervals disagree with audit')
    numeric = out/'critic'/f'{name}.npz'
    np.savez(numeric, state=state, rewards=rewards, returns=returns, human=human)
    columns = {k:audit[k].to_numpy(zero_copy_only=False) for k in audit.column_names
        if k in ('source_frame_index','source_action_frame_index','action_source_frame_index',
                 'observation_sequence','action_observation_sequence','source_timestamp',
                 'source_action_timestamp','source_timestamp_utc','source_action_timestamp_utc')
        or k.startswith('camera_sample_sequence.')}
    segments, temporal = temporal_segments(columns, n, info['fps'], min_frames=1)
    videos = {view:root/info['video_path'].format(**fmt, video_key='observation.images.'+camera)
              for view,camera in CAMERAS.items()}
    prompt = row.get('prompt') or row['tasks'][0]
    for j,(a,b) in enumerate(segments):
        segment_name = f'{name}_seg{j:03d}'
        arrays = out/'arrays'/f'{segment_name}.npz'
        np.savez(arrays, **trajectory_arrays(state[a:b], action[a:b]))
        traj = dict(trajectory_type='ongoing', time=segment_name, num_frames=b-a,
            arrays_path=str(arrays), observations={v:[dict(path=str(p),start=a,crop_bbox=None)] for v,p in videos.items()},
            source=dict(dataset=str(root),episode_index=index,start=a,end_exclusive=b),
            recap=dict(pool=pool,sidecar=str(out/'advantages'/f'{name}.npz'),offset=a),
            instruction={'general':[{'images':['observations.'+v for v in CAMERAS],
                'conversations':[{'from':'human','value':'The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n'+prompt},{'from':'gpt','value':''}]}]})
        (out/pool/'episodes'/f'{segment_name}.json').write_text(json.dumps(traj))
    images = out/'images'/f'{name}.npy'
    if not images.exists():
        packed = np.empty((n,3,3,224,224), dtype=np.uint8)
        for vi,path in enumerate(videos.values()):
            with av.open(str(path)) as container:
                container.streams.video[0].thread_count = 1
                batch, offset = [], 0
                def flush():
                    nonlocal offset
                    if not batch:return
                    tensor = torch.from_numpy(np.stack(batch)).permute(0,3,1,2)
                    resized = resize_with_pad(tensor,224,224)
                    packed[offset:offset+len(batch),vi] = resized.numpy()
                    offset += len(batch)
                    batch.clear()
                for frame in container.decode(video=0):
                    if offset+len(batch) >= n:raise ValueError(f'{name}: extra RGB frames')
                    batch.append(frame.to_ndarray(format='rgb24'))
                    if len(batch)==32:flush()
                flush()
                if offset!=n:raise ValueError(f'{name}: RGB length {offset} != {n}')
        temporary = images.with_suffix('.partial.npy')
        np.save(temporary, packed)
        temporary.replace(images)
    record = dict(name=name,dataset=str(root),episode_index=index,pool=pool,length=n,
        outcome=outcome,labeled=labeled,prompt=prompt,numeric=str(numeric),images=str(images),
        policy_frames=sum(b-a for a,b in segments),policy_segments=len(segments),temporal=temporal,
        human_frames=int((human==1).sum()),unknown_intervention_frames=int((human<0).sum()))
    temporary = done.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record));temporary.replace(done)
    return record


def main(args):
    torch.set_num_threads(1)
    out=args.output.resolve()
    if (out/'prepared.json').exists():
        print('Preparation already complete:',out,flush=True);return
    for sub in ('sft/episodes','rollout/episodes','critic','arrays','images','advantages','records'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    sources=[args.sft.resolve(),args.rollout.resolve()]
    jobs=[];metadata={}
    for root,pool in zip(sources,('sft','rollout')):
        info=json.loads((root/'meta/info.json').read_text())
        if info['fps']!=30 or info['features']['observation.state']['names']!=NAMES or info['cleaning']['action_shift_frames']!=3:
            raise ValueError(f'{root}: incompatible source schema')
        rows=[json.loads(l) for l in (root/'meta/episodes.jsonl').open()]
        if len(rows)!=info['total_episodes'] or sum(r['length'] for r in rows)!=info['total_frames']:
            raise ValueError(f'{root}: metadata totals disagree')
        if any(any(r.get(k) is False for k in ('data_valid','structurally_valid')) or r.get('integrity_error_code') for r in rows):
            raise ValueError(f'{root}: explicit invalid metadata requires review')
        metadata[str(root)]=digest(root/'meta/episodes.jsonl')
        jobs.extend((root,info,r,pool,out) for r in rows)
    records=[];counts=Counter()
    with ThreadPoolExecutor(args.workers) as executor,(out/'episodes.jsonl').open('w') as manifest:
        for i,r in enumerate(executor.map(prepare_one,jobs),1):
            records.append(r);manifest.write(json.dumps(r)+'\n');manifest.flush()
            counts[r['pool']+'_episodes']+=1
            counts[r['pool']+'_frames']+=r['policy_frames']
            counts[r['pool']+'_segments']+=r['policy_segments']
            if i%10==0:print(json.dumps(dict(processed=i,total=len(jobs),counts=dict(counts))),flush=True)
    shutil.copyfile(args.normalization,out/'normalize.json')
    report=dict(ready=True,**counts,sources=list(metadata),source_metadata_sha256=metadata,
        normalization_source=str(args.normalization.resolve()),normalization_sha256=digest(out/'normalize.json'),
        return_scale=-min(float(np.load(r['numeric'])['returns'].min()) for r in records if r['labeled']),
        gamma=1.0,failure_reward=-300.0,horizon=10,positive_fraction=0.3,image_size=224,
        critic='unmodified RLinf SigLIP2-so400m + Gemma3-270M + gemma_1m value expert + CLS + 201-bin head',
        label_mode='rlinf_global_quantile',human_label_override=False,
        unknown_outcome_policy='excluded from critic and threshold; unconditional policy samples',
        epoch_definition='larger pool once, smaller pool cycled; complete global half batches',
        action_shift_applied=0,independent_policy_evaluation=False,
        preprocessing='native RLinf resize_with_pad uint8 RGB to 224x224; native processor normalizes to [-1,1]')
    (out/'prepared.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[1]
    source=Path('/mnt/cfs/7rnh3z/kele/data/00_traindata')
    p.add_argument('--sft',type=Path,default=source/'20261016_20261019_1563')
    p.add_argument('--rollout',type=Path,default=source/'recap_755')
    p.add_argument('--output',type=Path,default=base/'data/recap_rlinf_1563_20261020')
    p.add_argument('--normalization',type=Path,default=base/'data/ethernet_middle_mixed_old2000_20261013/normalize.json')
    p.add_argument('--workers',type=int,default=12)
    main(p.parse_args())
