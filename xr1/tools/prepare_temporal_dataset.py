#!/usr/bin/env python3
"""Temporal-only segmentation and session-disjoint splits; no quality/verdict filtering."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from prepare_ethernet_dataset import CAMERAS, canonical_arrays, trajectory_arrays
from prepare_network_dataset import Statistics


def temporal_segments(columns, n, fps=30, min_frames=30):
    """Use acquisition proxies, not Cartesian jumps or success/quality labels.

    Cut observation/target gaps >1.5 nominal periods or nonmonotonic times;
    observation sequence and raw row gaps must be exactly one. Camera skips
    of <=3 samples are counted but retained; larger skips/reversals cut, and
    >=3 consecutive repeats mask the repeated frames. Guard shift3 targets
    before every internal cut so no action label crosses the cut.
    """
    cuts=set();invalid=np.zeros(n,dtype=bool);counts=Counter();evidence=[]
    def column(key):
        if key not in columns:return None
        x=np.asarray(columns[key],dtype=np.float64)
        if not np.isfinite(x).any():return None
        evidence.append(key);return x
    def transitions(key, kind):
        x=column(key)
        if x is None:return
        finite=np.isfinite(x[:-1]) & np.isfinite(x[1:]);d=np.diff(x)
        if kind=='sequence':bad=finite & (d!=1)
        else:bad=finite & ((d<=0) | (d>1.5/fps))
        indices=np.flatnonzero(bad)+1;cuts.update(indices.tolist());counts[key+':cuts']=len(indices)
    for k in ['source_frame_index','source_action_frame_index','action_source_frame_index','observation_sequence','action_observation_sequence']:
        transitions(k,'sequence')
    for k in ['source_timestamp','source_action_timestamp','source_timestamp_utc','source_action_timestamp_utc']:
        transitions(k,'time')
    for left,right,lo,hi,label in [
        ('source_frame_index','source_action_frame_index',3,3,'row_shift'),
        ('source_frame_index','action_source_frame_index',3,3,'row_shift'),
        ('observation_sequence','action_observation_sequence',3,3,'observation_shift'),
        ('source_timestamp_utc','source_action_timestamp_utc',1.5/fps,4.5/fps,'utc_shift'),
    ]:
        a,b=column(left),column(right)
        if a is None or b is None:continue
        finite=np.isfinite(a)&np.isfinite(b);d=b-a
        bad=finite & ((d<lo-1e-6)|(d>hi+1e-6));invalid|=bad;counts[label+':invalid_rows']+=int(bad.sum())
    for key in sorted(k for k in columns if k.startswith('camera_sample_sequence.')):
        x=column(key)
        if x is None:continue
        d=np.diff(x);finite=np.isfinite(x[:-1])&np.isfinite(x[1:])
        bad=finite & ((d<0)|(d>3));cuts.update((np.flatnonzero(bad)+1).tolist())
        counts[key+':cuts']=int(bad.sum());counts[key+':small_skips']=int((finite&(d>1)&(d<=3)).sum())
        repeated=finite&(d==0);edges=np.flatnonzero(np.diff(np.r_[False,repeated,False]))
        for a,b in zip(edges[::2],edges[1::2]):
            if b-a>=3:invalid[a+1:b+1]=True;counts[key+':long_repeat_rows']+=int(b-a)
    # Invalid rows become boundaries in addition to acquisition discontinuities.
    for i in np.flatnonzero(invalid):cuts.add(int(i));cuts.add(int(i)+1)
    boundaries=sorted({0,n,*cuts});segments=[];short=0;guarded=0
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        if start==end or invalid[start:end].any():continue
        safe_end=end-3 if end<n else end
        guarded+=min(3,end-start) if end<n else 0
        if safe_end-start>=min_frames:segments.append((start,safe_end))
        else:short+=max(0,safe_end-start)
    return segments,{'events':dict(counts),'cut_count':len(cuts-{0,n}),
        'invalid_rows':int(invalid.sum()),'shift_guard_rows':guarded,'short_fragment_rows':short,
        'evidence':sorted(set(evidence)),'retained_frames':sum(b-a for a,b in segments)}


def session_key(row, source):
    raw=row.get('original_source_session') or row.get('source_session')
    if not raw:raise ValueError(f'{source}: missing acquisition-session identity')
    found=re.search(r'sess_[^/]+',raw)
    return found.group() if found else 'coarse_group:'+raw


def split_sessions(records, seed=42):
    # Assign whole groups. Weight by retained frames; aim for 90/5/5 per source.
    # Prefer several small sessions; allow one session where all sessions are
    # larger than the nominal 5% target. Coarse source-day groups stay in train.
    groups=defaultdict(list)
    for r in records:
        if r['segments']:groups[r['session']].append(r)
    assignment={g:'train' for g in groups}
    source_totals=Counter()
    for r in records:source_totals[r['dataset']]+=r['temporal']['retained_frames']
    for source,total in source_totals.items():
        candidates=[g for g,rs in groups.items() if all(r['dataset']==source for r in rs)]
        candidates.sort(key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        target=total*0.05
        weights={g:sum(r['temporal']['retained_frames'] for r in groups[g]) for g in candidates}
        cap=max(target*0.6,min(weights.values())*1.1)
        for split in ('validation','test'):
            have=0
            for g in candidates:
                weight=weights[g]
                if assignment[g]=='train' and weight<=cap and (have==0 or abs(target-have-weight)<abs(target-have)):
                    assignment[g]=split;have+=weight
            if have==0:raise ValueError(f'cannot form a session-disjoint {split} split for {source}')
    return assignment


def audit_one(job):
    root,info,row=job;i=int(row['episode_index']);fmt={'episode_index':i,'episode_chunk':i//int(info['chunks_size'])}
    table=pq.read_table(root/info['audit']['path'].format(**fmt));n=int(row['length'])
    if len(table)!=n or not np.array_equal(table['frame_index'].to_numpy(),np.arange(n)):
        raise ValueError(f'{root}/{i}: audit index/length mismatch')
    keys=[k for k in table.column_names if k in ['source_frame_index','source_action_frame_index','action_source_frame_index',
        'observation_sequence','action_observation_sequence','source_timestamp','source_action_timestamp',
        'source_timestamp_utc','source_action_timestamp_utc'] or k.startswith('camera_sample_sequence.')]
    columns={k:table[k].to_numpy(zero_copy_only=False) for k in keys}
    segments,details=temporal_segments(columns,n,float(info['fps']))
    return {'dataset':str(root),'episode_index':i,'frames':n,'session':session_key(row,str(root)),
        'segments':segments,'temporal':details}


def prepare(args):
    out=args.output.resolve()
    if (out/'prepared.json').exists():raise ValueError('choose a new output directory; existing preparation is immutable')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for root in args.datasets:
        root=root.resolve();info=json.loads((root/'meta/info.json').read_text())
        rows=[json.loads(l) for l in (root/'meta/episodes.jsonl').open()]
        jobs.extend((root,info,r) for r in rows)
    records=[]
    with ThreadPoolExecutor(args.workers) as pool,(out/'temporal_audit.jsonl').open('w') as audit:
        for start in range(0,len(jobs),64):
            for r in pool.map(audit_one,jobs[start:start+64]):
                records.append(r);audit.write(json.dumps(r)+'\n')
            audit.flush()
            if start%640==0:print('temporal audit',len(records),'/',len(jobs),flush=True)
    assignments=split_sessions(records,args.seed)
    (out/'session_splits.json').write_text(json.dumps(assignments,indent=2))
    stats=Statistics(30);counts={s:Counter() for s in ['train','validation','test']};event_counts=Counter()
    source_counts=defaultdict(Counter)
    for s in counts:(out/s/'episodes').mkdir(parents=True,exist_ok=True)
    (out/'arrays').mkdir(exist_ok=True)
    with (out/'segment_manifest.jsonl').open('w') as manifest:
        for count,(job,r) in enumerate(zip(jobs,records),1):
            root,info,row=job;event_counts.update(r['temporal']['events'])
            if not r['segments']:continue
            split=assignments[r['session']];i=r['episode_index'];fmt={'episode_index':i,'episode_chunk':i//int(info['chunks_size'])}
            table=pq.read_table(root/info['data_path'].format(**fmt),columns=['observation.state','action'])
            state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
            action=np.asarray(table['action'].to_pylist(),dtype=np.float32)
            if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('unexpected nonfinite training input')
            counts[split]['source_episodes']+=1
            for j,(a,b) in enumerate(r['segments']):
                name=f'{root.name}_ep{i:06d}_seg{j:03d}';n=b-a
                arrays=trajectory_arrays(state[a:b],action[a:b])
                # Keep the checkpoint's 60-D layout, but no inactive action targets.
                for key in list(arrays):
                    if key.startswith('actions.') and not key.startswith('actions.left_'):arrays[key]=np.zeros_like(arrays[key])
                np.savez(out/'arrays'/f'{name}.npz',**arrays)
                videos={v:[{'path':str(root/info['video_path'].format(**fmt,video_key='observation.images.'+cam)),
                    'start':a,'crop_bbox':None}] for v,cam in CAMERAS.items()}
                prompt=row.get('prompt') or info['prompt']
                trajectory={'trajectory_type':'ongoing','time':name,'num_frames':n,'arrays_path':f'../../arrays/{name}.npz',
                    'observations':videos,'source':{'dataset':str(root),'episode_index':i,'start':a,'end_exclusive':b,'session':r['session']},
                    'instruction':{'general':[{'images':['observations.'+v for v in CAMERAS],
                    'conversations':[{'from':'human','value':'The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n'+prompt},{'from':'gpt','value':''}]}]}}
                (out/split/'episodes'/f'{name}.json').write_text(json.dumps(trajectory))
                manifest.write(json.dumps({'name':name,'split':split,**trajectory['source'],'frames':n})+'\n')
                counts[split]['segments']+=1;counts[split]['frames']+=n;source_counts[str(root)][split]+=n
                if split=='train':stats.add(canonical_arrays(state[a:b],action[a:b]),joints_in_degrees=True)
            if count%400==0:print('converted',count,'/',len(jobs),{k:dict(v) for k,v in counts.items()},flush=True)
    normalization=stats.result()
    for key in ['mean','std']:
        values=np.asarray(normalization[key]);values[:,7:]=0;normalization[key]=values.tolist()
    (out/'normalize.json').write_text(json.dumps(normalization))
    report={'sources':[str(p.resolve()) for p in args.datasets],'raw_episodes':len(records),
        'raw_frames':sum(r['frames'] for r in records),'quality_filtering':False,'verdict_filtering':False,
        'splits':{k:dict(v) for k,v in counts.items()},'frames_by_source':dict(source_counts),
        'dropped_episodes_no_30_frame_segment':sum(not r['segments'] for r in records),
        'removed_frames_temporal_or_short_fragments':sum(r['frames']-r['temporal']['retained_frames'] for r in records),
        'temporal_events':dict(event_counts),'active_action_dims':list(range(7)),
        'actions':'left relative EE xyz + axis-angle + relative left gripper; others masked',
        'normalization_source':'train split only','split_seed':args.seed,'split_unit':'original session, or coarse source-day when session unavailable',
        'temporal_policy':{'max_adjacent_time_s':0.05,'action_shift_utc_s':[0.05,0.15],
            'max_camera_sequence_skip':3,'camera_repeat_invalid_at':3,'min_segment_frames':30,'internal_cut_action_guard_frames':3},
        'limits':'No visual/task-quality audit. Sample counters are temporal proxies, not proof of RGB hardware synchronization.'}
    (out/'prepared.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('datasets',type=Path,nargs='+')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    prepare(p.parse_args())
