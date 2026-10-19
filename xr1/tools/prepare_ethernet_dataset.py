#!/usr/bin/env python3
"""Audit and adapt canonical 30-D LeRobot Ethernet datasets without modifying sources.

Source joints are degrees, XYZ meters, and quaternions XYZW. Actions have
already been shifted by three frames: this converter applies NO time shift.
Unknown verdicts are held out by default, independently of numeric integrity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from prepare_network_dataset import Statistics, quaternion_xyzw_to_matrix

CAMERAS = {'ego': 'headDepthCamera', 'wrist_left': 'leftCamera', 'wrist_right': 'rightCamera'}
NAMES = [f'{side}arm_state_{kind}_pos_{i}' for side in ('left', 'right')
         for kind in ('joint', 'cart') for i in range(7)]
NAMES.insert(14, 'leftarm_gripper_state_pos')
NAMES.append('rightarm_gripper_state_pos')

def canonical_arrays(state, action):
    # Existing normalization code expects [L joint, L pose, R joint, R pose, L grip, R grip].
    return {
        'frame_index': np.arange(len(state)),
        'observation.state.left_arm_joint': state[:, :7],
        'observation.state.left_arm_cart': state[:, 7:14],
        'observation.state.left_gripper_pos': state[:, 14:15],
        'observation.state.right_arm_joint': state[:, 15:22],
        'observation.state.right_arm_cart': state[:, 22:29],
        'observation.state.right_gripper_pos': state[:, 29:30],
        'action': action[:, list(range(14)) + list(range(15,29)) + [14,29]],
    }

def trajectory_arrays(state, action):
    arrays = {}
    for section, values in [('proprios', state), ('actions', action)]:
        for side, offset in [('left', 0), ('right', 15)]:
            arrays[f'{section}.{side}_ee_pos'] = values[:, offset+7:offset+10]
            arrays[f'{section}.{side}_ee_rotm'] = quaternion_xyzw_to_matrix(values[:, offset+10:offset+14]).reshape(-1,9).astype(np.float32)
            arrays[f'{section}.{side}_gripper_pos'] = values[:, offset+14:offset+15]
            if section == 'proprios':
                arrays[f'{section}.{side}_arm_joint'] = np.deg2rad(values[:, offset:offset+7]).astype(np.float32)
        arrays[f'{section}.waist_pos'] = np.zeros((len(values),1), dtype=np.float32)
    arrays['actions.base_vel'] = np.zeros((len(state),3), dtype=np.float32)
    return arrays

def inspect_episode(job):
    root, info, episode = job
    index = int(episode['episode_index'])
    substitutions = {'episode_index': index, 'episode_chunk': index//int(info['chunks_size'])}
    path = root / info['data_path'].format(**substitutions)
    result = {'dataset': str(root), 'episode_index': index, 'length': episode['length'],
              'operator_verdict': episode.get('operator_verdict'), 'errors': [], 'warnings': [],
              'source_session': episode.get('original_source_session', episode.get('source_session'))}
    state = action = None
    videos = {}
    try:
        table = pq.read_table(path)
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        count = len(state)
        if state.shape != (count,30) or action.shape != (count,30): raise ValueError('state/action shape mismatch')
        if count != episode['length'] or count < 30: raise ValueError('length mismatch or fewer than 30 frames')
        if not np.isfinite(state).all() or not np.isfinite(action).all(): raise ValueError('nonfinite state/action')
        if not np.array_equal(table['frame_index'].to_numpy(),np.arange(count)): raise ValueError('noncontiguous frame indices')
        if not np.all(table['episode_index'].to_numpy()==index): raise ValueError('episode index mismatch')
        timestamps=table['timestamp'].to_numpy()
        if np.any(np.diff(timestamps)<=0): raise ValueError('nonincreasing timestamps')
        result['timestamp_grid_max_error_s']=float(np.max(np.abs(timestamps-np.arange(count)/info['fps'])))
        result['action_equals_state_plus3']=bool(np.array_equal(action[:-3],state[3:]))
        if not result['action_equals_state_plus3']: result['warnings'].append('action differs from state[t+3]')
        q = np.concatenate([values[:, start:start+4] for values in (state,action) for start in (10,25)])
        norms = np.linalg.norm(q,axis=1)
        if np.any(norms<1e-8): raise ValueError('zero quaternion')
        result['quaternion_max_norm_error']=float(np.max(np.abs(norms-1)))
        if result['quaternion_max_norm_error'] > 0.01: result['warnings'].append('quaternion norm deviation > 0.01')
        result['max_position_step_m']=max(float(np.linalg.norm(np.diff(state[:,s:s+3],axis=0),axis=1).max()) for s in (7,22))
        if result['max_position_step_m']>0.1: result['warnings'].append('position step > 10 cm; inspect temporal cuts')
        result['gripper_min']=float(min(state[:,[14,29]].min(),action[:,[14,29]].min()))
        result['gripper_max']=float(max(state[:,[14,29]].max(),action[:,[14,29]].max()))
        result['numeric_sha256']=hashlib.sha256(state.tobytes()+action.tobytes()).hexdigest()
        for view, camera in CAMERAS.items():
            key='observation.images.'+camera
            video=root / info['video_path'].format(**substitutions,video_key=key)
            with av.open(str(video)) as container:
                stream=container.streams.video[0]
                if stream.frames != count: raise ValueError(f'{camera} container frame count {stream.frames} != {count}')
                if abs(float(stream.average_rate)-info['fps'])>0.01: raise ValueError(f'{camera} FPS mismatch')
                expected=info['features'][key]['shape']
                if (stream.height,stream.width)!=tuple(expected[:2]): raise ValueError(f'{camera} dimensions mismatch')
                # Deterministic, distributed decode sample; headers checked for every video.
                if index % 200 == 0 or index == info['total_episodes']-1:
                    first=next(container.decode(stream))
                    if first.to_ndarray(format='rgb24').shape[:2]!=tuple(expected[:2]): raise ValueError('decoded frame dimensions mismatch')
            videos[view]=[{'path':str(video.resolve()),'start':0,'crop_bbox':None}]
        result['video_headers_checked']=3
        result['sample_decoded_videos']=3 if index%200==0 or index==info['total_episodes']-1 else 0
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    if episode.get('structurally_valid') is False or episode.get('data_valid') is False:
        result['errors'].append('source explicitly marks episode invalid')
    if episode.get('quality_status') == 'degraded': result['warnings'].append('source quality_status=degraded')
    return result,state,action,videos

def prepare(args):
    out=args.output.resolve()
    if (out/'prepared.json').exists():
        raise SystemExit(f'{out} is already prepared; choose a new output directory to avoid stale entries')
    (out/'episodes').mkdir(parents=True,exist_ok=True)
    (out/'arrays').mkdir(exist_ok=True)
    jobs=[]
    for root in args.datasets:
        root=root.resolve();info=json.loads((root/'meta/info.json').read_text())
        for key in ('observation.state','action'):
            if info['features'][key]['names']!=NAMES: raise ValueError(f'{root}: unexpected {key} layout')
        if info.get('cleaning',{}).get('action_shift_frames')!=3: raise ValueError('expected already shifted actions')
        episodes=[json.loads(line) for line in (root/'meta/episodes.jsonl').open()]
        if len(episodes)!=info['total_episodes'] or sum(e['length'] for e in episodes)!=info['total_frames']:
            raise ValueError('metadata totals mismatch')
        jobs.extend((root,info,e) for e in episodes)
    stats=Statistics(30);summary=Counter();digests={};duplicates=[]
    with (out/'audit.jsonl').open('w') as audit, ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Bound retained numeric arrays while sharing filesystem I/O across threads.
        for start in range(0,len(jobs),32):
            batch=jobs[start:start+32]
            for job, (result,state,action,videos) in zip(batch,pool.map(inspect_episode,batch)):
                root,info,episode=job
                summary['episodes_scanned']+=1;summary['frames_scanned']+=episode['length']
                summary['video_headers_checked']+=result.get('video_headers_checked',0)
                summary['sample_decoded_videos']+=result.get('sample_decoded_videos',0)
                if result['warnings']: summary['episodes_with_warnings']+=1
                identity=f'{root.name}/episode_{episode["episode_index"]:06d}'
                digest=result.get('numeric_sha256')
                if digest:
                    if digest in digests: duplicates.append([digests[digest],identity])
                    else: digests[digest]=identity
                verdict=episode.get('operator_verdict')
                if result['errors']:
                    result['selection']='excluded_integrity';summary['excluded_integrity']+=1
                elif verdict!='success' and not (args.include_unlabeled and verdict is None):
                    result['selection']='held_out_verdict';summary['held_out_verdict']+=1
                else:
                    result['selection']='train';summary['train_episodes']+=1;summary['train_frames']+=len(state)
                    name=f'{root.name}_episode_{episode["episode_index"]:06d}'
                    np.savez(out/'arrays'/f'{name}.npz',**trajectory_arrays(state,action))
                    prompt=episode.get('prompt') or info['prompt']
                    trajectory={'trajectory_type':'success' if verdict=='success' else 'ongoing','time':name,'num_frames':len(state),
                        'arrays_path':f'../arrays/{name}.npz','observations':videos,
                        'source':{'dataset':str(root),'episode_index':episode['episode_index'],'operator_verdict':verdict},
                        'instruction':{'general':[{'images':['observations.'+v for v in CAMERAS],
                        'conversations':[{'from':'human','value':'The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n'+prompt},{'from':'gpt','value':''}]}]}}
                    (out/'episodes'/f'{name}.json').write_text(json.dumps(trajectory))
                    stats.add(canonical_arrays(state,action),joints_in_degrees=True)
                audit.write(json.dumps(result)+'\n')
            audit.flush()
            if start%320==0: print(dict(summary),flush=True)
    (out/'normalize.json').write_text(json.dumps(stats.result()))
    report={'summary':dict(summary),'sources':[str(p.resolve()) for p in args.datasets],
        'include_unlabeled':args.include_unlabeled,'exact_numeric_duplicates':duplicates,
        'additional_action_shift':0,'joint_input_units':'degrees','xyz_units':'meters','quaternion_order':'xyzw',
        'validation_split_created':False,'visual_task_success_reviewed':False,
        'video_validation':'all container headers; sampled first-frame decode, not full decode',
        'normalization':'exact mean/std over complete 30-frame windows of selected episodes; state quantiles over all selected frames'}
    (out/'prepared.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-unlabeled',action='store_true');p.add_argument('--workers',type=int,default=8)
    prepare(p.parse_args())
