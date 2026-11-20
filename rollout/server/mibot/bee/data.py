"""Read annotated USB observations; human actions are evaluation-only."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from mibot.utils.io import compose_state, normalize_quantile, resize_image
from mibot.server.runtime.client import Client

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    result = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                       2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                       2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1)
    return result.reshape(q.shape[:-1] + (3, 3)).astype(np.float32)


def robot_state(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (30,) or not np.isfinite(values).all():
        raise ValueError('Expected a finite canonical 30D observation')
    state = {}
    for side, offset in [('left', 0), ('right', 15)]:
        state[f'{side}_arm_joint'] = np.deg2rad(values[offset:offset+7])
        state[f'{side}_ee_pos'] = values[offset+7:offset+10].copy()
        state[f'{side}_ee_rotm'] = quaternion_matrix(values[offset+10:offset+14])
        state[f'{side}_gripper_pos'] = values[offset+14:offset+15].copy()
    return state


def make_batch(processor, images, instruction, state, norm, device):
    images = [resize_image(image, factor=32, max_pixels=160000) for image in images]
    messages = Client._messages(instruction, *images)
    batch = processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, return_tensors='pt',
        padding=True, images_kwargs={'do_resize': False},
    )
    packed = compose_state(state['left_gripper_pos'], state['left_arm_joint'],
                           state['right_gripper_pos'], state['right_arm_joint'])
    normalized = normalize_quantile(packed, norm['q01'], norm['q99'])
    batch['state'] = torch.from_numpy(normalized)[None]
    batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
    valid = norm['q99'][0, :16] > norm['q01'][0, :16]
    saturated = (packed[0, :16] < norm['q01'][0, :16]) | (packed[0, :16] > norm['q99'][0, :16])
    return batch, {'state_saturation_count': int((saturated & valid).sum()),
                   'state_valid_dimensions': int(valid.sum())}


def load_dataset(root):
    root = Path(root).resolve()
    info = json.loads((root/'meta/info.json').read_text())
    names = info['features']['observation.state']['names']
    expected = []
    for side in ('left', 'right'):
        expected += [f'{side}_joint_{i}_deg' for i in range(7)]
        expected += [f'{side}_eef_{axis}_m' for axis in ('x','y','z')]
        expected += [f'{side}_eef_q{axis}' for axis in ('x','y','z','w')]
        expected += [f'{side}_gripper_m']
    if list(names) != expected:
        raise ValueError('Dataset does not declare the supported USB state30 units/layout')
    if info['features']['action']['names'] != ['x_m','y_m','z_m','rotvec_x_rad','rotvec_y_rad','rotvec_z_rad']:
        raise ValueError('Expected absolute base-frame USB action6')
    for field in ('annotation.source', 'annotation.intervention', 'source.chunk_list_index'):
        if field not in info['features']:
            raise ValueError(f'Missing annotation: {field}')
    episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line]
    return root, info, episodes


def select_windows(root, info, episode, *, selection='human-chunks', start_frame=0, end_frame=None):
    """Respect original chunk boundaries, including short final human windows."""
    eid = int(episode['episode_index'])
    subst = {'episode_index': eid, 'episode_chunk': eid//int(info['chunks_size'])}
    annotation = json.loads((root/f'annotations/episode_{eid:06d}.json').read_text())
    table = pq.read_table(root/info['data_path'].format(**subst), columns=[
        'frame_index', 'source.step_id', 'source.chunk_list_index',
        'annotation.source', 'annotation.intervention', 'annotation.drop_transition'])
    arrays = {key: table[key].to_numpy() for key in table.column_names}
    if not np.array_equal(arrays['source.step_id'], np.arange(len(table))):
        raise ValueError('Source step IDs must match canonical frame indices')
    stop_selection = len(table) if end_frame is None else min(end_frame, len(table))
    windows = []
    for chunk_index, chunk in enumerate(annotation['chunks']):
        start, stop = int(chunk['step_start']), int(chunk['step_stop'])
        if not 0 <= start <= stop <= len(table):
            raise ValueError(f'Episode {eid}: invalid chunk bounds')
        if start == stop or int(chunk['source']) != 2:
            continue
        if not (np.all(arrays['source.chunk_list_index'][start:stop] == chunk_index)
                and np.all(arrays['annotation.source'][start:stop] == 2)
                and np.all(arrays['annotation.intervention'][start:stop])):
            raise ValueError(f'Episode {eid}: human chunk annotation mismatch')
        stride = 1 if selection == 'human-frames' else 10
        for frame in range(start, stop, stride):
            if start_frame <= frame < stop_selection:
                windows.append({'frame_index': frame, 'stop': min(stop,frame+10,stop_selection),
                                'chunk_index': chunk_index, 'chunk_start': start, 'chunk_stop': stop,
                                'source_episode_id': int(annotation['source_episode_id'])})
    return windows


def episode_samples(root, info, episode, windows):
    """Stream RGB at observation timestamps, ignoring trailing non-training frames."""
    eid = int(episode['episode_index'])
    subst = {'episode_index': eid, 'episode_chunk': eid // int(info['chunks_size'])}
    table = pq.read_table(root/info['data_path'].format(**subst))
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    truth = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if state.shape != (len(table), 30) or truth.shape != (len(table), 6):
        raise ValueError('USB observation/action shapes mismatch')
    if not np.isfinite(state).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite source observations/actions')
    timestamps = table['timestamp'].to_numpy()
    frames = table['frame_index'].to_numpy()
    if len(state) != int(episode['length']) or not np.array_equal(frames, np.arange(len(state))):
        raise ValueError(f'Episode {eid}: frame metadata mismatch')
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f'Episode {eid}: nonmonotonic observation timestamps')
    indices = [w['frame_index'] for w in windows]
    if indices != sorted(set(indices)) or any(i < 0 or i >= len(state) for i in indices):
        raise ValueError('Frame selection must be increasing, unique and within episode')
    tasks = {int(t['task_index']): t['task'] for t in
             (json.loads(line) for line in (root/'meta/tasks.jsonl').read_text().splitlines() if line)}
    task_ids = table['task_index'].to_numpy()
    with ExitStack() as stack:
        sources = []
        for camera in CAMERAS:
            path = root/info['video_path'].format(**subst, video_key='observation.images.'+camera)
            container = stack.enter_context(av.open(str(path)))
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            sources.append([iter(container.decode(stream)), None])
        for window in windows:
            index = window['frame_index']
            ts = float(timestamps[index])
            images, decoded_times = [], []
            for source in sources:
                while source[1] is None or float(source[1].time) < ts - 0.5/float(info['fps']):
                    source[1] = next(source[0], None)
                    if source[1] is None:
                        raise ValueError(f'Episode {eid}: video ended before frame {index}')
                    if source[1].time is None:
                        raise ValueError('Video frame lacks timestamp')
                decoded_ts = float(source[1].time)
                if abs(decoded_ts-ts) > 0.51/float(info['fps']):
                    raise ValueError(f'Episode {eid}: video/data misalignment at {index}')
                decoded_times.append(decoded_ts)
                images.append(source[1].to_image())
            if not all(bool(table['observation.camera_valid.'+camera][index].as_py()) for camera in CAMERAS):
                raise ValueError(f'Episode {eid}: invalid camera at frame {index}')
            yield {
                'episode_index': eid, 'frame_index': index, 'timestamp': ts,
                'observation': state[index], 'images': images, 'prompt': tasks[int(task_ids[index])],
                'video_timestamps': decoded_times,
                'chunk_index': window['chunk_index'], 'chunk_start': window['chunk_start'],
                'chunk_stop': window['chunk_stop'], 'source_episode_id': window['source_episode_id'],
                'source_observation_index': int(table['source.observation_index'][index].as_py()),
                'source_observation_timestamp': float(table['source.observation_timestamp'][index].as_py()),
                'source_observation_timestamp_valid': bool(table['source.observation_timestamp_valid'][index].as_py()),
                'episode_success': bool(table['annotation.episode_success'][index].as_py()),
                'drop_transition': table['annotation.drop_transition'].to_numpy()[index:window['stop']],
                # True actions are absolute base-frame poses; human references
                # in the original dataset are zero placeholders, never inputs.
                'truth': truth[index:window['stop']],
            }
