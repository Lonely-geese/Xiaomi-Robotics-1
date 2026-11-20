"""Read-only audit of a committed dataset snapshot; outputs only outside the dataset."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'recordings/usb_recap_epoch10_step15490'


def main():
    info = json.loads((DATA / 'meta/info.json').read_text())
    episodes = [json.loads(x) for x in (DATA / 'meta/episodes.jsonl').read_text().splitlines()
                if x.strip()][:info['total_episodes']]
    report = {'checked_at': datetime.now().isoformat(), 'dataset': str(DATA),
              'episodes': len(episodes), 'frames': 0, 'issues': [], 'episode_details': [],
              'video_headers_checked': 0, 'decoded_samples': 0}
    timing = {1: [], 2: []}
    offsets = {1: [], 2: []}
    sources = Counter()
    outcome = Counter()
    all_states = []
    index_offset = 0

    def check(condition, episode, message):
        if not condition:
            report['issues'].append({'episode': episode, 'issue': message})

    for meta in episodes:
        eid, n = meta['episode_index'], meta['length']
        sub = {'episode_index': eid, 'episode_chunk': eid // info['chunks_size']}
        table = pq.read_table(DATA / info['data_path'].format(**sub))
        col = lambda name: table[name].to_numpy()
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        source, stamp, seq = col('source'), col('hal_time_ns'), col('observation_sequence')
        ts = col('timestamp')
        target = np.minimum(np.arange(n) + 3, n - 1)
        dt = np.diff(stamp) / 1e9
        check(len(table) == n and state.shape == action.shape == (n, 30), eid, 'shape/length')
        check(np.isfinite(state).all() and np.isfinite(action).all(), eid, 'nonfinite values')
        check(np.array_equal(action, state[target]), eid, 'action shift mismatch')
        check(np.array_equal(col('frame_index'), np.arange(n)), eid, 'frame index')
        check(np.array_equal(col('index'), index_offset + np.arange(n)), eid, 'global index')
        check(np.all(col('episode_index') == eid), eid, 'episode index')
        check(np.allclose(ts, np.arange(n) / 30, atol=1e-5), eid, 'stored timestamps')
        check(np.all(dt > 0) and np.all(np.diff(seq) > 0), eid, 'nonmonotonic source timing')
        check(np.isin(source, [1, 2]).all(), eid, 'source labels')
        check(np.array_equal(col('intervene_flag'), source == 2), eid, 'intervention labels')
        check(np.all(col('is_success') == meta['success']), eid, 'success label')
        check(np.array_equal(col('done'), np.arange(n) == n - 1), eid, 'done label')
        check(np.array_equal(col('reward'), ((np.arange(n) == n-1) & meta['success']).astype(float)), eid, 'reward')
        check(int((source == 2).sum()) == meta['human_frames'], eid, 'human frame count')
        check(bool((source == 2).any()) == meta['human_intervened'], eid, 'human episode label')
        for q in (state[:, 10:14], state[:, 25:29]):
            check(np.max(np.abs(np.linalg.norm(q, axis=1)-1)) < 1e-3, eid, 'quaternion norm')
        check(np.all((state[:, [14, 29]] >= -1e-4) & (state[:, [14, 29]] <= .0901)), eid, 'gripper range')
        for label in (1, 2):
            timing[label].extend(dt[(source[:-1] == label) & (source[1:] == label)].tolist())
            valid = (source == label) & (source[target] == label) & (target < n - 1)
            offsets[label].extend(((stamp[target] - stamp) / 1e9)[valid].tolist())
        sources.update(source.tolist())
        outcome[('human_' if meta['human_intervened'] else 'policy_') + meta['outcome']] += 1
        transitions = np.flatnonzero(source[1:] != source[:-1]) + 1
        row = {'id': eid, 'frames': n, 'human_frames': meta['human_frames'],
               'success': meta['success'], 'bad_frames_reported': meta.get('bad_frames'),
               'sequence_gaps_reported': meta.get('sequence_gaps'),
               'source_transitions': transitions.tolist(),
               'action_crosses_source_boundary': int((source != source[target]).sum()),
               'max_source_gap_s': float(dt.max()),
               'gaps_over_1s': int((dt > 1).sum()),
               'real_span_s': float((stamp[-1] - stamp[0]) / 1e9),
               'stored_span_s': float(ts[-1]), 'videos': []}
        for key, feature in info['features'].items():
            if feature['dtype'] != 'video':
                continue
            path = DATA / info['video_path'].format(**sub, video_key=key)
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    check(stream.frames == n, eid, f'{key}: header frame count {stream.frames}')
                    check((stream.width, stream.height) == (848, 480), eid, f'{key}: dimensions')
                    check(float(stream.average_rate) == 30, eid, f'{key}: fps')
                    report['video_headers_checked'] += 1
                    samples = []
                    for frame_index in sorted({0, n // 2, n - 1}):
                        wanted = frame_index / 30
                        container.seek(int(wanted / stream.time_base), stream=stream, backward=True)
                        frame = next(f for f in container.decode(stream) if float(f.time) >= wanted - .001)
                        pixels = frame.to_ndarray(format='rgb24')
                        check(abs(float(frame.time) - wanted) < .001, eid, f'{key}: sample timestamp')
                        samples.append({'frame': frame_index, 'mean': float(pixels.mean()),
                                        'std': float(pixels.std())})
                        check(float(pixels.mean()) >= 5, eid, f'{key}: dark sample {frame_index}')
                        report['decoded_samples'] += 1
                    row['videos'].append({'key': key, 'frames': stream.frames, 'samples': samples})
            except Exception as exc:
                check(False, eid, f'{key}: {type(exc).__name__}: {exc}')
        report['episode_details'].append(row)
        all_states.append(state)
        report['frames'] += n
        index_offset += n
        if (eid + 1) % 10 == 0:
            print(f'Checked {eid + 1}/{len(episodes)} episodes', flush=True)
    state = np.concatenate(all_states)
    report.update(outcomes=dict(outcome), source_frames=dict(sources),
                  state_min=state.min(axis=0).tolist(), state_max=state.max(axis=0).tolist())
    for label, name in ((1, 'policy'), (2, 'human')):
        def quantiles(values):
            return dict(zip(['min', 'median', 'p99', 'max'], np.quantile(values, [0, .5, .99, 1]).tolist())) if values else {}
        report[name + '_frame_dt_s'] = quantiles(timing[label])
        report[name + '_action_lookahead_s'] = quantiles(offsets[label])
    report['action_crosses_source_boundary'] = sum(r['action_crosses_source_boundary'] for r in report['episode_details'])
    report['bad_frames_reported'] = sum(r['bad_frames_reported'] or 0 for r in report['episode_details'])
    report['sequence_gaps_reported'] = sum(r['sequence_gaps_reported'] or 0 for r in report['episode_details'])
    output = ROOT / 'runs' / ('dataset_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('episode_details', 'state_min', 'state_max')}, ensure_ascii=False))
    print(output)


if __name__ == '__main__':
    main()
