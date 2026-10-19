#!/usr/bin/env python3
"""Independent CPU verification of the exact mixed dataset before GPU launch."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path

from mmengine import Config
import numpy as np
import pyarrow.parquet as pq
import torch

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.data.collate.custom_collate import CustomCollate


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root/'xr1/data/ethernet_middle_mixed_20261013')
    parser.add_argument('--old-count', type=int, default=1500)
    parser.add_argument('--report-dir', type=Path, default=root/'.setup/middle_mixed_20261013')
    args = parser.parse_args()
    os.environ.setdefault('MAX_LENGTH', '20000')
    os.environ['XR1_STRICT_BATCH'] = '1'
    data = args.data.resolve()
    prepared = json.loads((data / 'prepared.json').read_text())
    assert prepared['ready'] is True
    records = [json.loads(line) for line in (data / 'segment_manifest.jsonl').open()]
    assert len({row['name'] for row in records}) == len(records)
    assert all(row['split'] == 'train' and row['verdict'] == 'success' for row in records)
    expected_files = {row['name'] + '.json' for row in records}
    assert expected_files == {p.name for p in (data / 'episodes').glob('*.json')}
    assert {row['name'] + '.npz' for row in records} == {p.name for p in (data / 'arrays').glob('*.npz')}
    assert sum(row['frames'] for row in records) == prepared['train_frames']
    assert (data / 'normalize.json').read_bytes() == Path(prepared['normalization_source']).read_bytes()
    assert hashlib.sha256((data / 'segment_manifest.jsonl').read_bytes()).hexdigest() == prepared['manifest_sha256']
    source_rows = {}
    unique = defaultdict(set)
    representatives = {}
    for row in records:
        dataset = row['dataset']
        if dataset not in source_rows:
            source_rows[dataset] = {entry['episode_index']: entry for entry in
                (json.loads(line) for line in (Path(dataset) / 'meta/episodes.jsonl').open())}
        original = source_rows[dataset][row['episode_index']]
        assert original['operator_verdict'] == 'success'
        assert original.get('data_valid') is not False
        assert original.get('structurally_valid') is not False
        bucket = row['origin'] + ':' + row['original_split']
        unique[bucket].add((dataset, row['episode_index']))
        expected_shift = 3 if row['origin'] == 'new_middle' else 0
        assert row['additional_action_shift_frames'] == expected_shift
        assert 0 <= row['start'] < row['end_exclusive'] <= original['length']
        assert row['frames'] == row['end_exclusive'] - row['start'] >= 30
        representatives.setdefault(bucket, row)
    counts = {key: len(values) for key, values in unique.items()}
    assert counts == {'new_middle:new': 721, 'old:train': args.old_count,
                      'old:validation': 491, 'old:test': 532}, counts
    all_keys = set().union(*unique.values())
    assert len(all_keys) == sum(counts.values()) == args.old_count + 1744

    config = Config(dict(train_datasets=dict(
        paths=[str(data / 'episodes' / (row['name'] + '.json')) for row in representatives.values()],
        normalization=str(data / 'normalize.json'), action_length=30, batch_size=12,
        augment=False, sample_stride=1, repeat_to_max_steps=False, active_action_dims=list(range(7)))))
    dataset = JsonDataset(config)
    by_file = defaultdict(list)
    for i, item in enumerate(dataset.samples):
        by_file[item['file']].append(i)
    samples, examples = [], []
    hashes = json.loads((data / 'arrays_sha256.json').read_text())
    for bucket, record in representatives.items():
        path = data / 'episodes' / (record['name'] + '.json')
        trajectory = json.loads(path.read_text())
        start, end = record['start'], record['end_exclusive']
        assert all(view[0]['start'] == start for view in trajectory['observations'].values())
        source = Path(record['dataset'])
        info = json.loads((source / 'meta/info.json').read_text())
        index = record['episode_index']
        source_path = source / info['data_path'].format(
            episode_index=index, episode_chunk=index // info['chunks_size'])
        table = pq.read_table(source_path, columns=['observation.state', 'action'])
        states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        targets = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        expected_action = states[start+3:end+3] if record['origin'] == 'new_middle' else targets[start:end]
        array_path = data / 'arrays' / (record['name'] + '.npz')
        assert hashlib.sha256(array_path.read_bytes()).hexdigest() == hashes[record['name']]
        with np.load(array_path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['actions.left_ee_pos'], expected_action[:, 7:10])
            np.testing.assert_array_equal(arrays['actions.left_gripper_pos'], expected_action[:, 14:15])
            np.testing.assert_array_equal(arrays['proprios.left_ee_pos'], states[start:end, 7:10])
            np.testing.assert_allclose(arrays['proprios.left_arm_joint'], np.deg2rad(states[start:end, :7]))
        choices = by_file[str(path)]
        for sample_index in (choices[0], choices[len(choices)//2], choices[-1]):
            item = dataset[sample_index]
            assert torch.isfinite(item['state']).all() and torch.isfinite(item['action']).all()
            assert torch.count_nonzero(item['action'][..., 7:]) == 0
            assert torch.count_nonzero(item['action_mask'][..., 7:]) == 0
            samples.append(item)
        examples.append({'bucket': bucket, 'name': record['name'], 'start': start, 'end': end})
    batch = CustomCollate()(samples)
    assert tuple(batch['action'].shape) == (12, 30, 60)
    assert batch['action_segments'].shape[0] == 12
    assert torch.isfinite(batch['pixel_values']).all()
    result = dict(passed=True, source_episode_counts=counts, segments=len(records),
                  train_frames=prepared['train_frames'], independent_evaluation=False,
                  batch_shapes={k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
                  examples=examples,
                  verified=['all manifest success/valid source labels', f'exact {args.old_count} old-train selection',
                            'all 721 new successes and 1023 valid old holdouts', 'no source overlap',
                            'inherited normalization byte equality', 'sampled numeric target alignment and units',
                            'sampled real video decoding at first/middle/last retained frames',
                            '12 real samples collated without truncation; inactive actions masked'])
    output = args.report_dir / 'independent_preflight.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
