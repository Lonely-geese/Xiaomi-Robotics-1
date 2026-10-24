import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mibot.data.dagger import classify_segments, expert_chunk_starts, is_validation, sha256
from mibot.data.datasets.dagger_dataset import ExpertSamples, DaggerJsonDataset
from mibot.data.recap import BalancedPoolSampler
from prepare_dagger_dataset import prepare
from verify_dagger_dataset import verify
from train_dagger import DaggerRunRecord
from mibot.data.datasets.dagger_dataset import ExactValidationBatchSampler
from mibot.models.runner.dagger_runner import DaggerRunner, LOSS_NAMES, validation_metrics


def test_takeover_is_failure_and_only_successful_human_segment_is_target():
    result = classify_segments('rollout', 'success', True, [0, 0, 1, 1])
    assert [s['event'] for s in result] == ['takeover_failure', 'correction_success']
    assert result[0]['terminal_label'] == -1 and result[0]['target_pool'] is None
    assert result[1]['terminal_label'] == 0 and result[1]['target_pool'] == 'correction'
    failed = classify_segments('rollout', 'failure', True, [0, 0, 1, 1])
    assert all(s['target_pool'] is None for s in failed)


@pytest.mark.parametrize('outcome,intervened,mask,reason', [
    ('unknown', True, [0, 1], 'unknown_outcome'),
    ('success', None, [-1, -1], 'unknown_intervention'),
    ('success', True, [0, 0], 'intervention_outside_retained_actions'),
    ('success', False, [0, 1], 'intervention_outside_retained_actions'),
])
def test_ambiguous_data_is_excluded(outcome, intervened, mask, reason):
    result = classify_segments('rollout', outcome, intervened, mask)
    assert len(result) == 1 and result[0]['event'] == reason and result[0]['target_pool'] is None


def test_demo_and_autonomous_success_are_distinct():
    assert classify_segments('sft', 'success', None, [-1, -1])[0]['target_pool'] == 'demo'
    assert classify_segments('rollout', 'success', False, [0, 0])[0]['target_pool'] is None


def test_chunk_filter_matches_rlinf_all_nonpadding_actions_rule():
    rng = np.random.default_rng(42)
    for n in (1, 3, 30, 90):
        flags = rng.integers(0, 2, n)
        for horizon in (1, 3, 30):
            raw = np.arange(n)[:, None] + np.arange(horizon)[None, :]
            in_episode = raw < n
            step_ok = np.ones_like(in_episode, dtype=bool)
            step_ok[in_episode] = flags[raw[in_episode]]
            np.testing.assert_array_equal(expert_chunk_starts(flags, horizon), np.flatnonzero(step_ok.all(axis=1)))
    # Human handback is not an episode ending: starts 2 and 3 are invalid.
    np.testing.assert_array_equal(expert_chunk_starts([0, 1, 1, 1, 0], 3), [1])
    np.testing.assert_array_equal(expert_chunk_starts([0, 1, 1, 1], 3), [1, 2, 3])


def make_source(tmp_path, name='input'):
    root = tmp_path/name
    dataset = tmp_path/(name+'_raw')
    (dataset/'meta').mkdir(parents=True)
    (root/'sft/episodes').mkdir(parents=True)
    (root/'rollout/episodes').mkdir(parents=True)
    (root/'arrays').mkdir()
    video = dataset/'rgb.mp4'
    video.write_bytes(b'fixture; no decoding in numerical audit')
    rows, records = [], []
    ids = [i for i in range(100) if not is_validation(str(dataset), i)][:3]
    for i, (pool, outcome, human) in zip(ids, [
        ('sft', 'success', [-1]*8),
        ('rollout', 'success', [0, 1, 1, 1, 0, 1, 1, 1]),
        ('rollout', 'failure', [0, 0, 1, 1, 1, 1, 1, 1]),
    ]):
        rows.append(dict(episode_index=i, human_intervened=None if pool=='sft' else True))
        numeric = root/f'{i}.npz'
        np.savez(numeric, human=np.array(human), state=np.arange(8))
        records.append(dict(name=f'ep{i}', dataset=str(dataset), episode_index=i,
            pool=pool, outcome=outcome, length=8, numeric=str(numeric)))
        arrays = root/'arrays'/f'{i}.npz'
        np.savez(arrays, **{'actions.x': np.arange(8)[:, None], 'proprios.x': np.arange(8)[:, None]+100})
        policy = dict(num_frames=8, arrays_path=str(arrays),
            observations=dict(ego=[dict(path=str(video), start=11, crop_bbox=None)]),
            source=dict(dataset=str(dataset), episode_index=i, start=0, end_exclusive=8),
            recap=dict(pool=pool, offset=0), instruction={})
        (root/pool/'episodes'/f'{i}.json').write_text(json.dumps(policy))
    path = dataset/'meta/episodes.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root/'normalize.json').write_text('{}')
    (root/'prepared.json').write_text(json.dumps(dict(ready=True,
        normalization_sha256=sha256(root/'normalize.json'), source_metadata_sha256={str(dataset):sha256(path)})))
    return root


def test_prepare_verify_preserves_offsets_filters_handback_and_refuses_overwrite(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path/'dagger'
    meta = prepare([source], output, horizon=3)
    assert meta['counts']['train_demo_frames'] == 8
    assert meta['counts']['train_correction_frames'] == 4  # one full chunk + terminal suffix
    assert verify(output)['expert_only']
    files = sorted((output/'correction/episodes').glob('*.json'))
    samples = ExpertSamples(files, horizon=3)
    assert len(samples) == 4
    for index in range(len(samples)):
        sample = samples[index]
        traj = json.loads(Path(sample['file']).read_text())
        assert traj['observations']['ego'][0]['start'] == 11+traj['source']['start']
        ds = DaggerJsonDataset.__new__(DaggerJsonDataset)
        ds.action_length = 3
        assert ds._action_steps(traj, sample['frame_index']) <= traj['num_frames']-sample['frame_index']
    with pytest.raises(ValueError, match='horizon'):
        ExpertSamples(files, horizon=30)
    with pytest.raises(ValueError, match='overwrite'):
        prepare([source], output, horizon=3)
    (output/'normalize.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Normalization'):
        verify(output)


def test_aggregation_is_additive_and_duplicate_parents_rejected(tmp_path):
    first, second = make_source(tmp_path, 'round1'), make_source(tmp_path, 'round2')
    meta = prepare([first, second], tmp_path/'all', horizon=3)
    assert meta['counts']['train_demo_frames'] == 16
    assert meta['counts']['train_correction_frames'] == 8
    verify(tmp_path/'all')
    import shutil
    shutil.copytree(first, tmp_path/'duplicate')
    with pytest.raises(ValueError, match='Duplicate parent'):
        prepare([first, tmp_path/'duplicate'], tmp_path/'bad', horizon=3)


def test_eight_rank_balanced_sampling_and_resume():
    demos = []
    for rank in range(8):
        sampler = BalancedPoolSampler(1280, 231, 32, 8, rank)
        batches = np.array(list(sampler)).reshape(-1, 32)
        assert np.all((batches < 1280).sum(1) == 16)
        demos.extend(batches[batches < 1280].tolist())
        sampler.start_batch = 4
        np.testing.assert_array_equal(list(sampler), batches[4:].ravel())
        sampler.set_epoch(1)
        assert sampler.start_batch == 0
    assert len(demos) == len(set(demos)) == 1280


def test_callback_removes_pool_metadata_before_xr1_forward_and_rejects_wrong_batch():
    callback = DaggerRunRecord('.', {})
    batch = dict(dagger_pool=torch.tensor([0, 1]*16), action=torch.zeros(32, 30, 20))
    callback.on_train_batch_start(None, None, batch, 0)
    assert set(batch) == {'action'}
    with pytest.raises(ValueError, match='16 demos'):
        callback.on_train_batch_start(None, None, dict(dagger_pool=torch.zeros(32)), 0)


def test_short_human_chunk_padding_does_not_contribute_to_flow_gradient():
    from mibot.models.VLA.XR1 import xr1
    model = xr1.__new__(xr1)
    torch.nn.Module.__init__(model)
    model.freq_excluded_dims = [17, 18, 19]
    pred = torch.zeros(2, 30, 60, requires_grad=True)
    target = torch.ones_like(pred)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[0, :3, :7] = True
    mask[1, :30, :7] = True
    mse, frequency = model.compute_flow_loss(pred, target, mask, torch.ones_like(pred))
    loss = mse + frequency
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad[0, :3, :7].abs().sum() > 0
    assert pred.grad[0, 3:].abs().sum() == 0
    assert pred.grad[:, :, 7:].abs().sum() == 0


@pytest.mark.parametrize('size', [1, 7, 63, 64, 65, 673, 7694])
def test_validation_covers_every_frame_once_with_equal_forward_counts(size):
    streams = [list(ExactValidationBatchSampler(size, 8, 8, rank)) for rank in range(8)]
    assert len({len(stream) for stream in streams}) == 1
    actual = [i for stream in streams for batch in stream for i in batch if i >= 0]
    assert sorted(actual) == list(range(size))
    assert all(batch == [-1] or all(i >= 0 for i in batch) for stream in streams for batch in stream)


def test_validation_macro_does_not_allow_large_demo_pool_to_hide_correction_loss():
    totals = torch.zeros(2, len(LOSS_NAMES)+1, dtype=torch.float64)
    totals[0, :-1] = 7694 * 1.
    totals[0, -1] = 7694
    totals[1, :-1] = 673 * 3.
    totals[1, -1] = 673
    metrics = validation_metrics(totals)
    assert metrics['val/demo_loss'] == 1.
    assert metrics['val/correction_loss'] == 3.
    assert metrics['val/loss'] == 2.


def test_validation_repeats_noise_restores_training_rng_and_ignores_dummy_batches():
    class FakeModel(torch.nn.Module):
        def forward(self, batch, return_loss):
            assert return_loss and set(batch) == {'action'}
            value = torch.rand(())
            return {name:value for name in LOSS_NAMES}
    runner = DaggerRunner(dict(model={}))
    runner.model = FakeModel()
    runner._trainer = SimpleNamespace(global_rank=0)
    runner.on_validation_epoch_start()
    def batch(real=True):
        return dict(action=torch.zeros(1, 30, 60), dagger_pool=torch.tensor([0]),
                    dagger_eval_weight=torch.tensor([int(real)]))
    state = torch.random.get_rng_state()
    first = runner.validation_step(batch(), 0, 0)
    assert torch.equal(state, torch.random.get_rng_state())
    second = runner.validation_step(batch(), 0, 0)
    assert first == second
    totals = runner._validation_totals.clone()
    runner.validation_step(batch(False), 2, 0)
    assert torch.equal(totals, runner._validation_totals)
