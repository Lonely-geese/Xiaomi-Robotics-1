import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compute_recap_advantages import finalize_advantages
from prepare_recap_events import prepare, sha256
from mibot.data.recap import nstep_advantages
from mibot.data.recap_events import (
    REWARD_VERSION, event_positive_labels, event_returns, event_segments,
    require_matching_data, validation_episode,
)


def test_takeover_terminates_before_success_and_handback_starts_new_segment():
    human = np.array([0]*4+[1]*3+[0]*5)
    spans, rejected = event_segments('rollout', 'success', True, human)
    assert not rejected
    assert spans == [(0,4,'takeover'), (4,7,'correction'), (7,12,'autonomous_success')]
    rewards, returns = event_returns(4, 'takeover', 12)
    assert rewards[-1] == -.5
    np.testing.assert_allclose(nstep_advantages(rewards, returns/1.1, 1.1, horizon=10), 0, atol=1e-7)
    # No prediction in the human segment is passed to the autonomous TD target.
    values = np.zeros(4)
    assert nstep_advantages(rewards, values, 1.1, horizon=10)[-1] == pytest.approx(-.5/1.1)


@pytest.mark.parametrize('outcome,flag,mask,reason', [
    ('unlabeled', False, [0,0], 'unknown_outcome'),
    ('success', None, [-1,-1], 'unknown_intervention'),
    ('failure', True, [0,0], 'intervention_outside_retained_actions'),
    ('success', True, [0,-1,1], 'unknown_intervention'),
])
def test_unknown_information_never_becomes_failure_or_autonomous_success(outcome,flag,mask,reason):
    kept, excluded = event_segments('rollout',outcome,flag,mask)
    assert kept == []
    assert excluded == [(0,len(mask),reason)]


def test_failed_human_actions_are_excluded_but_confirmed_autonomous_events_remain():
    kept, excluded = event_segments('rollout','failure',True,[0,0,1,1,0,0])
    assert kept == [(0,2,'takeover'), (4,6,'failure')]
    assert excluded == [(2,4,'correction_without_confirmed_success')]


def test_terminal_ranking_dominates_duration_and_native_support_is_respected():
    demo = event_returns(100,'demo_success',100)[1]
    success = event_returns(100,'autonomous_success',100)[1]
    takeover = event_returns(1,'takeover',100)[1]
    failure = event_returns(1,'failure',100)[1]
    np.testing.assert_array_equal(demo,success)
    assert success.min() > takeover.max() > failure.max()
    for event in ('demo_success','autonomous_success','takeover','failure'):
        values=event_returns(100,event,100)[1]/1.1
        assert np.all((values>=-1)&(values<=0))


def test_forced_labels_do_not_depend_on_critic_error():
    scores=np.array([-3.0,0.0,.2,np.nan])
    for event in ('demo_success','correction'):
        assert event_positive_labels(event,scores,.1).all()
    np.testing.assert_array_equal(event_positive_labels('failure',scores,0),[False,False,True,False])


def test_old_critic_rejected_even_when_raw_sources_match():
    old={'source_metadata_sha256':{'a':'same'}}
    new=dict(old,experiment_fingerprint='new-reward-spec')
    with pytest.raises(ValueError,match='retrain'):
        require_matching_data(old,new)
    require_matching_data(new,new)


def test_native_loader_reads_segment_image_offset_and_normalized_reward(tmp_path):
    import torch
    from train_recap_rlinf_value import NativeFrames, is_validation
    class Processor:
        def process_text(self, prompts):
            return dict(input_ids=torch.ones(1,2,dtype=torch.long),attention_mask=torch.ones(1,2,dtype=torch.bool))
    image_path=tmp_path/'rgb.npy'
    np.save(image_path,np.arange(12,dtype=np.uint8).reshape(12,1,1,1,1))
    rewards,returns=event_returns(3,'failure',12)
    path=tmp_path/'numeric.npz';np.savez(path,returns=returns)
    record=dict(length=3,numeric=str(path),images=str(image_path),image_start=7,
                prompt='insert',split='validation',dataset='source',episode_index=0)
    assert is_validation(record)
    ds=NativeFrames([record],1.1,Processor())
    assert ds[0][0].item()==7 and ds[2][0].item()==9
    assert ds[2][3].item()==pytest.approx(-1/1.1)


def test_critic_only_pipeline_does_not_start_advantages_or_policy(tmp_path):
    from types import SimpleNamespace
    from run_recap_pipeline import Pipeline
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.args=SimpleNamespace(stop_after='critic')
    pipeline.native=pipeline.events=True
    pipeline.repo=tmp_path
    pipeline.data=tmp_path/'data';pipeline.data.mkdir()
    (pipeline.data/'prepared.json').write_text('{}')
    pipeline.value=tmp_path/'value';pipeline.value.mkdir()
    (pipeline.value/'completed.json').write_text('{"steps": 8000}')
    (pipeline.value/'best.pt').touch()
    smoke=tmp_path/'value_smoke';smoke.mkdir()
    (smoke/'completed.json').write_text('{}')
    (tmp_path/'models').mkdir()
    (tmp_path/'models/recap_rlinf_backbones.json').write_text('{}')
    pipeline.adopt_orphan_stage=lambda:None
    commands=[];statuses=[]
    pipeline.command=lambda stage,command:commands.append(stage)
    pipeline.record=lambda **record:statuses.append(record)
    pipeline.run()
    assert commands==['data_audit']
    assert statuses[-1]['scope']=='critic_only'
    assert statuses[-1]['stage']=='complete'


def test_threshold_uses_only_training_autonomous_segments(tmp_path):
    (tmp_path/'advantages').mkdir()
    records=[]
    examples=[('sft','demo_success','train',[-100,-100]),
              ('rollout','correction','train',[np.nan,np.nan]),
              ('rollout','autonomous_success','train',[.1,.3]),
              ('rollout','failure','validation',[100,100])]
    for i,(pool,event,split,scores) in enumerate(examples):
        r=dict(name=str(i),pool=pool,event=event,split=split,length=2,labeled=event!='correction')
        records.append(r)
        np.savez(tmp_path/'advantages'/f'{i}.npz',values=np.zeros(2),advantage=scores,
            human=np.ones(2) if event in ('demo_success','correction') else np.zeros(2),
            value_checkpoint_digest='digest')
    meta=dict(reward_version=REWARD_VERSION,positive_fraction=.3,gamma=1,horizon=10,experiment_fingerprint='spec')
    result=finalize_advantages(tmp_path,tmp_path/'critic.pt',meta,records,True,'digest',{})
    assert result['threshold']==pytest.approx(.24)
    assert result['sft_positive']==2
    with np.load(tmp_path/'advantages/1.npz') as labels:
        assert labels['positive'].all()
    with np.load(tmp_path/'advantages/2.npz') as labels:
        assert labels['positive'].tolist()==[False,True]


def test_preparation_filters_sources_and_aligns_image_action_offsets(tmp_path):
    source=tmp_path/'cache';source.mkdir()
    (source/'normalize.json').write_text('{}')
    manifests=[];hashes={}
    for pool in ('sft','rollout'):
        dataset=tmp_path/pool;(dataset/'meta').mkdir(parents=True)
        ids=[i for i in range(1000) if not validation_episode(dict(dataset=str(dataset),episode_index=i))][:2]
        val=next(i for i in range(1000) if validation_episode(dict(dataset=str(dataset),episode_index=i)))
        metadata=[]
        for index in ids+[val]:
            human=np.array([0]*4+[1]*3+[0]*5) if pool=='rollout' else np.ones(12)
            outcome='unlabeled' if index==ids[1] and pool=='rollout' else 'success'
            metadata.append(dict(episode_index=index,outcome=outcome,human_intervened=True))
            name=f'{pool}{index}';numeric=source/f'{name}.npz'
            np.savez(numeric,state=np.zeros((12,30)),human=human)
            images=source/f'{name}.npy';np.save(images,np.arange(12))
            manifests.append(dict(name=name,dataset=str(dataset),episode_index=index,pool=pool,
                length=12,outcome=outcome,numeric=str(numeric),images=str(images),prompt='insert'))
            arrays=source/f'{name}_arrays.npz';np.savez(arrays,actions=np.arange(12))
            traj=dict(time=name,num_frames=12,arrays_path=str(arrays),
                observations={'ego':[dict(path='video.mp4',start=0)]},
                source=dict(dataset=str(dataset),episode_index=index,start=0,end_exclusive=12),
                recap=dict(pool=pool,offset=0,sidecar='old.npz'))
            dest=source/pool/'episodes';dest.mkdir(parents=True,exist_ok=True)
            (dest/f'{name}.json').write_text(json.dumps(traj))
        path=dataset/'meta/episodes.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in metadata))
        hashes[str(dataset)]=sha256(path)
    (source/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifests))
    (source/'prepared.json').write_text(json.dumps(dict(source_metadata_sha256=hashes,
        normalization_sha256=sha256(source/'normalize.json'))))
    out=tmp_path/'new'
    report=prepare(source,out)
    assert report['rollout_frames']==12  # Only one known, non-validation episode.
    assert report['sft_frames']==24
    parts=[json.loads(p.read_text()) for p in (out/'rollout/episodes').glob('*.json')]
    assert len(parts)==3
    for traj in parts:
        a,b=traj['source']['start'],traj['source']['end_exclusive']
        assert traj['observations']['ego'][0]['start']==a
        with np.load(traj['arrays_path']) as arrays:
            np.testing.assert_array_equal(arrays['actions'],np.arange(a,b))
        assert traj['recap']['offset']==0
    new=[json.loads(l) for l in (out/'episodes.jsonl').open()]
    assert all(r['policy_frames']==0 for r in new if r['split']=='validation')
    assert all(not r['labeled'] for r in new if r['event']=='correction')
    assert any(r['image_start']==7 and r['length']==5 for r in new)
    with pytest.raises(ValueError,match='overwrite'):
        prepare(source,out)
