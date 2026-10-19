from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_temporal_dataset import temporal_segments, split_sessions
from mibot.data.datamodule.base_datamodule import CompleteBatchSampler
from mibot.models.VLA.XR1 import xr1


def test_timestamp_gap_and_shift_guard():
    t=np.arange(100)/30
    t[50:]+=0.5
    segments, report=temporal_segments({'source_timestamp_utc':t},100)
    assert segments==[(0,47),(50,100)]
    assert report['shift_guard_rows']==3


def test_target_crossing_deleted_span_and_camera_freeze():
    obs=np.arange(100)/30
    target=obs+0.1
    target[47:50]+=0.5
    segments, report=temporal_segments({'source_timestamp_utc':obs,'source_action_timestamp_utc':target},100)
    assert report['invalid_rows']==3
    assert all(not (a<=49 and b>47) for a,b in segments)
    seq=np.arange(100,dtype=float);seq[40:46]=39
    segments, report=temporal_segments({'camera_sample_sequence.left':seq},100)
    assert report['invalid_rows']==6
    assert all(not (a<=45 and b>40) for a,b in segments)


def test_nominal_continuity_and_session_split_small_source():
    segments,_=temporal_segments({'observation_sequence':np.arange(100)},100)
    assert segments==[(0,100)]
    records=[{'dataset':'small','session':f'sess_{i}','segments':[(0,100)],'temporal':{'retained_frames':100}} for i in range(16)]
    assignment=split_sessions(records)
    assert set(assignment.values())=={'train','validation','test'}
    assert assignment==split_sessions(records)


def test_global_batch_exact_without_duplication():
    shards=[CompleteBatchSampler(range(1030),128,num_replicas=8,rank=i,shuffle=True,seed=42) for i in range(8)]
    samples=[list(s) for s in shards]
    assert all(len(s)==128 for s in samples)
    assert len(set(sum(samples,[])))==1024
    shards[0].set_epoch(1)
    assert list(shards[0])!=samples[0]


def test_left_only_losses_ignore_right_and_gradients():
    model=xr1.__new__(xr1);model.freq_excluded_dims=[17,18,19];model.n_choices=3
    pred=torch.randn(2,30,60,requires_grad=True);target=torch.randn_like(pred)
    mask=torch.zeros_like(pred,dtype=torch.int32);mask[:,:,:7]=1
    weight=torch.rand_like(pred)+0.1
    losses=model.compute_flow_loss(pred,target,mask,weight)
    other=target.detach().clone();other[:,:,7:]+=1000
    altered_weight=weight.clone();altered_weight[:,:,7:]*=1000
    again=model.compute_flow_loss(pred,other,mask,altered_weight)
    torch.testing.assert_close(sum(losses),sum(again))
    sum(losses).backward()
    assert torch.count_nonzero(pred.grad[:,:,7:])==0
    assert pred.grad[:,:,:7].abs().sum()>0
    choices=torch.randn(60,3*60,requires_grad=True);scores=torch.randn(2,3,requires_grad=True)
    a=model.compute_choice_loss(choices,scores,target.flatten(0,1),mask.flatten(0,1),torch.tensor([30,30]))
    b=model.compute_choice_loss(choices,scores,other.flatten(0,1),mask.flatten(0,1),torch.tensor([30,30]))
    torch.testing.assert_close(sum(a),sum(b))
    sum(a).backward()
    assert torch.count_nonzero(choices.grad.reshape(60,3,60)[:,:,7:])==0
