#!/usr/bin/env python3
"""Verify installed kernels, full checkpoint compatibility and a real data batch.

Does not launch a 5B training job. GPU probes use small tensors only.
Run after sourcing scripts/env.sh, from xr1/.
"""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_func
from mibot.models.VLA.XR1 import xr1, DecoderLayer
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.data.collate.custom_collate import CustomCollate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/ethernet_9007_verified'))
    p.add_argument('--checkpoint',type=Path,default=Path('pretrained_ckpt/model_states.pt'))
    p.add_argument('--output',type=Path,default=Path('../.setup/verification.json'))
    args=p.parse_args()
    report={'packages':{k:version(k) for k in ['torch','torchvision','transformers','lightning','deepspeed','flash-attn']}}
    torch.set_num_threads(4)
    report['cuda_available']=torch.cuda.is_available()
    if not report['cuda_available']: raise RuntimeError('CUDA unavailable')
    q,k,v=[torch.randn(1,32,8,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    y=flash_attn_func(q,k,v,causal=True)
    y.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q,k,v))
    report['flash_attention_forward_backward']=True
    layer=DecoderLayer().to(device='cuda',dtype=torch.bfloat16)
    opt=torch.optim.AdamW(layer.parameters(),lr=1e-5,fused=True)
    hidden=torch.randn(1,4,1024,device='cuda',dtype=torch.bfloat16)
    cache=[torch.randn(1,8,3,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    angles=(torch.ones(1,4,128,device='cuda',dtype=torch.bfloat16),torch.zeros(1,4,128,device='cuda',dtype=torch.bfloat16))
    mask=torch.ones(1,1,4,7,device='cuda',dtype=torch.bool)
    loss=layer(hidden,cache,angles,torch.randn(1,6,1024,device='cuda',dtype=torch.bfloat16),mask).float().square().mean()
    loss.backward();assert torch.isfinite(loss);opt.step()
    report['dit_layer_backward_fused_adamw_step']=True
    del q,k,v,y,layer,opt,hidden,cache,angles,mask,loss
    gc.collect();torch.cuda.empty_cache()
    # Use CPU construction: the model creates a validated Beta distribution,
    # which intentionally cannot be initialized on the meta device.
    model=xr1(full_finetune=True)
    ckpt=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    weights={k.removeprefix('model.'):v for k,v in ckpt['module'].items()}
    model.load_state_dict(weights,strict=True)
    report['checkpoint_strict_load']=True
    report['parameter_count']=sum(p.numel() for p in model.parameters())
    report['trainable_parameter_count']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['frozen_parameters']=[name for name,p in model.named_parameters() if not p.requires_grad]
    assert not report['frozen_parameters']
    print('Checkpoint strictly matched; all parameters trainable.',flush=True)
    del model,weights,ckpt;gc.collect()
    # Use one episode from each source and inspect the real image/action collation.
    files=sorted((args.data/'episodes').glob('*.json'))
    selected=[files[0],next(f for f in files if f.name.startswith('20260928_935'))]
    params={'max_steps':1,'train_datasets':{'batch_size':1,'action_length':30,'repeat_to_max_steps':False,
        'paths':[str(f) for f in selected],'normalization':str(args.data/'normalize.json')}}
    ds=JsonDataset(params);collate=CustomCollate()
    reports=[]
    for index in [0,int(JsonDataset._read_json(str(selected[0]))['num_frames'])]:
        batch=collate([ds[index]])
        assert batch['action'].shape==(1,30,60)
        assert batch['state'].shape==(1,1,60)
        assert torch.isfinite(batch['action']).all() and torch.isfinite(batch['state']).all()
        reports.append({'tokens':int(batch['input_ids'].shape[1]),'action_shape':list(batch['action'].shape),
            'image_grids':batch['image_grid_thw'].tolist()})
    report['real_dataset_batches']=reports
    report['full_5b_training_step']='not_run; insufficient free GPU memory while other jobs occupy the GPUs'
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
