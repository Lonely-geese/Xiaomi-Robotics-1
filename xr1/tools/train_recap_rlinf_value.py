#!/usr/bin/env python3
"""Train the native RLinf SigLIP2/Gemma3 categorical critic on two pools."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
import pickle
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from mibot.data.recap import BalancedPoolSampler
from mibot.models.recap_rlinf import build_critic,build_processor,observation,verify_upstream
from recap_recovery import atomic_torch_save


def is_validation(record):
    key=f"{record['dataset']}:{record['episode_index']}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%20==0


class NativeFrames(Dataset):
    def __init__(self,records,scale,processor):
        self.records,self.scale=records,scale
        self.ends=np.cumsum([r['length'] for r in records])
        self.text={p:processor.process_text([p]) for p in {r['prompt'] for r in records}}
        self.cache=OrderedDict()

    def __len__(self):return int(self.ends[-1])

    def __getitem__(self,index):
        episode=int(np.searchsorted(self.ends,index,side='right'))
        frame=index-(int(self.ends[episode-1]) if episode else 0)
        r=self.records[episode]
        if episode not in self.cache:
            with np.load(r['numeric']) as data:returns=data['returns'].copy()
            self.cache[episode]=(np.load(r['images'],mmap_mode='r'),returns)
            if len(self.cache)>64:self.cache.popitem(last=False)
        else:self.cache.move_to_end(episode)
        images,returns=self.cache[episode]
        t=self.text[r['prompt']]
        return (torch.from_numpy(images[frame].copy()),t['input_ids'][0],t['attention_mask'][0],
                torch.tensor(returns[frame]/self.scale,dtype=torch.float32))


def main(args):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world!=8:raise ValueError(f'This run requires 8 GPUs x 32 = 256; found WORLD_SIZE={world}')
    if world>1:dist.init_process_group('nccl')
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK','0')))
    torch.cuda.set_device(device)
    torch.manual_seed(42);np.random.seed(42)
    metadata=json.loads((args.data/'prepared.json').read_text())
    records=[json.loads(l) for l in (args.data/'episodes.jsonl').open()]
    records=[r for r in records if r['labeled']]
    training=sorted([r for r in records if not is_validation(r)],key=lambda r:r['pool']!='sft')
    validation=[r for r in records if is_validation(r)]
    processor=build_processor(args.gemma)
    sampler=BalancedPoolSampler(sum(r['length'] for r in training if r['pool']=='sft'),
        sum(r['length'] for r in training if r['pool']=='rollout'),32,world,rank)
    train=NativeFrames(training,metadata['return_scale'],processor)
    loader=DataLoader(train,batch_size=32,sampler=sampler,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=True,prefetch_factor=2 if args.workers else None)
    valsets={pool:NativeFrames([r for r in validation if r['pool']==pool],metadata['return_scale'],processor)
             for pool in ('sft','rollout')}
    valloaders={pool:DataLoader(ds,batch_size=32,sampler=range(rank,len(ds),world),num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0) for pool,ds in valsets.items()}
    # FP32 master parameters with BF16 autocast; model topology and native loss
    # are unchanged. This avoids rounding small updates into BF16 parameters.
    model=build_critic(args.siglip,args.gemma).float().to(device)
    network=DDP(model,device_ids=[device.index],find_unused_parameters=True) if world>1 else model
    # Native _apply_checkpoint recomputes image embeddings. Keep Gemma's own
    # gradient_checkpointing disabled because prefix KV caches are required.
    model.gradient_checkpointing_enabled=True
    groups=[dict(params=[p for n,p in model.named_parameters() if 'value_head' not in n],lr=5e-5),
            dict(params=[p for n,p in model.named_parameters() if 'value_head' in n],lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.0,s/500))
    args.output.mkdir(parents=True,exist_ok=True)
    step,epoch,start_batch,best=0,0,0,float('inf')
    resume=args.output/'resume.pt'
    previous=args.output/'resume.previous.pt'
    if args.resume and (resume.exists() or previous.exists()):
        try:
            saved=torch.load(resume if resume.exists() else previous,map_location='cpu',weights_only=False)
        except (OSError,RuntimeError,EOFError,ValueError,pickle.UnpicklingError):
            saved=torch.load(previous,map_location='cpu',weights_only=False)
            if rank==0 and resume.exists():resume.rename(resume.with_name(f'resume.corrupt.{time.time_ns()}.pt'))
            if world>1:dist.barrier()
        if saved['metadata']['source_metadata_sha256']!=metadata['source_metadata_sha256']:
            raise ValueError('Critic recovery belongs to different data')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step,epoch,start_batch,best=[saved[k] for k in ('steps','next_epoch','next_batch','best')]
        local=saved['rank_states'][rank]
        torch.set_rng_state(local['torch_rng']);torch.cuda.set_rng_state(local['cuda_rng'],device)
        np.random.set_state(local['numpy_rng'])
    record=dict(backend='native_rlinf',upstream_commit=verify_upstream(),metadata=metadata,
        config=model.config.to_dict(),steps=args.steps,global_batch=32*world,per_gpu_batch=32,
        optimizer='AdamW FP32 master + BF16 autocast; native RLinf parameter groups and LR warmup',
        checkpoint_selection='rollout episode-disjoint validation RMSE',
        training_episodes=[r['name'] for r in training],validation_episodes=[r['name'] for r in validation])
    if rank==0:
        (args.output/'run_record.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(stage='value_train',backend='native_rlinf',parameters=sum(p.numel() for p in model.parameters()),
            steps=args.steps,start_step=step,steps_per_epoch=sampler.steps,train_frames=len(train),
            validation_frames={k:len(v) for k,v in valsets.items()})),flush=True)

    def save_resume(next_epoch,next_batch):
        local=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),numpy_rng=np.random.get_state())
        states=[None]*world
        if world>1:dist.all_gather_object(states,local)
        else:states[0]=local
        if rank==0:
            atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                metadata=metadata,steps=step,next_epoch=next_epoch,next_batch=next_batch,best=best,rank_states=states),resume)
        if world>1:dist.barrier()

    def evaluate():
        network.eval();metrics=dict(step=step,epoch=epoch+1)
        for pool,val_loader in valloaders.items():
            total=torch.zeros(3,device=device,dtype=torch.float64)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for images,tokens,mask,target in val_loader:
                    images,tokens,mask,target=[x.to(device,non_blocking=True) for x in (images,tokens,mask,target)]
                    result=model(observation(images,tokens,mask,processor),target_values=target)
                    total+=torch.stack([result.loss.double()*len(target),((result.predicted_values.float()-target)**2).sum().double(),target.new_tensor(len(target)).double()])
            if world>1:dist.all_reduce(total)
            metrics[pool+'_val_loss']=float(total[0]/total[2])
            metrics[pool+'_val_rmse']=float((total[1]/total[2]).sqrt())
        if rank==0:
            print(json.dumps(metrics),flush=True)
            with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        network.train()
        return metrics

    while step<args.steps:
        sampler.set_epoch(epoch);sampler.start_batch=start_batch
        network.train();started=time.monotonic()
        for batch_index,(images,tokens,mask,target) in enumerate(loader,start=start_batch):
            images,tokens,mask,target=[x.to(device,non_blocking=True) for x in (images,tokens,mask,target)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=network(observation(images,tokens,mask,processor,training=True),target_values=target)
                loss=result.loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite native RLinf critic loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite native critic gradient')
            optimizer.step();scheduler.step();step+=1
            if step<=2 or step%10==0:
                reduced=loss.detach().float().clone()
                if world>1:dist.all_reduce(reduced);reduced/=world
                if rank==0:print(json.dumps(dict(stage='value_train',step=step,loss=float(reduced),lr=optimizer.param_groups[0]['lr'],
                    seconds_per_step=(time.monotonic()-started)/(batch_index-start_batch+1))),flush=True)
            if step%500==0 or step==args.steps:
                metrics=evaluate()
                improved=metrics['rollout_val_rmse']<best
                best=min(best,metrics['rollout_val_rmse'])
                if rank==0:
                    saved=dict(model=model.state_dict(),metadata=metadata,metrics=metrics,backend='native_rlinf',
                        config=model.config.to_dict(),upstream_commit=record['upstream_commit'])
                    atomic_torch_save(saved,args.output/'last.pt')
                    if improved:atomic_torch_save(saved,args.output/'best.pt')
                save_resume(epoch,batch_index+1)
            if step>=args.steps:break
        epoch+=1;start_batch=0
        if step<args.steps:save_resume(epoch,0)
    if rank==0:(args.output/'completed.json').write_text(json.dumps(dict(steps=step,epochs=epoch,best_rollout_val_rmse=best,backend='native_rlinf'),indent=2))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[1]
    p.add_argument('--data',type=Path,default=base/'data/recap_rlinf_1563_20261020')
    p.add_argument('--output',type=Path,default=base/'outputs/recap_rlinf_value_1563_20261020')
    p.add_argument('--siglip',type=Path,default=base.parent/'models/siglip2-so400m-patch14-224')
    p.add_argument('--gemma',type=Path,default=base.parent/'models/gemma-3-270m')
    p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())
