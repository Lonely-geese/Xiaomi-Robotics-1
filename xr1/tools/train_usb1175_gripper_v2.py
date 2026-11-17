"""Full XR1 asynchronous fine-tuning, with local metrics and epoch exports."""
import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timedelta
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('XR1_USB_PREPARED_DIR',str(Path(__file__).resolve().parents[2]/'prepared_gripper_v2'))
import torch
import torch.distributed as dist
from lightning import Trainer,seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from mibot.usb1175 import ROOT,PREPARED,USB1175Runner,USB1175DataModule

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def configuration(stage=2,dit_checkpoint=False):
    return dict(pretrained=str(ROOT/'models/official/model_states.pt'),
        model=dict(type='xr1',async_train=True,full_finetune=True,active_action_dims=list(range(7)),
                   freq_coefficient=1.0,freq_excluded_dims=[6,17,18,19],ffn_gradient_checkpointing=True),
        optimizer=dict(type='torch.optim.AdamW',params=dict(lr=1.0,betas=[0.9,0.95],weight_decay=0.1,eps=1e-8,fused=True)),
        scheduler=dict(type='mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup',params=dict(
            num_training_steps=23100,num_warmup_steps=693,warmup_lr_start=5e-7,max_lr=2e-5,min_lr=5e-6)))

class Record(Callback):
    def __init__(self,output,benchmark=False):
        self.output=Path(output);self.benchmark=benchmark;self.durations=deque(maxlen=90)
        self.best=math.inf;self.last_step=0;self.saved_epochs=set()
    def state_dict(self):return dict(best=self.best,saved_epochs=sorted(self.saved_epochs))
    def load_state_dict(self,state_dict):
        self.best=state_dict['best'];self.saved_epochs=set(state_dict['saved_epochs'])
    def on_fit_start(self,trainer,module):
        assert trainer.world_size==8 and trainer.accumulate_grad_batches==1
        assert module.model.training_repeat==4 and module.model.async_train and module.model.full_finetune
        assert all(p.requires_grad for p in module.parameters())
        self.gradient_seen=set();self.gradient_handles=[]
        for name,p in module.named_parameters():
            group=('vision' if '.visual.' in name else 'language' if '.language_model.' in name else 'dit' if '.dit.' in name else None)
            if group and group not in {g for g,_ in getattr(self,'gradient_watched',[])}:
                self.gradient_watched=getattr(self,'gradient_watched',[])+[(group,name)]
                def observed(grad,group=group):
                    self.gradient_seen.add(group)
                    return grad
                self.gradient_handles.append(p.register_hook(observed))
        assert {g for g,_ in self.gradient_watched}=={'vision','language','dit'}
        if trainer.is_global_zero:
            write(self.output/'initialization.json',dict(source='official',official_sha256=json.loads((ROOT/'models/official/download.json').read_text())['expected_sha256'],
                async_train=True,full_finetune=True,training_repeat=4,world_size=8,unique_samples_per_gpu=32,
                global_batch_size=256,gradient_accumulation_steps=1,stage=trainer.strategy.config['zero_optimization']['stage'],
                dit_checkpoint=os.environ.get('XR1_DIT_CHECKPOINT')=='1',trainable_parameters=sum(getattr(p,'ds_numel',p.numel()) for p in module.parameters()),
                steps_per_epoch=1155,epochs=20,total_steps=23100,validation_samples=128,proxies_used=False,gripper_repair_version=2,all_parameters_unfrozen=True))
    def on_train_start(self,trainer,module):
        assert len(trainer.train_dataloader)==1155
        self.last_time=time.perf_counter();self.last_step=trainer.global_step
        torch.cuda.reset_peak_memory_stats(module.device)
    def on_train_epoch_start(self,trainer,module):
        trainer.train_dataloader.sampler.set_epoch(trainer.current_epoch)
        self.last_time=time.perf_counter()
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        assert batch['action'].shape==(32,30,60)
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        loss=outputs['loss'].detach()
        finite=torch.isfinite(loss).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError('Nonfinite training loss')
        if batch_idx==2:
            assert self.gradient_seen=={'vision','language','dit'},self.gradient_seen
            if trainer.is_global_zero:write(self.output/'gradient_verified.json',dict(passed=True,branches=sorted(self.gradient_seen)))
            for handle in self.gradient_handles:handle.remove()
        norm=trainer.strategy.deepspeed_engine.get_global_grad_norm()
        if norm is not None and not math.isfinite(float(norm)):raise FloatingPointError('Nonfinite gradient norm')
        now=time.perf_counter();step=trainer.global_step
        duration=(now-self.last_time)/max(1,step-self.last_step)
        self.last_time=now;self.last_step=step
        if batch_idx>=5:self.durations.append(duration)
        if not trainer.is_global_zero or (step!=1 and step%10):return
        seconds=sum(self.durations)/len(self.durations) if self.durations else None
        metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1}
        record=dict(state='training',global_step=step,epoch=step/1155,max_steps=trainer.max_steps,
                    world_size=8,per_gpu_batch=32,global_batch_size=256,accumulation=1,
                    seconds_per_step=seconds,updated_at=datetime.now().astimezone().isoformat(),
                    global_gradient_norm=float(norm) if norm is not None else None,
                    eta_seconds=(trainer.max_steps-step)*seconds if seconds else None,
                    allocated_gib=torch.cuda.memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    metrics=metrics)
        write(self.output/'progress.json',record)
        with (self.output/'progress.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print('TRAIN_PROGRESS '+json.dumps(record),flush=True)
    def on_train_epoch_end(self,trainer,module):
        step=trainer.global_step
        epoch=0 if self.benchmark else trainer.current_epoch+1
        assert step==(100 if self.benchmark else epoch*1155),(epoch,step)
        assert epoch not in self.saved_epochs
        value=trainer.callback_metrics.get('val/loss')
        assert value is not None,'Validation must run before saving an epoch'
        val=float(value.detach().cpu());assert math.isfinite(val)
        self.saved_epochs.add(epoch)
        improved=val<self.best
        if improved:self.best=val
        # Stage 2 has replicated weights. Stage 3 must gather one parameter at a time on all ranks.
        state={};original_names={}
        for name,p in module.named_parameters(remove_duplicate=False):
            if id(p) in original_names:
                if trainer.is_global_zero:state[name]=state[original_names[id(p)]]
                continue
            original_names[id(p)]=name
            if hasattr(p,'ds_id'):
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                context=GatheredParameters([p],modifier_rank=None)
            else:context=nullcontext()
            with context:
                if trainer.is_global_zero:state[name]=p.detach().cpu().clone()
        if trainer.is_global_zero:
            for name,b in module.named_buffers():
                if name in module.state_dict():state[name]=b.detach().cpu().clone()
            target=self.output/'weights'/f'epoch{epoch:02d}-step{step}.pt';target.parent.mkdir(exist_ok=True)
            temporary=target.with_suffix('.partial')
            torch.save(dict(module=state,epoch=epoch,global_step=step,validation_loss=val,
                            normalization_sha256=hashlib.sha256((PREPARED/'normalize.json').read_bytes()).hexdigest()),temporary)
            temporary.replace(target)
            info=dict(epoch=epoch,global_step=step,validation_loss=val,path=str(target))
            write(self.output/'latest_weights.json',info)
            if improved:write(self.output/'best_validation.json',info)
            with (self.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
        del state
        trainer.strategy.barrier()
        # Atomic replacement retains one complete resumable optimizer/scheduler state.
        temporary=self.output/'resume.partial';latest=self.output/'resume';backup=self.output/'resume.previous'
        if trainer.is_global_zero and temporary.exists():shutil.rmtree(temporary)
        trainer.strategy.barrier()
        trainer.save_checkpoint(str(temporary),weights_only=False)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            if backup.exists():shutil.rmtree(backup)
            if latest.exists():latest.replace(backup)
            temporary.replace(latest)
            if backup.exists():shutil.rmtree(backup)
        trainer.strategy.barrier()
        self.last_time=time.perf_counter()
    def on_train_end(self,trainer,module):
        # Lightning moves the model to CPU during teardown, before on_fit_end.
        # NCCL reductions must therefore finish while the training device is live.
        memory=torch.tensor([torch.cuda.max_memory_allocated()/2**30,torch.cuda.max_memory_reserved()/2**30],device=module.device)
        dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        self.final_memory=memory.cpu().tolist()
    def on_fit_end(self,trainer,module):
        if trainer.is_global_zero:
            memory=self.final_memory
            expected=100 if self.benchmark else 23100
            assert trainer.global_step==expected
            if not self.benchmark:assert len(self.saved_epochs)==20
            seconds=sum(self.durations)/len(self.durations) if self.durations else None
            validation=trainer.callback_metrics.get('val/loss');assert validation is not None
            result=dict(state='completed',global_step=trainer.global_step,epochs=20 if not self.benchmark else None,
                        benchmark=self.benchmark,async_train=True,per_gpu_batch=32,global_batch_size=256,
                        accumulation=1,seconds_per_step=seconds,
                        p95_seconds_per_step=sorted(self.durations)[min(len(self.durations)-1,int(len(self.durations)*0.95))] if self.durations else None,
                        peak_allocated_gib=float(memory[0]),peak_reserved_gib=float(memory[1]),validation_loss=float(validation),
                        checkpoint_saved=(self.output/'latest_weights.json').exists() and (self.output/'resume/checkpoint').is_dir(),
                        updated_at=datetime.now().astimezone().isoformat())
            write(self.output/'completed.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true');parser.add_argument('--stage',type=int,choices=[2,3],default=2)
    parser.add_argument('--dit-checkpoint',action='store_true');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    if (ROOT/'queue_gripper_v2/STOP_REQUESTED.json').exists():
        raise SystemExit('New gripper-v2 training stopped by user.')
    proof=json.loads((PREPARED/'input_verified.json').read_text())
    assert proof['passed'] and proof['gripper_repair_version']==2
    assert proof['manifest_sha256']==hashlib.sha256((PREPARED/'manifest.json').read_bytes()).hexdigest()
    for key in list(os.environ):
        if key.lower().endswith('_proxy'):os.environ.pop(key)
    os.environ.update(NO_PROXY='*',no_proxy='*',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        XR1_DIT_CHECKPOINT='1' if args.dit_checkpoint else '0',MAX_LENGTH='32768',XR1_STRICT_BATCH='1')
    assert json.loads((PREPARED/'input_verified.json').read_text())['passed']
    assert json.loads((ROOT/'models/official/download.json').read_text())['verified']
    args.output.mkdir(parents=True,exist_ok=True)
    seed_everything(42,workers=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    params=configuration(args.stage,args.dit_checkpoint)
    if args.resume:params['pretrained']=None
    if int(os.environ.get('RANK','0'))==0:
        write(args.output/'config.json',dict(model=params,arguments=vars(args)|{'output':str(args.output),'resume':str(args.resume) if args.resume else None},
              source_dataset=str(PREPARED/'manifest.json'),camera_preprocessing='existing verified RGB crops',
              control='left end-effector: chunk-local xyz/rotvec; absolute gripper command through fixed anchor',validation_session_disjoint=True))
        shutil.copy2(PREPARED/'normalize.json',args.output/'normalize.json')
        shutil.copy2(PREPARED/'manifest.json',args.output/'dataset_manifest.json')
        shutil.copy2(PREPARED/'gripper_contract.json',args.output/'gripper_contract.json')
        shutil.copy2(ROOT/'xr1/mibot/gripper_contract_v2.py',args.output/'gripper_contract_v2.py')
        shutil.copy2(PREPARED/'input_verified.json',args.output/'input_verified.json')
    strategy=DeepSpeedStrategy(stage=args.stage,allgather_bucket_size=50_000_000,reduce_bucket_size=50_000_000,
        overlap_comm=True,contiguous_gradients=True,zero_allow_untested_optimizer=True,logging_batch_size_per_gpu=32,
        timeout=timedelta(minutes=30))
    trainer=Trainer(accelerator='gpu',devices=8,num_nodes=1,strategy=strategy,precision='bf16-mixed',
        max_epochs=1 if args.benchmark else 20,max_steps=100 if args.benchmark else 23100,
        limit_train_batches=100 if args.benchmark else 1.0,num_sanity_val_steps=0,
        accumulate_grad_batches=1,gradient_clip_val=1.0,gradient_clip_algorithm='norm',
        use_distributed_sampler=False,check_val_every_n_epoch=1,enable_checkpointing=False,
        enable_model_summary=False,enable_progress_bar=False,log_every_n_steps=10,
        logger=CSVLogger(str(args.output),name='csv'),callbacks=[Record(args.output,args.benchmark)],
        default_root_dir=str(args.output))
    trainer.fit(USB1175Runner(params),datamodule=USB1175DataModule(args.workers),ckpt_path=str(args.resume) if args.resume else None)

if __name__=='__main__':main()
