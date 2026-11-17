"""Cached observations and exact official XR1 token/target conventions."""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import copy,json,os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,DistributedSampler,default_collate
from lightning import LightningDataModule
import torch.distributed as dist
from mibot.data.datamodule.base_datamodule import CompleteBatchSampler
from mibot.data.collate.custom_collate import CustomCollate,STATE_ID,IM_START_ID
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.models.runner.base_runner import BaseRunner
from mibot.models import MIMODEL
from mibot.utils.io import normalize_quantile

ROOT=Path(__file__).resolve().parents[2]
PREPARED=Path(os.environ.get('XR1_USB_PREPARED_DIR',str(ROOT/'prepared')))

def prepared_path(path):
    """Allow portable manifests with paths relative to their prepared directory."""
    value=Path(path)
    return value if value.is_absolute() else PREPARED/value

@lru_cache(maxsize=96)
def array(path):return np.load(prepared_path(path),mmap_mode='r',allow_pickle=False)

class USB1175Dataset(Dataset):
    def __init__(self,split):
        manifest=json.loads((PREPARED/'manifest.json').read_text());self.split=split
        self.rows=[r for r in manifest['rows'] if r['split']==split]
        stats=json.loads((PREPARED/'normalize.json').read_text())
        self.mean=np.asarray(stats['mean'],np.float32)[:,:7];self.std=np.asarray(stats['std'],np.float32)[:,:7]
        self.q01=np.asarray(stats['q01'],np.float32);self.q99=np.asarray(stats['q99'],np.float32)
        self.ends=np.cumsum([r['length'] for r in self.rows]).tolist()
        lookup={r['uid']:i for i,r in enumerate(self.rows)}
        self.panel=[(lookup[r['uid']],r['frame']) for r in manifest['validation_panel']] if split=='val' else None
    def __len__(self):return len(self.panel) if self.panel is not None else self.ends[-1]
    def locate(self,index):
        if self.panel is not None:return self.panel[index]
        ep=bisect_right(self.ends,index);return ep,index-(self.ends[ep-1] if ep else 0)
    def __getitem__(self,index):
        ep,frame=self.locate(index);row=self.rows[ep];steps=min(30,row['length']-frame)
        values=(array(row['actions'])[frame]-self.mean)/(self.std+1e-6)
        action=np.zeros((30,60),np.float32);action[:,:7]=values
        mask=np.zeros((30,60),bool);mask[:steps,:7]=True;action[~mask]=0
        if 'gripper_mask' in row:
            mask[:,6] &= array(row['gripper_mask'])[frame]
            action[~mask]=0
        state=normalize_quantile(array(row['states'])[frame:frame+1],self.q01,self.q99)
        rgb=np.stack([array(p)[frame] for p in row['cameras']])
        return dict(rgb=torch.from_numpy(rgb),action=torch.from_numpy(action),action_mask=torch.from_numpy(mask),
            state=torch.from_numpy(state),vlm_action_target=torch.from_numpy(action[:steps].copy()),
            vlm_action_mask=torch.from_numpy(mask[:steps].copy()),vlm_action_actual_length=steps)

class CachedCollate:
    def __init__(self):
        self.templates=torch.load(PREPARED/'templates.pt',map_location='cpu',weights_only=True)
    def __call__(self,items):
        b=len(items);inputs=[self.templates[int(item['vlm_action_actual_length'])] for item in items]
        lengths=[x['input_ids'].shape[1] for x in inputs];offsets=np.r_[0,np.cumsum(lengths)].tolist()
        assert offsets[-1]<=32768,'Packed token budget exceeded; samples must never be dropped'
        out=dict(input_ids=torch.cat([x['input_ids'] for x in inputs],1),position_ids=torch.cat([x['position_ids'] for x in inputs],2),
            cu_seq_lens_q=torch.tensor(offsets,dtype=torch.int32),cu_seq_lens_k=torch.tensor(offsets,dtype=torch.int32),
            max_length_q=max(lengths),max_length_k=max(lengths),
            action_segments=torch.tensor(list(zip(offsets[:-1],offsets[1:])),dtype=torch.long),
            action_vlm_condition_segments=torch.tensor([[offsets[i],offsets[i]+x['prefix_end']] for i,x in enumerate(inputs)],dtype=torch.long))
        payloads=[dict(item) for item in items]
        targets=[p.pop('vlm_action_target') for p in payloads];masks=[p.pop('vlm_action_mask') for p in payloads]
        out.update(default_collate(payloads));out['vlm_action_target']=torch.cat(targets);out['vlm_action_mask']=torch.cat(masks)
        assert out['action'].shape[0]==b
        return out

def build_templates():
    reference=CustomCollate();manifest=json.loads((PREPARED/'manifest.json').read_text());prompt=manifest['rows'][0]['prompt']
    assert all(row['prompt']==prompt for row in manifest['rows']), 'Cached token templates require the same instruction'
    native=object.__new__(JsonDataset);native.action_length=30
    dummy=[Image.fromarray(np.zeros((256,256,3),np.uint8)) for _ in range(3)]
    templates={}
    for steps in range(1,31):
        p=native._prompt(dict(instruction=dict(general=[prompt])))
        p['conversations'] += [{'from':'human','value':'Robot state: <state>'},{'from':'gpt','value':''.join(f'<a_{i}>' for i in range(steps))+'<score>'}]
        messages=native._messages(p['conversations'],dummy)
        row=reference.processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',do_resize=False)
        position=reference._position_ids(row);ids=row['input_ids'][0];state=(ids==STATE_ID).nonzero()[0,0];starts=(ids==IM_START_ID).nonzero().flatten();end=int(starts[starts<state][-1])
        templates[steps]=dict(input_ids=row['input_ids'],position_ids=position,prefix_end=end,
                              image_grid_thw=row['image_grid_thw'])
    torch.save(templates,PREPARED/'templates.pt')

class USB1175DataModule(LightningDataModule):
    def __init__(self,workers=4):
        super().__init__();self.workers=workers;self.batch_size=32;self.collate=CachedCollate()
    def setup(self,stage=None):
        if not hasattr(self,'train_set'):self.train_set=USB1175Dataset('train');self.val_set=USB1175Dataset('val')
    def loader(self,dataset,train):
        world=dist.get_world_size() if dist.is_initialized() else 1;rank=dist.get_rank() if dist.is_initialized() else 0
        args=dict(num_replicas=world,rank=rank,shuffle=train,seed=42)
        sampler=CompleteBatchSampler(dataset,256,**args) if train else DistributedSampler(dataset,**args)
        return DataLoader(dataset,batch_size=32 if train else 16,sampler=sampler,drop_last=train,
            collate_fn=self.collate,num_workers=self.workers,pin_memory=True,persistent_workers=self.workers>0,
            prefetch_factor=2 if self.workers else None)
    def train_dataloader(self):return self.loader(self.train_set,True)
    def val_dataloader(self):return self.loader(self.val_set,False)

class USB1175Runner(BaseRunner):
    def configure_model(self):
        if hasattr(self,'model'):return
        self.model=MIMODEL.build(self._model)
        if self._pretrained:
            saved=torch.load(self._pretrained,map_location='cpu',mmap=True,weights_only=False)['module']
            expected=self.state_dict()
            assert expected.keys()==saved.keys(), 'Official checkpoint keys must match exactly'
            parameters=dict(self.named_parameters(remove_duplicate=False))
            for name,value in expected.items():
                parameter=parameters.get(name)
                shape=tuple(parameter.ds_shape) if parameter is not None and hasattr(parameter,'ds_shape') else tuple(value.shape)
                assert shape==tuple(saved[name].shape),(name,shape,saved[name].shape)
            if any(hasattr(p,'ds_id') for p in parameters.values()):
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                visited=set()
                for name,p in parameters.items():
                    if id(p) in visited:continue
                    visited.add(id(p))
                    with GatheredParameters([p],modifier_rank=0):
                        if dist.get_rank()==0:
                            with torch.no_grad():p.copy_(saved[name].to(device=p.device,dtype=p.dtype))
                for name,b in self.named_buffers():
                    if name in saved:b.copy_(saved[name].to(device=b.device,dtype=b.dtype))
            else:self.load_state_dict(saved,strict=True)
            del saved,expected
        assert self.model.async_train is True and self.model.full_finetune is True
        assert all(p.requires_grad for p in self.model.parameters()),'Full fine-tuning must not freeze parameters'
        if os.environ.get('XR1_DIT_CHECKPOINT','0')=='1':
            from torch.utils.checkpoint import checkpoint
            for layer in self.model.dit.layers:
                original=layer.forward
                def forward(*args,_forward=original,_layer=layer,**kwargs):
                    if _layer.training and torch.is_grad_enabled():return checkpoint(_forward,*args,use_reentrant=False,**kwargs)
                    return _forward(*args,**kwargs)
                layer.forward=forward
    def _prepare_images(self,batch):
        from transformers import AutoProcessor
        if not hasattr(self,'image_processor'):
            self.image_processor=AutoProcessor.from_pretrained(os.environ['XR1_VLM_PATH'],use_fast=True,local_files_only=True).image_processor
        rgb=batch.pop('rgb').flatten(0,1).permute(0,3,1,2)
        with torch.no_grad():
            image=self.image_processor(images=rgb,device=rgb.device,do_resize=False,return_tensors='pt')
            if not getattr(self,'pixel_device_verified',False):
                reference=self.image_processor(images=rgb[:1].cpu(),device='cpu',do_resize=False,return_tensors='pt')
                count=reference['pixel_values'].shape[0]
                torch.testing.assert_close(image['pixel_values'][:count].cpu(),reference['pixel_values'],rtol=0,atol=1e-6)
                torch.testing.assert_close(image['image_grid_thw'][:1].cpu(),reference['image_grid_thw'],rtol=0,atol=0)
                self.pixel_device_verified=True
        batch.update({key:value.to(self.device,non_blocking=True) for key,value in image.items()})
        return batch
    def training_step(self,batch,batch_idx):
        assert batch['action'].shape[0]==32 and self.trainer.accumulate_grad_batches==1
        return super().training_step(self._prepare_images(batch),batch_idx)
    def validation_step(self,batch,batch_idx):
        return super().validation_step(self._prepare_images(batch),batch_idx)
