"""Portable inference, truthful transition recording and incremental pose-IQL.

Hardware transport is supplied by the user's adapter. This module never invents
executed actions, rewards or successor observations and never sends robot commands.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mibot.bee.model import load_bee
from mibot.bee.offline import OfflineIQL
from mibot.bee.ensemble import combine_actions,constrain_to_correction
from mibot.bee.data import robot_state,make_batch
from mibot.bee.supervision import human_to_normalized,sha256
from mibot.utils.io import ACTION_EPS,recover_action,rotm2aa_batch

FIELDS=('z','proprio','reference','executed_action','valid_mask','next_z','next_proprio',
        'terminated','horizon','reward','discount','human','episode','frame')


def feature_signature(assets):
    keys=('stage1_sha256','base_sha256','normalization_sha256','features','action_coordinates')
    return hashlib.sha256(json.dumps({k:assets[k] for k in keys},sort_keys=True).encode()).hexdigest()


class RobotPolicy:
    def __init__(self,weights,checkpoint,device='cuda:0',asset_cache=None):
        from transformers import AutoProcessor
        from mibot.rlt.model import load_config,verify_package,validate_assets,build_model
        weights=Path(weights)
        config=load_config(weights/'config.yaml')
        verified=verify_package(weights,asset_cache)
        self.bee,self.metadata=load_bee(checkpoint,device)
        stage=torch.load(weights/'stage1_step20000.pt',map_location='meta',weights_only=True,mmap=True)
        validate_assets(config,stage,self.metadata,verified['sha256'])
        self.frozen=build_model(config,device,weights/'stage1_step20000.pt',profile='direct-cache')
        self.processor=AutoProcessor.from_pretrained(config['vlm_path'],local_files_only=True)
        self.processor.tokenizer.padding_side='right'
        self.norm={k:np.array(v,np.float32) for k,v in json.loads((weights/'normalize.json').read_text()).items()}
        self.device=device;self.signature=feature_signature(self.metadata['assets'])

    @torch.inference_mode()
    def observe(self,observation,seed=42):
        """obs: id, timestamp (seconds), state30, 3 PIL RGB images, instruction."""
        if not str(observation['id']) or not np.isfinite(observation['timestamp']):raise ValueError('Observation ID and real acquisition timestamp required')
        state=robot_state(observation['state30'])
        if len(observation['images'])!=3:raise ValueError('Expected ego/left wrist/right wrist images')
        batch,audit=make_batch(self.processor,observation['images'],observation['instruction'],state,self.norm,self.device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        prediction=self.frozen(batch)
        return dict(id=str(observation['id']),timestamp=float(observation['timestamp']),
                    state30=np.asarray(observation['state30'],np.float32).copy(),
                    z=prediction['z'][0].float().cpu().numpy(),proprio=batch['state'][0,0,:16].float().cpu().numpy(),
                    reference=prediction['normalized_action'][0,:10,:7].float().cpu().numpy(),
                    feature_signature=self.signature,seed=seed,state_saturation_count=audit['state_saturation_count'])

    @torch.inference_mode()
    def action(self,packet,mode='actor',actor_weight=.75,epsilon=.3):
        if packet['feature_signature']!=self.signature:raise ValueError('Feature interface mismatch')
        z,p,r=[torch.as_tensor(packet[k],device=self.device).float()[None] for k in ('z','proprio','reference')]
        if mode=='vla':action=r
        elif mode=='correction':action=self.bee.correct(z,p,r)['corrected']
        else:
            action=self.bee.actor_action(z,p,r)
            if mode in ('blend','constrained'):
                dist=self.bee.distribution(z,p,r)
                if mode=='blend':action=combine_actions(action,dist['corrected'],r,self.bee.editable,actor_weight=actor_weight)
                else:action=constrain_to_correction(action,dist['corrected'],dist['variance'],r,self.bee.editable,epsilon=epsilon)['action']
            elif mode!='actor':raise ValueError('Unknown output mode')
        normalized=action[0].cpu().numpy()
        raw=np.zeros((10,60),np.float32)
        raw[:,:7]=normalized*(self.norm['std'][:10,:7]+ACTION_EPS)+self.norm['mean'][:10,:7]
        targets=recover_action(raw,robot_state(packet['state30']))
        return dict(normalized_action=normalized,absolute_pose6=np.concatenate([targets['left_ee_pos'],rotm2aa_batch(targets['left_ee_rotm'])],-1),
                    gripper_m=targets['left_gripper_pos'].reshape(-1),mode=mode,actor_steps=self.bee.trained_steps,
                    observation_id=packet['id'])


def make_transition(packet,executed_pose6,rewards,terminated,next_packet,human,norm,episode,frame, gamma=.99):
    """The adapter must report the commands ACTUALLY executed after intervention/clipping.

    Homogeneous chunks only: split at human/autonomous switches. Unknown successor
    is an error, never a fabricated terminal. Missing gripper commands stay absent.
    """
    executed=np.asarray(executed_pose6,np.float32);rewards=np.asarray(rewards,np.float32)
    if executed.ndim!=2 or executed.shape[1]!=6 or not 1<=len(executed)<=10:raise ValueError('Expected 1..10 actually executed absolute pose6 commands')
    n=len(executed)
    if rewards.shape!=(n,) or not np.isfinite(rewards).all() or not np.isfinite(executed).all():raise ValueError('Real per-step rewards and finite executed commands required')
    if not isinstance(terminated,(bool,np.bool_)) or not isinstance(human,(bool,np.bool_)):raise ValueError('Explicit terminated/human booleans required')
    if not terminated:
        if next_packet is None:raise ValueError('Missing true successor observation')
        if packet['feature_signature']!=next_packet['feature_signature']:raise ValueError('Successor uses a different frozen interface')
        if packet['id']==next_packet['id'] or next_packet['timestamp']<=packet['timestamp']:raise ValueError('Successor must be newly acquired after execution')
    target,proprio=human_to_normalized(np.asarray(packet['state30'],np.float32),executed,np.asarray(packet['reference'][:n],np.float32),norm)
    if not np.allclose(proprio,packet['proprio'],atol=1e-6):raise ValueError('Proprio normalization mismatch')
    action=np.asarray(packet['reference'],np.float32).copy();action[:n]=target
    row=dict(z=np.asarray(packet['z'],np.float32),proprio=proprio,reference=np.asarray(packet['reference'],np.float32),executed_action=action,
             valid_mask=np.arange(10)<n,next_z=np.zeros_like(packet['z'],dtype=np.float32) if terminated else np.asarray(next_packet['z'],np.float32),
             next_proprio=np.zeros_like(proprio) if terminated else np.asarray(next_packet['proprio'],np.float32),
             terminated=bool(terminated),horizon=n,reward=sum(gamma**j*float(r) for j,r in enumerate(rewards)),discount=gamma**n,
             human=bool(human),episode=int(episode),frame=int(frame))
    trace=dict(feature_signature=packet['feature_signature'],observation_id=packet['id'],timestamp=packet['timestamp'],
               next_observation_id=None if terminated else next_packet['id'],next_timestamp=None if terminated else next_packet['timestamp'],
               executed_pose6=executed.tolist(),primitive_rewards=rewards.tolist(),state30=np.asarray(packet['state30']).tolist(),
               note='Critic omits unrecorded gripper commands; pose6+validity-mask encoding')
    return row,trace


def save_transition(path,row,trace):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**{k:np.asarray(row[k]) for k in FIELDS},trace_json=np.array(json.dumps(trace)))


def read_transition(path,signature):
    with np.load(path,allow_pickle=False) as a:
        trace=json.loads(str(a['trace_json']))
        if trace['feature_signature']!=signature:raise ValueError('Recorded transition feature signature mismatch')
        return {k:a[k].copy() for k in FIELDS}


class IncrementalLearner:
    def __init__(self,checkpoint,seed_replay,device='cuda:0'):
        self.model,self.meta=load_bee(checkpoint,device)
        if self.meta.get('training',{}).get('mode') not in ('iql','incremental_iql') or 'optimizers' not in self.meta:
            raise ValueError('A trained IQL checkpoint with optimizer state is required')
        self.learner=OfflineIQL(self.model,expectile=self.model.config.get('offline_expectile',.7),temperature=self.model.config.get('offline_temperature',3.))
        for k,optimizer in self.learner.optimizers.items():optimizer.load_state_dict(self.meta['optimizers'][k])
        self.learner.critic_updates=int(self.meta['critic_steps'])
        self.generator=torch.Generator();self.generator.set_state(self.meta['generator_state'].cpu())
        self.device=device;self.signature=feature_signature(self.meta['assets']);self.new=[]
        if sha256(seed_replay)!=self.meta['training']['replay_sha256']:raise ValueError('Seed replay hash mismatch')
        source=torch.load(seed_replay,map_location='cpu',weights_only=True)
        if source['episodes']!=self.meta['training']['episodes']:raise ValueError('Seed replay episode split mismatch')
        # Holdout episodes are NEVER replayed in training.
        ix=source['splits']['train'];self.seed={k:source['data'][k][ix].clone() for k in FIELDS}
        self.human=torch.where(self.seed['human'])[0]
        self._validate_batch(self.seed)

    def _validate_batch(self,b):
        count=len(b['z']);m=self.model
        shapes={'z':(count,m.z_dim),'proprio':(count,m.proprio_dim),'reference':(count,10,7),'executed_action':(count,10,7),
                'valid_mask':(count,10),'next_z':(count,m.z_dim),'next_proprio':(count,m.proprio_dim)}
        for k,v in b.items():
            if k in shapes and tuple(v.shape)!=shapes[k]:raise ValueError('Invalid replay shape: '+k)
            if not torch.isfinite(v).all():raise ValueError('Nonfinite replay field: '+k)
        if not ((b['horizon']>=1)&(b['horizon']<=10)).all():raise ValueError('Invalid horizon')
        expected=torch.arange(10)[None]<b['horizon'][:,None]
        if not torch.equal(b['valid_mask'].bool(),expected):raise ValueError('Horizon mask mismatch')
        if not torch.allclose(b['discount'].float(),.99**b['horizon'].float(),atol=1e-6):raise ValueError('Discount mismatch')

    def add(self,row):
        row={k:torch.as_tensor(row[k]).clone() for k in FIELDS}
        self._validate_batch({k:v[None] for k,v in row.items()});self.new.append(row)

    def update(self,batch_size=256):
        if batch_size%4:raise ValueError('Batch must be divisible by four')
        # With new data: 1/4 recent transitions, remaining 3/4 balanced seed replay/human.
        # Without new data: ordinary balanced replay/human; used only for package smoke.
        recent=batch_size//4 if self.new else 0
        rest=batch_size-recent
        all_ix=torch.randint(len(self.seed['z']),(rest//2,),generator=self.generator)
        human_ix=self.human[torch.randint(len(self.human),(rest-rest//2,),generator=self.generator)]
        ix=torch.cat([all_ix,human_ix]);b={k:v[ix] for k,v in self.seed.items()}
        if recent:
            selected=torch.randint(len(self.new),(recent,),generator=self.generator).tolist()
            b={k:torch.cat([v,torch.stack([self.new[i][k] for i in selected])]) for k,v in b.items()}
        float_fields=('z','proprio','reference','executed_action','next_z','next_proprio','reward','discount')
        b={k:(v.float() if k in float_fields else v).to(self.device) for k,v in b.items()}
        self.model.train()
        return self.learner.update(b,train_actor=True)

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        ckpt=dict(self.meta)
        ckpt.update(format='xr1-bee-v1',config=self.model.config,model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                    step=self.model.trained_steps,critic_steps=self.learner.critic_updates,value_steps=self.learner.critic_updates,
                    iteration=self.learner.critic_updates,
                    optimizers={k:v.state_dict() for k,v in self.learner.optimizers.items()},generator_state=self.generator.get_state(),
                    torch_rng_state=torch.get_rng_state())
        ckpt['training']=dict(self.meta['training'],mode='incremental_iql',incremental_samples=len(self.new),
                              continued_actor_lr=self.learner.optimizers['actor'].param_groups[0]['lr'],
                              replay_sampling='25% new + 75% balanced seed/human when new data available',
                              correction_frozen=True,scope='Incremental pose-IQL on collected real transitions, not original online BEE')
        ckpt['continuation_note']='Weights/Adam/RNG restored. Fixed saved learning rates; correction frozen. Reload transition files when restarting.'
        if 'validation' in ckpt:ckpt['source_validation']=ckpt.pop('validation')
        ckpt['evaluation_status']='new candidate; not evaluated'
        torch.save(ckpt,path)
