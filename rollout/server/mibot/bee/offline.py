"""IQL for audited offline pose chunks; deliberately distinct from online BEE.

Expectile V, reward-driven twin-Q TD, advantage-weighted policy extraction.
Deterministic residual mean uses fixed unit-variance Gaussian likelihood.
"""
import math
import torch
from mibot.bee.learning import adam, optimize


def expectile_loss(diff, expectile=.7):
    return (torch.where(diff > 0, expectile, 1-expectile)*diff.square()).mean()


def advantage_weights(q, value, temperature=3., max_weight=100.):
    return ((q-value).detach()*temperature).clamp(max=math.log(max_weight)).exp()


def pose_likelihood_loss(action, executed, valid_mask, weights):
    # Unit variance Gaussian NLL without the parameter-independent constant.
    # Tail padding and unobserved gripper never enter the objective.
    error=(action[...,:6]-executed[...,:6]).masked_fill(~valid_mask[...,None],0)
    return (.5*error.square().sum((1,2))*weights.detach()).mean()


def chunk_target(reward, discount, terminated, next_value):
    return reward + discount*next_value.masked_fill(terminated.bool(),0)


class OfflineIQL:
    def __init__(self, model, actor_lr=1e-4, expectile=.7, temperature=3.):
        if model.config.get('algorithm') != 'bee-offline-iql-v1':
            raise ValueError('Offline critic encoding must be explicitly configured')
        self.model=model.requires_grad_(False)
        for module in (model.actor,model.critics,model.value):module.requires_grad_(True)
        self.optimizers={k:adam(getattr(model,k).parameters(),actor_lr if k=='actor' else 3e-4)
                         for k in ('actor','critics','value')}
        self.expectile=expectile;self.temperature=temperature
        self.critic_updates=0

    def advantages(self,b):
        m=self.model
        with torch.no_grad():
            q=m.q_values(b['z'],b['proprio'],b['executed_action'],target=True,valid_mask=b['valid_mask']).min(-1).values
            value=m.state_value(b['z'],b['proprio'])
            weights=advantage_weights(q,value,self.temperature)
        return q,value,weights

    def update(self,b,train_actor=True):
        m=self.model
        q,_,_=self.advantages(b)
        value=m.state_value(b['z'],b['proprio'])
        vloss=expectile_loss(q-value,self.expectile)
        optimize(vloss,self.optimizers['value'],m.value.parameters())
        with torch.no_grad():
            next_value=torch.zeros_like(b['reward'])
            live=~b['terminated']
            if live.any():next_value[live]=m.state_value(b['next_z'][live],b['next_proprio'][live])
            target=chunk_target(b['reward'],b['discount'],b['terminated'],next_value)
        predictions=m.q_values(b['z'],b['proprio'],b['executed_action'],valid_mask=b['valid_mask'])
        qloss=(predictions-target[:,None]).square().sum(-1).mean()
        optimize(qloss,self.optimizers['critics'],m.critics.parameters())
        with torch.no_grad():
            for target_p,current in zip(m.target_critics.parameters(),m.critics.parameters()):target_p.lerp_(current,.005)
        self.critic_updates+=1
        q,value,weights=self.advantages(b)
        result=dict(q_loss=float(qloss.detach()),value_loss=float(vloss.detach()),q_mean=float(q.mean()),
                    advantage_std=float((q-value).std()),weight_mean=float(weights.mean()),weight_std=float(weights.std()),
                    weight_min=float(weights.min()),weight_max=float(weights.max()))
        if train_actor:
            action=m.policy(b['z'],b['proprio'],b['reference'])
            loss=pose_likelihood_loss(action,b['executed_action'],b['valid_mask'],weights)
            optimize(loss,self.optimizers['actor'],m.actor.parameters())
            m.trained_steps+=1;result['actor_loss']=float(loss.detach())
        return result
