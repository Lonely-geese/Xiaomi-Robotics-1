"""FP32 AdamW masters sharded by parameter, with replicated low-precision models.

The learner must average and clip model gradients before step(). Each parameter
has one optimizer owner; owners broadcast the rounded model weights afterwards.
Checkpoints contain one optimizer shard per rank, including unrounded masters.
"""

import torch
import torch.distributed as dist

from .distributed import rank, world_size


@torch.no_grad()
def clip_grad_norm_fp32(parameters, max_norm):
    """Accumulate the norm in FP32 even when stored gradients are BF16."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.)
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g, dtype=torch.float32) for g in gradients]))
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
    torch._foreach_mul_(gradients, scale)
    return norm


class ShardedMasterAdamW(torch.optim.Optimizer):
    def __init__(self, parameters, lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        parameters = list(parameters)
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))
        self.parameters = parameters
        self.shard_rank, self.shard_world_size = rank(), world_size()
        loads = [0] * self.shard_world_size
        self.owners = [0] * len(parameters)
        for index in sorted(range(len(parameters)), key=lambda i: -parameters[i].numel()):
            owner = min(range(self.shard_world_size), key=lambda r: loads[r])
            self.owners[index] = owner
            loads[owner] += parameters[index].numel()
        self.local_indices = [i for i, owner in enumerate(self.owners) if owner == self.shard_rank]
        self.masters = [torch.nn.Parameter(parameters[i].detach().float().clone()) for i in self.local_indices]
        self.inner = (torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay,
                                       betas=betas, eps=eps, foreach=False)
                      if self.masters else None)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ShardedMasterAdamW does not support closures")
        if self.inner is not None:
            for key in ("lr", "weight_decay", "betas", "eps"):
                self.inner.param_groups[0][key] = self.param_groups[0][key]
            for index, master in zip(self.local_indices, self.masters, strict=True):
                gradient = self.parameters[index].grad
                master.grad = None if gradient is None else gradient.detach().float()
            self.inner.step()
            for index, master in zip(self.local_indices, self.masters, strict=True):
                self.parameters[index].copy_(master)
            self.inner.zero_grad(set_to_none=True)
        if self.shard_world_size > 1:
            # Identical order on every rank, including parameters unused this step.
            for parameter, owner in zip(self.parameters, self.owners, strict=True):
                dist.broadcast(parameter, src=owner)

    def state_dict(self):
        return {"format": "sharded-master-adamw-v1", "rank": self.shard_rank,
                "world_size": self.shard_world_size, "owners": self.owners,
                "outer": super().state_dict(),
                "inner": self.inner.state_dict() if self.inner is not None else None,
                "masters": [p.detach() for p in self.masters]}

    @torch.no_grad()
    def load_state_dict(self, saved):
        if saved.get("format") != "sharded-master-adamw-v1":
            raise ValueError("expected an FP32 master optimizer shard")
        if (saved["rank"] != self.shard_rank or saved["world_size"] != self.shard_world_size
                or saved["owners"] != self.owners):
            raise ValueError("optimizer shard rank/world size/parameter ownership differs")
        if len(saved["masters"]) != len(self.masters):
            raise ValueError("optimizer master count differs")
        super().load_state_dict(saved["outer"])
        if self.inner is not None:
            self.inner.load_state_dict(saved["inner"])
        for index, master, value in zip(self.local_indices, self.masters, saved["masters"], strict=True):
            if value.dtype != torch.float32 or value.shape != master.shape:
                raise ValueError("invalid FP32 master tensor")
            master.copy_(value)
            self.parameters[index].copy_(master)

    @torch.no_grad()
    def load_legacy_adamw(self, saved):
        """Preserve old Adam moments/step, starting masters at saved model weights.

        Previously rounded-away updates cannot be recovered. Future updates use
        FP32 masters and FP32 moments, without resetting the LR schedule.
        """
        groups = saved["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(self.parameters):
            raise ValueError("legacy AdamW parameter layout differs")
        self.param_groups[0].update({k: groups[0][k] for k in ("lr", "weight_decay", "betas", "eps")})
        if self.inner is None:
            return
        group = dict(groups[0], params=list(range(len(self.masters))), foreach=False, fused=False)
        states = {}
        for local, global_index in enumerate(self.local_indices):
            old = saved["state"].get(groups[0]["params"][global_index])
            if old is not None:
                states[local] = {key: value.float() if isinstance(value, torch.Tensor) else value
                                 for key, value in old.items()}
        self.inner.load_state_dict({"state": states, "param_groups": [group]})
