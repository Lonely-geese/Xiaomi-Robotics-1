"""Synchronous data parallelism for the two independent LWD optimizers."""

import random

import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


@torch.no_grad()
def broadcast_model(model):
    if world_size() > 1:
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def synchronize_gradients(parameters, bucket_numel=8 * 1024 * 1024):
    """Average gradients before clipping/stepping, including rank-local unused parameters.

    FP32 communication avoids BF16 summation overflow. Large parameters are
    split as well, so even the vocabulary embedding uses bounded workspace.
    All ranks must call this for the same parameter sequence.
    """
    size = world_size()
    parameters = [p for p in parameters if p.requires_grad]
    if size == 1 or not parameters:
        return
    presence = torch.tensor([p.grad is not None for p in parameters],
                            device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(presence)
    bucket, count = [], 0

    def flush():
        flat = torch.cat([part.float() for part in bucket])
        dist.all_reduce(flat)
        flat.div_(size)
        offset = 0
        for part in bucket:
            part.copy_(flat[offset:offset + part.numel()])
            offset += part.numel()

    for parameter, used in zip(parameters, presence.tolist(), strict=True):
        if not used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()
        for part in parameter.grad.view(-1).split(bucket_numel):
            if bucket and count + part.numel() > bucket_numel:
                flush()
                bucket, count = [], 0
            bucket.append(part)
            count += part.numel()
    if bucket:
        flush()


def mean_metrics(metrics, device):
    if world_size() == 1:
        return metrics
    names = sorted(metrics)
    values = torch.tensor([metrics[key] for key in names], device=device, dtype=torch.float64)
    dist.all_reduce(values)
    return dict(zip(names, (values / world_size()).tolist(), strict=True))


def gather_rng():
    local = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    states = [None] * world_size()
    dist.all_gather_object(states, local)
    return {"world_size": world_size(), "rng_by_rank": states}


def restore_rng(checkpoint):
    saved = checkpoint.get("distributed", {})
    if saved.get("world_size", 1) != world_size():
        raise ValueError("exact LWD resume requires the same distributed world size")
    if world_size() > 1:
        local = saved["rng_by_rank"][rank()]
        torch.set_rng_state(local["torch"].cpu())
        random.setstate(local["python"])
        if local["cuda"] is not None:
            torch.cuda.set_rng_state(local["cuda"].cpu())
        return True
    return False
