#!/usr/bin/env python3
"""Fit a multimodal categorical return model, then write a checkpoint."""
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
from mibot.models.recap_value import RecapValueCritic
from prepare_recap_dataset import read_jsonl
from recap_recovery import atomic_torch_save


class CriticFrames(Dataset):
    def __init__(self, records, scale):
        self.records, self.scale = records, scale
        self.ends = np.cumsum([r['length'] for r in records])
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def episode(self, index):
        if index not in self.cache:
            record = self.records[index]
            with np.load(record['numeric'], allow_pickle=False) as data:
                numeric = {key:data[key] for key in ('state','returns')}
            self.cache[index] = (np.load(record['images'], mmap_mode='r'), numeric)
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side='right'))
        frame = index - (int(self.ends[episode-1]) if episode else 0)
        images, data = self.episode(episode)
        return (torch.from_numpy(images[frame].copy()), torch.from_numpy(data['state'][frame].copy()),
            torch.tensor(data['returns'][frame]/self.scale, dtype=torch.float32))


def is_validation(record):
    # Episode-disjoint critic diagnostics only. These episodes will still be
    # consumed by policy training and do not form an independent policy test.
    key = f"{record['dataset']}:{record['episode_index']}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0


def main(args):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('nccl')
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    metadata = json.loads((args.data/'prepared.json').read_text())
    scale = metadata['return_scale']
    records = [r for r in read_jsonl(args.data/'episodes.jsonl') if r['labeled']]
    training = sorted([r for r in records if not is_validation(r)], key=lambda r:r['pool'] != 'sft')
    validation = [r for r in records if is_validation(r)]
    sft_size = sum(r['length'] for r in training if r['pool'] == 'sft')
    rollout_size = sum(r['length'] for r in training if r['pool'] == 'rollout')
    sampler = BalancedPoolSampler(sft_size, rollout_size, args.batch_size, world, rank)
    data = CriticFrames(training, scale)
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers else None)
    valdata = CriticFrames(validation, scale)
    # No padded/duplicated examples in distributed validation.
    valloader = DataLoader(valdata, batch_size=args.batch_size,
        sampler=range(rank, len(valdata), world), num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    model = RecapValueCritic().to(device)
    total, squares, count = np.zeros(30), np.zeros(30), 0
    for r in training:
        with np.load(r['numeric'], allow_pickle=False) as numeric:
            state = numeric['state'].astype(np.float64)
        total += state.sum(0)
        squares += (state**2).sum(0)
        count += len(state)
    mean = total/count
    model.state_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=device))
    model.state_std.copy_(torch.tensor(np.sqrt(np.maximum(squares/count-mean**2, 1e-6)), dtype=torch.float32, device=device))
    network = DDP(model, device_ids=[device.index]) if distributed else model
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    args.output.mkdir(parents=True, exist_ok=True)
    best, steps, start_epoch, start_batch = float('inf'), 0, 0, 0
    restored_sum = None
    resume_path = args.output/'resume.pt'
    fallback = args.output/'resume.previous.pt'
    if args.resume and (resume_path.exists() or fallback.exists()):
        try:
            selected = resume_path if resume_path.exists() else fallback
            saved = torch.load(selected, map_location='cpu', weights_only=False)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            saved = torch.load(fallback, map_location='cpu', weights_only=False)
            if rank == 0 and resume_path.exists():
                resume_path.rename(resume_path.with_name(f'resume.corrupt.{time.time_ns()}.pt'))
            if distributed:
                dist.barrier()
        if saved['metadata']['source_metadata_sha256'] != metadata['source_metadata_sha256']:
            raise ValueError('Critic recovery checkpoint belongs to different data')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        best, steps = saved['best'], saved['steps']
        start_epoch, start_batch = saved['next_epoch'], saved['next_batch']
        local = saved['rank_states'][rank]
        torch.set_rng_state(local['torch_rng'])
        torch.cuda.set_rng_state(local['cuda_rng'], device)
        np.random.set_state(local['numpy_rng'])
        restored_sum = local['train_sum']
        if rank == 0:
            print(json.dumps(dict(resumed_from=str(resume_path), epoch=start_epoch, batch=start_batch, step=steps)), flush=True)

    def save_recovery(next_epoch, next_batch, train_sum):
        local = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
            numpy_rng=np.random.get_state(), train_sum=train_sum.detach().cpu())
        states = [None]*world
        if distributed:
            dist.all_gather_object(states, local)
        else:
            states[0] = local
        if rank == 0:
            atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                metadata=metadata, best=best, steps=steps, next_epoch=next_epoch,
                next_batch=next_batch, rank_states=states), resume_path)
        if distributed:
            dist.barrier()
    if rank == 0:
        print(json.dumps(dict(stage='value_train', sft_frames=sft_size, rollout_frames=rollout_size,
            validation_frames=len(valdata), steps_per_epoch=sampler.steps, epochs=args.epochs)), flush=True)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        offset = start_batch if epoch == start_epoch else 0
        sampler.start_batch = offset
        network.train()
        started = time.monotonic()
        train_sum = torch.zeros(2, device=device)
        if restored_sum is not None and epoch == start_epoch and offset:
            train_sum.copy_(restored_sum.to(device))
        batch_index = offset-1
        for batch_index, (images, state, targets) in enumerate(loader, start=offset):
            images, state, targets = [t.to(device, non_blocking=True) for t in (images,state,targets)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = network(images, state)
                loss = model.loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite value loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            train_sum += torch.stack([loss.detach()*len(targets), loss.new_tensor(len(targets))])
            steps += 1
            if rank == 0 and batch_index % 100 == 0:
                progress = dict(stage='value_train', epoch=epoch+1, step=steps,
                    batch_in_epoch=batch_index+1, steps_per_epoch=sampler.steps,
                    loss=float(loss.detach()), elapsed_s=time.monotonic()-started)
                print(json.dumps(progress), flush=True)
                (args.output/'progress.json').write_text(json.dumps(progress, indent=2))
            if steps % args.checkpoint_every == 0:
                save_recovery(epoch, batch_index+1, train_sum)
            if args.max_steps and steps >= args.max_steps:
                break
        # Save before validation, so an interruption during validation does not
        # repeat the training epoch. Preserve this rank's unreduced loss sum.
        next_batch = batch_index+1
        save_recovery(epoch, next_batch, train_sum)
        network.eval()
        errors = torch.zeros(3, device=device)
        with torch.no_grad():
            for images, state, targets in valloader:
                images, state, targets = [t.to(device, non_blocking=True) for t in (images,state,targets)]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images, state)
                    prediction = (logits.float().softmax(-1)*model.support).sum(-1)
                    loss = model.loss(logits, targets)
                errors += torch.stack([loss*len(targets), ((prediction-targets)**2).sum(), loss.new_tensor(len(targets))])
        if distributed:
            dist.all_reduce(errors)
            dist.all_reduce(train_sum)
        metrics = dict(epoch=epoch+1, steps=steps, train_cross_entropy=float(train_sum[0]/train_sum[1]),
            val_cross_entropy=float(errors[0]/errors[2]), val_rmse=float((errors[1]/errors[2]).sqrt()),
            diagnostics='episode-disjoint critic validation, not independent policy evaluation')
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with (args.output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics)+'\n')
            checkpoint = dict(model=model.state_dict(), metadata=metadata, metrics=metrics,
                training_episodes=[r['name'] for r in training], validation_episodes=[r['name'] for r in validation])
            atomic_torch_save(checkpoint, args.output/'last.pt')
            if metrics['val_rmse'] < best:
                best = metrics['val_rmse']
                atomic_torch_save(checkpoint, args.output/'best.pt')
        if distributed:
            dist.barrier()
        if args.max_steps and steps >= args.max_steps:
            break
        save_recovery(epoch+1, 0, torch.zeros_like(train_sum))
    if rank == 0:
        (args.output/'completed.json').write_text(json.dumps(dict(epochs=args.epochs if not args.max_steps else epoch+1,
            steps=steps, best_val_rmse=best), indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=base/'data/recap_20261020')
    parser.add_argument('--output', type=Path, default=base/'outputs/recap_value_20261020')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=500)
    main(parser.parse_args())
