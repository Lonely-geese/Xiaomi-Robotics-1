"""Two RECAP pools with compact frame indexes and action-aligned labels."""
from copy import deepcopy
from functools import lru_cache
import json

import numpy as np
import torch
import torch.distributed as dist
from mmengine import DATASETS
from torch.utils.data import ConcatDataset, DataLoader

from mibot.data.datamodule.base_datamodule import BaseDataModule
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.data.recap import BalancedPoolSampler


class FrameSamples:
    def __init__(self, files):
        self.files = files
        self.ends = np.cumsum([int(json.load(open(path))['num_frames']) for path in files])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.ends, index, side='right'))
        offset = int(self.ends[file_index-1]) if file_index else 0
        return dict(file=self.files[file_index], frame_index=int(index-offset))


class RecapJsonDataset(JsonDataset):
    def _samples(self):
        if self.repeat_to_max_steps or self.sample_stride != 1:
            raise ValueError('RECAP uses frame sampling controlled by BalancedPoolSampler')
        return FrameSamples(self.files)

    @staticmethod
    @lru_cache(maxsize=64)
    def _sidecar(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def _action_steps(self, traj, frame):
        steps = super()._action_steps(traj, frame)
        if traj['recap']['pool'] == 'rollout':
            offset = int(traj['recap']['offset']) + frame
            human = self._sidecar(traj['recap']['sidecar'])['human']
            # A model action chunk must never contain a human takeover target.
            changes = np.flatnonzero(human[offset:offset+steps] != human[offset])
            if len(changes):
                steps = int(changes[0])
        return steps

    def __getitem__(self, index):
        sample = self.samples[index]
        traj = self._read_json(sample['file'])
        result = super().__getitem__(index)
        meta = traj['recap']
        offset = int(meta['offset']) + sample['frame_index']
        sidecar = self._sidecar(meta['sidecar'])
        result['recap_positive'] = torch.tensor(bool(sidecar['positive'][offset]))
        result['recap_pool'] = torch.tensor(0 if meta['pool'] == 'sft' else 1)
        return result


@DATASETS.register_module()
class RecapDataModule(BaseDataModule):
    def setup(self, stage=None):
        if self.train_set is not None:
            return
        pools = []
        for name in ('sft_paths', 'rollout_paths'):
            params = deepcopy(self.params)
            params.train_datasets.paths = list(params.train_datasets[name])
            pools.append(RecapJsonDataset(params))
        self.sft_size, self.rollout_size = map(len, pools)
        self.train_set = ConcatDataset(pools)

    def train_dataloader(self):
        self.setup('fit')
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        cfg = self.params.train_datasets
        sampler = BalancedPoolSampler(self.sft_size, self.rollout_size,
            self.batch_size, world, rank, seed=int(self.params.get('seed', 42)))
        workers = int(cfg.get('num_workers', 4))
        return DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler,
            num_workers=workers, collate_fn=self.collate_fn, persistent_workers=workers > 0,
            prefetch_factor=2 if workers else None, pin_memory=True, drop_last=True)
