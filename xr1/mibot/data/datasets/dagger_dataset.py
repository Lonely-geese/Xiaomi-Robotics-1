"""Two expert-only pools; reuse XR1 RGB, relative-action and padding code."""
from copy import deepcopy
import json
import math

import numpy as np
import torch
import torch.distributed as dist
from mmengine import DATASETS
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from mibot.data.datamodule.base_datamodule import BaseDataModule
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.data.datasets.recap_dataset import ResumablePoolLoader
from mibot.data.recap import BalancedPoolSampler


class ExpertSamples:
    def __init__(self, files, horizon):
        self.files = files
        counts = []
        for path in files:
            with open(path) as handle:
                traj = json.load(handle)
            meta = traj['dagger']
            if meta['pool'] not in ('demo', 'correction') or not meta['expert_only']:
                raise ValueError('HG-DAgger accepts expert actions only')
            if meta['horizon'] != horizon:
                raise ValueError('Action horizon changed; rebuild the HG-DAgger dataset')
            count = int(meta['sample_frames'])
            if not 0 < count <= int(traj['num_frames']):
                raise ValueError('Invalid expert chunk-start count')
            counts.append(count)
        self.ends = np.cumsum(counts)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.ends, index, side='right'))
        offset = int(self.ends[file_index - 1]) if file_index else 0
        return dict(file=self.files[file_index], frame_index=int(index - offset))


class DaggerJsonDataset(JsonDataset):
    def _samples(self):
        if self.repeat_to_max_steps or self.sample_stride != 1:
            raise ValueError('HG-DAgger uses balanced frame sampling')
        return ExpertSamples(self.files, self.action_length)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        meta = self._read_json(self.samples[index]['file'])['dagger']
        result['dagger_pool'] = torch.tensor(0 if meta['pool'] == 'demo' else 1)
        return result


class DaggerValidationDataset(DaggerJsonDataset):
    def __getitem__(self, index):
        # -1 is an explicitly zero-weight placeholder for an otherwise empty
        # final distributed batch. All ranks execute the same number of forwards.
        result = super().__getitem__(max(index, 0))
        result['dagger_eval_weight'] = torch.tensor(0 if index == -1 else 1)
        return result


class ExactValidationBatchSampler(Sampler):
    """Visit each validation frame once, with no dropped or counted duplicates."""
    def __init__(self, size, batch_size, world=1, rank=0):
        if size <= 0 or batch_size <= 0 or world <= 0 or not 0 <= rank < world:
            raise ValueError('Invalid validation sampler configuration')
        self.size, self.batch_size, self.world, self.rank = size, batch_size, world, rank

    def __len__(self):
        return math.ceil(self.size / (self.batch_size * self.world))

    def __iter__(self):
        for step in range(len(self)):
            start = (step * self.world + self.rank) * self.batch_size
            yield list(range(start, min(start + self.batch_size, self.size))) if start < self.size else [-1]


@DATASETS.register_module()
class DaggerDataModule(BaseDataModule):
    def setup(self, stage=None):
        if self.train_set is not None:
            return
        pools = []
        for name in ('demo_paths', 'correction_paths'):
            params = deepcopy(self.params)
            params.train_datasets.paths = list(params.train_datasets[name])
            pools.append(DaggerJsonDataset(params))
        self.demo_size, self.correction_size = map(len, pools)
        self.train_set = ConcatDataset(pools)
        self.validation_sets = []
        for name in ('validation_demo_paths', 'validation_correction_paths'):
            params = deepcopy(self.params)
            params.train_datasets.paths = list(params.train_datasets[name])
            params.train_datasets.augment = False
            self.validation_sets.append(DaggerValidationDataset(params))

    def train_dataloader(self):
        self.setup('fit')
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = BalancedPoolSampler(self.demo_size, self.correction_size,
            self.batch_size, world, rank, seed=int(self.params.get('seed', 42)))
        workers = int(self.params.train_datasets.get('num_workers', 4))
        return ResumablePoolLoader(self.train_set, batch_size=self.batch_size, sampler=sampler,
            num_workers=workers, collate_fn=self.collate_fn, persistent_workers=workers > 0,
            prefetch_factor=2 if workers else None, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        self.setup('fit')
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        cfg = self.params.train_datasets
        workers = int(cfg.get('validation_workers', 2))
        return [DataLoader(dataset, batch_sampler=ExactValidationBatchSampler(
                    len(dataset), int(cfg.get('validation_batch_size', 8)), world, rank),
                num_workers=workers, collate_fn=self.collate_fn, persistent_workers=workers > 0,
                prefetch_factor=2 if workers else None, pin_memory=True,
                generator=torch.Generator().manual_seed(271828 + 100 * pool + rank))
                for pool, dataset in enumerate(self.validation_sets)]
