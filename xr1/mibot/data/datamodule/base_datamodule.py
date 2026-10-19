# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy
import torch.distributed as dist
from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler
from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


class CompleteBatchSampler(DistributedSampler):
    """Shuffle each epoch, then truncate to complete global optimizer batches."""
    def __init__(self, dataset, global_batch_size, **kwargs):
        super().__init__(dataset, drop_last=True, **kwargs)
        if global_batch_size % self.num_replicas:
            raise ValueError('global batch must be divisible by world size')
        self.total_size = len(dataset) // global_batch_size * global_batch_size
        self.num_samples = self.total_size // self.num_replicas
        if self.total_size == 0:
            raise ValueError('dataset smaller than a global batch')


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params = params
        self.batch_size = int(params.train_datasets.get('batch_size', 16))
        self.collate_fn = CustomCollate()
        self.train_set = self.val_set = self.test_set = None

    def _dataset(self, key):
        params = deepcopy(self.params)
        params['train_datasets'] = deepcopy(params[key])
        return JsonDataset(params)

    def setup(self, stage=None):
        if stage in (None, 'fit') and self.train_set is None:
            self.train_set = self._dataset('train_datasets')
        if stage in (None, 'fit', 'validate') and self.val_set is None and 'val_datasets' in self.params:
            self.val_set = self._dataset('val_datasets')
        if stage in (None, 'test') and self.test_set is None and 'test_datasets' in self.params:
            self.test_set = self._dataset('test_datasets')

    def _loader(self, dataset, key, train=False):
        if dataset is None:
            return []
        cfg = self.params[key]
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler_args = dict(num_replicas=world, rank=rank, shuffle=train, seed=42)
        if train and self.params.get('global_batch_size'):
            sampler = CompleteBatchSampler(dataset, self.params.global_batch_size, **sampler_args)
        else:
            sampler = DistributedSampler(dataset, **sampler_args)
        workers = int(cfg.get('num_workers', 4))
        return DataLoader(dataset, batch_size=int(cfg.get('batch_size', self.batch_size)),
            sampler=sampler, num_workers=workers, prefetch_factor=2 if workers else None,
            collate_fn=self.collate_fn, persistent_workers=workers > 0, pin_memory=True,
            drop_last=train and bool(self.params.get('global_batch_size')))

    def train_dataloader(self):
        self.setup('fit')
        return self._loader(self.train_set, 'train_datasets', True)

    def val_dataloader(self):
        return self._loader(self.val_set, 'val_datasets')

    def test_dataloader(self):
        return self._loader(self.test_set, 'test_datasets')
