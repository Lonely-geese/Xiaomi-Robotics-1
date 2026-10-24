"""HG-DAgger validation with separate, equally weighted expert pools."""
import json
from pathlib import Path

import torch
import torch.distributed as dist

from mibot.models import MIMODEL
from mibot.models.runner.base_runner import BaseRunner


LOSS_NAMES = ('loss', 'loss_mse', 'loss_freq', 'loss_l1', 'loss_score')
POOL_NAMES = ('demo', 'correction')


def validation_metrics(totals):
    """Sums are batch losses weighted by real frames, followed by frame count."""
    if totals.shape != (2, len(LOSS_NAMES) + 1) or not torch.isfinite(totals).all():
        raise ValueError('Invalid validation loss totals')
    if not (totals[:, -1] > 0).all():
        raise ValueError('Both validation pools must be nonempty')
    means = totals[:, :-1] / totals[:, -1:]
    metrics = {f'val/{pool}_{name}': means[p, i]
               for p, pool in enumerate(POOL_NAMES) for i, name in enumerate(LOSS_NAMES)}
    metrics.update({f'val/{name}': means[:, i].mean() for i, name in enumerate(LOSS_NAMES)})
    return metrics


@MIMODEL.register_module()
class DaggerRunner(BaseRunner):
    def on_validation_epoch_start(self):
        self._validation_totals = torch.zeros(2, len(LOSS_NAMES) + 1,
                                               dtype=torch.float64, device=self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pool = batch.pop('dagger_pool')
        weights = batch.pop('dagger_eval_weight')
        if not torch.all(pool == dataloader_idx):
            raise ValueError('Validation pool mismatch')
        if not (torch.all(weights == 1) or (len(weights) == 1 and weights.item() == 0)):
            raise ValueError('Dummy validation frames must occupy their own batch')
        count = int(weights.sum())
        devices = [batch['action'].device.index] if batch['action'].is_cuda else []
        # Fixed noise and diffusion times for each rank/batch/pool on every
        # validation pass; training RNG state is restored by fork_rng.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345 + 100000 * dataloader_idx + 1000 * self.global_rank + batch_idx)
            losses = self.model(batch, return_loss=True)
        values = torch.stack([losses[name].detach().double() for name in LOSS_NAMES])
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite HG-DAgger validation loss')
        self._validation_totals[dataloader_idx, :-1] += values * count
        self._validation_totals[dataloader_idx, -1] += count
        return losses['loss']

    def on_validation_epoch_end(self):
        totals = self._validation_totals
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        metrics = validation_metrics(totals)
        for name, value in metrics.items():
            self.log(name, value.float(), on_step=False, on_epoch=True,
                     sync_dist=False, add_dataloader_idx=False, prog_bar=name == 'val/loss')
        if self.trainer.sanity_checking:
            return
        expected = [len(ds) for ds in self.trainer.datamodule.validation_sets]
        if totals[:, -1].long().tolist() != expected:
            raise ValueError(f'Validation did not cover both complete pools: {totals[:, -1]} != {expected}')
        if self.trainer.is_global_zero:
            report = dict(global_step=self.global_step, epoch=self.current_epoch,
                frames=dict(zip(POOL_NAMES, expected)),
                **{name:float(value) for name, value in metrics.items()})
            root = Path(self.trainer.default_root_dir)
            root.mkdir(parents=True, exist_ok=True)
            with (root/'validation_metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(report) + '\n')
            (root/'validation_latest.json').write_text(json.dumps(report, indent=2))
            print('DAGGER_VALIDATION ' + json.dumps(report), flush=True)
