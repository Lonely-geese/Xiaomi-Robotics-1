#!/usr/bin/env python3
"""XR1 advantage-conditioned offline policy fine-tuning."""
import json
import os
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from mmengine import Config, DATASETS
from omegaconf import DictConfig, OmegaConf

import mibot.data
from mibot.models import MIMODEL
from mibot.utils.cfg_utils import strategy_helper
from train import TrainingProgress


class RestartableCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        trainer.strategy.barrier('checkpoint_files_written')
        if trainer.is_global_zero:
            Path(filepath, 'restart_ready.json').write_text(json.dumps(dict(
                global_step=trainer.global_step, epoch=trainer.current_epoch,
                world_size=trainer.world_size), indent=2))
        trainer.strategy.barrier('checkpoint_ready')


class RecapRunRecord(Callback):
    def __init__(self, resume=None):
        self.resume = resume

    def on_train_start(self, trainer, pl_module):
        data = trainer.datamodule
        actual = trainer.world_size*data.batch_size*trainer.accumulate_grad_batches
        if actual != 256 or data.batch_size != 32 or trainer.world_size != 8:
            raise ValueError(f'Required 8 GPUs x 32 = 256; actual global batch {actual}')
        if (trainer.global_step != 0 and not self.resume) or not pl_module.model.recap_enabled:
            raise ValueError('Expected a fresh RECAP optimizer and enabled condition')
        if any(not p.requires_grad for p in pl_module.parameters()):
            raise ValueError('RECAP policy run requires full parameter fine-tuning')
        if trainer.is_global_zero:
            root = Path(os.environ['RECAP_DATA_DIR'])
            manifest = json.loads((root/'prepared.json').read_text())
            if data.sft_size != manifest['sft_frames'] or data.rollout_size != manifest['rollout_frames']:
                raise ValueError('Pool frame counts disagree with preparation manifest')
            record = dict(initial_model=pl_module._pretrained,
                optimizer='restored with scheduler from checkpoint' if self.resume else 'fresh, no previous optimizer/scheduler state',
                resumed_from=self.resume, starting_step=trainer.global_step,
                world_size=trainer.world_size, batch_per_gpu=32, global_batch=256,
                local_pool_counts=dict(sft=16, rollout=16),
                sft_frames=data.sft_size, rollout_frames=data.rollout_size,
                epoch_definition=manifest['epoch_definition'],
                steps_per_epoch=trainer.num_training_batches,
                epochs=trainer.max_epochs, total_steps=trainer.max_steps,
                normalization_sha256=manifest['normalization_sha256'],
                independent_policy_evaluation=False,
                trainable_parameters=sum(p.numel() for p in pl_module.parameters()),
                advantages=json.loads((root/'advantages_ready.json').read_text()))
            target = Path(trainer.default_root_dir, 'run_record.json')
            if self.resume:
                with Path(trainer.default_root_dir, 'resume_history.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
            else:
                target.write_text(json.dumps(record, indent=2))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.train_dataloader.consumed_batches = batch_idx+1

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch['action'].shape[0] != 32 or int((batch['recap_pool'] == 0).sum()) != 16:
            raise ValueError('Each local batch must contain exactly 16 demos and 16 rollout samples')

    def on_train_end(self, trainer, pl_module):
        norm = float(pl_module.model.recap_condition_embedding.detach().float().norm())
        if not 0 < norm < float('inf'):
            raise ValueError('The new RECAP condition did not learn finite, nonzero weights')
        if trainer.is_global_zero:
            Path(trainer.default_root_dir, 'condition_verified.json').write_text(json.dumps(
                dict(condition_norm=norm, global_step=trainer.global_step), indent=2))


@hydra.main(config_path='../configs', config_name='recap', version_base=None)
def main(config: DictConfig):
    cfg = Config(OmegaConf.to_container(config, resolve=True))
    resume = cfg.trainer.ckpt_path
    if resume and not Path(resume, 'restart_ready.json').exists():
        raise ValueError('Recovery requires a completely saved RECAP checkpoint')
    root = Path(cfg.trainer.default_root_dir)/f'project_{cfg.trainer.project}'/cfg.trainer.exp_name
    cfg.trainer.default_root_dir = str(root)
    if int(os.environ.get('RANK', 0)) == 0:
        root.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(root/'config.yaml'))
        cfg.dump(str(root/'config.py'))
    seed_everything(cfg.trainer.pop('seed', 42)+int(os.environ.get('RANK', 0)), workers=True)
    strategy_helper(cfg)
    cfg.model.params.optimizer = cfg.trainer.pop('optimizer')
    cfg.model.params.scheduler = cfg.trainer.pop('scheduler')
    cfg.trainer.pop('project')
    cfg.trainer.pop('exp_name')
    cfg.trainer.pop('ckpt_path')
    data = DATASETS.build(cfg.data)
    model = MIMODEL.build(cfg.model)
    callbacks = [TrainingProgress(), RecapRunRecord(resume)]
    if cfg.trainer.get('enable_checkpointing', True):
        callbacks.append(RestartableCheckpoint(dirpath=str(root), every_n_epochs=1,
            every_n_train_steps=0, save_top_k=-1, save_last='link',
            save_on_train_epoch_end=True, filename='epoch{epoch:02d}-step{step}',
            auto_insert_metric_name=False, enable_version_counter=False))
        callbacks.append(RestartableCheckpoint(dirpath=str(root/'recovery'),
            every_n_train_steps=1000, every_n_epochs=0, save_top_k=2,
            monitor='step', mode='max', save_last=False, save_on_train_epoch_end=False,
            filename='step{step}', auto_insert_metric_name=False, enable_version_counter=False))
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=CSVLogger(str(root), name='csv'))
    trainer.fit(model=model, datamodule=data, ckpt_path=resume)
    if trainer.is_global_zero:
        (root/'completed.json').write_text(json.dumps(dict(global_step=trainer.global_step,
            epochs=trainer.current_epoch, independent_policy_evaluation=False), indent=2))


if __name__ == '__main__':
    main()
