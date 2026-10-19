# Copyright (C) 2026 Xiaomi Corporation.
import json
import logging
import os
import socket
import time
from collections import deque
from pathlib import Path
import warnings
import torch
import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from mmengine import DATASETS
from omegaconf import DictConfig, OmegaConf
from mibot.models import MIMODEL
from mibot.utils.cfg_utils import helper
import mibot.data

warnings.filterwarnings('ignore')


class TrainingProgress(Callback):
    """Record observed update speed and CUDA memory without synchronizing GPUs."""

    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        self.last_time = time.perf_counter()
        self.durations = deque(maxlen=50)

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero or trainer.global_step == self.last_step:
            return
        now = time.perf_counter()
        step = trainer.global_step
        duration = (now - self.last_time) / (step - self.last_step)
        self.last_step, self.last_time = step, now
        if step > 2:
            self.durations.append(duration)
        if step != 1 and step % 10:
            return
        seconds = sum(self.durations) / len(self.durations) if self.durations else None
        record = dict(hostname=socket.gethostname(), global_step=step,
            max_steps=trainer.max_steps, epoch=trainer.current_epoch,
            world_size=trainer.world_size, per_gpu_batch=trainer.datamodule.batch_size,
            gradient_accumulation_steps=trainer.accumulate_grad_batches,
            global_batch_size=trainer.world_size*trainer.datamodule.batch_size*trainer.accumulate_grad_batches,
            updated_unix=time.time(), seconds_per_update=seconds,
            measured_updates=len(self.durations),
            estimated_training_seconds_remaining=(trainer.max_steps-step)*seconds if seconds else None)
        if pl_module.device.type == 'cuda':
            record.update(peak_allocated_gib=torch.cuda.max_memory_allocated(pl_module.device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(pl_module.device)/2**30)
        target = Path(trainer.default_root_dir, 'training_progress.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(target)
        print('TRAIN_PROGRESS ' + json.dumps(record), flush=True)


class RunRecord(Callback):
    def on_fit_start(self, trainer, pl_module):
        params = getattr(trainer.datamodule, 'params', {})
        expected_batch = params.get('global_batch_size')
        if expected_batch:
            actual_batch = trainer.world_size * trainer.datamodule.batch_size * trainer.accumulate_grad_batches
            if actual_batch != expected_batch:
                raise ValueError(f'Expected global batch {expected_batch}, got {actual_batch}')
        if not trainer.is_global_zero:
            return
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        for logger in trainer.loggers:
            if isinstance(logger, SwanLabLogger):
                run = logger.experiment
                Path(trainer.default_root_dir, 'swanlab_run.json').write_text(
                    json.dumps({'id': run.id, 'url': run.url}, indent=2))


def prepare(cfg):
    serializable = OmegaConf.to_container(cfg, resolve=True)
    cfg = helper(cfg)
    datamodule = DATASETS.build(cfg.data)
    model = MIMODEL.build(cfg.model)
    interval = cfg.trainer.pop('save_interval', 10000)
    checkpoint_mode = cfg.pop('checkpoint_mode', 'steps')
    callbacks = [ModelSummary(max_depth=2), TrainingProgress()]
    if cfg.trainer.get('enable_checkpointing', True):
        checkpoint_args = dict(dirpath=cfg.trainer.default_root_dir, save_last=True,
            enable_version_counter=False)
        if checkpoint_mode == 'epochs':
            checkpoint_args.update(every_n_epochs=1, save_top_k=2, monitor='val/loss', mode='min',
                save_on_train_epoch_end=False, filename='epoch{epoch:02d}-step{step}', auto_insert_metric_name=False)
        else:
            checkpoint_args.update(every_n_train_steps=interval, save_top_k=-1)
        callbacks.append(ModelCheckpoint(**checkpoint_args))
    project = cfg.trainer.pop('project')
    name = cfg.trainer.pop('exp_name')
    options = cfg.pop('logging', {'type': 'wandb'})
    backend = options.pop('type')
    if backend == 'swanlab':
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        logger = [SwanLabLogger(project=project, experiment_name=name,
            log_dir=os.path.join(cfg.trainer.default_root_dir, 'swanlab'),
            save_dir=cfg.trainer.default_root_dir, config=serializable, **options),
            CSVLogger(cfg.trainer.default_root_dir, name='csv')]
        callbacks.append(RunRecord())
    else:
        logger = [WandbLogger(project=project, name=name, config=serializable)]
    cfg.trainer['callbacks'] = callbacks
    logging.getLogger('lightning.pytorch').setLevel(logging.INFO)
    return cfg, datamodule, model, logger


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path = cfg.trainer.pop('ckpt_path', None)
    test_after_fit = cfg.get('evaluation', {}).get('test_after_fit', False)
    trainer = Trainer(**cfg.trainer, logger=logger)
    succeeded = False
    try:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        fit_steps, fit_epochs = trainer.global_step, trainer.current_epoch
        if test_after_fit:
            results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
            if trainer.is_global_zero:
                Path(trainer.default_root_dir, 'test_metrics.json').write_text(json.dumps(results, indent=2))
        if trainer.is_global_zero:
            Path(trainer.default_root_dir, 'completed.json').write_text(json.dumps(
                {'global_step': fit_steps, 'epochs': fit_epochs, 'test_completed': test_after_fit}, indent=2))
        succeeded = True
    finally:
        if trainer.is_global_zero:
            import swanlab
            if swanlab.get_run() is not None:
                swanlab.finish(state='success' if succeeded else 'crashed',
                    error=None if succeeded else 'Training or evaluation failed; see local training log.')


if __name__ == '__main__':
    main()
