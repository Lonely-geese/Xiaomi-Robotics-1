#!/usr/bin/env python3
"""XR1 HG-DAgger: supervised flow matching on demonstrations + human corrections."""
import json
import math
import os
from pathlib import Path
import shutil

import hydra
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from mmengine import Config, DATASETS
from omegaconf import DictConfig, OmegaConf

import mibot.data.datasets.dagger_dataset  # register without changing live RECAP imports
import mibot.models.runner.dagger_runner
from mibot.models import MIMODEL
from mibot.utils.cfg_utils import strategy_helper
from train import TrainingProgress


class DaggerCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        trainer.strategy.barrier('dagger_checkpoint_files_written')
        if trainer.is_global_zero:
            record = json.loads(Path(trainer.default_root_dir, 'run_record.json').read_text())
            Path(filepath, 'restart_ready.json').write_text(json.dumps(dict(
                global_step=trainer.global_step, epoch=trainer.current_epoch,
                world_size=trainer.world_size, algorithm='HG-DAgger',
                experiment_fingerprint=record['experiment_fingerprint']), indent=2))
        trainer.strategy.barrier('dagger_checkpoint_ready')


class DaggerRunRecord(Callback):
    def __init__(self, data_root, manifest, resume=None):
        self.data_root, self.manifest, self.resume = Path(data_root), manifest, resume

    def on_train_start(self, trainer, pl_module):
        data = trainer.datamodule
        if (trainer.world_size, data.batch_size, trainer.accumulate_grad_batches) != (8, 32, 1):
            raise ValueError('HG-DAgger requires 8 GPUs x 32 = global batch 256')
        if pl_module.model.recap_enabled:
            raise ValueError('HG-DAgger must disable RECAP conditioning')
        if any(not p.requires_grad for p in pl_module.parameters()):
            raise ValueError('This HG-DAgger configuration requires full fine-tuning')
        if (data.demo_size, data.correction_size) != (
                self.manifest['counts']['train_demo_frames'], self.manifest['counts']['train_correction_frames']):
            raise ValueError('Expert pool sizes disagree with manifest')
        if trainer.is_global_zero:
            record = dict(algorithm='HG-DAgger', initial_model=pl_module._pretrained,
                resumed_from=self.resume, starting_step=trainer.global_step,
                experiment_fingerprint=self.manifest['experiment_fingerprint'],
                normalization_sha256=self.manifest['normalization_sha256'],
                world_size=8, batch_per_gpu=32, global_batch=256,
                local_pool_counts=dict(demo=16, correction=16),
                demo_frames=data.demo_size, correction_frames=data.correction_size,
                validation_frames=dict(demo=len(data.validation_sets[0]), correction=len(data.validation_sets[1])),
                validation='full held-out pools every half epoch; fixed noise, no augmentation; equal pool mean',
                steps_per_epoch=trainer.num_training_batches, epochs=trainer.max_epochs,
                total_steps=trainer.max_steps, critic_required=False, recap_enabled=False,
                trainable_parameters=sum(p.numel() for p in pl_module.parameters()),
                loss='XR1 supervised flow matching and frequency loss on expert actions',
                optimizer='restored' if self.resume else 'fresh')
            root = Path(trainer.default_root_dir)
            if self.resume:
                with (root/'resume_history.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
            else:
                (root/'run_record.json').write_text(json.dumps(record, indent=2))
            shutil.copyfile(self.data_root/'normalize.json', root/'normalize.json')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pools = batch.pop('dagger_pool')
        if len(pools) != 32 or int((pools == 0).sum()) != 16 or int((pools == 1).sum()) != 16:
            raise ValueError('Each GPU batch must contain exactly 16 demos and 16 human corrections')
        if 'recap_positive' in batch or 'recap_pool' in batch:
            raise ValueError('Unexpected RECAP supervision in HG-DAgger batch')

    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        if loader.sampler.epoch != trainer.current_epoch:
            loader.consumed_batches = 0
        loader.sampler.set_epoch(trainer.current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.train_dataloader.consumed_batches = batch_idx + 1
        loss = trainer.callback_metrics.get('train/loss_step')
        if loss is not None and not math.isfinite(float(loss)):
            raise ValueError('Non-finite HG-DAgger training loss')
        if trainer.is_global_zero and (trainer.global_step == 1 or trainer.global_step % 10 == 0):
            record = dict(global_step=trainer.global_step, epoch=trainer.current_epoch,
                loss=float(loss) if loss is not None else None,
                lr=trainer.optimizers[0].param_groups[0]['lr'])
            Path(trainer.default_root_dir, 'training_latest.json').write_text(json.dumps(record, indent=2))
            print('DAGGER_TRAIN ' + json.dumps(record), flush=True)


def configure(config):
    """Resolve epoch length from the actual pools before optimizer creation."""
    cfg = Config(OmegaConf.to_container(config, resolve=True))
    smoke = bool(cfg.pop('smoke_test', False))
    data_root = Path(cfg.pop('dagger_data_dir')).resolve()
    manifest = json.loads((data_root/'prepared.json').read_text())
    verification = json.loads((data_root/'verification.json').read_text())
    if not verification['verified'] or verification['experiment_fingerprint'] != manifest['experiment_fingerprint']:
        raise ValueError('Run verify_dagger_dataset.py before training')
    if (cfg.trainer.devices, cfg.data.params.train_datasets.batch_size,
            cfg.trainer.accumulate_grad_batches, cfg.trainer.num_nodes) != (8, 32, 1, 1):
        raise ValueError('Required training layout: one node, 8 GPUs, 32 per GPU, no accumulation')
    if cfg.model.params.model.recap_enabled:
        raise ValueError('RECAP conditioning must be disabled')
    steps = max(manifest['counts']['train_demo_frames'], manifest['counts']['train_correction_frames']) // 128
    if steps <= 0:
        raise ValueError('Not enough data for a complete global batch')
    total = steps * cfg.trainer.max_epochs
    cfg.trainer.max_steps = total
    cfg.data.params.max_steps = total
    cfg.trainer.scheduler.params.num_training_steps = total
    cfg.trainer.scheduler.params.num_warmup_steps = min(500, max(1, round(total * .03)))
    if smoke:
        cfg.trainer.max_steps = 2
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.val_check_interval = 1.0
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.scheduler.params.num_training_steps = 2
        cfg.trainer.scheduler.params.num_warmup_steps = 1
    if not Path(cfg.model.params.pretrained).is_file():
        raise ValueError('Initial policy checkpoint does not exist')
    return cfg, data_root, manifest


@hydra.main(config_path='../configs', config_name='dagger', version_base=None)
def main(config: DictConfig):
    cfg, data_root, manifest = configure(config)
    # Every rank fails consistently if immutable dataset hashes changed. The
    # launcher audits full arrays once; rank preflight checks the small manifests.
    from mibot.data.dagger import sha256
    for name, expected in manifest['manifests'].items():
        if sha256(data_root/name) != expected:
            raise ValueError(f'Dataset manifest changed: {name}')
    if sha256(data_root/'normalize.json') != manifest['normalization_sha256']:
        raise ValueError('Normalization changed')
    resume = cfg.trainer.ckpt_path
    root = Path(cfg.trainer.default_root_dir)/f'project_{cfg.trainer.project}'/cfg.trainer.exp_name
    if resume:
        if not Path(resume, 'restart_ready.json').is_file():
            raise ValueError('Resume requires a completely saved distributed checkpoint')
        marker = json.loads(Path(resume, 'restart_ready.json').read_text())
        if (marker.get('algorithm') != 'HG-DAgger' or marker.get('world_size') != 8
                or marker.get('experiment_fingerprint') != manifest['experiment_fingerprint']):
            raise ValueError('Resume checkpoint is not HG-DAgger with the same data and world size')
        old = json.loads((root/'run_record.json').read_text())
        if old['experiment_fingerprint'] != manifest['experiment_fingerprint']:
            raise ValueError('Dataset changed: start a new aggregation round with model weights, not optimizer resume')
    elif (root/'run_record.json').exists():
        raise ValueError('Output already contains a run; resume explicitly or choose a new experiment name')
    cfg.trainer.default_root_dir = str(root)
    if int(os.environ.get('RANK', 0)) == 0:
        root.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(root/'config.yaml'))
        cfg.dump(str(root/'config.py'))
    seed_everything(cfg.trainer.pop('seed', 42)+int(os.environ.get('RANK', 0)), workers=True)
    strategy_helper(cfg)
    cfg.model.params.optimizer = cfg.trainer.pop('optimizer')
    cfg.model.params.scheduler = cfg.trainer.pop('scheduler')
    for name in ('project', 'exp_name', 'ckpt_path'):
        cfg.trainer.pop(name)
    data, model = DATASETS.build(cfg.data), MIMODEL.build(cfg.model)
    callbacks = [TrainingProgress(), DaggerRunRecord(data_root, manifest, resume)]
    if cfg.trainer.get('enable_checkpointing', True):
        callbacks.append(DaggerCheckpoint(dirpath=str(root), every_n_epochs=1,
        every_n_train_steps=0, save_top_k=-1, save_last='link', save_on_train_epoch_end=True,
        filename='epoch{epoch:02d}-step{step}', auto_insert_metric_name=False, enable_version_counter=False))
        callbacks.append(DaggerCheckpoint(dirpath=str(root/'recovery'), every_n_train_steps=1000,
        every_n_epochs=0, save_top_k=2, monitor='step', mode='max', save_last=False,
        save_on_train_epoch_end=False, filename='step{step}', auto_insert_metric_name=False,
        enable_version_counter=False))
        callbacks.append(DaggerCheckpoint(dirpath=str(root/'best'), monitor='val/loss', mode='min',
        save_top_k=1, save_last=False, save_on_train_epoch_end=False,
        filename='step{step}-val{val/loss:.5f}', auto_insert_metric_name=False, enable_version_counter=False))
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=CSVLogger(str(root), name='csv'))
    trainer.fit(model=model, datamodule=data, ckpt_path=resume)
    if trainer.is_global_zero:
        (root/'completed.json').write_text(json.dumps(dict(global_step=trainer.global_step,
            epochs=trainer.current_epoch, experiment_fingerprint=manifest['experiment_fingerprint']), indent=2))


if __name__ == '__main__':
    main()
