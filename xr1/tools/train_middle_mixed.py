"""Train the approved mixture with fresh optimizer state and CSV/SwanLab logging."""
import json
import logging
import os
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from mmengine import Config, DATASETS
from omegaconf import DictConfig, OmegaConf

import mibot.data
from mibot.models import MIMODEL
from mibot.utils.cfg_utils import fill_num_nodes, strategy_helper
from train import TrainingProgress


def validate_config(cfg):
    data = cfg.data.params
    train = data.train_datasets
    if cfg.trainer.ckpt_path is not None:
        raise ValueError("Use model.params.pretrained; resuming old optimizer state is prohibited.")
    if "val_datasets" in data or "test_datasets" in data:
        raise ValueError("This dataset includes the former held-out episodes; no independent evaluation remains.")
    if cfg.evaluation.test_after_fit or cfg.trainer.limit_val_batches != 0:
        raise ValueError("Validation and test execution must be disabled for this run.")
    if not cfg.model.params.model.full_finetune:
        raise ValueError("This run requires full parameter fine-tuning.")
    if train.repeat_to_max_steps or train.sample_stride != 1:
        raise ValueError("Use natural per-frame sampling without repeats or source reweighting.")
    if list(train.active_action_dims) != list(range(7)):
        raise ValueError("The checkpoint requires the seven left-arm action dimensions.")
    batch = cfg.trainer.devices * train.batch_size * cfg.trainer.accumulate_grad_batches
    if batch != data.global_batch_size:
        raise ValueError(f"Global batch mismatch: {batch} versus {data.global_batch_size}")
    if cfg.trainer.scheduler.params.max_lr != 2e-6:
        raise ValueError("The approved peak learning rate is 2e-6.")


class MixedRunRecord(Callback):
    def on_fit_start(self, trainer, pl_module):
        data = trainer.datamodule
        batch = trainer.world_size * data.batch_size * trainer.accumulate_grad_batches
        if batch != data.params.global_batch_size or trainer.global_step != 0:
            raise ValueError("Expected the approved global batch and a fresh optimizer run starting at step zero.")
        prepared = Path(data.params.train_datasets.paths[0]).parent / "prepared.json"
        manifest = json.loads(prepared.read_text())
        expected = int(manifest.get("train_frames", manifest.get("splits", {}).get("train", {}).get("frames", 0)))
        if len(data.train_set) != expected:
            raise ValueError(f"Prepared frames {expected} != dataset samples {len(data.train_set)}")
        frozen = [name for name, param in pl_module.named_parameters() if not param.requires_grad]
        if frozen:
            raise ValueError(f"Unexpected frozen parameters: {frozen[:10]}")
        if trainer.is_global_zero:
            record = {
                "initial_model": pl_module._pretrained,
                "optimizer_and_scheduler": "fresh; no training checkpoint resumed",
                "global_batch_size": batch,
                "train_frames": expected,
                "sampling": "one shuffled pass over frames per epoch; no source reweighting",
                "dropped_tail_frames_per_epoch": expected % batch,
                "trainable_parameters": sum(param.numel() for param in pl_module.parameters()),
                "independent_validation": False,
                "independent_test": False,
            }
            Path(trainer.default_root_dir, "run_record.json").write_text(json.dumps(record, indent=2))
            if any(type(logger).__name__ == 'SwanLabLogger' for logger in trainer.loggers):
                from swanlab.integration.pytorch_lightning import SwanLabLogger
                for logger in trainer.loggers:
                    if isinstance(logger, SwanLabLogger):
                        run = logger.experiment
                        Path(trainer.default_root_dir, 'swanlab_run.json').write_text(
                            json.dumps({'id': run.id, 'url': run.url, 'mode': 'online'}, indent=2))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    validate_config(config)
    serializable = OmegaConf.to_container(config, resolve=True)
    cfg = Config(OmegaConf.to_container(config, resolve=True))
    cfg.trainer.default_root_dir = str(
        Path(cfg.trainer.default_root_dir) / f"project_{cfg.trainer.project}" / cfg.trainer.exp_name
    )
    fill_num_nodes(cfg)
    root = Path(cfg.trainer.default_root_dir)
    if int(os.environ.get("RANK", 0)) == 0:
        root.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(root / "config.yaml"))
    seed_everything(cfg.trainer.pop("seed", 42) + int(os.environ.get("RANK", 0)), workers=True)
    os.environ["_max_steps"] = str(cfg.trainer.max_steps)
    strategy_helper(cfg)
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")
    data = DATASETS.build(cfg.data)
    model = MIMODEL.build(cfg.model)
    cfg.trainer.pop("save_interval", None)
    project = cfg.trainer.pop("project")
    experiment_name = cfg.trainer.pop("exp_name")
    cfg.trainer.pop("ckpt_path")
    callbacks = [ModelSummary(max_depth=2), TrainingProgress(), MixedRunRecord()]
    if cfg.trainer.get("enable_checkpointing", True):
        callbacks.append(ModelCheckpoint(
            dirpath=str(root), every_n_epochs=1, every_n_train_steps=0,
            save_top_k=-1, monitor=None, save_last="link", save_on_train_epoch_end=True,
            filename="epoch{epoch:02d}-step{step}", auto_insert_metric_name=False,
            enable_version_counter=False,
        ))
    cfg.trainer.callbacks = callbacks
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    loggers = [CSVLogger(str(root), name="csv")]
    online = cfg.logging.type == 'swanlab'
    if online:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        if cfg.logging.mode != 'online':
            raise ValueError('SwanLab logging must upload online for this experiment')
        loggers.append(SwanLabLogger(project=project, experiment_name=experiment_name,
            log_dir=str(root/'swanlab'), save_dir=str(root), mode='online',
            public=False, config=serializable))
    trainer = Trainer(**cfg.trainer, logger=loggers)
    succeeded = False
    try:
        trainer.fit(model=model, datamodule=data, ckpt_path=None)
        if trainer.is_global_zero:
            Path(root, "completed.json").write_text(json.dumps({
                "global_step": trainer.global_step,
                "epochs": trainer.current_epoch,
                "test_completed": False,
                "independent_validation": False,
                "independent_test": False,
            }, indent=2))
        succeeded = True
    finally:
        if online and trainer.is_global_zero:
            import swanlab
            if swanlab.get_run() is not None:
                swanlab.finish(state='success' if succeeded else 'crashed',
                    error=None if succeeded else 'Training failed; see the local training log.')


if __name__ == "__main__":
    main()
