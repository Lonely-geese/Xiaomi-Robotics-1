# Copyright (C) 2026 Xiaomi Corporation.
from importlib import import_module
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from mibot.models import MIMODEL

logger = logging.get_logger(__name__)


@MIMODEL.register_module()
class BaseRunner(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")

    def configure_model(self) -> None:
        if hasattr(self, "model"):
            return
        self.model = MIMODEL.build(self._model)

        if self._pretrained:
            ckpt = torch.load(
                self._pretrained,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            info = self.load_state_dict(ckpt["module"], strict=False)
            if info.missing_keys or info.unexpected_keys:
                raise RuntimeError(f"Checkpoint does not match the configured model: {info}")
            logger.info(f"Loaded pretrained model from {self._pretrained}: {info}")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters())
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]]) -> Optimizer:
        module_params = cfg.get("params", {})
        parameters = list(parameters)

        no_decay = [
            "bias",
            "norm",
            "Norm",
            "ln",
            "Ln",
            "rotary_emb",
            "adaln",
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": module_params.get("weight_decay", 0.1),
            },
            {
                "params": [p for n, p in parameters if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_size = int(batch["action"].shape[0])
        self.log("train/token", float(batch["input_ids"].shape[1]), batch_size=batch_size)
        loss_dict = self.model(batch, return_loss=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("lr", self.optimizers().param_groups[0]["lr"], batch_size=batch_size)
        return loss_dict

    def _evaluation_step(self, batch, batch_idx, stage):
        batch_size = int(batch["action"].shape[0])
        devices = [batch["action"].device.index] if batch["action"].is_cuda else []
        # Fixed validation noise across epochs, without perturbing training RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(100000 * self.global_rank + batch_idx + 12345)
            losses = self.model(batch, return_loss=True)
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, batch_idx, "test")
