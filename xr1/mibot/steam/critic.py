"""Construct the vendored RLinf critic and its unchanged processor."""

from .data import CAMERAS
from .upstream.data.pair_dataset import BinaryPairDataCollator
from .upstream.models.steam.configuration import SteamConfig
from .upstream.models.steam.ensemble_modeling_critic import (
    EnsembleSteamCriticModel,
    build_ensemble_members,
)
from .upstream.models.steam.modeling_critic import SteamCriticModel
from .upstream.models.steam.processing import SteamProcessor


def build_critic(config):
    cfg = SteamConfig(
        vision_repo_id=config["vision_path"],
        language_repo_id=config["language_path"],
        precision="float32",
        num_bins=config.get("num_bins", 32),
        stride_k=config.get("k", 32),
        ensemble_size=config.get("ensemble_size", 3),
        ensemble_head_seed_base=0,
        freeze_vision_encoder=False,
        freeze_language_model=False,
        fusion_hidden_dim=512,
        dropout=0.1,
        label_smoothing=0.05,
        use_gradient_checkpointing=True,
    )
    model = EnsembleSteamCriticModel(
        cfg, build_ensemble_members(SteamCriticModel(cfg), cfg.ensemble_size, 0)
    )
    model.gradient_checkpointing_enable()
    processor = SteamProcessor(
        tokenizer_name_or_path=config["language_path"],
        image_keys=CAMERAS,
        do_augment=False,
    )
    return model, BinaryPairDataCollator(processor, num_bins=cfg.num_bins, train=False)
