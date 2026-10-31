"""XR1 backbone and observation-only RLToken features for USB."""

import os
from pathlib import Path

import torch
from torch import nn

from .data import sha256
from .upstream.modules.rlt_token_transformer import RLTTokenTransformer


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def unpack_prefix(hidden, segments):
    """Undo XR1 sequence packing, excluding all SFT target turns."""
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Expected packed XR1 hidden states [1, sequence, width]")
    lengths = segments[:, 1] - segments[:, 0]
    if (lengths <= 0).any():
        raise ValueError("Empty observation prefix")
    prefix = hidden.new_zeros(len(segments), int(lengths.max()), hidden.shape[-1])
    mask = torch.zeros(prefix.shape[:2], device=hidden.device, dtype=torch.bool)
    for i, (start, end) in enumerate(segments.tolist()):
        if start < 0 or end > hidden.shape[1]:
            raise ValueError("Prefix segment outside packed sequence")
        prefix[i, : end - start] = hidden[0, start:end].detach()
        mask[i, : end - start] = True
    return prefix, mask


class ActionCoordinates:
    """Bijective bounded RLT coordinates for XR1's unbounded z-score actions.

    u = tanh(z / scale); z = scale * atanh(u). Saturated values are rejected,
    never silently clipped. The actor's tanh is therefore retained unchanged.
    """

    def __init__(self, scale=5.0):
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("Action coordinate scale must be positive")

    def encode(self, normalized):
        result = torch.tanh(normalized.float() / self.scale)
        self._validate(result)
        return result

    def decode(self, bounded):
        self._validate(bounded)
        return self.scale * torch.atanh(bounded.float())

    @staticmethod
    def _validate(value):
        if not torch.isfinite(value).all() or (value.abs() >= 1).any():
            raise ValueError(
                "RLT action coordinate is nonfinite or saturated at +/-1; increase scale for data or inspect the policy"
            )


def load_xr1(config, device="cpu"):
    os.environ["XR1_VLM_PATH"] = config["vlm_path"]
    from transformers.modeling_utils import no_init_weights

    from mibot.models.VLA.XR1 import xr1

    with no_init_weights():
        model = xr1(
            full_finetune=True,
            active_action_dims=list(range(7)),
            ffn_gradient_checkpointing=True,
            async_train=True,
        )
    checkpoint = Path(config["base_checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = saved.get("module", saved.get("state_dict", saved))
    if not all(k.startswith("model.") for k in state):
        raise ValueError("Expected XR1 Lightning/DeepSpeed model.* checkpoint")
    state = {k[len("model.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state, saved
    return model.to(device).eval()


class XR1RLT(nn.Module):
    def __init__(self, base, rlt_config, *, mode="joint", alpha=1.0, action_scale=5.0):
        super().__init__()
        self.base = base
        self.rlt_config = dict(rlt_config)
        if self.rlt_config["input_dim"] != base.vlm.config.text_config.hidden_size:
            raise ValueError("RLT input_dim must match XR1 VLM hidden width")
        self.rlt_module = RLTTokenTransformer(**self.rlt_config)
        self.alpha = float(alpha)
        self.coordinates = ActionCoordinates(action_scale)
        self.set_stage1_mode(mode)

    def set_stage1_mode(self, mode):
        if mode not in ("joint", "rlt_only"):
            raise ValueError("mode must be joint or rlt_only")
        self.mode = mode
        self.base.requires_grad_(mode == "joint")
        # Match RLinf Stage 1's frozen VLM prefix. XR1 action experts and its
        # choice/score projectors retain their native SFT objective in joint mode.
        self.base.vlm.requires_grad_(False)
        # AdamW must keep trainable master weights and moments in fp32. Small
        # 1e-5 updates would otherwise round away in the imported bf16 weights.
        for parameter in self.base.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.rlt_module.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.vlm.eval()
        if self.mode == "rlt_only":
            self.base.eval()
        return self

    def forward(self, batch):
        batch = dict(batch)
        dense_prefix = batch.pop("rlt_prefix_length", None)
        if self.mode == "rlt_only":
            observation = batch["observation"]
            with torch.no_grad():
                inputs = {k: v for k, v in observation.items() if k != "state"}
                outputs = self.base.vlm(**inputs, use_cache=False, skip_logits=True)
            prefix, mask = (
                outputs.hidden_states.detach(),
                observation["attention_mask"].bool(),
            )
            vla_loss = prefix.new_zeros((), dtype=torch.float32)
        else:
            captured = []

            def capture(module, inputs, output):
                if dense_prefix is not None:
                    prefix = output.hidden_states[:, :dense_prefix].detach()
                    captured.append(
                        (prefix, batch["attention_mask"][:, :dense_prefix].bool())
                    )
                    return
                captured.append(
                    unpack_prefix(
                        output.hidden_states, batch["action_vlm_condition_segments"]
                    )
                )

            handle = self.base.vlm.register_forward_hook(capture)
            self.base.vlm._rlt_split_prefix = dense_prefix
            try:
                losses = self.base(dict(batch), return_loss=True)
            finally:
                handle.remove()
                self.base.vlm._rlt_split_prefix = None
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one XR1 VLM call")
            prefix, mask = captured[0]
            vla_loss = losses["loss"]
        # RLinf uses fp32 RLT reconstruction even when the VLA runs in bf16.
        with torch.autocast(
            device_type=prefix.device.type,
            enabled=getattr(self, "rlt_amp", False),
            dtype=torch.bfloat16,
        ):
            rlt_loss, details = self.rlt_module(prefix.float(), mask)
        return {
            "loss": rlt_loss + self.alpha * vla_loss,
            "rlt_loss": rlt_loss,
            "vla_loss": vla_loss,
            "z_rl": details["z_rl"],
        }

    @torch.no_grad()
    def extract(
        self, observation, *, chunk_len=10, noise=None, action_coordinates="bounded"
    ):
        """Frozen feature/reference contract consumed by RLinf's RLTMLPPolicy."""
        if self.training:
            raise ValueError("Call eval() before extracting frozen RLT features")
        if not 1 <= chunk_len <= 30:
            raise ValueError("Executed action chunk must be between 1 and 30 steps")
        if action_coordinates not in ("bounded", "zscore"):
            raise ValueError("Unknown action coordinates")
        captured = []

        def capture(module, inputs, output):
            captured.append(output.hidden_states.detach())

        handle = self.base.vlm.register_forward_hook(capture)
        b = observation["state"].shape[0]
        mask = torch.zeros(
            b, 30, 60, device=observation["state"].device, dtype=torch.bfloat16
        )
        mask[..., :7] = 1
        try:
            with torch.autocast(device_type=mask.device.type, dtype=torch.bfloat16):
                condition = self.base.prepare_flow_condition(observation, mask)
                # Native XR1 forward computes RoPE in the bf16 action dtype,
                # even when trainable master weights are fp32.
                condition["position_embeds"] = tuple(
                    x.to(torch.bfloat16) for x in condition["position_embeds"]
                )
                if noise is None:
                    noise = torch.randn_like(mask)
                if noise.shape != mask.shape:
                    raise ValueError("Noise must be [B,30,60]")
                reference = self.base._generate(noise.to(mask), condition) * mask
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one observation-only VLM pass")
        with torch.autocast(device_type=mask.device.type, enabled=False):
            z = self.rlt_module.encode_flat(
                captured[0].float(), observation["attention_mask"].bool()
            )
        return {
            "z_rl": z.float(),
            "proprio": observation["state"][:, 0, :16].float(),
            "ref_chunk": self.coordinates.encode(reference[..., :7])
            if action_coordinates == "bounded"
            else reference[..., :7].float(),
        }, reference.float()


def build_model(config, device="cpu", stage1_checkpoint=None):
    base = load_xr1(config, device)
    model = XR1RLT(
        base,
        config["rlt"],
        mode=config["stage1"]["mode"],
        alpha=config["stage1"]["alpha"],
        action_scale=config["action_scale"],
    )
    # Move only the sidecar: calling .to(dtype=bf16) on the wrapper would also
    # downcast the fp32 reconstruction network.
    model.rlt_module.to(device=device, dtype=torch.float32)
    if stage1_checkpoint:
        saved = torch.load(
            stage1_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        validate_assets(saved["assets"], config)
        if saved["config"]["rlt"] != config["rlt"]:
            raise ValueError("Stage 1 RLT architecture mismatch")
        if saved["config"]["action_scale"] != config["action_scale"]:
            raise ValueError("Stage 1 action coordinate mismatch")
        model.rlt_module.load_state_dict(saved["rlt_module"], strict=True)
        # Joint checkpoints store every changed non-VLM parameter; frozen VLM
        # weights remain in the immutable base checkpoint, whose hash is checked.
        expected = {k for k in base.state_dict() if not k.startswith("vlm.")}
        delta = saved["base_non_vlm"]
        if set(delta) != expected:
            raise ValueError("Incomplete Stage 1 XR1 action expert checkpoint")
        merged = base.state_dict()
        merged.update(delta)
        base.load_state_dict(merged, strict=True)
    return model


def asset_metadata(config):
    return {k: sha256(config[k]) for k in ("base_checkpoint", "normalization")}


def validate_assets(assets, config):
    if assets != asset_metadata(config):
        raise ValueError(
            "Base weights or normalization differ from the Stage 1 checkpoint"
        )


def save_stage1(path, model, config, optimizer, step, assets, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format="xr1-rlt-stage1-v1",
        config=config,
        assets=assets,
        step=step,
        rlt_module=model.rlt_module.state_dict(),
        base_non_vlm={
            k: v for k, v in model.base.state_dict().items() if not k.startswith("vlm.")
        },
        optimizer=optimizer.state_dict(),
        **extra,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
