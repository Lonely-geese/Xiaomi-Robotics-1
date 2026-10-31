"""Dense USB batches and exact causal-prefix splitting for a frozen XR1 VLM."""

import types

import torch
from torch.utils.data import default_collate

from mibot.data.collate.custom_collate import IM_START_ID, STATE_ID, CustomCollate


class DenseUSBCollator(CustomCollate):
    def __call__(self, items):
        rows = []
        payloads = []
        prefixes = []
        for item in items:
            inputs = self.processor.apply_chat_template(
                item["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                do_resize=False,
            )
            inputs["position_ids"] = self._position_ids(inputs)
            ids = inputs["input_ids"][0]
            state = (ids == STATE_ID).nonzero().flatten()
            starts = (ids == IM_START_ID).nonzero().flatten()
            if len(state) != 1:
                raise ValueError("Expected one state token per USB sample")
            prefixes.append(int(starts[starts < state[0]][-1]))
            rows.append(inputs)
            payloads.append({k: v for k, v in item.items() if k != "messages"})
        if len(set(prefixes)) != 1:
            raise ValueError(
                "Dense USB optimization requires equal observation prefix lengths"
            )
        length = max(x["input_ids"].shape[1] for x in rows)
        b = len(rows)
        ids = torch.full(
            (b, length), self.processor.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros(b, length, dtype=torch.bool)
        pos = torch.zeros(3, b, length, dtype=torch.long)
        for i, row in enumerate(rows):
            n = row["input_ids"].shape[1]
            ids[i, :n] = row["input_ids"][0]
            mask[i, :n] = True
            pos[:, i, :n] = row["position_ids"][:, 0]
        out = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": pos,
            "rlt_prefix_length": prefixes[0],
        }
        for key in ("pixel_values", "image_grid_thw"):
            out[key] = torch.cat([r[key] for r in rows])
        targets = [p.pop("vlm_action_target") for p in payloads]
        masks = [p.pop("vlm_action_mask") for p in payloads]
        out.update(default_collate(payloads))
        out["vlm_action_target"] = torch.cat(targets)
        out["vlm_action_mask"] = torch.cat(masks)
        return out


def configure_fast(model, *, rlt_amp=True, checkpoint_rlt=True, checkpoint_dit=True):
    """Keep fp32 parameters/Adam state; optionally use bf16 GEMMs and rematerialization."""
    from torch.utils.checkpoint import checkpoint
    from transformers import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    vlm = model.base.vlm
    lm = vlm.model.language_model
    original_lm = lm.forward
    original_vlm = vlm.forward

    def language_forward(self, *args, **kwargs):
        length = getattr(vlm, "_rlt_split_prefix", None)
        if length is None:
            return original_lm(*args, **kwargs)
        if args:
            raise ValueError("Split VLM expects keyword inputs")
        embeddings = kwargs["inputs_embeds"]
        positions = kwargs["position_ids"]
        mask = kwargs.get("attention_mask")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Training prefix must start without a KV cache")
        with torch.no_grad():
            pre = original_lm(
                inputs_embeds=embeddings[:, :length].detach(),
                position_ids=positions[:, :, :length],
                attention_mask=mask[:, :length] if mask is not None else None,
                use_cache=True,
                visual_pos_masks=kwargs["visual_pos_masks"][:, :length],
                deepstack_visual_embeds=kwargs.get("deepstack_visual_embeds"),
            )
        # A distinct cache object prevents suffix decoding from extending the
        # observation cache consumed by the action expert.
        suffix_cache = DynamicCache(
            ddp_cache_data=list(pre.past_key_values), config=self.config
        )
        suffix = original_lm(
            inputs_embeds=embeddings[:, length:],
            position_ids=positions[:, :, length:],
            attention_mask=mask,
            use_cache=True,
            past_key_values=suffix_cache,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=torch.cat(
                [pre.last_hidden_state, suffix.last_hidden_state], 1
            ),
            past_key_values=pre.past_key_values,
        )

    def vlm_forward(self, *args, **kwargs):
        result = original_vlm(*args, **kwargs)
        length = getattr(self, "_rlt_split_prefix", None)
        if length is not None:
            result.position_ids = result.position_ids[:, :, :length]
            result.attention_mask = result.attention_mask[:, :length]
        return result

    lm.forward = types.MethodType(language_forward, lm)
    vlm.forward = types.MethodType(vlm_forward, vlm)
    model.rlt_amp = bool(rlt_amp)
    if checkpoint_rlt:
        for layer in list(model.rlt_module.encoder.layers) + list(
            model.rlt_module.decoder.layers
        ):
            original = layer.forward

            def wrapped(*args, _forward=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            layer.forward = wrapped
    if checkpoint_dit:
        for layer in model.base.dit.layers:
            original = layer.forward

            def wrapped(*args, _forward=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            layer.forward = wrapped


def configure_fast_zero_checkpoint():
    """Avoid element-by-element ByteTensor(bytearray) conversion for large states."""
    import io

    import torch.distributed as dist
    import torch.distributed.optim.zero_redundancy_optimizer as zero

    def broadcast_object(
        obj, src_rank, group=dist.group.WORLD, device=None
    ):
        device = torch.device("cpu") if device is None else device
        if dist.get_rank() == src_rank:
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            # frombuffer views serialized bytes without converting billions of
            # individual Python integers; serialization and collective order match ZRO.
            data = torch.frombuffer(buffer.getbuffer(), dtype=torch.uint8).to(device)
            length = torch.tensor([data.numel()], dtype=torch.long, device=device)
            dist.broadcast(length, src=src_rank, group=group)
            dist.broadcast(data, src=src_rank, group=group)
            return obj
        length = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(length, src=src_rank, group=group)
        data = torch.empty(int(length.item()), dtype=torch.uint8, device=device)
        dist.broadcast(data, src=src_rank, group=group)
        return torch.load(
            io.BytesIO(data.cpu().numpy()), map_location=device, weights_only=False
        )

    zero._broadcast_object = broadcast_object
