"""Independent Gemma 3 + SigLIP state encoder for the shared V/Q branch."""

import torch
from torch import nn


class ValueBackbone(nn.Module):
    def __init__(self, language, vision, state_dim=60):
        super().__init__()
        self.language = language
        self.vision = vision
        self.feature_dim = language.config.hidden_size
        self.visual_projector = nn.Linear(vision.config.hidden_size, self.feature_dim)
        self.state_projector = nn.Linear(state_dim, self.feature_dim)
        self.readout = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

    @classmethod
    def from_pretrained(cls, language_path="google/gemma-3-270m-it",
                        vision_path="google/siglip-so400m-patch14-384", **kwargs):
        from transformers import AutoModelForCausalLM, SiglipVisionModel
        # Gemma's released checkpoint has `model.*` keys. In Transformers
        # 4.57.1 AutoModel/Gemma3TextModel does not strip that prefix and silently
        # initializes the decoder anew. Load the matching LM wrapper first.
        language, language_info = AutoModelForCausalLM.from_pretrained(
            language_path, output_loading_info=True, **kwargs)
        vision, vision_info = SiglipVisionModel.from_pretrained(
            vision_path, output_loading_info=True, **kwargs)
        for name, info in (("language", language_info), ("vision", vision_info)):
            errors = {key: info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")
                      if info.get(key)}
            if errors:
                raise ValueError(f"incomplete pretrained {name} encoder: {errors}")
        return cls(language.get_decoder(), vision)

    def forward(self, observation):
        # Each camera contributes its patch tokens. The final readout token can
        # attend to all vision, language and state tokens in the causal decoder.
        pixels = observation["pixel_values"]  # [B,V,3,H,W], SigLIP normalized
        if pixels.ndim != 5:
            raise ValueError("critic pixel_values must be [B,V,3,H,W]")
        batch, views = pixels.shape[:2]
        vision = self.vision(pixel_values=pixels.flatten(0, 1)).last_hidden_state
        patches = vision.shape[1]
        visual = self.visual_projector(vision).reshape(batch, views * patches, -1)
        text = self.language.get_input_embeddings()(observation["input_ids"])
        state = observation["state"].reshape(batch, -1, self.state_projector.in_features)
        state = self.state_projector(state)
        embeddings = torch.cat([visual, text, state, self.readout.expand(batch, -1, -1)], 1)
        text_mask = observation["attention_mask"].bool()
        view_mask = observation.get("view_mask", torch.ones(batch, views, device=pixels.device))
        visual_mask = view_mask.bool().repeat_interleave(patches, dim=1)
        suffix_mask = torch.ones(batch, state.shape[1] + 1, dtype=torch.bool, device=pixels.device)
        mask = torch.cat([visual_mask, text_mask, suffix_mask], 1)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        output = self.language(inputs_embeds=embeddings, attention_mask=mask,
                               position_ids=positions, use_cache=False, return_dict=True)
        return output.last_hidden_state[:, -1]
