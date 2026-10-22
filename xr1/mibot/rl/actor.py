"""Expose XR-1's actual DiT vector field to QAM without SFT auxiliary losses."""

import torch
from torch import nn


class XR1FlowActor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stage = "offline"

    def set_stage(self, stage):
        if stage not in ("offline", "online"):
            raise ValueError("unknown LWD stage")
        self.stage = stage
        self.model.zero_grad(set_to_none=True)
        self.model.requires_grad_(True)
        self.model.vlm.requires_grad_(stage == "offline")
        # These heads serve SFT choice/score prediction, not the flow policy.
        for name in ("state_projector_choice", "action_projector_choice", "score_projector_choice"):
            getattr(self.model, name).requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "online":
            self.model.vlm.eval()
        return self

    def prepare(self, observation, mask):
        return self.model.prepare_flow_condition(observation, mask)

    def velocity(self, condition, action, time):
        dtype = self.model.sink.weight.dtype
        return self.model.dit_forward(action.to(dtype), time.to(dtype), **condition).float()
