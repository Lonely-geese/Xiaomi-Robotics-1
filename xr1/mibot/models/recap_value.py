"""Task-specific multimodal categorical value critic for offline RECAP.

Unlike RLinf's SigLIP/Gemma critic, this compact critic learns from three 64px
RGB views and measured proprioception. The task is fixed to Ethernet insertion;
no outcome, frame index, return or human-intervention feature enters the model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RecapValueCritic(nn.Module):
    def __init__(self, bins=201):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(3*96*4+30, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, bins))
        self.register_buffer('support', torch.linspace(-1, 0, bins))
        self.register_buffer('state_mean', torch.zeros(30))
        self.register_buffer('state_std', torch.ones(30))

    def forward(self, images, state):
        batch = images.shape[0]
        visual = self.vision(images.flatten(0, 1).float()/255.0).reshape(batch, -1)
        state = ((state.float()-self.state_mean)/self.state_std.clamp_min(1e-3)).clamp(-10, 10)
        return self.head(torch.cat([visual, state], dim=-1))

    def value(self, images, state):
        return (self(images, state).float().softmax(-1)*self.support).sum(-1)

    def loss(self, logits, targets):
        # Linear interpolation between adjacent bins, including both endpoints.
        index = (targets.clamp(-1, 0)+1)*(len(self.support)-1)
        lower = index.floor().long()
        upper = (lower+1).clamp_max(len(self.support)-1)
        weight = index-lower
        logprobs = F.log_softmax(logits.float(), dim=-1)
        return (-logprobs.gather(1, lower[:, None]).squeeze(1)*(1-weight)
            - logprobs.gather(1, upper[:, None]).squeeze(1)*weight).mean()
