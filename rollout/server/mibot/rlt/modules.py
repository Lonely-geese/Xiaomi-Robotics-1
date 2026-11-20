"""State-dict compatible RLToken modules, not a recovered original implementation.

Parameterless operations are explicit hypotheses: pre-LN, GELU then SwiGLU,
bidirectional encoder with an appended latent token, broadcast-latent decoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, 2 * width)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.silu(gate)


class Block(nn.Module):
    def __init__(self, width, heads, mlp_ratio, dropout):
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), SwiGLU(hidden), nn.Linear(hidden, width))

    def forward(self, x, padding_mask=None, attn_mask=None):
        h = self.self_norm(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=padding_mask,
                               attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, prefix_seq_len, num_layers, num_heads, mlp_ratio, dropout_rate):
        super().__init__()
        self.rl_token_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.prefix_pos_enc = nn.Parameter(torch.zeros(prefix_seq_len, embed_dim))
        self.rl_token_pos_enc = nn.Parameter(torch.zeros(1, embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(num_layers)])

    def forward(self, prefix, valid_mask):
        n, length, _ = prefix.shape
        if length > len(self.prefix_pos_enc):
            raise ValueError(f"Prefix length {length} exceeds checkpoint capacity {len(self.prefix_pos_enc)}; no truncation allowed")
        if valid_mask.shape != prefix.shape[:2] or not valid_mask.bool().any(dim=1).all():
            raise ValueError("Expected at least one valid observation token per example")
        # Mask values before projection, including NaN padding supplied by callers.
        x = self.input_proj(prefix.masked_fill(~valid_mask.bool()[..., None], 0))
        x = x + self.prefix_pos_enc[:length]
        token = (self.rl_token_embed + self.rl_token_pos_enc).expand(n, 1, -1)
        x = torch.cat([x, token], dim=1)
        padding = torch.cat([~valid_mask.bool(), torch.zeros(n, 1, device=x.device, dtype=torch.bool)], dim=1)
        for layer in self.layers:
            x = layer(x, padding)
        return x[:, -1]


class Decoder(nn.Module):
    def __init__(self, input_dim, embed_dim, prefix_seq_len, num_layers, num_heads, mlp_ratio, dropout_rate):
        super().__init__()
        self.decoder_pos_enc = nn.Parameter(torch.zeros(prefix_seq_len, embed_dim))
        self.teacher_input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, z, length, valid_mask=None):
        if not 0 < length <= len(self.decoder_pos_enc):
            raise ValueError("Invalid decoder length")
        # teacher_input_proj is loaded strictly but is not used in this observation-only
        # reconstruction. Original teacher forcing/inference schedule is unknown.
        x = z[:, None] + self.decoder_pos_enc[:length]
        for layer in self.layers:
            x = layer(x, None if valid_mask is None else ~valid_mask.bool())
        return self.output_proj(x)

    def teacher_forced(self, z, prefix, valid_mask):
        """RLT paper Eq. 2: causal [z, stop_gradient(prefix[:-1])].

        Explicit new implementation, not proof of the legacy checkpoint graph.
        The decoder trains representation reconstruction, not VLA KV caches.
        """
        length = prefix.shape[1]
        if not 0 < length <= len(self.decoder_pos_enc):
            raise ValueError('Invalid decoder length')
        teacher = prefix.detach().masked_fill(~valid_mask.bool()[..., None], 0)
        x = torch.cat([z[:, None], self.teacher_input_proj(teacher[:, :-1])], dim=1)
        x = x + self.decoder_pos_enc[:length]
        padding = torch.cat([torch.zeros(len(z), 1, dtype=torch.bool, device=z.device),
                             ~valid_mask[:, :-1].bool()], dim=1)
        causal = torch.ones(length, length, dtype=torch.bool, device=z.device).triu(1)
        for layer in self.layers:
            x = layer(x, padding_mask=padding, attn_mask=causal)
        return self.output_proj(x)


class RLToken(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

    def forward(self, prefix, valid_mask):
        z = self.encoder(prefix, valid_mask)
        return z, self.decoder(z, prefix.shape[1], valid_mask)

    def reconstruction_loss(self, prefix, valid_mask):
        """Train encoder/decoder only; VLA input embeddings are stop-gradient."""
        z = self.encoder(prefix.detach(), valid_mask)
        decoded = self.decoder.teacher_forced(z, prefix, valid_mask)
        error = (decoded-prefix.detach()).square().sum(-1)
        return error.masked_fill(~valid_mask.bool(), 0).sum(1).mean()
