from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] (batch_first=True) or [T, B, D] (batch_first=False)."""
        if x.ndim == 3:
            # Assume first dim is sequence if shape[0] <= max_len and shape[1] != 1
            # We use a heuristic: if called from seq-first code, T is dim 0
            # For safety, we support both via the caller passing the right shape.
            seq_len = x.shape[1] if x.shape[0] > x.shape[1] or x.shape[0] == x.shape[1] else x.shape[0]
        return x + self.pe[:, :seq_len].transpose(0, 1) if x.shape[0] != x.shape[1] and x.shape[0] < 512 else x + self.pe[:, :x.shape[1]]


class SeqFirstLearnedPE(nn.Module):
    """Learned positional encoding for sequence-first tensors [T, B, D]."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, D]"""
        return x + self.pe[: x.shape[0]]


# ---------------------------------------------------------------------------
# Skip Transformer Encoder (U-Net style skip connections)
# Matches DART-main's SkipTransformerEncoder from mld/models/operator/cross_attention.py
# ---------------------------------------------------------------------------

class TransformerEncoderLayerCustom(nn.Module):
    """Standard post-norm transformer encoder layer with positional embedding injection."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SkipTransformerEncoder(nn.Module):
    """Transformer encoder with U-Net style skip connections.

    Requires num_layers to be odd. The layers are split into:
      - input_blocks: (num_layers - 1) // 2 layers
      - middle_block: 1 layer
      - output_blocks: (num_layers - 1) // 2 layers (with skip from input_blocks)
    """

    def __init__(self, encoder_layer: TransformerEncoderLayerCustom, num_layers: int, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1, f"SkipTransformerEncoder requires odd num_layers, got {num_layers}"

        num_block = (num_layers - 1) // 2
        self.input_blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_block)])
        self.middle_block = copy.deepcopy(encoder_layer)
        self.output_blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_block)])
        self.linear_blocks = nn.ModuleList([nn.Linear(2 * self.d_model, self.d_model) for _ in range(num_block)])

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        x = src
        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# AutoMldVae with Skip Transformer
# ---------------------------------------------------------------------------

class AutoMldVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: tuple[int, int] = (1, 256),
        h_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 7,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.latent_size = latent_dim[0]
        self.latent_width = latent_dim[1]
        self.h_dim = h_dim
        self.arch = arch

        self.input_proj = nn.Linear(nfeats, h_dim)
        self.encoder_pos = SeqFirstLearnedPE(h_dim)
        self.decoder_pos = SeqFirstLearnedPE(h_dim)

        encoder_layer = TransformerEncoderLayerCustom(
            d_model=h_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation,
        )
        encoder_norm = nn.LayerNorm(h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encoder_latent_proj = nn.Linear(h_dim, self.latent_width)

        if arch == "all_encoder":
            decoder_norm = nn.LayerNorm(h_dim)
            self.decoder = SkipTransformerEncoder(
                copy.deepcopy(encoder_layer), num_layers, decoder_norm
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}. Only 'all_encoder' is supported.")

        self.decoder_latent_proj = nn.Linear(self.latent_width, h_dim)
        self.output_proj = nn.Linear(h_dim, nfeats)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, h_dim))

        # Latent rescaling buffers
        self.register_buffer("latent_mean", torch.tensor(0.0))
        self.register_buffer("latent_std", torch.tensor(1.0))

    def encode(
        self,
        future_motion: torch.Tensor,
        history_motion: torch.Tensor,
        scale_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.distributions.Normal]:
        bs = future_motion.shape[0]
        x = torch.cat([history_motion, future_motion], dim=1)  # [B, H+F, nfeats]
        x = self.input_proj(x)  # [B, H+F, h_dim]
        x = x.permute(1, 0, 2)  # [T, B, h_dim] (seq-first for transformer)

        # Global motion tokens for distribution
        token = self.global_motion_token[:, None, :].expand(-1, bs, -1)  # [2*latent_size, B, h_dim]
        xseq = torch.cat([token, x], dim=0)  # [2*latent_size + T, B, h_dim]

        xseq = self.encoder_pos(xseq)
        dist_tokens = self.encoder(xseq)[: token.shape[0]]  # [2*latent_size, B, h_dim]
        dist_tokens = self.encoder_latent_proj(dist_tokens)  # [2*latent_size, B, latent_width]

        mu = dist_tokens[: self.latent_size]
        logvar = dist_tokens[self.latent_size :]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        if scale_latent:
            latent = latent / self.latent_std
        return latent, dist

    def decode(
        self,
        z: torch.Tensor,
        history_motion: torch.Tensor,
        nfuture: int,
        scale_latent: bool = False,
    ) -> torch.Tensor:
        if scale_latent:
            z = z * self.latent_std
        bs = history_motion.shape[0]

        z = self.decoder_latent_proj(z)  # [latent_size, B, h_dim]
        history = self.input_proj(history_motion).permute(1, 0, 2)  # [H, B, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device, dtype=z.dtype)

        xseq = torch.cat([z, history, queries], dim=0)  # [latent_size + H + F, B, h_dim]
        xseq = self.decoder_pos(xseq)
        output = self.decoder(xseq)[-nfuture:]  # [F, B, h_dim]

        output = self.output_proj(output)  # [F, B, nfeats]
        return output.permute(1, 0, 2).contiguous()  # [B, F, nfeats]

    @torch.no_grad()
    def fit_latent_scale(self, loader, device: torch.device, history_length: int,
                         future_length: int, num_primitives: int, max_batches: int = 0) -> tuple[float, float]:
        """Scan the loader and compute global latent mean/std for rescaling."""
        was_training = self.training
        self.eval()

        latents = []
        for batch_idx, batch in enumerate(loader, start=1):
            motion = batch["motion"].to(device)
            for primitive_idx in range(num_primitives):
                start = primitive_idx * future_length
                history = motion[:, start: start + history_length].contiguous()
                future = motion[:, start + history_length: start + history_length + future_length].contiguous()
                latent, _ = self.encode(future, history)
                latents.append(latent.detach().flatten().float())
            if max_batches and batch_idx >= max_batches:
                break

        all_latents = torch.cat(latents, dim=0)
        mean = all_latents.mean()
        std = (all_latents - mean).pow(2).mean().sqrt().clamp(min=1e-6)

        self.latent_mean = mean.detach().to(self.latent_mean.device)
        self.latent_std = std.detach().to(self.latent_std.device)

        if was_training:
            self.train()
        return float(mean.item()), float(std.item())
