from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[0]]


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.pe.pe[timesteps]).permute(1, 0, 2)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, n_blocks: int, activation: str = "gelu"):
        super().__init__()
        if activation == "gelu":
            act = nn.GELU
        elif activation == "lrelu":
            act = lambda: nn.LeakyReLU(0.2)
        else:
            act = nn.ReLU
        layers: list[nn.Module] = []
        for _ in range(n_blocks):
            layers.extend([nn.Linear(dim, dim), act()])
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(x) + x)


class DenoiserMLP(nn.Module):
    def __init__(
        self,
        h_dim: int = 512,
        n_blocks: int = 2,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: tuple[int, int] = (2, 272),
        noise_shape: tuple[int, int] = (1, 128),
        cond_mask_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_mask_prob = cond_mask_prob
        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.time_embed = TimestepEmbedder(h_dim)
        input_dim = h_dim + clip_dim + history_shape[0] * history_shape[1] + noise_shape[0] * noise_shape[1]
        self.input_proj = nn.Linear(input_dim, h_dim)
        self.backbone = MLPBlock(h_dim, noise_shape[0] * noise_shape[1], n_blocks=n_blocks, activation=activation)

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0:
            keep = torch.bernoulli(torch.full((cond.shape[0], 1), 1.0 - self.cond_mask_prob, device=cond.device))
            return cond * keep
        return cond

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, y: dict) -> torch.Tensor:
        batch = x_t.shape[0]
        emb_time = self.time_embed(timesteps).squeeze(0)
        emb_text = self.mask_cond(y["text_embedding"], force_mask=y.get("uncond", False))
        emb_history = y["history_motion_normalized"].reshape(batch, -1)
        emb_noise = x_t.reshape(batch, -1)
        h = torch.cat([emb_time, emb_text, emb_history, emb_noise], dim=-1)
        out = self.backbone(self.input_proj(h))
        return out.reshape(batch, *self.noise_shape)


class DenoiserTransformer(nn.Module):
    def __init__(
        self,
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: tuple[int, int] = (2, 272),
        noise_shape: tuple[int, int] = (1, 128),
        cond_mask_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_mask_prob = cond_mask_prob
        self.noise_shape = noise_shape
        self.time_embed = TimestepEmbedder(h_dim)
        self.pos_enc = SinusoidalPositionalEncoding(h_dim)
        self.text_proj = nn.Linear(clip_dim, h_dim)
        self.history_proj = nn.Linear(history_shape[1], h_dim)
        self.noise_proj = nn.Linear(noise_shape[1], h_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            activation=activation,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out = nn.Linear(h_dim, noise_shape[1])

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0:
            keep = torch.bernoulli(torch.full((cond.shape[0], 1), 1.0 - self.cond_mask_prob, device=cond.device))
            return cond * keep
        return cond

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, y: dict) -> torch.Tensor:
        emb_time = self.time_embed(timesteps)
        emb_text = self.text_proj(self.mask_cond(y["text_embedding"], force_mask=y.get("uncond", False))).unsqueeze(0)
        emb_history = self.history_proj(y["history_motion_normalized"]).permute(1, 0, 2)
        emb_noise = self.noise_proj(x_t).permute(1, 0, 2)
        xseq = self.pos_enc(torch.cat([emb_time, emb_text, emb_history, emb_noise], dim=0))
        output = self.encoder(xseq)[-self.noise_shape[0] :]
        return self.out(output).permute(1, 0, 2)


class ClassifierFreeGuidanceWrapper(nn.Module):
    """Wraps a denoiser model to apply classifier-free guidance at inference time.

    At each diffusion step, runs the model twice:
      1. Conditional pass (with text embedding)
      2. Unconditional pass (text embedding zeroed out)

    Final output = uncond + guidance_scale * (cond - uncond)

    The wrapped model must have been trained with cond_mask_prob > 0.
    """

    def __init__(self, model: nn.Module, guidance_scale: float = 5.0) -> None:
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, y: dict) -> torch.Tensor:
        # Conditional forward
        y["uncond"] = False
        out_cond = self.model(x_t, timesteps, y)

        # Unconditional forward
        y["uncond"] = True
        out_uncond = self.model(x_t, timesteps, y)

        # Reset flag
        y["uncond"] = False

        return out_uncond + self.guidance_scale * (out_cond - out_uncond)
