from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class AutoMldVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: tuple[int, int] = (1, 128),
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.latent_size = latent_dim[0]
        self.latent_width = latent_dim[1]
        self.arch = arch

        self.input_proj = nn.Linear(nfeats, h_dim)
        self.encoder_pos = LearnedPositionalEncoding(h_dim)
        self.decoder_pos = LearnedPositionalEncoding(h_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.encoder_latent_proj = nn.Linear(h_dim, self.latent_width)

        if arch == "all_encoder":
            dec_layer = nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        elif arch == "encoder_decoder":
            dec_layer = nn.TransformerDecoderLayer(
                d_model=h_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        self.decoder_latent_proj = nn.Linear(self.latent_width, h_dim)
        self.output_proj = nn.Linear(h_dim, nfeats)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, h_dim) * 0.02)

    def encode(
        self,
        future_motion: torch.Tensor,
        history_motion: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.distributions.Normal]:
        bs = future_motion.shape[0]
        x = torch.cat([history_motion, future_motion], dim=1)
        x = self.input_proj(x)
        token = self.global_motion_token.unsqueeze(0).expand(bs, -1, -1)
        xseq = self.encoder_pos(torch.cat([token, x], dim=1))
        dist_tokens = self.encoder(xseq)[:, : token.shape[1]]
        dist_tokens = self.encoder_latent_proj(dist_tokens).permute(1, 0, 2).contiguous()
        mu = dist_tokens[: self.latent_size]
        logvar = dist_tokens[self.latent_size :]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if scale is not None:
            latent = latent / scale
        return latent, dist

    def decode(
        self,
        z: torch.Tensor,
        history_motion: torch.Tensor,
        nfuture: int,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scale is not None:
            z = z * scale
        bs = history_motion.shape[0]
        z = self.decoder_latent_proj(z.permute(1, 0, 2).contiguous())
        history = self.input_proj(history_motion)
        queries = torch.zeros(bs, nfuture, z.shape[-1], device=z.device, dtype=z.dtype)
        if self.arch == "all_encoder":
            xseq = self.decoder_pos(torch.cat([z, history, queries], dim=1))
            output = self.decoder(xseq)[:, -nfuture:]
        else:
            tgt = self.decoder_pos(torch.cat([history, queries], dim=1))
            output = self.decoder(tgt=tgt, memory=z)[:, -nfuture:]
        return self.output_proj(output).contiguous()
