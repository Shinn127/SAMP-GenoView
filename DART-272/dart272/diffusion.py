from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999).float()


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.to(t.device).gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, num_steps: int = 10) -> None:
        self.num_steps = num_steps
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, model_kwargs: dict) -> torch.Tensor:
        x_start = model(x_t, t, model_kwargs["y"]).clamp(min=-5.0, max=5.0)
        mean, _, log_var = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    def p_sample_loop(self, model, shape: tuple[int, ...], model_kwargs: dict, device: torch.device) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, model_kwargs)
        return x_t
