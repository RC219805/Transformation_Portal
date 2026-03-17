"""Diffusion World Model for pipeline outcome generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformation_portal.dwm.schedule import DiffusionSchedule
from transformation_portal.dwm.unet_1d import UNet1D


class DiffusionWorldModel(nn.Module):
    """Diffusion model for generating pipeline outcomes.

    Generates latent representations of pipeline outcomes
    (metrics, image features, 3D features) conditioned on
    pipeline context.
    """

    def __init__(self, latent_dim: int, cond_dim: int, T: int = 1000) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.unet = UNet1D(latent_dim, cond_dim)
        self.sched = DiffusionSchedule(T)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass."""
        noise = torch.randn_like(x0)
        x_t = self.sched.q_sample(x0, t, noise)
        pred = self.unet(x_t, t.float().unsqueeze(-1) / self.sched.T, cond)
        return pred, noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate samples via DDIM-like sampling."""
        B = cond.size(0)
        x = torch.randn(B, self.latent_dim, device=cond.device)
        T = self.sched.T

        for i in reversed(range(steps)):
            t = torch.full((B,), int(i * (T / steps)), device=cond.device, dtype=torch.long)
            eps = self.unet(x, t.float().unsqueeze(-1) / T, cond)
            a = self.sched.alphas[t].unsqueeze(-1)
            a_bar = self.sched.alpha_bar[t].unsqueeze(-1)
            x0_pred = (x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()
            if i > 0:
                x = a_bar.sqrt() * x0_pred
            else:
                x = x0_pred

        return x
