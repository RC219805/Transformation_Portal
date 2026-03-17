"""1D UNet for latent diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class UNet1D(nn.Module):
    """Simple 1D UNet for latent space diffusion."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.down = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2)
        )
        self.mid = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2)
        )
        self.up = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.out = nn.Linear(dim, dim)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise."""
        t_emb = self.time_mlp(t)
        c = self.cond_proj(cond)
        h = x_t + t_emb + c
        h = self.down(h)
        h = self.mid(h)
        h = self.up(h)
        return self.out(h)
