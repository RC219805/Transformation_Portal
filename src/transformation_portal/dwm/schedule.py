"""Diffusion schedule for DWM."""

from __future__ import annotations

import torch


class DiffusionSchedule:
    """DDPM-style diffusion schedule."""

    def __init__(self, T: int = 1000) -> None:
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion process."""
        a_bar = self.alpha_bar[t].view(-1, 1)
        return (a_bar.sqrt() * x0) + ((1 - a_bar).sqrt() * noise)
