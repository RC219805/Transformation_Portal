"""MuZero-style latent planner for pipeline optimization.

This module implements a MuZero-style architecture adapted for pipeline
optimization, replacing explicit state with learned latent dynamics.

Components:
    - RepresentationNet: obs -> latent state
    - DynamicsNet: (state, action) -> (next_state, reward)
    - PredictionNet: state -> (policy, value)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNet(nn.Module):
    """Project observation to latent state.

    Maps raw pipeline observations (metrics, diff features, config)
    to a compact latent representation.
    """

    def __init__(self, obs_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.net(obs)


class DynamicsNet(nn.Module):
    """Predict next latent state and reward from (state, action).

    Learns the transition dynamics in latent space, avoiding
    the need for explicit pipeline state materialization.
    """

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, 32)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.state_head = nn.Linear(256, latent_dim)
        self.reward_head = nn.Linear(256, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward."""
        a_emb = self.action_emb(a)
        x = torch.cat([s, a_emb], dim=-1)
        h = self.net(x)

        next_s = self.state_head(h)
        reward = self.reward_head(h)

        return next_s, reward


class PredictionNet(nn.Module):
    """Predict policy and value from latent state.

    Used for both action selection during MCTS and
    value estimation for backup.
    """

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy distribution and value."""
        h = self.net(s)
        p = F.softmax(self.policy_head(h), dim=-1)
        v = self.value_head(h)
        return p, v


class MuZeroModel(nn.Module):
    """Complete MuZero model for pipeline optimization.

    Combines representation, dynamics, and prediction networks
    for latent-space planning with MCTS.

    Example:
        >>> model = MuZeroModel(obs_dim=64, action_dim=10)
        >>> obs = torch.randn(1, 64)
        >>> s, p, v = model.initial_inference(obs)
        >>> s2, p2, v2, r = model.recurrent_inference(s, torch.tensor([3]))
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.repr = RepresentationNet(obs_dim, latent_dim)
        self.dyn = DynamicsNet(latent_dim, action_dim)
        self.pred = PredictionNet(latent_dim, action_dim)

    def initial_inference(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation and predict policy/value.

        Used at the root of MCTS search.

        Returns:
            Tuple of (latent_state, policy, value)
        """
        s = self.repr(obs)
        p, v = self.pred(s)
        return s, p, v

    def recurrent_inference(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state, policy, value, and reward.

        Used during MCTS simulation steps.

        Returns:
            Tuple of (next_state, policy, value, reward)
        """
        s2, r = self.dyn(s, a)
        p, v = self.pred(s2)
        return s2, p, v, r

    def get_action_dim(self) -> int:
        """Return the action dimension."""
        return self.action_dim
