"""QMIX/VDN hybrid centralized critic.

This module provides the mixing network for multi-agent RL:
- QMIX: Hypernetwork-based mixing with monotonicity constraints
- VDN: Simple value decomposition (sum)
- Hybrid: Weighted combination of both
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy torch import
_torch = None
_nn = None
_F = None


def _get_torch():
    """Lazy import torch."""
    global _torch, _nn, _F
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            _torch = torch
            _nn = nn
            _F = F
        except ImportError:
            raise ImportError("PyTorch required for QMIX")
    return _torch, _nn, _F


class MixingNetwork:
    """QMIX mixing network with monotonicity via positive weights.

    Takes per-agent Q-values and global state, outputs Q_total.
    Uses hypernetworks to produce weights conditioned on global state.

    Architecture:
        agent_qs [B, n_agents] -> W1 (from hyper) -> hidden -> W2 -> Q_total [B, 1]
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 64,
    ) -> None:
        """Initialize mixing network.

        Args:
            n_agents: Number of agents
            state_dim: Global state dimension
            embed_dim: Embedding/hidden dimension
        """
        torch, nn, F = _get_torch()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetworks produce weights conditioned on global state
        # W1: [state_dim] -> [n_agents * embed_dim]
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )

        # b1: [state_dim] -> [embed_dim]
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
        )

        # W2: [state_dim] -> [embed_dim]
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # b2: [state_dim] -> [1]
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: Any, global_state: Any) -> Any:
        """Mix agent Q-values into Q_total.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            global_state: Global state [batch, state_dim]

        Returns:
            Q_total [batch, 1]
        """
        torch, _, F = _get_torch()

        B = agent_qs.size(0)

        # First layer weights (positive for monotonicity)
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)

        # agent_qs: [B, n_agents] -> [B, 1, n_agents]
        # bmm: [B, 1, n_agents] @ [B, n_agents, embed] = [B, 1, embed]
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)

        # [B, 1, embed] @ [B, embed, 1] = [B, 1, 1]
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.view(B, 1)

    def parameters(self):
        """Get all parameters."""
        params = []
        for module in [self.hyper_w1, self.hyper_b1, self.hyper_w2, self.hyper_b2]:
            params.extend(module.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "hyper_w1": self.hyper_w1.state_dict(),
            "hyper_b1": self.hyper_b1.state_dict(),
            "hyper_w2": self.hyper_w2.state_dict(),
            "hyper_b2": self.hyper_b2.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.hyper_w1.load_state_dict(state_dict["hyper_w1"])
        self.hyper_b1.load_state_dict(state_dict["hyper_b1"])
        self.hyper_w2.load_state_dict(state_dict["hyper_w2"])
        self.hyper_b2.load_state_dict(state_dict["hyper_b2"])


class CentralCritic:
    """Hybrid QMIX/VDN centralized critic.

    Combines:
    - QMIX: Hypernetwork-based mixing (captures complex coordination)
    - VDN: Simple sum (fast, stable baseline)

    Q_total = alpha * QMIX(q_i, s) + (1-alpha) * sum(q_i)

    Attributes:
        mixer: QMIX mixing network
        alpha: Blend factor (1.0 = pure QMIX, 0.0 = pure VDN)
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        alpha: float = 0.7,
        embed_dim: int = 64,
    ) -> None:
        """Initialize hybrid critic.

        Args:
            n_agents: Number of agents
            state_dim: Global state dimension
            alpha: QMIX weight (0-1)
            embed_dim: Mixing network embed dim
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.alpha = alpha

        self.mixer = MixingNetwork(n_agents, state_dim, embed_dim)

    def forward(self, agent_qs: Any, global_state: Any) -> Any:
        """Compute Q_total from agent Q-values.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            global_state: Global state [batch, state_dim]

        Returns:
            Q_total [batch, 1]
        """
        torch, _, _ = _get_torch()

        # QMIX component
        qmix = self.mixer(agent_qs, global_state)

        # VDN component (simple sum)
        vdn = agent_qs.sum(dim=1, keepdim=True)

        # Blend
        return self.alpha * qmix + (1.0 - self.alpha) * vdn

    def parameters(self):
        """Get parameters."""
        return self.mixer.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "mixer": self.mixer.state_dict(),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.mixer.load_state_dict(state_dict["mixer"])
        self.alpha = state_dict.get("alpha", self.alpha)


def create_critic(
    n_agents: int,
    state_dim: int,
    critic_type: str = "qmix",
    **kwargs: Any,
) -> CentralCritic:
    """Factory function to create critic.

    Args:
        n_agents: Number of agents
        state_dim: Global state dimension
        critic_type: "qmix", "vdn", or "hybrid"
        **kwargs: Additional arguments

    Returns:
        CentralCritic instance
    """
    if critic_type == "vdn":
        return CentralCritic(n_agents, state_dim, alpha=0.0, **kwargs)
    elif critic_type == "qmix":
        return CentralCritic(n_agents, state_dim, alpha=1.0, **kwargs)
    else:
        return CentralCritic(n_agents, state_dim, **kwargs)
