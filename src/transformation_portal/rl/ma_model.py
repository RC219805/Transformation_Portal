"""Multi-agent models: Per-agent Q-networks.

This module provides Q-network architectures for the multi-agent
RL optimizer where each node has its own policy network.
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
            raise ImportError("PyTorch required for RL models")
    return _torch, _nn, _F


class AgentNet:
    """Actor-Critic network for a single agent.

    Used with advantage actor-critic (A2C) style training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize agent network.

        Args:
            state_dim: State vector dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: Any) -> tuple[Any, Any]:
        """Forward pass.

        Args:
            x: State tensor

        Returns:
            Tuple of (action_logits, state_value)
        """
        torch, _, _ = _get_torch()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        h = self.shared(x)
        return self.policy(h), self.value(h)

    def act(self, x: Any, deterministic: bool = False) -> tuple[int, Any, Any]:
        """Select action from state.

        Args:
            x: State vector
            deterministic: If True, select argmax

        Returns:
            Tuple of (action_index, log_prob, value)
        """
        torch, _, F = _get_torch()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs[0, action])
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def parameters(self):
        """Get all parameters."""
        torch, nn, _ = _get_torch()

        params = []
        for module in [self.shared, self.policy, self.value]:
            params.extend(module.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "shared": self.shared.state_dict(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.shared.load_state_dict(state_dict["shared"])
        self.policy.load_state_dict(state_dict["policy"])
        self.value.load_state_dict(state_dict["value"])


class AgentQNet:
    """Q-Network for QMIX-style training.

    Outputs Q-values for all actions given state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize Q-network.

        Args:
            state_dim: State vector dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s: Any) -> Any:
        """Forward pass.

        Args:
            s: State tensor

        Returns:
            Q-values tensor [batch, actions]
        """
        torch, _, _ = _get_torch()

        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)

        return self.net(s)

    def select_action(self, s: Any, eps: float = 0.05) -> int:
        """Select action with epsilon-greedy.

        Args:
            s: State vector
            eps: Exploration probability

        Returns:
            Action index
        """
        torch, _, _ = _get_torch()
        import random

        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)

        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)

        if s.dim() == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            q = self.forward(s)
            return int(q.argmax(dim=-1).item())

    def parameters(self):
        """Get parameters."""
        return self.net.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {"net": self.net.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.net.load_state_dict(state_dict["net"])


def create_agent(
    state_dim: int,
    action_dim: int,
    model_type: str = "actor_critic",
    **kwargs: Any,
) -> AgentNet | AgentQNet:
    """Factory function to create agent network.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        model_type: "actor_critic" or "qnet"
        **kwargs: Additional arguments

    Returns:
        Agent network instance
    """
    if model_type == "qnet":
        return AgentQNet(state_dim, action_dim, **kwargs)
    return AgentNet(state_dim, action_dim, **kwargs)
