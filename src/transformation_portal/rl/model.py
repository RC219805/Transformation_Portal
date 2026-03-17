"""RL model: Policy and value networks for pipeline optimization.

This module provides neural network architectures for the RL optimizer:
- PolicyValueNet: Combined actor-critic network
- PolicyNet: Standalone policy network
- ValueNet: Standalone value network
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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


class PolicyValueNet:
    """Combined actor-critic network for RL optimization.

    Architecture:
        Input -> Shared MLP -> Policy Head (action logits)
                           -> Value Head (state value)

    Example:
        >>> model = PolicyValueNet(state_dim=100, action_dim=50)
        >>> state = torch.randn(1, 100)
        >>> action, log_prob, value = model.act(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialize network.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build shared layers
        layers = []
        in_dim = state_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Policy head (action logits)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head (state value)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Move to device
        self.device = torch.device("cpu")
        self._to_device()

    def _to_device(self) -> None:
        """Move model to device."""
        torch, _, _ = _get_torch()

        self.shared = self.shared.to(self.device)
        self.policy_head = self.policy_head.to(self.device)
        self.value_head = self.value_head.to(self.device)

    def to(self, device: str) -> "PolicyValueNet":
        """Move model to device.

        Args:
            device: Device name ('cpu', 'cuda', 'mps')

        Returns:
            self
        """
        torch, _, _ = _get_torch()
        self.device = torch.device(device)
        self._to_device()
        return self

    def forward(self, x: Any) -> tuple[Any, Any]:
        """Forward pass.

        Args:
            x: State tensor [batch, state_dim]

        Returns:
            Tuple of (action_logits, state_value)
        """
        torch, _, _ = _get_torch()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        # Shared features
        h = self.shared(x)

        # Heads
        logits = self.policy_head(h)
        value = self.value_head(h)

        return logits, value

    def act(
        self,
        x: Any,
        deterministic: bool = False,
    ) -> tuple[int, Any, Any]:
        """Select action from state.

        Args:
            x: State vector or tensor
            deterministic: If True, select argmax action

        Returns:
            Tuple of (action_index, log_probability, state_value)
        """
        torch, _, F = _get_torch()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        logits, value = self.forward(x)

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs[0, action])
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def get_action_probs(self, x: Any) -> Any:
        """Get action probabilities.

        Args:
            x: State vector or tensor

        Returns:
            Action probability tensor
        """
        torch, _, F = _get_torch()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)

    def evaluate_actions(
        self,
        states: Any,
        actions: Any,
    ) -> tuple[Any, Any, Any]:
        """Evaluate actions for given states (for PPO).

        Args:
            states: Batch of states
            actions: Batch of action indices

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        torch, _, F = _get_torch()

        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def parameters(self):
        """Get all parameters for optimizer."""
        torch, nn, _ = _get_torch()

        params = []
        for module in [self.shared, self.policy_head, self.value_head]:
            params.extend(module.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for saving."""
        return {
            "shared": self.shared.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.shared.load_state_dict(state_dict["shared"])
        self.policy_head.load_state_dict(state_dict["policy_head"])
        self.value_head.load_state_dict(state_dict["value_head"])

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path
        """
        torch, _, _ = _get_torch()
        torch.save(
            {
                "state_dict": self.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "PolicyValueNet":
        """Load model from file.

        Args:
            path: File path

        Returns:
            Loaded model
        """
        torch, _, _ = _get_torch()

        data = torch.load(path, weights_only=True)
        model = cls(
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
        )
        model.load_state_dict(data["state_dict"])
        logger.info("Loaded model from %s", path)
        return model


def create_model(
    state_dim: int,
    action_dim: int,
    **kwargs: Any,
) -> PolicyValueNet:
    """Factory function to create a model.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Additional model arguments

    Returns:
        PolicyValueNet instance
    """
    return PolicyValueNet(state_dim, action_dim, **kwargs)
