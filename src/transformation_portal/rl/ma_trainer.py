"""Multi-agent trainer: Decentralized A2C with centralized critic.

This module provides training algorithms for multi-agent RL:
- Per-agent actor updates (decentralized)
- Optional centralized critic (QMIX/VDN)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy torch import
_torch = None


def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            raise ImportError("PyTorch required for training")
    return _torch


@dataclass
class MATrainerConfig:
    """Configuration for multi-agent trainer."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 32
    use_central_critic: bool = True


class AgentTrainer:
    """Trainer for a single agent (actor-critic).

    Used in decentralized training where each agent has its own trainer.
    """

    def __init__(
        self,
        model: Any,
        actions: list[Any],
        config: MATrainerConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Agent model (AgentNet)
            actions: List of available actions
            config: Trainer configuration
        """
        torch = _get_torch()

        self.model = model
        self.actions = actions
        self.config = config or MATrainerConfig()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
        )

    def step(self, batch: list[tuple]) -> float:
        """Training step on batch.

        Args:
            batch: List of (state, action, reward, next_state, done, log_prob, value)

        Returns:
            Loss value
        """
        torch = _get_torch()
        import torch.nn.functional as F

        loss = torch.tensor(0.0)

        for s, a, r, s2, done, logp, v in batch:
            # Compute target
            target = r
            if not done:
                with torch.no_grad():
                    _, v2 = self.model.forward(torch.tensor(s2).unsqueeze(0))
                    target = r + self.config.gamma * v2.item()

            advantage = target - v.item()

            # Policy loss
            loss = loss + (-logp * advantage)

            # Value loss
            loss = loss + self.config.value_coef * F.mse_loss(v, torch.tensor([[target]]))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return float(loss.item())


class CentralizedTrainer:
    """Trainer with centralized critic (QMIX-style).

    Updates all agent networks + central critic together.
    """

    def __init__(
        self,
        agents: dict[str, Any],
        critic: Any,
        config: MATrainerConfig | None = None,
    ) -> None:
        """Initialize centralized trainer.

        Args:
            agents: Dictionary of node_id -> agent model
            critic: Central critic (CentralCritic)
            config: Trainer configuration
        """
        torch = _get_torch()

        self.agents = agents
        self.critic = critic
        self.config = config or MATrainerConfig()

        # Collect all parameters
        params = list(critic.parameters())
        for agent in agents.values():
            params.extend(agent.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

    def get_weights(self) -> dict[str, Any]:
        """Get all model weights."""
        return {
            "agents": {k: v.state_dict() for k, v in self.agents.items()},
            "critic": self.critic.state_dict(),
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        """Set all model weights."""
        for k, v in self.agents.items():
            if k in weights["agents"]:
                v.load_state_dict(weights["agents"][k])
        self.critic.load_state_dict(weights["critic"])

    def step(self, batch: list[dict[str, Any]]) -> float:
        """Training step on batch.

        Args:
            batch: List of transition dicts with:
                - states: dict[node_id -> state]
                - actions: dict[node_id -> action_idx]
                - reward: float
                - next_states: dict[node_id -> state]
                - global_state: array
                - done: bool

        Returns:
            Loss value
        """
        torch = _get_torch()
        import torch.nn.functional as F

        total_loss = torch.tensor(0.0)

        for item in batch:
            states = item["states"]
            actions = item["actions"]
            reward = item["reward"]
            global_state = item["global_state"]

            # Get Q-values from each agent
            q_values = []
            for node_id, agent in self.agents.items():
                if node_id not in states:
                    continue

                s = torch.tensor(states[node_id], dtype=torch.float32).unsqueeze(0)

                # For Q-nets, get Q(s, a)
                if hasattr(agent, "net"):
                    q = agent.forward(s)
                    a = actions.get(node_id, 0)
                    q_values.append(q[0, a])
                else:
                    # For actor-critic, use value
                    _, v = agent.forward(s)
                    q_values.append(v.squeeze())

            if not q_values:
                continue

            # Stack Q-values
            agent_qs = torch.stack(q_values).unsqueeze(0)  # [1, n_agents]

            # Global state
            gs = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)

            # Get Q_total from critic
            q_total = self.critic.forward(agent_qs, gs)

            # Target (simplified - could use target networks)
            target = torch.tensor([[reward]], dtype=torch.float32)

            # Loss
            loss = F.mse_loss(q_total, target)
            total_loss = total_loss + loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + [p for a in self.agents.values() for p in a.parameters()],
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return float(total_loss.item())
