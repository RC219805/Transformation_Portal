"""RL trainer: A2C + replay training algorithm.

This module provides the training algorithm for the RL optimizer:
- A2C (Advantage Actor-Critic) with off-policy replay
- Policy gradient updates
- Value function regression
- Entropy bonus for exploration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformation_portal.rl.action_space import RLAction
from transformation_portal.rl.replay import ReplayBuffer, Transition

if TYPE_CHECKING:
    from transformation_portal.rl.model import PolicyValueNet

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
            raise ImportError("PyTorch required for RL training")
    return _torch


@dataclass
class TrainerConfig:
    """Configuration for RL trainer."""

    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    batch_size: int = 32
    update_frequency: int = 4  # Steps between updates


class RLTrainer:
    """A2C trainer with experience replay.

    Example:
        >>> trainer = RLTrainer(model, action_list)
        >>> action_idx, logp, value = trainer.select_action(state)
        >>> trainer.store_transition(s, a, r, s2, done, logp, value)
        >>> loss = trainer.update()
    """

    def __init__(
        self,
        model: "PolicyValueNet",
        action_list: list[RLAction],
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PolicyValueNet instance
            action_list: List of available actions
            config: Trainer configuration
        """
        torch = _get_torch()

        self.model = model
        self.actions = action_list
        self.config = config or TrainerConfig()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=50000)

        # Training stats
        self.total_steps = 0
        self.total_updates = 0

    def select_action(
        self,
        state_vec: Any,
        deterministic: bool = False,
    ) -> tuple[int, Any, Any]:
        """Select action from state.

        Args:
            state_vec: Encoded state vector
            deterministic: If True, select best action

        Returns:
            Tuple of (action_index, log_probability, value)
        """
        torch = _get_torch()

        state = torch.tensor(state_vec, dtype=torch.float32)
        action_idx, logp, value = self.model.act(state, deterministic=deterministic)

        return action_idx, logp, value

    def get_action(self, action_idx: int) -> RLAction:
        """Get action by index.

        Args:
            action_idx: Action index

        Returns:
            RLAction instance
        """
        return self.actions[action_idx]

    def store_transition(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        log_prob: Any,
        value: Any,
    ) -> None:
        """Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            log_prob: Log probability of action
            value: Value estimate
        """
        self.buffer.add(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
            )
        )
        self.total_steps += 1

    def compute_loss(
        self,
        batch: list[Transition],
    ) -> tuple[Any, dict[str, float]]:
        """Compute A2C loss from batch.

        Args:
            batch: List of transitions

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        torch = _get_torch()
        import torch.nn.functional as F

        policy_losses = []
        value_losses = []
        entropies = []

        for t in batch:
            # Get current value and action probs
            state = torch.tensor(t.state, dtype=torch.float32).unsqueeze(0)
            logits, value = self.model.forward(state)

            # Compute target value
            if t.done:
                target = t.reward
            else:
                with torch.no_grad():
                    next_state = torch.tensor(t.next_state, dtype=torch.float32).unsqueeze(0)
                    _, next_value = self.model.forward(next_state)
                    target = t.reward + self.config.gamma * next_value.item()

            # Advantage
            advantage = target - value.item()

            # Policy loss (negative because we maximize expected return)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor([t.action]))
            policy_losses.append(-log_prob * advantage)

            # Value loss
            value_losses.append(F.mse_loss(value, torch.tensor([[target]])))

            # Entropy bonus
            entropies.append(dist.entropy())

        # Combine losses
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()

        total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        return total_loss, {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "total_loss": float(total_loss.item()),
        }

    def update(self) -> dict[str, float] | None:
        """Perform training update if ready.

        Returns:
            Loss dictionary or None if not enough samples
        """
        # Check if ready to update
        if len(self.buffer) < self.config.batch_size:
            return None

        if self.total_steps % self.config.update_frequency != 0:
            return None

        torch = _get_torch()

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)

        # Compute loss
        loss, loss_dict = self.compute_loss(batch)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()
        self.total_updates += 1

        logger.debug(
            "Update %d: loss=%.4f (policy=%.4f, value=%.4f, entropy=%.4f)",
            self.total_updates,
            loss_dict["total_loss"],
            loss_dict["policy_loss"],
            loss_dict["value_loss"],
            loss_dict["entropy"],
        )

        return loss_dict

    def step(self, batch: list[Transition]) -> float:
        """Single training step on provided batch.

        Args:
            batch: List of transitions

        Returns:
            Total loss value
        """
        torch = _get_torch()

        loss, _ = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.total_updates += 1

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save trainer state.

        Args:
            path: File path
        """
        torch = _get_torch()

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
            },
            path,
        )
        logger.info("Saved trainer to %s", path)

    def load(self, path: str) -> None:
        """Load trainer state.

        Args:
            path: File path
        """
        torch = _get_torch()

        data = torch.load(path, weights_only=True)  # nosec B614: trusted checkpoint
        self.model.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.total_steps = data["total_steps"]
        self.total_updates = data["total_updates"]
        logger.info("Loaded trainer from %s", path)
