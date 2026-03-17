"""MuZero trainer for learning latent dynamics.

This module provides training utilities for the MuZero model,
including loss computation and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.optim as optim

if TYPE_CHECKING:
    from transformation_portal.rl.muzero_model import MuZeroModel


@dataclass
class MuZeroTrainConfig:
    """Configuration for MuZero training."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    unroll_steps: int = 5
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    epochs: int = 10
    batch_size: int = 128


class MuZeroTrainer:
    """Trainer for MuZero models.

    Implements the MuZero training loop with:
    - Initial inference loss
    - Unrolled recurrent inference loss
    - Policy, value, and reward predictions

    Example:
        >>> trainer = MuZeroTrainer(model)
        >>> loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: "MuZeroModel",
        config: MuZeroTrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or MuZeroTrainConfig()

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        target_rewards: torch.Tensor,
        target_policies: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute MuZero losses.

        Args:
            obs: Observations [B, obs_dim]
            actions: Action sequences [B, T]
            target_values: Target values [B, T+1]
            target_rewards: Target rewards [B, T]
            target_policies: Target policy distributions [B, T+1, A]

        Returns:
            Dictionary of loss components
        """
        batch_size = obs.shape[0]
        unroll_steps = actions.shape[1]

        # Initial inference
        s, p, v = self.model.initial_inference(obs)

        # Initial losses
        value_loss = F.mse_loss(v.squeeze(-1), target_values[:, 0])
        policy_loss = F.cross_entropy(
            p.log(), target_policies[:, 0], reduction="mean"
        )
        reward_loss = torch.tensor(0.0, device=obs.device)

        # Unroll dynamics
        for t in range(unroll_steps):
            s, p, v, r = self.model.recurrent_inference(s, actions[:, t])

            # Accumulate losses
            value_loss = value_loss + F.mse_loss(
                v.squeeze(-1), target_values[:, t + 1]
            )
            reward_loss = reward_loss + F.mse_loss(
                r.squeeze(-1), target_rewards[:, t]
            )
            policy_loss = policy_loss + F.cross_entropy(
                p.log(), target_policies[:, t + 1], reduction="mean"
            )

        # Average over unroll steps
        n_steps = unroll_steps + 1
        value_loss = value_loss / n_steps
        policy_loss = policy_loss / n_steps
        reward_loss = reward_loss / max(unroll_steps, 1)

        # Total loss
        total_loss = (
            self.config.value_loss_weight * value_loss
            + self.config.reward_loss_weight * reward_loss
            + self.config.policy_loss_weight * policy_loss
        )

        return {
            "total": total_loss,
            "value": value_loss,
            "reward": reward_loss,
            "policy": policy_loss,
        }

    def train_step(self, batch: dict) -> dict[str, float]:
        """Execute single training step.

        Args:
            batch: Dictionary containing:
                - obs: [B, obs_dim]
                - actions: [B, T]
                - values: [B, T+1]
                - rewards: [B, T]
                - policies: [B, T+1, A]

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        losses = self.compute_loss(
            obs=batch["obs"],
            actions=batch["actions"],
            target_values=batch["values"],
            target_rewards=batch["rewards"],
            target_policies=batch["policies"],
        )

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}


def train_muzero(
    model: "MuZeroModel",
    dataset,
    config: MuZeroTrainConfig | None = None,
) -> list[dict[str, float]]:
    """Train MuZero model on dataset.

    Args:
        model: MuZero model to train
        dataset: Dataset providing batches
        config: Training configuration

    Returns:
        List of loss dictionaries per epoch
    """
    config = config or MuZeroTrainConfig()
    trainer = MuZeroTrainer(model, config)
    history = []

    for epoch in range(config.epochs):
        epoch_losses = []

        for batch in dataset.batches(config.batch_size):
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)

        # Average epoch losses
        avg_losses = {
            k: sum(l[k] for l in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0]
        }

        history.append(avg_losses)
        print(
            f"[MuZero] epoch {epoch} "
            f"loss={avg_losses['total']:.4f} "
            f"value={avg_losses['value']:.4f} "
            f"reward={avg_losses['reward']:.4f} "
            f"policy={avg_losses['policy']:.4f}"
        )

    return history
