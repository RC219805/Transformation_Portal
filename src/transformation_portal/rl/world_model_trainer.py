"""World model trainer for learning pipeline dynamics.

This module provides training utilities for the world model,
including dataset management and training loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Lazy imports
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
class WorldModelTransition:
    """A single transition for world model training."""

    state: Any  # Current state vector
    action: int  # Action index
    next_state: Any  # Next state vector
    metrics: list[float]  # [score, psnr, lpips, llava]


@dataclass
class WorldModelDataset:
    """Dataset for world model training."""

    transitions: list[WorldModelTransition] = field(default_factory=list)
    max_size: int = 100000

    def add(self, transition: WorldModelTransition) -> None:
        """Add transition to dataset."""
        self.transitions.append(transition)
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def __len__(self) -> int:
        return len(self.transitions)

    def batches(self, batch_size: int) -> Iterator[dict[str, Any]]:
        """Iterate over batches.

        Yields:
            Batches as dicts with state, action, next_state, metrics tensors
        """
        torch = _get_torch()
        import random

        indices = list(range(len(self.transitions)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            states = []
            actions = []
            next_states = []
            metrics = []

            for idx in batch_indices:
                t = self.transitions[idx]

                s = t.state
                if hasattr(s, "tolist"):
                    s = s.tolist()
                states.append(s)

                actions.append(t.action)

                ns = t.next_state
                if hasattr(ns, "tolist"):
                    ns = ns.tolist()
                next_states.append(ns)

                metrics.append(t.metrics)

            yield {
                "state": torch.tensor(states, dtype=torch.float32),
                "action": torch.tensor(actions, dtype=torch.long),
                "next_state": torch.tensor(next_states, dtype=torch.float32),
                "metrics": torch.tensor(metrics, dtype=torch.float32),
            }

    def save(self, path: str | Path) -> None:
        """Save dataset to file."""
        import json

        data = []
        for t in self.transitions:
            state = t.state.tolist() if hasattr(t.state, "tolist") else t.state
            next_state = (
                t.next_state.tolist()
                if hasattr(t.next_state, "tolist")
                else t.next_state
            )
            data.append(
                {
                    "state": state,
                    "action": t.action,
                    "next_state": next_state,
                    "metrics": t.metrics,
                }
            )

        Path(path).write_text(json.dumps(data))
        logger.info("Saved %d transitions to %s", len(data), path)

    @classmethod
    def load(cls, path: str | Path) -> "WorldModelDataset":
        """Load dataset from file."""
        import json

        data = json.loads(Path(path).read_text())
        dataset = cls()

        for item in data:
            dataset.add(
                WorldModelTransition(
                    state=item["state"],
                    action=item["action"],
                    next_state=item["next_state"],
                    metrics=item["metrics"],
                )
            )

        logger.info("Loaded %d transitions from %s", len(dataset), path)
        return dataset


@dataclass
class TrainingConfig:
    """Configuration for world model training."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    state_loss_weight: float = 1.0
    metrics_loss_weight: float = 1.0
    log_interval: int = 10


@dataclass
class TrainingResult:
    """Result of training run."""

    epochs: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)


def train_world_model(
    model: Any,
    dataset: WorldModelDataset,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train world model on dataset.

    Args:
        model: WorldModel instance
        dataset: Training dataset
        config: Training configuration

    Returns:
        TrainingResult with training history
    """
    torch = _get_torch()
    import torch.optim as optim

    config = config or TrainingConfig()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_history = []

    logger.info(
        "Training world model: %d epochs, %d samples",
        config.epochs,
        len(dataset),
    )

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataset.batches(config.batch_size):
            s = batch["state"]
            a = batch["action"]
            s2_target = batch["next_state"]
            m_target = batch["metrics"]

            # Forward
            next_state_pred, metrics_pred = model.forward(s, a)

            # Loss
            state_loss = (next_state_pred - s2_target).pow(2).mean()
            metrics_loss = (metrics_pred - m_target).pow(2).mean()

            loss = (
                config.state_loss_weight * state_loss
                + config.metrics_loss_weight * metrics_loss
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if epoch % config.log_interval == 0 or epoch == config.epochs - 1:
            logger.info(
                "[WorldModel] Epoch %d/%d, loss=%.4f",
                epoch + 1,
                config.epochs,
                avg_loss,
            )

    return TrainingResult(
        epochs=config.epochs,
        final_loss=loss_history[-1] if loss_history else 0.0,
        loss_history=loss_history,
    )


def train_ensemble(
    ensemble: Any,
    dataset: WorldModelDataset,
    config: TrainingConfig | None = None,
) -> list[TrainingResult]:
    """Train ensemble of world models.

    Each model is trained on a bootstrap sample of the dataset.

    Args:
        ensemble: EnsembleWorldModel instance
        dataset: Training dataset
        config: Training configuration

    Returns:
        List of TrainingResults for each model
    """
    import random

    config = config or TrainingConfig()
    results = []

    for i, model in enumerate(ensemble.models):
        logger.info("Training ensemble model %d/%d", i + 1, len(ensemble.models))

        # Bootstrap sample
        bootstrap_dataset = WorldModelDataset()
        for _ in range(len(dataset)):
            t = random.choice(dataset.transitions)
            bootstrap_dataset.add(t)

        result = train_world_model(model, bootstrap_dataset, config)
        results.append(result)

    return results
