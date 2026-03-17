"""World model for pipeline outcome prediction.

This module provides a learned world model that predicts pipeline
outcomes without executing the actual pipeline, enabling:
- Fast candidate evaluation (orders of magnitude cheaper)
- Lookahead planning for RL / beam search
- Pre-filtering bad actions before GPU use

Architecture:
    (state, action) -> WorldModel -> (predicted_next_state, predicted_metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
            raise ImportError("PyTorch required for world model")
    return _torch, _nn, _F


@dataclass
class WorldModelConfig:
    """Configuration for world model."""

    state_dim: int = 105
    action_dim: int = 50
    hidden_dim: int = 256
    action_embed_dim: int = 32
    num_metrics: int = 4  # score, psnr, lpips, llava
    dropout: float = 0.1


@dataclass
class PredictionResult:
    """Result of world model prediction."""

    next_state: Any  # Predicted next state vector
    metrics: Any  # Predicted metrics [score, psnr, lpips, llava]
    score: float = 0.0
    psnr: float = 0.0
    lpips: float = 0.0
    llava_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to metrics dict."""
        return {
            "score": self.score,
            "psnr": self.psnr,
            "lpips": self.lpips,
            "llava_score": self.llava_score,
        }


class WorldModel:
    """Neural network world model for pipeline prediction.

    Predicts:
    - Next state (encoded pipeline + metrics + diff)
    - Metrics (score, PSNR, LPIPS, LLaVA score)

    Given:
    - Current state
    - Action to take

    Example:
        >>> model = WorldModel(state_dim=105, action_dim=50)
        >>> next_state, metrics = model.forward(state, action)
    """

    def __init__(
        self,
        state_dim: int = 105,
        action_dim: int = 50,
        hidden_dim: int = 256,
        action_embed_dim: int = 32,
        num_metrics: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize world model.

        Args:
            state_dim: State vector dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            action_embed_dim: Action embedding dimension
            num_metrics: Number of metrics to predict
            dropout: Dropout probability
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_metrics = num_metrics

        # Action embedding
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Next state prediction head
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Metrics prediction head
        self.metrics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_metrics),
        )

    def forward(self, state: Any, action: Any) -> tuple[Any, Any]:
        """Forward pass.

        Args:
            state: Current state [batch, state_dim]
            action: Action indices [batch]

        Returns:
            Tuple of (predicted_next_state, predicted_metrics)
        """
        torch, _, _ = _get_torch()

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        # Embed action
        action_emb = self.action_embed(action)  # [batch, action_embed_dim]

        # Concatenate state and action embedding
        x = torch.cat([state, action_emb], dim=-1)

        # Encode
        h = self.encoder(x)

        # Predict
        next_state = self.next_state_head(h)
        metrics = self.metrics_head(h)

        return next_state, metrics

    def predict(self, state: Any, action: int) -> PredictionResult:
        """Predict next state and metrics (inference mode).

        Args:
            state: Current state vector
            action: Action index

        Returns:
            PredictionResult with predictions
        """
        torch, _, _ = _get_torch()

        with torch.no_grad():
            next_state, metrics = self.forward(state, action)

        # Extract scalar values
        metrics_np = metrics[0].cpu().numpy()

        return PredictionResult(
            next_state=next_state[0],
            metrics=metrics[0],
            score=float(metrics_np[0]),
            psnr=float(metrics_np[1]) if len(metrics_np) > 1 else 0.0,
            lpips=float(metrics_np[2]) if len(metrics_np) > 2 else 0.0,
            llava_score=float(metrics_np[3]) if len(metrics_np) > 3 else 0.0,
        )

    def simulate_rollout(
        self,
        state: Any,
        actions: list[int],
    ) -> tuple[float, list[PredictionResult]]:
        """Simulate multiple steps.

        Args:
            state: Initial state
            actions: List of actions to take

        Returns:
            Tuple of (total_score, list of predictions)
        """
        torch, _, _ = _get_torch()

        current_state = state
        total_score = 0.0
        predictions = []

        with torch.no_grad():
            for action in actions:
                pred = self.predict(current_state, action)
                predictions.append(pred)
                total_score += pred.score
                current_state = pred.next_state

        return total_score, predictions

    def parameters(self):
        """Get model parameters."""
        params = []
        params.extend(self.action_embed.parameters())
        params.extend(self.encoder.parameters())
        params.extend(self.next_state_head.parameters())
        params.extend(self.metrics_head.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "action_embed": self.action_embed.state_dict(),
            "encoder": self.encoder.state_dict(),
            "next_state_head": self.next_state_head.state_dict(),
            "metrics_head": self.metrics_head.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.action_embed.load_state_dict(state_dict["action_embed"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.next_state_head.load_state_dict(state_dict["next_state_head"])
        self.metrics_head.load_state_dict(state_dict["metrics_head"])

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        torch, _, _ = _get_torch()
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_metrics": self.num_metrics,
                },
            },
            path,
        )
        logger.info("Saved world model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "WorldModel":
        """Load model from file."""
        torch, _, _ = _get_torch()
        data = torch.load(path, weights_only=True)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        logger.info("Loaded world model from %s", path)
        return model


class EnsembleWorldModel:
    """Ensemble of world models for uncertainty estimation.

    Trains multiple models and uses disagreement to estimate
    prediction uncertainty. High uncertainty -> run real pipeline.

    Example:
        >>> ensemble = EnsembleWorldModel(n_models=5)
        >>> mean_score, std = ensemble.predict_with_uncertainty(state, action)
        >>> if std > threshold:
        ...     run_real_pipeline()
    """

    def __init__(
        self,
        n_models: int = 5,
        **model_kwargs: Any,
    ) -> None:
        """Initialize ensemble.

        Args:
            n_models: Number of models in ensemble
            **model_kwargs: Arguments for WorldModel
        """
        self.n_models = n_models
        self.models = [WorldModel(**model_kwargs) for _ in range(n_models)]

    def forward(self, state: Any, action: Any) -> tuple[Any, Any, Any, Any]:
        """Forward pass through all models.

        Returns:
            Tuple of (mean_next_state, mean_metrics, std_next_state, std_metrics)
        """
        torch, _, _ = _get_torch()

        next_states = []
        all_metrics = []

        for model in self.models:
            ns, m = model.forward(state, action)
            next_states.append(ns)
            all_metrics.append(m)

        next_states = torch.stack(next_states)
        all_metrics = torch.stack(all_metrics)

        return (
            next_states.mean(dim=0),
            all_metrics.mean(dim=0),
            next_states.std(dim=0),
            all_metrics.std(dim=0),
        )

    def predict_with_uncertainty(
        self,
        state: Any,
        action: int,
    ) -> tuple[float, float]:
        """Predict with uncertainty estimate.

        Args:
            state: Current state
            action: Action index

        Returns:
            Tuple of (mean_score, score_std)
        """
        torch, _, _ = _get_torch()

        with torch.no_grad():
            _, mean_metrics, _, std_metrics = self.forward(state, action)

        mean_score = float(mean_metrics[0, 0].item())
        std_score = float(std_metrics[0, 0].item())

        return mean_score, std_score

    def parameters(self):
        """Get all parameters."""
        params = []
        for model in self.models:
            params.extend(model.parameters())
        return params


def create_world_model(
    state_dim: int = 105,
    action_dim: int = 50,
    ensemble: bool = False,
    **kwargs: Any,
) -> WorldModel | EnsembleWorldModel:
    """Factory function to create world model.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        ensemble: Whether to create ensemble
        **kwargs: Additional arguments

    Returns:
        WorldModel or EnsembleWorldModel
    """
    if ensemble:
        return EnsembleWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs,
        )
    return WorldModel(state_dim=state_dim, action_dim=action_dim, **kwargs)
