"""Pipeline Foundation Model for generalized pipeline optimization.

This module implements a foundation model pretrained on pipeline execution
logs that can generalize across projects and domains. It combines:
- Self-supervised pretraining on historical pipeline data
- Transfer learning capabilities
- Few-shot adaptation to new pipelines

Architecture: Encoder-Decoder Transformer with:
- Pipeline state encoder (handles variable DAG structures)
- Execution trajectory modeling
- Multi-task prediction heads
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


@dataclass
class PipelineFoundationConfig:
    """Configuration for Pipeline Foundation Model."""

    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_nodes: int = 32
    max_seq_len: int = 256

    # Input dimensions
    node_state_dim: int = 64
    action_dim: int = 64
    metric_dim: int = 16

    # Pretraining
    mask_ratio: float = 0.15
    contrastive_temp: float = 0.07


class NodeTypeEmbedding(nn.Module):
    """Learnable embeddings for different node types."""

    def __init__(self, n_types: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_types, d_model)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_ids)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for graph + time dimensions."""

    def __init__(self, d_model: int, max_nodes: int, max_time: int) -> None:
        super().__init__()
        self.node_pos = nn.Embedding(max_nodes, d_model // 2)
        self.time_pos = nn.Embedding(max_time, d_model // 2)

    def forward(self, node_idx: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        """Combine node and time positional encodings.

        Args:
            node_idx: Node indices [B, T, N]
            time_idx: Time indices [B, T]

        Returns:
            Combined positional encoding [B, T, N, d_model]
        """
        node_emb = self.node_pos(node_idx)  # [B, T, N, d/2]
        time_emb = self.time_pos(time_idx)  # [B, T, d/2]

        # Broadcast time across nodes
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, node_idx.size(2), -1)

        return torch.cat([node_emb, time_emb], dim=-1)


class GraphEncoder(nn.Module):
    """Encodes DAG structure into fixed-size representation."""

    def __init__(self, config: PipelineFoundationConfig) -> None:
        super().__init__()
        self.config = config

        # Node state projection
        self.node_proj = nn.Linear(config.node_state_dim, config.d_model)

        # Graph attention layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )

        # Pooling for graph-level representation
        self.pool = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph to node and graph-level representations.

        Args:
            node_states: Node features [B, N, state_dim]
            adjacency: Optional adjacency matrix [B, N, N]

        Returns:
            Tuple of (node_embeddings [B, N, d_model], graph_embedding [B, d_model])
        """
        x = self.node_proj(node_states)

        # Create attention mask from adjacency if provided
        mask = None
        if adjacency is not None:
            # Invert: 0 = attend, 1 = mask
            mask = (1 - adjacency).bool()

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Graph-level pooling (attention-weighted mean)
        weights = F.softmax(self.pool(x).mean(dim=-1), dim=-1)
        graph_emb = (weights.unsqueeze(-1) * x).sum(dim=1)

        return x, graph_emb


class TrajectoryEncoder(nn.Module):
    """Encodes execution trajectories (state-action-reward sequences)."""

    def __init__(self, config: PipelineFoundationConfig) -> None:
        super().__init__()
        self.config = config

        # Token projections
        self.state_proj = nn.Linear(config.d_model, config.d_model)
        self.action_proj = nn.Linear(config.action_dim, config.d_model)
        self.reward_proj = nn.Linear(1, config.d_model)
        self.metric_proj = nn.Linear(config.metric_dim, config.d_model)

        # Positional encoding
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_encoder_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        graph_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        metrics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode trajectory sequence.

        Args:
            graph_states: Graph embeddings per step [B, T, d_model]
            actions: Action embeddings [B, T, action_dim]
            rewards: Rewards [B, T, 1]
            metrics: Optional metrics [B, T, metric_dim]

        Returns:
            Trajectory encoding [B, T, d_model]
        """
        b, t, _ = graph_states.shape

        # Project each modality
        state_emb = self.state_proj(graph_states)
        action_emb = self.action_proj(actions)
        reward_emb = self.reward_proj(rewards)

        # Combine
        x = state_emb + action_emb + reward_emb

        if metrics is not None:
            x = x + self.metric_proj(metrics)

        # Add positional encoding
        positions = torch.arange(t, device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)

        # Causal mask
        causal_mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()

        # Transform
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        return self.norm(x)


class PipelineFoundationModel(nn.Module):
    """Foundation Model for Pipeline Optimization.

    A pretrained model that can be fine-tuned for various pipeline
    optimization tasks. Supports:
    - Zero-shot transfer to new pipelines
    - Few-shot adaptation with minimal data
    - Multi-task learning (optimization, prediction, generation)

    Example:
        >>> config = PipelineFoundationConfig()
        >>> model = PipelineFoundationModel(config)
        >>> node_states = torch.randn(2, 5, 64)  # [B, N, state_dim]
        >>> actions = torch.randn(2, 10, 64)  # [B, T, action_dim]
        >>> rewards = torch.randn(2, 10, 1)
        >>> out = model(node_states, actions, rewards)
    """

    def __init__(self, config: PipelineFoundationConfig) -> None:
        super().__init__()
        self.config = config

        # Graph encoder
        self.graph_encoder = GraphEncoder(config)

        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(config)

        # Prediction heads
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        self.metric_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.metric_dim),
        )

        # Contrastive projection head (for pretraining)
        self.contrastive_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 128),
        )

    def encode_pipeline(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pipeline structure.

        Args:
            node_states: Node features [B, N, state_dim]
            adjacency: Optional adjacency matrix [B, N, N]

        Returns:
            Tuple of (node_embeddings, graph_embedding)
        """
        return self.graph_encoder(node_states, adjacency)

    def forward(
        self,
        node_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        metrics: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through foundation model.

        Args:
            node_states: Node features [B, N, state_dim] or [B, T, N, state_dim]
            actions: Actions [B, T, action_dim]
            rewards: Rewards [B, T, 1]
            metrics: Optional metrics [B, T, metric_dim]
            adjacency: Optional adjacency [B, N, N]

        Returns:
            Predictions dictionary
        """
        # Handle both single-step and sequence inputs
        if node_states.dim() == 3:
            # Single step: [B, N, state_dim]
            _, graph_emb = self.graph_encoder(node_states, adjacency)
            # Expand for trajectory
            t = actions.size(1)
            graph_emb = graph_emb.unsqueeze(1).expand(-1, t, -1)
        else:
            # Sequence: [B, T, N, state_dim]
            b, t, n, _ = node_states.shape
            graph_embs = []
            for step in range(t):
                _, g = self.graph_encoder(node_states[:, step], adjacency)
                graph_embs.append(g)
            graph_emb = torch.stack(graph_embs, dim=1)

        # Encode trajectory
        traj_emb = self.trajectory_encoder(graph_emb, actions, rewards, metrics)

        # Predictions from last timestep
        last = traj_emb[:, -1]

        return {
            "policy": self.policy_head(last),
            "value": self.value_head(last),
            "reward_pred": self.reward_head(last),
            "metric_pred": self.metric_head(last),
            "hidden": traj_emb,
            "contrastive": F.normalize(self.contrastive_head(last), dim=-1),
        }

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Single pretraining step with multiple objectives.

        Objectives:
        1. Masked prediction (predict masked states/actions)
        2. Contrastive learning (similar trajectories close)
        3. Metric prediction (predict pipeline metrics)

        Args:
            batch: Training batch

        Returns:
            Loss dictionary
        """
        node_states = batch["node_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        metrics = batch.get("metrics")
        adjacency = batch.get("adjacency")

        # Forward pass
        out = self.forward(node_states, actions, rewards, metrics, adjacency)

        losses = {}

        # Metric prediction loss
        if metrics is not None:
            target_metrics = metrics[:, -1]  # Last timestep
            losses["metric"] = F.mse_loss(out["metric_pred"], target_metrics)

        # Value prediction loss (if targets available)
        if "target_value" in batch:
            losses["value"] = F.mse_loss(out["value"].squeeze(-1), batch["target_value"])

        # Reward prediction loss
        if "target_reward" in batch:
            losses["reward"] = F.mse_loss(out["reward_pred"].squeeze(-1), batch["target_reward"])

        # Contrastive loss (if pairs available)
        if "positive_idx" in batch:
            # InfoNCE contrastive loss
            embeddings = out["contrastive"]
            positive_idx = batch["positive_idx"]

            # Compute similarity matrix and normalize by temperature
            logits = torch.matmul(embeddings, embeddings.T) / self.config.contrastive_temp

            # For InfoNCE, labels should be the indices of positive samples
            # If positive_idx[i] = j, then sample i's positive is sample j
            # We need to ensure positive_idx correctly represents the positive pair indices
            labels = positive_idx.to(logits.device)
            losses["contrastive"] = F.cross_entropy(logits, labels)

        # Total loss
        losses["total"] = sum(losses.values())

        return losses


class PipelineFoundationTrainer:
    """Trainer for Pipeline Foundation Model pretraining."""

    def __init__(
        self,
        model: PipelineFoundationModel,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.warmup_steps = warmup_steps
        self.step = 0

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute single training step."""
        self.model.train()
        self.step += 1

        # Learning rate warmup
        if self.step < self.warmup_steps:
            lr_scale = self.step / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * lr_scale

        # Forward + backward
        losses = self.model.pretrain_step(batch)

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}


def create_foundation_model(
    pretrained_path: str | None = None,
    config: PipelineFoundationConfig | None = None,
) -> PipelineFoundationModel:
    """Create or load a Pipeline Foundation Model.

    Args:
        pretrained_path: Optional path to pretrained weights
        config: Model configuration

    Returns:
        Initialized model
    """
    config = config or PipelineFoundationConfig()
    model = PipelineFoundationModel(config)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model
