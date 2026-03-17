"""Pipeline Foundation Model (PFM) core model.

Graph + Temporal Transformer architecture for learning from
pipeline execution logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PFMConfig:
    """Configuration for Pipeline Foundation Model."""

    # Architecture
    node_vocab_size: int = 32
    feature_dim: int = 16
    d_model: int = 256
    n_heads: int = 8
    n_graph_layers: int = 2
    n_temporal_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Sequence
    max_nodes: int = 8
    max_seq_len: int = 64

    # Output dimensions
    action_dim: int = 64
    metric_dim: int = 4


class GraphAttention(nn.Module):
    """Multi-head attention over graph nodes."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply graph attention.

        Args:
            x: Node features [B, N, D]
            mask: Optional attention mask [N, N]

        Returns:
            Updated features [B, N, D]
        """
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_out))


class TemporalBlock(nn.Module):
    """Transformer block for temporal sequence modeling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply temporal transformer block.

        Args:
            x: Sequence [B, T, D]
            mask: Optional causal mask [T, T]

        Returns:
            Updated sequence [B, T, D]
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        x = self.norm2(x + self.ffn(x))

        return x


class PipelineFoundationModel(nn.Module):
    """Pipeline Foundation Model (PFM).

    A foundation model that learns from pipeline execution logs
    and can be fine-tuned for various downstream tasks:
    - Policy prediction (RL)
    - Value estimation (planning)
    - Metric prediction (fast evaluation)

    Architecture:
        1. Node embedding + feature projection
        2. Graph attention (captures DAG structure)
        3. Temporal transformer (captures evolution)
        4. Task-specific heads

    Example:
        >>> config = PFMConfig()
        >>> model = PipelineFoundationModel(config)
        >>> node_ids = torch.randint(0, 8, (2, 10, 4))  # [B, T, N]
        >>> features = torch.randn(2, 10, 4, 16)  # [B, T, N, F]
        >>> out = model(node_ids, features)
    """

    def __init__(self, config: PFMConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.node_embed = nn.Embedding(config.node_vocab_size, config.d_model)
        self.feat_proj = nn.Linear(config.feature_dim, config.d_model)

        # Graph attention layers
        self.graph_layers = nn.ModuleList(
            [GraphAttention(config.d_model, config.n_heads // 2, config.dropout) for _ in range(config.n_graph_layers)]
        )

        # Temporal transformer (operates on flattened graph)
        self.temporal_dim = config.d_model * config.max_nodes
        self.temporal_proj_in = nn.Linear(self.temporal_dim, config.d_model)
        self.temporal_proj_out = nn.Linear(config.d_model, self.temporal_dim)

        self.temporal_layers = nn.ModuleList(
            [
                TemporalBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_temporal_layers)
            ]
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)

        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.temporal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.temporal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        self.metric_head = nn.Sequential(
            nn.Linear(self.temporal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.metric_dim),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(self.temporal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        node_ids: torch.Tensor,
        features: torch.Tensor,
        graph_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through PFM.

        Args:
            node_ids: Node indices [B, T, N]
            features: Node features [B, T, N, F]
            graph_mask: Optional graph attention mask [N, N]
            causal: Whether to use causal masking for temporal attention

        Returns:
            Dictionary with policy, value, metrics, reward predictions
        """
        b, t, n, _ = features.shape

        # Embed and project
        node_emb = self.node_embed(node_ids)  # [B, T, N, D]
        feat_emb = self.feat_proj(features)  # [B, T, N, D]
        x = node_emb + feat_emb

        # Apply graph attention at each timestep
        graph_outputs = []
        for step in range(t):
            h = x[:, step]  # [B, N, D]
            for layer in self.graph_layers:
                h = layer(h, graph_mask)
            graph_outputs.append(h)

        # Stack and flatten for temporal processing
        g = torch.stack(graph_outputs, dim=1)  # [B, T, N, D]
        g_flat = g.view(b, t, -1)  # [B, T, N*D]

        # Project to temporal dimension
        h = self.temporal_proj_in(g_flat)  # [B, T, D]

        # Add positional encoding
        h = h + self.pos_embed[:, :t]

        # Create causal mask if needed
        causal_mask = None
        if causal:
            causal_mask = torch.triu(torch.ones(t, t, device=h.device), diagonal=1).bool()

        # Apply temporal layers
        for layer in self.temporal_layers:
            h = layer(h, causal_mask)

        # Project back
        h = self.temporal_proj_out(h)  # [B, T, N*D]

        # Use last timestep for predictions
        last = h[:, -1]  # [B, N*D]

        return {
            "policy": F.softmax(self.policy_head(last), dim=-1),
            "value": self.value_head(last),
            "metrics": self.metric_head(last),
            "reward": self.reward_head(last),
            "hidden": h,
        }

    def get_policy(
        self,
        node_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Get policy distribution for action selection.

        Args:
            node_ids: Node indices [B, T, N]
            features: Node features [B, T, N, F]

        Returns:
            Policy distribution [B, action_dim]
        """
        out = self.forward(node_ids, features)
        return out["policy"]

    def get_value(
        self,
        node_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Get value estimate.

        Args:
            node_ids: Node indices [B, T, N]
            features: Node features [B, T, N, F]

        Returns:
            Value estimate [B, 1]
        """
        out = self.forward(node_ids, features)
        return out["value"]


class PFMForRL(nn.Module):
    """PFM adapted for reinforcement learning.

    Wraps PFM with RL-specific interfaces compatible with
    the existing RL training loop.
    """

    def __init__(self, pfm: PipelineFoundationModel) -> None:
        super().__init__()
        self.pfm = pfm

    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select action from state.

        Args:
            state: Encoded state (must include node_ids and features)
            deterministic: Whether to select greedily

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Assume state is already encoded
        out = self.pfm.forward(
            state["node_ids"].unsqueeze(0),
            state["features"].unsqueeze(0),
        )

        policy = out["policy"][0]
        value = out["value"][0]

        if deterministic:
            action = policy.argmax().item()
            log_prob = policy[action].log()
        else:
            dist = torch.distributions.Categorical(policy)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob, value
