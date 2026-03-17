"""Graph Transformer MuZero (GTMuZero) for unified pipeline planning.

This module implements a Graph Transformer architecture that unifies:
- DAG topology (graph attention over nodes)
- Temporal evolution (sequence attention across steps)
- Multi-agent coordination (per-node tokens + shared context)

It replaces separate components (GAT critic, temporal model, policy/value heads)
with one coherent architecture for end-to-end pipeline optimization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


class NodeEmbedding(nn.Module):
    """Project node-level state to embedding space."""

    def __init__(self, state_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed node states.

        Args:
            x: Node states [B, N, state_dim]

        Returns:
            Node embeddings [B, N, d_model]
        """
        return self.proj(x)


class GraphAttentionLayer(nn.Module):
    """Graph attention layer with DAG-aware masking.

    Implements attention over graph nodes with optional
    adjacency-based masking to respect DAG structure.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply graph attention.

        Args:
            x: Node features [B, N, d_model]
            edge_mask: Optional adjacency mask [N, N] (True = masked)

        Returns:
            Updated node features [B, N, d_model]
        """
        b, n, d = x.shape
        h = self.n_heads
        dk = self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x).view(b, n, h, dk).transpose(1, 2)  # [B, H, N, dk]
        k = self.k_proj(x).view(b, n, h, dk).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, dk).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)

        # Apply DAG mask if provided
        if edge_mask is not None:
            scores = scores.masked_fill(edge_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, H, N, dk]
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)

        # Residual + norm
        return self.layer_norm(x + out)


class TemporalTransformerLayer(nn.Module):
    """Transformer layer for temporal sequence modeling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            x: Sequence [B, T, d_model]
            causal_mask: Optional causal mask [T, T]

        Returns:
            Updated sequence [B, T, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class GraphTransformerMuZero(nn.Module):
    """Graph Transformer MuZero for unified pipeline planning.

    Combines graph attention (DAG structure) with temporal transformers
    (sequence evolution) for comprehensive pipeline optimization.

    Architecture:
        1. Node embedding: state -> d_model
        2. Graph attention blocks: capture DAG dependencies
        3. Temporal transformer: model evolution across time
        4. Prediction heads: policy, value, reward, next_state

    Example:
        >>> model = GraphTransformerMuZero(
        ...     state_dim=64, action_dim=10, n_nodes=3
        ... )
        >>> state_seq = torch.randn(2, 5, 3, 64)  # [B, T, N, D]
        >>> out = model(state_seq)
        >>> print(out["policy"].shape)  # [2, 10]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_nodes: int,
        d_model: int = 256,
        n_graph_layers: int = 2,
        n_temporal_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_nodes = n_nodes
        self.d_model = d_model

        # Node embedding
        self.node_embed = NodeEmbedding(state_dim, d_model)

        # Graph attention layers
        self.graph_layers = nn.ModuleList(
            [GraphAttentionLayer(d_model, n_heads=4, dropout=dropout) for _ in range(n_graph_layers)]
        )

        # Temporal transformer (operates on flattened node features)
        self.temporal_dim = d_model * n_nodes
        self.temporal_layers = nn.ModuleList(
            [TemporalTransformerLayer(self.temporal_dim, n_heads, dropout) for _ in range(n_temporal_layers)]
        )

        # Positional encoding for temporal dimension
        self.pos_embed = nn.Parameter(torch.randn(1, 128, self.temporal_dim) * 0.02)

        # Prediction heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.temporal_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.temporal_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.temporal_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.next_state_head = nn.Sequential(
            nn.Linear(self.temporal_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.temporal_dim),
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through Graph Transformer.

        Args:
            state_seq: Node states over time [B, T, N, state_dim]
            edge_mask: Optional DAG adjacency mask [N, N]
            causal: Whether to use causal masking for temporal attention

        Returns:
            Dictionary with policy, value, reward, next_state predictions
        """
        b, t, n, _ = state_seq.shape

        # Process each timestep through graph attention
        graph_outputs = []
        for step in range(t):
            x = self.node_embed(state_seq[:, step])  # [B, N, d_model]

            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_mask)

            graph_outputs.append(x)

        # Stack and flatten nodes: [B, T, N*d_model]
        h = torch.stack(graph_outputs, dim=1)
        h = h.view(b, t, self.temporal_dim)

        # Add positional encoding
        h = h + self.pos_embed[:, :t]

        # Create causal mask if needed
        causal_mask = None
        if causal:
            causal_mask = torch.triu(torch.ones(t, t, device=h.device), diagonal=1).bool()

        # Temporal transformer
        for temporal_layer in self.temporal_layers:
            h = temporal_layer(h, causal_mask)

        # Use last timestep for predictions
        last = h[:, -1]

        return {
            "policy": F.softmax(self.policy_head(last), dim=-1),
            "value": self.value_head(last),
            "reward": self.reward_head(last),
            "next_state": self.next_state_head(last),
            "hidden": h,
        }

    def initial_inference(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial inference for MCTS root.

        Args:
            obs: Initial observation [B, N, state_dim]

        Returns:
            Tuple of (hidden_state, policy, value)
        """
        # Single timestep
        state_seq = obs.unsqueeze(1)  # [B, 1, N, state_dim]
        out = self.forward(state_seq, causal=False)

        return out["hidden"][:, -1], out["policy"], out["value"]

    def recurrent_inference(
        self,
        hidden: torch.Tensor,
        history: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference with history context.

        Args:
            hidden: Current hidden state [B, temporal_dim]
            history: Previous states [B, T, N, state_dim]
            action: Action to take [B]

        Returns:
            Tuple of (next_hidden, policy, value, reward)
        """
        # For recurrent inference, we predict next state and continue
        out = self.forward(history, causal=True)

        return (
            out["hidden"][:, -1],
            out["policy"],
            out["value"],
            out["reward"],
        )


def build_dag_attention_mask(
    edge_index: torch.Tensor,
    n_nodes: int,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Build attention mask from DAG edges.

    Args:
        edge_index: Edge indices [2, E] (source, target)
        n_nodes: Number of nodes
        bidirectional: Allow attention in both directions

    Returns:
        Attention mask [N, N] where True = masked (no attention)
    """
    # Start with all masked (no attention)
    mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool)

    # Allow self-attention
    mask.diagonal().fill_(False)

    # Allow attention along edges
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]

        # dst can attend to src (upstream)
        for s, d in zip(src.tolist(), dst.tolist()):
            mask[d, s] = False
            if bidirectional:
                mask[s, d] = False

    return mask


class GraphTransformerMuZeroV2(nn.Module):
    """Enhanced GTMuZero with per-node action spaces.

    Supports multi-agent setting where each node has its own
    action space, while still coordinating through shared context.
    """

    def __init__(
        self,
        state_dim: int,
        node_action_dims: list[int],
        n_nodes: int,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.node_action_dims = node_action_dims

        # Shared encoder
        self.node_embed = NodeEmbedding(state_dim, d_model)

        self.graph_layers = nn.ModuleList([GraphAttentionLayer(d_model) for _ in range(2)])

        # Per-node policy heads
        self.policy_heads = nn.ModuleList([nn.Linear(d_model, action_dim) for action_dim in node_action_dims])

        # Shared value head (central critic)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * n_nodes, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass with per-node policies.

        Args:
            state: Node states [B, N, state_dim]
            edge_mask: Optional DAG mask [N, N]

        Returns:
            Dictionary with per-node policies and shared value
        """
        x = self.node_embed(state)

        for layer in self.graph_layers:
            x = layer(x, edge_mask)

        # Per-node policies
        policies = []
        for i, head in enumerate(self.policy_heads):
            p = F.softmax(head(x[:, i]), dim=-1)
            policies.append(p)

        # Central value
        flat = x.view(x.size(0), -1)
        value = self.value_head(flat)

        return {
            "policies": policies,  # List[Tensor], one per node
            "value": value,
            "hidden": x,
        }
