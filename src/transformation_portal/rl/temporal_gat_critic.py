"""Temporal GAT critic: Time-aware graph attention for sequential decisions.

This module provides a temporal extension to the GAT critic that:
- Processes sequences of pipeline states over time
- Uses temporal attention to capture evolution patterns
- Learns delayed effects (e.g., segmentation → reconstruction → material)
- Combines graph structure awareness with temporal dynamics

Architecture:
    Time t-2 → t-1 → t
       ↓        ↓      ↓
     GAT      GAT    GAT (per-timestep graph processing)
       ↓        ↓      ↓
    Temporal Attention (sequence modeling)
       ↓
    QMIX + VDN mixing
       ↓
    Q_total
"""

from __future__ import annotations

import logging
from typing import Any

from transformation_portal.rl.gat import GATLayer, create_gat_layer

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
            raise ImportError("PyTorch required for temporal GAT")
    return _torch, _nn, _F


class TemporalGATCritic:
    """Temporal Graph Attention critic with QMIX mixing.

    Combines:
    - GAT: Graph structure awareness (DAG topology)
    - Temporal attention: Sequence modeling across timesteps
    - QMIX: Monotonic value mixing
    - VDN: Stable fallback

    This enables learning patterns like:
    - "Lowering threshold → improves coverage BUT increases noise later"
    - "Early errors propagate through the DAG over time"

    Example:
        >>> critic = TemporalGATCritic(state_dim=28, n_agents=3, seq_len=3)
        >>> q_total = critic.forward(qs_seq, states_seq, global_seq, edges)
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        embed_dim: int = 64,
        seq_len: int = 3,
        gat_heads: int = 4,
        temporal_heads: int = 4,
        alpha: float = 0.8,
    ) -> None:
        """Initialize temporal GAT critic.

        Args:
            state_dim: Per-agent state dimension
            n_agents: Maximum number of agents
            embed_dim: Embedding dimension
            seq_len: Sequence length for temporal modeling
            gat_heads: Number of GAT attention heads
            temporal_heads: Number of temporal attention heads
            alpha: QMIX weight (1.0 = pure QMIX, 0.0 = pure VDN)
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.alpha = alpha

        # Node embedding
        self.node_embed = nn.Linear(state_dim, embed_dim)

        # GAT layers for spatial processing
        self.gat1 = create_gat_layer(embed_dim, embed_dim, heads=gat_heads)
        self.gat2 = create_gat_layer(embed_dim, embed_dim, heads=gat_heads)

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            batch_first=True,
        )

        # Temporal position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.1)

        # Context fusion
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # QMIX hypernetworks
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def _run_gat(self, x: Any, edge_index: Any) -> Any:
        """Run GAT layers on single timestep.

        Args:
            x: Node features [N, embed_dim]
            edge_index: Graph edges [2, E]

        Returns:
            Updated features [N, embed_dim]
        """
        _, _, F = _get_torch()

        x = F.relu(self.gat1.forward(x, edge_index))
        x = F.relu(self.gat2.forward(x, edge_index))
        return x

    def forward(
        self,
        agent_qs_seq: Any,
        agent_states_seq: Any,
        global_state_seq: Any,
        edge_index: Any,
    ) -> Any:
        """Compute Q_total from temporal sequence.

        Args:
            agent_qs_seq: Per-agent Q-values [batch, seq_len, n_agents]
            agent_states_seq: Per-agent states [batch, seq_len, n_agents, state_dim]
            global_state_seq: Global states [batch, seq_len, state_dim]
            edge_index: DAG edges [2, num_edges] (assumed constant over sequence)

        Returns:
            Q_total [batch, 1]
        """
        torch, _, F = _get_torch()

        B, T, N, D = agent_states_seq.shape

        # Process each timestep through GAT
        node_embeddings_seq = []

        for t in range(T):
            timestep_embeds = []

            for b in range(B):
                # Embed node states
                x = self.node_embed(agent_states_seq[b, t])  # [N, embed_dim]

                # Run GAT
                x = self._run_gat(x, edge_index)  # [N, embed_dim]

                timestep_embeds.append(x)

            # Stack batch: [B, N, embed_dim]
            timestep_embeds = torch.stack(timestep_embeds, dim=0)

            # Aggregate nodes to single embedding per batch: [B, embed_dim]
            node_embeddings_seq.append(timestep_embeds.mean(dim=1))

        # Stack sequence: [B, T, embed_dim]
        node_embeddings_seq = torch.stack(node_embeddings_seq, dim=1)

        # Add positional encoding
        node_embeddings_seq = node_embeddings_seq + self.pos_encoding[:, :T, :]

        # Temporal attention
        attn_out, attn_weights = self.temporal_attn(
            node_embeddings_seq,
            node_embeddings_seq,
            node_embeddings_seq,
        )

        # Use last timestep context (most recent)
        context = attn_out[:, -1, :]  # [B, embed_dim]
        context = self.context_proj(context)

        # Get Q-values and global state for last timestep
        agent_qs = agent_qs_seq[:, -1, :]  # [B, N]
        global_state = global_state_seq[:, -1, :]  # [B, state_dim]

        # QMIX Layer 1
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(B, N, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Fuse temporal context
        hidden = hidden + context.view(B, 1, self.embed_dim)

        # QMIX Layer 2
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)

        qmix = torch.bmm(hidden, w2) + b2
        qmix = qmix.view(B, 1)

        # VDN fallback
        vdn = agent_qs.sum(dim=1, keepdim=True)

        return self.alpha * qmix + (1.0 - self.alpha) * vdn

    def forward_with_attention(
        self,
        agent_qs_seq: Any,
        agent_states_seq: Any,
        global_state_seq: Any,
        edge_index: Any,
    ) -> tuple[Any, Any, list[Any]]:
        """Forward with attention weights for visualization.

        Returns:
            Tuple of (Q_total, temporal_attention_weights, gat_attention_weights)
        """
        torch, _, F = _get_torch()

        B, T, N, D = agent_states_seq.shape

        node_embeddings_seq = []
        all_gat_attentions = []

        for t in range(T):
            timestep_embeds = []
            timestep_gat_attn = []

            for b in range(B):
                x = self.node_embed(agent_states_seq[b, t])
                x, attn1 = self.gat1.forward(x, edge_index, return_attention=True)
                x = F.relu(x)
                x, attn2 = self.gat2.forward(x, edge_index, return_attention=True)
                x = F.relu(x)

                timestep_embeds.append(x)
                timestep_gat_attn.append((attn1, attn2))

            timestep_embeds = torch.stack(timestep_embeds, dim=0)
            node_embeddings_seq.append(timestep_embeds.mean(dim=1))
            all_gat_attentions.append(timestep_gat_attn)

        node_embeddings_seq = torch.stack(node_embeddings_seq, dim=1)
        node_embeddings_seq = node_embeddings_seq + self.pos_encoding[:, :T, :]

        attn_out, temporal_attn_weights = self.temporal_attn(
            node_embeddings_seq,
            node_embeddings_seq,
            node_embeddings_seq,
        )

        context = self.context_proj(attn_out[:, -1, :])

        agent_qs = agent_qs_seq[:, -1, :]
        global_state = global_state_seq[:, -1, :]

        w1 = torch.abs(self.hyper_w1(global_state)).view(B, N, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        hidden = hidden + context.view(B, 1, self.embed_dim)

        w2 = torch.abs(self.hyper_w2(global_state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        qmix = torch.bmm(hidden, w2) + b2
        qmix = qmix.view(B, 1)

        vdn = agent_qs.sum(dim=1, keepdim=True)
        q_total = self.alpha * qmix + (1.0 - self.alpha) * vdn

        return q_total, temporal_attn_weights, all_gat_attentions

    def parameters(self):
        """Get all parameters."""
        params = list(self.node_embed.parameters())
        params.extend(self.gat1.parameters())
        params.extend(self.gat2.parameters())
        params.extend(self.temporal_attn.parameters())
        params.append(self.pos_encoding)
        params.extend(self.context_proj.parameters())
        params.extend(self.hyper_w1.parameters())
        params.extend(self.hyper_b1.parameters())
        params.extend(self.hyper_w2.parameters())
        params.extend(self.hyper_b2.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "node_embed": self.node_embed.state_dict(),
            "gat1": self.gat1.state_dict(),
            "gat2": self.gat2.state_dict(),
            "temporal_attn": self.temporal_attn.state_dict(),
            "pos_encoding": self.pos_encoding,
            "context_proj": self.context_proj.state_dict(),
            "hyper_w1": self.hyper_w1.state_dict(),
            "hyper_b1": self.hyper_b1.state_dict(),
            "hyper_w2": self.hyper_w2.state_dict(),
            "hyper_b2": self.hyper_b2.state_dict(),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.node_embed.load_state_dict(state_dict["node_embed"])
        self.gat1.load_state_dict(state_dict["gat1"])
        self.gat2.load_state_dict(state_dict["gat2"])
        self.temporal_attn.load_state_dict(state_dict["temporal_attn"])
        self.pos_encoding.data.copy_(state_dict["pos_encoding"])
        self.context_proj.load_state_dict(state_dict["context_proj"])
        self.hyper_w1.load_state_dict(state_dict["hyper_w1"])
        self.hyper_b1.load_state_dict(state_dict["hyper_b1"])
        self.hyper_w2.load_state_dict(state_dict["hyper_w2"])
        self.hyper_b2.load_state_dict(state_dict["hyper_b2"])
        self.alpha = state_dict.get("alpha", self.alpha)


def create_temporal_gat_critic(
    state_dim: int,
    n_agents: int,
    seq_len: int = 3,
    **kwargs: Any,
) -> TemporalGATCritic:
    """Factory function to create temporal GAT critic.

    Args:
        state_dim: State dimension
        n_agents: Number of agents
        seq_len: Sequence length
        **kwargs: Additional arguments

    Returns:
        TemporalGATCritic instance
    """
    return TemporalGATCritic(state_dim, n_agents, seq_len=seq_len, **kwargs)
