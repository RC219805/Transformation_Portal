"""GAT-based centralized critic for topology-aware multi-agent coordination.

This module provides a Graph Attention Network-based critic that uses
the actual DAG topology for credit assignment and coordination.

Features:
- GAT message passing over DAG edges
- QMIX-style monotonic mixing
- VDN fallback for stability
- Understands dependency relationships

Benefits:
- Better credit assignment (upstream vs downstream)
- Learns patterns like "SAM2 errors propagate to NVDIFFREC"
- Topology-aware coordination
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
            raise ImportError("PyTorch required for GAT critic")
    return _torch, _nn, _F


class GATQMIXCritic:
    """GAT-based centralized critic with QMIX mixing.

    Uses Graph Attention Networks to process the DAG topology,
    enabling the critic to understand dependency relationships
    and perform better credit assignment.

    Architecture:
        Per-node states -> Node embedding -> GAT layers -> Context
        Agent Q-values + Context -> QMIX mixing -> Q_total

    Example:
        >>> critic = GATQMIXCritic(state_dim=28, n_agents=3)
        >>> q_total = critic.forward(agent_qs, agent_states, global_state, edge_index)
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        embed_dim: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 2,
        alpha: float = 0.8,
        gat_dropout: float = 0.0,
    ) -> None:
        """Initialize GAT-QMIX critic.

        Args:
            state_dim: Per-agent state dimension
            n_agents: Maximum number of agents
            embed_dim: Embedding dimension
            gat_heads: Number of GAT attention heads
            num_gat_layers: Number of GAT layers
            alpha: QMIX weight (1.0 = pure QMIX, 0.0 = pure VDN)
            gat_dropout: Dropout in GAT layers
        """
        torch, nn, F = _get_torch()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.alpha = alpha

        # Node embedding
        self.node_embed = nn.Linear(state_dim, embed_dim)

        # GAT layers
        self.gat_layers = []
        for i in range(num_gat_layers):
            gat = create_gat_layer(
                in_dim=embed_dim,
                out_dim=embed_dim,
                heads=gat_heads,
                dropout=gat_dropout,
            )
            self.gat_layers.append(gat)

        # Context projection
        self.context_proj = nn.Linear(embed_dim, embed_dim)

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
        """Run GAT layers.

        Args:
            x: Node features [N, embed_dim]
            edge_index: Graph edges [2, E]

        Returns:
            Updated features [N, embed_dim]
        """
        _, _, F = _get_torch()

        for gat in self.gat_layers:
            x = F.relu(gat.forward(x, edge_index))

        return x

    def forward(
        self,
        agent_qs: Any,
        agent_states: Any,
        global_state: Any,
        edge_index: Any,
    ) -> Any:
        """Compute Q_total using GAT-enhanced mixing.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            agent_states: Per-agent states [batch, n_agents, state_dim]
            global_state: Global state [batch, state_dim]
            edge_index: DAG edges [2, num_edges]

        Returns:
            Q_total [batch, 1]
        """
        torch, _, F = _get_torch()

        B, N, D = agent_states.shape

        # Process each batch item through GAT
        # (GAT operates on single graphs, so we loop over batch)
        embeddings = []

        for b in range(B):
            # Embed node states
            x = self.node_embed(agent_states[b])  # [N, embed_dim]

            # Run GAT
            x = self._run_gat(x, edge_index)  # [N, embed_dim]

            embeddings.append(x)

        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=0)  # [B, N, embed_dim]

        # Aggregate to context (mean pooling)
        context = embeddings.mean(dim=1)  # [B, embed_dim]
        context = self.context_proj(context)  # [B, embed_dim]

        # QMIX Layer 1
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(B, N, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)

        # [B, 1, N] @ [B, N, embed] = [B, 1, embed]
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Fuse GAT context
        hidden = hidden + context.view(B, 1, self.embed_dim)

        # QMIX Layer 2
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)

        qmix = torch.bmm(hidden, w2) + b2  # [B, 1, 1]
        qmix = qmix.view(B, 1)

        # VDN fallback
        vdn = agent_qs.sum(dim=1, keepdim=True)

        # Hybrid
        return self.alpha * qmix + (1.0 - self.alpha) * vdn

    def forward_with_attention(
        self,
        agent_qs: Any,
        agent_states: Any,
        global_state: Any,
        edge_index: Any,
    ) -> tuple[Any, list[Any]]:
        """Forward pass with attention weights for visualization.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            agent_states: Per-agent states [batch, n_agents, state_dim]
            global_state: Global state [batch, state_dim]
            edge_index: DAG edges [2, num_edges]

        Returns:
            Tuple of (Q_total, list of attention weights per layer)
        """
        torch, _, F = _get_torch()

        B, N, D = agent_states.shape
        all_attentions = []

        embeddings = []

        for b in range(B):
            x = self.node_embed(agent_states[b])
            batch_attn = []

            for gat in self.gat_layers:
                x, attn = gat.forward(x, edge_index, return_attention=True)
                x = F.relu(x)
                batch_attn.append(attn)

            embeddings.append(x)
            all_attentions.append(batch_attn)

        embeddings = torch.stack(embeddings, dim=0)
        context = self.context_proj(embeddings.mean(dim=1))

        # QMIX mixing (same as forward)
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

        return q_total, all_attentions

    def parameters(self):
        """Get all parameters."""
        params = list(self.node_embed.parameters())
        for gat in self.gat_layers:
            params.extend(gat.parameters())
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
            "gat_layers": [g.state_dict() for g in self.gat_layers],
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
        for i, gat_state in enumerate(state_dict["gat_layers"]):
            self.gat_layers[i].load_state_dict(gat_state)
        self.context_proj.load_state_dict(state_dict["context_proj"])
        self.hyper_w1.load_state_dict(state_dict["hyper_w1"])
        self.hyper_b1.load_state_dict(state_dict["hyper_b1"])
        self.hyper_w2.load_state_dict(state_dict["hyper_w2"])
        self.hyper_b2.load_state_dict(state_dict["hyper_b2"])
        self.alpha = state_dict.get("alpha", self.alpha)


def create_gat_critic(
    state_dim: int,
    n_agents: int,
    **kwargs: Any,
) -> GATQMIXCritic:
    """Factory function to create GAT-QMIX critic.

    Args:
        state_dim: State dimension
        n_agents: Number of agents
        **kwargs: Additional arguments

    Returns:
        GATQMIXCritic instance
    """
    return GATQMIXCritic(state_dim, n_agents, **kwargs)
