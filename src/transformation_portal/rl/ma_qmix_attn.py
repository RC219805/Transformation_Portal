"""QMIX + Attention Central Critic for context-aware multi-agent coordination.

This module provides an enhanced QMIX critic with attention over agent
embeddings, enabling the critic to focus on problematic agents based
on the global context.

Features:
- Monotonic QMIX constraint (positive weights)
- Multi-head attention over agent embeddings
- Hybrid with VDN fallback
- Optional agent masking for inactive nodes
"""

from __future__ import annotations

import logging
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
            raise ImportError("PyTorch required for QMIX attention")
    return _torch, _nn, _F


class AgentEmbed:
    """Project per-agent local state to embedding space.

    Maps agent-specific state features to a shared embedding space
    for attention-based mixing.
    """

    def __init__(self, state_dim: int, embed_dim: int = 64) -> None:
        """Initialize embedding network.

        Args:
            state_dim: Input state dimension
            embed_dim: Output embedding dimension
        """
        torch, nn, F = _get_torch()

        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, s_i: Any) -> Any:
        """Embed agent state.

        Args:
            s_i: Agent state [batch, state_dim]

        Returns:
            Embedding [batch, embed_dim]
        """
        return self.net(s_i)

    def parameters(self):
        """Get parameters."""
        return self.net.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {"net": self.net.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.net.load_state_dict(state_dict["net"])


class AttnMixer:
    """QMIX-style mixing with multi-head attention over agent embeddings.

    Combines:
    - Hypernetwork-based mixing (QMIX)
    - Multi-head attention for context-aware coordination
    - Monotonicity via positive weights (abs)

    The attention mechanism allows the critic to focus on specific
    agents based on the global context (e.g., focusing on SAM2 when
    there are segmentation issues).
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
    ) -> None:
        """Initialize attention mixer.

        Args:
            n_agents: Number of agents
            state_dim: Global state dimension
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
        """
        torch, nn, F = _get_torch()

        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Attention projections
        # Query from global state, Keys/Values from agent embeddings
        self.q_proj = nn.Linear(state_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # QMIX hypernetworks (positive via abs for monotonicity)
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

        # Attention context fusion
        self.context_proj = nn.Linear(embed_dim, embed_dim)

    def _attention(
        self,
        global_state: Any,
        agent_embeds: Any,
        mask: Any = None,
    ) -> tuple[Any, Any]:
        """Compute multi-head attention over agent embeddings.

        Args:
            global_state: Global state [batch, state_dim]
            agent_embeds: Agent embeddings [batch, n_agents, embed_dim]
            mask: Optional attention mask [batch, n_agents]

        Returns:
            Tuple of (context, attention_weights)
        """
        torch, _, F = _get_torch()

        B, N, E_dim = agent_embeds.shape

        # Project query from global state
        q = self.q_proj(global_state)  # [B, embed_dim]

        # Project keys and values from agent embeddings
        k = self.k_proj(agent_embeds)  # [B, N, embed_dim]
        v = self.v_proj(agent_embeds)  # [B, N, embed_dim]

        # Reshape for multi-head attention
        q = q.view(B, self.n_heads, self.head_dim)  # [B, H, D]
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Scaled dot-product attention
        # q: [B, H, D], k: [B, H, N, D]
        attn_scores = torch.einsum("bhd,bhnd->bhn", q, k) / (self.head_dim**0.5)  # [B, H, N]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1)  # [B, H, N]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N]

        # Compute context
        # attn_weights: [B, H, N], v: [B, H, N, D]
        context = torch.einsum("bhn,bhnd->bhd", attn_weights, v)  # [B, H, D]

        # Reshape back
        context = context.reshape(B, self.embed_dim)  # [B, embed_dim]

        # Output projection
        context = self.o_proj(context)

        return context, attn_weights

    def forward(
        self,
        agent_qs: Any,
        agent_embeds: Any,
        global_state: Any,
        mask: Any = None,
    ) -> tuple[Any, Any]:
        """Mix agent Q-values with attention-enhanced QMIX.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            agent_embeds: Agent embeddings [batch, n_agents, embed_dim]
            global_state: Global state [batch, state_dim]
            mask: Optional agent mask [batch, n_agents]

        Returns:
            Tuple of (Q_total [batch, 1], attention_weights)
        """
        torch, _, F = _get_torch()

        B = agent_qs.size(0)

        # Compute attention context
        context, attn_weights = self._attention(global_state, agent_embeds, mask)

        # QMIX Layer 1: agent_qs -> hidden
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)

        # [B, 1, n_agents] @ [B, n_agents, embed_dim] = [B, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Fuse attention context
        context_proj = self.context_proj(context).view(B, 1, self.embed_dim)
        hidden = hidden + context_proj

        # QMIX Layer 2: hidden -> Q_total
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # [B, 1, 1]

        return q_total.view(B, 1), attn_weights

    def parameters(self):
        """Get all parameters."""
        params = []
        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.hyper_w1,
            self.hyper_b1,
            self.hyper_w2,
            self.hyper_b2,
            self.context_proj,
        ]:
            params.extend(module.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "q_proj": self.q_proj.state_dict(),
            "k_proj": self.k_proj.state_dict(),
            "v_proj": self.v_proj.state_dict(),
            "o_proj": self.o_proj.state_dict(),
            "hyper_w1": self.hyper_w1.state_dict(),
            "hyper_b1": self.hyper_b1.state_dict(),
            "hyper_w2": self.hyper_w2.state_dict(),
            "hyper_b2": self.hyper_b2.state_dict(),
            "context_proj": self.context_proj.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.q_proj.load_state_dict(state_dict["q_proj"])
        self.k_proj.load_state_dict(state_dict["k_proj"])
        self.v_proj.load_state_dict(state_dict["v_proj"])
        self.o_proj.load_state_dict(state_dict["o_proj"])
        self.hyper_w1.load_state_dict(state_dict["hyper_w1"])
        self.hyper_b1.load_state_dict(state_dict["hyper_b1"])
        self.hyper_w2.load_state_dict(state_dict["hyper_w2"])
        self.hyper_b2.load_state_dict(state_dict["hyper_b2"])
        self.context_proj.load_state_dict(state_dict["context_proj"])


class CentralCriticQMIXAttn:
    """Hybrid QMIX + Attention + VDN Central Critic.

    Combines:
    - QMIX: Hypernetwork-based mixing with monotonicity
    - Attention: Context-aware coordination across agents
    - VDN: Simple sum fallback for stability

    The attention mechanism enables the critic to:
    - Focus on problematic agents based on global context
    - Better credit assignment when some agents have issues
    - Faster convergence through targeted coordination
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        alpha: float = 0.8,
    ) -> None:
        """Initialize hybrid critic.

        Args:
            n_agents: Number of agents
            state_dim: Global state dimension
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            alpha: QMIX weight (1.0 = pure QMIX, 0.0 = pure VDN)
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.alpha = alpha

        self.embed = AgentEmbed(state_dim, embed_dim)
        self.mixer = AttnMixer(n_agents, state_dim, embed_dim, n_heads)

    def forward(
        self,
        agent_qs: Any,
        agent_states: Any,
        global_state: Any,
        mask: Any = None,
    ) -> tuple[Any, Any]:
        """Compute Q_total from agent Q-values and states.

        Args:
            agent_qs: Per-agent Q-values [batch, n_agents]
            agent_states: Per-agent states [batch, n_agents, state_dim]
            global_state: Global state [batch, state_dim]
            mask: Optional agent mask [batch, n_agents]

        Returns:
            Tuple of (Q_total [batch, 1], attention_weights)
        """
        torch, _, _ = _get_torch()

        B, N, S = agent_states.shape

        # Embed each agent's state
        embeds = []
        for i in range(N):
            embeds.append(self.embed.forward(agent_states[:, i, :]))
        embeds = torch.stack(embeds, dim=1)  # [B, n_agents, embed_dim]

        # QMIX + attention
        qmix, attn_weights = self.mixer.forward(agent_qs, embeds, global_state, mask)

        # VDN component (simple sum)
        vdn = agent_qs.sum(dim=1, keepdim=True)

        # Hybrid blend
        q_total = self.alpha * qmix + (1.0 - self.alpha) * vdn

        return q_total, attn_weights

    def parameters(self):
        """Get all parameters."""
        params = list(self.embed.parameters())
        params.extend(self.mixer.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        return {
            "embed": self.embed.state_dict(),
            "mixer": self.mixer.state_dict(),
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.embed.load_state_dict(state_dict["embed"])
        self.mixer.load_state_dict(state_dict["mixer"])
        self.alpha = state_dict.get("alpha", self.alpha)


def create_attention_critic(
    n_agents: int,
    state_dim: int,
    **kwargs: Any,
) -> CentralCriticQMIXAttn:
    """Factory function to create attention-based critic.

    Args:
        n_agents: Number of agents
        state_dim: State dimension
        **kwargs: Additional arguments

    Returns:
        CentralCriticQMIXAttn instance
    """
    return CentralCriticQMIXAttn(n_agents, state_dim, **kwargs)
