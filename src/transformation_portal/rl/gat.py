"""Graph Attention Network (GAT) layer implementation.

This module provides a pure PyTorch implementation of Graph Attention
Networks without external dependencies. Suitable for small graphs
like pipeline DAGs.

Reference:
    Veličković et al., "Graph Attention Networks", ICLR 2018
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
            raise ImportError("PyTorch required for GAT")
    return _torch, _nn, _F


class GATLayer:
    """Single Graph Attention Network layer.

    Implements multi-head attention over graph nodes, where attention
    weights are learned based on node features and graph structure.

    Supports:
    - Multi-head attention
    - LeakyReLU activation for attention scores
    - Edge-based message passing
    - Residual connections (optional)

    Example:
        >>> gat = GATLayer(in_dim=64, out_dim=64, heads=4)
        >>> x = torch.randn(5, 64)  # 5 nodes
        >>> edge_index = torch.tensor([[0,1,2], [1,2,3]])  # 3 edges
        >>> out = gat.forward(x, edge_index)
        >>> print(out.shape)  # [5, 64]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = False,
    ) -> None:
        """Initialize GAT layer.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            heads: Number of attention heads
            dropout: Dropout probability
            negative_slope: LeakyReLU negative slope
            residual: Whether to use residual connection
        """
        torch, nn, F = _get_torch()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual

        # Linear transformation for node features
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)

        # Attention parameters (per head)
        # a = [a_src || a_dst], split into source and destination
        self.a_src = nn.Parameter(torch.randn(heads, out_dim))
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim))

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim * heads))

        # Residual projection if dimensions don't match
        if residual and in_dim != out_dim * heads:
            self.res_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        else:
            self.res_proj = None

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize layer parameters."""
        torch, nn, _ = _get_torch()

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)

    def forward(
        self,
        x: Any,
        edge_index: Any,
        return_attention: bool = False,
    ) -> Any:
        """Forward pass through GAT layer.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edges [2, num_edges]
            return_attention: Whether to return attention weights

        Returns:
            Updated node features [num_nodes, out_dim]
            or tuple of (features, attention_weights) if return_attention
        """
        torch, _, F = _get_torch()

        N = x.size(0)  # Number of nodes
        H = self.heads
        D = self.out_dim

        # Transform features: [N, in_dim] -> [N, H*D] -> [N, H, D]
        h = self.W(x)
        h = h.view(N, H, D)

        # Handle empty graph
        if edge_index.numel() == 0:
            out = h.mean(dim=1)  # Average over heads
            if return_attention:
                return out, None
            return out

        src, dst = edge_index  # [E], [E]
        E = src.size(0)

        # Get source and destination node features
        h_src = h[src]  # [E, H, D]
        h_dst = h[dst]  # [E, H, D]

        # Compute attention scores
        # e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
        e_src = (h_src * self.a_src).sum(dim=-1)  # [E, H]
        e_dst = (h_dst * self.a_dst).sum(dim=-1)  # [E, H]
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)

        # Softmax over incoming edges per destination node
        # Use scatter operations for efficiency
        attn = torch.zeros(E, H, device=x.device, dtype=x.dtype)

        for i in range(N):
            mask = dst == i
            if mask.any():
                attn[mask] = F.softmax(e[mask], dim=0)

        # Apply dropout to attention weights
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Aggregate: weighted sum of source features
        # out_i = Σ_j α_ij · h_j
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)

        for i in range(E):
            out[dst[i]] += attn[i].unsqueeze(-1) * h_src[i]

        # Average over heads
        out = out.mean(dim=1)  # [N, D]

        # Add bias
        out = out + self.bias[:D]  # Use first D elements of bias

        # Residual connection
        if self.residual:
            if self.res_proj is not None:
                res = self.res_proj(x)
            else:
                res = x
            out = out + res[:, :D] if res.size(1) > D else out + res

        if return_attention:
            return out, attn

        return out

    def parameters(self):
        """Get layer parameters."""
        params = list(self.W.parameters())
        params.append(self.a_src)
        params.append(self.a_dst)
        params.append(self.bias)
        if self.res_proj is not None:
            params.extend(self.res_proj.parameters())
        return params

    def state_dict(self) -> dict[str, Any]:
        """Get state dict."""
        state = {
            "W": self.W.state_dict(),
            "a_src": self.a_src,
            "a_dst": self.a_dst,
            "bias": self.bias,
        }
        if self.res_proj is not None:
            state["res_proj"] = self.res_proj.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict."""
        self.W.load_state_dict(state_dict["W"])
        self.a_src.data.copy_(state_dict["a_src"])
        self.a_dst.data.copy_(state_dict["a_dst"])
        self.bias.data.copy_(state_dict["bias"])
        if self.res_proj is not None and "res_proj" in state_dict:
            self.res_proj.load_state_dict(state_dict["res_proj"])


class GATv2Layer:
    """GATv2 layer with improved attention mechanism.

    GATv2 applies the nonlinearity before computing attention,
    making it more expressive than the original GAT.

    Reference:
        Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        share_weights: bool = False,
    ) -> None:
        """Initialize GATv2 layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            heads: Number of attention heads
            dropout: Dropout probability
            share_weights: Share weights between source and target
        """
        torch, nn, F = _get_torch()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.share_weights = share_weights

        # Separate linear transformations for source and target
        self.W_src = nn.Linear(in_dim, out_dim * heads, bias=False)
        if share_weights:
            self.W_dst = self.W_src
        else:
            self.W_dst = nn.Linear(in_dim, out_dim * heads, bias=False)

        # Attention vector (applied after nonlinearity)
        self.att = nn.Parameter(torch.randn(heads, out_dim))

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: Any, edge_index: Any) -> Any:
        """Forward pass.

        Args:
            x: Node features [N, in_dim]
            edge_index: Graph edges [2, E]

        Returns:
            Updated features [N, out_dim]
        """
        torch, _, F = _get_torch()

        N = x.size(0)
        H = self.heads
        D = self.out_dim

        # Transform
        h_src = self.W_src(x).view(N, H, D)
        h_dst = self.W_dst(x).view(N, H, D)

        if edge_index.numel() == 0:
            return h_src.mean(dim=1)

        src, dst = edge_index
        E = src.size(0)

        # GATv2: apply nonlinearity before attention
        # e_ij = a · LeakyReLU(W_src·h_i + W_dst·h_j)
        combined = F.leaky_relu(h_src[src] + h_dst[dst], negative_slope=0.2)
        e = (combined * self.att).sum(dim=-1)  # [E, H]

        # Softmax per destination
        attn = torch.zeros(E, H, device=x.device, dtype=x.dtype)
        for i in range(N):
            mask = dst == i
            if mask.any():
                attn[mask] = F.softmax(e[mask], dim=0)

        # Aggregate
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        for i in range(E):
            out[dst[i]] += attn[i].unsqueeze(-1) * h_src[src[i]]

        return out.mean(dim=1) + self.bias

    def parameters(self):
        """Get parameters."""
        params = list(self.W_src.parameters())
        if not self.share_weights:
            params.extend(self.W_dst.parameters())
        params.append(self.att)
        params.append(self.bias)
        return params


def create_gat_layer(
    in_dim: int,
    out_dim: int,
    version: str = "v1",
    **kwargs: Any,
) -> GATLayer | GATv2Layer:
    """Factory function to create GAT layer.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        version: "v1" for original GAT, "v2" for GATv2
        **kwargs: Additional arguments

    Returns:
        GAT layer instance
    """
    if version == "v2":
        return GATv2Layer(in_dim, out_dim, **kwargs)
    return GATLayer(in_dim, out_dim, **kwargs)
