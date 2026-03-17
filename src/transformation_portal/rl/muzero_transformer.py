"""Transformer-based MuZero for sequence modeling over pipeline evolution.

This module implements a Transformer-based variant of MuZero that models
temporal evolution explicitly, supporting long-horizon credit assignment
and trajectory-aware planning.

Architecture:
    - Single Transformer backbone (replaces fθ, gθ, hθ)
    - Sequence-to-sequence modeling
    - Policy, value, reward prediction heads
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to input."""
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.pos_emb(pos)


class MuZeroTransformer(nn.Module):
    """Transformer-based MuZero model.

    Processes observation-action sequences to predict next state,
    policy, value, and reward. Enables sequence-aware planning
    that captures long-range dependencies.

    Example:
        >>> model = MuZeroTransformer(obs_dim=64, action_dim=10)
        >>> obs_seq = torch.randn(2, 5, 64)  # batch=2, seq_len=5
        >>> act_seq = torch.randint(0, 10, (2, 5))
        >>> out = model(obs_seq, act_seq)
        >>> print(out["policy"].shape)  # [2, 10]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projections
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_emb = nn.Embedding(action_dim, d_model)

        # Positional encoding
        self.pos = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process observation-action sequence.

        Args:
            obs_seq: Observation sequence [B, T, obs_dim]
            action_seq: Action sequence [B, T]
            mask: Optional attention mask [B, T]

        Returns:
            Dictionary with policy, value, reward, next_state predictions
        """
        # Embed observations and actions
        obs_emb = self.obs_proj(obs_seq)
        act_emb = self.action_emb(action_seq)

        # Combine embeddings
        x = obs_emb + act_emb
        x = self.pos(x)

        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()

        # Transform
        h = self.transformer(x, mask=mask)

        # Use last position for predictions
        last = h[:, -1]

        return {
            "policy": F.softmax(self.policy_head(last), dim=-1),
            "value": self.value_head(last),
            "reward": self.reward_head(last),
            "next_state": self.state_head(last),
            "hidden": h,
        }

    def initial_inference(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial inference for MCTS root.

        Args:
            obs: Single observation [B, obs_dim]

        Returns:
            Tuple of (hidden_state, policy, value)
        """
        # Create single-step sequence
        obs_seq = obs.unsqueeze(1)  # [B, 1, obs_dim]
        # Use zero action for initial step
        action_seq = torch.zeros(
            obs.size(0), 1, dtype=torch.long, device=obs.device
        )

        out = self.forward(obs_seq, action_seq)

        return out["hidden"][:, -1], out["policy"], out["value"]

    def recurrent_inference(
        self,
        hidden: torch.Tensor,
        context_obs: torch.Tensor,
        context_actions: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference with context.

        Args:
            hidden: Current hidden state [B, d_model]
            context_obs: Previous observations [B, T, obs_dim]
            context_actions: Previous actions [B, T]
            action: New action [B]

        Returns:
            Tuple of (next_hidden, policy, value, reward)
        """
        # Append new action
        new_actions = torch.cat(
            [context_actions, action.unsqueeze(1)], dim=1
        )

        # Use dummy observation for new position
        dummy_obs = torch.zeros(
            context_obs.size(0), 1, self.obs_dim, device=context_obs.device
        )
        new_obs = torch.cat([context_obs, dummy_obs], dim=1)

        out = self.forward(new_obs, new_actions)

        return (
            out["hidden"][:, -1],
            out["policy"],
            out["value"],
            out["reward"],
        )


class MuZeroTransformerV2(nn.Module):
    """Enhanced Transformer MuZero with GPT-style architecture.

    Uses causal attention for autoregressive modeling of
    pipeline state evolution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # Token type embeddings
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.action_embed = nn.Embedding(action_dim + 1, d_model)  # +1 for start token
        self.reward_embed = nn.Linear(1, d_model)

        # Positional encoding
        self.pos = LearnedPositionalEncoding(d_model, max_len=512)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Prediction heads
        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Linear(d_model, 1)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with interleaved state-action-reward tokens.

        Args:
            states: State sequence [B, T, obs_dim]
            actions: Action sequence [B, T]
            rewards: Optional reward sequence [B, T, 1]

        Returns:
            Predictions dictionary
        """
        b, t, _ = states.shape

        # Embed each modality
        state_emb = self.obs_embed(states)
        action_emb = self.action_embed(actions)

        if rewards is None:
            rewards = torch.zeros(b, t, 1, device=states.device)
        reward_emb = self.reward_embed(rewards)

        # Interleave: [s_0, a_0, r_0, s_1, a_1, r_1, ...]
        # For simplicity, we'll stack and process
        x = state_emb + action_emb + reward_emb
        x = self.pos(x)

        # Causal mask
        mask = torch.triu(
            torch.ones(t, t, device=x.device), diagonal=1
        ).bool()

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln_f(x)
        last = x[:, -1]

        return {
            "policy": F.softmax(self.policy_head(last), dim=-1),
            "value": self.value_head(last),
            "reward": self.reward_head(last),
            "hidden": x,
        }
