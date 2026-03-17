"""RL-based pipeline optimizer module.

This module provides reinforcement learning components for autonomous
pipeline optimization that learns policies over DAG configurations
across runs.

Components:
- action_space: Discrete + parameterized action definitions
- state_encoder: State feature encoding
- model: Policy and value networks
- replay: Experience replay buffer
- trainer: A2C + replay trainer
- env: Pipeline environment wrapper
- optimize_rl: Main training loop
- policy_guard: Safety gates
"""

from transformation_portal.rl.action_space import (
    ACTION_TEMPLATES,
    RLAction,
    enumerate_actions,
)
from transformation_portal.rl.state_encoder import (
    DIFF_TYPES,
    SEVERITIES,
    encode_state,
    get_state_dim,
)

__all__ = [
    "ACTION_TEMPLATES",
    "RLAction",
    "enumerate_actions",
    "DIFF_TYPES",
    "SEVERITIES",
    "encode_state",
    "get_state_dim",
]
