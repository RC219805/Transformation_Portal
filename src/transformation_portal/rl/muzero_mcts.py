"""MuZero MCTS implementation for latent-space planning.

This module implements Monte Carlo Tree Search optimized for
MuZero-style latent dynamics models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from transformation_portal.rl.muzero_model import MuZeroModel


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 50
    c_puct: float = 1.5
    discount: float = 0.99
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


class MuZeroNode:
    """Node in the MCTS tree.

    Stores visit counts, value estimates, and children
    for tree search in latent space.
    """

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MuZeroNode] = {}
        self.reward = 0.0
        self.latent_state: torch.Tensor | None = None

    def value(self) -> float:
        """Return mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0


def ucb_score(parent: MuZeroNode, child: MuZeroNode, config: MCTSConfig) -> float:
    """Compute UCB score for action selection.

    Uses PUCT formula from AlphaZero/MuZero.
    """
    exploration_bonus = math.log((parent.visit_count + config.c_puct + 1) / config.c_puct) + config.c_puct
    exploration_bonus *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = exploration_bonus * child.prior
    value_score = child.value()

    return prior_score + value_score


def select_child(node: MuZeroNode, config: MCTSConfig) -> tuple[int, MuZeroNode]:
    """Select best child using UCB."""
    best_score = -float("inf")
    best_action = -1
    best_child = None

    for action, child in node.children.items():
        score = ucb_score(node, child, config)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is None:
        raise ValueError("No children to select from")

    return best_action, best_child


def expand_node(
    node: MuZeroNode,
    latent_state: "torch.Tensor",
    policy: "torch.Tensor",
) -> None:
    """Expand node with children for each action.

    Args:
        node: Node to expand
        latent_state: Latent state at this node
        policy: Policy distribution over actions
    """
    node.latent_state = latent_state
    policy_np = policy[0].detach().cpu().numpy()

    for action, prior in enumerate(policy_np):
        node.children[action] = MuZeroNode(prior=prior)


def add_exploration_noise(
    node: MuZeroNode,
    config: MCTSConfig,
) -> None:
    """Add Dirichlet noise to root node priors for exploration."""
    import numpy as np

    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))

    frac = config.root_exploration_fraction
    for idx, action in enumerate(actions):
        node.children[action].prior = node.children[action].prior * (1 - frac) + noise[idx] * frac


def backpropagate(
    path: list[MuZeroNode],
    value: float,
    discount: float,
) -> None:
    """Backpropagate value through the search path."""
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def run_mcts(
    model: "MuZeroModel",
    obs: "torch.Tensor",
    config: MCTSConfig | None = None,
) -> tuple[int, dict[int, float]]:
    """Run MCTS search from observation.

    Args:
        model: MuZero model for inference
        obs: Initial observation [1, obs_dim]
        config: MCTS configuration

    Returns:
        Tuple of (best_action, action_visit_counts)
    """
    import torch

    if config is None:
        config = MCTSConfig()

    # Initial inference at root
    s, p, v = model.initial_inference(obs)
    root = MuZeroNode(prior=1.0)
    expand_node(root, s, p)
    add_exploration_noise(root, config)

    # Run simulations
    for _ in range(config.num_simulations):
        node = root
        path = [node]
        state = root.latent_state

        # SELECT: traverse tree
        while node.is_expanded():
            action, node = select_child(node, config)
            path.append(node)

            if node.latent_state is not None:
                state = node.latent_state

        # EXPAND: use dynamics model
        parent = path[-2] if len(path) > 1 else root
        action = next(a for a, c in parent.children.items() if c is node)

        s2, p2, v2, r = model.recurrent_inference(state, torch.tensor([action], device=state.device))

        node.reward = r.item()
        expand_node(node, s2, p2)

        # BACKUP
        backpropagate(path, v2.item(), config.discount)

    # Select action with most visits
    visit_counts = {action: child.visit_count for action, child in root.children.items()}
    best_action = max(visit_counts, key=lambda a: visit_counts[a])

    return best_action, visit_counts


def get_action_probs(
    visit_counts: dict[int, float],
    temperature: float = 1.0,
) -> dict[int, float]:
    """Convert visit counts to action probabilities.

    Args:
        visit_counts: Visit count per action
        temperature: Temperature for softmax (0 = greedy, 1 = proportional)

    Returns:
        Action probability distribution
    """
    total = sum(v ** (1 / temperature) for v in visit_counts.values())
    return {a: (v ** (1 / temperature)) / total for a, v in visit_counts.items()}
