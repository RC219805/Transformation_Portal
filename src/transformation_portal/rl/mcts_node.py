"""MCTS node for tree search planning.

This module provides the node structure for Monte Carlo Tree Search
(MCTS) used in AlphaZero-style pipeline planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCTSNode:
    """Node in MCTS search tree.

    Represents a state in the pipeline optimization search tree.
    Stores visit counts, values, and policy priors for PUCT selection.

    Attributes:
        state: Encoded state vector
        parent: Parent node (None for root)
        action_from_parent: Action that led to this node
        children: Dictionary mapping actions to child nodes
        N: Visit count
        W: Total value accumulated
        Q: Mean value (W/N)
        P: Prior probabilities for each action
    """

    state: Any
    parent: "MCTSNode | None" = None
    action_from_parent: int | None = None

    # Statistics
    N: int = 0  # Visit count
    W: float = 0.0  # Total value
    Q: float = 0.0  # Mean value

    # Prior probabilities and children
    P: dict[int, float] = field(default_factory=dict)
    children: dict[int, "MCTSNode | None"] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children expanded)."""
        return len(self.children) == 0

    def is_expanded(self) -> bool:
        """Check if node has been expanded with actions."""
        return len(self.P) > 0

    def expand(self, actions: list[int], priors: list[float]) -> None:
        """Expand node with available actions and prior probabilities.

        Args:
            actions: List of action indices
            priors: Prior probabilities for each action
        """
        for action, prior in zip(actions, priors):
            self.P[action] = prior
            self.children[action] = None  # Not yet visited

    def update(self, value: float) -> None:
        """Update node statistics after simulation.

        Args:
            value: Value to backpropagate
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def get_child(self, action: int) -> "MCTSNode | None":
        """Get child node for action.

        Args:
            action: Action index

        Returns:
            Child node or None if not visited
        """
        return self.children.get(action)

    def add_child(self, action: int, child_state: Any) -> "MCTSNode":
        """Add child node for action.

        Args:
            action: Action index
            child_state: State of child node

        Returns:
            New child node
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            action_from_parent=action,
        )
        self.children[action] = child
        return child

    def get_action_visits(self) -> dict[int, int]:
        """Get visit counts for each action.

        Returns:
            Dictionary mapping action -> visit count
        """
        visits = {}
        for action, child in self.children.items():
            visits[action] = child.N if child else 0
        return visits

    def get_action_values(self) -> dict[int, float]:
        """Get mean values for each action.

        Returns:
            Dictionary mapping action -> mean value
        """
        values = {}
        for action, child in self.children.items():
            values[action] = child.Q if child else 0.0
        return values

    def best_action(self, by: str = "visits") -> int | None:
        """Get best action according to criterion.

        Args:
            by: "visits" for most visited, "value" for highest value

        Returns:
            Best action index or None
        """
        if not self.children:
            return None

        if by == "value":
            stats = self.get_action_values()
        else:
            stats = self.get_action_visits()

        if not stats:
            return None

        return max(stats.items(), key=lambda x: x[1])[0]

    def get_policy(self, temperature: float = 1.0) -> dict[int, float]:
        """Get policy distribution from visit counts.

        The probability of an action is ``N(a) ** (1 / temperature)``
        normalised over all actions (AlphaZero visit-count policy).

        Args:
            temperature: Visit-count temperature (0 = greedy with ties split
                equally, 1 = proportional to visit counts)

        Returns:
            Dictionary mapping action -> probability (sums to 1)
        """
        visits = self.get_action_visits()

        if not visits:
            return {}

        best = max(visits.values())

        if temperature <= 0:
            # Greedy; ties share the probability mass equally
            n_best = sum(1 for v in visits.values() if v == best)
            return {a: (1.0 / n_best if v == best else 0.0) for a, v in visits.items()}

        if best <= 0:
            # No visits yet: uniform
            return {a: 1.0 / len(visits) for a in visits}

        # (N / N_max) ** (1 / T) is N ** (1 / T) rescaled, so it cannot overflow
        weights = {a: (v / best) ** (1.0 / temperature) for a, v in visits.items()}
        total = sum(weights.values())

        return {a: w / total for a, w in weights.items()}

    def depth(self) -> int:
        """Get depth of node in tree."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def path_to_root(self) -> list["MCTSNode"]:
        """Get path from this node to root.

        Returns:
            List of nodes from this node to root
        """
        path = [self]
        node = self
        while node.parent is not None:
            path.append(node.parent)
            node = node.parent
        return path

    def __repr__(self) -> str:
        return f"MCTSNode(N={self.N}, Q={self.Q:.3f}, children={len(self.children)})"
