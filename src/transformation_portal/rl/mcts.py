"""Monte Carlo Tree Search (MCTS) for pipeline optimization.

This module provides AlphaZero-style MCTS planning that combines:
- World model for fast simulation
- Policy prior from learned policies
- PUCT selection for exploration/exploitation balance
- Tree search for lookahead planning

Enables near-optimal decision making by searching over possible
pipeline modifications before committing to execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

from transformation_portal.rl.mcts_node import MCTSNode

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 50  # Number of MCTS simulations per search
    c_puct: float = 1.5  # PUCT exploration constant
    rollout_depth: int = 3  # Depth for world model rollout
    temperature: float = 1.0  # Temperature for action selection
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha (exploration)
    dirichlet_weight: float = 0.25  # Weight of Dirichlet noise


def puct_select(node: MCTSNode, c_puct: float = 1.5) -> int | None:
    """Select action using PUCT formula.

    PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

    Args:
        node: Current node
        c_puct: Exploration constant

    Returns:
        Selected action index
    """
    if not node.children:
        return None

    best_score = float("-inf")
    best_action = None

    sqrt_n = math.sqrt(node.N + 1)

    for action in node.children:
        child = node.children[action]

        # Q-value (exploitation)
        q = child.Q if child else 0.0

        # Visit count
        n = child.N if child else 0

        # Prior (from policy)
        p = node.P.get(action, 0.0)

        # PUCT score
        u = c_puct * p * sqrt_n / (1 + n)
        score = q + u

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def add_dirichlet_noise(
    priors: dict[int, float],
    alpha: float = 0.3,
    weight: float = 0.25,
) -> dict[int, float]:
    """Add Dirichlet noise to priors for exploration.

    Args:
        priors: Action prior probabilities
        alpha: Dirichlet alpha parameter
        weight: Weight of noise vs original priors

    Returns:
        Noisy priors
    """
    try:
        import numpy as np
    except ImportError:
        return priors

    actions = list(priors.keys())
    if not actions:
        return priors

    noise = np.random.dirichlet([alpha] * len(actions))

    noisy_priors = {}
    for i, action in enumerate(actions):
        noisy_priors[action] = (1 - weight) * priors[action] + weight * noise[i]

    return noisy_priors


class MCTS:
    """Monte Carlo Tree Search planner.

    Uses a world model for fast simulation and policy priors
    for guiding the search. Implements AlphaZero-style PUCT
    selection and backup.

    Example:
        >>> mcts = MCTS(world_model, action_fn, config)
        >>> best_action = mcts.search(initial_state)
    """

    def __init__(
        self,
        world_model: Any,
        action_fn: Callable[[Any], tuple[list[int], list[float]]],
        config: MCTSConfig | None = None,
    ) -> None:
        """Initialize MCTS.

        Args:
            world_model: World model for simulation
            action_fn: Function that returns (actions, priors) for a state
            config: MCTS configuration
        """
        self.world_model = world_model
        self.action_fn = action_fn
        self.config = config or MCTSConfig()

    def search(self, root_state: Any) -> int:
        """Run MCTS search from root state.

        Args:
            root_state: Initial state vector

        Returns:
            Best action index
        """
        # Create root node
        root = MCTSNode(state=root_state)

        # Initial expansion
        actions, priors = self.action_fn(root_state)
        if not actions:
            logger.warning("No actions available for MCTS")
            return 0

        # Normalize priors
        prior_sum = sum(priors)
        if prior_sum > 0:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(actions)] * len(actions)

        root.expand(actions, priors)

        # Add exploration noise at root
        if self.config.dirichlet_weight > 0:
            root.P = add_dirichlet_noise(
                root.P,
                self.config.dirichlet_alpha,
                self.config.dirichlet_weight,
            )

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)

        # Select best action
        best_action = root.best_action(by="visits")

        if best_action is None:
            best_action = actions[0] if actions else 0

        return best_action

    def _simulate(self, root: MCTSNode) -> None:
        """Run one MCTS simulation.

        Steps:
        1. SELECT: Traverse tree using PUCT
        2. EXPAND: Expand leaf node
        3. SIMULATE: Rollout using world model
        4. BACKUP: Update values along path
        """
        node = root
        path = [node]

        # SELECT: Traverse to leaf
        while not node.is_leaf():
            action = puct_select(node, self.config.c_puct)

            if action is None:
                break

            child = node.get_child(action)

            if child is None:
                # Need to expand this action
                break

            node = child
            path.append(node)

        # EXPAND: If at unexpanded action, create child
        if node.is_expanded():
            action = puct_select(node, self.config.c_puct)

            if action is not None and node.children[action] is None:
                # Simulate transition
                pred = self.world_model.predict(node.state, action)
                child_state = pred.next_state

                # Create child node
                child = node.add_child(action, child_state)

                # Expand child with actions
                child_actions, child_priors = self.action_fn(child_state)
                if child_actions:
                    prior_sum = sum(child_priors)
                    if prior_sum > 0:
                        child_priors = [p / prior_sum for p in child_priors]
                    else:
                        child_priors = [1.0 / len(child_actions)] * len(child_actions)
                    child.expand(child_actions, child_priors)

                path.append(child)
                node = child

        # SIMULATE: Rollout from leaf
        value = self._rollout(node.state)

        # BACKUP: Update values along path
        for n in reversed(path):
            n.update(value)

    def _rollout(self, state: Any, depth: int | None = None) -> float:
        """Rollout from state using world model.

        Args:
            state: Starting state
            depth: Rollout depth (defaults to config)

        Returns:
            Average predicted score over rollout
        """
        depth = depth or self.config.rollout_depth
        total_score = 0.0
        current_state = state

        for _ in range(depth):
            # Get action (greedy from prior)
            actions, priors = self.action_fn(current_state)

            if not actions:
                break

            # Select action with highest prior
            best_idx = priors.index(max(priors))
            action = actions[best_idx]

            # Simulate step
            pred = self.world_model.predict(current_state, action)
            total_score += pred.score
            current_state = pred.next_state

        return total_score / depth if depth > 0 else 0.0

    def get_policy(self, root_state: Any) -> dict[int, float]:
        """Get action probabilities after search.

        Args:
            root_state: State to search from

        Returns:
            Dictionary mapping action -> probability
        """
        # Create and expand root
        root = MCTSNode(state=root_state)
        actions, priors = self.action_fn(root_state)

        if not actions:
            return {}

        prior_sum = sum(priors)
        if prior_sum > 0:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(actions)] * len(actions)

        root.expand(actions, priors)

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)

        # Return policy from visit counts
        return root.get_policy(self.config.temperature)


def mcts_search(
    root_state: Any,
    world_model: Any,
    action_fn: Callable[[Any], tuple[list[int], list[float]]],
    num_simulations: int = 50,
    c_puct: float = 1.5,
) -> int:
    """Convenience function for MCTS search.

    Args:
        root_state: Initial state
        world_model: World model for simulation
        action_fn: Function returning (actions, priors)
        num_simulations: Number of simulations
        c_puct: PUCT exploration constant

    Returns:
        Best action index
    """
    config = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct)
    mcts = MCTS(world_model, action_fn, config)
    return mcts.search(root_state)


class MCTSPipelineOptimizer:
    """MCTS-based pipeline optimizer.

    Combines MCTS planning with real execution for hybrid
    optimization that balances speed and accuracy.

    Example:
        >>> optimizer = MCTSPipelineOptimizer(world_model, run_fn, eval_fn)
        >>> best_pipeline = optimizer.optimize(initial_pipeline)
    """

    def __init__(
        self,
        world_model: Any,
        run_fn: Callable,
        eval_fn: Callable,
        action_list: list[Any],
        encode_fn: Callable,
        config: MCTSConfig | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            world_model: World model for simulation
            run_fn: Pipeline runner function
            eval_fn: Evaluation function
            action_list: List of available actions
            encode_fn: Function to encode pipeline to state
            config: MCTS configuration
        """
        self.world_model = world_model
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.actions = action_list
        self.encode_fn = encode_fn
        self.config = config or MCTSConfig()

    def _action_fn(self, state: Any) -> tuple[list[int], list[float]]:
        """Get actions and uniform priors."""
        action_indices = list(range(len(self.actions)))
        priors = [1.0 / len(self.actions)] * len(self.actions)
        return action_indices, priors

    def plan_step(self, pipeline: dict[str, Any]) -> int:
        """Plan single step using MCTS.

        Args:
            pipeline: Current pipeline

        Returns:
            Best action index
        """
        state = self.encode_fn(pipeline)
        mcts = MCTS(self.world_model, self._action_fn, self.config)
        return mcts.search(state)

    def optimize(
        self,
        pipeline: dict[str, Any],
        steps: int = 5,
        validate_top_k: int = 2,
    ) -> dict[str, Any]:
        """Optimize pipeline using MCTS + real validation.

        Args:
            pipeline: Initial pipeline
            steps: Number of planning steps
            validate_top_k: Number of candidates to validate with real execution

        Returns:
            Optimized pipeline
        """
        from transformation_portal.execution_graph.patcher import apply_fix

        # Generate candidates using MCTS
        candidates = []

        for _ in range(validate_top_k * 2):
            current = pipeline

            for _ in range(steps):
                action_idx = self.plan_step(current)
                action = self.actions[action_idx]

                try:
                    current = apply_fix(current, action)
                except Exception as e:
                    logger.warning("Apply fix failed: %s", e)
                    break

            # Predict score
            state = self.encode_fn(current)
            pred = self.world_model.predict(state, 0)  # Dummy action for scoring
            candidates.append((current, pred.score))

        # Select top-K by predicted score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:validate_top_k]

        # Validate with real execution
        best_pipeline = pipeline
        best_score = float("-inf")

        for candidate, _ in top_candidates:
            try:
                output = self.run_fn(candidate)
                metrics = self.eval_fn(output)
                score = metrics.get("score", 0.0)

                if score > best_score:
                    best_score = score
                    best_pipeline = candidate

            except Exception as e:
                logger.warning("Validation failed: %s", e)

        return best_pipeline
