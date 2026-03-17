"""Multi-agent environment: Joint action execution.

This module provides the environment wrapper for multi-agent RL
where multiple agents take actions simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from transformation_portal.rl.ma_action_space import NodeAction
from transformation_portal.rl.ma_state import encode_global, encode_state, get_node_config

logger = logging.getLogger(__name__)


@dataclass
class MAStepResult:
    """Result of multi-agent environment step."""

    states: dict[str, Any]  # node_id -> state vector
    global_state: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class MultiAgentEnv:
    """Multi-agent pipeline environment.

    Multiple agents (one per node) take actions simultaneously.
    Environment applies all actions and returns shared reward.

    Example:
        >>> env = MultiAgentEnv(run_fn, eval_fn, diff_fn, node_ids)
        >>> states = env.reset(pipeline)
        >>> result = env.step(joint_actions)
    """

    def __init__(
        self,
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        eval_fn: Callable[[dict[str, Any]], dict[str, float]],
        diff_fn: Callable[[dict[str, Any]], dict[str, Any]],
        node_ids: list[str],
        action_cost: float = 0.01,
        max_steps: int = 50,
    ) -> None:
        """Initialize environment.

        Args:
            run_fn: Pipeline runner function
            eval_fn: Evaluation function
            diff_fn: Semantic diff function
            node_ids: List of agent node IDs
            action_cost: Cost per action
            max_steps: Maximum steps per episode
        """
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.diff_fn = diff_fn
        self.node_ids = node_ids
        self.action_cost = action_cost
        self.max_steps = max_steps

        # Episode state
        self.current_pipeline: dict[str, Any] = {}
        self.current_metrics: dict[str, float] = {}
        self.current_diff: dict[str, Any] = {}
        self.prev_score: float = 0.0
        self.step_count: int = 0

    def reset(self, pipeline: dict[str, Any]) -> dict[str, Any]:
        """Reset environment.

        Args:
            pipeline: Initial pipeline configuration

        Returns:
            Dictionary of node_id -> initial state
        """
        self.current_pipeline = pipeline
        self.step_count = 0

        # Run initial pipeline
        output = self.run_fn(pipeline)
        self.current_metrics = self.eval_fn(output)
        self.current_diff = self.diff_fn(pipeline)

        self.prev_score = self.current_metrics.get("score", 0.0)

        # Encode states for each agent
        states = {}
        for node_id in self.node_ids:
            node_cfg = get_node_config(pipeline, node_id)
            states[node_id] = encode_state(
                node_cfg,
                self.current_metrics,
                self.current_diff,
                node_id,
            )

        logger.info(
            "Multi-agent env reset: %d agents, initial score=%.4f",
            len(self.node_ids),
            self.prev_score,
        )

        return states

    def step(self, joint_actions: list[NodeAction]) -> MAStepResult:
        """Take joint action in environment.

        Args:
            joint_actions: List of actions from all agents

        Returns:
            MAStepResult with new states, reward, done
        """
        from transformation_portal.evals.self_healing import FixSuggestion
        from transformation_portal.execution_graph.patcher import apply_fix

        # Apply all actions
        new_pipeline = self.current_pipeline
        applied_count = 0

        for action in joint_actions:
            # Convert to fix
            fix = FixSuggestion(
                type=action.action_type.split("_")[0],
                target_node=action.node_id,
                action=action.action_type,
                params=action.params,
                confidence=0.8,
                rationale=f"MA action: {action.action_type}",
            )

            try:
                new_pipeline = apply_fix(new_pipeline, fix)
                applied_count += 1
            except Exception as e:
                logger.warning("Action %s failed: %s", action.action_type, e)

        # Run modified pipeline
        try:
            output = self.run_fn(new_pipeline)
            metrics = self.eval_fn(output)
            diff = self.diff_fn(new_pipeline)

            new_score = metrics.get("score", 0.0)

            # Compute reward (shared)
            reward = new_score - self.prev_score - self.action_cost * applied_count

            success = True
            error = None

        except Exception as e:
            logger.error("Pipeline run failed: %s", e)
            reward = -0.5
            metrics = self.current_metrics
            diff = self.current_diff
            new_pipeline = self.current_pipeline
            new_score = self.prev_score
            success = False
            error = str(e)

        # Update state
        self.current_pipeline = new_pipeline
        self.current_metrics = metrics
        self.current_diff = diff
        self.prev_score = new_score
        self.step_count += 1

        # Check termination
        done = self.step_count >= self.max_steps

        # Encode new states
        states = {}
        for node_id in self.node_ids:
            node_cfg = get_node_config(new_pipeline, node_id)
            states[node_id] = encode_state(node_cfg, metrics, diff, node_id)

        global_state = encode_global(metrics, diff)

        logger.debug(
            "MA step %d: %d actions, reward=%.4f, score=%.4f",
            self.step_count,
            applied_count,
            reward,
            new_score,
        )

        return MAStepResult(
            states=states,
            global_state=global_state,
            reward=reward,
            done=done,
            info={
                "success": success,
                "error": error,
                "score": new_score,
                "applied_actions": applied_count,
            },
        )

    @property
    def n_agents(self) -> int:
        """Get number of agents."""
        return len(self.node_ids)


class MockMultiAgentEnv(MultiAgentEnv):
    """Mock environment for testing."""

    def __init__(self, node_ids: list[str]) -> None:
        """Initialize mock environment."""

        def mock_run(p):
            return {"output": "mock"}

        def mock_eval(o):
            import random

            return {"score": random.uniform(0.3, 0.9)}

        def mock_diff(p):
            return {"changes": []}

        super().__init__(
            run_fn=mock_run,
            eval_fn=mock_eval,
            diff_fn=mock_diff,
            node_ids=node_ids,
        )
