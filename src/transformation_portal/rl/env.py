"""RL environment wrapper: Pipeline execution as MDP.

This module wraps the pipeline execution system as a Markov Decision
Process environment for RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from transformation_portal.rl.action_space import RLAction
from transformation_portal.rl.state_encoder import encode_state

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of environment step."""

    state: Any  # Encoded state vector
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class PipelineEnv:
    """Pipeline execution environment for RL.

    Wraps pipeline execution as an MDP:
    - State: Encoded pipeline config + metrics + diff
    - Action: Fix application
    - Reward: Score improvement - cost
    - Transition: Apply fix -> re-run -> evaluate

    Example:
        >>> env = PipelineEnv(run_fn, eval_fn, diff_fn, actions)
        >>> state = env.reset(initial_pipeline)
        >>> result = env.step(action_idx)
    """

    def __init__(
        self,
        run_fn: Callable[[dict[str, Any]], dict[str, Any]],
        eval_fn: Callable[[dict[str, Any]], dict[str, float]],
        diff_fn: Callable[[dict[str, Any]], dict[str, Any]],
        action_list: list[RLAction],
        action_cost: float = 0.01,
        max_steps: int = 50,
    ) -> None:
        """Initialize environment.

        Args:
            run_fn: Pipeline runner function
            eval_fn: Evaluation function (returns metrics dict)
            diff_fn: Semantic diff function
            action_list: List of available actions
            action_cost: Cost per action (subtracted from reward)
            max_steps: Maximum steps per episode
        """
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.diff_fn = diff_fn
        self.actions = action_list
        self.action_cost = action_cost
        self.max_steps = max_steps

        # Episode state
        self.current_pipeline: dict[str, Any] = {}
        self.current_metrics: dict[str, float] = {}
        self.current_diff: dict[str, Any] = {}
        self.prev_score: float = 0.0
        self.step_count: int = 0
        self.score_history: list[float] = []

    def reset(self, pipeline: dict[str, Any]) -> Any:
        """Reset environment with new pipeline.

        Args:
            pipeline: Initial pipeline configuration

        Returns:
            Initial state vector
        """
        self.current_pipeline = pipeline
        self.step_count = 0
        self.score_history = []

        # Run initial pipeline
        output = self.run_fn(pipeline)
        self.current_metrics = self.eval_fn(output)
        self.current_diff = self.diff_fn(pipeline)

        self.prev_score = self.current_metrics.get("score", 0.0)
        self.score_history.append(self.prev_score)

        # Encode state
        state = encode_state(
            self.current_pipeline,
            self.current_metrics,
            self.current_diff,
            self.score_history,
        )

        logger.info("Environment reset: initial score=%.4f", self.prev_score)

        return state

    def step(self, action_idx: int) -> StepResult:
        """Take action in environment.

        Args:
            action_idx: Index of action to take

        Returns:
            StepResult with new state, reward, done flag
        """
        from transformation_portal.execution_graph.patcher import apply_fix
        from transformation_portal.evals.self_healing import FixSuggestion

        action = self.actions[action_idx]

        # Convert action to fix suggestion
        fix = FixSuggestion(
            type=action.action_type.split("_")[0],
            target_node=action.node,
            action=action.action_type,
            params=action.params,
            confidence=0.8,
            rationale=f"RL action: {action.action_type}",
        )

        try:
            # Apply fix
            new_pipeline = apply_fix(self.current_pipeline, fix)

            # Run modified pipeline
            output = self.run_fn(new_pipeline)
            metrics = self.eval_fn(output)
            diff = self.diff_fn(new_pipeline)

            new_score = metrics.get("score", 0.0)

            # Compute reward
            reward = new_score - self.prev_score - self.action_cost

            # Update state
            self.current_pipeline = new_pipeline
            self.current_metrics = metrics
            self.current_diff = diff
            self.prev_score = new_score
            self.score_history.append(new_score)

            success = True
            error = None

        except Exception as e:
            logger.warning("Action failed: %s", e)

            # Penalize failed actions
            reward = -0.1
            success = False
            error = str(e)

        self.step_count += 1

        # Check termination
        done = self.step_count >= self.max_steps

        # Encode new state
        state = encode_state(
            self.current_pipeline,
            self.current_metrics,
            self.current_diff,
            self.score_history,
        )

        logger.debug(
            "Step %d: action=%s, reward=%.4f, score=%.4f",
            self.step_count,
            action.action_type,
            reward,
            self.prev_score,
        )

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            info={
                "success": success,
                "error": error,
                "score": self.prev_score,
                "action": action.to_dict(),
            },
        )

    def get_action_mask(self) -> list[bool]:
        """Get mask of valid actions.

        Returns:
            Boolean mask for each action
        """
        # For now, all actions are valid
        # Could be extended to mask invalid actions based on pipeline state
        return [True] * len(self.actions)

    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        from transformation_portal.rl.state_encoder import get_state_dim

        return get_state_dim()

    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return len(self.actions)


class MockPipelineEnv(PipelineEnv):
    """Mock environment for testing without real pipeline execution."""

    def __init__(self, action_list: list[RLAction]) -> None:
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
            action_list=action_list,
        )
