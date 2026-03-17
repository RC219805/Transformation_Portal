"""RL optimization: Main training loop for pipeline optimization.

This module provides the main training loop that integrates all RL
components for autonomous pipeline optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from transformation_portal.rl.action_space import enumerate_actions
from transformation_portal.rl.env import PipelineEnv
from transformation_portal.rl.model import PolicyValueNet, create_model
from transformation_portal.rl.replay import ReplayBuffer, Transition
from transformation_portal.rl.state_encoder import encode_state, get_state_dim
from transformation_portal.rl.trainer import RLTrainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class RLOptimizationResult:
    """Result of RL optimization."""

    best_pipeline: dict[str, Any]
    best_score: float
    iterations: int
    total_reward: float
    score_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_score": self.best_score,
            "iterations": self.iterations,
            "total_reward": self.total_reward,
            "score_history": self.score_history,
            "loss_history": self.loss_history,
        }


@dataclass
class RLOptimizationConfig:
    """Configuration for RL optimization."""

    max_iterations: int = 50
    batch_size: int = 32
    min_buffer_size: int = 100
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    save_model: bool = True
    model_path: str = "models/rl_optimizer.pt"
    log_frequency: int = 10


def train_rl(
    env: PipelineEnv,
    trainer: RLTrainer,
    initial_pipeline: dict[str, Any],
    config: RLOptimizationConfig | None = None,
) -> RLOptimizationResult:
    """Train RL optimizer on pipeline.

    Args:
        env: Pipeline environment
        trainer: RL trainer
        initial_pipeline: Starting pipeline configuration
        config: Optimization configuration

    Returns:
        RLOptimizationResult with best pipeline

    Example:
        >>> result = train_rl(env, trainer, pipeline)
        >>> print(f"Best score: {result.best_score:.4f}")
    """
    config = config or RLOptimizationConfig()

    # Reset environment
    state = env.reset(initial_pipeline)

    # Track best
    best_pipeline = initial_pipeline
    best_score = env.prev_score
    total_reward = 0.0

    score_history = [best_score]
    loss_history: list[float] = []

    logger.info("Starting RL optimization: %d iterations", config.max_iterations)

    for step in range(config.max_iterations):
        # Select action
        action_idx, log_prob, value = trainer.select_action(state)

        # Take step
        result = env.step(action_idx)

        # Store transition
        trainer.store_transition(
            state=state,
            action=action_idx,
            reward=result.reward,
            next_state=result.state,
            done=result.done,
            log_prob=log_prob,
            value=value,
        )

        total_reward += result.reward
        state = result.state

        # Track best
        current_score = result.info.get("score", 0.0)
        if current_score > best_score:
            best_score = current_score
            best_pipeline = env.current_pipeline
            logger.info("New best score: %.4f at step %d", best_score, step)

        score_history.append(current_score)

        # Training update
        loss_dict = trainer.update()
        if loss_dict is not None:
            loss_history.append(loss_dict["total_loss"])

        # Logging
        if step % config.log_frequency == 0:
            logger.info(
                "Step %d: score=%.4f, reward=%.4f, total_reward=%.4f",
                step,
                current_score,
                result.reward,
                total_reward,
            )

        # Check termination
        if result.done:
            logger.info("Episode done at step %d", step)
            break

    # Save model
    if config.save_model:
        try:
            model_path = Path(config.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(str(model_path))
        except Exception as e:
            logger.warning("Failed to save model: %s", e)

    return RLOptimizationResult(
        best_pipeline=best_pipeline,
        best_score=best_score,
        iterations=step + 1,
        total_reward=total_reward,
        score_history=score_history,
        loss_history=loss_history,
    )


def create_rl_optimizer(
    run_fn: Callable[[dict[str, Any]], dict[str, Any]],
    eval_fn: Callable[[dict[str, Any]], dict[str, float]],
    diff_fn: Callable[[dict[str, Any]], dict[str, Any]],
    model_path: str | None = None,
) -> tuple[PipelineEnv, RLTrainer]:
    """Create RL optimizer components.

    Args:
        run_fn: Pipeline runner function
        eval_fn: Evaluation function
        diff_fn: Semantic diff function
        model_path: Optional path to load existing model

    Returns:
        Tuple of (environment, trainer)

    Example:
        >>> env, trainer = create_rl_optimizer(run_fn, eval_fn, diff_fn)
        >>> result = train_rl(env, trainer, initial_pipeline)
    """
    # Enumerate actions
    actions = enumerate_actions()
    logger.info("Action space: %d actions", len(actions))

    # Create environment
    env = PipelineEnv(
        run_fn=run_fn,
        eval_fn=eval_fn,
        diff_fn=diff_fn,
        action_list=actions,
    )

    # Get dimensions
    state_dim = get_state_dim()
    action_dim = len(actions)

    # Create model
    model = create_model(state_dim, action_dim)

    # Load existing model if provided
    if model_path:
        try:
            model = PolicyValueNet.load(model_path)
            logger.info("Loaded model from %s", model_path)
        except Exception as e:
            logger.warning("Failed to load model: %s", e)

    # Create trainer
    trainer = RLTrainer(model, actions)

    return env, trainer


def optimize_pipeline_rl(
    pipeline: dict[str, Any],
    run_fn: Callable[[dict[str, Any]], dict[str, Any]],
    eval_fn: Callable[[dict[str, Any]], dict[str, float]],
    diff_fn: Callable[[dict[str, Any]], dict[str, Any]],
    config: RLOptimizationConfig | None = None,
    model_path: str | None = None,
) -> RLOptimizationResult:
    """High-level function to optimize pipeline using RL.

    Args:
        pipeline: Initial pipeline configuration
        run_fn: Pipeline runner
        eval_fn: Evaluation function
        diff_fn: Diff function
        config: Optimization config
        model_path: Optional model path

    Returns:
        RLOptimizationResult

    Example:
        >>> result = optimize_pipeline_rl(pipeline, run, eval, diff)
    """
    env, trainer = create_rl_optimizer(run_fn, eval_fn, diff_fn, model_path)
    return train_rl(env, trainer, pipeline, config)
