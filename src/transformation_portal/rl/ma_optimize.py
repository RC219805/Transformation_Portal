"""Multi-agent optimization: Main training loop.

This module provides the orchestration for multi-agent RL training
where each node has a specialist agent that learns cooperatively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from transformation_portal.rl.ma_action_space import (
    DEFAULT_AGENT_NODES,
    NodeAction,
    enumerate_node_actions,
)
from transformation_portal.rl.ma_comm import MessageBus
from transformation_portal.rl.ma_env import MultiAgentEnv
from transformation_portal.rl.ma_model import AgentNet, create_agent
from transformation_portal.rl.ma_qmix import CentralCritic, create_critic
from transformation_portal.rl.ma_state import get_node_config, get_state_dim
from transformation_portal.rl.ma_trainer import AgentTrainer, CentralizedTrainer

logger = logging.getLogger(__name__)


@dataclass
class MAOptimizationResult:
    """Result of multi-agent optimization."""

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
class AgentState:
    """State for a single agent."""

    model: Any
    trainer: AgentTrainer | None
    actions: list[NodeAction]
    replay: list[tuple] = field(default_factory=list)
    last_transition: tuple | None = None


def create_agents(
    node_ids: list[str] | None = None,
    state_dim: int | None = None,
) -> dict[str, AgentState]:
    """Create agent states for each node.

    Args:
        node_ids: List of node IDs (defaults to DEFAULT_AGENT_NODES)
        state_dim: State dimension (defaults to auto-computed)

    Returns:
        Dictionary of node_id -> AgentState
    """
    node_ids = node_ids or DEFAULT_AGENT_NODES
    state_dim = state_dim or get_state_dim()

    agents = {}

    for node_id in node_ids:
        actions = enumerate_node_actions(node_id)
        action_dim = len(actions)

        if action_dim == 0:
            logger.warning("No actions for node %s, skipping", node_id)
            continue

        model = create_agent(state_dim, action_dim)
        trainer = AgentTrainer(model, actions)

        agents[node_id] = AgentState(
            model=model,
            trainer=trainer,
            actions=actions,
        )

        logger.info("Created agent for %s: %d actions", node_id, action_dim)

    return agents


def optimize_multi_agent(
    env: MultiAgentEnv,
    agents: dict[str, AgentState],
    initial_pipeline: dict[str, Any],
    iterations: int = 20,
    batch_size: int = 32,
    use_communication: bool = True,
) -> MAOptimizationResult:
    """Run multi-agent optimization loop.

    Args:
        env: Multi-agent environment
        agents: Dictionary of agent states
        initial_pipeline: Starting pipeline
        iterations: Number of training iterations
        batch_size: Batch size for training
        use_communication: Whether to use message bus

    Returns:
        MAOptimizationResult
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for multi-agent optimization")

    pipeline = initial_pipeline
    bus = MessageBus() if use_communication else None

    # Reset environment
    states = env.reset(pipeline)

    best_pipeline = pipeline
    best_score = env.prev_score
    total_reward = 0.0

    score_history = [best_score]
    loss_history: list[float] = []

    logger.info(
        "Starting multi-agent optimization: %d agents, %d iterations",
        len(agents),
        iterations,
    )

    for t in range(iterations):
        joint_actions: list[NodeAction] = []

        # Each agent selects action
        for node_id, agent in agents.items():
            if node_id not in states:
                continue

            s = states[node_id]
            a_idx, logp, v = agent.model.act(torch.tensor(s).unsqueeze(0))
            action = agent.actions[a_idx]

            joint_actions.append(action)

            # Publish intent
            if bus:
                bus.publish(node_id, {"intent": action.action_type})

            # Store for training
            agent.last_transition = (s, a_idx, logp, v)

        # Environment step
        result = env.step(joint_actions)
        reward = result.reward
        total_reward += reward

        # Store transitions
        for node_id, agent in agents.items():
            if agent.last_transition is None:
                continue
            if node_id not in result.states:
                continue

            s, a_idx, logp, v = agent.last_transition
            s2 = result.states[node_id]

            agent.replay.append((s, a_idx, reward, s2, result.done, logp, v))

        # Training updates
        for node_id, agent in agents.items():
            if agent.trainer is None:
                continue

            buf = agent.replay
            if len(buf) >= 10:
                batch = buf[-batch_size:]
                loss = agent.trainer.step(batch)
                loss_history.append(loss)

        # Track best
        current_score = result.info.get("score", 0.0)
        if current_score > best_score:
            best_score = current_score
            best_pipeline = env.current_pipeline
            logger.info("New best score: %.4f at iteration %d", best_score, t)

        score_history.append(current_score)

        # Clear message bus
        if bus:
            bus.clear()

        # Update states
        states = result.states
        pipeline = env.current_pipeline

        if t % 5 == 0:
            logger.info(
                "Iteration %d: score=%.4f, reward=%.4f",
                t,
                current_score,
                reward,
            )

        if result.done:
            logger.info("Episode done at iteration %d", t)
            break

    return MAOptimizationResult(
        best_pipeline=best_pipeline,
        best_score=best_score,
        iterations=t + 1,
        total_reward=total_reward,
        score_history=score_history,
        loss_history=loss_history,
    )


def optimize_with_central_critic(
    env: MultiAgentEnv,
    initial_pipeline: dict[str, Any],
    iterations: int = 20,
    batch_size: int = 32,
) -> MAOptimizationResult:
    """Run optimization with centralized QMIX critic.

    Args:
        env: Multi-agent environment
        initial_pipeline: Starting pipeline
        iterations: Training iterations
        batch_size: Batch size

    Returns:
        MAOptimizationResult
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required")

    from transformation_portal.rl.ma_state import get_global_dim

    # Create agents
    agents_state = create_agents(env.node_ids)
    agent_models = {k: v.model for k, v in agents_state.items()}

    # Create central critic
    n_agents = len(agent_models)
    global_dim = get_global_dim()
    critic = create_critic(n_agents, global_dim, critic_type="hybrid")

    # Create centralized trainer
    trainer = CentralizedTrainer(agent_models, critic)

    # Reset environment
    states = env.reset(initial_pipeline)

    best_pipeline = initial_pipeline
    best_score = env.prev_score
    total_reward = 0.0

    replay: list[dict[str, Any]] = []
    score_history = [best_score]
    loss_history: list[float] = []

    logger.info("Starting QMIX optimization: %d agents", n_agents)

    for t in range(iterations):
        joint_actions: list[NodeAction] = []
        actions_dict: dict[str, int] = {}

        # Each agent selects action
        for node_id, agent in agents_state.items():
            if node_id not in states:
                continue

            s = states[node_id]
            a_idx, _, _ = agent.model.act(torch.tensor(s).unsqueeze(0))
            action = agent.actions[a_idx]

            joint_actions.append(action)
            actions_dict[node_id] = a_idx

        # Environment step
        result = env.step(joint_actions)

        # Store transition
        replay.append(
            {
                "states": {k: v.copy() for k, v in states.items()},
                "actions": actions_dict,
                "reward": result.reward,
                "next_states": {k: v.copy() for k, v in result.states.items()},
                "global_state": result.global_state.copy(),
                "done": result.done,
            }
        )

        total_reward += result.reward

        # Train
        if len(replay) >= batch_size:
            batch = replay[-batch_size:]
            loss = trainer.step(batch)
            loss_history.append(loss)

        # Track best
        current_score = result.info.get("score", 0.0)
        if current_score > best_score:
            best_score = current_score
            best_pipeline = env.current_pipeline

        score_history.append(current_score)
        states = result.states

        if result.done:
            break

    return MAOptimizationResult(
        best_pipeline=best_pipeline,
        best_score=best_score,
        iterations=t + 1,
        total_reward=total_reward,
        score_history=score_history,
        loss_history=loss_history,
    )
