"""Ray-distributed training components for multi-agent RL.

This module provides distributed training using Ray:
- RayReplay: Shared replay buffer
- RolloutWorker: Parallel environment workers
- Learner: Central parameter server
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Check Ray availability
_ray = None


def _get_ray():
    """Lazy import Ray."""
    global _ray
    if _ray is None:
        try:
            import ray

            _ray = ray
        except ImportError:
            raise ImportError("Ray required for distributed training")
    return _ray


def create_ray_replay(capacity: int = 200000):
    """Create Ray-distributed replay buffer.

    Args:
        capacity: Buffer capacity

    Returns:
        Ray actor handle
    """
    ray = _get_ray()

    @ray.remote
    class RayReplay:
        """Ray actor for shared replay buffer."""

        def __init__(self, capacity: int):
            self.buf: list[Any] = []
            self.capacity = capacity

        def add(self, item: Any) -> None:
            """Add item to buffer."""
            self.buf.append(item)
            if len(self.buf) > self.capacity:
                self.buf.pop(0)

        def add_batch(self, items: list[Any]) -> None:
            """Add multiple items."""
            for item in items:
                self.add(item)

        def sample(self, batch_size: int) -> list[Any]:
            """Sample random batch."""
            import random

            return random.sample(self.buf, min(len(self.buf), batch_size))

        def size(self) -> int:
            """Get buffer size."""
            return len(self.buf)

        def clear(self) -> None:
            """Clear buffer."""
            self.buf.clear()

    return RayReplay.remote(capacity)


def create_rollout_worker(env_factory: Callable, agent_weights: dict[str, Any] | None = None):
    """Create Ray rollout worker.

    Args:
        env_factory: Factory function to create environment
        agent_weights: Initial agent weights

    Returns:
        Ray actor handle
    """
    ray = _get_ray()

    @ray.remote
    class RolloutWorker:
        """Ray actor for parallel rollouts."""

        def __init__(self, env_factory: Callable, initial_weights: dict | None):
            self.env = env_factory()
            self.weights = initial_weights or {}

        def set_weights(self, weights: dict[str, Any]) -> None:
            """Update agent weights."""
            self.weights = weights

        def rollout(self, pipeline: dict[str, Any], steps: int = 10) -> list[dict[str, Any]]:
            """Execute rollout and collect transitions.

            Args:
                pipeline: Initial pipeline
                steps: Number of steps

            Returns:
                List of transitions
            """
            from transformation_portal.rl.ma_action_space import enumerate_node_actions
            from transformation_portal.rl.ma_model import create_agent
            from transformation_portal.rl.ma_state import get_state_dim

            trajectory: list[dict[str, Any]] = []

            # Reset environment
            states = self.env.reset(pipeline)

            for _ in range(steps):
                joint_actions = []
                actions_dict = {}

                # Select actions
                for node_id in self.env.node_ids:
                    if node_id not in states:
                        continue

                    # Get action (simplified - would use actual policy)
                    import random

                    actions = enumerate_node_actions(node_id)
                    if actions:
                        action = random.choice(actions)
                        joint_actions.append(action)
                        actions_dict[node_id] = action.index

                # Step
                result = self.env.step(joint_actions)

                trajectory.append(
                    {
                        "states": {k: v.tolist() for k, v in states.items()},
                        "actions": actions_dict,
                        "reward": result.reward,
                        "next_states": {k: v.tolist() for k, v in result.states.items()},
                        "global_state": result.global_state.tolist(),
                        "done": result.done,
                    }
                )

                states = result.states

                if result.done:
                    break

            return trajectory

    return RolloutWorker.remote(env_factory, agent_weights)


def create_learner(
    agents: dict[str, Any],
    critic: Any,
    state_dim: int,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
):
    """Create Ray learner actor.

    Args:
        agents: Agent models
        critic: Central critic
        state_dim: State dimension
        learning_rate: Learning rate
        gamma: Discount factor

    Returns:
        Ray actor handle
    """
    ray = _get_ray()

    @ray.remote
    class Learner:
        """Ray actor for centralized learning."""

        def __init__(
            self,
            agents: dict,
            critic: Any,
            state_dim: int,
            lr: float,
            gamma: float,
        ):
            import torch

            self.agents = agents
            self.critic = critic
            self.gamma = gamma

            # Collect parameters
            params = list(critic.parameters())
            for a in agents.values():
                params.extend(a.parameters())

            self.optimizer = torch.optim.Adam(params, lr=lr)

        def get_weights(self) -> dict[str, Any]:
            """Get all model weights."""
            return {
                "agents": {k: v.state_dict() for k, v in self.agents.items()},
                "critic": self.critic.state_dict(),
            }

        def set_weights(self, weights: dict[str, Any]) -> None:
            """Set all model weights."""
            for k, v in self.agents.items():
                if k in weights["agents"]:
                    v.load_state_dict(weights["agents"][k])
            self.critic.load_state_dict(weights["critic"])

        def train_step(self, batch: list[dict[str, Any]]) -> float:
            """Training step on batch.

            Args:
                batch: List of transitions

            Returns:
                Loss value
            """
            import torch
            import torch.nn.functional as F

            total_loss = torch.tensor(0.0)

            for item in batch:
                states = item["states"]
                actions = item["actions"]
                reward = item["reward"]
                global_state = item["global_state"]

                # Get Q-values
                q_values = []
                for node_id, agent in self.agents.items():
                    if node_id not in states:
                        continue

                    s = torch.tensor(states[node_id], dtype=torch.float32).unsqueeze(0)
                    q = agent.forward(s)
                    a = actions.get(node_id, 0)

                    if hasattr(q, "shape") and len(q.shape) > 1:
                        q_values.append(q[0, a])
                    else:
                        q_values.append(q.squeeze())

                if not q_values:
                    continue

                agent_qs = torch.stack(q_values).unsqueeze(0)
                gs = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)

                q_total = self.critic.forward(agent_qs, gs)
                target = torch.tensor([[reward]], dtype=torch.float32)

                total_loss = total_loss + F.mse_loss(q_total, target)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            return float(total_loss.item())

    return Learner.remote(agents, critic, state_dim, learning_rate, gamma)


def train_distributed(
    env_factory: Callable,
    agents: dict[str, Any],
    critic: Any,
    initial_pipeline: dict[str, Any],
    num_workers: int = 4,
    iterations: int = 100,
    steps_per_rollout: int = 10,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run distributed multi-agent training.

    Args:
        env_factory: Factory to create environment
        agents: Agent models
        critic: Central critic
        initial_pipeline: Starting pipeline
        num_workers: Number of rollout workers
        iterations: Training iterations
        steps_per_rollout: Steps per worker rollout
        batch_size: Training batch size

    Returns:
        Training results
    """
    ray = _get_ray()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    logger.info(
        "Starting distributed training: %d workers, %d iterations",
        num_workers,
        iterations,
    )

    # Create actors
    replay = create_ray_replay()
    learner = create_learner(agents, critic, state_dim=28)

    workers = [create_rollout_worker(env_factory) for _ in range(num_workers)]

    loss_history = []

    for it in range(iterations):
        # Broadcast weights
        weights = ray.get(learner.get_weights.remote())
        for w in workers:
            w.set_weights.remote(weights)

        # Collect rollouts
        futures = [w.rollout.remote(initial_pipeline, steps_per_rollout) for w in workers]
        trajectories = ray.get(futures)

        # Add to replay
        for traj in trajectories:
            ray.get(replay.add_batch.remote(traj))

        # Train
        buffer_size = ray.get(replay.size.remote())
        if buffer_size >= batch_size:
            batch = ray.get(replay.sample.remote(batch_size))
            loss = ray.get(learner.train_step.remote(batch))
            loss_history.append(loss)

            if it % 10 == 0:
                logger.info("Iteration %d: loss=%.4f, buffer=%d", it, loss, buffer_size)

    return {
        "iterations": iterations,
        "loss_history": loss_history,
        "final_weights": ray.get(learner.get_weights.remote()),
    }
