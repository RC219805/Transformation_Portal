"""RL experience replay buffer.

This module provides replay buffers for off-policy RL training:
- ReplayBuffer: Simple FIFO buffer with random sampling
- PrioritizedReplayBuffer: Priority-based sampling
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple


class Transition(NamedTuple):
    """A single transition tuple."""

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    log_prob: Any = None
    value: Any = None


@dataclass
class BatchedTransitions:
    """Batched transitions for training."""

    states: Any
    actions: Any
    rewards: Any
    next_states: Any
    dones: Any
    log_probs: Any = None
    values: Any = None


class ReplayBuffer:
    """Simple FIFO replay buffer with random sampling.

    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(Transition(s, a, r, s2, done))
        >>> batch = buffer.sample(32)
    """

    def __init__(self, capacity: int = 50000) -> None:
        """Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Args:
            transition: Transition tuple to add
        """
        self.buffer.append(transition)

    def add_batch(self, transitions: list[Transition]) -> None:
        """Add multiple transitions.

        Args:
            transitions: List of transitions
        """
        for t in transitions:
            self.add(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample random transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def sample_batched(self, batch_size: int) -> BatchedTransitions | None:
        """Sample and batch transitions.

        Args:
            batch_size: Number of transitions

        Returns:
            BatchedTransitions or None if not enough samples
        """
        if len(self.buffer) < batch_size:
            return None

        try:
            import numpy as np
        except ImportError:
            return None

        transitions = self.sample(batch_size)

        return BatchedTransitions(
            states=np.array([t.state for t in transitions]),
            actions=np.array([t.action for t in transitions]),
            rewards=np.array([t.reward for t in transitions]),
            next_states=np.array([t.next_state for t in transitions]),
            dones=np.array([t.done for t in transitions]),
            log_probs=[t.log_prob for t in transitions],
            values=[t.value for t in transitions],
        )

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples.

        Args:
            min_size: Minimum required samples

        Returns:
            True if ready for sampling
        """
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Uses TD-error based priorities for sampling.

    Example:
        >>> buffer = PrioritizedReplayBuffer(capacity=10000)
        >>> buffer.add(transition, priority=1.0)
        >>> batch, indices, weights = buffer.sample(32, beta=0.4)
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
    ) -> None:
        """Initialize buffer.

        Args:
            capacity: Maximum capacity
            alpha: Priority exponent (0 = uniform, 1 = full priority)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[Transition] = []
        self.priorities: list[float] = []
        self.position = 0
        self.max_priority = 1.0

    def add(self, transition: Transition, priority: float | None = None) -> None:
        """Add transition with priority.

        Args:
            transition: Transition to add
            priority: Priority value (defaults to max)
        """
        priority = priority or self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority**self.alpha)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority**self.alpha

        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> tuple[list[Transition], list[int], list[float]]:
        """Sample with priority weighting.

        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent

        Returns:
            Tuple of (transitions, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], [], []

        # Compute sampling probabilities
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]

        # Sample indices
        indices = random.choices(
            range(len(self.buffer)),
            weights=probs,
            k=min(batch_size, len(self.buffer)),
        )

        # Compute importance sampling weights
        min_prob = min(probs)
        weights = []
        for idx in indices:
            prob = probs[idx]
            weight = (len(self.buffer) * prob) ** (-beta)
            weights.append(weight)

        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]

        transitions = [self.buffer[i] for i in indices]

        return transitions, indices, weights

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Transition indices
            priorities: New priorities (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class RolloutBuffer:
    """Buffer for on-policy algorithms (PPO, A2C).

    Stores complete rollouts for batch updates.
    """

    def __init__(self) -> None:
        """Initialize buffer."""
        self.states: list[Any] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[Any] = []
        self.log_probs: list[Any] = []
        self.dones: list[bool] = []

    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        value: Any,
        log_prob: Any,
        done: bool,
    ) -> None:
        """Add a step to the rollout.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Episode done flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[list[float], list[float]]:
        """Compute returns and advantages using GAE.

        Args:
            last_value: Bootstrap value for incomplete episode
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []

        gae = 0.0
        next_value = last_value

        # Reverse iteration for GAE
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                delta = self.rewards[i] - self.values[i].item()
                gae = delta
            else:
                delta = (
                    self.rewards[i]
                    + gamma * next_value
                    - self.values[i].item()
                )
                gae = delta + gamma * gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i].item())

            next_value = self.values[i].item()

        return returns, advantages

    def clear(self) -> None:
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        """Get rollout length."""
        return len(self.states)
