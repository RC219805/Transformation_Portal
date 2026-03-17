"""Temporal replay buffer for sequence-based RL.

This module provides replay buffers that store and sample
temporal sequences (episodes) rather than individual transitions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalTransition:
    """A single transition in a temporal sequence.

    Attributes:
        states: Per-agent states dict
        actions: Per-agent actions dict
        reward: Step reward
        global_state: Global state
        done: Episode done flag
        info: Additional info
    """

    states: dict[str, Any]
    actions: dict[str, int]
    reward: float
    global_state: Any
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A complete episode (sequence of transitions)."""

    transitions: list[TemporalTransition] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    edge_index: Any = None  # DAG structure (assumed constant)

    def add(self, transition: TemporalTransition) -> None:
        """Add transition to episode."""
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.length += 1

    def __len__(self) -> int:
        return self.length

    def get_sequence(self, start: int, length: int) -> list[TemporalTransition]:
        """Get a subsequence from the episode."""
        end = min(start + length, len(self.transitions))
        return self.transitions[start:end]


class TemporalReplayBuffer:
    """Replay buffer for temporal sequences.

    Stores complete episodes and samples contiguous subsequences
    of specified length for temporal RL training.

    Example:
        >>> buffer = TemporalReplayBuffer(capacity=1000, seq_len=3)
        >>> buffer.add_episode(episode)
        >>> batch = buffer.sample(32)
    """

    def __init__(
        self,
        capacity: int = 50000,
        seq_len: int = 3,
    ) -> None:
        """Initialize buffer.

        Args:
            capacity: Maximum number of episodes to store
            seq_len: Sequence length for sampling
        """
        self.episodes: list[Episode] = []
        self.capacity = capacity
        self.seq_len = seq_len

    def add_episode(self, episode: Episode) -> None:
        """Add a complete episode to the buffer.

        Args:
            episode: Episode to add
        """
        if len(episode) >= self.seq_len:
            self.episodes.append(episode)

            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def add_transitions(
        self,
        transitions: list[TemporalTransition],
        edge_index: Any = None,
    ) -> None:
        """Add transitions as a new episode.

        Args:
            transitions: List of transitions
            edge_index: DAG structure
        """
        if len(transitions) < self.seq_len:
            return

        episode = Episode(edge_index=edge_index)
        for t in transitions:
            episode.add(t)

        self.add_episode(episode)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        """Sample batch of temporal sequences.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of sequence dicts with:
            - states_seq: [seq_len, n_agents, state_dim]
            - actions_seq: [seq_len, n_agents]
            - rewards_seq: [seq_len]
            - global_states_seq: [seq_len, state_dim]
            - edge_index: DAG edges
        """
        if not self.episodes:
            return []

        batch = []

        for _ in range(batch_size):
            # Sample random episode
            episode = random.choice(self.episodes)

            if len(episode) < self.seq_len:
                continue

            # Sample random starting point
            max_start = len(episode) - self.seq_len
            start = random.randint(0, max_start)

            # Extract sequence
            sequence = episode.get_sequence(start, self.seq_len)

            batch.append(
                {
                    "transitions": sequence,
                    "edge_index": episode.edge_index,
                }
            )

        return batch

    def sample_formatted(
        self,
        batch_size: int,
        node_ids: list[str],
    ) -> dict[str, Any] | None:
        """Sample and format batch for critic input.

        Args:
            batch_size: Number of sequences
            node_ids: List of agent node IDs

        Returns:
            Formatted batch dict or None if not enough samples
        """
        try:
            import numpy as np
        except ImportError:
            return None

        raw_batch = self.sample(batch_size)
        if not raw_batch:
            return None

        # Determine dimensions from first sample
        first_seq = raw_batch[0]["transitions"]
        n_agents = len(node_ids)

        # Collect formatted data
        states_seqs = []
        actions_seqs = []
        rewards_seqs = []
        global_states_seqs = []

        for item in raw_batch:
            seq = item["transitions"]

            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_globals = []

            for t in seq:
                # Per-agent states
                agent_states = []
                agent_actions = []

                for nid in node_ids:
                    if nid in t.states:
                        state = t.states[nid]
                        if hasattr(state, "tolist"):
                            state = state.tolist()
                        agent_states.append(state)
                    else:
                        agent_states.append([0.0] * 28)  # Default

                    agent_actions.append(t.actions.get(nid, 0))

                seq_states.append(agent_states)
                seq_actions.append(agent_actions)
                seq_rewards.append(t.reward)

                global_state = t.global_state
                if hasattr(global_state, "tolist"):
                    global_state = global_state.tolist()
                seq_globals.append(global_state)

            states_seqs.append(seq_states)
            actions_seqs.append(seq_actions)
            rewards_seqs.append(seq_rewards)
            global_states_seqs.append(seq_globals)

        return {
            "states_seq": np.array(states_seqs),  # [B, T, N, D]
            "actions_seq": np.array(actions_seqs),  # [B, T, N]
            "rewards_seq": np.array(rewards_seqs),  # [B, T]
            "global_states_seq": np.array(global_states_seqs),  # [B, T, D]
            "edge_index": raw_batch[0]["edge_index"],  # Assume same DAG
        }

    def __len__(self) -> int:
        """Get number of episodes."""
        return len(self.episodes)

    def total_transitions(self) -> int:
        """Get total number of transitions across all episodes."""
        return sum(len(ep) for ep in self.episodes)

    def is_ready(self, min_episodes: int = 10) -> bool:
        """Check if buffer has enough data for training."""
        return len(self.episodes) >= min_episodes

    def clear(self) -> None:
        """Clear the buffer."""
        self.episodes.clear()
