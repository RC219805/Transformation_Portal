"""Multi-agent communication: Message bus for agent coordination.

This module provides a lightweight message passing system for
agents to share intents and coordinate actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """A message from an agent.

    Attributes:
        sender: Agent/node ID
        intent: What the agent intends to do
        params: Action parameters
        priority: Message priority
        timestamp: When message was sent
    """

    sender: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: int = 0


class MessageBus:
    """Lightweight message bus for multi-agent coordination.

    Allows agents to publish intents and read others' messages
    to enable cooperative behavior.

    Example:
        >>> bus = MessageBus()
        >>> bus.publish("sam2", {"intent": "increase_coverage"})
        >>> messages = bus.read_all()
    """

    def __init__(self) -> None:
        """Initialize message bus."""
        self._messages: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._step: int = 0

    def publish(self, node_id: str, payload: dict[str, Any]) -> None:
        """Publish a message from a node.

        Args:
            node_id: Sender node ID
            payload: Message payload (should include 'intent')
        """
        self._messages[node_id] = {
            **payload,
            "sender": node_id,
            "timestamp": self._step,
        }

    def read(self, node_id: str) -> dict[str, Any] | None:
        """Read message from a specific node.

        Args:
            node_id: Node to read from

        Returns:
            Message payload or None
        """
        return self._messages.get(node_id)

    def read_all(self) -> dict[str, dict[str, Any]]:
        """Read all current messages.

        Returns:
            Dictionary of node_id -> message
        """
        return self._messages.copy()

    def read_others(self, exclude_node: str) -> dict[str, dict[str, Any]]:
        """Read messages from other nodes.

        Args:
            exclude_node: Node to exclude

        Returns:
            Messages from other nodes
        """
        return {k: v for k, v in self._messages.items() if k != exclude_node}

    def clear(self) -> None:
        """Clear all messages (call at end of step)."""
        # Archive to history
        if self._messages:
            self._history.append(
                {
                    "step": self._step,
                    "messages": self._messages.copy(),
                }
            )
        self._messages.clear()
        self._step += 1

    def get_history(self, last_n: int = 10) -> list[dict[str, Any]]:
        """Get recent message history.

        Args:
            last_n: Number of recent steps

        Returns:
            List of historical message snapshots
        """
        return self._history[-last_n:]

    def encode_messages(self, exclude_node: str | None = None) -> list[float]:
        """Encode messages as feature vector.

        Args:
            exclude_node: Optional node to exclude

        Returns:
            Flattened feature vector of message intents
        """
        # Simple encoding: count of each intent type
        intent_counts: dict[str, int] = {}

        for node_id, msg in self._messages.items():
            if exclude_node and node_id == exclude_node:
                continue

            intent = msg.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Fixed set of known intents
        known_intents = [
            "increase_coverage",
            "increase_iterations",
            "adjust_roughness",
            "adjust_metalness",
            "enable_blending",
            "apply_denoising",
        ]

        return [float(intent_counts.get(i, 0)) for i in known_intents]


class CoordinationProtocol:
    """Protocol for agent coordination.

    Defines rules for how agents should coordinate their actions
    based on messages from other agents.
    """

    # Action compatibility matrix
    # If agent A does action X, agent B should/shouldn't do Y
    COMPATIBILITY = {
        ("increase_coverage", "increase_iterations"): 1.0,  # Compatible
        ("increase_coverage", "adjust_roughness"): 0.5,  # Neutral
        ("adjust_roughness", "adjust_metalness"): 0.3,  # Slightly conflicting
    }

    @staticmethod
    def get_compatibility(action_a: str, action_b: str) -> float:
        """Get compatibility score between two actions.

        Args:
            action_a: First action type
            action_b: Second action type

        Returns:
            Compatibility score (0-1, higher = more compatible)
        """
        key = (action_a, action_b)
        if key in CoordinationProtocol.COMPATIBILITY:
            return CoordinationProtocol.COMPATIBILITY[key]

        # Reverse key
        key_rev = (action_b, action_a)
        if key_rev in CoordinationProtocol.COMPATIBILITY:
            return CoordinationProtocol.COMPATIBILITY[key_rev]

        return 0.5  # Default neutral

    @staticmethod
    def compute_coordination_bonus(
        agent_action: str,
        other_messages: dict[str, dict[str, Any]],
    ) -> float:
        """Compute coordination bonus for an action.

        Args:
            agent_action: Action the agent wants to take
            other_messages: Messages from other agents

        Returns:
            Bonus/penalty value
        """
        bonus = 0.0

        for node_id, msg in other_messages.items():
            other_intent = msg.get("intent", "")
            compat = CoordinationProtocol.get_compatibility(agent_action, other_intent)
            bonus += (compat - 0.5) * 0.1  # Small adjustment

        return bonus
