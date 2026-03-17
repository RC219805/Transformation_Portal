"""Multi-agent action space: Per-node action definitions.

This module defines node-specific action spaces for the multi-agent
RL optimizer where each node learns its own policy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeAction:
    """A single action for a specific node.

    Attributes:
        node_id: Target node ID
        action_type: Type of action
        params: Action parameters
        index: Index in node's action list
    """

    node_id: str
    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    index: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "action_type": self.action_type,
            "params": self.params,
            "index": self.index,
        }

    def to_fix(self) -> dict[str, Any]:
        """Convert to fix suggestion format."""
        return {
            "type": self.action_type.split("_")[0],
            "target_node": self.node_id,
            "action": self.action_type,
            "params": self.params,
            "confidence": 0.8,
            "rationale": f"Multi-agent action: {self.action_type}",
            "priority": 5,
            "reversible": True,
        }


# Per-node action templates
# Format: {node_id: [(action_type, {param: [values]}), ...]}
NODE_ACTIONS: dict[str, list[tuple[str, dict[str, list[Any]]]]] = {
    "sam2": [
        ("increase_mask_coverage", {"threshold": [0.2, 0.3, 0.4, 0.5]}),
        ("expand_prompt_set", {"include_negative": [True, False]}),
    ],
    "nvdiffrec": [
        ("increase_iterations", {"steps": [200, 400, 600, 800]}),
        ("increase_mesh_resolution", {"subdivisions": [1, 2, 3]}),
    ],
    "material_backend": [
        ("adjust_roughness_prior", {"bias": [-0.2, -0.1, 0.0, 0.1, 0.2]}),
        ("adjust_metalness_prior", {"bias": [-0.2, -0.1, 0.0, 0.1, 0.2]}),
        ("adjust_texture_quality", {"detail_level": ["low", "medium", "high"]}),
    ],
    "depth_backend": [
        ("increase_resolution", {"scale": [1.0, 1.5, 2.0]}),
    ],
    "postprocess": [
        ("enable_seam_blending", {"blend_radius": [4, 8, 16]}),
        ("apply_denoising", {"strength": [0.3, 0.5, 0.7]}),
    ],
    "color_grading": [
        ("adjust_tone_curve", {"contrast": [0.95, 1.0, 1.05, 1.1]}),
    ],
}

# Default nodes for multi-agent setup
DEFAULT_AGENT_NODES = ["sam2", "nvdiffrec", "material_backend"]


def enumerate_node_actions(node_id: str) -> list[NodeAction]:
    """Enumerate all actions for a specific node.

    Args:
        node_id: Node identifier

    Returns:
        List of all possible NodeAction instances

    Example:
        >>> actions = enumerate_node_actions("sam2")
        >>> print(f"SAM2 has {len(actions)} actions")
    """
    templates = NODE_ACTIONS.get(node_id, [])
    actions: list[NodeAction] = []

    for action_type, param_grid in templates:
        if not param_grid:
            actions.append(
                NodeAction(
                    node_id=node_id,
                    action_type=action_type,
                    params={},
                    index=len(actions),
                )
            )
        else:
            keys = list(param_grid.keys())
            value_lists = [param_grid[k] for k in keys]

            for values in itertools.product(*value_lists):
                params = dict(zip(keys, values))
                actions.append(
                    NodeAction(
                        node_id=node_id,
                        action_type=action_type,
                        params=params,
                        index=len(actions),
                    )
                )

    return actions


def get_all_node_actions() -> dict[str, list[NodeAction]]:
    """Get actions for all nodes.

    Returns:
        Dictionary mapping node_id -> list of actions
    """
    return {node_id: enumerate_node_actions(node_id) for node_id in NODE_ACTIONS}


def get_action_dims() -> dict[str, int]:
    """Get action dimension for each node.

    Returns:
        Dictionary mapping node_id -> action count
    """
    return {node_id: len(enumerate_node_actions(node_id)) for node_id in NODE_ACTIONS}
