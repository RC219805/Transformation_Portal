"""RL action space: Discrete + parameterized actions for pipeline optimization.

This module defines the action space for the RL optimizer, reusing
the fix types from the self-healing system but structured for RL.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RLAction:
    """A single RL action representing a pipeline modification.

    Attributes:
        node: Target node ID
        action_type: Type of action (matches self-healing actions)
        params: Parameters for the action
        index: Optional index in action list
    """

    node: str
    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    index: int = -1

    def to_fix_suggestion(self) -> dict[str, Any]:
        """Convert to fix suggestion format for patcher."""
        return {
            "type": self.action_type.split("_")[0],  # Derive type from action
            "target_node": self.node,
            "action": self.action_type,
            "params": self.params,
            "confidence": 0.8,  # Default confidence for RL actions
            "rationale": f"RL-selected: {self.action_type}",
            "priority": 5,
            "reversible": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node": self.node,
            "action_type": self.action_type,
            "params": self.params,
            "index": self.index,
        }


# Action templates with parameter grids
# Format: (node_id, action_type, {param: [values]})
ACTION_TEMPLATES: list[tuple[str, str, dict[str, list[Any]]]] = [
    # Segmentation actions
    (
        "sam2",
        "increase_mask_coverage",
        {"threshold": [0.2, 0.3, 0.4, 0.5]},
    ),
    (
        "sam2",
        "expand_prompt_set",
        {"include_negative": [True, False]},
    ),
    # Reconstruction actions
    (
        "nvdiffrec",
        "increase_iterations",
        {"steps": [200, 400, 600, 800]},
    ),
    (
        "nvdiffrec",
        "increase_mesh_resolution",
        {"subdivisions": [1, 2, 3]},
    ),
    # Material actions
    (
        "material_backend",
        "adjust_roughness_prior",
        {"bias": [-0.2, -0.1, 0.0, 0.1, 0.2]},
    ),
    (
        "material_backend",
        "adjust_metalness_prior",
        {"bias": [-0.2, -0.1, 0.0, 0.1, 0.2]},
    ),
    (
        "material_backend",
        "adjust_texture_quality",
        {"detail_level": ["low", "medium", "high"]},
    ),
    # Depth actions
    (
        "depth_backend",
        "increase_resolution",
        {"scale": [1.0, 1.5, 2.0]},
    ),
    # Postprocess actions
    (
        "postprocess",
        "enable_seam_blending",
        {"blend_radius": [4, 8, 16]},
    ),
    (
        "postprocess",
        "apply_denoising",
        {"strength": [0.3, 0.5, 0.7]},
    ),
    # Color grading
    (
        "color_grading",
        "adjust_tone_curve",
        {"contrast": [0.95, 1.0, 1.05, 1.1]},
    ),
]


def enumerate_actions(templates: list | None = None) -> list[RLAction]:
    """Enumerate all possible actions from templates.

    Creates discrete actions by taking cartesian product of
    parameter values for each action template.

    Args:
        templates: Action templates (defaults to ACTION_TEMPLATES)

    Returns:
        List of all possible RLAction instances with indices

    Example:
        >>> actions = enumerate_actions()
        >>> print(f"Total actions: {len(actions)}")
        >>> print(actions[0])
    """
    templates = templates or ACTION_TEMPLATES
    actions: list[RLAction] = []

    for node, action_type, param_grid in templates:
        if not param_grid:
            # No parameters
            actions.append(
                RLAction(
                    node=node,
                    action_type=action_type,
                    params={},
                    index=len(actions),
                )
            )
        else:
            # Cartesian product of parameter values
            keys = list(param_grid.keys())
            value_lists = [param_grid[k] for k in keys]

            for values in itertools.product(*value_lists):
                params = dict(zip(keys, values))
                actions.append(
                    RLAction(
                        node=node,
                        action_type=action_type,
                        params=params,
                        index=len(actions),
                    )
                )

    return actions


def get_action_dim(templates: list | None = None) -> int:
    """Get dimension of action space.

    Args:
        templates: Action templates

    Returns:
        Number of discrete actions
    """
    return len(enumerate_actions(templates))


def action_to_index(action: RLAction, action_list: list[RLAction]) -> int:
    """Get index of action in action list.

    Args:
        action: Action to find
        action_list: List of all actions

    Returns:
        Index of action, or -1 if not found
    """
    for i, a in enumerate(action_list):
        if a.node == action.node and a.action_type == action.action_type:
            if a.params == action.params:
                return i
    return -1


def index_to_action(index: int, action_list: list[RLAction]) -> RLAction:
    """Get action by index.

    Args:
        index: Action index
        action_list: List of all actions

    Returns:
        RLAction at index

    Raises:
        IndexError: If index out of bounds
    """
    return action_list[index]


# Action categories for hierarchical selection
ACTION_CATEGORIES = {
    "segmentation": ["increase_mask_coverage", "expand_prompt_set"],
    "reconstruction": ["increase_iterations", "increase_mesh_resolution"],
    "material": ["adjust_roughness_prior", "adjust_metalness_prior", "adjust_texture_quality"],
    "depth": ["increase_resolution"],
    "postprocess": ["enable_seam_blending", "apply_denoising"],
    "color": ["adjust_tone_curve"],
}


def get_actions_by_category(
    category: str,
    action_list: list[RLAction] | None = None,
) -> list[RLAction]:
    """Get actions belonging to a category.

    Args:
        category: Action category name
        action_list: List of all actions

    Returns:
        Filtered list of actions
    """
    action_list = action_list or enumerate_actions()
    action_types = ACTION_CATEGORIES.get(category, [])
    return [a for a in action_list if a.action_type in action_types]
