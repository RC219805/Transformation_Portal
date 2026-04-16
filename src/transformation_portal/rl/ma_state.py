"""Multi-agent state features for local and global context.

This module provides state encoding for the multi-agent RL optimizer,
combining global context with node-specific local features.
"""

from __future__ import annotations

from typing import Any

# Lazy numpy import
_np = None


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        try:
            import numpy as np

            _np = np
        except ImportError:
            raise ImportError("NumPy required for state encoding")
    return _np


# Semantic diff categories
DIFF_TYPES = ["geometry", "texture", "missing", "artifact", "semantic"]
SEVERITIES = ["low", "medium", "high"]

# Config keys per node type
NODE_CONFIG_KEYS = {
    "sam2": ["threshold", "iou_threshold", "include_negative"],
    "nvdiffrec": ["steps", "subdivisions", "refine"],
    "material_backend": ["roughness_bias", "metalness_bias", "detail_level"],
    "depth_backend": ["resolution_scale"],
    "postprocess": ["seam_blending", "blend_radius", "denoise_strength"],
    "color_grading": ["contrast", "saturation"],
}

# Default config keys for unknown nodes
DEFAULT_CONFIG_KEYS = ["threshold", "steps", "bias"]


def encode_global(
    metrics: dict[str, float],
    diff: dict[str, Any],
) -> Any:  # Returns numpy array
    """Encode global state features.

    Args:
        metrics: Evaluation metrics
        diff: Semantic diff result

    Returns:
        Global state feature vector
    """
    np = _get_numpy()
    feats: list[float] = []

    # Metrics features
    feats.append(metrics.get("score", 0.0))
    feats.append(metrics.get("psnr", 0.0) / 50.0)
    feats.append(1.0 - min(metrics.get("lpips", 0.0), 1.0))
    feats.append(metrics.get("ssim", 0.0))
    feats.append(metrics.get("llava_score", 0.0))

    # Diff histogram
    hist: dict[tuple[str, str], int] = {}
    for t in DIFF_TYPES:
        for s in SEVERITIES:
            hist[(t, s)] = 0

    for change in diff.get("changes", []):
        ctype = change.get("type", "semantic").lower()
        severity = change.get("severity", "medium").lower()

        if ctype not in DIFF_TYPES:
            ctype = "semantic"
        if severity not in SEVERITIES:
            severity = "medium"

        hist[(ctype, severity)] += 1

    # Flatten histogram (normalized)
    max_changes = 10.0
    for t in DIFF_TYPES:
        for s in SEVERITIES:
            feats.append(min(hist[(t, s)] / max_changes, 1.0))

    return np.array(feats, dtype=np.float32)


def encode_local(
    node_cfg: dict[str, Any],
    node_id: str | None = None,
) -> Any:  # Returns numpy array
    """Encode node-specific local state.

    Args:
        node_cfg: Node configuration dict
        node_id: Optional node ID for type-specific encoding

    Returns:
        Local state feature vector
    """
    np = _get_numpy()

    # Get config keys for this node type
    config_keys = NODE_CONFIG_KEYS.get(node_id or "", DEFAULT_CONFIG_KEYS)

    feats: list[float] = []

    for key in config_keys:
        value = node_cfg.get(key, 0.0)

        # Normalize based on key
        if key == "steps":
            value = float(value) / 1000.0
        elif key == "threshold" or key == "iou_threshold":
            value = float(value)
        elif key == "blend_radius":
            value = float(value) / 20.0
        elif key == "resolution_scale":
            value = float(value) / 3.0
        elif key in ("roughness_bias", "metalness_bias", "bias"):
            value = (float(value) + 0.5) / 1.0
        elif key == "denoise_strength" or key == "strength":
            value = float(value)
        elif key == "contrast":
            value = (float(value) - 0.8) / 0.4
        elif key == "subdivisions":
            value = float(value) / 5.0
        elif key in ("seam_blending", "refine", "include_negative"):
            value = 1.0 if value else 0.0
        elif key == "detail_level":
            level_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
            value = level_map.get(str(value).lower(), 0.5)
        else:
            value = float(value) if isinstance(value, (int, float)) else 0.0

        feats.append(value)

    # Pad to fixed size
    while len(feats) < 8:
        feats.append(0.0)

    return np.array(feats[:8], dtype=np.float32)


def encode_state(
    node_cfg: dict[str, Any],
    metrics: dict[str, float],
    diff: dict[str, Any],
    node_id: str | None = None,
) -> Any:  # Returns numpy array
    """Encode full state (global + local) for a node agent.

    Args:
        node_cfg: Node configuration
        metrics: Evaluation metrics
        diff: Semantic diff result
        node_id: Optional node ID

    Returns:
        Combined state feature vector
    """
    np = _get_numpy()

    global_feats = encode_global(metrics, diff)
    local_feats = encode_local(node_cfg, node_id)

    return np.concatenate([global_feats, local_feats])


def get_global_dim() -> int:
    """Get dimension of global state."""
    # 5 metrics + 15 diff histogram
    return 5 + len(DIFF_TYPES) * len(SEVERITIES)


def get_local_dim() -> int:
    """Get dimension of local state."""
    return 8


def get_state_dim() -> int:
    """Get total state dimension."""
    return get_global_dim() + get_local_dim()


def get_node_config(pipeline: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Extract node config from pipeline.

    Args:
        pipeline: Pipeline configuration
        node_id: Node ID to find

    Returns:
        Node config dict (empty if not found)
    """
    for node in pipeline.get("nodes", []):
        if node.get("id") == node_id:
            return node.get("config", {})
    return {}
