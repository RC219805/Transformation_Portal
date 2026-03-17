"""RL state encoder: Compact feature encoding for pipeline states.

This module provides state encoding for the RL optimizer, converting
pipeline configurations, metrics, and semantic diff results into
fixed-size feature vectors.
"""

from __future__ import annotations

from typing import Any

# Lazy numpy import for environments without numpy
_np = None


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        try:
            import numpy as np

            _np = np
        except ImportError:
            raise ImportError("NumPy required for RL state encoding")
    return _np


# Semantic diff categories
DIFF_TYPES = ["geometry", "texture", "missing", "artifact", "semantic"]
SEVERITIES = ["low", "medium", "high"]

# Maximum nodes to encode (for fixed-size vectors)
MAX_NODES = 10

# Config keys to encode per node
NODE_CONFIG_KEYS = [
    "threshold",
    "steps",
    "roughness_bias",
    "metalness_bias",
    "resolution_scale",
    "blend_radius",
    "denoise_strength",
    "contrast",
]


def get_state_dim() -> int:
    """Get dimension of state vector.

    Returns:
        Size of encoded state vector
    """
    # Metrics: 5 values
    # Diff histogram: 5 types * 3 severities = 15
    # Node configs: MAX_NODES * len(NODE_CONFIG_KEYS) = 10 * 8 = 80
    # History: 5 recent scores
    return 5 + 15 + MAX_NODES * len(NODE_CONFIG_KEYS) + 5


def encode_metrics(metrics: dict[str, float]) -> list[float]:
    """Encode evaluation metrics.

    Args:
        metrics: Evaluation metrics dict

    Returns:
        List of normalized metric values
    """
    return [
        metrics.get("score", 0.0),
        metrics.get("psnr", 0.0) / 50.0,  # Normalize PSNR (typical range 20-50)
        1.0 - min(metrics.get("lpips", 0.0), 1.0),  # Invert LPIPS
        metrics.get("ssim", 0.0),
        metrics.get("llava_score", 0.0),
    ]


def encode_diff_histogram(semantic_diff: dict[str, Any]) -> list[float]:
    """Encode semantic diff as histogram.

    Args:
        semantic_diff: Semantic diff result

    Returns:
        Flattened histogram of change counts
    """
    # Initialize histogram
    hist: dict[tuple[str, str], int] = {}
    for t in DIFF_TYPES:
        for s in SEVERITIES:
            hist[(t, s)] = 0

    # Count changes
    for change in semantic_diff.get("changes", []):
        ctype = change.get("type", "semantic").lower()
        severity = change.get("severity", "medium").lower()

        # Clamp to known types
        if ctype not in DIFF_TYPES:
            ctype = "semantic"
        if severity not in SEVERITIES:
            severity = "medium"

        hist[(ctype, severity)] += 1

    # Flatten to list (normalized)
    max_changes = 10.0  # Normalize by max expected changes
    features = []
    for t in DIFF_TYPES:
        for s in SEVERITIES:
            features.append(min(hist[(t, s)] / max_changes, 1.0))

    return features


def encode_node_config(node: dict[str, Any]) -> list[float]:
    """Encode a single node's configuration.

    Args:
        node: Node configuration dict

    Returns:
        List of config values
    """
    config = node.get("config", {})
    features = []

    for key in NODE_CONFIG_KEYS:
        value = config.get(key, 0.0)

        # Normalize based on expected ranges
        if key == "steps":
            value = float(value) / 1000.0
        elif key == "threshold":
            value = float(value)
        elif key == "blend_radius":
            value = float(value) / 20.0
        elif key == "resolution_scale":
            value = float(value) / 3.0
        elif key in ("roughness_bias", "metalness_bias"):
            value = (float(value) + 0.5) / 1.0  # Map [-0.5, 0.5] to [0, 1]
        elif key == "denoise_strength":
            value = float(value)
        elif key == "contrast":
            value = (float(value) - 0.8) / 0.4  # Map [0.8, 1.2] to [0, 1]
        else:
            value = float(value) if isinstance(value, (int, float)) else 0.0

        features.append(value)

    return features


def encode_pipeline_config(pipeline: dict[str, Any]) -> list[float]:
    """Encode pipeline node configurations.

    Args:
        pipeline: Pipeline configuration

    Returns:
        Flattened node config features
    """
    nodes = pipeline.get("nodes", [])
    features = []

    for i in range(MAX_NODES):
        if i < len(nodes):
            features.extend(encode_node_config(nodes[i]))
        else:
            # Pad with zeros
            features.extend([0.0] * len(NODE_CONFIG_KEYS))

    return features


def encode_history(history: list[float], window: int = 5) -> list[float]:
    """Encode recent score history.

    Args:
        history: List of recent scores
        window: Number of recent scores to include

    Returns:
        Padded list of scores
    """
    # Take last N scores
    recent = history[-window:] if history else []

    # Pad to fixed size
    while len(recent) < window:
        recent.insert(0, 0.0)

    return recent


def encode_state(
    pipeline: dict[str, Any],
    metrics: dict[str, float],
    semantic_diff: dict[str, Any],
    history: list[float] | None = None,
) -> "Any":  # Returns numpy array
    """Encode full state for RL agent.

    Combines pipeline configuration, evaluation metrics, semantic
    diff summary, and score history into a fixed-size feature vector.

    Args:
        pipeline: Pipeline configuration dict
        metrics: Evaluation metrics
        semantic_diff: Semantic diff result
        history: Recent score history

    Returns:
        numpy array of state features

    Example:
        >>> state = encode_state(pipeline, metrics, diff)
        >>> print(f"State shape: {state.shape}")
    """
    np = _get_numpy()

    features: list[float] = []

    # 1. Metrics (5 features)
    features.extend(encode_metrics(metrics))

    # 2. Diff histogram (15 features)
    features.extend(encode_diff_histogram(semantic_diff))

    # 3. Node configs (MAX_NODES * NODE_CONFIG_KEYS features)
    features.extend(encode_pipeline_config(pipeline))

    # 4. History (5 features)
    features.extend(encode_history(history or []))

    return np.array(features, dtype=np.float32)


def decode_state_summary(state_vector: "Any") -> dict[str, Any]:
    """Decode state vector into human-readable summary.

    Args:
        state_vector: Encoded state vector

    Returns:
        Dictionary with decoded components
    """
    np = _get_numpy()

    # Extract components
    idx = 0

    # Metrics
    metrics_end = 5
    metrics_vec = state_vector[idx:metrics_end]
    metrics = {
        "score": float(metrics_vec[0]),
        "psnr": float(metrics_vec[1]) * 50.0,
        "lpips": 1.0 - float(metrics_vec[2]),
        "ssim": float(metrics_vec[3]),
        "llava_score": float(metrics_vec[4]),
    }
    idx = metrics_end

    # Diff histogram
    diff_end = idx + 15
    diff_vec = state_vector[idx:diff_end]
    diff_summary = {"total_changes": float(np.sum(diff_vec) * 10.0)}
    idx = diff_end

    # Skip node configs
    config_end = idx + MAX_NODES * len(NODE_CONFIG_KEYS)
    idx = config_end

    # History
    history = list(state_vector[idx : idx + 5])

    return {
        "metrics": metrics,
        "diff_summary": diff_summary,
        "recent_scores": history,
    }
