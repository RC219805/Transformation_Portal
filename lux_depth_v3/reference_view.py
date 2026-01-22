"""Reference view selection strategies for multi-view depth estimation.

Automatic reference view selection based on class token features from DA3's
vision transformer. Selects the most suitable reference frame when processing
multiple input views (≥3).

Based on DA3 official implementation with sophisticated feature analysis.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from enum import Enum
from dataclasses import dataclass

import numpy as np


class RefViewStrategy(str, Enum):
    """Reference view selection strategies."""

    SADDLE_BALANCED = "saddle_balanced"
    SADDLE_SIM_RANGE = "saddle_sim_range"
    FIRST = "first"
    MIDDLE = "middle"


@dataclass
class RefViewSelectionResult:
    """Result of reference view selection."""

    selected_index: int
    strategy: RefViewStrategy
    scores: Optional[np.ndarray] = None  # Per-view scores
    metrics: Optional[dict] = None  # Additional metrics

    def __str__(self) -> str:
        return f"Selected view {self.selected_index} using strategy '{self.strategy.value}'"


class ReferenceViewSelector:
    """
    Automatic reference view selection for multi-view depth estimation.

    Analyzes class token features from all input views and intelligently
    selects the most suitable reference frame.

    Based on DA3 official implementation with class token analysis.

    Strategies:
        - saddle_balanced: Balanced features across multiple metrics (default)
        - saddle_sim_range: Maximizes similarity range (wide baseline)
        - middle: Uses middle view (recommended for temporal sequences)
        - first: Always uses first view (for pre-sorted inputs)

    Examples:
        >>> # Default strategy
        >>> selector = ReferenceViewSelector()
        >>> result = selector.select(num_views=5, class_tokens=tokens)
        >>> print(result.selected_index)

        >>> # Video sequence
        >>> selector = ReferenceViewSelector(strategy=RefViewStrategy.MIDDLE)
        >>> result = selector.select(num_views=10)
        >>> print(result.selected_index)  # 5
    """

    def __init__(self, strategy: RefViewStrategy = RefViewStrategy.SADDLE_BALANCED):
        """
        Initialize selector.

        Args:
            strategy: Reference view selection strategy
        """
        self.strategy = strategy

    def select(self, num_views: int, class_tokens: Optional[np.ndarray] = None) -> RefViewSelectionResult:
        """
        Select reference view from multiple inputs.

        Args:
            num_views: Number of input views
            class_tokens: Class token features (num_views, feature_dim)
                         Required for saddle_* strategies

        Returns:
            RefViewSelectionResult with selected index and metrics
        """
        # No reordering for 1-2 views
        if num_views < 3:
            return RefViewSelectionResult(
                selected_index=0,
                strategy=RefViewStrategy.FIRST,
                metrics={"reason": "num_views < 3, no reordering"},
            )

        if self.strategy == RefViewStrategy.FIRST:
            return self._select_first(num_views)
        elif self.strategy == RefViewStrategy.MIDDLE:
            return self._select_middle(num_views)
        elif self.strategy == RefViewStrategy.SADDLE_BALANCED:
            if class_tokens is None:
                raise ValueError("class_tokens required for saddle_balanced strategy")
            return self._select_saddle_balanced(class_tokens)
        elif self.strategy == RefViewStrategy.SADDLE_SIM_RANGE:
            if class_tokens is None:
                raise ValueError("class_tokens required for saddle_sim_range strategy")
            return self._select_saddle_sim_range(class_tokens)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_first(self, num_views: int) -> RefViewSelectionResult:
        """Always select first view."""
        return RefViewSelectionResult(
            selected_index=0,
            strategy=RefViewStrategy.FIRST,
            metrics={"reason": "first view strategy"},
        )

    def _select_middle(self, num_views: int) -> RefViewSelectionResult:
        """Select middle view (good for temporal sequences)."""
        middle_idx = num_views // 2
        return RefViewSelectionResult(
            selected_index=middle_idx,
            strategy=RefViewStrategy.MIDDLE,
            metrics={"reason": "middle view for temporal sequence"},
        )

    def _select_saddle_balanced(self, class_tokens: np.ndarray) -> RefViewSelectionResult:
        """
        Select view with balanced features across multiple metrics.

        Selects view closest to median (0.5) across three normalized metrics:
        - Similarity score (avg cosine similarity with other views)
        - Feature norm (L2 norm of features)
        - Feature variance (variance across dimensions)

        Args:
            class_tokens: (num_views, feature_dim) class token features

        Returns:
            RefViewSelectionResult with selected view
        """
        num_views, feature_dim = class_tokens.shape

        # Normalize class tokens
        normalized_tokens = class_tokens / (np.linalg.norm(class_tokens, axis=-1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        similarity_matrix = normalized_tokens @ normalized_tokens.T

        # Metric 1: Average similarity with other views
        similarity_scores = np.mean(similarity_matrix, axis=1)

        # Metric 2: Feature norm
        feature_norms = np.linalg.norm(class_tokens, axis=-1)

        # Metric 3: Feature variance
        feature_variances = np.var(class_tokens, axis=-1)

        # Normalize all metrics to [0, 1]
        def normalize_metric(values):
            min_val, max_val = values.min(), values.max()
            if max_val - min_val < 1e-8:
                return np.ones_like(values) * 0.5
            return (values - min_val) / (max_val - min_val)

        norm_similarity = normalize_metric(similarity_scores)
        norm_norms = normalize_metric(feature_norms)
        norm_variances = normalize_metric(feature_variances)

        # Find view closest to 0.5 across all metrics
        distances_to_median = np.abs(norm_similarity - 0.5) + np.abs(norm_norms - 0.5) + np.abs(norm_variances - 0.5)

        selected_idx = int(np.argmin(distances_to_median))

        return RefViewSelectionResult(
            selected_index=selected_idx,
            strategy=RefViewStrategy.SADDLE_BALANCED,
            scores=distances_to_median,
            metrics={
                "similarity_scores": similarity_scores.tolist(),
                "feature_norms": feature_norms.tolist(),
                "feature_variances": feature_variances.tolist(),
                "normalized_similarity": norm_similarity.tolist(),
                "normalized_norms": norm_norms.tolist(),
                "normalized_variances": norm_variances.tolist(),
                "distances_to_median": distances_to_median.tolist(),
            },
        )

    def _select_saddle_sim_range(self, class_tokens: np.ndarray) -> RefViewSelectionResult:
        """
        Select view with largest similarity range.

        Identifies "saddle point" views that are highly similar to some
        views but dissimilar to others, making them information-rich
        anchor points.

        Args:
            class_tokens: (num_views, feature_dim) class token features

        Returns:
            RefViewSelectionResult with selected view
        """
        num_views, feature_dim = class_tokens.shape

        # Normalize class tokens
        normalized_tokens = class_tokens / (np.linalg.norm(class_tokens, axis=-1, keepdims=True) + 1e-8)

        # Compute pairwise cosine similarity
        similarity_matrix = normalized_tokens @ normalized_tokens.T

        # For each view, compute similarity range (max - min with other views)
        similarity_ranges = []
        for i in range(num_views):
            # Get similarities to other views (exclude self)
            other_sims = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i + 1 :]])
            sim_range = other_sims.max() - other_sims.min()
            similarity_ranges.append(sim_range)

        similarity_ranges = np.array(similarity_ranges)
        selected_idx = int(np.argmax(similarity_ranges))

        return RefViewSelectionResult(
            selected_index=selected_idx,
            strategy=RefViewStrategy.SADDLE_SIM_RANGE,
            scores=similarity_ranges,
            metrics={
                "similarity_ranges": similarity_ranges.tolist(),
                "similarity_matrix": similarity_matrix.tolist(),
            },
        )


def select_reference_view(
    num_views: int,
    strategy: str = "saddle_balanced",
    class_tokens: Optional[np.ndarray] = None,
) -> RefViewSelectionResult:
    """
    Convenience function for reference view selection.

    Args:
        num_views: Number of input views
        strategy: Strategy name (saddle_balanced/saddle_sim_range/first/middle)
        class_tokens: Class token features (required for saddle_* strategies)

    Returns:
        RefViewSelectionResult with selected index

    Examples:
        >>> # Default strategy
        >>> result = select_reference_view(5, class_tokens=tokens)
        >>> print(result.selected_index)

        >>> # Video sequence
        >>> result = select_reference_view(10, strategy="middle")
        >>> print(result.selected_index)  # 5
    """
    strategy_enum = RefViewStrategy(strategy)
    selector = ReferenceViewSelector(strategy=strategy_enum)
    return selector.select(num_views, class_tokens)
