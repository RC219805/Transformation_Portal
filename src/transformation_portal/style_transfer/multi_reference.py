"""Multi-reference style blending for IP-Adapter.

Utilities for blending multiple style references to create composite
styles. Useful for:
- Combining complementary aspects from different references
- Creating custom "house styles" from portfolios
- Balancing multiple aesthetic goals
- Style interpolation and exploration

Example:
    >>> blender = MultiReferenceBlender()
    >>>
    >>> # Blend three references
    >>> result = blender.blend_references([
    ...     ("warm_tones.jpg", 0.5),
    ...     ("dramatic_light.jpg", 0.3),
    ...     ("editorial_composition.jpg", 0.2)
    ... ])
"""

import logging
from typing import List, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


class MultiReferenceBlender:
    """Blend multiple style references for composite styles.

    Provides advanced blending strategies for combining style features
    from multiple reference images.

    Example:
        >>> from transformation_portal.style_transfer import MultiReferenceBlender
        >>>
        >>> blender = MultiReferenceBlender()
        >>>
        >>> # Weighted blend
        >>> blended = blender.weighted_blend([
        ...     (features1, 0.5),
        ...     (features2, 0.3),
        ...     (features3, 0.2)
        ... ])
    """

    # Blending strategies
    STRATEGIES = ["weighted", "max", "min", "average", "interpolate"]

    def __init__(self, device: Optional[str] = None):
        """Initialize multi-reference blender.

        Args:
            device: Computation device
        """
        self.device = device or self._detect_device()
        logger.info(f"MultiReferenceBlender initialized on {self.device}")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def weighted_blend(
        self,
        features_weights: List[Tuple[torch.Tensor, float]],
        normalize: bool = True
    ) -> torch.Tensor:
        """Blend features using weighted average.

        Args:
            features_weights: List of (features, weight) tuples
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        features_list = [f for f, _ in features_weights]
        weights = [w for _, w in features_weights]

        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()

        # Weighted sum
        blended = sum(
            features * weight
            for features, weight in zip(features_list, weights)
        )

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Weighted blend of {len(features_list)} references")

        return blended

    def max_blend(
        self,
        features_list: List[torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """Blend features using element-wise maximum.

        Emphasizes strongest features from each reference.

        Args:
            features_list: List of feature tensors
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        # Stack and take max
        stacked = torch.stack(features_list, dim=0)
        blended, _ = torch.max(stacked, dim=0)

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Max blend of {len(features_list)} references")

        return blended

    def min_blend(
        self,
        features_list: List[torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """Blend features using element-wise minimum.

        Emphasizes common features across references.

        Args:
            features_list: List of feature tensors
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        # Stack and take min
        stacked = torch.stack(features_list, dim=0)
        blended, _ = torch.min(stacked, dim=0)

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Min blend of {len(features_list)} references")

        return blended

    def average_blend(
        self,
        features_list: List[torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """Blend features using simple average.

        Args:
            features_list: List of feature tensors
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        # Average
        stacked = torch.stack(features_list, dim=0)
        blended = torch.mean(stacked, dim=0)

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Average blend of {len(features_list)} references")

        return blended

    def interpolate_blend(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        alpha: float,
        normalize: bool = True
    ) -> torch.Tensor:
        """Interpolate between two feature sets.

        Args:
            features1: First feature tensor
            features2: Second feature tensor
            alpha: Interpolation factor (0 = features1, 1 = features2)
            normalize: Normalize final features

        Returns:
            Interpolated feature tensor
        """
        # Linear interpolation
        blended = (1 - alpha) * features1 + alpha * features2

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Interpolated blend (alpha={alpha:.2f})")

        return blended

    def hierarchical_blend(
        self,
        feature_groups: List[List[Tuple[torch.Tensor, float]]],
        group_weights: List[float],
        normalize: bool = True
    ) -> torch.Tensor:
        """Hierarchical blending with grouped references.

        First blends within groups, then blends groups together.
        Useful for combining different aspects (e.g., lighting + color + composition).

        Args:
            feature_groups: List of groups, each containing (features, weight) tuples
            group_weights: Weights for each group
            normalize: Normalize final features

        Returns:
            Blended feature tensor

        Example:
            >>> # Lighting group: 50%, Color group: 30%, Composition group: 20%
            >>> result = blender.hierarchical_blend(
            ...     feature_groups=[
            ...         [(light_ref1, 0.6), (light_ref2, 0.4)],  # Lighting
            ...         [(color_ref1, 0.5), (color_ref2, 0.5)],  # Color
            ...         [(comp_ref1, 1.0)]                       # Composition
            ...     ],
            ...     group_weights=[0.5, 0.3, 0.2]
            ... )
        """
        # Blend within each group
        group_blends = [
            self.weighted_blend(group, normalize=False)
            for group in feature_groups
        ]

        # Blend groups
        blended = self.weighted_blend(
            list(zip(group_blends, group_weights)),
            normalize=normalize
        )

        logger.info(f"Hierarchical blend of {len(feature_groups)} groups")

        return blended

    def adaptive_blend(
        self,
        features_list: List[torch.Tensor],
        target_features: torch.Tensor,
        temperature: float = 1.0,
        normalize: bool = True
    ) -> torch.Tensor:
        """Adaptive blending based on similarity to target.

        Automatically weights references based on their similarity
        to a target style.

        Args:
            features_list: List of reference feature tensors
            target_features: Target style features
            temperature: Softmax temperature (lower = more selective)
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        # Compute similarities to target
        similarities = torch.stack([
            torch.nn.functional.cosine_similarity(
                features,
                target_features,
                dim=-1
            )
            for features in features_list
        ])

        # Apply softmax to get weights
        weights = torch.nn.functional.softmax(
            similarities / temperature,
            dim=0
        )

        # Weighted blend
        blended = sum(
            features * weight
            for features, weight in zip(features_list, weights)
        )

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(
            f"Adaptive blend (weights: {weights.cpu().numpy()})"
        )

        return blended

    def sequential_blend(
        self,
        features_list: List[torch.Tensor],
        blend_factor: float = 0.5,
        normalize: bool = True
    ) -> torch.Tensor:
        """Sequential blending applying each reference progressively.

        Args:
            features_list: List of feature tensors
            blend_factor: Blending factor for each step
            normalize: Normalize final features

        Returns:
            Blended feature tensor
        """
        # Start with first reference
        blended = features_list[0]

        # Progressively blend in remaining references
        for features in features_list[1:]:
            blended = (1 - blend_factor) * blended + blend_factor * features

        # Normalize if requested
        if normalize:
            blended = blended / blended.norm(dim=-1, keepdim=True)

        logger.info(f"Sequential blend of {len(features_list)} references")

        return blended

    def create_style_palette(
        self,
        base_features: torch.Tensor,
        variations: List[torch.Tensor],
        num_samples: int = 5
    ) -> List[torch.Tensor]:
        """Create style palette with variations around base.

        Args:
            base_features: Base style features
            variations: Variation feature tensors
            num_samples: Number of palette samples

        Returns:
            List of style palette samples
        """
        palette = [base_features]  # Include base

        # Create interpolations to each variation
        for variation in variations[:num_samples - 1]:
            # Sample at midpoint
            interpolated = self.interpolate_blend(
                base_features,
                variation,
                alpha=0.5,
                normalize=True
            )
            palette.append(interpolated)

        logger.info(f"Created style palette with {len(palette)} samples")

        return palette

    def find_optimal_blend(
        self,
        features_list: List[torch.Tensor],
        target_features: torch.Tensor,
        num_trials: int = 100
    ) -> Tuple[List[float], float]:
        """Find optimal blend weights to match target.

        Uses random search to find weights that best approximate target.

        Args:
            features_list: List of reference features
            target_features: Target style features
            num_trials: Number of random trials

        Returns:
            Tuple of (best weights, best similarity)
        """
        best_similarity = -1.0
        best_weights = None

        for _ in range(num_trials):
            # Random weights
            weights = torch.rand(len(features_list), device=self.device)
            weights = weights / weights.sum()

            # Blend
            blended = sum(
                features * weight
                for features, weight in zip(features_list, weights)
            )
            blended = blended / blended.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = torch.nn.functional.cosine_similarity(
                blended,
                target_features,
                dim=-1
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_weights = weights.cpu().numpy().tolist()

        logger.info(
            f"Optimal blend found (similarity={best_similarity:.3f}): "
            f"weights={best_weights}"
        )

        return best_weights, best_similarity

    def __repr__(self) -> str:
        return f"MultiReferenceBlender(device='{self.device}')"


# Export
__all__ = ['MultiReferenceBlender']
