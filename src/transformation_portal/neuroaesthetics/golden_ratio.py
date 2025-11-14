"""Golden ratio analysis and composition optimization.

The golden ratio (φ ≈ 1.618) appears throughout architectural history:
- The Parthenon
- Le Corbusier's Modulor system
- Renaissance compositions

While neurological evidence for inherent superiority remains speculative,
systematic proportional relationships create measurable visual coherence.

This module:
- Analyzes existing compositions for golden ratio adherence
- Generates optimal feature placement grids
- Scores compositional balance
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895


@dataclass
class GoldenRatioAnalysis:
    """Results of golden ratio analysis.

    Attributes:
        score: Adherence score (0-1, higher = better alignment)
        grid_points: Golden ratio grid intersection points
        feature_positions: Detected feature locations
        alignments: Feature alignments with grid points
        recommendations: Composition improvement suggestions
    """
    score: float
    grid_points: np.ndarray
    feature_positions: List[Tuple[int, int]]
    alignments: List[Dict]
    recommendations: List[str]


class GoldenRatioAnalyzer:
    """Analyze and optimize compositions using golden ratio principles.

    The golden ratio creates natural-feeling proportions. This analyzer:
    - Evaluates how well key features align with golden ratio grid
    - Identifies optimal placement for architectural elements
    - Provides actionable composition recommendations

    Example:
        >>> analyzer = GoldenRatioAnalyzer()
        >>> analysis = analyzer.analyze("luxury_interior.jpg")
        >>> print(f"Golden ratio score: {analysis.score:.2f}")
        >>> print(f"Recommendations: {analysis.recommendations}")
        >>>
        >>> # Visualize golden ratio grid
        >>> grid_viz = analyzer.visualize_grid("luxury_interior.jpg")
    """

    def __init__(
        self,
        tolerance: float = 0.05,  # 5% tolerance for alignment
        min_feature_strength: float = 0.1
    ):
        """Initialize golden ratio analyzer.

        Args:
            tolerance: Alignment tolerance (proportion of image dimension)
            min_feature_strength: Minimum feature strength to consider
        """
        self.tolerance = tolerance
        self.min_feature_strength = min_feature_strength

        logger.info(f"GoldenRatioAnalyzer initialized (tolerance={tolerance})")

    def analyze(
        self,
        image: Union[str, np.ndarray, Image.Image],
        detect_features: bool = True
    ) -> GoldenRatioAnalysis:
        """Analyze image composition using golden ratio.

        Args:
            image: Input image
            detect_features: Automatically detect salient features

        Returns:
            GoldenRatioAnalysis with score and recommendations
        """
        # Load image
        image_np = self._load_image(image)
        h, w = image_np.shape[:2]

        # Generate golden ratio grid
        grid_points = self._generate_grid(w, h)

        # Detect features
        if detect_features:
            feature_positions = self._detect_features(image_np)
        else:
            feature_positions = []

        # Calculate alignments
        alignments = self._calculate_alignments(
            feature_positions,
            grid_points,
            w, h
        )

        # Calculate overall score
        score = self._calculate_score(alignments, len(feature_positions))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            score, alignments, grid_points, w, h
        )

        return GoldenRatioAnalysis(
            score=score,
            grid_points=grid_points,
            feature_positions=feature_positions,
            alignments=alignments,
            recommendations=recommendations
        )

    def _generate_grid(
        self,
        width: int,
        height: int,
        include_phi_reciprocal: bool = True
    ) -> np.ndarray:
        """Generate golden ratio grid points.

        Creates intersection points at golden ratio divisions:
        - Primary: 1/φ ≈ 0.618
        - Secondary: 1/φ² ≈ 0.382

        Args:
            width: Image width
            height: Image height
            include_phi_reciprocal: Include reciprocal divisions

        Returns:
            Array of grid points (N, 2) as [x, y] coordinates
        """
        points = []

        # Primary golden ratio divisions
        phi_reciprocal = 1 / PHI  # ≈ 0.618

        # Vertical lines at golden ratio
        x_positions = [
            int(width * phi_reciprocal),  # Left golden point
            int(width * (1 - phi_reciprocal)),  # Right golden point
        ]

        # Horizontal lines at golden ratio
        y_positions = [
            int(height * phi_reciprocal),  # Top golden point
            int(height * (1 - phi_reciprocal)),  # Bottom golden point
        ]

        # Also include center lines (traditional rule of thirds)
        x_positions.extend([width // 3, 2 * width // 3])
        y_positions.extend([height // 3, 2 * height // 3])

        # Generate intersection points
        for x in x_positions:
            for y in y_positions:
                points.append([x, y])

        # Add edge points at golden ratio
        for x in x_positions:
            points.append([x, 0])
            points.append([x, height])

        for y in y_positions:
            points.append([0, y])
            points.append([width, y])

        return np.array(points)

    def _detect_features(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Detect salient features using corner detection and saliency.

        Args:
            image: RGB image

        Returns:
            List of (x, y) feature positions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect corners (architectural features)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=30
        )

        feature_positions = []

        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                feature_positions.append((int(x), int(y)))

        # Also use saliency detection for prominent objects
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(image)

        if success:
            # Find local maxima in saliency map
            threshold = saliency_map.max() * 0.7
            salient_points = np.argwhere(saliency_map > threshold)

            # Cluster and take centroids
            if len(salient_points) > 0:
                # Simple clustering: grid-based
                h, w = saliency_map.shape
                grid_size = 50

                for i in range(0, h, grid_size):
                    for j in range(0, w, grid_size):
                        region = salient_points[
                            (salient_points[:, 0] >= i) &
                            (salient_points[:, 0] < i + grid_size) &
                            (salient_points[:, 1] >= j) &
                            (salient_points[:, 1] < j + grid_size)
                        ]

                        if len(region) > 10:  # Significant cluster
                            centroid_y = int(region[:, 0].mean())
                            centroid_x = int(region[:, 1].mean())
                            feature_positions.append((centroid_x, centroid_y))

        return feature_positions

    def _calculate_alignments(
        self,
        features: List[Tuple[int, int]],
        grid_points: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict]:
        """Calculate how well features align with grid points.

        Args:
            features: Feature positions
            grid_points: Golden ratio grid points
            width: Image width
            height: Image height

        Returns:
            List of alignment dictionaries
        """
        alignments = []

        tolerance_px_x = width * self.tolerance
        tolerance_px_y = height * self.tolerance

        for feat_x, feat_y in features:
            # Find closest grid point
            distances = np.sqrt(
                (grid_points[:, 0] - feat_x) ** 2 +
                (grid_points[:, 1] - feat_y) ** 2
            )

            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            closest_point = grid_points[closest_idx]

            # Calculate alignment score
            # Score decreases with distance, 0 at tolerance threshold
            if closest_distance < tolerance_px_x:
                alignment_score = 1.0 - (closest_distance / tolerance_px_x)
            else:
                alignment_score = 0.0

            alignments.append({
                'feature_position': (feat_x, feat_y),
                'closest_grid_point': tuple(closest_point),
                'distance': closest_distance,
                'alignment_score': alignment_score,
                'is_aligned': closest_distance < tolerance_px_x
            })

        return alignments

    def _calculate_score(
        self,
        alignments: List[Dict],
        total_features: int
    ) -> float:
        """Calculate overall golden ratio adherence score.

        Args:
            alignments: Feature alignments
            total_features: Total number of features

        Returns:
            Score from 0-1 (higher = better alignment)
        """
        if not alignments:
            return 0.5  # Neutral score if no features

        # Average alignment score of all features
        alignment_scores = [a['alignment_score'] for a in alignments]
        avg_alignment = np.mean(alignment_scores)

        # Bonus for having features on multiple grid points
        aligned_count = sum(1 for a in alignments if a['is_aligned'])
        coverage_bonus = min(aligned_count / 4, 1.0) * 0.2  # Up to 20% bonus

        score = avg_alignment * 0.8 + coverage_bonus

        return min(score, 1.0)

    def _generate_recommendations(
        self,
        score: float,
        alignments: List[Dict],
        grid_points: np.ndarray,
        width: int,
        height: int
    ) -> List[str]:
        """Generate composition recommendations.

        Args:
            score: Overall alignment score
            alignments: Feature alignments
            grid_points: Grid points
            width: Image width
            height: Image height

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if score >= 0.8:
            recommendations.append(
                "Excellent golden ratio composition! "
                "Key features align well with golden divisions."
            )
        elif score >= 0.6:
            recommendations.append(
                "Good composition with some golden ratio alignment. "
                "Consider minor adjustments for optimal balance."
            )
        else:
            recommendations.append(
                "Composition could benefit from golden ratio principles. "
                "Consider repositioning key elements."
            )

        # Identify poorly aligned features
        poor_alignments = [a for a in alignments if a['alignment_score'] < 0.3]

        if poor_alignments:
            recommendations.append(
                f"{len(poor_alignments)} key features are not aligned "
                "with golden ratio divisions. Consider cropping or reframing."
            )

        # Find empty grid points (opportunities)
        used_points = set(a['closest_grid_point'] for a in alignments if a['is_aligned'])
        phi_reciprocal = 1 / PHI

        key_points = [
            (int(width * phi_reciprocal), int(height * phi_reciprocal)),
            (int(width * (1 - phi_reciprocal)), int(height * phi_reciprocal)),
            (int(width * phi_reciprocal), int(height * (1 - phi_reciprocal))),
            (int(width * (1 - phi_reciprocal)), int(height * (1 - phi_reciprocal))),
        ]

        empty_key_points = [
            p for p in key_points
            if not any(
                np.sqrt((p[0] - up[0])**2 + (p[1] - up[1])**2) < width * 0.05
                for up in used_points
            )
        ]

        if empty_key_points:
            recommendations.append(
                f"Consider placing focal elements at unused golden points "
                f"(particularly at {len(empty_key_points)} key intersections)."
            )

        return recommendations

    def visualize_grid(
        self,
        image: Union[str, np.ndarray, Image.Image],
        analysis: Optional[GoldenRatioAnalysis] = None,
        line_color: Tuple[int, int, int] = (255, 215, 0),  # Gold color
        line_thickness: int = 2,
        show_features: bool = True
    ) -> np.ndarray:
        """Visualize golden ratio grid on image.

        Args:
            image: Input image
            analysis: Pre-computed analysis (computes if None)
            line_color: RGB color for grid lines
            line_thickness: Line thickness in pixels
            show_features: Show detected features

        Returns:
            Image with golden ratio grid overlay
        """
        # Load image
        image_np = self._load_image(image).copy()
        h, w = image_np.shape[:2]

        # Get or compute analysis
        if analysis is None:
            analysis = self.analyze(image)

        # Draw grid lines
        phi_reciprocal = 1 / PHI

        # Vertical lines
        for x_ratio in [phi_reciprocal, 1 - phi_reciprocal, 1/3, 2/3]:
            x = int(w * x_ratio)
            cv2.line(image_np, (x, 0), (x, h), line_color, line_thickness)

        # Horizontal lines
        for y_ratio in [phi_reciprocal, 1 - phi_reciprocal, 1/3, 2/3]:
            y = int(h * y_ratio)
            cv2.line(image_np, (0, y), (w, y), line_color, line_thickness)

        # Draw grid intersection points
        for point in analysis.grid_points:
            cv2.circle(
                image_np,
                tuple(point.astype(int)),
                5,
                line_color,
                -1
            )

        # Show features if requested
        if show_features and analysis.feature_positions:
            for feat_x, feat_y in analysis.feature_positions:
                # Color based on alignment
                alignment = next(
                    (a for a in analysis.alignments
                     if a['feature_position'] == (feat_x, feat_y)),
                    None
                )

                if alignment and alignment['is_aligned']:
                    color = (0, 255, 0)  # Green for aligned
                else:
                    color = (255, 0, 0)  # Red for not aligned

                cv2.circle(image_np, (feat_x, feat_y), 8, color, 2)

        # Add score text
        score_text = f"Golden Ratio Score: {analysis.score:.2f}"
        cv2.putText(
            image_np,
            score_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            image_np,
            score_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            line_color,
            2,
            cv2.LINE_AA
        )

        return image_np

    def get_optimal_crop(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_aspect: Optional[float] = None
    ) -> Tuple[int, int, int, int]:
        """Calculate optimal crop using golden ratio.

        Args:
            image: Input image
            target_aspect: Target aspect ratio (uses golden ratio if None)

        Returns:
            Crop box as (x1, y1, x2, y2)
        """
        # Load image
        image_np = self._load_image(image)
        h, w = image_np.shape[:2]

        # Use golden ratio as default aspect ratio
        if target_aspect is None:
            target_aspect = PHI

        current_aspect = w / h

        if current_aspect > target_aspect:
            # Too wide - crop width
            new_width = int(h * target_aspect)
            x_offset = (w - new_width) // 2
            return (x_offset, 0, x_offset + new_width, h)
        else:
            # Too tall - crop height
            new_height = int(w / target_aspect)
            y_offset = (h - new_height) // 2
            return (0, y_offset, w, y_offset + new_height)

    def _load_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image as RGB numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        else:
            pil_img = Image.open(image).convert("RGB")
            return np.array(pil_img)

    def __repr__(self) -> str:
        return (
            f"GoldenRatioAnalyzer(tolerance={self.tolerance})"
        )
