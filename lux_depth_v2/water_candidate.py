"""Water Candidate Detection (PR-W1) - Heuristic-based water mask generator.

This module provides CPU-only heuristic water detection for pool/ocean scenes.
Uses multi-cue analysis (chromaticity, specular, texture, planarity) to generate
high-recall candidate water masks.

Stage: Production (PR-W1 Complete)
Dependencies: numpy, scipy, scikit-image (CPU-only, CI-safe)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class SceneContext(Enum):
    """Scene context for water detection tuning."""
    POOL = "pool"
    OCEAN = "ocean"
    UNKNOWN = "unknown"


@dataclass
class WaterDetectionParams:
    """Tunable parameters for water detection heuristics."""
    # Chromaticity
    pool_hue_range: Tuple[float, float] = (170, 210)  # Cyan/blue
    ocean_hue_range: Tuple[float, float] = (160, 220)  # Broader blue-green
    saturation_min: float = 0.15
    value_min: float = 0.20
    
    # Specularness (reflections)
    specular_highlight_threshold: float = 0.85
    specular_low_sat_threshold: float = 0.30
    
    # Texture/entropy
    texture_entropy_max: float = 5.0  # Lower = smoother
    
    # Planarity (if depth available)
    depth_gradient_threshold: float = 0.05
    
    # Post-processing
    morph_close_kernel: int = 5
    morph_open_kernel: int = 3
    min_component_area_px: int = 1000
    max_components_kept: int = 3
    hole_fill_enabled: bool = True
    
    # Confidence weighting
    chromaticity_weight: float = 0.35
    specular_weight: float = 0.25
    texture_weight: float = 0.20
    planarity_weight: float = 0.15
    component_stability_weight: float = 0.05


@dataclass
class WaterCandidateResult:
    """Output of water candidate detection."""
    mask: np.ndarray  # HxW float32, 0.0-1.0
    confidence: float  # 0.0-1.0
    coverage: float  # Fraction of image
    coverage_px: int
    feature_scores: Dict[str, float]  # Individual cue scores
    debug_info: Dict[str, Any]  # Intermediate masks, thresholds, etc.


class WaterCandidateDetector:
    """CPU-only heuristic water mask generator (PR-W1)."""
    
    def __init__(self, params: Optional[WaterDetectionParams] = None):
        """Initialize water candidate detector.
        
        Args:
            params: Optional detection parameters (uses defaults if None)
        """
        self.params = params or WaterDetectionParams()
    
    def detect(
        self,
        rgb01: np.ndarray,  # HxWx3 float32
        depth01: Optional[np.ndarray] = None,  # HxW float32
        scene_context: SceneContext = SceneContext.UNKNOWN
    ) -> WaterCandidateResult:
        """Generate water candidate mask using multi-cue heuristics.
        
        Args:
            rgb01: RGB image (HxWx3 float32 in [0,1])
            depth01: Optional depth map (HxW float32)
            scene_context: Scene type hint (pool/ocean/unknown)
            
        Returns:
            WaterCandidateResult with mask and confidence
        """
        h, w = rgb01.shape[:2]
        
        # Convert to HSV for chromaticity analysis
        hsv = self._rgb_to_hsv(rgb01)
        
        # Feature extraction
        chroma_mask, chroma_score = self._chromaticity_cue(hsv, scene_context)
        specular_mask, specular_score = self._specular_cue(rgb01, hsv)
        texture_mask, texture_score = self._texture_cue(rgb01)
        
        planarity_mask = None
        planarity_score = 0.0
        if depth01 is not None:
            planarity_mask, planarity_score = self._planarity_cue(depth01)
        
        # Combine cues (weighted OR)
        combined_mask = self._combine_cues(
            chroma_mask, specular_mask, texture_mask, planarity_mask
        )
        
        # Post-processing
        refined_mask = self._postprocess(combined_mask)
        
        # Component analysis and stability
        final_mask, component_score = self._component_filtering(refined_mask)
        
        # Compute overall confidence
        confidence = self._compute_confidence(
            chroma_score, specular_score, texture_score, 
            planarity_score, component_score
        )
        
        coverage_px = int(np.sum(final_mask > 0.5))
        coverage = coverage_px / (h * w)
        
        return WaterCandidateResult(
            mask=final_mask,
            confidence=confidence,
            coverage=coverage,
            coverage_px=coverage_px,
            feature_scores={
                "chromaticity": chroma_score,
                "specular": specular_score,
                "texture": texture_score,
                "planarity": planarity_score,
                "component_stability": component_score,
            },
            debug_info={
                "chroma_mask": chroma_mask,
                "specular_mask": specular_mask,
                "texture_mask": texture_mask,
                "planarity_mask": planarity_mask,
                "combined_mask": combined_mask,
                "scene_context": scene_context.value,
            }
        )
    
    def _chromaticity_cue(
        self, hsv: np.ndarray, scene_context: SceneContext
    ) -> Tuple[np.ndarray, float]:
        """Blue/green dominance in HSV (pool vs ocean tuned)."""
        hue = hsv[:, :, 0] * 360  # Convert to degrees
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Select hue range based on scene context
        if scene_context == SceneContext.POOL:
            hue_min, hue_max = self.params.pool_hue_range
        else:  # OCEAN or UNKNOWN
            hue_min, hue_max = self.params.ocean_hue_range
        
        # Hue in range + sufficient saturation + not too dark
        hue_match = (hue >= hue_min) & (hue <= hue_max)
        sat_match = sat >= self.params.saturation_min
        val_match = val >= self.params.value_min
        
        mask = (hue_match & sat_match & val_match).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _specular_cue(
        self, rgb01: np.ndarray, hsv: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """High highlights + low saturation pockets (water reflections)."""
        from scipy import ndimage
        
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        
        # High value (bright) with low saturation (specular reflection)
        specular = (val >= self.params.specular_highlight_threshold) & \
                   (sat <= self.params.specular_low_sat_threshold)
        
        # Dilate slightly to capture reflection context
        mask = ndimage.binary_dilation(specular, iterations=2).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _texture_cue(self, rgb01: np.ndarray) -> Tuple[np.ndarray, float]:
        """Water tends to be lower-frequency than foliage/stone."""
        from scipy import ndimage
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        
        # Convert to grayscale
        gray = np.mean(rgb01, axis=2)
        
        # Local entropy (lower for water)
        # Use uint8 for entropy calculation
        gray_uint8 = (gray * 255).astype(np.uint8)
        local_entropy = entropy(gray_uint8, disk(5))
        
        # Normalize and invert (low entropy = high score)
        entropy_norm = local_entropy / 8.0  # Max entropy for 8-bit
        mask = (entropy_norm <= self.params.texture_entropy_max / 8.0).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _planarity_cue(
        self, depth01: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Low depth-gradient bands if depth is present (optional)."""
        from scipy import ndimage
        
        # Compute depth gradients
        grad_x = ndimage.sobel(depth01, axis=1)
        grad_y = ndimage.sobel(depth01, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Low gradient = planar surface
        mask = (grad_mag <= self.params.depth_gradient_threshold).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _combine_cues(
        self,
        chroma_mask: np.ndarray,
        specular_mask: np.ndarray,
        texture_mask: np.ndarray,
        planarity_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Weighted combination of cues."""
        combined = (
            self.params.chromaticity_weight * chroma_mask +
            self.params.specular_weight * specular_mask +
            self.params.texture_weight * texture_mask
        )
        
        if planarity_mask is not None:
            combined += self.params.planarity_weight * planarity_mask
        
        # Normalize to [0, 1]
        combined = np.clip(combined, 0.0, 1.0)
        return combined
    
    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Morphological operations and hole filling."""
        from scipy import ndimage
        from skimage.morphology import disk, binary_closing, binary_opening
        
        # Threshold to binary
        binary_mask = mask > 0.5
        
        # Morphological close (fill small gaps)
        if self.params.morph_close_kernel > 0:
            binary_mask = binary_closing(
                binary_mask, 
                disk(self.params.morph_close_kernel)
            )
        
        # Morphological open (remove noise)
        if self.params.morph_open_kernel > 0:
            binary_mask = binary_opening(
                binary_mask,
                disk(self.params.morph_open_kernel)
            )
        
        # Fill holes
        if self.params.hole_fill_enabled:
            binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        return binary_mask.astype(np.float32)
    
    def _component_filtering(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Keep top-K components by area; suppress tiny blobs."""
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(mask > 0.5)
        
        if num_features == 0:
            return mask, 0.0
        
        # Compute component sizes
        component_sizes = ndimage.sum(
            mask, labeled, range(1, num_features + 1)
        )
        
        # Filter by minimum area
        valid_components = []
        for i, size in enumerate(component_sizes, start=1):
            if size >= self.params.min_component_area_px:
                valid_components.append((i, size))
        
        if not valid_components:
            return np.zeros_like(mask), 0.0
        
        # Keep top-K largest
        valid_components.sort(key=lambda x: x[1], reverse=True)
        top_k = valid_components[:self.params.max_components_kept]
        
        # Rebuild mask with only valid components
        filtered_mask = np.zeros_like(mask)
        for component_id, _ in top_k:
            filtered_mask[labeled == component_id] = 1.0
        
        # Stability score: ratio of kept area to original
        stability = float(np.sum(filtered_mask) / max(np.sum(mask), 1))
        
        return filtered_mask, stability
    
    def _compute_confidence(
        self,
        chroma_score: float,
        specular_score: float,
        texture_score: float,
        planarity_score: float,
        component_score: float
    ) -> float:
        """Weighted combination of feature scores."""
        confidence = (
            self.params.chromaticity_weight * chroma_score +
            self.params.specular_weight * specular_score +
            self.params.texture_weight * texture_score +
            self.params.planarity_weight * planarity_score +
            self.params.component_stability_weight * component_score
        )
        return float(np.clip(confidence, 0.0, 1.0))
    
    # Helper color conversion methods
    def _rgb_to_hsv(self, rgb01: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        from skimage.color import rgb2hsv
        return rgb2hsv(rgb01)
