"""Water Candidate Detection (PR-W1) - Heuristic-based water mask generator.

This module provides CPU-only heuristic water detection for pool/ocean scenes.
Uses multi-cue analysis (chromaticity, specular, texture, planarity) to generate
high-recall candidate water masks.

Stage: Production (PR-W1 Complete)
Dependencies: numpy, scipy (optional), scikit-image (optional), torch (optional)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional dependencies (graceful degradation if missing)
try:
    from scipy import ndimage as scipy_ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_ndimage = None

try:
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import disk as skimage_disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    skimage_entropy = None
    skimage_disk = None


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
    
    # PR-W1.2: Confidence suppressors (false trigger reduction)
    suppressors_enabled: bool = True
    
    # Flat blue surface suppressor (targets blue walls)
    flat_surface_suppressor_enabled: bool = True
    flat_surface_edge_energy_threshold: float = 0.02  # Low edge energy = flat
    flat_surface_specular_fraction_threshold: float = 0.10  # Low specular = not water
    flat_surface_penalty: float = 0.5  # Confidence *= 0.5
    
    # Architectural glass suppressor (targets glass buildings)
    glass_suppressor_enabled: bool = True
    glass_edge_alignment_threshold: float = 0.15  # High axis-alignment (0°/90°)
    glass_grid_score_threshold: float = 0.25  # Grid-like gradient pattern
    glass_penalty: float = 0.6  # Confidence *= 0.6


@dataclass
class WaterCandidateResult:
    """Output of water candidate detection."""
    mask: np.ndarray  # HxW float32, 0.0-1.0
    confidence: float  # 0.0-1.0
    coverage: float  # Fraction of image
    coverage_px: int
    feature_scores: Dict[str, float]  # Individual cue scores
    debug_info: Dict[str, Any]  # Intermediate masks, thresholds, etc.
    suppressor_telemetry: Dict[str, Any] = None  # PR-W1.2: Suppressor diagnostics


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
        
        # PR-W1.2: Apply confidence suppressors (false trigger reduction)
        # Always call suppressors (they handle enabled flag internally)
        confidence, suppressor_telemetry = self._apply_suppressors(
            confidence=confidence,
            rgb01=rgb01,
            hsv=hsv,
            final_mask=final_mask,
            specular_mask=specular_mask,
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
            },
            suppressor_telemetry=suppressor_telemetry,
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
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        
        # High value (bright) with low saturation (specular reflection)
        specular = (val >= self.params.specular_highlight_threshold) & \
                   (sat <= self.params.specular_low_sat_threshold)
        
        # Dilate slightly to capture reflection context (if scipy available)
        if SCIPY_AVAILABLE:
            mask = scipy_ndimage.binary_dilation(specular, iterations=2).astype(np.float32)
        else:
            # Fallback: simple mask without dilation
            mask = specular.astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _texture_cue(self, rgb01: np.ndarray) -> Tuple[np.ndarray, float]:
        """Water tends to be lower-frequency than foliage/stone."""
        if not SKIMAGE_AVAILABLE:
            # Fallback: return neutral mask
            mask = np.ones_like(rgb01[:, :, 0], dtype=np.float32) * 0.5
            return mask, 0.5
        
        # Convert to grayscale
        gray = np.mean(rgb01, axis=2)
        
        # Local entropy (lower for water)
        # Use uint8 for entropy calculation
        gray_uint8 = (gray * 255).astype(np.uint8)
        local_entropy = skimage_entropy(gray_uint8, skimage_disk(5))
        
        # Normalize and invert (low entropy = high score)
        entropy_norm = local_entropy / 8.0  # Max entropy for 8-bit
        mask = (entropy_norm <= self.params.texture_entropy_max / 8.0).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _planarity_cue(
        self, depth01: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Low depth-gradient bands if depth is present (optional)."""
        if not SCIPY_AVAILABLE:
            # Fallback: return neutral mask
            mask = np.ones_like(depth01, dtype=np.float32) * 0.5
            return mask, 0.5
        
        # Compute depth gradients
        grad_x = scipy_ndimage.sobel(depth01, axis=1)
        grad_y = scipy_ndimage.sobel(depth01, axis=0)
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
        if not SCIPY_AVAILABLE or not SKIMAGE_AVAILABLE:
            # Fallback: simple threshold
            return (mask > 0.5).astype(np.float32)
        
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
            binary_mask = scipy_ndimage.binary_fill_holes(binary_mask)
        
        return binary_mask.astype(np.float32)
    
    def _component_filtering(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Keep top-K components by area; suppress tiny blobs."""
        if not SCIPY_AVAILABLE:
            # Fallback: return mask as-is
            return mask, 1.0
        
        # Label connected components
        labeled, num_features = scipy_ndimage.label(mask > 0.5)
        
        if num_features == 0:
            return mask, 0.0
        
        # Compute component sizes
        component_sizes = scipy_ndimage.sum(
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
    
    def _apply_suppressors(
        self,
        confidence: float,
        rgb01: np.ndarray,
        hsv: np.ndarray,
        final_mask: np.ndarray,
        specular_mask: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply confidence suppressors to reduce false triggers (PR-W1.2).
        
        Args:
            confidence: Base confidence score
            rgb01: RGB image (HxWx3 float32)
            hsv: HSV image (HxWx3 float32)
            final_mask: Final water mask
            specular_mask: Specular highlights mask
            
        Returns:
            Tuple of (adjusted_confidence, telemetry_dict)
        """
        original_confidence = confidence
        telemetry = {
            "original_confidence": float(original_confidence),
            "suppressors_applied": [],
        }
        
        # If suppressors globally disabled, return early with telemetry
        if not self.params.suppressors_enabled:
            telemetry["final_confidence"] = float(confidence)
            telemetry["total_suppression"] = 0.0
            return confidence, telemetry
        
        # Flat blue surface suppressor (targets blue walls)
        if self.params.flat_surface_suppressor_enabled:
            is_flat_surface, flat_metrics = self._detect_flat_blue_surface(
                rgb01, hsv, final_mask, specular_mask
            )
            telemetry["flat_surface_detector"] = flat_metrics
            
            if is_flat_surface:
                confidence *= self.params.flat_surface_penalty
                telemetry["suppressors_applied"].append("flat_surface")
                telemetry["flat_surface_penalty"] = float(self.params.flat_surface_penalty)
        
        # Architectural glass suppressor (targets glass buildings)
        if self.params.glass_suppressor_enabled:
            is_glass, glass_metrics = self._detect_architectural_glass(rgb01, final_mask)
            telemetry["glass_detector"] = glass_metrics
            
            if is_glass:
                confidence *= self.params.glass_penalty
                telemetry["suppressors_applied"].append("architectural_glass")
                telemetry["glass_penalty"] = float(self.params.glass_penalty)
        
        telemetry["final_confidence"] = float(confidence)
        telemetry["total_suppression"] = float(original_confidence - confidence)
        
        return confidence, telemetry
    
    def _detect_flat_blue_surface(
        self,
        rgb01: np.ndarray,
        hsv: np.ndarray,
        mask: np.ndarray,
        specular_mask: np.ndarray,
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect flat blue surfaces (e.g., painted walls) to suppress false triggers.
        
        Uses edge energy and specular fraction to identify non-water flat surfaces.
        
        Args:
            rgb01: RGB image (HxWx3 float32)
            hsv: HSV image (HxWx3 float32)
            mask: Water candidate mask
            specular_mask: Specular highlights mask
            
        Returns:
            Tuple of (is_flat_surface, metrics_dict)
        """
        # Compute edge energy within masked region
        gray = np.mean(rgb01, axis=2)
        
        if SCIPY_AVAILABLE:
            # Sobel edge detection
            grad_x = scipy_ndimage.sobel(gray, axis=1)
            grad_y = scipy_ndimage.sobel(gray, axis=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        else:
            # Fallback: simple gradient
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute edge energy within mask (0-1 normalized)
        masked_edges = edge_magnitude * (mask > 0.5)
        total_edge_energy = float(np.mean(masked_edges))
        
        # Compute specular fraction within mask
        masked_specular = specular_mask * (mask > 0.5)
        specular_fraction = float(np.sum(masked_specular) / max(np.sum(mask > 0.5), 1))
        
        # Detection logic: low edge energy + low specular = flat blue wall
        is_flat = (
            total_edge_energy < self.params.flat_surface_edge_energy_threshold and
            specular_fraction < self.params.flat_surface_specular_fraction_threshold
        )
        
        metrics = {
            "edge_energy": total_edge_energy,
            "specular_fraction": specular_fraction,
            "is_flat_surface": is_flat,
            "edge_threshold": float(self.params.flat_surface_edge_energy_threshold),
            "specular_threshold": float(self.params.flat_surface_specular_fraction_threshold),
        }
        
        return is_flat, metrics
    
    def _detect_architectural_glass(
        self,
        rgb01: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect architectural glass patterns (grid-like, axis-aligned edges).
        
        Uses edge orientation analysis to identify rectilinear window/building patterns.
        
        Args:
            rgb01: RGB image (HxWx3 float32)
            mask: Water candidate mask
            
        Returns:
            Tuple of (is_glass, metrics_dict)
        """
        gray = np.mean(rgb01, axis=2)
        
        if not SCIPY_AVAILABLE:
            # Fallback: cannot reliably detect glass without gradient analysis
            return False, {
                "edge_alignment_score": 0.0,
                "grid_score": 0.0,
                "is_glass": False,
                "reason": "scipy_unavailable",
            }
        
        # Compute gradients
        grad_x = scipy_ndimage.sobel(gray, axis=1)
        grad_y = scipy_ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute edge orientations (in radians)
        # Use arctan2 for full 360° range, then normalize to [0, π]
        edge_orientation = np.arctan2(grad_y, grad_x)  # -π to π
        edge_orientation = np.abs(edge_orientation)  # 0 to π
        
        # Focus on edges within mask (only where we detected water)
        mask_binary = mask > 0.5
        if not np.any(mask_binary):
            return False, {
                "edge_alignment_score": 0.0,
                "grid_score": 0.0,
                "is_glass": False,
                "reason": "no_mask",
            }
        
        # Apply mask to edge magnitude
        masked_edges = edge_magnitude.copy()
        masked_edges[~mask_binary] = 0
        
        # Threshold edges (top 20% strongest within mask)
        mask_edge_values = masked_edges[mask_binary]
        if len(mask_edge_values) == 0 or np.max(mask_edge_values) == 0:
            return False, {
                "edge_alignment_score": 0.0,
                "grid_score": 0.0,
                "is_glass": False,
                "reason": "no_edges_in_mask",
            }
        
        edge_threshold = np.percentile(mask_edge_values, 80)
        strong_edges = (masked_edges > edge_threshold) & mask_binary
        
        if np.sum(strong_edges) < 10:
            return False, {
                "edge_alignment_score": 0.0,
                "grid_score": 0.0,
                "is_glass": False,
                "reason": "insufficient_strong_edges",
            }
        
        # Compute alignment to horizontal/vertical axes
        # Bin orientations: 0° (horizontal), 90° (vertical) ± tolerance
        orientations = edge_orientation[strong_edges]
        
        # Horizontal: near 0 or π (±15°)
        tolerance = 0.26  # ~15 degrees in radians
        horizontal_aligned = np.sum((orientations < tolerance) | (orientations > (np.pi - tolerance)))
        # Vertical: near π/2 (±15°)
        vertical_aligned = np.sum(np.abs(orientations - np.pi / 2) < tolerance)
        
        total_strong_edges = len(orientations)
        alignment_score = float((horizontal_aligned + vertical_aligned) / total_strong_edges)
        
        # Grid score: spatial regularity (simplified heuristic)
        # High grid score = edges cluster along rows/columns
        edge_coords = np.argwhere(strong_edges)
        if len(edge_coords) > 20:
            # Check if edges align in rows/columns
            # Count distinct rows and columns with edges
            unique_rows = len(np.unique(edge_coords[:, 0]))
            unique_cols = len(np.unique(edge_coords[:, 1]))
            h, w = rgb01.shape[:2]
            
            # If edges span many rows/cols but are axis-aligned, likely a grid
            # Grid has low row/col diversity relative to total size
            row_density = unique_rows / h
            col_density = unique_cols / w
            
            # If alignment is high and edges are distributed, compute grid score
            if alignment_score > 0.3:
                # High alignment + moderate spread = grid
                grid_score = alignment_score * min(row_density + col_density, 1.0)
            else:
                grid_score = 0.0
        else:
            grid_score = 0.0
        
        # Detection logic: high axis-alignment or grid pattern
        is_glass = (
            alignment_score > self.params.glass_edge_alignment_threshold or
            grid_score > self.params.glass_grid_score_threshold
        )
        
        metrics = {
            "edge_alignment_score": alignment_score,
            "grid_score": grid_score,
            "is_glass": is_glass,
            "alignment_threshold": float(self.params.glass_edge_alignment_threshold),
            "grid_threshold": float(self.params.glass_grid_score_threshold),
            "horizontal_aligned_fraction": float(horizontal_aligned / total_strong_edges),
            "vertical_aligned_fraction": float(vertical_aligned / total_strong_edges),
            "total_strong_edges": int(total_strong_edges),
        }
        
        return is_glass, metrics
    
    # Helper color conversion methods
    def _rgb_to_hsv(self, rgb01: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        from skimage.color import rgb2hsv
        return rgb2hsv(rgb01)
