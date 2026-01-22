"""Depth-zoning preprocessing layer (Stage A.5).

Generates depth-conditioned zone masks between DA3 depth estimation (Stage A)
and V2 enhancement (Stage B).

Architecture:
    Stage A (DA3) → Stage A.5 (Depth Zones) → Stage B (V2)

Reference:
    - docs/architecture/DEPTH_ZONING_SPEC_1PAGE.md
    - docs/architecture/DEPTH_ZONING_ARCHITECTURE.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal
import json
import logging

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class DepthZoneConfig:
    """Configuration for depth-zoning preprocessing layer (Stage A.5)."""

    # Execution mode
    enabled: bool = False  # Default: disabled
    mode: Literal["off", "preview", "apply"] = "off"
    # off: Skip depth-zoning entirely
    # preview: Generate zone masks and diagnostics, do NOT apply operators
    # apply: Generate masks AND pass to V2 (operators in Phase 4)

    # Zone thresholds (4 values for 4 zones: Z1, Z2, Z3, Z4)
    percentiles: Tuple[float, float, float] = (10.0, 35.0, 65.0)
    # Z1 (foreground): [0, P10]
    # Z2 (mid-foreground): (P10, P35]
    # Z3 (mid-background): (P35, P65]
    # Z4 (far field): (P65, 1.0]

    # Valid depth mask configuration (NEW - CRITICAL)
    exclude_invalid_depth: bool = True  # Exclude NaN/Inf from percentile computation
    sky_saturation_threshold: float = 0.95  # Exclude depth > 0.95 if brightness > 0.85
    min_valid_coverage_pct: float = 30.0  # Warn if valid depth < 30%
    valid_depth_threshold: float = 0.30  # Minimum valid pixel coverage (0.0-1.0)

    # Zone refinement
    morphology_iterations: int = 2  # Morphological cleanup iterations
    blend_sigma: float = 5.0  # Gaussian smoothing for smooth transitions
    min_zone_area_pct: float = 0.02  # Suppress zones smaller than 2% of image

    # Sky handling heuristic (CONSERVATIVE - Phase 2 Hardening)
    apply_sky_heuristic: bool = False  # Default: OFF (conservative)
    sky_brightness_threshold: float = 0.90  # Conservative (was 0.85)
    z4_sky_sharpening_weight: float = 0.1  # Reduced sharpening for sky regions

    # Depth convention override (Phase 2 Hardening)
    depth_convention: str = "auto"  # "auto"|"near_to_far_increasing"|"near_to_far_decreasing"

    # Canary test mode (Phase 4)
    exaggerate_foreground: bool = False  # Force 90% Z1 for canary testing

    # Per-zone operator config (PHASE 4 - NOT IMPLEMENTED IN MVP)
    enable_local_operators: bool = False
    zone_micro_contrast: Tuple[float, float, float, float] = (0.15, 0.10, 0.05, 0.0)
    zone_clarity: Tuple[float, float, float, float] = (0.20, 0.10, 0.05, 0.0)
    zone_sharpening: Tuple[float, float, float, float] = (1.0, 0.8, 0.5, 0.3)
    zone_exposure_offset: Tuple[float, float, float, float] = (-0.05, 0.0, 0.05, 0.10)

    # Artifacts
    save_zone_masks: bool = True  # Save Z1-Z4 as separate 16-bit PNGs
    save_preview_overlay: bool = True  # Save color-coded visualization
    save_zone_stats: bool = True  # Save JSON with histograms and coverage

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.mode in ["off", "preview", "apply"], f"Invalid mode: {self.mode}"
        assert len(self.percentiles) == 3, "Must provide 3 percentiles for 4 zones"
        assert all(0 < p < 100 for p in self.percentiles), "Percentiles must be in (0, 100)"
        assert self.percentiles == tuple(sorted(self.percentiles)), "Percentiles must be increasing"
        assert 0.0 <= self.blend_sigma <= 50.0, "blend_sigma must be in [0, 50]"
        assert 0.0 <= self.min_valid_coverage_pct <= 100.0, "min_valid_coverage_pct must be in [0, 100]"
        assert 0.0 <= self.valid_depth_threshold <= 1.0, "valid_depth_threshold must be in [0, 1]"
        assert self.morphology_iterations >= 0, "morphology_iterations must be >= 0"
        assert self.depth_convention in ["auto", "near_to_far_increasing", "near_to_far_decreasing"], (
            f"Invalid depth_convention: {self.depth_convention}"
        )


@dataclass
class ZoneArtifacts:
    """Results from depth-zoning processing."""

    zones: np.ndarray  # (H, W, 4) zone weights [Z1, Z2, Z3, Z4], sum to 1.0 per pixel
    thresholds: Dict[str, float]  # {"P10": 0.10, "P35": 0.35, "P65": 0.65}
    depth_normalized: np.ndarray  # (H, W) depth map [0, 1] float32
    coverage_stats: Dict[str, float]  # {"Z1_pct": 8.5, "Z2_pct": ...}
    zone_stats: Dict[str, Dict]  # Per-zone mean/std depth, brightness
    depth_convention: Dict[str, Any]  # Direction, min, max, p50, valid_coverage_pct


class DepthZoneGenerator:
    """Generate depth-based processing zones with diagnostics."""

    def __init__(self, config: DepthZoneConfig):
        """Initialize zone generator.

        Args:
            config: Depth zone configuration
        """
        self.config = config
        config.validate()
        self._sky_mask = None  # Track sky mask for diagnostic overlay

    def generate_zones(self, depth: np.ndarray, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Generate 4-zone masks from depth map.

        Args:
            depth: (H, W) depth map (near=small, far=large)
            image: (H, W, 3) RGB image for sky heuristic (optional)

        Returns:
            zones: (H, W, 4) zone weights [Z1, Z2, Z3, Z4], sum to 1.0 per pixel
            stats: dict with coverage %, depth histograms, percentile drift
        """
        # CANARY TEST MODE: Force exaggerated foreground
        if self.config.exaggerate_foreground:
            logger.warning("CANARY TEST MODE: Forcing 90% Z1 foreground weight")
            zones = self.generate_exaggerated_foreground_zones(depth, foreground_weight=0.90)

            # Generate valid mask and depth stats for logging
            valid_mask = self._generate_valid_depth_mask(depth, image)
            depth_convention = self._validate_depth_convention(depth, valid_mask)

            # Compute stats for logging
            stats = {
                "coverage": {"Z1": 90.0, "Z2": 3.3, "Z3": 3.3, "Z4": 3.3},
                "thresholds": {"P10": 0.0, "P35": 0.0, "P65": 0.0},
                "exaggerated": True,
                "valid_coverage": float(valid_mask.sum() / valid_mask.size),
                "depth_convention": depth_convention,
            }

            return zones, stats

        # 1. Generate valid depth mask
        valid_mask = self._generate_valid_depth_mask(depth, image)
        valid_coverage = valid_mask.sum() / valid_mask.size

        # 2. Validate depth convention
        depth_convention = self._validate_depth_convention(depth, valid_mask)

        # 3. Check for low valid coverage (fallback to uniform zones)
        if valid_coverage < self.config.valid_depth_threshold:
            logger.warning(
                f"Valid depth coverage {valid_coverage:.1%} < {self.config.valid_depth_threshold:.1%}, using uniform zones"
            )
            zones = self._generate_uniform_zones(depth.shape)
            stats = {
                "fallback": "uniform",
                "valid_coverage": float(valid_coverage),
                "depth_convention": depth_convention,
                "coverage": {"Z1": 0.25, "Z2": 0.25, "Z3": 0.25, "Z4": 0.25},
            }
            return zones, stats

        # 4. Compute percentiles on valid pixels only
        valid_depth = depth[valid_mask]
        p_values = np.percentile(valid_depth, self.config.percentiles)
        thresholds = {f"P{int(p)}": float(v) for p, v in zip(self.config.percentiles, p_values)}

        # 5. Generate zone masks (hard thresholds)
        zone_masks = self._generate_zone_masks(depth, p_values, valid_mask)

        # 6. Apply morphology cleanup
        if self.config.morphology_iterations > 0:
            zone_masks = self._apply_morphology(zone_masks)

        # 7. Blend zones with Gaussian
        zones = self._blend_zones(zone_masks)

        # 8. Apply sky brightness heuristic if image provided
        if image is not None and self.config.mode != "off":
            zones = self._apply_sky_heuristic(zones, image)

        # 9. Compute diagnostics
        stats = self._compute_zone_stats(depth, zones, valid_mask, thresholds, depth_convention)

        return zones, stats

    def _generate_valid_depth_mask(self, depth: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """Exclude NaN, Inf, and saturated sky from percentile computation.

        Saturation exclusion is CONSERVATIVE to avoid deleting real surfaces.

        Args:
            depth: (H, W) depth map
            image: (H, W, 3) RGB image for brightness calculation (optional)

        Returns:
            valid_mask: (H, W) boolean array, True for valid pixels
        """
        valid_mask = np.ones(depth.shape, dtype=bool)

        # Track exclusion counts for logging
        nan_mask = ~np.isfinite(depth)
        valid_mask &= ~nan_mask

        # Exclude out-of-range values
        out_of_range_mask = (depth < 0.0) | (depth > 1.0)
        valid_mask &= ~out_of_range_mask

        # Exclude low saturation (very near surfaces) - CONSERVATIVE
        # Only exclude if spatially contiguous (likely invalid, not real surface)
        low_sat_mask = np.zeros_like(depth, dtype=bool)
        low_sat_threshold = 0.001  # Near-zero
        low_sat_candidates = depth < low_sat_threshold

        if low_sat_candidates.sum() > 0.05 * depth.size:  # >5% of image
            # Use connected components to exclude only large contiguous regions
            try:
                labeled, num_features = ndimage.label(low_sat_candidates)
                for i in range(1, num_features + 1):
                    component = labeled == i
                    if component.sum() > 0.02 * depth.size:  # >2% of image
                        low_sat_mask |= component
            except Exception as e:
                logger.debug(f"Low saturation connected component analysis failed: {e}")

        valid_mask &= ~low_sat_mask

        # Exclude high saturation (sky) - CONSERVATIVE
        # Only exclude if bright AND spatially contiguous AND top of frame
        high_sat_mask = np.zeros_like(depth, dtype=bool)
        high_sat_candidates = depth > 0.999

        if high_sat_candidates.sum() > 0.05 * depth.size:
            # Sky is typically in upper portion of frame
            upper_half = high_sat_candidates[: depth.shape[0] // 2, :]
            if upper_half.sum() > 0.1 * upper_half.size:  # >10% of upper half
                high_sat_mask = high_sat_candidates

        valid_mask &= ~high_sat_mask

        # Optional: Exclude saturated sky (depth > threshold AND brightness > threshold)
        # This is additional to the spatial heuristics above
        sky_brightness_mask = np.zeros_like(depth, dtype=bool)
        if image is not None and self.config.exclude_invalid_depth:
            # Compute brightness (mean of RGB channels)
            brightness = image.mean(axis=2) if image.ndim == 3 else image

            # Sky heuristic: high depth AND high brightness
            sky_brightness_mask = (depth > self.config.sky_saturation_threshold) & (
                brightness > self.config.sky_brightness_threshold
            )
            valid_mask &= ~sky_brightness_mask

        # LOG EXCLUSION STATS (critical for debugging)
        nan_pct = nan_mask.sum() / depth.size * 100
        out_of_range_pct = out_of_range_mask.sum() / depth.size * 100
        low_sat_pct = low_sat_mask.sum() / depth.size * 100
        high_sat_pct = high_sat_mask.sum() / depth.size * 100
        sky_brightness_pct = sky_brightness_mask.sum() / depth.size * 100
        valid_pct = valid_mask.sum() / depth.size * 100

        logger.info(
            f"Valid depth exclusion: NaN/Inf={nan_pct:.1f}%, out_of_range={out_of_range_pct:.1f}%, "
            f"low_sat={low_sat_pct:.1f}%, high_sat={high_sat_pct:.1f}%, "
            f"sky_brightness={sky_brightness_pct:.1f}%, VALID={valid_pct:.1f}%"
        )

        # Warn if valid coverage is too low
        if valid_pct < 85.0:
            logger.warning(
                f"Valid depth coverage {valid_pct:.1f}% is lower than expected (typically >85%). "
                f"Check for depth map artifacts or sky-heavy images."
            )

        return valid_mask

    def _validate_depth_convention(self, depth: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        """Validate depth encoding convention and return stats.

        Args:
            depth: (H, W) depth map
            valid_mask: (H, W) boolean mask of valid pixels

        Returns:
            dict with direction, min, max, p50, valid_coverage_pct, detected, override

        Raises:
            ValueError: If depth convention is inverted (far→near) and auto mode
        """
        valid_depth = depth[valid_mask]

        if valid_depth.size == 0:
            logger.error("No valid depth pixels found")
            return {
                "final": "INVALID_NO_DATA",
                "detected": "INVALID_NO_DATA",
                "override": False,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "p50_depth": 0.0,
                "valid_coverage_pct": 0.0,
            }

        min_depth = float(valid_depth.min())
        max_depth = float(valid_depth.max())
        p50_depth = float(np.percentile(valid_depth, 50))
        valid_coverage_pct = float(valid_mask.sum() / valid_mask.size * 100)

        # Detect direction
        # For near→far encoding: p10 should be < p90
        # For far→near encoding: p10 would be > p90
        p10 = np.percentile(valid_depth, 10)
        p90 = np.percentile(valid_depth, 90)

        # Check for uniform or inverted depth
        # Use a small epsilon to account for numerical precision
        depth_range = p90 - p10
        if depth_range < 1e-6:
            # Nearly uniform depth - not inverted, just no variation
            logger.warning(
                f"Depth is nearly uniform (p10={p10:.3f}, p90={p90:.3f}, range={depth_range:.6f}). "
                f"Zone-based processing may not be meaningful."
            )
            detected_direction = "near_to_far_increasing"  # Assume correct convention
        elif p10 >= p90:
            # Truly inverted depth
            detected_direction = "near_to_far_decreasing"
            logger.warning(
                f"Detected INVERTED depth convention (p10={p10:.3f} >= p90={p90:.3f}). Expected near→far increasing."
            )
        else:
            detected_direction = "near_to_far_increasing"

        # Override if manual convention specified
        if self.config.depth_convention != "auto":
            final_direction = self.config.depth_convention
            if final_direction != detected_direction:
                logger.warning(
                    f"MANUAL OVERRIDE: depth_convention={final_direction} "
                    f"(auto-detected: {detected_direction}). Recording in zone_stats.json"
                )
        else:
            final_direction = detected_direction
            logger.info(f"Auto-detected depth direction: {final_direction}")

        # Raise error if auto mode detects inverted depth
        if final_direction == "near_to_far_decreasing" and self.config.depth_convention == "auto":
            logger.error(
                f"Depth convention INVERTED (p10={p10:.3f} >= p90={p90:.3f}). "
                f"Expected near→far increasing. Aborting. Use depth_convention override if intentional."
            )
            raise ValueError("Depth encoding mismatch: inverted depth convention (far→near)")

        return {
            "direction": final_direction,  # Main key for backward compat
            "final": final_direction,
            "detected": detected_direction,
            "override": (self.config.depth_convention != "auto"),
            "min_depth": min_depth,
            "max_depth": max_depth,
            "p50_depth": p50_depth,
            "p10": float(p10),
            "p90": float(p90),
            "valid_coverage_pct": valid_coverage_pct,
        }

    def _generate_uniform_zones(self, shape: Tuple[int, int]) -> np.ndarray:
        """Fallback: uniform zones when valid coverage is low.

        Args:
            shape: (H, W) output shape

        Returns:
            zones: (H, W, 4) with equal weights 0.25
        """
        H, W = shape
        zones = np.full((H, W, 4), 0.25, dtype=np.float32)
        return zones

    def generate_exaggerated_foreground_zones(self, depth: np.ndarray, foreground_weight: float = 0.90) -> np.ndarray:
        """
        Generate exaggerated foreground zones for canary testing.

        Forces most pixels to Z1 (foreground) to test if zones affect V2 output.

        Args:
            depth: (H, W) depth map
            foreground_weight: Weight for Z1 (default 0.90 = 90%)

        Returns:
            zones: (H, W, 4) with exaggerated Z1 weight
        """
        h, w = depth.shape
        zones = np.zeros((h, w, 4), dtype=np.float32)

        # Force 90% Z1, distribute remaining 10% across Z2/Z3/Z4
        zones[:, :, 0] = foreground_weight  # Z1 = 90%
        remaining_weight = (1.0 - foreground_weight) / 3
        zones[:, :, 1] = remaining_weight  # Z2 = 3.3%
        zones[:, :, 2] = remaining_weight  # Z3 = 3.3%
        zones[:, :, 3] = remaining_weight  # Z4 = 3.3%

        return zones

    def _generate_zone_masks(self, depth: np.ndarray, p_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Generate hard zone masks from percentile thresholds.

        Args:
            depth: (H, W) depth map
            p_values: (3,) percentile values [P10, P35, P65]
            valid_mask: (H, W) boolean mask of valid pixels

        Returns:
            zone_masks: (H, W, 4) binary masks for each zone
        """
        H, W = depth.shape
        zone_masks = np.zeros((H, W, 4), dtype=np.float32)

        # Z1: depth <= P10
        zone_masks[:, :, 0] = (depth <= p_values[0]) & valid_mask

        # Z2: P10 < depth <= P35
        zone_masks[:, :, 1] = (depth > p_values[0]) & (depth <= p_values[1]) & valid_mask

        # Z3: P35 < depth <= P65
        zone_masks[:, :, 2] = (depth > p_values[1]) & (depth <= p_values[2]) & valid_mask

        # Z4: depth > P65
        zone_masks[:, :, 3] = (depth > p_values[2]) & valid_mask

        return zone_masks

    def _apply_morphology(self, zone_masks: np.ndarray) -> np.ndarray:
        """Clean up zone boundaries with morphology.

        Args:
            zone_masks: (H, W, 4) zone masks

        Returns:
            cleaned_masks: (H, W, 4) morphologically cleaned masks
        """
        cleaned_masks = np.zeros_like(zone_masks)

        for i in range(4):
            # Binary opening (erosion + dilation) to remove noise
            mask = zone_masks[:, :, i]
            for _ in range(self.config.morphology_iterations):
                mask = ndimage.binary_opening(mask, iterations=1)
            cleaned_masks[:, :, i] = mask.astype(np.float32)

        return cleaned_masks

    def _blend_zones(self, zone_masks: np.ndarray) -> np.ndarray:
        """Apply Gaussian blending for smooth transitions.

        Args:
            zone_masks: (H, W, 4) zone masks

        Returns:
            blended_zones: (H, W, 4) with soft transitions, normalized to sum=1
        """
        if self.config.blend_sigma <= 0:
            # No blending, just normalize
            return self._normalize_zones(zone_masks)

        blended = np.zeros_like(zone_masks)

        for i in range(4):
            # Gaussian blur for soft transitions
            blended[:, :, i] = gaussian_filter(zone_masks[:, :, i], sigma=self.config.blend_sigma, mode="reflect")

        # Normalize to sum to 1.0 per pixel
        return self._normalize_zones(blended)

    def _normalize_zones(self, zones: np.ndarray) -> np.ndarray:
        """Normalize zone masks to sum to 1.0 per pixel.

        Args:
            zones: (H, W, 4) zone weights

        Returns:
            normalized: (H, W, 4) with sum=1.0 per pixel
        """
        zone_sum = zones.sum(axis=2, keepdims=True)
        # Avoid division by zero - if all zones are zero, distribute uniformly
        zero_mask = zone_sum < 1e-6
        zone_sum = np.where(zero_mask, 1.0, zone_sum)

        normalized = zones / zone_sum

        # For pixels where all zones were zero, set to uniform distribution
        if zero_mask.any():
            normalized[zero_mask.squeeze(-1), :] = 0.25

        return normalized

    def _apply_sky_heuristic(self, zones: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Reduce Z4 weight in bright regions (conservative sky detection).

        Uses EXPLICIT MASS TRANSFER to guarantee reduction survives normalization.

        Algorithm:
        1. Detect sky candidates (Z4-heavy + bright + upper frame)
        2. Scale Z4 to target weight (e.g., 0.3)
        3. Transfer REMOVED MASS to Z3 (mid-background)
        4. Renormalize (guarantees sum=1.0)

        CONSERVATIVE: Only reduces weight, never zeros out. Requires explicit config.

        Args:
            zones: (H, W, 4) zone weights
            image: (H, W, 3) RGB image

        Returns:
            adjusted_zones: (H, W, 4) with reduced Z4 in sky regions (or unchanged)
        """
        if not self.config.apply_sky_heuristic:
            return zones  # No-op if disabled

        # Convert to grayscale luminance (ITU-R BT.709)
        if image.ndim == 3:
            luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        else:
            luminance = image

        # Sky candidates: Z4 regions with high brightness
        z4_weight = zones[:, :, 3]
        sky_candidates = (luminance > self.config.sky_brightness_threshold) & (z4_weight > 0.5)

        # Only apply in upper half of frame (sky is rarely bottom-heavy)
        upper_half_mask = np.zeros_like(sky_candidates)
        upper_half_mask[: zones.shape[0] // 2, :] = True
        sky_mask = sky_candidates & upper_half_mask

        # EXPLICIT MASS TRANSFER (prevents normalization cancellation)
        zones_adjusted = zones.copy()

        if sky_mask.any():
            w4_before = zones_adjusted[sky_mask, 3].copy()
            target_weight = 0.3  # Target Z4 weight in sky
            w4_new = w4_before * target_weight  # Scale to target
            removed_mass = w4_before - w4_new  # Mass to transfer

            zones_adjusted[sky_mask, 3] = w4_new  # Reduce Z4
            zones_adjusted[sky_mask, 2] += removed_mass  # Transfer to Z3 (mid-background)

            # Renormalize (now guaranteed to preserve reduction)
            zones_adjusted = self._normalize_zones(zones_adjusted)

            # Verify reduction occurred (assertion for debugging)
            z4_mean_before = w4_before.mean()
            z4_mean_after = zones_adjusted[sky_mask, 3].mean()
            assert z4_mean_after < z4_mean_before, (
                f"Sky heuristic failed: Z4 not reduced ({z4_mean_before:.3f} → {z4_mean_after:.3f})"
            )

            # Log sky coverage and effect
            sky_pct = sky_mask.sum() / sky_mask.size * 100
            z4_reduction = (1 - z4_mean_after / z4_mean_before) * 100
            logger.info(
                f"Sky heuristic: {sky_pct:.1f}% pixels adjusted, Z4 reduced by {z4_reduction:.1f}% (transferred to Z3)"
            )

        # Store sky mask for diagnostic overlay (save to instance for preview generation)
        self._sky_mask = sky_mask

        return zones_adjusted

    def _compute_zone_stats(
        self,
        depth: np.ndarray,
        zones: np.ndarray,
        valid_mask: np.ndarray,
        thresholds: Dict[str, float],
        depth_convention: Dict[str, Any],
    ) -> dict:
        """Compute coverage %, histograms, percentile drift.

        Args:
            depth: (H, W) depth map
            zones: (H, W, 4) zone weights
            valid_mask: (H, W) boolean mask
            thresholds: Dict of percentile thresholds
            depth_convention: Dict with depth direction and stats

        Returns:
            stats: dict with zone coverage, histograms, drift metrics
        """
        H, W = depth.shape
        total_pixels = H * W

        # Compute coverage percentages
        coverage = {}
        for i, zone_id in enumerate(["Z1", "Z2", "Z3", "Z4"]):
            coverage[zone_id] = float(zones[:, :, i].sum() / total_pixels * 100)

        # Compute per-zone depth statistics
        zone_stats = {}
        for i, zone_id in enumerate(["Z1", "Z2", "Z3", "Z4"]):
            zone_mask = zones[:, :, i] > 0.1  # Pixels with significant zone weight
            if zone_mask.sum() > 0:
                zone_depth = depth[zone_mask]
                zone_stats[zone_id] = {
                    "mean_depth": float(zone_depth.mean()),
                    "std_depth": float(zone_depth.std()),
                    "area_percent": coverage[zone_id],
                }
            else:
                zone_stats[zone_id] = {
                    "mean_depth": 0.0,
                    "std_depth": 0.0,
                    "area_percent": 0.0,
                }

        # Compute percentile drift (morphology + blending effects)
        valid_depth = depth[valid_mask]
        if valid_depth.size > 0:
            p05_original = float(np.percentile(valid_depth, 5))
            p50_original = float(np.percentile(valid_depth, 50))
            p95_original = float(np.percentile(valid_depth, 95))

            # After zoning, compute effective depth distribution
            # (weighted by zone masks)
            # For simplicity, just report original percentiles
            percentile_drift = {
                "p05": 0.0,  # Placeholder
                "p50": 0.0,
                "p95": 0.0,
            }
        else:
            percentile_drift = {"p05": 0.0, "p50": 0.0, "p95": 0.0}

        return {
            "coverage": coverage,
            "zone_stats": zone_stats,
            "thresholds": thresholds,
            "depth_convention": depth_convention,
            "percentile_drift": percentile_drift,
        }

    def save_preview(self, zones: np.ndarray, output_path: Path) -> None:
        """Save color-coded zone preview with optional sky mask overlay.

        Zone colors:
        - Z1 (foreground): Red
        - Z2 (mid-foreground): Green
        - Z3 (mid-background): Blue
        - Z4 (far field): Yellow
        - Sky mask (if applied): Purple tint overlay

        Args:
            zones: (H, W, 4) zone weights
            output_path: Output path for RGB preview PNG
        """
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        H, W = zones.shape[:2]

        # Color mapping: Z1=red, Z2=green, Z3=blue, Z4=yellow
        colors = np.array(
            [
                [255, 0, 0],  # Z1: red
                [0, 255, 0],  # Z2: green
                [0, 0, 255],  # Z3: blue
                [255, 255, 0],  # Z4: yellow
            ],
            dtype=np.uint8,
        )

        # Blend colors by zone weights
        preview = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(4):
            preview += zones[:, :, i : i + 1] * colors[i]

        preview = np.clip(preview, 0, 255).astype(np.uint8)

        # Sky mask overlay (purple tint) if sky heuristic was applied
        if hasattr(self, "_sky_mask") and self._sky_mask is not None and self.config.apply_sky_heuristic:
            if self._sky_mask.any():
                # Purple tint: blend with [128, 0, 128]
                sky_overlay = np.zeros_like(preview)
                sky_overlay[self._sky_mask] = [128, 0, 128]  # Purple
                preview = (preview.astype(np.float32) * 0.7 + sky_overlay * 0.3).astype(np.uint8)
                logger.info(f"Added purple overlay for {self._sky_mask.sum()} sky-adjusted pixels")

        # Save as PNG
        Image.fromarray(preview).save(output_path)
        logger.info(f"Saved zone preview to {output_path}")

    def save_zone_masks(self, zones: np.ndarray, output_dir: Path, stem: str) -> None:
        """Save individual zone masks as 16-bit PNGs.

        Args:
            zones: (H, W, 4) zone weights
            output_dir: Output directory
            stem: File stem for naming
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, zone_id in enumerate(["Z1", "Z2", "Z3", "Z4"]):
            # Convert to 16-bit (0-65535)
            mask_u16 = (zones[:, :, i] * 65535).astype(np.uint16)

            # Save as 16-bit grayscale PNG
            output_path = output_dir / f"{stem}_zones_{zone_id}.png"
            Image.fromarray(mask_u16, mode="I;16").save(output_path)
            logger.debug(f"Saved {zone_id} mask to {output_path}")

    def save_stats_json(self, stats: dict, output_path: Path) -> None:
        """Save zone statistics to JSON.

        Args:
            stats: Statistics dictionary
            output_path: Output path for JSON file
        """
        with output_path.open("w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved zone stats to {output_path}")


def apply_zone_operators(
    image: np.ndarray,
    zones: np.ndarray,
    config: DepthZoneConfig,
) -> np.ndarray:
    """Apply depth-conditioned photometric operators.

    Phase 4 - Manual Implementation Required:
    - Micro-contrast per zone
    - Clarity/sharpening (zone-weighted)
    - Exposure offsets (preserve energy)
    - Tone curves (per-zone falloff)

    DO NOT implement this with Copilot. Requires manual photometric tuning.

    Args:
        image: (H, W, 3) RGB image [0, 1] float32
        zones: (H, W, 4) zone weights
        config: DepthZoneConfig with operator parameters

    Returns:
        Enhanced image (H, W, 3) [0, 1] float32

    Raises:
        NotImplementedError: Always (Phase 4 not implemented)
    """
    if not config.enable_local_operators:
        return image  # No-op

    raise NotImplementedError(
        "Phase 4: apply_zone_operators requires manual photometric tuning. "
        "Copilot should NOT implement artistic/color science logic. "
        "See DEPTH_ZONING_ARCHITECTURE.md Section 6 for design requirements.\n\n"
        "Implementation checklist:\n"
        "  1. Micro-contrast boost (unsharp mask, per-zone strength)\n"
        "  2. Clarity enhancement (high-pass filter, per-zone radius)\n"
        "  3. Sharpening (frequency-selective, per-zone weight)\n"
        "  4. Exposure offset (LAB L-channel, energy-preserving)\n"
    )
