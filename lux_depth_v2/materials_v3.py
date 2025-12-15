"""Materials V3 Engine - Advanced Material Understanding & EfficientSAM Integration.

Materials V3 builds on Materials V2 with:
- Real confidence semantics (not placeholder masks)
- Smarter EfficientSAM prompt generation (from SegFormer peaks, not box centers)
- Expanded taxonomy (semantic + material layers)
- Edge-aware response gating
- Scene-aware parameterization (optional lighting integration)

Stage: Scaffolding (disabled by default)
Status: Development
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .logging_utils import setup_logging
from .backends.prompt_generation import PromptGenerationConfig  # Use PR-2 tested config


log = setup_logging(__name__)


@dataclass
class WaterCandidateReport:
    """Water detection telemetry (PR-W0: always present when Materials V3 enabled).
    
    Provides observability into water detection regardless of source:
    - SegFormer-detected water (source="segformer")
    - Heuristic-detected water (source="heuristic")
    - No water detected (source="none")
    
    PR-W2 adds optional mask field for debugging/visualization.
    PR-W3 adds edge refinement tracking.
    PR-W4 adds two-stage gating telemetry.
    """
    present: bool  # Water detected and passed thresholds
    coverage: float  # 0.0-1.0 (fraction of image)
    coverage_px: int  # Absolute pixel count
    confidence: float  # 0.0-1.0 (detection confidence)
    source: str  # segformer|heuristic|efficientsam_refined|none
    reason: str  # Explanation (e.g., "heuristic_confidence_0.750", "water_detection_disabled")
    mask: Optional[np.ndarray] = None  # PR-W2: Optional mask for debugging
    # PR-W3: Edge refinement tracking
    edge_refined: bool = False  # True if edge refinement was applied
    edge_refinement_boundary_px: int = 0  # Boundary pixel count (for BF1 gating)
    edge_refinement_applied: bool = False  # True if refinement successful
    # PR-W4: Two-stage gating telemetry
    confidence_raw: float = 0.0  # Pre-suppressor confidence
    confidence_after_suppressors: float = 0.0  # Post-suppressor, pre-boost
    confidence_final: float = 0.0  # Post-boost, pre-injection gate
    saturation_boost_applied: bool = False  # True if saturation boost was applied
    candidate_stage_passed: bool = False  # Stage A: Candidate detection
    injection_stage_passed: bool = False  # Stage B: Injection decision
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (excludes mask field)."""
        return {
            "present": self.present,
            "coverage": float(self.coverage),
            "coverage_px": int(self.coverage_px),
            "confidence": float(self.confidence),
            "source": self.source,
            "reason": self.reason,
            "edge_refined": self.edge_refined,
            "edge_refinement_boundary_px": int(self.edge_refinement_boundary_px),
            "edge_refinement_applied": self.edge_refinement_applied,
            # PR-W4: Two-stage gating telemetry
            "confidence_raw": float(self.confidence_raw),
            "confidence_after_suppressors": float(self.confidence_after_suppressors),
            "confidence_final": float(self.confidence_final),
            "saturation_boost_applied": self.saturation_boost_applied,
            "candidate_stage_passed": self.candidate_stage_passed,
            "injection_stage_passed": self.injection_stage_passed,
        }


class MaterialTaxonomy(str, Enum):
    """Material taxonomy depth for segmentation."""
    
    BASE = "base"  # SegFormer buckets only (8-12 classes)
    EXPANDED = "expanded"  # Semantic + material layers (18-24 classes)
    FULL = "full"  # Future: full PBR taxonomy (40+ classes)


class RefinementStrategy(str, Enum):
    """EfficientSAM refinement strategy."""
    
    OFF = "off"  # No refinement
    CANARY = "canary"  # Glass/water/foliage only (Stage 6 validated)
    SELECTIVE = "selective"  # Auto-select based on confidence
    AGGRESSIVE = "aggressive"  # All materials (development only)


@dataclass
class ConfidenceSemantics:
    """Real confidence semantics for Materials V3.
    
    Unlike Materials V2 (placeholder), V3 distinguishes:
    - Base confidence (SegFormer raw output)
    - Refined confidence (post-EfficientSAM + fusion)
    - Edge confidence (boundary-specific)
    - Final confidence (gated + blended)
    """
    
    # Global thresholds
    base_threshold: float = 0.50  # SegFormer mask threshold
    refined_threshold: float = 0.45  # EfficientSAM can be slightly lower
    edge_threshold: float = 0.30  # Edge band lower threshold
    
    # Per-material overrides
    material_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'wood': 0.65,
        'metal': 0.60,
        'glass': 0.40,  # Inherently low confidence
        'water': 0.35,  # Highly variable
        'stone': 0.65,
        'fabric': 0.55,
        'foliage': 0.45,
        'polished': 0.45,
    })
    
    # Edge-aware gating
    use_edge_confidence: bool = True
    edge_band_width: float = 0.20  # Fraction of mask perimeter
    
    # Quality gates
    min_coverage: float = 0.01  # Minimum mask coverage (% of image)
    max_coverage: float = 0.95  # Maximum mask coverage (sanity check)
    
    def get_threshold(self, material_type: str, is_edge: bool = False) -> float:
        """Get threshold for specific material type and region."""
        base = self.material_thresholds.get(material_type, self.base_threshold)
        if is_edge and self.use_edge_confidence:
            return max(self.edge_threshold, base * 0.7)  # Lower at edges
        return base


@dataclass
class EdgeAwareGating:
    """Edge-aware response gating (core vs edge differentiation)."""
    
    enabled: bool = True
    
    # Core vs edge thresholds (reuse fusion config concept)
    core_threshold: float = 0.70
    edge_low: float = 0.20
    edge_high: float = 0.70
    
    # Response strength multipliers
    core_strength: float = 1.0
    edge_strength: float = 0.8  # Slightly conservative at edges
    
    # Edge detection method
    edge_method: str = "confidence_gradient"  # confidence_gradient | sobel | both


@dataclass
class ExpandedTaxonomyConfig:
    """Expanded material taxonomy (semantic + material layers)."""
    
    enabled: bool = False  # Feature gate
    
    # Semantic layer (SegFormer→buckets)
    semantic_classes: List[str] = field(default_factory=lambda: [
        'sky', 'building', 'wall', 'floor', 'ceiling',
        'window', 'door', 'furniture', 'vegetation',
        'water', 'ground', 'road', 'sidewalk'
    ])
    
    # Material layer (what matters for response)
    material_classes: List[str] = field(default_factory=lambda: [
        'wood_grain', 'wood_smooth',
        'stone_paver', 'stone_wall', 'stone_counter',
        'metal_brushed', 'metal_polished',
        'glass_clear', 'glass_frosted',
        'water_surface', 'water_volume',
        'fabric_matte', 'fabric_glossy',
        'stucco', 'painted_plaster',
        'ceramic_tile', 'ceramic_glazed',
    ])
    
    # Mapping: semantic→material (simple heuristics for now)
    semantic_to_material_map: Dict[str, List[str]] = field(default_factory=lambda: {
        'window': ['glass_clear', 'glass_frosted', 'metal_brushed'],
        'floor': ['wood_grain', 'stone_paver', 'ceramic_tile'],
        'wall': ['stucco', 'painted_plaster', 'wood_smooth'],
        'water': ['water_surface', 'water_volume'],
        # etc.
    })


@dataclass
class MaterialsV3Config:
    """Complete Materials V3 configuration.
    
    Disabled by default; opt-in via CLI --materials-v3.
    """
    
    enabled: bool = False  # Master feature gate
    
    # Taxonomy
    taxonomy: MaterialTaxonomy = MaterialTaxonomy.BASE
    expanded: ExpandedTaxonomyConfig = field(default_factory=ExpandedTaxonomyConfig)
    
    # Confidence & gating
    confidence: ConfidenceSemantics = field(default_factory=ConfidenceSemantics)
    edge_gating: EdgeAwareGating = field(default_factory=EdgeAwareGating)
    
    # EfficientSAM integration
    refine_edges: RefinementStrategy = RefinementStrategy.OFF
    prompt_gen: PromptGenerationConfig = field(default_factory=PromptGenerationConfig)
    
    # Safety guards (OOM prevention from Stage 6)
    max_megapixels: float = 30.0  # Hard stop for EfficientSAM
    max_dimension: int = 6000
    
    # Caching
    cache_dir: Optional[Path] = None
    cache_enabled: bool = False
    
    # Backend compatibility
    backend: str = 'segformer'  # Inherit from SegmentationConfig
    
    # Lighting-aware tuning (optional, deferred until lighting validated)
    lighting_aware: bool = False
    
    # PR-4B: Pixel operations (canary-only)
    apply_pixel_ops: bool = False  # Master gate for pixel modifications
    glass_response_enabled: bool = False  # Specific toggle for glass
    
    # PR-4B validation-only override:
    # When True, bypass response_plan.should_refine for glass so we can validate
    # pixel ops behavior at least once. Must remain False in production presets.
    force_glass_pixel_ops: bool = False
    
    # PR-4D: Stone pixel operations
    stone_response_enabled: bool = False  # Specific toggle for stone
    force_stone_pixel_ops: bool = False  # Validation-only override (dev-only)
    
    # PR-W0/W2: Water detection (opt-in only)
    water_detection_enabled: bool = False  # Master gate for water candidate detection
    
    # PR-W4: Two-stage gating (decouples candidate detection from injection decision)
    water_candidate_threshold: float = 0.25  # Stage A: Candidate detection (high recall)
    water_candidate_confidence_threshold: float = 0.4  # Stage B: Injection decision (precision)
    water_min_coverage: float = 0.05  # Minimum coverage (5% of image) to inject
    
    # PR-W4: Saturation boost for low-saturation pools (controlled experiment)
    water_saturation_boost_enabled: bool = True  # Enable saturation-based confidence boost
    water_saturation_boost_amount: float = 0.15  # Boost for low-saturation candidates
    water_saturation_boost_threshold: float = 0.20  # Saturation threshold for boost
    
    # PR-W3: Water edge refinement (opt-in, after candidate exists)
    water_edge_refinement_enabled: bool = False  # Master gate for edge refinement
    water_edge_refinement_min_confidence: float = 0.5  # Only refine high-confidence candidates
    water_edge_refinement_min_boundary_px: int = 100  # Avoid BF1 failures on tiny boundaries


class MaterialsV3Engine:
    """Materials V3 processing engine (scaffolding only).
    
    Stage: Scaffolding
    TODO:
    - Implement __init__ with config validation
    - Implement process() entrypoint
    - Implement _generate_prompts_from_mask()
    - Implement _apply_edge_aware_response()
    - Wire into LuxPipelineV2
    """
    
    def __init__(self, config: MaterialsV3Config):
        """Initialize Materials V3 engine.
        
        Args:
            config: Materials V3 configuration
            
        Raises:
            NotImplementedError: Scaffolding only
        """
        self.config = config
        
        # PR-W2: Initialize water detector (lazy, only when enabled)
        self.water_detector = None
        if config.water_detection_enabled:
            try:
                from .water_candidate import WaterCandidateDetector, SCIPY_AVAILABLE, SKIMAGE_AVAILABLE
                if not (SCIPY_AVAILABLE and SKIMAGE_AVAILABLE):
                    log.warning(
                        "Water detection enabled but scipy/scikit-image not available. "
                        "Water detection will use fallback behavior (reduced quality)."
                    )
                self.water_detector = WaterCandidateDetector()
                log.info("Water candidate detection enabled")
            except ImportError as e:
                log.warning(f"Water detection enabled but dependencies missing: {e}")
                self.water_detector = None
        
        if config.enabled:
            log.info("Materials V3 enabled (experimental)")
            log.info(f"  Taxonomy: {config.taxonomy}")
            log.info(f"  Refinement: {config.refine_edges}")
            log.info(f"  Max MP: {config.max_megapixels}")
        
        # TODO: Initialize backends, caches, etc.
    
    def _audit_class_presence(
        self,
        raw_materials: dict,
        canonical_materials: dict,
        requested_targets: Optional[List[str]] = None,
    ) -> dict:
        """Audit class presence for debugging taxonomy/coverage issues.
        
        This addresses Stage 6 "water missing" problems by reporting:
        - What classes the segmenter actually emitted
        - How they mapped to canonical keys
        - Which requested targets are missing and why
        
        Args:
            raw_materials: Original output from segmenter
            canonical_materials: After taxonomy normalization
            requested_targets: Optional list of expected classes (glass, water, foliage)
            
        Returns:
            Audit report dict with diagnostics
        """
        if requested_targets is None:
            requested_targets = ["glass", "water", "foliage"]
        
        from .materials_v3_taxonomy import normalize_material_name
        
        audit = {
            "emitted_classes": sorted(raw_materials.keys()),
            "emitted_count": len(raw_materials),
            "canonical_classes": sorted(canonical_materials.keys()),
            "canonical_count": len(canonical_materials),
            "requested_targets": requested_targets,
            "target_status": {},
        }
        
        # Check each requested target
        for target in requested_targets:
            canonical_target = normalize_material_name(target)
            
            if canonical_target in canonical_materials:
                mask = canonical_materials[canonical_target]
                if isinstance(mask, np.ndarray):
                    if mask.dtype == bool:
                        coverage_px = int(mask.sum())
                    else:
                        coverage_px = int((mask > 0.5).sum())
                else:
                    coverage_px = 0
                
                status = {
                    "present": True,
                    "canonical_name": canonical_target,
                    "coverage_pixels": coverage_px,
                    "reason": "found" if coverage_px > 0 else "zero_coverage",
                }
            else:
                # Try to find which emitted classes might have mapped
                possible_sources = [
                    k for k in raw_materials.keys()
                    if normalize_material_name(k) == canonical_target
                ]
                
                status = {
                    "present": False,
                    "canonical_name": canonical_target,
                    "coverage_pixels": 0,
                    "reason": "not_emitted_by_segmenter",
                    "possible_raw_names": possible_sources,
                }
            
            audit["target_status"][target] = status
        
        # Identify unmapped classes (emitted but not in canonical)
        unmapped = [k for k in raw_materials.keys() if normalize_material_name(k) not in canonical_materials]
        if unmapped:
            audit["unmapped_classes"] = sorted(unmapped)
            audit["warning"] = f"{len(unmapped)} emitted classes did not map to canonical names"
        
        return audit
    
    def _build_water_audit(
        self,
        raw_materials: dict,
        canonical_materials: dict,
        water_candidate: WaterCandidateReport,
        h: int,
        w: int,
    ) -> dict:
        """Build water-specific audit for class presence audit.
        
        PR-W0/W2: Water audit includes both SegFormer and candidate detection metrics.
        
        Args:
            raw_materials: Original output from segmenter
            canonical_materials: After taxonomy normalization
            water_candidate: Water candidate detection report
            h, w: Image dimensions for coverage calculation
            
        Returns:
            Water audit dict with raw_present, raw_coverage, candidate_* fields
        """
        from .materials_v3_taxonomy import normalize_material_name
        
        total_pixels = h * w
        
        # Check if SegFormer emitted water
        raw_water_present = "water" in raw_materials
        canonical_water_present = "water" in canonical_materials
        
        # Compute raw coverage from SegFormer
        raw_coverage = 0.0
        if raw_water_present:
            water_mask = raw_materials["water"]
            if isinstance(water_mask, np.ndarray):
                if water_mask.dtype == bool:
                    raw_coverage = float(water_mask.sum() / total_pixels)
                else:
                    raw_coverage = float((water_mask > 0.5).sum() / total_pixels)
        elif canonical_water_present:
            # May have been mapped from different name
            water_mask = canonical_materials["water"]
            if isinstance(water_mask, np.ndarray):
                if water_mask.dtype == bool:
                    raw_coverage = float(water_mask.sum() / total_pixels)
                else:
                    raw_coverage = float((water_mask > 0.5).sum() / total_pixels)
        
        return {
            "raw_present": raw_water_present or canonical_water_present,
            "raw_coverage": raw_coverage,
            "candidate_present": water_candidate.present,
            "candidate_coverage": water_candidate.coverage,
            "candidate_source": water_candidate.source,
        }
    
    def _infer_scene_context(
        self,
        canonical_materials: dict,
    ):
        """Infer pool vs ocean vs unknown from materials.
        
        PR-W2: Simple heuristic scene context for water detection tuning.
        
        Args:
            canonical_materials: Canonical material dict
            
        Returns:
            SceneContext enum value
        """
        from .water_candidate import SceneContext
        
        # Simple heuristic: if building/architecture present -> pool
        # If large sky/horizon -> ocean, otherwise unknown
        if "building" in canonical_materials or "wall" in canonical_materials:
            return SceneContext.POOL
        elif "sky" in canonical_materials:
            sky_mask = canonical_materials["sky"]
            if isinstance(sky_mask, np.ndarray):
                if sky_mask.dtype == bool:
                    sky_coverage = float(sky_mask.sum() / sky_mask.size)
                else:
                    sky_coverage = float((sky_mask > 0.5).sum() / sky_mask.size)
                
                if sky_coverage > 0.3:
                    return SceneContext.OCEAN
        
        return SceneContext.UNKNOWN
    
    def _rgb_to_hsv_vectorized(self, rgb01: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV (vectorized, no dependencies).
        
        Args:
            rgb01: RGB image (HxWx3 float32 in [0,1])
            
        Returns:
            HSV image (HxWx3 float32 in [0,1])
        """
        # Vectorized RGB to HSV conversion (numpy only, no skimage)
        r, g, b = rgb01[:, :, 0], rgb01[:, :, 1], rgb01[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        delta = maxc - minc
        s = np.where(maxc != 0, delta / maxc, 0)
        
        # Hue calculation
        rc = np.where(delta != 0, (maxc - r) / delta, 0)
        gc = np.where(delta != 0, (maxc - g) / delta, 0)
        bc = np.where(delta != 0, (maxc - b) / delta, 0)
        
        h = np.zeros_like(r)
        h = np.where((maxc == r) & (delta != 0), bc - gc, h)
        h = np.where((maxc == g) & (delta != 0), 2.0 + rc - bc, h)
        h = np.where((maxc == b) & (delta != 0), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0  # Normalize to [0, 1]
        
        return np.stack([h, s, v], axis=2)
    
    def _detect_water_candidate(
        self,
        rgb01: np.ndarray,
        depth01: Optional[np.ndarray],
        canonical_materials: dict,
    ) -> WaterCandidateReport:
        """Run water candidate detection (PR-W2).
        
        Checks if SegFormer already provided water, otherwise runs heuristic detector.
        
        Args:
            rgb01: RGB image (HxWx3 float32 in [0,1])
            depth01: Optional depth map (HxW float32)
            canonical_materials: Canonical materials dict
            
        Returns:
            WaterCandidateReport with detection results
        """
        h, w = rgb01.shape[:2]
        total_pixels = h * w
        
        # If water detection disabled, return disabled report
        if not self.config.water_detection_enabled:
            return WaterCandidateReport(
                present=False,
                coverage=0.0,
                coverage_px=0,
                confidence=0.0,
                source="none",
                reason="water_detection_disabled",
                mask=None,
            )
        
        # Check if SegFormer already provided water with sufficient coverage
        if "water" in canonical_materials:
            water_mask = canonical_materials["water"]
            if isinstance(water_mask, np.ndarray):
                if water_mask.dtype == bool:
                    coverage = float(water_mask.sum() / total_pixels)
                else:
                    coverage = float((water_mask > 0.5).sum() / total_pixels)
                
                coverage_px = int(coverage * total_pixels)
                
                # If SegFormer water has sufficient coverage, use it
                if coverage >= self.config.water_min_coverage:
                    return WaterCandidateReport(
                        present=True,
                        coverage=coverage,
                        coverage_px=coverage_px,
                        confidence=1.0,  # Trust SegFormer when it emits water
                        source="segformer",
                        reason="segformer_emitted_sufficient_coverage",
                        mask=water_mask.astype(np.float32) if water_mask.dtype == bool else water_mask,
                    )
        
        # Run heuristic detector (if available)
        if self.water_detector is None:
            # Detector failed to initialize (missing dependencies)
            return WaterCandidateReport(
                present=False,
                coverage=0.0,
                coverage_px=0,
                confidence=0.0,
                source="none",
                reason="water_detector_unavailable_missing_dependencies",
                mask=None,
            )
        
        scene_context = self._infer_scene_context(canonical_materials)
        result = self.water_detector.detect(rgb01, depth01, scene_context)
        
        # PR-W4: Two-stage gating implementation
        # Stage 1: Extract raw confidence (before suppressors)
        confidence_raw = result.confidence
        if result.suppressor_telemetry:
            # If suppressors were applied, extract original confidence
            confidence_raw = result.suppressor_telemetry.get("original_confidence", result.confidence)
        
        # Stage 2: After suppressors (already applied in detector)
        confidence_after_suppressors = result.confidence
        
        # Stage 3: Saturation boost (controlled experiment for low-saturation pools)
        # PR-W4B/V1: Stricter gating - rescue legitimate pools, block glass negatives
        saturation_boost_applied = False
        confidence_final = confidence_after_suppressors
        
        if self.config.water_saturation_boost_enabled and result.mask is not None:
            # Extract suppressor telemetry to check which suppressors fired
            glass_grid_suppressed = False
            
            if result.suppressor_telemetry:
                suppressors_applied = result.suppressor_telemetry.get("suppressors_applied", [])
                glass_grid_suppressed = "architectural_glass" in suppressors_applied
            
            # Compute average saturation in detected region
            h, w = rgb01.shape[:2]
            hsv = self._rgb_to_hsv_vectorized(rgb01)
            masked_saturation = hsv[:, :, 1] * (result.mask > 0.5)
            avg_saturation = float(np.sum(masked_saturation) / max(np.sum(result.mask > 0.5), 1))
            
            # V1: Apply boost when low saturation detected, UNLESS glass suppressor fired
            # Glass suppressor indicates architectural glass (false positive), not a pool
            # Flat surface suppressor indicates a desaturated pool (legitimate, needs rescue)
            if (avg_saturation < self.config.water_saturation_boost_threshold and
                not glass_grid_suppressed):
                confidence_final = min(1.0, confidence_after_suppressors + self.config.water_saturation_boost_amount)
                saturation_boost_applied = True
        
        # Stage A: Candidate detection (high recall threshold)
        candidate_stage_passed = confidence_raw >= self.config.water_candidate_threshold
        
        # Stage B: Injection decision (precision threshold, after suppressors and boost)
        # CRITICAL: Suppressors should NOT veto if saturation boost can rescue
        # Saturation boost is a controlled experiment for low-saturation pools
        # Only veto if final boosted confidence still fails threshold
        injection_stage_passed = confidence_final >= self.config.water_candidate_confidence_threshold
        
        # Final injection decision
        present = injection_stage_passed and result.coverage >= self.config.water_min_coverage
        
        # PR-W3: Optional edge refinement (only if injection passes)
        refined_mask = None
        edge_refined = False
        edge_refinement_boundary_px = 0
        edge_refinement_applied = False
        
        if present and self.config.water_edge_refinement_enabled:
            if confidence_final >= self.config.water_edge_refinement_min_confidence:
                refined_mask = self._refine_water_edges(
                    rgb01=rgb01,
                    water_candidate_mask=result.mask,
                    water_confidence=confidence_final
                )
                
                if refined_mask is not None:
                    edge_refined = True
                    edge_refinement_applied = True
                    # Update coverage with refined mask
                    h, w = rgb01.shape[:2]
                    refined_coverage = float((refined_mask > 0.5).sum() / (h * w))
                    refined_coverage_px = int((refined_mask > 0.5).sum())
                    
                    # Compute boundary pixels for reporting
                    boundary_mask = self._extract_boundary(result.mask, width=5)
                    edge_refinement_boundary_px = int(np.sum(boundary_mask))
                    
                    return WaterCandidateReport(
                        present=present,
                        coverage=refined_coverage,
                        coverage_px=refined_coverage_px,
                        confidence=confidence_final,  # Report final confidence
                        source="efficientsam_refined",
                        reason=f"two_stage_gating_final_{confidence_final:.3f}_edge_refined",
                        mask=refined_mask,
                        edge_refined=edge_refined,
                        edge_refinement_boundary_px=edge_refinement_boundary_px,
                        edge_refinement_applied=edge_refinement_applied,
                        # PR-W4: Two-stage telemetry
                        confidence_raw=confidence_raw,
                        confidence_after_suppressors=confidence_after_suppressors,
                        confidence_final=confidence_final,
                        saturation_boost_applied=saturation_boost_applied,
                        candidate_stage_passed=candidate_stage_passed,
                        injection_stage_passed=injection_stage_passed,
                    )
        
        # No refinement applied, return heuristic result with two-stage telemetry
        return WaterCandidateReport(
            present=present,
            coverage=result.coverage,
            coverage_px=result.coverage_px,
            confidence=confidence_final,  # Report final confidence
            source="heuristic",
            reason=f"two_stage_gating_final_{confidence_final:.3f}",
            mask=result.mask,
            edge_refined=edge_refined,
            edge_refinement_boundary_px=edge_refinement_boundary_px,
            edge_refinement_applied=edge_refinement_applied,
            # PR-W4: Two-stage telemetry
            confidence_raw=confidence_raw,
            confidence_after_suppressors=confidence_after_suppressors,
            confidence_final=confidence_final,
            saturation_boost_applied=saturation_boost_applied,
            candidate_stage_passed=candidate_stage_passed,
            injection_stage_passed=injection_stage_passed,
        )
    
    def _should_inject_water_candidate(
        self,
        water_candidate: WaterCandidateReport,
    ) -> bool:
        """Decide if candidate should be added to canonical materials (PR-W2).
        
        Args:
            water_candidate: Water candidate detection report
            
        Returns:
            True if candidate passes thresholds and should be injected
        """
        return (
            water_candidate.present and
            water_candidate.source == "heuristic" and  # Only inject heuristics, not SegFormer
            water_candidate.coverage >= self.config.water_min_coverage and
            water_candidate.confidence >= self.config.water_candidate_confidence_threshold
        )
    
    def _build_water_material(
        self,
        water_candidate: WaterCandidateReport,
    ) -> np.ndarray:
        """Convert water candidate to material mask (PR-W2).
        
        Args:
            water_candidate: Water candidate detection report
            
        Returns:
            Water mask as numpy array (HxW float32)
        """
        # Return the mask from candidate (already float32)
        return water_candidate.mask
    
    def _extract_boundary(
        self,
        mask: np.ndarray,
        width: int = 5
    ) -> np.ndarray:
        """Extract boundary region of mask (PR-W3).
        
        Args:
            mask: Binary or float mask (HxW)
            width: Boundary width in pixels
            
        Returns:
            Boundary mask as float32
        """
        from scipy import ndimage
        
        # Convert to binary if needed
        binary_mask = mask > 0.5
        
        # Dilate and erode to get boundary
        dilated = ndimage.binary_dilation(binary_mask, iterations=width)
        eroded = ndimage.binary_erosion(binary_mask, iterations=width)
        boundary = (dilated & ~eroded).astype(np.float32)
        
        return boundary
    
    def _sample_prompts_from_mask(
        self,
        mask: np.ndarray,
        confidence_threshold: float = 0.7,
        num_samples: int = 5
    ) -> List[Tuple[int, int]]:
        """Sample point prompts from high-confidence regions (PR-W3).
        
        Args:
            mask: Float mask (HxW)
            confidence_threshold: Sample only where mask > threshold
            num_samples: Target number of prompts
            
        Returns:
            List of (y, x) coordinates
        """
        # Find high-confidence pixels
        high_conf = mask > confidence_threshold
        y_coords, x_coords = np.where(high_conf)
        
        if len(y_coords) == 0:
            # Fallback: sample from any positive region
            high_conf = mask > 0.5
            y_coords, x_coords = np.where(high_conf)
        
        if len(y_coords) == 0:
            return []
        
        # Sample uniformly
        num_samples = min(num_samples, len(y_coords))
        indices = np.linspace(0, len(y_coords) - 1, num_samples, dtype=int)
        
        prompts = [(int(y_coords[i]), int(x_coords[i])) for i in indices]
        return prompts
    
    def _compute_roi_bbox(
        self,
        mask: np.ndarray,
        padding: int = 50
    ) -> Tuple[int, int, int, int]:
        """Compute ROI bounding box around mask region (PR-W3).
        
        Args:
            mask: Binary or float mask (HxW)
            padding: Padding pixels around bbox
            
        Returns:
            (y0, y1, x0, x1) bounding box
        """
        # Find non-zero regions
        binary_mask = mask > 0.5
        y_coords, x_coords = np.where(binary_mask)
        
        if len(y_coords) == 0:
            # Fallback: full image
            h, w = mask.shape
            return (0, h, 0, w)
        
        # Compute bbox with padding
        y0 = max(0, y_coords.min() - padding)
        y1 = min(mask.shape[0], y_coords.max() + padding)
        x0 = max(0, x_coords.min() - padding)
        x1 = min(mask.shape[1], x_coords.max() + padding)
        
        return (int(y0), int(y1), int(x0), int(x1))
    
    def _crop_to_roi(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop image to ROI (PR-W3).
        
        Args:
            image: Image array (HxW or HxWxC)
            bbox: (y0, y1, x0, x1) bounding box
            
        Returns:
            Cropped image
        """
        y0, y1, x0, x1 = bbox
        return image[y0:y1, x0:x1]
    
    def _uncrop_from_roi(
        self,
        mask_roi: np.ndarray,
        bbox: Tuple[int, int, int, int],
        full_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Map ROI mask back to full resolution (PR-W3).
        
        Args:
            mask_roi: Mask in ROI coordinates
            bbox: (y0, y1, x0, x1) bounding box
            full_shape: (H, W) of full image
            
        Returns:
            Full-resolution mask
        """
        y0, y1, x0, x1 = bbox
        full_mask = np.zeros(full_shape, dtype=mask_roi.dtype)
        full_mask[y0:y1, x0:x1] = mask_roi
        return full_mask
    
    def _refine_water_edges(
        self,
        rgb01: np.ndarray,
        water_candidate_mask: np.ndarray,
        water_confidence: float
    ) -> Optional[np.ndarray]:
        """Refine water edges using EfficientSAM (PR-W3).
        
        Only runs after candidate exists and passes thresholds.
        
        Args:
            rgb01: RGB image (HxWx3 float32 in [0,1])
            water_candidate_mask: Water candidate mask (HxW float32)
            water_confidence: Confidence score
            
        Returns:
            Refined mask if successful, None otherwise
        """
        # Gate 1: Feature disabled
        if not self.config.water_edge_refinement_enabled:
            return None
        
        # Gate 2: Confidence too low
        if water_confidence < self.config.water_edge_refinement_min_confidence:
            log.debug(
                f"PR-W3: Skipping edge refinement (confidence {water_confidence:.3f} "
                f"< threshold {self.config.water_edge_refinement_min_confidence})"
            )
            return None
        
        # Gate 3: Extract boundary and check size (BF1 avoidance)
        boundary_mask = self._extract_boundary(water_candidate_mask, width=5)
        boundary_px = int(np.sum(boundary_mask))
        
        if boundary_px < self.config.water_edge_refinement_min_boundary_px:
            log.debug(
                f"PR-W3: Skipping edge refinement (boundary {boundary_px}px "
                f"< min {self.config.water_edge_refinement_min_boundary_px}px)"
            )
            return None
        
        # Generate prompts from high-confidence regions
        prompts = self._sample_prompts_from_mask(
            water_candidate_mask,
            confidence_threshold=0.7,
            num_samples=5
        )
        
        if len(prompts) == 0:
            log.debug("PR-W3: No high-confidence prompts found, skipping refinement")
            return None
        
        # Compute ROI bbox around candidate region
        roi_bbox = self._compute_roi_bbox(water_candidate_mask, padding=50)
        rgb_roi = self._crop_to_roi(rgb01, roi_bbox)
        
        # Try to run EfficientSAM
        try:
            # Check if EfficientSAM backend is available
            from .backends.efficientsam_backend import (
                EfficientSAMBackend,
                EfficientSAMNotAvailable,
                PointPrompt
            )
            
            # Adjust prompts to ROI coordinates and create PointPrompt objects
            y0, y1, x0, x1 = roi_bbox
            h_roi, w_roi = rgb_roi.shape[:2]
            
            # Convert to normalized coordinates and PointPrompt objects
            roi_prompts = []
            for y, x in prompts:
                # Adjust to ROI coordinates
                y_roi = y - y0
                x_roi = x - x0
                # Normalize to [0, 1]
                y_norm = y_roi / h_roi
                x_norm = x_roi / w_roi
                # Create PointPrompt (label=1 for foreground)
                roi_prompts.append(PointPrompt(x=x_norm, y=y_norm, label=1))
            
            # Initialize backend (lazy load)
            sam_backend = EfficientSAMBackend()
            
            # Convert RGB to uint8 if needed (EfficientSAM expects uint8 or float32)
            rgb_roi_uint8 = (rgb_roi * 255).astype(np.uint8) if rgb_roi.dtype == np.float32 else rgb_roi
            
            # Run segmentation
            sam_mask_roi = sam_backend.segment(
                rgb_roi_uint8,
                prompts=roi_prompts
            )
            
            # Map back to full resolution
            refined_mask = self._uncrop_from_roi(sam_mask_roi, roi_bbox, rgb01.shape[:2])
            
            log.info(
                f"PR-W3: Edge refinement successful "
                f"(boundary={boundary_px}px, prompts={len(prompts)})"
            )
            return refined_mask
            
        except Exception as e:
            # Graceful degradation: log warning and return None
            log.warning(f"PR-W3: Edge refinement failed ({e}), using original mask")
            return None
    
    def process(
        self,
        image: np.ndarray,
        segmentation_result: dict,
        depth_map: Optional[np.ndarray] = None,
    ) -> dict:
        """Process materials with V3 enhancements (Plan Mode).
        
        PR-3A: Implements plan+stats mode (no pixel changes yet).
        Canonicalizes material keys, computes per-class stats, decides refinement.
        
        Args:
            image: RGB image (HxWx3)
            segmentation_result: Output from material_segmentation (V2 or V3)
            depth_map: Optional depth map (HxW)
            
        Returns:
            Segmentation result with V3 metadata attached
        """
        if not self.config.enabled:
            # Pass-through when disabled
            return segmentation_result
        
        # Import taxonomy helpers
        from .materials_v3_taxonomy import (
            normalize_material_name,
            normalize_material_dict,
            get_material_metadata,
            should_refine_material,
        )
        
        # Import response planning (PR-4A)
        from .materials_v3_response import (
            ResponsePlanConfig,
            generate_response_plan,
        )
        
        # Extract masks from segmentation result
        # Segmentation result is typically dict with 'materials' key containing masks
        if 'materials' not in segmentation_result:
            log.warning("No 'materials' key in segmentation_result; V3 pass-through")
            return segmentation_result
        
        raw_materials = segmentation_result['materials']
        
        # PR-3A Step 1: Canonicalize material keys
        canonical_materials = normalize_material_dict(raw_materials)
        
        # PR-W2: Water candidate detection (before per-class stats)
        water_candidate = self._detect_water_candidate(
            rgb01=image,  # Assume image is already [0,1] float32
            depth01=depth_map,
            canonical_materials=canonical_materials,
        )
        
        # PR-W2: Inject water candidate if it passes thresholds
        if self._should_inject_water_candidate(water_candidate):
            water_mask = self._build_water_material(water_candidate)
            canonical_materials["water"] = water_mask
            # Also inject into original materials dict so it's visible to downstream
            segmentation_result['materials']["water"] = water_mask
            log.info(
                f"PR-W2: Injected heuristic water mask "
                f"(confidence={water_candidate.confidence:.3f}, "
                f"coverage={water_candidate.coverage:.3f})"
            )
        
        # NEW: Class presence audit (addresses Stage 6 "water missing" issue)
        class_audit = self._audit_class_presence(
            raw_materials, canonical_materials, requested_targets=["glass", "water", "foliage"]
        )
        
        # PR-3A Step 2: Compute per-class stats
        h, w = image.shape[:2] if image.ndim >= 2 else (1, 1)
        total_pixels = h * w
        
        per_class_stats = {}
        
        for canonical_name, mask in canonical_materials.items():
            metadata = get_material_metadata(canonical_name)
            
            # Compute coverage
            if isinstance(mask, np.ndarray):
                if mask.dtype == bool:
                    coverage = mask.sum() / total_pixels
                    mean_conf = 1.0 if mask.any() else 0.0
                    edge_conf = mean_conf  # simplified for boolean masks
                else:
                    # Float confidence mask
                    coverage = (mask > metadata.confidence_threshold).sum() / total_pixels
                    mean_conf = float(mask.mean())
                    
                    # Edge confidence: compute from boundary band
                    edge_conf = self._compute_edge_confidence(mask, metadata)
            else:
                # Fallback for unexpected types
                coverage = 0.0
                mean_conf = 0.0
                edge_conf = 0.0
            
            # PR-3A Step 3: Decide should_refine per class
            refine_decision = should_refine_material(
                canonical_name,
                refinement_strategy=self.config.refine_edges.value,
            )
            
            per_class_stats[canonical_name] = {
                "coverage": float(coverage),
                "mean_confidence": float(mean_conf),
                "edge_confidence": float(edge_conf),
                "should_refine": refine_decision,
                "refinement_priority": metadata.refinement_priority,
                "threshold": metadata.confidence_threshold,
            }
        
        # PR-3A Step 4: Attach V3 metadata to result (no pixel modifications)
        # PR-W0/W2: Add water audit to class_presence_audit
        water_audit = self._build_water_audit(
            raw_materials=raw_materials,
            canonical_materials=canonical_materials,
            water_candidate=water_candidate,
            h=h,
            w=w,
        )
        class_audit["water"] = water_audit
        
        segmentation_result['materials_v3'] = {
            "enabled": True,
            "taxonomy": self.config.taxonomy.value,
            "refinement_strategy": self.config.refine_edges.value,
            "per_class_stats": per_class_stats,
            "canonical_materials": list(canonical_materials.keys()),
            "class_presence_audit": class_audit,  # NEW: diagnose missing classes
            "water_candidate": water_candidate.to_dict(),  # PR-W0/W2: Water detection report (JSON-safe)
        }
        
        # PR-4A: Generate response plan (PR-4C: pass RGB for edge signals)
        response_plan_config = ResponsePlanConfig()
        response_plan = generate_response_plan(
            canonical_materials=canonical_materials,
            config=response_plan_config,
            strategy=self.config.refine_edges.value,
            intent="client",  # TODO: get from auto-preset context
            quality_tier="max",  # TODO: get from pipeline config
            rgb_image=image,  # PR-4C: for edge signal computation
        )
        
        segmentation_result['materials_v3_response_plan'] = response_plan
        
        # Still return original masks (no pixel changes in PR-3A/PR-4A)
        return segmentation_result
    
    def _compute_edge_confidence(
        self,
        mask: np.ndarray,
        metadata,
    ) -> float:
        """Compute mean confidence in edge band.
        
        Uses edge_gating config to define edge band width.
        
        Args:
            mask: Float confidence mask (HxW)
            metadata: MaterialMetadata with thresholds
            
        Returns:
            Mean confidence in edge band
        """
        from scipy.ndimage import binary_erosion, binary_dilation
        
        # Create binary mask from confidence
        binary = mask > metadata.confidence_threshold
        
        if not binary.any():
            return 0.0
        
        # Edge band: pixels near boundary
        # Use edge_gating config if available
        edge_width = getattr(self.config.edge_gating, 'edge_low', 0.20)
        iterations = max(1, int(edge_width * 10))  # heuristic
        
        # Erode and dilate to get boundary band
        eroded = binary_erosion(binary, iterations=iterations)
        dilated = binary_dilation(binary, iterations=iterations)
        edge_band = dilated & ~eroded
        
        if not edge_band.any():
            return float(mask.mean())  # fallback
        
        return float(mask[edge_band].mean())
    
    def get_v3_report(self, segmentation_result: Optional[dict] = None) -> dict:
        """Get Materials V3 processing report.
        
        Args:
            segmentation_result: Optional result dict from process()
        
        Returns:
            Report dict with V3-specific metrics
        """
        report = {
            "enabled": self.config.enabled,
            "taxonomy": self.config.taxonomy.value if isinstance(self.config.taxonomy, MaterialTaxonomy) else str(self.config.taxonomy),
            "refinement_strategy": self.config.refine_edges.value if isinstance(self.config.refine_edges, RefinementStrategy) else str(self.config.refine_edges),
            "edge_gating_enabled": self.config.edge_gating.enabled,
        }
        
        # Include per-class stats if available
        if segmentation_result and 'materials_v3' in segmentation_result:
            v3_data = segmentation_result['materials_v3']
            report.update({
                "per_class_stats": v3_data.get("per_class_stats", {}),
                "canonical_materials": v3_data.get("canonical_materials", []),
            })
        
        return report

    def apply_glass_response_if_enabled(
        self,
        image: np.ndarray,
        segmentation_result: dict,
        response_plan: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Apply glass pixel response if enabled and glass is present.
        
        PR-4B: Glass-only pixel operations (canary)
        
        Args:
            image: HxWx3 float32 RGB in [0,1]
            segmentation_result: Result from material_segmentation
            response_plan: Response plan from PR-4A
            
        Returns:
            Enhanced image (HxWx3 float32) + pixel_ops_stats dict
        """
        # Check if pixel ops are enabled
        pixel_ops_enabled = getattr(self.config, 'apply_pixel_ops', False)
        
        if not pixel_ops_enabled:
            return image, {"enabled": False, "reason": "disabled_by_config"}
        
        # Check if glass should be enhanced per response plan
        per_class = response_plan.get("per_class", {})
        glass_plan = per_class.get("glass", {})
        
        should_refine = bool(glass_plan.get("should_refine", False))
        plan_reason = (
            glass_plan.get("refine_reason")
            or glass_plan.get("skip_reason")
            or glass_plan.get("reason")
            or None
        )
        
        forced = bool(getattr(self.config, "force_glass_pixel_ops", False))
        if forced:
            # Validation-only: force apply to prove pixel ops correctness.
            should_refine = True
            plan_reason = "force_glass_pixel_ops"
        
        if not should_refine:
            return image, {
                "enabled": True,
                "applied_to": [],
                "applied": False,
                "reason": plan_reason or "plan_skip_no_reason",
                "forced": forced,
            }
        
        # Extract glass mask
        canonical_materials = segmentation_result.get("materials", {})
        from .materials_v3_taxonomy import normalize_material_dict
        normalized = normalize_material_dict(canonical_materials)
        
        glass_mask = normalized.get("glass")
        if glass_mask is None:
            return image, {"enabled": False, "reason": "glass_mask_missing"}
        
        # Convert mask to numpy float32 if needed
        if hasattr(glass_mask, 'cpu'):  # torch tensor
            glass_mask = glass_mask.cpu().numpy()
        if glass_mask.ndim == 4:  # (1,1,H,W)
            glass_mask = glass_mask[0, 0]
        elif glass_mask.ndim == 3:  # (1,H,W)
            glass_mask = glass_mask[0]
        
        glass_mask = glass_mask.astype(np.float32)
        
        # Apply glass response
        from .materials_v3_pixel_ops import (
            GlassResponseConfig,
            apply_glass_response,
        )
        
        glass_cfg = GlassResponseConfig()
        enhanced, stats = apply_glass_response(image, glass_mask, glass_cfg, glass_plan)
        
        log.info(
            f"PR-4B Glass Response: "
            f"core={stats['core_pixels']}px, edge={stats['edge_pixels']}px, "
            f"mean_delta={stats['mean_delta_core']:.4f}"
        )
        
        return enhanced, {
            "enabled": True,
            "applied": True,
            "applied_to": ["glass"],
            "forced": forced,
            "reason": plan_reason,
            "glass_stats": stats,
        }

    def apply_stone_response_if_enabled(
        self,
        image: np.ndarray,
        segmentation_result: dict,
        response_plan: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Apply stone pixel response if enabled and stone is present.
        
        PR-4D: Stone-only pixel operations (canary)
        
        Args:
            image: HxWx3 float32 RGB in [0,1]
            segmentation_result: Result from material_segmentation
            response_plan: Response plan from PR-4A
            
        Returns:
            Enhanced image (HxWx3 float32) + pixel_ops_stats dict
        """
        # Check if pixel ops are enabled
        pixel_ops_enabled = getattr(self.config, 'apply_pixel_ops', False)
        stone_enabled = getattr(self.config, 'stone_response_enabled', False)
        
        if not pixel_ops_enabled or not stone_enabled:
            return image, {"enabled": False, "reason": "disabled_by_config"}
        
        # Check if stone should be enhanced per response plan
        per_class = response_plan.get("per_class", {})
        stone_plan = per_class.get("stone", {})
        
        should_refine = bool(stone_plan.get("should_refine", False))
        plan_reason = (
            stone_plan.get("refine_reason")
            or stone_plan.get("skip_reason")
            or stone_plan.get("reason")
            or None
        )
        
        forced = bool(getattr(self.config, "force_stone_pixel_ops", False))
        if forced:
            # Validation-only: force apply to prove pixel ops correctness.
            should_refine = True
            plan_reason = "force_stone_pixel_ops"
        
        if not should_refine:
            return image, {
                "enabled": True,
                "applied_to": [],
                "applied": False,
                "reason": plan_reason or "plan_skip_no_reason",
                "forced": forced,
            }
        
        # Extract stone mask
        canonical_materials = segmentation_result.get("materials", {})
        from .materials_v3_taxonomy import normalize_material_dict
        normalized = normalize_material_dict(canonical_materials)
        
        stone_mask = normalized.get("stone")
        if stone_mask is None:
            return image, {"enabled": False, "reason": "stone_mask_missing"}
        
        # Convert mask to numpy float32 if needed
        if hasattr(stone_mask, 'cpu'):  # torch tensor
            stone_mask = stone_mask.cpu().numpy()
        if stone_mask.ndim == 4:  # (1,1,H,W)
            stone_mask = stone_mask[0, 0]
        elif stone_mask.ndim == 3:  # (1,H,W)
            stone_mask = stone_mask[0]
        
        stone_mask = stone_mask.astype(np.float32)
        
        # Apply stone response
        from .materials_v3_pixel_ops_stone import (
            StoneResponseConfig,
            apply_stone_response,
        )
        
        stone_cfg = StoneResponseConfig()
        enhanced, stats = apply_stone_response(image, stone_mask, stone_cfg, stone_plan)
        
        if stats.get('applied', False):
            log.info(
                f"PR-4D Stone Response: "
                f"core={stats.get('core_px', 0)}px, edge={stats.get('edge_px', 0)}px, "
                f"mean_delta={stats.get('mean_delta', 0):.4f}, "
                f"halo_risk={stats.get('halo_risk', 'N/A')}"
            )
        
        return enhanced, {
            "enabled": True,
            "applied": stats.get('applied', False),
            "applied_to": ["stone"] if stats.get('applied', False) else [],
            "forced": forced,
            "reason": plan_reason if stats.get('applied', False) else stats.get('reason', 'unknown'),
            "stone_stats": stats,
        }

