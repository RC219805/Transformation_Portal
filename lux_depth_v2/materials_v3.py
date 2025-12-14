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
        segmentation_result['materials_v3'] = {
            "enabled": True,
            "taxonomy": self.config.taxonomy.value,
            "refinement_strategy": self.config.refine_edges.value,
            "per_class_stats": per_class_stats,
            "canonical_materials": list(canonical_materials.keys()),
            "class_presence_audit": class_audit,  # NEW: diagnose missing classes
        }
        
        # PR-4A: Generate response plan (no pixel ops)
        response_plan_config = ResponsePlanConfig()
        response_plan = generate_response_plan(
            canonical_materials=canonical_materials,
            config=response_plan_config,
            strategy=self.config.refine_edges.value,
            intent="client",  # TODO: get from auto-preset context
            quality_tier="max",  # TODO: get from pipeline config
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
