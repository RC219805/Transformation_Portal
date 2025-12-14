# PR Structure: Reliable Water Masks for Materials V3

**Philosophy**: Treat water as a **detector + mask generator**, not as refinement of a SegFormer class that may never emit it.

---

## PR-W0: Water Observability + Contract (Report-Only, Zero Behavior Change)

### Goal
Make water measurable everywhere, even when absent. Establish telemetry foundation.

### Implementation

#### 1. Extend Report Schema (`lux_depth_v2/materials_v3.py`)

```python
@dataclass
class WaterCandidateReport:
    """Water detection telemetry (always present when Materials V3 enabled)."""
    present: bool
    coverage: float  # 0.0-1.0
    coverage_px: int
    confidence: float  # 0.0-1.0
    source: str  # segformer|heuristic|efficientsam_refined|none
    reason: str  # not_emitted_by_segmenter|heuristic_triggered|confidence_too_low|...
    
@dataclass
class MaterialsV3Report:
    # ... existing fields ...
    water_candidate: WaterCandidateReport
    class_presence_audit: dict[str, dict[str, Any]]
```

#### 2. Extend Class Presence Audit

```python
# In MaterialsV3Engine._build_class_presence_audit()
audit["water"] = {
    "raw_present": "water" in segformer_classes_emitted,
    "raw_coverage": segformer_coverage.get("water", 0.0),
    "candidate_present": water_candidate.present,
    "candidate_coverage": water_candidate.coverage,
    "candidate_source": water_candidate.source,
}
```

#### 3. Default Behavior (Disabled)

```python
# In MaterialsV3Engine.process()
if not self.config.water_detection_enabled:
    water_candidate = WaterCandidateReport(
        present=False,
        coverage=0.0,
        coverage_px=0,
        confidence=0.0,
        source="none",
        reason="water_detection_disabled"
    )
```

### Configuration Addition

```python
@dataclass
class MaterialsV3Config:
    # ... existing fields ...
    water_detection_enabled: bool = False  # Opt-in only
```

### Tests (torch-free, `tests/test_materials_v3_water.py`)

```python
def test_water_candidate_report_schema_present():
    """Water candidate report always exists when Materials V3 enabled."""
    config = MaterialsV3Config(water_detection_enabled=False)
    engine = MaterialsV3Engine(config)
    report = engine.process(rgb_dummy, depth_dummy)
    
    assert hasattr(report, 'water_candidate')
    assert report.water_candidate.source == "none"
    assert report.water_candidate.reason == "water_detection_disabled"

def test_water_audit_in_class_presence():
    """Class presence audit includes water metrics."""
    report = process_dummy_scene()
    assert "water" in report.class_presence_audit
    assert "raw_present" in report.class_presence_audit["water"]
    assert "candidate_present" in report.class_presence_audit["water"]

def test_no_behavior_change_when_disabled():
    """Pipeline output identical when water detection disabled."""
    config_disabled = MaterialsV3Config(water_detection_enabled=False)
    config_baseline = MaterialsV3Config()
    
    # Process same image
    result_disabled = process_with_config(config_disabled)
    result_baseline = process_with_config(config_baseline)
    
    # Materials, masks, refinements identical
    assert_materials_identical(result_disabled, result_baseline)
```

### Acceptance Criteria

- ✅ `water_candidate` report block present in all Materials V3 outputs
- ✅ Class presence audit includes water metrics
- ✅ Zero pipeline behavior change when `water_detection_enabled=False`
- ✅ Tests pass without torch/ML dependencies

---

## PR-W1: WaterCandidateDetector (Heuristic Mask Generator, CPU-Only, CI-Safe)

### Goal
Produce high-recall candidate water mask for pool/ocean scenes using cheap, robust heuristics.

### Implementation

#### Create `lux_depth_v2/water_candidate.py`

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from enum import Enum

class SceneContext(Enum):
    POOL = "pool"
    OCEAN = "ocean"
    UNKNOWN = "unknown"

@dataclass
class WaterDetectionParams:
    """Tunable parameters for water detection heuristics."""
    # Chromaticity
    pool_hue_range: tuple[float, float] = (170, 210)  # Cyan/blue
    ocean_hue_range: tuple[float, float] = (160, 220)  # Broader blue-green
    saturation_min: float = 0.15
    value_min: float = 0.20
    
    # Specularness (reflections)
    specular_highlight_threshold: float = 0.85
    specular_low_sat_threshold: float = 0.30
    
    # Texture/entropy
    texture_entropy_max: float = 5.0  # Lower = smoother
    frequency_band_threshold: float = 0.25
    
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
    feature_scores: dict[str, float]  # Individual cue scores
    debug_info: dict[str, Any]  # Intermediate masks, thresholds, etc.

class WaterCandidateDetector:
    """CPU-only heuristic water mask generator."""
    
    def __init__(self, params: Optional[WaterDetectionParams] = None):
        self.params = params or WaterDetectionParams()
    
    def detect(
        self,
        rgb01: np.ndarray,  # HxWx3 float32
        depth01: Optional[np.ndarray] = None,  # HxW float32
        scene_context: SceneContext = SceneContext.UNKNOWN
    ) -> WaterCandidateResult:
        """
        Generate water candidate mask using multi-cue heuristics.
        
        Returns high-recall mask with confidence score.
        """
        h, w = rgb01.shape[:2]
        
        # Convert to HSV and Lab for chromaticity analysis
        hsv = self._rgb_to_hsv(rgb01)
        lab = self._rgb_to_lab(rgb01)
        
        # Feature extraction
        chroma_mask, chroma_score = self._chromaticity_cue(hsv, lab, scene_context)
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
        self, hsv: np.ndarray, lab: np.ndarray, scene_context: SceneContext
    ) -> tuple[np.ndarray, float]:
        """Blue/green dominance in HSV/Lab (pool vs ocean tuned)."""
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
    ) -> tuple[np.ndarray, float]:
        """High highlights + low saturation pockets (water reflections)."""
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        
        # High value (bright) with low saturation (specular reflection)
        specular = (val >= self.params.specular_highlight_threshold) & \
                   (sat <= self.params.specular_low_sat_threshold)
        
        # Dilate slightly to capture reflection context
        from scipy import ndimage
        mask = ndimage.binary_dilation(specular, iterations=2).astype(np.float32)
        score = float(np.mean(mask))
        
        return mask, score
    
    def _texture_cue(self, rgb01: np.ndarray) -> tuple[np.ndarray, float]:
        """Water tends to be lower-frequency than foliage/stone."""
        from scipy import ndimage
        
        # Convert to grayscale
        gray = np.mean(rgb01, axis=2)
        
        # Local entropy (lower for water)
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        
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
    ) -> tuple[np.ndarray, float]:
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
    ) -> tuple[np.ndarray, float]:
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
    
    def _rgb_to_lab(self, rgb01: np.ndarray) -> np.ndarray:
        """Convert RGB to Lab."""
        from skimage.color import rgb2lab
        return rgb2lab(rgb01)
```

### Tests (`tests/test_water_candidate_detector.py`, torch-free)

```python
import numpy as np
import pytest
from lux_depth_v2.water_candidate import (
    WaterCandidateDetector,
    SceneContext,
    WaterDetectionParams
)

def test_synthetic_pool_yields_mask():
    """Synthetic pool-like image yields mask above threshold."""
    # Create 512x512 image with pool-blue color
    rgb = np.zeros((512, 512, 3), dtype=np.float32)
    rgb[:, :, 2] = 0.7  # Blue channel
    rgb[:, :, 1] = 0.3  # Green channel
    
    detector = WaterCandidateDetector()
    result = detector.detect(rgb, scene_context=SceneContext.POOL)
    
    assert result.coverage > 0.3, "Pool scene should have significant coverage"
    assert result.confidence > 0.3, "Pool scene should have decent confidence"

def test_synthetic_foliage_yields_near_zero():
    """Synthetic foliage-like image yields near-zero mask."""
    # Create 512x512 image with green foliage color
    rgb = np.zeros((512, 512, 3), dtype=np.float32)
    rgb[:, :, 1] = 0.6  # Green dominant
    rgb[:, :, 0] = 0.3  # Some red
    
    # Add noise for texture
    noise = np.random.rand(512, 512, 3) * 0.1
    rgb = np.clip(rgb + noise, 0, 1)
    
    detector = WaterCandidateDetector()
    result = detector.detect(rgb)
    
    assert result.coverage < 0.2, "Foliage scene should have low water coverage"

def test_component_filtering_works():
    """Component filtering removes tiny blobs."""
    # Create mask with large region + tiny blobs
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:400, 100:400] = 1.0  # Large region
    mask[10:15, 10:15] = 1.0  # Tiny blob
    mask[500:505, 500:505] = 1.0  # Another tiny blob
    
    detector = WaterCandidateDetector()
    filtered, stability = detector._component_filtering(mask)
    
    # Should keep only large region
    assert np.sum(filtered) < np.sum(mask), "Should remove tiny blobs"
    assert np.sum(filtered[100:400, 100:400]) > 0, "Should keep large region"

def test_depth_planarity_cue():
    """Planarity cue detects low-gradient regions."""
    # Create flat depth map
    depth = np.ones((512, 512), dtype=np.float32) * 0.5
    
    detector = WaterCandidateDetector()
    planarity_mask, score = detector._planarity_cue(depth)
    
    assert score > 0.9, "Flat surface should score high on planarity"

def test_feature_scores_in_result():
    """Result contains individual feature scores."""
    rgb = np.random.rand(256, 256, 3).astype(np.float32)
    
    detector = WaterCandidateDetector()
    result = detector.detect(rgb)
    
    assert "chromaticity" in result.feature_scores
    assert "specular" in result.feature_scores
    assert "texture" in result.feature_scores
    assert "component_stability" in result.feature_scores
```

### Acceptance Criteria

- ✅ `WaterCandidateDetector` produces masks for pool-like images
- ✅ Low false-positive rate on foliage/stone textures
- ✅ Component filtering removes noise effectively
- ✅ Confidence scores reflect mask quality
- ✅ All tests pass without torch/ML dependencies
- ✅ Debug info available for tuning

---

## PR-W2: Integration into Materials V3 Canonical Materials

### Goal
Make water available to planning/gating even when SegFormer misses it.

### Implementation

#### Extend `MaterialsV3Engine.process()`

```python
class MaterialsV3Engine:
    def __init__(self, config: MaterialsV3Config):
        self.config = config
        self.water_detector = WaterCandidateDetector() if config.water_detection_enabled else None
    
    def process(self, rgb01, depth01=None):
        # ... existing SegFormer processing ...
        
        # Canonical materials from SegFormer
        canonical_materials = self._build_canonical_materials(segformer_output)
        
        # Water candidate detection
        water_candidate_report = self._detect_water_candidate(
            rgb01, depth01, canonical_materials
        )
        
        # Inject water if candidate passes thresholds
        if self._should_inject_water_candidate(water_candidate_report):
            canonical_materials["water"] = self._build_water_material(
                water_candidate_report
            )
        
        # ... rest of pipeline ...
        
        return MaterialsV3Report(
            canonical_materials=canonical_materials,
            water_candidate=water_candidate_report,
            # ... other fields ...
        )
    
    def _detect_water_candidate(
        self, 
        rgb01: np.ndarray,
        depth01: Optional[np.ndarray],
        canonical_materials: dict
    ) -> WaterCandidateReport:
        """Run water candidate detection."""
        
        if not self.config.water_detection_enabled:
            return WaterCandidateReport(
                present=False, coverage=0.0, coverage_px=0,
                confidence=0.0, source="none",
                reason="water_detection_disabled"
            )
        
        # Check if SegFormer already provided water
        if "water" in canonical_materials:
            water_mat = canonical_materials["water"]
            if water_mat.coverage >= self.config.water_min_coverage:
                return WaterCandidateReport(
                    present=True,
                    coverage=water_mat.coverage,
                    coverage_px=int(water_mat.coverage * rgb01.shape[0] * rgb01.shape[1]),
                    confidence=1.0,  # Trust SegFormer when it emits water
                    source="segformer",
                    reason="segformer_emitted_sufficient_coverage"
                )
        
        # Run heuristic detector
        scene_context = self._infer_scene_context(canonical_materials)
        result = self.water_detector.detect(rgb01, depth01, scene_context)
        
        return WaterCandidateReport(
            present=result.confidence >= self.config.water_candidate_confidence_threshold,
            coverage=result.coverage,
            coverage_px=result.coverage_px,
            confidence=result.confidence,
            source="heuristic",
            reason=f"heuristic_confidence_{result.confidence:.3f}"
        )
    
    def _should_inject_water_candidate(
        self, water_candidate: WaterCandidateReport
    ) -> bool:
        """Decide if candidate should be added to canonical materials."""
        return (
            water_candidate.present and
            water_candidate.coverage >= self.config.water_min_coverage and
            water_candidate.confidence >= self.config.water_candidate_confidence_threshold
        )
    
    def _build_water_material(
        self, water_candidate: WaterCandidateReport
    ) -> Material:
        """Convert water candidate to Material object."""
        return Material(
            class_name="water",
            coverage=water_candidate.coverage,
            mask=water_candidate.mask,  # Store in result
            source="heuristic",
            confidence=water_candidate.confidence,
        )
    
    def _infer_scene_context(
        self, canonical_materials: dict
    ) -> SceneContext:
        """Infer pool vs ocean vs unknown from materials."""
        # Simple heuristic: if building/architecture present -> pool
        # If large sky/horizon -> ocean
        # Otherwise unknown
        
        if "building" in canonical_materials or "wall" in canonical_materials:
            return SceneContext.POOL
        elif "sky" in canonical_materials and canonical_materials["sky"].coverage > 0.3:
            return SceneContext.OCEAN
        return SceneContext.UNKNOWN
```

#### Configuration Updates

```python
@dataclass
class MaterialsV3Config:
    # ... existing fields ...
    
    # Water detection
    water_detection_enabled: bool = False  # Opt-in only
    water_candidate_confidence_threshold: float = 0.4
    water_min_coverage: float = 0.05  # 5% of image
```

### Safety Rules

1. **Opt-in only**: `water_detection_enabled=False` by default
2. **Canary preset**: Enable only in experimental presets initially
3. **SegFormer priority**: If SegFormer emits water with sufficient coverage, use it
4. **Gating thresholds**: Confidence and coverage must both pass
5. **Source tracking**: Always mark heuristic-derived water as `source="heuristic"`

### Tests

```python
def test_water_injected_when_segformer_missing():
    """Water candidate injected when SegFormer misses it."""
    # Create pool-like image
    rgb = create_pool_synthetic()
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_candidate_confidence_threshold=0.3
    )
    engine = MaterialsV3Engine(config)
    result = engine.process(rgb)
    
    assert "water" in result.canonical_materials
    assert result.canonical_materials["water"].source == "heuristic"
    assert result.water_candidate.present

def test_segformer_water_preferred_over_heuristic():
    """SegFormer water takes priority when available."""
    # Mock SegFormer output with water
    segformer_with_water = mock_segformer_output(classes=["water", "sky"])
    
    config = MaterialsV3Config(water_detection_enabled=True)
    # ... process ...
    
    assert result.canonical_materials["water"].source == "segformer"
    assert result.water_candidate.source == "segformer"

def test_water_not_injected_below_thresholds():
    """Water candidate not injected if confidence/coverage too low."""
    rgb = create_ambiguous_scene()
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_candidate_confidence_threshold=0.6,
        water_min_coverage=0.1
    )
    # ... process ...
    
    assert "water" not in result.canonical_materials
    assert result.water_candidate.present == False
```

### Acceptance Criteria

- ✅ Water injected into canonical materials when SegFormer misses it
- ✅ SegFormer water preferred when available
- ✅ Confidence and coverage thresholds enforced
- ✅ Source tracking accurate (`segformer` vs `heuristic`)
- ✅ Disabled by default; opt-in via config

---

## PR-W3: Optional Edge Refinement (EfficientSAM After Candidate Exists)

### Goal
Improve water mask boundaries without relying on SegFormer vocabulary.

### Implementation

#### Extend Configuration

```python
@dataclass
class MaterialsV3Config:
    # ... existing fields ...
    
    # Water edge refinement
    water_edge_refinement_enabled: bool = False
    water_edge_refinement_min_confidence: float = 0.5
    water_edge_refinement_min_boundary_px: int = 100
```

#### Integration in Materials V3

```python
def _refine_water_edges(
    self,
    rgb01: np.ndarray,
    water_candidate_mask: np.ndarray,
    water_confidence: float
) -> Optional[np.ndarray]:
    """Refine water edges using EfficientSAM (only after candidate exists)."""
    
    if not self.config.water_edge_refinement_enabled:
        return None
    
    if water_confidence < self.config.water_edge_refinement_min_confidence:
        return None
    
    # Compute boundary pixels
    boundary_mask = self._extract_boundary(water_candidate_mask)
    boundary_px = int(np.sum(boundary_mask))
    
    if boundary_px < self.config.water_edge_refinement_min_boundary_px:
        return None  # Avoid degenerate BF1 failures
    
    # Generate prompts from high-confidence regions
    prompts = self._sample_prompts_from_mask(
        water_candidate_mask,
        confidence_threshold=0.7,
        num_samples=5
    )
    
    # ROI crop around candidate region
    roi_bbox = self._compute_roi_bbox(water_candidate_mask, padding=50)
    rgb_roi = self._crop_to_roi(rgb01, roi_bbox)
    
    # Run EfficientSAM
    sam_mask = self.efficient_sam.segment(
        rgb_roi,
        point_prompts=prompts,
        multimask_output=False
    )
    
    # Map back to full resolution
    refined_mask = self._uncrop_from_roi(sam_mask, roi_bbox, rgb01.shape[:2])
    
    return refined_mask

def _extract_boundary(self, mask: np.ndarray, width: int = 5) -> np.ndarray:
    """Extract boundary region of mask."""
    from scipy import ndimage
    
    # Dilate and subtract original to get boundary
    dilated = ndimage.binary_dilation(mask > 0.5, iterations=width)
    eroded = ndimage.binary_erosion(mask > 0.5, iterations=width)
    boundary = (dilated & ~eroded).astype(np.float32)
    
    return boundary
```

#### Update Report Schema

```python
@dataclass
class WaterCandidateReport:
    # ... existing fields ...
    edge_refined: bool = False
    edge_refinement_boundary_px: int = 0
    edge_refinement_applied: bool = False
```

### Tests

```python
def test_edge_refinement_only_after_candidate():
    """Edge refinement runs only when candidate exists."""
    rgb = create_pool_synthetic()
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_edge_refinement_enabled=True,
        water_candidate_confidence_threshold=0.4
    )
    # ... process ...
    
    assert result.water_candidate.edge_refinement_applied

def test_edge_refinement_skipped_low_confidence():
    """Edge refinement skipped when confidence too low."""
    config = MaterialsV3Config(
        water_edge_refinement_min_confidence=0.8
    )
    # ... process with low-confidence water ...
    
    assert not result.water_candidate.edge_refinement_applied

def test_edge_refinement_skipped_small_boundary():
    """Edge refinement skipped when boundary too small (BF1 avoidance)."""
    # Create tiny water blob
    config = MaterialsV3Config(
        water_edge_refinement_min_boundary_px=1000
    )
    # ... process ...
    
    assert not result.water_candidate.edge_refinement_applied
```

### Acceptance Criteria

- ✅ Edge refinement runs only after candidate exists
- ✅ Confidence and boundary pixel thresholds enforced
- ✅ ROI cropping reduces SAM overhead
- ✅ Prompts sampled from high-confidence regions
- ✅ Graceful degradation when boundary too small

---

## PR-W4: Validation Harness (Pool + Ocean)

### Goal
Make water detection decision defensible with automated quality metrics.

### Implementation

#### Create `scripts/prw_water_validation.py`

```python
#!/usr/bin/env python3
"""
Water candidate validation harness for pool and ocean scenes.

Produces:
- Coverage sanity checks
- Boundary pixel statistics
- Edge alignment vs gradients (primary metric)
- Stability across perturbations
- False-positive checks on non-water scenes
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List
import json

from lux_depth_v2.materials_v3 import MaterialsV3Engine, MaterialsV3Config
from lux_depth_v2.water_candidate import SceneContext

@dataclass
class ValidationResult:
    """Single validation test result."""
    image_path: str
    scene_type: str  # pool|ocean|non_water
    
    # Coverage
    coverage: float
    coverage_px: int
    
    # Confidence
    confidence: float
    source: str
    
    # Edge quality (primary metric)
    edge_alignment_score: float  # vs image gradients
    boundary_px: int
    
    # Stability
    stability_score: float  # across perturbations
    
    # False positive check
    is_false_positive: bool
    
    # Performance
    processing_time_ms: float

class WaterValidationHarness:
    """Validation harness for water detection."""
    
    def __init__(self, config: MaterialsV3Config):
        self.engine = MaterialsV3Engine(config)
    
    def validate_dataset(
        self,
        image_paths: List[Path],
        ground_truth_labels: dict[str, str]  # path -> scene_type
    ) -> List[ValidationResult]:
        """Run validation on dataset."""
        results = []
        
        for img_path in image_paths:
            result = self.validate_single(
                img_path,
                expected_scene=ground_truth_labels.get(str(img_path), "unknown")
            )
            results.append(result)
        
        return results
    
    def validate_single(
        self, img_path: Path, expected_scene: str
    ) -> ValidationResult:
        """Validate single image."""
        import time
        from PIL import Image
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        rgb01 = np.array(img, dtype=np.float32) / 255.0
        
        # Process
        start = time.perf_counter()
        report = self.engine.process(rgb01)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Extract water candidate
        water = report.water_candidate
        
        # Compute edge alignment
        edge_score = self._compute_edge_alignment(
            rgb01, water.mask if hasattr(water, 'mask') else None
        )
        
        # Compute stability
        stability = self._compute_stability(rgb01)
        
        # Check false positive
        is_fp = (expected_scene == "non_water" and water.present)
        
        return ValidationResult(
            image_path=str(img_path),
            scene_type=expected_scene,
            coverage=water.coverage,
            coverage_px=water.coverage_px,
            confidence=water.confidence,
            source=water.source,
            edge_alignment_score=edge_score,
            boundary_px=self._count_boundary_pixels(water.mask) if hasattr(water, 'mask') else 0,
            stability_score=stability,
            is_false_positive=is_fp,
            processing_time_ms=elapsed_ms
        )
    
    def _compute_edge_alignment(
        self, rgb01: np.ndarray, mask: Optional[np.ndarray]
    ) -> float:
        """
        Primary metric: edge alignment vs image gradients.
        
        High score = mask boundaries align with image edges.
        """
        if mask is None:
            return 0.0
        
        from scipy import ndimage
        
        # Compute image gradients
        gray = np.mean(rgb01, axis=2)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract mask boundary
        boundary = self._extract_boundary(mask)
        
        # Measure overlap between boundary and high-gradient regions
        grad_threshold = np.percentile(grad_mag, 75)
        high_grad = (grad_mag >= grad_threshold).astype(np.float32)
        
        overlap = np.sum(boundary * high_grad)
        max_overlap = np.sum(boundary)
        
        score = overlap / max(max_overlap, 1)
        return float(score)
    
    def _compute_stability(self, rgb01: np.ndarray) -> float:
        """
        Stability across minor perturbations (resize/compress jitter).
        
        High score = consistent detection under perturbations.
        """
        from PIL import Image
        
        # Baseline detection
        baseline_report = self.engine.process(rgb01)
        baseline_coverage = baseline_report.water_candidate.coverage
        
        # Perturbation 1: slight resize
        h, w = rgb01.shape[:2]
        resized = np.array(
            Image.fromarray((rgb01 * 255).astype(np.uint8)).resize(
                (int(w * 0.95), int(h * 0.95))
            )
        ).astype(np.float32) / 255.0
        resized_report = self.engine.process(resized)
        
        # Perturbation 2: JPEG compression simulation (add slight noise)
        noisy = rgb01 + np.random.randn(*rgb01.shape) * 0.01
        noisy = np.clip(noisy, 0, 1)
        noisy_report = self.engine.process(noisy)
        
        # Compute coverage variance
        coverages = [
            baseline_coverage,
            resized_report.water_candidate.coverage,
            noisy_report.water_candidate.coverage
        ]
        std = np.std(coverages)
        
        # Low variance = high stability
        stability = 1.0 - min(std * 5, 1.0)  # Scale to [0, 1]
        return float(stability)
    
    def _extract_boundary(self, mask: np.ndarray, width: int = 3) -> np.ndarray:
        """Extract boundary of mask."""
        from scipy import ndimage
        dilated = ndimage.binary_dilation(mask > 0.5, iterations=width)
        eroded = ndimage.binary_erosion(mask > 0.5, iterations=width)
        return (dilated & ~eroded).astype(np.float32)
    
    def _count_boundary_pixels(self, mask: np.ndarray) -> int:
        """Count boundary pixels."""
        boundary = self._extract_boundary(mask)
        return int(np.sum(boundary))
    
    def generate_report(
        self, results: List[ValidationResult], output_path: Path
    ):
        """Generate JSON validation report."""
        # Summary statistics
        pool_results = [r for r in results if r.scene_type == "pool"]
        ocean_results = [r for r in results if r.scene_type == "ocean"]
        non_water_results = [r for r in results if r.scene_type == "non_water"]
        
        summary = {
            "total_images": len(results),
            "pool_scenes": len(pool_results),
            "ocean_scenes": len(ocean_results),
            "non_water_scenes": len(non_water_results),
            
            # Coverage stats
            "pool_avg_coverage": np.mean([r.coverage for r in pool_results]) if pool_results else 0,
            "ocean_avg_coverage": np.mean([r.coverage for r in ocean_results]) if ocean_results else 0,
            
            # Edge alignment (primary metric)
            "pool_avg_edge_alignment": np.mean([r.edge_alignment_score for r in pool_results]) if pool_results else 0,
            "ocean_avg_edge_alignment": np.mean([r.edge_alignment_score for r in ocean_results]) if ocean_results else 0,
            
            # Stability
            "overall_avg_stability": np.mean([r.stability_score for r in results]),
            
            # False positives
            "false_positive_count": sum(r.is_false_positive for r in results),
            "false_positive_rate": sum(r.is_false_positive for r in results) / max(len(non_water_results), 1),
            
            # Performance
            "avg_processing_time_ms": np.mean([r.processing_time_ms for r in results]),
        }
        
        report = {
            "summary": summary,
            "results": [vars(r) for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report written to {output_path}")
        print(f"\nSummary:")
        print(f"  Pool scenes: {len(pool_results)}, avg edge alignment: {summary['pool_avg_edge_alignment']:.3f}")
        print(f"  Ocean scenes: {len(ocean_results)}, avg edge alignment: {summary['ocean_avg_edge_alignment']:.3f}")
        print(f"  False positive rate: {summary['false_positive_rate']:.2%}")
        print(f"  Avg stability: {summary['overall_avg_stability']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Water validation harness")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True,
                       help="JSON mapping image_path -> scene_type")
    parser.add_argument("--output", type=Path, default=Path("water_validation_report.json"))
    args = parser.parse_args()
    
    # Load ground truth
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    
    # Find images
    image_paths = list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.png"))
    
    # Run validation
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_edge_refinement_enabled=True
    )
    harness = WaterValidationHarness(config)
    results = harness.validate_dataset(image_paths, ground_truth)
    
    # Generate report
    harness.generate_report(results, args.output)

if __name__ == "__main__":
    main()
```

#### Ground Truth Format

```json
{
  "data/pool_001.jpg": "pool",
  "data/pool_002.jpg": "pool",
  "data/ocean_001.jpg": "ocean",
  "data/foliage_001.jpg": "non_water",
  "data/interior_001.jpg": "non_water"
}
```

### Acceptance Criteria

- ✅ Validation harness runs on pool/ocean/non-water scenes
- ✅ Edge alignment metric (primary) computed for all detections
- ✅ Stability metric tracks consistency across perturbations
- ✅ False-positive rate computed for non-water scenes
- ✅ JSON report with summary statistics
- ✅ Performance metrics (processing time) included

---

## Implementation Sequence

1. **PR-W0** (1-2 days): Report schema and observability
   - Merge fast, zero risk, enables telemetry

2. **PR-W1** (3-5 days): Heuristic detector
   - Independent module, heavily tested
   - Can iterate on parameters post-merge

3. **PR-W2** (2-3 days): Materials V3 integration
   - Depends on W0 and W1
   - Canary preset initially, gradual rollout

4. **PR-W3** (2-3 days): Edge refinement (optional)
   - Can ship without this if heuristics sufficient
   - Adds polish but not critical path

5. **PR-W4** (2-3 days): Validation harness
   - Can develop in parallel with W1-W3
   - Provides confidence for production enablement

**Total Timeline**: ~2-3 weeks with sequential merges, or ~10 days with parallel development.

---

## Quality Gates

Each PR must pass:

1. **Tests**: All new tests pass (torch-free for W0, W1, W2)
2. **Linting**: flake8/pylint clean
3. **Documentation**: Inline docstrings + README updates
4. **Performance**: No regression in Materials V3 baseline (when disabled)
5. **Semantic Consistency**: Aligns with Materials V3 architecture
6. **Security**: No new CVEs introduced

---

## Success Metrics (Post-Deployment)

Track in production telemetry:

- **Detection Rate**: % of pool/ocean scenes with water detected
- **False Positive Rate**: % of non-water scenes with false detections
- **Edge Quality**: Average edge alignment score
- **Stability**: Coverage variance across perturbations
- **Performance**: p50/p95 processing time overhead

**Target Thresholds**:
- Detection rate ≥ 85% for pool scenes
- False positive rate ≤ 5% for non-water scenes
- Edge alignment ≥ 0.6 (60% boundary-gradient overlap)
- Stability ≥ 0.8 (coverage std ≤ 0.04)
- Processing overhead ≤ 50ms per image

---

## Rollout Strategy

1. **Canary Preset** (Week 1-2): Enable in `experimental_water_detection` preset only
2. **Validation** (Week 3): Run validation harness on production dataset
3. **Gradual Rollout** (Week 4-6): Enable in pool/ocean presets if metrics pass
4. **Default Enable** (Week 7+): Make default after 2 weeks of stable metrics

**Rollback Plan**: Single config flag (`water_detection_enabled=False`) disables entire subsystem.

---

## Open Questions

1. **Scene context inference**: Current heuristic (building → pool, sky → ocean). Sufficient?
2. **Parameter tuning**: Initial thresholds are educated guesses. Plan for A/B testing?
3. **EfficientSAM integration**: SAM version (1 vs 2)? MobileSAM for speed?
4. **Depth dependency**: How critical is depth for planarity cue? Optional or required?
5. **Multi-material interactions**: How does water candidate interact with glass/sky separation?

---

## References

- **Materials V3 Architecture**: `lux_depth_v2/materials_v3.py`
- **EfficientSAM Integration**: `lux_depth_v2/sam_integration.py` (if exists)
- **Depth Pipeline**: `lux_depth_v2/pipeline.py`
- **Test Patterns**: `tests/test_materials_v3*.py`
