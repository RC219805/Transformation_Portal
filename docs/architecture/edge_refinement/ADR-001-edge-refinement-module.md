# ADR 001: Edge Refinement Module Architecture

**Status**: Proposed (Design Phase - Feature Freeze)
**Date**: December 20, 2025
**Deciders**: Transformation Portal Architect
**Technical Story**: Structure scene improvement (25% → 60%+) via edge-aware post-processing

---

## Context

Current depth processing pipelines (lux_depth_v2, lux_depth_v3) achieve strong results for material enhancement and depth-aware processing, but structure scene quality remains at ~25% in validation metrics. Analysis indicates that **edge-aware refinement** can improve structure preservation through:

1. **Bilateral Filtering**: Edge-preserving smoothing that maintains structural boundaries
2. **Guided Filter**: Fast edge-aware filtering using depth maps as guidance
3. **Edge-Guided Enhancement**: Targeted sharpening and contrast enhancement along structural edges

Additionally, preliminary analysis suggests that **input size sweep** (518px → 1022px) may improve depth estimation quality for architectural scenes with fine structural details.

### Current Limitations

1. **Structure Preservation**: Existing pipelines lack dedicated edge-aware processing
2. **Architectural Detail**: Fine structural details (railings, window frames, moldings) suffer from over-smoothing
3. **Depth-Structure Alignment**: Depth maps and structural edges are not consistently aligned
4. **Configuration Complexity**: No unified configuration for edge refinement parameters

### Feature Freeze Constraint

This ADR is written during the **active feature freeze** (until Jan 10, 2026). Implementation is prohibited until the freeze lifts. This document serves as **design preparation** for rapid implementation in January.

---

## Decision

We will implement a **modular edge refinement subsystem** within `lux_depth_v2` that:

1. **Integrates cleanly** with the existing Golden Path architecture
2. **Operates as opt-in** (disabled by default, preserves backward compatibility)
3. **Provides dedicated configuration** for refinement parameters
4. **Supports multiple algorithms** (bilateral, guided filter, edge-guided enhancement)
5. **Maintains testability** through isolated unit tests per algorithm

### Architecture Design

```
lux_depth_v2/
├── pipeline.py                    # MODIFIED: Add refinement hook
├── config.py                      # MODIFIED: Add refinement configuration
├── upscaling.py                   # Unchanged
├── material_segmentation.py       # Unchanged
└── refinement/                    # NEW: Edge refinement subsystem
    ├── __init__.py                # Public API exports
    ├── bilateral_filter.py        # Bilateral filtering implementation
    ├── guided_filter.py           # Guided filter implementation
    ├── edge_enhancer.py           # Edge-guided enhancement
    ├── structure_scorer.py        # Structure quality metrics
    └── config.py                  # Refinement-specific configuration
```

### Integration Points

#### 1. Pipeline Hook (lux_depth_v2/pipeline.py)

```python
class LuxDepthV2Pipeline:
    def process_image(self, image: np.ndarray, config: PipelineConfig) -> np.ndarray:
        # Existing stages
        depth_map = self.estimate_depth(image)
        enhanced = self.enhance_materials(image, depth_map)

        # NEW: Optional refinement stage
        if config.enable_edge_refinement:
            enhanced = self._apply_edge_refinement(enhanced, depth_map, config)

        return enhanced

    def _apply_edge_refinement(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        config: PipelineConfig
    ) -> np.ndarray:
        """Apply edge-aware refinement to preserve structural details."""
        from lux_depth_v2.refinement import EdgeRefinementPipeline

        refiner = EdgeRefinementPipeline(config.refinement_config)
        return refiner.refine(image, depth_map)
```

#### 2. Configuration Extension (lux_depth_v2/config.py)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RefinementConfig:
    """Configuration for edge-aware refinement."""

    # Bilateral filtering
    enable_bilateral: bool = True
    bilateral_sigma_space: float = 5.0
    bilateral_sigma_color: float = 0.1

    # Guided filter
    enable_guided_filter: bool = True
    guided_filter_radius: int = 8
    guided_filter_eps: float = 0.01

    # Edge enhancement
    enable_edge_enhancement: bool = True
    edge_enhancement_strength: float = 0.3
    edge_detection_threshold: float = 0.15

    # Structure preservation
    structure_weight: float = 0.5  # Balance between smoothing and preservation
    min_structure_score: float = 0.6  # Target structure quality score


@dataclass
class PipelineConfig:
    """Main pipeline configuration (EXTENDED)."""

    # Existing fields
    preset: str = "interior_luxury"
    input_size: int = 518  # NEW: Configurable input size for sweep experiments

    # NEW: Refinement configuration
    enable_edge_refinement: bool = False
    refinement_config: Optional[RefinementConfig] = None

    @classmethod
    def apply_preset(cls, preset: str) -> "PipelineConfig":
        """Apply named preset with refinement settings."""
        configs = {
            "interior_luxury": cls(
                preset="interior_luxury",
                enable_edge_refinement=False,  # Disabled by default
            ),
            "interior_structure_enhanced": cls(  # NEW PRESET
                preset="interior_structure_enhanced",
                enable_edge_refinement=True,
                refinement_config=RefinementConfig(
                    bilateral_sigma_space=5.0,
                    guided_filter_radius=8,
                    edge_enhancement_strength=0.3,
                    structure_weight=0.6,  # Higher structure preservation
                ),
            ),
            # ... other presets
        }
        return configs.get(preset, cls())
```

#### 3. CLI Integration (lux_depth_v2/cli.py)

```python
@app.command()
def process(
    input_dir: Path,
    output_dir: Path,
    preset: str = "interior_luxury",
    enable_edge_refinement: bool = False,  # NEW FLAG
    input_size: int = 518,  # NEW: Input size sweep parameter
):
    """Process images with optional edge refinement."""
    config = PipelineConfig.apply_preset(preset)
    config.enable_edge_refinement = enable_edge_refinement
    config.input_size = input_size

    pipeline = LuxDepthV2Pipeline(config)
    # ... processing logic
```

---

## Refinement Algorithms

### 1. Bilateral Filter

**Purpose**: Edge-preserving smoothing that reduces noise while maintaining structural boundaries.

**Parameters**:
- `sigma_space`: Spatial extent (default: 5.0)
- `sigma_color`: Color similarity threshold (default: 0.1)

**Implementation Strategy**:
```python
def bilateral_filter(image: np.ndarray, sigma_space: float, sigma_color: float) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.

    Uses OpenCV's bilateralFilter with automatic parameter scaling
    based on image resolution.
    """
    # Scale parameters for image resolution
    h, w = image.shape[:2]
    d = int(sigma_space * 2) + 1  # Diameter derived from sigma_space

    # Apply bilateral filter (preserves edges)
    filtered = cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color * 255,  # Scale to [0, 255]
        sigmaSpace=sigma_space
    )

    return filtered
```

**Testing Strategy**:
- Unit test: Verify edge preservation on synthetic test images (sharp edges + noise)
- Property test: Ensure output is bounded and preserves data type
- Validation test: Measure structure quality score improvement on real architectural images

### 2. Guided Filter

**Purpose**: Fast edge-aware filtering using depth map as guidance to align structural boundaries.

**Parameters**:
- `radius`: Filter radius (default: 8)
- `eps`: Regularization parameter (default: 0.01)

**Implementation Strategy**:
```python
def guided_filter(image: np.ndarray, depth_map: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Apply guided filter using depth map as guidance.

    Aligns image edges with depth discontinuities for improved
    structure preservation.
    """
    # Implementation options:
    # 1. cv2.ximgproc.guidedFilter (requires opencv-contrib)
    # 2. Custom implementation (for dependency control)

    # For now, use cv2.ximgproc if available, fallback to bilateral
    try:
        from cv2 import ximgproc
        filtered = ximgproc.guidedFilter(
            guide=depth_map,
            src=image,
            radius=radius,
            eps=eps
        )
    except ImportError:
        # Fallback: Use bilateral filter
        filtered = bilateral_filter(image, sigma_space=radius, sigma_color=eps)

    return filtered
```

**Testing Strategy**:
- Unit test: Verify depth-guided edge alignment on synthetic depth + image pairs
- Integration test: Ensure compatibility with Depth Anything V2 depth maps
- Dependency test: Verify graceful fallback when cv2.ximgproc unavailable

### 3. Edge-Guided Enhancement

**Purpose**: Targeted sharpening and contrast enhancement along structural edges.

**Parameters**:
- `enhancement_strength`: Enhancement intensity (default: 0.3)
- `edge_detection_threshold`: Edge detection sensitivity (default: 0.15)

**Implementation Strategy**:
```python
def edge_guided_enhancement(
    image: np.ndarray,
    depth_map: np.ndarray,
    strength: float,
    threshold: float
) -> np.ndarray:
    """
    Apply edge-guided enhancement for structural detail preservation.

    Detects edges from depth map and applies targeted sharpening
    and contrast enhancement.
    """
    # 1. Detect edges from depth map
    edges = cv2.Canny(
        (depth_map * 255).astype(np.uint8),
        threshold1=threshold * 255 * 0.5,
        threshold2=threshold * 255
    )

    # 2. Create edge mask (dilated for local enhancement)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_mask = edge_mask.astype(np.float32) / 255.0

    # 3. Apply unsharp masking along edges
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    # 4. Blend sharpened and original using edge mask
    enhanced = (
        image * (1 - edge_mask[:, :, np.newaxis]) +
        sharpened * edge_mask[:, :, np.newaxis]
    )

    return enhanced
```

**Testing Strategy**:
- Unit test: Verify edge detection and selective enhancement on synthetic test images
- Visual test: Compare before/after enhancement on architectural renders
- Metric test: Measure sharpness improvement (Laplacian variance) in edge regions

---

## Input Size Sweep Experiment

**Hypothesis**: Larger input sizes (1022px vs 518px) improve depth estimation quality for architectural scenes with fine structural details.

**Experimental Design**:
```python
# Test input sizes: [518, 640, 768, 896, 1022]
# Metrics: Structure score, edge sharpness, depth consistency
# Dataset: 10-20 architectural renders (interior + exterior)

def run_input_size_sweep(
    image_paths: List[Path],
    sizes: List[int],
    output_dir: Path
):
    """Execute input size sweep experiment."""
    results = []

    for size in sizes:
        for image_path in image_paths:
            config = PipelineConfig(input_size=size)
            pipeline = LuxDepthV2Pipeline(config)

            # Process image
            result = pipeline.process_image(load_image(image_path), config)

            # Compute metrics
            metrics = {
                "input_size": size,
                "structure_score": compute_structure_score(result),
                "edge_sharpness": compute_edge_sharpness(result),
                "processing_time": pipeline.last_processing_time,
            }
            results.append(metrics)

    # Generate sweep report
    generate_sweep_report(results, output_dir)
```

**Metrics**:
1. **Structure Score**: Frequency of high-frequency content in structural regions (FFT-based)
2. **Edge Sharpness**: Laplacian variance along detected edges
3. **Processing Time**: Latency vs quality tradeoff analysis

---

## Consequences

### Positive

1. ✅ **Structure Improvement**: Target 25% → 60%+ structure score
2. ✅ **Modular Design**: Edge refinement isolated in dedicated subsystem
3. ✅ **Backward Compatibility**: Disabled by default, no breaking changes
4. ✅ **Testability**: Each algorithm has dedicated unit tests
5. ✅ **Golden Path Aligned**: Integrates cleanly with lux_depth_v2 architecture

### Negative

1. ⚠️ **Processing Overhead**: Edge refinement adds 10-20% latency per image
2. ⚠️ **Dependency Risk**: Guided filter requires opencv-contrib (fallback to bilateral)
3. ⚠️ **Configuration Complexity**: Refinement config adds 6 new parameters
4. ⚠️ **Input Size Tradeoff**: Larger input sizes increase memory usage (518px: ~200MB, 1022px: ~800MB)

### Neutral

1. 📊 **Validation Required**: Structure score improvement must be validated on production dataset
2. 📊 **Parameter Tuning**: Refinement parameters may require per-preset calibration
3. 📊 **Performance Profiling**: Input size sweep must balance quality vs latency

---

## Implementation Roadmap

### Phase 1: Design & Infrastructure (Dec 20 - Jan 10) - **FEATURE FREEZE**

- [x] Write ADR (this document)
- [ ] Design API contracts for refinement algorithms
- [ ] Create test infrastructure (test harnesses, synthetic test data)
- [ ] Draft requirements-edge-refinement.txt
- [ ] Prepare validation datasets (10-20 architectural renders)

### Phase 2: Module Scaffolding (Jan 10 - Jan 13)

- [ ] Create `lux_depth_v2/refinement/` directory structure
- [ ] Implement configuration classes (RefinementConfig)
- [ ] Write unit test skeletons for each algorithm
- [ ] Install dependencies (opencv-contrib, scipy)

### Phase 3: Algorithm Implementation (Jan 13 - Jan 24)

- [ ] Implement bilateral_filter.py (Jan 13-14)
- [ ] Implement guided_filter.py (Jan 15-16)
- [ ] Implement edge_enhancer.py (Jan 17-18)
- [ ] Implement structure_scorer.py (Jan 19-20)
- [ ] Write comprehensive unit tests (Jan 21-24)

### Phase 4: Integration (Jan 24 - Jan 27)

- [ ] Integrate refinement hook into pipeline.py
- [ ] Extend PipelineConfig with refinement settings
- [ ] Add CLI flags (--enable-edge-refinement, --input-size)
- [ ] Create "interior_structure_enhanced" preset

### Phase 5: Validation (Jan 27 - Feb 7)

- [ ] Execute input size sweep experiment (Jan 27-28)
- [ ] Run structure score validation on production dataset (Jan 29-31)
- [ ] Analyze results and tune parameters (Feb 1-3)
- [ ] Generate validation report (Feb 4-7)

**Target**: Structure score improvement from 25% → 60%+ by Feb 7, 2026

---

## Alternatives Considered

### Alternative 1: Modify Existing Pipeline (Rejected)

**Approach**: Add edge refinement directly to existing lux_depth_v2 pipeline without dedicated module.

**Rejection Rationale**:
- Violates single responsibility principle
- Increases pipeline.py complexity
- Difficult to test in isolation
- Couples refinement logic with core processing

### Alternative 2: Separate Tool (Rejected)

**Approach**: Create standalone edge refinement tool (e.g., `lux-edge-refiner` CLI).

**Rejection Rationale**:
- Requires users to run multiple tools in sequence
- Increases cognitive load (anti-Golden Path)
- Duplicates depth estimation (inefficient)
- Harder to configure consistent pipelines

### Alternative 3: Post-Processing Plugin System (Future Consideration)

**Approach**: Generic plugin architecture for post-processing modules.

**Deferred Rationale**:
- Over-engineering for single use case
- Increases architectural complexity
- Plugin discovery and loading adds overhead
- Can be implemented later if more post-processing modules emerge

**Decision**: Start with dedicated refinement module, refactor to plugin system if 3+ post-processing modules are needed.

---

## References

- **Golden Path Architecture**: `/docs/DECISION_GUIDE.md`
- **Lux Depth V2 Documentation**: `/lux_depth_v2/README.md`
- **Structure Validation Metrics**: `/docs/validation/structure_metrics.md` (to be created)
- **Bilateral Filtering**: Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images" (1998)
- **Guided Filter**: He et al., "Guided Image Filtering" (2013)

---

## Approval

**Status**: ✅ Approved for implementation (after feature freeze lifts on Jan 10, 2026)

**Architect Sign-Off**: This design respects the Golden Path architecture, maintains backward compatibility, and provides a clear path to 60%+ structure score improvement.

**Next Steps**:
1. Review this ADR during design review (Jan 9, 2026)
2. Begin implementation on Jan 10, 2026
3. Complete Phase 2-5 according to roadmap
