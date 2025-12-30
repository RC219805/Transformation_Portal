# Edge-Aware Depth Refinement - Usage Guide

**Status**: Infrastructure module (feature freeze until Jan 10, 2026)
**ADR**: [ADR-001-edge-refinement-module.md](../architecture/edge_refinement/ADR-001-edge-refinement-module.md)

## Overview

The edge refinement module provides edge-aware post-processing for depth maps to improve structural quality in architectural rendering pipelines. It implements 5 core refinement techniques:

1. **Bilateral Filtering** - Edge-preserving smoothing
2. **Guided Filter** - RGB-guided edge-aware smoothing
3. **Edge-Guided Enhancement** - Targeted structural detail enhancement
4. **Gradient Consistency Filtering** - Smooth away from edges, sharp at boundaries
5. **Segment-Aware Refinement** - Reduce cross-segment smoothing

## Installation

**Note**: Installation is prohibited until feature freeze lifts (Jan 10, 2026).

```bash
# After Jan 10, 2026:
pip install -r lux_depth_v2/requirements-edge-refinement.txt
```

Dependencies:
- `numpy>=1.23`
- `opencv-python>=4.8`
- `scipy>=1.9.0`

## Quick Start

### Basic Usage

```python
from lux_depth_v2.edge_refinement import refine_depth_edge_aware, RefinementPreset
import numpy as np

# Load your depth map and RGB image
depth = load_depth_map("render_depth.tif")  # HxW float32, normalized [0, 1]
rgb = load_rgb_image("render.jpg")          # HxWx3 uint8

# Apply edge-aware refinement with balanced preset
refined_depth = refine_depth_edge_aware(
    depth,
    rgb,
    preset=RefinementPreset.BALANCED
)

# Save refined depth
save_depth_map(refined_depth, "refined_depth.tif")
```

### Pipeline Integration

```python
from lux_depth_v2.edge_refinement import EdgeRefinementPipeline, EdgeRefinementConfig

# Create custom configuration
config = EdgeRefinementConfig(
    enable_bilateral=True,
    bilateral_sigma_color=75.0,
    bilateral_sigma_space=75.0,

    enable_guided=True,
    guided_radius=8,
    guided_eps=0.01,

    enable_edge_enhancement=True,
    edge_enhancement_strength=0.3,

    enable_gradient_smoothing=True,
    gradient_weight=0.5,

    structure_weight=0.5
)

# Initialize pipeline
pipeline = EdgeRefinementPipeline(config)

# Refine depth map
refined = pipeline.refine(depth, rgb)
```

### Preset-Based Configuration

Three curated presets are available:

```python
from lux_depth_v2.edge_refinement import RefinementPreset, EdgeRefinementConfig

# Subtle refinement (minimal processing, 15% enhancement)
subtle = EdgeRefinementConfig.from_preset(RefinementPreset.SUBTLE)

# Balanced refinement (recommended, 30% enhancement)
balanced = EdgeRefinementConfig.from_preset(RefinementPreset.BALANCED)

# Aggressive refinement (maximum enhancement, 50%)
aggressive = EdgeRefinementConfig.from_preset(RefinementPreset.AGGRESSIVE)
```

## Module Documentation

### Module 1: Bilateral Filtering

**Purpose**: Edge-preserving smoothing to reduce noise while maintaining structural boundaries.

```python
from lux_depth_v2.edge_refinement import bilateral_depth_filter

filtered = bilateral_depth_filter(
    depth_map,
    d=9,                    # Filter diameter (0 = auto-compute)
    sigma_color=75.0,       # Color similarity threshold (0-255)
    sigma_space=75.0        # Spatial extent in pixels
)
```

**Use Cases**:
- Remove depth estimation noise
- Smooth surfaces while preserving architectural edges
- Pre-processing for downstream modules

**Algorithm**: Tomasi & Manduchi (1998) - Bilateral Filtering for Gray and Color Images

---

### Module 2: Guided Filter

**Purpose**: Fast edge-aware filtering using RGB image as guidance to align depth edges with structural boundaries.

```python
from lux_depth_v2.edge_refinement import guided_filter_depth

filtered = guided_filter_depth(
    depth_map,
    rgb_image,
    radius=8,               # Filter radius in pixels
    eps=0.01                # Regularization (controls edge preservation)
)
```

**Use Cases**:
- Align depth edges with RGB structural boundaries
- Improve depth-image consistency
- Fast edge-aware smoothing (O(1) complexity)

**Algorithm**: He et al. (2013) - Guided Image Filtering

**Note**: Requires `opencv-contrib-python` for optimized implementation. Falls back to custom implementation if unavailable.

---

### Module 3: Edge-Guided Enhancement

**Purpose**: Targeted sharpening and contrast enhancement along structural edges.

```python
from lux_depth_v2.edge_refinement import enhance_edges_with_guidance

enhanced = enhance_edges_with_guidance(
    depth_map,
    rgb_image,
    strength=0.3,           # Enhancement intensity (0.0-1.0)
    threshold=40.0          # Edge detection sensitivity (0-255)
)
```

**Use Cases**:
- Sharpen architectural details (railings, moldings, window frames)
- Enhance structural edges detected from RGB
- Targeted unsharp masking along boundaries

**Algorithm**: Canny edge detection + selective unsharp masking

---

### Module 4: Gradient Consistency Filtering

**Purpose**: Enforce gradient smoothness away from edges while allowing sharp transitions at boundaries.

```python
from lux_depth_v2.edge_refinement import gradient_smoothness

smoothed = gradient_smoothness(
    depth_map,
    rgb_image,
    gradient_weight=0.5     # Weight for gradient alignment (0.0-1.0)
)
```

**Use Cases**:
- Reduce gradient noise in smooth regions
- Preserve sharp transitions at structural boundaries
- Improve depth-RGB gradient consistency

**Algorithm**: Sobel gradient analysis + edge-aware bilateral smoothing

---

### Module 5: Segment-Aware Refinement

**Purpose**: Refine depth within segments while reducing cross-segment smoothing.

```python
from lux_depth_v2.edge_refinement import segment_aware_refine

refined = segment_aware_refine(
    depth_map,
    segmentation_mask,      # Segment labels (HxW int32/uint8)
    filter_radius=5         # Radius for intra-segment smoothing
)
```

**Use Cases**:
- Material-based depth refinement
- Object-based smoothing
- Preserve sharp transitions between segments

**Algorithm**: Per-segment bilateral filtering with boundary preservation

---

## CLI Integration (Coming Jan 10, 2026)

Edge refinement will be integrated into the `lux-depth-v2` CLI with an opt-in flag:

```bash
# Process with edge refinement enabled (after Jan 10, 2026)
lux-depth-v2 \
    --input-dir renders/ \
    --output-dir refined/ \
    --preset interior_luxury \
    --enable-edge-refinement \
    --refinement-preset balanced
```

**CLI Flags**:
- `--enable-edge-refinement` - Enable edge-aware refinement (default: False)
- `--refinement-preset` - Refinement strength preset (subtle/balanced/aggressive)

## Configuration Reference

### EdgeRefinementConfig

```python
@dataclass
class EdgeRefinementConfig:
    # Bilateral filtering
    enable_bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Guided filter
    enable_guided: bool = True
    guided_radius: int = 8
    guided_eps: float = 0.01

    # Edge enhancement
    enable_edge_enhancement: bool = True
    edge_enhancement_strength: float = 0.3
    edge_detection_threshold: float = 40.0

    # Gradient smoothing
    enable_gradient_smoothing: bool = True
    gradient_weight: float = 0.5

    # Global settings
    structure_weight: float = 0.5
    max_image_dim: int = 4096  # Resource exhaustion prevention
```

### Preset Configurations

| Parameter | Subtle | Balanced | Aggressive |
|-----------|--------|----------|------------|
| `bilateral_sigma_color` | 50.0 | 75.0 | 100.0 |
| `bilateral_sigma_space` | 50.0 | 75.0 | 100.0 |
| `edge_enhancement_strength` | 0.15 | 0.3 | 0.5 |
| `gradient_weight` | 0.3 | 0.5 | 0.7 |
| `structure_weight` | 0.4 | 0.5 | 0.6 |

## Workflow Examples

### Example 1: Noise Reduction for Interior Rendering

```python
from lux_depth_v2.edge_refinement import EdgeRefinementConfig, EdgeRefinementPipeline

# Configure for noise reduction with edge preservation
config = EdgeRefinementConfig(
    enable_bilateral=True,
    bilateral_sigma_color=50.0,
    bilateral_sigma_space=50.0,

    enable_guided=True,
    guided_radius=8,

    enable_edge_enhancement=False,  # Disable sharpening
    enable_gradient_smoothing=True,
    gradient_weight=0.4
)

pipeline = EdgeRefinementPipeline(config)
refined = pipeline.refine(noisy_depth, rgb_image)
```

### Example 2: Structural Detail Enhancement

```python
# Configure for maximum structural detail preservation
config = EdgeRefinementConfig(
    enable_bilateral=False,  # Skip smoothing

    enable_guided=True,
    guided_radius=12,
    guided_eps=0.005,  # Stronger edge preservation

    enable_edge_enhancement=True,
    edge_enhancement_strength=0.5,  # Strong sharpening
    edge_detection_threshold=30.0,

    enable_gradient_smoothing=True,
    gradient_weight=0.7,  # High edge preservation

    structure_weight=0.7
)

pipeline = EdgeRefinementPipeline(config)
refined = pipeline.refine(depth, rgb_image)
```

### Example 3: Segment-Based Material Refinement

```python
from lux_depth_v2.edge_refinement import refine_depth_edge_aware, segment_aware_refine

# Step 1: Global edge-aware refinement
refined = refine_depth_edge_aware(
    depth,
    rgb,
    preset=RefinementPreset.BALANCED
)

# Step 2: Segment-aware refinement for materials
segmentation = load_material_segmentation("materials.png")  # Wood, metal, glass, etc.
final = segment_aware_refine(refined, segmentation, filter_radius=6)
```

## Performance Characteristics

**Throughput** (estimated, M4 Max CPU):
- Bilateral filter: ~50 images/hour (512x512)
- Guided filter: ~40 images/hour (512x512)
- Edge enhancement: ~60 images/hour (512x512)
- Full pipeline (all modules): ~25-30 images/hour (512x512)

**Memory Requirements**:
- 512x512 image: ~10-20 MB RAM
- 1024x1024 image: ~40-80 MB RAM
- 2048x2048 image: ~160-320 MB RAM

**Latency** (single image, 512x512):
- Bilateral filter: ~70-100ms
- Guided filter: ~90-120ms
- Edge enhancement: ~60-80ms
- Full pipeline: ~120-180ms

## Security Considerations

The edge refinement module includes security hardening:

- **CWE-703**: Input validation prevents invalid dimensions and types
- **CWE-834**: Resource exhaustion prevention via bounded kernel sizes and max image dimensions
- **Memory safety**: All operations use bounded buffers with proper clipping

**Maximum image dimension**: 4096x4096 (configurable via `max_image_dim`)

## Testing

Run the test suite:

```bash
# Run all edge refinement tests
pytest lux_depth_v2/tests/test_edge_refinement.py -v

# Run with coverage
pytest lux_depth_v2/tests/test_edge_refinement.py --cov=lux_depth_v2.edge_refinement

# Run specific test class
pytest lux_depth_v2/tests/test_edge_refinement.py::TestBilateralDepthFilter -v
```

**Test Coverage**: 15+ test cases covering:
- All 5 refinement modules
- Input validation and error handling
- Pipeline integration
- Configuration presets
- Integration workflows
- Security boundaries (CWE-703, CWE-834)

## Validation Metrics

Track structure improvement using these metrics:

```python
from lux_depth_v2.edge_refinement import EdgeRefinementPipeline
import cv2

# Process image
refined = pipeline.refine(depth, rgb)

# Metric 1: Edge sharpness (Laplacian variance)
def edge_sharpness(depth):
    laplacian = cv2.Laplacian(depth, cv2.CV_32F)
    return laplacian.var()

# Metric 2: Structure score (frequency content in structural regions)
def structure_score(depth):
    # FFT-based high-frequency content analysis
    # Implementation in lux_depth_v2/validation/structure_metrics.py
    pass

# Compare before/after
print(f"Edge sharpness improvement: {edge_sharpness(refined) / edge_sharpness(depth):.2f}x")
```

**Target Metrics** (from ADR-001):
- Structure score improvement: 25% → 60%+ (target: 2.4x)
- Edge sharpness: 10-30% improvement
- Processing overhead: <20% latency increase

## Troubleshooting

### Issue: Guided filter fallback warning

**Symptom**: Warning message "opencv-contrib not available, falling back to custom implementation"

**Solution**: Install opencv-contrib for optimized guided filter:
```bash
pip install opencv-contrib-python>=4.8
```

### Issue: Over-smoothing edges

**Symptom**: Architectural edges are too smooth after refinement

**Solution**:
1. Reduce `bilateral_sigma_color` (e.g., 50.0 → 30.0)
2. Decrease `guided_eps` for stronger edge preservation (e.g., 0.01 → 0.005)
3. Increase `structure_weight` (e.g., 0.5 → 0.7)
4. Use SUBTLE preset instead of BALANCED

### Issue: Insufficient noise reduction

**Symptom**: Depth map still noisy after refinement

**Solution**:
1. Increase `bilateral_sigma_color` (e.g., 75.0 → 100.0)
2. Increase `bilateral_sigma_space` (e.g., 75.0 → 100.0)
3. Use AGGRESSIVE preset
4. Apply bilateral filter multiple times

### Issue: Processing too slow

**Symptom**: Refinement takes >500ms per image

**Solution**:
1. Disable unused modules (e.g., `enable_gradient_smoothing=False`)
2. Reduce `bilateral_d` (e.g., 9 → 5)
3. Reduce `guided_radius` (e.g., 8 → 4)
4. Process at lower resolution and upscale

## References

- **ADR-001**: [Edge Refinement Module Architecture](../architecture/edge_refinement/ADR-001-edge-refinement-module.md)
- **Bilateral Filtering**: Tomasi & Manduchi (1998) - Bilateral Filtering for Gray and Color Images
- **Guided Filter**: He et al. (2013) - Guided Image Filtering
- **Lux Depth V2 Documentation**: [lux_depth_v2/README.md](../lux_depth_v2/README.md)

## Future Enhancements

Planned for post-freeze implementation (Feb 2026+):

1. **Structure quality metrics** - FFT-based structure score computation
2. **Auto-parameter tuning** - Automatic refinement parameter selection based on depth quality
3. **Multi-scale processing** - Pyramid-based refinement for large images
4. **GPU acceleration** - CUDA/MPS implementations for 3-5x speedup
5. **Temporal consistency** - Video depth refinement with temporal coherence

## Contact

For questions or issues, please:
1. Review ADR-001 for architectural context
2. Check test suite for usage examples
3. Consult repository documentation in `docs/`
4. File issues with `[edge-refinement]` tag after Jan 10, 2026
