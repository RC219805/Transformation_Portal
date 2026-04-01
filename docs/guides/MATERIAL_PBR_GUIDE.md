# Material PBR Integration Guide

**Version:** 5.1.0
**Status:** Stable
**Last Updated:** 2026-04-01

---

## Overview

The Material PBR Integration provides production-ready physics-based rendering (PBR) texture generation for luxury real estate post-production. This system generates 6 PBR texture maps (albedo, normal, roughness, metallic, ambient occlusion, height) from input RGB images with optional depth and material hints.

### Key Features

- ✅ **Deterministic CPU baseline** - Always available, zero ML dependencies
- ✅ **Optional GPU acceleration** - PBRFusion backend with auto-fallback
- ✅ **Depth-aware processing** - Enhanced normals and AO using geometry
- ✅ **Material intelligence** - 8 PBR-accurate material presets
- ✅ **Production performance** - 4.28s/MP @ 12MP (meets <5s/MP target)
- ✅ **Commercial licensing** - Apache 2.0 (stable + canary)

---

## Quick Start

### 1. Basic Usage (Stable Preset)

```python
from transformation_portal.spatial_ai.materials import MaterialBackend

# Initialize backend (CPU-only, always works)
backend = MaterialBackend(backend="heuristic", device="cpu")

# Generate PBR textures
result = backend.generate_pbr_textures(rgb=your_image)

# Access textures
albedo = result.albedo              # [0,1] RGB diffuse color
normal = result.normal              # [-1,1] tangent-space normals
roughness = result.roughness        # [0,1] 0=smooth, 1=rough
metallic = result.metallic          # [0,1] 0=dielectric, 1=metal
ao = result.ambient_occlusion       # [0,1] 0=occluded, 1=lit
height = result.height              # [0,1] normalized displacement

# Access metadata (may be None for older/custom backends)
metadata = result.metadata
if metadata is not None:
    print(f"Backend: {metadata.backend}")
    print(f"Normal scale: {metadata.normal_scale}")
else:
    print("Metadata not available for this backend")
```

### 2. With Material Hints (Enhanced Quality)

```python
# Specify material type for optimized PBR properties
result = backend.generate_pbr_textures(
    rgb=your_image,
    material_hint="wood"  # wood, stone, metal, glass, fabric, concrete, plastic, ceramic
)

# Material properties are automatically optimized
properties = result.properties
print(f"Roughness mean: {properties.roughness_mean:.3f}")
print(f"Metallic mean: {properties.metallic_mean:.3f}")
```

### 3. With Depth (Maximum Quality)

```python
# Provide depth map for geometry-aware processing
result = backend.generate_pbr_textures(
    rgb=your_image,
    depth=depth_map,        # Optional: normalized depth [0,1]
    material_hint="stone"
)

# Depth improves:
# - Normal map quality (5× depth-aware scale)
# - AO generation (concavity detection)
# - Height map accuracy
```

---

## Preset Selection

### Stable v5.1.0 (`material_pbr.yaml`)

**Use When:**
- Production deployments requiring determinism
- CPU-only environments (no GPU available)
- Guaranteed reproducibility needed
- Commercial licensing required

**Backend:** Heuristic (CPU-only, zero ML deps)
**Performance:** 4.28s/MP @ 12MP
**Stability:** SHA256-locked, CI-enforced immutability

```bash
python scripts/enhance_image.py \
  --input input.tif \
  --preset config/presets/material_pbr.yaml \
  --output output/
```

### Canary v5.0.0-canary (`material_pbr_canary.yaml`)

**Use When:**
- GPU acceleration available (CUDA/MPS)
- Higher quality requirements
- Experimental features acceptable
- Auto-fallback to CPU is acceptable

**Backend:** PBRFusion (GPU primary) + Heuristic (CPU fallback)
**Performance:** ~2-3s/MP @ 12MP on GPU (estimated)
**Note:** Requires `PBRFUSION_PATH` environment variable

```bash
export PBRFUSION_PATH=/path/to/pbrfusion
python scripts/enhance_image.py \
  --input input.tif \
  --preset config/presets/material_pbr_canary.yaml \
  --output output/
```

---

## Material Preset Reference

### PBR-Accurate Material Properties

| Material  | Roughness Range | Metallic Range | Use Case                          |
|-----------|-----------------|----------------|-----------------------------------|
| Metal     | 0.1 - 0.3       | 0.9 - 1.0      | Stainless steel, chrome fixtures  |
| Glass     | 0.02 - 0.1      | 0.0            | Windows, mirrors, gloss surfaces  |
| Wood      | 0.4 - 0.7       | 0.0            | Hardwood floors, cabinetry        |
| Stone     | 0.6 - 0.9       | 0.0            | Marble, granite, countertops      |
| Fabric    | 0.7 - 0.95      | 0.0            | Upholstery, textiles, rugs        |
| Concrete  | 0.8 - 0.95      | 0.0            | Walls, floors, architectural      |
| Plastic   | 0.3 - 0.6       | 0.0            | Modern fixtures, furniture        |
| Ceramic   | 0.2 - 0.4       | 0.0            | Tile, porcelain, bathroom         |

**Note:** Ranges are physically accurate for real-world materials in luxury real estate contexts.

---

## Integration with SAM2 Segmentation

### Full Pipeline Example

```python
from transformation_portal.spatial_ai.segmentation import MaterialClassifier, SAM2Segmenter
from transformation_portal.spatial_ai.materials import MaterialBackend

# 1. Segment image into zones
segmenter = SAM2Segmenter(model="sam2_hiera_large")
masks = segmenter.segment(image)

# 2. Classify materials per segment
classifier = MaterialClassifier()
material_labels = classifier.classify(image, masks)

# 3. Generate PBR textures per segment
backend = MaterialBackend(backend="heuristic", device="cpu")

for mask, material in zip(masks, material_labels):
    segment_rgb = image * mask[..., None]

    result = backend.generate_pbr_textures(
        rgb=segment_rgb,
        material_hint=material,
        depth=depth_map
    )

    # Use PBR textures for rendering...
```

---

## Performance Optimization

### Environment Configuration

**CPU (Default):**
```bash
# No configuration needed
python scripts/enhance_image.py --preset config/presets/material_pbr.yaml --input input.tif
```

**MPS (Apple Silicon):**
```bash
# Auto-detected if available
python scripts/enhance_image.py --preset config/presets/material_pbr.yaml --input input.tif
```

**CUDA (NVIDIA):**
```bash
# Auto-detected if available
CUDA_VISIBLE_DEVICES=0 python scripts/enhance_image.py \
  --preset config/presets/material_pbr_canary.yaml \
  --input input.tif
```

### Batch Processing

```python
from pathlib import Path
from transformation_portal.spatial_ai.materials import MaterialBackend

backend = MaterialBackend(backend="heuristic", device="cpu")

for input_path in Path("input/").glob("*.tif"):
    rgb = load_tiff(input_path)
    result = backend.generate_pbr_textures(rgb)

    # Save PBR maps...
    save_tiff(output_path / f"{input_path.stem}_albedo.tif", result.albedo)
    save_tiff(output_path / f"{input_path.stem}_normal.tif", result.normal)
    # ... etc
```

---

## Troubleshooting

### Issue: "PBRFusion backend not available"

**Solution:** Canary preset requires PBRFusion installation:

```bash
# Option 1: Set PBRFUSION_PATH
export PBRFUSION_PATH=/path/to/pbrfusion

# Option 2: Use stable preset (CPU-only)
python scripts/enhance_image.py --preset config/presets/material_pbr.yaml --input input.tif
```

**Note:** System auto-falls back to heuristic if PBRFusion unavailable.

### Issue: Low-quality normal maps

**Causes:**
1. Missing depth map (provide `depth=` for best results)
2. Low-resolution input (<1MP)
3. Flat surfaces (normals require texture variation)

**Solutions:**
- Provide depth map from Depth Anything V3 or Depth Pro
- Increase `normal_scale` in preset (default: 5.0)
- Enable `bilateral_filtering` for noise reduction

### Issue: Overly dark ambient occlusion

**Causes:**
1. Deep concavities (expected behavior)
2. High-contrast surfaces
3. Material preset mismatch

**Solutions:**
- Adjust `ao_blend_ratio` in preset (default: 0.7 concavity, 0.3 variance)
- Use correct material hint (`glass` has minimal AO)
- Reduce AO strength in post-processing

---

## API Reference

### `MaterialBackend.generate_pbr_textures()`

```python
def generate_pbr_textures(
    rgb: np.ndarray,                    # [H,W,3] RGB image, gamma=1.0, [0,1] float32
    depth: Optional[np.ndarray] = None, # [H,W] depth map, normalized [0,1]
    material_hint: Optional[str] = None # "wood", "stone", "metal", etc.
) -> PBRTextures:
    """
    Generate PBR texture maps from RGB image.

    Returns:
        PBRTextures with:
        - albedo: [H,W,3] diffuse color
        - normal: [H,W,3] tangent-space normals
        - roughness: [H,W] surface roughness
        - metallic: [H,W] metalness
        - ambient_occlusion: [H,W] occlusion
        - height: [H,W] displacement
        - properties: MaterialProperties metadata
        - metadata: PBRGenerationMetadata (backend, params)
    """
```

### `PBRTextures` Dataclass

```python
@dataclass
class PBRTextures:
    albedo: np.ndarray              # [H,W,3] float32 [0,1]
    normal: np.ndarray              # [H,W,3] float32 [-1,1]
    roughness: np.ndarray           # [H,W] float32 [0,1]
    metallic: np.ndarray            # [H,W] float32 [0,1]
    ambient_occlusion: np.ndarray   # [H,W] float32 [0,1]
    height: np.ndarray              # [H,W] float32 [0,1]
    properties: MaterialProperties
    metadata: Optional[PBRGenerationMetadata] = None
```

---

## Quality Firewall Thresholds

**Stable Preset (`material_pbr.yaml`):**

| Metric                  | Threshold      | Status  |
|-------------------------|----------------|---------|
| Mean latency (s/MP)     | < 5.0          | ✅ 4.28 |
| P95 latency (s/MP)      | < 7.0          | ✅ 5.32 |
| Memory (MB)             | < 1000         | ✅ <500 |
| Determinism             | Bitwise        | ✅      |
| Value ranges validated  | 100%           | ✅      |

**Measured on:** Python 3.11.14, NumPy 1.26.4, macOS arm64 (M3 Pro)

---

## Best Practices

### 1. Always Provide Material Hints When Known

```python
# ✅ Good: Explicit material for accurate PBR properties
result = backend.generate_pbr_textures(rgb, material_hint="wood")

# ❌ Suboptimal: Generic defaults used
result = backend.generate_pbr_textures(rgb)
```

### 2. Use Depth Maps for Architectural Scenes

```python
# ✅ Good: Depth improves normals + AO quality
result = backend.generate_pbr_textures(rgb, depth=depth_map)

# ✅ Acceptable: Works without depth, lower quality
result = backend.generate_pbr_textures(rgb)
```

### 3. Validate Output Ranges

```python
result = backend.generate_pbr_textures(rgb)

# Verify PBR contract
assert result.albedo.min() >= 0 and result.albedo.max() <= 1
assert result.normal.min() >= -1 and result.normal.max() <= 1
assert result.roughness.min() >= 0 and result.roughness.max() <= 1
```

### 4. Cache Backend Instances for Batch Processing

```python
# ✅ Good: Reuse backend (avoids re-initialization)
backend = MaterialBackend(backend="heuristic", device="cpu")
for image in images:
    result = backend.generate_pbr_textures(image)

# ❌ Inefficient: Creates new backend per image
for image in images:
    backend = MaterialBackend(backend="heuristic", device="cpu")
    result = backend.generate_pbr_textures(image)
```

---

## Roadmap

### Phase 6: Gaussian Splatting Integration (Planned)

- Integration with 3D Gaussian Splatting (3DGS)
- SuGaR surface reconstruction
- Multi-view PBR consistency
- Real-time rendering pipeline

### Future Enhancements

- Anisotropic roughness support
- Subsurface scattering for translucent materials
- Clearcoat layers for automotive/luxury finishes
- HDR environment map integration

---

## Support & Resources

- **Architecture Docs:** `docs/architecture/ADR-027-material-classification.md`
- **Performance Baselines:** `docs/performance/PHASE5_PBR_BASELINES.md`
- **Migration Guide:** `docs/guides/MATERIAL_PBR_MIGRATION.md`
- **Contract Reference:** `src/transformation_portal/spatial_ai/materials/contracts.py`
- **Protocol Spec:** `src/transformation_portal/spatial_ai/materials/protocol.py`

---

## License

**Stable Preset:** Apache 2.0 (commercial use allowed)
**Canary Preset:** Apache 2.0 (PBRFusion model)

All heuristic backend code: Apache 2.0
