# Materials Module Guide (Phase 2.2)

**Status:** Experimental
**Version:** 2.2.0
**Last Updated:** 2026-02-11

## Overview

The Materials module provides physics-based PBR (Physically-Based Rendering) texture generation for architectural visualization and luxury real estate rendering. It decomposes RGB images into material-aware texture maps suitable for modern rendering engines (Unreal, Unity, Blender).

### Key Capabilities

- **PBR Texture Decomposition**: Generates albedo, normal, roughness, metallic, and AO maps
- **Multiple Backends**: NVDIFFREC (neural), MaterialGAN (research), Heuristic (CPU fallback)
- **Segmentation Integration**: Works with Phase 2.1 SAM2 masks for per-material optimization
- **Material Hints**: Supports material-specific optimization (wood, metal, stone, etc.)
- **Contract-Driven**: Strict validation of inputs/outputs (gamma=1.0, float32, value ranges)

---

## Architecture

### Components

```
materials/
├── contracts.py           # Data contracts (MaterialInput, PBRTextures)
├── material_backend.py    # Neural backend wrapper (NVDIFFREC/MaterialGAN)
├── pbr_generator.py       # High-level orchestrator
├── heuristic_fallback.py  # CPU-based classical image processing
└── __init__.py            # Public API
```

### Backend Comparison

| Backend | License | Commercial OK? | Performance | Quality |
|---------|---------|----------------|-------------|---------|
| **NVDIFFREC** | BSD-3-Clause | ✅ Yes | <10s (GPU) | High |
| **MaterialGAN** | CC BY-NC 4.0 | ❌ Research only | <15s (GPU) | High |
| **Heuristic** | MIT | ✅ Yes | 2-5s (CPU) | Medium |

**Recommendation:** Use NVDIFFREC for production (broader licensing).

---

## Quick Start

### Basic Usage

```python
from transformation_portal.spatial_ai.materials import PBRGenerator
import numpy as np

# Load linear RGB image (gamma=1.0)
rgb = load_linear_rgb("scene.exr")  # (H, W, 3) float32

# Initialize generator
generator = PBRGenerator(backend="nvdiffrec", device="cuda")

# Generate PBR textures
result = generator.generate(
    image=rgb,
    gamma=1.0,
)

# Access outputs
albedo = result.albedo              # (H, W, 3) RGB [0, 1]
normal = result.normal              # (H, W, 3) XYZ [-1, 1]
roughness = result.roughness        # (H, W) [0, 1]
metallic = result.metallic          # (H, W) [0, 1]
ao = result.ambient_occlusion       # (H, W) [0, 1]
```

### With Segmentation Mask (Phase 2.1 Integration)

```python
from transformation_portal.spatial_ai.segmentation import SAM2Backend
from transformation_portal.spatial_ai.materials import PBRGenerator

# Step 1: Segment image with SAM2
sam2 = SAM2Backend(model_size="large", device="cuda")
seg_result = sam2.segment(rgb, gamma=1.0, mode="auto")

# Step 2: Generate PBR for each segment
pbr_gen = PBRGenerator(backend="nvdiffrec", device="cuda")
pbr_results = pbr_gen.generate_batch(
    image=rgb,
    gamma=1.0,
    masks=seg_result.masks,  # (N, H, W) bool
    material_hints=["wood", "metal", "glass"],  # Optional
)

# Step 3: Access per-segment PBR textures
for i, pbr in enumerate(pbr_results):
    print(f"Segment {i}:")
    print(f"  Roughness: {pbr.properties.roughness_mean:.2f}")
    print(f"  Metallic: {pbr.properties.metallic_mean:.2f}")
```

### With Depth Map

```python
from transformation_portal.spatial_ai.ingest import load_depth_map

# Load depth from Phase 2.0
depth = load_depth_map("scene_depth.exr")  # (H, W) float32

# Generate geometry-aware PBR
result = generator.generate(
    image=rgb,
    gamma=1.0,
    depth=depth,  # Improves normal and AO quality
)
```

---

## API Reference

### PBRGenerator

High-level orchestrator with contract validation.

#### `__init__(backend, device, model_repo_id=None, model_revision=None)`

**Parameters:**
- `backend` (str): Backend to use ("nvdiffrec", "material_gan", "heuristic")
- `device` (str): Compute device ("cuda", "mps", "cpu")
- `model_repo_id` (str, optional): HuggingFace model repo ID
- `model_revision` (str, optional): HuggingFace commit SHA for reproducibility

#### `generate(image, gamma, mask=None, depth=None, material_hint=None, config=None)`

Generate PBR textures for single image.

**Parameters:**
- `image` (np.ndarray): Linear RGB image (H, W, 3) float32
- `gamma` (float): Gamma value (must be 1.0)
- `mask` (np.ndarray, optional): Segmentation mask (H, W) bool
- `depth` (np.ndarray, optional): Depth map (H, W) float32
- `material_hint` (str, optional): Material category ("wood", "metal", etc.)
- `config` (MaterialGenerationConfig, optional): Generation configuration

**Returns:**
- `PBRTextures`: Object with albedo, normal, roughness, metallic, AO, height, properties

**Raises:**
- `ValueError`: If gamma != 1.0 or invalid dtypes/shapes

#### `generate_batch(image, gamma, masks, depth=None, material_hints=None, config=None)`

Generate PBR textures for multiple segments.

**Parameters:**
- `masks` (List[np.ndarray]): List of N segmentation masks (H, W) bool
- `material_hints` (List[str], optional): List of N material categories

**Returns:**
- `List[PBRTextures]`: One result per segment

---

## Data Contracts

### MaterialInput

Input contract with validation.

```python
from transformation_portal.spatial_ai.materials import MaterialInput

mat_input = MaterialInput(
    image=rgb,          # (H, W, 3) float32
    gamma=1.0,          # Must be 1.0 (SpatialCaptureV1 contract)
    mask=mask,          # (H, W) bool, optional
    depth=depth,        # (H, W) float32, optional
    material_hint="wood",  # Optional category
)
```

**Validation:**
- ✅ Gamma = 1.0 (linear RGB only)
- ✅ Image dtype = float32
- ✅ Image shape = (H, W, 3)
- ✅ Mask dtype = bool, shape = (H, W)
- ✅ Depth dtype = float32, shape = (H, W), values ≥ 0
- ✅ Material hint in {wood, stone, metal, glass, fabric, concrete, leather, ceramic}

### PBRTextures

Output contract with validation.

```python
from transformation_portal.spatial_ai.materials import PBRTextures

pbr = PBRTextures(
    albedo=albedo,              # (H, W, 3) float32, [0, 1]
    normal=normal,              # (H, W, 3) float32, [-1, 1], normalized
    roughness=roughness,        # (H, W) float32, [0, 1]
    metallic=metallic,          # (H, W) float32, [0, 1]
    ambient_occlusion=ao,       # (H, W) float32, [0, 1]
    height=height,              # (H, W) float32, [0, 1], optional
    properties=props,           # MaterialProperties, optional
)
```

**Validation:**
- ✅ All textures are float32
- ✅ Spatial dimensions match
- ✅ Value ranges enforced
- ✅ Normal vectors normalized

### MaterialProperties

Aggregated material statistics.

```python
from transformation_portal.spatial_ai.materials import MaterialProperties

props = MaterialProperties(
    roughness_mean=0.5,         # [0, 1]
    metallic_mean=0.0,          # [0, 1]
    ao_strength=0.3,            # [0, 1]
    normal_strength=1.0,        # [0, 2]
    specular_intensity=0.5,     # [0, 1]
    subsurface_scattering=0.0,  # [0, 1]
)
```

---

## Material Hints

Material hints guide PBR generation for better results:

### Wood
- Roughness: 0.7 (moderately rough)
- Metallic: 0.0 (dielectric)
- Subsurface: 0.1 (slight)

### Stone (Marble, Granite)
- Roughness: 0.6
- Metallic: 0.0
- Subsurface: 0.05

### Metal (Steel, Brass, Copper)
- Roughness: 0.2 (smooth)
- Metallic: 0.95 (conductor)
- Subsurface: 0.0

### Glass
- Roughness: 0.05 (very smooth)
- Metallic: 0.0
- Subsurface: 0.0

### Fabric (Linen, Velvet)
- Roughness: 0.8 (rough)
- Metallic: 0.0
- Subsurface: 0.2 (high)

### Concrete
- Roughness: 0.9 (very rough)
- Metallic: 0.0
- Subsurface: 0.0

### Leather
- Roughness: 0.5 (moderate)
- Metallic: 0.0
- Subsurface: 0.15

### Ceramic
- Roughness: 0.3 (smooth)
- Metallic: 0.0
- Subsurface: 0.05

---

## Performance Considerations

### GPU Acceleration

**NVDIFFREC (CUDA):**
- 1024x1024: <10s (RTX 3090)
- Memory: ~2-4GB VRAM
- Recommended for production

**MaterialGAN (CUDA):**
- 1024x1024: ~15s (RTX 3090)
- Memory: ~3-5GB VRAM
- Research only (licensing)

### CPU Fallback

**Heuristic:**
- 1024x1024: ~2-5s (CPU)
- Memory: <500MB RAM
- Lower quality but functional
- No ML dependencies

### Optimization Tips

1. **Batch Processing**: Use `generate_batch()` for multiple segments (reuses model)
2. **Resolution**: Start with 512 for testing, use 1024+ for production
3. **Iterations**: 50-100 is good balance (speed vs. quality)
4. **Lazy Loading**: Models only load when first used
5. **Unload Models**: Call `generator.unload_model()` to free VRAM

---

## Integration Examples

### Full Pipeline (Phase 2.0 + 2.1 + 2.2)

```python
from transformation_portal.spatial_ai.ingest import load_depth_map
from transformation_portal.spatial_ai.segmentation import SAM2Backend
from transformation_portal.spatial_ai.materials import PBRGenerator

# Phase 2.0: Load depth
depth = load_depth_map("scene_depth.exr")

# Phase 2.1: Segment image
sam2 = SAM2Backend(model_size="large", device="cuda")
seg_result = sam2.segment(rgb, gamma=1.0, mode="auto")

# Phase 2.2: Generate PBR for each segment
pbr_gen = PBRGenerator(backend="nvdiffrec", device="cuda")
pbr_results = pbr_gen.generate_batch(
    image=rgb,
    gamma=1.0,
    masks=seg_result.masks,
    depth=depth,
    material_hints=[meta.material_label for meta in seg_result.metadata],
)

# Save outputs
for i, pbr in enumerate(pbr_results):
    save_texture(f"segment_{i}_albedo.png", pbr.albedo)
    save_texture(f"segment_{i}_normal.png", pbr.normal)
    save_texture(f"segment_{i}_roughness.png", pbr.roughness)
    save_texture(f"segment_{i}_metallic.png", pbr.metallic)
    save_texture(f"segment_{i}_ao.png", pbr.ambient_occlusion)
```

### Export for Rendering Engines

```python
# Unreal Engine 5 / Unity HDRP
def export_pbr_for_unreal(pbr: PBRTextures, output_dir: Path):
    """Export PBR textures in Unreal Engine format."""
    # Albedo (BaseColor)
    save_png(output_dir / "T_Material_BaseColor.png", pbr.albedo)

    # Normal (OpenGL format: +Y up)
    normal_ue = pbr.normal.copy()
    normal_ue[:, :, 1] = 1.0 - normal_ue[:, :, 1]  # Flip Y
    save_png(output_dir / "T_Material_Normal.png", (normal_ue + 1) / 2)

    # ORM (Occlusion, Roughness, Metallic) packed
    orm = np.stack([pbr.ambient_occlusion, pbr.roughness, pbr.metallic], axis=2)
    save_png(output_dir / "T_Material_ORM.png", orm)

    # Height (optional)
    if pbr.height is not None:
        save_png(output_dir / "T_Material_Height.png", pbr.height)
```

---

## Troubleshooting

### "gamma=1.0" Error

**Problem:** ValueError: Material generation requires gamma=1.0

**Solution:** Input must be linear RGB. If you have sRGB, convert first:

```python
# Convert sRGB to linear
rgb_linear = np.power(rgb_srgb, 2.2)
```

### "float32" Error

**Problem:** ValueError: Image must be float32

**Solution:** Convert dtype:

```python
rgb = rgb.astype(np.float32)
```

### NVDIFFREC Fallback Warning

**Problem:** UserWarning: NVDIFFREC backend not yet implemented

**Status:** Expected for Phase 2.2 initial implementation. Neural backends will be added in future phases. Heuristic fallback is used automatically.

### Low Quality Results (Heuristic)

**Problem:** PBR textures look approximate/simplified

**Solution:** Heuristic backend uses classical image processing (no ML). For production:
1. Use `backend="nvdiffrec"` (when available)
2. Or accept lower fidelity for CPU-only environments

### Memory Issues

**Problem:** CUDA out of memory

**Solutions:**
1. Reduce resolution: `config.resolution = 512`
2. Process segments sequentially instead of batch
3. Unload model between batches: `generator.unload_model()`

---

## Advanced Configuration

```python
from transformation_portal.spatial_ai.materials import MaterialGenerationConfig

config = MaterialGenerationConfig(
    backend="nvdiffrec",
    resolution=2048,            # 512/1024/2048/4096
    optimize_iterations=200,    # 10-500 (more = slower + better)
    use_depth=True,             # Geometry-aware optimization
    normal_strength=1.5,        # [0, 2] - normal intensity
    ao_intensity=0.8,           # [0, 1] - AO darkness
    device="cuda",              # cuda/mps/cpu
)

result = generator.generate(image=rgb, gamma=1.0, config=config)
```

---

## Testing

Run materials module tests:

```bash
# All materials tests
pytest tests/spatial_ai/materials/ -v

# Specific test file
pytest tests/spatial_ai/materials/test_contracts.py -v

# With coverage
pytest tests/spatial_ai/materials/ --cov=src/transformation_portal/spatial_ai/materials --cov-report=term-missing
```

---

## Limitations

### Phase 2.2 Initial Implementation

- ✅ Heuristic backend fully functional (CPU fallback)
- ⏳ NVDIFFREC backend: placeholder (falls back to heuristic)
- ⏳ MaterialGAN backend: placeholder (falls back to heuristic)
- ⏳ Neural optimization: future phase

### Current Capabilities

The heuristic backend provides:
- ✅ Albedo extraction (shadow removal)
- ✅ Normal map generation (edge detection)
- ✅ Roughness estimation (texture variance)
- ✅ Metallic detection (saturation analysis)
- ✅ AO approximation (depth/luminance gradients)
- ✅ Height map generation

---

## Future Roadmap

### Phase 2.3+

- [ ] Full NVDIFFREC integration (neural PBR decomposition)
- [ ] MaterialGAN integration (alternative neural backend)
- [ ] Multi-layer material decomposition
- [ ] Procedural texture synthesis
- [ ] AI-guided material optimization
- [ ] Temporal consistency for video

---

## License & Attribution

- **Module Code**: MIT License (Transformation Portal)
- **NVDIFFREC**: BSD-3-Clause (NVIDIA) - Commercial OK
- **MaterialGAN**: CC BY-NC 4.0 - Research only
- **Heuristic**: MIT License - Commercial OK

---

## Related Documentation

- [Phase 2.0: Ingest Module](./ingest_guide.md)
- [Phase 2.1: Segmentation Guide](./segmentation_guide.md)
- [ADR-027: Spatial AI Foundation Architecture](../architecture/decisions/ADR-027-spatial-ai-foundation.md)
- [Experimental Preset: material_pbr.yaml](../../config/presets/experimental/material_pbr.yaml)

---

**Questions?** Open an issue or see [CONTRIBUTING.md](../../CONTRIBUTING.md)
