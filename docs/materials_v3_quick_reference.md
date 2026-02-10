# Materials V3 Enhancements - Quick Reference

## New Pixel Operations (Phase 1)

### Water - `reflection_enhance`
**Purpose:** Enhance reflections and clarity for water surfaces (pools, lakes, ocean)

**Usage:**
```python
from transformation_portal.lux_depth_v3.pixel_ops_registry import OP_REGISTRY

water_op = OP_REGISTRY["water"]["reflection_enhance"]
result = water_op.op(image, mask, {"strength": 0.10})
```

**Parameters:**
- `strength`: Enhancement strength (default: 0.10)
  - Range: 0.0-0.3 (0.10 = subtle, 0.20 = moderate, 0.30 = strong)
  - Recommendation: Keep ≤0.15 for natural appearance

**Effect:**
- Combines brightness boost + contrast enhancement
- Subtle enhancement to maintain natural water appearance
- Enhances reflectivity typical of water surfaces

---

### Foliage - `vibrance_boost`
**Purpose:** Boost green channel vibrance for vegetation

**Usage:**
```python
foliage_op = OP_REGISTRY["foliage"]["vibrance_boost"]
result = foliage_op.op(image, mask, {"strength": 0.08})
```

**Parameters:**
- `strength`: Enhancement strength (default: 0.08)
  - Range: 0.0-0.15 (0.08 = subtle, 0.12 = moderate, 0.15 = maximum)
  - Recommendation: Keep ≤0.10 to avoid oversaturation

**Effect:**
- Selectively enhances green channel
- Makes vegetation more vibrant
- Preserves natural color balance

---

## Material Mask Exposure (Phase 2)

### Feature
Materials V3 now exposes material segmentation masks to V2 enhancement pipeline.

### Usage

**Access masks from Materials V3 result:**
```python
from transformation_portal.lux_depth_v3.materials_v3 import MaterialsV3Engine

engine = MaterialsV3Engine(config)
result = engine.process(image, segmentation_result, depth_map)

# Masks are now available
material_masks = result["material_masks"]
# Example: {"glass": mask_array, "water": mask_array, ...}
```

**Masks format:**
```python
material_masks: Dict[str, np.ndarray]
# Each mask is (H, W) float32 array with values 0.0-1.0
```

### Integration Status

**Current:**
- ✅ Masks captured in Materials V3 result
- ✅ Passed to V2Runner (with logging)
- ⚠️ Not serialized to V2 subprocess (requires disk serialization)

**Future:**
- Serialize masks to temporary directory
- Pass mask path to V2 subprocess
- Enable material-aware V2 enhancement

**Workaround (In-Process V2):**
```python
from transformation_portal.lux_depth_v3.v2_enhance import enhance_image

# When using V2 enhancement in-process (not subprocess):
enhance_image(
    input_path=input_path,
    output_path=output_path,
    depth_map_path=depth_path,
    material_masks=material_masks,  # ✅ Works in-process
    config=v2_config,
)
```

---

## Segmentation Backend (Phase 3)

### Configuration

**Enable segmentation:**
```python
from transformation_portal.lux_depth_v3.config import EnhanceConfig

config = EnhanceConfig(
    enable_materials_v3=True,
    enable_material_segmentation=True,  # NEW
    material_segmentation_backend="stub",  # Options: stub, efficientam
)
```

### Backend Options

#### Stub Backend (Default)
```python
config.material_segmentation_backend = "stub"
```
- Returns empty masks `{}`
- Zero overhead
- For testing/development when no segmentation needed

#### EfficientSAM Backend (Future)
```python
config.material_segmentation_backend = "efficientam"
```
- Not yet implemented
- Falls back to stub with warning
- Future: Automatic material segmentation using EfficientSAM

### Usage

**Programmatic:**
```python
from transformation_portal.lux_depth_v3.segmentation_backend import segment_materials

masks = segment_materials(image, config)
# Returns: Dict[str, np.ndarray] mapping material names to masks
```

**CLI (when available):**
```bash
python -m transformation_portal.lux_depth_v3 enhance \
  --enable-materials-v3 \
  --enable-material-segmentation \
  --material-segmentation-backend efficientam \
  input.jpg output/
```

---

## Complete Example

### Process Image with All New Features

```python
import numpy as np
from pathlib import Path
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Configure pipeline
config = EnhanceConfig(
    enable_materials_v3=True,
    apply_pixel_ops=True,
    enable_material_segmentation=True,
    material_segmentation_backend="stub",

    # Materials V3 settings
    refinement_strategy="canary",  # Apply water + foliage ops
    min_coverage_px=500,
    min_mean_conf=0.2,
)

# Create orchestrator
output_dir = Path("output")
orchestrator = EnhanceOrchestrator(config, output_dir)

# Process image
result = orchestrator.enhance_image("input.jpg")

print(f"Status: {result['status']}")
print(f"Manifest: {result['manifest']}")
```

### Check Which Ops Were Applied

```python
from transformation_portal.lux_depth_v3.manifest import CombinedManifest

manifest = CombinedManifest.load(result['manifest'])

if manifest.materials_v3:
    pixel_ops = manifest.materials_v3.pixel_ops
    print(f"Ops applied: {pixel_ops['applied']}")
    # Example: ["glass.brightness_boost", "water.reflection_enhance"]
```

---

## Canary Set Summary

**Current Status:**
```
✅ glass       → brightness_boost, edge_contrast
✅ stone       → microcontrast
✅ water       → reflection_enhance (NEW)
✅ foliage     → vibrance_boost (NEW)
⏳ wood        → (future)
⏳ metal       → (future)
⏳ fabric      → (future)
⏳ stucco      → (future)
```

**Automatic Application:**
When Materials V3 is enabled with `refinement_strategy="canary"`, pixel operations are automatically applied to detected materials in the canary set (glass, stone, water, foliage).

---

## Testing

### Test Water Operations
```python
import numpy as np
from transformation_portal.lux_depth_v3.pixel_ops_registry import water_reflection_enhance

# Create test image and mask
image = np.ones((256, 256, 3), dtype=np.uint8) * 128
mask = np.zeros((256, 256), dtype=np.float32)
mask[50:200, 50:200] = 0.8  # Water region

# Apply enhancement
result = water_reflection_enhance(image, mask, {"strength": 0.10})

# Check result
assert result.shape == image.shape
assert result.dtype == image.dtype
```

### Test Foliage Operations
```python
from transformation_portal.lux_depth_v3.pixel_ops_registry import foliage_vibrance_boost

# Create green-ish test image
image = np.ones((256, 256, 3), dtype=np.uint8)
image[..., 0] = 50   # Red
image[..., 1] = 120  # Green
image[..., 2] = 60   # Blue

mask = np.ones((256, 256), dtype=np.float32)

# Apply enhancement
result = foliage_vibrance_boost(image, mask, {"strength": 0.08})

# Green channel should be enhanced
assert result[..., 1].mean() > image[..., 1].mean()
```

### Test Segmentation Backend
```python
from transformation_portal.lux_depth_v3.segmentation_backend import segment_materials
from transformation_portal.lux_depth_v3.config import EnhanceConfig

config = EnhanceConfig(
    enable_material_segmentation=True,
    material_segmentation_backend="stub"
)

image = np.ones((256, 256, 3), dtype=np.uint8) * 128
masks = segment_materials(image, config)

assert isinstance(masks, dict)
assert len(masks) == 0  # Stub returns empty
```

---

## Troubleshooting

### Issue: Water/Foliage ops not applied

**Check:**
1. Materials V3 enabled: `config.enable_materials_v3 = True`
2. Pixel ops enabled: `config.apply_pixel_ops = True`
3. Refinement strategy: `config.refinement_strategy = "canary"`
4. Material coverage: Mask must have ≥500 pixels (default `min_coverage_px`)
5. Confidence: Mean mask confidence ≥0.2 (default `min_mean_conf`)

**Debug:**
```python
# Check pixel ops telemetry
manifest = CombinedManifest.load(manifest_path)
if manifest.materials_v3:
    telemetry = manifest.materials_v3.pixel_ops
    print(f"Applied: {telemetry['applied']}")
    print(f"Blocked: {telemetry['blocked']}")
```

### Issue: Material masks not in V2

**Expected Behavior:**
- Masks are logged but not passed to V2 subprocess
- This is documented as "future work" (requires serialization)

**Workaround:**
Use in-process V2 enhancement instead of subprocess:
```python
from transformation_portal.lux_depth_v3.v2_enhance import enhance_image
# Pass material_masks directly
```

### Issue: EfficientSAM backend warning

**Expected:**
```
WARNING: EfficientSAM backend not yet implemented. Falling back to stub.
```

**Resolution:**
This is expected. EfficientSAM integration is future work. Use `backend="stub"` for now.

---

## Performance Notes

**Water/Foliage Ops:**
- Overhead: ~1-2ms per operation
- Only runs when material detected and coverage threshold met
- No impact when materials not present

**Material Masks:**
- Zero overhead (data passing only)
- Logging: <1ms

**Segmentation Backend (Stub):**
- Zero overhead (immediate return)
- EfficientSAM (future): Expected ~100-200ms per image

**Overall:** No measurable performance regression

---

## Version Information

**Materials V3 Version:** 3.1
**Schema Version:** 1.0
**Implementation Date:** 2025-02-10

**Changes in 3.1:**
- Added water pixel operations
- Added foliage pixel operations
- Exposed material masks to V2 enhancement
- Added configurable segmentation backend
- Maintained 100% backward compatibility

---

## Next Steps

1. **Try New Ops:** Test water and foliage enhancements on real images
2. **Monitor Masks:** Check V2Runner logs for material mask availability
3. **Plan EfficientSAM:** Review future integration requirements
4. **Expand Ops:** Implement wood, metal, fabric, stucco operations

For questions or issues, see: `docs/materials_v3_enhancements_summary.md`
