# Scene Enhancement Quick Reference

**For:** Implementation Team  
**Phase 1 Tasks:** 3 actionable items, 9-14 hours total  
**Expected Impact:** 40-60% quality improvement  
**Priority:** P0 - Critical

---

## Task 1: Activate SegFormer-B5 Backend

**Effort:** 2-4 hours  
**Impact:** Confidence 9.9% → 42% (3-5x improvement)  
**Risk:** LOW

### Changes Required

**File:** `lux_depth_v2/config.py`

**Location 1: Preset Definitions (Line ~450)**

```python
# FIND (in apply_preset method):
if self.preset == Preset.INTERIOR_LUXURY_APEX_QUALITY:
    self.segmentation = SegmentationConfig(
        backend="auto",  # ❌ CHANGE THIS
        # ...
    )

# REPLACE WITH:
if self.preset == Preset.INTERIOR_LUXURY_APEX_QUALITY:
    self.segmentation = SegmentationConfig(
        backend="segformer",  # ✅ EXPLICIT ACTIVATION
        segformer_model="nvidia/segformer-b5-finetuned-ade-640-640",
        input_long_side=2048,  # Already optimal for APEX
        min_confidence=0.15,   # Already optimal
        allow_downloads=True,  # Already set
    )
```

**Location 2: Default SegmentationConfig (Line ~38)**

```python
# OPTIONAL: Change default from "auto" to "segformer"
@dataclass
class SegmentationConfig:
    backend: str = "segformer"  # Change from "auto"
    # ...
```

### Testing

```bash
# Re-run pool scene
lux-depth-v2 \
  --input input_images/750_Picacho/Source_TIFFs/750Picacho_Pool_16bit.tiff \
  --preset interior_luxury_apex_quality \
  --output-dir output_pool_segformer_validation

# Check report.json
cat output_pool_segformer_validation/*_report.json | \
  python3 -m json.tool | \
  grep -A10 "materials_v2_metadata"

# Expected results:
# "confidence_avg": 0.35-0.45  (was 0.099)
# "high_confidence_pct": 0.50-0.65  (was 0.14)
# "is_high_quality": true  (was false)
```

### Validation Criteria

- [ ] `confidence_avg` > 0.35
- [ ] `high_confidence_pct` > 0.50
- [ ] `is_high_quality` = true
- [ ] Processing time < 12s (pool scene)
- [ ] No VRAM errors on M4 Max

---

## Task 2: Material Property Schema

**Effort:** 4-6 hours  
**Impact:** Material response accuracy +30-40%  
**Risk:** LOW

### Implementation

**File:** `lux_depth_v2/material_profiles.py` (ADD NEW SECTION)

```python
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

@dataclass
class MaterialProperties:
    """Physics-based material properties for enhanced response."""
    
    # Surface characteristics
    roughness: float = 0.5           # 0=mirror, 1=diffuse
    specular: float = 0.1            # Highlight intensity
    metallic: float = 0.0            # 0=dielectric, 1=conductor
    albedo: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # Base color RGB
    
    # Optical properties
    transmission: float = 0.0        # Transparency
    ior: float = 1.5                # Index of refraction
    anisotropy: float = 0.0         # Directional reflections
    
    # Enhancement guidance
    clarity_boost: float = 1.0      # Multiplier for clarity
    saturation_mult: float = 1.0    # Saturation adjustment
    exposure_bias: float = 0.0      # Exposure offset
    contrast_mult: float = 1.0      # Contrast multiplier
    
    # Material response strength (per-material override)
    response_strength: float = 1.0  # 0=disabled, 1=full

# Material database
MATERIAL_PROPERTIES: Dict[str, MaterialProperties] = {
    # Architecture
    "stucco": MaterialProperties(
        roughness=0.85,
        specular=0.05,
        albedo=(0.95, 0.95, 0.92),
        clarity_boost=0.8,
        saturation_mult=0.98,
        response_strength=0.85,
    ),
    
    "concrete": MaterialProperties(
        roughness=0.75,
        specular=0.15,
        albedo=(0.6, 0.6, 0.6),
        clarity_boost=0.9,
        response_strength=0.9,
    ),
    
    # Water features
    "pool_tile_mosaic": MaterialProperties(
        roughness=0.15,
        specular=0.7,
        albedo=(0.15, 0.35, 0.55),
        clarity_boost=1.3,
        saturation_mult=1.15,
        response_strength=1.2,
    ),
    
    "pool_water_surface": MaterialProperties(
        roughness=0.05,
        specular=0.9,
        transmission=0.85,
        ior=1.33,
        clarity_boost=1.5,
        saturation_mult=1.08,
        response_strength=1.1,
    ),
    
    # Vegetation
    "vegetation_trees": MaterialProperties(
        roughness=0.7,
        albedo=(0.15, 0.45, 0.18),
        clarity_boost=1.1,
        saturation_mult=1.08,
        response_strength=1.0,
    ),
    
    "vegetation_shrubs": MaterialProperties(
        roughness=0.75,
        albedo=(0.18, 0.48, 0.20),
        clarity_boost=1.05,
        saturation_mult=1.05,
        response_strength=0.95,
    ),
    
    # Existing materials (refined)
    "wood": MaterialProperties(
        roughness=0.6,
        specular=0.25,
        anisotropy=0.3,
        albedo=(0.45, 0.35, 0.25),
        clarity_boost=1.1,
        saturation_mult=1.05,
        response_strength=1.0,
    ),
    
    "metal": MaterialProperties(
        roughness=0.2,
        specular=0.8,
        metallic=1.0,
        clarity_boost=1.2,
        saturation_mult=0.95,
        response_strength=1.15,
    ),
    
    "glass": MaterialProperties(
        roughness=0.1,
        specular=0.85,
        transmission=0.9,
        ior=1.52,
        clarity_boost=1.4,
        saturation_mult=1.0,
        response_strength=1.05,
    ),
    
    "stone": MaterialProperties(
        roughness=0.65,
        specular=0.2,
        albedo=(0.55, 0.52, 0.48),
        clarity_boost=0.95,
        saturation_mult=1.02,
        response_strength=0.9,
    ),
    
    "sky": MaterialProperties(
        roughness=0.0,
        specular=0.0,
        clarity_boost=0.7,
        saturation_mult=1.05,
        exposure_bias=-0.02,
        response_strength=0.5,
    ),
    
    "foliage": MaterialProperties(  # Alias for vegetation_trees
        roughness=0.7,
        albedo=(0.15, 0.45, 0.18),
        clarity_boost=1.1,
        saturation_mult=1.08,
        response_strength=1.0,
    ),
}

def get_material_properties(material_type: str) -> MaterialProperties:
    """Get properties for material type, with fallback."""
    return MATERIAL_PROPERTIES.get(
        material_type,
        MaterialProperties()  # Default neutral properties
    )
```

### Integration Point

**File:** `lux_depth_v2/material_profiles.py` (MODIFY: apply_material_response)

```python
def apply_material_response(
    rgb: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    cfg,
    zone_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> MaterialMods:
    """Apply material-specific enhancements."""
    
    # ... existing code ...
    
    # NEW: Use material properties
    from .material_properties import get_material_properties
    
    for mat_name, mask in masks.items():
        props = get_material_properties(mat_name)
        
        # Apply property-guided adjustments
        # Scale strength by material-specific response_strength
        effective_strength = cfg.material_strength * props.response_strength
        
        # Clarity: boosted by material clarity_boost
        clarity_mult += mask * (props.clarity_boost - 1.0) * effective_strength
        
        # Saturation: adjusted by material saturation_mult
        sat_mult += mask * (props.saturation_mult - 1.0) * effective_strength
        
        # ... etc
```

### Testing

```bash
# Verify material properties are loaded
python3 -c "
from lux_depth_v2.material_profiles import MATERIAL_PROPERTIES, get_material_properties
print('Materials defined:', list(MATERIAL_PROPERTIES.keys()))
print('Stucco roughness:', get_material_properties('stucco').roughness)
print('Pool tile specular:', get_material_properties('pool_tile_mosaic').specular)
"

# Re-run pool scene and verify material response is stronger
lux-depth-v2 \
  --input input_images/750_Picacho/Source_TIFFs/750Picacho_Pool_16bit.tiff \
  --preset interior_luxury_apex_quality \
  --output-dir output_pool_material_properties_test
```

### Validation Criteria

- [ ] 12+ materials in `MATERIAL_PROPERTIES`
- [ ] Properties applied in `apply_material_response()`
- [ ] Pool water shows higher clarity than stucco
- [ ] No regression in processing time

---

## Task 3: Hybrid Depth Zones

**Effort:** 3-4 hours  
**Impact:** Depth-aware processing accuracy +20-30%  
**Risk:** LOW

### Implementation

**File:** `lux_depth_v2/pipeline.py`

**Location: Add new function before `_compute_zone_masks`**

```python
def _detect_scene_type(rgb: torch.Tensor, depth: np.ndarray) -> str:
    """Detect if scene is interior or exterior based on heuristics."""
    
    # Heuristic 1: Sky presence (top 20% of image, blue+bright)
    h, w = rgb.shape[2], rgb.shape[3]
    top_region = rgb[:, :, 0:int(h*0.2), :]
    
    r = top_region[:, 0]
    g = top_region[:, 1]
    b = top_region[:, 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # Sky signature: blue-dominant + bright
    is_blue = (b > r + 0.08) & (b > g + 0.05)
    is_bright = luma > 0.3
    sky_pct = (is_blue & is_bright).float().mean().item()
    
    # Heuristic 2: Depth range
    depth_range = depth.max() - depth.min()
    depth_90th_percentile = np.percentile(depth, 90)
    
    # Exterior: large depth range, sky present
    if sky_pct > 0.15 or depth_90th_percentile > 0.7:
        return "exterior"
    else:
        return "interior"

def _compute_zone_masks_hybrid(
    depth: np.ndarray,
    cfg,
    rgb: Optional[torch.Tensor] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute depth zones with scene-aware strategy.
    
    Args:
        depth: Normalized depth map [0, 1]
        cfg: Pipeline config with fg_q, bg_q
        rgb: Optional RGB image for scene detection
    
    Returns:
        (foreground, midground, background) masks
    """
    
    scene_type = "interior"
    if rgb is not None:
        scene_type = _detect_scene_type(rgb, depth)
    
    if scene_type == "exterior":
        # Metric-based zones for outdoor scenes
        # Assumes depth is normalized [0, 1] where 1=far
        
        # Foreground: 0-2m (assuming max depth ~50m → 0.04 normalized)
        fg_threshold = 0.04
        fg_mask = depth < fg_threshold
        
        # Background: >10m (0.20 normalized)
        bg_threshold = 0.20
        bg_mask = depth > bg_threshold
        
        # Midground: 2-10m
        mid_mask = ~(fg_mask | bg_mask)
        
    else:
        # Percentile-based for interior (compressed depth range)
        fg_percentile = cfg.fg_q * 100  # e.g., 35
        bg_percentile = cfg.bg_q * 100  # e.g., 65
        
        fg_threshold = np.percentile(depth, fg_percentile)
        bg_threshold = np.percentile(depth, bg_percentile)
        
        fg_mask = depth < fg_threshold
        bg_mask = depth > bg_threshold
        mid_mask = ~(fg_mask | bg_mask)
    
    return fg_mask, mid_mask, bg_mask
```

**Location: Modify `_compute_zone_masks` to use hybrid function**

```python
def _compute_zone_masks(
    depth: np.ndarray,
    cfg,
    manual_masks: Optional[Dict[str, np.ndarray]] = None,
    rgb: Optional[torch.Tensor] = None  # NEW parameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute zone masks from depth."""
    
    if manual_masks:
        # Use manual masks if provided
        return (
            manual_masks.get("foreground", np.ones_like(depth, dtype=bool)),
            manual_masks.get("midground", np.zeros_like(depth, dtype=bool)),
            manual_masks.get("background", np.zeros_like(depth, dtype=bool)),
        )
    
    # Use hybrid method (scene-aware)
    return _compute_zone_masks_hybrid(depth, cfg, rgb)
```

**Location: Update pipeline call (Line ~350 in `process_one`)**

```python
# FIND:
fg, mid, bg = _compute_zone_masks(depth_np, cfg, manual_masks=manual_zone_masks)

# REPLACE WITH:
fg, mid, bg = _compute_zone_masks(
    depth_np,
    cfg,
    manual_masks=manual_zone_masks,
    rgb=rgb_input  # Pass RGB for scene detection
)
```

### Testing

```bash
# Test exterior scene (pool)
lux-depth-v2 \
  --input input_images/750_Picacho/Source_TIFFs/750Picacho_Pool_16bit.tiff \
  --preset interior_luxury_apex_quality \
  --output-dir output_pool_hybrid_zones

# Test interior scene (kitchen)
lux-depth-v2 \
  --input input_images/750_Picacho/Source_TIFFs/750Picacho_Kitchen_16bit.tiff \
  --preset interior_luxury_apex_quality \
  --output-dir output_kitchen_hybrid_zones

# Verify scene detection
python3 -c "
import numpy as np
import torch
from lux_depth_v2.pipeline import _detect_scene_type
from lux_depth_v2 import io_utils

# Load pool image
rgb = io_utils.read_image_any('input_images/750_Picacho/Source_TIFFs/750Picacho_Pool_16bit.tiff')
rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
depth = np.random.rand(rgb.shape[0], rgb.shape[1])  # Dummy depth

scene = _detect_scene_type(rgb_torch, depth)
print(f'Pool scene type: {scene}')  # Expected: exterior

# Load kitchen image
rgb_k = io_utils.read_image_any('input_images/750_Picacho/Source_TIFFs/750Picacho_Kitchen_16bit.tiff')
rgb_k_torch = torch.from_numpy(rgb_k).permute(2, 0, 1).unsqueeze(0).float() / 255.0
depth_k = np.random.rand(rgb_k.shape[0], rgb_k.shape[1])

scene_k = _detect_scene_type(rgb_k_torch, depth_k)
print(f'Kitchen scene type: {scene_k}')  # Expected: interior
"
```

### Validation Criteria

- [ ] Pool scene detected as "exterior"
- [ ] Kitchen scene detected as "interior"
- [ ] Exterior uses metric thresholds (0.04, 0.20)
- [ ] Interior uses percentiles (35, 65)
- [ ] Atmospheric perspective more pronounced in pool scene

---

## Validation Protocol

### Phase 1 Complete Checklist

**Task 1: SegFormer Activation**
- [ ] `backend="segformer"` in APEX preset
- [ ] Pool confidence > 35%
- [ ] Kitchen confidence > 35%
- [ ] `is_high_quality = true` for both scenes

**Task 2: Material Properties**
- [ ] 12+ materials in `MATERIAL_PROPERTIES` dict
- [ ] Properties applied in `apply_material_response()`
- [ ] Pool water clarity > stucco clarity
- [ ] No processing time regression

**Task 3: Hybrid Depth Zones**
- [ ] Scene type detection implemented
- [ ] Pool detected as exterior
- [ ] Kitchen detected as interior
- [ ] Depth zones visually accurate

### Regression Testing

```bash
# Run full test suite
cd lux_depth_v2
pytest tests/ -v

# Run specific validation tests
pytest tests/test_segmentation.py -k segformer
pytest tests/test_material_properties.py
pytest tests/test_depth_zones.py

# Benchmark performance
make benchmark-lux-depth-v2

# Expected results:
# - Pool: 10-12s (was 9.7s, acceptable +6-23%)
# - Kitchen: 54-58s (was 53.4s, acceptable +1-9%)
# - Confidence: 35-45% (was 9.9%, target met)
```

### Visual Validation

**Pool Scene:**
1. Open `output_pool_segformer_validation/750Picacho_Pool_16bit_master16.tif`
2. Check pool water: should be vibrant, sharp reflections
3. Check stucco: should be matte, uniform, not over-sharpened
4. Check vegetation: should be saturated, detailed foliage
5. Compare to Phase 0 output: significant quality improvement visible

**Kitchen Scene:**
1. Open kitchen APEX output
2. Check wood cabinets: warm tones, anisotropic highlights
3. Check metal fixtures: high specular, sharp
4. Check stone countertops: textured, medium saturation

---

## Rollback Plan

If Phase 1 introduces regressions:

### Emergency Rollback (5 minutes)

```bash
# Revert config changes
git checkout lux_depth_v2/config.py
git checkout lux_depth_v2/material_profiles.py
git checkout lux_depth_v2/pipeline.py

# Rebuild and test
pip install -e .
make test-fast
```

### Partial Rollback

**Keep SegFormer, rollback others:**
```python
# config.py - Keep backend="segformer"
# Delete material_properties.py additions
# Revert pipeline.py depth zone changes
```

---

## Success Metrics

### Quantitative

- [ ] Pool confidence: 9.9% → 35%+ (3.5x improvement)
- [ ] Kitchen confidence: 16.6% → 35%+ (2.1x improvement)
- [ ] High-confidence coverage: 14% → 50%+
- [ ] Processing time increase: <20%
- [ ] Quality gate: FAILED → PASSED

### Qualitative

- [ ] Pool water boundaries are crisp and accurate
- [ ] Stucco walls are uniform (not over-processed)
- [ ] Vegetation detail is preserved
- [ ] Material-specific enhancements are visible
- [ ] Client deliverable quality: "Good" → "Exceptional"

---

## Support

**Questions:** Tag @transformation-portal-architect  
**Issues:** Open GitHub issue with label `phase1-enhancement`  
**Documentation:** See `docs/architecture/SCENE_DESCRIPTION_ENHANCEMENT_ROADMAP.md`

**Estimated Timeline:**
- Task 1: 2-4 hours
- Task 2: 4-6 hours
- Task 3: 3-4 hours
- Testing: 2-3 hours
- **Total: 11-17 hours (2 business days)**

**Target Completion:** 2025-12-15
