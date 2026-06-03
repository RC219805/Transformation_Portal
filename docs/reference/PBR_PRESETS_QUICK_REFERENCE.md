# PBR Presets Quick Reference

**For full documentation, see:** `docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md`

## Quick Start

```python
from transformation_portal.lux_depth_v3 import (
    EnhanceOrchestrator,
    STANDARD_QUALITY,  # or PREMIUM_QUALITY, FAST_PREVIEW
)
from pathlib import Path

# Use preset directly
orchestrator = EnhanceOrchestrator(STANDARD_QUALITY, output_root=Path("./output"))
```

## Available Presets

### Quality Tiers

| Preset | Throughput | Quality | Use Case |
|--------|-----------|---------|----------|
| `FAST_PREVIEW` | 500-700 img/hr | Draft | Quick iteration, internal review |
| `STANDARD_QUALITY` | 200-250 img/hr | Production | Typical batch workflows |
| `PREMIUM_QUALITY` | 100-150 img/hr | Maximum | Hero shots, client deliverables |

### Material-Specific

| Preset | Optimized For | Key Characteristics |
|--------|--------------|---------------------|
| `WOOD_OPTIMIZED` | Hardwood floors, cabinetry | High normal detail, grain preservation |
| `METAL_OPTIMIZED` | Fixtures, appliances | Lower roughness, smooth surfaces |
| `GLASS_OPTIMIZED` | Windows, mirrors | Heavy smoothing, flat normals |
| `STONE_OPTIMIZED` | Countertops, tile | High texture detail, deep grout shadows |
| `FABRIC_OPTIMIZED` | Upholstery, curtains | Moderate detail, natural folds |

## Usage Examples

### Example 1: Standard Batch Processing

```python
from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, STANDARD_QUALITY
from pathlib import Path

output_root = Path("./pbr_output")
orchestrator = EnhanceOrchestrator(STANDARD_QUALITY, output_root)

for img_path in Path("./input_images").glob("*.jpg"):
    result = orchestrator.enhance_image(img_path, input_root=Path("./input_images"))
    print(f"✓ {img_path.name}")
```

### Example 2: Get Preset by Name

```python
from transformation_portal.lux_depth_v3 import get_preset, EnhanceOrchestrator

# Load preset dynamically
config = get_preset("premium")  # or "standard", "draft", "wood", etc.
orchestrator = EnhanceOrchestrator(config, output_root)
```

### Example 3: List Available Presets

```python
from transformation_portal.lux_depth_v3 import list_presets

print("Available PBR presets:")
for name in list_presets():
    print(f"  - {name}")
```

### Example 4: Custom Preset Based on Standard

```python
from transformation_portal.lux_depth_v3 import STANDARD_QUALITY, EnhanceOrchestrator
from dataclasses import replace

# Customize preset
custom_config = replace(
    STANDARD_QUALITY,
    pbr_normal_strength=1.5,  # Increase normal detail
    pbr_ao_bias=0.40,         # Darker AO
)

orchestrator = EnhanceOrchestrator(custom_config, output_root)
```

## Key Parameters

### Normal Map
- `pbr_normal_strength`: Gradient intensity (0.5-2.0, default 1.0)
- `pbr_normal_blur_radius`: Pre-blur smoothing (0=sharp, 3=smooth)

### Roughness Map
- `pbr_roughness_strength`: Surface detail intensity (0.5-1.5)
- `pbr_roughness_blur_radius`: Smoothing kernel (2-6)

### Ambient Occlusion
- `pbr_ao_strength`: Shadow darkness (0.5-1.5)
- `pbr_ao_blur_radius`: Occlusion spread (3-10)
- `pbr_ao_bias`: Brightness offset (0.0=dark, 1.0=bright)

### Critical Setting
- `save_float_depth=True`: **Always enable** for production quality (prevents quantization artifacts)

## Output Files

```
output_root/
├── image_name_depth.png              # 16-bit depth visualization
├── image_name_depth_float.npy        # High-precision depth (if save_float_depth=True)
├── image_name_normal.png             # RGB normal map
├── image_name_roughness.png          # Grayscale roughness map
├── image_name_ao.png                 # Grayscale ambient occlusion map
└── image_name_manifest.json          # Processing metadata
```

## Troubleshooting

**Flat/low-contrast PBR maps?**
→ Enable `save_float_depth=True`, increase strength parameters

**Noisy normals on smooth surfaces?**
→ Increase `pbr_normal_blur_radius` (try 2-3)

**Over-darkened AO?**
→ Increase `pbr_ao_bias` (try 0.55) or reduce `pbr_ao_strength`

**Lost fine detail?**
→ Reduce blur radii, ensure `save_float_depth=True`

## Performance Tips

1. **Two-pass workflow**: Use `FAST_PREVIEW` first to validate depth quality, then re-run with `PREMIUM_QUALITY` (depth cached = 10-20x faster)

2. **Material batching**: Group images by dominant material and use material-specific presets

3. **Hybrid approach**: Use `STANDARD_QUALITY` for batch, `PREMIUM_QUALITY` for hero shots only

## References

- **Full Guide**: `docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md`
- **Module**: `src/transformation_portal/lux_depth_v3/pbr_presets.py`
- **Tests**: `tests/test_pbr_presets.py`
- **PBR Algorithm**: `src/transformation_portal/lux_depth_v3/pbr.py`
