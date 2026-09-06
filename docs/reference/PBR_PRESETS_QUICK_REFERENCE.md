# PBR Presets Quick Reference

**For full documentation, see:** `docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md`

## Quick Start

```python
from transformation_portal.lux_depth_v3 import (
    EnhanceOrchestrator,
    STANDARD_QUALITY,  # or PREMIUM_QUALITY, FAST_PREVIEW
)
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from pathlib import Path

# Use the preset with one exact prepared input selection
input_root = Path("./input_images").resolve()
input_files = sorted(input_root.glob("*.jpg"))
prepared = prepare_lux_execution(STANDARD_QUALITY, input_root, input_files)
orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root=Path("./output"))
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
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from pathlib import Path

input_root = Path("./input_images").resolve()
image_paths = sorted(input_root.glob("*.jpg"))
output_root = Path("./pbr_output")
prepared = prepare_lux_execution(STANDARD_QUALITY, input_root, image_paths)
orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
results = orchestrator.enhance_batch(
    prepared.input_root,
    input_files=list(prepared.input_files),
)
for img_path, result in zip(prepared.input_files, results):
    print(f"{result['status']}: {img_path.name}")
```

### Example 2: Get Preset by Name

```python
from transformation_portal.lux_depth_v3 import get_preset, EnhanceOrchestrator
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from pathlib import Path

# Load preset dynamically
config = get_preset("premium")  # or "standard", "draft", "wood", etc.
input_root = Path("./input_images").resolve()
input_files = sorted(input_root.glob("*.jpg"))
prepared = prepare_lux_execution(config, input_root, input_files)
orchestrator = EnhanceOrchestrator.from_prepared(prepared, Path("./output"))
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
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from dataclasses import replace
from pathlib import Path

# Customize preset
custom_config = replace(
    STANDARD_QUALITY,
    pbr_normal_strength=1.5,  # Increase normal detail
    pbr_ao_bias=0.40,         # Darker AO
)

input_root = Path("./input_images").resolve()
input_files = sorted(input_root.glob("*.jpg"))
prepared = prepare_lux_execution(custom_config, input_root, input_files)
orchestrator = EnhanceOrchestrator.from_prepared(prepared, Path("./output"))
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
├── depth/
│   ├── <input-key>_depth.png    # 16-bit depth visualization
│   ├── <input-key>_depth.npy    # High-precision depth (if enabled)
│   └── <input-key>_depth_metadata.json # Depth provenance and statistics
├── pbr/
│   ├── <input-key>_normal.png   # RGB normal map
│   ├── <input-key>_roughness.png # Grayscale roughness map
│   └── <input-key>_ao.png       # Grayscale ambient occlusion map
├── manifests/
│   ├── <input-key>_combined.json     # Processing metadata
│   ├── batch_<batch-id>.json
│   └── execution_evidence_<batch-id>.json # Detached completion record
├── run_card_<batch-id>.json          # Reproducibility card when enabled
└── run_card_<batch-id>.self.json     # Run-card self-integrity sidecar
```

Use the paths returned by `enhance_batch` and the PBR paths recorded in the
combined manifest; do not reconstruct names from the source stem.

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
