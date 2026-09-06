# PBR Processor Quick Start Guide

**Version:** 2.0
**Component:** Standalone PBR Map Generator
**Purpose:** Generate Physically Based Rendering maps from existing depth data

---

## What is PBRProcessor?

`PBRProcessor` is a standalone API for generating PBR maps (normal, roughness, ambient occlusion) from depth data **without running the full depth estimation pipeline**.

### Key Benefits

- **2.3x faster** than full pipeline when regenerating PBR from existing depth
- **Memory-efficient** mode for custom post-processing workflows
- **No orchestrator dependency** - simple, focused API
- **Material presets** for wood, metal, glass, stone, fabric
- **Batch processing** support with progress tracking

---

## 5-Minute Tutorial

### Step 1: Installation

Ensure Transformation Portal is installed:

```bash
cd Transformation_Portal
pip install -e .
```

### Step 2: Verify Depth Files Exist

PBRProcessor requires an explicit existing depth-map path. For an
`EnhanceOrchestrator` run, retain the exact `result["depth_float_path"]`
returned by `enhance_batch`; the evidence-bound name includes the input
extension and a stable path hash, so do not reconstruct it from the source
stem. The standalone tutorial below uses a caller-managed depth file instead:

```bash
# Check the caller-managed depth file used by this tutorial
test -f output/scene1_depth.npy
```

### Step 3: Generate PBR Maps

**Python script** (`generate_pbr.py`):

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Configure preset
config = get_preset("premium").to_pbr_config()

# Generate PBR from cached depth
paths = PBRProcessor.from_cached_depth(
    depth_path=Path("output/scene1_depth.npy"),
    config=config,
    output_dir=Path("output/pbr/"),
    base_name="scene1"
)

print(f"✓ Normal map: {paths['normal']}")
print(f"✓ Roughness: {paths['roughness']}")
print(f"✓ AO map: {paths['ao']}")
```

**Run:**
```bash
python generate_pbr.py
```

**Output:**
```
output/pbr/
├── scene1_normal.png      # RGB normal map (tangent space)
├── scene1_roughness.png   # Grayscale roughness
└── scene1_ao.png          # Ambient occlusion
```

### Step 4: Verify Output

Check generated maps:

```python
from PIL import Image

# Load and inspect
normal = Image.open("output/pbr/scene1_normal.png")
print(f"Normal map size: {normal.size}")
print(f"Normal map mode: {normal.mode}")  # Should be RGB

roughness = Image.open("output/pbr/scene1_roughness.png")
print(f"Roughness mode: {roughness.mode}")  # Should be L (grayscale)
```

---

## Common Use Cases

### Use Case 1: Iterating on PBR Parameters

**Scenario**: You've run depth estimation once and want to test different PBR presets.

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

depth_path = Path("output/luxury_kitchen_depth.npy")

# Try different presets
for preset_name in ["standard", "premium", "wood", "metal"]:
    config = get_preset(preset_name).to_pbr_config()
    output_dir = Path(f"output/pbr_{preset_name}/")

    paths = PBRProcessor.from_cached_depth(
        depth_path=depth_path,
        config=config,
        output_dir=output_dir,
        base_name="luxury_kitchen"
    )

    print(f"✓ Generated {preset_name} preset")

# Review outputs and choose best for final deliverable
```

**Performance**: ~1.2s per preset vs ~2.8s if re-running full pipeline.

### Use Case 2: Batch Processing Existing Depths

**Scenario**: You have 100 depth files and need PBR maps for all.

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Input directory with depth files
depth_dir = Path("output/estate_depths/")
depth_files = sorted(depth_dir.glob("*_depth.npy"))

# Configure preset
config = get_preset("premium").to_pbr_config()
pbr_dir = Path("output/estate_pbr/")

# Process all files
for i, depth_file in enumerate(depth_files, 1):
    base_name = depth_file.stem.replace("_depth", "")

    paths = PBRProcessor.from_cached_depth(
        depth_path=depth_file,
        config=config,
        output_dir=pbr_dir,
        base_name=base_name
    )

    print(f"[{i}/{len(depth_files)}] ✓ {base_name}")

print(f"\nProcessed {len(depth_files)} images")
# Throughput: ~3,000 images/hour vs ~1,277 with full pipeline
```

### Use Case 3: Memory-Only Processing

**Scenario**: Custom post-processing without intermediate file I/O.

```python
import numpy as np
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset
from PIL import Image

# Load depth array
depth = np.load("output/scene1_depth.npy")

# Generate PBR in memory (no file writes)
config = get_preset("standard").to_pbr_config()
processor = PBRProcessor(config=config, output_dir=None)

maps = processor.from_depth(depth, save=False)

# Access maps as NumPy arrays
normal_map = maps["normal"]      # Shape: (H, W, 3), dtype: uint8
roughness_map = maps["roughness"]  # Shape: (H, W), dtype: uint8
ao_map = maps["ao"]               # Shape: (H, W), dtype: uint8

# Custom processing
ao_enhanced = (ao_map * 1.5).clip(0, 255).astype(np.uint8)

# Save final result
Image.fromarray(ao_enhanced).save("output/custom_ao.png")
```

**Performance**: Fastest option (~1.16s for 24MP), no disk I/O overhead.

### Use Case 4: Material-Specific Processing

**Scenario**: Process architectural renders with material-optimized presets.

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Material classification (manual or automated)
scenes = {
    "hardwood_floor": "wood",
    "marble_countertop": "stone",
    "glass_facade": "glass",
    "metal_fixtures": "metal",
    "linen_curtains": "fabric",
}

for scene_name, material in scenes.items():
    depth_path = Path(f"output/{scene_name}_depth.npy")
    config = get_preset(material).to_pbr_config()

    paths = PBRProcessor.from_cached_depth(
        depth_path=depth_path,
        config=config,
        output_dir=Path(f"output/{material}_pbr/"),
        base_name=scene_name
    )

    print(f"✓ {scene_name} → {material} preset")
```

---

## Available Presets

### Quality Tiers

| Preset | Use Case | Normal Detail | AO Bias | Speed |
|--------|----------|---------------|---------|-------|
| `standard` | Batch processing | Balanced | 0.45 | Fast |
| `premium` | Hero shots | Maximum | 0.35 | Medium |
| `draft` | Quick preview | Low | 0.55 | Fastest |

### Material-Optimized

| Preset | Optimized For | Key Settings |
|--------|---------------|--------------|
| `wood` | Hardwood, grain texture | High normal strength, no blur |
| `metal` | Polished metal surfaces | Low roughness strength |
| `glass` | Windows, mirrors | Heavy smoothing, bright AO |
| `stone` | Marble, granite, tile | High detail, dark AO for grout |
| `fabric` | Textiles, upholstery | Moderate detail, balanced AO |

**Access presets:**

```python
from transformation_portal.lux_depth_v3 import get_preset, list_presets

# List all available
print(list_presets())
# ['standard', 'premium', 'draft', 'wood', 'metal', 'glass', 'stone', 'fabric']

# Get specific preset
config = get_preset("wood").to_pbr_config()
```

---

## Troubleshooting

### Issue: "Depth file not found"

**Error:**
```
FileNotFoundError: Depth file not found: output/scene1_depth.npy
```

**Solution:**
- Verify the caller-managed depth file exists: `test -f output/scene1_depth.npy`
- For orchestrator-generated depth, pass
  `Path(result["depth_float_path"])` from the successful `enhance_batch`
  result instead of guessing `output/scene1_depth.npy`
- If needed, run the full orchestrator first and retain that returned result
- Check file path is correct (relative or absolute)

### Issue: "Expected 2D depth array"

**Error:**
```
ValueError: Expected 2D depth array, got shape (1080, 1920, 3)
```

**Solution:**
- Depth should be grayscale (H, W), not RGB (H, W, 3)
- If you have RGB depth visualization, extract single channel:
  ```python
  from PIL import Image
  import numpy as np

  depth_rgb = np.array(Image.open("depth.png"))
  depth = depth_rgb[:, :, 0]  # Use R channel
  np.save("depth.npy", depth / 255.0)  # Normalize to [0, 1]
  ```

### Issue: "Depth contains NaN or Inf values"

**Error:**
```
ValueError: Depth contains NaN or Inf values
```

**Solution:**
- Check depth array for invalid values:
  ```python
  import numpy as np
  depth = np.load("depth.npy")
  print(f"NaN count: {np.isnan(depth).sum()}")
  print(f"Inf count: {np.isinf(depth).sum()}")
  ```
- Clean depth array:
  ```python
  depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
  np.save("depth_cleaned.npy", depth)
  ```

### Issue: Output maps are all uniform/flat

**Symptom**: Normal map is solid blue, AO is all gray

**Cause**: Depth map is flat (all same value) or has very low variation

**Solution:**
- Verify depth has meaningful variation:
  ```python
  import numpy as np
  depth = np.load("depth.npy")
  print(f"Min: {depth.min()}, Max: {depth.max()}, Std: {depth.std()}")
  ```
- If depth is flat, re-run depth estimation with higher quality model
- Check source image has actual depth variation

### Issue: PBR generation is slow

**Expected performance** (256x256 image):
- Memory-only: ~0.1s
- With file I/O: ~0.3s

**If slower:**
- Check disk I/O speed (SSD recommended)
- Verify depth array dtype is `float32` (not `float64`)
- Close other memory-intensive applications
- For large batches, process in smaller chunks

---

## Performance Benchmarks

**Test configuration:**
- Image size: 24MP (6000×4000)
- Hardware: Apple M4 Max, 48GB RAM
- Depth model: `da3-metric` (commercial-safe)

| Workflow | Time (ms) | Throughput (img/hr) |
|----------|-----------|---------------------|
| Full Orchestrator | 2,800 | 1,277 |
| PBRProcessor (file) | 1,200 | 3,000 |
| PBRProcessor (memory) | 1,160 | 3,100 |

**Speedup comparison:**
- **Single image**: 2.3-2.4x faster than orchestrator
- **10 preset iterations**: 2x faster (13.7s vs 28s)
- **Batch processing**: 2.3x higher throughput

---

## Advanced: Custom Parameter Overrides

Override preset parameters for fine-grained control:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor
from transformation_portal.lux_depth_v3.pbr import PBRConfig

# Create custom config (not using preset)
custom_config = PBRConfig(
    normal_strength=1.8,       # Very strong normals
    normal_blur_radius=0,       # No smoothing (sharp edges)
    roughness_strength=0.8,     # Lower roughness (smoother)
    roughness_blur_radius=5,    # Heavy roughness smoothing
    ao_strength=1.3,            # Stronger AO
    ao_blur_radius=8,           # Wide AO spread
    ao_bias=0.3,                # Darker shadows
)

processor = PBRProcessor(config=custom_config, output_dir=Path("output/custom/"))
maps = processor.from_depth(depth, save=True, base_name="custom_scene")
```

**Parameter guidelines:**
- `normal_strength`: 0.5-2.0 (higher = more pronounced surface detail)
- `normal_blur_radius`: 0-5 (0 = sharp, 5 = very smooth)
- `roughness_strength`: 0.5-2.0 (higher = rougher surfaces)
- `roughness_blur_radius`: 0-10 (higher = smoother transitions)
- `ao_strength`: 0.5-2.0 (higher = darker occlusion)
- `ao_blur_radius`: 0-15 (higher = wider shadow spread)
- `ao_bias`: 0.0-1.0 (0 = dark, 1 = bright, 0.5 = neutral)

---

## Integration with Full Pipeline

PBRProcessor complements the full orchestrator pipeline:

### Workflow 1: Depth First, PBR Later

```python
from dataclasses import replace
from pathlib import Path
from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, PBRProcessor, get_preset
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution

# Step 1: Run orchestrator for depth only
config = replace(get_preset("premium"), generate_pbr=False)

input_root = Path("input").resolve()
image_path = input_root / "scene1.jpg"
prepared = prepare_lux_execution(config, input_root, [image_path])
orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root=Path("output/"))
result = orchestrator.enhance_batch(
    prepared.input_root,
    input_files=list(prepared.input_files),
)[0]
if result["status"] != "ok" or not result["depth_float_path"]:
    raise RuntimeError("Depth generation did not produce the requested float map")

# Step 2: Generate PBR separately (allows parameter iteration)
pbr_config = config.to_pbr_config()
paths = PBRProcessor.from_cached_depth(
    depth_path=Path(result["depth_float_path"]),
    config=pbr_config,
    output_dir=Path("output/pbr/"),
    base_name="scene1"
)
```

### Workflow 2: PBR from External Depth

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Use depth from external source (MiDaS, ZoeDepth, etc.)
external_depth_path = Path("external/zoe_depth.npy")

config = get_preset("standard").to_pbr_config()
paths = PBRProcessor.from_cached_depth(
    depth_path=external_depth_path,
    config=config,
    output_dir=Path("output/pbr/"),
    base_name="external_scene"
)
```

---

## Next Steps

- **Full documentation**: `docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md`
- **Production example**: `examples/process_750_picacho_pbr.py`
- **API reference**: `src/transformation_portal/lux_depth_v3/pbr_processor.py`
- **Preset catalog**: `src/transformation_portal/lux_depth_v3/pbr_presets.py`
- **Integration tests**: `tests/test_pbr_processor.py`

---

**Last Updated**: 2026-02-01
**Version**: 2.0
