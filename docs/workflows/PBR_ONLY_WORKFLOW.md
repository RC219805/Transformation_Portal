# PBR-Only Workflow Guide

This guide describes how to use the PBR (Physically Based Rendering) map generation features without requiring the full V2 enhancement pipeline.

## Overview

The PBR processing module is now decoupled from the V2 enhancement workflow, allowing you to:

1. Generate PBR maps from existing depth files
2. Run PBR-only workflows without the V2 script dependency
3. Use simplified APIs for common use cases
4. Batch process multiple depth maps efficiently

## Quick Start: PBR from Cached Depth

### Option 1: Python API (Recommended)

```python
from pathlib import Path
from transformation_portal.lux_depth_v3.pbr_processor import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset

# Load a preset configuration
preset = get_preset("premium")
pbr_config = preset.to_pbr_config()

# Generate PBR maps from cached depth file
paths = PBRProcessor.from_cached_depth(
    depth_path=Path("output/scene1_depth.npy"),
    config=pbr_config,
    output_dir=Path("output/pbr/"),
    base_name="scene1"
)

# Output paths
print(f"Normal map: {paths['normal']}")
print(f"Roughness map: {paths['roughness']}")
print(f"Ambient occlusion: {paths['ao']}")
```

### Option 2: CLI

```bash
# Using preset
python -m transformation_portal.lux_depth_v3.pbr_cli \
    --depth output/scene1_depth.npy \
    --preset premium \
    --output output/pbr/

# Custom parameters
python -m transformation_portal.lux_depth_v3.pbr_cli \
    --depth output/scene1_depth.npy \
    --normal-strength 1.8 \
    --roughness-strength 1.5 \
    --ao-strength 1.3 \
    --output output/pbr/
```

## Batch Processing

Process multiple depth files at once:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3.pbr_processor import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset

# Setup
depth_dir = Path("output/depth/")
pbr_dir = Path("output/pbr/")
config = get_preset("premium").to_pbr_config()

# Process all .npy depth files
for depth_file in depth_dir.glob("*.npy"):
    base_name = depth_file.stem

    try:
        paths = PBRProcessor.from_cached_depth(
            depth_path=depth_file,
            config=config,
            output_dir=pbr_dir,
            base_name=base_name
        )
        print(f"✓ Processed {base_name}")
    except Exception as e:
        print(f"✗ Failed {base_name}: {e}")
```

## Disabling V2 Enhancement

If you only need depth estimation and PBR maps (no V2 enhancement):

### Option 1: Python API

```python
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.config import EnhanceConfig

# Configure for PBR-only mode (V2 disabled)
config = EnhanceConfig(
    enable_v2=False,           # Disable V2 stage
    generate_pbr=True,         # Enable PBR generation
    pbr_normal_strength=1.5,
    pbr_ao_strength=1.2,
    depth_device="mps",        # Use Apple Silicon acceleration
)

orchestrator = EnhanceOrchestrator(config=config, output_root=Path("output/"))

# Process - will generate depth + PBR, skip V2
result = orchestrator.process_image(Path("input/image.jpg"))
```

### Option 2: CLI (New in v2.0.0+)

```bash
# PBR-only workflow (disable V2)
lux-depth-v3 \
    --input-dir "./input_images" \
    --output-dir "./output/pbr_only" \
    --enable-v2 "off" \
    --pbr "on" \
    --quality-tier "apex" \
    --depth-device "mps"

# Alternative: Skip V2 via preset control
lux-depth-v3 \
    --input-dir "./input_images" \
    --output-dir "./output/pbr_only" \
    --v2-preset "none" \
    --pbr "on" \
    --quality-tier "apex"
```

**V2 Control Flags:**
- `--enable-v2 on|off`: Master switch for V2 enhancement (default: on)
- `--v2-preset PRESET|none`: V2 preset or "none" to skip (default: default)

## Simplified Device Configuration

New simplified API for device selection:

```python
from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine

# Simple - just pass device string
engine = DA3InferenceEngine("mps")  # or "cpu", "cuda", "auto"

# Old style still works for backward compatibility
from transformation_portal.lux_depth_v3.config import DA3Config
config = DA3Config()
engine = DA3InferenceEngine(config)
```

## Available Presets

### Quality Presets
- **`draft`**: Fast processing, lower quality
- **`standard`**: Balanced speed and quality (default)
- **`premium`**: Highest quality, slower processing

### Material-Specific Presets
- **`wood`**: Optimized for wood surfaces
- **`metal`**: Sharp normals, low roughness
- **`glass`**: Smooth normals, very low roughness
- **`stone`**: High roughness, strong AO
- **`fabric`**: Soft normals, medium roughness

## Troubleshooting

### Missing V2 Script Error

If you see:
```
FileNotFoundError: V2 enhancement script not found: scripts/enhance_image.py
```

**Solutions:**

1. **Disable V2 via CLI** (easiest):
```bash
lux-depth-v3 --input-dir ./input_images --output-dir ./output --enable-v2 "off" --pbr "on"
```

2. **Disable V2 via Python API**:
```python
config = EnhanceConfig(enable_v2=False)
# OR
config = EnhanceConfig(v2_preset=None)
```

3. **Create placeholder script** (allows V2 pass-through):
   - The script `scripts/enhance_image.py` is now included as a placeholder
   - Provides pass-through behavior (copies input → output)
   - Replace with full enhancement logic when ready

**Note:** As of v2.0.0+, the V2 enhancement stage can be controlled via CLI flags, eliminating the need to modify code for PBR-only workflows.

### NaN/Inf in Depth

If you encounter:
```
ValueError: Depth contains NaN or Inf values
```

**Solution**: The PBR processor validates inputs. Clean your depth data:
```python
import numpy as np
depth = np.load("depth.npy")
depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
```

## Performance Tips

1. **Use `.npy` depth files**: 10-20x faster than PNG
2. **Batch processing**: Process multiple files in one session
3. **Memory-only mode**: Skip file I/O when not needed (`save=False`)
4. **Preset selection**: Use `draft` for iteration, `premium` for final
5. **Apple Silicon**: Use `depth_device="mps"` for 3-5x speedup

## Further Reading

- [PBR Integration Architecture](../architecture/ADR-001-PBR-Integration-Architecture.md)
- [PBR Implementation Review](../architecture/PBR_IMPLEMENTATION_REVIEW_2026-02-01.md)
- [PBR Configuration Guide](../guides/PBR_ENHANCE_CONFIG_GUIDE.md)
