# TIFF Enhancement Pipeline

## Comprehensive orchestration for dramatic enhancement of large 16/32-bit architectural TIFFs

---

## Pipeline Architecture

### Stage Sequence & Rationale

```
Input TIFFs (16/32-bit)
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 1: HDR Enhancement                    │
│ (realize_v8_unified.py)                     │
│                                             │
│ • Preserves high bit depth                 │
│ • Initial color grading & exposure         │
│ • Tone curve application                   │
│ • Output: *_ENH.tif (16/32-bit)           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 2: Depth Prediction                   │
│ (depth_predict_coreml.py)                   │
│                                             │
│ • CoreML accelerated (Apple Neural Engine) │
│ • DepthAnything V2 model                   │
│ • Output: *_depth16.png                    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 3: AgX Tone Mapping                   │
│ (agx_batch_processor.py)                    │
│                                             │
│ • Professional filmic tone mapping         │
│ • Per-image auto-exposure                  │
│ • Scene-referred to display-referred       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 4: Semantic Segmentation              │
│ (run_detectron2_panoptic_batch.py)         │
│                                             │
│ • Detectron2 panoptic segmentation         │
│ • Sky & building masks                     │
│ • Output: *_mask_sky.png, *_mask_building  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STAGE 5: Depth-Aware Effects                │
│ (depth_tools.py)                            │
│                                             │
│ • Atmospheric haze (depth-weighted)        │
│ • Clarity enhancement (depth-selective)    │
│ • Depth-of-field (optional)                │
│ • Combines depth + masks + enhanced RGB    │
└─────────────────────────────────────────────┘
    ↓
Final Enhanced Images
```

### Why This Sequence?

1. **HDR Enhancement First**: Maximize dynamic range preservation before any neural network processing
2. **Depth on Enhanced Images**: Better depth estimation from properly exposed, color-graded images
3. **Tone Mapping Before Segmentation**: Display-referred images yield better semantic segmentation
4. **Effects Last**: Final compositing stage has access to all computed artifacts (depth, masks, tonemap)

---

## Installation

### Core Requirements

```bash
# Core dependencies
pip install numpy pillow coremltools imageio tifffile tqdm pyyaml

# AgX tone mapping (optional but recommended)
pip install opencolorio

# Depth effects optimization (optional)
pip install opencv-python scipy

# Segmentation (required for Stage 4)
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

### M4 Max Optimizations

The pipeline automatically leverages:
- **Apple Neural Engine (ANE)** for CoreML depth prediction
- **Metal Performance Shaders** for image operations
- **Unified memory architecture** for efficient data transfers

---

## Quick Start

### Basic Usage

```bash
python tiff_enhancement_pipeline.py \
    --input-dir /path/to/tiffs \
    --output-dir /path/to/outputs \
    --preset dramatic \
    --depth-effects haze clarity \
    --workers 4
```

### Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `natural` | Subtle, realistic enhancement | Documentation, archival |
| `punchy` | Moderate contrast, vibrant | Marketing materials |
| `dramatic` | **High contrast, cinematic** | Hero shots, portfolio |
| `golden` | Warm, atmospheric | Sunset/sunrise scenes |
| `vibrant` | Maximum saturation, clarity | Social media, web |

### Advanced Usage

```bash
python tiff_enhancement_pipeline.py \
    --input-dir ./raw_tiffs \
    --output-dir ./enhanced \
    --preset dramatic \
    --tone-curve agx-high \
    --ocio-config /path/to/agx/config.ocio \
    --depth-effects haze clarity dof \
    --workers 8 \
    --bitdepth 16 \
    --keep-intermediates \
    --device mps
```

### Skip Stages

If you already have intermediate outputs:

```bash
# Skip enhancement and depth, start from tone mapping
python tiff_enhancement_pipeline.py \
    --input-dir ./tiffs \
    --output-dir ./outputs \
    --skip-stages 1 2 \
    --preset dramatic
```

---

## Performance Optimization

### M4 Max Tuning (100-200MB TIFFs)

**Recommended settings:**

```bash
--workers 6           # M4 Max has 12/14 cores, leave headroom
--device cpu          # For Detectron2 (MPS support varies)
--bitdepth 16         # Balance quality vs. file size
```

**Memory considerations:**
- Each worker consumes ~2-4GB during processing
- 100MB TIFF → ~400MB decompressed in memory
- Pipeline peak: ~20-30GB for 4-6 workers

### Processing Time Estimates (M4 Max)

For a 4000×3000 px, 150MB TIFF:

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1: Enhancement | 8-12s | Multi-threaded numpy |
| Stage 2: Depth | 3-5s | CoreML/ANE accelerated |
| Stage 3: Tone Mapping | 2-4s | OCIO optimized |
| Stage 4: Segmentation | 6-10s | Detectron2 inference |
| Stage 5: Depth Effects | 4-8s per effect | OpenCV/scipy |
| **Total** | **~30-45s/image** | With haze + clarity |

### Batch Processing

For 100 images:
- Serial: ~50-75 minutes
- Parallel (6 workers): ~15-25 minutes

---

## Configuration Files

### Using YAML Configs

Create `pipeline_config.yaml`:

```yaml
# Enhancement parameters
preset: dramatic
tone_curve: agx-high
exposure_ev: 0.5
contrast: 1.12
saturation: 1.08

# Depth effects
depth_effects:
  - haze
  - clarity

haze:
  strength: 0.22
  near: 15.0
  far: 85.0

clarity:
  amount: 0.15
  radius: 3

# Output
out_bitdepth: 16
quality_jpeg: 95
```

Load with:

```bash
python tiff_enhancement_pipeline.py \
    --config pipeline_config.yaml \
    --input-dir ./tiffs \
    --output-dir ./outputs
```

---

## Troubleshooting

### Common Issues

**1. "CoreML model not found"**
```bash
# Download DepthAnything V2 CoreML model
# Update MODEL_PATH in depth_predict_coreml.py
```

**2. "AgX tone mapping failed"**
```bash
# Install OpenColorIO
pip install opencolorio

# Download AgX config
git clone https://github.com/sobotka/AgX
# Set --ocio-config to downloaded config.ocio
```

**3. "Out of memory"**
```bash
# Reduce workers
--workers 2

# Or process stages separately
--skip-stages 4 5  # Run 1-3 first
--skip-stages 1 2 3  # Then run 4-5
```

**4. "Detectron2 installation failed (M4 Max)"**
```bash
# Use CPU wheels
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

---

## Output Structure

```
outputs/
├── pipeline_manifest.json        # Processing metadata
├── 01_enhance/                  # Stage 1 outputs
│   ├── image001_ENH.tif
│   └── image002_ENH.tif
├── 02_depth/                    # Stage 2 outputs
│   ├── image001_depth16.png
│   ├── image001_depth8_vis.png
│   └── ...
├── 03_tonemap/                  # Stage 3 outputs
│   ├── image001_ENH.jpg
│   └── ...
├── 04_masks/                    # Stage 4 outputs
│   ├── image001_mask_sky.png
│   ├── image001_mask_building.png
│   └── ...
└── 05_final/                    # ★ Final enhanced images
    ├── image001_depthhaze.tif
    ├── image001_depthclarity.tif
    └── ...
```

Use `--keep-intermediates` to preserve all stages; otherwise intermediates are cleaned up automatically.

---

## API Usage

### Python API

```python
from pathlib import Path
from tiff_enhancement_pipeline import Pipeline, PipelineConfig

# Configure
config = PipelineConfig(
    input_dir=Path("./tiffs"),
    output_dir=Path("./outputs"),
    stage1_enhance=Path("./outputs/01_enhance"),
    stage2_depth=Path("./outputs/02_depth"),
    stage3_tonemap=Path("./outputs/03_tonemap"),
    stage4_masks=Path("./outputs/04_masks"),
    stage5_final=Path("./outputs/05_final"),
    preset="dramatic",
    tone_curve="agx",
    depth_effects=["haze", "clarity"],
    workers=6,
    out_bitdepth=16
)

# Execute
pipeline = Pipeline(config)
success = pipeline.execute()
```

### Custom Stage Execution

```python
from tiff_enhancement_pipeline import Stage1Enhance, PipelineConfig

config = PipelineConfig(...)
stage = Stage1Enhance(config)
success, files_processed = stage.execute()
```

---

## Advanced Topics

### Custom Depth Effects

Modify `depth_tools.py` parameters in Stage5DepthEffects:

```python
# In tiff_enhancement_pipeline.py, Stage5DepthEffects.execute()
if effect == "haze":
    cmd.extend([
        "--strength", "0.25",      # Increase haze intensity
        "--haze-color", "0.92", "0.94", "0.98",  # Custom color
        "--near", "10.0",          # Start closer
        "--far", "90.0",           # Extend further
        "--mids-gain", "1.05"      # Boost midtones
    ])
```

### Multi-Project Workflows

```bash
# Process multiple projects
for project in project_*/; do
    python tiff_enhancement_pipeline.py \
        --input-dir "$project/raw" \
        --output-dir "$project/enhanced" \
        --preset dramatic \
        --workers 6
done
```

### Archival Workflows

For maximum quality preservation:

```bash
python tiff_enhancement_pipeline.py \
    --input-dir ./tiffs \
    --output-dir ./archive \
    --preset natural \
    --bitdepth 32 \
    --keep-intermediates \
    --tone-curve agx-base
```

---

## Technical Details

### Depth Map Format

- **16-bit PNG** normalized to [0, 65535]
- Closer = darker, Further = brighter
- Inverse depth representation (1/z)

### Tone Mapping Pipeline

1. **Scene-linear** → Enhanced image in linear light
2. **AgX/Filmic** → Display-referred via OCIO
3. **sRGB OETF** → Encoded for display

### Mask Generation

- **Sky**: COCO categories with "sky" in name
- **Building**: "building", "wall", "house", "structure"
- Binary masks: 255 = object present, 0 = absent

---

## Credits

- **DepthAnything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **AgX**: Troy Sobotka's OCIO config
- **Detectron2**: Facebook AI Research
- **Pipeline Architecture**: Optimized for architectural visualization workflows

---

## License

MIT License - See individual tool licenses for dependencies.

---

## Support

For issues or feature requests, ensure all dependencies are installed and check the troubleshooting section. The pipeline is designed to fail gracefully with informative error messages at each stage.
