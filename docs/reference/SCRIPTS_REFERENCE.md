# Scripts Reference Guide
**Transformation Portal - Complete Script Inventory**

## Overview

Complete reference for all scripts in the Transformation Portal, organized by function and with usage examples.

---

## Table of Contents

1. [Core Processing Pipelines](#core-processing-pipelines)
2. [Enhancement Utilities](#enhancement-utilities)
3. [Depth Processing](#depth-processing)
4. [Development Tools](#development-tools)
5. [Setup & Installation](#setup--installation)
6. [Testing & Validation](#testing--validation)
7. [Legacy & Experimental](#legacy--experimental)

---

## Core Processing Pipelines

### 1. `lux_render_pipeline.py`
**Purpose**: AI-powered architectural render enhancement with ControlNet and SDXL

**Features:**
- Edge-preserving enhancement (ControlNet Canny)
- Depth-aware refinement (ControlNet Depth)
- Real-ESRGAN 4x upscaling
- Material Response finishing
- Brand overlay support

**Usage:**
```bash
python lux_render_pipeline.py \
  --input input_images/kitchen.jpg \
  --output output/ \
  --preset architectural \
  --upscale 4
```

**Options:**
- `--preset`: architectural, interior, exterior
- `--controlnet`: canny, depth, both
- `--upscale`: 2, 4 (Real-ESRGAN)
- `--material-response`: 0.0-1.0
- `--brand-overlay`: path to logo PNG

---

### 2. `luxury_video_master_grader.py`
**Purpose**: Professional video color grading with LUT application

**Features:**
- 16+ cinematic LUT presets
- HDR tone mapping (PQ, HLG)
- ProRes 422 HQ output
- Frame rate conformance
- Color metadata preservation

**Usage:**
```bash
python luxury_video_master_grader.py \
  --input video.mp4 \
  --output graded.mov \
  --preset signature_estate \
  --lut-strength 0.7
```

**Presets:**
- `signature_estate`: Kodak 2393 film emulation
- `golden_hour_courtyard`: Warm sunset tones
- `coastal_breeze`: Cool, airy aesthetics
- `urban_lux`: Modern architectural look

---

### 3. `luxury_tiff_batch_processor.py`
**Purpose**: 16-bit TIFF batch processing with LUT application

**Features:**
- Preserve 16-bit precision
- Batch directory processing
- GPS/IPTC metadata preservation
- Progress tracking
- Output directory mirroring

**Usage:**
```bash
python luxury_tiff_batch_processor_cli.py \
  --input input_dir/ \
  --output output_dir/ \
  --preset warm_estate \
  --workers 4
```

**Options:**
- `--preset`: Choose from 10+ presets
- `--workers`: Parallel processing (default: 4)
- `--recursive`: Process subdirectories
- `--preserve-metadata`: Keep EXIF/IPTC

---

### 4. `pro_pipeline.py`
**Purpose**: Professional end-to-end rendering pipeline

**Features:**
- Depth-aware processing
- Material Response integration
- Multi-stage enhancement
- Linear colorspace workflow
- Quality validation

**Usage:**
```bash
python pro_pipeline.py \
  --input kitchen.jpg \
  --output output/ \
  --stages all \
  --quality premium
```

**Stages:**
- depth: Depth estimation
- enhance: AI enhancement
- material: Material Response
- grade: Color grading
- upscale: 4K upscaling

---

## Enhancement Utilities

### 5. `material_response.py`
**Purpose**: Physics-based surface enhancement

**Features:**
- Material detection (wood, metal, glass, fabric, stone)
- Per-material enhancement curves
- Micro-contrast preservation
- Highlight protection

**Usage:**
```bash
python material_response.py \
  --input render.jpg \
  --output enhanced.jpg \
  --strength 0.7 \
  --materials wood metal glass
```

---

### 6. `agx_batch_processor.py`
**Purpose**: AgX tone mapping for architectural renders

**Features:**
- Filmic tone mapping
- Batch processing
- Highlight roll-off
- Shadow detail preservation

**Usage:**
```bash
python agx_batch_processor.py \
  --input renders/ \
  --output graded/ \
  --exposure 0.0 \
  --contrast 1.1
```

---

### 7. `run_aerial_enhancement.py`
**Purpose**: Aerial photography enhancement

**Features:**
- Atmospheric haze removal
- Color temperature correction
- Horizon leveling
- GPS metadata preservation

**Usage:**
```bash
python scripts/pipelines/run_aerial_enhancement.py \
  --input aerial.jpg \
  --output enhanced.jpg \
  --resolution 2048
```

---

## Depth Processing

### 8. `depth_anything_v2.py`
**Purpose**: Depth Anything V2 integration

**Features:**
- Monocular depth estimation
- CoreML optimization (Apple Silicon)
- Batch processing
- Depth map visualization

**Usage:**
```bash
python depth_anything_v2.py \
  --input image.jpg \
  --output depth.png \
  --model small \
  --device mps
```

**Models:**
- small: Fastest (24ms on M4)
- base: Balanced
- large: Highest quality

---

### 9. `depth_predict_coreml.py`
**Purpose**: CoreML-accelerated depth prediction

**Features:**
- Apple Neural Engine optimization
- 10-20x speedup on M-series
- LRU caching
- Batch inference

**Usage:**
```bash
python depth_predict_coreml.py \
  --input batch/ \
  --output depth_maps/ \
  --cache
```

---

### 10. `depth_tools.py`
**Purpose**: Depth processing utilities

**Features:**
- Depth map normalization
- Zone-based segmentation
- Atmospheric effects
- Depth-aware filtering

**Usage:**
```python
from depth_tools import normalize_depth, apply_atmospheric

depth = normalize_depth(raw_depth)
result = apply_atmospheric(image, depth, strength=0.3)
```

---

## Development Tools

### 11. `codebase_philosophy_auditor.py`
**Purpose**: Code quality and philosophy compliance auditing

**Features:**
- Decision annotation checking
- Complexity analysis
- Documentation coverage
- Philosophy alignment scoring

**Usage:**
```bash
python codebase_philosophy_auditor.py \
  --target src/ \
  --output audit_report.json
```

---

### 12. `decision_decay_dashboard.py`
**Purpose**: Temporal contract monitoring

**Features:**
- Decision annotation tracking
- Expiration monitoring
- Decay detection
- Dashboard visualization

**Usage:**
```bash
python decision_decay_dashboard.py \
  --scan . \
  --output dashboard.html
```

---

### 13. `temporal_evolution.py`
**Purpose**: Code evolution tracking

**Features:**
- Git history analysis
- LOC trend tracking
- Complexity evolution
- Refactoring detection

**Usage:**
```bash
python temporal_evolution.py \
  --repo . \
  --since "2025-01-01" \
  --output evolution.json
```

---

### 14. `evolutionary_checkpoint.py`
**Purpose**: Version snapshot creation

**Features:**
- Codebase snapshots
- Diff generation
- Rollback support
- Change tracking

**Usage:**
```bash
python evolutionary_checkpoint.py \
  --create "v1.2.0" \
  --tag "stable-release"
```

---

### 15. `visualize_material_assignments.py`
**Purpose**: Material detection visualization

**Features:**
- Material heatmaps
- Detection confidence overlay
- Multi-material comparison
- Debug visualization

**Usage:**
```bash
python scripts/utilities/visualize_material_assignments.py \
  render.jpg \
  --output viz.png \
  --clusters 8
```

---

## Setup & Installation

### 16. `install_models.py`
**Purpose**: Interactive model installation

**Features:**
- Depth Anything V2 download
- Real-ESRGAN weights
- ControlNet models
- Progress tracking

**Usage:**
```bash
python scripts/setup/install_models.py --dry-run
```

---

### 17. `install_models_auto.py`
**Purpose**: Automated model installation

**Features:**
- Non-interactive setup
- Parallel downloads
- Checksum verification
- Resume support

**Usage:**
```bash
python scripts/setup/install_models_auto.py --skip-optional
```

---

### 18. `download_depth_models.py`
**Purpose**: Depth model download utility

**Features:**
- Model variant selection
- CoreML conversion
- Cache management

**Usage:**
```bash
python scripts/setup/download_depth_models.py \
  --model depth \
  --verify-only
```

---

## Testing & Validation

### 19. `verify_example_paths.py`
**Purpose**: Example code validation

**Features:**
- Path existence checking
- Import validation
- Example execution tests
- Error reporting

**Usage:**
```bash
python verify_example_paths.py \
  --check-all \
  --fix-paths
```

---

### 20. `format_utils.py`
**Purpose**: File format utilities

**Features:**
- Format detection
- Conversion helpers
- Metadata preservation
- Validation

**Usage:**
```python
from format_utils import detect_format, convert_image

fmt = detect_format("image.tif")
convert_image("input.tif", "output.jpg", quality=95)
```

---

## Pipeline Utilities

### 21. `create_board_textures.py`
**Purpose**: Texture generation for materials

**Features:**
- Procedural textures
- Tileable patterns
- Material presets
- Export to PNG/TIFF

**Usage:**
```bash
python create_board_textures.py \
  --material wood \
  --size 2048x2048 \
  --output textures/
```

---

### 22. `synthetic_viewer.py`
**Purpose**: Training data visualization

**Features:**
- Synthetic dataset browser
- Annotation overlay
- Batch preview
- Export selected

**Usage:**
```bash
python -c "from transformation_portal.perceptual.synthetic_viewer import SyntheticViewer; print(SyntheticViewer())"
```

---

## Helper Scripts (scripts/ directory)

### 23. `scripts/download_samples.py`
**Purpose**: Download sample images for testing

**Features:**
- Test image sets
- Various formats
- Metadata included
- Progress tracking

**Usage:**
```bash
python scripts/download_samples.py \
  --category architectural \
  --count 10
```

---

### 24. `scripts/install_modules.py`
**Purpose**: Python module installation helper

**Features:**
- Dependency checking
- Virtual environment setup
- Optional extras installation
- Version compatibility

**Usage:**
```bash
python scripts/install_modules.py --extras ml,tiff
```

---

## Best Practices

### General Guidelines

1. **Always use virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Check dependencies first**
   ```bash
   pip install -r requirements.txt
   ```

3. **Use --dry-run for validation**
   ```bash
   python script.py --input file.jpg --dry-run
   ```

4. **Enable verbose output for debugging**
   ```bash
   python script.py --verbose
   ```

5. **Backup originals before batch processing**
   ```bash
   cp -r input/ input_backup/
   ```

### Performance Tips

- Use `--workers` for parallel processing
- Enable caching with `--cache` when available
- Process in batches for large datasets
- Monitor memory with `--profile`

### Quality Control

- Validate outputs with `--check-quality`
- Compare before/after with visualization scripts
- Keep processing logs with `--log output.log`
- Use presets as starting points, then fine-tune

---

## Script Dependencies

### Core Dependencies
- Python 3.11+
- NumPy, Pillow, scipy
- PyTorch 2.0+ (for ML pipelines)
- FFmpeg 6+ (for video processing)

### Optional Dependencies
- transformers (Depth Anything V2)
- diffusers (Stable Diffusion, ControlNet)
- realesrgan (4x upscaling)
- tifffile (16-bit TIFF support)
- coremltools (Apple Neural Engine)

### Install All
```bash
pip install -e ".[all]"
```

---

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install -r requirements.txt
pip install -e ".[ml]"
```

**Memory Issues:**
```bash
python script.py --batch-size 1 --workers 1
```

**GPU Not Detected:**
```bash
python script.py --device cpu
```

**Model Download Fails:**
```bash
python scripts/setup/install_models.py --dry-run
python scripts/setup/install_models_auto.py --skip-optional
```

---

## Quick Reference

| Task | Script | Command |
|------|--------|---------|
| Enhance render | `lux_render_pipeline.py` | `--input img.jpg --upscale 4` |
| Grade video | `luxury_video_master_grader.py` | `--preset signature_estate` |
| Batch TIFF | `luxury_tiff_batch_processor.py` | `--input dir/ --workers 4` |
| Depth map | `depth_anything_v2.py` | `--model small --device mps` |
| Material viz | `scripts/utilities/visualize_material_assignments.py` | `--clusters 8` |
| Install models | `scripts/setup/install_models.py` | `--dry-run` |
| Code audit | `codebase_philosophy_auditor.py` | `--target src/` |

---

**Last Updated**: 2026-06-03
**Version**: 2.0.0
**Total Scripts**: 24+
