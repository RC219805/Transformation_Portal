# Unified Luxury Pipeline (Compatibility)

**Compatibility/convenience image-finishing facade for luxury real estate rendering, architectural visualization, and editorial post-production.**

For governed production depth, PBR, Materials V3, APEX, run-card, and portal/orchestrator workflows, use `lux-depth-v3` and the `transformation_portal.lux_depth_v3` package. This pipeline remains available for legacy callers and lightweight multi-format finishing workflows.

## Overview

The Unified Luxury Pipeline combines the best aspects of earlier image-finishing pipeline implementations into a single compatibility surface:

- **Multi-format output generation** with proper bit-depth handling (from `premium_pipeline_fixed.py`)
- **Modular stage-based architecture** with graceful failure handling (from `pro_pipeline.py`)
- **Architectural context intelligence** and scene detection (from `context_aware_pro_pipeline.py`)
- **VFX capabilities** and depth-aware enhancements (from `realize_v8_unified.py`)

## Key Features

### 🎯 Multi-Format Output System
Generate all required deliverables in one pass:
- **Master TIFF** (16-bit, full resolution) - archival master
- **Web 4K** (3840px JPEG, Q96) - web-optimized
- **Print 8K** (7680px JPEG, Q98) - print-ready
- **Social** (1080px JPEG, Q92) - Instagram/social media
- **Magazine** (2048px JPEG, Q95) - magazine layouts

### 🚀 Processing Profiles
Trade quality for speed based on your needs:
- **PREMIUM** - Maximum quality (2-5 min/image, full AI pipeline)
- **BALANCED** - Quality/speed balance (30-90 sec/image, selective AI)
- **PERFORMANCE** - Fast preview (10-30 sec/image, minimal AI)

### 🏗️ Scene Intelligence
Automatic detection and optimization for:
- **Interior** - Enhanced clarity, controlled contrast
- **Exterior** - Boosted saturation, atmospheric effects
- **Aerial** - Maximum clarity, aerial perspective
- **Auto** - Automatic scene type detection

### 🎨 Processing Stages
Modular pipeline with optional stages:
1. **Load & Validate** (required) - Image loading and metadata extraction
2. **Scene Detection** (optional) - Automatic scene type detection
3. **Depth Processing** (optional) - Depth Anything V2 with CoreML optimization
4. **Material Response** (optional) - Physics-based surface enhancement
5. **VFX Effects** (optional) - Bloom, fog, depth-of-field
6. **Color Grading** (optional) - Professional color grading with LUT support
7. **Output Generation** (required) - Multi-format output rendering

### 📊 Statistics & Monitoring
- Per-stage timing breakdown
- Success/failure tracking
- Memory usage monitoring
- JSON statistics export

## Quick Start

### Basic Usage

```python
from pathlib import Path
from transformation_portal.pipelines import (
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
    ProcessingProfile,
    SceneType,
    OutputFormat
)

# Create configuration
config = UnifiedPipelineConfig(
    scene_type=SceneType.INTERIOR,
    profile=ProcessingProfile.BALANCED,
    output_dir=Path("output")
)

# Initialize pipeline
pipeline = UnifiedLuxuryPipeline(config)

# Process image
outputs = pipeline.process(Path("input/kitchen.jpg"))

# Access outputs
print(f"Master: {outputs['master']}")
print(f"Web: {outputs['web']}")
print(f"Print: {outputs['print']}")
```

### Convenience Functions

```python
from transformation_portal.pipelines import process_luxury_render

# Quick single image processing
outputs = process_luxury_render(
    "input.jpg",
    profile=ProcessingProfile.PREMIUM
)
```

### Batch Processing

```python
from transformation_portal.pipelines import batch_process_luxury_renders

# Process entire directory
results = batch_process_luxury_renders(
    input_dir=Path("renders"),
    output_dir=Path("output"),
    profile=ProcessingProfile.BALANCED,
    parallel_io=True,
    io_prefetch_size=2,
    io_saver_workers=2,
)

print(f"Processed {len(results)} images")
```

`parallel_io` is opt-in for compatibility. It overlaps input loading and output
writing for batch runs while preserving result ordering and per-image failure
isolation. Keep it disabled when comparing against older timing baselines or
when debugging per-stage behavior.

## Configuration Guide

### UnifiedPipelineConfig Parameters

#### Scene Configuration
```python
config = UnifiedPipelineConfig(
    scene_type=SceneType.AUTO,  # AUTO, INTERIOR, EXTERIOR, AERIAL
    profile=ProcessingProfile.BALANCED  # PREMIUM, BALANCED, PERFORMANCE
)
```

#### Output Configuration
```python
config = UnifiedPipelineConfig(
    output_formats=[
        OutputFormat.MASTER_TIFF,
        OutputFormat.WEB_4K,
        OutputFormat.PRINT_8K
    ],
    output_dir=Path("output"),
    parallel_outputs=True  # Generate formats in parallel
)
```

#### Processing Stages
```python
config = UnifiedPipelineConfig(
    enable_depth=True,              # Depth-aware processing
    enable_material_response=True,  # Material Response technology
    enable_vfx=False,               # VFX effects (bloom, fog, DOF)
    enable_color_grading=True       # Professional color grading
)
```

#### Enhancement Parameters
```python
config = UnifiedPipelineConfig(
    exposure=0.15,      # Exposure adjustment (-2.0 to 2.0 EV)
    contrast=1.10,      # Contrast multiplier (0.5 to 2.0)
    saturation=1.05,    # Saturation multiplier (0.0 to 2.0)
    clarity=0.18        # Clarity enhancement (0.0 to 1.0)
)
```

#### Color Grading
```python
config = UnifiedPipelineConfig(
    lut_path=Path("assets/luts/film_emulation/Kodak_2393.cube"),
    lut_strength=0.7  # LUT strength (0.0 to 1.0)
)
```

#### Advanced Options
```python
config = UnifiedPipelineConfig(
    device="auto",              # auto, cpu, cuda, mps
    preserve_metadata=True,     # Preserve EXIF/IPTC/XMP
    parallel_outputs=True,      # Generate requested output formats in parallel
    parallel_io=False,          # Opt in to overlapped batch load/save I/O
    io_prefetch_size=2,         # Images to prefetch when parallel_io=True
    io_saver_workers=2,         # Background output workers when parallel_io=True
    save_intermediates=False    # Save depth maps, etc.
)
```

## Usage Examples

### Example 1: Premium Quality Hero Shot

```python
config = UnifiedPipelineConfig(
    scene_type=SceneType.INTERIOR,
    profile=ProcessingProfile.PREMIUM,
    output_formats=[
        OutputFormat.MASTER_TIFF,
        OutputFormat.PRINT_8K,
        OutputFormat.WEB_4K
    ],
    enable_depth=True,
    enable_material_response=True,
    exposure=0.15,
    contrast=1.10,
    saturation=1.05,
    clarity=0.18,
    save_intermediates=True
)

pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("greatroom.exr")

print(f"Processing time: {pipeline.stats.total_time:.2f}s")
pipeline.save_stats()
```

### Example 2: Fast Preview for Client Review

```python
config = UnifiedPipelineConfig(
    profile=ProcessingProfile.PERFORMANCE,
    output_formats=[OutputFormat.WEB_4K],
    enable_depth=False,
    enable_material_response=False,
    exposure=0.1,
    contrast=1.05
)

pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("preview.jpg")
# Fast processing: ~10-30 seconds
```

### Example 3: Social Media Workflow

```python
config = UnifiedPipelineConfig(
    profile=ProcessingProfile.BALANCED,
    output_formats=[
        OutputFormat.SOCIAL,   # 1080p for Instagram
        OutputFormat.WEB_4K    # 4K for website
    ],
    saturation=1.12,  # Boosted for social
    contrast=1.10,
    clarity=0.15,
    enable_material_response=True
)

pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("lifestyle_shot.jpg")
```

### Example 4: Print Production

```python
config = UnifiedPipelineConfig(
    profile=ProcessingProfile.PREMIUM,
    output_formats=[
        OutputFormat.MASTER_TIFF,  # 16-bit archive
        OutputFormat.PRINT_8K      # Print file
    ],
    contrast=1.08,      # Conservative for print
    saturation=1.05,
    clarity=0.12,
    preserve_metadata=True,
    save_intermediates=True
)

pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("architectural_hero.exr")
```

### Example 5: Batch Processing with Auto Scene Detection

```python
config = UnifiedPipelineConfig(
    scene_type=SceneType.AUTO,  # Auto-detect each image
    profile=ProcessingProfile.BALANCED,
    output_dir=Path("output/batch")
)

pipeline = UnifiedLuxuryPipeline(config)

input_images = list(Path("renders").glob("*.jpg"))
results = pipeline.batch_process(input_images)

print(pipeline.stats.summary())
```

### Example 6: Custom LUT Grading

```python
config = UnifiedPipelineConfig(
    lut_path=Path("assets/luts/film_emulation/Kodak_2393.cube"),
    lut_strength=0.7,
    exposure=0.05,
    contrast=1.08,
    saturation=1.05
)

pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("exterior.jpg")
```

### Example 7: Runtime Parameter Overrides

```python
config = UnifiedPipelineConfig(
    profile=ProcessingProfile.BALANCED,
    output_dir=Path("output")
)

pipeline = UnifiedLuxuryPipeline(config)

# Process different images with different parameters
pipeline.process("bright_kitchen.jpg", exposure=0.0)
pipeline.process("dark_bedroom.jpg", exposure=0.3)
pipeline.process("dramatic_staircase.jpg", contrast=1.15, enable_vfx=True)
```

## Output Format Specifications

| Format | Resolution | Quality | DPI | Use Case |
|--------|-----------|---------|-----|----------|
| **Master TIFF** | Full (16-bit) | Lossless | 300 | Archival master, compositing |
| **Print 8K** | 7680px max | JPEG Q98 | 300 | Large format printing |
| **Web 4K** | 3840px max | JPEG Q96 | 72 | Website hero images |
| **Magazine 2K** | 2048px max | JPEG Q95 | 150 | Magazine layouts |
| **Social** | 1080px max | JPEG Q92 | 72 | Instagram, social media |

### Bit Depth Handling

- **Master TIFF**: 16-bit RGB when `tifffile` is available, 8-bit fallback
- **All JPEGs**: 8-bit RGB with 4:4:4 chroma (no subsampling)
- **Metadata**: EXIF, IPTC, XMP, GPS coordinates preserved when available
- **Color profiles**: ICC profiles preserved in JPEG outputs

## Processing Profiles

### PREMIUM Profile
- **Speed**: 2-5 minutes per 4K image (M4 Max)
- **AI Strength**: 0.45
- **AI Steps**: 30
- **Depth Model**: Large (best quality)
- **Material Strength**: 0.7
- **Best for**: Final deliverables, hero shots, print production

### BALANCED Profile (Default)
- **Speed**: 30-90 seconds per 4K image
- **AI Strength**: 0.35
- **AI Steps**: 20
- **Depth Model**: Base (good quality)
- **Material Strength**: 0.65
- **Best for**: Standard workflow, client presentations

### PERFORMANCE Profile
- **Speed**: 10-30 seconds per 4K image
- **AI Strength**: 0.25
- **AI Steps**: 15
- **Depth Model**: Small (fast)
- **Material Strength**: 0.5
- **Best for**: Quick previews, iteration, review cycles

## Scene Type Optimization

### Interior (Automatically Detected or Manual)
- Enhanced clarity (≥0.15)
- Controlled contrast (≤1.12)
- Focus on material surfaces
- Balanced color temperature

### Exterior
- Boosted saturation (+5%)
- Atmospheric haze effects
- Natural lighting enhancement
- Sky/foreground separation

### Aerial
- Maximum clarity (≥0.20)
- Aerial perspective effects
- Enhanced landscape features
- High saturation for vibrancy

### Auto Detection Heuristics
- **Aerial**: High sky ratio (>70% brightness in top third), low variance
- **Interior**: Low sky ratio (<40%), high brightness variance (>0.15)
- **Exterior**: Medium sky ratio, moderate variance

## Statistics & Monitoring

### Accessing Statistics

```python
pipeline = UnifiedLuxuryPipeline(config)
outputs = pipeline.process("input.jpg")

# Print summary
print(pipeline.stats.summary())

# Access individual metrics
print(f"Total time: {pipeline.stats.total_time:.2f}s")
print(f"Images processed: {pipeline.stats.images_processed}")
print(f"Stage times: {pipeline.stats.stage_times}")

# Save to JSON
stats_path = pipeline.save_stats()
```

### Statistics JSON Format

```json
{
  "total_time": 45.67,
  "images_processed": 10,
  "images_failed": 0,
  "stage_times": {
    "Load & Validate": 1.2,
    "Scene Detection": 0.8,
    "Depth Processing": 15.4,
    "Material Response": 8.3,
    "Color Grading": 5.1,
    "Output Generation": 14.9
  },
  "output_files": {
    "input/image1.jpg": [
      "output/image1_MASTER.tiff",
      "output/image1_WEB_4K.jpg"
    ]
  },
  "config": {
    "profile": "balanced",
    "scene_type": "auto",
    "device": "mps",
    "output_formats": ["master", "web", "print", "social", "magazine"]
  }
}
```

## Performance Benchmarks

### Apple M4 Max (16-core CPU, 40-core GPU, CoreML)

| Profile | Resolution | Time | Throughput |
|---------|-----------|------|------------|
| PREMIUM | 4K | 2-5 min | 120-300 images/hour |
| BALANCED | 4K | 30-90 sec | 400-1200 images/hour |
| PERFORMANCE | 4K | 10-30 sec | 1200-3600 images/hour |

*Throughput assumes batch processing with optimizations*

### Device Recommendations

- **Apple Silicon (M1/M2/M3/M4)**: Use `device="mps"` for optimal performance
- **NVIDIA GPU**: Use `device="cuda"` with CUDA 11.8+
- **CPU Only**: Use `device="cpu"` (slower but works everywhere)
- **Auto**: Use `device="auto"` for automatic detection

## Graceful Degradation

The pipeline uses a modular architecture with optional stages that can fail without halting processing:

### Required Stages
- **Load & Validate**: Must succeed (file loading)
- **Output Generation**: Must succeed (file writing)

### Optional Stages (Graceful Failure)
- **Scene Detection**: Falls back to manual scene type
- **Depth Processing**: Continues without depth awareness
- **Material Response**: Continues with basic enhancement
- **VFX Effects**: Continues without VFX
- **Color Grading**: Continues with basic adjustments

### Example Behavior

```python
config = UnifiedPipelineConfig(
    enable_depth=True,  # May fail without depth models
    enable_material_response=True
)

pipeline = UnifiedLuxuryPipeline(config)

# If depth processing fails:
# - Warning is logged
# - Depth stage is skipped
# - Pipeline continues with remaining stages
# - Outputs are still generated
outputs = pipeline.process("input.jpg")
```

## Metadata Preservation

### Preserved Metadata
- **EXIF**: Camera settings, timestamps, software
- **IPTC**: Copyright, keywords, captions
- **XMP**: Adobe metadata, ratings, labels
- **GPS**: Location coordinates
- **ICC Profile**: Color profiles for JPEGs

### Format Support
- **TIFF**: Full metadata support
- **JPEG**: EXIF, IPTC, XMP, ICC profile
- **PNG**: Limited metadata support

## Integration with Existing Pipelines

### Depth Pipeline Integration

The unified pipeline integrates with the existing Depth Anything V2 pipeline:

```python
# Depth processing is automatically used when enabled
config = UnifiedPipelineConfig(
    enable_depth=True,
    depth_model="depth-anything-v2-small"  # or base/large
)
```

### Material Response Integration

Integrates with the Material Response technology:

```python
config = UnifiedPipelineConfig(
    enable_material_response=True,
    # Automatically adjusts based on scene type
    scene_type=SceneType.INTERIOR
)
```

## Troubleshooting

### Common Issues

#### "Depth pipeline not available"
```
Solution: Install depth dependencies or disable depth processing:
config.enable_depth = False
```

#### "tifffile not available for 16-bit TIFF"
```
Solution: Install tifffile:
pip install tifffile imagecodecs

Or accept 8-bit fallback (automatic)
```

#### "Out of memory"
```
Solution: Reduce batch size or use PERFORMANCE profile:
config.profile = ProcessingProfile.PERFORMANCE
```

#### Slow processing on Apple Silicon
```
Solution: Ensure MPS is being used:
print(pipeline.device)  # Should show 'mps'

If showing 'cpu', install PyTorch with MPS support
```

## Testing

Comprehensive test suite with 38 tests:

```bash
# Run all tests
pytest tests/test_unified_luxury_pipeline.py -v

# Run specific test class
pytest tests/test_unified_luxury_pipeline.py::TestOutputGeneration -v

# Run with coverage
pytest tests/test_unified_luxury_pipeline.py --cov
```

### Test Coverage
- ✓ All output formats
- ✓ Bit depth preservation
- ✓ Metadata preservation
- ✓ Profile selection
- ✓ Scene detection accuracy
- ✓ Graceful failure handling
- ✓ Statistics tracking
- ✓ Batch processing
- ✓ Edge cases (small images, grayscale, etc.)

## API Reference

See module docstrings for detailed API documentation:

```python
from transformation_portal.pipelines import (
    UnifiedLuxuryPipeline,     # Main pipeline class
    UnifiedPipelineConfig,     # Configuration dataclass
    ProcessingProfile,         # PREMIUM, BALANCED, PERFORMANCE
    SceneType,                 # INTERIOR, EXTERIOR, AERIAL, AUTO
    OutputFormat,              # MASTER_TIFF, WEB_4K, PRINT_8K, etc.
    PipelineStage,             # Stage tracking
    PipelineStatistics,        # Statistics tracking
    process_luxury_render,     # Convenience function
    batch_process_luxury_renders  # Batch convenience function
)
```

## License

Part of the Transformation Portal project.

## Contributing

See main repository documentation for contribution guidelines.
