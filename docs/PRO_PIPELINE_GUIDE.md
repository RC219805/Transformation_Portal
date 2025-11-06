# Professional Pipeline - User Guide

## Overview

The **Transformation Portal Professional Pipeline** is a fully-integrated orchestrator that combines all major pipeline components into a unified, production-ready workflow for luxury real estate rendering and architectural visualization.

## What It Does

The Pro Pipeline integrates five core processing stages:

1. **Depth-Aware Processing** - Depth Anything V2 with CoreML optimization
2. **AI Enhancement** - Stable Diffusion XL, ControlNet, Real-ESRGAN
3. **Material Response** - Physics-based surface enhancement
4. **Professional Color Grading** - LUT application, AgX tone mapping
5. **Finishing** - Sharpening, clarity, micro-contrast

## Quick Start

### Installation

```bash
# Ensure the package is installed
cd /path/to/Transformation_Portal
pip install -e .

# Install optional dependencies
pip install -e ".[ml]"    # For AI enhancement
pip install -e ".[tiff]"  # For 16-bit TIFF support
```

### Basic Usage

```bash
# Process a single image with a preset
python pro_pipeline.py process image.jpg --preset architectural-hero --out ./enhanced

# Batch process a directory
python pro_pipeline.py batch ./renders --preset interior-dramatic --out ./final

# List available presets
python pro_pipeline.py list-presets

# Show version and capabilities
python pro_pipeline.py version
```

## Available Presets

The Pro Pipeline includes 10 professionally-tuned presets:

| Preset | Use Case | Stages Enabled |
|--------|----------|----------------|
| `architectural-hero` | Dramatic hero shots | All stages (full enhancement) |
| `interior-dramatic` | High-contrast interiors | Depth + Material + Grading + Finishing |
| `exterior-golden-hour` | Warm outdoor scenes | All stages with golden hour LUT |
| `aerial-estate` | Aerial photography | Depth + Material + Grading (no AI) |
| `pool-luxury` | Pool/water features | Depth + Material + Grading + Finishing |
| `kitchen-bright` | Bright kitchen spaces | Conservative enhancement |
| `bedroom-cozy` | Warm bedroom aesthetic | Material + Grading with warm tone |
| `bathroom-spa` | Spa-like bathrooms | Material + Clarity enhancement |
| `courtyard-natural` | Natural outdoor spaces | Balanced enhancement |
| `custom` | Manual configuration | Configure each stage individually |

## Command Reference

### `process` - Single Image

Process a single image through the pipeline.

```bash
python pro_pipeline.py process INPUT_IMAGE [OPTIONS]
```

**Arguments:**
- `INPUT_IMAGE` - Path to the input image (required)

**Options:**
- `--out, -o PATH` - Output directory (default: `./output`)
- `--preset, -p PRESET` - Pipeline preset to use (default: `architectural-hero`)
- `--format, -f FORMAT` - Output format: jpg, png, tiff (default: `tiff`)
- `--bits INTEGER` - Bit depth for TIFF: 8, 16, 32 (default: `16`)
- `--device DEVICE` - Device: auto, cpu, cuda, mps (default: `auto`)
- `--quality, -q QUALITY` - Quality: draft, standard, high, ultra (default: `high`)

**Stage Toggles:**
- `--depth-aware / --no-depth` - Enable/disable depth processing (default: enabled)
- `--ai-enhance / --no-ai` - Enable/disable AI enhancement (default: enabled)
- `--material-response / --no-material` - Enable/disable Material Response (default: enabled)
- `--color-grading / --no-grading` - Enable/disable color grading (default: enabled)
- `--finishing / --no-finishing` - Enable/disable finishing (default: enabled)

**Other Options:**
- `--keep-intermediates` - Keep intermediate outputs
- `--dry-run` - Show what would be done without processing

**Examples:**

```bash
# Basic usage with preset
python pro_pipeline.py process render.jpg --preset interior-dramatic --out ./final

# Disable AI enhancement for faster processing
python pro_pipeline.py process render.jpg --preset architectural-hero --no-ai --out ./quick

# Export as high-quality JPEG
python pro_pipeline.py process render.jpg --format jpg --out ./deliverables

# Process with only specific stages
python pro_pipeline.py process render.jpg --depth-aware --material-response --no-ai --no-grading

# Dry run to preview configuration
python pro_pipeline.py process render.jpg --preset exterior-golden-hour --dry-run
```

### `batch` - Multiple Images

Process multiple images in a directory.

```bash
python pro_pipeline.py batch INPUT_DIR [OPTIONS]
```

**Arguments:**
- `INPUT_DIR` - Directory containing input images (required)

**Options:**
- Same as `process` command, plus:
- `--workers, -w INTEGER` - Number of parallel workers (default: `4`)
- `--pattern PATTERN` - File pattern to match (default: `*.{jpg,jpeg,png,tiff,tif}`)

**Examples:**

```bash
# Batch process with preset
python pro_pipeline.py batch ./renders --preset architectural-hero --out ./final

# Process with multiple workers
python pro_pipeline.py batch ./renders --preset interior-dramatic --workers 8 --out ./final

# Process only TIFF files
python pro_pipeline.py batch ./renders --pattern "*.tiff" --out ./final
```

### `list-presets` - Show Presets

List all available presets with descriptions.

```bash
python pro_pipeline.py list-presets
```

### `version` - Show Version

Display version and capabilities.

```bash
python pro_pipeline.py version
```

## Performance

### Expected Processing Times

**Single Image (4K resolution):**
- With AI Enhancement: 2-5 minutes (M4 Max with CoreML + MPS)
- Without AI Enhancement: 30-60 seconds (M4 Max with CoreML)
- CPU only: 5-10 minutes

**Batch Processing:**
- 400-600 images/hour with optimizations (no AI)
- 50-100 images/hour with full AI enhancement
- 200-300 images/hour with selective AI

**Performance Tips:**
1. Use `--no-ai` for faster processing when AI refinement isn't needed
2. Increase `--workers` for batch processing (optimal: CPU cores - 2)
3. Use `--quality standard` instead of `high` for drafts
4. Enable CoreML on Apple Silicon for 3-5x depth processing speedup
5. Use CUDA or MPS acceleration when available

### Device Selection

The pipeline auto-detects the best available device:

- **Apple Silicon (M1/M2/M3/M4)**: Uses MPS for PyTorch, CoreML for depth
- **NVIDIA GPU**: Uses CUDA for all ML operations
- **CPU only**: Falls back to CPU processing

Manual device selection:
```bash
python pro_pipeline.py process image.jpg --device mps  # Force MPS on Mac
python pro_pipeline.py process image.jpg --device cuda  # Force CUDA
python pro_pipeline.py process image.jpg --device cpu   # Force CPU
```

## Configuration

### Using Configuration Files

Create a custom configuration file based on `config/pro_pipeline_config.yaml`:

```yaml
# my_config.yaml
global:
  quality: ultra
  output_format: tiff
  bit_depth: 16

stages:
  depth:
    model: depth-anything-v2-large
    clarity:
      amount: 0.20
  
  material:
    strength: 0.75
  
  grading:
    contrast: 1.15
    saturation: 1.12
```

Use the configuration:
```bash
python pro_pipeline.py process image.jpg --config my_config.yaml
```

### Custom Presets

You can define custom presets in the configuration file:

```yaml
presets:
  my-custom-preset:
    description: "Custom preset for my workflow"
    stages:
      depth:
        enabled: true
        clarity:
          amount: 0.25
      material:
        enabled: true
        strength: 0.80
      grading:
        enabled: true
        contrast: 1.15
```

## Output Formats

### TIFF (Default)

Best for:
- Professional archival
- 16-bit color precision
- Metadata preservation
- Further editing

Settings:
- Bit depth: 8, 16, or 32-bit
- Compression: Adobe Deflate (lossless)
- Metadata: EXIF, IPTC, XMP preserved

### JPEG

Best for:
- Client deliverables
- Web use
- Email sharing
- Smaller file sizes

Settings:
- Quality: 95 (optimized)
- Progressive encoding
- EXIF metadata preserved

### PNG

Best for:
- Web graphics
- Transparency needs
- Lossless compression
- Medium file sizes

Settings:
- Compression level: 6 (balanced)
- Metadata: Limited (PNG text chunks)

## Troubleshooting

### Common Issues

**1. "No module named 'transformation_portal'"**

Solution:
```bash
cd /path/to/Transformation_Portal
pip install -e .
```

**2. Slow processing times**

Solutions:
- Use `--no-ai` to skip AI enhancement
- Reduce `--quality` to `standard`
- Check that CUDA/MPS is being used (see console output)
- Reduce image resolution before processing

**3. Out of memory errors**

Solutions:
- Reduce `--workers` count
- Use `--quality draft` or `standard`
- Process smaller batches
- Close other applications

**4. AI enhancement not working**

Requirements:
- PyTorch must be installed
- GPU with 8GB+ VRAM recommended
- Models must be downloaded (automatic on first run)

**5. Depth processing very slow**

Solutions:
- Install CoreML models on Apple Silicon
- Ensure `--device mps` or `--device cuda` is being used
- Check that models are cached (second run should be faster)

### Debug Mode

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python pro_pipeline.py process image.jpg --preset architectural-hero
```

### Performance Profiling

Generate a performance report:
```bash
python pro_pipeline.py batch ./renders --preset architectural-hero --out ./final 2>&1 | tee performance.log
```

The report will show:
- Total processing time
- Per-stage timing
- Average time per image
- Throughput (images/hour)

## Advanced Usage

### Pipeline Chaining

Chain multiple presets for complex workflows:

```bash
# Step 1: Initial enhancement
python pro_pipeline.py process image.jpg --preset interior-dramatic --out ./stage1

# Step 2: Additional AI refinement
python pro_pipeline.py process stage1/image_interior-dramatic.tiff \\
  --preset architectural-hero --depth-aware --no-material --out ./final
```

### Custom Stage Configuration

Override preset settings with command-line options:

```bash
# Use preset but disable specific stages
python pro_pipeline.py process image.jpg --preset architectural-hero \\
  --no-ai --no-finishing --out ./custom
```

### Integration with Other Pipelines

The Pro Pipeline can be integrated with existing workflows:

```python
# In your Python script
from pro_pipeline import ProPipeline, ProPipelineConfig, PipelinePreset

# Create configuration
config = ProPipelineConfig(
    input_path=Path("image.jpg"),
    output_dir=Path("./output"),
    preset=PipelinePreset.ARCHITECTURAL_HERO,
)

# Process
pipeline = ProPipeline(config)
result = pipeline.process_image(Path("image.jpg"))
```

## Best Practices

### For Architectural Rendering

1. **Use depth-aware processing** - Provides natural depth cues
2. **Enable Material Response** - Enhances surface realism
3. **Apply appropriate LUT** - Match location aesthetic
4. **Use 16-bit TIFF** - Preserve tonal range
5. **Enable finishing** - Professional polish

Recommended preset: `architectural-hero`

### For Interior Photography

1. **Adjust exposure** - Interior scenes often need lifting
2. **Enable clarity** - Reveals detail without over-sharpening
3. **Use conservative Material Response** - Avoid over-enhancement
4. **Apply warm color grading** - Creates inviting atmosphere
5. **Skip AI enhancement** - Preserve natural look

Recommended preset: `interior-dramatic` or `kitchen-bright`

### For Aerial Photography

1. **Enable atmospheric haze** - Natural aerial perspective
2. **Boost clarity** - Compensate for atmospheric effects
3. **Skip AI enhancement** - Preserve scale and context
4. **Enhance greens and blues** - Vegetation and sky
5. **Use depth-guided processing** - Separate foreground/background

Recommended preset: `aerial-estate`

### For Client Deliverables

1. **Use architectural-hero preset** - Maximum quality
2. **Export as TIFF** - Archival master
3. **Also export JPEG** - Preview/sharing
4. **Include metadata** - Copyright, description
5. **Batch process** - Consistent look across set

## FAQ

**Q: Which preset should I use?**  
A: Start with `architectural-hero` for maximum quality. For specific use cases (interiors, aerials, pools), use the corresponding preset.

**Q: How long does processing take?**  
A: 30 seconds to 5 minutes per 4K image, depending on stages enabled and hardware.

**Q: Can I process RAW files?**  
A: Convert RAW to TIFF first using Adobe Camera Raw or similar. The pipeline works best with TIFF/PNG/JPEG.

**Q: Does it work on Windows?**  
A: Yes! The pipeline is cross-platform (Windows, macOS, Linux).

**Q: Can I use multiple GPUs?**  
A: Currently single-GPU only. Use `--workers` for parallel CPU processing.

**Q: How do I customize a preset?**  
A: Create a custom config file based on `config/pro_pipeline_config.yaml` and modify the preset settings.

**Q: What's the difference between quality levels?**  
A: 
- `draft`: Fast preview (reduced resolution, fewer steps)
- `standard`: Good balance (default settings)
- `high`: Maximum quality (more steps, higher resolution)
- `ultra`: Extreme quality (very slow, experimental)

**Q: Can I process videos?**  
A: Not directly. Extract frames, process with batch mode, then reassemble. Or use `luxury_video_master_grader.py` for video-specific workflows.

## Support

**Documentation:**
- Full Pipeline Documentation: `docs/PIPELINE_OPERATIONS_GUIDE.md`
- Architecture Guide: `docs/ARCHITECTURE.md`
- Performance Tips: `docs/PERFORMANCE_OPTIMIZATION.md`

**Getting Help:**
- GitHub Issues: https://github.com/RC219805/Transformation_Portal/issues
- GitHub Discussions: https://github.com/RC219805/Transformation_Portal/discussions

**Reporting Bugs:**
1. Include console output with `--dry-run`
2. Specify hardware (CPU, GPU, RAM)
3. Attach sample image if possible
4. Describe expected vs actual behavior

## Version History

**v1.0.0** (November 2025)
- Initial release
- 10 professionally-tuned presets
- 5-stage integrated pipeline
- Batch processing support
- Comprehensive CLI interface
- Full test coverage
