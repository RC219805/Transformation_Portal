# Transformation Portal Examples

This directory contains example scripts demonstrating how to use various features of the Transformation Portal toolkit.

## Prerequisites

Before running any examples, you must install the package in development mode:

```bash
# From the repository root directory
pip install -e .

# Or with optional dependencies for specific features
pip install -e ".[ml]"      # For ML/AI features (depth estimation, enhancement)
pip install -e ".[tiff]"    # For 16-bit TIFF processing
pip install -e ".[all]"     # Install all features
```

This installation step is **required** for the examples to work correctly. The examples import modules from the package, which must be installed to be accessible.

## Available Examples

### Depth Pipeline Examples

#### `simple_process.py`
Basic single-image processing with the depth-aware pipeline.

```bash
python examples/simple_process.py input.jpg output/
```

Demonstrates:
- Loading a default configuration
- Processing a single image with depth estimation
- Saving enhanced results with depth maps

#### `custom_pipeline.py`
Build a custom processing pipeline with specific parameters.

```bash
python examples/custom_pipeline.py input.jpg output.jpg
```

Demonstrates:
- Manual pipeline construction
- Custom processor parameters
- Depth-aware denoising, tone mapping, atmospheric effects
- Depth visualization

#### `batch_process.py`
Batch process multiple images with progress tracking.

```bash
python examples/batch_process.py input_dir/ output_dir/ --preset interior
```

Options:
- `--preset`: Choose from `default`, `interior`, or `exterior`
- `--pattern`: File pattern (e.g., `"*.png"`, `"render_*.jpg"`)
- `--no-depth`: Skip saving depth maps
- `--no-viz`: Skip depth visualizations

Demonstrates:
- Batch processing with presets
- Progress tracking
- Cache statistics

### Format Validation

#### `validate_file_formats.py`
Validate image and video formats before processing.

```bash
# Validate single file
python examples/validate_file_formats.py render.jpg

# Scan directory
python examples/validate_file_formats.py --scan images/

# Show all supported formats
python examples/validate_file_formats.py --formats
```

Demonstrates:
- Format validation and recommendations
- Supported format detection
- Pipeline recommendations
- Batch processing suggestions

### VFX and Enhancement

#### `vfx_extension_example.py`
Apply depth-guided visual effects to architectural renderings.

```bash
python examples/vfx_extension_example.py
```

Demonstrates:
- Available VFX presets
- CLI usage examples
- Depth-guided effects application

#### `enhance_aerial_example.py`
Apply MBAR board materials to aerial photographs.

```bash
python examples/enhance_aerial_example.py
```

Note: Requires an input image at `input_images/montecito_aerial.jpg`.

Demonstrates:
- Material-based aerial enhancement
- Custom texture paths
- Color clustering for material assignment

## Troubleshooting

### Module Not Found Errors

If you see errors like:
```
ModuleNotFoundError: No module named 'depth_pipeline'
```

This means the package is not installed. Run:
```bash
pip install -e .
```

### Missing Optional Dependencies

Some examples require optional dependencies:

- Depth processing: `pip install -e ".[ml]"`
- TIFF processing: `pip install -e ".[tiff]"`
- All features: `pip install -e ".[all]"`

### FFmpeg for Video Processing

Video-related examples require FFmpeg to be installed:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Development Workflow

When developing or modifying examples:

1. Install in editable mode: `pip install -e ".[dev]"`
2. Make changes to example scripts
3. Test by running the examples
4. Ensure examples follow the established patterns

## Additional Resources

- Main README: `../README.md`
- Depth Pipeline Documentation: `../docs/depth_pipeline/DEPTH_PIPELINE_README.md`
- Supported File Formats: `../SUPPORTED_FILE_FORMATS.md`
- Architecture Overview: `../docs/ARCHITECTURE.md`

## Notes

- All examples assume you're running them from the repository root (not from the `examples/` directory)
- Example files and outputs should be placed in appropriate directories (e.g., `input_images/`, `processed_images/`)
- Some examples require specific input files to exist
