# Format Support Overview

**Transformation Portal** | Complete File Format Documentation

---

## Document Index

This directory contains comprehensive documentation for file format support across all Transformation Portal pipelines.

### 📚 Available Documentation

| Document | Description | Best For |
|----------|-------------|----------|
| [SUPPORTED_FILE_FORMATS.md](../SUPPORTED_FILE_FORMATS.md) | Complete format specification (350+ lines) | Detailed reference, technical specs |
| [FILE_FORMAT_QUICK_REFERENCE.md](../FILE_FORMAT_QUICK_REFERENCE.md) | One-page quick reference (150+ lines) | Daily use, printable cheat sheet |
| [README.md](../README.md) | Main project documentation | Getting started, overview |

---

## Quick Answer

### "What file formats are supported?"

**Images**: PNG, JPEG, TIFF, WebP, BMP + 8 more (13 total)  
**Videos**: MP4, MOV, AVI, MKV, WebM + 2 more (7 total)

All formats are **case-insensitive** and work across all pipelines with automatic format detection.

---

## Getting Started

### 1. Check Your Files

Use the validation tool to check if your files are supported:

```bash
# Single file
python examples/validate_file_formats.py image.jpg

# Entire directory
python examples/validate_file_formats.py --scan ./images/

# Show all supported formats
python examples/validate_file_formats.py --formats
```

### 2. Install Format Support

For maximum quality (16-bit TIFF support):

```bash
# Install TIFF support
pip install -e ".[tiff]"

# Or install all extras
pip install -e ".[all]"
```

For video processing:

```bash
# Linux/Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 3. Process Your Files

Choose the appropriate pipeline based on your file type:

```bash
# TIFF files → Luxury TIFF Batch Processor
python luxury_tiff_batch_processor.py input/ output/ --preset signature

# General images → Depth Pipeline
python depth_pipeline/pipeline.py --input render.jpg --output enhanced.jpg

# AI enhancement → Lux Render Pipeline
python lux_render_pipeline.py --input bedroom.jpg --out ./final

# Videos → Video Master Grader
python luxury_video_master_grader.py --input tour.mp4 --output graded.mov
```

---

## Format Validation (Programmatic)

### Python API

```python
from format_utils import (
    validate_format,
    get_format_info,
    suggest_output_format,
    UnsupportedFormatError,
)

# Validate format
try:
    validate_format('photo.jpg', 'image')
    print("✅ Supported")
except UnsupportedFormatError as e:
    print(f"❌ {e}")

# Get detailed information
info = get_format_info('render.tiff')
print(f"Format: {info['extension']}")
print(f"Luxury grade: {info['is_luxury']}")
for rec in info['recommendations']:
    print(f"  • {rec}")

# Get smart output suggestion
output_ext = suggest_output_format('input.jpg', preserve_quality=True)
print(f"Suggested output: {output_ext}")
```

### Command Line

```bash
# Validate and get recommendations
python examples/validate_file_formats.py photo.jpg

# Scan directory for supported files
python examples/validate_file_formats.py --scan ./images/

# List all supported formats
python examples/validate_file_formats.py --formats
```

---

## Format Support Matrix

### By Pipeline

| Pipeline | Optimized For | Recommended Formats |
|----------|--------------|---------------------|
| **Luxury TIFF Batch Processor** | 16-bit precision | `.tiff`, `.tif` |
| **Depth Pipeline** | Depth-aware processing | `.png`, `.tiff`, `.jpg` |
| **Lux Render Pipeline** | AI enhancement | `.png`, `.tiff` |
| **Material Response** | Surface enhancement | `.tiff`, `.png` |
| **Video Master Grader** | Color grading | `.mov`, `.mp4` |

### By Use Case

| Use Case | Input Format | Processing | Output Format |
|----------|--------------|------------|---------------|
| **Architectural Rendering** | TIFF (16-bit) | Depth Pipeline | PNG/TIFF |
| **Real Estate Photography** | TIFF (16-bit) | TIFF Batch Processor | TIFF (16-bit) |
| **Web Delivery** | Any | Any Pipeline | JPEG (95%) |
| **Print Production** | TIFF (16-bit) | Any Pipeline | TIFF (16-bit) |
| **Video Production** | MOV/MP4 | Video Grader | ProRes MOV |

---

## Common Scenarios

### Scenario 1: Professional Photography Workflow

```bash
# Step 1: Convert RAW to 16-bit TIFF (use Lightroom/Darktable)
# Step 2: Batch process TIFFs
python luxury_tiff_batch_processor.py \
    ./raw_tiffs/ \
    ./enhanced_tiffs/ \
    --preset signature_estate \
    --recursive

# Step 3: Export web versions as JPEG
# (Use --compression option or separate export)
```

### Scenario 2: Architectural Rendering Pipeline

```bash
# Step 1: Export renders as PNG or TIFF
# Step 2: Generate depth maps (if available)
# Step 3: Enhance with depth awareness
python depth_pipeline/pipeline.py \
    --input render.png \
    --output enhanced/ \
    --config config/interior_preset.yaml

# Step 4: Optional AI refinement
python lux_render_pipeline.py \
    --input enhanced/render_enhanced.png \
    --out final/ \
    --prompt "luxury interior, natural daylight"
```

### Scenario 3: Video Color Grading

```bash
# Step 1: Export video from NLE as ProRes or H.264
# Step 2: Apply color grading
python luxury_video_master_grader.py \
    --input property_tour.mp4 \
    --output graded_tour.mov \
    --preset golden_hour_courtyard \
    --tone-map Hable

# Step 3: Export for web (if needed)
ffmpeg -i graded_tour.mov -c:v libx265 -crf 23 web_tour.mp4
```

---

## Troubleshooting

### "Unsupported format" Error

**Check**: File extension is recognized
```bash
python examples/validate_file_formats.py yourfile.xyz
```

**Solution**: Convert to supported format or check for typos

---

### "16-bit support requires tifffile" Warning

**Check**: TIFF support installation
```bash
python -c "import tifffile; print('✓ Installed')"
```

**Solution**: Install TIFF support
```bash
pip install -e ".[tiff]"
```

---

### "FFmpeg not found" Error

**Check**: FFmpeg installation
```bash
ffmpeg -version
```

**Solution**: Install FFmpeg
```bash
# Linux/Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org
```

---

### RAW Files Not Working

**Problem**: Camera RAW files (.cr2, .nef, .arw) not supported

**Solution**: Convert to 16-bit TIFF using:
- Adobe Lightroom
- Darktable (open-source)
- RawTherapee (open-source)
- Capture One

Then process the TIFF files normally.

---

## Performance Notes

### Format Processing Speed (4K image on M4 Max)

| Format | Load Time | Save Time | Memory Usage |
|--------|-----------|-----------|--------------|
| JPEG | 50ms | 100ms | 32 MB |
| PNG | 100ms | 200ms | 32 MB |
| TIFF 8-bit | 100ms | 150ms | 32 MB |
| TIFF 16-bit | 200ms | 300ms | 64 MB |

### Batch Processing Throughput

| Pipeline | Throughput | Bottleneck |
|----------|------------|------------|
| Format validation | 10,000+ files/sec | I/O |
| TIFF Batch Processor | 400-600 images/hour | CPU |
| Depth Pipeline | 400-600 images/hour | GPU/CoreML |
| Lux Render (AI) | 50-100 images/hour | GPU/VRAM |

---

## Best Practices

### ✅ Do

- **Use TIFF** for professional/print work (16-bit precision)
- **Use PNG** for web renders (lossless, good compression)
- **Use JPEG** for web delivery (95%+ quality)
- **Install tifffile** for full 16-bit support
- **Validate formats** before batch processing
- **Preserve originals** during processing

### ❌ Don't

- Use JPEG for intermediate processing (lossy)
- Mix formats in batch operations (inconsistent results)
- Process RAW files directly (convert to TIFF first)
- Ignore format warnings (may indicate quality loss)
- Use BMP unless necessary (inefficient format)

---

## Format Conversion

### Convert to Supported Format

```bash
# JPEG to PNG (lossless upgrade)
python -c "from PIL import Image; Image.open('photo.jpg').save('photo.png')"

# PNG to TIFF (16-bit upgrade with tifffile)
python -c "from PIL import Image; import numpy as np; import tifffile; \
img = np.array(Image.open('render.png')); \
tifffile.imwrite('render.tiff', img)"

# TIFF to JPEG (web delivery)
python -c "from PIL import Image; Image.open('photo.tiff').save('photo.jpg', quality=95)"
```

### Batch Conversion

```bash
# PNG to TIFF
find input/ -name "*.png" -exec python -c \
  "from PIL import Image; import sys; \
   Image.open(sys.argv[1]).save(sys.argv[1].replace('.png', '.tiff'))" {} \;

# TIFF to JPEG (95% quality)
find input/ -name "*.tiff" -exec python -c \
  "from PIL import Image; import sys; \
   Image.open(sys.argv[1]).save(sys.argv[1].replace('.tiff', '.jpg'), quality=95)" {} \;
```

---

## API Reference

### Core Functions

```python
# Format detection
is_supported_image_format(path) -> bool
is_supported_video_format(path) -> bool
is_supported_tiff_format(path) -> bool
is_luxury_format(path) -> bool

# Validation
validate_format(path, allowed_types='image', raise_error=True) -> bool

# Information
get_format_info(path) -> dict
suggest_output_format(path, preserve_quality=True) -> str
get_supported_formats_summary() -> dict

# Utilities
normalize_extension(path) -> str
format_help_text(format_type='image') -> str
```

Full API documentation in `format_utils.py` module.

---

## Testing

### Run Format Tests

```bash
# Run format validation tests
pytest tests/test_format_utils.py -v

# Run with coverage
pytest tests/test_format_utils.py --cov=format_utils --cov-report=term

# Run specific test class
pytest tests/test_format_utils.py::TestValidateFormat -v
```

### Manual Testing

```bash
# Test format detection
python format_utils.py

# Test validation tool
python examples/validate_file_formats.py test.jpg

# Test directory scanning
python examples/validate_file_formats.py --scan ./test_images/
```

---

## Additional Resources

### Internal Documentation

- [SUPPORTED_FILE_FORMATS.md](../SUPPORTED_FILE_FORMATS.md) - Complete format specification
- [FILE_FORMAT_QUICK_REFERENCE.md](../FILE_FORMAT_QUICK_REFERENCE.md) - Printable quick reference
- [README.md](../README.md) - Project overview
- [DEPTH_PIPELINE_README.md](../DEPTH_PIPELINE_README.md) - Depth pipeline specifics

### External Resources

- [PIL/Pillow Documentation](https://pillow.readthedocs.io/) - Image format support
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Video formats
- [TIFF Specification](https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf) - TIFF format details

---

## Support

### Getting Help

1. **Check documentation** - Most questions answered in SUPPORTED_FILE_FORMATS.md
2. **Validate format** - Use `python examples/validate_file_formats.py yourfile`
3. **Check tests** - See `tests/test_format_utils.py` for usage examples
4. **Report issues** - GitHub Issues for bugs or feature requests

### Common Questions

**Q: Do I need special software to process images?**  
A: No, all required libraries are in requirements.txt. Optional: tifffile for 16-bit TIFF.

**Q: What's the best format for...?**  
A: See "Format Support Matrix" above or use `get_format_info()` function.

**Q: Can I batch convert formats?**  
A: Yes, see "Format Conversion" section above.

**Q: Is there a file size limit?**  
A: No hard limit, depends on available RAM. 4K+ images need 8-16GB for ML pipelines.

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Maintainer**: Transformation Portal Team

---

*For questions or improvements to this documentation, please open an issue on GitHub.*
