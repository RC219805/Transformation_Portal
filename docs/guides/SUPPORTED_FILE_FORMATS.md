# Supported File Formats

This document details the image and video file format support across all Transformation Portal pipelines.

## Quick Reference

| Pipeline | Primary Formats | Notes |
|----------|----------------|-------|
| **Luxury TIFF Batch Processor** | `.tif`, `.tiff` | 16-bit precision with `tifffile` |
| **Depth Pipeline** | `.jpg`, `.png`, `.tiff`, `.webp` | Any PIL-supported RGB format |
| **Lux Render Pipeline** | `.jpg`, `.png`, `.tiff`, `.bmp` | AI enhancement, any PIL format |
| **Material Response** | `.jpg`, `.png`, `.tiff`, `.webp` | Physics-based surface enhancement |
| **Video Master Grader** | `.mp4`, `.mov`, `.avi`, `.mkv` | FFmpeg-based processing |

---

## Image Formats

### Universal Support (All Image Pipelines)

The following formats are supported by all image processing tools through PIL/Pillow:

- **PNG** (`.png`) - Lossless compression, alpha channel support
- **JPEG** (`.jpg`, `.jpeg`) - Lossy compression, widely compatible
- **TIFF** (`.tif`, `.tiff`) - High-fidelity, metadata preservation
- **WebP** (`.webp`) - Modern format with good compression
- **BMP** (`.bmp`) - Uncompressed bitmap format

All image formats are **case-insensitive** (e.g., `.PNG`, `.Png`, `.png` are all accepted).

### Extended Format Support

These formats work with PIL-based tools but may have limited optimization:

- **GIF** (`.gif`) - Animation support (first frame used)
- **ICO** (`.ico`) - Icon format
- **PPM/PGM/PBM** (`.ppm`, `.pgm`, `.pbm`) - Netpbm formats
- **TGA** (`.tga`) - Truevision Targa

---

## Pipeline-Specific Details

### 1. Luxury TIFF Batch Processor

**Optimized For**: Professional 16-bit TIFF photography with metadata preservation

**Recommended Formats**:
- `.tif`, `.tiff` (16-bit with `tifffile` installed)

**Requirements**:
- **16-bit precision**: Requires `tifffile` and `imagecodecs` packages
- **HDR workflows**: Requires `tifffile` for high dynamic range support
- **Without tifffile**: Falls back to 8-bit PIL processing

**Installation**:
```bash
pip install -e ".[tiff]"  # Includes tifffile + imagecodecs
```

**Metadata Preservation**:
- IPTC tags (copyright, caption, keywords)
- EXIF data (camera settings, GPS coordinates)
- XMP metadata (Adobe metadata)
- ICC color profiles

**Example**:
```bash
# Process directory of 16-bit TIFFs with Material Response
python luxury_tiff_batch_processor.py \
    --input ./raw_tiffs \
    --output ./enhanced \
    --preset signature_estate \
    --recursive
```

**Supported Input**: `.tif`, `.tiff`, `.TIF`, `.TIFF`  
**Output**: 16-bit TIFF (with tifffile) or 8-bit TIFF (PIL fallback)

---

### 2. Depth Pipeline (Depth Anything V2)

**Optimized For**: Depth-aware architectural rendering and enhancement

**Supported Formats**:
- All PIL-supported formats (converted to RGB internally)
- Depth maps: 16-bit PNG recommended for precision

**Format Recommendations**:
- **Input renders**: PNG, TIFF (lossless quality)
- **Depth maps**: 16-bit PNG (e.g., `scene_depth16.png`)
- **Final output**: PNG for web, TIFF for print

**Example**:
```bash
# Process architectural render with depth-aware enhancement
python depth_pipeline/pipeline.py \
    --input living_room.jpg \
    --output enhanced/ \
    --config config/interior_preset.yaml
```

**Supported Input**: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`, `.bmp`  
**Output**: Matches input format or specified via `--output-format`

---

### 3. Lux Render Pipeline

**Optimized For**: AI-powered render refinement with ControlNet and SDXL

**Supported Formats**:
- All PIL-supported formats
- Converts to RGB for AI processing
- Logo overlays: PNG with alpha channel recommended

**Format Recommendations**:
- **Input**: PNG or TIFF for best quality
- **Logos**: PNG with transparency (`.png` with RGBA)
- **Output**: PNG for final delivery

**Example**:
```bash
# AI-enhance bedroom render with branding
python lux_render_pipeline.py \
    --input bedroom_render.jpg \
    --out ./final \
    --prompt "luxury bedroom interior, natural daylight" \
    --material-response \
    --logo ./brand/logo.png \
    --brand_text "The Veridian | Penthouse 21B"
```

**Supported Input**: Any PIL format (`.jpg`, `.png`, `.tiff`, `.bmp`, `.webp`)  
**Output**: PNG (default), TIFF (with `--output-tiff`)

---

### 4. Material Response System

**Optimized For**: Physics-based surface enhancement (wood, metal, glass, textiles)

**Supported Formats**:
- All PIL-supported RGB formats
- Works best with high-resolution input (4K+)

**Format Recommendations**:
- **Input**: TIFF or PNG for detail preservation
- **Output**: TIFF (16-bit) for maximum quality

**Example**:
```bash
# Enhance materials in architectural photo
python material_response.py \
    --input facade.jpg \
    --output facade_enhanced.tiff \
    --surfaces wood metal glass \
    --strength 0.75
```

**Supported Input**: `.jpg`, `.png`, `.tiff`, `.webp`, `.bmp`  
**Output**: Matches input format, TIFF recommended

---

### 5. Board Material Aerial Enhancer

**Optimized For**: Texture-based enhancement for aerial and exterior shots

**Supported Formats**:
- Standard image formats (PNG, JPEG, TIFF)
- Custom textures: PNG recommended for quality

**Example**:
```bash
# Enhance aerial photo with custom textures
python board_material_aerial_enhancer.py \
    aerial_montecito.jpg \
    enhanced_aerial.jpg \
    --textures textures/custom/
```

**Supported Input**: `.jpg`, `.png`, `.tiff`  
**Output**: Matches input format

---

## Video Formats

### Luxury Video Master Grader

**Optimized For**: Professional video color grading with LUTs and HDR support

**Supported Formats**:
- **MP4** (`.mp4`) - H.264, H.265/HEVC
- **MOV** (`.mov`) - ProRes, Apple Intermediate Codec
- **AVI** (`.avi`) - Uncompressed, DV
- **MKV** (`.mkv`) - Matroska container
- **WebM** (`.webm`) - VP8/VP9 codec

**HDR Support**:
- **PQ (Perceptual Quantizer)** - HDR10, HDR10+
- **HLG (Hybrid Log-Gamma)** - Broadcast HDR
- Automatic tone mapping for HDR → SDR conversion

**Output Formats**:
- **Default**: ProRes 422 HQ (`.mov`)
- **H.264**: MP4 with configurable bitrate
- **H.265**: HEVC for smaller file sizes

**Example**:
```bash
# Grade video with film emulation LUT
python luxury_video_master_grader.py \
    --input property_tour.mp4 \
    --output graded_tour.mov \
    --preset golden_hour_courtyard \
    --tone-map Hable
```

**Supported Input**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`  
**Output**: `.mov` (ProRes), `.mp4` (H.264/H.265)

---

## Format Recommendations by Use Case

### Architectural Rendering

| Stage | Format | Reason |
|-------|--------|--------|
| Raw renders | TIFF (16-bit) | Maximum dynamic range |
| Depth maps | PNG (16-bit) | Lossless, smaller than TIFF |
| AI enhancement | PNG | Quality with reasonable file size |
| Final delivery | TIFF/PNG | Print: TIFF, Web: PNG |

### Real Estate Photography

| Stage | Format | Reason |
|-------|--------|--------|
| Camera RAW → TIFF | TIFF (16-bit) | Preserve metadata, HDR |
| Batch processing | TIFF | Material Response precision |
| Web delivery | JPEG (95% quality) | Balance quality/size |
| Print delivery | TIFF (16-bit) | Maximum quality |

### Video Production

| Stage | Format | Reason |
|-------|--------|--------|
| Editing proxies | MP4 (H.264) | Efficient editing |
| Master grading | ProRes 422 HQ | High quality, fast codec |
| Final delivery | H.265 (4K+) | Small size, streaming-ready |
| Archival | ProRes 4444 XQ | Lossless, alpha channel |

---

## Technical Constraints

### File Size Limits

- **Image processing**: No hard limits, but 8K+ images may require 16GB+ RAM for ML pipelines
- **Video processing**: FFmpeg handles files up to system limits
- **Batch processing**: Uses streaming/chunking for memory efficiency

### Bit Depth Support

| Pipeline | 8-bit | 16-bit | 32-bit Float |
|----------|-------|--------|--------------|
| TIFF Batch Processor | ✅ | ✅* | ❌ |
| Depth Pipeline | ✅ | ✅** | ❌ |
| Lux Render | ✅ | ✅** | ❌ |
| Material Response | ✅ | ✅* | ❌ |

\* Requires `tifffile` package  
\*\* Input accepted, converted to 8-bit for processing, output as 16-bit

### Color Space Support

- **Input**: sRGB, Adobe RGB (via ICC profiles), Display P3
- **Processing**: sRGB color space (internally)
- **Output**: Preserves input color profile when possible
- **HDR**: PQ (SMPTE ST 2084), HLG (ITU-R BT.2100) for video

---

## Error Handling

### Unsupported Format Detection

All pipelines include format validation:

```python
from pathlib import Path

def validate_image_format(path: Path) -> bool:
    """Check if image format is supported."""
    supported = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp', '.bmp'}
    return path.suffix.lower() in supported
```

**Error Messages**:
- `"Unsupported image format: .xyz"` - Format not recognized
- `"Unable to open image: [path]"` - Corrupted file or access issue
- `"16-bit support requires tifffile"` - Missing optional dependency

### Fallback Behavior

1. **16-bit TIFF without tifffile**: Falls back to 8-bit PIL processing
2. **Corrupted images**: Skipped in batch mode, error logged
3. **Unsupported formats**: Early validation with clear error messages

---

## Dependencies for Format Support

### Core Image Support (PIL/Pillow)

```bash
pip install Pillow  # Included in requirements.txt
```

Enables: PNG, JPEG, TIFF (8-bit), BMP, GIF, WebP

### High-Fidelity TIFF (Optional)

```bash
pip install tifffile imagecodecs
# Or use extras:
pip install -e ".[tiff]"
```

Enables: 16-bit TIFF, advanced compression (LZW, JPEG, ZIP)

### Video Processing

```bash
# System dependency (not Python package)
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Verify installation:
ffmpeg -version
```

Enables: Video format support, LUT application, HDR tone mapping

---

## Examples by Format

### Processing TIFF Files

```bash
# 16-bit TIFF with metadata preservation
python luxury_tiff_batch_processor.py \
    --input ./tiff_photos \
    --output ./enhanced_tiffs \
    --preset signature_estate \
    --recursive
```

### Processing JPEG Files

```bash
# AI-enhance JPEG renders
python lux_render_pipeline.py \
    --input 'drafts/*.jpg' \
    --out ./enhanced \
    --prompt "luxury interior, natural light" \
    --steps 30 --strength 0.45
```

### Processing PNG Files

```bash
# Depth-aware enhancement for PNG renders
python depth_pipeline/pipeline.py \
    --input render.png \
    --output enhanced/ \
    --config config/interior_preset.yaml
```

### Mixed Format Batch

```bash
# Process directory with mixed formats (converted to TIFF output)
find ./input -type f \( -name "*.jpg" -o -name "*.png" \) \
    -exec python lux_render_pipeline.py --input {} --out ./output \;
```

---

## FAQ

### Q: Can I process RAW files (.cr2, .nef, .arw)?

**A**: Not directly. Convert RAW files to 16-bit TIFF using:
- Adobe Camera Raw / Lightroom
- Darktable (open-source)
- RawTherapee (open-source)

Then process with Luxury TIFF Batch Processor.

### Q: Does the system support HDR image formats?

**A**: Yes, for video (PQ, HLG tone curves). For images:
- 16-bit TIFF captures extended dynamic range
- OpenEXR not directly supported (convert to 16-bit TIFF first)

### Q: What about alpha channel preservation?

**A**: Alpha channels are:
- **Preserved**: PNG with transparency (logo overlays)
- **Removed**: Most pipelines convert to RGB (no alpha)
- **Output with alpha**: Save as PNG with 4-channel mode

### Q: Can I output to different formats than input?

**A**: Yes, specify output format:
```bash
# JPEG input → TIFF output
python material_response.py --input photo.jpg --output enhanced.tiff
```

### Q: Are file extensions case-sensitive?

**A**: No, all extensions are case-insensitive:
- `.jpg`, `.JPG`, `.Jpg` all recognized
- `.tiff`, `.TIFF`, `.Tiff` all recognized

---

## Performance Notes

### Format Speed Comparison

| Format | Read Speed | Write Speed | File Size | Best For |
|--------|-----------|-------------|-----------|----------|
| JPEG | ⚡⚡⚡ | ⚡⚡⚡ | Smallest | Web, fast preview |
| PNG | ⚡⚡ | ⚡ | Medium | Lossless, web |
| TIFF (8-bit) | ⚡⚡ | ⚡⚡ | Large | Editing |
| TIFF (16-bit) | ⚡ | ⚡ | Largest | Print, archival |
| WebP | ⚡⚡ | ⚡⚡ | Small | Modern web |

### Memory Usage

- **JPEG/PNG**: ~3-4x file size (decompression)
- **TIFF (8-bit)**: ~1.5x file size
- **TIFF (16-bit)**: ~3x file size
- **AI processing**: +8-16GB VRAM for models

---

## See Also

- [README.md](README.md) - Main documentation
- [DEPTH_PIPELINE_README.md](DEPTH_PIPELINE_README.md) - Depth processing specifics (overview)
- [docs/depth_pipeline/DEPTH_PIPELINE_README.md](docs/depth_pipeline/DEPTH_PIPELINE_README.md) - Depth pipeline configuration and advanced documentation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [tests/TEST_STATUS.md](tests/TEST_STATUS.md) - Test coverage

---

**Last Updated**: October 2025  
**Maintainer**: Transformation Portal Team
