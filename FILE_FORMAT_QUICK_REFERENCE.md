# File Format Quick Reference

**Transformation Portal** | Image & Video Format Support

---

## 📸 Image Formats

| Format | Extension | Use Case | Quality | Notes |
|--------|-----------|----------|---------|-------|
| **TIFF** | `.tif`, `.tiff` | Professional/Print | 16-bit | Requires `tifffile` for full precision |
| **PNG** | `.png` | Web/Lossless | 8-bit | Supports transparency, good for renders |
| **JPEG** | `.jpg`, `.jpeg` | Web/Delivery | 8-bit | Lossy, use quality ≥95 |
| **WebP** | `.webp` | Modern Web | 8-bit | Good compression, lossy/lossless modes |
| **BMP** | `.bmp` | Uncompressed | 8-bit | Large files, rarely used |
| **GIF** | `.gif` | Animation | 8-bit | Limited colors, first frame used |

**All formats are case-insensitive** (`.PNG` = `.png`)

---

## 🎬 Video Formats

| Format | Extension | Use Case | Notes |
|--------|-----------|----------|-------|
| **MP4** | `.mp4` | Universal/Streaming | H.264, H.265/HEVC codecs |
| **MOV** | `.mov` | Professional/Editing | ProRes, QuickTime codecs |
| **AVI** | `.avi` | Legacy | Various codecs |
| **MKV** | `.mkv` | High-quality/Archive | Matroska container |
| **WebM** | `.webm` | Web Streaming | VP8/VP9 codecs |

**HDR Support**: PQ (HDR10), HLG (Hybrid Log-Gamma)

---

## 🔧 Pipeline Quick Match

| Need to... | Use This Pipeline | Best Format |
|------------|-------------------|-------------|
| Process 16-bit TIFFs | `luxury_tiff_batch_processor.py` | `.tiff` |
| Enhance architectural renders | `depth_pipeline/pipeline.py` | `.png`, `.tiff` |
| AI-powered enhancement | `lux_render_pipeline.py` | `.png`, `.tiff` |
| Material surface enhancement | `material_response.py` | `.tiff`, `.png` |
| Video color grading | `luxury_video_master_grader.py` | `.mov`, `.mp4` |

---

## ⚡ Quick Commands

### Validate File Format
```bash
python examples/validate_file_formats.py image.jpg
python examples/validate_file_formats.py --scan directory/
python examples/validate_file_formats.py --formats
```

### Process Images
```bash
# TIFF batch processing
python luxury_tiff_batch_processor.py input/ output/ --preset signature

# Depth-aware enhancement
python depth_pipeline/pipeline.py --input render.jpg --output enhanced.jpg

# AI enhancement
python lux_render_pipeline.py --input bedroom.jpg --out ./final --prompt "luxury interior"
```

### Process Videos
```bash
python luxury_video_master_grader.py --input tour.mp4 --output graded.mov --preset golden_hour
```

---

## 💎 Luxury Formats (Recommended)

Best for professional/commercial work:
- **TIFF** (`.tif`, `.tiff`) - 16-bit precision, metadata preservation
- **PNG** (`.png`) - Lossless, good for web and renders

---

## 🚨 Common Issues

### "16-bit support requires tifffile"
**Solution**: `pip install -e ".[tiff]"`

### "FFmpeg not found"
**Solution**: 
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from ffmpeg.org

### "Unsupported format"
**Solution**: Convert to supported format or check file extension

### RAW files (.cr2, .nef, .arw)
**Solution**: Convert to 16-bit TIFF using Lightroom, Darktable, or RawTherapee

---

## 📊 Format Comparison

### File Size (1920×1080 image)

| Format | Approx. Size | Speed | Use When |
|--------|-------------|-------|----------|
| JPEG (95%) | 1-2 MB | ⚡⚡⚡ | Web, fast preview |
| PNG | 3-5 MB | ⚡⚡ | Web, transparency needed |
| TIFF (8-bit) | 6-8 MB | ⚡⚡ | Editing workflow |
| TIFF (16-bit) | 12-16 MB | ⚡ | Print, archival |
| WebP | 0.5-1.5 MB | ⚡⚡ | Modern web |

### Processing Speed (4K image)

| Operation | Time (M4 Max) | Memory |
|-----------|---------------|--------|
| Format validation | <1ms | <1 MB |
| TIFF load (8-bit) | 50-100ms | 32 MB |
| TIFF load (16-bit) | 100-200ms | 64 MB |
| Depth estimation | 24-65ms | 2 GB |
| AI enhancement | 5-30s | 8-16 GB |

---

## 🎯 Workflow Recommendations

### Architectural Rendering
1. **Raw Render** → 16-bit TIFF (from render engine)
2. **Depth Maps** → 16-bit PNG (smaller than TIFF)
3. **AI Enhancement** → PNG or TIFF
4. **Final Delivery** → PNG (web) or TIFF (print)

### Real Estate Photography
1. **Camera RAW** → 16-bit TIFF (via Lightroom)
2. **Batch Processing** → TIFF (preserve metadata)
3. **Web Gallery** → JPEG (95% quality)
4. **Print Files** → 16-bit TIFF (ProPhoto RGB)

### Video Production
1. **Editing Proxies** → MP4 H.264 (fast)
2. **Master Grading** → ProRes 422 HQ MOV
3. **Web Delivery** → MP4 H.265 (4K)
4. **Archive** → ProRes 4444 XQ (lossless)

---

## 📖 More Information

- **Detailed Docs**: [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md)
- **Main README**: [README.md](README.md)
- **Depth Pipeline**: [DEPTH_PIPELINE_README.md](DEPTH_PIPELINE_README.md)
- **Detailed Depth Pipeline Docs**: [docs/depth_pipeline/DEPTH_PIPELINE_README.md](docs/depth_pipeline/DEPTH_PIPELINE_README.md)
- **Detailed Depth Pipeline Docs**: [docs/depth_pipeline/DEPTH_PIPELINE_README.md](docs/depth_pipeline/DEPTH_PIPELINE_README.md)

---

## 🔍 Format Detection (Python)

```python
from format_utils import (
    is_supported_image_format,
    get_format_info,
    validate_format,
)

# Check if supported
if is_supported_image_format('photo.jpg'):
    print("✅ Supported")

# Get detailed info
info = get_format_info('render.tiff')
print(info['recommendations'])

# Validate with error handling
try:
    validate_format('document.pdf', 'image')
except UnsupportedFormatError as e:
    print(f"❌ {e}")
```

---

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| What formats work? | PNG, JPEG, TIFF, WebP, BMP + more (see above) |
| Do I need 16-bit? | Recommended for print/professional work |
| What about RAW? | Convert to TIFF first |
| Video formats? | MP4, MOV, AVI, MKV (FFmpeg required) |
| Case-sensitive? | No, `.PNG` = `.png` |
| File size limit? | No hard limit (system memory dependent) |

---

**Print this page** for quick reference while working! 🖨️

**Version**: 1.0 | **Updated**: October 2025
