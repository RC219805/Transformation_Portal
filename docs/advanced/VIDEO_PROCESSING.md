# Video Processing

**⚠️ SEPARATE DOMAIN** - Not part of image processing Golden Path.

---

## Quick Decision

**Use Video Processing if**:
- You're processing **video files** (not images)
- Need ProRes 422 HQ masters
- Require HDR tone mapping
- Want LUT-based color grading

**Use Golden Path if**:
- You're processing **images** (not videos)

---

## Overview

Video processing uses a **separate tool** (`luxury_video_master_grader.py`) for:
- **ProRes 422 HQ** output (broadcast-quality)
- **HDR Detection** and automatic tone mapping
- **LUT Grading** (film emulation, location aesthetics)
- **Frame Rate Conformance** (23.976, 24, 25, 29.97, 30 fps)
- **Color Metadata** preservation

---

## Usage

### Basic Usage

```bash
python luxury_video_master_grader.py \
  --input input.mp4 \
  --output graded.mov \
  --preset signature_estate \
  --lut assets/luts/film_emulation/Kodak_2393.cube
```

### Presets

- `signature_estate` - Luxury real estate signature look
- `golden_hour_courtyard` - Warm, sunset-toned grading
- `modern_minimalist` - Clean, high-contrast aesthetic
- `heritage_warmth` - Classic, timeless color palette

---

## Features

### HDR Tone Mapping

Automatically detects HDR transfer functions (PQ, HLG) and applies tone mapping:

```bash
python luxury_video_master_grader.py \
  --input hdr_video.mp4 \
  --output sdr_video.mov \
  --tone-map hable \
  --preset signature_estate
```

**Tone map operators**: `hable`, `reinhard`, `mobius`, `filmic`

### LUT Stacking

Apply multiple LUTs for complex grading:

```bash
python luxury_video_master_grader.py \
  --input video.mp4 \
  --output graded.mov \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --lut assets/luts/location_aesthetic/Santa_Barbara_Coastal.cube \
  --strength 0.75
```

### Frame Rate Conformance

Conform to cinema/broadcast standards:

```bash
python luxury_video_master_grader.py \
  --input variable_fps.mp4 \
  --output conform.mov \
  --fps 23.976
```

---

## Requirements

**FFmpeg 6+** with ProRes codec support:

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Verify
ffmpeg -version
ffmpeg -encoders | grep prores
```

---

## Performance

| Video Length | Resolution | Processing Time | Output Size |
|--------------|------------|-----------------|-------------|
| 1 minute | 1080p | ~30 seconds | ~500 MB |
| 5 minutes | 4K | ~5 minutes | ~5 GB |
| 10 minutes | 1080p | ~3 minutes | ~2.5 GB |

**Trade-off**: ProRes 422 HQ creates large files but maintains broadcast quality.

---

## Output Formats

**Default**: ProRes 422 HQ (`.mov`)

**Alternatives**:
- H.264 (web delivery): `--codec libx264`
- H.265 (smaller files): `--codec libx265`
- DNxHD (Avid workflow): `--codec dnxhd`

---

## Troubleshooting

**"Codec not found"**
```bash
# Check FFmpeg codec support
ffmpeg -codecs | grep prores
```

**"Invalid frame rate"**
```bash
# Supported rates: 23.976, 24, 25, 29.97, 30
# Use --fps to conform
```

**"Out of disk space"**
```bash
# ProRes is large - ensure 1GB per minute of 1080p video
df -h
```

---

## Why Separate from Image Processing?

**Different domains**:
- Video: Temporal consistency, frame rate, codecs, audio
- Images: Spatial quality, batch processing, archival formats

**Different tools**:
- Video: FFmpeg, ProRes, LUT grading
- Images: Depth processing, material response, AI enhancement

**Different workflows**:
- Video: Linear timeline, color grading, delivery specs
- Images: Parallel processing, detail enhancement, format conversion

---

## Related Documentation

- **[Golden Path](../../QUICKSTART.md)** - Image processing
- **[Advanced README](README.md)** - Advanced workflows
- **FFmpeg Documentation**: https://ffmpeg.org/documentation.html

---

## Full Tool Documentation

```bash
python luxury_video_master_grader.py --help
```

---

*Video processing is a separate domain with its own best practices. Don't conflate it with image processing.*
