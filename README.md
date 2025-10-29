[![CI](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/python-app.yml/badge.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs)

# 800 Picacho Lane — Professional LUT Collection

## Overview

A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Quickstart

Install the package and run your first TIFF enhancement:

```bash
pip install picacho-lane-luts
python luxury_tiff_batch_processor.py input_folder output_folder --preset signature
````

For advanced render finishing, see [Material Response Finishing for Neural Renders](#material-response-finishing-for-neural-renders).

## Table of Contents

* [Overview](#overview)
* [Quickstart](#quickstart)
* [Collection Contents](#collection-contents)
* [Innovation](#innovation)
* [Usage](#usage)

  * [Material Response Finishing for Neural Renders](#material-response-finishing-for-neural-renders)
* [Developer Setup](#developer-setup)

  * [Install Dependencies](#install-dependencies)
  * [Test Shortcuts](#test-shortcuts)
* [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
* [Luxury Video Master Grader](#luxury-video-master-grader)
* [HDR Production Pipeline](#hdr-production-pipeline)
* [Board Material Aerial Enhancer](#board-material-aerial-enhancer)
* [Decision Decay Dashboard](#decision-decay-dashboard)
* [License](#license)

## Collection Contents

* **Film Emulation**: Kodak 2393, FilmConvert Nitrate
* **Location Aesthetic**: Montecito Golden Hour, Spanish Colonial Warm
* **Material Response**: Revolutionary physics-based surface enhancement

---

## Innovation

**Material Response LUTs** analyze and enhance how different surfaces interact with light—shifting from purely global color transforms to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials.

---

## Usage

1. Import into DaVinci Resolve, Premiere Pro, or other color-grading software.
2. Apply at **60–80% opacity** initially.
3. Stack multiple LUTs for complex material interactions.

### Material Response Finishing for Neural Renders

[`lux_render_pipeline.py`](./lux_render_pipeline.py) exposes a **Material Response** finishing layer that reinforces wood grain, textile separation, fireplace warmth, and atmospheric haze directly from the command line. Enable it with `--material-response` to activate detail boosts, contact shadowing, and volumetric tinting that better fuse interior renders with their exterior vistas.

> **Requires:** the `[ml]` extras installed (`pip install -e ".[ml]"`) and a GPU-enabled environment for optimal performance.

Example:

```bash
python lux_render_pipeline.py \
  --input bedroom_render.jpg \
  --out ./enhanced_bedroom \
  --prompt "minimalist bedroom interior..." \
  --material-response --texture-boost 0.28 ...
```

---

## Developer Setup

### Install Dependencies

* Create a `.venv` (optional but recommended)

* Install the project requirements:

  ```bash
  python -m pip install .
  ```

  or mirror CI:

  ```bash
  python -m pip install -r requirements.txt
  ```

* Add optional extras:

  ```bash
  pip install -e ".[tiff]"   # 16-bit TIFF processing
  pip install -e ".[dev]"    # pytest, linting
  pip install -e ".[ml]"     # ML extras for lux_render_pipeline
  pip install -e ".[all]"    # everything
  ```

### Console Scripts

After installation, the following command-line tools are available:

* [`luxury_tiff_batch_processor.py`](./luxury_tiff_batch_processor.py) — batch process TIFFs
* [`luxury_video_master_grader.py`](./luxury_video_master_grader.py) — video grading
* [`lux_render_pipeline.py`](./lux_render_pipeline.py) — AI-powered render refinement
* [`decision_decay_dashboard.py`](./decision_decay_dashboard.py) — codebase philosophy audits

---

### Test Shortcuts

Use the bundled Makefile:

```bash
make test-fast
make test-full
```

**Tip:** `make ci` runs linting (`flake8`, `pylint`) and fast tests, mirroring GitHub Actions.

---

## Luxury TIFF Batch Processor

[`luxury_tiff_batch_processor.py`](./luxury_tiff_batch_processor.py) is a high-end workflow for polishing large-format TIFF photography prior to digital launch. It preserves metadata, honors 16-bit source files (via [`tifffile`](https://pypi.org/project/tifffile/)), and layers tonal and chroma refinements tuned for luxury real-estate storytelling.

**Features:**
- 16-bit TIFF support with metadata preservation
- Multiple processing presets (Signature, Vivid, Natural, Moody)
- Batch processing with progress tracking
- Non-destructive workflow

**Usage:**
```bash
python luxury_tiff_batch_processor.py input_folder output_folder --preset signature
```

---

## Luxury Video Master Grader

[`luxury_video_master_grader.py`](./luxury_video_master_grader.py) brings the same curated aesthetic to motion content using FFmpeg.

**Features:**
- FFmpeg-based video color grading
- LUT application for consistent look
- Support for multiple video formats
- Batch processing capabilities

**Usage:**
```bash
python luxury_video_master_grader.py input_video.mp4 output_video.mp4 --lut path/to/lut.cube
```

---

## HDR Production Pipeline

[`hdr_production_pipeline.sh`](./hdr_production_pipeline.sh) orchestrates a full HDR finishing pass, combining ACES tone mapping, adaptive debanding, and halation.

**Features:**
- ACES color space workflow
- Adaptive debanding and grain
- Halation and bloom effects
- Production-grade HDR output

**Usage:**
```bash
./hdr_production_pipeline.sh input.exr output.mp4
```

---

## Board Material Aerial Enhancer

[`board_material_aerial_enhancer.py`](./board_material_aerial_enhancer.py) applies MBAR-approved material palettes to aerials using clustering and texture blending.

**Features:**
- K-means clustering for material segmentation
- Material-aware palette assignment
- Texture-based enhancement
- Board-approved aesthetic compliance

For full documentation, see [Palette Assignment Guide](./08_Documentation/Palette_Assignment_Guide.md).

**Usage:**
```bash
python board_material_aerial_enhancer.py aerial_image.jpg output_enhanced.jpg
```

---

## Decision Decay Dashboard

[`decision_decay_dashboard.py`](./decision_decay_dashboard.py) surfaces temporal contracts, codebase philosophy violations, and brand color token drift in one terminal dashboard.

**Features:**
- Codebase philosophy auditing
- Temporal contract monitoring
- Brand consistency checking
- Terminal-based dashboard interface

**Usage:**
```bash
python decision_decay_dashboard.py
```

---

## License

Professional use permitted with attribution.

---

**Author:** Richard Cheetham
**Brand:** Carolwood Estates · RACLuxe Division
**Contact:** [info@racluxe.com](mailto:info@racluxe.com)
