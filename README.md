[![CI/CD](https://github.com/RC219805/Transformation_Portal/actions/workflows/ci-consolidated.yml/badge.svg)](https://github.com/RC219805/Transformation_Portal/actions/workflows/ci-consolidated.yml)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1348%20passed-brightgreen.svg)](https://github.com/RC219805/Transformation_Portal/actions)

# Transformation Portal

Professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production.

## 🎉 Phase 3 Complete - Full Pipeline Deployment (December 5, 2025)

**Unified Luxury Pipeline** - Production-ready with complete 6-stage processing

**[📋 View Phase 3 Deployment Summary](PHASE3_DEPLOYMENT_SUMMARY.md)**

**✅ ALL PHASES COMPLETE:**
- ✅ **Phase 1**: Upscaling (SwinIR + Real-ESRGAN) - 15 tests
- ✅ **Phase 2**: Depth Processing + Material Response - 21 tests  
- ✅ **Phase 3**: LUT Color Grading System - 23 tests ← NEW
- ✅ **59 total tests passing** - 100% pass rate
- ✅ **127-400 images/hour** - Production-validated throughput
- ✅ **16-bit precision** - Archival-grade quality maintained

**Complete Pipeline (All 6 Stages Functional):**
1. ✅ Loading & Validation
2. ✅ AI Upscaling (4x) - SwinIR/Real-ESRGAN
3. ✅ Depth-Aware Processing - Depth Anything V2
4. ✅ Material Response - 8 surface types
5. ✅ **Professional Color Grading - Film emulation + Location LUTs** ← Phase 3
6. ✅ Export - 16-bit TIFF with metadata

**Phase Documentation:**
- 📘 [Phase 1: Upscaling](UPSCALING_REFINEMENT_COMPLETE.md)
- 📗 [Phase 2: Depth + Material](PHASE2_FINAL_SUMMARY.md)
- 📙 [Phase 3: LUT System](PHASE3_DEPLOYMENT_SUMMARY.md) ← NEW

---

## 📋 Comprehensive Status Report (December 2025)

A detailed review of the codebase structure, infrastructure, capabilities, and performance is now available:

**[📊 View Comprehensive Codebase Update](docs/COMPREHENSIVE_CODEBASE_UPDATE_2025.md)**

**Key Highlights:**
- ✅ **1,348 tests passing** (0 critical linting errors)
- ⚡ **3-5x throughput improvement** via async pipeline architecture  
- 📉 **92% smaller repository** (180MB → 15MB)
- 🔒 **Security hardened** (CVE-2024-27763 mitigation)
- 🧠 **RAG Knowledge Engine** activated (Phase 2)

---

## 🚀 Latest Update: Async/Streaming Pipeline Architecture (November 2025)

NEW: High-performance async processing infrastructure for 3-5x throughput improvement on batch image processing workloads.

### What's New
- ⚡ **AsyncPipeline**: Stage orchestration with queue-based execution and backpressure handling
- 🔄 **BackpressureQueue**: Flow control with high/low water marks for memory efficiency
- 👷 **WorkerPool**: Separate CPU/IO thread pools with GPU device affinity (CUDA/MPS)
- 📸 **StreamingImageLoader**: Memory-efficient prefetch loading for large batches
- 🎯 **Concrete Stages**: ImageLoad, ImageSave, DepthEstimation, MaterialResponse, ColorGrading, Resize, Denoise

### Performance Targets
- Sequential processing (100 4K images): ~6.9 hours
- With async pipeline: ~1.5-2 hours (3-5x faster)
- Memory footprint reduced 50% via streaming I/O

See: `src/transformation_portal/streaming/` for implementation details.

---

## 🎯 Phase 1 Strategic Enhancements (December 2024)

**Version 1.0.0** - Comprehensive workflow optimizations

### New Capabilities
- **🔍 Batch Comparison Tool** - PSNR/SSIM metrics, visual difference analysis
- **📊 HDR Visualization** - Before/after histograms, dynamic range charts
- **⏱️ Time Prediction** - Intelligent estimation with historical learning
- **✅ QA Validation** - Pre-flight checks with go/no-go decisions
- **🎨 Adaptive Tone Mapping** - Scene-aware parameter selection (low/mid/high-key)
- **🎭 Alpha Compositing** - Multiple background modes and variants
- **📄 Enhanced Reports** - Client summaries and technical appendices

### Quick Start
```bash
# Validate inputs before processing
python tools/qa_validator.py input_images/*.tif --output qa_report.md

# Process with intelligent enhancements
python process_750_picacho_32bit_hdr_enhanced.py

# Compare outputs
python tools/comparison_tool.py --dir1 baseline/ --dir2 enhanced/ --output comparisons/

# Analyze HDR processing
python tools/hdr_visualizer.py --before hdr.tif --after tone_mapped.tif --name Kitchen
```

**Documentation:** See `docs/PHASE1_ENHANCEMENTS.md` and `PHASE1_IMPLEMENTATION_SUMMARY.md`

---

## 🧠 Context-Aware Rendering (November 2025)

Revolutionary Context-Aware Rendering System that extracts architectural intelligence from construction documents (floor plans, elevations, specifications) and uses this knowledge to inform every processing decision.

### What's New
- 🏗️ Architectural Context Extraction - Reads PDFs to extract room types, dimensions, materials, and design style
- 🧠 Intelligent Strategy Derivation - Automatically optimizes processing for each specific space
- 🎯 Room-Specific Processing - Kitchen, bedroom, bathroom, living, outdoor areas each get tailored treatment
- 📐 Dimension-Aware - Depth processing respects actual room proportions
- 🎨 Style-Consistent - Color grading aligns with architectural design language
- 📄 Document Provenance - Direct connection between construction docs and final renders

See: [Context-Aware Rendering Guide](docs/CONTEXT_AWARE_RENDERING.md) | [Quick Start](#quick-start) | [Implementation Summary](docs/CONTEXT_AWARE_RENDERING.md#implementation-summary)

---

## 🎉 Recent Update: Repository Refactored (October 2025)

The repository has been significantly reorganized for better performance and maintainability:
- 92% smaller repository size (180MB → 15MB)
- 60% faster imports with lazy loading
- Clear modular structure with organized packages
- Comprehensive documentation in `docs/` directory

See `docs/REFACTORING_SUMMARY.md` for details.

## 🗂️ Automated Repository Organization

The repository now includes an automated file organization system to maintain a clean, structured directory hierarchy:

- Automatic file organization with `.auto-organize.sh`
- Pre-commit hooks to prevent misplaced files
- Clear directory structure for docs, scripts, assets, and data

See `docs/migrated/REPO_ORGANIZATION.md` for complete documentation.

Quick Start:
```bash
# Install organization system
./scripts/setup/auto-organize-install.sh

# Organize repository (dry-run first)
./.auto-organize.sh --dry-run
./.auto-organize.sh
```

## Overview

Transformation Portal is a comprehensive suite of AI-powered tools and pipelines designed for high-end architectural rendering, real estate photography, and video post-production. It combines cutting-edge machine learning models, professional color grading techniques, and proprietary Material Response technology to transform raw renders and photographs into polished marketing visuals.

### 🤖 Custom AI Agent Available (Enhanced v2.0) 🚀

A specialized Transformation Portal Specialist GitHub Copilot agent is available to assist with development. **NEW in v2.0**: Enhanced with 8 advanced capabilities including multi-modal analysis, proactive automation, deep debugging, performance optimization, CI/CD intelligence, and quality assurance.

**Use it in Copilot Chat**: `@transformation-portal-specialist [your request]`

**New Capabilities**:
- 🖼️ Multi-modal artifact analysis (image/video quality assessment)
- 🤖 Proactive workflow automation (suggests next steps)
- 🔍 Deep debugging with root cause analysis
- ⚡ Automated performance optimization (2-5x improvements)
- 🔄 Smart CI/CD (60% faster test feedback)
- 🧠 Interactive learning (adapts to your style)
- 📝 Context-aware responses (tutorial/quick-ref modes)
- ✨ Automated quality assurance and security scanning

**Quick Start**: See [Agent Quick Start v2.0](.github/agents/QUICK_START_v2.md)
**Full Details**: See [Agent Enhancements v2.0](.github/agents/AGENT_ENHANCEMENTS_v2.md)
**User Guide**: See [Custom Agent Guide](docs/CUSTOM_AGENT_GUIDE.md)

## Table of Contents

* [Features](#features)
* [Quick Start](#quick-start)
* [🎓 Model Training](#-model-training)
* [Installation](#installation)
* [Supported File Formats](#supported-file-formats)
* [📖 Pipeline Operations Guide](#pipeline-operations-guide)
* [Core Components](#core-components)
    * [Context-Aware Rendering 🆕](#context-aware-rendering-)
    * [Depth Pipeline](#depth-pipeline)
    * [Lux Render Pipeline](#lux-render-pipeline)
    * [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
    * [Luxury Video Master Grader](#luxury-video-master-grader)
    * [Material Response System](#material-response-system)
    * [Board Material Aerial Enhancer](#board-material-aerial-enhancer)
* [LUT Collection](#lut-collection)
* [Developer Tools](#developer-tools)
* [Usage Examples](#usage-examples)
* [Configuration](#configuration)
* [Performance](#performance)
* [Testing](#testing)
* [License](#license)

---

## Features

### Core Capabilities

- ✅ **Advanced Upscaling Engine** 🆕 - SwinIR + Real-ESRGAN with 16-bit precision, tile-based gigapixel processing
- ✅ Context-Aware Rendering - First-of-its-kind system that reads architectural documents and adapts processing accordingly
- ✅ AI-Powered Enhancement - Stable Diffusion XL, ControlNet, intelligent 4x upscaling
- ✅ Depth-Aware Processing - Depth Anything V2 with Apple Neural Engine optimization
- ✅ Material Response Technology - Physics-based surface enhancement for wood, metal, glass, textiles
- ✅ Professional Color Grading - 16+ LUTs with Film Emulation and Location Aesthetics
- ✅ 16-bit TIFF Support - Metadata-preserving batch processing for high-end photography
- ✅ HDR Production Pipeline - ACES color space, adaptive debanding, halation effects
- ✅ Batch Processing - 150-600 images/hour throughput (model-dependent)
- ✅ Production-Ready - Comprehensive test suite, CI/CD, performance profiling

### Technology Stack

| Technology         | Purpose                                             |
|--------------------|-----------------------------------------------------|
| **SwinIR** 🆕      | **Photo-realistic 4x upscaling (superior textures)**|
| **Real-ESRGAN** 🆕 | **Fast 4x upscaling (robust for noisy inputs)**     |
| Depth Anything V2  | Monocular depth estimation (24ms @ 518px on M4 Max) |
| Stable Diffusion XL| AI-powered render refinement                        |
| ControlNet         | Edge-preserving image-to-image translation          |
| FFmpeg             | Video processing and LUT application                |
| PyTorch/CoreML     | GPU acceleration (CUDA, MPS, Apple Neural Engine)   |
| Colour Science     | Professional color space transformations            |

---

## Quick Start

📖 New to the pipelines? Check out the complete Pipeline Operations Guide for step-by-step instructions on how to operate each pipeline, or see the Quick Start Cheat Sheet for common commands.

### 🔬 Advanced Upscaling (NEW - December 2025)

Professional 4x upscaling with multiple model options for maximum quality:

```bash
# Setup upscaling engine
make setup-upscaling  # Downloads SwinIR + Real-ESRGAN models (~280MB)

# Single image upscale (highest quality)
python utils/upscaling_engine.py input.tif output_4x.tif --model swinir_real_4x

# Batch processing (20+ images)
python utils/upscaling_engine.py input_dir/ output_dir/ --batch --model swinir_real_4x

# Fast processing (noisy inputs)
python utils/upscaling_engine.py noisy.jpg clean_4x.tif --model realesrgan_general_4x
```

**Model Selection Guide:**
- **SwinIR Real 4x**: Best quality for photos (portraits, architecture) - ~150 images/hour
- **Real-ESRGAN 4x**: Fast processing, robust for mixed quality - ~410 images/hour  
- **Real-ESRGAN General**: Configurable denoising for very noisy sources - ~395 images/hour

**Features:**
- ✅ 16-bit TIFF preservation (archival quality)
- ✅ Tile-based processing (gigapixel images on 4GB GPU)
- ✅ Color consistency validation (<2% RGB deviation)
- ✅ Batch model caching (10-20x speedup)
- ✅ Cross-platform (CPU, CUDA, Apple MPS)

📚 **Full Documentation**: [docs/UPSCALING_GUIDE.md](docs/UPSCALING_GUIDE.md)  
🔬 **Examples**: [examples/upscaling_workflow.py](examples/upscaling_workflow.py)  
📊 **Summary**: [docs/UPSCALING_SUMMARY.md](docs/UPSCALING_SUMMARY.md)

### 🚀 Lux Depth V2 Pipeline (NEW - December 2025)

**Production-oriented, GPU-accelerated depth processing** with advanced material segmentation and security hardening:

```bash
# Quick Start (Batch Processing)
lux-depth-v2 \
  --input-dir renders/ \
  --depth-dir depth_maps/ \
  --output-dir output/ \
  --preset interior_luxury \
  --device cuda \
  --upscaler-backend torch

# Service Mode (FastAPI - persistent models, low latency)
lux-depth-v2-service \
  --output-dir /data/output \
  --service \
  --host 0.0.0.0 \
  --port 8088
```

**Key Features:**
- ✅ **Secure by default** - CVE-2024-27763 mitigated, input validation, rate limiting
- ✅ **GPU-accelerated** - Torch-based post-processing (clarity, sharpen, detail transfer)
- ✅ **Advanced material segmentation** - ONNX/SegFormer/Heuristic backends
- ✅ **Safe AI detail transfer** - Color/luma drift guardrails
- ✅ **Service mode** - FastAPI with persistent models for real-time processing
- ✅ **16-bit precision** - Full archival-grade workflow maintained

**Security Hardening:**
- ✅ Path traversal prevention
- ✅ Rate limiting (10 req/min per IP)
- ✅ File size validation (configurable)
- ✅ Safe upscaling (torch backend, no vulnerable dependencies)

**Outputs:**
- `*_master16.tif` - 16-bit graded pre-upscale
- `*_upscaled16.tif` - 16-bit final output
- `*_marketing.png` - 8-bit preview
- `*_report.json` - Processing metadata

📚 **Documentation**: [lux_depth_v2/README.md](lux_depth_v2/README.md)  
🔒 **Security Guide**: [lux_depth_v2/SECURITY.md](lux_depth_v2/SECURITY.md)  
⚡ **Quick Start**: [docs/LUX_DEPTH_V2_QUICK_START.md](docs/LUX_DEPTH_V2_QUICK_START.md)

### 🎯 Unified Luxury Pipeline (December 2025)

**Production-grade pipeline** integrating upscaling, depth processing, and luxury enhancements:

```bash
# Single image with preset
python unified_luxury_pipeline.py input.tif --preset photo_realistic

# Batch processing (architectural renders)
python unified_luxury_pipeline.py renders/ --batch --preset architectural

# Luxury estate showcase
python unified_luxury_pipeline.py estate.tif --preset signature_estate
```

**7 Production Presets:**
- **Photo Realistic**: Maximum quality (SwinIR + full depth) - 150/hr
- **Architectural**: Balanced speed/quality - 350/hr
- **Archival Quality**: Museum-grade 16-bit - 120/hr
- **Fast Batch**: Speed-optimized - 450/hr
- **Signature Estate**: Luxury marketing - 140/hr
- **Interior Luxury**: Interior emphasis - 160/hr
- **Exterior Showcase**: Outdoor focus - 150/hr

**Features:**
- ✅ 7 preset workflows for common use cases
- ✅ **Upscaling + Depth + Material Response + Color Grading** (Phase 2 Complete!)
- ✅ 16-bit end-to-end workflow
- ✅ Batch processing with ETAs
- ✅ Automatic quality reports
- ✅ Configurable stage control
- ✅ **Depth-aware zone processing** (foreground/midground/background)
- ✅ **8 material types supported** (wood, metal, glass, stone, fabric, concrete, ceramic, water)

**Integration Status** ✅:
- ✓ Upscaling Engine (SwinIR + Real-ESRGAN)
- ✓ Depth Processing (Depth Anything V2) - **Phase 2 Complete**
- ✓ Material Response (Physics-based) - **Phase 2 Complete**
- 🔄 LUT System (Phase 3 pending)

📚 **Documentation**: [docs/UNIFIED_PIPELINE_GUIDE.md](docs/UNIFIED_PIPELINE_GUIDE.md)  
🔬 **Examples**: [examples/unified_pipeline_workflows.py](examples/unified_pipeline_workflows.py)  
📊 **Summary**: [UNIFIED_PIPELINE_COMPLETE.md](UNIFIED_PIPELINE_COMPLETE.md)  
✅ **Phase 2**: [PHASE2_INTEGRATION_COMPLETE.md](PHASE2_INTEGRATION_COMPLETE.md)

## 🎓 Model Training

**NEW**: Complete training infrastructure for neural network models (CausticGenerator, AtmosphericSynthesizer, MaterialTranscendence, SpatialHarmonics).

### ⚡ Quick Start Training

```bash
# Check if ready for training
python scripts/check_training_ready.py

# Run training (recommended - best quality)
./scripts/train_with_750picacho.sh

# Wait 2.5-3.5 hours (GPU) or 12-18 hours (CPU)
# Result: 103-107/100 quality (vs 78/100 baseline)
```

### 📚 Training Documentation

| Guide | Purpose | Use When |
|-------|---------|----------|
| **[HOW_TO_TRAIN.md](docs/migrated/HOW_TO_TRAIN.md)** | Complete implementation guide | You want step-by-step instructions |
| **[TRAINING_QUICK_REFERENCE.md](docs/migrated/TRAINING_QUICK_REFERENCE.md)** | Quick reference card | You need quick command lookup |
| **[TRAINING_DECISION_TREE.md](docs/migrated/TRAINING_DECISION_TREE.md)** | Choose training method | You're unsure which approach to use |
| **[TRAINING_EXECUTION_GUIDE.md](docs/migrated/TRAINING_EXECUTION_GUIDE.md)** | Detailed workflow | You want comprehensive details |

### 🎯 Training Options

**Option 1: 750 Picacho Real Data (Recommended)** ⭐
```bash
./scripts/train_with_750picacho.sh
# Time: 2.5-3.5h (GPU), Quality: 103-107/100
```

**Option 2: Synthetic Data (Faster)**
```bash
./scripts/quickstart_training.sh
# Time: 2-3h (GPU), Quality: 100-103/100
```

**Option 3: Custom Data (Advanced)**
```bash
python src/enhancements/train_hyper_reality.py --data-dir my_data/
# Custom dataset training
```

### ✅ Requirements

- Python 3.10+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- GPU recommended (CUDA or Apple Silicon MPS)
- 10GB+ disk space

### 📈 Expected Results

- **Quality:** 103-107/100 (from 78/100 baseline)
- **PSNR:** +13-15 dB improvement
- **SSIM:** +28-31% improvement
- **Materials:** Excellent realism
- **Architecture:** Room-aware enhancements

**Status:** ✅ Infrastructure complete and validated

## Installation

### Dependency Management

Transformation Portal uses a **layered dependency system** managed in the `requirements/` directory. This system provides fine-grained control over core, ML, development, and CI dependencies, and is compatible with modern Python dependency management tools (e.g., `pip-tools`).

```text
requirements/
├── base.in      # Core runtime dependencies (human-editable)
├── base.txt     # Core runtime dependencies (compiled, for pip install)
├── ml.in        # ML/AI dependencies (human-editable)
├── ml.txt       # ML/AI dependencies (compiled)
├── dev.in       # Development tools (human-editable)
├── dev.txt      # Development tools (compiled)
├── ci.in        # CI/test dependencies (human-editable)
└── ci.txt       # CI/test dependencies (compiled)
### Quick Installation

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Option 1 (Recommended for contributors): full dev environment**

```bash
pip install -r requirements-dev.txt
pip install -e .
```

This installs:

- Runtime dependencies
- Test dependencies
- Developer tooling (linting, formatting, type checking)

**Option 2: runtime + tests (CI-like environment)**

```bash
pip install -r requirements-ci.txt
pip install -e .
```

**Option 3: minimal runtime only**

```bash
pip install -r requirements.txt
pip install -e .
```

> **Important**: Installing the package in editable mode with `pip install -e .` is **required** for:
> - Using CLI console scripts (`luxury-tiff-batch`, etc.)
> - Importing from the `transformation_portal` package
> - Running the test suite correctly  
> The editable installation makes the package importable and registers command-line entry points.

### Optional Extras

If your packaging configuration defines extras like `[tiff]`, `[ml]`, `[dev]`, `[all]`, you can install them as:

```bash
# Optional: Install extras (if defined)
pip install -e ".[tiff]"   # 16-bit TIFF processing
pip install -e ".[ml]"     # ML extras for AI pipelines
pip install -e ".[dev]"    # pytest, linting, dev tooling
pip install -e ".[all]"    # everything
```

### Verify Installation

```bash
# Test depth pipeline
python -c "from depth_pipeline import ArchitecturalDepthPipeline; print('✓ Depth Pipeline ready')"

# Test Material Response
python -c "from material_response import MaterialResponse; print('✓ Material Response ready')"

# Run test suite
make test-fast
```

### Process Your First Image

```bash
# Depth-aware enhancement
python depth_pipeline/pipeline.py --input render.jpg --output enhanced.jpg

# TIFF batch processing (requires pip install -e . first)
luxury-tiff-batch input_folder/ output_folder/ --preset signature

# AI render refinement
python lux_render_pipeline.py --input bedroom.jpg --out ./enhanced --prompt "luxury bedroom interior" --material-response
```

---

## Supported File Formats

Transformation Portal supports a wide range of image and video formats across its pipelines.

### Image Formats

**Universal Support (All image pipelines):**

- PNG (.png) - Lossless, alpha channel support
- JPEG (.jpg, .jpeg) - Lossy, widely compatible
- TIFF (.tif, .tiff) - 16-bit precision with metadata preservation
- WebP (.webp) - Modern compression
- BMP (.bmp) - Uncompressed bitmap

All formats are case-insensitive (`.PNG`, `.Png`, `.png` all accepted).

### Video Formats

**Luxury Video Master Grader:**

- MP4 (.mp4) - H.264, H.265/HEVC
- MOV (.mov) - ProRes, QuickTime
- AVI (.avi) - Various codecs
- MKV (.mkv) - Matroska container
- HDR support: PQ (HDR10), HLG (Hybrid Log-Gamma)

### Pipeline-Specific Recommendations

| Pipeline               | Best Format      | Notes                                         |
|------------------------|------------------|-----------------------------------------------|
| TIFF Batch Processor   | 16-bit TIFF      | Requires `tifffile` for full precision        |
| Depth Pipeline         | PNG, TIFF        | Lossless for architectural rendering          |
| Lux Render Pipeline    | PNG, TIFF        | AI enhancement works with any PIL format      |
| Material Response      | TIFF, PNG        | High-res input recommended (4K+)              |
| Video Grading          | ProRes MOV       | Master format for color grading               |

### Installation for Full Format Support

```bash
# Install base image support (included in requirements.txt)
pip install Pillow

# Install high-fidelity TIFF support (16-bit precision)
pip install tifffile imagecodecs

# FFmpeg required for video (system package)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

---

## Core Components

### 🧠 Context-Aware Rendering System (NEW!)

Revolutionary intelligence-driven rendering that reads architectural documents and adapts processing to each specific space.

#### Workflow:
1. Extract Context - Parse PDFs (floor plans, elevations, specs) to extract architectural intelligence
2. Derive Strategy - Automatically determine optimal processing for each rendering
3. Process with Intelligence - Apply context-aware depth, materials, and color grading

#### Capabilities:
- ✅ Document Intelligence - Extracts room types, dimensions, materials, design style from PDFs
- ✅ Room-Specific Optimization - Kitchen, bedroom, bathroom, living, outdoor each get tailored processing
- ✅ Material-Aware - Prioritizes enhancement for actual materials used in design
- ✅ Style-Consistent - Color grading respects architectural design language
- ✅ Dimension-Aware - Depth processing accounts for actual room proportions
- ✅ Three Quality Tiers - Standard (30-45s), Premium (60-90s), Ultimate (3-5min with 4K)

#### Example Workflow:
```bash
# Step 1: Extract architectural context from PDF
python scripts/architectural_context_extractor.py     "750_Picacho_Plans.pdf"     --output extracted_context     --verbose

# Step 2: Process rendering with context intelligence
python scripts/premium_context_pipeline.py     "input_images/Kitchen_Rendering.tiff"     --context "extracted_context/750_Picacho_context.json"     --quality premium     --output output_premium

# Or use shortcut:
bash scripts/context_aware_quickstart.sh  # Interactive guide
```

#### What Gets Extracted:
- Room types and dimensions (e.g., Kitchen: 18' x 14.5', 10' ceiling)
- Material palette (Quartzite, Stainless Steel, White Oak)
- Design style (Modern, Traditional, Contemporary, etc.)
- Floor plans and elevations (as images)
- Project metadata (name, number, address)

#### How It Adapts:
- Kitchen → Bright lighting, balanced depth, metal/stone/glass emphasis, neutral temperature
- Bedroom → Soft lighting, atmospheric depth, wood/fabric emphasis, warm temperature
- Bathroom → Soft lighting, stone/glass/metal, spa aesthetic, neutral temperature
- Outdoor → Natural lighting, atmospheric depth, stone/concrete, enhanced perspective

#### Performance:
- Context extraction: 5-60 seconds (depending on PDF size)
- Strategy derivation: < 100ms
- Full pipeline: 30-45s (standard), 60-90s (premium), 3-5min (ultimate)

---

### 🌟 Professional Pipeline (NEW!)

Fully-integrated orchestrator combining all pipeline stages into a unified, production-ready workflow.

#### Capabilities:
- ✅ 5-Stage Integration - Depth → AI → Material → Grading → Finishing
- ✅ 10 Professional Presets - Optimized for common use cases
- ✅ Batch Processing - 400-600 images/hour throughput
- ✅ Intelligent Stage Management - Enable/disable stages as needed
- ✅ Performance Optimized - Apple Silicon (CoreML + MPS) + CUDA support

#### Quick Start:
```bash
# Single image with preset
python pro_pipeline.py process render.jpg --preset architectural-hero --out ./enhanced

# Batch processing
python pro_pipeline.py batch ./renders --preset interior-dramatic --out ./final

# List available presets
python pro_pipeline.py list-presets
```

---

### Depth Pipeline

State-of-the-art depth-aware image processing using Depth Anything V2 for architectural rendering.

#### Key Features:
- Monocular depth estimation (24-65ms per image on M4 Max)
- Apple Neural Engine optimization via CoreML
- Depth-aware denoising with edge preservation
- Zone-based tone mapping (AgX, Reinhard, Filmic)
- Atmospheric effects (haze, aerial perspective)
- Depth-guided clarity enhancement
- LRU caching for 10-20x speedup in iterative workflows

#### Usage:
```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load preset configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Process single image
result = pipeline.process_render('interior.jpg')
pipeline.save_result(result, 'output/')

# Batch process
from pathlib import Path
image_paths = Path('input/').glob('*.jpg')
results = pipeline.batch_process(image_paths, output_dir='output/')
```

---

### Lux Render Pipeline

AI-powered render refinement combining ControlNet, Stable Diffusion, and intelligent upscaling.

#### Example:
```bash
python lux_render_pipeline.py   --input bedroom_render.jpg   --out ./enhanced   --prompt "minimalist bedroom interior, natural daylight, oak wood floor"   --neg "low detail, cartoon, blurry"   --width 1024 --height 768 --steps 30 --strength 0.45   --material-response --texture-boost 0.28   --brand_text "The Veridian | Penthouse 21B" --logo ./brand/logo.png
```

---

### Luxury TIFF Batch Processor

High-end workflow for polishing large-format TIFF photography with metadata preservation.

#### Usage:
```bash
# After installing the package (pip install -e .)
luxury-tiff-batch input_folder/ output_folder/ --preset signature
```

---

### Luxury Video Master Grader

FFmpeg-based video color grading with LUT application and batch processing.

#### Usage:
```bash
python luxury_video_master_grader.py input_video.mp4 output_video.mp4 --lut path/to/lut.cube
```

---

### Material Response System

Proprietary surface-aware rendering technology that analyzes and enhances how different materials interact with light.

```python
from material_response import MaterialResponse, SurfaceType

mr = MaterialResponse()
result = mr.enhance(
    image,
    surfaces=[SurfaceType.WOOD, SurfaceType.METAL, SurfaceType.GLASS],
    strength=0.7
)
```

---

### Board Material Aerial Enhancer

Material-aware palette assignment for aerial photography using clustering and texture blending.

```bash
python board_material_aerial_enhancer.py aerial_image.jpg output_enhanced.jpg
```

---

## LUT Collection

Professional color grading LUTs for film emulation and location aesthetics.

---

## Managing Dependencies 🔧

### For Contributors

Use the development requirements for reproducible local environments:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

This installs:

- Core runtime dependencies
- Test dependencies
- Developer tooling (linting, formatting, type checking)

### Adding New Dependencies

The project uses a layered dependency system in the `requirements/` directory. To add a new dependency:

1. **Edit the appropriate `.in` file** in the `requirements/` directory:
   - **Core runtime dependency** → edit `requirements/base.in`
   - **ML/AI dependency** → edit `requirements/ml.in`
   - **Test-only dependency** → edit `requirements/dev.in`
   - **CI/CD tooling** → edit `requirements/ci.in`

2. **Recompile the pinned requirements**:
   ```bash
   cd requirements/
   make compile
   ```

3. **Commit both `.in` and `.txt` files**:
   ```bash
   git add requirements/
   git commit -m "Add new dependency: package-name"
   ```

After changes, refresh your environment:

```bash
pip install -r requirements/all.txt
pip install -e .
```

### Updating Dependencies

To update all dependencies to their latest allowed versions:

```bash
cd requirements/
make update
```

This respects the version constraints in `.in` files while finding the newest compatible versions. After updating, commit both `.in` and `.txt` files.

---

## Developer Tools

- Decision Decay Dashboard
- HDR Production Pipeline
- Prophetic Orchestrator
- Temporal Evolution

(See relevant docs in `tools/` and `docs/` for details.)

---

## Testing

### Run Test Suite

```bash
# Fast tests only
make test-fast

# Full test suite
make test-full

# CI simulation (linting + tests)
make ci

# Specific test file
pytest tests/test_depth_pipeline.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Run async pipeline tests (requires pytest-asyncio)
pytest tests/test_async_pipeline.py tests/test_async_stages.py -v
```

### Test Dependencies

The test suite requires `pytest-asyncio` for async tests. This is included in:
- Development dependencies: `pip install -e ".[dev]"`
- CI dependencies: `pip install -r requirements-ci.txt`

If running tests locally and encountering async test failures, ensure pytest-asyncio is installed:
```bash
pip install pytest-asyncio
```

---

## Requirements

### System Requirements
- Python: 3.10+
- OS: macOS (M1/M2/M3/M4), Linux, Windows
- RAM: 16GB minimum, 36GB+ recommended for batch processing
- GPU: CUDA-capable GPU or Apple Silicon with Neural Engine

### Python Dependencies (Overview)

**Core (see `requirements.txt` for full list):**
- numpy>=1.24,<3
- Pillow>=10.0.0,<12
- scipy>=1.10,<2
- torch>=2.0,<3
- typer>=0.12,<1
- diffusers, transformers, controlnet-aux, realesrgan, torchvision, opencv-python
- tifffile, imagecodecs

---

## License

Professional use permitted with attribution.

**Component Licenses:**
- Pipeline code: Proprietary with attribution requirements
- Depth Anything V2 Small: Apache 2.0 License
- Depth Anything V2 Base/Large: CC-BY-NC-4.0 (non-commercial)
- LUT Collection: Attribution required

---

## Support and Contact

Author: Richard Cheetham  
Brand: Carolwood Estates · RACLuxe Division  
Email: info@racluxe.com

Resources:

- GitHub Issues: Bug reports and feature requests
- Documentation: See inline code documentation and `docs/`
- Examples: See `examples/` directory

---

**Last Updated: 2025-11-27**
