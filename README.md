[![CI/CD](https://github.com/RC219805/Transformation_Portal/actions/workflows/ci-consolidated.yml/badge.svg)](https://github.com/RC219805/Transformation_Portal/actions/workflows/ci-consolidated.yml)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1348%20passed-brightgreen.svg)](https://github.com/RC219805/Transformation_Portal/actions)

# Transformation Portal

**Production-grade image processing service for architectural visualization and luxury real estate rendering.**

---

## ⚡ Quick Start (2 Minutes)

**New user?** → [**QUICKSTART.md**](QUICKSTART.md) ← **Start here**

```bash
# Install and process images
pip install -e .
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury
```

**That's it.** Security-hardened, production-validated, 127-400 images/hour.

---

## 🎯 The Golden Path

**`lux_depth_v2`** is the primary workflow for 95% of use cases:

✅ **Security Hardened** - CVE-2024-27763 mitigated, input validation  
✅ **Production Validated** - 1,348 tests passing, 127-400 img/hr throughput  
✅ **Deployment Ready** - Docker stack, Prometheus, health checks  
✅ **Feature Frozen** - Predictable, stable, no breaking changes  
✅ **16-bit Precision** - Archival-grade quality maintained

### When to Use the Golden Path

**Use `lux_depth_v2` if**:
- You're processing architectural renders or images
- You need reliable, predictable behavior
- You want security-hardened processing
- You're deploying to production

**Key Resources**:
- 📖 [**QUICKSTART.md**](QUICKSTART.md) - Get started in 2 minutes
- 📘 [Phase 2 User Guide](docs/PHASE2_USER_GUIDE.md) - Complete walkthrough
- ⚡ [Quick Reference Card](docs/QUICK_REFERENCE_PHASE2.md) - CLI cheat sheet
- 🔒 [Security Guide](lux_depth_v2/SECURITY.md) - Best practices

---

## 🔀 Advanced Workflows

**Only use these if the Golden Path doesn't meet your needs.**

| Workflow | Use When | Documentation |
|----------|----------|---------------|
| **Async Pipeline** | 1000+ images, need 3-5x throughput | [docs/advanced/ASYNC_PIPELINE.md](docs/advanced/ASYNC_PIPELINE.md) |
| **Context-Aware** | Document-driven intelligence | [docs/advanced/CONTEXT_AWARE_RENDERING.md](docs/advanced/CONTEXT_AWARE_RENDERING.md) |
| **Material Response** | Custom material enhancement | [docs/advanced/MATERIAL_RESPONSE.md](docs/advanced/MATERIAL_RESPONSE.md) |
| **Video Processing** | Processing video files (not images) | [docs/advanced/VIDEO_PROCESSING.md](docs/advanced/VIDEO_PROCESSING.md) |

**Research/Experimental**: [docs/research/](docs/research/) (⚠️ NOT production-ready)

---

## 🐳 Production Deployment

```bash
# 1. Clone and configure
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal
cp deployment/.env.production.example .env.production

# 2. Start Lux Depth V2 service (CPU)
docker-compose up -d lux-depth-v2-service

# 3. Verify service is running
curl http://localhost:8088/health

# 4. Process an image
curl -X POST http://localhost:8088/v2/process \
  -F "image=@input.jpg" \
  -F "preset=interior_luxury"
```

### GPU-Accelerated Deployment

```bash
# Start GPU service (requires NVIDIA Docker)
docker-compose up -d lux-depth-v2-gpu

# Verify GPU is available
docker exec lux-depth-v2-gpu python -c "import torch; print(torch.cuda.is_available())"

# Process with GPU acceleration (3-5x faster)
curl -X POST http://localhost:8089/v2/process \
  -F "image=@input.jpg" \
  -F "preset=exterior_showcase"
```

### Full Production Stack (with Monitoring)

```bash
# Start complete stack: service + Prometheus + Grafana
docker-compose -f deployment/docker-compose.production.yml up -d

# Access services:
# - Lux Depth V2 API: http://localhost:8088
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/<password-from-env>)
```

### Key Resources

**Getting Started:**
- **📖 [Phase 2 User Guide](docs/PHASE2_USER_GUIDE.md)** - 🆕 **Start here!** Complete walkthrough with examples
- **⚡ [Quick Reference Card](docs/QUICK_REFERENCE_PHASE2.md)** - 🆕 One-page cheat sheet for CLI and common use cases
- **📊 [Quality Tiers Guide](docs/QUALITY_TIERS.md)** - Preset comparison and selection guide

**Technical Documentation:**
- **📖 [Phase 2 Deployment Guide](docs/PHASE2_DEPLOYMENT_GUIDE.md)** - Complete setup instructions
- **🔒 [Security Guide](lux_depth_v2/SECURITY.md)** - Security best practices
- **📚 [Lux Depth V2 README](lux_depth_v2/README.md)** - Module documentation
- **⚙️ [Environment Config](deployment/.env.production.example)** - Configuration reference
- **🚀 [Performance Benchmarks](docs/PHASE2_PERFORMANCE.md)** - Throughput and timing data
- **🔧 [CI/CD Integration](docs/CI_PHASE2_INTEGRATION.md)** - Automated testing and deployment

### Security Validation

All deployments pass automated security checks:

```bash
# Verify CVE-2024-27763 mitigation
python scripts/utilities/verify_no_basicsr_imports.py --check-pkg

# Run full security audit
make security-audit

# Check Docker image security
docker exec lux-depth-v2-service python -c "import basicsr"
# Expected: ImportError (correct - package not present)
```

### Architecture Highlights

**Service Components**:
- `lux-depth-v2-service` - Main FastAPI service (CPU)
- `lux-depth-v2-gpu` - GPU-accelerated variant (CUDA)
- `lux-depth-v2-worker` - Batch processing worker
- `prometheus` - Metrics collection
- `grafana` - Visualization dashboards

**Security Features**:
- Non-root user execution
- Input validation and sanitization
- Rate limiting (10 req/min default)
- File size limits (100MB default)
- Health check endpoints
- Automated vulnerability scanning

**Monitoring Metrics**:
- `lux_depth_requests_total` - Request count
- `lux_depth_request_duration_seconds` - Latency
- `lux_depth_errors_total` - Error tracking
- `lux_depth_queue_size` - Processing queue
- `lux_depth_gpu_memory_bytes` - GPU utilization

---

## 📚 Documentation

**Production Use**:
- [QUICKSTART.md](QUICKSTART.md) - Get started in 2 minutes
- [docs/PHASE2_USER_GUIDE.md](docs/PHASE2_USER_GUIDE.md) - Complete user guide
- [docs/QUICK_REFERENCE_PHASE2.md](docs/QUICK_REFERENCE_PHASE2.md) - CLI reference
- [lux_depth_v2/SECURITY.md](lux_depth_v2/SECURITY.md) - Security best practices
- [deployment/](deployment/) - Docker deployment guides

**Advanced Users**:
- [docs/advanced/](docs/advanced/) - Advanced workflows (async, context-aware, material, video)
- [docs/architecture/](docs/architecture/) - System architecture

**Researchers**:
- [docs/research/](docs/research/) - Experimental features (⚠️ NOT production-ready)

**Development**:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [tests/](tests/) - Test suite (1,348 tests)
- [.github/workflows/](.github/workflows/) - CI/CD pipelines

---

## 🏗️ Architecture

**Core System**: `lux_depth_v2/` (feature-frozen production module)

**6-Stage Pipeline**:
1. Loading & Validation
2. AI Upscaling (SwinIR/Real-ESRGAN) - 4x resolution
3. Depth Processing (Depth Anything V2) - Architectural depth awareness
4. Material Response - 8 surface types (wood, metal, glass, fabric, stone, concrete, leather, water)
5. Color Grading - Film emulation + location LUTs
6. Export - 16-bit TIFF with metadata preservation

**Performance**:
- Throughput: 127-400 images/hour (batch)
- Latency: 2-5 seconds/image (service mode)
- GPU Acceleration: 3-5x faster with CUDA
- Memory: < 4GB per image (16-bit TIFF)

**Security**:
- CVE-2024-27763 mitigated (no vulnerable packages)
- Input validation (path traversal protection)
- Rate limiting (10-100 req/min)
- Non-root containers
- Automated security scanning (CI/CD)

---

## 📚 Reference Material

**Everything below this point is reference documentation for advanced users, contributors, and researchers.**

**New users and production deployers should stop here.** You have everything you need to deploy and use the system effectively.

---

# Appendix A — Historical & Reference Documentation

## 🧪 Testing & Quality

**Test Coverage**:
- 1,348 tests passing (100% pass rate)
- Unit, integration, and performance tests
- Python 3.10, 3.11, 3.12 tested
- CPU and GPU configurations validated

**Run Tests**:
```bash
# Fast tests (development)
make test-fast

# Full test suite
make test-full

# Linting
make lint

# Security audit
make security-audit
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Core Stability Policy**:
- `lux_depth_v2/` is **feature-frozen** (security/bugs/performance only)
- Advanced features follow standard development practices
- Experimental features in `docs/research/` have minimal governance

---

## 📜 License

Attribution-based license. See repository for details.

---

## 🆘 Support

**Issues**: Open GitHub issue with error logs  
**Security**: See [SECURITY.md](SECURITY.md)  
**Questions**: Check [docs/](docs/) first, then open discussion

---

*Built with discipline. Operated with confidence.*
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

- ✅ **Lux Depth V2 Pipeline** 🆕🔥 - Production-ready depth processing with GPU acceleration, FastAPI service, security hardening
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

### Lux Depth V2 Features (Phase 2 - December 2025)

**Processing Pipeline**:
- 🎨 **GPU-Accelerated Post-Processing** - Torch-based grading, clarity, sharpening (3-5x faster)
- 🔍 **Advanced Material Segmentation** - ONNX/SegFormer/Heuristic backends for 8 material types
- 🌊 **Depth-Aware Enhancement** - Zone-based tone mapping respects depth information
- 🎯 **Safe AI Detail Transfer** - Color/luma drift guardrails prevent artifacts
- 📐 **UHR Tiling** - Process 324MP+ images with configurable tile size/overlap
- 🎚️ **Production Presets** - Interior luxury, exterior showcase, architectural detail

**Service & Deployment**:
- 🚀 **FastAPI REST API** - RESTful endpoints for real-time processing
- 🔒 **Security Hardened** - CVE-2024-27763 mitigation, input validation, rate limiting
- 📊 **Prometheus Metrics** - Request rate, latency, errors, GPU memory tracking
- 🐳 **Docker Containerization** - Multi-stage builds, health checks, resource limits
- 👤 **Non-root Execution** - All containers run as unprivileged users
- ⚖️ **Load Balancing Ready** - Horizontal scaling with multiple service instances

**Quality & Validation**:
- ✅ **Production Validation Framework** - Synthetic reference and real-world modes
- 📈 **Multiple Metrics** - SSIM, PSNR, LPIPS, NIMA for quality assessment
- 🔬 **Baseline Comparison** - Compare against industry tools (Topaz, Adobe)
- 📝 **Reproducibility Stamping** - Git commit, config hash, device info in reports
- 🛡️ **AI Safety Checks** - Automatic validation prevents color/luma drift

**Performance**:
- ⚡ **24-65ms** per image for depth estimation (M4 Max, 518px)
- 🏎️ **127-400 images/hour** batch throughput (CPU vs GPU)
- 🧠 **Persistent Models** - Service mode keeps models loaded for low latency
- 💾 **16-bit Precision** - End-to-end high-quality workflow
- 🔄 **Async Processing** - Non-blocking API with queue management

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

## 🎓 Advanced: Model Training

⚠️ **Advanced Feature** - Not part of default production workflow

Neural network training infrastructure for researchers and advanced users who need custom model adaptation.

**Requires**: GPU, 10GB+ disk, 2-3 hours training time  
**Production users**: Use pre-trained models in `lux-depth-v2` instead

📚 **Complete training documentation**: [docs/training/TRAINING_GUIDE.md](docs/training/TRAINING_GUIDE.md)

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
