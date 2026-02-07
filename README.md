[![CI](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml)
[![APEX Performance](https://github.com/RC219805/Transformation_Portal/actions/workflows/apex_performance.yml/badge.svg?branch=main)](https://github.com/RC219805/Transformation_Portal/actions/workflows/apex_performance.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Release](https://img.shields.io/github/v/release/RC219805/Transformation_Portal?sort=semver)](https://github.com/RC219805/Transformation_Portal/releases)

# Transformation Portal

Professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production.

**📊 [Performance Dashboard](https://rc219805.github.io/Transformation_Portal/apex/)** | **📈 [Latest Metrics](https://rc219805.github.io/Transformation_Portal/apex/latest.html)**

---

## Current Release: v2.0.0 (Golden Path)

Transformation Portal v2.0.0 is the first stable release with production-ready contracts and preset governance.

Key improvements in v2.0.0:
- Versioned API contracts (schema-aligned payloads)
- Preset stability taxonomy (stable / canary / experimental) discoverable via CLI
- Service hardening including `/ready` for readiness checks

Quick discovery:
```bash
lux-depth-v2 --list-stable
lux-depth-v2 --describe-preset interior_luxury

# If console scripts aren't on PATH, run as module:
python -m lux_depth_v2 --list-stable
python -m lux_depth_v2 --describe-preset interior_luxury
```

Install the release:
```bash
pip install "git+https://github.com/RC219805/Transformation_Portal.git@v2.0.0"
```

---

## Major Feature: Context-Aware Rendering (Nov 2025)

Context-Aware Rendering extracts architectural intelligence from construction documents (floor plans, elevations, specifications) and uses that context to inform processing decisions.
- Architectural context extraction from PDFs (room types, dimensions, materials, design style)
- Room-specific strategy derivation (kitchen, bedroom, bath, living, outdoor)
- Dimension-aware depth decisions (proportion-respecting depth logic)
- Style-consistent color decisions aligned to design language
- Document provenance: explicit linkage from construction docs → final render decisions

Docs:
- docs/CONTEXT_AWARE_RENDERING.md

---

## What this repository provides

Core capabilities:
- Context-aware rendering workflows (document-informed processing)
- Depth-aware enhancement (monocular depth + depth-guided processing)
- **PBR Map Generation** (Physically Based Rendering maps: normal, roughness, AO)
- AI-powered refinement (optional ML stack)
- Material Response technology (surface-aware finishing)
- Professional grading looks (LUT library for film/location/material aesthetics)
- TIFF workflows (high bit-depth + metadata preservation, where supported)
- Video grading workflows (FFmpeg-based pipelines)

---

## Depth Models: Commercial vs. Research

Transformation Portal supports depth models across two tiers with different licensing and use cases.

### Production (Commercial)
- **Depth Anything V3 (V2 commercial variant):** Fully supported, production-ready
- **Use for:** Commercial applications, products, revenue-generating services
- **Licensing:** Commercial-friendly licensing
- **Default:** All standard presets use this tier

### Research & Non-Commercial
- **Depth Anything V3.1 (DA3 1.1, CC BY-NC 4.0):** Available for research/academic use only
- **Use for:** Academic research, benchmarking, non-profit projects
- **Licensing:** CC BY-NC 4.0 (non-commercial research only)
- **Enabled by:** Setting `non_commercial_ok=True` in EnhanceConfig
- **Example Preset:** `depth-anything-v3.1-research-m4` (Apple Silicon optimized)

**Important:** DA3 1.1 is prohibited for commercial use. If you plan to use these models in a commercial product or service, use the commercial DA3 V2 variants instead. See [ADR-0015: DA3 1.1 Non-Commercial Research Tier](docs/architecture/adr-0015-da3-1-1-non-commercial-research-tier.md) for detailed governance.

### Research Preset Example

```python
from transformation_portal.lux_depth_v3 import EnhanceConfig, Preset

# Non-commercial research (requires explicit opt-in)
config = EnhanceConfig(
    preset=Preset.RESEARCH_DA31_M4,
    non_commercial_ok=True,  # Acknowledge CC BY-NC 4.0 restrictions
    depth_device="mps",       # Apple Silicon
)
```

---

## Backend Selection

Lux Depth V3 supports multiple depth estimation backends with automatic fallback for robustness.

### Available Backends

| Backend | Model | License | Focal Length | Metric Depth | Checkpoint Required |
|---------|-------|---------|--------------|--------------|---------------------|
| `da3` (default) | Depth Anything V3 | MIT | ❌ | ❌ | No (auto-download) |
| `depth_pro` | Apple Depth Pro | Apple ML Research | ✅ | ✅ | Yes (1.9 GB) |

### Usage

**Default (DA3):**
```bash
lux-depth-v3 --input-dir ./input --output-dir ./output
```

**Depth Pro (requires license acceptance):**
```bash
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --depth-backend depth_pro \
  --accept-apple-depth-pro-research-license true \
  --non-commercial-ok true
```

**Python API:**
```python
from transformation_portal.lux_depth_v3 import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from pathlib import Path

# Using Depth Pro
config = EnhanceConfig(
    depth_backend="depth_pro",
    depth_pro_checkpoint_path="checkpoints/depth_pro.pt",
    accept_apple_depth_pro_research_license=True,
    non_commercial_ok=True,
    depth_device="cpu",
    enable_v2=False,
)

orchestrator = EnhanceOrchestrator(config, Path("./output"))
```

### Fallback Behavior

If the requested backend is unavailable (missing checkpoint or dependencies), the system automatically falls back to DA3 with a warning logged. This ensures robustness in production environments.

### Backend Metadata

All processing manifests include backend selection metadata:
- `requested_backend`: User's requested backend
- `resolved_backend`: Actually used backend
- `resolution_status`: "success" or "fallback"
- `resolution_reason`: Explanation if fallback occurred

See [ADR-019: Backend Registry Integration](docs/architecture/decisions/ADR-019-REVISED-DECISION.md) for architectural details.

---

## Optional Dependencies

### Depth Pro (Experimental)

Apple's Depth Pro model for metric depth estimation. **Experimental tier** - for research and evaluation only.

**Installation:**
```bash
pip install depth-pro
```

**Checkpoint Download (1.9 GB):**
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
```

**License Requirements (Research-Only):**

Depth Pro uses the Apple Machine Learning Research License (AMLR), which restricts usage to **non-commercial research only**. To use Depth Pro, you must explicitly acknowledge both:

```python
from transformation_portal.lux_depth_v3 import EnhanceConfig

config = EnhanceConfig(
    depth_backend="depth_pro",
    non_commercial_ok=True,                          # Required: Acknowledge non-commercial use
    accept_apple_depth_pro_research_license=True,    # Required: Accept Apple AMLR license
    depth_device="mps",  # Apple Silicon (or "cpu" for fallback)
)
```

**⚠️ Important:** This model cannot be used for:
- Commercial products or services
- Revenue-generating applications
- Paid client work

See [Apple AMLR License](https://github.com/apple/ml-depth-pro/blob/main/LICENSE) for full terms.

**Presets:**
- `depth_pro_metric_mps.yaml` - Apple Silicon optimized
- `depth_pro_metric_cpu.yaml` - CPU fallback

**Hardware Requirements:**
- Optimized for Apple Silicon (MPS device)
- Fallback to CPU supported
- Memory: ~2 GB for model + checkpoint

**Tier Status:** Experimental - use at your own risk. Default backend remains Depth Anything V3.

---

## PBR Map Generation

**New in v2.0**: Standalone PBR processor for generating Physically Based Rendering maps from depth data.

### Quick Start - PBR Only

Generate PBR maps from existing depth:

```python
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Use premium quality preset
config = get_preset("premium").to_pbr_config()

# Generate from cached depth (2.3x faster than full pipeline)
paths = PBRProcessor.from_cached_depth(
    depth_path="output/scene1_depth.npy",
    config=config,
    output_dir="output/pbr/",
    base_name="scene1"
)

# Output: scene1_normal.png, scene1_roughness.png, scene1_ao.png
```

### When to Use PBRProcessor vs Full Pipeline

**Use PBRProcessor** (standalone) when:
- You already have depth maps and only need PBR
- Iterating on PBR parameters (2.3x faster than re-running depth)
- Integrating PBR into custom workflows
- Processing depth from external sources

**Use Orchestrator** (full pipeline) when:
- Starting from RGB images (need depth estimation)
- Running complete enhancement workflow
- Need depth + PBR + V2 enhancement in one pass

### Available Presets

**Quality Tiers:**
- `standard` - Balanced quality/speed (typical batch processing)
- `premium` - Maximum quality (hero shots, marketing)
- `draft` - Fast preview (internal review)

**Material-Optimized:**
- `wood` - Emphasizes grain texture
- `metal` - Lower roughness for polished surfaces
- `glass` - Heavy smoothing for flat surfaces
- `stone` - High detail for texture
- `fabric` - Moderate parameters for textiles

### Performance Benefits

- **PBR-only workflow**: ~3,000 images/hour (vs ~1,277 for full pipeline)
- **Memory-only mode**: No file I/O overhead
- **Iterative tuning**: 2x faster when testing multiple presets

See [PBR Processor Quick Start](docs/PBR_PROCESSOR_QUICKSTART.md) for detailed guide.

---

## Quick Start

1) Clone (recommended for development / local ops)

```bash
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install (choose your environment)

**Option A - Minimal runtime**
```bash
pip install -r requirements.txt
pip install -e .
```

**Option B - Runtime + tests (CI-like)**
```bash
pip install -r requirements-ci.txt
pip install -e .
```

**Option C - Full dev environment**
```bash
pip install -r requirements-dev.txt
pip install -e .
```

3) Verify installation

```bash
python verify_core.py
```

---

## Dependency Management

This repo uses two layers:
1. Convenience pinned files at repo root:
   - `requirements.txt`
   - `requirements-ci.txt`
   - `requirements-dev.txt`
   - `requirements-lint.txt`

2. Source-of-truth layered inputs in `requirements/` for maintainers:

```
requirements/
├── base.in      # Core runtime deps (human-editable)
├── base.txt     # Compiled/pinned
├── ml.in        # ML/AI deps (human-editable)
├── ml.txt       # Compiled/pinned
├── dev.in       # Dev deps (human-editable)
├── dev.txt      # Compiled/pinned
├── ci.in        # CI/test deps (human-editable)
└── ci.txt       # Compiled/pinned
```

If you update `.in` files, recompile and commit both `.in` and `.txt` outputs:
```bash
cd requirements/
make compile
```

---

## Repository Layout (high level)

```
assets/       # LUTs, branding, look assets
config/       # YAML presets and configuration
docs/         # Architecture, guides, reports
examples/     # Usage examples
requirements/ # Layered dependency sources (pip-tools style)
scripts/      # Operational scripts / pipeline runners
src/          # Installable package source
tests/        # pytest suite
tools/        # Dev/ops tools (manifests, audits, utilities)
workflows/    # Workflow artifacts / operational workflow utilities
```

---

## Supported File Formats (summary)

**Images:**
- PNG, JPEG, TIFF/TIF, WebP, BMP (case-insensitive)

**Video:**
- MP4, MOV, AVI, MKV (codec/container dependent)
- HDR pipelines supported where FFmpeg metadata and filters allow (PQ/HLG workflows)

---

## System Requirements
- Python: 3.11+
- FFmpeg: 6+ (for video workflows)
- Hardware: CPU-only supported; GPU/Apple Silicon acceleration optional depending on pipeline

CI note:
- Core tests run on Python 3.11 and 3.12
- ML tests run on Python 3.11
- Lint runs on Python 3.12

---

## Testing

Fast local run (mirrors CI core suite):
```bash
pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
```

ML tests (requires ML extras):
```bash
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
```

All tests except slow:
```bash
pytest -v tests/ -ra -m "not slow" --maxfail=1
```

Repo Make targets may exist (see Makefile):
```bash
make test-fast
make test-full
make ci
```

---

## Performance Monitoring

Transformation Portal includes performance regression detection via the Performance Ledger tool.

### Capture Baseline

```bash
python tools/performance_ledger.py \
  --manifests-dir output/prod_run/manifests \
  --output docs/performance/baselines/v2.1.0-baseline.json \
  --version "v2.1.0" \
  --backend "da3" \
  --quality-tier "standard"
```

### Compare Against Baseline

```bash
python tools/performance_ledger.py \
  --baseline docs/performance/baselines/v2.0.0-post-pr841.json \
  --compare output/test_run/manifests \
  --output perf_report.md
```

Exit codes:
- `0`: No regressions detected
- `1`: Regressions detected (blocks merge)

### Regression Thresholds

- **p95 > 10% worse:** Tail latency regression
- **mean > 15% worse:** Average performance regression
- **failure_rate > 0%:** Any new failures

See [Performance Monitoring Guide](docs/performance/README.md) for details.

---

## Documentation

**📖 Start with:** [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)

The Documentation Map is your single source of truth for finding guides, references, and technical documentation.

### Essential Docs
- **[DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)** - Complete documentation index
- **[API Documentation](docs/api/)** - Full API reference (Sphinx)
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Detailed installation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute

### Quick Links
- **API Reference:** [docs/api/](docs/api/)
- **Pipelines:** [docs/pipeline/](docs/pipeline/)
- **CI/CD:** [docs/ci/](docs/ci/)
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Lux Depth V3 CLI Guide:** [docs/LUX_DEPTH_V3_CLI_GUIDE.md](docs/LUX_DEPTH_V3_CLI_GUIDE.md)
- **Lux Depth V3 Troubleshooting:** [docs/LUX_DEPTH_V3_TROUBLESHOOTING.md](docs/LUX_DEPTH_V3_TROUBLESHOOTING.md)

---

## License

Professional use permitted with attribution.

Component licenses:
- Pipeline code: proprietary with attribution requirements
- Depth Anything V3 (commercial variant): Commercial-friendly licensing
- Depth Anything V3.1 (DA3 1.1): CC BY-NC-4.0 (non-commercial research only) ⚠️
- LUT collection: attribution required

**⚠️ Important:** DA3 1.1 is non-commercial only. Commercial applications must use DA3 V2 or equivalent commercially-licensed depth models. See [Depth Models: Commercial vs. Research](#depth-models-commercial-vs-research) above.

---

## Support and Contact

Author: Richard Cheetham
Brand: Carolwood Estates · RACLuxe Division
Email: info@racluxe.com

Resources:
- GitHub Issues: bug reports and feature requests
- Documentation: docs/
- Examples: examples/

---

Last Updated: 2026-01-31
