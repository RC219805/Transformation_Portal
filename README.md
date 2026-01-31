[![CI](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/RC219805/Transformation_Portal/actions/workflows/build.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Release](https://img.shields.io/github/v/release/RC219805/Transformation_Portal?sort=semver)](https://github.com/RC219805/Transformation_Portal/releases)

# Transformation Portal

Professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production.

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
```

Install the release:
```bash
pip install "git+https://github.com/RC219805/Transformation_Portal.git@v2.0.0"
```

⸻

## Major Feature: Context-Aware Rendering (Nov 2025)

Context-Aware Rendering extracts architectural intelligence from construction documents (floor plans, elevations, specifications) and uses that context to inform processing decisions.
- Architectural context extraction from PDFs (room types, dimensions, materials, design style)
- Room-specific strategy derivation (kitchen, bedroom, bath, living, outdoor)
- Dimension-aware depth decisions (proportion-respecting depth logic)
- Style-consistent color decisions aligned to design language
- Document provenance: explicit linkage from construction docs → final render decisions

Docs:
- docs/CONTEXT_AWARE_RENDERING.md

⸻

## What this repository provides

Core capabilities:
- Context-aware rendering workflows (document-informed processing)
- Depth-aware enhancement (monocular depth + depth-guided processing)
- AI-powered refinement (optional ML stack)
- Material Response technology (surface-aware finishing)
- Professional grading looks (LUT library for film/location/material aesthetics)
- TIFF workflows (high bit-depth + metadata preservation, where supported)
- Video grading workflows (FFmpeg-based pipelines)

⸻

⸻

## Quick Start

1) Clone (recommended for development / local ops)

```bash
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install (choose your environment)

**Option A — Minimal runtime**
```bash
pip install -r requirements.txt
pip install -e .
```

**Option B — Runtime + tests (CI-like)**
```bash
pip install -r requirements-ci.txt
pip install -e .
```

**Option C — Full dev environment**
```bash
pip install -r requirements-dev.txt
pip install -e .
```

3) Verify installation

```bash
python verify_core.py
```

⸻

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

⸻

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

⸻

## Supported File Formats (summary)

**Images:**
- PNG, JPEG, TIFF/TIF, WebP, BMP (case-insensitive)

**Video:**
- MP4, MOV, AVI, MKV (codec/container dependent)
- HDR pipelines supported where FFmpeg metadata and filters allow (PQ/HLG workflows)

⸻

## System Requirements
- Python: 3.10+
- FFmpeg: 6+ (for video workflows)
- Hardware: CPU-only supported; GPU/Apple Silicon acceleration optional depending on pipeline

CI note:
- Core tests run on Python 3.10 and 3.12
- ML tests run on Python 3.11
- Lint runs on Python 3.12

⸻

## Testing

Fast local run:
```bash
pytest -v tests/ -k "not (ml or slow)"
```

ML/slow tests (requires ML extras):
```bash
pytest -v tests/ -k "ml or slow"
```

Repo Make targets may exist (see Makefile):
```bash
make test-fast
make test-full
make ci
```

⸻

## License

Professional use permitted with attribution.

Component licenses:
- Pipeline code: proprietary with attribution requirements
- Depth Anything V2 Small: Apache 2.0
- Depth Anything V2 Base/Large: CC-BY-NC-4.0 (non-commercial)
- LUT collection: attribution required

⸻

## Support and Contact

Author: Richard Cheetham
Brand: Carolwood Estates · RACLuxe Division
Email: info@racluxe.com

Resources:
- GitHub Issues: bug reports and feature requests
- Documentation: docs/
- Examples: examples/

⸻

Last Updated: 2026-01-31
