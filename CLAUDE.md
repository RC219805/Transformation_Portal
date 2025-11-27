# CLAUDE.md - AI Assistant Guide for Transformation Portal

## Project Overview

**Transformation Portal** is a professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production. The codebase combines AI-powered enhancement, depth-aware processing, and professional color grading into production-ready pipelines.

### Core Capabilities
- **Context-Aware Rendering**: Extracts architectural intelligence from PDFs (floor plans, elevations) to inform processing decisions
- **AI-Powered Enhancement**: Stable Diffusion XL, ControlNet, Real-ESRGAN for intelligent upscaling
- **Depth-Aware Processing**: Depth Anything V2 with Apple Neural Engine (CoreML) optimization
- **Material Response Technology**: Physics-based surface enhancement for wood, metal, glass, textiles
- **Professional Color Grading**: 16+ LUTs with Film Emulation and Location Aesthetics
- **Batch Processing**: 400-600 images/hour throughput on M4 Max

---

## Repository Structure

```
Transformation_Portal/
├── src/                              # Main installable packages
│   ├── transformation_portal/        # Core package (25+ submodules)
│   │   ├── depth/                   # Depth pipeline (models, processors, utils)
│   │   ├── pipelines/               # Processing pipelines (lux_render, unified_luxury)
│   │   ├── processors/              # Material response, video grading
│   │   ├── plugins/                 # Plugin system for extensions
│   │   ├── streaming/               # Streaming/checkpoint processing
│   │   ├── utils/                   # Shared utilities (validation, formatting)
│   │   └── ...                      # Other submodules (events, compat, vlm, etc.)
│   ├── luxury_tiff_batch_processor/ # TIFF batch processing CLI
│   └── enhancements/                # Model training infrastructure
├── tests/                            # pytest test suite (~64 test files, 70+ test functions)
│   ├── conftest.py                  # Shared pytest configuration
│   ├── foundation/                  # Foundation layer tests
│   └── perceptual/                  # Perceptual quality tests
├── scripts/                          # Utility and automation scripts
│   ├── setup/                       # Installation scripts
│   ├── pipelines/                   # Pipeline-specific scripts
│   ├── analysis/                    # Analysis and reporting tools
│   └── automation/                  # CI/CD and automation scripts
├── config/                           # YAML configuration presets
├── docs/                             # Documentation (35+ subdirectories)
├── assets/                           # Assets (LUTs, brand, projects)
│   ├── luts/                        # Color grading LUTs (.cube files)
│   │   ├── film_emulation/          # Kodak, FilmConvert emulations
│   │   ├── location_aesthetic/      # Location-specific color profiles
│   │   └── material_response/       # Physics-based surface LUTs
│   └── brand/                       # Brand assets (logos, colors)
├── data/                             # Sample data and datasets
├── basicsr_tp/                       # Vendored BasicSR (security-hardened)
├── .github/
│   ├── workflows/                   # CI/CD workflows (10+ workflows)
│   ├── agents/                      # Copilot agent configurations
│   └── copilot-instructions.md      # GitHub Copilot guide
├── requirements/                     # Layered dependency management
├── pyproject.toml                    # Package configuration
├── Makefile                          # Development task automation
└── README.md                         # Main documentation
```

---

## Development Setup

### Prerequisites
- Python 3.10+ (CI currently tests on 3.11 to conserve resources)
- FFmpeg 6+ (for video processing)
- Git
- Optional: CUDA-capable GPU or Apple Silicon (M1/M2/M3/M4) with MPS

### Quick Start

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install for development (REQUIRED - editable mode)
pip install -r requirements-dev.txt
pip install -e .

# Verify installation
make test-fast

# Run linting
make lint
```

### Installation Options

| Option | Command | Purpose |
|--------|---------|---------|
| Development | `pip install -r requirements-dev.txt && pip install -e .` | Full dev environment with tests and linting |
| CI/Testing | `pip install -r requirements-ci.txt && pip install -e .` | Minimal for running tests |
| Runtime only | `pip install -r requirements.txt && pip install -e .` | Production dependencies only |
| ML extras | `pip install -e ".[ml]"` | AI/ML dependencies (torch, diffusers, etc.) |
| All extras | `pip install -e ".[all]"` | Everything |

**IMPORTANT**: The `pip install -e .` (editable install) is **required** for:
- CLI console scripts (`luxury-tiff-batch`, etc.)
- Importing from `transformation_portal` package
- Running the test suite correctly

---

## Tech Stack

### Core Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Primary language |
| PyTorch 2.0+ | ML/AI framework |
| Depth Anything V2 | Monocular depth estimation |
| Stable Diffusion XL | AI render refinement |
| ControlNet | Edge-preserving image-to-image |
| Real-ESRGAN | Intelligent 4x upscaling |
| FFmpeg | Video processing |
| CoreML | Apple Neural Engine acceleration |
| Typer | CLI framework |

### Key Dependencies

```
# Core
numpy>=1.24,<2.3.0    # Note: pinned for opencv-python compatibility
Pillow>=10.0.0,<12
scipy>=1.15,<1.16     # Pinned for Python 3.10 compatibility
PyYAML>=6.0,<7
typer>=0.10.0,<1
tifffile, imagecodecs # 16-bit TIFF support

# ML/AI (optional)
torch>=2.0,<3
diffusers>=0.20,<1
transformers>=4.35.0,<5
controlnet-aux>=0.0.6,<1
sentence-transformers>=2.2.0,<6

# Dev/Testing
pytest>=8.0,<10
pytest-cov>=4.0,<8
hypothesis>=6.0,<7
flake8>=7.0
pylint>=3.0
```

---

## Coding Standards

### Python Style

- **PEP 8** compliance with 127-character max line length
- Use **type hints** where appropriate
- Use **dataclasses** for configuration objects
- Prefer **pathlib.Path** over string paths
- Use **f-strings** for string formatting

### File Naming

- Python scripts: `lowercase_with_underscores.py`
- Shell scripts: `lowercase_with_underscores.sh`
- Test files: `test_<module_name>.py`

### Code Organization

```python
# Good practices for this codebase:

# 1. Use lazy loading for heavy ML imports
def process_image(image_path):
    # Import heavy dependencies only when needed
    import torch
    from diffusers import StableDiffusionPipeline
    ...

# 2. Use descriptive variable names matching domain
preset = "signature_estate"
filter_graph = build_filter_chain()
tone_map_config = ZoneToneMapConfig()

# 3. Document complex algorithms
def apply_zone_tone_mapping(image, depth_map):
    """Apply zone-based tone mapping respecting depth information.

    Uses AgX/Reinhard/Filmic operators depending on zone.
    Foreground gets different treatment than background.
    """
    ...

# 4. Separate concerns: CLI, business logic, I/O
```

### Linting Configuration

- **flake8**: Critical errors only (`E9,F63,F7,F82`)
- **pylint**: Non-blocking in CI (many rules disabled, see `.pylintrc`)
- **mypy**: Type checking (optional, see `mypy.ini`)
- **Max line length**: 127 characters

### Directories Excluded from Linting

```
# .pylintrc ignore list:
- build, dist, .venv, .eggs
- tools/deprecated
- src/transformation_portal, src/luxury_tiff_batch_processor  # Still maturing
- scripts
- .backup_local, .github/agents
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py                     # Shared pytest configuration
├── foundation/                     # Foundation layer tests
├── perceptual/                     # Perceptual quality tests
├── test_*.py                       # Module-specific tests (60+ files)
└── TEST_STATUS.md                  # Test coverage status
```

### Running Tests

```bash
# Fast tests (recommended for development)
make test-fast

# Full test suite (with parallel execution if xdist installed)
make test-full

# Specific module
pytest tests/test_depth_tools.py -v

# With coverage
pytest tests/ --cov=src/transformation_portal --cov-report=html

# Skip heavy tests
pytest -k 'not video_master_grader' tests/

# Run structure validation
make test-structure
```

### Test Best Practices

1. **Mock external dependencies** (FFmpeg, ML models, file I/O) to avoid CI timeouts
2. **Use pytest fixtures** from `tests/__init__.py` for common setup
3. **Use hypothesis** for property-based testing of mathematical functions
4. **Keep tests fast**: Avoid loading large ML models unless necessary
5. **Document optional dependencies**: Mark tests requiring `tifffile`, `torch`, etc.

### CI Test Configuration

- Python 3.11 only (conserving CI resources)
- CPU-only PyTorch (`torch+cpu`) to save ~6GB disk space
- Coverage threshold: 35%
- Tests skipped: `test_realesrgan_integration.py`, `test_coreml_integration.py`

---

## CI/CD

### GitHub Actions Workflows

| Workflow | Purpose |
|----------|---------|
| `python-app.yml` | Main CI: lint → test → deploy |
| `build.yml` | Extended build with matrix testing |
| `codeql.yml` | Security scanning (CodeQL) |
| `quality-gate.yml` | Quality enforcement |
| `dependency-submission.yml` | Dependency graph updates |
| `smart-issue-management.yml` | Issue automation |
| `submit-pypi.yml` | PyPI publishing |

### CI Pipeline Stages

1. **Lint** (non-blocking): flake8 critical errors, pylint, mypy
2. **Test**: pytest with coverage on Python 3.11
3. **Deploy**: Build package, publish to Test PyPI (on main push)
4. **Cleanup**: Remove temporary files and caches

### Local CI Simulation

```bash
# Quick CI check
make ci

# Full CI simulation
make ci-full

# Validate workflow configs
make validate-ci
```

---

## Common Tasks

### Adding a New Pipeline Preset

1. Create YAML config in `config/` (use existing presets as templates)
2. Define parameters: depth model, tone mapping, denoising, effects
3. Add tests in `tests/test_pipeline.py`
4. Document in README

### Adding Dependencies

1. Edit appropriate `.in` file in `requirements/`:
   - Runtime: `requirements/base.in`
   - ML/AI: `requirements/ml.in`
   - Dev: `requirements/dev.in`
   - CI: `requirements/ci.in`
2. Recompile: `cd requirements/ && make compile`
3. Commit both `.in` and `.txt` files

### Working with CLI Commands

Entry points defined in `pyproject.toml`:
- `transform-render` → `transformation_portal.cli:render_cli`
- `transform-process` → `transformation_portal.cli:process_cli`
- `luxury-tiff-batch` → `luxury_tiff_batch_processor.cli:main`

### Running Pipelines

```bash
# Depth-aware enhancement
python -m transformation_portal.depth.pipeline --input render.jpg --output enhanced.jpg

# TIFF batch processing
luxury-tiff-batch input_folder/ output_folder/ --preset signature

# Context-aware rendering
python scripts/architectural_context_extractor.py "plans.pdf" --output context/
python scripts/premium_context_pipeline.py "image.tiff" --context "context/data.json"
```

---

## Key Architectural Patterns

### Lazy Loading

Heavy ML dependencies are imported only when needed:

```python
def load_model():
    # Lazy import to speed up CLI startup
    import torch
    from transformers import AutoModel
    ...
```

### Plugin System

Located in `src/transformation_portal/plugins/`:
- Extensible architecture for custom processors
- Plugin discovery and registration
- Configuration via YAML

### Streaming Processing

Located in `src/transformation_portal/streaming/`:
- Checkpoint-based processing for large batches
- Progress tracking with `tqdm`
- Resume capability for interrupted jobs

### Compatibility Layer

Located in `src/transformation_portal/compat/`:
- Shims for optional dependencies
- Version-specific code paths
- Decorator-based feature detection

---

## Security Considerations

### CVE-2024-27763 Mitigation

BasicSR vulnerability is mitigated via vendored `basicsr_tp` package:
- Only RRDBNet architecture extracted
- SLURM distributed utilities removed
- All vulnerable code eliminated

Verify mitigation:
```bash
python verify_no_basicsr_imports.py
```

### Input Validation

- `src/transformation_portal/utils/input_validation.py` provides comprehensive validation
- File path sanitization
- Parameter bounds checking
- Shell command escaping (when FFmpeg subprocess is used)

### Error Handling

- `src/transformation_portal/utils/error_handling.py` provides structured error handling
- Graceful degradation for optional features
- Detailed error messages for debugging

---

## Performance Considerations

### Benchmarks (M4 Max)

- Depth estimation: 24-65ms per image
- Batch throughput: 400-600 images/hour
- Context extraction: 5-60 seconds (PDF-dependent)

### Optimization Tips

1. **LRU caching**: 10-20x speedup for repeated computations
2. **CoreML models**: 3-5x speedup on Apple Silicon
3. **Batch processing**: Use multiprocessing for independent operations
4. **Early validation**: Check models/files before long operations
5. **Memory profiling**: Use `memory-profiler` for optimization

### GPU/Acceleration Support

- **CUDA**: NVIDIA GPU support
- **MPS**: Apple Metal Performance Shaders
- **CoreML**: Apple Neural Engine (M-series chips)

---

## Troubleshooting

### Common Import Errors

```bash
# Missing optional dependencies
pip install -e ".[ml]"  # For torch, diffusers, etc.
pip install -e ".[all]" # For everything

# Ensure editable install
pip install -e .
```

### Test Failures

```bash
# Isolate failing test
pytest tests/test_<module>.py -v

# Check FFmpeg
ffmpeg -version

# Check Python version (requires 3.10+)
python --version

# Run minimal fast tests
make test-fast
```

### Linting Errors

```bash
# Auto-fix common issues
autopep8 --in-place --max-line-length=127 <file.py>

# Check specific file
flake8 <file.py> --count --select=E9,F63,F7,F82 --show-source
```

### Memory Issues

- Large images (4K+) require 8-16GB RAM
- Use `--batch-size 1` to reduce memory
- Consider downsampling before processing
- Close other applications when running ML models

---

## Important Conventions

### When Making Changes

1. **Read before editing**: Always read existing code before modifying
2. **Test with real files**: Use samples from `data/sample_images/`
3. **Preserve metadata**: IPTC, XMP, GPS data should survive processing
4. **Backward compatibility**: Existing scripts may be in client production use
5. **Document performance**: Include throughput/memory requirements
6. **Profile before optimizing**: Use `memory-profiler` or `cProfile`

### Commit Guidelines

- Clear, descriptive commit messages
- Reference issue numbers when applicable
- Keep commits focused on single changes
- Run `make ci` before pushing

### Pull Request Guidelines

- Run full test suite: `make test-full`
- Update documentation if needed
- Add tests for new functionality
- Ensure CI passes

---

## Quick Reference

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make test-fast` | Run fast subset of tests |
| `make test-full` | Run entire test suite |
| `make lint` | Run flake8 + pylint |
| `make ci` | Local CI checks (lint + test-fast) |
| `make ci-full` | Comprehensive CI simulation |
| `make setup` | Install package in editable mode |
| `make clean` | Remove cache files and build artifacts |
| `make quality-check` | All quality validations |

### Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration, dependencies |
| `Makefile` | Development task automation |
| `.pylintrc` | Pylint configuration |
| `mypy.ini` | Type checking configuration |
| `pytest.ini` | Pytest configuration |
| `requirements/*.txt` | Layered dependencies |

### Documentation

- Main README: `README.md`
- Architecture: `docs/ARCHITECTURE.md`
- Depth Pipeline: `docs/depth_pipeline/`
- Performance: `docs/PERFORMANCE_OPTIMIZATION.md`
- Refactoring: `docs/REFACTORING_SUMMARY.md`

---

*Last Updated: 2025-11-27*
