# Transformation Portal - Repository Structure

## Overview

This repository maintains a **dual structure** to support both:
1. **Legacy root-level scripts** - Production-ready standalone tools
2. **Modern package structure** - Installable package in `src/transformation_portal/`

This design allows for gradual migration while maintaining backward compatibility with existing workflows.

## Directory Structure

```
Transformation_Portal/
├── src/transformation_portal/       # Modern package structure (WIP)
│   ├── processors/                  # Video/image processors
│   ├── pipelines/                   # Processing pipelines
│   ├── analyzers/                   # Analysis tools
│   ├── enhancers/                   # Enhancement utilities
│   ├── rendering/                   # Rendering tools
│   └── utils/                       # Shared utilities
│
├── Root-level scripts (production):
│   ├── luxury_video_master_grader.py    # Video color grading CLI
│   ├── luxury_tiff_batch_processor.py   # 16-bit TIFF processing
│   ├── lux_render_pipeline.py           # AI-powered render refinement
│   ├── material_response.py             # Material Response core
│   ├── depth_tools.py                   # Depth estimation utilities
│   └── ... (other production scripts)
│
├── depth_pipeline/                  # Depth Anything V2 integration
├── tests/                           # Test suite (tests root-level scripts)
├── config/                          # YAML configuration presets
├── docs/                            # Documentation
└── pyproject.toml                   # Package configuration
```

## Dependency Management

### Core Dependencies (Always Installed)
Located in `pyproject.toml` under `[project.dependencies]`:
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `scipy` - Scientific computing
- `typer` - CLI framework
- `tqdm` - Progress bars

### Optional Dependencies
Install with `pip install -e ".[<extra>]"`:

#### `[tiff]` - 16-bit TIFF Processing
```bash
pip install -e ".[tiff]"
```
- `tifffile` - Advanced TIFF handling
- `imagecodecs` - Codec support

#### `[ai]` - AI-Powered Enhancement
```bash
pip install -e ".[ai]"
```
- `torch` - PyTorch framework
- `diffusers` - Stable Diffusion pipelines
- `controlnet-aux` - ControlNet models
- `realesrgan` - AI upscaling

#### `[ml]` - Depth Pipeline & Advanced Processing
```bash
pip install -e ".[ml]"
```
- `transformers` - Depth Anything V2
- `torchvision` - Vision models
- `opencv-python` - Computer vision
- `scikit-learn` - Machine learning
- `scikit-image` - Image processing algorithms
- `PyYAML` - Configuration files
- `colour-science` - Color space transforms
- `coremltools` - Apple Neural Engine
- `psutil` - Performance monitoring
- `memory-profiler` - Memory profiling

#### `[dev]` - Development Tools
```bash
pip install -e ".[dev]"
```
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `hypothesis` - Property-based testing
- `flake8` - Linting
- `pylint` - Static analysis

#### `[all]` - Everything
```bash
pip install -e ".[all]"
```
Installs all optional dependencies: tiff, ai, ml, and dev.

## Usage Patterns

### For Testing (CI/CD)
Tests load **root-level scripts directly** using `importlib`, so package installation is not required:
```bash
pip install -r requirements-ci.txt
pytest
```

### For Development (Root-Level Scripts)
Work with production scripts without installing the package:
```bash
pip install -r requirements-ci.txt
python luxury_video_master_grader.py --help
```

### For Development (Package)
Install package with necessary extras:
```bash
# Basic installation (core dependencies only)
pip install -e .

# With AI features
pip install -e ".[ai]"

# With all ML features
pip install -e ".[ml]"

# Full development setup
pip install -e ".[all]"
```

### For Production
Install from git with required extras:
```bash
pip install "git+https://github.com/RC219805/Transformation_Portal.git#egg=transformation-portal[ai,ml]"
```

## Why This Structure?

### Legacy Root-Level Scripts
- **Proven in production** - Used by clients for real projects
- **Standalone execution** - No package installation needed
- **Fast iteration** - Direct editing and testing
- **CLI-first design** - Optimized for command-line use

### Modern Package Structure
- **Modular design** - Better code organization
- **Import reusability** - Share code between scripts
- **Distribution ready** - Can be published to PyPI
- **Type hints** - Better IDE support and type checking
- **Future migration path** - Gradually move to package structure

## Testing Strategy

Tests are designed to work with **root-level scripts** without requiring package installation:

```python
# tests/test_luxury_video_master_grader.py
def load_module() -> ModuleType:
    """Load root-level module directly"""
    module_path = Path(__file__).resolve().parent.parent / "luxury_video_master_grader.py"
    spec = importlib.util.spec_from_file_location("luxury_video_master_grader", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
```

This approach:
- ✅ Avoids heavy ML dependencies in CI
- ✅ Tests actual production scripts
- ✅ Fast test execution (no model loading)
- ✅ Easy to debug and maintain

## Migration Path

The repository is gradually migrating from root-level scripts to the package structure:

1. **Phase 1** (Current): Dual structure with root-level scripts primary
2. **Phase 2**: Package structure complete, root-level scripts deprecated
3. **Phase 3**: Full migration to package, CLI entry points via `[project.scripts]`

During Phase 1:
- New features can go in either location
- Root-level scripts are production-ready
- Package structure is work-in-progress
- Tests focus on root-level scripts

## CI/CD Configuration

### GitHub Actions Workflow
The CI workflow (`build.yml`) is optimized for testing root-level scripts:

```yaml
- name: Install dependencies
  run: |
    pip install -r requirements-ci.txt
    # Skip package installation - tests load root-level scripts directly
    # To install package: pip install -e ".[ai,ml]"
    pip install pytest
```

This approach:
- Installs only lightweight CI dependencies
- Skips heavy ML packages (torch, diffusers, etc.)
- Allows tests to run quickly
- Reduces CI failure rate from dependency conflicts

### Requirements Files

- **requirements.txt** - Full dependencies for development (includes ML)
- **requirements-ci.txt** - Lightweight dependencies for CI testing
- **requirements-dev.txt** - Development tools (pytest, linting)

## Best Practices

### When to Use Root-Level Scripts
- Client deliverables and production workflows
- Quick prototyping and iteration
- Scripts that need direct execution
- Tools with minimal dependencies

### When to Use Package Structure
- Shared utilities used by multiple scripts
- Complex modules with many dependencies
- Code intended for import by other projects
- Features targeting future package distribution

### Dependency Guidelines
- Keep **core dependencies minimal** (numpy, Pillow, scipy, typer, tqdm)
- Put **ML/AI dependencies in optionals** (torch, diffusers, transformers)
- Use **extras for feature groups** ([ai], [ml], [tiff], [dev])
- Document **which extras are needed** for each feature

## Troubleshooting

### "pip install -e ." fails with heavy dependencies
**Solution**: Install with specific extras only:
```bash
pip install -e ".[tiff,dev]"  # Skip AI/ML dependencies
```

### Tests can't import modules
**Solution**: Tests load root-level scripts directly, no installation needed:
```bash
pip install -r requirements-ci.txt
pytest
```

### CI fails with torch/diffusers errors
**Solution**: CI should NOT install package. Update `.github/workflows/build.yml`:
```yaml
# Remove or comment out:
# pip install -e .
```

### Want to use package structure
**Solution**: Install with required extras:
```bash
pip install -e ".[ai,ml]"  # Full ML features
```

## Future Improvements

- [ ] Complete package structure migration
- [ ] Deprecate root-level scripts with migration guide
- [ ] Publish to PyPI with proper versioning
- [ ] Add pre-commit hooks for code quality
- [ ] Unified CLI entry point via package
- [ ] Comprehensive package documentation
- [ ] API reference documentation

## Questions?

See also:
- `README.md` - Main project documentation
- `docs/ARCHITECTURE.md` - System design
- `pyproject.toml` - Package configuration
- `.github/workflows/build.yml` - CI configuration
