# Copilot Instructions for Transformation Portal

## Project Overview

**Transformation Portal** is a professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production. The repository combines:

- **AI-Powered Enhancement**: Stable Diffusion XL, ControlNet, and Real-ESRGAN for intelligent upscaling
- **Depth-Aware Processing**: Depth Anything V2 with Apple Neural Engine optimization for architectural rendering
- **Material Response Technology**: Physics-based surface enhancement for wood, metal, glass, and textiles
- **Professional Color Grading**: 16+ LUTs with Film Emulation and Location Aesthetics
- **Batch Processing**: High-throughput TIFF and video processing pipelines
- **Production-Ready**: Comprehensive test suite with CI/CD, performance profiling

## Repository Structure

```
.
├── depth_pipeline/             # Depth Anything V2 integration with CoreML
│   ├── pipeline.py            # Main depth-aware processing pipeline
│   ├── processors/            # Depth-based image processors
│   └── models/                # ML model configurations
├── src/transformation_portal/ # Installable package (WIP)
├── tests/                      # pytest test suite (70+ tests)
├── config/                     # YAML configuration presets
├── docs/                       # Architecture and performance documentation
├── 01_Film_Emulation/          # Kodak and FilmConvert LUT emulations
├── 02_Location_Aesthetic/      # Location-specific color profiles
├── 03_Material_Response/       # Physics-based surface enhancement LUTs
├── 08_Documentation/           # Version history and technical guides
├── 09_Client_Deliverables/     # Brand assets and production deliverables
├── lux_render_pipeline.py      # AI-powered render refinement
├── luxury_tiff_batch_processor.py  # 16-bit TIFF batch processing CLI
├── luxury_video_master_grader.py   # Video grading with FFmpeg
├── material_response.py        # Material Response core implementation
├── depth_tools.py              # Depth estimation utilities
├── hdr_production_pipeline.sh  # HDR finishing workflow
├── codebase_philosophy_auditor.py  # Code quality auditing tool
└── decision_decay_dashboard.py     # Temporal contract monitoring
```

## Getting Started

### Prerequisites
- Python 3.10+ (CI tests on 3.10, 3.11, 3.12)
- FFmpeg 6+ (for video processing)
- Git
- Optional: CUDA-capable GPU or Apple Silicon with MPS for ML pipelines

### Setup
```bash
# Clone and navigate to repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (pytest, flake8, pylint)
pip install -r requirements-dev.txt

# Optional: Install extras for specific features
pip install -e ".[tiff]"   # 16-bit TIFF processing with tifffile
pip install -e ".[ml]"     # ML extras (transformers, torchvision, CoreML)
pip install -e ".[dev]"    # Development tools
pip install -e ".[all]"    # Everything

# Run tests to verify setup
make test-fast

# Run full test suite (if xdist installed, runs in parallel)
make test-full

# Run linting
make lint
```

## Tech Stack

- **Languages**: Python 3.10+, Shell (Bash), TypeScript (minimal)
- **Key Dependencies**: 
  - **ML/AI**: PyTorch 2.0+, Diffusers, ControlNet-aux, transformers, Real-ESRGAN
  - **Image Processing**: NumPy, Pillow, scipy, scikit-image, tifffile, imagecodecs
  - **Video Processing**: FFmpeg 6+
  - **Color Science**: colour-science for ACES/ODT transforms
  - **Depth Estimation**: Depth Anything V2 (via transformers)
  - **Apple Acceleration**: coremltools for Apple Neural Engine (M-series chips)
  - **CLI**: Typer for user-friendly command-line interfaces
- **Testing**: pytest, pytest-cov, pytest-xdist (parallel tests), hypothesis (property testing)
- **Linting**: flake8, pylint
- **Performance**: psutil, memory-profiler for optimization

## Coding Standards

### Python
- Follow PEP 8 style guidelines
- Maximum line length: 127 characters (as configured in CI)
- Use type hints where appropriate
- Use dataclasses for configuration objects
- Prefer pathlib.Path over string paths
- Use f-strings for string formatting

### Code Organization
- Keep functions focused and single-purpose
- Use descriptive variable names (e.g., `preset`, `filter_graph`, `tone_map_config`)
- Document complex algorithms with docstrings
- Separate concerns: CLI parsing, business logic, and I/O operations

### File Naming
- Python scripts: lowercase with underscores (e.g., `luxury_video_master_grader.py`)
- Shell scripts: lowercase with underscores (e.g., `hdr_production_pipeline.sh`)
- Test files: `test_` prefix (e.g., `test_luxury_video_master_grader.py`)

## Key Concepts

### Depth Pipeline
- Uses **Depth Anything V2** for monocular depth estimation (24-65ms per image on M4 Max)
- Apple Neural Engine optimization via CoreML on M-series chips
- Depth-aware processing: denoising, tone mapping, atmospheric effects, clarity enhancement
- Zone-based tone mapping (AgX, Reinhard, Filmic) respects depth information
- LRU caching provides 10-20x speedup in iterative workflows
- Configuration via YAML presets in `config/` directory

### Material Response Technology
- Proprietary surface-aware rendering that analyzes material types (wood, metal, glass, fabric, stone)
- Applies physics-based enhancements respecting highlights, midtones, and micro-contrast
- Works in conjunction with depth information and LUTs for realistic material rendering
- Surface-type detection and per-material enhancement strength adjustments

### LUT Processing
- LUTs are stored in `.cube` format in categorized directories:
  - `01_Film_Emulation/` - Kodak and FilmConvert emulations
  - `02_Location_Aesthetic/` - Location-specific color profiles  
  - `03_Material_Response/` - Physics-based surface enhancement
- LUT strength is typically applied at 60-80% opacity
- LUTs can be stacked for complex material interactions
- Applied via FFmpeg for video or custom Python implementation for images

### AI Enhancement Pipeline (Lux Render)
- Edge-preserving AI enhancement via ControlNet (Canny, Depth)
- SDXL refinement for photorealistic architectural details
- Real-ESRGAN 4x upscaling for resolution enhancement
- Material Response finishing layer for surface realism
- Brand overlay and text placement for marketing deliverables

### Video Processing
- Default output: ProRes 422 HQ masters
- HDR detection (PQ, HLG) and automatic tone mapping
- Frame rate conformance to cinema/broadcast standards (23.976, 24, 25, 29.97, 30 fps)
- Preset-driven workflows (e.g., `signature_estate`, `golden_hour_courtyard`)
- Color metadata preservation (`color_primaries`, `color_trc`, `colorspace`)

### Image Processing
- Preserve 16-bit TIFF precision when `tifffile` is available
- Maintain IPTC/XMP metadata and GPS coordinates across processing
- Support for batch processing with directory tree mirroring
- Preset-based adjustments for exposure, contrast, saturation, clarity, glow
- Progress tracking and batch statistics for large-scale operations

## Testing Guidelines

- Write unit tests for new functions in the `tests/` directory
- Follow naming convention: `test_<module_name>.py` for test files
- Use pytest fixtures for common setup (see `tests/__init__.py` for shared fixtures)
- Mock external dependencies (FFmpeg, file I/O, ML models) when appropriate to avoid CI timeouts
- Test edge cases: missing files, invalid parameters, HDR content, various image formats
- Use hypothesis for property-based testing of mathematical functions
- Run fast tests during development: `make test-fast`
- Run full suite before committing: `make test-full` or `pytest`
- CI runs tests on Python 3.10, 3.11, 3.12 with both CPU and GPU configurations
- Keep tests fast: avoid loading large ML models in unit tests unless necessary
- Document any tests that require optional dependencies (e.g., `tifffile`, `torch`)

## Common Tasks

### Adding a New Depth Pipeline Preset
1. Create YAML configuration file in `config/` directory (e.g., `config/new_preset.yaml`)
2. Define parameters: depth model, tone mapping, denoising, atmospheric effects
3. Use existing presets like `config/interior_preset.yaml` as templates
4. Document the preset's intended use case and parameter rationale
5. Add tests in `tests/test_pipeline.py` for the new preset

### Adding a New Image/Video Preset
1. Define preset in the appropriate script (e.g., `PRESETS` dict in `luxury_video_master_grader.py`)
2. Include descriptive name, LUT path, and parameter values
3. Add documentation in the preset's `notes` field
4. Update README with example usage
5. Test with representative sample images/videos

### Modifying FFmpeg Filters
1. Use the `build_filter_graph()` pattern to construct filter chains
2. Always validate filter syntax with `--dry-run` option
3. Preserve color metadata (`color_primaries`, `color_trc`, `colorspace`)
4. Test with both SDR and HDR sources
5. Consider HDR tone mapping operators (Hable, Reinhard, Mobius) when applicable

### Adding CLI Options
1. Use Typer for Python CLIs (consistent with existing scripts)
2. Provide helpful descriptions and sensible defaults
3. Group related options with comments (e.g., `# Render config`, `# Models`)
4. Support `--dry-run` for inspection without execution
5. Add `--verbose` output for debugging when appropriate

### Optimizing Performance
1. Profile with `memory-profiler` or `psutil` for memory-intensive operations
2. Consider LRU caching (`functools.lru_cache`) for repeated computations
3. Use lazy loading for ML models to reduce import times
4. Implement batch processing for I/O-bound operations
5. Document performance characteristics (throughput, memory usage) in docstrings

## Repository-Specific Notes

### Depth Processing
- Depth Anything V2 is the primary depth estimation model
- CoreML variants provide 3-5x speedup on Apple Silicon (M1/M2/M3/M4)
- Depth maps are normalized and cached for iterative workflows
- Zone-based processing applies different enhancements to foreground/midground/background
- See `docs/depth_pipeline/DEPTH_PIPELINE_README.md` for detailed documentation

### HDR Handling
- Automatically detect HDR transfer functions (PQ, HLG) from video metadata
- Apply tone mapping with configurable operators (Hable, Reinhard, Mobius)
- Preserve HDR10 and Dolby Vision metadata when available
- Use ACES ODT for broadcast compliance and color space transforms
- Test with `zscale` filter in FFmpeg for HDR-to-SDR conversion

### Performance Considerations
- Depth pipeline: 24-65ms per image on M4 Max, 400-600 images/hour batch throughput
- Video processing can be GPU-intensive (CUDA/MPS/Metal support)
- ML pipelines (Stable Diffusion, ControlNet) require 8GB+ VRAM for optimal performance
- Batch operations should report progress using `tqdm` or custom progress tracking
- Use multiprocessing for independent image operations (see `material_response.py`)
- Provide early validation before long-running operations (model loading, file checks)
- Recent refactoring reduced repo size by 92% (180MB → 15MB) and improved import speed by 60%

### Brand/Client Deliverables
- Support for logo overlays (PNG with alpha channel) in `lux_render_pipeline.py`
- Maintain GPS coordinates and IPTC/XMP metadata when available
- Follow naming conventions: `{basename}_{preset}.{ext}`
- Create timestamped output directories for productions (e.g., `output_2025-10-30_14-30/`)
- Brand assets located in `09_Client_Deliverables/Lantern_Logo_Implementation_Kit/`

## Documentation

- Keep README.md synchronized with tool capabilities and feature updates
- Update `08_Documentation/Version_History/changelog.md` for significant changes
- Architecture documentation in `docs/ARCHITECTURE.md`
- Refactoring notes in `docs/REFACTORING_SUMMARY.md` and `docs/REFACTORING_2025.md`
- Performance optimization guidance in `docs/PERFORMANCE_OPTIMIZATION.md`
- Depth pipeline specifics in `docs/depth_pipeline/DEPTH_PIPELINE_README.md`
- Include usage examples with common parameter combinations in README
- Document all presets with their intended use cases and parameter rationale
- Add inline comments for complex algorithms (depth processing, tone mapping, material detection)

## CI/CD

- GitHub Actions workflows in `.github/workflows/`:
  - `build.yml`: Main CI with linting and tests (matrix: Python 3.10/3.11/3.12, CPU/GPU)
  - `codeql.yml`: Security scanning with CodeQL
  - `summary.yml`: PR summary and metadata checks
- Linting: flake8 (critical errors), pylint (non-blocking warnings)
- Tests run on push and pull request to `main` branch
- Tests must pass before merging
- Python 3.10 is the minimum supported version (CI tests 3.10, 3.11, 3.12)
- CI uses `requirements-ci.txt` for minimal dependency installation
- Free disk space cleanup runs before tests due to large ML models
- Exclude deprecated code from linting: `deprecated/`, `src/transformation_portal/`, `scripts/`

## When Making Changes

1. **Understand the pipeline**: Video/image processing pipelines are complex and order-dependent
   - Depth estimation → Material detection → Color grading → Tone mapping → Sharpening
   - Order matters for quality and performance
2. **Test with real files**: Use sample images from `data/sample_images/` when available
3. **Validate FFmpeg commands**: Use `--dry-run` to inspect generated commands before execution
4. **Consider backward compatibility**: Existing scripts may be in production use by clients
5. **Update documentation**: Keep README, docs/depth_pipeline/DEPTH_PIPELINE_README.md, and examples current
6. **Preserve metadata**: IPTC, XMP, and GPS data should survive processing
   - Use `Pillow.Image.info` for metadata preservation
   - Consider `tifffile` for 16-bit TIFF with full metadata support
7. **Profile before optimizing**: Use `memory-profiler` or `cProfile` to identify bottlenecks
8. **Lazy load ML models**: Import heavy dependencies only when needed to speed up CLI startup
9. **Document performance**: Include throughput numbers (images/hour) and memory requirements
10. **Consider Apple Silicon**: Test CoreML optimizations on M-series chips when available

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For ML features: `pip install -e ".[ml]"` (includes torch, diffusers, transformers)
- For TIFF support: `pip install -e ".[tiff]"` (includes tifffile, imagecodecs)
- Missing hypothesis: `pip install hypothesis` (required for some tests)
- Check for conflicting package versions: `pip list | grep <package>`

**Test Failures**
- Run individual test files to isolate issues: `pytest tests/test_<module>.py -v`
- Check that FFmpeg is available: `ffmpeg -version`
- Verify Python version: `python --version` (requires 3.10+)
- Some tests require optional dependencies (tifffile, torch): skip with `-k 'not <test_name>'`
- Use `make test-fast` to run quick tests without heavy ML models
- CI may timeout on ML tests: mock model loading in unit tests

**Linting Errors**
- Fix automatically when possible: `autopep8 --in-place --max-line-length=127 <file.py>`
- Check line length: Maximum 127 characters (CI enforced)
- Verify imports are used: flake8 will flag unused imports (F401)
- Pylint is non-blocking in CI: address critical issues first
- Excluded from linting: `deprecated/`, `src/transformation_portal/`, `scripts/`

**FFmpeg Processing Issues**
- Use `--dry-run` to inspect commands before execution
- Check source file compatibility: `ffprobe <input_file>`
- Verify LUT file paths are correct and `.cube` files exist
- For HDR issues, ensure zimg support: `ffmpeg -filters | grep zscale`
- Check color space metadata: `ffprobe -show_streams <input_file> | grep color`

**Depth Pipeline Issues**
- Ensure Depth Anything V2 model is downloaded (automatic on first run)
- Check GPU/MPS availability: `torch.cuda.is_available()` or `torch.backends.mps.is_available()`
- CoreML models require macOS 13+ and M-series chip
- Out of memory: reduce batch size or image resolution
- Slow performance: check if using CPU fallback instead of GPU/CoreML

**Memory Issues**
- Large images (4K+) require 8-16GB RAM for ML pipelines
- Use `--batch-size 1` to reduce memory usage
- Close other applications when running ML models
- Consider downsampling images before processing if memory-constrained

## Additional Resources

- **Main README**: `README.md` - Feature overview, installation, usage examples
- **Depth Pipeline**: `docs/depth_pipeline/DEPTH_PIPELINE_README.md` - Comprehensive depth processing guide
- **Architecture**: `docs/ARCHITECTURE.md` - System design and component relationships
- **Performance**: `docs/PERFORMANCE_OPTIMIZATION.md` - Optimization strategies and benchmarks
- **Refactoring**: `docs/REFACTORING_SUMMARY.md` - Recent changes and improvements (Oct 2025)
- **LUT Documentation**: `03_Material_Response/_Material_Response_Technical_Guide.md` (if exists)
- **Version History**: `08_Documentation/Version_History/changelog.md` - Change tracking
- **Test Status**: `tests/TEST_STATUS.md` - Current test coverage and known issues
- **Brand Assets**: `09_Client_Deliverables/Lantern_Logo_Implementation_Kit/` - Logos and color tokens
- **Workflow Fixes**: `WORKFLOW_BUGS_FIXED.md` - Documented bug fixes and solutions

## Code Examples

### Using the Depth Pipeline
```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load preset configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Process single image
result = pipeline.process_render('interior.jpg')
pipeline.save_result(result, 'output/')

# Batch process with progress tracking
from pathlib import Path
image_paths = list(Path('input/').glob('*.jpg'))
results = pipeline.batch_process(image_paths, output_dir='output/')
```

### Using Material Response
```python
from material_response import MaterialResponse, SurfaceType

# Initialize Material Response
mr = MaterialResponse()

# Analyze and enhance image
result = mr.enhance(
    image,
    surfaces=[SurfaceType.WOOD, SurfaceType.METAL, SurfaceType.GLASS],
    strength=0.7,
    preserve_highlights=True
)
```

### Adding a New LUT Preset
```python
# In luxury_video_master_grader.py or luxury_tiff_batch_processor.py
PRESETS = {
    "my_new_preset": PresetConfig(
        name="My New Preset",
        lut="01_Film_Emulation/Kodak_2393.cube",
        notes="Custom look for specific project",
        # Adjustments
        exposure=0.0,
        contrast=1.08,
        saturation=1.05,
        # Optional enhancements
        clarity=0.15,
        grain=0.012,
    ),
}
```

### Running Tests for Specific Module
```bash
# Fast tests (recommended during development)
make test-fast

# Test a single module
pytest tests/test_luxury_video_master_grader.py -v

# Test with coverage
pytest --cov=luxury_video_master_grader tests/test_luxury_video_master_grader.py

# Run specific test
pytest tests/test_luxury_video_master_grader.py::test_assess_frame_rate_detects_vfr -v

# Skip tests requiring optional dependencies
pytest -k 'not video_master_grader' tests/

# Full test suite with parallel execution
make test-full
```

### Using Decision Annotations
```python
# Document intentional deviations from coding standards
# Decision: allow_wildcard_import - tight integration with plugin API
from plugin_api import *  # noqa: F403

# Decision: undocumented_public_api - docstring inherited from base class
class CustomProcessor:
    def process(self):
        pass

# Decision: complex_function - inherent complexity of tone mapping algorithm
def apply_tone_mapping(image, zones, operators):  # noqa: C901
    # Complex but necessary logic
    pass
```

### Performance Profiling
```python
from memory_profiler import profile

@profile
def batch_process_images(image_paths):
    """Profile memory usage during batch processing."""
    results = []
    for path in image_paths:
        result = process_image(path)
        results.append(result)
    return results

# Run with: python -m memory_profiler script.py
```
