# Repository Architecture

## Overview

This document describes the architecture of the Transformation Portal after the October 2025 refactoring. It explains the design decisions, module organization, and best practices for development.

## Design Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- **Pipelines**: Orchestrate multi-step workflows
- **Processors**: Core data transformation engines
- **Enhancers**: Specialized improvement algorithms
- **Analyzers**: Code and workflow analysis tools
- **Utils**: Shared functionality with no business logic

### 2. Explicit Over Implicit
- Clear import paths: `from transformation_portal.processors.material_response.engine import MaterialResponseEngine`
- Explicit dependencies listed in each module
- No magic imports or hidden behavior

### 3. Backward Compatibility
- Root-level files maintained for existing code (v0.1.x)
- Deprecation warnings for future removal
- Migration tools provided

### 4. Performance First
- Lazy imports to reduce startup time
- Efficient file I/O with pathlib
- Caching where beneficial
- Batch processing support

### 5. Developer Friendly
- Consistent structure across modules
- Comprehensive documentation
- Clear error messages
- Type hints where beneficial

## Module Structure

### Top-Level Organization

```
transformation_portal/              # Repository root
├── src/                          # Installable package source
│   ├── transformation_portal/    # Main package
│   ├── tp/                       # Short alias package
│   └── luxury_tiff_batch_processor/  # TIFF batch processor
├── scripts/                      # Operational scripts and pipeline runners
├── config/                       # YAML presets and configuration
├── assets/                       # LUTs, branding, look assets
├── docs/                         # Documentation
├── tests/                        # Pytest suite
├── tools/                        # Dev/ops tools (manifests, audits)
├── workflows/                    # Workflow artifacts / ComfyUI workflows
├── requirements/                 # Layered dependency sources (pip-tools)
└── schemas/                      # JSON schemas for contracts
```

### Package Organization (src/transformation_portal/)

```
transformation_portal/
├── __init__.py                   # Package root with lazy imports
├── lux_depth_v3/                 # 🔑 Main depth processing pipeline
│   ├── __main__.py               # CLI entry point
│   ├── orchestrator.py           # Pipeline orchestration
│   ├── config.py                 # Configuration management
│   └── ...                       # Preprocessing, postprocessing, etc.
├── depth/                        # Depth estimation backends
│   ├── backends/                 # DA3, Depth Pro backends
│   └── registry.py               # Backend selection
├── pipelines/                    # High-level workflows
│   ├── lux_render_pipeline.py    # AI-powered render refinement
│   └── depth_tools.py            # Depth processing utilities
├── processors/                   # Core processing engines
│   ├── luxury_video_master_grader.py  # Video color grading
│   └── material_response/        # Material-aware processing
├── enhancers/                    # Specialized enhancement
├── analyzers/                    # Analysis & monitoring
├── rendering/                    # Rendering workflows
├── ingest/                       # RAW/TIFF ingest with provenance
├── attestation/                  # Archive attestation CLI
├── comfyui/                      # ComfyUI workflow integration
├── spatial_ai/                   # Spatial AI features
├── determinism/                  # Cross-ISA determinism tools
├── utils/                        # Shared utilities
└── cli/                          # CLI entry points
```

## Module Responsibilities

### Pipelines
**Purpose**: Orchestrate complex, multi-step workflows

**Characteristics**:
- High-level APIs
- Coordinate multiple processors
- Handle configuration and state
- Provide progress tracking
- Error handling and recovery

**Example**:
```python
# lux_render_pipeline.py
from transformation_portal.processors.material_response.engine import MaterialResponseEngine
from transformation_portal.utils.color_science import apply_lut

def process_render(image_path, config):
    """Complete render refinement pipeline."""
    # 1. Load and preprocess
    image = load_image(image_path)

    # 2. AI enhancement
    enhanced = ai_enhance(image, config)

    # 3. Material response
    if config.material_response:
        material_engine = MaterialResponseEngine.from_config(
            {"profile": "luxury_interior"}
        )
        enhanced = material_engine.apply(enhanced)

    # 4. Color grading
    result = apply_lut(enhanced, config.lut)

    return result
```

### Processors
**Purpose**: Core data transformation engines

**Characteristics**:
- Stateful or stateless operations
- Well-defined input/output contracts
- Reusable across pipelines
- Optimized for performance
- Extensive error checking

**Example**:
```python
# material_response/engine.py
from transformation_portal.processors.material_response.engine import MaterialResponseEngine

engine = MaterialResponseEngine.from_config(
    {"profile": "luxury_interior", "texture_boost": 0.25}
)
enhanced_image = engine.apply(image)
```

### Enhancers
**Purpose**: Specialized improvement algorithms

**Characteristics**:
- Domain-specific enhancements
- Often stateless functions
- Composable with other enhancers
- Well-documented parameters

**Example**:
```python
# enhance_aerial.py
def enhance_aerial(image, settings):
    """Enhance aerial photography with atmospheric effects."""
    # Apply enhancements
    return enhanced_image
```

### Analyzers
**Purpose**: Code quality and workflow analysis

**Characteristics**:
- Introspection and reporting
- Non-invasive analysis
- Dashboard and CLI interfaces
- Monitoring and alerting

### Utils
**Purpose**: Shared utility functions

**Characteristics**:
- No business logic
- Pure functions preferred
- Well-tested
- Broadly applicable

**Anti-patterns to avoid**:
- Don't put business logic in utils
- Don't create circular dependencies
- Don't add too many utilities (creates bloat)

## Dependency Management

### Internal Dependencies

**Allowed**:
- Utils can import from standard library only
- Enhancers can import from utils
- Processors can import from utils and enhancers
- Pipelines can import from all lower layers
- Analyzers can import from all layers (read-only)

**Forbidden**:
- No circular dependencies
- Utils cannot import from other transformation_portal modules
- Processors cannot import from pipelines

### External Dependencies

**Core Dependencies** (required for all):
- numpy, Pillow, scipy, typer

**Optional Dependencies**:
- `[tiff]`: tifffile, imagecodecs
- `[ml]`: torch, diffusers, transformers, controlnet-aux
- `[dev]`: pytest, flake8, pylint

**Adding New Dependencies**:
1. Check if existing dependency can be used
2. Evaluate size and maintenance status
3. Add to appropriate optional-dependencies group
4. Update documentation

## Data Flow

### Typical Processing Flow

```
Input
  ↓
[Pipeline] ← coordinates
  ↓
[Processor] ← transforms
  ↓
[Enhancer] ← improves
  ↓
[Processor] ← finalizes
  ↓
Output
```

### Example: Render Processing

```
Raw Render (render.jpg)
  ↓
lux_render_pipeline.py
  ├─→ Load image
  ├─→ AI enhancement (Stable Diffusion + ControlNet)
  ├─→ Material Response enhancement
  │     └─→ material_response/core.py
  │           ├─→ Detect materials
  │           ├─→ Apply physics-based enhancements
  │           └─→ Return enhanced
  ├─→ Color grading (LUT application)
  ├─→ Sharpening and export
  └─→ Save result (render_enhanced.jpg)
```

## Testing Strategy

### Test Organization

```
tests/
├── test_pipelines/
├── test_processors/
├── test_enhancers/
├── test_analyzers/
├── test_rendering/
└── test_utils/
```

### Testing Principles

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test module interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Verify speed requirements
5. **Regression Tests**: Prevent breaking changes

### Test Coverage Goals

- Utils: 95%+ coverage
- Processors: 85%+ coverage
- Pipelines: 75%+ coverage
- Enhancers: 80%+ coverage

## Configuration Management

### YAML Configuration Pattern

```yaml
# config/preset_name.yaml
pipeline:
  name: "signature_estate"

processor:
  material_response:
    enabled: true
    strength: 0.7

  color_grading:
    lut: "assets/luts/location_aesthetic/Montecito_Golden_Hour.cube"
    opacity: 0.75

output:
  format: "tiff"
  bit_depth: 16
  quality: 100
```

### Loading Configuration

```python
from pathlib import Path
import yaml

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)

# Usage
config = load_config('config/signature_estate.yaml')
```

## Error Handling

### Standard Error Pattern

```python
class TransformationPortalError(Exception):
    """Base exception for all transformation portal errors."""
    pass

class ProcessingError(TransformationPortalError):
    """Error during image processing."""
    pass

class ConfigurationError(TransformationPortalError):
    """Invalid configuration."""
    pass

# Usage
def process_image(image_path, config):
    if not image_path.exists():
        raise ProcessingError(f"Image not found: {image_path}")

    if not validate_config(config):
        raise ConfigurationError(f"Invalid config: {config}")

    # Process...
```

### Error Recovery

- Provide helpful error messages
- Include context in exceptions
- Log errors with traceback
- Offer suggestions for fixes

## Performance Considerations

### Lazy Imports

```python
# Good: Lazy import for optional functionality
def render_with_ai(image):
    from transformation_portal.pipelines import lux_render_pipeline
    return lux_render_pipeline.process(image)

# Avoid: Top-level import of heavy dependencies
# from transformation_portal.pipelines import lux_render_pipeline
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def load_model(model_name):
    """Load model with caching."""
    return expensive_model_load(model_name)
```

### Batch Processing

```python
def process_batch(images, batch_size=10):
    """Process images in batches for better memory usage."""
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        yield process_images(batch)
```

## Versioning and Compatibility

### Semantic Versioning

- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Deprecation Policy

1. Mark feature as deprecated
2. Add deprecation warning
3. Document migration path
4. Remove after 2 minor versions

### Example Deprecation

```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

## Future Architecture Goals

### Completed (v2.0.0 - Current)
- [x] Stable public API contracts (schema-aligned payloads)
- [x] Preset stability taxonomy (stable / canary / experimental)
- [x] Service hardening with `/ready` readiness checks
- [x] Context-aware rendering workflows
- [x] Unified CLI interface (`lux-depth-v3`)
- [x] Backend registry with automatic fallback
- [x] Performance monitoring (APEX System)

### Near Term (v2.1.x)
- [ ] Enhanced plugin architecture
- [ ] Additional depth backend integrations
- [ ] Async batch processing support
- [ ] Extended RAW format support

### Future (v3.0.0)
- [ ] Distributed processing
- [ ] Web API improvements
- [ ] Enterprise features
- [ ] CoreML optimization for Apple Silicon

## Contributing

### Adding a New Module

1. Choose appropriate package (pipelines, processors, etc.)
2. Create module with clear docstring
3. Add unit tests (aim for 80%+ coverage)
4. Update package `__init__.py` if needed
5. Add documentation
6. Update CHANGELOG

### Code Review Checklist

- [ ] Follows existing code style
- [ ] Has docstrings and type hints
- [ ] Includes tests
- [ ] No circular dependencies
- [ ] Performance considerations addressed
- [ ] Error handling in place
- [ ] Documentation updated

## Questions?

- See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines
- See [TROUBLESHOOTING.md](../guides/TROUBLESHOOTING.md) for common issues
- Open an issue on GitHub for specific questions
