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
- Clear import paths: `from transformation_portal.processors.material_response.core import MaterialResponse`
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
transformation_portal/
├── src/transformation_portal/    # Main package
├── scripts/                      # Standalone utilities  
├── data/                         # Data files (gitignored)
├── docs/                         # Documentation
├── tests/                        # Test suite
├── depth_pipeline/               # Existing depth processing
├── luxury_tiff_batch_processor/  # Existing TIFF processor
├── tools/                        # Editorial tools
└── [LUT directories]/           # Film emulation & location LUTs
```

### Package Organization (src/transformation_portal/)

```
transformation_portal/
├── __init__.py                   # Package root with lazy imports
├── pipelines/                    # High-level workflows
│   ├── __init__.py
│   ├── lux_render_pipeline.py   # AI-powered render refinement
│   ├── depth_tools.py            # Depth processing utilities
│   └── dreaming_pipeline.py     # Visualization pipeline
├── processors/                   # Core processing engines
│   ├── __init__.py
│   ├── luxury_video_master_grader.py  # Video color grading
│   └── material_response/             # Material-aware processing
│       ├── __init__.py
│       ├── core.py                    # Main MaterialResponse class
│       └── optimizer.py               # Optimization algorithms
├── enhancers/                    # Specialized enhancement
│   ├── __init__.py
│   ├── enhance_aerial.py         # Aerial photography enhancement
│   ├── enhance_pool_aerial.py    # Pool-specific enhancement
│   ├── board_material_aerial_enhancer.py  # Material palette assignment
│   └── update_enhance_aerial.py  # Updated aerial enhancement
├── analyzers/                    # Analysis & monitoring
│   ├── __init__.py
│   ├── decision_decay_dashboard.py      # Code philosophy dashboard
│   ├── codebase_philosophy_auditor.py   # Philosophy auditing
│   └── parse_workflows.py               # Workflow parser
├── rendering/                    # Rendering workflows
│   ├── __init__.py
│   ├── coastal_estate_render.py         # Coastal estate preset
│   ├── golden_hour_courtyard_workflow.py  # Golden hour preset
│   └── process_renderings_750.py        # Batch rendering
├── utils/                        # Shared utilities
│   ├── __init__.py
│   ├── color_science.py          # Color space operations
│   └── helpers.py                # General helpers
└── cli/                          # CLI entry points
    ├── __init__.py
    ├── render.py                 # Rendering CLI
    ├── process.py                # Processing CLI
    └── analyze.py                # Analysis CLI
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
from transformation_portal.processors.material_response.core import MaterialResponse
from transformation_portal.utils.color_science import apply_lut

def process_render(image_path, config):
    """Complete render refinement pipeline."""
    # 1. Load and preprocess
    image = load_image(image_path)
    
    # 2. AI enhancement
    enhanced = ai_enhance(image, config)
    
    # 3. Material response
    if config.material_response:
        mr = MaterialResponse()
        enhanced = mr.enhance(enhanced)
    
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
# material_response/core.py
class MaterialResponse:
    """Physics-based surface enhancement."""
    
    def __init__(self, config=None):
        self.config = config or default_config()
    
    def enhance(self, image, surfaces=None):
        """Enhance materials in image."""
        # Material-aware processing
        return enhanced_image
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
    lut: "02_Location_Aesthetic/Montecito_Golden_Hour.cube"
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

### Short Term (v0.2.0)
- [ ] Complete import migration
- [ ] Remove root-level duplicates
- [ ] Unified CLI interface
- [ ] Comprehensive type hints

### Medium Term (v0.3.0)
- [ ] Plugin architecture
- [ ] Async processing support
- [ ] Distributed processing
- [ ] Web API

### Long Term (v1.0.0)
- [ ] Stable public API
- [ ] Comprehensive documentation
- [ ] Performance optimizations
- [ ] Enterprise features

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

- See [REFACTORING_2025.md](REFACTORING_2025.md) for migration guide
- See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for performance tips
- Open an issue on GitHub for specific questions
