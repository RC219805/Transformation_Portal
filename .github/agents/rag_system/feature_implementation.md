# Feature Implementation Template

**Use this template for**: Adding new features to pipelines (depth effects, material enhancements, LUT presets, processors)

---

## Feature Request

**Feature Name**: `{FEATURE_NAME}`

**Description**: 
```
{DETAILED_FEATURE_DESCRIPTION}
```

**Target Pipeline/Component**: 
- [ ] Depth Pipeline (`depth_pipeline/`)
- [ ] Lux Render Pipeline (`lux_render_pipeline.py`)
- [ ] Video Master Grader (`luxury_video_master_grader.py`)
- [ ] TIFF Batch Processor (`luxury_tiff_batch_processor.py`)
- [ ] Material Response (`material_response.py`)
- [ ] Other: `{SPECIFY}`

**Context/Use Case**:
```
{BUSINESS_CONTEXT_OR_USE_CASE}
```

---

## Requirements Analysis

### 1. Core Functionality
**What the feature must do**:
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] Requirement 3

**Expected Inputs**:
- Input type: `{IMAGE/VIDEO/DEPTH_MAP/CONFIG}`
- Format: `{TIFF/JPG/MP4/YAML}`
- Size constraints: `{MIN_SIZE - MAX_SIZE}`

**Expected Outputs**:
- Output type: `{IMAGE/VIDEO/CONFIG}`
- Format: `{TIFF/JPG/MP4/JSON}`
- Metadata preservation: `{YES/NO/PARTIAL}`

### 2. Edge Cases to Consider
- [ ] Missing input files
- [ ] Invalid parameters (negative values, out-of-range)
- [ ] HDR vs SDR content (for video features)
- [ ] Different image formats (8-bit vs 16-bit)
- [ ] GPU/MPS unavailable (CPU fallback)
- [ ] Large images (4K+) - memory constraints
- [ ] Batch processing with mixed file types

### 3. Performance Requirements
**Target Performance**:
- Processing time: `{X ms/image or Y fps for video}`
- Memory usage: `{Max RAM consumption}`
- GPU/MPS acceleration: `{REQUIRED/OPTIONAL/NOT_NEEDED}`
- Batch throughput: `{X images/hour or Y minutes/video}`

**Optimization Strategies**:
- [ ] LRU caching for repeated operations
- [ ] Lazy loading of ML models
- [ ] Batch processing optimization
- [ ] CoreML optimization for Apple Silicon
- [ ] Multiprocessing for independent operations

### 4. Dependencies
**New Python Packages**:
```python
# Add to requirements.txt or setup.py
{PACKAGE_NAME}=={VERSION}  # {REASON}
```

**New ML Models**:
- Model name: `{MODEL_NAME}`
- Source: `{HUGGINGFACE_HUB/GITHUB/OTHER}`
- Size: `{MODEL_SIZE_GB}`
- Acceleration: `{CUDA/MPS/COREML/CPU}`

**New LUTs/Assets**:
- LUT file: `assets/luts/{CATEGORY}/{LUT_NAME}.cube`
- Brand assets: `assets/brand/{ASSET_NAME}/`

---

## Implementation Plan

### Step 1: Configuration Setup
**Files to create/modify**:
- [ ] `config/{PRESET_NAME}.yaml` - Add configuration preset

**Example configuration**:
```yaml
# config/{PRESET_NAME}.yaml
feature_name: {FEATURE_NAME}
enabled: true
parameters:
  intensity: 0.5
  mode: "auto"
  fallback: "cpu"

# Performance tuning
batch_size: 4
cache_size: 128
```

### Step 2: Core Implementation
**Files to create/modify**:

**Option A: New Processor** (for depth_pipeline)
- [ ] `depth_pipeline/processors/{FEATURE_NAME}.py` - New processor class

```python
# depth_pipeline/processors/{FEATURE_NAME}.py
from typing import Optional
import numpy as np
from PIL import Image

class {FeatureName}Processor:
    """
    {FEATURE_DESCRIPTION}
    
    Performance: ~{X}ms per image on M4 Max
    """
    
    def __init__(self, intensity: float = 0.5):
        """
        Args:
            intensity: Effect strength (0.0-1.0)
        """
        self.intensity = intensity
    
    def process(
        self, 
        image: Image.Image, 
        depth_map: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Apply {FEATURE_NAME} effect to image.
        
        Args:
            image: Input PIL Image
            depth_map: Optional depth information (0.0=near, 1.0=far)
        
        Returns:
            Processed PIL Image
        """
        # Implementation here
        pass
```

**Option B: Extend Existing Script** (for presets)
- [ ] `{SCRIPT_NAME}.py` - Add preset to PRESETS dictionary

```python
# Example: luxury_video_master_grader.py
PRESETS = {
    "{preset_name}": PresetConfig(
        name="{Display Name}",
        lut="assets/luts/{category}/{lut_file}.cube",
        notes="{Description of aesthetic and use case}",
        # Adjustments
        exposure=0.0,
        contrast=1.08,
        saturation=1.05,
        # Optional enhancements
        clarity=0.15,
        glow=0.02,
        grain=0.012,
    ),
}
```

### Step 3: Pipeline Integration
**Files to modify**:
- [ ] `depth_pipeline/pipeline.py` - Integrate new processor (if applicable)
- [ ] `{MAIN_SCRIPT}.py` - Add CLI options and workflow integration

**Integration pattern**:
```python
# depth_pipeline/pipeline.py
from .processors.{feature_name} import {FeatureName}Processor

class ArchitecturalDepthPipeline:
    def __init__(self, config: Dict):
        # ... existing init ...
        if config.get('enable_{feature_name}', False):
            self.{feature_name}_processor = {FeatureName}Processor(
                intensity=config.get('{feature_name}_intensity', 0.5)
            )
    
    def process_render(self, image_path: Path) -> ProcessedResult:
        # ... existing processing ...
        
        # Apply new feature
        if hasattr(self, '{feature_name}_processor'):
            result.image = self.{feature_name}_processor.process(
                result.image, 
                depth_map=result.depth_map
            )
        
        return result
```

### Step 4: CLI Integration
**Files to modify**:
- [ ] `{SCRIPT_NAME}.py` - Add CLI arguments

```python
# CLI integration using Typer
@app.command()
def process(
    input_path: Path,
    output_dir: Path,
    # ... existing args ...
    enable_{feature_name}: bool = typer.Option(
        False,
        "--{feature-name}/--no-{feature-name}",
        help="{Feature description}"
    ),
    {feature_name}_intensity: float = typer.Option(
        0.5,
        "--{feature-name}-intensity",
        help="Effect strength (0.0-1.0)"
    ),
):
    """Process with {FEATURE_NAME}."""
    config = {
        'enable_{feature_name}': enable_{feature_name},
        '{feature_name}_intensity': {feature_name}_intensity,
    }
    # ... rest of implementation
```

---

## Testing Strategy

### Test Files to Create/Modify
- [ ] `tests/test_{feature_name}.py` - Unit tests for new feature
- [ ] `tests/integration/test_{pipeline}_pipeline.py` - Integration tests
- [ ] `tests/test_{script_name}.py` - CLI integration tests

### Unit Tests
```python
# tests/test_{feature_name}.py
import pytest
from PIL import Image
import numpy as np
from {module} import {FeatureName}Processor

class Test{FeatureName}Processor:
    """Test suite for {FeatureName}Processor."""
    
    def test_basic_processing(self):
        """Test basic processing workflow."""
        processor = {FeatureName}Processor(intensity=0.5)
        image = Image.new('RGB', (100, 100), color='red')
        
        result = processor.process(image)
        
        assert isinstance(result, Image.Image)
        assert result.size == image.size
    
    def test_with_depth_map(self):
        """Test processing with depth information."""
        processor = {FeatureName}Processor(intensity=0.7)
        image = Image.new('RGB', (100, 100), color='blue')
        depth_map = np.random.rand(100, 100).astype(np.float32)
        
        result = processor.process(image, depth_map=depth_map)
        
        assert isinstance(result, Image.Image)
    
    @pytest.mark.parametrize("intensity", [0.0, 0.5, 1.0])
    def test_intensity_range(self, intensity):
        """Test different intensity values."""
        processor = {FeatureName}Processor(intensity=intensity)
        image = Image.new('RGB', (50, 50), color='green')
        
        result = processor.process(image)
        
        assert result is not None
    
    def test_invalid_intensity_raises_error(self):
        """Test that invalid intensity raises ValueError."""
        with pytest.raises(ValueError):
            {FeatureName}Processor(intensity=1.5)
    
    def test_preserves_metadata(self):
        """Test that image metadata is preserved."""
        processor = {FeatureName}Processor(intensity=0.5)
        image = Image.new('RGB', (100, 100))
        image.info['dpi'] = (300, 300)
        
        result = processor.process(image)
        
        assert result.info.get('dpi') == (300, 300)
```

### Integration Tests
```python
# tests/integration/test_depth_pipeline.py
def test_{feature_name}_integration(tmp_path):
    """Test {FEATURE_NAME} integration in full pipeline."""
    from depth_pipeline import ArchitecturalDepthPipeline
    
    config = {
        'enable_{feature_name}': True,
        '{feature_name}_intensity': 0.7,
    }
    
    pipeline = ArchitecturalDepthPipeline(config)
    
    # Create test image
    test_image = tmp_path / "test.jpg"
    Image.new('RGB', (512, 512), color='red').save(test_image)
    
    # Process
    result = pipeline.process_render(test_image)
    
    assert result.image is not None
    assert result.processing_time_ms > 0
```

### Property-Based Tests (using hypothesis)
```python
# tests/test_{feature_name}_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    intensity=st.floats(min_value=0.0, max_value=1.0),
    width=st.integers(min_value=10, max_value=1000),
    height=st.integers(min_value=10, max_value=1000),
)
def test_{feature_name}_properties(intensity, width, height):
    """Property-based test for {FEATURE_NAME}."""
    processor = {FeatureName}Processor(intensity=intensity)
    image = Image.new('RGB', (width, height), color='red')
    
    result = processor.process(image)
    
    # Properties that should always hold
    assert result.size == (width, height)
    assert result.mode == 'RGB'
```

### Edge Case Tests
- [ ] Test with missing depth map (should work without it)
- [ ] Test with very small images (< 10x10 pixels)
- [ ] Test with very large images (8K resolution)
- [ ] Test with different image modes (RGB, RGBA, L, I;16)
- [ ] Test CPU fallback when GPU/MPS unavailable
- [ ] Test batch processing with mixed formats

### Performance Tests
```python
# tests/test_{feature_name}_performance.py
import time
from PIL import Image
import pytest

@pytest.mark.slow
def test_{feature_name}_performance_benchmark():
    """Benchmark processing performance."""
    processor = {FeatureName}Processor(intensity=0.5)
    image = Image.new('RGB', (2048, 2048), color='blue')
    
    start = time.perf_counter()
    result = processor.process(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Should process 2K image in under 100ms on modern hardware
    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.2f}ms"
```

---

## Documentation

### 1. Docstring Updates
**Files to document**:
- [ ] Add comprehensive docstrings to new classes/functions
- [ ] Include performance notes in docstrings
- [ ] Document parameter ranges and constraints

### 2. README Updates
**Section to update**: `README.md`

```markdown
### {FEATURE_NAME}

**Description**: {One-sentence feature description}

**Use Cases**:
- {Use case 1}
- {Use case 2}

**Usage Example**:
```python
from {module} import {FeatureName}Processor

processor = {FeatureName}Processor(intensity=0.7)
result = processor.process(image, depth_map=depth_map)
```

**CLI Usage**:
```bash
python {script_name}.py input.jpg output/ --{feature-name} --{feature-name}-intensity 0.7
```

**Performance**: ~{X}ms per image on M4 Max, {Y} images/hour batch throughput

**Parameters**:
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| intensity | float | 0.0-1.0 | 0.5 | Effect strength |
| mode | str | auto/manual | auto | Processing mode |
```

### 3. Configuration Documentation
**File to update**: `config/README.md` or inline comments

```yaml
# config/{preset_name}.yaml
# {FEATURE_NAME} Configuration
#
# Use this preset for: {USE_CASE}
# Performance: ~{X}ms per image
# Recommended for: {CONTENT_TYPE}

{feature_name}:
  enabled: true
  intensity: 0.5  # Effect strength (0.0=off, 1.0=maximum)
  mode: "auto"    # Processing mode: auto, aggressive, conservative
```

---

## Validation Checklist

### Code Quality
- [ ] Follows PEP 8 style guidelines (max line length: 127 chars)
- [ ] Type hints added where appropriate
- [ ] Docstrings follow NumPy/Google style
- [ ] No flake8 critical errors
- [ ] Pylint warnings addressed or annotated with Decision comments
- [ ] Code is DRY (Don't Repeat Yourself)

### Testing
- [ ] All unit tests pass: `pytest tests/test_{feature_name}.py -v`
- [ ] Integration tests pass: `pytest tests/integration/ -v`
- [ ] Edge cases covered
- [ ] Performance benchmarks documented
- [ ] Tests work without optional dependencies (with appropriate skips)

### Performance
- [ ] Processing time documented in docstring
- [ ] Memory usage profiled for large images
- [ ] GPU/MPS acceleration implemented (if applicable)
- [ ] LRU caching added for repeated operations
- [ ] Batch processing optimized

### Metadata & Compatibility
- [ ] IPTC/XMP metadata preserved
- [ ] GPS coordinates preserved (if present)
- [ ] Color metadata preserved (for video)
- [ ] Works with 8-bit and 16-bit images
- [ ] Works with RGB, RGBA, L modes

### Documentation
- [ ] README.md updated with usage examples
- [ ] Configuration documented
- [ ] Performance characteristics documented
- [ ] CLI help text accurate (`python {script}.py --help`)

### CI/CD
- [ ] All CI checks pass
- [ ] Tests pass on Python 3.10, 3.11, 3.12
- [ ] Linting passes (flake8, pylint)
- [ ] No new security vulnerabilities (CodeQL)

---

## Response Format (JSON Schema)

```json
{
  "summary": "Add {FEATURE_NAME} to {PIPELINE_NAME}",
  "files": [
    {
      "path": "depth_pipeline/processors/{feature_name}.py",
      "patch": "Create new processor class with process() method",
      "description": "Core implementation of {FEATURE_NAME} effect"
    },
    {
      "path": "config/{preset_name}.yaml",
      "patch": "Add {feature_name}_intensity: 0.5 parameter",
      "description": "Configuration for new feature"
    },
    {
      "path": "depth_pipeline/pipeline.py",
      "patch": "Integrate {FeatureName}Processor in process_render() workflow",
      "description": "Pipeline integration"
    }
  ],
  "tests": [
    "tests/test_{feature_name}.py",
    "tests/integration/test_depth_pipeline.py::test_{feature_name}_integration"
  ],
  "explanation": "Implementation uses depth information to modulate effect intensity across spatial zones. Near objects receive minimal effect, distant objects receive maximum effect based on depth_map values. Performance is ~{X}ms per image on M4 Max due to vectorized NumPy operations.",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "depth_pipeline/processors/atmospheric.py",
      "snippet": "Similar depth-based blending pattern",
      "relevance": "Shows correct approach for depth-modulated effects"
    }
  ]
}
```

---

## Few-Shot Examples from Repository

### Example 1: Adding Depth-Based Atmospheric Haze

**Input**:
```
Add depth-based atmospheric haze effect to the depth pipeline
```

**Output**:
```json
{
  "summary": "Add atmospheric haze effect based on depth information",
  "files": [
    {
      "path": "depth_pipeline/processors/atmospheric.py",
      "patch": "Add AtmosphericHazeProcessor class with depth-based fog blending",
      "description": "New processor for atmospheric effects using depth maps"
    },
    {
      "path": "config/exterior_preset.yaml",
      "patch": "Add haze_intensity: 0.3 to atmospheric effects section",
      "description": "Configure default haze for exterior renders"
    },
    {
      "path": "depth_pipeline/pipeline.py",
      "patch": "Integrate AtmosphericHazeProcessor in process_render() after tone mapping",
      "description": "Add haze effect to processing pipeline"
    }
  ],
  "tests": [
    "tests/test_atmospheric_processor.py",
    "tests/integration/test_depth_pipeline.py::test_atmospheric_haze_integration"
  ],
  "explanation": "Atmospheric haze is implemented by blending a fog color proportional to depth distance. Uses normalized depth maps (0.0=near, 1.0=far) to determine haze intensity per-pixel. Fog color can be configured (default: light blue #E0F0FF). Performance impact is ~8-12ms per 2K image using vectorized NumPy operations.",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "depth_pipeline/processors/zone_mapper.py",
      "snippet": "def apply_zone_processing(depth_map, zones): ...",
      "relevance": "Shows similar depth-based effect modulation pattern"
    }
  ]
}
```

### Example 2: Adding New LUT Preset

**Input**:
```
Add new LUT preset for warm California sunset aesthetic
```

**Output**:
```json
{
  "summary": "Add California Golden Hour LUT preset to video grader",
  "files": [
    {
      "path": "luxury_video_master_grader.py",
      "patch": "Add 'sunset_estate' preset with California_Golden_Hour.cube LUT",
      "description": "New preset in PRESETS dictionary with warm color grading"
    }
  ],
  "tests": [
    "tests/test_luxury_video_master_grader.py::test_preset_exists",
    "tests/test_luxury_video_master_grader.py::test_sunset_estate_preset_parameters"
  ],
  "explanation": "Adding a new preset is straightforward: define PresetConfig with LUT path (assets/luts/location_aesthetic/California_Golden_Hour.cube), exposure (+0.15 for warm lift), contrast (1.10), saturation (1.08 for enhanced warmth). LUT file should be created/sourced separately. No pipeline changes needed - preset system is fully extensible.",
  "confidence": 0.95,
  "citations": [
    {
      "file_path": "luxury_video_master_grader.py",
      "snippet": "PRESETS = { 'signature_estate': PresetConfig(...), ... }",
      "relevance": "Shows existing preset pattern to follow"
    }
  ]
}
```

---

## Repository-Specific Patterns

### Pattern 1: Depth-Based Processing
```python
# Always normalize depth maps to 0.0-1.0 range
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Use depth to modulate effect intensity
near_mask = depth_map < 0.3  # Foreground
mid_mask = (depth_map >= 0.3) & (depth_map < 0.7)  # Midground
far_mask = depth_map >= 0.7  # Background

# Apply different processing to each zone
result[near_mask] = apply_foreground_effect(image[near_mask])
result[mid_mask] = apply_midground_effect(image[mid_mask])
result[far_mask] = apply_background_effect(image[far_mask])
```

### Pattern 2: Metadata Preservation
```python
# Always preserve PIL Image.info dict
original_info = image.info.copy()
result = process_image(image)
result.info = original_info

# For TIFF files with tifffile
if tifffile_available:
    with tifffile.TiffFile(input_path) as tif:
        metadata = tif.pages[0].tags
        # ... processing ...
        tifffile.imwrite(output_path, result, metadata=metadata)
```

### Pattern 3: Optional Dependencies
```python
# Graceful fallback for optional dependencies
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    warnings.warn("tifffile not available, using Pillow for TIFF (8-bit only)")

# Use in code
if TIFFFILE_AVAILABLE:
    image = tifffile.imread(path)
else:
    image = Image.open(path)
```

### Pattern 4: LRU Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def estimate_depth(image_hash: str) -> np.ndarray:
    """Cached depth estimation (10-20x speedup)."""
    image = load_from_hash(image_hash)
    return depth_model.estimate(image)
```

### Pattern 5: Progress Tracking
```python
from tqdm import tqdm

def batch_process(image_paths: List[Path]):
    """Process multiple images with progress bar."""
    results = []
    for path in tqdm(image_paths, desc="Processing images"):
        result = process_image(path)
        results.append(result)
    return results
```

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
