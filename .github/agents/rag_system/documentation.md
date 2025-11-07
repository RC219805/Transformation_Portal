# Documentation Template

**Use this template for**: Documenting new features, pipelines, presets, APIs, and usage examples

---

## Documentation Types

**Choose documentation type**:
- [ ] **Feature Documentation** - Document new pipeline feature or processor
- [ ] **API Documentation** - Document public functions, classes, methods
- [ ] **Usage Guide** - How-to guide for specific workflows
- [ ] **Configuration Guide** - Document preset or YAML configuration
- [ ] **Troubleshooting Guide** - Common issues and solutions
- [ ] **Performance Documentation** - Benchmarks and optimization tips

---

## Feature Documentation Template

### Location
- README.md (user-facing features)
- docs/{feature_name}.md (detailed documentation)
- Inline docstrings (API reference)

### Structure

```markdown
## {Feature Name}

**Status**: ✅ Stable | ⚠️ Beta | 🚧 Experimental

**Since**: v{VERSION_NUMBER}

**Description**: {One-sentence description of what the feature does}

{Detailed explanation of the feature, its purpose, and benefits. Include:
- What problem it solves
- When to use it
- Key capabilities
}

### Use Cases

**Best for**:
- {Use case 1}
- {Use case 2}
- {Use case 3}

**Not recommended for**:
- {Anti-pattern 1}
- {Anti-pattern 2}

### Quick Start

**Basic Usage**:
```python
from {module} import {FeatureName}

# Initialize
feature = {FeatureName}(
    intensity=0.7,
    mode="auto"
)

# Process
result = feature.process(input_image)
```

**CLI Usage**:
```bash
python {script}.py input.jpg output/ \
    --{feature-flag} \
    --{feature-param} 0.7 \
    --verbose
```

### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `intensity` | float | 0.0-1.0 | 0.5 | Effect strength |
| `mode` | str | auto/manual | auto | Processing mode |
| `preserve_highlights` | bool | true/false | true | Protect bright areas |

**Parameter Details**:

**`intensity`**: Effect strength
- `0.0`: No effect (pass-through)
- `0.3-0.5`: Subtle enhancement (recommended for most cases)
- `0.7-1.0`: Strong effect (use with caution)

**`mode`**: Processing strategy
- `auto`: Automatically determine best approach based on content
- `manual`: Use fixed parameters (faster but less adaptive)

### Examples

**Example 1: Interior Architectural Render**
```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Process single image
result = pipeline.process_render('interior.jpg')

# Save output
pipeline.save_result(result, output_dir='output/')
```

**Example 2: Batch Processing**
```python
from pathlib import Path
from tqdm import tqdm

# Setup
input_dir = Path('renders/')
output_dir = Path('enhanced/')
output_dir.mkdir(exist_ok=True)

# Process all images
image_paths = list(input_dir.glob('*.jpg'))
for image_path in tqdm(image_paths, desc="Processing"):
    result = pipeline.process_render(image_path)
    pipeline.save_result(result, output_dir)

print(f"Processed {len(image_paths)} images")
```

**Example 3: Custom Configuration**
```python
# Custom configuration for exterior scenes
config = {
    'depth_model': {'name': 'vits', 'backend': 'coreml'},
    'tone_mapping': {
        'enabled': True,
        'operator': 'agx',
        'zones': {
            'foreground': {'exposure': 0.0, 'saturation': 1.05},
            'background': {'exposure': 0.15, 'saturation': 0.9}
        }
    },
    'atmospheric': {
        'enabled': True,
        'haze_intensity': 0.25
    }
}

pipeline = ArchitecturalDepthPipeline(config)
```

### Performance

**Benchmarks** (M4 Max, macOS 14.5):
- 512x512: ~15ms per image
- 2048x2048: ~45ms per image
- 4096x4096: ~180ms per image

**Throughput** (batch processing):
- 1K images: ~450 images/hour
- 2K images: ~400 images/hour
- 4K images: ~100 images/hour

**Memory Usage**:
- 512x512: ~50MB
- 2K: ~200MB
- 4K: ~800MB

**Optimization Tips**:
```python
# Enable caching for iterative workflows (10-20x speedup)
pipeline = ArchitecturalDepthPipeline(config, cache_size=128)

# Use smaller model for speed
config['depth_model']['name'] = 'vits'  # Fastest

# Batch processing with multiprocessing
from multiprocessing import Pool

def process_single(image_path):
    return pipeline.process_render(image_path)

with Pool(processes=4) as pool:
    results = pool.map(process_single, image_paths)
```

### Integration

**With Material Response**:
```python
from material_response import MaterialResponse

# Combine depth processing with material enhancement
depth_result = pipeline.process_render(image_path)
material_enhanced = MaterialResponse().enhance(
    depth_result.image,
    surfaces=['wood', 'metal', 'glass'],
    strength=0.7
)
```

**With LUT Application**:
```python
# Apply LUT after depth processing
from {module} import apply_lut

depth_result = pipeline.process_render(image_path)
graded = apply_lut(
    depth_result.image,
    lut_path='assets/luts/film_emulation/Kodak_2393.cube',
    strength=0.8
)
```

### Configuration Reference

**YAML Configuration** (`config/{feature}_preset.yaml`):
```yaml
# Full configuration example
{feature_name}:
  enabled: true
  intensity: 0.7
  mode: "auto"
  
  # Advanced options
  advanced:
    preserve_highlights: true
    depth_modulated: false
    fallback_mode: "cpu"
```

See [Configuration Guide](config/README.md) for all options.

### Troubleshooting

**Issue: Feature produces no visible effect**

Check intensity parameter:
```python
# Too low - no effect
feature = {FeatureName}(intensity=0.0)  # ✗

# Recommended range
feature = {FeatureName}(intensity=0.5)  # ✓
```

**Issue: Out of memory errors**

Reduce batch size or image resolution:
```python
# Process in tiles for large images
config['performance']['tile_size'] = 2048
```

**Issue: Slow processing**

Enable GPU/CoreML acceleration:
```python
config['depth_model']['backend'] = 'coreml'  # For Apple Silicon
config['depth_model']['backend'] = 'cuda'    # For NVIDIA GPUs
```

See [Troubleshooting Guide](#troubleshooting) for more issues.

### API Reference

**Class: `{ClassName}`**

```python
class {ClassName}:
    """
    {Brief description}
    
    {Detailed description}
    
    Args:
        param1: Description
        param2: Description
    
    Attributes:
        attr1: Description
        attr2: Description
    
    Example:
        >>> feature = {ClassName}(param1="value")
        >>> result = feature.process(image)
    """
    
    def __init__(self, param1: str, param2: float = 0.5):
        ...
    
    def process(self, image: Image.Image) -> Image.Image:
        """
        Process image with {FEATURE_NAME}.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Processed PIL Image
        
        Raises:
            ValueError: If image is invalid
            RuntimeError: If processing fails
        """
        ...
```

### Related Documentation

- [Depth Pipeline Guide](docs/depth_pipeline/DEPTH_PIPELINE_README.md)
- [Material Response Guide](assets/luts/material_response/_Material_Response_Technical_Guide.md)
- [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)
- [API Reference](docs/API_REFERENCE.md)

### Changelog

**v1.2.0** (2025-11-06)
- Added `preserve_highlights` parameter
- Improved performance by 25% with optimized NumPy operations
- Fixed metadata preservation bug

**v1.1.0** (2025-10-15)
- Added depth modulation support
- New `auto` mode for adaptive processing

**v1.0.0** (2025-09-01)
- Initial release

---

## API Documentation Template

### Docstring Format (NumPy Style)

```python
def function_name(
    param1: str,
    param2: int,
    param3: Optional[float] = None
) -> Dict[str, Any]:
    """
    Brief one-line summary of what the function does.
    
    More detailed explanation of the function's purpose, behavior,
    and any important details users should know.
    
    Parameters
    ----------
    param1 : str
        Description of param1. Include expected format, constraints,
        or example values.
    param2 : int
        Description of param2. Mention valid ranges if applicable.
    param3 : float, optional
        Description of optional parameter. Include default behavior
        when not provided. Default is None.
    
    Returns
    -------
    dict
        Description of return value structure:
        - 'key1': Description of key1
        - 'key2': Description of key2
    
    Raises
    ------
    ValueError
        When param1 is empty or param2 is negative.
    RuntimeError
        When processing fails due to system resources.
    
    Notes
    -----
    Additional important information:
    - Performance characteristics
    - Thread safety
    - Side effects
    
    Examples
    --------
    Basic usage:
    
    >>> result = function_name("test", 42)
    >>> result['key1']
    'expected_value'
    
    With optional parameter:
    
    >>> result = function_name("test", 42, param3=1.5)
    
    See Also
    --------
    related_function : Related functionality
    other_module.function : Alternative approach
    
    References
    ----------
    .. [1] Author Name, "Paper Title", Journal, Year.
           URL: https://example.com/paper
    """
    # Implementation
    pass
```

### Class Documentation

```python
class ClassName:
    """
    Brief one-line description of the class.
    
    Detailed explanation of what the class does, its purpose,
    and how it should be used.
    
    Parameters
    ----------
    param1 : str
        Description of initialization parameter.
    param2 : float, optional
        Description of optional parameter. Default is 0.5.
    
    Attributes
    ----------
    attr1 : str
        Description of public attribute.
    attr2 : int
        Description of another attribute.
    
    Methods
    -------
    method1(arg)
        Brief description of method1.
    method2()
        Brief description of method2.
    
    Examples
    --------
    Create an instance and use it:
    
    >>> obj = ClassName("value", param2=0.7)
    >>> result = obj.method1(input_data)
    
    Notes
    -----
    - Thread safety: This class is thread-safe.
    - Performance: O(n) for method1, O(1) for method2.
    
    See Also
    --------
    RelatedClass : Similar functionality
    """
    
    def __init__(self, param1: str, param2: float = 0.5):
        """
        Initialize {ClassName}.
        
        Parameters
        ----------
        param1 : str
            Parameter description.
        param2 : float, optional
            Parameter description. Default is 0.5.
        """
        self.param1 = param1
        self.param2 = param2
```

---

## Usage Guide Template

```markdown
# {Task Name} Guide

**Goal**: {What you'll accomplish with this guide}

**Prerequisites**:
- Python 3.10+
- Required packages: `{package1}`, `{package2}`
- Optional: GPU with CUDA or Apple Silicon

**Time**: ~{X} minutes

---

## Overview

{Brief explanation of the workflow and what it accomplishes}

**Steps**:
1. {Step 1 summary}
2. {Step 2 summary}
3. {Step 3 summary}

---

## Step 1: {Step Title}

{Explanation of what this step does and why}

```bash
# Commands for this step
command1
command2
```

**Expected output**:
```
Expected console output
```

**Troubleshooting**:
- **Issue**: {Common problem}
  - **Solution**: {How to fix it}

---

## Step 2: {Step Title}

{Explanation}

```python
# Python code for this step
from module import Class

# Setup
obj = Class(param="value")

# Execute
result = obj.process(input_data)
```

**Tips**:
- 💡 {Helpful tip or best practice}
- ⚠️ {Warning or common pitfall}

---

## Complete Example

**Full workflow script**:
```python
#!/usr/bin/env python3
"""
{Script description}

Usage:
    python workflow.py input_dir/ output_dir/
"""

from pathlib import Path
from {module} import {Class}

def main():
    # Configuration
    config = {
        'param1': 'value1',
        'param2': 0.7,
    }
    
    # Initialize
    processor = {Class}(config)
    
    # Process
    input_dir = Path('input/')
    output_dir = Path('output/')
    output_dir.mkdir(exist_ok=True)
    
    for input_file in input_dir.glob('*.jpg'):
        result = processor.process(input_file)
        output_file = output_dir / input_file.name
        result.save(output_file)
        print(f"Processed: {input_file.name}")

if __name__ == '__main__':
    main()
```

---

## Next Steps

**What to try next**:
- [ ] {Advanced topic or extension}
- [ ] {Integration with another feature}
- [ ] {Optimization or customization}

**Further Reading**:
- [Related Guide](link)
- [API Reference](link)
- [Performance Tips](link)
```

---

## Configuration Documentation Template

```markdown
# {Configuration Name} Reference

**File**: `config/{config_name}.yaml`

**Purpose**: {What this configuration is for}

**Recommended for**: {Content type or use case}

---

## Configuration Overview

```yaml
# config/{config_name}.yaml
{
  # Paste full configuration file here
}
```

---

## Parameter Reference

### Section: {Section Name}

**Purpose**: {What this section configures}

#### Parameter: `{parameter_name}`

- **Type**: `{type}`
- **Range**: `{min}-{max}` or `{option1}|{option2}`
- **Default**: `{default_value}`
- **Description**: {Detailed description}

**Effect**:
- Low values (`{range}`): {Effect description}
- Medium values (`{range}`): {Effect description}
- High values (`{range}`): {Effect description}

**Example**:
```yaml
{parameter_name}: {example_value}  # {Why this value}
```

---

## Preset Variations

### Variation 1: {Name}
**Use for**: {Use case}

```yaml
# Key parameters
param1: value1
param2: value2
```

**Example output**: [Link to example image]

### Variation 2: {Name}
**Use for**: {Different use case}

```yaml
# Key parameters
param1: different_value
param2: different_value
```

---

## Performance Notes

- **Processing time**: ~{X}ms per image (2K)
- **Memory usage**: ~{Y}MB peak
- **Recommended batch size**: {N}

**Optimization**:
```yaml
# For speed
{parameter}: {speed_value}

# For quality
{parameter}: {quality_value}
```
```

---

## Troubleshooting Guide Template

```markdown
# Troubleshooting: {Component Name}

Common issues and solutions for {COMPONENT_NAME}.

---

## Issue: {Problem Title}

**Symptoms**:
- {Symptom 1}
- {Symptom 2}

**Error Message** (if applicable):
```
{Paste error message}
```

**Cause**: {Root cause explanation}

**Solution**:

1. {Step 1}
   ```bash
   command1
   ```

2. {Step 2}
   ```python
   code_snippet()
   ```

3. Verify fix:
   ```bash
   verification_command
   ```

**Prevention**: {How to avoid this issue in the future}

---

## Issue: {Another Problem}

{Follow same structure}
```

---

## Documentation Checklist

### Before Publishing
- [ ] All code examples tested and working
- [ ] Parameter ranges verified
- [ ] Performance numbers accurate and current
- [ ] Links to related docs valid
- [ ] Spelling and grammar checked
- [ ] Screenshots/diagrams added (if applicable)
- [ ] Version number and date updated
- [ ] Added to main README table of contents

### Quality Standards
- [ ] Examples use realistic data/parameters
- [ ] Troubleshooting covers common issues
- [ ] API docs follow NumPy docstring format
- [ ] No deprecated APIs referenced
- [ ] Performance tips included where relevant
- [ ] Metadata preservation documented
- [ ] Platform-specific notes (macOS, Linux, Windows)

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
