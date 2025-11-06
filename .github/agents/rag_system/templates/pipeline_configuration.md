# Pipeline Configuration Template

**Use this template for**: Creating new YAML configurations for depth pipeline, video grader, batch processors, and custom presets

---

## Configuration Overview

**Configuration Name**: `{CONFIG_NAME}`

**Target Pipeline**:
- [ ] Depth Pipeline (`depth_pipeline/`) → Save to: `config/{config_name}.yaml`
- [ ] Video Grader Preset → Modify: `luxury_video_master_grader.py`
- [ ] TIFF Processor Preset → Modify: `luxury_tiff_batch_processor.py`
- [ ] Custom Pipeline → Create: `config/custom/{config_name}.yaml`

**Use Case**:
```
{DESCRIBE_INTENDED_USE_CASE}

Examples:
- "Interior architectural renders with dramatic depth-of-field"
- "Exterior daytime scenes with atmospheric haze"
- "Aerial photography with gradient tone mapping"
- "Product photography with material enhancement"
```

**Content Type**:
- [ ] Interior Architecture
- [ ] Exterior Architecture
- [ ] Aerial/Drone Photography
- [ ] Product Photography
- [ ] Portrait/People
- [ ] Landscape
- [ ] Other: `{SPECIFY}`

---

## Depth Pipeline Configuration

### Template: `config/{config_name}.yaml`

```yaml
# {CONFIG_NAME} Configuration
# Use case: {USE_CASE_DESCRIPTION}
# Performance: ~{X}ms per image on M4 Max
# Recommended for: {CONTENT_TYPE}

# Model Configuration
depth_model:
  name: "depth_anything_v2_vits"  # Options: vits, vitb, vitl
  backend: "auto"  # Options: auto, coreml, cuda, cpu
  precision: "fp16"  # Options: fp32, fp16
  cache_predictions: true

# Depth Processing
depth_processing:
  normalize: true
  smoothing_sigma: 1.0  # Gaussian smoothing (0.0 = off)
  clip_range: [0.0, 1.0]  # Normalize to this range

# Zone-Based Tone Mapping
tone_mapping:
  enabled: true
  operator: "agx"  # Options: agx, reinhard, filmic, hable, aces
  
  # Zone definitions (depth ranges)
  zones:
    foreground:
      depth_range: [0.0, 0.3]  # Near objects
      exposure: 0.0
      contrast: 1.0
      saturation: 1.0
    
    midground:
      depth_range: [0.3, 0.7]  # Mid-distance
      exposure: 0.05
      contrast: 1.05
      saturation: 1.02
    
    background:
      depth_range: [0.7, 1.0]  # Far objects
      exposure: 0.1
      contrast: 1.1
      saturation: 0.95  # Slight desaturation for distance

# Depth-Aware Denoising
denoising:
  enabled: true
  method: "bilateral"  # Options: bilateral, nlmeans, guided
  strength: 0.5  # 0.0-1.0
  depth_modulated: true  # Stronger denoising on distant objects

# Atmospheric Effects
atmospheric:
  enabled: false
  haze_intensity: 0.0  # 0.0-1.0
  haze_color: [224, 240, 255]  # RGB [0-255] - light blue
  fog_density: 0.0  # 0.0-1.0
  depth_falloff: 2.0  # Exponential falloff rate

# Clarity Enhancement
clarity:
  enabled: true
  strength: 0.15  # 0.0-1.0
  radius: 5  # pixels
  preserve_highlights: true
  depth_modulated: false

# Color Grading
color_grading:
  enabled: true
  lut_path: ""  # Path to .cube LUT file (optional)
  lut_strength: 0.8  # 0.0-1.0 if LUT is used
  
  # Manual adjustments
  exposure: 0.0  # -2.0 to +2.0 (EV)
  contrast: 1.0  # 0.5 to 2.0
  saturation: 1.0  # 0.0 to 2.0
  vibrance: 0.0  # -1.0 to +1.0
  temperature: 0  # -100 to +100 (Kelvin shift)
  tint: 0  # -100 to +100 (green-magenta)

# Sharpening
sharpening:
  enabled: true
  method: "unsharp_mask"  # Options: unsharp_mask, high_pass, clarity
  amount: 0.5  # 0.0-2.0
  radius: 1.0  # pixels
  threshold: 0  # 0-255 (edges to ignore)
  depth_modulated: true  # Less sharpening on distant objects

# Material Response (if integrated)
material_response:
  enabled: false
  surfaces: []  # Options: wood, metal, glass, fabric, stone
  strength: 0.7
  preserve_highlights: true

# Output Settings
output:
  format: "tiff"  # Options: tiff, jpg, png, exr
  bit_depth: 16  # 8 or 16 (for tiff/png)
  quality: 95  # 1-100 (for jpg)
  preserve_metadata: true
  preserve_gps: true

# Performance Tuning
performance:
  batch_size: 4
  num_workers: 4  # Parallel processing threads
  cache_size: 128  # LRU cache size for depth predictions
  use_gpu: true
  gpu_memory_fraction: 0.8  # Max GPU memory to use (0.0-1.0)

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  show_progress: true
  save_depth_maps: false  # Save depth maps alongside outputs
  save_zone_maps: false  # Save zone visualization maps
```

---

### Parameter Reference

#### Depth Model Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `name` | str | vits/vitb/vitl | vits | Model size (vits=fastest, vitl=most accurate) |
| `backend` | str | auto/coreml/cuda/cpu | auto | Processing backend (auto selects best available) |
| `precision` | str | fp32/fp16 | fp16 | Floating point precision (fp16 faster, slightly less accurate) |
| `cache_predictions` | bool | true/false | true | Cache depth predictions for repeated images (10-20x speedup) |

**Performance**:
- `vits`: 24-35ms/image (M4 Max CoreML)
- `vitb`: 45-60ms/image (M4 Max CoreML)
- `vitl`: 60-80ms/image (M4 Max CoreML)

#### Zone-Based Tone Mapping

Zones divide the image by depth for differential processing:
- **Foreground** (0.0-0.3): Objects close to camera
- **Midground** (0.3-0.7): Mid-distance objects
- **Background** (0.7-1.0): Distant objects/sky

**Common Patterns**:
- **Interior**: Bright foreground, darker background
  - foreground: `exposure: 0.1, contrast: 1.0`
  - background: `exposure: -0.1, contrast: 1.1`
  
- **Exterior**: Even foreground, lighter background (sky)
  - foreground: `exposure: 0.0, contrast: 1.05`
  - background: `exposure: 0.15, saturation: 0.9`
  
- **Aerial**: Compressed depth, uniform tone mapping
  - All zones: `exposure: 0.0, contrast: 1.08, saturation: 1.1`

#### Atmospheric Effects

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `haze_intensity` | float | 0.0-1.0 | 0.0 | Overall haze strength |
| `haze_color` | RGB | [0-255] | [224,240,255] | Fog/haze color (typically light blue) |
| `fog_density` | float | 0.0-1.0 | 0.0 | Ground fog density |
| `depth_falloff` | float | 1.0-5.0 | 2.0 | How quickly haze increases with distance |

**Use Cases**:
- **Morning mist**: `haze_intensity: 0.3, fog_density: 0.2, haze_color: [240, 245, 250]`
- **Distant mountains**: `haze_intensity: 0.5, depth_falloff: 3.0, haze_color: [200, 220, 240]`
- **Urban smog**: `haze_intensity: 0.4, haze_color: [200, 200, 200]`

#### Clarity Enhancement

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `strength` | float | 0.0-1.0 | 0.15 | Clarity effect strength |
| `radius` | int | 1-20 | 5 | Effect radius in pixels |
| `preserve_highlights` | bool | true/false | true | Protect highlights from over-enhancement |
| `depth_modulated` | bool | true/false | false | Apply more clarity to foreground |

**Guidelines**:
- **Architectural interiors**: `strength: 0.2, radius: 8`
- **Product photography**: `strength: 0.25, radius: 3`
- **Landscapes**: `strength: 0.15, radius: 10`

#### Color Grading

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `lut_path` | str | path | "" | Path to .cube LUT file |
| `lut_strength` | float | 0.0-1.0 | 0.8 | LUT opacity/strength |
| `exposure` | float | -2.0 to +2.0 | 0.0 | Exposure compensation (EV) |
| `contrast` | float | 0.5-2.0 | 1.0 | Global contrast multiplier |
| `saturation` | float | 0.0-2.0 | 1.0 | Color saturation (1.0=neutral) |
| `vibrance` | float | -1.0 to +1.0 | 0.0 | Smart saturation (affects muted colors more) |
| `temperature` | int | -100 to +100 | 0 | Warmth (negative=cooler, positive=warmer) |
| `tint` | int | -100 to +100 | 0 | Green-magenta shift |

**Common LUT Paths**:
```yaml
# Film Emulation
lut_path: "assets/luts/film_emulation/Kodak_2393.cube"
lut_path: "assets/luts/film_emulation/Kodak_Vision3_250D.cube"

# Location Aesthetic
lut_path: "assets/luts/location_aesthetic/California_Golden_Hour.cube"
lut_path: "assets/luts/location_aesthetic/Nordic_Cool.cube"

# Material Response
lut_path: "assets/luts/material_response/Wood_Enhancement.cube"
lut_path: "assets/luts/material_response/Metal_Polish.cube"
```

---

## Video Grader Preset Configuration

### Template: Add to `luxury_video_master_grader.py`

```python
# In PRESETS dictionary
PRESETS = {
    "{preset_name}": PresetConfig(
        name="{Display Name}",
        lut="assets/luts/{category}/{lut_file}.cube",
        notes="{Description of aesthetic, use case, and inspiration}",
        
        # Global Adjustments
        exposure=0.0,      # -1.0 to +1.0 (EV)
        contrast=1.0,      # 0.5 to 2.0
        saturation=1.0,    # 0.0 to 2.0
        
        # Enhancement (optional)
        clarity=0.0,       # 0.0 to 0.5 (micro-contrast)
        glow=0.0,          # 0.0 to 0.1 (highlight bloom)
        grain=0.0,         # 0.0 to 0.05 (film grain)
        vignette=0.0,      # 0.0 to 0.5 (edge darkening)
        
        # Advanced (optional)
        hdr_tone_map="none",  # none, hable, reinhard, mobius
        temperature=0,        # -100 to +100 (Kelvin)
        tint=0,              # -100 to +100 (green-magenta)
    ),
}
```

### Example Presets

```python
# Warm Sunset Estate
"sunset_estate": PresetConfig(
    name="Sunset Estate",
    lut="assets/luts/location_aesthetic/California_Golden_Hour.cube",
    notes="Warm golden hour aesthetic for California luxury estates. "
          "Enhanced warmth, lifted shadows, gentle contrast.",
    exposure=0.15,
    contrast=1.08,
    saturation=1.10,
    clarity=0.18,
    grain=0.012,
    temperature=15,
),

# Nordic Minimalist
"nordic_minimal": PresetConfig(
    name="Nordic Minimal",
    lut="assets/luts/location_aesthetic/Nordic_Cool.cube",
    notes="Cool, clean aesthetic for Scandinavian design. "
          "Slightly desaturated, high clarity, blue-shifted.",
    exposure=0.05,
    contrast=1.12,
    saturation=0.92,
    clarity=0.22,
    temperature=-10,
    tint=-5,
),

# Cinematic Moody
"cinematic_moody": PresetConfig(
    name="Cinematic Moody",
    lut="assets/luts/film_emulation/Kodak_2383.cube",
    notes="Film-inspired moody look with crushed blacks and muted colors. "
          "Great for dramatic architectural videos.",
    exposure=-0.05,
    contrast=1.15,
    saturation=0.88,
    clarity=0.15,
    grain=0.018,
    vignette=0.25,
),
```

---

## TIFF Batch Processor Preset

### Template: Add to `luxury_tiff_batch_processor.py`

```python
# In PRESETS dictionary
"{preset_name}": {
    "name": "{Display Name}",
    "description": "{Detailed description of aesthetic and use case}",
    
    # Adjustments
    "exposure": 0.0,      # -2.0 to +2.0 (EV)
    "contrast": 1.0,      # 0.5 to 2.0
    "saturation": 1.0,    # 0.0 to 2.0
    "clarity": 0.0,       # 0.0 to 1.0
    
    # Color (optional)
    "temperature": 0,     # -100 to +100
    "tint": 0,           # -100 to +100
    
    # LUT (optional)
    "lut": "",           # Path to .cube file
    "lut_strength": 0.8, # 0.0 to 1.0
    
    # Sharpening (optional)
    "sharpen": 0.0,      # 0.0 to 2.0
    "sharpen_radius": 1.0,
    
    # Effects (optional)
    "glow": 0.0,         # 0.0 to 0.2
    "grain": 0.0,        # 0.0 to 0.1
},
```

---

## Configuration Validation

### Required Validations

**Before deploying configuration**:

1. **File Existence**
   ```python
   # Validate LUT file exists
   from pathlib import Path
   
   lut_path = Path(config['color_grading']['lut_path'])
   assert lut_path.exists(), f"LUT not found: {lut_path}"
   assert lut_path.suffix == '.cube', f"LUT must be .cube format"
   ```

2. **Parameter Ranges**
   ```python
   # Validate parameter ranges
   assert 0.0 <= config['clarity']['strength'] <= 1.0
   assert -2.0 <= config['color_grading']['exposure'] <= 2.0
   assert config['depth_model']['name'] in ['vits', 'vitb', 'vitl']
   ```

3. **Logical Consistency**
   ```python
   # Zone depth ranges should not overlap
   zones = config['tone_mapping']['zones']
   for zone_name, zone_config in zones.items():
       start, end = zone_config['depth_range']
       assert 0.0 <= start < end <= 1.0
   ```

4. **Performance Constraints**
   ```python
   # Warn about performance-intensive settings
   if config['depth_model']['name'] == 'vitl':
       logger.warning("vitl model is slower (~80ms/image). Consider vits for faster processing.")
   
   if config['performance']['batch_size'] > 8:
       logger.warning("Large batch size may cause OOM on systems with < 16GB RAM")
   ```

---

## Testing Configuration

### Manual Testing Workflow

```bash
# 1. Create test configuration
cp config/default_config.yaml config/test_config.yaml
# Edit test_config.yaml with new parameters

# 2. Test with single image
python depth_pipeline/cli.py \
    --config config/test_config.yaml \
    --input data/sample_images/interior_01.jpg \
    --output output/test/

# 3. Inspect output
# - Visual quality check
# - Depth map accuracy (if saved)
# - Processing time
# - Metadata preservation

# 4. Batch test (small set)
python depth_pipeline/cli.py \
    --config config/test_config.yaml \
    --input data/sample_images/ \
    --output output/batch_test/ \
    --verbose

# 5. Performance benchmark
python -m depth_pipeline.benchmark \
    --config config/test_config.yaml \
    --num-images 10
```

### Automated Validation

```python
# tests/test_config_validation.py
import pytest
import yaml
from pathlib import Path

def test_config_file_valid_yaml():
    """Test that all config files are valid YAML."""
    config_dir = Path('config/')
    for config_file in config_dir.glob('*.yaml'):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert config is not None

def test_config_has_required_fields():
    """Test that config has all required fields."""
    with open('config/test_config.yaml') as f:
        config = yaml.safe_load(f)
    
    required_sections = [
        'depth_model',
        'tone_mapping',
        'output',
    ]
    
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"

def test_lut_paths_exist():
    """Test that referenced LUT files exist."""
    with open('config/test_config.yaml') as f:
        config = yaml.safe_load(f)
    
    lut_path = config.get('color_grading', {}).get('lut_path', '')
    if lut_path:
        assert Path(lut_path).exists(), f"LUT not found: {lut_path}"

@pytest.mark.parametrize("param,min_val,max_val", [
    ('clarity.strength', 0.0, 1.0),
    ('color_grading.exposure', -2.0, 2.0),
    ('color_grading.saturation', 0.0, 2.0),
])
def test_parameter_ranges(param, min_val, max_val):
    """Test that parameters are within valid ranges."""
    with open('config/test_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Navigate nested dict
    keys = param.split('.')
    value = config
    for key in keys:
        value = value[key]
    
    assert min_val <= value <= max_val, \
        f"{param} out of range: {value} not in [{min_val}, {max_val}]"
```

---

## Configuration Examples by Use Case

### Interior Architecture

**Focus**: Dramatic depth, enhanced foreground clarity, warm color

```yaml
# config/interior_preset.yaml
tone_mapping:
  enabled: true
  zones:
    foreground:
      depth_range: [0.0, 0.35]
      exposure: 0.1
      contrast: 1.0
      saturation: 1.05
    background:
      depth_range: [0.65, 1.0]
      exposure: -0.1
      contrast: 1.15

clarity:
  enabled: true
  strength: 0.22
  depth_modulated: true

color_grading:
  temperature: 10  # Slight warmth
  lut_path: "assets/luts/location_aesthetic/Interior_Warm.cube"
```

### Exterior Architecture

**Focus**: Even lighting, atmospheric depth, natural color

```yaml
# config/exterior_preset.yaml
tone_mapping:
  enabled: true
  zones:
    foreground:
      depth_range: [0.0, 0.3]
      exposure: 0.0
      saturation: 1.05
    background:
      depth_range: [0.7, 1.0]
      exposure: 0.15
      saturation: 0.92  # Desaturate sky

atmospheric:
  enabled: true
  haze_intensity: 0.25
  depth_falloff: 2.5

clarity:
  enabled: true
  strength: 0.18
```

### Aerial Photography

**Focus**: Compressed depth, high clarity, vibrant color

```yaml
# config/aerial_preset.yaml
tone_mapping:
  enabled: true
  operator: "agx"
  zones:
    # Uniform tone mapping (minimal depth variation)
    foreground:
      depth_range: [0.0, 0.5]
      contrast: 1.08
      saturation: 1.12
    background:
      depth_range: [0.5, 1.0]
      contrast: 1.08
      saturation: 1.12

clarity:
  enabled: true
  strength: 0.25
  radius: 8

atmospheric:
  enabled: false  # No haze for aerial
```

### Product Photography

**Focus**: Sharp detail, material enhancement, neutral color

```yaml
# config/product_preset.yaml
depth_model:
  name: "vits"  # Fast for batch processing

clarity:
  enabled: true
  strength: 0.30
  radius: 3
  preserve_highlights: true

sharpening:
  enabled: true
  amount: 0.8
  radius: 1.0

material_response:
  enabled: true
  surfaces: ["metal", "glass", "wood"]
  strength: 0.75

color_grading:
  saturation: 1.05
  vibrance: 0.1
```

---

## Performance Optimization Tips

### For Speed (High Throughput)
```yaml
depth_model:
  name: "vits"          # Fastest model
  backend: "coreml"     # Use Apple Neural Engine if available
  precision: "fp16"

performance:
  batch_size: 8
  cache_size: 256       # Large cache for repeated images
  
# Disable expensive effects
atmospheric:
  enabled: false
denoising:
  enabled: false
```

### For Quality (Best Results)
```yaml
depth_model:
  name: "vitl"          # Most accurate model
  precision: "fp32"

denoising:
  enabled: true
  method: "nlmeans"     # Slower but better quality
  strength: 0.7

sharpening:
  enabled: true
  method: "high_pass"   # Better edge preservation
  amount: 0.6
```

### For Memory Efficiency (Large Images)
```yaml
performance:
  batch_size: 1         # Process one image at a time
  cache_size: 32        # Smaller cache
  gpu_memory_fraction: 0.6

# Disable memory-intensive features
logging:
  save_depth_maps: false
  save_zone_maps: false
```

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
