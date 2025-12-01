# End-to-End 4K Rendering Enhancement Pipeline

A comprehensive, production-ready pipeline for enhancing architectural renders and luxury real estate imagery to 4K resolution with intelligent quality feedback.

## Overview

The 4K Rendering Enhancement Pipeline integrates the best features from the Transformation Portal ecosystem:

- **Depth Estimation**: Monocular depth using Depth Anything V2 (with CoreML/MPS acceleration)
- **Material Response Technology**: Physics-based surface enhancement for realistic materials
- **Intelligent Tone Mapping**: AgX, Filmic, Reinhard, and ACES operators
- **Color Grading**: Professional LUT support with saturation and vibrance controls
- **AI Enhancement**: ControlNet guidance integration point
- **4K Upscaling**: Lanczos with Real-ESRGAN integration point
- **RAG Quality Feedback**: Iterative quality assessment and refinement

## Quick Start

### Basic Usage

```python
from transformation_portal.pipelines import Rendering4KPipeline

# Create pipeline from preset
pipeline = Rendering4KPipeline.from_preset("luxury_estate")

# Process single image
result = pipeline.process("input.jpg", output_dir="output/")

# Check quality score
print(f"Quality Score: {result.quality_score:.2%}")
```

### CLI Usage

```bash
# Process single image
python -m transformation_portal.pipelines.rendering_4k_pipeline \
    -i input.jpg -o output/ --preset luxury_estate

# Batch process directory
python -m transformation_portal.pipelines.rendering_4k_pipeline \
    -d inputs/ -o outputs/ --preset editorial

# Preview mode (fast, lower quality)
python -m transformation_portal.pipelines.rendering_4k_pipeline \
    -i input.jpg -o output/ --preset preview
```

## Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Balanced settings | General purpose |
| `luxury_estate` | Optimized for interiors | Real estate interiors |
| `aerial_exterior` | Golden hour warmth | Aerial and exterior shots |
| `editorial` | Maximum quality | Magazine/print quality |
| `preview` | Fast preview | Quick iterations |

## Pipeline Stages

The pipeline processes images through 9 sequential stages:

1. **Input Validation**: Load and verify input image
2. **Depth Estimation**: Generate depth map (optional)
3. **Tone Mapping**: HDR compression with AgX/Filmic/ACES
4. **Material Response**: Surface texture enhancement
5. **Color Grading**: Temperature, saturation, vibrance
6. **AI Enhancement**: ControlNet guidance (optional)
7. **Upscaling**: Scale to 4K resolution
8. **Quality Assessment**: RAG-based quality metrics
9. **Output Generation**: Save TIFF masters and JPEG delivery

## Configuration

### YAML Configuration

Create custom configurations in YAML format:

```yaml
# config/my_preset.yaml
name: "my_custom_preset"
description: "Custom configuration"
quality_level: "high"

depth:
  enabled: true
  model_variant: "small"
  num_zones: 3

tone_mapping:
  method: "agx"
  exposure: 0.0
  contrast: 1.05

material_response:
  enabled: true
  strength: 0.75
  texture_boost: 0.3

color_grading:
  saturation: 1.08
  vibrance: 1.12
  temperature_shift: [1.0, 0.98, 0.95]

upscaling:
  enabled: true
  target_resolution: [3840, 2160]

quality_feedback:
  enabled: true
  min_quality_threshold: 0.75
  auto_adjust: true

output:
  master_tiff_16bit: true
  delivery_jpeg: true
  jpeg_quality: 95
```

Load custom configuration:

```python
pipeline = Rendering4KPipeline.from_yaml("config/my_preset.yaml")
```

### Programmatic Configuration

```python
from transformation_portal.pipelines.rendering_4k_pipeline import (
    PipelineConfig,
    DepthConfig,
    ToneMappingConfig,
    ToneMappingMethod,
    MaterialResponseConfig,
    ColorGradingConfig,
    UpscalingConfig,
    QualityFeedbackConfig,
    OutputConfig,
    QualityLevel,
    Rendering4KPipeline,
)

config = PipelineConfig(
    name="custom",
    quality_level=QualityLevel.ULTRA,
    depth=DepthConfig(enabled=True, num_zones=4),
    tone_mapping=ToneMappingConfig(
        method=ToneMappingMethod.AGX,
        exposure=0.1,
        contrast=1.1,
    ),
    material_response=MaterialResponseConfig(
        strength=0.8,
        texture_boost=0.35,
    ),
    color_grading=ColorGradingConfig(
        saturation=1.12,
        vibrance=1.15,
        temperature_shift=(1.05, 1.0, 0.95),  # Warm
    ),
    upscaling=UpscalingConfig(
        target_resolution=(3840, 2160),
        preserve_sharpness=True,
    ),
    quality_feedback=QualityFeedbackConfig(
        enabled=True,
        min_quality_threshold=0.8,
        auto_adjust=True,
    ),
)

pipeline = Rendering4KPipeline(config)
```

## Quality Assessment

The RAG-based quality feedback system evaluates images on multiple metrics:

| Metric | Description | Weight |
|--------|-------------|--------|
| Sharpness | Edge clarity (Laplacian variance) | 25% |
| Contrast | Luminance distribution | 20% |
| Colorfulness | Color saturation balance | 20% |
| Exposure | Brightness balance | 20% |
| Noise | Noise level (penalty) | 15% |

### Quality Feedback Loop

When quality falls below threshold, the system suggests adjustments:

```python
result = pipeline.process("input.jpg", output_dir="output/")

if result.quality_score < 0.75:
    # System automatically suggests adjustments
    # Example output:
    # {
    #     'clarity_boost': 0.2,
    #     'contrast_increase': 0.1,
    #     'saturation_boost': 0.05
    # }
```

## Output Files

The pipeline generates several output files:

| File | Format | Description |
|------|--------|-------------|
| `{name}_MASTER.tiff` | 16-bit TIFF | Master file for further editing |
| `{name}_DELIVERY.jpg` | Progressive JPEG | Client delivery |
| `{name}_depth.png` | 8-bit PNG | Depth visualization |
| `{name}_quality_report.json` | JSON | Processing metrics |

## Performance

### Benchmarks (M4 Max, 36GB RAM)

| Image Size | Processing Time | Throughput |
|------------|-----------------|------------|
| 1024×768 | ~500ms | ~7,200/hr |
| 2048×1536 | ~1.2s | ~3,000/hr |
| 4096×3072 | ~3.5s | ~1,000/hr |

### Optimization Tips

1. **Depth Caching**: Enable `depth.cache_enabled` for iterative workflows
2. **Preview Mode**: Use `preview` preset for parameter tuning
3. **Batch Processing**: Use `batch_process()` for multiple images
4. **Memory**: Disable `master_tiff_16bit` if not needed

## Integration Points

### Depth Anything V2

The pipeline includes integration points for the full Depth Anything V2 model:

```python
# Simple fallback (always available)
from transformation_portal.pipelines.rendering_4k_pipeline import estimate_depth_simple

# Full model (requires optional deps)
# Connect via the DepthConfig.backend setting
```

### Real-ESRGAN

For higher quality upscaling, configure the ESRGAN backend:

```yaml
upscaling:
  method: "esrgan"  # Requires: pip install realesrgan
```

### ControlNet

AI enhancement via ControlNet is available when ML dependencies are installed:

```yaml
ai_enhancement:
  enabled: true
  use_controlnet: true
  use_depth_guidance: true
  prompt: "photorealistic luxury interior..."
```

## Troubleshooting

### Common Issues

**ImportError: scipy not found**
```bash
pip install scipy
```

**Low quality scores**
- Increase `material_response.strength`
- Adjust `tone_mapping.contrast`
- Enable `quality_feedback.auto_adjust`

**Slow processing**
- Use `preview` preset for testing
- Disable depth estimation if not needed
- Reduce `upscaling.target_resolution`

## API Reference

### Rendering4KPipeline

```python
class Rendering4KPipeline:
    """Main pipeline class."""

    @classmethod
    def from_preset(cls, preset_name: str) -> "Rendering4KPipeline":
        """Create from built-in preset."""

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Rendering4KPipeline":
        """Create from YAML config file."""

    def process(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Process single image."""

    def batch_process(
        self,
        input_paths: List[Path],
        output_dir: Path,
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """Process multiple images."""

    def clear_cache(self):
        """Clear depth cache."""
```

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    image: Image.Image
    depth_map: Optional[np.ndarray]
    quality_metrics: Optional[QualityMetrics]
    stage_metrics: List[StageMetrics]
    total_duration_ms: float
    iterations: int
    output_paths: Dict[str, Path]
    config_used: Optional[PipelineConfig]

    @property
    def quality_score(self) -> float:
        """Overall quality score (0-1)."""
```

## See Also

- [Unified Luxury Pipeline](UNIFIED_LUXURY_PIPELINE.md)
- [Elite Pipeline Guide](ELITE_PIPELINE_GUIDE.md)
- [Pro Pipeline Guide](PRO_PIPELINE_GUIDE.md)
- [Depth Pipeline Documentation](depth_pipeline/DEPTH_PIPELINE_README.md)
