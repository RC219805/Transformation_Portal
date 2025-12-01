# Unified Pipeline Migration Guide

## Overview

The Unified Pipeline Architecture (Phase 1) introduces a centralized orchestration system that combines all existing processing pipelines into a single, recipe-driven workflow. This document describes the architecture, migration path, and usage patterns.

## Architecture

### Pipeline Orchestrator

The `UnifiedPipeline` class in `src/transformation_portal/pipeline_unified.py` serves as the central orchestrator. It:

1. **Loads YAML recipes** that define processing stages and parameters
2. **Executes stages** in sequence with error recovery
3. **Generates outputs** in configurable formats
4. **Reports progress** with detailed timing and statistics

```python
from transformation_portal.pipeline_unified import UnifiedPipeline

# Load from recipe file
pipeline = UnifiedPipeline.from_recipe("config/recipes/signature_estate.yaml")

# Process single image
result = pipeline.process_single("input.jpg")

# Batch process with dry-run preview
results = pipeline.process_batch("inputs/*.jpg", "outputs/", dry_run=True)
```

### Processing Stages

The pipeline follows a stage-based architecture:

| Stage | Description | Optional |
|-------|-------------|----------|
| `depth_estimation` | Depth Anything V2 depth map generation | Yes |
| `ai_enhancement` | SDXL/ControlNet enhancement | Yes |
| `material_response` | Physics-based surface enhancement | Yes |
| `color_grading` | LUT application and color adjustments | Yes |
| `photo_finishing` | ACES, bloom, vignette, grain | Yes |
| `branding` | Logo/text overlay | Yes |

### YAML Recipes

Recipes define the processing workflow in a declarative format:

```yaml
name: "Signature Estate"
description: "Flagship Kodak 2393 emulation"

stages:
  - depth_estimation
  - material_response
  - color_grading
  - photo_finishing

material_response:
  enabled: true
  profile: "luxury_interior"
  texture_boost: 0.25

color_grading:
  enabled: true
  lut: "assets/luts/film_emulation/Kodak/Kodak_2393_D55.cube"
  lut_strength: 0.85

output:
  format: "tiff"
  quality: 95
```

## File Structure

```
src/transformation_portal/
├── pipeline_unified.py          # Core orchestrator
├── config_loader.py             # YAML recipe loader
├── processors/
│   └── material_response/
│       ├── engine.py           # Material Response Engine
│       └── profiles.py         # Material profiles
└── utils/
    └── recipe_validator.py     # Schema validation

config/
├── recipes/                    # Recipe presets
│   ├── signature_estate.yaml
│   ├── golden_hour_courtyard.yaml
│   ├── interior_neutral_luxe.yaml
│   └── video_cinematic_hdr.yaml
└── schemas/
    └── recipe_schema.json      # JSON Schema for validation

tests/
├── integration/
│   └── test_unified_pipeline.py
└── unit/
    └── test_material_response_engine.py
```

## Built-in Recipes

### Signature Estate
- **LUT:** Kodak 2393 D55
- **Profile:** Luxury Interior
- **Use Case:** Flagship real estate interiors
- **Output:** 16-bit TIFF

### Golden Hour Courtyard
- **LUT:** Montecito Golden Hour HDR
- **Profile:** Exterior Courtyard
- **Use Case:** Outdoor spaces at magic hour
- **Output:** High-quality JPEG

### Interior Neutral Luxe
- **LUT:** FilmConvert Nitrate
- **Profile:** Luxury Interior
- **Use Case:** Modern neutral interiors
- **Output:** 16-bit TIFF

### Video Cinematic HDR
- **LUT:** Kodak 2393 D55 HDR
- **Profile:** Luxury Interior (light)
- **Use Case:** 4K video frames
- **Output:** 16-bit TIFF

## Material Response Profiles

The `profiles.py` module defines preset material profiles:

| Profile | Use Case |
|---------|----------|
| `luxury_interior` | Default interior spaces |
| `wood_floor_oak` | Hardwood floor enhancement |
| `marble_stone` | Marble and natural stone |
| `textile_linen` | Upholstery and bedding |
| `metal_brushed` | Stainless steel and metal |
| `glass_window` | Glass and window surfaces |
| `exterior_courtyard` | Outdoor spaces |
| `aerial_estate` | Drone photography |

## CLI Commands

### Process Images

```bash
# Run pipeline with recipe
transform-process pipeline process \
    -i "renders/*.exr" \
    -o outputs/ \
    -r config/recipes/signature_estate.yaml

# Dry-run preview
transform-process pipeline process \
    -i "renders/*.exr" \
    -o outputs/ \
    -r config/recipes/signature_estate.yaml \
    --dry-run
```

### List Available Recipes

```bash
transform-process pipeline list-recipes
transform-process pipeline list-recipes -d custom/recipes/
```

### Validate Recipe

```bash
transform-process pipeline validate-recipe config/recipes/signature_estate.yaml
transform-process pipeline validate-recipe custom_recipe.yaml -v
```

## Migration from Existing Pipelines

### From lux_render_pipeline.py

The `lux_render_pipeline.py` functionality is preserved and can be accessed via:

1. **Direct import** (unchanged):
   ```python
   from transformation_portal.pipelines.lux_render_pipeline import LuxuryRenderPipeline
   ```

2. **Via unified pipeline** (new):
   ```python
   pipeline = UnifiedPipeline.from_recipe("config/recipes/signature_estate.yaml")
   result = pipeline.process_single("input.jpg")
   ```

### From unified_luxury_pipeline.py

The existing `unified_luxury_pipeline.py` remains functional. The new unified pipeline provides:

- Recipe-driven configuration
- Stage-based architecture
- Dry-run preview
- Better error recovery

### Migration Steps

1. **Identify your workflow**: Determine which stages you currently use
2. **Create a recipe**: Define your workflow in YAML format
3. **Test with dry-run**: Preview the processing plan
4. **Run processing**: Execute with the unified pipeline
5. **Compare outputs**: Verify quality matches expectations

## Configuration Reference

### Recipe Schema

```yaml
name: string (required)
description: string
stages: array of stage names (required)

depth_estimation:
  enabled: boolean
  model: string (default: "depth-anything-v2-small")
  device: "auto" | "cpu" | "cuda" | "mps"

material_response:
  enabled: boolean
  profile: profile name
  texture_boost: 0.0-1.0
  ambient_occlusion: 0.0-1.0
  highlight_warmth: 0.0-1.0
  # ... other parameters

color_grading:
  enabled: boolean
  lut: path to .cube file
  lut_strength: 0.0-1.0
  contrast: 0.5-2.0
  saturation: 0.0-2.0
  warmth: -0.5 to 0.5
  exposure: -2.0 to 2.0

photo_finishing:
  enabled: boolean
  aces: boolean
  bloom:
    enabled: boolean
    threshold: 0.0-1.0
    intensity: 0.0-1.0
  vignette:
    enabled: boolean
    strength: 0.0-1.0
  grain:
    enabled: boolean
    amount: 0.0-0.1

branding:
  enabled: boolean
  logo: path to logo file
  text: overlay text

output:
  format: "jpeg" | "png" | "tiff" | "exr"
  quality: 1-100
  bit_depth: 8 | 16 | 32
```

### Environment Variables

Recipes support environment variable expansion:

```yaml
color_grading:
  lut: "${LUT_BASE_PATH}/Kodak_2393_D55.cube"
```

### Relative Paths

Paths in recipes are resolved relative to the recipe file location.

## API Reference

### UnifiedPipeline

```python
class UnifiedPipeline:
    @classmethod
    def from_recipe(cls, recipe_path: Path) -> UnifiedPipeline:
        """Create pipeline from YAML recipe file."""

    def process_single(self, input_path: Path) -> ProcessingResult:
        """Process a single image through the pipeline."""

    def process_batch(
        self,
        input_glob: str,
        output_dir: Path,
        mode: str = "default",
        dry_run: bool = False
    ) -> BatchResult:
        """Process multiple images matching a glob pattern."""
```

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str]
    stages_executed: List[str]
    stage_times: Dict[str, float]
    total_time: float
```

### MaterialResponseEngine

```python
class MaterialResponseEngine:
    @classmethod
    def from_config(cls, config_dict: Dict) -> MaterialResponseEngine:
        """Create engine from configuration dictionary."""

    def apply(
        self,
        image: Image.Image,
        profile: Optional[str] = None,
        strength: float = 1.0
    ) -> Image.Image:
        """Apply Material Response enhancement to an image."""
```

## Performance Considerations

- **Dry-run first**: Always preview with `--dry-run` before batch processing
- **Stage selection**: Disable unused stages for faster processing
- **Device selection**: Use GPU/MPS when available for ML stages
- **Output format**: JPEG is faster; TIFF preserves quality
- **Batch size**: Process in batches to manage memory

## Troubleshooting

### Recipe Validation Errors

```bash
# Validate recipe with verbose output
transform-process pipeline validate-recipe my_recipe.yaml -v
```

### Processing Failures

Check the `ProcessingResult.error_message` for detailed error information:

```python
result = pipeline.process_single("input.jpg")
if not result.success:
    print(f"Error: {result.error_message}")
    print(f"Stages completed: {result.stages_executed}")
```

### Performance Issues

- Monitor stage times in `ProcessingResult.stage_times`
- Disable expensive stages (depth, AI) for quick previews
- Use the PERFORMANCE profile for faster processing

## Zero Breaking Changes

The Phase 1 implementation maintains full backward compatibility:

- All existing CLIs continue to work
- Existing imports are unchanged
- No required migration for existing code

## Next Steps

Phase 2 will introduce:
- Parallel stage execution
- GPU memory management
- Advanced caching
- Additional recipe presets
