# Examples Directory

Example code demonstrating various features and workflows.

## Structure
- **pipelines/** - Pipeline usage examples
- **rag/** - RAG system examples
- **workflows/** - Complete workflow demonstrations

## Quick Examples: PBRProcessor API

### Single File Processing

Generate PBR maps from existing depth:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Configure premium quality preset
config = get_preset("premium").to_pbr_config()

# Generate PBR maps from cached depth
paths = PBRProcessor.from_cached_depth(
    depth_path=Path("output/scene1_depth.npy"),
    config=config,
    output_dir=Path("output/pbr/"),
    base_name="scene1"
)

print(f"Normal map: {paths['normal']}")
print(f"Roughness: {paths['roughness']}")
print(f"AO map: {paths['ao']}")
```

### Batch Processing

Process multiple depth files with different presets:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

# Input depth files
depth_dir = Path("output/depths/")
depth_files = list(depth_dir.glob("*_depth.npy"))

# Process with each preset
for preset_name in ["standard", "premium", "wood", "metal"]:
    config = get_preset(preset_name).to_pbr_config()
    output_dir = Path(f"output/pbr_{preset_name}/")

    for depth_file in depth_files:
        base_name = depth_file.stem.replace("_depth", "")

        paths = PBRProcessor.from_cached_depth(
            depth_path=depth_file,
            config=config,
            output_dir=output_dir,
            base_name=base_name
        )

        print(f"Processed {base_name} with {preset_name} preset")
```

### Memory-Only Mode

Generate PBR maps without file I/O (useful for custom post-processing):

```python
import numpy as np
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset
from PIL import Image

# Load depth array
depth = np.load("output/scene1_depth.npy")

# Configure processor
config = get_preset("standard").to_pbr_config()
processor = PBRProcessor(config=config, output_dir=None)

# Generate maps in memory
maps = processor.from_depth(depth, save=False)

# Custom post-processing
normal_map = maps["normal"]
roughness_map = maps["roughness"]
ao_map = maps["ao"]

# Apply custom adjustments (e.g., AO intensity)
ao_adjusted = (ao_map * 1.2).clip(0, 255).astype(np.uint8)

# Save custom outputs
Image.fromarray(normal_map).save("output/custom_normal.png")
Image.fromarray(ao_adjusted).save("output/custom_ao.png")
```

### Custom Parameter Overrides

Override preset parameters for fine-tuning:

```python
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset
from transformation_portal.lux_depth_v3.pbr import PBRConfig

# Start with standard preset
preset = get_preset("standard")
config = preset.to_pbr_config()

# Override specific parameters
custom_config = PBRConfig(
    normal_strength=1.5,  # Stronger than standard
    normal_blur_radius=0,  # No blur (sharper)
    roughness_strength=config.roughness_strength,
    roughness_blur_radius=config.roughness_blur_radius,
    ao_strength=1.2,  # Stronger AO
    ao_blur_radius=config.ao_blur_radius,
    ao_bias=0.35  # Darker shadows
)

# Process with custom config
processor = PBRProcessor(config=custom_config, output_dir="output/custom_pbr/")
maps = processor.from_depth(depth, save=True, base_name="scene1")
```

## Production Examples

### RAG Workflows

**rag/run_rag_workflow.py** - End-to-end RAG workflow runner for the repository.

Run:
```bash
python examples/rag/run_rag_workflow.py
```

### Luxury Estate Pipeline Usage

**pipelines/elite_architectural_pipeline_examples.py** - Interactive usage
examples for the elite architectural pipeline implementation under
`scripts/pipelines/`.

Run:
```bash
python examples/pipelines/elite_architectural_pipeline_examples.py
```

**pipelines/luxury_estate_pipeline_examples.py** - Usage examples for the
legacy luxury-estate master pipeline implementation under `scripts/pipelines/`.

Run:
```bash
python examples/pipelines/luxury_estate_pipeline_examples.py
```

### PBR Map Generation

**process_750_picacho_pbr.py** - Production-ready example for luxury real estate PBR processing

Demonstrates optimal PBR map generation for the 750 Picacho Primary Bedroom using the new Lux Depth V3 presets.

**Features:**
- Material-aware preset selection (premium, wood, stone, glass, fabric)
- Production-quality output for hero shot marketing
- Comprehensive source file analysis and validation
- Integration with existing 750 Picacho processing workflows
- Detailed performance reporting and troubleshooting

**Quick Start:**
```bash
# Process with premium quality (recommended for hero shots)
python examples/process_750_picacho_pbr.py

# Emphasize hardwood flooring detail
python examples/process_750_picacho_pbr.py --preset wood

# Validate configuration without processing
python examples/process_750_picacho_pbr.py --dry-run

# List available presets and material recommendations
python examples/process_750_picacho_pbr.py --list-presets
```

**Preset Recommendations for 750 Picacho:**
- `premium` - **RECOMMENDED** for hero shot marketing (max quality, 6-8s/image)
- `wood` - Emphasize hardwood grain and plank texture
- `fabric` - Emphasize bedding and textile weave patterns
- `glass` - Emphasize windows and reflective surfaces

**Expected Outputs:**
```
output_750_picacho_pbr/
├── 750Picacho_PrimaryBedroom_Ultimate_depth.png       # 16-bit depth visualization
├── 750Picacho_PrimaryBedroom_Ultimate_depth_float.npy # High-precision depth array
├── 750Picacho_PrimaryBedroom_Ultimate_normal.png      # RGB normal map
├── 750Picacho_PrimaryBedroom_Ultimate_roughness.png   # Grayscale roughness
├── 750Picacho_PrimaryBedroom_Ultimate_ao.png          # Ambient occlusion
└── 750Picacho_PrimaryBedroom_Ultimate_manifest.json   # Processing metadata
```

**Material Analysis:**
The 750 Picacho Primary Bedroom contains:
- Hardwood flooring (wide-plank, satin finish) - 15-20% of frame
- Premium textiles (bedding, curtains) - 30-40% of frame
- Architectural glass (windows, mirrors) - 10-15% of frame
- Stone surfaces (visible bathroom) - 5-10% of frame
- Metal accents (fixtures, hardware) - 5% of frame

**Integration:**
- Compatible with existing depth-aware processing (`process_750_picacho_depth_aware.py`)
- Follows `output_750_picacho_*` naming convention
- PBR maps ready for 3D workflows and compositing
- Depth maps cached for 10-20x speedup in iterative workflows

**Performance:**
- Premium quality: ~6-8 seconds/image (first run with depth estimation)
- Subsequent runs: ~0.3-0.5 seconds (depth cached)
- Memory: ~5.5 GB peak with METRIC_LARGE model
- Throughput: 100-150 images/hour

**Documentation:**
- Full configuration guide: `docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md`
- Quick reference: `docs/reference/PBR_PRESETS_QUICK_REFERENCE.md`
- Preset module: `src/transformation_portal/lux_depth_v3/pbr_presets.py`

### Other Examples

**pbr_preset_example.py** - General-purpose PBR preset demonstration
**simple_process.py** - Basic image processing workflow
**batch_process.py** - Batch processing with progress tracking
**custom_pipeline.py** - Custom pipeline configuration
