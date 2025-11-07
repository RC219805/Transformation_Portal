# Architectural Context Integration Guide

## Overview

The **Architectural Context Engine** integrates architectural documentation (floor plans, elevations, dimensions, specifications) into the AI-powered rendering pipeline. This enables **context-aware processing** that understands spatial relationships, material specifications, and design intent.

## Key Benefits

1. **Enhanced AI Prompts**: Automatically enriched with project-specific context
2. **Space-Aware Processing**: Different treatments for kitchens vs. living rooms vs. exteriors
3. **Material Intelligence**: Surfaces rendered based on actual specifications
4. **Dimension-Informed Depth**: Spatial understanding improves depth estimation
5. **Design Intent Preservation**: Style and aesthetic goals maintained throughout processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Input Sources                               │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Image File  │  Floor Plans │  Elevations  │  Specs/Notes  │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       │              │              │               │
       v              v              v               v
┌─────────────────────────────────────────────────────────────┐
│          Architectural Context Extractor                     │
├─────────────────────────────────────────────────────────────┤
│  • Filename parsing (project, space type, style)            │
│  • PDF text extraction (dimensions, materials)               │
│  • Pattern matching (design intent, specifications)          │
│  • Context caching (JSON for reuse)                          │
└──────────────────────────┬──────────────────────────────────┘
                           v
┌─────────────────────────────────────────────────────────────┐
│             Architectural Context Object                     │
├─────────────────────────────────────────────────────────────┤
│  Project: 750 Picacho Lane                                   │
│  Space: Kitchen (12' x 14', 10' ceiling)                     │
│  Materials: White oak flooring, quartz counters, brass       │
│  Design Intent: Open concept, natural light, luxury finishes │
│  Style: Bright, airy atmosphere                              │
└──────────────────────────┬──────────────────────────────────┘
                           v
┌─────────────────────────────────────────────────────────────┐
│          Context-Aware Processing Pipeline                   │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ Depth Stage  │ Material     │ AI Enhanced  │ Upscaling     │
│ (config per  │ Response     │ (enriched    │ (optional 4x) │
│  space type) │ (spec-based) │  prompts)    │               │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       v              v              v               v
┌─────────────────────────────────────────────────────────────┐
│                    Final Output                              │
├─────────────────────────────────────────────────────────────┤
│  • Photorealistic rendering with architectural accuracy      │
│  • Context summary (dimensions, materials, processing)       │
│  • Cached context for iterative refinement                   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Architectural Context Engine (`architectural_context_engine.py`)

Core system for extracting and managing architectural context.

**Key Classes:**
- `ArchitecturalContext`: Complete context dataclass
- `DimensionInfo`: Spatial dimensions (width, length, height, area)
- `MaterialSpec`: Material specifications (type, finish, color)
- `SpatialContext`: Space relationships and adjacencies
- `ArchitecturalContextExtractor`: Extraction from files and PDFs
- `ContextAwareRenderingPipeline`: Integration with rendering

**Features:**
- Filename pattern recognition (project, space, style)
- PDF text extraction and parsing
- Dimension extraction (12' x 14', 10' ceiling height)
- Material specification parsing
- Design intent identification
- JSON caching for reuse

### 2. Context-Aware Pro Pipeline (`context_aware_pro_pipeline.py`)

Professional rendering pipeline with architectural intelligence.

**Processing Stages:**
1. **Context Extraction**: Parse image filename + PDF documents
2. **Prompt Enhancement**: Enrich base prompt with context
3. **Depth Processing**: Space-type-aware depth estimation
4. **Material Response**: Specification-based surface enhancement
5. **AI Enhancement**: Context-informed Stable Diffusion refinement

**Space-Type Mapping:**
- Kitchen → Interior preset, stone/metal/glass materials
- Living Room → Interior preset, wood/fabric/glass materials
- Pool Area → Exterior preset, stone/water materials
- Automatic config selection based on space type

## Usage

### Quick Start

```bash
# Basic usage (image only)
python context_aware_pro_pipeline.py input_images/kitchen.jpg

# With architectural PDFs
python context_aware_pro_pipeline.py \
    input_images/kitchen.jpg \
    --pdf floor_plans.pdf \
    --pdf elevations.pdf

# Custom prompt + 4x upscaling
python context_aware_pro_pipeline.py \
    input_images/kitchen.jpg \
    --pdf floor_plans.pdf \
    --prompt "magazine-quality luxury kitchen" \
    --upscale-4x

# Disable specific stages
python context_aware_pro_pipeline.py \
    input_images/kitchen.jpg \
    --no-depth \
    --no-material
```

### Programmatic Usage

```python
from pathlib import Path
from context_aware_pro_pipeline import ContextAwareProPipeline

# Initialize pipeline
pipeline = ContextAwareProPipeline(
    output_dir=Path("output_context_aware_pro")
)

# Process with PDFs
outputs = pipeline.process_image(
    image_path=Path("input_images/kitchen.jpg"),
    pdf_documents=[
        Path("docs/floor_plans.pdf"),
        Path("docs/elevations.pdf")
    ],
    enable_depth=True,
    enable_material_response=True,
    enable_ai_enhancement=True,
    upscale_4x=False
)

# Access outputs
print(f"Depth output: {outputs['depth']}")
print(f"Material output: {outputs['material']}")
print(f"AI enhanced: {outputs['ai_enhanced']}")
```

### Context Extraction Only

```python
from architectural_context_engine import ArchitecturalContextExtractor
from pathlib import Path

extractor = ArchitecturalContextExtractor()

# Extract from filename
context = extractor.extract_from_filename(
    "Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg"
)

print(f"Project: {context.project_name}")
print(f"Space: {context.space_type}")
print(f"Style: {', '.join(context.style_notes)}")

# Extract from PDF
pdf_context = extractor.extract_from_pdf(
    Path("docs/floor_plans.pdf")
)

print(f"Dimensions: {pdf_context.dimensions}")
print(f"Materials: {len(pdf_context.materials)}")
```

### Enhanced Prompt Generation

```python
from architectural_context_engine import ContextAwareRenderingPipeline
from pathlib import Path

pipeline = ContextAwareRenderingPipeline()

# Generate enhanced prompt
enhanced_prompt = pipeline.enhance_prompt(
    base_prompt="photorealistic architectural rendering",
    image_path=Path("input_images/kitchen.jpg"),
    pdf_documents=[Path("docs/floor_plans.pdf")]
)

print(enhanced_prompt)
# Output: "photorealistic architectural rendering, kitchen, 
#          12' x 14' space, 10' ceiling height, materials: white oak flooring, 
#          quartz counters, brass hardware, open concept, natural light, 
#          bright airy atmosphere"
```

## File Naming Conventions

The system recognizes structured filenames to extract context automatically:

### Project Identification
- `750Picacho` → "750 Picacho Lane"
- `MBAR` → "MBAR Project"
- Pattern: `{ProjectCode}_{SpaceType}_{Style}`

### Space Types (Auto-Detection)
- `kitchen` → Kitchen
- `greatroom`, `living` → Living Room
- `master`, `bedroom` → Bedroom
- `pool` → Pool Area
- `courtyard` → Courtyard
- `exterior` → Exterior

### Style Hints
- `bright` → "bright, airy atmosphere"
- `moody` → "dramatic, moody lighting"
- `twilight` → "twilight ambiance"
- `golden` → "golden hour lighting"
- `coastal` → "coastal contemporary style"

### Examples
```
Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg
  → Project: 750 Picacho Lane
  → Space: Kitchen
  → Style: bright, airy atmosphere

Coastal_Estate_Greatroom_Twilight.tiff
  → Project: Coastal Estate
  → Space: Living Room
  → Style: twilight ambiance, coastal contemporary style
```

## PDF Document Processing

### Supported Documents
- Floor plans (dimensions, room names)
- Elevations (ceiling heights, window counts)
- Material schedules (finishes, manufacturers)
- Specification sheets (design intent, style notes)

### Extracted Information

**Dimensions:**
```
Pattern: "12' x 14'" or "12'-0" x 14'-6""
Result: DimensionInfo(width=12.0, length=14.0, area=168.0)

Pattern: "ceiling height: 10'-0""
Result: DimensionInfo(ceiling_height=10.0)
```

**Materials:**
```
Pattern: "white oak flooring"
Result: MaterialSpec(material_type="wood", location="floor", finish="white oak")

Pattern: "quartz countertop"
Result: MaterialSpec(material_type="stone", location="counter", finish="quartz")
```

**Design Intent:**
```
Keywords: "open concept", "natural light", "indoor-outdoor"
Result: design_intent = ["open concept", "natural light", "indoor-outdoor"]
```

### PDF Requirements

```bash
# Install PDF parsing library
pip install PyPDF2

# Or use pikepdf (alternative)
pip install pikepdf
```

## Context Caching

Contexts are cached as JSON files for rapid reuse:

```json
{
  "project_name": "750 Picacho Lane",
  "space_type": "kitchen",
  "dimensions": {
    "width": 12.0,
    "length": 14.0,
    "ceiling_height": 10.0,
    "area": 168.0,
    "unit": "feet"
  },
  "materials": [
    {
      "material_type": "wood",
      "location": "floor",
      "finish": "white oak"
    }
  ],
  "design_intent": ["open concept", "natural light"],
  "style_notes": ["bright airy atmosphere"]
}
```

**Cache Location:** `extracted_context/{image_stem}_context.json`

**Benefits:**
- Instant loading on subsequent runs
- Manual editing for corrections
- Version control friendly
- Shareable across team

## Integration with Existing Pipelines

### Depth Pipeline Integration

```python
# Context determines depth config
config_map = {
    SpaceType.KITCHEN: "config/interior_preset.yaml",
    SpaceType.EXTERIOR: "config/exterior_preset.yaml",
}

config = config_map.get(context.space_type, "config/interior_preset.yaml")
pipeline = ArchitecturalDepthPipeline.from_config(config)
```

### Material Response Integration

```python
# Context materials → surface types
surface_map = {
    "wood": SurfaceType.WOOD,
    "stone": SurfaceType.STONE,
    "metal": SurfaceType.METAL,
}

surfaces = [surface_map[mat.material_type] for mat in context.materials]
enhanced = mr.enhance(image, surfaces=surfaces, strength=0.75)
```

### AI Enhancement Integration

```python
# Context-enhanced prompts
base_prompt = "photorealistic architectural rendering"
enhanced_prompt = context.to_enhanced_prompt(base_prompt)

# Use in Stable Diffusion pipeline
result = sd_pipeline(
    prompt=enhanced_prompt,
    image=input_image,
    strength=0.75
)
```

## Advanced Features

### Custom Context Creation

```python
from architectural_context_engine import (
    ArchitecturalContext,
    DimensionInfo,
    MaterialSpec,
    SpaceType
)

context = ArchitecturalContext(
    project_name="Custom Project",
    space_type=SpaceType.KITCHEN,
    dimensions=DimensionInfo(
        width=12.0,
        length=14.0,
        ceiling_height=10.0
    ),
    materials=[
        MaterialSpec(
            material_type="wood",
            location="floor",
            finish="white oak"
        )
    ],
    design_intent=["modern", "minimalist"],
    style_notes=["bright", "airy"]
)

# Save for reuse
context.save(Path("extracted_context/custom_context.json"))
```

### Batch Processing with Context

```python
from pathlib import Path

pipeline = ContextAwareProPipeline()

# Process all images in directory
input_dir = Path("input_images")
pdf_docs = [
    Path("docs/floor_plans.pdf"),
    Path("docs/elevations.pdf")
]

for image_path in input_dir.glob("*.jpg"):
    outputs = pipeline.process_image(
        image_path=image_path,
        pdf_documents=pdf_docs,
        enable_depth=True,
        enable_material_response=True,
        enable_ai_enhancement=True
    )
    print(f"Processed: {image_path.name}")
```

## Troubleshooting

### No PDF Parsing Available
```
Warning: PyPDF2 not installed
```
**Solution:** `pip install PyPDF2`

### Context Not Detected from Filename
```
Project: Unknown Project
Space: None
```
**Solution:** Rename file to match conventions or create custom context

### PDFs Not Providing Context
**Possible causes:**
- PDF is image-based (no text layer)
- Text format not recognized
- OCR required

**Solution:** Use OCR or create context manually

### Material/Depth Processing Skipped
```
Warning: Material processing failed
```
**Check:**
- Dependencies installed (`pip install -r requirements.txt`)
- Models downloaded (`python install_models_auto.py`)
- Sufficient memory/VRAM

## Performance

### Context Extraction
- Filename parsing: < 1ms
- PDF parsing (5 pages): 100-500ms
- Context caching: 10-20ms read, 20-50ms write

### Pipeline Processing
- Context-aware depth: +5-10% overhead vs. standard depth
- Context-aware material: +2-5% overhead vs. standard material
- Overall impact: < 10% slower, significantly higher quality

## Best Practices

1. **Filename Conventions**: Use structured naming for automatic context
2. **PDF Organization**: Keep architectural docs in project folders
3. **Context Review**: Check cached JSON files for accuracy
4. **Iterative Refinement**: Run multiple times with same PDFs (uses cache)
5. **Manual Corrections**: Edit JSON cache for fine-tuning
6. **Version Control**: Commit context caches with code

## Future Enhancements

- [ ] OCR support for image-based PDFs
- [ ] DWG/DXF CAD file parsing
- [ ] 3D model integration (OBJ, FBX)
- [ ] BIM integration (Revit, IFC)
- [ ] Natural language design brief parsing
- [ ] Multi-view consistency (floor plan → render matching)
- [ ] Automated material library matching
- [ ] Photogrammetry integration

## Examples

See `examples/` directory for complete workflows:
- `example_kitchen_context.py` - Kitchen rendering with floor plans
- `example_batch_context.py` - Batch processing with shared PDFs
- `example_custom_context.py` - Manual context creation

## Support

For issues or questions:
1. Check documentation: `docs/ARCHITECTURAL_CONTEXT_INTEGRATION.md`
2. Review examples: `examples/`
3. Check logs: Pipeline provides detailed logging
4. Create issue: Include context JSON and processing summary
