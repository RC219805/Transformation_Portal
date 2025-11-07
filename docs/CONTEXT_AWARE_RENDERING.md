# Context-Aware Rendering System
**Transformation Portal - Intelligent Architectural Visualization**

## Overview

The Context-Aware Rendering System represents a paradigm shift in architectural visualization: **every processing decision is informed by actual architectural intelligence extracted from construction documents**.

Instead of generic "one-size-fits-all" processing, this system:
- **Reads** floor plans, elevations, and specifications
- **Understands** room types, dimensions, materials, and design intent
- **Adapts** depth processing, material enhancement, and color grading accordingly
- **Delivers** contextually optimized renderings that respect the architect's vision

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CONSTRUCTION DOCUMENTS                      │
│         (Floor Plans, Elevations, Specifications)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            ARCHITECTURAL CONTEXT EXTRACTOR                   │
│  • Extracts room types, dimensions, materials               │
│  • Identifies design style and intent                       │
│  • Captures project metadata                                │
│  • Processes embedded images (plans, sections)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              [ProjectContext]
           (Structured Intelligence)
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          CONTEXT-AWARE RENDERING PIPELINE                    │
│  • Derives optimal strategy for each rendering              │
│  • Generates processing configurations                      │
│  • Adapts to room type and materials                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PREMIUM PROCESSING STAGES                       │
│                                                              │
│  Stage 1: Depth-Aware Processing                            │
│           └─ Depth Anything V2 + Context                    │
│                                                              │
│  Stage 2: Material Response                                 │
│           └─ Surface-aware enhancement                      │
│                                                              │
│  Stage 3: Color Grading                                     │
│           └─ Context-driven LUT selection                   │
│                                                              │
│  Stage 4: Ultimate Enhancement (optional)                   │
│           └─ AI upscaling + refinement                      │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Architectural Context Extractor
**File**: `architectural_context_extractor.py`

Extracts architectural intelligence from PDFs:

**Capabilities:**
- **Room Detection**: Identifies kitchens, bathrooms, bedrooms, living spaces, outdoor areas
- **Dimension Extraction**: Parses room dimensions (width x depth)
- **Material Analysis**: Detects wood, stone, metal, glass, concrete, fabrics
- **Style Inference**: Determines design style (Modern, Traditional, Mediterranean, etc.)
- **Image Extraction**: Captures embedded floor plans and elevations
- **Metadata Capture**: Project name, number, address, architect

**Usage:**
```bash
# Extract context from architectural PDF
python scripts/architectural_context_extractor.py \
    "path/to/plans.pdf" \
    --output extracted_context \
    --verbose

# Outputs:
# - extracted_context/{project}_context.json
# - extracted_context/{project}_images/page*.png
```

**Output Structure:**
```json
{
  "project_name": "750 PICACHO LANE",
  "project_number": "24098.00",
  "design_style": "Modern Luxury",
  "materials_palette": ["wood", "stone", "glass", "metal"],
  "floors": ["1st Floor", "2nd Floor"],
  "rooms": {
    "kitchen_0": {
      "name": "Kitchen",
      "dimensions": [18.0, 14.5],
      "floor_level": "1st Floor",
      "ceiling_height": 10.0,
      "materials": ["stone", "metal", "glass"],
      "features": ["island", "wine storage"]
    }
  }
}
```

### 2. Context-Aware Rendering Pipeline
**File**: `context_aware_rendering.py`

Derives optimal rendering strategies from architectural context:

**Strategy Derivation:**
- **Room Type Identification**: Analyzes filename to determine space type
- **Material Prioritization**: Focuses enhancement on primary materials
- **Lighting Adaptation**: Adjusts tone mapping for room function
- **Depth Emphasis**: Configures foreground/background processing
- **Color Temperature**: Sets warm/neutral/cool based on design style

**Room-Specific Strategies:**

| Room Type | Materials | Lighting | Depth | Temperature | Enhancement |
|-----------|-----------|----------|-------|-------------|-------------|
| Kitchen | Metal, Stone, Wood, Glass | Bright | Balanced | Neutral | 0.75 |
| Bathroom | Stone, Glass, Metal, Tile | Soft | Balanced | Neutral | 0.70 |
| Bedroom | Wood, Fabric, Leather | Soft | Atmospheric | Warm | 0.60 |
| Living | Wood, Fabric, Stone | Ambient | Balanced | Warm | 0.70 |
| Outdoor | Stone, Concrete, Wood | Natural | Atmospheric | Neutral | 0.80 |

**Usage:**
```bash
# Generate context-aware strategy
python scripts/context_aware_rendering.py \
    "input_images/Kitchen_Rendering.tiff" \
    --context "extracted_context/project_context.json" \
    --output output_strategies
```

### 3. Premium Context-Aware Pipeline
**File**: `premium_context_pipeline.py`

Full end-to-end processing with context intelligence:

**Quality Tiers:**
- **Standard**: Depth + Material + Color
- **Premium**: Standard + Advanced tone mapping
- **Ultimate**: Premium + AI upscaling (4K)

**Processing Stages:**
1. **Strategy Derivation**: Analyze context and determine optimal approach
2. **Depth Processing**: Context-aware depth estimation and zone-based enhancement
3. **Material Response**: Surface-specific enhancement respecting material types
4. **Color Grading**: Context-driven LUT selection and application
5. **Ultimate Enhancement**: AI-powered upscaling and refinement (Ultimate only)

**Usage:**
```bash
# Process with premium quality
python scripts/premium_context_pipeline.py \
    "input_images/Kitchen_Rendering.tiff" \
    --context "path/to/plans.pdf" \
    --quality premium \
    --output output_premium

# Quality levels:
# standard  - Fast processing, good quality
# premium   - Balanced quality/speed (recommended)
# ultimate  - Maximum quality, slower (4K upscale)
```

## Workflow

### Complete End-to-End Workflow

```bash
# Step 1: Extract architectural context from plans
python scripts/architectural_context_extractor.py \
    "documents/750_Picacho_Plans.pdf" \
    --output extracted_context \
    --verbose

# Step 2: Process renderings with context intelligence
python scripts/premium_context_pipeline.py \
    "input_images/Kitchen_Rendering.tiff" \
    --context "extracted_context/750_Picacho_Plans_context.json" \
    --quality premium \
    --output output_premium

# Step 3: Review outputs
# - output_premium/Kitchen_Rendering_strategy.json (strategy config)
# - output_premium/Kitchen_Rendering_depth.tiff (depth processed)
# - output_premium/Kitchen_Rendering_material.tiff (material enhanced)
# - output_premium/Kitchen_Rendering_graded.tiff (final premium output)
```

### Batch Processing Multiple Rooms

```bash
# Process all renderings for a project
for render in input_images/750Picacho_*.tiff; do
    python scripts/premium_context_pipeline.py \
        "$render" \
        --context "extracted_context/750_Picacho_context.json" \
        --quality premium \
        --output "output_750Picacho"
done
```

## Intelligence Features

### Context-Driven Adaptations

1. **Material-Aware Enhancement**
   - Kitchen with stone countertops → Enhanced specular highlights
   - Bedroom with wood floors → Warmer tone, preserved grain texture
   - Bathroom with glass shower → Controlled reflections, clarity boost

2. **Room-Function Optimization**
   - Kitchens: Bright, balanced depth, crisp details
   - Bedrooms: Soft lighting, atmospheric depth, warm tones
   - Outdoor spaces: Natural lighting, enhanced atmospheric perspective

3. **Design Style Consistency**
   - Modern/Contemporary → Neutral temperature, higher enhancement
   - Traditional → Warm temperature, subtle enhancement
   - Industrial → Cooler temperature, material texture emphasis

4. **Dimension-Aware Processing**
   - Large rooms → Enhanced atmospheric perspective
   - Small rooms → Balanced depth, avoid over-processing
   - Open-plan spaces → Careful zone transitions

### Quality Metrics

The system tracks quality improvements:
- **Context Match Score**: How well processing aligns with architectural intent
- **Material Fidelity**: Accuracy of surface rendering
- **Spatial Coherence**: Depth and dimension representation
- **Style Consistency**: Adherence to design language

## Technical Details

### Dependencies

**Core:**
- `PyMuPDF` (fitz) - PDF processing
- `Pillow` - Image handling
- `NumPy` - Numerical operations

**Pipeline Integration:**
- Depth Anything V2 - Monocular depth estimation
- Material Response - Physics-based surface enhancement
- Luxury TIFF Processor - Color grading and LUT application

### Performance

**Context Extraction:**
- Small PDF (< 50 pages): 5-10 seconds
- Large PDF (200+ pages): 30-60 seconds
- Image extraction: ~100-200ms per embedded image

**Strategy Derivation:**
- Instant (< 100ms)

**Full Premium Pipeline:**
- Standard: 30-45 seconds per image
- Premium: 60-90 seconds per image
- Ultimate: 3-5 minutes per image (includes 4K upscaling)

### File Formats

**Input:**
- PDFs: Construction documents, specifications
- Images: TIFF, PNG, JPEG (16-bit TIFF recommended)

**Output:**
- Context: JSON (structured data)
- Strategies: JSON (processing configurations)
- Images: 16-bit TIFF (preserves quality)

## Examples

### Example 1: Luxury Kitchen Rendering

**Input**: `Kitchen_Rendering.tiff` from 750 Picacho Lane project

**Context Extracted:**
- Room: Kitchen, 18' x 14.5', 10' ceiling
- Materials: Quartzite counters, stainless appliances, white oak floors
- Style: Modern Transitional

**Strategy Derived:**
- Primary materials: Stone (quartzite), Metal (stainless), Wood (oak), Glass
- Lighting: Bright (kitchen function)
- Depth: Balanced (workable space)
- Temperature: Neutral (modern aesthetic)
- Enhancement: 0.75 (high-end finishes)
- LUT: `signature_estate`

**Processing:**
1. Depth zones emphasize island and cabinetry detail
2. Material Response enhances stone specular, metal reflections, wood grain
3. Color grading maintains neutral temperature with subtle warmth
4. Final output: Photorealistic kitchen with accurate material rendering

### Example 2: Primary Bathroom

**Input**: `Bathroom_Rendering.tiff`

**Context Extracted:**
- Room: Primary Bathroom, 12' x 16', spa shower
- Materials: Marble tile, frameless glass, polished nickel
- Style: Contemporary Luxury

**Strategy Derived:**
- Primary materials: Stone (marble), Glass (frameless), Metal (nickel)
- Lighting: Soft (spa ambiance)
- Depth: Balanced
- Temperature: Neutral (clean aesthetic)
- Enhancement: 0.70
- LUT: `serene_spa`

**Result**: Spa-like rendering with accurate marble veining, controlled glass reflections, subtle metallic accents

## Benefits

### For Architects & Designers
- **Design Intent Preserved**: Processing respects architectural vision
- **Material Accuracy**: Realistic representation of specified finishes
- **Style Consistency**: Automated adherence to design language
- **Time Savings**: No manual tweaking per rendering

### For Visualization Studios
- **Intelligent Automation**: Context-aware processing reduces manual work
- **Consistent Quality**: Repeatable results across project
- **Scalability**: Batch process entire projects with confidence
- **Client Confidence**: Provenance from actual construction documents

### For Real Estate Marketing
- **Authentic Representation**: Renderings match architectural reality
- **Material Storytelling**: Accurate depiction of luxury finishes
- **Contextual Coherence**: All renderings share visual language
- **Buyer Trust**: Transparent connection to architectural plans

## Future Enhancements

### Planned Features
1. **3D Model Integration**: Import BIM/Revit models for spatial understanding
2. **Lighting Analysis**: Time-of-day and seasonal lighting simulation
3. **Material Library**: Expanded physics-based material database
4. **Style Transfer**: Learn from architect's previous projects
5. **Quality Prediction**: Pre-processing quality estimation
6. **Interactive Refinement**: Real-time strategy adjustment

### Research Directions
- **Semantic Segmentation**: ML-based room and object identification
- **View Classification**: Automatic perspective and viewpoint analysis
- **Material Classification**: Neural network material recognition
- **Style Embedding**: Deep learning design style understanding

## Troubleshooting

### Common Issues

**"Could not identify room type"**
- Ensure filename includes room name (e.g., `Kitchen`, `Bedroom`)
- Check that PDF contains recognizable floor plans
- Manually specify room type via CLI (future feature)

**"Depth processing failed"**
- Verify `depth_pipeline/` is available
- Check that input image is valid TIFF/PNG
- Ensure sufficient memory (8GB+ recommended)

**"Material Response skipped"**
- Verify `material_response.py` exists in root
- Check that strategy includes `material_config`

**"No materials detected in PDF"**
- PDF may be image-only (no OCR text)
- Run OCR on PDF first: `ocrmypdf input.pdf output.pdf`
- Manually specify materials in strategy

## Support

**Documentation**: `/docs/context_aware_rendering/`
**Examples**: `/examples/context_aware/`
**Issue Tracker**: GitHub Issues

---

**Transformation Portal** - Where architectural intelligence meets AI-powered visualization.
