# Architectural Workflow Guide: Montecito Estate Rendering

Production workflow guide for luxury real estate rendering in Santa Barbara's unique coastal micro-climate.

## Table of Contents

- [Introduction](#introduction)
- [The Montecito Micro-Climate](#the-montecito-micro-climate)
- [Physics-Based Approach](#physics-based-approach)
- [Signature Atmospheric Conditions](#signature-atmospheric-conditions)
  - [Sundowner Effect](#sundowner-effect-high-clarity)
  - [Marine Layer Effect](#marine-layer-june-gloom)
- [Complete Rendering Workflow](#complete-rendering-workflow)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Introduction

This guide demonstrates the **physics-based atmospheric rendering workflow** specifically optimized for Montecito and Santa Barbara coastal estates. Unlike traditional sky replacement tools that naively composite any sky onto any photo, Transformation Portal uses **shadow analysis**, **atmospheric physics**, and **auto-correction guardrails** to ensure optically-consistent results.

**Who This Is For**:
- Architectural visualization artists
- Real estate photographers
- Marketing agencies for luxury properties
- 3D rendering professionals

**What You'll Learn**:
- How to leverage Montecito's unique atmospheric conditions
- Physics-based sky replacement with shadow consistency
- Handling the "Sundowner" effect (exceptional clarity)
- Simulating the "Marine Layer" (June Gloom fog)
- Complete production workflow from raw render to final delivery

---

## The Montecito Micro-Climate

### Geographic Context

**Location**: 34.4°N, 119.7°W (Santa Barbara County, California)
**Elevation**: Sea level to 500ft for most estates
**Climate**: Mediterranean with coastal marine influence
**Unique Features**: Sundowner winds, Marine layer fog, exceptional air clarity

### Why This Matters for Rendering

Traditional rendering tools use generic atmospheric models. Montecito's micro-climate has **distinctive optical characteristics**:

1. **Sundowner Winds** (Fall): Hot offshore winds create turbidity ~1.3 (exceptionally clear)
2. **Marine Layer** (Spring/Summer): Low fog at ~500ft elevation creates diffuse, cool-toned lighting
3. **Coastal Aerosols**: Salt spray and moisture affect atmospheric scattering
4. **Sun Path**: 34.4°N latitude means specific seasonal sun angles

The Transformation Portal includes **pre-calibrated atmospheric parameters** for these conditions.

---

## Physics-Based Approach

### Traditional Sky Replacement (Naive)

```
1. Mask out old sky (Photoshop magic wand)
2. Paste new sky (any sky from internet)
3. Blend edges
4. Done ✓ (but physically wrong)
```

**Problems**:
- ❌ New sky sun position conflicts with existing shadows
- ❌ Atmospheric perspective doesn't match
- ❌ Lighting temperature inconsistent
- ❌ Depth cues broken

### Transformation Portal Approach (Physics-Informed)

```
1. Shadow Analysis: Detect dominant light direction from scene geometry
2. Consistency Check: Compare requested sky vs. measured shadows
3. Auto-Correction: If conflict detected, suggest physically-correct parameters
4. Atmospheric Integration: Apply depth-aware atmospheric scattering
5. Volumetric Blending: Not just edge blending, but 3D atmospheric mixing
6. Done ✓ (optically consistent)
```

**Advantages**:
- ✅ Shadow-consistent sky placement
- ✅ Physically-accurate atmospheric perspective
- ✅ Correct lighting temperature and color
- ✅ Depth-aware integration

---

## Signature Atmospheric Conditions

### Sundowner Effect (High Clarity)

**What It Is**:
The Sundowner is a meteorological phenomenon unique to Santa Barbara where hot, dry offshore winds (Santa Ana-type) descend from the mountains, creating **exceptionally clear atmospheric conditions** with turbidity as low as 1.3 (pristine clarity).

**Visual Characteristics**:
- **Turbidity**: 1.3-1.5 (vs. typical 2.0-3.0)
- **Visibility**: 50+ miles (Channel Islands clearly visible)
- **Color**: Deep blue sky with minimal haze
- **Shadows**: Sharp, high-contrast
- **Temperature**: Warm, golden light
- **Time**: Late afternoon/evening (5-7 PM typical)

**When to Use**:
- Fall season (September-November)
- Late afternoon golden hour shots
- Showcase ocean/mountain views
- Emphasize architectural detail and shadows

**Workflow Example**:

```python
from transformation_portal.atmosphere import LocationPresets, SkyBlender, SkyGANGenerator
import cv2
import numpy as np

# Load your architectural render
image = cv2.imread('estate_exterior.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get Sundowner atmospheric parameters
presets = LocationPresets()

# Request Sundowner golden hour (5:30 PM in Fall)
sky_params = presets.get_sky_parameters(
    location="montecito",
    time_of_day=17.5,      # 5:30 PM (decimal hours)
    season="fall",          # Sundowner season
    condition="sundowner"   # Exceptional clarity
)

atmo_params = presets.get_atmospheric_parameters(
    location="montecito",
    condition="sundowner"
)

# Verify parameters
print(f"Sun Azimuth: {sky_params.sun_azimuth:.1f}° (expect ~220-240° for SW)")
print(f"Sun Elevation: {sky_params.sun_elevation:.1f}° (expect ~15-30° for golden hour)")
print(f"Turbidity: {atmo_params.turbidity:.1f} (expect ~1.3 for Sundowner)")
print(f"Haze Density: {atmo_params.haze_density:.2f} (expect ~0.05-0.10)")

# Initialize physics engine
blender = SkyBlender()

# Execute smart render with physics guardrails
result, suggestion = blender.smart_render(
    source_image=image,
    sky_params=sky_params,
    atmo_params=atmo_params,
    auto_correct=True,     # Enable shadow analysis
    strict_physics=False   # Suggest corrections, don't reject
)

# Check for physics violations
if suggestion.confidence < 0.8:
    print("\n⚠️  PHYSICS ALERT:")
    print(f"   Requested Sun: {suggestion.original_request_azimuth:.1f}°")
    print(f"   Measured Shadows: {suggestion.measured_source_azimuth:.1f}°")
    print(f"   Recommendation: {suggestion.message}")
    print(f"\n   Proceeding with CORRECTED parameters...")

    # Use corrected parameters for final render
    result, _ = blender.smart_render(
        source_image=image,
        sky_params=suggestion.suggested_params,  # Use corrected!
        atmo_params=atmo_params,
        auto_correct=False  # Already corrected
    )

# Save result
cv2.imwrite('output/sundowner_estate.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("✅ Sundowner render complete!")
```

**Expected Output**:
- Deep blue sky (minimal haze)
- Sharp mountain/island silhouettes
- Warm golden light on architecture
- High contrast shadows
- Visible fine atmospheric detail

**Physics Guardrails Example**:

If you accidentally request a **West-facing sun** (azimuth 270°) but your scene has **East-facing shadows** (measured azimuth 90°):

```
⚠️  PHYSICS ALERT:
   Requested Sun: 270.0° (West)
   Measured Shadows: 95.0° (East)
   Confidence: 0.85
   Recommendation: "Sun position conflicts with existing shadows.
                    Suggested azimuth: 95.0° for shadow consistency."
```

The system **auto-corrects** the parameters to avoid the optical impossibility.

---

### Marine Layer Effect (June Gloom)

**What It Is**:
The Marine Layer is a persistent fog/cloud layer at low altitude (~500ft) common in late spring/early summer. It creates **diffuse, cool-toned lighting** with soft shadows.

**Visual Characteristics**:
- **Fog Altitude**: ~500ft (150m) above sea level
- **Visibility**: Reduced, 1-5 miles
- **Color**: Cool blue-gray tones
- **Shadows**: Soft, low-contrast
- **Light Quality**: Diffuse, overcast-like
- **Temperature**: Cool, muted light
- **Time**: Morning through early afternoon typical

**When to Use**:
- Spring/Summer season (May-July)
- Overcast/moody aesthetic
- Emphasize softness and serenity
- Properties with ocean views showing coastal atmosphere

**Workflow Example**:

```python
from transformation_portal.atmosphere import LocationPresets, SkyBlender, AtmosphericModel
import cv2

# Load render
image = cv2.imread('coastal_estate.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get Marine Layer parameters
presets = LocationPresets()

sky_params = presets.get_sky_parameters(
    location="montecito",
    time_of_day=10.0,        # Mid-morning
    season="summer",          # Marine layer season
    condition="marine_layer"  # Foggy conditions
)

atmo_params = presets.get_atmospheric_parameters(
    location="montecito",
    condition="marine_layer"
)

# Verify parameters
print(f"Turbidity: {atmo_params.turbidity:.1f} (expect ~3.0-5.0 for fog)")
print(f"Haze Density: {atmo_params.haze_density:.2f} (expect ~0.4-0.7)")

# Marine layer fog settings
if hasattr(atmo_params, 'marine_layer'):
    ml = atmo_params.marine_layer
    print(f"Fog Altitude: {ml.altitude_m:.0f}m (~500ft)")
    print(f"Fog Thickness: {ml.thickness_m:.0f}m")
    print(f"Fog Density: {ml.density:.2f}")

# Execute render
blender = SkyBlender()
result, suggestion = blender.smart_render(
    source_image=image,
    sky_params=sky_params,
    atmo_params=atmo_params,
    auto_correct=True
)

# Apply cool color grade for marine layer aesthetic
from transformation_portal.color_grading import apply_lut
result = apply_lut(result, "assets/luts/location_aesthetic/Coastal_Cool.cube", strength=0.7)

# Save
cv2.imwrite('output/marine_layer_estate.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("✅ Marine layer render complete!")
```

**Expected Output**:
- Gray-blue overcast sky
- Soft diffuse lighting
- Cool color temperature
- Atmospheric depth (fog layers visible)
- Muted shadows

---

## Complete Rendering Workflow

### End-to-End Production Pipeline

**Stage 1: Raw Render Preparation**
```bash
# Assume you have raw 3D renders from Blender/3ds Max/V-Ray
# Files: bedroom_raw.exr, kitchen_raw.exr, exterior_raw.exr

# Create processing directory
mkdir -p workflow/01_raw
mkdir -p workflow/02_depth
mkdir -p workflow/03_sky
mkdir -p workflow/04_material
mkdir -p workflow/05_color
mkdir -p workflow/06_final

# Copy raw renders
cp renders/*.exr workflow/01_raw/
```

---

**Stage 2: Depth Estimation**
```bash
# Generate depth maps for all renders
python run_depth_estimation.py \
  --input-dir workflow/01_raw/ \
  --output-dir workflow/02_depth/ \
  --model depth_anything_v2_large \
  --use-coreml  # Apple Silicon acceleration
```

**Output**: Depth maps for shadow analysis and atmospheric processing

---

**Stage 3: Sky Replacement (Physics-Based)**
```python
# Script: workflow/03_process_sky.py
from transformation_portal.atmosphere import LocationPresets, SkyBlender
import cv2
from pathlib import Path

presets = LocationPresets()
blender = SkyBlender()

# Process each exterior render
for image_path in Path("workflow/01_raw").glob("exterior_*.jpg"):
    print(f"\n🏠 Processing: {image_path.name}")

    # Load
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sundowner golden hour (adapt as needed)
    sky_params = presets.get_sky_parameters(
        location="montecito",
        time_of_day=17.5,
        season="fall",
        condition="sundowner"
    )

    atmo_params = presets.get_atmospheric_parameters(
        location="montecito",
        condition="sundowner"
    )

    # Smart render with physics
    result, suggestion = blender.smart_render(
        source_image=image,
        sky_params=sky_params,
        atmo_params=atmo_params,
        auto_correct=True,
        strict_physics=False
    )

    # Handle corrections
    if suggestion.confidence < 0.8:
        print(f"   ⚠️  Auto-corrected: {suggestion.message}")

    # Save
    output_path = Path("workflow/03_sky") / image_path.name
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved: {output_path}")
```

**Run**:
```bash
python workflow/03_process_sky.py
```

---

**Stage 4: Material Response Enhancement**
```bash
# Apply physics-based surface enhancement
python material_response.py \
  --input-dir workflow/03_sky/ \
  --output-dir workflow/04_material/ \
  --surfaces wood metal glass stone \
  --strength 0.75 \
  --preserve-highlights \
  --batch-size 16
```

**Output**: Enhanced material rendering with realistic surface properties

---

**Stage 5: Color Grading**
```bash
# Apply LUT for film emulation
python luxury_tiff_batch_processor.py \
  --input-dir workflow/04_material/ \
  --output-dir workflow/05_color/ \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --lut-strength 0.7 \
  --exposure 0.1 \
  --contrast 1.08 \
  --saturation 1.05 \
  --clarity 0.15
```

---

**Stage 6: AI Enhancement (Optional)**
```bash
# Final AI refinement with ControlNet
python lux_render_pipeline.py \
  --input-dir workflow/05_color/ \
  --output-dir workflow/06_final/ \
  --preset luxury_estate \
  --controlnet-strength 0.7 \
  --upscale 4x \
  --brand-overlay assets/brand/lantern_logo/logo.png
```

---

**Stage 7: Delivery Export**
```bash
# Export final deliverables
# 16-bit TIFF masters
cp workflow/06_final/*.tiff deliverables/masters/

# Web-optimized JPEGs
mogrify -path deliverables/web/ -format jpg -quality 90 -resize 2560x workflow/06_final/*.tiff

# Thumbnails
mogrify -path deliverables/thumbnails/ -format jpg -quality 85 -resize 800x workflow/06_final/*.tiff
```

---

## Real-World Examples

### Example 1: Coastal Estate at Golden Hour

**Scenario**: 4-bedroom oceanfront estate, need hero shots for marketing.

**Requirements**:
- Sundowner atmospheric clarity
- Golden hour lighting (5:30 PM)
- Ocean and Channel Islands visible
- Warm, inviting aesthetic

**Workflow**:
```bash
# 1. Depth estimation
python run_depth_estimation.py \
  --input estate_exterior.jpg \
  --output depth/ \
  --use-coreml

# 2. Sky replacement with Sundowner
python -c "
from transformation_portal.atmosphere import LocationPresets, SkyBlender
import cv2

presets = LocationPresets()
blender = SkyBlender()

image = cv2.imread('estate_exterior.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sky_params = presets.get_sky_parameters('montecito', 17.5, 'fall', 'sundowner')
atmo_params = presets.get_atmospheric_parameters('montecito', 'sundowner')

result, _ = blender.smart_render(image, sky_params, atmo_params, auto_correct=True)
cv2.imwrite('sky_replaced.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
"

# 3. Material enhancement
python material_response.py \
  --input sky_replaced.jpg \
  --output material/ \
  --surfaces wood metal glass stone \
  --strength 0.75

# 4. Color grading
python luxury_tiff_batch_processor.py \
  --input material/sky_replaced.jpg \
  --output final/ \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --exposure 0.15 \
  --contrast 1.08

# 5. AI upscale
python lux_render_pipeline.py \
  --input final/sky_replaced.jpg \
  --output deliverables/ \
  --upscale 4x \
  --brand-overlay logo.png
```

**Result**: Hero shot with physically-accurate Sundowner atmosphere, 4x upscaled to 8K.

---

### Example 2: Modern Interior with June Gloom

**Scenario**: Contemporary bedroom interior, need soft, serene mood.

**Requirements**:
- Marine layer atmosphere
- Soft diffuse lighting
- Cool color palette
- Minimize harsh shadows

**Workflow**:
```python
# Script: process_marine_layer_interior.py
from transformation_portal.atmosphere import LocationPresets, SkyBlender
from transformation_portal.color_grading import apply_lut
import cv2

presets = LocationPresets()
blender = SkyBlender()

# Load
image = cv2.imread('bedroom_interior.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Marine layer parameters (overcast morning)
sky_params = presets.get_sky_parameters(
    location="montecito",
    time_of_day=10.0,
    season="summer",
    condition="marine_layer"
)

atmo_params = presets.get_atmospheric_parameters(
    location="montecito",
    condition="marine_layer"
)

# Render with soft lighting
result, _ = blender.smart_render(
    source_image=image,
    sky_params=sky_params,
    atmo_params=atmo_params,
    auto_correct=True
)

# Cool color grade
result = apply_lut(result, "assets/luts/location_aesthetic/Coastal_Cool.cube", strength=0.6)

# Save
cv2.imwrite('bedroom_marine_layer.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("✅ Marine layer interior complete!")
```

**Result**: Soft, serene interior with cool coastal aesthetic.

---

### Example 3: Batch Processing 50 Renders

**Scenario**: Full property shoot with 50 mixed interior/exterior renders.

**Requirements**:
- Batch process entire set
- Consistent aesthetic across all images
- Optimize for performance
- Preserve metadata

**Workflow**:
```bash
# Use unified recipe-based pipeline
cat > config/recipes/montecito_estate_batch.yaml << EOF
name: "Montecito Estate Batch"
description: "Complete processing for 50-image estate shoot"
version: "1.0"

stages:
  - depth_estimation
  - sky_replacement
  - material_response
  - color_grading

depth:
  model: "depth_anything_v2_large"
  use_coreml: true

sky_replacement:
  location: "montecito"
  condition: "sundowner"
  time_of_day: 17.5
  season: "fall"
  auto_correct: true

material_response:
  surfaces: [wood, metal, glass, stone, fabric]
  strength: 0.75

color_grading:
  lut: "assets/luts/film_emulation/Kodak_2393.cube"
  lut_strength: 0.7
  exposure: 0.1
  contrast: 1.08

output:
  format: "tiff"
  bit_depth: 16
  preserve_metadata: true
EOF

# Execute batch
python -m transformation_portal process \
  --input "raw_renders/*.jpg" \
  --recipe config/recipes/montecito_estate_batch.yaml \
  --output deliverables/ \
  --parallel \
  --log-level info
```

**Performance**: ~400-600 images/hour on M4 Max = ~5-8 minutes for 50 images

---

## Best Practices

### 1. Always Enable Auto-Correction

```python
# GOOD: Physics guardrails enabled
result, suggestion = blender.smart_render(
    ...,
    auto_correct=True  # ✅
)

# BAD: No physics checking
result = naive_sky_replace(image, new_sky)  # ❌ Can create impossible lighting
```

**Why**: Auto-correction detects shadow conflicts and prevents optically impossible renders.

---

### 2. Validate Atmospheric Parameters

```python
# Check that parameters match your intent
print(f"Sun Azimuth: {sky_params.sun_azimuth:.1f}°")
print(f"Sun Elevation: {sky_params.sun_elevation:.1f}°")
print(f"Turbidity: {atmo_params.turbidity:.1f}")

# Verify against expected values
assert 1.0 <= atmo_params.turbidity <= 2.0, "Sundowner should have low turbidity"
assert 15 <= sky_params.sun_elevation <= 35, "Golden hour should be low sun"
```

---

### 3. Preserve Confidence Scores

```python
result, suggestion = blender.smart_render(...)

# Log confidence for quality control
with open('render_log.txt', 'a') as f:
    f.write(f"{image_name}, confidence: {suggestion.confidence:.2f}\n")

# Alert on low confidence
if suggestion.confidence < 0.5:
    print(f"⚠️  LOW CONFIDENCE: {suggestion.message}")
    # Manual review recommended
```

**Confidence Score Interpretation**:
- **0.0**: Flat gray image (no scene lighting detected, no hallucination!)
- **0.0-0.5**: Weak/ambiguous shadow information
- **0.5-0.8**: Moderate confidence, minor correction suggested
- **0.8-1.0**: High confidence, shadows clearly detected

---

### 4. Use Seasonal Presets

```python
# Match season to shoot date
# Fall: Sundowner clarity
sky_params = presets.get_sky_parameters("montecito", 17.5, "fall", "sundowner")

# Summer: Marine layer
sky_params = presets.get_sky_parameters("montecito", 10.0, "summer", "marine_layer")

# Winter: Cool clear
sky_params = presets.get_sky_parameters("montecito", 15.0, "winter", "clear")

# Spring: Mild
sky_params = presets.get_sky_parameters("montecito", 16.0, "spring", "clear")
```

---

### 5. Batch Processing Strategy

```bash
# For large batches, use parallel processing
python material_response.py \
  --input-dir renders/ \
  --output-dir output/ \
  --parallel \
  --batch-size 32  # Tune based on available RAM

# Monitor memory usage
# M4 Max: batch_size 32-64 typical
# 16GB RAM: batch_size 8-16
# 8GB RAM: batch_size 1-4
```

---

## Troubleshooting

### Problem: "Physics Violation Detected" on Every Image

**Symptom**: All renders trigger auto-correction warnings.

**Likely Cause**: Your 3D renders have inconsistent sun position vs. requested sky.

**Solutions**:
1. **Re-render in 3D**: Match sun position in render to Montecito location
2. **Use Measured Parameters**: Apply the `suggested_params` from the auto-correction
3. **Accept Deviation**: Set `strict_physics=False` and log deviations for review

```python
# Option 2: Use auto-corrected parameters
_, suggestion = blender.smart_render(..., auto_correct=True)

# Re-render with corrected parameters
result, _ = blender.smart_render(
    source_image=image,
    sky_params=suggestion.suggested_params,  # Use these!
    atmo_params=atmo_params,
    auto_correct=False
)
```

---

### Problem: Sky Looks Too Hazy for Sundowner

**Symptom**: Sundowner renders have too much atmospheric haze.

**Cause**: Turbidity parameter too high.

**Solution**:
```python
# Manually override turbidity
atmo_params.turbidity = 1.3  # Pristine Sundowner clarity

# Or create custom parameters
from transformation_portal.atmosphere import AtmosphericParameters

custom_atmo = AtmosphericParameters(
    turbidity=1.3,           # ✅ Very clear
    haze_density=0.05,       # ✅ Minimal haze
    aerosol_scale_height=1200,
    ozone_concentration=0.35
)
```

---

### Problem: Marine Layer Fog Too Subtle

**Symptom**: June Gloom renders don't show enough fog.

**Cause**: Marine layer density too low.

**Solution**:
```python
from transformation_portal.atmosphere import MarineLayerParameters

# Enhance marine layer
marine_layer = MarineLayerParameters(
    altitude_m=500,        # Low fog altitude
    thickness_m=300,       # ✅ Thicker fog layer
    density=0.7,           # ✅ Higher density
    boundary_sharpness=0.3
)

atmo_params.marine_layer = marine_layer
```

---

### Problem: Batch Processing Running Out of Memory

**Symptom**: `OutOfMemoryError` or system slowdown during batch processing.

**Solutions**:
```bash
# 1. Reduce batch size
--batch-size 1

# 2. Process in segments
python material_response.py --input-dir renders/segment1/ ...
python material_response.py --input-dir renders/segment2/ ...

# 3. Reduce image resolution before processing
mogrify -resize 2048x renders/*.jpg

# 4. Use CPU instead of GPU (slower but less memory)
export TRANSFORMERS_DEVICE=cpu
```

---

## Summary

**Key Takeaways**:

1. **Physics-Based = Quality**: Shadow analysis prevents optically impossible renders
2. **Montecito-Specific**: Pre-calibrated for Sundowner and Marine Layer
3. **Auto-Correction**: System detects conflicts and suggests fixes
4. **Confidence Scores**: 0.0 = no hallucination, high scores = strong detection
5. **Production-Ready**: 400-600 images/hour batch throughput

**Workflow Checklist**:
- ✅ Choose correct season and condition (Sundowner vs. Marine Layer)
- ✅ Enable `auto_correct=True` for shadow analysis
- ✅ Validate atmospheric parameters match intent
- ✅ Monitor confidence scores for quality control
- ✅ Use batch processing for efficiency

**Next Steps**:
- Review CLI Reference: `docs/cli/CLI_REFERENCE.md`
- Explore SkyGAN Guide: `docs/SKYGAN_ATMOSPHERIC_RENDERING.md`
- Try examples: `examples/montecito_workflow/`

---

**Document Version**: 1.0
**Last Updated**: January 28, 2026
**System Version**: Transformation Portal v2.0
