# AI Enhancement Guide for Luxury Real Estate

This guide covers the revolutionary AI-powered visual enhancement technologies integrated into the Transformation Portal for luxury real estate photography.

## Table of Contents

1. [Overview](#overview)
2. [Vision-Language Models (VLMs)](#vision-language-models)
3. [Semantic Segmentation](#semantic-segmentation)
4. [Neuroaesthetics Optimization](#neuroaesthetics-optimization)
5. [Quality Metrics](#quality-metrics)
6. [Montecito/Santa Barbara Presets](#montecitosanta-barbara-presets)
7. [Complete Workflow](#complete-workflow)

---

## Overview

The Transformation Portal now includes cutting-edge AI technologies for luxury real estate imagery:

- **LLaVA-1.5** - Scene understanding and quality validation
- **SAM + CLIP** - Material-aware semantic segmentation
- **Neuroaesthetics** - Golden ratio, color harmony, spatial frequency optimization
- **LPIPS & FID** - Perceptual quality metrics
- **Location-specific presets** - Montecito/Santa Barbara atmospheric rendering

### Technology Stack

```
Vision-Language: LLaVA-1.5 (13B parameters)
Segmentation: SAM (Segment Anything Model)
Classification: CLIP (ViT-L/14)
Metrics: LPIPS, FID, Traditional (PSNR, SSIM)
Diffusion: Stable Diffusion XL (existing), FLUX (pending)
```

---

## Vision-Language Models

### LLaVA-1.5 Integration

LLaVA provides intelligent scene understanding and quality assessment:

```python
from transformation_portal.vlm import LLaVAProcessor, SceneAnalyzer, QualityValidator

# Initialize processor
processor = LLaVAProcessor(
    model_id="llava-hf/llava-1.5-13b-hf",
    quantization=True  # 4-bit quantization: 24GB -> 8GB VRAM
)

# Analyze scene
analysis = processor.analyze_image(
    "luxury_kitchen.jpg",
    prompt="Analyze this architectural image for room type, style, materials, and luxury features"
)
print(analysis)
```

### Scene Analysis

Structured scene understanding:

```python
from transformation_portal.vlm import SceneAnalyzer

analyzer = SceneAnalyzer()
analysis = analyzer.analyze("luxury_interior.jpg")

print(f"Space: {analysis.space_type}")  # interior/exterior/aerial
print(f"Room: {analysis.room_type}")  # kitchen/bathroom/living/etc
print(f"Style: {analysis.architectural_style}")  # modern/traditional/mediterranean
print(f"Materials: {analysis.materials}")  # [marble, wood, glass, ...]
print(f"Luxury features: {analysis.luxury_features}")

# Get processing recommendations based on scene
recommendations = analyzer.get_processing_recommendations(analysis)
print(f"Suggested preset: {recommendations['suggested_preset']}")
```

### Quality Validation

Automated quality assessment:

```python
from transformation_portal.vlm import QualityValidator

validator = QualityValidator()

# Validate enhanced image
report = validator.validate("enhanced_image.jpg", detailed=True)

print(f"Overall status: {report.overall_status}")  # PASS/WARNING/FAIL
print(f"Overall score: {report.overall_score}/10")
print(f"Passed: {report.passed_validation}")

# Check specific aspects
for score in report.scores:
    print(f"{score.aspect}: {score.score}/10 - {score.status}")

# Compare original vs enhanced
comparison = validator.validate_enhancement(
    original="original.jpg",
    enhanced="enhanced.jpg"
)

if comparison["enhancement_validation"]["quality_improved"]:
    print("Enhancement improved quality!")
```

---

## Semantic Segmentation

### Material-Aware Processing

Combine SAM and CLIP for intelligent material segmentation:

```python
from transformation_portal.segmentation import MaterialSegmenter

# Initialize segmenter
segmenter = MaterialSegmenter()

# Segment by materials
segments = segmenter.segment_materials(
    "luxury_kitchen.jpg",
    materials=["marble", "wood", "glass", "metal", "stainless steel"],
    min_segment_area=500,
    confidence_threshold=0.3
)

# Analyze results
print(f"Found {len(segments)} material segments")

for seg in segments:
    print(f"Material: {seg.material} (confidence: {seg.confidence:.2f})")
    print(f"Area: {seg.area} pixels")

# Get enhancement recommendations
recommendations = segmenter.get_enhancement_recommendations(segments)
print(recommendations)

# Visualize segmentation
viz = segmenter.visualize_materials("luxury_kitchen.jpg", segments, alpha=0.5)
```

### Individual Segment Processing

Process different materials with appropriate settings:

```python
# Find all marble surfaces
marble_masks = segmenter.get_material_masks(segments, "marble")

# Apply marble-specific enhancement
for mask in marble_masks:
    # Enhance veining, preserve color, boost clarity
    apply_marble_enhancement(image, mask)

# Find glass/water features
glass_segments = [s for s in segments if s.material in ["glass", "water"]]

# Enhance reflections and transparency
for seg in glass_segments:
    enhance_transparency_and_reflections(image, seg.mask)
```

### SAM Standalone Usage

```python
from transformation_portal.segmentation import SAMSegmenter

segmenter = SAMSegmenter(model_type="vit_h")  # Highest quality

# Automatic segmentation
masks = segmenter.segment_automatic("image.jpg", min_area=500)

# Point-prompted segmentation
point_coords = [[500, 300]]  # Click on object
mask = segmenter.segment_from_points("image.jpg", point_coords)

# Box-prompted segmentation
box = [100, 100, 500, 400]  # [x1, y1, x2, y2]
mask = segmenter.segment_from_box("image.jpg", box)
```

### CLIP Standalone Usage

```python
from transformation_portal.segmentation import CLIPClassifier

classifier = CLIPClassifier()

# Classify image
materials = classifier.classify_materials("countertop.jpg")
print(materials)  # {'marble': 0.85, 'granite': 0.10, ...}

# Detect luxury features
features = classifier.detect_features("luxury_bath.jpg", threshold=0.1)
print(features)  # [('high ceiling', 0.75), ('chandelier', 0.60), ...]

# Create semantic map
semantic_map, labels = classifier.create_semantic_map(
    "kitchen.jpg",
    masks=sam_masks,
    categories=["marble", "wood", "glass", "metal"]
)
```

---

## Neuroaesthetics Optimization

### Golden Ratio Analysis

Compositional analysis using golden ratio (φ ≈ 1.618):

```python
from transformation_portal.neuroaesthetics import GoldenRatioAnalyzer

analyzer = GoldenRatioAnalyzer()

# Analyze composition
analysis = analyzer.analyze("architectural_photo.jpg")

print(f"Golden ratio score: {analysis.score:.2f}")  # 0-1
print(f"Recommendations: {analysis.recommendations}")

# Visualize golden ratio grid
grid_viz = analyzer.visualize_grid("architectural_photo.jpg")

# Get optimal crop
crop_box = analyzer.get_optimal_crop("image.jpg", target_aspect=1.618)
```

### Color Harmony Analysis

CIELAB perceptual color analysis:

```python
from transformation_portal.neuroaesthetics import ColorHarmonyAnalyzer

analyzer = ColorHarmonyAnalyzer(num_colors=5)

# Analyze color harmony
analysis = analyzer.analyze("interior.jpg")

print(f"Harmony score: {analysis.harmony_score:.2f}")
print(f"Harmony type: {analysis.harmony_type}")  # analogous/complementary/warm/cool
print(f"Temperature: {analysis.temperature:.2f}")  # -1=cool, 1=warm

# Emotional profile
print(f"Emotional associations:")
for emotion, score in analysis.emotional_profile.items():
    print(f"  {emotion}: {score:.2f}")

# Check for disharmony
if analysis.disharmony_factors:
    print("Issues detected:")
    for issue in analysis.disharmony_factors:
        print(f"  - {issue}")
```

### Spatial Frequency Balance

Visual comfort through frequency analysis:

```python
from transformation_portal.neuroaesthetics import SpatialFrequencyAnalyzer

analyzer = SpatialFrequencyAnalyzer()

# Analyze frequency distribution
analysis = analyzer.analyze("photo.jpg")

print(f"LSF (structure): {analysis.lsf_energy:.2f}")
print(f"MSF (detail): {analysis.msf_energy:.2f}")
print(f"HSF (texture): {analysis.hsf_energy:.2f}")
print(f"Balance score: {analysis.balance_score:.2f}")
print(f"Visual comfort: {analysis.visual_comfort_score:.2f}")

# Visualize spectrum
spectrum_viz = analyzer.create_frequency_visualization("photo.jpg")
```

### Emotional Optimization

Integrated neuroaesthetics optimization:

```python
from transformation_portal.neuroaesthetics import EmotionalOptimizer, EmotionalTarget

optimizer = EmotionalOptimizer()

# Analyze complete profile
profile = optimizer.analyze("luxury_property.jpg")

print(f"Overall quality: {profile.overall_quality:.2f}")
print(f"Emotional scores:")
for emotion, score in profile.emotional_scores.items():
    print(f"  {emotion}: {score:.2f}")

# Optimization priorities
print("Prioritize:")
for aspect, importance in profile.optimization_priority:
    print(f"  {aspect}: {importance:.2f}")

# Optimize for specific emotion
strategy = optimizer.optimize_for_emotion(
    "property.jpg",
    target_emotion=EmotionalTarget.ASPIRATION
)

print(f"Target: {strategy['target_emotion']}")
print(f"Current score: {strategy['current_score']:.2f}")
print(f"Gaps: {strategy['gaps']}")
print(f"Adjustments: {strategy['recommended_adjustments']}")
print(f"Parameters: {strategy['processing_parameters']}")
```

**Emotional Targets:**

- `NOSTALGIA` - Warm palettes, natural materials, heritage details
- `ASPIRATION` - High spatial quality, abundant light, golden ratio
- `DESIRE` - Quality craftsmanship, premium materials, believable luxury
- `LUXURY` - Low saturation sophistication, high lightness
- `COMFORT` - Warm analogous colors, high visual comfort
- `SERENITY` - Cool colors, high visual comfort, balanced frequencies
- `ENERGY` - High saturation, complementary colors

---

## Quality Metrics

### LPIPS (Perceptual Similarity)

Learned perceptual similarity - more accurate than PSNR/SSIM:

```python
from transformation_portal.metrics import LPIPSMetric

metric = LPIPSMetric(network='alex')  # or 'vgg', 'squeeze'

# Calculate perceptual distance
distance = metric.calculate("original.jpg", "enhanced.jpg")

print(f"LPIPS distance: {distance:.4f}")

# Interpret results
interpretation = metric.interpret(distance)
print(f"Similarity: {interpretation['similarity']}")  # very_similar/similar/different
print(f"Quality: {interpretation['quality']}")  # excellent/good/acceptable/poor
print(f"Acceptable: {interpretation['acceptable_for_enhancement']}")

# Batch processing
distances = metric.calculate_batch(
    ["orig1.jpg", "orig2.jpg", "orig3.jpg"],
    ["enh1.jpg", "enh2.jpg", "enh3.jpg"]
)
```

**LPIPS Thresholds:**
- < 0.1: Very similar (excellent)
- 0.1-0.2: Similar (good)
- 0.2-0.3: Somewhat different (acceptable)
- \> 0.3: Different (poor)

### FID (Distribution Matching)

Fréchet Inception Distance - validates photorealism:

```python
from transformation_portal.metrics import FIDMetric

metric = FIDMetric()

# Compare distributions
real_images = ["real1.jpg", "real2.jpg", ..., "real50.jpg"]
enhanced_images = ["enh1.jpg", "enh2.jpg", ..., "enh50.jpg"]

fid_score = metric.calculate(real_images, enhanced_images, batch_size=32)

print(f"FID score: {fid_score:.2f}")

# Interpret
interpretation = metric.interpret(fid_score)
print(f"Quality: {interpretation['quality']}")
print(f"Photorealistic: {interpretation['photorealistic']}")
print(f"Description: {interpretation['description']}")
```

**FID Thresholds:**
- < 10: Excellent (nearly indistinguishable)
- 10-20: Very good
- 20-50: Good
- \> 50: Poor (distribution mismatch)

### Traditional Metrics

PSNR, SSIM for reference:

```python
from transformation_portal.metrics import TraditionalMetrics

metrics = TraditionalMetrics()

# Calculate all metrics
results = metrics.calculate_all("original.jpg", "enhanced.jpg")

print(f"PSNR: {results['psnr']:.2f} dB")  # Typically 20-50 dB
print(f"SSIM: {results['ssim']:.4f}")  # 0-1, higher = better
```

---

## Montecito/Santa Barbara Presets

Location-specific enhancement for coastal California luxury properties.

### Available Presets

1. **montecito_estate** - Mediterranean luxury estates
2. **coastal_contemporary** - Modern coastal architecture
3. **hope_ranch_traditional** - Traditional estates
4. **riviera_view** - Hillside properties with views
5. **coastal_beach** - Beachfront properties
6. **santa_barbara_golden_hour** - Golden hour optimization
7. **marine_layer_morning** - Soft morning light

### Usage

```python
import yaml

# Load presets
with open('config/montecito_santa_barbara_presets.yaml') as f:
    presets = yaml.safe_load(f)

# Get specific preset
montecito_settings = presets['presets']['montecito_estate']

# Apply to pipeline
pipeline = UnifiedLuxuryPipeline(
    color_temperature=montecito_settings['color_grading']['temperature'],
    saturation=montecito_settings['color_grading']['saturation'],
    # ... other parameters
)
```

### Preset Characteristics

**Montecito Estate:**
- Warm Mediterranean light (+8 temperature)
- Enhanced stone and tile materials
- Golden ratio target: 0.80
- Emotion: nostalgia_luxury

**Coastal Contemporary:**
- Neutral-warm palette (+3 temperature)
- Muted sophisticated saturation (0.98)
- Enhanced glass and metal
- Emotion: aspiration_serenity

**Hope Ranch Traditional:**
- Very warm (+10 temperature)
- Enhanced wood and heritage materials
- Soft atmospheric depth
- Emotion: nostalgia_comfort

**Riviera View:**
- Balanced warm (+5 temperature)
- Atmospheric depth for views
- Enhanced glass (view framing)
- Emotion: aspiration_energy

**Santa Barbara Golden Hour:**
- Maximum warmth (+12 temperature)
- Golden atmospheric glow
- Enhanced all reflective surfaces
- Emotion: nostalgia_aspiration

### Seasonal Adjustments

```python
# Get seasonal settings
season_settings = presets['seasonal_adjustments']['fall']

# Fall: Sundowner season with exceptional clarity
clarity_boost = season_settings['clarity_enhancement']  # 1.2
warmth_boost = season_settings['warmth_boost']  # 1.1
```

### Atmospheric Parameters

Location-specific atmospheric modeling (34.4°N):

```yaml
atmospheric_model:
  location:
    latitude: 34.4
    longitude: -119.7

  marine_layer:
    typical_height: 500  # feet
    density_range: [0.3, 0.7]

  sundowner:
    clarity_boost: 1.3
    temperature_shift: +5

  aerosols:
    marine_aerosol: 0.6
    humidity_typical: 0.65
```

---

## Complete Workflow

### End-to-End Enhancement Pipeline

```python
from transformation_portal.vlm import SceneAnalyzer, QualityValidator
from transformation_portal.segmentation import MaterialSegmenter
from transformation_portal.neuroaesthetics import EmotionalOptimizer
from transformation_portal.metrics import LPIPSMetric, FIDMetric

# Step 1: Scene Understanding
print("1. Analyzing scene...")
scene_analyzer = SceneAnalyzer()
scene = scene_analyzer.analyze("input.jpg")

print(f"Room: {scene.room_type}, Style: {scene.architectural_style}")
recommendations = scene_analyzer.get_processing_recommendations(scene)

# Step 2: Material Segmentation
print("2. Segmenting materials...")
material_segmenter = MaterialSegmenter()
segments = material_segmenter.segment_materials("input.jpg")

enhancement_strategy = material_segmenter.get_enhancement_recommendations(segments)
print(f"Detected materials: {[s.material for s in segments]}")

# Step 3: Neuroaesthetics Analysis
print("3. Analyzing aesthetics...")
aesthetic_optimizer = EmotionalOptimizer()
aesthetic_profile = aesthetic_optimizer.analyze("input.jpg")

print(f"Overall quality: {aesthetic_profile.overall_quality:.2f}")
print(f"Optimization priorities: {aesthetic_profile.optimization_priority}")

# Step 4: Apply Enhancement
print("4. Applying enhancement...")
# Use your existing pipeline with insights from above
from transformation_portal.pipelines import UnifiedLuxuryPipeline

pipeline = UnifiedLuxuryPipeline(
    preset=recommendations['suggested_preset'],
    enhancement_strength=recommendations['enhancement_strength']
)

result = pipeline.process("input.jpg", output_dir="output/")

# Step 5: Quality Validation
print("5. Validating quality...")
lpips = LPIPSMetric()
lpips_score = lpips.calculate("input.jpg", result['master'])

validator = QualityValidator()
quality_report = validator.validate(result['master'])

print(f"LPIPS: {lpips_score:.4f}")
print(f"Quality status: {quality_report.overall_status}")
print(f"Passed: {quality_report.passed_validation}")

if not quality_report.passed_validation:
    print("Issues:", quality_report.artifacts)
    print("Recommendations:", quality_report.recommendations)
```

### Batch Processing with Quality Control

```python
import glob
from pathlib import Path

# Initialize components
scene_analyzer = SceneAnalyzer()
validator = QualityValidator()
lpips_metric = LPIPSMetric()

# Process batch
input_images = glob.glob("input_images/*.jpg")
results = []

for img_path in input_images:
    print(f"Processing {img_path}...")

    # Analyze
    scene = scene_analyzer.analyze(img_path)

    # Process
    output = pipeline.process(
        img_path,
        preset=scene_analyzer.get_processing_recommendations(scene)['suggested_preset']
    )

    # Validate
    quality = validator.validate(output['master'])
    lpips_score = lpips_metric.calculate(img_path, output['master'])

    # Record results
    results.append({
        'input': img_path,
        'output': output['master'],
        'quality_passed': quality.passed_validation,
        'quality_score': quality.overall_score,
        'lpips': lpips_score,
        'scene_type': scene.room_type.value
    })

    # Flag issues
    if not quality.passed_validation or lpips_score > 0.2:
        print(f"  WARNING: Quality issues detected")
        print(f"    Score: {quality.overall_score:.2f}/10")
        print(f"    LPIPS: {lpips_score:.4f}")

# Summary
passed = sum(1 for r in results if r['quality_passed'])
print(f"\nProcessed {len(results)} images")
print(f"Passed quality: {passed}/{len(results)}")
print(f"Average LPIPS: {np.mean([r['lpips'] for r in results]):.4f}")
```

---

## Installation

### Core Dependencies

```bash
# Install via requirements.txt
pip install -r requirements.txt

# Additional installations
pip install lpips  # Quality metrics
pip install git+https://github.com/facebookresearch/segment-anything.git  # SAM
```

### Model Downloads

**LLaVA:**
```python
# Downloaded automatically on first use
# Requires ~13GB for model + ~8GB VRAM (with quantization)
```

**SAM:**
```bash
# Download checkpoint from:
# https://github.com/facebookresearch/segment-anything#model-checkpoints

# Recommended: vit_h (highest quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir -p ~/.cache/sam/
mv sam_vit_h_4b8939.pth ~/.cache/sam/
```

### Hardware Requirements

**Minimum:**
- GPU: RTX 3060 12GB or RTX 4060 Ti 16GB
- RAM: 32GB
- Storage: 50GB for models

**Recommended:**
- GPU: RTX 4080 16GB or RTX 4090 24GB
- RAM: 64GB
- Storage: 100GB SSD

**Professional:**
- GPU: RTX 4090 24GB or dual RTX 4080
- RAM: 128GB
- Storage: 500GB NVMe SSD

---

## Best Practices

### Quality Validation Gates

Always validate enhancements:

```python
def validate_enhancement(original, enhanced):
    """Quality gate for production use."""

    # Perceptual similarity
    lpips = LPIPSMetric()
    lpips_score = lpips.calculate(original, enhanced)

    if lpips_score > 0.2:
        return False, "Excessive perceptual change"

    # VLM quality assessment
    validator = QualityValidator()
    report = validator.validate(enhanced)

    if not report.passed_validation:
        return False, f"Quality issues: {report.artifacts}"

    # Structural preservation
    comparison = validator.validate_enhancement(original, enhanced)

    if comparison['enhancement_validation']['new_artifacts_introduced']:
        return False, "New artifacts introduced"

    return True, "Passed all quality gates"
```

### Iterative Refinement

Use feedback loops:

```python
max_iterations = 3
strength = 0.5

for i in range(max_iterations):
    # Enhance
    enhanced = enhance_image(original, strength)

    # Validate
    passed, message = validate_enhancement(original, enhanced)

    if passed:
        print(f"Passed on iteration {i+1}")
        break

    # Reduce strength and retry
    strength *= 0.8
    print(f"Iteration {i+1} failed: {message}. Reducing strength to {strength:.2f}")
```

---

## Troubleshooting

### Out of Memory

```python
# Use 4-bit quantization for LLaVA
processor = LLaVAProcessor(quantization=True)  # 24GB -> 8GB

# Use smaller SAM model
segmenter = SAMSegmenter(model_type="vit_b")  # vs vit_h

# Process in batches
for batch in image_batches:
    results = process_batch(batch)
    torch.cuda.empty_cache()  # Clear GPU memory
```

### Slow Processing

```python
# Use faster models where appropriate
lpips = LPIPSMetric(network='alex')  # Fastest
# vs 'vgg' (slower, more accurate)

# Skip detailed analysis for batch processing
analysis = scene_analyzer.analyze(img, detailed=False)

# Reduce segmentation resolution
segments = segmenter.segment_materials(
    img,
    max_segments=20  # vs 50 default
)
```

---

## References

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Segment Anything (SAM)](https://segment-anything.com/)
- [CLIP](https://openai.com/research/clip)
- [LPIPS](https://arxiv.org/abs/1801.03924)
- [FID](https://arxiv.org/abs/1706.08500)

---

## Support

For issues or questions, see the main project documentation or open an issue on GitHub.
