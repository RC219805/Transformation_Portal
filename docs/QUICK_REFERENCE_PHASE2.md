# Phase 2 Quick Reference Card

**Lux Depth V2 - Phase 2 Features Cheat Sheet**

---

## One-Line Commands

### Basic Processing

```bash
# Standard quality (fastest)
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury

# Max quality (balanced)
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury_max_quality

# APEX quality (maximum)
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury_apex_quality
```

### Phase 2 Features

```bash
# All Phase 2 features (CLIP + Lighting)
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury --enable-phase2

# CLIP materials only
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury --clip-materials

# Lighting detection only
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury --lighting-detection

# Auto-select preset based on scene
lux-depth-v2 --input-file image.jpg --output-dir out/ --auto-preset --enable-phase2
```

### Batch Processing

```bash
# Process entire folder (Standard)
lux-depth-v2 --input-dir renders/ --output-dir final/ --preset interior_luxury

# Batch with Phase 2 (GPU accelerated)
lux-depth-v2 --input-dir renders/ --output-dir final/ --preset interior_luxury_max_quality \
  --enable-phase2 --device cuda --batch-size 8
```

### Advanced

```bash
# Natural language material query
lux-depth-v2 --input-file pool.jpg --output-dir out/ --preset interior_luxury_apex_quality \
  --enable-phase2 --material-query "water that reflects sunlight"

# Override lighting detection
lux-depth-v2 --input-file sunset.jpg --output-dir out/ --preset interior_luxury_apex_quality \
  --lighting-detection --lighting-override golden_hour

# Custom confidence threshold
lux-depth-v2 --input-file kitchen.jpg --output-dir out/ --preset interior_luxury_max_quality \
  --clip-materials --material-confidence 0.6
```

---

## Preset Comparison Table

| Preset | Speed | Quality | Memory | Use Case |
|--------|-------|---------|--------|----------|
| `interior_luxury` | 45-50s | Good | 6-8 GB | Bulk batches, previews |
| `interior_luxury_max_quality` | 60-65s | High | 8-12 GB | Client marketing, web |
| `interior_luxury_apex_quality` | 50-55s | Maximum | 8-12 GB | Portfolio, print, archival |

**Times:** Based on 81MP images (9504x8504 @ 16-bit TIFF) on M4 Max

---

## Phase 2 Feature Matrix

| Feature | Flag | Cost (per image) | Benefit |
|---------|------|------------------|---------|
| **CLIP Materials** | `--clip-materials` | +50-100ms, +600MB | 28 material classes, natural language |
| **Lighting Detection** | `--lighting-detection` | +10-20ms, +10MB | 9 time-of-day types, adaptive tone map |
| **Both** | `--enable-phase2` | +60-120ms, +610MB | Full scene understanding |

---

## Material Classes (28 Total)

### Core Materials (8)
`wood`, `metal`, `glass`, `water`, `fabric`, `stone`, `ceramic`, `polished`

### Architecture (6)
`stucco_wall`, `stone_column`, `aluminum_frame`, `wood_structure`, `concrete_surface`, `tile_surface`

### Hardscape (4)
`pool_tile_mosaic`, `pool_deck_paver`, `stone_paver`, `concrete_deck`

### Water (3)
`pool_water_surface`, `pool_water_volume`, `water_feature`

### Vegetation (5)
`tree_canopy`, `flowering_tree`, `shrub`, `grass`, `succulent`

### Sky (2)
`sky_gradient`, `mountain_distant`

---

## Lighting Conditions (9 Types)

| Condition | R/B Ratio | Characteristics |
|-----------|-----------|-----------------|
| `dawn` | 0.8-1.0 | Cool blue, low contrast |
| `sunrise` | 1.2-1.5 | Warm horizon, directional |
| `morning` | 1.0-1.2 | Clear, moderate contrast |
| `midday` | 0.9-1.1 | Bright, harsh shadows |
| `afternoon` | 1.1-1.4 | Warm, strong directional |
| `golden_hour` | 1.4-2.0 | Warm glow, soft shadows |
| `twilight` | 0.7-0.9 | Blue hour, low light |
| `night` | <0.7 | Artificial light, high ISO |
| `overcast` | 0.9-1.1 | Flat light, low contrast |

---

## Decision Trees

### Which Preset Should I Use?

```
Is this a PREVIEW/DRAFT?
  → YES: interior_luxury (45-50s)

Is this CLIENT-FACING?
  → WEB/SOCIAL: interior_luxury_max_quality (60-65s)
  → PRINT (36"+): interior_luxury_apex_quality (50-55s)

Is this PORTFOLIO/HERO?
  → YES: interior_luxury_apex_quality (50-55s)
```

### Should I Enable Phase 2?

```
Is scene COMPLEX? (mixed materials, challenging lighting)
  → YES: --enable-phase2

Is this BATCH PROCESSING? (10+ images)
  → YES: --enable-phase2 (auto-optimization per image)

Is scene SIMPLE? (single room, consistent light)
  → MAYBE: Try without first, add if needed

Is SPEED CRITICAL? (tight deadline)
  → NO: --disable-phase2 (saves 60-120ms per image)
```

---

## Common Use Cases

### Wedding Photography (Golden Hour Outdoor)

```bash
lux-depth-v2 --input-dir ceremony/ --output-dir final/ \
  --preset interior_luxury_apex_quality \
  --lighting-detection --lighting-override golden_hour
```

**Why:** Golden hour needs warm enhancement and soft contrast.

### Real Estate Listing (Mixed Interior Rooms)

```bash
lux-depth-v2 --input-dir property/ --output-dir marketing/ \
  --preset interior_luxury_max_quality \
  --enable-phase2 --batch-size 4
```

**Why:** Auto-detects lighting per room, material-aware enhancements.

### Architectural Portfolio (Hero Shots)

```bash
lux-depth-v2 --input-file hero_pool.jpg --output-dir portfolio/ \
  --preset interior_luxury_apex_quality \
  --enable-phase2 --device cuda
```

**Why:** Maximum quality, full scene understanding, worth the time.

### Quick Client Comp (Internal Review)

```bash
lux-depth-v2 --input-dir rough_cuts/ --output-dir comps/ \
  --preset interior_luxury \
  --disable-phase2
```

**Why:** Speed over perfection, Phase 2 overhead not needed for drafts.

### Pool/Exterior (Sunset, Complex Materials)

```bash
lux-depth-v2 --input-file pool_sunset.jpg --output-dir final/ \
  --preset interior_luxury_apex_quality \
  --enable-phase2 \
  --material-query "water surfaces and stone pavers"
```

**Why:** Multiple materials (water, stone, sky), golden hour lighting.

---

## Performance Tips

### Speed Optimization

```bash
# Disable Phase 2 for speed
lux-depth-v2 --input-dir batch/ --output-dir out/ --preset interior_luxury --disable-phase2

# Use GPU acceleration
lux-depth-v2 --input-dir batch/ --output-dir out/ --preset interior_luxury --device cuda

# Increase batch size (GPU)
lux-depth-v2 --input-dir batch/ --output-dir out/ --preset interior_luxury --batch-size 16 --device cuda
```

### Quality Optimization

```bash
# APEX quality with all Phase 2 features
lux-depth-v2 --input-file hero.jpg --output-dir out/ --preset interior_luxury_apex_quality --enable-phase2

# Increase material confidence (more conservative)
lux-depth-v2 --input-file image.jpg --output-dir out/ --preset interior_luxury --clip-materials --material-confidence 0.7

# Override incorrect lighting detection
lux-depth-v2 --input-file twilight.jpg --output-dir out/ --preset interior_luxury --lighting-override twilight
```

---

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| **Too slow** | `--disable-phase2` or `--device cuda` |
| **Materials wrong** | `--material-confidence 0.6` (more conservative) |
| **Lighting wrong** | `--lighting-override <condition>` |
| **Model not found** | `--allow-downloads` or manual download |
| **Out of memory** | `--batch-size 1` or `--preset interior_luxury` |
| **Quality not good enough** | `--preset interior_luxury_apex_quality --enable-phase2` |

---

## Output Files

For each input image, you get:

| File | Format | Description |
|------|--------|-------------|
| `*_master16.tif` | 16-bit TIFF | Graded, pre-upscale (archival) |
| `*_upscaled16.tif` | 16-bit TIFF | Final upscaled (archival) |
| `*_marketing.png` | 8-bit PNG | Web/social media ready |
| `*_preview.jpg` | JPEG | Quick preview (small) |
| `*_report.json` | JSON | Processing metadata |

---

## Python API Quick Reference

### Basic Processing

```python
from lux_depth_v2.pipeline import DepthPipeline
from lux_depth_v2.config import PipelineConfig

# Load preset
config = PipelineConfig.from_preset('interior_luxury_max_quality')
config.enable_phase2 = True

# Process
pipeline = DepthPipeline(config)
result = pipeline.process_image('input.jpg')
```

### CLIP Materials

```python
from lux_depth_v2.materials_v2 import CLIPMaterialClassifier
import torch

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = CLIPMaterialClassifier(device)

# Classify
rgb = torch.rand(1, 3, 512, 512)
scores = classifier.classify_image(rgb)
# {'wood': 0.65, 'metal': 0.42, ...}

# Natural language query
mask = classifier.query_natural_language(rgb, "reflective metal surfaces")
```

### Lighting Detection

```python
from lux_depth_v2.lighting_detector import LightingConditionDetector

# Detect
detector = LightingConditionDetector()
lighting = detector.detect(rgb_array)
# 'golden_hour', 'twilight', 'midday', etc.

# Get adaptive parameters
params = detector.get_adaptive_params(lighting)
# {'exposure': 0.1, 'warmth': 0.15, 'contrast': 0.95}
```

---

## Links

- 📖 [Full User Guide](PHASE2_USER_GUIDE.md)
- 📊 [Quality Tiers](QUALITY_TIERS.md)
- 🚀 [Performance Benchmarks](PHASE2_PERFORMANCE.md)
- 🔧 [CI/CD Integration](CI_PHASE2_INTEGRATION.md)

---

**Last Updated:** December 12, 2025  
**Version:** Phase 2 Production (CLIP + Lighting Detection)
