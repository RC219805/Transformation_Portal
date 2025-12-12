# Phase 2 User Guide: Intelligent Material & Lighting Detection

**Last Updated:** December 12, 2025  
**Pipeline:** lux_depth_v2  
**Status:** Production-Ready

---

## What is Phase 2?

Phase 2 adds **intelligent scene understanding** to your image processing workflow:

- 🎯 **CLIP Material Classification** - Automatically identifies 28 material types (wood, metal, glass, water, stone, etc.)
- 🌅 **Lighting Detection** - Recognizes time-of-day and adjusts processing (golden hour, twilight, midday, etc.)
- 🔄 **Adaptive Processing** - Parameters automatically adjust based on what's in your scene
- 🎨 **Natural Language Queries** - Find materials using plain English ("reflective surfaces", "warm wood tones")

**Bottom Line:** Better results with less manual tuning, especially for luxury real estate and architectural photography.

---

## Quick Start - Enable Phase 2

### Command Line (Simple)

```bash
# Process a single image with Phase 2 auto-detection
lux-depth-v2 \
  --input-file mansion_pool.jpg \
  --output-dir output/ \
  --preset interior_luxury \
  --enable-phase2
```

### Command Line (Full Control)

```bash
# Batch processing with explicit Phase 2 configuration
lux-depth-v2 \
  --input-dir renders/ \
  --output-dir output/ \
  --preset interior_luxury_apex_quality \
  --enable-phase2 \
  --clip-materials \
  --lighting-detection \
  --device cuda
```

### Python API

```python
from lux_depth_v2.pipeline import DepthPipeline
from lux_depth_v2.config import PipelineConfig

# Create config with Phase 2 enabled
config = PipelineConfig.from_preset('interior_luxury_apex_quality')
config.enable_phase2 = True
config.clip_materials = True
config.lighting_detection = True

# Run pipeline
pipeline = DepthPipeline(config)
result = pipeline.process_image('mansion_pool.jpg')
```

---

## Phase 2 Features Explained

### 1. CLIP Material Classification

**What it does:** Analyzes your image and identifies 28 different material types using AI.

**Why it matters:** Different materials need different processing:
- **Glass** → Preserve transparency, enhance reflections
- **Wood** → Warm tone mapping, texture enhancement
- **Water** → Surface reflections, color purity
- **Metal** → Specular highlights, contrast boost

**Material Classes (28 Total):**

| Category | Materials |
|----------|-----------|
| **Core** (8) | wood, metal, glass, water, fabric, stone, ceramic, polished |
| **Architecture** (6) | stucco_wall, stone_column, aluminum_frame, wood_structure, concrete_surface, tile_surface |
| **Hardscape** (4) | pool_tile_mosaic, pool_deck_paver, stone_paver, concrete_deck |
| **Water** (3) | pool_water_surface, pool_water_volume, water_feature |
| **Vegetation** (5) | tree_canopy, flowering_tree, shrub, grass, succulent |
| **Sky** (2) | sky_gradient, mountain_distant |

**Natural Language Queries:**

```python
# Find all reflective surfaces
mask = classifier.query_natural_language(image, "reflective surfaces")

# Find warm wood tones
mask = classifier.query_natural_language(image, "warm natural wood")

# Find water features
mask = classifier.query_natural_language(image, "water that reflects light")
```

### 2. Lighting Detection

**What it does:** Identifies time-of-day and lighting conditions from your image.

**Why it matters:** Different lighting needs different tone mapping:
- **Golden Hour** → Warm enhancement, gentle contrast
- **Twilight** → Blue hour color, shadow detail recovery
- **Midday** → Highlight protection, haze reduction
- **Overcast** → Contrast boost, saturation lift

**Lighting Conditions (9 Types):**

| Condition | Characteristics | Auto-Adjustments |
|-----------|----------------|------------------|
| **Dawn** | Cool blue sky, low contrast | +Contrast, +Shadow detail |
| **Sunrise** | Warm horizon, directional light | +Warmth, +Glow |
| **Morning** | Clear light, moderate contrast | Balanced, +Clarity |
| **Midday** | Bright, high contrast, harsh shadows | -Contrast, +HDR tone map |
| **Afternoon** | Warm, strong directional light | +Detail, moderate warmth |
| **Golden Hour** | Warm glow, soft shadows | +Warmth, +Glow, -Contrast |
| **Twilight** | Blue hour, low light | +Blue tone, +Shadow detail |
| **Night** | Artificial light, high ISO | +Denoise, +Shadow recovery |
| **Overcast** | Flat light, low contrast | +Contrast, +Saturation |

**Adaptive Processing Example:**

```python
# Automatic lighting detection
detector = LightingConditionDetector()
lighting = detector.detect(image)  # Returns: "golden_hour"

# Auto-adjust tone mapping
if lighting == "golden_hour":
    config.exposure += 0.1      # Brighter
    config.warmth += 0.15       # More orange/red
    config.contrast *= 0.95     # Softer contrast
```

---

## When to Use Each Quality Tier

Not sure which preset to use? Follow this decision tree:

### Decision Tree

```
START: What type of project?

├─ Preview/Draft?
│  └─ Use: interior_luxury (STANDARD)
│     ⏱️ ~45-50s per image
│     💰 Fastest, good quality
│
├─ Client Marketing?
│  ├─ Web/Social Media?
│  │  └─ Use: interior_luxury_max_quality (MAX)
│  │     ⏱️ ~60-65s per image
│  │     💰 Best balance
│  │
│  └─ Print/Large Format?
│     └─ Use: interior_luxury_apex_quality (APEX)
│        ⏱️ ~50-55s per image
│        💰 Maximum quality
│
└─ Portfolio/Archival/Hero Shot?
   └─ Use: interior_luxury_apex_quality (APEX)
      ⏱️ ~50-55s per image
      💰 Flagship quality, worth the time
```

### Quick Reference Table

| Use Case | Preset | Speed | Best For |
|----------|--------|-------|----------|
| **Internal Reviews** | `interior_luxury` | ⚡⚡⚡ Fast | Rough cuts, client comps |
| **Website Hero Images** | `interior_luxury_max_quality` | ⚡⚡ Moderate | Marketing, social media |
| **Print Materials** | `interior_luxury_max_quality` | ⚡⚡ Moderate | Brochures, magazines |
| **Portfolio Pieces** | `interior_luxury_apex_quality` | ⚡ Slower | Best-of-best, archival |
| **Large Format Prints** | `interior_luxury_apex_quality` | ⚡ Slower | 36"+ width, exhibitions |

**Processing Times:** Based on 81MP images (9504x8504 @ 16-bit TIFF)

---

## Real-World Examples

### Example 1: Kitchen Interior (Phase 2 Auto-Detection)

**Scenario:** High-end kitchen with stainless appliances, marble counters, wood cabinets.

```bash
lux-depth-v2 \
  --input-file kitchen.jpg \
  --output-dir output/ \
  --preset interior_luxury_apex_quality \
  --enable-phase2
```

**What Phase 2 Detected:**
- **Materials:** stainless steel (42%), marble (28%), wood (18%), glass (12%)
- **Lighting:** morning (clear, directional light)
- **Auto-Adjustments:**
  - ✅ Enhanced metal reflections (appliances)
  - ✅ Warmed wood tones (cabinets)
  - ✅ Preserved glass transparency (windows)
  - ✅ Balanced contrast for morning light

**Result:** Natural, balanced rendering without manual tuning.

### Example 2: Pool Exterior (Golden Hour)

**Scenario:** Luxury pool at sunset with stone pavers, water features, mountain views.

```bash
lux-depth-v2 \
  --input-file pool_sunset.jpg \
  --output-dir output/ \
  --preset exterior_luxury_apex_quality \
  --enable-phase2 \
  --lighting-detection
```

**What Phase 2 Detected:**
- **Materials:** pool_water_surface (35%), stone_paver (25%), sky_gradient (20%), tree_canopy (15%)
- **Lighting:** golden_hour (warm, low-angle sun)
- **Auto-Adjustments:**
  - ✅ +15% warmth (golden glow)
  - ✅ +0.1 exposure (brighter highlights)
  - ✅ -5% contrast (softer transitions)
  - ✅ Enhanced water reflections
  - ✅ Preserved sky gradient

**Result:** Cinematic golden-hour look with minimal effort.

### Example 3: Batch Processing (50+ Images)

**Scenario:** Full property shoot with mixed interiors/exteriors.

```bash
lux-depth-v2 \
  --input-dir property_shoot/ \
  --output-dir final_deliverables/ \
  --preset interior_luxury_max_quality \
  --enable-phase2 \
  --batch-size 4 \
  --device cuda
```

**What Phase 2 Did:**
- ✅ Automatically detected scene type per image (interior vs exterior)
- ✅ Adjusted lighting per image (some morning, some afternoon, some twilight)
- ✅ Applied material-specific enhancements per image
- ✅ Maintained consistent quality across all 50 images

**Result:** Professional consistency without manual per-image tweaking.

---

## Command-Line Flags Reference

### Phase 2 Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-phase2` | `false` | Enable all Phase 2 features (CLIP + Lighting) |
| `--clip-materials` | `false` | Enable CLIP material classification only |
| `--lighting-detection` | `false` | Enable lighting detection only |
| `--disable-phase2` | - | Explicitly disable Phase 2 features |

### Material Classification Options

| Flag | Default | Description |
|------|---------|-------------|
| `--material-confidence` | `0.4` | Minimum confidence threshold (0.0-1.0) |
| `--material-query` | - | Natural language material query |
| `--fuse-segformer` | `true` | Blend CLIP with SegFormer spatial priors |

### Lighting Detection Options

| Flag | Default | Description |
|------|---------|-------------|
| `--lighting-adaptive` | `true` | Auto-adjust parameters based on detected lighting |
| `--lighting-override` | - | Force specific lighting condition |
| `--lighting-sensitivity` | `0.5` | Detection sensitivity (0.0=conservative, 1.0=aggressive) |

### Quality Preset Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `interior_luxury` | Quality tier (see below) |
| `--auto-preset` | `false` | Let CLIP choose preset based on scene |

**Available Presets:**
- `interior_luxury` - Standard quality (45-50s)
- `interior_luxury_max_quality` - Max quality (60-65s)
- `interior_luxury_apex_quality` - APEX quality (50-55s)
- `exterior_luxury_apex_quality` - Exterior APEX (coming soon)

---

## Performance Considerations

### Phase 2 Overhead

| Feature | Initialization | Per-Image | Memory |
|---------|---------------|-----------|--------|
| **CLIP Model** | ~2-3s (first time) | ~50-100ms | +600MB |
| **Lighting Detection** | None | ~10-20ms | +10MB |
| **Total Phase 2** | ~2-3s | ~60-120ms | +610MB |

**Bottom Line:** Phase 2 adds ~2-4% overhead to total processing time but delivers significantly better results.

### Batch Processing Optimization

```bash
# Amortize model loading across many images
lux-depth-v2 \
  --input-dir large_batch/ \
  --output-dir output/ \
  --preset interior_luxury \
  --enable-phase2 \
  --batch-size 8 \
  --device cuda
```

**Throughput:**
- **Without Phase 2:** 70-80 images/hour (standard)
- **With Phase 2:** 68-76 images/hour (standard, ~3% slower)
- **APEX Quality:** 60-70 images/hour (with Phase 2)

### GPU Acceleration

Phase 2 CLIP classification benefits significantly from GPU:

| Device | CLIP Time per Image |
|--------|---------------------|
| CPU (16-core M4 Max) | ~100-150ms |
| GPU (CUDA) | ~30-50ms |
| MPS (Apple Silicon) | ~40-60ms |

---

## Frequently Asked Questions

### Q: Do I need Phase 2 for good results?

**A:** No, but it helps significantly for:
- **Mixed-material scenes** (kitchens, pools, multi-surface areas)
- **Challenging lighting** (golden hour, twilight, harsh midday)
- **Batch processing** (automatic per-image optimization)

If you're processing simple scenes with consistent lighting, Phase 1 features (depth + materials v1) may be sufficient.

### Q: Can I use Phase 2 with older presets?

**A:** Yes! Phase 2 features are **additive**:

```bash
# Use Phase 1 preset with Phase 2 intelligence
lux-depth-v2 \
  --preset interior_luxury \
  --enable-phase2
```

### Q: What's the difference between CLIP and SegFormer materials?

**A:**

| Feature | SegFormer | CLIP |
|---------|-----------|------|
| **Approach** | Semantic segmentation (spatial) | Zero-shot classification (conceptual) |
| **Strengths** | Precise boundaries, fast | Understands concepts, no training |
| **Best For** | Well-defined regions | Complex materials, natural language |
| **Speed** | Fast (~50ms) | Moderate (~100ms) |

**Recommended:** Use both with fusion (`--fuse-segformer`) for best results.

### Q: Can I disable specific Phase 2 features?

**A:** Yes:

```bash
# CLIP materials only (no lighting detection)
lux-depth-v2 --clip-materials --preset interior_luxury

# Lighting detection only (no CLIP materials)
lux-depth-v2 --lighting-detection --preset interior_luxury

# All Phase 2 features
lux-depth-v2 --enable-phase2 --preset interior_luxury
```

### Q: Does Phase 2 work with exterior scenes?

**A:** Yes! Phase 2 expanded materials specifically for exteriors:
- Pool surfaces (tile, water, deck pavers)
- Vegetation (trees, shrubs, grass)
- Hardscape (stone, concrete)
- Sky and mountains

Use `--preset exterior_luxury_apex_quality` (coming soon) or `interior_luxury_apex_quality` with `--enable-phase2`.

### Q: How accurate is lighting detection?

**A:** Very good for clear conditions, moderate for edge cases:
- ✅ Excellent: Golden hour, twilight, midday, overcast
- ✅ Good: Morning, afternoon, sunrise, dawn
- ⚠️ Moderate: Mixed lighting (indoor + outdoor), artificial light

You can override with `--lighting-override golden_hour` if detection is incorrect.

### Q: Can I train custom material models?

**A:** Not yet, but it's on the roadmap:
- 📋 Phase 2.5: Custom ONNX model support
- 📋 Phase 3.0: Fine-tuning CLIP on custom datasets

For now, use natural language queries (`--material-query "custom description"`) for custom materials.

---

## Troubleshooting

### Issue: "CLIP model not found"

**Solution:**

```bash
# Download CLIP model manually
python -c "from transformers import CLIPProcessor, CLIPModel; \
  CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
  CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

# Or allow auto-download (requires internet)
lux-depth-v2 --enable-phase2 --allow-downloads
```

### Issue: "Phase 2 features too slow"

**Solution:**

```bash
# Disable Phase 2 for speed-critical workflows
lux-depth-v2 --disable-phase2 --preset interior_luxury

# Or use GPU acceleration
lux-depth-v2 --enable-phase2 --device cuda
```

### Issue: "Material detection inaccurate"

**Solution:**

```bash
# Increase confidence threshold (more conservative)
lux-depth-v2 --enable-phase2 --material-confidence 0.6

# Use natural language query for specific materials
lux-depth-v2 --enable-phase2 --material-query "polished marble countertops"

# Disable CLIP fusion (use SegFormer only)
lux-depth-v2 --clip-materials --fuse-segformer false
```

### Issue: "Lighting detection wrong (detected golden_hour but it's midday)"

**Solution:**

```bash
# Override lighting detection
lux-depth-v2 --lighting-detection --lighting-override midday

# Reduce detection sensitivity (less aggressive)
lux-depth-v2 --lighting-detection --lighting-sensitivity 0.3
```

---

## Next Steps

- 📖 [Quality Tiers Guide](QUALITY_TIERS.md) - Detailed preset comparison
- 🚀 [Performance Benchmarks](PHASE2_PERFORMANCE.md) - Throughput and timing data
- 🔧 [CI/CD Integration](CI_PHASE2_INTEGRATION.md) - Automated testing and deployment
- 💡 [Phase 2 Implementation Guide](../lux_depth_v2/PHASE2_IMPLEMENTATION_GUIDE.md) - Technical deep dive

**Support:** https://github.com/RC219805/Transformation_Portal/issues

---

**Pro Tip:** Start with `--enable-phase2` and `interior_luxury_max_quality` for most projects. Only disable Phase 2 if you're hitting performance constraints or need maximum throughput.
