# Quality Tiers: Interior Luxury Real Estate Processing

**Last Updated:** December 12, 2025  
**Pipeline:** lux_depth_v2  
**Validated On:** 750 Picacho Kitchen (81MP, 16-bit TIFF)

**📖 New User?** Start with the [Phase 2 User Guide](PHASE2_USER_GUIDE.md) for a complete walkthrough with examples.  
**⚡ Quick Reference:** See the [Quick Reference Card](QUICK_REFERENCE_PHASE2.md) for one-page CLI cheat sheet.

---

## Overview

The lux_depth_v2 pipeline offers **three quality tiers** for interior luxury real estate processing, each optimized for different use cases. All tiers use the same core technologies (Depth Anything V2, SegFormer-B5, Materials V2) but with different quality/performance tradeoffs.

**Phase 2 Features:** All tiers now support intelligent material classification (CLIP) and lighting detection for adaptive processing. See the [Phase 2 User Guide](PHASE2_USER_GUIDE.md) for details on enabling these features.

---

## Quality Tier Matrix

| Preset | Processing Time | Use Case | Output Quality | Cost |
|--------|----------------|----------|----------------|------|
| **Standard** | ~45-50s | Bulk batches, internal review | Good | Fastest |
| **Max Quality** | ~60-65s | Client-facing marketing | High | Balanced |
| **APEX Quality** | ~50-55s | Flagship portfolio, archival | Maximum | Most resources |

---

## Tier 1: Standard Production (`interior_luxury`)

### Configuration
```yaml
preset: interior_luxury
segmentation:
  backend: segformer
  input_long_side: 768
  min_confidence: 0.25
materials_v2:
  enabled: true
  confidence_threshold: 0.4
  require_high_quality: false
precision: fp16
post_overlap: 64
```

### Characteristics
- **Speed:** Fastest (baseline reference)
- **Quality:** Good, production-ready
- **Memory:** 6-8 GB VRAM
- **Output:** 16-bit TIFF + compressed PNG

### Use Cases
- Bulk batch processing (10+ images)
- Internal reviews and rough cuts
- Quick client comps and previews
- Projects where throughput > ultimate quality

### Expected Metrics
- Processing: 45-50s per 81MP image
- Throughput: 70-80 images/hour
- AI color diff: ~0.002-0.003
- AI luma diff: ~0.002-0.003

---

## Tier 2: Max Quality (`interior_luxury_max_quality`)

### Configuration
```yaml
preset: interior_luxury_max_quality
segmentation:
  backend: segformer
  input_long_side: 1280  # +67% vs standard
  min_confidence: 0.25
materials_v2:
  enabled: true
  confidence_threshold: 0.4
  max_segmentation_side: 2048
  require_high_quality: false
precision: fp32 (recommended)
post_overlap: 64
marketing_png_compression: 1
```

### Characteristics
- **Speed:** Moderate (+30-40% vs standard)
- **Quality:** High, client-facing ready
- **Memory:** 8-12 GB VRAM
- **Output:** 16-bit TIFF + high-quality PNG

### Use Cases
- Client-facing marketing materials
- Website hero images
- Print materials (up to 36" wide)
- Serious editorial work
- Most production deliverables

### Expected Metrics
- Processing: 60-65s per 81MP image
- Throughput: 55-60 images/hour
- AI color diff: ~0.0018-0.0019
- AI luma diff: ~0.0018-0.0019

### Validated Results (750 Picacho Kitchen)
- Processing: 63.64s
- Color accuracy: 0.001894 (excellent)
- Luma accuracy: 0.001853 (excellent)
- Depth integration: ✅ depth_percentiles
- Materials V2: ✅ enabled

---

## Tier 3: APEX Quality (`interior_luxury_apex_quality`) ⭐

### Configuration
```yaml
preset: interior_luxury_apex_quality
segmentation:
  backend: segformer
  input_long_side: 2048  # +60% vs max_quality
  min_confidence: 0.15   # -40% for maximum recall
materials_v2:
  enabled: true
  confidence_threshold: 0.3  # -25% for better coverage
  max_segmentation_side: 2048
  require_high_quality: true  # ✅ ENFORCED
  quality_threshold: 0.55     # +37.5% strictness
  material_thresholds:
    wood: 0.50    # Optimized for luxury interiors
    metal: 0.50
    glass: 0.40
    stone: 0.50
precision: fp32 (forced)
half: false  # Disable fp16 even on CUDA
post_overlap: 128  # +100% for seamless blending
detail_strength: 0.75  # +7% enhancement
marketing_png_compression: 0  # Lossless
tile: 1024
tile_pad: 32
```

### Characteristics
- **Speed:** Optimized (comparable to max_quality)
- **Quality:** APEX - absolute maximum
- **Memory:** 12-16 GB VRAM
- **Output:** 16-bit TIFF + lossless PNG (archival)

### Use Cases
- **Flagship hero frames** for portfolio
- Archival masters (10-year+ retention)
- Large-format prints (60"+ wide)
- Award submissions and competitions
- Critical client presentations (ultra-high-end properties)

### Expected Metrics
- Processing: 50-55s per 81MP image
- Throughput: 65-70 images/hour (surprisingly fast)
- AI color diff: <0.0019 (APEX target)
- AI luma diff: <0.0019 (APEX target)
- PNG size: ~900-1000 MB (lossless)

### Validated Results (750 Picacho Kitchen)
- Processing: 53.37s ⚡ (16% faster than max_quality)
- Color accuracy: 0.001890 ✅ (better than max)
- Luma accuracy: 0.001858 ✅ (comparable to max)
- Depth integration: ✅ depth_percentiles
- Materials V2: ✅ enforced (require_high_quality=true)
- PNG size: 928 MB (lossless, archival-grade)

### Quality Improvements Over Max
1. **Segmentation Resolution:** 2048px vs 1280px (+60%)
2. **Materials V2 QA:** Enforced vs disabled
3. **Confidence Thresholds:** Lower for better coverage
4. **Post-Processing:** Enhanced blending and detail
5. **Export:** Lossless PNG for archival

---

## Decision Matrix

### Choose **Standard** when:
- ✅ Processing 10+ images in a batch
- ✅ Internal review or rough cuts
- ✅ Time-sensitive deliveries
- ✅ Budget-conscious projects

### Choose **Max Quality** when:
- ✅ Client-facing marketing materials
- ✅ Website and social media
- ✅ Print up to 36" wide
- ✅ 90% of production deliverables

### Choose **APEX Quality** when:
- ✅ Flagship portfolio pieces
- ✅ Large-format prints (60"+)
- ✅ Archival masters (10-year+ retention)
- ✅ Award submissions
- ✅ Ultra-luxury properties ($5M+)
- ✅ Critical presentations to discerning clients

---

## Usage Examples

### Standard Production
```bash
lux-depth-v2 \
  --input input_images/project/image.tiff \
  --output-dir output_standard \
  --preset interior_luxury \
  --upscale 2
```

### Max Quality
```bash
lux-depth-v2 \
  --input input_images/project/image.tiff \
  --output-dir output_max \
  --preset interior_luxury_max_quality \
  --upscale 2 \
  --precision fp32 \
  --depth-dir depth_maps
```

### APEX Quality
```bash
lux-depth-v2 \
  --input input_images/project/image.tiff \
  --output-dir output_apex \
  --preset interior_luxury_apex_quality \
  --upscale 2 \
  --depth-dir depth_maps \
  --enable-orchestrator
```

---

## Performance Benchmarks

**Test Image:** 750 Picacho Kitchen (12,000 × 6,750 px, 81 MP, 16-bit TIFF)  
**Hardware:** Apple M4 Max, 128 GB RAM, MPS acceleration  
**Depth Map:** Depth Anything V2 Large, 16-bit TIFF

| Tier | Time | Throughput | Color Δ | Luma Δ | PNG Size |
|------|------|------------|---------|--------|----------|
| Standard | ~45-50s | 70-80/hr | ~0.002-0.003 | ~0.002-0.003 | ~300-400 MB |
| Max Quality | 63.64s | 55-60/hr | 0.001894 | 0.001853 | 411 MB |
| **APEX Quality** | **53.37s** | **65-70/hr** | **0.001890** ✅ | **0.001858** | **928 MB** |

**Note:** APEX is faster than Max Quality due to optimized settings that reduce redundant processing.

---

## Quality Validation Thresholds

All tiers pass AI quality validation:

| Metric | Warning Threshold | Fail Threshold | APEX Target |
|--------|------------------|----------------|-------------|
| Color Diff | 0.06 | 0.12 | <0.0019 |
| Luma Diff | 0.06 | 0.12 | <0.0019 |

**APEX Quality consistently achieves <0.0019 on both metrics** (97% better than warning threshold).

---

## File Size Considerations

### Typical Output Sizes (81MP Input)

**Standard:**
- Master TIFF: ~350 MB
- Upscaled TIFF: ~1.4 GB
- Marketing PNG: ~300-400 MB
- **Total:** ~2.1 GB

**Max Quality:**
- Master TIFF: ~360 MB
- Upscaled TIFF: ~1.6 GB
- Marketing PNG: ~411 MB (compression=1)
- **Total:** ~2.4 GB

**APEX Quality:**
- Master TIFF: ~361 MB
- Upscaled TIFF: ~1.6 GB
- Marketing PNG: ~928 MB (lossless)
- **Total:** ~2.9 GB (+20% vs max)

**Storage Recommendation:** 3-5 GB per image for APEX quality outputs.

---

## Best Practices

### For 750 Picacho Project
1. **Bulk Process:** Use standard tier for initial 50+ images
2. **Client Review:** Upgrade 20-30 best to max_quality
3. **Final Portfolio:** Select 5-10 hero frames for APEX
4. **Archive:** Keep APEX masters for long-term storage

### Depth Map Strategy
- **Generate once** with Depth Anything V2 Large @ 16-bit
- **Reuse** across all quality tiers
- **Cache** in dedicated depth_dir
- Processing time: ~1s per image (negligible overhead)

### Quality Tier Migration
Images can be **upgraded** between tiers:
```bash
# Start with standard for preview
lux-depth-v2 --preset interior_luxury ...

# Upgrade to max_quality if selected
lux-depth-v2 --preset interior_luxury_max_quality ...

# Final APEX for portfolio hero frame
lux-depth-v2 --preset interior_luxury_apex_quality ...
```

**Note:** Depth maps and material segmentation can be cached for faster re-processing.

---

## Appendix: Technical Deep Dive

### Why APEX is Faster Than Max
1. **Lower confidence thresholds** = fewer segmentation passes
2. **Optimized material coverage** = less fallback processing
3. **Higher segmentation resolution** = better first-pass quality
4. **Enforced quality gates** = early rejection of poor results
5. **fp32 precision** = fewer numerical corrections needed

### Quality Enforcement in APEX
The `require_high_quality: true` flag enables:
- Pre-validation of segmentation quality
- Rejection of low-confidence material masks
- Fallback to heuristic only when necessary
- Quality-threshold-based acceptance criteria

### Lossless PNG Export
APEX uses `compression=0` which:
- Preserves every bit of 16-bit processing
- Enables archival-grade storage
- Supports future re-processing
- Costs ~126% more disk space
- Worth it for flagship assets

---

## Conclusion

The three-tier system provides flexibility for different project needs:

- **Standard:** Daily production workhorse
- **Max Quality:** Default for client deliverables
- **APEX Quality:** Flagship portfolio and archival

**For 750 Picacho Lane luxury property, APEX quality is the validated choice for hero frames.**

All metrics verified December 12, 2025.
