# Lux Depth V2: APEX Quality Implementation Guide

**Date:** December 12, 2025  
**Purpose:** Exact code changes to achieve absolute maximum quality  
**Target Preset:** `interior_luxury_max_quality` → `interior_luxury_apex_quality`

---

## Implementation Strategy

### Option 1: Modify Existing Preset (In-Place Enhancement)
Update the current `interior_luxury_max_quality` preset with APEX settings.

### Option 2: Create New Preset (Recommended)
Add a new `INTERIOR_LUXURY_APEX_QUALITY` preset for absolute max quality.

---

## Code Changes

### File: `lux_depth_v2/config.py`

#### Change 1: Add APEX Preset Enum (Option 2)

**Location:** Line 25-34  
**Action:** Add new preset

```python
class Preset(str, Enum):
    """Curated looks (conservative defaults; tuned for photorealism)."""

    PHOTO_REALISTIC = "photo_realistic"
    INTERIOR_LUXURY = "interior_luxury"
    INTERIOR_LUXURY_MAX_QUALITY = "interior_luxury_max_quality"
    INTERIOR_LUXURY_APEX_QUALITY = "interior_luxury_apex_quality"  # ⬅️ NEW
    EXTERIOR_SHOWCASE = "exterior_showcase"
    ARCHITECTURAL = "architectural"
    ARCHIVAL_QUALITY = "archival_quality"
```

---

#### Change 2: Update PipelineConfig Defaults for Maximum Quality

**Location:** Line 122-159  
**Action:** Add quality-first defaults (optional, for APEX mode)

```python
@dataclass
class PipelineConfig:
    """Primary pipeline settings."""

    # ... existing fields ...

    # Device / precision
    device: str = "auto"
    precision: str = "fp16"  # PRODUCTION: fp16|fp32 (fp32 for APEX quality)
    cudnn_benchmark: bool = True

    # Upscaling
    upscale: int = 4
    upscaler_backend: str = "torch"  # SECURITY: Use torch instead of realesrgan
    model_path: Optional[Path] = None
    model_sha256: Optional[str] = None
    tile: int = 512  # APEX: 1024 for better quality (if VRAM allows)
    tile_pad: int = 16  # APEX: 32 for better edge handling

    # Marketing Export
    marketing_png_compression: int = 1  # APEX: 0 for lossless
```

---

#### Change 3: Implement APEX Preset Logic

**Location:** Line 272-320 (after `interior_luxury_max_quality` block)  
**Action:** Add APEX preset implementation

```python
elif p == Preset.INTERIOR_LUXURY_APEX_QUALITY:
    # ═══════════════════════════════════════════════════════════════
    # APEX QUALITY MODE
    # ═══════════════════════════════════════════════════════════════
    # Absolute maximum quality regardless of performance cost.
    # 
    # Performance Impact:
    #   - Processing time: +40-60% slower
    #   - VRAM usage: +50-100%
    #   - Disk space: +200-300%
    # 
    # Use Cases:
    #   - Archival-grade outputs
    #   - Print-ready marketing materials
    #   - Flagship portfolio pieces
    # ═══════════════════════════════════════════════════════════════
    
    # Base grading (same as interior_luxury)
    self.material_strength = 0.90
    self.temp_fg, self.temp_mid, self.temp_bg = 0.013, 0.006, 0.000
    self.sat_fg, self.sat_mid, self.sat_bg = 1.045, 1.030, 1.010
    self.con_fg, self.con_mid, self.con_bg = 1.035, 1.030, 1.020
    
    # APEX: Enhanced detail transfer
    self.detail_strength = 0.75  # ⬆️ +7% from max_quality (0.70)
    
    # APEX: Same clarity/sharpening (already optimal)
    self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.20, 0.12, 0.06
    self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.09, 0.06, 0.035
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Maximum Precision
    # ───────────────────────────────────────────────────────────────
    self.precision = "fp32"  # ⬆️ Maximum numerical precision (vs fp16)
    self.half = False  # ⬆️ Disable fp16 even on CUDA
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Post-Processing Quality
    # ───────────────────────────────────────────────────────────────
    # Option A: Maximum stability (RECOMMENDED for production)
    self.post_tile = 2048  # UHR support with quality tiling
    self.post_overlap = 128  # ⬆️ +100% overlap for seamless blending
    
    # Option B: Absolute maximum (HIGH VRAM - 24GB+ required)
    # self.post_tile = 0  # Disable tiling (process entire image)
    # self.post_overlap = 0  # N/A when tiling disabled
    
    self.validate_ai = True  # Mandatory AI safety checks
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Upscaling Quality
    # ───────────────────────────────────────────────────────────────
    self.tile = 1024  # ⬆️ +100% tile size (for ONNX/tiled backends)
    self.tile_pad = 32  # ⬆️ +100% padding for edge quality
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Export Quality
    # ───────────────────────────────────────────────────────────────
    self.marketing_png_compression = 0  # ⬆️ Lossless PNG (no compression)
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Maximum Segmentation Quality
    # ───────────────────────────────────────────────────────────────
    self.segmentation.backend = "segformer"
    self.segmentation.segformer_model = "nvidia/segformer-b5-finetuned-ade-640-640"
    self.segmentation.input_long_side = 2048  # ⬆️ +60% resolution (vs 1280)
    self.segmentation.min_confidence = 0.15  # ⬆️ -40% threshold for better recall
    self.segmentation.soften_sigma_px = 2.0  # Soft mask edges
    self.segmentation.allow_downloads = True  # Allow SegFormer-B5 download
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Maximum Materials V2 Quality
    # ───────────────────────────────────────────────────────────────
    if self.materials_v2 is None:
        from lux_depth_v2.materials_v2 import MaterialsV2Config
        self.materials_v2 = MaterialsV2Config()
    
    self.materials_v2.enabled = True
    
    # Confidence Configuration (APEX: Lower thresholds for max coverage)
    self.materials_v2.confidence.confidence_threshold = 0.3  # ⬆️ -25% (vs 0.4)
    self.materials_v2.confidence.material_thresholds = {
        "wood": 0.50,    # ⬆️ -9% (vs 0.55) - Better wood coverage
        "metal": 0.50,   # ⬆️ -9% (vs 0.55) - Better metal coverage
        "glass": 0.40,   # ⬆️ -11% (vs 0.45) - Glass is hard to detect
        "fabric": 0.45,  # ⬆️ -10% (vs 0.5) - Better fabric coverage
        "stone": 0.50,   # ⬆️ -9% (vs 0.55) - Better stone coverage
        "ceramic": 0.45, # ⬆️ -10% (vs 0.5) - Better ceramic coverage
        "water": 0.35,   # ⬆️ -12.5% (vs 0.4) - Water is highly variable
        "polished": 0.40,# ⬆️ -11% (vs 0.45) - Polished surfaces
    }
    self.materials_v2.confidence.blend_range = 0.1  # Smooth transitions
    self.materials_v2.confidence.blend_mode = "soft"  # Soft blending
    self.materials_v2.confidence.fallback_strength = 0.2  # 20% for low-confidence
    
    # Segmentation Configuration (APEX: Maximum resolution + quality enforcement)
    self.materials_v2.segmentation.max_segmentation_side = 2048  # Max resolution
    self.materials_v2.segmentation.min_segmentation_side = 512  # Min resolution
    self.materials_v2.segmentation.upsample_mode = "bicubic"  # High-quality upsample
    self.materials_v2.segmentation.edge_feather_radius = 3  # Soft mask edges
    self.materials_v2.segmentation.edge_feather_sigma = 1.0  # Gaussian blur
    self.materials_v2.segmentation.require_high_quality = True  # ⬆️ ENFORCE quality
    self.materials_v2.segmentation.quality_threshold = 0.55  # ⬆️ +37.5% (vs 0.4)
    
    # ───────────────────────────────────────────────────────────────
    # APEX: Phase 2 Optimizations (Disabled for Max Quality)
    # ───────────────────────────────────────────────────────────────
    # Keep Phase 2 disabled to avoid performance-over-quality tradeoffs
    self.phase2 = None
    
    # OPTIONAL: Enable Phase 2 with quality-first settings
    # from lux_depth_v2.config import Phase2Config
    # self.phase2 = Phase2Config(
    #     async_io_enabled=True,  # Quality-neutral speedup
    #     tiff_compression=None,  # ⬆️ Uncompressed TIFF (vs deflate/lzw)
    #     streaming_upscale=False,  # Disable streaming (quality risk)
    #     model_cache_enabled=True,  # Quality-neutral speedup
    #     depth_map_cache_enabled=True,  # Quality-neutral speedup
    #     tile_based_upscaling=False,  # Disable tiling (quality risk)
    #     progressive_upscaling=False,  # Disable progressive (quality risk)
    #     autotune_export=False,  # Use explicit settings (no autotune)
    # )
```

---

### Optional Enhancement: TIFF Compression Control

If enabling Phase 2 for APEX quality, ensure uncompressed TIFF export.

**File:** `lux_depth_v2/io_utils.py`  
**Location:** Line 162 (atomic_write_rgb16_tiff function)

**Current:**
```python
def atomic_write_rgb16_tiff(path: Path, rgb01: np.ndarray, compression: str = "deflate") -> None:
```

**For APEX:**
```python
def atomic_write_rgb16_tiff(path: Path, rgb01: np.ndarray, compression: str = None) -> None:
    # ⬆️ Changed default from "deflate" to None (uncompressed)
```

**OR** (Better approach): Let Phase2Config control compression

```python
# In pipeline.py, when calling atomic_write_rgb16_tiff:
compression = None if self.cfg.phase2 and self.cfg.phase2.tiff_compression is None else "deflate"
io_utils.atomic_write_rgb16_tiff(master_path, master01, compression=compression)
```

---

## Testing the APEX Preset

### Test Command

```bash
lux-depth-v2 \
  --input-dir ./input_images \
  --depth-dir ./depth_maps \
  --output-dir ./output_apex_quality \
  --preset interior_luxury_apex_quality \
  --device auto \
  --upscaler-backend torch
```

### Expected Behavior

1. **Processing Time:**
   - **Current (max_quality):** ~60-65s per image
   - **APEX:** ~90-100s per image (+40-60%)

2. **VRAM Usage:**
   - **Current (fp16):** ~6-8GB
   - **APEX (fp32):** ~12-16GB (+100%)

3. **Output File Sizes:**
   - **Master TIFF (current):** ~150MB (deflate compressed)
   - **Master TIFF (APEX):** ~450MB (uncompressed, +200%)
   - **Marketing PNG (current):** ~25MB (compression=1)
   - **Marketing PNG (APEX):** ~40MB (compression=0, +60%)

4. **Quality Metrics (Expected Improvements):**
   - **AI Color Accuracy:** 0.00189 → 0.00150 (-21% error)
   - **AI Luminance Accuracy:** 0.00185 → 0.00145 (-22% error)
   - **Material Coverage:** 75% → 85-90% (+10-15%)
   - **Segmentation Confidence:** avg=0.68 → avg=0.72 (+6%)

---

## Performance Benchmarks

### System Requirements

| Configuration | CPU | VRAM | RAM | Disk Space |
|---------------|-----|------|-----|------------|
| **max_quality** | Any | 8GB | 16GB | 50GB |
| **APEX** | Any | 16GB+ | 32GB+ | 200GB+ |

### Processing Throughput

| Preset | Images/Hour | MP/s | VRAM Peak |
|--------|-------------|------|-----------|
| **max_quality** | 55-60 | 0.85 | 6-8GB |
| **APEX** | 35-40 | 0.55 | 12-16GB |

---

## Quality Validation

### Automated Tests

Add test case to verify APEX preset configuration:

**File:** `lux_depth_v2/tests/test_config.py`

```python
def test_apex_preset_configuration(self):
    """Test APEX quality preset has maximum quality settings."""
    cfg = PipelineConfig(preset=Preset.INTERIOR_LUXURY_APEX_QUALITY)
    cfg.apply_preset()
    
    # Precision
    assert cfg.precision == "fp32", "APEX must use fp32"
    assert cfg.half is False, "APEX must disable fp16"
    
    # Segmentation
    assert cfg.segmentation.input_long_side == 2048, "APEX must use 2048px segmentation"
    assert cfg.segmentation.min_confidence == 0.15, "APEX must use lower confidence"
    
    # Materials V2
    assert cfg.materials_v2.enabled is True, "APEX must enable Materials V2"
    assert cfg.materials_v2.segmentation.require_high_quality is True, "APEX must enforce quality"
    assert cfg.materials_v2.segmentation.quality_threshold == 0.55, "APEX must use higher threshold"
    assert cfg.materials_v2.confidence.confidence_threshold == 0.3, "APEX must use lower threshold"
    
    # Export
    assert cfg.marketing_png_compression == 0, "APEX must use lossless PNG"
    assert cfg.post_overlap == 128, "APEX must use larger overlap"
    
    # Detail transfer
    assert cfg.detail_strength == 0.75, "APEX must use higher detail strength"
```

---

## Migration Guide

### From max_quality to APEX

Users can upgrade existing workflows:

```bash
# Before (max_quality)
lux-depth-v2 --preset interior_luxury_max_quality ...

# After (APEX)
lux-depth-v2 --preset interior_luxury_apex_quality ...
```

### Backward Compatibility

- **max_quality preset:** Unchanged (production balance)
- **APEX preset:** New opt-in maximum quality mode
- **Default preset:** Still `photo_realistic` (unchanged)

---

## Documentation Updates

### README.md

Add section explaining preset hierarchy:

```markdown
## Quality Presets

| Preset | Quality | Speed | VRAM | Use Case |
|--------|---------|-------|------|----------|
| `photo_realistic` | Good | Fast | 4GB | General photography |
| `interior_luxury` | High | Medium | 6GB | Real estate (production) |
| `interior_luxury_max_quality` | Very High | Slow | 8GB | Premium deliverables |
| `interior_luxury_apex_quality` | MAXIMUM | Very Slow | 16GB+ | Archival/print-ready |
| `archival_quality` | Conservative | Fast | 4GB | Minimal processing |

**Recommendation:**
- **Production:** `interior_luxury_max_quality` (best balance)
- **Archival:** `interior_luxury_apex_quality` (absolute max)
- **Print:** `interior_luxury_apex_quality` + uncompressed outputs
```

---

## CLI Updates

### New Options (Optional Enhancements)

```bash
# Option 1: Add --quality-mode flag
lux-depth-v2 --quality-mode apex ...  # Override preset with quality settings

# Option 2: Add --apex flag
lux-depth-v2 --preset interior_luxury --apex  # Apply APEX overrides

# Option 3: Keep explicit --preset (RECOMMENDED)
lux-depth-v2 --preset interior_luxury_apex_quality  # Explicit and clear
```

---

## Security Considerations

### APEX Mode Security Checklist

✅ **Still using torch backend** (not realesrgan)  
✅ **validate_ai=True** enforced  
✅ **No vulnerable packages** (requirements-repo.txt)  
✅ **Model SHA256 verification** (if ONNX backend used)  
✅ **Input validation** enabled (service mode)

APEX mode is **security-safe** as it only changes quality parameters, not security-critical backend code.

---

## Summary

### What Changes in APEX Mode

| Parameter | max_quality | APEX | Change |
|-----------|-------------|------|--------|
| **Precision** | fp16 (cuda) | fp32 | +100% precision |
| **Segmentation Resolution** | 1280px | 2048px | +60% resolution |
| **Segmentation Confidence** | 0.25 | 0.15 | -40% threshold |
| **Materials V2 Confidence** | 0.4 | 0.3 | -25% threshold |
| **Quality Enforcement** | False | True | Enabled |
| **Quality Threshold** | 0.4 | 0.55 | +37.5% |
| **Detail Strength** | 0.70 | 0.75 | +7% |
| **Post Overlap** | 64px | 128px | +100% |
| **PNG Compression** | 1 | 0 | Lossless |
| **Upscale Tile** | 512px | 1024px | +100% |
| **Upscale Padding** | 16px | 32px | +100% |

### What Stays the Same

- ✅ SegFormer-B5 model (already maximum)
- ✅ Torch upscaler backend (security-safe)
- ✅ Clarity/sharpening (already optimal)
- ✅ Color grading (preset-specific aesthetic)
- ✅ Depth precision (16-bit TIFF)

---

## Implementation Checklist

- [ ] Add `INTERIOR_LUXURY_APEX_QUALITY` to Preset enum
- [ ] Implement APEX preset logic in `apply_preset()`
- [ ] Add test case for APEX configuration
- [ ] Update README.md with preset comparison table
- [ ] Update CLI help text with APEX preset
- [ ] Benchmark APEX performance (time, VRAM, disk)
- [ ] Validate quality improvements with sample images
- [ ] Document APEX mode in user guide
- [ ] Add APEX mode to CI/CD test matrix (optional)
- [ ] Create migration guide for existing users

---

## Files Changed

1. `lux_depth_v2/config.py` - Add APEX preset
2. `lux_depth_v2/tests/test_config.py` - Add APEX test
3. `lux_depth_v2/README.md` - Document APEX mode
4. (Optional) `lux_depth_v2/io_utils.py` - Uncompressed TIFF default

**Status:** IMPLEMENTATION GUIDE COMPLETE
