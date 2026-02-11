# APEX Research Ultra — Quick Implementation Guide

**ADR:** ADR-026
**Status:** Experimental
**Preset:** `config/presets/experimental/apex_research_ultra.yaml`

---

## Executive Summary

**APEX Research Ultra** is an experimental research-grade workflow that extends ADR-025 with:

1. **Multi-model depth ensemble** (Depth Pro + DA3 1.1 + DepthCrafter)
2. **SAM2 video segmentation** with temporal propagation
3. **Physics-based materials** (MaterialGAN or NVDIFFREC)
4. **Geometric verification** (3D Gaussian Splatting)
5. **Vision-language QA** (LLaVA-1.6 34B)

**Expected Quality:** ≥15% improvement over ADR-025
**Expected Performance:** 2.4x slower than ADR-025 (~8-20s for 4K)
**License:** Research-only (non-commercial)

---

## Quick Start

### 1. Prerequisites

**System Requirements:**
- High-end GPU/NPU (≥16GB VRAM)
- ≥32GB RAM
- Python 3.11+

**License Acknowledgment:**
You must explicitly acknowledge research-only usage:

```python
from transformation_portal.lux_depth_v3.config import EnhanceConfig

config = EnhanceConfig(
    preset="experimental/apex_research_ultra",
    non_commercial_ok=True,
    accept_research_tools_license=True,
    spatial_ai_linear_ingest=True
)
```

**Dependencies:**
See `docs/architecture/ADR-026-apex-research-ultra.md` for full installation guide.

---

### 2. Basic Usage (Single Image)

```bash
# Minimal command
python -m transformation_portal.lux_depth_v3 enhance \
  --input projects/example/source.tiff \
  --output output_ultra/ \
  --preset experimental/apex_research_ultra.yaml \
  --non-commercial-ok true \
  --accept-research-tools-license true
```

**Expected Output:**
```
output_ultra/
├── source_linear.exr                    # Float32 HDR intermediate
├── source_provenance.json               # Decode recipe + hashes
├── source_depth_ensemble.npz            # Multi-model depth + confidence
├── source_depth_variance.png            # Model agreement visualization
├── source_materials_v3_sam2.npz         # SAM2 segmentation
├── source_albedo.exr                    # Physics-based albedo (linear)
├── source_roughness.png                 # Physics-based roughness (16-bit)
├── source_metallic.png                  # Physics-based metallic (16-bit)
├── source_enhanced_apex_research_ultra.tiff  # Final output (16-bit)
├── source_quality_report.json           # LLaVA quality assessment
└── source_manifest.json                 # Full provenance manifest
```

---

### 3. Multi-View Workflow (with Geometric Verification)

For ≥3 views, 3D Gaussian Splatting reconstruction is automatically enabled:

```bash
python -m transformation_portal.lux_depth_v3 enhance \
  --input-dir projects/example/views/ \
  --output-dir output_ultra/ \
  --preset experimental/apex_research_ultra.yaml \
  --non-commercial-ok true \
  --accept-research-tools-license true
```

**Additional Output:**
```
output_ultra/
└── recon/
    ├── scene.ply                        # 3DGS point cloud
    ├── scene_meta.json                  # Poses + config + RMSE metrics
    ├── view_001_depth_consistency_error.png  # Heatmap
    ├── view_002_depth_consistency_error.png
    └── view_003_depth_consistency_error.png
```

**Quality Gate:**
- If geometric RMSE > 5%, warning is logged
- Scene is still processed (fail_on_high_rmse: false)

---

### 4. Video Workflow (with Temporal Consistency)

SAM2 enables temporal propagation for video sequences:

```bash
python -m transformation_portal.lux_depth_v3 enhance \
  --input-dir projects/example/video_frames/ \
  --output-dir output_ultra/ \
  --preset experimental/apex_research_ultra.yaml \
  --non-commercial-ok true \
  --accept-research-tools-license true \
  --video-mode true
```

**Key Features:**
- SAM2 tracks objects across frames (reduces flicker)
- DepthCrafter provides temporal depth consistency
- Material boundaries are temporally coherent

---

## Key Differences vs ADR-025 (Stable Research)

| Feature | ADR-025 (Stable) | ADR-026 (Ultra) |
|---------|------------------|-----------------|
| **Depth** | Depth Pro (single) | Ensemble (3 models) |
| **Segmentation** | SAM vit_h | SAM2 Hiera Large (video) |
| **Materials** | Heuristic PBR | Physics-based (MaterialGAN) |
| **Reconstruction** | None | 3DGS (optional, multi-view) |
| **Validation** | LLaVA-1.5 13B | LLaVA-1.6 34B |
| **Ingest** | 8-bit sRGB | Linear float32 (Spatial AI) |
| **Performance** | ~3.5s (4K) | ~8-20s (4K) |
| **Quality** | Baseline | +15% (target) |
| **Stability** | Stable | Experimental |

---

## Configuration Flags

### Required Flags (License Compliance)

```yaml
compliance:
  non_commercial_ok: true                        # REQUIRED
  accept_research_tools_license: true            # REQUIRED
  spatial_ai_linear_ingest_required: true        # REQUIRED
```

### Optional Customization

```yaml
# Depth ensemble weights (must sum to 1.0)
depth:
  models:
    - name: depth_pro
      weight: 0.5    # Increase for more metric accuracy
    - name: da3_1.1_nested_giant_large
      weight: 0.3    # Increase for more edge detail
    - name: depthcrafter
      weight: 0.2    # Increase for more temporal consistency

# Materials backend selection
materials:
  backend: materialgan    # Fast (200ms)
  # backend: nvdiffrec    # Slow (10s), physics-accurate

# Reconstruction RMSE threshold
reconstruction:
  depth_consistency_threshold: 0.05  # 5% = "plausible"
  fail_on_high_rmse: false           # Log warning only
```

---

## Quality Validation

### Automated Quality Assessment (LLaVA-1.6 34B)

Every image receives a multi-dimensional quality report:

```json
{
  "overall_score": 8.7,
  "dimensions": {
    "depth_plausibility": 9.2,
    "material_realism": 8.5,
    "enhancement_quality": 8.3,
    "architectural_correctness": 9.0
  },
  "flags": [
    "Minor halo detected near window edge (severity: low)"
  ],
  "recommendations": [
    "Consider reducing bilateral_sigma_spatial to 4.0"
  ]
}
```

**Interpretation:**
- **Overall ≥8.0:** Research-grade quality achieved
- **7.0-7.9:** Acceptable, review flags
- **<7.0:** Manual review required

---

## Fallback Behavior

All stages have graceful degradation:

1. **Depth Ensemble Failure** → Single Depth Pro → DA3 Metric Large
2. **SAM2 OOM** → SAM vit_h (ADR-025)
3. **MaterialGAN Failure** → Heuristic PBR (ADR-025)
4. **LLaVA 34B OOM** → LLaVA-1.5 13B (ADR-025)
5. **Reconstruction (<3 views)** → Skip stage

**Result:** Workflow always completes, degrades to ADR-025 quality in worst case.

---

## Performance Tuning

### Memory Optimization

```yaml
# Reduce VRAM usage (trade quality for memory)
depth:
  models:
    - name: depth_pro    # Keep (primary)
    - name: da3_1.1_nested_giant_large   # Remove (largest)
    # - name: depthcrafter  # Remove (video-only)

validation:
  backend: llava_1.5_13b  # Smaller model (8GB vs 16GB)
```

### Speed Optimization

```yaml
# Skip expensive stages
reconstruction:
  enabled: false  # Save ~12s

materials:
  backend: heuristic_pbr  # Save ~10s (MaterialGAN → heuristic)

validation:
  backend: none  # Skip validation (save ~2.5s)
```

**Result:** ~3.5s total (same as ADR-025)

---

## Troubleshooting

### Issue: "LicenseRestrictionError: requires non_commercial_ok=True"

**Solution:** Add required flags:

```bash
--non-commercial-ok true \
--accept-research-tools-license true
```

### Issue: "Out of memory (OOM) during depth ensemble"

**Solution:** Reduce ensemble to 2 models or use single Depth Pro:

```yaml
depth:
  backend: depth_pro  # Override ensemble
```

### Issue: "Reconstruction RMSE >10%, geometric consistency violated"

**Interpretation:** Multi-view geometry is inconsistent (camera poses may be inaccurate).

**Solution:**
1. Check input images are from same scene
2. Verify camera intrinsics (if provided)
3. Increase `depth_consistency_threshold` to 0.10 (more lenient)

### Issue: "SAM2 checkpoint not found"

**Solution:** SAM2 integration is experimental. Fallback to SAM vit_h:

```yaml
segmentation:
  backend: sam_vit_h  # Override SAM2
```

---

## Benchmarking

### Run Benchmark Suite

```bash
pytest tests/benchmarks/test_adr026_quality_validation.py \
  --benchmark-only \
  -v
```

**Expected Metrics (vs ADR-025):**
- Depth MAE: ≥15% reduction
- Material IoU: ≥10% improvement
- Geometric RMSE: <5% error
- LLaVA score: ≥8.0/10

---

## Promotion Criteria (Experimental → Stable)

**Requirements:**
1. ✅ ≥3 successful full-project validations
2. ✅ All benchmark metrics met (≥15% depth improvement, etc.)
3. ✅ Artist preference ≥70% (Ultra vs ADR-025)
4. ✅ No critical failures (crashes, license violations)

**Timeline:** 6-8 weeks (estimated)

---

## References

- **Full ADR:** `docs/architecture/ADR-026-apex-research-ultra.md`
- **Preset:** `config/presets/experimental/apex_research_ultra.yaml`
- **ADR-025 (Stable):** `docs/architecture/ADR-025-apex-research-workflow.md`
- **Spatial AI Foundation:** Issue #890, `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md`

---

## Example: Full Pipeline Execution

```bash
# Complete APEX Research Ultra workflow
python -m transformation_portal.lux_depth_v3 enhance \
  --input-dir projects/750_picacho/source_tiffs/ \
  --output-dir output_apex_ultra_750_picacho/ \
  --preset experimental/apex_research_ultra.yaml \
  --non-commercial-ok true \
  --accept-research-tools-license true \
  --device auto \
  --strict-mode true

# Expected timeline (M4 Ultra, 5x 4K TIFFs, multi-view):
# Stage 0: Linear ingest           ~0.8s  (5 files @ 150ms each)
# Stage 1: Depth ensemble           ~17.5s (5 files @ 3.5s each)
# Stage 2: SAM2 segmentation        ~6.0s  (5 files @ 1.2s each)
# Stage 3: MaterialGAN              ~1.0s  (5 files @ 200ms each)
# Stage 4: 3DGS reconstruction      ~12.0s (one-time, 5 views)
# Stage 5: Enhancement              ~4.0s  (5 files @ 800ms each)
# Stage 6: LLaVA validation         ~12.5s (5 files @ 2.5s each)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Total:                            ~53.8s

# Quality report (expected):
# - Depth MAE: 0.085 (vs 0.100 ADR-025) → 15% improvement ✅
# - Material IoU: 0.88 (vs 0.80 ADR-025) → 10% improvement ✅
# - Geometric RMSE: 3.2% (vs 5.0% threshold) → plausible ✅
# - LLaVA score: 8.7/10 (vs 7.0 threshold) → excellent ✅
```

---

**Status:** Experimental (v0.1.0)
**Last Updated:** 2026-02-11
**Author:** Transformation Portal Architect
