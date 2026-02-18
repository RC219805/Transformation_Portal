# Material PBR Migration Guide

**Version:** 5.0.0
**Migration Path:** Experimental → Stable
**Last Updated:** 2026-02-18

---

## Overview

This guide documents the migration path from experimental PBR implementations to the stable v5.0.0 Material PBR Integration. It covers API changes, preset upgrades, performance expectations, and rollback procedures.

---

## Migration Checklist

### Pre-Migration Assessment

- [ ] Identify current PBR usage (scripts, pipelines, notebooks)
- [ ] Verify Python 3.11+ installed (minimum supported version)
- [ ] Review current preset configurations
- [ ] Baseline current performance metrics
- [ ] Document custom material presets (if any)
- [ ] Identify dependencies on experimental features
- [ ] Plan testing strategy for migrated code

### Migration Execution

- [ ] Update preset references (experimental → stable)
- [ ] Update API calls (tuple unpacking → structured return)
- [ ] Validate material hints against new preset taxonomy
- [ ] Update CI/CD pipelines
- [ ] Run regression tests
- [ ] Verify output quality (visual inspection)
- [ ] Performance validation (meets Quality Firewall)
- [ ] Update documentation/runbooks

### Post-Migration Validation

- [ ] All tests passing (unit + integration)
- [ ] Production validation on sample TIFFs
- [ ] Performance within expected ranges
- [ ] No quality regressions vs baseline
- [ ] Monitoring/alerting updated
- [ ] Team training completed

---

## Breaking Changes Summary

### ⚠️ API Changes (Breaking)

**Old (pre-v5.0.0, now breaks):**
```python
albedo, normal, roughness, metallic, ao, height, properties = backend.generate_pbr_textures(rgb)
```

**New (Recommended):**
```python
result = backend.generate_pbr_textures(rgb)
albedo = result.albedo
normal = result.normal
# ... etc
```

**Migration Strategy:**
- Tuple unpacking is no longer supported in v5.0.0+
- Update all call sites to use `PBRTextures` attributes
- Expected error on old usage: `TypeError: cannot unpack non-iterable PBRTextures object`

### ✅ Non-Breaking Changes

- Preset taxonomy (stable/canary/experimental) - additive only
- Material hint names - backward compatible
- Backend selection logic - same API
- Device placement - auto-detection unchanged

---

## Preset Migration

### From: Experimental Preset

**Old Path:** `config/presets/material_pbr_experimental.yaml`

**Migration Options:**

#### Option 1: Stable (Recommended for Production)

```yaml
# config/presets/material_pbr.yaml
version: "5.0.0"
tier: stable
backend: heuristic  # CPU-only, deterministic
```

**Use when:**
- Production deployments
- Determinism required
- CPU-only environments
- Commercial licensing needed

**Changes from experimental:**
- No GPU dependencies (heuristic only)
- Locked preset (SHA256 CI enforcement)
- Documented Quality Firewall thresholds

#### Option 2: Canary (For Early Adopters)

```yaml
# config/presets/material_pbr_canary.yaml
version: "5.0.0-canary"
tier: canary
backend: pbr_fusion  # GPU primary, CPU fallback
```

**Use when:**
- GPU acceleration available
- Higher quality requirements
- Willing to track canary changes
- Auto-fallback acceptable

**Changes from experimental:**
- PBRFusion backend (Apache 2.0)
- Requires `PBRFUSION_PATH` environment variable
- Auto-falls back to heuristic if unavailable

#### Option 3: Stay on Experimental (Not Recommended)

**Experimental preset still exists but:**
- No stability guarantees
- May break without notice
- Not covered by Quality Firewall
- Use only for research/prototyping

---

## Code Migration Examples

### Example 1: Basic PBR Generation

**Before:**
```python
from transformation_portal.spatial_ai.materials import MaterialBackend

backend = MaterialBackend(backend="heuristic", device="cpu")
albedo, normal, roughness, metallic, ao, height, properties = backend.generate_pbr_textures(rgb)

# Use textures...
save_texture(albedo, "albedo.png")
```

**After (Recommended):**
```python
from transformation_portal.spatial_ai.materials import MaterialBackend

backend = MaterialBackend(backend="heuristic", device="cpu")
result = backend.generate_pbr_textures(rgb)

# Access via structured return
save_texture(result.albedo, "albedo.png")
save_texture(result.normal, "normal.png")

# Access metadata (NEW)
print(f"Backend: {result.metadata.backend_version}")
print(f"Normal scale: {result.metadata.normal_scale}")
```

**Migration Effort:** Low (15 minutes per script)

---

### Example 2: Material-Specific Processing

**Before:**
```python
# Experimental: material hints were unstable
result = backend.generate_pbr_textures(rgb, material_hint="wood_floor")  # May not exist
```

**After:**
```python
# Stable: PBR-accurate material taxonomy
result = backend.generate_pbr_textures(rgb, material_hint="wood")  # Standardized

# Supported materials (v5.0.0):
# - wood, stone, metal, glass, fabric, concrete, plastic, ceramic
```

**Migration Effort:** Low (validate material hint names)

---

### Example 3: Batch Processing Pipeline

**Before:**
```python
# Experimental: No performance guarantees
for image_path in images:
    backend = MaterialBackend(backend="heuristic")  # Re-init per image (inefficient)
    albedo, *_ = backend.generate_pbr_textures(load_image(image_path))
```

**After:**
```python
# Stable: Optimized batch processing
backend = MaterialBackend(backend="heuristic", device="cpu")  # Init once

for image_path in images:
    rgb = load_image(image_path)
    result = backend.generate_pbr_textures(rgb)

    # Meets Quality Firewall: <5s/MP @ 12MP
    save_pbr_textures(result, output_path)
```

**Migration Effort:** Medium (refactor batch loops)

---

### Example 4: Integration with SAM2

**Before:**
```python
# Experimental: Manual material detection
masks = segmenter.segment(image)
for mask in masks:
    # Guess material from RGB statistics
    material = guess_material(image, mask)  # Custom logic
    result = backend.generate_pbr_textures(image * mask, material_hint=material)
```

**After:**
```python
# Stable: Formal material classifier integration
from transformation_portal.spatial_ai.materials import MaterialClassifier

masks = segmenter.segment(image)
classifier = MaterialClassifier()
material_labels = classifier.classify(image, masks)

for mask, material in zip(masks, material_labels):
    result = backend.generate_pbr_textures(
        rgb=image * mask[..., None],
        material_hint=material,  # CLIP-based classification
        depth=depth_map          # NEW: geometry-aware
    )
```

**Migration Effort:** Medium (integrate MaterialClassifier)

---

## Performance Comparison

### Experimental vs Stable Baselines

| Metric                  | Experimental (v4.x) | Stable (v5.0.0) | Change  |
|-------------------------|---------------------|-----------------|---------|
| Small TIFF (0.06 MP)    | ~0.4s               | 0.29s           | 🟢 -28% |
| Large TIFF (12 MP)      | ~60s                | 51.39s          | 🟢 -14% |
| Latency (s/MP)          | ~5.0-6.0            | 4.28            | 🟢 -15% |
| Memory (MB)             | ~600                | <500            | 🟢 -17% |
| Determinism             | ⚠️ Partial          | ✅ Bitwise      | 🟢      |
| Backend availability    | ⚠️ GPU-dependent    | ✅ Always (CPU) | 🟢      |

**Key Improvements:**
- Enhanced bilateral filtering (Phase 5C)
- Depth-aware normal generation (5× scale)
- Optimized concavity AO (70%/30% blend)
- Vectorized material preset lookups

---

## Feature Comparison Matrix

| Feature                        | Experimental | Stable v5.0.0 | Canary v5.0.0-canary |
|--------------------------------|--------------|---------------|----------------------|
| **Backend**                    |              |               |                      |
| Heuristic (CPU)                | ✅           | ✅            | ✅ (fallback)        |
| PBRFusion (GPU)                | ❌           | ❌            | ✅                   |
| NVDIFFREC                      | ❌           | ❌            | ❌ (future)          |
| **Features**                   |              |               |                      |
| Basic PBR generation           | ✅           | ✅            | ✅                   |
| Depth-aware processing         | ⚠️ Basic     | ✅ Enhanced   | ✅ Enhanced          |
| Material presets (count)       | 4            | 8             | 8                    |
| Bilateral filtering            | ❌           | ✅            | ✅                   |
| Concavity-based AO             | ⚠️ Simple    | ✅ Advanced   | ✅ Advanced          |
| Artifact metadata              | ❌           | ✅            | ✅                   |
| **Guarantees**                 |              |               |                      |
| Determinism                    | ⚠️           | ✅ Bitwise    | ⚠️ GPU-dependent     |
| Quality Firewall               | ❌           | ✅            | ✅                   |
| CI stability guard             | ❌           | ✅ SHA256     | ⚠️ Flexible          |
| Commercial licensing           | ⚠️           | ✅ Apache 2.0 | ✅ Apache 2.0        |
| Production support             | ❌           | ✅            | ⚠️ Canary            |

---

## Rollback Plan

### If Migration Fails

#### Option 1: Revert to Experimental Preset

```bash
# Emergency rollback (immediate)
git checkout main -- config/presets/material_pbr_experimental.yaml
python scripts/enhance_image.py --preset config/presets/material_pbr_experimental.yaml --input input.tif
```

**Risks:**
- Loses Phase 5C enhancements (bilateral, concavity AO)
- No Quality Firewall protection
- Experimental instability

#### Option 2: Pin to Stable with Custom Overrides

```yaml
# config/presets/material_pbr_custom.yaml
extends: config/presets/material_pbr.yaml  # Stable base

material_pbr:
  generation:
    # Override specific params while keeping stable foundation
    normal_scale: 3.0  # Lower than stable default (5.0)
```

**Recommended for:** Gradual migration, A/B testing

#### Option 3: Parallel Validation (Safest)

```bash
# Run both presets on same input
python scripts/enhance_image.py --preset config/presets/material_pbr_experimental.yaml --input test.tif --output output/exp/
python scripts/enhance_image.py --preset config/presets/material_pbr.yaml --input test.tif --output output/stable/

# Compare outputs
python tools/compare_pbr_outputs.py output/exp/ output/stable/
```

**Use for:** Risk-averse production migrations

---

## Testing Strategy

### Recommended Test Plan

#### Phase 1: Unit Testing (1 day)

```bash
# Run material tests
pytest tests/spatial_ai/materials/ -v

# Expected: 62/62 tests passing
# Validates: contracts, backends, material presets
```

#### Phase 2: Integration Testing (1 day)

```bash
# Test with SAM2 pipeline
python scripts/validate_pbr_phase5d.py \
  --input input_images/BECW0138.TIF \
  --preset config/presets/material_pbr.yaml

# Validate:
# - All 6 PBR maps generated
# - MaterialProperties populated
# - Performance within Quality Firewall
```

#### Phase 3: Production Validation (2 days)

```bash
# Test on representative luxury TIFF dataset
for tiff in production_samples/*.TIF; do
    python scripts/enhance_image.py \
      --preset config/presets/material_pbr.yaml \
      --input "$tiff" \
      --output "validation_output/"
done

# Validate:
# - Visual quality (manual inspection)
# - Performance consistency
# - Memory usage
# - No crashes/exceptions
```

#### Phase 4: Performance Regression Testing (1 day)

```bash
# Benchmark against baseline
python tools/performance_ledger.py \
  --test materials_heuristic_phase5c_production \
  --compare-to materials_experimental_v4

# Validate:
# - Latency regressions <10% (p95)
# - Memory regressions <15%
# - No failure rate increase
```

---

## Common Issues & Solutions

### Issue: "TypeError: cannot unpack non-iterable PBRTextures object"

**Symptom:**
```python
TypeError: cannot unpack non-iterable PBRTextures object
```

**Cause:** Using old tuple unpacking with new structured return

**Solution:**
```python
# ❌ Old (breaks)
albedo, normal, *_ = backend.generate_pbr_textures(rgb)

# ✅ New (works)
result = backend.generate_pbr_textures(rgb)
albedo = result.albedo
```

---

### Issue: "Material hint not recognized"

**Symptom:**
```python
Warning: Unknown material hint 'wood_floor', using defaults
```

**Cause:** Experimental material names not in stable taxonomy

**Solution:**
```python
# ❌ Experimental names
material_hint="wood_floor"      # Not in stable preset
material_hint="brushed_metal"   # Not in stable preset

# ✅ Stable taxonomy
material_hint="wood"   # Covers all wood types
material_hint="metal"  # Covers all metal types
```

**Valid hints:** wood, stone, metal, glass, fabric, concrete, plastic, ceramic

---

### Issue: "Performance regression vs experimental"

**Symptom:** Stable preset slower than experimental on specific images

**Possible Causes:**
1. Bilateral filtering overhead (Phase 5C enhancement)
2. Depth-aware normal computation (5× scale)
3. Advanced concavity AO (70%/30% blend)

**Solutions:**

**Option 1:** Disable optional enhancements
```yaml
# config/presets/material_pbr_lean.yaml
extends: config/presets/material_pbr.yaml

material_pbr:
  generation:
    bilateral_filtering: false  # Skip bilateral (faster)
    normal_scale: 3.0           # Lower scale (faster)
```

**Option 2:** Use canary preset (GPU acceleration)
```bash
export PBRFUSION_PATH=/path/to/pbrfusion
python scripts/enhance_image.py --preset config/presets/material_pbr_canary.yaml --input input.tif
```

---

### Issue: "Missing PBRGenerationMetadata"

**Symptom:**
```python
AttributeError: 'PBRTextures' object has no attribute 'metadata'
```

**Cause:** Using old experimental backend without metadata support

**Solution:** Upgrade to stable v5.0.0 (metadata guaranteed)

---

## Migration Support

### Resources

- **Technical Guide:** `docs/guides/MATERIAL_PBR_GUIDE.md`
- **Performance Baselines:** `docs/performance/PHASE5_PBR_BASELINES.md`
- **Architecture:** `docs/architecture/ADR-027-material-classification.md`
- **Protocol Spec:** `src/transformation_portal/spatial_ai/materials/protocol.py`

### Timelines

**Typical Migration:** 2-4 days (small codebase)
**Complex Migration:** 1-2 weeks (large production system)

**Recommended Timeline:**
- Week 1: Assessment + planning
- Week 2: Code migration + unit testing
- Week 3: Integration testing + validation
- Week 4: Production deployment + monitoring

---

## Success Criteria

### Migration Complete When:

- ✅ All scripts/pipelines using stable preset
- ✅ 100% test coverage passing
- ✅ Production validation on representative dataset
- ✅ Performance within Quality Firewall thresholds
- ✅ No quality regressions vs baseline
- ✅ Team trained on new API
- ✅ Documentation updated
- ✅ Monitoring/alerting updated
- ✅ Rollback plan tested and documented

---

## Version History

| Version | Date       | Changes                                    |
|---------|------------|--------------------------------------------|
| 5.0.0   | 2026-02-18 | Initial stable release                     |
| 4.x     | 2025-2026  | Experimental implementations (deprecated)  |

---

## License

Apache 2.0 (stable + canary presets)
