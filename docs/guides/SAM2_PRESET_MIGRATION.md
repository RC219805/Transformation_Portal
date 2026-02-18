# SAM2 Preset Migration Guide

**Date:** 2026-02-18
**Phase:** 4E (Promotion to Stable)

## Overview

SAM2 segmentation presets have been promoted from experimental to stable/canary tiers after completing Phase 4 validation.

## Preset Taxonomy

### Production Presets (New)

**✅ `sam2_segmentation` (Stable)**
- **Use for:** Production workflows
- **Stability:** Stable, validated
- **Version:** 4.0.0
- **Status:** Production-ready
- **Tests:** 42/42 passing
- **Performance:** Quality Firewall validated
- **Location:** `config/presets/sam2_segmentation.yaml`

**✅ `sam2_segmentation_canary` (Canary)**
- **Use for:** Pre-production testing
- **Stability:** Canary (testing before stable)
- **Version:** 4.0.0-canary
- **Status:** Feature-complete, validation pending
- **Features:** All stable features + full config options
- **Location:** `config/presets/sam2_segmentation_canary.yaml`

### Development Preset

**⚠️ `experimental/sam2_segmentation` (Experimental)**
- **Use for:** Development and testing only
- **Stability:** Experimental
- **Version:** 4.0.0-dev
- **Status:** Development features, relaxed quality gates
- **Location:** `config/presets/experimental/sam2_segmentation.yaml`

## Migration Paths

### From Experimental to Stable

**Old (Phase 3):**
```bash
transformation_portal spatial-ai segment \
  --preset experimental/sam2_segmentation \
  --input scene.tiff \
  --output output/
```

**New (Phase 4E - Stable):**
```bash
transformation_portal spatial-ai segment \
  --preset sam2_segmentation \
  --input scene.tiff \
  --output output/
```

**Changes:**
- ✅ No breaking changes
- ✅ All parameters backward compatible
- ✅ Material classification now available (disabled by default)
- ✅ Performance validated

### From Experimental to Canary

Use canary for early access to validated features before stable promotion:

```bash
transformation_portal spatial-ai segment \
  --preset sam2_segmentation_canary \
  --input scene.tiff \
  --output output/
```

## Feature Comparison

| Feature | Experimental | Canary | Stable |
|---------|-------------|--------|--------|
| Auto mode | ✅ | ✅ | ✅ |
| Prompted mode (points/bbox) | ✅ | ✅ | ✅ |
| Video tracking | ✅ | ✅ | ✅ |
| Material classification | ✅ | ✅ | ✅ (opt-in) |
| Quality Firewall | Shadow | Enforced | Enforced |
| Test coverage | 42/42 | 42/42 | 42/42 |
| Production-ready | ❌ | ✅ | ✅ |
| Stability guarantee | None | Canary | Stable |

## Breaking Changes

**None.** All presets are backward compatible with Phase 3 configurations.

## New Features (Phase 4)

### Phase 4A: Video Tracking
```yaml
segmentation:
  mode: "video"  # NEW: temporal tracking

temporal:
  enabled: true  # Enable video mode
  iou_threshold: 0.5
```

### Phase 4D: Material Classification
```yaml
material_classification:
  enabled: true  # NEW: CLIP-based labeling
  confidence_threshold: 0.3
```

## Deprecation Notice

**⚠️ `experimental/sam2_segmentation` remains available for development** but should not be used in production. It will continue to receive experimental features before promotion to canary/stable.

## Upgrade Checklist

- [ ] Review new preset options (canary vs stable)
- [ ] Test with canary preset in staging
- [ ] Update CI/CD to use stable preset
- [ ] Update documentation references
- [ ] Remove experimental preset from production configs

## Performance Characteristics

All presets share the same baseline performance (Phase 4C validated):

**Auto mode (512x512, MPS, large model):**
- Mean latency: 13.38s
- P95 latency: 13.38s
- Peak memory: 1.67 GB
- Quality Firewall thresholds: +15% mean, +10% p95

**With material classification enabled:**
- +2-3s overhead per image
- +500MB memory for CLIP model

## Rollback Plan

If issues arise after migration to stable/canary:

1. Revert to experimental preset:
   ```bash
   --preset experimental/sam2_segmentation
   ```

2. Report issue to development team

3. Experimental preset maintains Phase 3 behavior

## Support

- **Stable preset issues:** File as production bugs
- **Canary preset issues:** File as canary feedback
- **Experimental preset issues:** File as development notes

## References

- Phase 4E Promotion: [commit 052bd384]
- SAM2 Integration Guide: `docs/guides/SAM2_INTEGRATION_GUIDE.md`
- Performance Baselines: `docs/performance/sam2_benchmarks.md`
- Test Suite: `tests/spatial_ai/segmentation/`

## FAQ

**Q: Should I use stable or canary?**
A: Use stable for production. Use canary for pre-production testing of new features.

**Q: Will experimental preset be removed?**
A: No, it remains for development. Do not use in production.

**Q: Are there any breaking changes?**
A: No, all presets are backward compatible.

**Q: What if I need custom features?**
A: Fork the canary preset and customize. Experimental preset allows more flexibility.

**Q: How do I enable material classification?**
A: Set `material_classification.enabled: true` in your config.

**Q: What's the performance impact of stable vs experimental?**
A: Same performance. Quality gates differ (stable is stricter).
