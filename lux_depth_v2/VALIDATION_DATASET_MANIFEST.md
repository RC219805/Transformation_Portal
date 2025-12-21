# Validation Dataset Manifest

**Purpose**: Ensure validation dataset integrity over time  
**Created**: December 21, 2025  
**Status**: LOCKED during Week 2-3 validation

---

## Dataset Definition

This manifest locks the validation dataset for Phase 2 edge refinement testing. Changes to this manifest or the images it references require explicit approval via feature freeze template.

---

## Representative Test Images (10 required)

**Naming Convention**: `{type}_{scene}_{id}_[SENTINEL].tiff`
- Sentinel images flagged for future regression testing
- SHA256 checksums lock dataset integrity

### Interior Scenes (4 images)

| ID | Scene Type | Critical Feature | Edge Case | Filename | SHA256 | Sentinel | Status |
|----|------------|------------------|-----------|----------|--------|----------|--------|
| 1 | Interior bedroom | Glass partition/shower | Glass-on-glass boundaries | interior_bedroom_01_SENTINEL.tiff | TBD | ✅ | 📋 Pending |
| 2 | Interior kitchen | Backsplash tiles | High-frequency edges | interior_kitchen_01.tiff | TBD | - | 📋 Pending |
| 3 | Interior great room | Large windows | Mixed depth zones | interior_great_room_01.tiff | TBD | - | 📋 Pending |
| 4 | Interior bathroom | Mirror surfaces | Specular highlights | interior_bathroom_01.tiff | TBD | - | 📋 Pending |

### Exterior Scenes (4 images)

| ID | Scene Type | Critical Feature | Edge Case | Filename | SHA256 | Sentinel | Status |
|----|------------|------------------|-----------|----------|--------|----------|--------|
| 5 | Exterior pool | Water/deck edge | Depth discontinuity | exterior_pool_01.tiff | TBD | - | ✅ Available |
| 6 | Exterior facade | White stucco | Low-contrast edges | exterior_facade_01_SENTINEL.tiff | TBD | ✅ | 📋 Pending |
| 7 | Exterior courtyard | Railing/horizon | Thin structures | exterior_courtyard_01.tiff | TBD | - | 📋 Pending |
| 8 | Exterior garden | Foliage | High-frequency organic | exterior_garden_01.tiff | TBD | - | 📋 Pending |

### Aerial Scenes (2 images)

| ID | Scene Type | Critical Feature | Edge Case | Filename | SHA256 | Sentinel | Status |
|----|------------|------------------|-----------|----------|--------|----------|--------|
| 9 | Aerial exterior | Roof edges | Multi-scale edges | aerial_exterior_01.tiff | TBD | - | 📋 Pending |
| 10 | Twilight exterior | Low light | Challenging illumination | twilight_exterior_01.tiff | TBD | - | 📋 Pending |

---

## Sentinel Images (Regression Tests)

**Purpose**: Known failure modes for future CI regression detection

**Designated Sentinels** (2 minimum):
1. **`interior_bedroom_01_SENTINEL.tiff`**
   - Known failure: Glass-on-glass boundary detection
   - Why sentinel: Consistent across lighting conditions
   - Use case: Bathroom/shower glass edge refinement

2. **`exterior_facade_01_SENTINEL.tiff`**
   - Known failure: Low-contrast white stucco edges
   - Why sentinel: Common in luxury real estate
   - Use case: Mediterranean/Spanish architecture

**Sentinel Criteria**:
- Represents known failure mode
- Reproducible across runs
- Representative of common use case
- Suitable for automated CI regression checks

---

## Validation Matrix

For each image:
- **Baseline**: No edge refinement
- **Subtle**: `--edge-refinement --refinement-preset subtle`
- **Balanced**: `--edge-refinement --refinement-preset balanced`
- **Aggressive**: `--edge-refinement --refinement-preset aggressive`

**Total**: 10 images × 4 configs = **40 test runs**

---

## Integrity Checks

### Manual Verification (Week 2)
```bash
# Generate checksums for all validation images
cd validation_images/
shasum -a 256 *.tiff > ../VALIDATION_CHECKSUMS.txt
```

### CI Enforcement (Future Enhancement)
```yaml
# .github/workflows/validation-integrity-check.yml
# Fail if validation images change without approval
- name: Verify validation dataset
  run: |
    shasum -a 256 -c VALIDATION_CHECKSUMS.txt
```

---

## Modification Policy

### During Feature Freeze (Dec 20 - Jan 10)
- ❌ **BLOCKED**: Adding/removing images
- ❌ **BLOCKED**: Changing image checksums
- ✅ **ALLOWED**: Updating status column (TBD → ✅ Available)
- ✅ **ALLOWED**: Adding SHA256 checksums

### After Freeze (Jan 10+)
- Requires: Feature freeze check template
- Approval: Architecture review required
- Rationale: Must document why baseline changed

---

## Dataset Acquisition

### Sources
- Client projects (anonymized, permission granted)
- Public datasets (CC-BY or CC0 licensed)
- Synthetic renders (for specific edge cases)

### Requirements
- **Format**: 16-bit TIFF or 8-bit PNG
- **Resolution**: Minimum 2048px on long side
- **Quality**: Archival-grade, no compression artifacts
- **Licensing**: Clear for use in validation/testing

---

## Validation Results Tracking

### Output Directory Structure
```
validation_results/
├── baseline/
│   ├── image_001_baseline/
│   ├── image_002_baseline/
│   └── ...
├── subtle/
│   ├── image_001_subtle/
│   └── ...
├── balanced/
│   ├── image_001_balanced/
│   └── ...
└── aggressive/
    ├── image_001_aggressive/
    └── ...
```

### Metrics to Collect
- Edge F1 score (Canny edge detection baseline)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Processing time
- Peak memory usage
- Visual quality assessment (manual)

---

## Change Log

| Date | Change | Approver | Reason |
|------|--------|----------|--------|
| 2025-12-21 | Initial manifest created | Architecture | Gap 2 remediation |
| TBD | Image checksums added | TBD | Week 2 dataset gathering |

---

**Status**: Locked for Week 2-3 validation  
**Next Review**: After validation completion (Dec 27, 2025)
