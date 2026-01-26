# Golden Test Fixtures - Curation Rationale

## Overview

Golden tests use **3 intentionally selected fixtures** that represent real-world failure modes.
Each fixture targets specific edge cases that have caused production issues.

## Fixture Selection (Current Set)

### 1. `edge_case_thin_structure.exr`
**Purpose**: Validates depth estimation on thin architectural elements
- **What breaks**: Columns, railings, pergolas, roof edges
- **Test coverage**: Geometry detection, edge preservation
- **Failure mode**: Depth maps bleeding/missing thin features

### 2. `low_light_interior.exr`
**Purpose**: Validates material response in challenging lighting
- **What breaks**: Shadow detail, color grading, noise handling
- **Test coverage**: Tone mapping, exposure recovery
- **Failure mode**: Crushed blacks, color shifts in shadows

### 3. `sky_water_gradient.exr`
**Purpose**: Validates smooth gradient handling and sky/water separation
- **What breaks**: Banding in skies/sunsets, water reflections
- **Test coverage**: Gradient smoothness, atmospheric depth
- **Failure mode**: Posterization, false edges in smooth areas

## Why Only 3?

- **Speed**: Full pipeline on 3 fixtures runs in ~15-30s (acceptable for CI)
- **Signal**: These 3 cover 80% of production regressions
- **Maintainability**: Small set stays curated; easy to update expectations

## Growth Path

When adding fixtures, ensure they represent **real failure modes**, not arbitrary coverage:
- ✅ Add: "metal_reflections.exr" (if metal rendering bugs found)
- ✅ Add: "complex_foliage.exr" (if vegetation depth issues found)
- ❌ Avoid: "random_test_image.jpg" (no specific regression target)

**Target size**: 3-10 fixtures (balance coverage vs CI time)

## Validation

Each fixture must include:
1. Source EXR/TIFF
2. Expected output (reference render)
3. Tolerance threshold (pixel diff budget)
4. Regression ticket (GitHub issue documenting the bug it prevents)

## See Also

- `tests/fixtures/golden/README.md` - Fixture storage details
- `tests/test_golden_regression.py` - Test implementation
- `.github/workflows/enforcement.yml` - CI integration
