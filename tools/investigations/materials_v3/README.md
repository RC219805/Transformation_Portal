# Materials V3 Diagnostic Tools

**Purpose**: Analysis scripts used during Materials V3 development to diagnose pixel operations, sky/water detection, and rendering issues.

**Status**: Archive (historical diagnostic tools, preserved for reference)

---

## Overview

This directory contains the diagnostic scripts used to investigate and fix critical bugs during **Materials V3** development (November 2024 - February 2026).

All scripts are preserved as-is from the development cycle, with minimal updates for compatibility with the current codebase structure.

---

## Available Tools

### 1. `diagnose_sky_issue.py`

**Purpose**: Diagnose sky detection and color grading issues

**Original Use Case**: Investigated why sky regions in coastal properties were showing inconsistent color temperature and gradient artifacts.

**What It Does**:
- Loads a test image
- Runs sky detection heuristic
- Applies sky pixel operations (dehaze, gradient_smooth, temperature_shift)
- Saves diagnostic outputs (masks, before/after comparisons)
- Generates visual diff with amplified changes

**Usage**:
```bash
# Use default paths (original investigation)
python tools/investigations/materials_v3/diagnose_sky_issue.py

# Analyze a specific directory
python tools/investigations/materials_v3/diagnose_sky_issue.py \
  --input depth_maps/ \
  --output debug_sky/

# Analyze specific depth map files
python tools/investigations/materials_v3/diagnose_sky_issue.py \
  --input depth_maps/ \
  --files aerial_depth.png pool_depth.png

# Outputs (console):
#   Depth statistics (min, max, mean, percentiles)
#   Zone analysis (foreground/background/midground)
#   Sky region analysis (for aerial/pool images)
#   Metadata check (from JSON sidecar files)
```

**Key Findings** (from original investigation):
- Sky gradient smoothing needed to eliminate banding artifacts in sunset skies
- Temperature shift of +150K (warm) appropriate for luxury real estate
- Dehaze critical for coastal properties with marine layer fog

**Related Investigation**: [`sky_water_color_grading_analysis.md`](../../../docs/investigations/materials_v3/sky_water_color_grading_analysis.md)

---

### 2. `create_sky_comparison.py`

**Purpose**: Generate side-by-side visual comparisons for sky processing validation

**Original Use Case**: Used to validate that sky pixel operations were actually applying (diagnosed NumPy view aliasing bug when all diffs showed zero).

**What It Does**:
- Processes an image with and without sky operations
- Creates a 3-panel comparison grid (before, after, diff)
- Amplifies diff for visibility
- Saves high-resolution comparison for inspection

**Usage**:
```bash
python tools/investigations/materials_v3/create_sky_comparison.py \
  --before output_old/aerial.tiff \
  --after output_new/aerial.tiff \
  --output output/materials_v3/comparison_sky.png

# Disable sky region cropping (show full image)
python tools/investigations/materials_v3/create_sky_comparison.py \
  --before before.jpg \
  --after after.jpg \
  --output output/materials_v3/comparison_sky.png \
  --no-crop

# Output: comparison image with 3 panels (before, after, difference visualization)
```

**Key Findings** (from original investigation):
- Detected zero delta across all sky operations (NumPy view aliasing bug)
- Revealed that `.copy()` was missing in pixel ops executor
- Validated fix by showing non-zero delta after `.copy()` added

**Related Investigation**: [`edge_artifacts_primary_bedroom.md`](../../../docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md)

---

## Installation & Dependencies

### Requirements

These scripts use the Transformation Portal codebase and standard dependencies:

```bash
# From repository root
pip install -e .

# Or install specific dependencies
pip install numpy opencv-python matplotlib Pillow
```

### Python Version

- **Minimum**: Python 3.11+
- **Tested**: Python 3.11, 3.12

---

## Usage Patterns

### Pattern 1: Reproduce a Historical Bug

To understand how a bug manifested and was diagnosed:

1. **Read the investigation report** (in `docs/investigations/materials_v3/`)
2. **Find the corresponding script** (this directory)
3. **Run the script** with similar inputs to original investigation
4. **Compare outputs** to investigation findings

Example:
```bash
# Reproduce sky gradient banding issue
python tools/investigations/materials_v3/diagnose_sky_issue.py \
  --input test_images/sunset_sky.jpg \
  --output debug/

# Expected: Gradient banding visible in debug/before.png, resolved in debug/after.png
```

---

### Pattern 2: Debug a New Issue

To use these scripts as templates for new investigations:

1. **Copy script** and rename for your investigation
2. **Modify inputs/outputs** to match your scenario
3. **Add new diagnostic outputs** as needed
4. **Document findings** in a new investigation report

Example:
```bash
# Create new diagnostic script
cp tools/investigations/materials_v3/diagnose_sky_issue.py \
   tools/investigations/materials_v3/diagnose_glass_issue.py

# Edit to focus on glass material
# Run and document findings
```

---

### Pattern 3: Regression Testing

To validate that a historical bug has not regressed:

1. **Run original diagnostic script** with original inputs
2. **Compare outputs** to expected (documented in investigation)
3. **Assert key metrics** (e.g., delta > 0, no artifacts in specific regions)

Example:
```bash
# Regression test: NumPy view aliasing bug should not recur
python tools/investigations/materials_v3/create_sky_comparison.py \
  --input regression_tests/sky_baseline.jpg \
  --output regression_tests/current_sky.png

# Manually inspect: diff panel should show non-zero changes
# (If zero, view aliasing bug has returned!)
```

---

## Diagnostic Output Interpretation

### Visual Diff Panels

All comparison scripts generate 3-panel outputs:

```
┌─────────────┬─────────────┬─────────────┐
│   BEFORE    │    AFTER    │  DIFF (×10) │
│ (original)  │ (processed) │ (amplified) │
└─────────────┴─────────────┴─────────────┘
```

**What to look for**:

- **DIFF panel all black**: No changes applied (possible bug!)
- **DIFF panel shows edges only**: Feathering issue
- **DIFF panel shows uniform regions**: Global color shift
- **DIFF panel shows localized patches**: Material-specific ops working

---

### Mask Outputs

Sky detection scripts save intermediate masks:

- **`sky_mask.png`**: Detected sky region (white = sky, black = not sky)
  - Should show smooth boundaries (feathering applied)
  - Should cover expected sky regions (upper frame typically)
  - Should not bleed into non-sky (trees, buildings)

**Diagnostic questions**:

- Is mask too aggressive? (Check confidence threshold, bbox, heuristics)
- Is mask too conservative? (Adjust detection parameters)
- Are edges ragged? (Check feathering sigma)

---

## Script Maintenance

### Current Status

**These scripts are archived** - they are preserved as historical artifacts, not actively maintained.

**Expected compatibility**:

| Script | Compatibility | Notes |
|--------|--------------|-------|
| `diagnose_sky_issue.py` | ✅ Should work | May need path updates if repo reorganized |
| `create_sky_comparison.py` | ✅ Should work | Uses stable core APIs |

**If a script breaks**:

1. Check import paths (codebase may have reorganized)
2. Check API changes (Materials V3 APIs are stable, but upstream deps may change)
3. Consider script may be intentionally historical (snapshot of development state)

---

## Related Documentation

### Investigation Reports

- **[Materials V3 Investigations Index](../../../docs/investigations/materials_v3/README.md)**: All investigation reports
- **[Edge Artifacts](../../../docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md)**: Primary Bedroom case study
- **[Sky/Water Analysis](../../../docs/investigations/materials_v3/sky_water_color_grading_analysis.md)**: Color grading investigation

### Methodology

- **[Diagnostic Methodology](../../../docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md)**: How these scripts were used

### Production Code

- **[`pixel_ops_executor.py`](../../../src/transformation_portal/lux_depth_v3/pixel_ops_executor.py)**: Final implementation
- **[`bootstrap/sky_seed.py`](../../../src/transformation_portal/lux_depth_v3/bootstrap/sky_seed.py)**: Sky detection

---

## Contributing

**This is an archive directory** - scripts are preserved as-is for historical reference.

**If you want to add a new diagnostic script**:

1. Create it in this directory
2. Follow the naming pattern: `diagnose_<issue>_<material>.py`
3. Include a docstring explaining:
   - What issue it diagnoses
   - How to run it
   - What outputs it generates
   - Link to investigation report (if any)
4. Update this README with a new section

**Do NOT**:

- Modify historical scripts (preserve original diagnostic behavior)
- Remove scripts (even if outdated, they document development process)

---

## Extraction Context

These scripts were extracted from PR #934 as **Phase 2** of the extraction strategy:

- **Phase 1**: Depth identity fix (PR #936) - ✅ MERGED
- **Phase 2**: Investigation docs + tools - ⏳ CURRENT
- **Phases 3-6**: Config presets, ML upscaling, SAM2, production docs

See [PR #934 Extraction Strategy](../../../docs/project-status/PR934_EXTRACTION_STRATEGY.md) for full plan.

---

**Last Updated**: February 14, 2026
**Status**: ✅ Phase 2 extraction complete
