# Phase 2 Execution Plan: Investigation Documentation

**Created**: 2024-02-14
**Priority**: 📚 HIGH (diagnostic methodology documentation)
**Effort**: 2-3 hours
**Risk**: 🟢 LOW
**Value**: ⭐⭐⭐⭐
**Timeline**: Days 2-3 (execute in next 24-48 hours)

---

## Executive Summary

**Phase 2** extracts and organizes Materials V3 investigation reports and diagnostic scripts from the development cycle. Based on repository analysis, **most content already exists** in the current main branch. This phase focuses on:

1. **Organization**: Consolidating scattered investigation docs into a structured subdirectory
2. **Completion**: Extracting any remaining investigation files from development branches
3. **Documentation**: Creating comprehensive README files and cross-references
4. **Preservation**: Ensuring diagnostic methodology is well-documented for future use

**Key Finding**: The repository already contains the primary investigation documentation:
- ✅ 2 major investigation reports (Primary Bedroom artifacts, Sky/Water)
- ✅ 2 diagnostic scripts (diagnose_sky_issue.py, create_sky_comparison.py)
- ✅ Phase A/B completion reports with bug analysis
- ✅ Materials V3 implementation summaries

**What's Needed**: Organization, additional diagnostics from branches, and comprehensive indexing.

---

## Part 1: Content Inventory

### 1.1 Existing Investigation Documentation (Already in Main)

#### Investigation Reports (2 files)
1. **`docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md`** (80+ lines)
   - **Topic**: Edge artifacts and halos in Materials V3 processing
   - **Status**: ✅ Already extracted
   - **Value**: Documents V2 enhancement artifact amplification bug
   - **Key Finding**: Mask feathering fix failed, V2 is root cause

2. **`docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md`** (80+ lines)
   - **Topic**: Sky/water color degradation analysis
   - **Status**: ✅ Already extracted
   - **Value**: Confirms no degradation, documents expected behavior
   - **Key Finding**: Color changes are preset-based enhancements, not bugs

#### Diagnostic Scripts (2 files, root level)
3. **`diagnose_sky_issue.py`** (5.4 KB)
   - **Purpose**: Analyzes depth maps to diagnose sky region issues
   - **Status**: ⚠️ Needs relocation to `tools/investigations/`
   - **Dependencies**: PIL, numpy (core deps)
   - **Usage**: Depth map statistics and zone analysis

4. **`create_sky_comparison.py`** (4.4 KB)
   - **Purpose**: Creates side-by-side comparisons for sky fixes
   - **Status**: ⚠️ Needs relocation to `tools/investigations/`
   - **Dependencies**: PIL, numpy (core deps)
   - **Usage**: Visual validation of fixes

#### Related Materials V3 Documentation (Already Extracted)
5. **`docs/materials/PHASE_A_COMPLETE.md`** - Bug fixes and hardening
6. **`docs/materials/PHASE_B_COMPLETE.md`** - Sky as first-class material
7. **`MATERIALS_V3_IMPLEMENTATION_SUMMARY.md`** - Production integration
8. **`MATERIALS_V3_EXECUTIVE_SUMMARY.md`** - High-level overview

### 1.2 Potentially Missing Investigation Files

**Search Required**: Check development branches for:
- Additional diagnostic scripts (pixel ops analysis, depth analysis)
- Iteration reports (fix attempts, A/B comparisons)
- Performance profiling data
- Edge case documentation
- Failed approach documentation

**Branches to Check**:
```bash
bugfix/materials-v3-critical-fixes
feature/materials-v3-water-sky-synthesis
feature/materials-v3-production-integration
```

### 1.3 Dependencies Analysis

**Documentation Files**:
- ✅ **No dependencies** - Pure markdown
- ⚠️ **Path references**: May reference old file locations
- ⚠️ **Image references**: May reference uncommitted test images

**Diagnostic Scripts**:
- ✅ **Core dependencies only**: PIL, numpy (already in requirements.txt)
- ⚠️ **Import risk**: LOW - use standard library + numpy
- ✅ **No ML models**: No heavy dependencies

---

## Part 2: Extraction Strategy

### 2.1 Documentation Organization

**Goal**: Consolidate all Materials V3 investigations into a dedicated subdirectory

**Method**: Move + Organize (not git cherry-pick, since files already exist)

**Target Structure**:
```
docs/investigations/materials_v3/
├── README.md                              # Overview of all investigations (NEW)
├── METHODOLOGY.md                         # Diagnostic approach guide (NEW)
├── INDEX.md                               # Quick reference table (NEW)
├── artifacts/
│   └── PRIMARY_BEDROOM_EDGE_ARTIFACTS.md  # Moved from parent dir
├── color_grading/
│   └── SKY_WATER_INVESTIGATION.md         # Moved from parent dir
├── depth_analysis/
│   └── (any depth-related investigations)
├── pixel_ops/
│   └── (pixel operations debugging)
└── assets/
    └── (supporting images if any)
```

**Alternative (Simpler)**:
```
docs/investigations/materials_v3/
├── README.md                              # Index + methodology
├── edge_artifacts_primary_bedroom.md      # Renamed for clarity
├── sky_water_color_analysis.md            # Renamed for clarity
├── depth_identity_bug_analysis.md         # If found in branches
├── pixel_ops_zero_delta_investigation.md  # If found in branches
└── assets/                                # Supporting materials
    └── (diagrams, sample outputs)
```

### 2.2 Diagnostic Scripts Organization

**Goal**: Create a dedicated investigations tooling directory

**Method**: Move from root to structured location

**Target Structure**:
```
tools/investigations/
├── README.md                      # Usage guide (NEW)
├── materials_v3/
│   ├── diagnose_sky_issue.py      # Moved from root
│   ├── create_sky_comparison.py   # Moved from root
│   ├── analyze_depth_zones.py     # If found in branches
│   └── profile_pixel_ops.py       # If found in branches
└── __init__.py                    # Make it a package
```

**Updates Required**:
- Add usage documentation to each script
- Add `--help` flags if missing
- Ensure scripts work with current main (test imports)

### 2.3 Supporting Materials

**Configuration Examples**:
- Link to `config/materials_v3_production.yaml`
- Document test presets used in investigations

**Test Fixtures**:
- Document which test images were used (if public)
- Provide download links or references

---

## Part 3: File Structure Design (Final Recommendation)

### Recommended Structure: Flat + Clear Naming

**Rationale**: Easier to navigate, less nesting, clearer purpose

```
docs/investigations/materials_v3/
├── README.md                                    # Overview + Quick Links
├── DIAGNOSTIC_METHODOLOGY.md                    # How we debug Materials V3
├── edge_artifacts_primary_bedroom.md            # Artifact investigation
├── sky_water_color_grading_analysis.md          # Color change validation
├── depth_identity_bug.md                        # [Extract if exists]
├── pixel_ops_zero_delta_bug.md                  # [Extract if exists]
├── mask_feathering_failed_fix.md                # [Extract if exists]
└── assets/
    ├── primary_bedroom_before_after.png         # [If exists]
    └── depth_zone_visualization.png             # [If exists]

tools/investigations/materials_v3/
├── README.md                                    # Tool usage guide
├── diagnose_sky_issue.py                        # Depth map analyzer
├── create_sky_comparison.py                     # Visual comparison generator
├── analyze_pixel_ops_delta.py                   # [Extract if exists]
└── profile_materials_performance.py             # [Extract if exists]
```

**Key Files to Create**:

1. **`docs/investigations/materials_v3/README.md`** (250-300 lines)
   - Investigation index with summaries
   - Cross-references to Phase A/B reports
   - Links to diagnostic tools
   - How to use this documentation

2. **`docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md`** (200-250 lines)
   - Materials V3 debugging approach
   - Common issues and diagnostic steps
   - Tool usage patterns
   - Performance profiling techniques
   - How to validate fixes

3. **`tools/investigations/materials_v3/README.md`** (150-200 lines)
   - Tool descriptions and usage
   - Example commands
   - Output interpretation
   - Integration with main pipeline

---

## Part 4: Execution Checklist

### Phase 2.1: Preparation (10 minutes)

**Branch Management**:
- [ ] Fetch all remote branches: `git fetch --all`
- [ ] Create Phase 2 branch: `git checkout -b docs/materials-v3-investigations`
- [ ] Verify clean working directory: `git status`

**Directory Creation**:
- [ ] `mkdir -p docs/investigations/materials_v3/assets`
- [ ] `mkdir -p tools/investigations/materials_v3`
- [ ] `touch tools/investigations/__init__.py`
- [ ] `touch tools/investigations/materials_v3/__init__.py`

**Time Estimate**: 10 minutes

---

### Phase 2.2: Search for Missing Investigation Files (20 minutes)

**Search Development Branches**:
```bash
# Checkout and search each branch
for branch in bugfix/materials-v3-critical-fixes \
              feature/materials-v3-water-sky-synthesis \
              feature/materials-v3-production-integration; do
  echo "=== Searching $branch ==="
  git checkout $branch 2>/dev/null || continue

  # Find investigation-related files
  find . -type f -name "*investigation*.md" -o -name "*bug*.md" -o -name "*diagnose*.py"

  # Look for analysis scripts
  find . -type f -name "*analyze*.py" -o -name "*profile*.py"

  # Check for Materials V3 docs not in main
  git diff main --name-only | grep -E "materials|pixel.*ops|depth"
done

# Return to Phase 2 branch
git checkout docs/materials-v3-investigations
```

**Documentation**:
- [ ] List all discovered files: `docs/project-status/phase_2_discovered_files.txt`
- [ ] Note which branch each file came from
- [ ] Identify any duplicate content

**Time Estimate**: 20 minutes

---

### Phase 2.3: Reorganize Existing Files (30 minutes)

**Move Investigation Reports**:
```bash
cd "$(git rev-parse --show-toplevel)"

# Move and rename for clarity
git mv docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md \
       docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md

git mv docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md \
       docs/investigations/materials_v3/sky_water_color_grading_analysis.md
```

**Move Diagnostic Scripts**:
```bash
# Move scripts from root to tools
git mv diagnose_sky_issue.py tools/investigations/materials_v3/
git mv create_sky_comparison.py tools/investigations/materials_v3/
```

**Update File References**:
- [ ] Check for any docs linking to old paths
- [ ] Update DOCUMENTATION_MAP.md
- [ ] Update any README references

**Time Estimate**: 30 minutes

---

### Phase 2.4: Extract Additional Files (If Found) (40 minutes)

**For Each Discovered File**:
```bash
# Example: Extract from feature branch
git checkout feature/materials-v3-water-sky-synthesis

# Cherry-pick specific file to temp location
git show HEAD:path/to/investigation.md > /tmp/investigation.md

# Switch back to Phase 2 branch
git checkout docs/materials-v3-investigations

# Copy file to new location
cp /tmp/investigation.md docs/investigations/materials_v3/pixel_ops_analysis.md

# Stage for commit
git add docs/investigations/materials_v3/pixel_ops_analysis.md
```

**Update Checklist**:
- [ ] Extract depth-related investigations
- [ ] Extract pixel ops debugging docs
- [ ] Extract performance profiling reports
- [ ] Extract failed fix attempts (learning material)

**Time Estimate**: 20-40 minutes (depends on findings)

---

### Phase 2.5: Create Documentation (60 minutes)

**README.md (Main Index)**:
```bash
cat > docs/investigations/materials_v3/README.md << 'EOF'
# Materials V3 Investigations - Documentation Index

This directory contains investigation reports and diagnostic methodology from the Materials V3 development cycle (2024-2026).

## Purpose

These documents serve as:
- **Historical record** of bugs discovered and fixed
- **Diagnostic methodology** for future Materials V3 debugging
- **Training material** for understanding edge cases
- **Architecture insights** into pixel ops, segmentation, and V2 integration

## Investigation Reports

### Edge Artifacts
- [**Primary Bedroom Edge Artifacts**](edge_artifacts_primary_bedroom.md)
  - **Issue**: White halos and blue contamination at material boundaries
  - **Root Cause**: V2 enhancement amplifies Materials V3 edge transitions
  - **Status**: Documented, awaiting proper fix
  - **Key Learning**: Mask feathering alone cannot fix V2 amplification

### Color Grading
- [**Sky/Water Color Analysis**](sky_water_color_grading_analysis.md)
  - **Issue**: User concern about color degradation
  - **Root Cause**: No degradation - preset-based enhancement as designed
  - **Status**: Resolved, behavior validated
  - **Key Learning**: Document preset expectations upfront

[Additional investigations listed here...]

## Diagnostic Tools

See [tools/investigations/materials_v3/](../../../tools/investigations/materials_v3/) for:
- Depth map analyzers
- Visual comparison generators
- Pixel ops profilers
- Performance benchmarking tools

## Related Documentation

- [Phase A Complete](../../materials/PHASE_A_COMPLETE.md) - Bug fixes and hardening
- [Phase B Complete](../../materials/PHASE_B_COMPLETE.md) - Sky as first-class material
- [Implementation Summary](../../../MATERIALS_V3_IMPLEMENTATION_SUMMARY.md)
- [Diagnostic Methodology](DIAGNOSTIC_METHODOLOGY.md)

## How to Use This Documentation

1. **Debugging a Similar Issue**: Search for relevant investigation reports
2. **Understanding Root Causes**: Read the analysis sections
3. **Validating Fixes**: Use the diagnostic tools
4. **Learning from Failures**: Review failed fix attempts

---

**Last Updated**: 2024-02-14
**Maintainer**: Transformation Portal Team
EOF
```

**DIAGNOSTIC_METHODOLOGY.md**:
```bash
cat > docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md << 'EOF'
# Materials V3 Diagnostic Methodology

This document describes the systematic approach to debugging Materials V3 issues.

## Quick Diagnosis Flowchart

```
Issue Reported
     ↓
1. Isolate Stage (V3 Pixel Ops vs V2 Enhancement vs Depth)
     ↓
2. Collect Telemetry (manifest.json, masks, depth maps)
     ↓
3. Quantitative Analysis (delta maps, statistics)
     ↓
4. Hypothesis Formation
     ↓
5. Targeted Fix + Validation
     ↓
6. Document Findings
```

## Stage 1: Isolation

### Determine Which Pipeline Stage is Responsible

**Tools**:
- Manifest telemetry: `output/*/manifest.json`
- Intermediate outputs: `output/*/temp/`
- Disable flags: `apply_pixel_ops=False`, `enable_v2_enhancement=False`

**Questions to Answer**:
- Does the issue occur without Materials V3? (`enable_materials_v3=False`)
- Does the issue occur without V2 enhancement? (`enable_v2_enhancement=False`)
- Does the issue occur with stub segmentation? (`enable_material_segmentation=False`)

### Example: Primary Bedroom Edge Artifacts

```python
# Test 1: Disable Materials V3 pixel ops
run_enhancement(..., apply_pixel_ops=False)
# Result: Artifacts still present → Not pixel ops bug

# Test 2: Disable V2 enhancement
run_enhancement(..., enable_v2_enhancement=False)
# Result: Artifacts disappear → V2 is amplifying edge transitions
```

## Stage 2: Telemetry Collection

### Manifest Analysis

**Key Fields**:
```json
{
  "materials_v3": {
    "pixel_ops_applied": [...],
    "overlap_resolution": {...},
    "material_coverage": {...}
  },
  "depth_estimation": {
    "backend": "depth_anything_v2",
    "model_variant": "small"
  },
  "v2_enhancement": {
    "preset": "luxury_estate"
  }
}
```

### Mask Visualization

```python
# Load and visualize material masks
import numpy as np
from PIL import Image

masks = np.load("output/*/temp/*_masks.npz")
for material, mask in masks.items():
    print(f"{material}: coverage={mask.mean():.2%}")
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"debug_{material}.png")
```

## Stage 3: Quantitative Analysis

### Delta Maps

```python
def compute_delta_map(before: np.ndarray, after: np.ndarray) -> dict:
    """Compute pixel-wise change statistics."""
    delta = after.astype(np.float32) - before.astype(np.float32)

    return {
        "mean_abs_delta": np.abs(delta).mean(),
        "max_delta": np.abs(delta).max(),
        "edge_artifact_percent": compute_edge_artifacts(delta),
        "spatial_distribution": analyze_spatial_pattern(delta)
    }
```

### Statistical Validation

**Thresholds** (from experience):
- **Edge artifacts**: >5% delta at material boundaries = 🔴 CRITICAL
- **White halos**: >40% pixels with RGB increase = 🔴 CRITICAL
- **Color contamination**: Localized blue/red excess = 🟡 INVESTIGATE
- **Global shift**: <3% mean delta = ✅ ACCEPTABLE

## Stage 4: Hypothesis Formation

### Common Materials V3 Bug Patterns

| Pattern | Symptoms | Likely Cause |
|---------|----------|--------------|
| **Edge halos** | Bright/colored edges around materials | V2 amplification or mask feathering |
| **Color bleeding** | Material A color appears in Material B | Overlap resolution or mask quality |
| **Zero delta** | Pixel ops applied but no change | Normalization bug or identity operation |
| **Crash on masks** | 3D mask shape error | Segmentation backend returns (H,W,1) |
| **Performance degradation** | >2x slowdown | Missing cache or redundant normalization |

### Tools for Each Pattern

- **Edge halos**: `tools/investigations/materials_v3/create_sky_comparison.py`
- **Depth issues**: `tools/investigations/materials_v3/diagnose_sky_issue.py`
- **Performance**: `pytest --profile` + memory-profiler

## Stage 5: Fix Validation

### Validation Checklist

**Before Merging a Fix**:
- [ ] Quantitative metrics improve (delta maps show reduction)
- [ ] No regression on other test images
- [ ] Telemetry confirms expected behavior
- [ ] Tests added for the specific bug
- [ ] Performance impact documented

**Example: Mask Feathering Fix (FAILED)**:
```
✅ Implementation: Gaussian blur applied correctly
✅ Telemetry: Feather sigma logged
❌ Validation: Artifacts identical before/after
→ Conclusion: Root cause is V2, not mask edges
```

## Stage 6: Documentation

### Investigation Report Template

```markdown
# [Issue Name] Investigation

**Date**: YYYY-MM-DD
**Status**: [Resolved|Pending|Documented]

## Issue Description
[User-reported symptoms]

## Quantitative Measurements
[Delta maps, statistics, thresholds]

## Root Cause Analysis
[Isolation steps, hypothesis testing]

## Attempted Fixes
[What was tried, results]

## Resolution
[Final fix or workaround]

## Key Learnings
[What we learned for next time]
```

---

## Diagnostic Tool Reference

### Built-in Tools

1. **`diagnose_sky_issue.py`**
   - **Purpose**: Depth map zone analysis
   - **Usage**: `python tools/investigations/materials_v3/diagnose_sky_issue.py --depth output/*/depth.tif`
   - **Output**: Percentile statistics, zone distributions

2. **`create_sky_comparison.py`**
   - **Purpose**: Side-by-side visual comparison
   - **Usage**: `python tools/investigations/materials_v3/create_sky_comparison.py --before input.tif --after output.tif`
   - **Output**: Comparison image highlighting differences

### External Tools

- **ImageMagick compare**: `compare -metric RMSE before.tif after.tif delta.png`
- **Python**: `from skimage.metrics import structural_similarity`
- **Photoshop**: Layer difference blending mode for visual inspection

---

## Performance Profiling

### Memory Usage

```bash
python -m memory_profiler scripts/enhance_image.py --input test.tif --output out.tif
```

### Time Profiling

```bash
pytest --profile --profile-svg tests/materials/
```

### Expected Baselines

- **Pixel ops overhead**: <50ms for 4K image
- **Mask feathering**: <20ms per material
- **Overlap resolution**: <10ms
- **Total V3 overhead**: <100ms

---

**Last Updated**: 2024-02-14
**Contributors**: Materials V3 Team
EOF
```

**Tools README.md**:
```bash
cat > tools/investigations/materials_v3/README.md << 'EOF'
# Materials V3 Diagnostic Tools

Diagnostic scripts for investigating Materials V3 pixel operations, segmentation, and enhancement issues.

## Available Tools

### 1. `diagnose_sky_issue.py`

**Purpose**: Analyzes depth maps to diagnose sky region detection and depth value issues.

**Usage**:
```bash
python tools/investigations/materials_v3/diagnose_sky_issue.py \
  --depth output/750Picacho_Aerial/depth.tif \
  --name "Aerial Test"
```

**Output**:
- Depth map statistics (min, max, mean, median, std)
- Percentile distribution (p5, p10, p25, p50, p75, p90, p95)
- Zone analysis (foreground, midground, background, sky)
- Histogram visualization

**Use Cases**:
- Verify sky is detected in far depth
- Check depth map quality
- Diagnose zone-based enhancement issues

---

### 2. `create_sky_comparison.py`

**Purpose**: Creates side-by-side visual comparisons of before/after images, focusing on sky regions.

**Usage**:
```bash
python tools/investigations/materials_v3/create_sky_comparison.py \
  --before input/750Picacho_Aerial_master16.tif \
  --after output/750Picacho_Aerial/enhanced.tif \
  --output comparison_sky_fix.png \
  --crop-sky
```

**Output**:
- Side-by-side comparison image
- Optional crop to sky region (top 40%)
- Downscaled for easy viewing
- Labels for before/after

**Use Cases**:
- Visual validation of sky enhancement fixes
- Quick QA for color grading changes
- Documentation for bug reports

---

## Installation

These tools use core dependencies already in `requirements.txt`:
- `Pillow` (image loading)
- `numpy` (numerical analysis)

No additional installation required if repository is set up.

---

## Integration with Main Pipeline

### Using Diagnostic Tools in CI/CD

**Example: Automated Depth Analysis**:
```yaml
# .github/workflows/materials-v3-qa.yml
- name: Analyze Depth Maps
  run: |
    python tools/investigations/materials_v3/diagnose_sky_issue.py \
      --depth output/test_aerial/depth.tif \
      --json output/depth_analysis.json
```

### Using Tools in Development

**Typical Workflow**:
```bash
# 1. Run enhancement
python scripts/enhance_image.py \
  --input input/aerial.tif \
  --output output/aerial_v1/ \
  --preset materials_v3_production

# 2. Diagnose depth if sky issues occur
python tools/investigations/materials_v3/diagnose_sky_issue.py \
  --depth output/aerial_v1/depth.tif

# 3. Create comparison
python tools/investigations/materials_v3/create_sky_comparison.py \
  --before input/aerial.tif \
  --after output/aerial_v1/enhanced.tif \
  --output debug/aerial_comparison.png
```

---

## Contributing New Tools

When adding a new diagnostic tool:

1. **Follow naming convention**: `analyze_*.py` or `diagnose_*.py`
2. **Add `--help` flag**: Use `argparse` for CLI
3. **Document usage**: Add section to this README
4. **Add example**: Include sample command
5. **Test with current main**: Ensure imports work

**Template**:
```python
#!/usr/bin/env python3
"""Brief description of what this tool diagnoses."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Diagnostic logic here

if __name__ == "__main__":
    main()
```

---

## Related Documentation

- [Investigation Reports](../../../docs/investigations/materials_v3/)
- [Diagnostic Methodology](../../../docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md)
- [Materials V3 Overview](../../../MATERIALS_V3_EXECUTIVE_SUMMARY.md)

---

**Last Updated**: 2024-02-14
**Maintainer**: Materials V3 Team
EOF
```

**Time Estimate**: 60 minutes

---

### Phase 2.6: Update File References (20 minutes)

**Update Moved Files**:
```bash
# Update any internal links in investigation reports
cd docs/investigations/materials_v3

# Example: Fix broken links in edge_artifacts_primary_bedroom.md
sed -i '' 's|../materials/PHASE_A_COMPLETE.md|../../materials/PHASE_A_COMPLETE.md|g' *.md
sed -i '' 's|../../MATERIALS_V3|../../../MATERIALS_V3|g' *.md
```

**Update DOCUMENTATION_MAP.md**:
```bash
# Add Phase 2 entries
cat >> DOCUMENTATION_MAP.md << 'EOF'

## Materials V3 Investigations

- [Investigation Index](docs/investigations/materials_v3/README.md)
- [Diagnostic Methodology](docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md)
- [Edge Artifacts Investigation](docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md)
- [Sky/Water Analysis](docs/investigations/materials_v3/sky_water_color_grading_analysis.md)
- [Diagnostic Tools](tools/investigations/materials_v3/README.md)
EOF
```

**Time Estimate**: 20 minutes

---

### Phase 2.7: Validation (20 minutes)

**Test Diagnostic Scripts**:
```bash
# Test that scripts execute without errors
python tools/investigations/materials_v3/diagnose_sky_issue.py --help
python tools/investigations/materials_v3/create_sky_comparison.py --help

# Verify imports work
python -c "from tools.investigations.materials_v3 import diagnose_sky_issue"
```

**Check Markdown Linting**:
```bash
# Lint all investigation docs
find docs/investigations/materials_v3 -name "*.md" -exec \
  npx markdownlint-cli {} +

# Or use make target if exists
make lint-docs
```

**Verify Links**:
```bash
# Check for broken links
find docs/investigations/materials_v3 -name "*.md" -exec \
  grep -H "]\(" {} \; | grep -E "\.\./.*\.md"

# Manually verify each link resolves
```

**Checklist**:
- [ ] All markdown files lint clean
- [ ] All Python scripts have `--help` flags
- [ ] Scripts execute without import errors
- [ ] No absolute paths in documentation
- [ ] All relative links resolve correctly
- [ ] No sensitive data or credentials

**Time Estimate**: 20 minutes

---

### Phase 2.8: Commit and PR Creation (15 minutes)

**Commit Strategy**:
```bash
# Commit in logical groups
git add docs/investigations/materials_v3/README.md \
        docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md \
        tools/investigations/materials_v3/README.md

git commit -m "docs(materials-v3): add investigation index and methodology"

git add docs/investigations/materials_v3/*.md
git commit -m "docs(materials-v3): reorganize investigation reports"

git add tools/investigations/materials_v3/*.py
git commit -m "tools(materials-v3): relocate diagnostic scripts"

git add DOCUMENTATION_MAP.md
git commit -m "docs: update documentation map with Phase 2 investigations"
```

**Push and Create PR**:
```bash
git push -u origin docs/materials-v3-investigations

gh pr create \
  --title "docs(materials-v3): extract and organize investigation reports (Phase 2)" \
  --base main \
  --body-file docs/project-status/PHASE_2_PR_TEMPLATE.md
```

**Time Estimate**: 15 minutes

---

## Part 5: PR Template

**File**: `docs/project-status/PHASE_2_PR_TEMPLATE.md`

```markdown
## Summary

Extracts and organizes **Materials V3 investigation reports** and **diagnostic scripts** as **Phase 2** of the PR #934 extraction strategy.

Consolidates scattered investigation documentation into a structured subdirectory with comprehensive indexing and diagnostic methodology.

## What's Included

### Investigation Reports
1. **Edge Artifacts - Primary Bedroom**
   - Documents white halos and blue contamination at material boundaries
   - Root cause: V2 enhancement amplifies Materials V3 edge transitions
   - Key learning: Mask feathering alone cannot fix V2 amplification issues

2. **Sky/Water Color Grading Analysis**
   - Validates color changes are preset-based enhancements, not degradation
   - Quantitative analysis of sky/water regions
   - Confirms expected behavior per luxury_estate preset

### Diagnostic Tools
1. **`diagnose_sky_issue.py`**
   - Depth map analyzer with zone statistics
   - Usage: Diagnose sky detection and depth value issues

2. **`create_sky_comparison.py`**
   - Visual comparison generator for before/after validation
   - Usage: QA for sky enhancement and color grading fixes

### Documentation
1. **Investigation Index** (`README.md`)
   - Complete catalog of all investigations
   - Cross-references to Phase A/B reports
   - Links to diagnostic tools

2. **Diagnostic Methodology** (`DIAGNOSTIC_METHODOLOGY.md`)
   - Systematic approach to debugging Materials V3
   - Stage-by-stage isolation techniques
   - Quantitative analysis methods
   - Fix validation checklist

3. **Tool Usage Guide** (`tools/investigations/materials_v3/README.md`)
   - Installation requirements
   - Example commands
   - Integration with main pipeline

## File Organization

```
docs/investigations/materials_v3/
├── README.md                                # Investigation index
├── DIAGNOSTIC_METHODOLOGY.md                # Debugging guide
├── edge_artifacts_primary_bedroom.md        # Artifact investigation
├── sky_water_color_grading_analysis.md      # Color analysis
└── assets/                                  # Supporting materials

tools/investigations/materials_v3/
├── README.md                                # Tool usage guide
├── diagnose_sky_issue.py                    # Depth analyzer
├── create_sky_comparison.py                 # Visual comparator
└── __init__.py                              # Package marker
```

## Value Proposition

- 📚 **Preserves diagnostic methodology** from Materials V3 development cycle
- 🔍 **Documents bug investigation findings** (edge artifacts, color grading)
- 🛠️ **Provides reusable analysis tools** for future debugging
- 📖 **Training material** for team on systematic debugging approaches
- 🏗️ **Historical record** of failed fixes and lessons learned

## Testing

✅ **All markdown files lint clean**
- Checked with markdownlint-cli
- No broken internal links

✅ **Diagnostic scripts execute without errors**
- Both scripts have `--help` flags
- Import dependencies verified (PIL, numpy)
- No ML model dependencies

✅ **File references updated**
- Relative paths adjusted for new structure
- Cross-references to Phase A/B docs working
- DOCUMENTATION_MAP.md updated

✅ **No sensitive data**
- No absolute paths
- No credentials or API keys
- Test image references are generic

## Backward Compatibility

✅ **100% Compatible** - Documentation and tooling only, no code changes to production pipeline

**What Changed**:
- File locations (moved from `docs/investigations/` to `docs/investigations/materials_v3/`)
- Script locations (moved from root to `tools/investigations/materials_v3/`)

**What Didn't Change**:
- Production pipeline code
- Test suite
- Configuration files
- Dependencies

## Extraction Context

This is **Phase 2** of the PR #934 extraction strategy:

- **Phase 1**: Depth identity fix (PR #936) - ✅ Merged
- **Phase 2**: Investigation docs - ⏳ Current PR
- **Phase 3**: Config presets - 📅 Planned (Week 1)
- **Phase 4**: Test fixtures - 📅 Planned (Week 1)
- **Phase 5**: Performance optimizations - 📅 Planned (Week 2)
- **Phase 6**: Documentation updates - 📅 Planned (Week 2)

See [Extraction Strategy](docs/project-status/PR934_EXTRACTION_STRATEGY.md) for full plan.

## Related PRs

- PR #934: Original PR (to be closed after all phases extracted)
- PR #935: Materials V3 Phases A+B (merged 2024-02-14)
- PR #936: Depth identity fix - Phase 1 (merged 2024-02-13)

## Review Checklist

- [ ] All investigation reports are complete and accurate
- [ ] Diagnostic tools have clear usage documentation
- [ ] File structure is logical and easy to navigate
- [ ] Cross-references work correctly
- [ ] No regressions in documentation linking
- [ ] README files provide clear entry points

---

**Files Changed**: ~10-15 files
**Lines Added**: ~800-1000 (documentation)
**Risk**: 🟢 LOW (docs/tools only)
**Effort**: ⏱️ 2-3 hours
**Value**: ⭐⭐⭐⭐ (high - preserves investigation knowledge)
```

---

## Part 6: Risk Assessment Matrix

### Documentation Files

| Risk Type | Level | Mitigation |
|-----------|-------|------------|
| **Merge Conflicts** | 🟢 LOW | Separate directory, no overlap with active work |
| **Staleness** | 🟡 MEDIUM | May reference code that changed in PR #935 |
| **Link Breakage** | 🟡 MEDIUM | Update all relative paths after move |
| **Maintenance Burden** | 🟢 LOW | Static historical docs, minimal updates needed |

**Mitigation Steps**:
1. ✅ Audit all file references and update paths
2. ✅ Add "historical investigation" context notes where needed
3. ✅ Cross-reference to current Phase A/B completion reports
4. ✅ Validate all links before PR submission

### Diagnostic Scripts

| Risk Type | Level | Mitigation |
|-----------|-------|------------|
| **Import Errors** | 🟡 MEDIUM | Production imports may have changed in PR #935 |
| **Dependency Changes** | 🟢 LOW | Use only core deps (PIL, numpy) |
| **Execution Failures** | 🟢 LOW | Scripts are standalone analyzers |
| **API Incompatibility** | 🟡 MEDIUM | May use deprecated internal APIs |

**Mitigation Steps**:
1. ✅ Test each script executes `--help` without errors
2. ✅ Check imports against current main
3. ✅ Update any deprecated API calls
4. ✅ Add "last validated" date to tool README

### Supporting Materials

| Risk Type | Level | Mitigation |
|-----------|-------|------------|
| **Missing Assets** | 🟡 MEDIUM | Investigation may reference uncommitted images |
| **Large Files** | 🟢 LOW | Docs only, no large binaries expected |
| **Sensitive Data** | 🟢 LOW | No production data in investigations |

**Mitigation Steps**:
1. ✅ Identify any image references and provide placeholders
2. ✅ Document which test images were used (without committing them)
3. ✅ Ensure no absolute paths or sensitive information

---

## Part 7: Quick Start Commands

### Complete Phase 2 Execution Script

**Archived file**:
`archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh`

This script is retained as historical evidence only. It performs branch
checkout, pull, stash, commit, and push operations from 2024 guidance. Prefer
manual execution for any current recovery work.

```bash
#!/usr/bin/env bash
set -e  # Exit on error

echo "=== Phase 2: Materials V3 Investigation Documentation Extraction ==="
echo ""

# Configuration
PHASE_BRANCH="docs/materials-v3-investigations"
BASE_DIR="$(git rev-parse --show-toplevel)"

cd "$BASE_DIR"

#------------------------------------------------------------------------------
# Step 1: Preparation (10 minutes)
#------------------------------------------------------------------------------
echo "Step 1: Preparation..."

# Fetch all branches
git fetch --all

# Create Phase 2 branch from main
git checkout main
git pull origin main
git checkout -b "$PHASE_BRANCH"

# Create directories
mkdir -p docs/investigations/materials_v3/assets
mkdir -p tools/investigations/materials_v3
touch tools/investigations/__init__.py
touch tools/investigations/materials_v3/__init__.py

echo "✅ Directories created"

#------------------------------------------------------------------------------
# Step 2: Search for Additional Files (20 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 2: Searching development branches for additional investigation files..."

# Search each development branch
for branch in bugfix/materials-v3-critical-fixes \
              feature/materials-v3-water-sky-synthesis \
              feature/materials-v3-production-integration; do
  echo ""
  echo "=== Searching $branch ==="

  if git checkout "$branch" 2>/dev/null; then
    echo "Branch: $branch" >> docs/project-status/phase_2_discovered_files.txt

    find . -type f \( -name "*investigation*.md" -o -name "*bug*.md" -o -name "*diagnose*.py" -o -name "*analyze*.py" \) \
      >> docs/project-status/phase_2_discovered_files.txt || true

    git diff main --name-only | grep -E "materials|pixel.*ops|depth" \
      >> docs/project-status/phase_2_discovered_files.txt || true

    echo "" >> docs/project-status/phase_2_discovered_files.txt
  else
    echo "⚠️  Branch $branch not found, skipping..."
  fi
done

# Return to Phase 2 branch
git checkout "$PHASE_BRANCH"

echo "✅ Branch search complete (see docs/project-status/phase_2_discovered_files.txt)"

#------------------------------------------------------------------------------
# Step 3: Reorganize Existing Files (30 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 3: Reorganizing existing investigation files..."

# Move investigation reports (if they exist)
if [ -f "docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md" ]; then
  git mv docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md \
         docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md
  echo "✅ Moved PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md"
fi

if [ -f "docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md" ]; then
  git mv docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md \
         docs/investigations/materials_v3/sky_water_color_grading_analysis.md
  echo "✅ Moved SKY_WATER_INVESTIGATION_SUMMARY.md"
fi

# Move diagnostic scripts (if they exist)
if [ -f "diagnose_sky_issue.py" ]; then
  git mv diagnose_sky_issue.py tools/investigations/materials_v3/
  echo "✅ Moved diagnose_sky_issue.py"
fi

if [ -f "create_sky_comparison.py" ]; then
  git mv create_sky_comparison.py tools/investigations/materials_v3/
  echo "✅ Moved create_sky_comparison.py"
fi

echo "✅ File reorganization complete"

#------------------------------------------------------------------------------
# Step 4: Create Documentation (60 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 4: Creating documentation..."
echo "⚠️  Manual step required - create README files:"
echo "  1. docs/investigations/materials_v3/README.md"
echo "  2. docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md"
echo "  3. tools/investigations/materials_v3/README.md"
echo ""
echo "Press Enter when documentation is created..."
read -r

#------------------------------------------------------------------------------
# Step 5: Update File References (20 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 5: Updating file references..."

# Update internal links in moved files
cd docs/investigations/materials_v3

# Fix relative paths (adjust based on actual content)
for file in *.md; do
  if [ -f "$file" ]; then
    # Update references to materials/ directory
    sed -i.bak 's|../materials/|../../materials/|g' "$file"

    # Update references to root-level docs
    sed -i.bak 's|../../MATERIALS_V3|../../../MATERIALS_V3|g' "$file"

    # Clean up backup files
    rm -f "${file}.bak"
  fi
done

cd "$BASE_DIR"

echo "✅ File references updated"

#------------------------------------------------------------------------------
# Step 6: Validation (20 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 6: Validating extraction..."

# Test diagnostic scripts
echo "Testing diagnostic tools..."
python tools/investigations/materials_v3/diagnose_sky_issue.py --help > /dev/null 2>&1 \
  && echo "✅ diagnose_sky_issue.py --help works" \
  || echo "❌ diagnose_sky_issue.py --help failed"

python tools/investigations/materials_v3/create_sky_comparison.py --help > /dev/null 2>&1 \
  && echo "✅ create_sky_comparison.py --help works" \
  || echo "❌ create_sky_comparison.py --help failed"

# Check for broken links (manual verification needed)
echo ""
echo "⚠️  Manual validation required:"
echo "  1. Check all markdown files lint clean"
echo "  2. Verify all relative links resolve"
echo "  3. Ensure no absolute paths or sensitive data"
echo ""
echo "Press Enter when validation is complete..."
read -r

#------------------------------------------------------------------------------
# Step 7: Commit (15 minutes)
#------------------------------------------------------------------------------
echo ""
echo "Step 7: Committing changes..."

# Commit in logical groups
git add docs/investigations/materials_v3/README.md \
        docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md \
        tools/investigations/materials_v3/README.md 2>/dev/null || true

git commit -m "docs(materials-v3): add investigation index and methodology" || true

git add docs/investigations/materials_v3/*.md 2>/dev/null
git commit -m "docs(materials-v3): reorganize investigation reports" || true

git add tools/investigations/materials_v3/*.py 2>/dev/null
git commit -m "tools(materials-v3): relocate diagnostic scripts" || true

git add DOCUMENTATION_MAP.md 2>/dev/null
git commit -m "docs: update documentation map with Phase 2 investigations" || true

echo "✅ Changes committed"

#------------------------------------------------------------------------------
# Step 8: Push and Create PR
#------------------------------------------------------------------------------
echo ""
echo "Step 8: Push and create PR..."

git push -u origin "$PHASE_BRANCH"

echo ""
echo "Create PR with:"
echo "  gh pr create \\"
echo "    --title 'docs(materials-v3): extract and organize investigation reports (Phase 2)' \\"
echo "    --base main \\"
echo "    --body-file docs/project-status/PHASE_2_PR_TEMPLATE.md"
echo ""
echo "Or visit: https://github.com/YOUR_ORG/Transformation_Portal/compare/main...$PHASE_BRANCH"
echo ""

echo "=== Phase 2 Extraction Complete ==="
```

Make this script executable:
```bash
chmod +x archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh
```

---

## Success Criteria

### Phase 2 Complete When:

✅ **Organization**:
- [ ] All investigation reports consolidated in `docs/investigations/materials_v3/`
- [ ] All diagnostic scripts in `tools/investigations/materials_v3/`
- [ ] Clear directory structure with logical naming

✅ **Documentation**:
- [ ] Comprehensive README.md index created
- [ ] Diagnostic methodology documented
- [ ] Tool usage guides written
- [ ] Cross-references to Phase A/B reports working

✅ **Validation**:
- [ ] All markdown files lint clean
- [ ] All diagnostic scripts execute `--help` successfully
- [ ] No broken relative links
- [ ] No absolute paths or sensitive data

✅ **Integration**:
- [ ] DOCUMENTATION_MAP.md updated
- [ ] File references updated to new paths
- [ ] Related docs link to investigation reports

✅ **PR Quality**:
- [ ] PR created with comprehensive description
- [ ] Linked to PR #934, #935, #936
- [ ] CI passes (linting, link checking if configured)
- [ ] No merge conflicts with main

---

## Time Budget Breakdown

| Phase | Task | Estimated Time | Actual Time |
|-------|------|----------------|-------------|
| **2.1** | Preparation | 10 minutes | _____ |
| **2.2** | Search branches | 20 minutes | _____ |
| **2.3** | Reorganize files | 30 minutes | _____ |
| **2.4** | Extract additional | 40 minutes | _____ |
| **2.5** | Create docs | 60 minutes | _____ |
| **2.6** | Update references | 20 minutes | _____ |
| **2.7** | Validation | 20 minutes | _____ |
| **2.8** | Commit & PR | 15 minutes | _____ |
| **TOTAL** | | **~3.5 hours** | _____ |

**Contingency**: +30 minutes for unexpected issues

---

## Appendix A: File Inventory Template

Use this template to track discovered files:

```
# Phase 2 Discovered Files

## Branch: bugfix/materials-v3-critical-fixes
- [ ] docs/...
- [ ] scripts/...

## Branch: feature/materials-v3-water-sky-synthesis
- [ ] docs/...
- [ ] tools/...

## Branch: feature/materials-v3-production-integration
- [ ] docs/...

## Files Already in Main
- [x] docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md
- [x] docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md
- [x] diagnose_sky_issue.py (root)
- [x] create_sky_comparison.py (root)
- [x] docs/materials/PHASE_A_COMPLETE.md
- [x] docs/materials/PHASE_B_COMPLETE.md

## Extraction Decisions
- File X: Extract to docs/investigations/materials_v3/X.md
- File Y: Skip (duplicates existing content)
- File Z: Merge with existing investigation report
```

---

## Appendix B: Quick Reference - Git Commands

```bash
# Fetch all branches
git fetch --all

# List all remote branches
git branch -r | grep materials

# Checkout specific branch
git checkout feature/materials-v3-water-sky-synthesis

# Show files that differ from main
git diff main --name-only

# Show specific file from another branch
git show branch-name:path/to/file.md

# Cherry-pick specific file to temp
git show HEAD:path/to/file.md > /tmp/file.md

# Move file with git (preserves history)
git mv old/path.md new/path.md

# Stage all files in directory
git add docs/investigations/materials_v3/

# Commit with message
git commit -m "docs(materials-v3): description"

# Push new branch
git push -u origin docs/materials-v3-investigations

# Create PR via GitHub CLI
gh pr create --title "Title" --base main --body "Description"
```

---

## Appendix C: Markdown Linting Commands

```bash
# Install markdownlint-cli (if not already installed)
npm install -g markdownlint-cli

# Lint all investigation docs
markdownlint docs/investigations/materials_v3/**/*.md

# Auto-fix simple issues
markdownlint --fix docs/investigations/materials_v3/**/*.md

# Check for broken links (using markdown-link-check)
npm install -g markdown-link-check
find docs/investigations/materials_v3 -name "*.md" -exec markdown-link-check {} \;
```

---

**Document Version**: 1.0
**Last Updated**: 2024-02-14
**Author**: Transformation Portal Specialist
**Status**: Ready for Execution
