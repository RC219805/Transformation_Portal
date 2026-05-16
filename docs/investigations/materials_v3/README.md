# Materials V3 Investigation Documentation

**Status**: Archive
**Phase**: Materials V3 Water/Sky Synthesis Development
**Timeline**: November 2024 - February 2026

---

## Overview

This directory contains investigation reports and diagnostic findings from the **Materials V3** development cycle, specifically focused on:

- Pixel operations quality issues (edge artifacts, color shifts)
- Sky and water detection heuristics
- Material segmentation challenges
- Production deployment diagnostics

These investigations directly informed the final implementation delivered in:
- **PR #935**: Materials V3 Phases A+B (Water/Sky Synthesis) - MERGED
- **PR #936**: Depth manifest accuracy fix - MERGED

---

## Investigation Reports

### 1. Edge Artifacts - Primary Bedroom Case Study
**File**: [`edge_artifacts_primary_bedroom.md`](edge_artifacts_primary_bedroom.md)

**Summary**: Investigation of edge artifacts appearing in Primary Bedroom rendering after Materials V3 initial integration.

**Key Findings**:
- Root cause: 3D mask shape bug (H, W, 1) causing numpy indexing failures
- Secondary cause: Feathering edge clipping due to insufficient bbox padding
- Impact: ~15% of luxury real estate images showed visible artifacts

**Resolution**: Phase A1 (canonical mask helper) + A2 (bbox padding)

**Value**: Documented the bug that motivated the entire Phase A hardening effort.

---

### 2. Sky/Water Color Grading Analysis
**File**: [`sky_water_color_grading_analysis.md`](sky_water_color_grading_analysis.md)

**Summary**: Analysis of sky and water color treatment in luxury real estate context.

**Key Findings**:
- Sky needs soft feathering (sigma=5.0), water needs moderate (sigma=2.5)
- Dehaze operation critical for coastal properties (fog/marine layer common)
- Temperature shift must be per-material (sky +150K warm, water neutral)
- Gradient smoothing eliminates banding artifacts in sunset skies

**Resolution**: Phase B (Sky as First-Class Material) + per-material feathering config

**Value**: Established the pixel operations taxonomy for sky/water materials.

---

### 3. Response Planner Audit - April 2026
**File**: [`response_planner_audit_2026_04.md`](response_planner_audit_2026_04.md)

**Summary**: Current-code audit of Materials V3 response planning after Phase A/B hardening.

**Key Findings**:
- Root cause: response planner still lacks mask canonicalization even though the pixel-ops executor has it
- Policy drift: taxonomy marks `sky` canary, but planner rejects it as `not_in_canary_set`
- Config drift: taxonomy defines `none`, config comments mention `disabled`, and planner ignores the strategy value
- Performance issue: full-image Sobel gradients are recomputed once per present material

**Resolution**: Pending response-planner hardening pass.

**Value**: Converts a broad external-style audit into a repo-grounded remediation checklist.

---

## Diagnostic Methodology

For detailed debugging approaches and techniques used during Materials V3 development, see:

**[`DIAGNOSTIC_METHODOLOGY.md`](DIAGNOSTIC_METHODOLOGY.md)** (NEW)

Topics covered:
- Visual diff workflows
- Isolation testing patterns
- NumPy shape debugging
- Telemetry-driven diagnosis
- Regression test design

---

## Analysis Scripts

All diagnostic scripts used during these investigations have been preserved in:

**[`tools/investigations/materials_v3/`](../../../tools/investigations/materials_v3/)**

See that directory's README for usage instructions.

---

## Historical Context

### Development Timeline

| Date | Event |
|------|-------|
| Nov 2024 | Materials V3 initial integration (edge artifacts discovered) |
| Dec 2024 | Primary Bedroom investigation begins |
| Jan 2026 | Sky/water analysis conducted |
| Feb 2026 | Phase A (hardening) + Phase B (sky) implemented |
| Feb 14, 2026 | PR #935 merged to production |

### Why Archive These?

Even though the bugs are fixed, these investigations provide:

1. **Training material** for new team members on debugging approaches
2. **Test case motivation** - every test traces back to a real production failure
3. **Design rationale** - why certain API choices were made
4. **Institutional knowledge** - preserves context that would otherwise be lost

---

## Related Documentation

### Implementation Docs
- **[Materials V3 Roadmap](../../materials/MATERIALS_V3_ROADMAP_IMPLEMENTATION_PLAN.md)**: Full 5-phase plan
- **[Phase A Complete](../../materials/PHASE_A_COMPLETE.md)**: Pixel ops hardening summary
- **[Phase B Complete](../../materials/PHASE_B_COMPLETE.md)**: Sky material summary

### Production Docs
- **[Materials V3 Quick Reference](../../materials/ROADMAP_QUICK_REF.md)**: User-facing guide
- **[ADR-048](../../architecture/ADR-048-materials-v3-production-integration.md)**: Governance decision (renumbered 2026-05-16 from ADR-030)

### Code
- **[`pixel_ops_executor.py`](../../../src/transformation_portal/lux_depth_v3/pixel_ops_executor.py)**: Production implementation
- **[`bootstrap/sky_seed.py`](../../../src/transformation_portal/lux_depth_v3/bootstrap/sky_seed.py)**: Sky detection heuristic

---

## Contributing

**This is an archive directory** - investigations are historical documentation, not active development.

If you discover a new bug or need to conduct a new investigation:

1. Create a new investigation report in this directory
2. Follow the structure established in existing reports
3. Link to the relevant GitHub issue/PR
4. Add to this README's "Investigation Reports" section

---

## Extraction Context

These files were extracted from PR #934 as **Phase 2** of the extraction strategy:

- **Phase 1**: Depth identity fix (PR #936) - ✅ MERGED
- **Phase 2**: Investigation docs - ⏳ CURRENT
- **Phases 3-6**: Config presets, ML upscaling, SAM2, production docs

See [PR #934 Extraction Strategy](../../project-status/PR934_EXTRACTION_STRATEGY.md) for full plan.

---

**Last Updated**: February 14, 2026
**Status**: ✅ Phase 2 extraction complete
