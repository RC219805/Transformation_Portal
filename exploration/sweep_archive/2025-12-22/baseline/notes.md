# Baseline Run - 0779a57

**Date**: 2025-12-22 00:32:08
**Status**: Reference (quality ceiling)

## Parameters

All parameters at default values from commit 0779a57.

## Outputs

- 6 images × 5 formats = 30 files
- JSON metrics: See `metrics_*.json`
- Processing time: See console output

## Notes

This is the locked quality ceiling. All parameter sweeps must improve on this
baseline **without introducing new artifacts**.

**Evaluation criteria**:
- Visible improvement (but not noticed as "AI-enhanced")
- Zero new artifacts (halos, banding, color shifts)
- Reversible via parameter rollback
- Explainable in <1 paragraph

**Next**: Execute single-parameter sweeps (Phase 1)
