# Stage 6 PR-3C: Boundary Metrics Integration

## Purpose

Replace IoU-based promotion gating with **boundary-focused metrics** that actually measure edge refinement quality.

## What Changed

### New Metrics Module
- `lux_depth_v2/metrics/boundary_metrics.py` — production-grade boundary scoring

### New Stage 6 Runner
- `scripts/stage6_ab_with_boundary_metrics.py` — A/B test with edge quality scoring

### Key Metrics

1. **Boundary F1** (primary): Precision/recall on edge band pixels
2. **Trimap IoU**: Separate scoring for core/boundary/background
3. **Edge Alignment**: Correlation with image gradients

## Usage

### Run the full A/B test

```bash
cd /Users/rc/Transformation_Portal
python scripts/stage6_ab_with_boundary_metrics.py
```

**Outputs:**
- `outputs/stage6_pr3c/` — per-scene A/B results
- `outputs/stage6_pr3c/stage6_pr3c_summary.json` — decision report

### Promotion Gate (Automated)

The script emits a `promote_to_default_apex` decision based on:

✅ **PROMOTE** only if:
- Boundary F1 ≥ 0.95 on ≥3/5 scenes (preserves baseline quality)
- No scene regresses badly (BF1 < 0.85)
- Max regression < 0.05

Otherwise: **KEEP CANARY-ONLY**

### Interpreting Results

```json
{
  "decision": {
    "promote_to_default_apex": false,
    "scenes_with_improvement": 2,
    "scenes_with_regression": 1,
    "max_regression": 0.18,
    "rationale": "BF1 improvements: 2/5, regressions: 1, max_regression: 0.1800"
  }
}
```

**Boundary F1 interpretation:**
- `BF1 ≥ 0.95`: Canary matches/improves baseline edges
- `0.85 ≤ BF1 < 0.95`: Acceptable variation
- `BF1 < 0.85`: Significant edge divergence (regression)

**Edge alignment:**
- Higher = mask boundary aligns better with image gradients
- Useful secondary signal when BF1 is moderate

## Dependencies

All dependencies already present:
- `scipy` (for distance transforms, morphology)
- `numpy`, `PIL`

## Tests

```bash
pytest tests/test_stage6_pr3c.py -v
```

Tests cover:
- Boundary band extraction
- F1 score computation
- Trimap IoU
- Edge alignment
- Degenerate cases (empty masks, shape mismatches)

## Expected Runtime

- Per-scene A/B: 50–70s (APEX tier on 81MP images)
- Full 5-scene matrix: ~6–8 minutes

## Decision Workflow

1. Run Stage 6 PR-3C A/B
2. Inspect `stage6_pr3c_summary.json`
3. If `promote_to_default_apex: true`:
   - Update APEX presets to use `FUSED` by default
   - Commit with rationale from summary
4. If `promote_to_default_apex: false`:
   - Keep canary-only
   - Document why (include BF1 scores)
   - Stop further EfficientSAM work until new refinement strategy

## Notes

- **Masks must be saved by pipeline** for comparison (currently placeholder logic in script)
- If masks aren't being saved, the script will report `masks_missing` per class
- Visual diff crops are still recommended as final sanity check (but not required for automated decision)

## Future Improvements

- Auto-extract masks from `result` dict instead of loading from disk
- Add depth-edge alignment as tertiary metric
- Generate visual diff crops automatically for high-delta regions
