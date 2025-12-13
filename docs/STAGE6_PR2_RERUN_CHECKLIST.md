# Stage 6 PR-2 Rerun Checklist

## Pre-Flight

- [x] PR-2 merged to `main` (prompt strategy + ROI refinement)
- [x] CI green (all workflows passing)
- [ ] Workspace clean (`make clean`)
- [ ] EfficientSAM model present (`efficientsam_s.onnx`)

## Execution

```bash
# 1. Clean workspace
make clean

# 2. Verify model availability
python -c "
from lux_depth_v2.backends.efficientsam_backend import EfficientSAMBackend
b = EfficientSAMBackend(model_name='efficientsam_s', lazy_load=True)
print('Model available:', b.available)
"

# 3. Run PR-2 rerun with comparison
python scripts/stage6_rerun_pr2.py
```

## Expected Improvements (PR-2 Goals)

- **Higher fusion_applied rate**: Mask-driven prompts should increase usefulness
- **Better IoU on Kitchen/Pool**: ROI cropping + better prompts should align better
- **No OOM on Bathroom**: Skip guards should prevent crashes
- **Runtime acceptable**: ROI cropping may actually reduce latency

## Decision Criteria

### ✅ Promote EfficientSAM to default APEX if:

1. Fusion applies in **≥4/5 scenes** (was 2/5)
2. IoU improvements **≥0.10** on Kitchen glass / Pool foliage
3. **No visual artifacts** in diff crops (halos, spill)
4. Runtime delta **<+20%** for APEX tier

### ⚠️ Keep canary-only if:

- Fusion rate improves but artifacts increase
- IoU improvements marginal (<0.05)
- Runtime cost unacceptable for hero frames

### ❌ Disable EfficientSAM entirely if:

- Regressions in baseline quality
- New failure modes introduced
- ROI/prompt logic creates instability

## Outputs to Inspect

1. **Summary JSON**: `outputs/stage6_pr2_rerun/pr2_comparison.json`
2. **Per-scene reports**: `outputs/stage6_ab/*/canary/*_report.json`
3. **Visual diffs**: Run `scripts/stage6_visual_diff.py` on top deltas
4. **Logs**: Check for skip reasons, ROI sizes, prompt counts

## Next Steps After Rerun

- [ ] Review comparison summary
- [ ] Generate visual diff crops (if promising)
- [ ] Update Materials V3 plan based on findings
- [ ] Document decision in session summary
