# Production Depth Validation: Quick Reference

## Current Status

**Running**: `production_depth_validation_fixed.py`  
**PID**: Check with `ps aux | grep production_depth_validation_fixed`  
**Output**: `outputs/validation_blocker_fixes_test/`

## Monitor Progress

```bash
# Check if still running
ps aux | grep production_depth_validation_fixed | grep -v grep

# Check memory usage
ps aux | grep python | grep production

# Check output files
ls -lh outputs/validation_blocker_fixes_test/

# View validation report (when complete)
cat outputs/validation_blocker_fixes_test/validation_report.json | python3 -m json.tool | head -50
```

## Expected Timeline

- **First Image** (Aerial 6000×3600): ~2-3 minutes (35 tiles)
- **Per Tile**: ~1-2 seconds inference + blending
- **Total Dataset** (6 images): ~12-18 minutes

## What To Check When Complete

### 1. Execution Success
```bash
cat outputs/validation_blocker_fixes_test/validation_report.json | grep -E "total_images|succeeded|failed|overall_status"
```

Expected:
```json
"total_images": 6,
"succeeded": 6,
"failed": 0,
"overall_status": "COMPLETE"
```

### 2. Seam Quality
```bash
cat outputs/validation_blocker_fixes_test/validation_report.json | python3 -m json.tool | grep -A3 "seam_validation"
```

Target: `seam_passed >= 5/6` (≥83% pass rate)

### 3. Edge Quality Summary
```bash
cat outputs/validation_blocker_fixes_test/validation_report.json | python3 -m json.tool | grep -A10 '"quality"'
```

Target: 
- Lenient: `≥ 4/6` (≥67% pass rate)
- Strict: `≥ 2/6` (≥33% pass rate for first production run)

## Interpreting Results

### Seam Boundary Ratio
- **< 1.1**: Excellent (imperceptible seams)
- **1.1 - 1.2**: Good (pass threshold)
- **1.2 - 1.3**: Borderline (may be visible on smooth regions)
- **> 1.3**: Fail (visible grid artifacts)

### Edge F1 Score
- **≥ 0.70**: Excellent (production-grade masking)
- **0.60 - 0.70**: Good (strict pass)
- **0.30 - 0.60**: Acceptable (lenient pass, usable for DOF)
- **< 0.30**: Poor (soft boundaries)

### Chamfer Distance (px)
- **< 3**: Excellent (sub-pixel alignment)
- **3 - 5**: Good (strict pass)
- **5 - 15**: Acceptable (lenient pass)
- **> 15**: Poor (misaligned edges)

## Success Criteria

### Pilot Deployment (Minimum Bar)
- ✅ All images execute without crash
- ✅ Seam pass rate ≥ 80% (5/6 images)
- ✅ Lenient quality pass rate ≥ 50% (3/6 images)
- ✅ No catastrophic failures (negative correlation, 100× edge count)

### Full Production (Target Bar)
- 🎯 All images execute without crash
- 🎯 Seam pass rate ≥ 90% (5+/6 images)
- 🎯 Strict quality pass rate ≥ 50% (3/6 images)
- 🎯 Lenient quality pass rate ≥ 80% (5/6 images)

## Next Steps After Validation

1. **Review validation report**: `outputs/validation_blocker_fixes_test/validation_report.json`
2. **Visual inspection**: Check edge overlay images for quality
3. **Analyze failures**: Identify if seam, edge, or alignment issues
4. **Proceed to Materials V3 integration** if validation passes

---

**Document Version**: 2.0  
**Last Updated**: 2025-12-18 10:12 AM
