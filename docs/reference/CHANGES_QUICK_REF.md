# APEX Fixes - Quick Reference Card

## Summary
Fixed 3 bugs + 4 optimizations from 750 Picacho production run.
**Status:** ✅ All tests passing (175/175) | **Risk:** 🟢 LOW

---

## What's New

### 1. Run Card Emission ✨
**What:** JSON file with reproducibility metadata
**Where:** `output/<batch_id>/run_card_<batch_id>.json`
**Contains:** Config fingerprint, git revision, runtime stats, outliers
**Control:** `--emit-run-card on/off` (default: on)

### 2. Outlier Detection ⚠️
**What:** Warns when image takes >5× median runtime
**Example:** `⚠️ Runtime outlier detected: image.tif took 8.50s (5.9× median of 1.45s)`
**Stored:** In batch manifest stats["outliers"]
**Threshold:** Configurable (default: 5.0×)

### 3. No More Empty zones/ 🗑️
**What:** zones/ directory no longer created if unused
**Why:** Reduces filesystem clutter
**Future:** Will be created on-demand when zoning features enabled

---

## For Users

### Before This Fix
```bash
lux-depth-v3 --input-dir images/ --emit-run-card on
# ✗ No run card emitted (flag ignored)
# ✗ Empty zones/ directory created
# ✗ No warning for slow images
```

### After This Fix
```bash
lux-depth-v3 --input-dir images/ --emit-run-card on
# ✓ Run card JSON emitted
# ✓ No empty zones/ directory
# ✓ Warnings for images >5× median runtime
```

---

## For Developers

### Modified Files
- `src/transformation_portal/lux_depth_v3/batch_stats.py` (enhanced)
- `src/transformation_portal/lux_depth_v3/orchestrator.py` (enhanced)

### New Files
- `tests/test_apex_artifact_assertions.py` (10 tests)
- `docs/historical/APEX_BUG_FIXES_IMPLEMENTATION_REPORT.md`
- `docs/guides/RUNTIME_SKEW_INVESTIGATION.md`

### API Changes
**None** - All changes are backward compatible

### New Functions
```python
# batch_stats.py
def detect_runtime_outliers(
    image_name: str,
    runtime_s: float,
    runtimes: List[float],
    threshold_multiplier: float = 5.0,
) -> Optional[Tuple[str, Dict[str, Any]]]

# orchestrator.py
def _emit_run_card(
    self,
    batch_id: str,
    start_time: str,
    end_time: str,
    results: List[Dict[str, Any]],
    runtime_stats: Dict[str, Any],
    outliers: List[Dict[str, Any]],
) -> None
```

---

## Testing

### Run All Tests
```bash
pytest tests/test_apex*.py tests/test_lux_depth_v3*.py -v
# Expected: 175 PASSED, 1 SKIPPED
```

### Run Just New Tests
```bash
pytest tests/test_apex_artifact_assertions.py -v
# Expected: 10 PASSED
```

---

## Deployment

### Pre-Merge Checklist
- [x] All tests passing (175/175)
- [x] No regressions detected
- [x] Documentation complete
- [x] Performance overhead negligible (<0.1s)
- [x] Backward compatible

### Post-Merge Verification
1. Run APEX pipeline on test images
2. Verify run card JSON created
3. Check batch manifest has "outliers" field
4. Confirm zones/ not created (unless zoning enabled)

---

## Troubleshooting

### Q: Run card not generated?
**A:** Check `--emit-run-card` flag is "on" (default: on)

### Q: No outlier warnings?
**A:** All images within 5× median (expected behavior)

### Q: zones/ directory still exists?
**A:** Created in old runs before this fix (safe to delete)

---

## Performance

- **Outlier detection:** ~0.001s per batch
- **Run card emission:** ~0.01-0.05s per batch
- **Total overhead:** <0.1s per batch ✅

---

**Quick Links:**
- [Detailed Report](../historical/APEX_BUG_FIXES_IMPLEMENTATION_REPORT.md)
- [Runtime Investigation](../guides/RUNTIME_SKEW_INVESTIGATION.md)
- [Test Suite](../../tests/test_apex_artifact_assertions.py)
