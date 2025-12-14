# PR-W4: Correctness Fixes Applied

## Three Critical Semantic Drift Issues Fixed

### 1. ✅ `is_false_positive` Now Equals `is_false_trigger` 

**Problem**: Legacy field was hardcoded to `False`, violating stated contract that it's an alias.

**Fix**: 
```python
is_false_trigger = (not should_detect and detected)
is_fp = is_false_trigger  # legacy alias, same semantics
```

**Impact**: Downstream consumers now see consistent values, not contradictory fields.

---

### 2. ✅ Use `present` for "detected", Not Coverage/Confidence

**Problem**: "Detected" was inferred from `coverage > 0 and confidence > 0`, which becomes unstable as detector scoring evolves.

**Fix**:
- Added `detected: bool` field to `ValidationResult`
- Set `detected = water_dict.get("present", False)` from detector's explicit boolean
- Updated recall computation to use `r.detected` flag

**Code**:
```python
# ValidationResult dataclass
detected: bool  # detector's explicit present flag

# In validate_single()
detected = water_dict.get('present', False)
is_false_trigger = (not should_detect and detected)

# In generate_report()
pool_detected = [r for r in pool_true if r.detected]
ocean_detected = [r for r in ocean_true if r.detected]
```

**Impact**: Recall and false trigger metrics remain stable even when coverage/confidence scoring changes.

---

### 3. ✅ .gitignore Patterns Now Match Class Subfolders

**Problem**: Pattern `data/water_*/images/*.jpg` does NOT match:
- `data/water_v0/images/pool/pool_0001.jpg`
- `data/water_v0/images/ocean/ocean_0001.jpg`

**Fix**: Use recursive glob patterns:
```gitignore
# Recursive patterns to catch class subfolders (pool/, ocean/)
data/water_*/images/**/*.jpg
data/water_*/images/**/*.jpeg
data/water_*/images/**/*.png
```

**Impact**: Full-res images in `pool/` and `ocean/` subfolders are now properly ignored.

---

## Test Results

All 3 deterministic tests passing after fixes:

```
============================= test session starts ===============================
tests/test_prw_water_validation_deterministic.py::test_stability_deterministic_with_seed PASSED
tests/test_prw_water_validation_deterministic.py::test_stability_different_with_different_seed PASSED
tests/test_prw_water_validation_deterministic.py::test_full_validation_deterministic PASSED
============================== 3 passed in 0.16s =================================
```

---

## Files Modified

1. **scripts/prw_water_validation.py**:
   - Added `detected: bool` to `ValidationResult`
   - Fixed `is_false_positive = is_false_trigger` (legacy alias)
   - Updated `generate_report()` to use `r.detected` instead of coverage/confidence

2. **.gitignore**:
   - Changed `data/water_*/images/*.jpg` → `data/water_*/images/**/*.jpg` (recursive)

---

## Validation

✅ Tests pass (3/3)  
✅ No false positives from `is_false_positive != is_false_trigger`  
✅ Recall computed from explicit detector boolean  
✅ Subfolder images properly ignored by git

---

## Status

**Ready for PR-W4 commit**: All three correctness nits fixed, tests passing, no semantic drift.
