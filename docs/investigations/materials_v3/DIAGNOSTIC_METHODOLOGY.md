# Materials V3 Diagnostic Methodology

**Purpose**: Document the debugging approaches and techniques used during Materials V3 development.
**Audience**: Engineers debugging image processing pipelines, Materials V3 contributors, QA engineers.

---

## Overview

This guide captures the **diagnostic methodology** used to identify and fix critical bugs during Materials V3 development, including:

- The **NumPy view aliasing bug** (pixel ops zero delta)
- The **3D mask shape bug** (segmentation crashes)
- **Edge artifacts** in production renders
- **Sky/water color grading** inconsistencies

---

## Core Diagnostic Patterns

### 1. Visual Diff Workflow

**When to use**: Pixel-level quality regressions (subtle color shifts, artifacts, unexpected changes)

**Steps**:

1. **Generate before/after pairs**:
   ```bash
   # Run with old code
   git checkout main
   python -m transformation_portal.lux_depth_v3.orchestrator \
     --input input.jpg --output before/

   # Run with new code
   git checkout feature-branch
   python -m transformation_portal.lux_depth_v3.orchestrator \
     --input input.jpg --output after/
   ```

2. **Create comparison visual**:
   ```python
   import cv2
   import numpy as np

   before = cv2.imread("before/output.png")
   after = cv2.imread("after/output.png")
   diff = cv2.absdiff(before, after)

   # Amplify diff for visibility
   diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

   # Side-by-side comparison
   comparison = np.hstack([before, after, diff_amplified])
   cv2.imwrite("comparison.png", comparison)
   ```

3. **Inspect diff regions**:
   - Look for edge artifacts → feathering/bbox issue
   - Look for global shifts → color space conversion bug
   - Look for material-specific issues → pixel ops bug

**Example**: [`create_sky_comparison.py`](../../../tools/investigations/materials_v3/create_sky_comparison.py) used this approach to diagnose sky color grading.

---

### 2. Isolation Testing

**When to use**: Complex failures where root cause is unclear

**Steps**:

1. **Isolate the stage**:
   ```python
   # Test segmentation in isolation
   from transformation_portal.lux_depth_v3.segmentation_backend import SAM2Backend

   backend = SAM2Backend(...)
   result = backend.segment(image, ...)

   # Inspect output shapes
   for key, mask in result.items():
       print(f"{key}: {mask.shape}, dtype={mask.dtype}")
   ```

2. **Isolate the operation**:
   ```python
   # Test single pixel op
   from transformation_portal.lux_depth_v3.pixel_ops_registry import PIXEL_OPS_REGISTRY

   op_def = PIXEL_OPS_REGISTRY["sky"]["sky_dehaze"]
   before = region.copy()  # Critical: .copy() to avoid view aliasing
   after = op_def.func(region, **params)
   delta = np.abs(after - before).sum()
   print(f"Delta: {delta}")  # Should be > 0 if op does anything
   ```

3. **Test with minimal fixture**:
   ```python
   # Use tiny synthetic image
   image = np.random.rand(64, 64, 3).astype(np.float32)
   mask = np.zeros((64, 64), dtype=bool)
   mask[16:48, 16:48] = True  # Small square region

   # Run through suspected buggy code
   result = process_material(image, mask, ...)
   ```

**Example**: The NumPy view aliasing bug was caught by isolation testing that showed `delta == 0` for all pixel operations.

---

### 3. Shape Debugging

**When to use**: NumPy indexing errors, unexpected broadcast behavior, tensor dimension mismatches

**Diagnostic checklist**:

```python
def diagnose_shapes(masks: dict):
    """Print diagnostic info for all masks in a materials dict."""
    for material, mask in masks.items():
        print(f"\n{material}:")
        print(f"  shape: {mask.shape}")
        print(f"  ndim: {mask.ndim}")
        print(f"  dtype: {mask.dtype}")
        print(f"  min/max: {mask.min():.3f} / {mask.max():.3f}")

        # Check for common shape bugs
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                print(f"  ⚠️  Shape is (H, W, 1) - squeeze needed?")
            elif mask.shape[0] == 1:
                print(f"  ⚠️  Shape is (1, H, W) - squeeze needed?")

        # Check for boolean vs float confusion
        if mask.dtype == bool and mask.max() > 1:
            print(f"  ⚠️  Boolean mask with values > 1")

        # Check for unexpected NaNs
        if np.isnan(mask).any():
            nan_count = np.isnan(mask).sum()
            print(f"  ⚠️  Contains {nan_count} NaN values")
```

**Common shape bugs**:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `ValueError: too many indices` | Mask is 3D (H,W,1), code expects 2D | `mask = np.squeeze(mask)` |
| `ValueError: operands could not be broadcast` | Incompatible shapes in binary op | Canonicalize shapes first |
| `IndexError: too many values to unpack` | `ys, xs = np.where(mask)` with 3D mask | Squeeze before np.where |

**Example**: The 3D mask bug was diagnosed by adding shape checks and discovering segmentation backends returned `(H, W, 1)` instead of `(H, W)`.

---

### 4. Telemetry-Driven Diagnosis

**When to use**: Production issues, intermittent failures, performance regressions

**Setup telemetry**:

```python
class PixelOpsExecutor:
    def apply_ops(self, ...):
        stats = {"operations": {}, "timing": {}, "deltas": {}}

        for material_key, ops in ...:
            op_start = time.perf_counter()

            before = region.copy()
            after = self._apply_op(region, op)
            delta = np.abs(after - before).sum()

            elapsed = time.perf_counter() - op_start

            stats["operations"][material_key] = len(ops)
            stats["timing"][material_key] = elapsed
            stats["deltas"][f"{material_key}/{op_name}"] = delta

        # Log only if something unexpected
        if any(d == 0 for d in stats["deltas"].values()):
            logger.warning(f"Zero delta detected: {stats}")

        return output, stats
```

**Analyze telemetry**:

```bash
# Extract telemetry from logs
grep "Zero delta detected" logs/processing.log | jq '.deltas'

# Look for patterns:
# - Which materials always have zero delta? (Indicates op not applying)
# - Which ops are slowest? (Performance regression)
# - Are there outliers? (Edge case failures)
```

**Example**: Telemetry showing `delta=0` for all materials led to discovery of the NumPy view aliasing bug.

---

### 5. Regression Test Design

**When to use**: After fixing a bug, to prevent reintroduction

**Steps**:

1. **Reproduce the bug**:
   ```python
   def test_reproduces_bug():
       """Reproduce the exact failure before fix."""
       # Use the buggy code path
       result = buggy_function(...)

       # Assert the bug manifests
       assert result == WRONG_VALUE, "Bug no longer reproduces!"
   ```

2. **Apply the fix**:
   ```python
   def _canonical_mask(mask):
       """Fix: Normalize 3D masks to 2D."""
       if mask.ndim == 3:
           mask = np.squeeze(mask)
       return mask.astype(np.float32)
   ```

3. **Write regression test**:
   ```python
   def test_handles_3d_masks():
       """Regression: 3D mask shape bug (Phase A1)."""
       mask_3d = np.random.rand(64, 64, 1)  # Buggy shape

       # Should not crash
       canonical = _canonical_mask(mask_3d)

       # Should be 2D
       assert canonical.ndim == 2
       assert canonical.shape == (64, 64)
   ```

4. **Add to CI**:
   - Mark with `@pytest.mark.regression`
   - Link to original bug report in docstring
   - Include in PR gating checks

**Example**: `test_canonical_mask_squeezes_3d_arrays` in `test_materials_v3_phase_a_hardening.py`.

---

## Investigation Workflow (End-to-End)

### Phase 1: Discovery (0.5-1 hour)

1. **Capture the symptom**:
   - Screenshot of visual artifact
   - Error traceback
   - Unexpected output value
   - Performance regression data

2. **Establish baseline**:
   ```bash
   git checkout main
   python reproduce_issue.py  # Should NOT show issue

   git checkout feature-branch
   python reproduce_issue.py  # Should show issue
   ```

3. **Narrow the scope**:
   - Which materials affected? (sky only? all?)
   - Which operations affected? (feathering? color grading?)
   - Which inputs trigger it? (large images? specific formats?)

---

### Phase 2: Root Cause Analysis (1-3 hours)

1. **Add diagnostic logging**:
   ```python
   logger.debug(f"Shape before op: {region.shape}")
   logger.debug(f"Delta after op: {np.abs(after - before).sum()}")
   ```

2. **Isolate the stage**:
   - Test segmentation alone
   - Test pixel ops alone
   - Test materials detection alone

3. **Use binary search**:
   ```bash
   # If bug introduced in last 10 commits
   git bisect start
   git bisect bad HEAD
   git bisect good HEAD~10

   # Git will checkout commits; run test each time
   python test_reproducer.py && git bisect good || git bisect bad
   ```

4. **Inspect intermediate values**:
   ```python
   import ipdb; ipdb.set_trace()  # Debugger breakpoint

   # At breakpoint, inspect:
   # - mask.shape, mask.dtype
   # - np.where(mask) returns
   # - operation inputs/outputs
   ```

---

### Phase 3: Hypothesis Validation (0.5-1 hour)

1. **Form hypothesis**:
   > "I think the bug is caused by NumPy view aliasing in pixel ops, where `before = output[y0:y1, x0:x1]` creates a view not a copy, so `delta = abs(after - before)` always shows zero because both point to the same memory."

2. **Design minimal test**:
   ```python
   import numpy as np

   output = np.ones((100, 100, 3))
   before = output[10:20, 10:20]  # VIEW (bug)
   after = output[10:20, 10:20] * 2
   output[10:20, 10:20] = after

   delta = np.abs(after - before).sum()
   print(f"Delta: {delta}")  # Will be 0 (bug confirmed!)

   # Now test fix
   before = output[10:20, 10:20].copy()  # COPY (fix)
   after = output[10:20, 10:20] * 2
   output[10:20, 10:20] = after

   delta = np.abs(after - before).sum()
   print(f"Delta: {delta}")  # Will be > 0 (fix validated!)
   ```

3. **Validate in isolation**:
   - Does minimal test reproduce the bug?
   - Does the fix resolve it in minimal test?
   - Does the fix resolve it in integration test?

---

### Phase 4: Fix Implementation (1-2 hours)

1. **Apply minimal fix**:
   - Change as few lines as possible
   - Add comments explaining WHY
   - Update docstrings if behavior changes

2. **Add regression test**:
   - Test the specific bug scenario
   - Test edge cases (empty mask, full-image mask, etc.)
   - Link to investigation report in docstring

3. **Validate fix doesn't break anything**:
   ```bash
   # Run full test suite
   pytest tests/ -v

   # Run on production test set
   python scripts/validate_on_test_set.py

   # Visual inspection of outputs
   python scripts/generate_comparison_grid.py
   ```

---

### Phase 5: Documentation (0.5-1 hour)

1. **Write investigation report**:
   - Symptom (with screenshot/error)
   - Root cause (with code snippet)
   - Fix (with before/after)
   - Validation (test results)

2. **Update architecture docs** (if pattern is generalizable):
   - Add to "Common Pitfalls" section
   - Update best practices
   - Link to investigation

3. **Share learnings**:
   - Team Slack post
   - PR description includes investigation link
   - Add to onboarding materials

---

## Common Pitfalls & Solutions

### Pitfall 1: NumPy View vs Copy Confusion

**Symptom**: Delta calculations show zero, operations appear to do nothing.

**Root Cause**: `array[slice]` returns a VIEW not a COPY.

**Solution**:
```python
# ❌ WRONG: View aliasing
before = output[y0:y1, x0:x1]
output[y0:y1, x0:x1] = after
delta = np.abs(after - before)  # Zero! Both point to same memory

# ✅ CORRECT: Explicit copy
before = output[y0:y1, x0:x1].copy()
output[y0:y1, x0:x1] = after
delta = np.abs(after - before)  # Correct!
```

---

### Pitfall 2: 3D Mask Shape Assumptions

**Symptom**: `IndexError: too many values to unpack`, `ValueError: too many indices`.

**Root Cause**: Some backends return masks as `(H, W, 1)` instead of `(H, W)`.

**Solution**:
```python
def _canonical_mask(mask):
    """Normalize mask to 2D float32."""
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    return mask.astype(np.float32)
```

**When to use**: At every entry point that accepts masks from external sources.

---

### Pitfall 3: Feathering Edge Clipping

**Symptom**: Artifacts appear at material boundaries, especially edges of image.

**Root Cause**: Gaussian blur kernel extends beyond bbox, causing edge values to clip.

**Solution**:
```python
def _expand_bbox_with_padding(bbox, sigma, image_shape):
    """Expand bbox by 3*sigma to prevent feathering edge clipping."""
    y0, y1, x0, x1 = bbox
    pad = int(math.ceil(3 * sigma))

    y0 = max(0, y0 - pad)
    y1 = min(image_shape[0], y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(image_shape[1], x1 + pad)

    return (y0, y1, x0, x1)
```

---

### Pitfall 4: Mask Overlap Ambiguity

**Symptom**: Materials overlap (e.g., sky + glass), pixel ops applied twice, resulting in over-processing.

**Root Cause**: Multiple materials claim the same pixels.

**Solution**:
```python
def _resolve_overlaps(materials, priorities, image_shape):
    """Assign each pixel to highest-priority material."""
    assigned = np.zeros(image_shape[:2], dtype=np.uint8)
    resolved = {}

    for material, mask in sorted(materials.items(), key=lambda x: priorities.get(x[0], 0), reverse=True):
        # Only assign unassigned pixels
        available = (assigned == 0) & (mask > 0.5)
        resolved[material] = available.astype(np.float32)
        assigned[available] = 1  # Mark as assigned

    return resolved
```

---

## Tools & Scripts

### Diagnostic Scripts

All scripts used during Materials V3 investigations are in [`tools/investigations/materials_v3/`](../../../tools/investigations/materials_v3/):

- **`diagnose_sky_issue.py`**: Diagnose sky color grading issues
- **`create_sky_comparison.py`**: Generate visual comparisons for sky processing

See that directory's README for usage.

---

### Useful Snippets

**Quick shape inspector**:
```python
def inspect(name, arr):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}")
```

**Visual diff generator**:
```python
def visual_diff(before, after, amplify=10):
    diff = np.abs(after.astype(float) - before.astype(float))
    amplified = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return amplified
```

**Telemetry logger**:
```python
import json
def log_telemetry(stats, path="telemetry.json"):
    with open(path, "a") as f:
        f.write(json.dumps(stats) + "\n")
```

---

## Lessons Learned

### 1. Always Copy, Never Assume Views Are Safe

NumPy view semantics are subtle. When in doubt, use `.copy()`.

### 2. Canonicalize Inputs at Boundaries

Don't trust external data shapes. Normalize at every entry point.

### 3. Telemetry Beats Intuition

Add logging/metrics early. Zero-delta telemetry caught the view aliasing bug that visual inspection missed.

### 4. Minimal Reproducers Are Gold

The faster you can reproduce a bug (ideally <10 lines), the faster you can fix it.

### 5. Document While Fresh

Write investigation reports immediately after fixing bugs, not weeks later. Context fades fast.

---

## Related Documentation

- **[Investigation Reports](README.md)**: Specific bug investigations
- **[Phase A Summary](../../materials/PHASE_A_COMPLETE.md)**: Hardening work motivated by these bugs
- **[Phase B Summary](../../materials/PHASE_B_COMPLETE.md)**: Sky material work
- **[Testing Best Practices](../../testing/BEST_PRACTICES.md)**: Regression test patterns

---

**Last Updated**: February 14, 2026
**Maintainer**: Materials V3 Team
