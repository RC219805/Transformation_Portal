# ADR-031: Test Dependency Isolation Contract

**Status:** Accepted
**Date:** 2026-02-15
**Authors:** Transformation Portal Team
**Related:** Issue #796 (CI Health & Stability), PR #949 (CI P0 Blockers)

---

## Context

On Feb 13-15, 2026, main branch CI experienced a 100% failure rate (20 consecutive failures) due to test dependency isolation violations. Two critical patterns emerged:

### Problem 1: Import-Before-Mock Anti-Pattern

Tests marked with `@pytest.mark.ml` were failing in CI's offline environment:

```python
# ❌ BAD: Imports transformers BEFORE patching
@pytest.mark.ml
def test_something():
    with patch("transformers.CLIPModel"):  # ModuleNotFoundError here!
        ...
```

**Root cause:** `@patch()` imports the module to find the attribute **before** applying the mock. In CI's offline mode (no transformers installed), this fails immediately.

### Problem 2: PIL uint16 RGB Mode Misuse

Tests assumed PIL supported uint16 RGB mode:

```python
# ❌ BAD: PIL doesn't support this
img = Image.fromarray(uint16_array, mode="RGB")  # TypeError!
```

**Root cause:** PIL only supports RGB mode for uint8, and I;16 mode for uint16 grayscale. uint16 RGB requires specialized libraries (tifffile, imageio).

### Impact

- **CI unavailable for 3 days** (no regression detection)
- **Merge friction** (broken main blocked new work)
- **Trust erosion** (developers unsure if tests are reliable)

---

## Decision

We establish a **Test Dependency Isolation Contract** with three enforcement layers:

### 1. Architecture Rules (Documented)

#### Rule: ML-Marked Tests MUST Handle Missing Dependencies Gracefully

**Pattern A: Module-level import guard** (RECOMMENDED)
```python
# At module top
try:
    import transformers
    import torch
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# In test class
@pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies required")
def test_with_transformers(self):
    # Safe to import/use transformers here
    ...
```

**Pattern B: pytest.importorskip()**
```python
def test_with_transformers(self):
    transformers = pytest.importorskip("transformers")
    # Safe to use transformers here
    ...
```

**Pattern C: Conditional execution**
```python
@pytest.mark.ml
def test_with_transformers(self):
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers not installed")
    # Safe to use transformers here
    ...
```

#### Rule: Core Tests MUST NOT Import ML Dependencies

Tests NOT marked `@pytest.mark.ml` must not import:
- `transformers`
- `torch` (except via safe device detection utilities)
- `diffusers`
- `depth_anything_v3` (or other ML backends)

**Rationale:** Core CI job runs without ML dependencies for speed (~30s vs ~5min).

#### Rule: Use Appropriate Libraries for Data Types

- **uint8 images:** Use PIL (Image.fromarray with mode="RGB")
- **uint16 grayscale:** Use PIL (mode="I;16")
- **uint16 RGB:** Use `tifffile.imwrite()` or `imageio.imwrite()`
- **Float32/HDR:** Use OpenEXR (if available) or tifffile

**Anti-patterns:**
- ❌ `Image.fromarray(uint16_array, mode="RGB")` → TypeError
- ❌ Forcing uint16 into uint8 without justification
- ❌ Mock libraries instead of using real offline-safe alternatives

---

### 2. Static Analysis (Pre-commit Hook)

**File:** `scripts/check_ml_test_isolation.sh`

**Checks:**
1. Scan for `@patch("transformers` or `@patch("torch` patterns
2. Verify corresponding module has import guard OR `pytest.importorskip()`
3. Fail with actionable error message if violation detected

**Example error:**
```
ERROR: ML mock detected without import guard

File: tests/spatial_ai/segmentation/test_material_classifier.py
Line: 42
Pattern: @patch("transformers.CLIPModel")

ML dependencies may not be installed in CI (offline mode).
Use one of these patterns:

1. Module-level guard:
   try:
       import transformers
       HAS_ML = True
   except ImportError:
       HAS_ML = False

   @pytest.mark.skipif(not HAS_ML, ...)

2. pytest.importorskip():
   transformers = pytest.importorskip("transformers")

See: docs/architecture/ADR-031-test-dependency-isolation.md
```

---

### 3. CI Runtime Validation

**Workflow:** `.github/workflows/ci-quality-firewall.yml`

**Job:** `test-isolation`

**Check:**
```bash
# Verify core tests don't import ML dependencies
pytest tests/ -m "not ml" --collect-only 2>&1 | \
  grep -i "transformers\|torch" && exit 1 || true
```

**Outcome:**
- ✅ **Pass:** Core tests isolated (no ML imports detected)
- ❌ **Fail:** Core test attempted ML import (fast-fail with clear diagnostic)

---

## Rationale

### Why Three Enforcement Layers?

1. **Documentation (ADR):** Educates developers, provides reference
2. **Pre-commit Hook:** Catches violations at commit time (fastest feedback)
3. **CI Validation:** Safety net if hook bypassed (definitive truth)

### Why Not Just Install ML Deps Everywhere?

- **Speed:** Core CI runs in ~30s without ML deps, ~5min with
- **Offline safety:** Prevents model downloads in CI (determinism + cost)
- **Clarity:** Forces explicit marking of ML-dependent tests

### Why Module-level Guards Over importorskip()?

**Preference:** Module-level guards (Pattern A)

**Advantages:**
- Single import check per module (not per test)
- Clear intent at module top
- Easier to audit (grep for `HAS_ML_DEPS`)
- Skip decorators are more visible

**Disadvantages:**
- Slightly more boilerplate
- Need to update if adding new ML deps

**When to use importorskip():** Single-test optional dependency (rare case).

---

## Consequences

### Positive

✅ **Prevents regression:** Import-before-mock failures impossible
✅ **Faster feedback:** Pre-commit hook catches violations locally
✅ **Clear contract:** Developers know the rules
✅ **Maintainable:** Enforcement is automated, not manual review

### Negative

⚠️ **Boilerplate:** Module-level guards add ~8 lines per test module
⚠️ **Hook complexity:** Pre-commit hook must parse Python (regex fragile)
⚠️ **CI cost:** +30s for test isolation validation job

### Mitigations

- Boilerplate: Acceptable trade-off for reliability
- Hook complexity: Start with simple regex, evolve to AST parsing if needed
- CI cost: Runs in parallel, doesn't block other jobs

---

## Examples

### Good: ML Test with Module-level Guard

```python
# tests/spatial_ai/segmentation/test_sam2_backend.py

try:
    import torch
    from transformers import AutoModel
    HAS_SAM2_DEPS = True
except ImportError:
    HAS_SAM2_DEPS = False

import pytest


class TestSAM2Backend:
    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_SAM2_DEPS, reason="SAM2 dependencies required")
    def test_segment_with_sam2(self):
        # Safe to use torch/transformers here
        backend = SAM2Backend(device="cpu")
        result = backend.segment(...)
        assert result.masks.shape == (1, 512, 512)
```

### Good: Core Test with Safe PIL Usage

```python
# tests/spatial_ai/ingest/test_linear_decoder.py

import tifffile
import numpy as np
from PIL import Image


class TestLinearDecoder:
    def test_uint16_tiff_decode(self, tmp_path):
        """Test 16-bit TIFF decoding."""
        # Use tifffile for uint16 RGB (PIL doesn't support this mode)
        test_img = (np.random.rand(100, 100, 3) * 65535).astype(np.uint16)
        test_img_path = tmp_path / "test.tiff"
        tifffile.imwrite(test_img_path, test_img)

        result = decode(test_img_path, gamma=1.0)
        assert result.input_format == "TIFF"
```

### Bad: Import-Before-Mock Anti-Pattern

```python
# ❌ This will fail in offline CI

@pytest.mark.ml
def test_with_mock(self):
    with patch("transformers.CLIPModel"):  # ModuleNotFoundError!
        ...
```

**Fix:** Add module-level guard and skip decorator (see Good example above).

### Bad: PIL uint16 RGB Misuse

```python
# ❌ This will fail: PIL doesn't support uint16 RGB

img = Image.fromarray(uint16_array, mode="RGB")  # TypeError!
```

**Fix:** Use tifffile or imageio for uint16 RGB operations.

---

## Enforcement Checklist

- [x] ADR published (this document)
- [x] Pre-commit hook implemented (`scripts/check_ml_test_isolation.sh`)
- [x] Hook integrated (`.pre-commit-config.yaml`)
- [x] CI validation job added (`.github/workflows/ci-quality-firewall.yml` - job `test-isolation`)
- [x] Existing violations fixed (PR #949)
- [ ] Documentation updated (`CONTRIBUTING.md`)

---

## References

- **Issue #796:** CI Health & Stability
- **PR #949:** CI P0 Blockers - ML Import Hygiene + PIL uint16 Fixtures
- **Root cause analysis:** Issue #796 comment (2026-02-15)
- **PIL documentation:** https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
- **pytest.importorskip():** https://docs.pytest.org/en/stable/how-to/skipping.html#skipping-on-a-missing-import-dependency

---

## Revision History

- **2026-02-15:** Initial version (ADR-031 created)
- Status: **Accepted** (enforcement in progress)
