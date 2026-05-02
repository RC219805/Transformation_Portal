# Bug Report: Code Review of Reformatting + Feature Changes

**Date:** 2025-01-XX
**Scope:** Full diff review (~12,263 lines) across 14 source files and 4 test files
**Nature:** Mixed reformatting (line-length reduction) + behavioral changes + new features

---

## Bugs

### BUG-1: `pragma: no cover` comment misplaced (postprocessing.py)

**Severity:** Medium
**File:** `src/transformation_portal/lux_depth_v3/postprocessing.py:41-42`

The `# pragma: no cover` comment was moved from the `except` line to the line *above* it:

```python
# BEFORE (correct):
    except Exception as exc:  # pragma: no cover - optional dependency
# AFTER (broken):
    # pragma: no cover - optional dependency
    except Exception as exc:
```

`coverage.py` only recognizes `# pragma: no cover` when it appears on the line of the construct it should exclude. Placing it on a standalone comment line above the `except` block means the `except` body is no longer excluded from coverage. This may cause CI coverage regressions.

**Fix:** Move the pragma back to the `except` line:

```python
    except Exception as exc:  # pragma: no cover
```

---

### BUG-2: `cv2.error` exception type dropped from bilateral filter fallback (postprocessing.py)

**Severity:** Medium
**File:** `src/transformation_portal/lux_depth_v3/postprocessing.py:258-262`

The original code dynamically added `cv2.error` to the caught exception types for the joint bilateral filter:

```python
# BEFORE:
opencv_error = getattr(opencv, "error", None)
expected_joint_filter_errors = (AttributeError, TypeError, ValueError)
if isinstance(opencv_error, type) and issubclass(opencv_error, Exception):
    expected_joint_filter_errors += (opencv_error,)
# ...
except expected_joint_filter_errors as exc:
```

The new code hardcodes only three types:

```python
# AFTER:
except (
    AttributeError,
    TypeError,
    ValueError,
) as exc:
```

**Impact:** When `cv2.ximgproc.jointBilateralFilter` raises `cv2.error` (OpenCV's native exception, e.g., for unsupported image formats or internal failures), the error now falls through to the generic `except Exception` handler, which logs it as "Unexpected joint bilateral filter failure" instead of the correct "Joint bilateral filter unavailable/incompatible" message. The fallback path still works, but the diagnostic logging is misleading and the exception flow is less precise.

**Fix:** Re-add dynamic `cv2.error` detection:

```python
_cv2_errors = [AttributeError, TypeError, ValueError]
_cv2_error_cls = getattr(opencv, "error", None)
if isinstance(_cv2_error_cls, type) and issubclass(_cv2_error_cls, Exception):
    _cv2_errors.append(_cv2_error_cls)
# ...
except tuple(_cv2_errors) as exc:
```

---

### BUG-3: Misleading `DepthResult` constructor formatting (orchestrator.py)

**Severity:** Low (correctness OK, maintenance hazard)
**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py:1869-1875`

```python
result_candidate = DepthResult(
    depth_map=cached_depth,
    original_image=preprocessed_uint8,
    metadata=cache_metadata, depth_units="meters"
    if backend_id == "depth_pro" else "relative",
    backend_id=backend_id, device="cache",
    dtype="float32", input_size=original_shape,)
```

While Python parses this correctly (the conditional binds to `depth_units`, not `metadata`), the formatting strongly implies the ternary applies to `metadata`. A future maintainer could easily misread or mis-refactor this.

**Fix:** Use explicit line breaks per keyword argument:

```python
result_candidate = DepthResult(
    depth_map=cached_depth,
    original_image=preprocessed_uint8,
    metadata=cache_metadata,
    depth_units="meters" if backend_id == "depth_pro" else "relative",
    backend_id=backend_id,
    device="cache",
    dtype="float32",
    input_size=original_shape,
)
```

---

### BUG-4: Bizarre `is` / `None or` line splitting (orchestrator.py)

**Severity:** Low (correctness OK, readability hazard)
**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py:2418-2423`

```python
if self.config.generate_pbr and (
    pbr_assets
    is
    None or
    not self._verify_pbr_outputs(
        pbr_assets)):
```

The `is` keyword sits alone on its own line. While syntactically valid, this is extremely unusual Python formatting. Any linter or formatter (black, ruff) would flag this.

**Fix:**

```python
if self.config.generate_pbr and (
    pbr_assets is None
    or not self._verify_pbr_outputs(pbr_assets)
):
```

---

## Behavioral Changes (Not Just Formatting)

These are intentional changes bundled into the reformatting diff. They should be called out separately for reviewers.

### CHANGE-1: `v2_preset` now defaults to `"default"` via `or "default"`

**Files:** `orchestrator.py` (two locations: V2 runner call and V2Metadata construction)

```python
# BEFORE:
preset=self.config.v2_preset,
# AFTER:
preset=(self.config.v2_preset or "default"),
```

If `v2_preset` is `None` or `""`, behavior changes. Previously `None` would propagate; now it silently becomes `"default"`.

### CHANGE-2: `DepthResult` import path changed

**File:** `postprocessing.py`

```python
# BEFORE:
from .inference import DepthResult
# AFTER:
from ..depth.backends.protocol import DepthResult
```

Both the `TYPE_CHECKING` import and the runtime import in `fuse_multiview` were updated. The protocol `DepthResult` has a `depth` property alias for backward compatibility, so this is safe but changes the dependency graph.

### CHANGE-3: `DepthResult` positional → keyword constructor in `fuse_multiview`

**File:** `postprocessing.py:377-381`

```python
# BEFORE:
return DepthResult(fused, results[0].original_image, metadata={"fusion_mode": ...})
# AFTER:
return DepthResult(depth_map=fused, original_image=results[0].original_image, metadata={"fusion_mode": ...})
```

Good change — makes the call self-documenting and resilient to field reordering.

### CHANGE-4: `object.__setattr__` for frozen/protected result objects

**File:** `orchestrator.py:1933-1937`

```python
# BEFORE:
result_candidate.depth = depth_candidate
# AFTER:
object.__setattr__(result_candidate, "depth", depth_candidate)
```

Bypasses potential `__setattr__` overrides or frozen dataclass restrictions. Intentional fix.

### CHANGE-5: `linear_decoder.py` — (3,4) and (4,3) color matrix support

**File:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py:834-605`

New feature: `_select_valid_color_matrix` now accepts LibRaw/rawpy (3,4) and (4,3) matrix layouts by deterministically contracting to 3×3. Well-tested with two new unit tests.

### CHANGE-6: `EnhancementStage` — depth map resize on dimension mismatch

**File:** `src/transformation_portal/stage_graph/stages/enhancement.py:223-293`

New defensive feature: when depth map shape doesn't match image shape, it's resized with bilinear interpolation before tone mapping. Includes a new `_resize_depth_map` static method. Covered by two new tests.

---

## New Tests Added

| Test File | Test Name | Purpose |
| --- | --- | --- |
| `test_linear_decoder.py` | `test_3x4_color_matrix_is_contracted_to_3x3` | Validates 3×4 matrix contraction |
| `test_linear_decoder.py` | `test_4x3_rgb_xyz_matrix_fallback_is_contracted_to_3x3` | Validates 4×3 matrix contraction |
| `test_enhancement_stage.py` | `test_enhancement_stage_resizes_mismatched_depth_map` | Validates depth resize |
| `test_v2_enhance.py` | `test_enhance_image_with_mismatched_depth_dimensions` | E2E depth resize |
| `test_app_orchestrator_runtime.py` | `test_portal_verbose_quiet_conflict_is_notified_and_blocked` | UI mutual-exclusion |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_verbose_and_quiet_combination` | CLI mutual-exclusion |
| `test_app_orchestrator_runtime.py` | `test_argv_normalization_ignores_sam2_checkpoint_path_when_backend_is_not_sam2` | SAM2 arg filtering |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_invalid_raw_ingest_mode` | Input validation |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_invalid_raw_wb_mode` | Input validation |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_invalid_raw_demosaic` | Input validation |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_invalid_reconstruction_tier` | Input validation |
| `test_app_orchestrator_runtime.py` | `test_argv_rejects_invalid_log_level` | Input validation |
| `test_app_orchestrator_runtime.py` | `test_portal_surfaces_pre_run_diagnostics_and_expected_outputs` | Portal HTML |
| `test_app_orchestrator_runtime.py` | `test_portal_exposes_run_card_quick_actions` | Portal HTML |
| `test_lux_depth_v3_cli.py` | `test_invalid_grouping_mode` | CLI validation |
| `test_lux_depth_v3_cli.py` | `test_reconstruction_requires_non_commercial` | License gate |
| `test_lux_depth_v3_cli.py` | `test_reconstruction_requires_research_tools_license` | License gate |
| `test_lux_depth_v3_cli.py` | `test_cameras_sidecar_path_must_exist` | Path validation |
| `test_lux_depth_v3_cli.py` | `test_reconstruction_flags_wire_into_config` | Config wiring |
| `test_lux_depth_v3_cli.py` | `test_reconstruction_flags_in_help` | Help text |

---

## General Observations

### Reformatting Quality

The bulk of this diff (~90%) is line-length reformatting. The approach is aggressive — many lines are broken at 40-50 characters rather than the typical 79/88/120 column limits. This creates:

- Single-word-per-line patterns that are harder to scan than the originals
- `getattr` attribute names split across string literals (e.g., `"apex_depth_max_high_" "saturation_fraction"`) — correct via Python string concatenation but fragile and grep-unfriendly
- Dictionary keys split across string literals (e.g., `"reconstruction_scene" "_manifest_path"`) — same issue
- Comments truncated to the point of losing context (e.g., docstrings reduced from full sentences to fragments)

### Recommendation

- **BUG-1 and BUG-2** should be fixed before merge
- **BUG-3 and BUG-4** are style issues but should be cleaned up
- The behavioral changes (CHANGE-1 through CHANGE-6) should be reviewed and approved independently of the reformatting
- Consider running the full test suite (`make test-full`) to catch any coverage regressions from BUG-1
