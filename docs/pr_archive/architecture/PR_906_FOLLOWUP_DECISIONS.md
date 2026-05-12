# Architectural Decisions: PR #906 Follow-Up Issues

**Date:** 2026-02-11
**Architect:** Transformation Portal Architect
**Context:** Follow-up to PR #906 Final Verification

---

## Purpose

This document provides **binding architectural decisions** for issues #5 and #6 identified in the PR #906 verification. These issues require Architect input because they involve API contracts and behavioral changes.

Issues #4 and #7 are trivial fixes that do not require architectural decisions.

---

## Issue #5: LinearDecoder `validate_contract=False` Is Non-Functional

### Problem Summary
The `validate_contract` parameter exists but doesn't work because `LinearIngestResult.__post_init__` always enforces gamma==1.0.

### Decision: **REMOVE `validate_contract` PARAMETER**

**Rationale:**
1. **SpatialCaptureV1 contract is non-negotiable.** The entire purpose of the Spatial AI ingest pipeline is to produce gamma=1.0 linear light. This is not a preference; it's a contract invariant.

2. **Simplicity over flexibility.** If we never intend to support gamma≠1.0, don't pretend we do. A simple, clear API is better than a fake override.

3. **Fail fast.** If someone tries to use gamma≠1.0, we should fail immediately at construction time, not later during decode.

4. **No known use case.** Phase 1 and Phase 2 specs don't require gamma overrides. If a future phase does, we can add it back with proper threading.

### Implementation Directive

**Change 1:** Remove `validate_contract` parameter from `LinearDecoder.__init__`:

```python
class LinearDecoder:
    def __init__(self, gamma: float = 1.0, bit_depth: int = 32):
        """Initialize linear decoder.

        Args:
            gamma: Gamma for decode (must be 1.0 for linear light).
            bit_depth: Output bit depth (32 for float32).

        Raises:
            ValueError: If gamma != 1.0.
        """
        if abs(gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Linear ingest requires gamma=1.0 (SpatialCaptureV1 contract), got {gamma}.\n"
                "This decoder is for linear light only. For display-ready output, "
                "use transformation_portal.lux_depth_v3.raw_loader instead."
            )

        self.gamma = gamma
        self.bit_depth = bit_depth
```

**Change 2:** Keep `LinearIngestResult.__post_init__` validation as-is (always enforces gamma==1.0).

**Change 3:** Update docstring and examples to remove all references to `validate_contract`.

**Change 4:** Add migration note to CHANGELOG:
```markdown
### Breaking Changes
- `LinearDecoder.__init__` no longer accepts `validate_contract` parameter
- gamma=1.0 is now strictly enforced (was always enforced in practice)
- If you need non-linear decode, use `lux_depth_v3.raw_loader` instead
```

**Testing:**
- Add test that verifies gamma≠1.0 raises ValueError at construction
- Verify existing tests still pass
- Check that no internal code uses `validate_contract=False`

---

## Issue #6: EXR Fallback Clips HDR Values

### Problem Summary
When OpenEXR is not installed, the fallback saves as 16-bit TIFF and clips values >1.0, violating HDR preservation claims.

### Decision: **FAIL LOUDLY ON HDR DATA WITHOUT OPENEXR**

**Rationale:**
1. **Research integrity.** Silently clipping HDR data could corrupt training datasets. Better to fail loudly than silently produce wrong results.

2. **OpenEXR is required anyway.** Phase 1 requirements include OpenEXR in `requirements.txt`. The fallback exists for defensive coding, not as a supported path.

3. **Clear user guidance.** A loud failure with actionable error message is better than a warning that gets ignored.

4. **Preserves correctness over convenience.** If we claim "HDR preservation," we must enforce it.

### Implementation Directive

**Change:** Modify `_save_exr` to detect HDR data and fail if OpenEXR unavailable:

```python
def _save_exr(self, linear_rgb: np.ndarray, output_dir: Path, stem: str) -> Path:
    """Save linear RGB as EXR (float32 HDR).

    Args:
        linear_rgb: Float32 RGB array.
        output_dir: Output directory.
        stem: Output filename stem.

    Returns:
        Path to saved EXR file.

    Raises:
        RuntimeError: If HDR data detected but OpenEXR not available.
    """
    output_path = output_dir / f"{stem}_linear.exr"

    try:
        import Imath
        import OpenEXR

        # ... existing OpenEXR save logic ...

    except ImportError:
        # Check if data contains HDR values
        max_value = linear_rgb.max()
        has_hdr = max_value > 1.0

        if has_hdr:
            raise RuntimeError(
                f"HDR data detected (max value: {max_value:.3f}) but OpenEXR not installed.\n\n"
                "Cannot preserve HDR range with fallback TIFF format (would clip to 1.0).\n\n"
                "Install OpenEXR to preserve HDR:\n"
                "  pip install OpenEXR\n\n"
                "Or disable EXR export if you don't need it:\n"
                "  decoder.decode(input_path, emit_exr=False)\n\n"
                "See: docs/architecture/ADR-026-apex-research-ultra.md"
            )

        # SDR data (max <= 1.0): safe to use 16-bit TIFF fallback
        logger.warning(
            "OpenEXR not available, using 16-bit TIFF fallback.\n"
            "This is only suitable for SDR data (max value <= 1.0).\n"
            "Install OpenEXR for proper HDR support: pip install OpenEXR"
        )

        output_path = output_dir / f"{stem}_linear.tiff"
        img_uint16 = np.clip(linear_rgb * 65535, 0, 65535).astype(np.uint16)
        img = Image.fromarray(img_uint16, mode="RGB")
        img.save(output_path, format="TIFF", compression="lzw")

    logger.debug(f"Saved linear output: {output_path}")
    return output_path
```

**Testing:**
- Add test: `test_hdr_data_without_openexr_raises_error()`
  - Mock `import OpenEXR` to raise ImportError
  - Create HDR data with values >1.0
  - Verify RuntimeError is raised with clear message
- Add test: `test_sdr_data_without_openexr_uses_fallback()`
  - Mock `import OpenEXR` to raise ImportError
  - Create SDR data with values <=1.0
  - Verify fallback succeeds with warning

**Documentation:**
- Update ADR-026 to explicitly state OpenEXR is required for HDR data
- Add troubleshooting section to README with OpenEXR install instructions

---

## Implementation Priority

| Issue | Decision | Complexity | Priority |
|-------|----------|------------|----------|
| #4 (Test names) | Trivial fix | Low | High (quick win) |
| #7 (Docstring) | Trivial fix | Low | High (quick win) |
| #5 (validate_contract) | Remove parameter | Medium | Medium |
| #6 (EXR fallback) | Fail loudly | Medium | Medium |

**Recommended order:**
1. Fix #4 and #7 first (quick wins, ~30 min)
2. Implement #5 (medium effort, ~1 hour)
3. Implement #6 (medium effort, ~1 hour)
4. Full test suite + documentation updates (~1 hour)

**Total estimated effort:** 3.5 hours

---

## Breaking Changes Summary

### Issue #5: `validate_contract` Removal

**Breaking:** Yes (API parameter removed)
**Impact:** Low (parameter was non-functional, unlikely to be used)
**Migration:** Remove `validate_contract` argument from any `LinearDecoder()` calls

### Issue #6: HDR Fallback Behavior

**Breaking:** Yes (new RuntimeError in fallback path)
**Impact:** Low (fallback path not expected in normal operation)
**Migration:** Install OpenEXR or ensure data is SDR-only

---

## Enforcement and Review

### Implementation Review Checklist

When reviewing the follow-up PR, verify:

**Issue #5:**
- [ ] `validate_contract` parameter removed from `__init__`
- [ ] Error message mentions `lux_depth_v3.raw_loader` as alternative
- [ ] No references to `validate_contract` in docstrings/examples
- [ ] CHANGELOG has breaking change note
- [ ] Test added for gamma≠1.0 rejection

**Issue #6:**
- [ ] HDR detection checks `linear_rgb.max() > 1.0`
- [ ] RuntimeError raised with actionable message
- [ ] SDR fallback still works (with warning)
- [ ] Tests for both HDR rejection and SDR fallback
- [ ] ADR-026 updated to state OpenEXR requirement

**Issue #4:**
- [ ] Test uses distinct model names (e.g., `"synthetic"`, `"synthetic_2"`)
- [ ] Test actually creates two model results in fusion
- [ ] Test still passes

**Issue #7:**
- [ ] Docstring matches return value
- [ ] References protocol documentation pattern

---

## Post-Implementation Actions

After follow-up PR is merged:

1. **Update ADR-026** to reflect new OpenEXR requirement enforcement
2. **Update README** with clear OpenEXR installation instructions
3. **Run full test suite** to verify no regressions
4. **Document in CHANGELOG** under "Breaking Changes" section
5. **Close verification issue** linking to this decision document

---

## Architectural Sign-Off

These decisions are **binding** for the follow-up PR implementation.

Deviations require explicit Architect approval and updated ADR.

**Architect:** Transformation Portal Architect
**Date:** 2026-02-11
**Status:** APPROVED FOR IMPLEMENTATION

---

## References

- Verification report: `docs/pr_archive/architecture/PR_906_FINAL_VERIFICATION.md`
- Follow-up tracking: `docs/pr_archive/architecture/PR_906_FOLLOWUP_ISSUES.md`
- ADR-026: `docs/architecture/ADR-026-apex-research-ultra.md`
- ADR-023: `docs/architecture/ADR-023-spatial-ai-ingest-isolation.md`
