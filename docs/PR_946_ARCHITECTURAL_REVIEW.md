# PR #946 Architectural Review
# Spatial AI Foundation Phase I — Linear Ingest Pipeline

**Review Date:** 2025-02-15
**Reviewer:** Transformation Portal Architect
**PR Branch:** `feature/spatial-ai-linear-ingest-phase1`
**Target Branch:** `main`
**Commit:** `be3ec61f`

---

## Executive Summary

**RECOMMENDATION: ✅ APPROVE WITH MINOR PRE-MERGE HARDENING**

PR #946 delivers a **production-ready, well-architected linear ingest pipeline** that successfully achieves Phase I objectives. The implementation demonstrates strong architectural discipline with:

- **Complete isolation** from rendering pipelines (ADR-023 compliance verified)
- **Deterministic linear contract** (gamma=1.0, float32, strict 8-bit rejection)
- **Comprehensive provenance** (40+ EXIF fields, SHA-256 hashing, versioned schemas)
- **Fail-fast validation** (hard failures on constraint violations)
- **Excellent test coverage** (73 tests passing, 74% coverage estimate)

### Approval Status

**Pre-merge requirements:**
- 2 high-priority hardening items (estimated 6-8 hours)
- 1 medium-priority item for clean future integration (2-3 hours)

**Total additional effort:** 8-11 hours
**Risk if deferred:** Medium to High (see detailed analysis below)

---

## Implementation Quality Assessment

### Strong Points ✅

1. **Architectural Isolation (ADR-023 Compliance)**
   - Complete separation from `lux_depth_v3.raw_loader` ✅
   - Zero shared decode logic ✅
   - CI test verifies no cross-contamination ✅
   - Clear module warnings in docstrings ✅

2. **Linear Light Contract (SpatialCaptureV1)**
   - Gamma=1.0 enforcement (hard-coded, non-negotiable) ✅
   - Float32 dtype validation ✅
   - HDR preservation (values >1.0 allowed) ✅
   - 8-bit rejection via `strict_ingest=True` flag ✅

3. **Provenance Depth**
   - 40+ EXIF fields captured (camera, lens, exposure settings) ✅
   - SHA-256 hashing for both input files and output tensors ✅
   - Timestamp tracking ✅
   - Value range metadata (min/max, HDR detection) ✅
   - ADR references embedded ✅

4. **Schema Versioning**
   - Current version: `1.0.0` ✅
   - Forward/backward compatibility checks ✅
   - Unsupported versions fail loudly ✅
   - Pydantic validation for manifests ✅

5. **Fail-Fast Guardrails**
   - Bit depth validation ✅
   - Dtype enforcement ✅
   - Gamma validation ✅
   - Range validation (NaN/Inf/negative detection) ✅
   - Clear, actionable error messages ✅

6. **Documentation Excellence**
   - 507-line architecture guide ✅
   - 649-line user guide ✅
   - 230-line example workflow ✅
   - ADR-023 and ADR-026 references ✅
   - Clear docstrings with examples ✅

7. **Test Coverage**
   - 73 passing tests ✅
   - Multiple test classes covering:
     - Linear decoder functionality
     - Provenance capture
     - Manifest schema validation
     - Validator guardrails
     - ADR-023 compliance
   - Determinism test (content hash reproducibility) ✅

8. **Zero Torch Dependencies**
   - Pure NumPy/PIL/rawpy implementation ✅
   - No CUDA/GPU dependencies ✅
   - CPU-only, lightweight ✅

---

## Pre-Merge Hardening Recommendations

### Priority Classification

- **P0 (Blocking):** Must fix before merge — security/correctness critical
- **P1 (High):** Should fix before merge — prevents future issues, low implementation risk
- **P2 (Medium):** Fix in Phase I.1 follow-up — improves quality but not blocking
- **P3 (Low):** Defer to Phase II — research/optimization work

---

## 1. Color Space Normalization Boundary (P1 — High Priority)

### Issue

Current implementation:
- Provenance captures `color_space: "linear_sRGB"` as a **static default**
- RAW decoder uses `rawpy.ColorSpace.sRGB` hardcoded
- **No validation** that output is actually in declared color space
- **No failure** if RAW is decoded without explicit camera → working space matrix

### Risk Assessment

**Risk if deferred:** ⚠️ **HIGH**

When integrating with SuGaR/3DGS/Gaussian splatting (Phase II):
- Color space mismatches cause **silent rendering errors** (wrong colors, gamut clipping)
- Mixed color spaces in training data → **model confusion** (learns color space drift, not scene content)
- Cannot retroactively fix: requires **re-ingesting all training data** if wrong

**Impact:**
- Breaks Phase II integration
- Invalidates existing training datasets
- Forces complete re-ingest (expensive for large datasets)

### Recommendation: FIX BEFORE MERGE

**Effort:** 4-5 hours
**Complexity:** Medium

**Required changes:**

1. **Add `color_space` to `LinearIngestResult`:**
   ```python
   @dataclass
   class LinearIngestResult:
       # ... existing fields ...
       color_space: str  # NEW: Actual output color space
   ```

2. **Add `color_space` validation to `LinearDecoder.decode()`:**
   ```python
   def _decode_raw(self, path: Path, format_str: str) -> Tuple[np.ndarray, Tuple[int, int], str]:
       """Decode RAW, returning (array, size, color_space)."""
       with rawpy.imread(str(path)) as raw:
           # Detect if camera color matrix is available
           if not raw.camera_whitebalance or not raw.color_matrix:
               raise ColorSpaceError(
                   f"RAW file {path.name} lacks camera color matrix. "
                   f"Cannot guarantee linear_sRGB output. "
                   f"Export to 16-bit TIFF with known color space instead."
               )

           rgb = raw.postprocess(...)
           return rgb, (rgb.shape[0], rgb.shape[1]), "linear_sRGB"
   ```

3. **Update manifest schema to make `color_space` required:**
   ```python
   class DatasetMetadataV1(BaseModel):
       color_space: str = Field(..., description="Color space (must match all images)")

       @field_validator("color_space")
       @classmethod
       def validate_color_space(cls, v):
           if v not in ["linear_sRGB", "linear_ACEScg"]:  # Phase I: sRGB only
               raise ValueError(f"Unsupported color space: {v}")
           return v
   ```

4. **Add test:**
   ```python
   def test_raw_color_space_validation(tmp_path):
       """Test that RAW files without color matrix are rejected."""
       # Create corrupted RAW (no color matrix)
       # Should raise ColorSpaceError
   ```

**Why before merge:**
- Low implementation risk (additive change)
- Prevents Phase II integration blockers
- Ensures all Phase I data is usable in Phase II
- Clean contract now vs expensive migration later

**Alternative (NOT recommended):**
- Document `linear_sRGB` as "best-effort, not validated"
- Accept risk of color space drift
- **Architect rejects this:** Violates fail-fast philosophy

---

## 2. RAW Demosaic Determinism Pinning (P1 — High Priority)

### Issue

Current implementation:
- RAW decoder uses `rawpy.DemosaicAlgorithm.AHD` (Adaptive Homogeneity-Directed)
- **No validation** that rawpy/LibRaw version matches expected behavior
- **No cross-platform determinism test**

### Risk Assessment

**Risk if deferred:** ⚠️ **MEDIUM-HIGH**

LibRaw algorithm implementations can drift across versions:
- **Example:** LibRaw 0.20 → 0.21 changed DCB demosaic default parameters
- Different platforms (macOS/Linux) may use different LibRaw builds
- Result: **Same RAW file → different tensors → content hash mismatch**

**Impact:**
- Training data not reproducible across machines
- Cannot verify dataset integrity after transfer
- Difficult to debug model training issues (is it model or data drift?)

### Recommendation: FIX BEFORE MERGE

**Effort:** 2-3 hours
**Complexity:** Low

**Required changes:**

1. **Add rawpy/LibRaw version capture to provenance:**
   ```python
   import rawpy

   transform = TransformMetadata(
       gamma=gamma,
       bit_depth=bit_depth,
       dtype=dtype,
       color_space=color_space,
       demosaic_method="AHD",
       demosaic_library=f"rawpy/{rawpy.__version__}",  # NEW
       libraw_version=raw.libraw_version,  # NEW (from rawpy.RawPy object)
   )
   ```

2. **Add cross-platform determinism test:**
   ```python
   @pytest.mark.slow
   def test_raw_demosaic_determinism(tmp_path):
       """Test that RAW demosaic is deterministic across runs."""
       # Use a known RAW file (check in test fixture or download)
       raw_path = download_test_raw()  # e.g., from DNG spec samples

       # Decode twice
       result1 = decode(raw_path, gamma=1.0)
       result2 = decode(raw_path, gamma=1.0)

       # Content hashes MUST match
       assert result1.content_hash == result2.content_hash

       # Also check pixel-level equality
       np.testing.assert_array_equal(result1.linear_rgb, result2.linear_rgb)
   ```

3. **Document rawpy version pinning in requirements:**
   ```text
   # requirements/raw.txt
   rawpy>=0.18.1,<0.19.0  # Pin minor version for demosaic stability
   ```

**Why before merge:**
- Reproducibility is a Phase I **core requirement**
- Low implementation risk
- Prevents future debugging nightmares
- Easy to add now, harder to retrofit later

---

## 3. Manifest DAG Forward Compatibility (P2 — Medium Priority)

### Issue

Current manifest schema:
- Tracks individual images with provenance
- **No field for artifact lineage** (which artifact was derived from which)
- Phase II will need DAG tracking for multi-stage pipelines (ingest → depth → material → reconstruction)

### Risk Assessment

**Risk if deferred:** ⚠️ **MEDIUM**

Without lineage fields:
- Phase II will need to **extend schema to v2.0.0** (breaking change)
- Cannot retroactively add lineage to Phase I datasets
- Migration complexity increases

**With forward-compatible fields:**
- Phase I manifests work in Phase II unchanged
- Clean schema evolution path
- Optional field → no validation burden in Phase I

### Recommendation: FIX BEFORE MERGE (if effort allows)

**Effort:** 2-3 hours
**Complexity:** Low

**Required changes:**

1. **Add optional `parent_artifact_hash` to `ImageMetadataV1`:**
   ```python
   class ImageMetadataV1(BaseModel):
       # ... existing fields ...
       parent_artifact_hash: Optional[str] = Field(
           None,
           description="SHA-256 hash of parent artifact (for DAG lineage)"
       )
       pipeline_stage: Optional[str] = Field(
           None,
           description="Pipeline stage that produced this artifact (e.g., 'linear_ingest', 'depth_estimation')"
       )
   ```

2. **Update builder to accept optional lineage:**
   ```python
   @dataclass
   class ImageManifestEntry:
       # ... existing fields ...
       parent_artifact_hash: Optional[str] = None
       pipeline_stage: str = "linear_ingest"  # Default for Phase I
   ```

3. **Add test for schema forward compatibility:**
   ```python
   def test_manifest_with_dag_fields_validates():
       """Test that DAG fields are optional and validate."""
       # Create manifest with DAG fields
       # Should validate and serialize correctly
       # Phase I code should ignore these fields
   ```

**Why before merge:**
- Schema v1.0.0 is **immutable once released**
- Adding fields later requires v2.0.0 (breaking change)
- Optional fields cost nothing in Phase I
- Enables clean Phase II integration

**Acceptable to defer if:**
- Team prioritizes faster Phase I delivery
- Accept schema v2.0.0 for Phase II
- Plan explicit migration tooling

**Architect preference:** Fix now (low cost, high future value)

---

## Items Already Well-Addressed ✅

### Memory Discipline

**User concern:** Torch import spike (+183 MiB)

**Status:** ✅ **NOT APPLICABLE**

- Implementation uses **zero torch imports** ✅
- Pure NumPy/PIL/rawpy (minimal memory footprint) ✅
- No CUDA context, no GPU dependencies ✅
- Verified via grep (no torch references in codebase)

**No action needed.**

---

### Determinism Expansion Tests

**User concern:** Cross-platform determinism, RAW demosaic drift

**Status:** ⚠️ **PARTIALLY ADDRESSED**

- Content hash reproducibility test exists ✅
- **Missing:** Cross-platform test (covered in Rec #2 above)
- **Missing:** LibRaw version tracking (covered in Rec #2 above)

**Action:** See Recommendation #2 (P1 — High Priority)

---

## Items to Defer (Explicitly Out of Scope for Phase I)

### 1. ACEScg Color Space Support (Phase II)

**Status:** Documented as Phase II feature ✅

- Current: `linear_sRGB` only
- Planned: ACEScg for wider gamut (luxury materials)
- **Defer to Phase II** per roadmap

**No action needed.**

---

### 2. Multi-Exposure HDR Merge (Phase II)

**Status:** Documented as Phase II feature ✅

- Current: Single-exposure linear decode
- Planned: Bracket merge with alignment
- **Defer to Phase II** per roadmap

**No action needed.**

---

### 3. ML-Based Demosaic (Phase III)

**Status:** Documented as Phase III exploration ✅

- Current: LibRaw AHD (industry standard)
- Exploration: Neural demosaic for quality improvement
- **Defer to Phase III** per roadmap

**No action needed.**

---

## Code Quality & Style Assessment

### Strengths ✅

1. **Consistent module structure:**
   - `__init__.py` with clear public API exports
   - Exceptions in dedicated module
   - Validators separated from core logic
   - Provenance isolated from decode logic

2. **Clear naming:**
   - `LinearDecoder` (not ambiguous `RawLoader`)
   - `LinearIngestResult` (explicit contract)
   - `ProvenanceCapture` (clear responsibility)

3. **Type hints:**
   - All public APIs have type annotations ✅
   - Dataclasses used for structured data ✅
   - Pydantic models for validation ✅

4. **Error messages:**
   - Actionable remediation guidance ✅
   - Context about *why* constraint violated ✅
   - Examples of correct usage ✅

5. **Docstrings:**
   - Module-level warnings about rendering vs training ✅
   - Class-level architecture references (ADR-023, ADR-026) ✅
   - Function-level examples ✅

### Minor Style Notes (Non-blocking)

1. **Provenance capture could use builder pattern:**
   ```python
   # Current: Many parameters
   prov = capture.capture(source_path, tensor, gamma, bit_depth, dtype, color_space, ...)

   # Possible improvement (defer to refactor if needed):
   prov = (ProvenanceBuilder()
           .source(source_path)
           .tensor(tensor)
           .transform(gamma=1.0, bit_depth=32)
           .build())
   ```
   **Recommendation:** Defer to future refactor if API becomes unwieldy

2. **Test organization:**
   - Tests well-organized by class ✅
   - Could benefit from parametrized tests for format coverage
   - **Recommendation:** Consider parametrization in Phase I.1

---

## Security Assessment

### Reviewed Security Aspects ✅

1. **Path Traversal:**
   - Uses `Path` objects consistently ✅
   - No `shell=True` subprocess calls ✅
   - File I/O through safe libraries (PIL, rawpy, tifffile) ✅

2. **Unsafe Deserialization:**
   - JSON only (no pickle) ✅
   - Pydantic validation on load ✅

3. **Input Validation:**
   - Format detection by extension (safe) ✅
   - Dtype validation prevents unexpected types ✅
   - Range validation prevents NaN/Inf propagation ✅

4. **Dependency Supply Chain:**
   - Core deps: NumPy, PIL, rawpy (mature, widely audited) ✅
   - Optional deps: OpenEXR, tifffile (industry standard) ✅
   - No experimental ML dependencies ✅

### No Security Concerns Identified ✅

---

## CI/CD & Reproducibility Assessment

### Current CI Coverage

Based on diff and test structure:
- ✅ Tests run in CI (pytest)
- ✅ ADR-023 compliance test (no cross-imports)
- ✅ Schema validation tests
- ✅ Determinism test (content hash reproducibility)

### Recommendations for CI Enhancement (P3 — Defer)

1. **Add pre-commit hook for isolation boundary:**
   ```yaml
   # .pre-commit-config.yaml
   - repo: local
     hooks:
       - id: spatial-ai-isolation
         name: Verify Spatial AI isolation boundary
         entry: python scripts/verify_pipeline_isolation.py
         language: system
         pass_filenames: false
   ```

2. **Add dependency tier enforcement:**
   - Verify `requirements/raw.txt` only has approved deps
   - Prevent accidental torch/cuda additions

**Recommendation:** Defer to Phase I.1 cleanup sprint

---

## Documentation Assessment

### Completeness ✅

1. **Architecture documentation:** 507 lines ✅
2. **User guide:** 649 lines ✅
3. **Example workflow:** 230 lines ✅
4. **ADR references:** ADR-023, ADR-026 ✅
5. **Inline docstrings:** Comprehensive ✅

### Forward Compatibility Documentation

**Recommendation:** Add migration guide (P3 — Defer to Phase II)

When Phase II extends to ACEScg/multi-exposure:
- Document schema v1.0.0 → v2.0.0 migration
- Provide migration script
- Explain breaking vs non-breaking changes

**Not blocking for Phase I.**

---

## Diff Summary

**Files changed:** 17
**Lines added:** 4,416
**Lines removed:** 45

**Breakdown:**
- Core implementation: ~2,000 lines
- Tests: ~1,300 lines
- Documentation: ~1,400 lines
- Examples: ~230 lines
- Demo artifacts: ~160 lines

**Assessment:** Well-balanced implementation-to-test-to-docs ratio ✅

---

## Final Recommendations

### Pre-Merge Checklist

**P1 (High Priority) — Must Fix Before Merge:**

- [ ] **Color Space Normalization Boundary** (4-5 hours)
  - Add `color_space` to `LinearIngestResult`
  - Validate RAW color matrix availability
  - Update manifest schema
  - Add test for color space validation

- [ ] **RAW Demosaic Determinism Pinning** (2-3 hours)
  - Capture rawpy/LibRaw versions in provenance
  - Add cross-platform determinism test
  - Pin rawpy version in requirements

**P2 (Medium Priority) — Should Fix if Time Allows:**

- [ ] **Manifest DAG Forward Compatibility** (2-3 hours)
  - Add optional `parent_artifact_hash` field
  - Add optional `pipeline_stage` field
  - Add forward compatibility test

**Total additional effort:** 8-11 hours (P1+P2)

---

### Merge Decision

**✅ APPROVE WITH CONDITIONS**

**Conditions:**
1. Complete P1 items before merge (required)
2. P2 item recommended but not blocking (architect preference: include)

**Rationale:**
- Core implementation is **production-ready**
- Architecture is **sound and well-isolated**
- Test coverage is **comprehensive**
- Documentation is **excellent**
- P1 items are **low-risk additions** that prevent future blockers
- P2 item enables **clean Phase II integration**

**Risk if merged without P1 fixes:**
- Color space issues discovered in Phase II → **expensive re-ingest**
- Demosaic non-determinism → **dataset integrity problems**

**Risk if merged without P2 fix:**
- Schema v2.0.0 required for Phase II → **migration complexity**
- Still manageable, but messier than adding optional fields now

---

## Post-Merge Follow-Up Items (Phase I.1)

**P3 (Low Priority) — Defer to Phase I.1:**

1. **CI Hardening:**
   - Pre-commit hook for isolation boundary
   - Dependency tier enforcement script

2. **Performance Profiling:**
   - Baseline memory usage metrics
   - Batch processing optimization (if needed)

3. **Documentation:**
   - Schema migration guide (stub for Phase II)
   - Troubleshooting common RAW errors

4. **Test Enhancements:**
   - Parametrized tests for format coverage
   - Integration test with real RAW samples

**Estimated effort:** 8-12 hours total (can be distributed over Phase I.1)

---

## Conclusion

PR #946 represents **high-quality architectural work** that successfully delivers Phase I objectives with:

- **Strong isolation boundaries** (ADR-023 compliance)
- **Rigorous linear contract enforcement** (no compromise on quality)
- **Comprehensive provenance tracking** (audit-grade metadata)
- **Fail-fast validation** (no silent corruption)
- **Excellent documentation** (architecture + user guide + examples)
- **Robust test coverage** (73 tests, determinism verified)

**With P1 hardening items addressed** (8-11 hours additional work), this PR will provide a **solid foundation for Phase II** and ensure **long-term dataset integrity**.

**Architect recommendation:** APPROVE after P1 items completed.

---

**Review completed by:** Transformation Portal Architect
**Date:** 2025-02-15
**Authority:** Final decision per governance policy (docs/architecture/agent_governance.md)
