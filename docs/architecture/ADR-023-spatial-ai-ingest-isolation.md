# ADR-023: Spatial AI Ingest Isolation Boundary

**Status:** Approved (Mandatory)
**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** Spatial AI Foundation Roadmap, DATA_CONTRACT.md v1.0.0
**Enforcement status:** The original enforcement script exists at
`scripts/security/verify_pipeline_isolation.py`, but it is no longer a green
description of the live codebase without re-scoping; see the current-state note
below.

---

## Executive Summary

**Decision:** The `spatial_ai` ingest pipeline and `lux_depth_v3` rendering pipeline MUST maintain complete isolation in RAW decode logic. No shared decode code is permitted.

**Rationale:** Different pipelines have fundamentally incompatible requirements:
- **Rendering:** 8-bit sRGB, tone-mapped, gamma 2.2 (perceptual display)
- **Training:** 12-16 bit linear, ACEScg/linear sRGB, gamma 1.0 (physical sensor data)

**Consequence:** Mixing these concerns creates silent cross-contamination risk that destroys both rendering quality and training data fidelity.

---

## Current Codebase Verification (2026-05-12)

Running `python3 scripts/security/verify_pipeline_isolation.py` against the live
tree currently reports violations. The failures are not documentation typos:

- `lux_depth_v3` imports spatial reconstruction and segmentation contracts in
  several runtime paths.
- `src/transformation_portal/spatial_ai/ingest/raw_worker.py` imports
  `lux_depth_v3.raw_loader` for its `load_rgb` bridge.

Treat the blanket cross-import examples below as the original ADR enforcement
model, not as current green CI evidence. The still-relevant architectural
boundary is raw pixel-decode contamination: changes touching RAW decode policy
must either restore/re-scope mechanical enforcement or document an Architect
decision that supersedes the older blanket-import rule.

---

## Context

### Current State

`lux_depth_v3/raw_loader.py` decodes RAW files for rendering and
APEX-compatible ingest bridges:
- Input: CR2/NEF/ARW (RAW formats)
- Default legacy output: 8-bit gamma-encoded sRGB numpy array (H, W, 3)
- Governed APEX-compatible output: 16-bit linear RGB when callers explicitly
  request `output_linear=True` and `output_bps=16`
- Transform: Auto/camera white balance, demosaic, and caller-selected gamma /
  linear output policy

The legacy default is **correct for display-oriented compatibility** but
catastrophic if accidentally reused for training:
- 8-bit quantization destroys shadow/highlight detail
- Gamma-encoded sRGB makes light intensity non-linear (breaks physics)
- sRGB gamut can clip saturated colors (luxury materials exceed sRGB)

### Spatial AI Requirements (DATA_CONTRACT.md v1.0.0)

`spatial_ai` requires:
- Input: Same RAW formats
- Output: 16-bit linear ACEScg (or linear sRGB Phase I MVP)
- Transform: Deterministic demosaic, linear gamma 1.0, wide gamut preservation

These are **mutually exclusive** with rendering requirements.

### Integration Risk Scenario

**Without this ADR:**

Developer thinks: "Both pipelines need RAW decode, let's share code!"

```python
# WRONG: Shared decode with mode parameter
def load_raw(path: Path, mode: str = "rendering") -> np.ndarray:
    if mode == "rendering":
        return decode_8bit_srgb(path)
    elif mode == "training":
        return decode_linear_acescg(path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

**Failure modes:**
1. Default parameter wrong → silent pipeline contamination
2. Mode flag forgotten → training gets sRGB, rendering gets linear
3. Shared LibRaw configuration → incompatible settings leak between modes
4. Future refactoring breaks one pipeline while "improving" the other

**This ADR prevents this entire class of errors.**

---

## Decision

### 1. Complete Isolation Mandate

**DECISION:** Zero shared decode logic between pipelines.

**Implementation:**

```
src/transformation_portal/
  lux_depth_v3/
    raw_loader.py          # Rendering-only (UNCHANGED, frozen)
  spatial_ai/
    ingest/
      linear_decoder.py    # Training-only (NEW, isolated)
```

**Allowed:**
- Shared metadata parsing only:
  ```python
  # utils/raw_metadata.py (NEW, metadata extraction only)
  def extract_raw_exif(path: Path) -> dict:
      """Extract EXIF without decoding pixel data."""
  ```

**Forbidden:**
- Shared pixel decode logic
- Shared LibRaw configuration
- Shared white balance / demosaic / color transform utilities

### 2. Enforcement Strategy

**Original CI Lint Rule Model:**

```python
# scripts/security/verify_pipeline_isolation.py

def test_no_spatial_imports_in_lux_depth():
    """lux_depth_v3 must not import spatial_ai modules."""
    lux_files = glob("src/transformation_portal/lux_depth_v3/**/*.py")
    for filepath in lux_files:
        with open(filepath) as f:
            content = f.read()
            assert "from transformation_portal.spatial_ai" not in content
            assert "import transformation_portal.spatial_ai" not in content

def test_no_lux_depth_imports_in_spatial_ingest():
    """spatial_ai.ingest must not import lux_depth_v3 decode logic."""
    spatial_files = glob("src/transformation_portal/spatial_ai/ingest/**/*.py")
    for filepath in spatial_files:
        with open(filepath) as f:
            content = f.read()
            assert "from transformation_portal.lux_depth_v3.raw_loader" not in content
```

**Original CI integration model:**
```yaml
# .github/workflows/spatial-contract.yml
- name: Verify Pipeline Isolation
  run: python scripts/security/verify_pipeline_isolation.py
```

### 3. Documentation Requirements

**Every RAW decode file MUST include header:**

```python
# lux_depth_v3/raw_loader.py
"""RAW decode for rendering pipeline ONLY.

WARNING: This decoder outputs 8-bit sRGB for perceptual display.
DO NOT use for training data (destroys linear light relationships).

For training ingest, use: spatial_ai.ingest.linear_decoder
"""
```

```python
# spatial_ai/ingest/linear_decoder.py
"""RAW decode for training pipeline ONLY.

WARNING: This decoder outputs linear ACEScg for physics-based learning.
DO NOT use for rendering (no tone mapping, will look washed out).

For rendering, use: lux_depth_v3.raw_loader
"""
```

### 4. Exception Process

**If shared code is proposed in the future:**

1. MUST escalate to Architect (per governance policy)
2. MUST provide ADR explaining:
   - Why isolation is insufficient
   - How cross-contamination will be prevented
   - What enforcement prevents silent mode confusion
3. MUST include migration plan for existing code
4. MUST update CI enforcement to verify new boundary

**Default answer:** Duplication is cheaper than contamination.

---

## Consequences

### Positive

✅ **Zero risk of silent cross-contamination**
- Rendering never accidentally gets linear data
- Training never accidentally gets tone-mapped data

✅ **Independent evolution**
- Rendering can optimize for perceptual quality
- Training can optimize for physical accuracy
- No coordination tax between teams

✅ **Clear ownership**
- `lux_depth_v3` owns rendering quality
- `spatial_ai` owns training fidelity
- No shared failure modes

✅ **Enforcement is mechanical**
- CI fails immediately if isolation violated
- No human vigilance required

### Negative

⚠️ **Code duplication**
- LibRaw initialization duplicated
- EXIF parsing duplicated (mitigated by `utils/raw_metadata.py`)
- Approximately 100-200 lines of duplicated boilerplate

**Architect Assessment:** This cost is negligible compared to cross-contamination risk.

### Neutral

- More files to maintain (marginal cost)
- Clearer separation makes debugging easier (net positive)

---

## Alternatives Considered

### Alternative 1: Shared Decode with Mode Parameter

```python
def load_raw(path: Path, output_mode: Literal["rendering", "training"]) -> np.ndarray:
    # ... shared logic with conditionals
```

**Rejected:**
- High risk of mode confusion
- Shared configuration state (LibRaw settings)
- Future refactoring breaks both pipelines
- No mechanical enforcement of correctness

### Alternative 2: Shared Base Class with Overrides

```python
class RawDecoder(ABC):
    def decode(self, path: Path) -> np.ndarray:
        ...

class RenderingRawDecoder(RawDecoder):
    # 8-bit sRGB implementation

class TrainingRawDecoder(RawDecoder):
    # Linear ACEScg implementation
```

**Rejected:**
- Creates coupling through shared interface
- Shared interface implies shared assumptions (dangerous)
- Overrides can silently break Liskov substitution
- More complex than simple isolation

### Alternative 3: Plugin Architecture with Registry

```python
RawDecoderRegistry.register("rendering", RenderingDecoder)
RawDecoderRegistry.register("training", TrainingDecoder)
decoder = RawDecoderRegistry.get(mode)
```

**Rejected:**
- Over-engineered for two pipelines
- Registry lookup adds indirection (harder to trace)
- Still requires mode parameter (same confusion risk)
- No enforcement that correct decoder is used

**Decision:** Complete isolation is simplest, safest, and most enforceable.

---

## Migration Plan

### Phase 1: Immediate (Before Spatial AI Milestone 0)

1. ✅ Approve this ADR
2. ✅ Create `scripts/security/verify_pipeline_isolation.py`
3. ⚠️ Re-scope or replace the isolation check before treating it as a current
   green CI gate
4. ✅ Document headers in `lux_depth_v3/raw_loader.py`

### Phase 2: Spatial AI Implementation (Milestone 2)

1. Create `spatial_ai/ingest/linear_decoder.py` (no imports from `lux_depth_v3`)
2. Create or identify the approved metadata-only sharing boundary
3. Re-establish mechanical enforcement for the current boundary

### Phase 3: Audit (After Phase I Complete)

1. Review both decoders for correctness
2. Verify no accidental coupling
3. Document any shared utilities in ADR amendment

---

## Success Criteria

This ADR is successful if:

1. ⚠️ Mechanical enforcement is re-scoped to the current codebase and fails on
   unsafe raw pixel-decode contamination
2. ⚠️ Any approved cross-package contract sharing is explicit and covered by
   tests or a superseding ADR
3. ✅ Both pipelines can evolve independently without coordination
4. ✅ Zero reports of cross-contamination (rendering gets linear, training gets sRGB)

---

## References

- Spatial AI Foundation Roadmap: `docs/spatial_ai/ROADMAP.md`
- Data Contract: `docs/spatial_ai/DATA_CONTRACT.md` v1.0.0
- Governance Policy: `docs/architecture/agent_governance.md`
- Architectural Review: `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md`

---

**Approval:** Transformation Portal Architect
**Enforcement:** Mandatory CI gate (non-bypassable)
**Review Date:** 2027-02-11 (12 months)
