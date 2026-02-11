# ADR-027: Phase 2 Spatial AI Extension Architecture

**Status:** Proposed
**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-023 (Isolation), ADR-026 (APEX Research Ultra), Phase 1 (PR #906), Phase 1.1 (PR #907)
**Enforcement:** CI gates (isolation, HF revisions), contract validation

---

## Executive Summary

**Decision:** Extend the Spatial AI Foundation (Phase 1) with three advanced capabilities:

1. **SAM2 Integration** — Temporal-consistent segmentation for material/object boundaries
2. **MaterialGAN Integration** — Physics-based BRDF estimation and PBR texture generation
3. **3D Gaussian Splatting** — Depth-guided scene reconstruction for geometric verification

**Design Principle:** Respect Phase 1.1 constraints. Design for rigor, not against it.

**Architectural Position:** Phase 2 components are **first-class members** of `spatial_ai` namespace, not add-ons. They extend the linear ingest foundation with perception, material understanding, and geometric reasoning.

---

## Context

### Phase 1 Foundation (PR #906)

Phase 1 delivered:
- **Linear ingest pipeline:** `spatial_ai/ingest/linear_decoder.py` (gamma=1.0 enforcement)
- **Depth ensemble backend:** Depth Pro + DA3 1.1 adaptive fusion
- **Provenance capture:** Deterministic decode recipes with content hashing

### Phase 1.1 Hardening (PR #907)

Phase 1.1 introduced three enforceable constraints:

1. **I/O Contract Discipline**
   - `gamma=1.0` always enforced (no override parameter)
   - `strict_ingest` flag for 8-bit rejection (lane-based policy)
   - EXR export fail-loud (RuntimeError if OpenEXR missing)

2. **Governance Gates**
   - AST-based isolation checker (ADR-023 enforcement)
   - CI blocks cross-pipeline imports (`lux_depth_v3` ↔ `spatial_ai`)

3. **Reproducibility Gates**
   - HuggingFace revision pinning (commit SHAs, not `main`)
   - Placeholders allowed only in `experimental/` presets
   - CI validates compliance before merge

### Research Gap Analysis

Phase 1 provides **spatial foundation** (linear RGB, metric depth) but lacks:
- **Material understanding:** No semantic segmentation or BRDF estimation
- **Temporal consistency:** Single-frame depth, no video tracking
- **Geometric verification:** Depth maps not validated against 3D reconstruction

**Phase 2 closes these gaps** while respecting Phase 1.1 constraints.

---

## Decision

### 1. Namespace Structure

**DECISION: Extend `spatial_ai/` with four new modules**

```
src/transformation_portal/
├── spatial_ai/                    # Spatial AI namespace (Phase 1 + 2)
│   ├── ingest/                    # Phase 1 (COMPLETE)
│   │   └── linear_decoder.py
│   │
│   ├── segmentation/              # Phase 2.1 (NEW)
│   │   ├── sam2_backend.py        # SAM2 model wrapper
│   │   ├── mask_processor.py      # Temporal tracking, refinement
│   │   ├── material_classifier.py # Optional CLIP integration
│   │   └── contracts.py           # Data contracts
│   │
│   ├── materials/                 # Phase 2.2 (NEW)
│   │   ├── material_gan.py        # NVDIFFREC wrapper
│   │   ├── pbr_generator.py       # PBR texture generation
│   │   ├── brdf_estimator.py      # BRDF parameter inference
│   │   └── contracts.py           # Data contracts
│   │
│   ├── reconstruction/            # Phase 2.3 (NEW)
│   │   ├── gaussian_splat.py      # 3DGS scene reconstruction
│   │   ├── depth_guidance.py      # Depth-guided optimization
│   │   ├── mesh_export.py         # PLY, OBJ export
│   │   └── contracts.py           # Data contracts
│   │
│   └── orchestration/             # Phase 2.4 (NEW)
│       ├── spatial_pipeline.py    # End-to-end workflow
│       └── validation.py          # Cross-component validation
│
├── core/                          # Shared utilities (OK to import)
│   └── geometry/                  # NEW: Camera, transforms, depth utils
│       ├── camera.py
│       ├── transforms.py
│       └── depth_utils.py
│
└── lux_depth_v3/                  # Rendering pipeline (ISOLATED)
```

**Rationale:**
- Clear module boundaries (one concern per module)
- Isolation from `lux_depth_v3` (ADR-023 compliance)
- Shared geometry utilities in `core/` (reusable across pipelines)

---

### 2. Import Policy

**DECISION: Enforce strict import boundaries**

#### Allowed Imports

```python
# ✅ ALLOWED: Phase 2 → Phase 1 (same pipeline)
from transformation_portal.spatial_ai.ingest import LinearDecoder

# ✅ ALLOWED: Phase 2 → core utilities
from transformation_portal.core.geometry.camera import CameraIntrinsics
from transformation_portal.core.geometry.depth_utils import normalize_depth

# ✅ ALLOWED: Phase 2 internal (within spatial_ai)
from transformation_portal.spatial_ai.segmentation import SAM2Backend
from transformation_portal.spatial_ai.materials import MaterialEstimator
```

#### Forbidden Imports

```python
# ❌ FORBIDDEN: Phase 2 → lux_depth_v3 (ADR-023 violation)
from transformation_portal.lux_depth_v3.utils import anything

# ❌ FORBIDDEN: lux_depth_v3 → spatial_ai (ADR-023 violation)
from transformation_portal.spatial_ai.ingest import anything
```

**Enforcement:**
- CI runs `scripts/security/verify_pipeline_isolation.py` (AST-based)
- Violations block merge
- No workarounds (e.g., `importlib.import_module`) allowed

---

### 3. Contract Design

**DECISION: Explicit input/output contracts for each component**

#### SAM2 Segmentation Contract

```python
@dataclass
class SegmentationInput:
    """Input contract for SAM2 segmentation."""
    image: np.ndarray              # (H, W, 3) float32 linear RGB [0, ∞)
    gamma: float = 1.0             # MUST be 1.0 (enforced)
    mode: Literal["auto", "points", "bbox", "video"]
    prompts: Optional[List[Dict]] = None
    prev_masks: Optional[np.ndarray] = None
    frame_idx: Optional[int] = None

@dataclass
class SegmentationResult:
    """Output contract from SAM2 segmentation."""
    masks: np.ndarray              # (N, H, W) bool
    scores: np.ndarray             # (N,) float32 [0, 1]
    metadata: List[MaskMetadata]
    temporal_ids: Optional[np.ndarray] = None  # Video tracking IDs
```

#### MaterialGAN Contract

```python
@dataclass
class MaterialInput:
    """Input contract for material estimation."""
    rgb: np.ndarray                # (H, W, 3) float32 linear RGB
    depth: np.ndarray              # (H, W) float32 metric depth
    normals: Optional[np.ndarray] = None
    camera: Optional[CameraIntrinsics] = None
    gamma: float = 1.0             # MUST be 1.0
    masks: Optional[np.ndarray] = None  # From SAM2

@dataclass
class MaterialResult:
    """Output contract from material estimation."""
    pbr_textures: PBRTextures      # Albedo, normal, roughness, metallic, AO
    material_params: Dict[str, Any]  # BRDF parameters
    quality_metrics: Dict[str, float]
```

#### 3DGS Reconstruction Contract

```python
@dataclass
class ReconstructionInput:
    """Input contract for 3D reconstruction."""
    rgb_images: List[np.ndarray]   # [(H, W, 3)] float32 linear RGB
    depth_maps: List[np.ndarray]   # [(H, W)] float32 metric depth
    cameras: List[CameraIntrinsics]
    masks: Optional[List[np.ndarray]] = None
    gamma: float = 1.0             # MUST be 1.0

@dataclass
class ReconstructionResult:
    """Output contract from 3D reconstruction."""
    gaussians: GaussianCloud       # 3D primitives
    mesh: Optional[Mesh] = None    # Extracted mesh
    quality_metrics: Dict[str, float]
    render_fn: Callable            # Novel view synthesis
```

**Contract Enforcement:**
- `__post_init__` validation (gamma=1.0, dtype checks)
- Explicit errors (ValueError, not silent corruption)
- Documented preconditions and postconditions

---

### 4. Phase 1.1 Constraint Compliance

**DECISION: Make Phase 2 compatible with Phase 1.1 rigor from day one**

#### OpenEXR Preflight Checks

**Pattern:**
```python
def check_exr_support():
    """Verify OpenEXR/Imath available before emit_exr=True."""
    try:
        import OpenEXR
        import Imath
        return True
    except ImportError:
        return False

# In orchestration
if config.emit_exr and not check_exr_support():
    raise RuntimeError(
        "EXR output requested but OpenEXR/Imath not installed.\n"
        "Install with: pip install OpenEXR Imath\n"
        "Or set emit_exr=False in preset."
    )
```

**Benefit:** Fail-fast with clear message, not after 20 seconds of processing.

#### Lane-Based Strictness

**Development Lane:**
```yaml
# config/presets/experimental/spatial_ai_dev.yaml
ingest:
  strict_ingest: false  # Allow 8-bit for quick iteration
  emit_exr: false       # Skip EXR overhead
```

**Research Lane:**
```yaml
# config/presets/experimental/spatial_ai_research.yaml
ingest:
  strict_ingest: true   # Reject 8-bit (enforce precision)
  emit_exr: true        # Require EXR (HDR preservation)
```

**Benefit:** Velocity preserved in dev, rigor enforced in research.

#### Experimental Preset Policy

**Workflow:**
1. Create preset in `config/presets/experimental/` with HF placeholder
2. Develop and test locally
3. Verify HuggingFace commit hash manually
4. Update preset with pinned revision
5. Promote to `config/presets/` (stable)
6. CI validates on merge

**Example:**
```yaml
# config/presets/experimental/sam2_video.yaml (development)
segmentation:
  model: sam2_large
  revision: "NEEDS_VERIFICATION_0000000000000000000000"  # OK in experimental

# config/presets/sam2_video.yaml (promoted)
segmentation:
  model: sam2_large
  # Verified 2026-02-15 from https://huggingface.co/facebook/sam2-hiera-large/commits/main
  revision: "a1b2c3d4e5f6789012345678901234567890abcd"
```

---

### 5. Model Selection & Licensing

**DECISION: Choose models with compatible licenses and clear tier restrictions**

#### SAM2 (Segmentation)

| Model | License | Tier Restriction |
|-------|---------|------------------|
| SAM2 Base | Apache 2.0 | None (commercial OK) |
| SAM2 Large | Apache 2.0 | None (commercial OK) |

**Decision:** Use SAM2 Large for quality, SAM2 Base for speed. No tier restrictions.

#### MaterialGAN (PBR Textures)

| Model | License | Tier Restriction |
|-------|---------|------------------|
| NVDIFFREC | BSD-3-Clause | None (commercial OK) |
| MaterialGAN | CC BY-NC 4.0 | Research only |

**Decision:** Use NVDIFFREC (better license). Add MaterialGAN later if quality gap justifies tier restriction.

#### 3D Gaussian Splatting

| Model | License | Tier Restriction |
|-------|---------|------------------|
| 3DGS (Inria) | Inria (research) | `tier: apex_research` or higher |
| NeuS2 | MIT | None (commercial OK) |

**Decision:** Use 3DGS for research tier. Offer NeuS2 as commercial alternative.

**Enforcement:**
```python
# In orchestration
if config.reconstruction.method == "3dgs" and config.tier not in ["apex_research", "apex_research_ultra"]:
    raise ValueError(
        "3DGS is research-licensed (Inria). "
        "Use tier: apex_research or higher, or switch to NeuS2 (commercial OK)."
    )
```

---

### 6. Testing Strategy

**DECISION: Comprehensive test coverage with isolation compliance checks**

#### Test Pyramid

```
E2E Integration (3 tests)
├── Full spatial pipeline
├── Multi-view reconstruction
└── Cross-component validation

Component Integration (12 tests)
├── SAM2 + LinearDecoder
├── MaterialGAN + Depth
└── 3DGS + Depth

Unit Tests (40+ tests)
├── Model loading
├── Contracts
└── Post-processing
```

#### Required Test Types

| Component | Unit Tests | Coverage Target |
|-----------|------------|-----------------|
| SAM2 Backend | 10 | ≥90% |
| MaterialGAN | 12 | ≥85% |
| 3DGS | 10 | ≥85% |
| Orchestration | 8 | ≥90% |

#### Isolation Compliance Testing

**Pre-commit Check:**
```bash
# Run before every commit
python scripts/security/verify_pipeline_isolation.py
# Expected: PASS (0 violations)
```

**CI Check:**
```yaml
# .github/workflows/enforcement.yml
- name: Verify Pipeline Isolation
  run: python scripts/security/verify_pipeline_isolation.py
```

#### HF Revision Validation Testing

**CI Check:**
```bash
# Run in CI before merge
python scripts/validation/validate_hf_revisions.py
# Expected: PASS (all stable presets have commit SHAs)
```

---

## Consequences

### Positive

✅ **Clear architectural boundaries**
- SAM2, MaterialGAN, 3DGS are separate modules (low coupling)
- No circular dependencies
- Isolation enforced by CI

✅ **Phase 1.1 constraint compliance**
- OpenEXR preflight prevents late failures
- Lane-based strictness (dev fast, research strict)
- Experimental presets enable rapid iteration

✅ **Reproducibility by design**
- HF revision pinning enforced (stable presets)
- Contract validation prevents silent errors
- Provenance capture continues from Phase 1

✅ **License clarity**
- Tier restrictions enforced mechanically
- Clear commercial vs research boundaries
- Documented alternatives (e.g., NeuS2 vs 3DGS)

✅ **Testing rigor**
- Comprehensive unit + integration + E2E tests
- Isolation compliance automated (CI)
- HF revision validation automated (CI)

### Negative

⚠️ **Code duplication**
- Cannot import from `lux_depth_v3` (ADR-023)
- Small helpers duplicated (normalize_depth, etc.)
- ~100-200 lines duplicated vs shared

**Architect Assessment:** Acceptable. Duplication is cheaper than coupling.

⚠️ **Dependency complexity**
- Phase 2 adds 6+ new dependencies (SAM2, NVDIFFREC, 3DGS, etc.)
- Version conflicts possible
- Install complexity increases

**Mitigation:** Tiered dependencies (`pip install .[spatial-ai]`), fallback modes, clear docs.

⚠️ **GPU memory pressure**
- Multi-model workflows (SAM2 + MaterialGAN + 3DGS) may exceed VRAM
- OOM crashes possible

**Mitigation:** Explicit model unloading, batch size limits, CPU fallback, VRAM requirements documented.

### Neutral

- More files to maintain (marginal cost)
- Clearer separation makes debugging easier (net positive)
- Preset promotion workflow adds process (one-time cost per model)

---

## Alternatives Considered

### Alternative 1: Integrate into lux_depth_v3

**Rejected:**
- Violates ADR-023 (pipeline isolation)
- Mixes rendering and training concerns (different gamma, gamut, bit depth)
- High risk of cross-contamination (silent corruption)

### Alternative 2: Create Separate Repository

**Rejected:**
- Fragments codebase (coordination overhead)
- Loses shared infrastructure (CI, testing, docs)
- Complicates cross-pipeline workflows (ingest + depth + segmentation)

### Alternative 3: Monolithic spatial_ai Module

**Rejected:**
- Poor modularity (SAM2, MaterialGAN, 3DGS conflated)
- Harder to test in isolation
- Difficult to maintain over time

**Decision:** Module-per-capability (chosen approach) balances modularity and cohesion.

---

## Migration Plan

### Phase 1 → Phase 2 Compatibility

**Backward Compatibility Guarantee:**
- Phase 1 code unchanged (`spatial_ai/ingest/` untouched)
- Phase 1 tests continue passing
- No breaking changes to existing APIs

**Phase 2 Additions:**
- New modules under `spatial_ai/` (additive, not destructive)
- New presets in `experimental/` (opt-in)
- New CLI commands (`transformation_portal spatial-ai ...`)

**Rollout Strategy:**
1. Phase 2.1 (SAM2) — Standalone, independent of Phase 1
2. Phase 2.2 (MaterialGAN) — Integrates with Phase 1 depth, Phase 2.1 segmentation
3. Phase 2.3 (3DGS) — Integrates with Phase 1 depth
4. Phase 2.4 (Orchestration) — Unifies all stages

**No forced migration:** Phase 1 users unaffected unless they opt-in to Phase 2 features.

---

## Success Criteria

Phase 2 is successful when:

### Technical Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | ≥85% | `pytest --cov` |
| CI green | 100% | All checks passing |
| Isolation compliance | 100% | AST checker passes |
| HF revision compliance | 100% (stable) | Validation script passes |
| Performance regression | <5% vs Phase 1 | Benchmark suite |

### Functional Criteria

| Capability | Success Criteria |
|------------|------------------|
| SAM2 Segmentation | Segments 512x512 in <3s on GPU |
| MaterialGAN PBR | Generates 1024x1024 PBR in <10s on GPU |
| 3DGS Reconstruction | 3-view scene in <30s, RMSE <2% |
| E2E Pipeline | Full workflow in <60s (512x512, GPU) |

### Governance Criteria

| Policy | Compliance Check |
|--------|------------------|
| ADR-023 Isolation | Zero forbidden imports |
| HF Revision Pinning | All stable presets have SHAs |
| Experimental Presets | Placeholders only in experimental/ |
| Security Posture | No path traversal, no unsafe pickle |

---

## References

- **Implementation Plan:** `docs/architecture/PHASE2_IMPLEMENTATION_PLAN.md`
- **Quick Reference:** `docs/architecture/PHASE2_QUICKREF.md`
- **ADR-023:** Pipeline isolation requirements
- **ADR-026:** APEX Research Ultra architecture
- **Phase 1 PR:** #906 (Spatial AI Foundation)
- **Phase 1.1 PR:** #907 (Contract integrity hardening)
- **Compatibility Checklist:** Session file `phase2_compatibility_checklist.md`

---

## Approval

**Status:** Proposed
**Review Date:** 2026-02-11
**Approver:** Transformation Portal Architect (required)
**Implementation Start:** Upon approval
**Review Interval:** 12 months (2027-02-11)

---

**Amendments:**
- None (initial version)

---

*This ADR is binding. Deviations require explicit superseding ADR with migration plan.*
