# Spatial AI Foundation Roadmap — Architectural Review

**Reviewer:** Transformation Portal Architect
**Date:** 2026-02-11
**Authority:** Final architectural decision per `docs/architecture/agent_governance.md`
**Review Target:** `docs/spatial_ai/ROADMAP.md` implementation translation
**Status:** APPROVED with MANDATORY modifications

**Current codebase note (2026-05-12):** This remains a point-in-time
architectural review. Phase I implementation now exists under
`src/transformation_portal/spatial_ai/`; tests live under `tests/spatial_ai/`
plus root and `tests/evals/` APEX files, not a `tests/apex/` directory.
Where this review says "to be created", check the current ADRs and code first.

---

## Executive Summary

**Overall Verdict: APPROVED with CONDITIONS**

The Spatial AI Foundation roadmap demonstrates strong repository-grounded discipline and correctly applies existing architectural patterns (protocol-based backends, APEX testing philosophy, contract enforcement). The Phase I scope (linear ingest + contract validation + catalog) is architecturally sound and represents a feasible foundational wedge.

**However, this approval is CONDITIONAL on addressing the Critical Blockers and Mandatory Modifications identified below before any Phase I implementation begins.**

### Key Strengths

1. ✅ **Correct pattern reuse**: Protocol + backend + stub + registry matches `DepthBackend` (ADR-019)
2. ✅ **Governance alignment**: Hard contract failures, provenance gates, banned dependency enforcement
3. ✅ **CI testing discipline**: Synthetic PR lane, real nightly checks (mirrors APEX)
4. ✅ **Dependency tiering**: Optional `pyproject.toml` `spatial-ai` extra;
   no checked-in `requirements/spatial-ai.in` lock lane exists today
5. ✅ **C2PA reality check**: Correctly identifies EXR isn't C2PA-signable, proposes pragmatic hybrid

### Critical Issues Requiring Immediate Resolution

1. 🚨 **CONTRACT CONFLICT**: ACEScg vs EXR color space (existing DATA_CONTRACT.md frozen ACEScg, roadmap assumes EXR)
2. 🚨 **INTEGRATION COUPLING**: Missing explicit isolation boundary for `lux_depth_v3` RAW ingest path
3. 🚨 **SECURITY POSTURE**: Insufficient untrusted input handling for spatial catalog queries
4. 🚨 **DEPENDENCY GOVERNANCE**: Apache Iceberg supply chain risk not assessed

---

## 1. Architectural Soundness Assessment

### 1.1 Module Structure and Placement

**Decision: APPROVED**

The proposed `src/transformation_portal/spatial_ai/` structure correctly parallels `lux_depth_v3/`:

```
src/transformation_portal/
  lux_depth_v3/          # Existing luxury rendering pipeline
  spatial_ai/            # NEW: Spatial AI Foundation (parallel, not nested)
    contracts/           # ✅ Correct: contract-first design
    protocols/           # ✅ Correct: backend protocol definitions
    catalog/             # ✅ Correct: spatiotemporal ledger
    ingest/              # ✅ Correct: linear decode pipeline
    provenance/          # ✅ Correct: sidecar + optional C2PA
```

**Rationale:**
- Preserves pipeline isolation (no cross-imports initially)
- Allows independent evolution and versioning
- Follows existing precedent (`depth/`, `lux_depth_v3/`, `segmentation/` as sibling packages)

**Rejected Alternative:**
- Nesting under `lux_depth_v3/spatial_ai/` would create false dependency and complicate future decoupling

### 1.2 Protocol-Based Backend Pattern Application

**Decision: APPROVED**

The roadmap correctly proposes following ADR-019's `DepthBackend` pattern for all spatial backends:

| Backend Type | Roadmap Proposes | Pattern Match | Verdict |
|--------------|------------------|---------------|---------|
| Provenance | `ProvenanceBackend` protocol + Stub/HashSidecar/C2PA | ✅ Matches `DepthBackend` | APPROVED |
| Catalog | `CatalogBackend` protocol + Stub/SQLite/Iceberg | ✅ Matches `DepthBackend` | APPROVED |
| Pose | `PoseBackend` protocol + Stub/COLMAP/MASt3R | ✅ Matches `DepthBackend` | APPROVED |
| Reconstruction | `ReconstructionBackend` protocol + Stub/Gsplat/Nerfstudio | ✅ Matches `DepthBackend` | APPROVED |

**Enforcement Pattern (from ADR-019):**
```python
class SpatialBackend(Protocol):
    name: str
    license_type: LicenseType
    requires_checkpoint: bool

    def compute(...) -> Result: ...
    def get_cache_key(...) -> str: ...
    def ensure_available() -> None: ...

    @classmethod
    def required_packages(cls) -> list[str]: ...
```

This is the **canonical pattern** for all spatial backends. No deviations permitted.

### 1.3 Directory Layout and Artifact Placement

**Decision: APPROVED with MODIFICATION**

**Approved:**
```
config/spatial_ai/           # YAML presets (like config/lux_depth_v3/)
docs/spatial_ai/             # ADRs + implementation reports
tests/spatial_ai/            # Unit + integration tests
scripts/spatial_ai/          # Runners (local + CI)
```

**MANDATORY MODIFICATION:**
```diff
- tools/spatial_ai/          # validators, audits, ledger tools
+ src/transformation_portal/spatial_ai/validation/  # validators as importable modules
+ tools/spatial_ai_audit.py  # CLI wrapper (like tools/performance_ledger.py)
```

**Rationale:**
- Validators should be importable library code (for CI + programmatic use)
- `tools/` is for standalone CLI utilities, not reusable logic
- Matches existing `lux_depth_v3/validation/` precedent (if created)

---

## 2. Contract & Governance Alignment

### 2.1 CRITICAL CONFLICT: Color Space Specification

**Status: BLOCKER — Must be resolved before Phase I begins**

**Conflict Identified:**

| Document | Color Space | Status |
|----------|-------------|--------|
| `docs/spatial_ai/DATA_CONTRACT.md` v1.0.0 | **ACEScg (AP1 primaries)** | Frozen, canonical |
| Roadmap Milestone 0 | **EXR (unspecified color space)** | Assumed, not explicit |

**Problem:**
- DATA_CONTRACT.md (already committed, frozen v1.0.0) mandates ACEScg linear as canonical training space
- Roadmap proposes EXR as "compute intermediate" but doesn't specify color space
- EXR supports multiple color spaces (linear sRGB, ACEScg, raw sensor RGB, etc.)
- **Silent mismatch = training on wrong color space = model collapse**

**MANDATORY RESOLUTION:**

1. **Update roadmap Milestone 0** to explicitly state:
   ```
   encoding.color_space: enum[acescg, linear_srgb, raw_sensor]  # REQUIRED
   encoding.container: enum[tiff, dng, exr]
   ```

2. **Contract validator MUST fail** if:
   - EXR lacks embedded color space metadata, OR
   - Color space is not ACEScg (per DATA_CONTRACT.md v1.0.0)

3. **Linear decoder MUST output**:
   - RAW → ACEScg linear (via LibRaw + OCIO/colour-science transforms)
   - EXR with embedded `chromaticities` attribute matching ACEScg AP1 primaries

**Alternative (requires ADR):**
- If ACEScg is too restrictive, supersede DATA_CONTRACT.md v1.0.0 with an ADR allowing "linear sRGB for Phase I MVP, ACEScg for production"
- This would require migration plan and provenance versioning

**Architect Decision:**
- I **strongly recommend** keeping ACEScg frozen per DATA_CONTRACT.md
- RAW workflows already need colour-science/OCIO for proper debayer
- Luxury real estate materials (gold, polished marble, glass) exceed sRGB gamut

### 2.2 SpatialCaptureV1 Contract Schema

**Decision: APPROVED with ADDITIONS**

Roadmap proposes:
```python
@dataclass
class SpatialCaptureV1:
    asset_id: UUID
    encoding: SpatialEncodingSpec
    sensor: SensorMetadata
    provenance: SpatialProvenanceSpec
    license: LicensePolicy
    policy: GovernancePolicy
```

**MANDATORY ADDITIONS:**

1. **Schema version field** (for migration safety):
   ```python
   schema_version: str = "1.0.0"  # DEPTH_ARTIFACT_SCHEMA_VERSION pattern
   ```

2. **Explicit color space** (resolves blocker above):
   ```python
   @dataclass
   class SpatialEncodingSpec:
       container: Literal["tiff", "dng", "exr"]
       dtype: Literal["uint16", "float16", "float32"]
       is_linear: bool  # MUST be True
       color_space: Literal["acescg"]  # Frozen per DATA_CONTRACT.md v1.0.0
       white_level: Optional[int]
       black_level: Optional[int]
   ```

3. **Hash chain integrity** (for provenance validation):
   ```python
   @dataclass
   class SpatialProvenanceSpec:
       source_sha256: str  # Hash of original RAW/TIFF bytes
       derived_sha256: str  # Hash of canonical EXR/TIFF output
       decode_recipe_hash: str  # Deterministic hash of decode config
       sidecar_json: str  # Path to JSON sidecar
       c2pa_manifest: Optional[str]  # Path to C2PA manifest (if available)
       actions: list[ProvenanceAction]  # Created, decoded, redacted, etc.
   ```

4. **Temporal indexing fields** (Milestone 1 catalog requirement):
   ```python
   capture_time: datetime  # ISO 8601 UTC
   project_id: str  # Maps to projects/<project_name>
   visit_id: str  # Longitudinal: multi-visit per property
   sequence_index: Optional[int]  # Frame order within visit
   spatial_index: Optional[str]  # H3 cell ID or Morton code
   ```

**Pattern Match:**
- Mirrors `DepthArtifact` (see `lux_depth_v3/contracts/depth_artifact.py`)
- Includes `DepthProvenance`, `LicenseTier`, `CameraIntrinsics` equivalents

### 2.3 APEX Contract Culture Alignment

**Decision: APPROVED**

Roadmap correctly applies APEX discipline:

| APEX Pattern | Spatial AI Application | Verdict |
|--------------|------------------------|---------|
| Contract-first design | `SpatialCaptureV1` schema precedes ingest code | ✅ APPROVED |
| Hard failure on schema drift | CI gate fails on uint8/non-linear/missing fields | ✅ APPROVED |
| Synthetic fixtures | Good/bad fixture matrix in `tests/spatial_ai/` | ✅ APPROVED |
| Provenance as binding | Hash chain + sidecar required, not optional | ✅ APPROVED |

**Example CI Gate (from roadmap):**
```yaml
# .github/workflows/spatial_contract.yml
- name: Validate Spatial Contracts
  run: |
    python -m transformation_portal.spatial_ai validate-contract \
      --paths data/ input_images/ projects/ tests/fixtures/ \
      --strict-ingest \
      --fail-on-schema-drift
```

This is **exactly correct** and matches existing `quality-gate.yml` enforcement posture.

---

## 3. CI/Testing Strategy Assessment

### 3.1 Synthetic PR Gating Philosophy

**Decision: APPROVED**

Roadmap correctly proposes:

**PR Lane (quality-gate.yml / spatial-contract.yml):**
- ✅ Contract validation (schema checks, dtype/range invariants)
- ✅ Shape tests (model instantiation, forward pass dimensions)
- ✅ Synthetic ingest fixtures (tiny RAW/TIFF → linear tensor round trips)
- ✅ Subprocess-based end-to-end (no GPU, stub backends only)

**Nightly Lane (nightly.yml / spatial-nightly.yml):**
- ✅ Benchmark markers (`@pytest.mark.benchmark`)
- ✅ Optional CUDA/DALI paths (real decode backends)
- ✅ Short smoke trains (I-JEPA mini-fixture for 10 steps)
- ✅ Consistency validator jobs (physics violation detection)

**Pattern Match:**
- Exactly mirrors APEX Phase 3 architecture (PR #884)
- Uses same marker taxonomy: `synthetic`, `real`, `ml`, `benchmark`

### 3.2 Test Marker Taxonomy and Isolation

**Decision: APPROVED with CLARIFICATION**

**Roadmap proposes:**
```python
@pytest.mark.synthetic  # Stub backends, no deps
@pytest.mark.real       # Real models/backends
@pytest.mark.ml         # Requires torch
@pytest.mark.benchmark  # Performance measurement
```

**MANDATORY CLARIFICATION:**

Add spatial-specific markers (parallel to existing `apex`, `efficientsam`):

```python
@pytest.mark.spatial_ingest    # Linear decode tests
@pytest.mark.spatial_catalog   # Ledger/query tests
@pytest.mark.spatial_geometry  # Pose/reconstruction tests
@pytest.mark.spatial_jepa      # Foundation model tests
```

**CI Lane Assignment:**
```yaml
# PR lane (fast, synthetic)
pytest tests/spatial_ai/ -m "not real and not ml and not benchmark"

# Nightly lane (deep, real)
pytest tests/spatial_ai/ -m "real or benchmark" --maxfail=5
```

This matches the APEX split between workflow-level synthetic/real modes in
`.github/workflows/apex_performance.yml` and the current root
`tests/test_apex_*.py` / `tests/evals/test_apex_*.py` files.

### 3.3 Definition of Done for Phase I

**Decision: APPROVED with ENFORCEMENT ADDITIONS**

**Roadmap DoD:**
1. ✅ Linear ingest produces float tensors + sidecars
2. ✅ Contract validation fails on uint8/non-linear/missing fields
3. ✅ CI exit code non-zero on contract violations

**MANDATORY ADDITIONS:**

4. **Registry integration test:**
   - Verify `SpatialBackendRegistry.get_backend("hashsidecar")` resolves correctly
   - Verify unknown backend raises `BackendNotFoundError` in strict mode

5. **Provenance hash chain validation:**
   - Tamper test: modify EXR bytes → hash mismatch → hard failure
   - Determinism test: re-decode same RAW → identical `decode_recipe_hash`

6. **Catalog query isolation test:**
   - Insert synthetic captures into SQLite catalog
   - Query by `project_id`, `visit_id`, `spatial_index`
   - Verify no SQL injection via malformed inputs (security posture)

7. **Documentation completeness:**
   - ADR for Spatial Capture Contract (Milestone 0)
   - ADR for Linear Ingest Strategy (Milestone 2)
   - Migration guide: "How to convert existing TIFF workflow to spatial ingest"

---

## 4. Dependency Management Assessment

### 4.1 Tiered Dependency Strategy

**Decision: APPROVED**

Roadmap correctly proposed an optional spatial dependency lane parallel to the
ML lane. The current repo exposes a `spatial-ai` optional extra in
`pyproject.toml`; it does not currently have a checked-in
`requirements/spatial-ai.in` lock lane.

**Current Package Metadata State:**
```
pyproject.toml
  [project.optional-dependencies]
    spatial-ai = [...]  # Optional spatial research dependencies
```

**Approval Rationale:**
- Maintains existing pattern (no disruption to `lux_depth_v3` installs)
- Keeps spatial deps optional (users not doing spatial work don't pay cost)
- Allows independent versioning and constraints

**MANDATORY ADDITION:**

Already present in `requirements/constraints.txt` under ADR-024:
```txt
# apache-iceberg  # Java interop (py4j), supply chain risk not assessed
apache-iceberg>=9999.0.0
```

**Rationale:**
- Iceberg has complex Java interop (PyIceberg wraps Java libs via py4j)
- Supply chain risk not yet assessed
- SQLite Tier A backend is sufficient for Phase I
- Unblock Iceberg in Phase II only after:
  1. License + maintenance audit
  2. Reproducible build verification
  3. CI integration test (synthetic Iceberg table CRUD)

### 4.2 Registry-Backed Dependency Validation

**Decision: APPROVED**

Roadmap proposes:
```python
# src/transformation_portal/spatial_ai/backends/registry.py
class SpatialBackendRegistry:
    def get_backend(self, name: str, strict: bool = False) -> SpatialBackend:
        if strict and name not in self._backends:
            raise BackendNotFoundError(...)
        # License validation
        # Dependency checks via backend.ensure_available()
```

**Pattern Match:**
- Exactly mirrors `DepthBackendRegistry` (ADR-019, PR #880)
- License gating follows `non_commercial_ok` precedent

**MANDATORY ENFORCEMENT:**

Add CI check (parallel to APEX backend validation):
```yaml
# .github/workflows/spatial-contract.yml
- name: Validate Spatial Backend Registry
  run: |
    python -m transformation_portal.spatial_ai.validation.registry_audit \
      --fail-on-missing-deps \
      --fail-on-unlicensed
```

This script should:
1. Enumerate all registered backends
2. Call `backend.ensure_available()` for non-stub backends
3. Verify `backend.required_packages()` are importable (in nightly lane only)
4. Fail if license requirements aren't documented

### 4.3 RAW Decode Dependency Risk

**Decision: APPROVED with RISK MITIGATION**

**Dependency:**
- `rawpy>=0.18.0` (LibRaw Python wrapper)

**Risk Assessment:**
- ✅ **License:** LGPL 2.1 (library), CDDL/LGPL dual (LibRaw core) — acceptable for use
- ✅ **Maintenance:** Active (last release 2023, LibRaw actively maintained)
- ⚠️ **Security:** RAW parsers are complex (buffer overflow history in LibRaw 0.19.x)
- ⚠️ **Supply chain:** Wraps native C++ library (build complexity)

**MANDATORY RISK MITIGATION:**

1. **Pin `rawpy` to tested version** in `requirements/spatial-ai.txt`:
   ```txt
   rawpy==0.18.1  # Pinned: security-audited, no known CVEs as of 2026-02-11
   ```

2. **Add Dependabot monitoring** for rawpy CVEs:
   ```yaml
   # .github/dependabot.yml (add to existing)
   - package-ecosystem: "pip"
     directory: "/requirements"
     schedule:
       interval: "weekly"
     allow:
       - dependency-name: "rawpy"
   ```

3. **Input sanitization** (already planned, but emphasize):
   - Filename sanitization (no path traversal)
   - File size limits (default: 500MB per DATA_CONTRACT.md)
   - Magic number verification (reject non-RAW headers)

4. **Sandboxed decode** (future hardening, not Phase I blocker):
   - Consider `bubblewrap` or container-based RAW decode for production
   - Document as "recommended for enterprise deployment" in SECURITY.md

---

## 5. Integration Risk Assessment

### 5.1 CRITICAL RISK: lux_depth_v3 RAW Ingest Coupling

**Status: HIGH PRIORITY — Must define explicit boundary**

**Risk Scenario:**

Current state:
- `lux_depth_v3` has `raw_loader.py` that decodes RAW → 8-bit sRGB for rendering
- Spatial AI requires RAW → linear ACEScg for training

**If we're not careful:**
1. Developer modifies `raw_loader.py` to "add linear mode"
2. Code becomes conditional mess: `if spatial_mode: ... else: ...`
3. Rendering pipeline accidentally gets linear input → visual breakage
4. Spatial pipeline accidentally gets sRGB input → training collapse

**MANDATORY MITIGATION:**

**Option A (Recommended): Total Separation**
```python
# lux_depth_v3/raw_loader.py (UNCHANGED, rendering-only)
def load_raw_for_rendering(path: Path) -> np.ndarray:
    """8-bit sRGB for lux rendering (existing behavior, frozen)."""
    # ... existing code ...

# spatial_ai/ingest/linear_decoder.py (NEW, training-only)
def decode_raw_linear_acescg(path: Path) -> LinearCaptureResult:
    """ACEScg linear for spatial training (Milestone 2)."""
    # ... new code, no shared logic with raw_loader.py ...
```

**Option B (Shared Utilities Only):**
- Extract shared RAW metadata parsing to `utils/raw_metadata.py`
- Keep decode logic completely separate

**Enforcement:**
1. **CI Lint Rule:**
   ```python
   # scripts/security/verify_pipeline_isolation.py
   def test_no_spatial_imports_in_lux_depth():
       """Lux Depth V3 must not import spatial_ai modules."""
       assert "spatial_ai" not in lux_depth_v3_imports()
   ```

2. **ADR Required:**
   - ADR-XXX: "Spatial AI Ingest Isolation Boundary"
   - Decision: "No shared RAW decode code between rendering and training pipelines"
   - Rationale: "Prevent silent rendering/training pipeline cross-contamination"

### 5.2 Provenance Backward Compatibility

**Risk:** Existing `lux_depth_v3` depth maps have provenance in `DepthProvenance` format, not `SpatialProvenanceSpec`.

**Mitigation Strategy:**

**Decision: APPROVED (roadmap already handles this)**

Roadmap correctly proposes:
```python
# SpatialCaptureV1 contract
pipeline_manifest_path: str | None  # Link to existing lux-depth manifest
```

**Enhancement:**
- Add migration utility:
  ```python
  # src/transformation_portal/spatial_ai/validation/migrate_provenance.py
  def convert_depth_provenance_to_spatial(
      depth_artifact: DepthArtifact
  ) -> SpatialProvenanceSpec:
      """Convert lux_depth_v3 provenance to spatial format."""
  ```

- Document in `docs/spatial_ai/MIGRATION.md`:
  - "How to index existing lux_depth_v3 outputs in spatial catalog"
  - "Provenance schema differences and compatibility layer"

### 5.3 Catalog Query Security (Spatial Injection Risk)

**Status: BLOCKER — Must address before catalog implementation**

**Risk Scenario:**

Roadmap proposes catalog queries like:
```python
catalog.query(project_id="750 Picacho")
catalog.query(spatial_bounds=(lat_min, lat_max, lon_min, lon_max))
```

**If implemented naively:**
```python
# INSECURE (DO NOT IMPLEMENT)
query = f"SELECT * FROM captures WHERE project_id = '{project_id}'"
```

**Attacker input:**
```python
project_id = "'; DROP TABLE captures; --"
```

**MANDATORY MITIGATION:**

1. **Parameterized queries ONLY:**
   ```python
   # catalog/ledger_sqlite.py
   def query_by_project(self, project_id: str) -> list[SpatialCaptureV1]:
       cursor.execute(
           "SELECT * FROM captures WHERE project_id = ?",
           (project_id,)  # ✅ SAFE: parameterized
       )
   ```

2. **Input validation at API boundary:**
   ```python
   # validation/query_sanitizer.py
   def validate_project_id(project_id: str) -> str:
       if not re.match(r'^[a-zA-Z0-9_\-\./ ]+$', project_id):
           raise ValueError("Invalid project_id format")
       if len(project_id) > 256:
           raise ValueError("project_id too long")
       return project_id
   ```

3. **Security test requirement:**
   ```python
   # tests/spatial_ai/test_catalog_security.py
   def test_sql_injection_resistance():
       """Catalog queries must resist SQL injection."""
       malicious_inputs = [
           "'; DROP TABLE captures; --",
           "1' OR '1'='1",
           "'; UPDATE captures SET project_id='pwned'; --"
       ]
       for malicious in malicious_inputs:
           with pytest.raises(ValueError):
               catalog.query_by_project(malicious)
   ```

4. **Document in SECURITY.md:**
   - "Spatial catalog queries use parameterized SQL only"
   - "All user inputs validated before catalog access"
   - "SQLite connection created with `check_same_thread=False` forbidden (thread safety)"

---

## 6. Phase I Feasibility Assessment

### 6.1 Milestone Sequencing

**Decision: APPROVED**

Roadmap proposes:
```
M0: Contracts + Provenance → M1: Catalog → M2: Linear Ingest
```

**Architect Assessment:**

✅ **Correct order:**
- Contract schema defines what catalog stores
- Catalog exists before ingest produces samples
- Ingest is the "write path," catalog is the "read path"

**Alternative Considered (and Rejected):**
```
M2 (Ingest) → M0 (Contracts) → M1 (Catalog)
```
Rejected because: "Write before schema" leads to schema drift and rework.

### 6.2 PR Stack Realism

**Roadmap Proposes:**
```
PR M0.1: Contracts + validator (schema only, no I/O)
PR M0.2: Provenance sidecar + hashing (JSON I/O, hash utils)
PR M0.3: Optional C2PA backend (c2pa-python integration)
PR M1.1: Catalog protocol + SQLite (Tier A)
PR M1.2: Optional Iceberg backend (Tier B, deferred)
PR M2.1: LinearDecoder + sidecar (RAW → linear tensor)
PR M2.2: Spatial ingest CLI
```

**Architect Assessment:**

✅ **Size is reasonable:**
- Each PR is focused (single responsibility)
- PRs build incrementally (no "big bang" merge)
- Early PRs are low-risk (schema + validation, no ML deps)

⚠️ **Risk: PR M2.1 Complexity**

PR M2.1 combines:
- RAW decoding (LibRaw via rawpy)
- Color space transform (RAW sensor RGB → ACEScg)
- Noise estimation (Poisson-Gaussian parameter fit)
- EXIF extraction (pyexif or ExifTool)

**Recommendation:**
- Split PR M2.1 into two PRs:
  - **PR M2.1a:** RAW → linear RGB (sensor space, no color transform yet)
  - **PR M2.1b:** Color space transform + ACEScg validation

**Rationale:**
- Isolates LibRaw risk from color science risk
- Allows incremental review (RAW decode correctness, then color correctness)
- Reduces PR review cognitive load

### 6.3 Resource Requirements (Personnel, Time, Compute)

**Phase I Personnel Estimate:**

| Milestone | Estimated Effort | Blocker Risk |
|-----------|------------------|--------------|
| M0 (Contracts) | 1-2 weeks | Low (schema design, no ML) |
| M1 (Catalog) | 2-3 weeks | Medium (SQLite schema, query design, security) |
| M2 (Linear Ingest) | 3-4 weeks | High (RAW decode, color science, noise) |

**Total Phase I:** 6-9 weeks (single full-time contributor)

**Compute Requirements (Phase I):**
- PR lane: CPU-only (synthetic fixtures, no GPU)
- Nightly lane: Optional MPS/CUDA for smoke tests (not required for merge)
- No large-scale training (deferred to Phase II)

**Architect Assessment:**

✅ **Feasible for Phase I** (assuming single experienced contributor)

⚠️ **Risk Factor:** Color science expertise

- ACEScg transform from RAW requires deep color management knowledge
- LibRaw defaults may not preserve ACEScg gamut
- Consider:
  1. Hiring color science consultant (1-2 week contract)
  2. Deferring ACEScg to Phase II (use linear sRGB for MVP, migrate later via ADR)
  3. Partnering with ACES/OCIO community for validation

**Recommendation:**
- Start with **linear sRGB for Phase I MVP** (simpler, well-understood)
- Add **ACEScg in Phase II** after foundation is stable
- Update DATA_CONTRACT.md v1.0.0 → v1.1.0 with migration plan
- This reduces Phase I risk significantly

---

## 7. Missing Considerations and Gaps

### 7.1 Temporal Consistency Validation (Missing from Roadmap)

**Gap Identified:**

Roadmap emphasizes longitudinal structure (`project_id → visit_id → timeline`) but doesn't specify:

1. **How to detect temporal anomalies:**
   - Frame B claims to be "after" Frame A, but lighting conditions regress
   - Visit 2 shows "additions" that contradict Visit 1 geometry

2. **How to enforce causal constraints:**
   - "Chair cannot disappear and reappear without explicit deletion/addition event"
   - "Wall cannot move (unless renovation logged)"

**MANDATORY ADDITION:**

Add to Phase III (Milestone 9 — Consistency Loop):

```python
# src/transformation_portal/spatial_ai/validation/temporal_consistency.py
def validate_visit_sequence(
    visit_a: list[SpatialCaptureV1],
    visit_b: list[SpatialCaptureV1]
) -> ConsistencyReport:
    """Validate temporal consistency between visits.

    Checks:
    - Causal time ordering (visit_b.capture_time > visit_a.capture_time)
    - Geometry drift (3DGS-CD residual thresholds)
    - Lighting consistency (no drastic HDR histogram shifts)
    - Object permanence (tracked objects don't vanish)
    """
```

**CI Integration:**
```yaml
# nightly.yml (add to spatial deep checks)
- name: Temporal Consistency Validation
  run: |
    python -m transformation_portal.spatial_ai.validation.temporal_consistency \
      --project-id "750_picacho" \
      --fail-on-causality-violation
```

### 7.2 PII Redaction Policy (Underspecified)

**Gap Identified:**

Roadmap mentions "PII redaction" in Milestone 10 but doesn't define:

1. **What counts as PII in spatial data?**
   - Faces (obviously)
   - License plates (obviously)
   - House numbers / street addresses visible in photos?
   - Personal documents in scene (mail, photos on walls)?
   - Identifiable artwork (copyrighted paintings)?

2. **Redaction strategy:**
   - Semantic inpainting (replace face with generic texture)?
   - Bounding box blur (loses spatial coherence)?
   - Full-frame rejection (too aggressive)?
   - Depth-aware inpainting (preserve 3D structure)?

**MANDATORY ADDITION:**

Add ADR:
- **ADR-XXX: Spatial PII Redaction Policy**
- Define PII taxonomy (faces, plates, documents, etc.)
- Choose redaction method (recommend: depth-aware semantic inpainting)
- Document provenance action: `redacted` with bounding boxes logged

Add to `src/transformation_portal/presence_security/`:
```python
# src/transformation_portal/presence_security/spatial_redactor.py
def redact_spatial_capture(
    capture: SpatialCaptureV1,
    redaction_policy: RedactionPolicy
) -> SpatialCaptureV1:
    """Depth-aware PII redaction for spatial captures."""
```

**Security Requirement:**
- Redaction MUST be logged in provenance chain
- Original unredacted captures MUST NOT be exported via World API
- Catalog MUST support `redacted_only=True` query filter

### 7.3 Model Weight Provenance (Missing from JEPA Milestones)

**Gap Identified:**

Roadmap proposes I-JEPA (M6) and V-JEPA (M7) training but doesn't specify:

1. **Where are model weights stored?**
   - `checkpoints/` (current depth model location)?
   - `models/spatial_jepa/` (new dedicated location)?
   - Remote blob storage (S3/GCS)?

2. **How are weights versioned and hashed?**
   - Git LFS (not recommended for >100MB models)?
   - DVC (Data Version Control)?
   - Custom hash manifest (like `DepthProvenance.checkpoint_sha256`)?

3. **How to reproduce training run?**
   - Training data snapshot (Iceberg table commit hash)?
   - Hyperparameters + code version?
   - Random seed + device config?

**MANDATORY ADDITION:**

Add to Milestone 6 (I-JEPA):

```python
# src/transformation_portal/spatial_ai/jepa/provenance.py
@dataclass
class ModelProvenance:
    """Training provenance for JEPA models."""
    model_id: str  # "i-jepa-v1.0-20260301"
    checkpoint_sha256: str
    training_data_snapshot: str  # Iceberg commit hash
    training_config_hash: str  # Hash of hyperparameters
    code_version: str  # Git commit SHA
    device_config: dict  # GPU model, CUDA version, etc.
    training_duration_hours: float
    final_loss: float
    evaluation_metrics: dict
```

**Storage Strategy:**
- Small models (<100MB): Git LFS
- Large models (>100MB): S3/GCS with manifest in repo
- Manifest pattern (like Hugging Face revision pins):
  ```yaml
  # config/spatial_ai/model_registry.yaml
  i-jepa-v1.0:
    storage: "s3://transformation-portal-models/i-jepa-v1.0.pth"
    sha256: "abc123..."
    license: "Apache-2.0"
    training_provenance: "s3://.../i-jepa-v1.0-provenance.json"
  ```

### 7.4 Backward Compatibility Strategy (Missing for Phase Transitions)

**Gap Identified:**

What happens when:
- Phase I (SQLite catalog) transitions to Phase II (Iceberg catalog)?
- Phase I (linear sRGB) transitions to Phase II (ACEScg)?
- SpatialCaptureV1 schema needs to evolve to V2?

**MANDATORY ADDITION:**

Add to roadmap:

**Schema Evolution Policy:**
```python
# src/transformation_portal/spatial_ai/contracts/migrations.py
def migrate_v1_to_v2(v1_capture: SpatialCaptureV1) -> SpatialCaptureV2:
    """Migrate contract schema with backward compatibility."""
```

**Catalog Migration Strategy:**
```python
# scripts/spatial_ai/migrate_sqlite_to_iceberg.py
def migrate_catalog(
    sqlite_path: Path,
    iceberg_catalog_uri: str
) -> MigrationReport:
    """One-time migration: SQLite → Iceberg."""
```

**Versioning Invariant:**
- All spatial artifacts MUST include `schema_version` field
- Older versions supported via migration adapters (no breaking changes)
- Deprecation cycle: 1 major version warning, 2 major versions removed

---

## 8. Recommendations and Required Modifications

### 8.1 MANDATORY PRE-PHASE-I WORK

Before any Phase I implementation PRs:

1. **[BLOCKER] Resolve ACEScg vs EXR color space conflict**
   - Decision: Update roadmap to mandate ACEScg, OR
   - Alternative: Create ADR superseding DATA_CONTRACT.md with "linear sRGB for Phase I MVP"
   - Owner: Architect (me)
   - Timeline: Before M0 PR

2. **[DONE WITH DRIFT] Create ADR-023: Spatial AI Ingest Isolation Boundary**
   - Mandate: No shared RAW decode code between `lux_depth_v3` and `spatial_ai`
   - Current state: ADR-023 exists, but its older blanket cross-import
     enforcement no longer matches the live codebase; see the ADR current-state
     note before treating `scripts/security/verify_pipeline_isolation.py` as a
     green CI gate.
   - Owner: Architect (me)
   - Timeline: Before M2 PR

3. **[DONE] Ban Apache Iceberg pending supply chain audit**
   - `apache-iceberg>=9999.0.0` is present in `requirements/constraints.txt`
     and tracked by ADR-024.
   - SQLite Tier A backend only for Phase I
   - Unblock in Phase II after ADR approval
   - Owner: Architect (me)
   - Timeline: Immediate (next commit)

4. **[HIGH] Create ADR-XXX: Spatial PII Redaction Policy**
   - Define PII taxonomy for spatial data
   - Choose redaction method (depth-aware inpainting)
   - Document provenance logging requirements
   - Owner: Specialist (with Architect review)
   - Timeline: Before M10 (Productionization)

5. **[MEDIUM] Document Model Weight Provenance Strategy**
   - Add `ModelProvenance` schema to contracts
   - Choose storage strategy (S3 + manifest vs Git LFS)
   - Document reproducibility requirements
   - Owner: Specialist
   - Timeline: Before M6 (I-JEPA)

### 8.2 RECOMMENDED ROADMAP ADJUSTMENTS

**Phase I De-risking:**

1. **Use linear sRGB for Phase I MVP** (instead of ACEScg)
   - Rationale: Simpler transform, faster iteration, lower color science expertise requirement
   - Migration: Phase II adds ACEScg via provenance-tracked reprocessing
   - Trade-off: Accepted gamut limitation for luxury materials (revisit in Phase II)

2. **Split PR M2.1 into two PRs:**
   - M2.1a: RAW → linear RGB (sensor space)
   - M2.1b: Color space transform + validation
   - Rationale: Isolates LibRaw risk from color science risk

3. **Add temporal consistency validation to Phase III:**
   - New: Milestone 9.5 (between consistency loop and productionization)
   - Validates causal time ordering and geometry drift
   - Prevents training on temporally inconsistent sequences

**Phase II Enhancements:**

4. **Defer Iceberg to Phase II** (keep SQLite Tier A for Phase I)
   - Rationale: Supply chain risk, complexity, overkill for initial scale
   - Trigger: When catalog exceeds 1M samples or multi-node queries needed

5. **Add "bridge mode" in Phase I** (not deferred to Phase III)
   - Allow: `lux-depth-v3 ... --emit-spatial-capture on`
   - Rationale: Enables gradual migration of existing projects to spatial catalog
   - Implementation: Orchestrator writes both depth artifact + spatial capture sidecar

### 8.3 CRITICAL SUCCESS FACTORS

For Phase I to succeed:

1. **Architect must remain engaged** (per governance policy)
   - Review all spatial PRs before merge
   - Approve any ADR deviations
   - Enforce "stop and escalate" protocol

2. **Color science validation** (external review recommended)
   - If using ACEScg: Partner with ACES community for transform validation
   - If using linear sRGB: Document gamut limitations and migration plan

3. **Security posture must be proactive** (not reactive)
   - Catalog query parameterization from day 1
   - PII redaction policy before any production deployment
   - Provenance integrity checks in every validator

4. **Documentation must match enforcement**
   - Every "MUST" in DATA_CONTRACT.md requires CI gate
   - Every banned dependency requires `constraints.txt` entry
   - Every security claim requires test coverage

---

## 9. Final Architectural Decision

### 9.1 Approval Status

**APPROVED with CONDITIONS**

The Spatial AI Foundation roadmap is architecturally sound and demonstrates strong repository-grounded discipline. The protocol-based backend pattern, APEX testing philosophy, and contract enforcement approach are exemplary.

**However, implementation MUST NOT proceed until:**

1. ✅ ACEScg vs EXR color space conflict resolved (ADR or roadmap update)
2. ✅ ADR-XXX: Spatial AI Ingest Isolation Boundary created
3. ✅ Apache Iceberg banned in `constraints.txt` pending audit
4. ✅ Catalog query security hardened (parameterized SQL + input validation)

### 9.2 Phase I Authorization

**Phase I (Milestones 0-3) is AUTHORIZED** subject to:

- Mandatory pre-work completed (Section 8.1)
- All PRs follow established governance (agent_governance.md)
- Architect reviews all spatial PRs before merge
- Security tests pass in CI (catalog injection, provenance tampering)

**Phase II/III are NOT YET AUTHORIZED** (require separate architectural review)

### 9.3 Escalation Reminder

Per `docs/architecture/agent_governance.md`:

Any implementation work that:
- Deviates from this review's decisions
- Encounters architectural ambiguity
- Proposes new dependencies or security posture changes

**MUST stop and escalate** to Architect before proceeding.

**Silence is not approval.**

---

## Appendices

### Appendix A: Quick Reference Checklist for Specialist

**Before starting Phase I implementation:**

- [ ] Read this review in full
- [ ] Verify ACEScg color space decision (Architect will clarify)
- [ ] Confirm ADR-023 (Ingest Isolation) still matches the intended boundary
- [ ] Verify `apache-iceberg>=9999.0.0` in `constraints.txt`
- [ ] Review existing `DepthBackend` pattern (ADR-019)
- [ ] Review existing `DepthArtifact` contract (lux_depth_v3/contracts/)
- [ ] Understand APEX workflow modes and current `tests/test_apex_*.py` /
      `tests/evals/test_apex_*.py` coverage
- [ ] Read SECURITY.md untrusted input handling requirements

**For every Phase I PR:**

- [ ] Follows protocol + backend + stub pattern
- [ ] Includes contract schema version field
- [ ] Includes synthetic test fixtures (good + bad cases)
- [ ] Includes security test (injection, tampering, etc.)
- [ ] Updates ADR if deviating from roadmap
- [ ] Tags Architect for review (@transformation-portal-architect)

### Appendix B: Referenced Repository Artifacts

**Existing Patterns to Reuse:**

- `src/transformation_portal/depth/backends/protocol.py` — DepthBackend Protocol
- `src/transformation_portal/lux_depth_v3/contracts/depth_artifact.py` — Contract example
- `docs/architecture/ADR-019-depth-backend-unification.md` — Backend architecture
- `tests/test_apex_*.py` and `tests/evals/test_apex_*.py` — current APEX test coverage
- `.github/workflows/quality-gate.yml` — PR lane enforcement
- `.github/workflows/nightly.yml` — Deep check strategy

**New Artifacts This Review Requires:**

- `docs/architecture/ADR-023-spatial-ai-ingest-isolation.md` (EXISTS; review current-state drift before using its enforcement script)
- `docs/architecture/ADR-XXX-spatial-pii-redaction-policy.md` (HIGH)
- `requirements/constraints.txt` update (ban Iceberg) (DONE; see ADR-024)
- `src/transformation_portal/spatial_ai/validation/query_sanitizer.py` (BLOCKER)

---

**Review Complete.**

**Next Action:** Architect to create mandatory ADRs and update constraints.txt before authorizing Phase I implementation.

**Specialist:** Wait for explicit "Phase I GO" signal from Architect before beginning Milestone 0 PRs.
