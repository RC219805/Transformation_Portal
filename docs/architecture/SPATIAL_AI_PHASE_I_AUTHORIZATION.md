# Spatial AI Foundation — Phase I Authorization

**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Reviewed Document:** `docs/spatial_ai/ROADMAP.md` implementation translation
**Architectural Review:** `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md`

---

## Authorization Status

**Phase I (Milestones 0-3) is AUTHORIZED to proceed** subject to conditions below.

**Phase II/III are NOT YET AUTHORIZED** (require separate architectural review).

---

## Mandatory Pre-Work (BLOCKERS — Must Complete Before M0 PR)

### ✅ COMPLETED

1. **[BLOCKER] Architectural Review**
   - Document: `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md`
   - Status: ✅ Complete
   - Verdict: APPROVED with CONDITIONS

2. **[BLOCKER] ADR-023: Spatial AI Ingest Isolation Boundary**
   - Document: `docs/architecture/ADR-023-spatial-ai-ingest-isolation.md`
   - Status: ✅ Approved
   - Enforcement: CI script created

3. **[BLOCKER] ADR-024: Apache Iceberg Dependency Ban**
   - Document: `docs/architecture/ADR-024-apache-iceberg-ban.md`
   - Status: ✅ Approved
   - Enforcement: `requirements/constraints.txt` updated

4. **[BLOCKER] CI Enforcement: Pipeline Isolation**
   - Script: `scripts/security/verify_pipeline_isolation.py`
   - Status: ✅ Created and tested
   - Next: Add to `.github/workflows/spatial-contract.yml` (M0 PR)

### ⏳ PENDING (Specialist to resolve before M0 PR)

5. **[BLOCKER] Resolve ACEScg vs EXR Color Space Conflict**
   - Issue: DATA_CONTRACT.md frozen to ACEScg, roadmap assumes EXR without color space spec
   - Architect Recommendation: Update roadmap Milestone 0 to mandate:
     ```python
     encoding.color_space: Literal["acescg"]  # Frozen per DATA_CONTRACT.md v1.0.0
     ```
   - Alternative: Create ADR superseding DATA_CONTRACT.md allowing "linear sRGB for Phase I MVP"
   - **DECISION REQUIRED:** Specialist should propose resolution, Architect will approve
   - **TIMELINE:** Before M0.1 PR (Contracts + validator)

---

## Phase I Implementation Rules

### Governance

1. **All spatial PRs MUST be reviewed by Architect** before merge
   - Tag: `@transformation-portal-architect`
   - Per: `docs/architecture/agent_governance.md`

2. **Escalation protocol applies**
   - Dependency changes → escalate
   - Security posture changes → escalate
   - Cross-pipeline contracts → escalate
   - ADR conflicts → escalate

3. **Silence is not approval**
   - Explicit Architect "LGTM" required for merge
   - No auto-merge, no "assumed approval"

### Technical Constraints

1. **Pattern adherence:**
   - Protocol + backend + stub + registry (ADR-019 pattern)
   - Contract-first design (APEX discipline)
   - Synthetic PR lane, real nightly checks

2. **Isolation boundaries:**
   - No `lux_depth_v3` imports in `spatial_ai` (ADR-023)
   - No `spatial_ai` imports in `lux_depth_v3` (ADR-023)
   - CI enforces automatically

3. **Dependency governance:**
   - No Iceberg until ADR-024 superseded (supply chain audit)
   - SQLite Tier A backend only for Phase I
   - Catalog must support future Iceberg migration (schema compatibility)

4. **Security posture:**
   - Parameterized SQL only (no string interpolation)
   - Input validation at all API boundaries
   - Provenance hash chain verification
   - PII redaction policy documented before M10

### Quality Gates

1. **Every PR must include:**
   - Synthetic test fixtures (good + bad cases)
   - Security tests (injection, tampering, isolation)
   - Contract schema validation
   - CI passing (all gates green)

2. **Definition of Done for Phase I:**
   - Linear ingest produces float tensors + sidecars
   - Contract validation fails on uint8/non-linear/missing fields
   - CI exit code non-zero on violations
   - Registry integration test passes
   - Provenance hash chain validation passes
   - Catalog query isolation test passes
   - Documentation complete (ADRs, migration guides)

---

## Milestone-Specific Approval

### Milestone 0: Contracts + Provenance

**Status:** AUTHORIZED (pending color space resolution)

**PRs:**
- M0.1: Contracts + validator
- M0.2: Provenance sidecar + hashing
- M0.3: Optional C2PA backend

**Blockers:**
- ⏳ Resolve ACEScg color space conflict (see above)

**Go/No-Go:** Specialist proposes color space resolution → Architect approves → M0.1 PR proceeds

### Milestone 1: Catalog

**Status:** AUTHORIZED (SQLite Tier A only)

**PRs:**
- M1.1: Catalog protocol + SQLite implementation
- M1.2: Optional Iceberg backend — ❌ BLOCKED (ADR-024)

**Constraints:**
- SQLite only (no Iceberg)
- Schema must be Iceberg-compatible (future migration)
- Query security tests mandatory (SQL injection resistance)

**Go/No-Go:** M0 complete → M1 proceeds

### Milestone 2: Linear Ingest

**Status:** AUTHORIZED (with split recommendation)

**PRs:**
- M2.1a: RAW → linear RGB (recommended split)
- M2.1b: Color space transform + validation (recommended split)
- M2.2: Spatial ingest CLI

**Constraints:**
- Complete isolation from `lux_depth_v3.raw_loader` (ADR-023)
- Color space matches M0 resolution (ACEScg or linear sRGB)
- Noise estimation documented but may be deferred to M2.3

**Recommendation:**
- Split M2.1 into two PRs to isolate LibRaw risk from color science risk
- Consider linear sRGB for Phase I MVP (defer ACEScg to Phase II)

**Go/No-Go:** M1 complete → M2 proceeds

### Milestone 3: I/O Architecture

**Status:** AUTHORIZED (CPU-only, benchmark markers)

**Constraints:**
- PR lane: CPU decode only (synthetic fixtures)
- Nightly lane: Optional CUDA/DALI paths
- Benchmark markers: `@pytest.mark.benchmark`
- No GPU required for CI passing

**Go/No-Go:** M2 complete → M3 proceeds

---

## Phase II/III (NOT YET AUTHORIZED)

**Milestones 4-10 (Geometry, JEPA, VL, Production):**
- Status: ❌ NOT AUTHORIZED
- Requires: Separate architectural review after Phase I complete
- Trigger: Phase I Definition of Done achieved
- Owner: Architect (new review cycle)

**Do NOT begin Phase II implementation until:**
1. Phase I Definition of Done verified
2. Architect conducts Phase II architectural review
3. Explicit Phase II authorization issued

---

## Enforcement Mechanisms

### CI Gates (Mandatory for PR merge)

1. **Contract Validation:**
   ```yaml
   # .github/workflows/spatial-contract.yml (to be created in M0.1 PR)
   - name: Validate Spatial Contracts
     run: python -m transformation_portal.spatial_ai validate-contract --strict
   ```

2. **Pipeline Isolation:**
   ```yaml
   # .github/workflows/spatial-contract.yml (add in M0.1 PR)
   - name: Verify Pipeline Isolation
     run: python scripts/security/verify_pipeline_isolation.py
   ```

3. **Dependency Governance:**
   ```yaml
   # .github/workflows/quality-gate.yml (existing, applies automatically)
   - name: Install with Constraints
     run: pip install -c requirements/constraints.txt --build-constraint requirements/constraints.txt -r requirements/spatial-ai.txt
   ```
   (Iceberg install will fail due to constraints.txt ban)

4. **Security Testing:**
   ```yaml
   # .github/workflows/spatial-contract.yml (add in M1.1 PR)
   - name: Catalog Security Tests
     run: pytest tests/spatial_ai/test_catalog_security.py -v
   ```

### Human Review (Mandatory for PR merge)

1. **Architect approval required:**
   - All spatial PRs
   - Tag: `@transformation-portal-architect`
   - No auto-merge

2. **Escalation required:**
   - Dependency changes
   - Security posture changes
   - Cross-pipeline contracts
   - ADR conflicts

---

## Communication Protocol

### For Specialist

**Starting implementation:**
1. Resolve color space blocker (propose resolution to Architect)
2. Wait for explicit "GO" from Architect
3. Begin M0.1 PR (Contracts + validator)

**During implementation:**
1. Tag Architect on every spatial PR
2. Escalate immediately if blocked or uncertain
3. Do NOT assume approval (silence ≠ approval)

**After Phase I:**
1. Verify Definition of Done
2. Request Phase II architectural review
3. Wait for explicit Phase II authorization

### For Architect

**Responsibilities:**
1. Review all spatial PRs before merge
2. Respond to escalations within 48 hours
3. Approve/reject/request-changes explicitly
4. Conduct Phase II review when triggered

**Review Criteria:**
1. ADR compliance (ADR-023, ADR-024)
2. Pattern adherence (protocol + backend + stub)
3. Security posture (injection, tampering, isolation)
4. Quality gates (tests, documentation, CI)

---

## Success Criteria (Phase I Complete)

Phase I is complete when:

1. ✅ All Milestones 0-3 PRs merged
2. ✅ Definition of Done achieved (see Section "Quality Gates")
3. ✅ CI passing (all gates green)
4. ✅ Documentation complete (ADRs, migration guides)
5. ✅ No open security issues (SQL injection, hash tampering, etc.)
6. ✅ Catalog contains >100 synthetic captures (smoke test data)
7. ✅ Specialist can run end-to-end: RAW → linear ingest → catalog query

**Then:** Trigger Phase II architectural review.

---

## References

- Architectural Review: `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md`
- Roadmap: `docs/spatial_ai/ROADMAP.md`
- Data Contract: `docs/spatial_ai/DATA_CONTRACT.md`
- ADR-023: `docs/architecture/ADR-023-spatial-ai-ingest-isolation.md`
- ADR-024: `docs/architecture/ADR-024-apache-iceberg-ban.md`
- Governance Policy: `docs/architecture/agent_governance.md`
- Enforcement Script: `scripts/security/verify_pipeline_isolation.py`

---

**Authorized By:** Transformation Portal Architect
**Date:** 2026-02-11
**Next Review:** Phase II architectural review (after Phase I DoD)
