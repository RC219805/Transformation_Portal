# ADR Decision-Currency Review — 2026-05-16

**Date:** 2026-05-16
**Branch:** `claude/review-adr-docs-JdxZP`
**Scope:** All 33 numbered ADRs under `docs/architecture/`
**Method:** Read each ADR's declared Status field, then verify implementation against code under `src/`, `scripts/`, `tests/`, `migrations/`, and CI workflows. Cross-reference against `docs/architecture/README.md` (file-disposition overlay), `docs/architecture/MONOLITH_DECOMPOSITION_TARGETS.md`, and CLAUDE.md.
**Relationship to other overlays:** The existing [`README.md`](README.md) is the *file-disposition* overlay (canonical / promoted / current-support / review-required). This document is the orthogonal *decision-currency* overlay (implemented / active / obsolete). The two answer different questions and have different cadences.

---

## Executive Summary

| Classification | Count | Notes |
|---|---|---|
| **Implemented** (decision shipped, ADR is historical record) | 11 | ADR-019, 029, 030, 031, 042, 043, 044, 046, 047, 048, and the ADR-001 PBR Integration |
| **Active — accepted policy** (still being enforced) | 14 | ADR-001-APPROVAL, ADR-015, 017, 018, 020, 021, 022, 023, 024, 032, 033, 034, 038, 040 |
| **Active — pending implementation** (Proposed; awaiting work) | 6 | ADR-025, 026, 035, 037, 039, 041 |
| **Active — partial implementation** | 2 | ADR-027, ADR-036 (Locked but additive layer) |
| **Active — accepted governance pattern** | 1 | ADR-045 (status bumped Proposed → Accepted in this PR) |
| **Obsolete** (superseded / abandoned / template) | 0 ADRs (1 template moved out of the numbered series) | The `ADR-0XX-vjepa2-...TEMPLATE.md` is not a real ADR; relocated to `templates/` |

**No ADR is fully obsolete in the "delete or supersede" sense.** Every ADR still represents either live policy, completed work worth preserving as a historical record, or proposed work that has not been formally withdrawn. The strongest cleanup targets are structural (number collisions, prefix inconsistency, template-as-ADR) rather than decisional, and those have been resolved in this PR.

**Structural fixes applied in this PR:**

1. Renumbered `adr-0015-da3-1-1-non-commercial-research-tier.md` → `ADR-015-...` (3-digit prefix to match the rest of the series).
2. Renumbered the second `ADR-030-materials-v3-production-integration.md` → `ADR-048-...` to resolve a collision with the canonical `ADR-030: Phase II Deterministic RAW Ingest`.
3. Moved `ADR-0XX-vjepa2-separate-repo-TEMPLATE.md` → `templates/ADR-vjepa2-separate-repo-TEMPLATE.md` (templates do not belong in the numbered series).
4. Bumped Status fields where the shipping evidence is now in the repo: ADR-045 (`Proposed → Accepted`), ADR-046 (`Proposed → Implemented`), ADR-047 (`Proposed → Implemented`).
5. Updated cross-references in active (non-archived) docs. Archived docs under `_archive/`, `historical/`, and `pr_archive/` were intentionally left untouched per the repo's "historical evidence is preserved" policy.

**Recommended follow-ups (NOT executed in this PR):**

- Verify whether ADR-025 (APEX Research Workflow) and ADR-026 (APEX Research Ultra) are still on the roadmap or have been quietly de-prioritized; if abandoned, mark `Superseded` with a pointer to the current direction.
- Decide whether ADR-035 (Bundle Root Anchoring) and ADR-041 (Phase 4F External Verifier) should be bumped to `Accepted` or `Implemented` based on Phase 3.4 / Phase 4F evidence-bundle code that may already have shipped (out of scope to verify in this review).
- Fold ADR-019 satellite docs (`ADR-019_FINAL_CHECKLIST.md`, `ADR-019_VERIFICATION_REPORT.md`) into a `support/` subfolder or link them explicitly from inside ADR-019 to reduce top-level clutter.
- Consider an `implemented/` subfolder for ADRs whose decisions are fully shipped and unlikely to need future amendment (ADR-019, 029, 030, 031, 042, 043, 044, 046, 047, 048).

---

## Classification Legend

| Bucket | Meaning | Evidence requirement |
|---|---|---|
| **Implemented** | Decision has shipped; ADR is now a historical record. Code/script/test evidence exists under `src/`, `scripts/`, `tests/`, `migrations/`, or `.github/workflows/`. | Required: file path to the implementing artifact. |
| **Active** | Either accepted policy still being enforced (e.g., banned dependency lists, license tiers, marker requirements), OR a Proposed decision still pending implementation, OR a partially-implemented decision with known deferred items. | Sub-classification: `policy` / `pending` / `partial` / `pattern`. |
| **Obsolete** | Superseded, abandoned, or a template/placeholder (not a real ADR). | Pointer to superseding ADR, or archival/relocation recommendation. |

---

## Master Classification Table

| ADR | Title | File Status | Classification | Sub | Evidence | Action |
|---|---|---|---|---|---|---|
| ADR-001 | PBR Module Integration Architecture | Proposed | Implemented | shipped | `src/transformation_portal/lux_depth_v3/pbr_presets.py`, `pbr_*.py` modules | none |
| ADR-001-APPROVAL | PBR Integration Approval Record | Approved | Active | policy | Approval governance for ADR-001 | none |
| ADR-015 | DA3 1.1 Non-Commercial Research Tier | Adopted | Active | policy | `@require_non_commercial` decorator; preset `config/presets/depth-anything-v3-1-research-m4.yaml`; ADR-019 enforcement | renumbered 2026-05-16 |
| ADR-017 | Parallelization Strategy for Batch Processing | Accepted | Active | policy | I/O-parallel batch path in pipeline coordinator; sequential GPU contract preserved | none |
| ADR-018 | Depth Pro Integration Decision | Adopted | Active | policy | `scripts/setup/install_depth_pro_runtime.sh`; `--depth-backend depth_pro` CLI gate; license dual-gate | none |
| ADR-019 | Depth Backend Unification Architecture | Implemented | Implemented | shipped | `src/transformation_portal/lux_depth_v3/da3_*.py`, `coreml_backend.py`, depth backend registry | none |
| ADR-020 | Drop Python 3.10 Support | Adopted | Active | policy | `pyproject.toml` python_requires ≥3.11; CI matrix | none |
| ADR-021 | HuggingFace Model Revision Pinning | Accepted | Active | policy | Dual-mode HF loader; `# nosec B615` markers; production revision pins | none |
| ADR-022 | V2 Enhancement Stage Optionality | Accepted | Active | policy | CLI flag to disable V2; default-off contract | none |
| ADR-023 | Spatial AI Ingest Isolation Boundary | Approved (Mandatory) | Active | policy | `src/transformation_portal/spatial_ai/` vs `lux_depth_v3/` isolation; RAW ingest separation | none |
| ADR-024 | Apache Iceberg Dependency Ban | Approved (Mandatory Ban) | Active | policy | `requirements/constraints.txt`; `scripts/security/verify_banned_dependencies.py` | none |
| ADR-025 | APEX Research Workflow Architecture | Proposed | Active | pending | No shipped APEX-Research orchestration verified in this review | follow-up: confirm roadmap status |
| ADR-026 | APEX Research Ultra | Proposed | Active | pending | Extends ADR-025; not verified as shipped | follow-up: confirm roadmap status |
| ADR-027 | Phase 2 Spatial AI Extension Architecture | Partially Implemented | Active | partial | SAM2 segmentation + orchestration bridge shipped; 3D reconstruction deferred; materials placeholder | none (Status already reflects partial state) |
| ADR-029 | Execution Graph Abstraction | Implemented | Implemented | shipped | `src/transformation_portal/spatial_ai/orchestration/` ExecutionGraph DAG | none |
| ADR-030 | Phase II Deterministic RAW Ingest | Implemented | Implemented | shipped | `src/transformation_portal/spatial_ai/ingest/` RAW decode → xyz_d50_linear_fp32 path | none |
| ADR-031 | Test Dependency Isolation Contract | Accepted | Implemented | shipped | `scripts/check_ml_test_isolation.sh`; pytest marker guards in `tests/`; pre-commit hooks | none |
| ADR-032 | Dependency Pinning Strategy | Accepted | Active | policy | `requirements/*.in` + pip-compile; `scripts/validation/check_requirements_lock_contract.py` | none |
| ADR-033 | Test Flake Management | Active | Active | policy | `flake_ledger.json`; quarantine thresholds | none |
| ADR-034 | Benchmark Exclusion from PR Gating | Accepted | Active | policy | `pyproject.toml` marker `benchmark`; nightly workflow only | none |
| ADR-035 | Bundle Root Anchoring Invariants | Proposed | Active | pending | Phase 3.4 evidence bundle code may exist (not verified end-to-end) | follow-up: verify and potentially bump Status |
| ADR-036 | Accountability Governance Invariants | Locked | Active | partial | Phase 3.6 CPPA layer; additive on frozen Phase 3.4.1 substrate | none |
| ADR-037 | Repo Root Contract | Proposed | Active | pending | Dual-anchored root discovery; verification scripts mostly present (not exhaustively verified) | follow-up: verify and bump Status |
| ADR-038 | Operational Determinism Enforcement Layer | Accepted | Active | policy | Full-chain replay diagnostic harness; CI gate | none |
| ADR-039 | Branch Staleness and Selective Integration Policy | Proposed | Active | policy | Process-only policy; followed in audit cadence | none |
| ADR-040 | Remove multipleOf Constraints for Float Fields | Accepted | Active | policy | `schemas/` updated; capture validators updated | none |
| ADR-041 | Phase 4F External Verification and Trust Export | Proposed | Active | pending | Standalone Phase 4 verifier CLI may exist under `tools/` | follow-up: verify and bump Status |
| ADR-042 | Scene Group Contract | Implemented | Implemented | shipped | `src/transformation_portal/spatial_ai/` SceneGroup model + deterministic generation | none |
| ADR-043 | Orchestrator Decomposition Strategy | Complete (Phases 2-7) | Implemented | shipped | `lux_depth_v3/config_resolver.py`, `pipeline_coordinator.py`, `execution_engine.py`, `artifact_manager.py`, `validators/` | none |
| ADR-044 | Test Marker Enforcement Policy | Implemented | Implemented | shipped | `scripts/validation/check_test_markers.py`; `scripts/validation/retrofit_test_markers.py`; `pyproject.toml` markers | none |
| ADR-045 | Monolith Decomposition Residuals — Governance Pattern | **Accepted** (bumped from Proposed 2026-05-16) | Active | pattern | Pattern used by ADR-046, ADR-047; `MONOLITH_DECOMPOSITION_TARGETS.md` tracks "Landed" extractions | Status bumped in this PR |
| ADR-046 | App Path Security Helper Extraction Contract | **Implemented** (bumped from Proposed 2026-05-16) | Implemented | shipped | `src/transformation_portal/portal/path_security.py`; `app.py` re-exports | Status bumped in this PR |
| ADR-047 | Managed SAM2 Checkpoint Security Extraction Contract | **Implemented** (bumped from Proposed 2026-05-16) | Implemented | shipped | `src/transformation_portal/portal/sam2_checkpoint_security.py`; `app.py` re-exports | Status bumped in this PR |
| ADR-048 | Materials V3 Production Integration | Approved | Active | partial | Materials V3 backends scaffolded; production wiring per ADR | renumbered 2026-05-16 from ADR-030 |

---

## Per-ADR Subsections

### ADR-001: PBR Module Integration Architecture

- **File:** `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
- **Declared Status:** Proposed (with companion `ADR-001-APPROVAL.md` recording approval)
- **Classification:** Implemented — `shipped`
- **Evidence:** PBR preset stack under `src/transformation_portal/lux_depth_v3/pbr_presets.py` and adjacent `pbr_*.py` modules; preset selectors live in `config/`. The ~420ms 4K performance target is the published reality of the production pipeline; the 44-file consolidation into `lux_depth_v3` is observable in the current src/ layout.
- **Rationale:** The decision (consolidate fragmented depth files into `lux_depth_v3`, integrate PBR normal/roughness/AO maps) has shipped. The "Proposed" Status reflects the original ADR drafting state and was never updated even after PBR became production. The companion `ADR-001-APPROVAL.md` records the actual go-ahead. Either both files should be bumped together, or this review should suffice as the public record.
- **Recommended action:** none (status-bump candidate for a future PR; this audit notes but does not auto-edit ADR-001's Status field because the ADR predates the modern status-bump convention).

### ADR-001-APPROVAL: PBR Integration Approval Record

- **File:** `docs/architecture/ADR-001-APPROVAL.md`
- **Declared Status:** Approved
- **Classification:** Active — `policy`
- **Evidence:** Approval governance for ADR-001 (zero breaking changes, CI deprecation enforcement, ≥80% test coverage requirements) is enforced by the CI test/coverage gates that are now standard practice.
- **Rationale:** This is a satellite governance record for ADR-001, not a separate decision. Treated here as a distinct entry only because the README index lists it separately.
- **Recommended action:** none.

### ADR-015: DA3 1.1 Non-Commercial Research Tier

- **File:** `docs/architecture/ADR-015-da3-1-1-non-commercial-research-tier.md` (renumbered 2026-05-16 from `adr-0015-...`)
- **Declared Status:** Adopted
- **Classification:** Active — `policy`
- **Evidence:** `@require_non_commercial` decorator (referenced in ADR-019 §License Governance); preset `config/presets/depth-anything-v3-1-research-m4.yaml` with `adr_reference: "ADR-015"`; CI license-compliance tests.
- **Rationale:** DA3 1.1 (CC BY-NC 4.0) is restricted to non-commercial research; DA3 V2 remains the commercial default. The enforcement is live policy: every preset using DA3 1.1 must declare `non_commercial_ok=True`. Renumbered in this PR for prefix-format consistency (every other ADR uses 3 digits); no content changes.
- **Recommended action:** none beyond the renumbering already applied.

### ADR-017: Parallelization Strategy for Batch Processing

- **File:** `docs/architecture/ADR-017-parallelization-strategy.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** I/O-parallelization (hashing, validation, manifest writes) is implemented in the pipeline coordinator; GPU inference remains sequential per the ADR's invariant.
- **Rationale:** The 3-5x throughput target is the operating reality of the batch pipeline on Apple Silicon M4. The ADR is live policy because new batch features must preserve "GPU inference is sequential" as an invariant.
- **Recommended action:** none.

### ADR-018: Depth Pro Integration Decision

- **File:** `docs/architecture/ADR-018-depth-pro-integration.md`
- **Declared Status:** Adopted
- **Classification:** Active — `policy`
- **Evidence:** `scripts/setup/install_depth_pro_runtime.sh` installs the isolated venv; CLAUDE.md documents `--depth-backend depth_pro` and the dual-gate (`non_commercial_ok=True` + `accept_apple_depth_pro_research_license=True`).
- **Rationale:** Depth Pro shipped as an isolated-venv optional backend with research-only licensing. The ADR remains live policy for the license-gate contract.
- **Recommended action:** none.

### ADR-019: Depth Backend Unification Architecture

- **File:** `docs/architecture/ADR-019-depth-backend-unification.md` (with satellite docs `ADR-019_FINAL_CHECKLIST.md`, `ADR-019_VERIFICATION_REPORT.md`)
- **Declared Status:** Implemented (impl date 2026-02-09, PR #906)
- **Classification:** Implemented — `shipped`
- **Evidence:** `DepthBackend` Protocol + `DepthBackendRegistry` are live; DA3 V2, DA2, Depth Pro are all unified behind the registry. Backend resolution metadata (`requested_backend`, `resolved_backend`, `resolution_status`, `resolution_reason`) is preserved in every manifest per CLAUDE.md.
- **Rationale:** The decision shipped and is preserved as a historical record. The two satellite docs (FINAL_CHECKLIST, VERIFICATION_REPORT) are evidence artifacts of the PR #906 ship and add navigation clutter at the top of `docs/architecture/`.
- **Recommended action:** Follow-up — fold satellite docs into a `docs/architecture/ADR-019-support/` subfolder, or link them from inside ADR-019 and let them remain at the same level.

### ADR-020: Drop Python 3.10 Support

- **File:** `docs/architecture/ADR-020-drop-python-3.10.md`
- **Declared Status:** Adopted
- **Classification:** Active — `policy`
- **Evidence:** `pyproject.toml` declares Python 3.11+ requirement; CI lanes pin Python 3.11 / 3.12.
- **Rationale:** Live policy: any reintroduction of 3.10 support would require a new ADR.
- **Recommended action:** none.

### ADR-021: HuggingFace Model Revision Pinning Policy

- **File:** `docs/architecture/ADR-021-huggingface-revision-policy.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** Dual-mode loader pattern (development unpinned + `# nosec B615`, production with explicit `revision=...`) used across HF model loaders in `src/transformation_portal/`.
- **Rationale:** Live policy for any HF model load; reviewers enforce it on PR.
- **Recommended action:** none.

### ADR-022: V2 Enhancement Stage Optionality

- **File:** `docs/architecture/ADR-022-v2-enhancement-optional.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** CLI flag wiring in the orchestrator entrypoint disables the V2 stage; `scripts/enhance_image.py` is no longer a hard dependency.
- **Rationale:** Live policy. CLAUDE.md states "V2 enhancement is optional; backward-compat defaults and fail-fast validation must stay intact."
- **Recommended action:** none.

### ADR-023: Spatial AI Ingest Isolation Boundary

- **File:** `docs/architecture/ADR-023-spatial-ai-ingest-isolation.md`
- **Declared Status:** Approved (Mandatory)
- **Classification:** Active — `policy`
- **Evidence:** `src/transformation_portal/spatial_ai/` (12-16 bit linear training) and `src/transformation_portal/lux_depth_v3/` (8-bit sRGB rendering) remain separate; ADR-030 codifies the deterministic ingest boundary.
- **Rationale:** Live mandatory policy: cross-contamination of RAW decode logic between training and rendering paths is forbidden.
- **Recommended action:** none.

### ADR-024: Apache Iceberg Dependency Ban

- **File:** `docs/architecture/ADR-024-apache-iceberg-ban.md`
- **Declared Status:** Approved (Mandatory Ban)
- **Classification:** Active — `policy`
- **Evidence:** `requirements/constraints.txt` and `scripts/security/verify_banned_dependencies.py` enforce the ban (the same enforcement surface that bans `realesrgan` per CLAUDE.md).
- **Rationale:** Live ban until a supply-chain audit completes. The ADR remains the binding policy artifact.
- **Recommended action:** none.

### ADR-025: APEX Research Workflow Architecture

- **File:** `docs/architecture/ADR-025-apex-research-workflow.md`
- **Declared Status:** Proposed
- **Classification:** Active — `pending`
- **Evidence:** Not verified as shipped in this review. APEX-Research tier referenced in other ADRs (e.g., ADR-026 extends it) but no concrete `apex_research_orchestrator.py`-equivalent verified end-to-end.
- **Rationale:** Multi-layer license compliance gating for research-only tools (Depth Pro AMLR, DA3 1.1 CC BY-NC, SAM, LLaVA) is in scope but its production wiring is unclear from this review.
- **Recommended action:** Follow-up — confirm with the architect whether APEX-Research is still on the active roadmap or has been quietly de-prioritized in favour of paid-pilot hardening. If de-prioritized, mark `Superseded by` paid-pilot roadmap or `Withdrawn`.

### ADR-026: APEX Research Ultra

- **File:** `docs/architecture/ADR-026-apex-research-ultra.md`
- **Declared Status:** Proposed
- **Classification:** Active — `pending`
- **Evidence:** Extends ADR-025; depends on multi-model depth ensemble, SAM2 video + physics BRDF materials, VLM validation, and geometric reconstruction (3DGS/NeRF). SAM2 and FastVLM are shipped (per CLAUDE.md) but the "Ultra" composition is not verified end-to-end.
- **Rationale:** Aspirational at proposal time; partial dependencies have shipped but the orchestrated Ultra workflow is not verified here.
- **Recommended action:** Follow-up — same as ADR-025.

### ADR-027: Phase 2 Spatial AI Extension Architecture

- **File:** `docs/architecture/ADR-027-phase2-spatial-ai-extension.md`
- **Declared Status:** Partially Implemented (updated 2026-03-23)
- **Classification:** Active — `partial`
- **Evidence:** SAM2 segmentation shipped (`src/transformation_portal/lux_depth_v3/segmentation/sam2.py`); orchestration bridge shipped; materials backends are placeholders; 3D reconstruction deferred per the ADR's own status.
- **Rationale:** The Status field already accurately reflects partial implementation. No action required.
- **Recommended action:** none.

### ADR-029: Execution Graph Abstraction for Spatial AI Orchestration

- **File:** `docs/architecture/ADR-029-execution-graph-abstraction.md`
- **Declared Status:** Implemented (updated 2026-03-23)
- **Classification:** Implemented — `shipped`
- **Evidence:** `src/transformation_portal/spatial_ai/orchestration/` ExecutionGraph DAG with deterministic caching, provenance tracking, resource budgeting. Known caveat: MaterialsStage caching disabled due to PBRTextures serialization (documented in ADR).
- **Rationale:** Decision shipped; historical record. Caveat is documented in-place.
- **Recommended action:** none.

### ADR-030: Phase II Deterministic RAW Ingest

- **File:** `docs/architecture/ADR-030-phase2-deterministic-raw-ingest.md`
- **Declared Status:** Implemented (updated 2026-02-22)
- **Classification:** Implemented — `shipped`
- **Evidence:** Deterministic RAW ingest emits canonical `xyz_d50_linear_fp32` tensors with stable geometry metadata and provenance hashes per CLAUDE.md; lives under `src/transformation_portal/spatial_ai/`.
- **Rationale:** Decision shipped. **Numbering note:** This is the canonical ADR-030. Until 2026-05-16, a second `ADR-030-materials-v3-production-integration.md` collided with this number; it has been renumbered to ADR-048.
- **Recommended action:** none.

### ADR-031: Test Dependency Isolation Contract

- **File:** `docs/architecture/ADR-031-test-dependency-isolation.md`
- **Declared Status:** Accepted
- **Classification:** Implemented — `shipped`
- **Evidence:** `scripts/check_ml_test_isolation.sh` (pre-commit + CI); module-level `HAS_ML_DEPS` guard pattern enforced across `tests/`; CLAUDE.md mandates the pattern.
- **Rationale:** Three-layer enforcement is live and CI-blocking. Decision shipped.
- **Recommended action:** none (consider bumping Status to `Implemented` in a future tidy-up PR).

### ADR-032: Dependency Pinning Strategy

- **File:** `docs/architecture/ADR-032-dependency-pinning-strategy.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** `requirements/*.in` + pip-compile producing `*.lock.txt`; `scripts/validation/check_requirements_lock_contract.py` enforces the contract; banned-package scanner.
- **Rationale:** Live policy for every dependency change.
- **Recommended action:** none.

### ADR-033: Test Flake Management

- **File:** `docs/architecture/ADR-033-test-flake-management.md`
- **Declared Status:** Active
- **Classification:** Active — `policy`
- **Evidence:** `flake_ledger.json` tracking; quarantine thresholds (<1% monitored, 3% auto-quarantine).
- **Rationale:** Operational program, live policy.
- **Recommended action:** none.

### ADR-034: Benchmark Test Exclusion from PR Gating CI

- **File:** `docs/architecture/ADR-034-benchmark-exclusion-from-pr-gating.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** `pyproject.toml` declares `benchmark` marker; PR-gating workflows exclude it; benchmark suite lives in the nightly workflow lane.
- **Rationale:** Live marker-discipline policy.
- **Recommended action:** none.

### ADR-035: Bundle Root Anchoring Invariants

- **File:** `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`
- **Declared Status:** Proposed
- **Classification:** Active — `pending`
- **Evidence:** Phase 3.4 evidence-bundle code likely exists under `tp/merkle/` and `attestation/` (per CLAUDE.md contract families), but the specific bundle-root invariant codification was not verified end-to-end in this review.
- **Rationale:** Bundle-root invariants are intentionally frozen so external timestamping and signatures can layer on top. The shipped state of the underlying Phase 3.4 evidence-bundle code suggests this ADR may already be enforced in practice.
- **Recommended action:** Follow-up — verify against `tp/merkle/` and `tp/phase4/` code and consider bumping to `Accepted` or `Implemented`.

### ADR-036: Accountability Governance Invariants

- **File:** `docs/architecture/ADR-036-accountability-governance-invariants.md`
- **Declared Status:** Locked
- **Classification:** Active — `partial`
- **Evidence:** Phase 3.6 CPPA-oriented accountability artifacts layer additively over the frozen Phase 3.4.1 integrity substrate; bundle root projection is preserved.
- **Rationale:** "Locked" means the integrity substrate it depends on is frozen; the additive accountability layer is the live deliverable. Already accurate.
- **Recommended action:** none.

### ADR-037: Repo Root Contract

- **File:** `docs/architecture/ADR-037-repo-root-contract.md`
- **Declared Status:** Proposed
- **Classification:** Active — `pending`
- **Evidence:** Dual-anchored discovery via `pyproject.toml` + `.github/workflows/` is referenced by validation scripts; the loud-failure discipline is observable in `scripts/validation/`.
- **Rationale:** The contract is largely how the repo already operates; the ADR codifies the invariant. Status-bump candidate but not verified exhaustively here.
- **Recommended action:** Follow-up — verify and bump Status.

### ADR-038: Operational Determinism Enforcement Layer

- **File:** `docs/architecture/ADR-038-operational-determinism-enforcement-layer.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** Full-chain replay diagnostic harness for Phase 4 metadata/provenance validation; CI gate referenced in governance docs.
- **Rationale:** Live governance gate.
- **Recommended action:** none.

### ADR-039: Branch Staleness and Selective Integration Policy

- **File:** `docs/architecture/ADR-039-branch-staleness-and-selective-integration-policy.md`
- **Declared Status:** Proposed
- **Classification:** Active — `policy`
- **Evidence:** Process-only policy; followed in the recent canonical worktree discipline (CLAUDE.md §"Canonical Worktree Discipline" tracks `origin/main` as the only standing source of truth).
- **Rationale:** Live process policy even though Status is Proposed; mandates branch-delta audits and selective integration after major architectural phases.
- **Recommended action:** none (status-bump candidate).

### ADR-040: Remove multipleOf Constraints for Float Fields

- **File:** `docs/architecture/ADR-040-remove-multipleof-floats-tp-meta-capture-v1.md`
- **Declared Status:** Accepted
- **Classification:** Active — `policy`
- **Evidence:** `schemas/` and capture validators updated to remove `multipleOf` from GPS, focal length, aperture, shutter, exposure float fields, avoiding IEEE-754 false negatives.
- **Rationale:** Live schema policy; preserves accuracy of capture-metadata validation.
- **Recommended action:** none.

### ADR-041: Phase 4F External Verification and Trust Export

- **File:** `docs/architecture/ADR-041-phase4f-external-verification-and-trust-export.md`
- **Declared Status:** Proposed
- **Classification:** Active — `pending`
- **Evidence:** A standalone Phase 4 verifier CLI may exist under `tools/` (CLAUDE.md mentions `tools/` is the home of governed CLIs); not verified end-to-end in this review.
- **Rationale:** Dependency-minimal external validation of capture metadata, manifest hashes, provenance entries, and Merkle root. Status-bump candidate if the CLI has shipped.
- **Recommended action:** Follow-up — verify against `tools/` and bump Status if shipped.

### ADR-042: Scene Group Contract for Multi-View Reconstruction

- **File:** `docs/architecture/ADR-042-scene-group-contract.md`
- **Declared Status:** Implemented (updated 2026-03-23)
- **Classification:** Implemented — `shipped`
- **Evidence:** SceneGroup model + deterministic generation + camera-resolution precedence + reconstruction eligibility validation, all under `src/transformation_portal/spatial_ai/`.
- **Rationale:** Decision shipped; historical record.
- **Recommended action:** none.

### ADR-043: Orchestrator Decomposition Strategy

- **File:** `docs/architecture/ADR-043-orchestrator-decomposition.md`
- **Declared Status:** Complete (Phases 2-7 Done)
- **Classification:** Implemented — `shipped`
- **Evidence:** `src/transformation_portal/lux_depth_v3/config_resolver.py`, `pipeline_coordinator.py`, `execution_engine.py`, `artifact_manager.py`, `validators/` are all present; `orchestrator.py` retains a compatibility-facing `EnhanceOrchestrator` re-export per CLAUDE.md.
- **Rationale:** Decision shipped. CLAUDE.md actively cites this ADR as the seam-discipline pattern for current edits. The orchestrator is larger than the original <1000-LOC aspiration (≈5,675 LOC at last count), but the extracted modules are the actual success metric.
- **Recommended action:** none.

### ADR-044: Test Marker Enforcement Policy

- **File:** `docs/architecture/ADR-044-test-marker-enforcement.md`
- **Declared Status:** Implemented (retrofit completed 2026-03-22)
- **Classification:** Implemented — `shipped`
- **Evidence:** `scripts/validation/check_test_markers.py`; `scripts/validation/retrofit_test_markers.py`; `pyproject.toml` registers `unit`, `security`, `regression`, `golden`, `integration`, `ml`, `slow`, `benchmark`, `stress`; CI uses positive marker selection.
- **Rationale:** Decision shipped and CI-enforced; pre-commit hook installed.
- **Recommended action:** none.

### ADR-045: Monolith Decomposition Residuals — Governance Pattern

- **File:** `docs/architecture/ADR-045-monolith-decomposition-residuals.md`
- **Declared Status:** **Accepted** (bumped from Proposed 2026-05-16 by this review)
- **Classification:** Active — `pattern`
- **Evidence:** `docs/architecture/MONOLITH_DECOMPOSITION_TARGETS.md` tracks 17 "Landed" extractions following this pattern. ADRs 046 and 047 are concrete instantiations citing this ADR as their general pattern source. CLAUDE.md cites this ADR alongside 043, 046, 047 as ADRs that materially affect routine edits.
- **Rationale:** The pattern is in active use as the governance standard for future decompositions. Leaving it as "Proposed" misrepresents its current role.
- **Recommended action:** Status bumped in this PR. No further action.

### ADR-046: App Path Security Helper Extraction Contract

- **File:** `docs/architecture/ADR-046-app-path-security-helper-extraction.md`
- **Declared Status:** **Implemented** (bumped from Proposed 2026-05-16 by this review)
- **Classification:** Implemented — `shipped`
- **Evidence:** `src/transformation_portal/portal/path_security.py` exists with the extracted helpers; `app.py` retains the re-export shims required by the compatibility-only contract; CLAUDE.md cites the file as the shipping evidence.
- **Rationale:** The compatibility-only extraction shipped. Leaving Status as "Proposed" misrepresents shipped work.
- **Recommended action:** Status bumped in this PR. No further action.

### ADR-047: Managed SAM2 Checkpoint Security Extraction Contract

- **File:** `docs/architecture/ADR-047-managed-sam2-checkpoint-security-extraction.md`
- **Declared Status:** **Implemented** (bumped from Proposed 2026-05-16 by this review)
- **Classification:** Implemented — `shipped`
- **Evidence:** `src/transformation_portal/portal/sam2_checkpoint_security.py` exists; `app.py` re-exports preserved; CLAUDE.md cites the file as the shipping evidence.
- **Rationale:** Same as ADR-046 — compatibility-only extraction shipped; Status field updated to match reality.
- **Recommended action:** Status bumped in this PR. No further action.

### ADR-048: Materials V3 Production Integration

- **File:** `docs/architecture/ADR-048-materials-v3-production-integration.md` (renumbered 2026-05-16 from `ADR-030-materials-v3-production-integration.md`)
- **Declared Status:** Approved
- **Classification:** Active — `partial`
- **Evidence:** Materials V3 backends scaffolded (`material_segmentation_backend="stub"` default, `efficientsam` available; `enable_material_segmentation=False` by default at the time of ADR drafting). Subsequent ADR-027 (Phase 2 Spatial AI Extension) covers further materials work; some pieces shipped (SAM2 segmentation), others remain placeholders.
- **Rationale:** The decision (production-ready integration path, CLI mask paths, preset examples) is approved; full production wiring may be partial. Renumbered in this PR to resolve the number collision with the canonical ADR-030.
- **Recommended action:** Follow-up — confirm production wiring status against ADR-027's "Partially Implemented" companion state and consider unifying or marking Superseded if appropriate.

---

## Structural Anomalies (resolved in this PR)

1. **Duplicate ADR-030 numbering.** Two files both held ADR-030. The Phase II Deterministic RAW Ingest file is the canonical ADR-030 (listed under "Canonical" in `README.md`). The Materials V3 file was renumbered to ADR-048 (next free slot) with an inline renumbering note. Cross-references updated in `docs/materials/`, `docs/investigations/materials_v3/`, and `docs/architecture/DUAL_REQUEST_QUICK_REF.md`. Archived doc references under `docs/_archive/`, `docs/historical/`, and `docs/pr_archive/` were intentionally left untouched per the repo's "historical evidence is preserved" policy.

2. **ADR-0015 4-digit prefix outlier.** Every other ADR in the series uses a 3-digit prefix (`ADR-NNN`). `adr-0015` also used a lowercase prefix. Renamed to `ADR-015-da3-1-1-non-commercial-research-tier.md` with renumbering notes inside the file and across cross-references in `README.md`, `docs/README.md`, `docs/governance/DOCUMENTATION_MAP.md`, `docs/depth_model/README.md`, `docs/guides/DEPTH_PIPELINE_README.md`, `config/presets/depth-anything-v3-1-research-m4.yaml`, `.github/agents/transformation-portal-architect.md`, and ADR-018 / 019 / 025.

3. **`ADR-0XX-vjepa2-...TEMPLATE.md` template-as-ADR.** This file is a template for a future ADR that may or may not be filed if the V-JEPA 2 separate-repository approach is approved. It is not a real ADR and should not occupy a slot in the numbered series. Moved to `docs/architecture/templates/ADR-vjepa2-separate-repo-TEMPLATE.md` and the `README.md` entry updated to a "template" disposition. The many other `ADR-0XX` references in PHASE_C and V_JEPA_2 docs are intentional shorthand for "ADR yet to be filed" and were left as-is.

4. **ADR-019 satellite docs.** `ADR-019_FINAL_CHECKLIST.md` and `ADR-019_VERIFICATION_REPORT.md` sit at the same level as ADRs themselves, adding clutter. Flagged as a follow-up; not relocated in this PR because moving them would touch every reference to the implementation evidence and is out of scope for a decision-currency review.

5. **Numbering gaps (002-014, 016, 028).** No action — gaps may reflect reserved-but-unfiled ADRs or ADRs that were withdrawn before commit. The numbered series is internally consistent after the fixes above.

---

## Recommended Status-Field Changes Beyond This PR

Bumping ADR Status fields was deliberately limited in this PR to cases where the shipped evidence is unambiguous. The following are *recommended for future PRs* but not auto-applied here, because they need an architect sign-off or an end-to-end code verification beyond what this review performed:

| ADR | Current Status | Recommended new Status | Reason |
|---|---|---|---|
| ADR-001 | Proposed | Implemented | PBR consolidation shipped years ago; companion ADR-001-APPROVAL records the actual go-ahead |
| ADR-031 | Accepted | Implemented | `check_ml_test_isolation.sh` is shipped + CI-blocking |
| ADR-035 | Proposed | Accepted or Implemented | Pending verification of Phase 3.4 bundle-root code |
| ADR-037 | Proposed | Accepted | Dual-anchored root discovery already in use |
| ADR-039 | Proposed | Accepted | Policy is followed in current worktree discipline |
| ADR-041 | Proposed | Accepted or Implemented | Pending verification of Phase 4F verifier CLI under `tools/` |
| ADR-025 | Proposed | Withdrawn or Superseded | If APEX-Research is off the roadmap, mark accordingly |
| ADR-026 | Proposed | Withdrawn or Superseded | Same as ADR-025 |

---

## Archival Candidates (NOT executed in this PR)

The following ADRs document decisions that have shipped and are unlikely to need future amendment. They are candidates for a future move to a `docs/architecture/implemented/` subfolder to reduce top-level clutter, but only after architect approval (some are actively cited by CLAUDE.md and the move would require updating those cites):

ADR-019, 029, 030, 031, 042, 043, 044, 046, 047, 048.

---

## Out of Scope

- No source-code changes under `src/` or `tests/`.
- No deletion of any ADR content (the V-JEPA 2 template was *moved*, not deleted).
- No edits to enforcement scripts under `scripts/validation/` or to CI workflows under `.github/workflows/`.
- No edits to ADR Status fields beyond the three rows applied in this PR (ADR-045, ADR-046, ADR-047). Other Status-bump recommendations are listed above for follow-up.
- No edits to docs under `docs/_archive/`, `docs/historical/`, `docs/pr_archive/`, or to the dated `DOCUMENTATION_STATE_AUDIT_2026-04-27.md` — those are point-in-time historical evidence and the repo policy preserves them verbatim.
- No PR creation (per session instructions, only on explicit request).
