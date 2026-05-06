# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Transformation Portal.

## Active ADRs

ADRs are **binding decisions** that define the repository's architecture, security posture, and governance. Deviations require an explicit superseding ADR with migration plan.

### Core Architecture

| ADR | Title | Status | Date | Authority |
|-----|-------|--------|------|-----------|
| [ADR-001](ADR-001-PBR-Integration-Architecture.md) | PBR Integration Architecture | Accepted | 2025 | Architect |
| [ADR-017](ADR-017-parallelization-strategy.md) | Parallelization Strategy | Accepted | 2025 | Architect |
| [ADR-018](ADR-018-depth-pro-integration.md) | Depth Pro Integration | Accepted | 2026-02 | Architect |
| [ADR-019](ADR-019-depth-backend-unification.md) | Depth Backend Unification | Proposed | 2026-02-02 | Architect |
| [ADR-023](ADR-023-spatial-ai-ingest-isolation.md) | Spatial AI Ingest Isolation | Accepted | 2026 | Architect |
| [ADR-027](ADR-027-phase2-spatial-ai-extension.md) | Phase 2 Spatial AI Extension | Proposed | 2026-02-11 | Architect |
| **[ADR-029](ADR-029-execution-graph-abstraction.md)** | **Execution Graph Abstraction** | **Proposed** | **2026-02-12** | **Architect** |
| **[ADR-030](ADR-030-phase2-deterministic-raw-ingest.md)** | **Phase II Deterministic RAW Ingest** | **Proposed** | **2026-02-20** | **Architect** |
| **[ADR-035](ADR-035-bundle-root-anchoring-invariants.md)** | **Bundle Root Anchoring Invariants** | **Proposed** | **2026-02-24** | **Architect** |
| **[ADR-038](ADR-038-operational-determinism-enforcement-layer.md)** | **Operational Determinism Enforcement Layer** | **Accepted** | **2026-02-26** | **Architect** |
| **[ADR-039](ADR-039-branch-staleness-and-selective-integration-policy.md)** | **Branch Staleness and Selective Integration Policy** | **Proposed** | **2026-02-26** | **Architect** |
| **[ADR-041](ADR-041-phase4f-external-verification-and-trust-export.md)** | **Phase 4F External Verification and Trust Export** | **Proposed** | **2026-02-27** | **Architect** |

### Dependency & Security

| ADR | Title | Status | Date | Authority |
|-----|-------|--------|------|-----------|
| [ADR-020](ADR-020-drop-python-3.10.md) | Drop Python 3.10 Support | Accepted | 2026 | Architect |
| [ADR-021](ADR-021-huggingface-revision-policy.md) | HuggingFace Revision Pinning | Accepted | 2026-02-03 | Architect |
| [ADR-024](ADR-024-apache-iceberg-ban.md) | Apache Iceberg Ban | Accepted | 2026 | Architect |
| [ADR-046](ADR-046-app-path-security-helper-extraction.md) | App Path Security Helper Extraction Contract | Proposed | 2026-05-06 | Architect |
| [ADR-047](ADR-047-managed-sam2-checkpoint-security-extraction.md) | Managed SAM2 Checkpoint Security Extraction Contract | Proposed | 2026-05-06 | Architect |

### Tier & Licensing

| ADR | Title | Status | Date | Authority |
|-----|-------|--------|------|-----------|
| [ADR-0015](adr-0015-da3-1-1-non-commercial-research-tier.md) | DA3 1.1 Non-Commercial Research Tier | Accepted | 2025 | Architect |
| [ADR-022](ADR-022-v2-enhancement-optional.md) | V2 Enhancement Optional | Accepted | 2026 | Architect |
| [ADR-025](ADR-025-apex-research-workflow.md) | APEX Research Workflow | Accepted | 2026 | Architect |
| [ADR-026](ADR-026-apex-research-ultra.md) | APEX Research Ultra | Accepted | 2026 | Architect |

---

## ADR Lifecycle

### Status Values

- **Proposed** — Under review, not yet binding
- **Accepted** — Approved and binding
- **Superseded** — Replaced by newer ADR (see "Supersedes" field)
- **Deprecated** — No longer recommended, but not yet replaced

### Creating New ADRs

1. **Check for conflicts**: Review existing ADRs to avoid duplication
2. **Use template**: Follow format of ADR-030 or ADR-029 (most recent)
3. **Required sections**:
   - Executive Summary (decision, principle, abstractions)
   - Context (current state, problem statement)
   - Decision (core abstractions, examples, consequences)
   - Alternatives Considered (why rejected)
   - Implementation Plan (phased rollout)
   - Success Metrics (measurable criteria)
   - References (related ADRs, code, docs)
4. **Enforcement**: Specify CI gates, tests, or manual review requirements
5. **Get approval**: Architect authority required for architectural decisions
6. **Update this index**: Add entry to relevant table above

### Superseding Existing ADRs

When replacing an ADR:

1. Create new ADR with `Supersedes: ADR-XXX` in header
2. Include migration plan (breaking changes, timeline, rollback)
3. Update old ADR status to "Superseded" with pointer to new ADR
4. Update this index to reflect supersession

---

## Architectural Invariants

These are repository-level rules enforced by ADRs. **Exceptions require an ADR.**

### Modularity and Coupling (ADR-023, ADR-027)

- ✅ Pipelines may share interfaces and contracts, not internal implementations
- ❌ No pipeline imports another pipeline's internal modules
- ✅ Shared utilities belong in `core/` with clear ownership

### Determinism and Reproducibility (ADR-021, ADR-027, ADR-029, ADR-030, ADR-038)

- ✅ HuggingFace models pinned to commit SHAs (not `main`)
- ✅ Same inputs → same outputs (deterministic execution)
- ✅ Provenance tracking for all ML artifacts
- ✅ Full-chain Phase 4 operational replay gate in CI

### Security and Supply Chain (ADR-021, ADR-024)

- ✅ Treat media files, filenames, metadata as hostile inputs
- ❌ No unsafe deserialization (e.g., `pickle`) without justification
- ❌ No banned dependencies (see ADR-024)
- ✅ CI enforces policy checks (pre-commit + GitHub Actions)

### Resource Management (ADR-029)

- ✅ Fail-fast on insufficient resources (no silent OOM)
- ✅ Explicit resource budgets (GPU memory, CPU memory)
- ✅ Deterministic scaling laws (memory = f(resolution, batch))

---

## Phase-Based Architecture

### Phase 1: Spatial AI Foundation (ADR-023)

**Foundation:** Linear ingest pipeline with deterministic decode

- Gamma=1.0 enforcement (no override)
- Provenance capture (content hashing)
- Strict ingest policy (8-bit rejection)

### Phase 2: Perception & Materials (ADR-027, ADR-030)

**Extension:** Segmentation, materials, 3D reconstruction

- Certified bounded determinism at ingest boundary (`xyz_d50_linear_fp32`) for cross-ISA parity (ADR-030)
- SAM2 integration (temporal-consistent segmentation)
- MaterialGAN/NVDIFFREC (PBR texture generation)
- 3D Gaussian Splatting (geometric verification)
- Tier enforcement (research licenses gated)

### Phase 3: Execution Graph Orchestration (ADR-029) ← **NEW**

**Orchestration:** DAG-based execution with caching and provenance

- Execution graph abstraction (explicit DAG)
- Content-addressed artifact store (deterministic caching)
- Automatic provenance tracking (lineage queries)
- Fail-fast resource budgeting (OOM prevention)

---

## Governance

**Authority:** Transformation Portal Architect

**Escalation:** See `docs/architecture/agent_governance.md`

**Binding Rule:** Existing ADRs are binding. Deviations require:
- Explicit superseding ADR
- Migration plan with rollback strategy
- Architect approval

**Silence is not approval.** Escalations require clear direction.

---

## Quick Links

- **Governance Policy:** [agent_governance.md](agent_governance.md)
- **CI Gate Pattern:** [ci_gate_pattern.md](ci_gate_pattern.md)
- **Phase 3 Foundation:** Session file `PHASE3_FOUNDATION.md`
- **Performance Ledger:** `/docs/performance/PERFORMANCE_LEDGER_README.md`
- **Security Policy:** `/SECURITY.md`

---

*Last Updated: 2026-02-26*
