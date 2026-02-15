# V-JEPA 2 Integration Roadmap — Architectural Assessment

**Reviewer:** Transformation Portal Architect
**Date:** 2026-02-15
**Authority:** Final architectural decision per `docs/architecture/agent_governance.md`
**Status:** **DECISION RENDERED**

---

## Executive Summary

**DECISION: NO-GO for integration into Transformation Portal repository**

The proposed V-JEPA 2 integration roadmap (10 milestones: world model training data export with token-mask schedules, motion descriptors, action conditioning, and multiprocessing safety) represents a **fundamental scope expansion** that conflicts with the repository's core mission and current architectural health.

### Primary Rationale

1. **Mission Misalignment**: V-JEPA 2 world model training is **research AI/ML infrastructure**, not luxury real estate rendering or architectural visualization tooling.

2. **Scope Creep**: The repository is already managing a complex multi-modal expansion (Spatial AI Phase I in PR #946). Adding video world model training data export creates a second, orthogonal expansion vector before the first is validated.

3. **Maintenance Burden**: The proposal adds 10 new milestones (M0-M9) on top of existing roadmap commitments. The repository does not have the organizational capacity to execute and maintain both tracks simultaneously.

4. **Opportunity Cost**: Resources allocated to V-JEPA 2 integration directly compete with completing and stabilizing the Spatial AI Foundation Phase I, which is already in-flight and aligned with repository goals.

### Recommended Path Forward

**ALTERNATIVE 1 (Strongly Recommended):** Create separate `transformation-portal-world-model` repository
- Cleanly separates research ML from production rendering
- Allows independent evolution and dependency management
- Can consume Transformation Portal outputs as inputs (decoupled integration)
- Reduces blast radius of experimental work

**ALTERNATIVE 2 (If staying in-repo):** Defer until Spatial AI Foundation Phase I is **complete and validated**
- Phase I must reach production stability first
- V-JEPA 2 integration would be evaluated as Phase III or later
- Requires separate ADR and architectural review at that time

---

## 1. Strategic Alignment Assessment

### 1.1 Core Mission Analysis

**Repository Mission (per README.md):**
> Professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production.

**Core Capabilities (current):**
- Depth-aware enhancement (monocular depth + depth-guided processing)
- PBR map generation (normal, roughness, AO)
- Material Response technology (surface-aware finishing)
- Professional grading workflows (LUT library, video grading)
- TIFF workflows (high bit-depth + metadata preservation)

**Spatial AI Foundation Phase I (in-flight, PR #946):**
- Linear training data ingest (RAW/TIFF → float32 tensors)
- Provenance capture and manifest schema
- High-fidelity data preservation for ML training
- **Rationale:** Enable future spatial intelligence research using clean, high-fidelity data from luxury real estate captures

### 1.2 V-JEPA 2 Proposal Analysis

**Proposed Capabilities:**
- Video world model training data export
- Token-mask schedules over spatiotemporal tubelets
- Motion descriptors (object-centric trajectories)
- Tiered action conditioning (Tier A inferred events, Tier B robot action/state streams)
- Multiprocessing/CUDA safety enforcement
- Durable atomic writes with directory fsync
- COCO-style RLE mask formats

**Intended Use Case:**
- Training video world models (V-JEPA 2 architecture)
- Predictive representation learning over video
- Spatiotemporal reasoning and prediction

### 1.3 Alignment Verdict: ❌ MISALIGNMENT

| Dimension | Assessment | Evidence |
|-----------|------------|----------|
| **Core Mission** | ❌ Orthogonal | World model training is ML research infrastructure, not rendering/visualization |
| **User Persona** | ❌ Different | Current: ArchViz professionals, real estate marketers. Proposed: ML researchers training foundation models |
| **Output Artifacts** | ❌ Different | Current: Enhanced images, PBR maps, graded video. Proposed: Training tuples, token masks, motion trajectories |
| **Quality Posture** | ⚠️ Conflict | Current: Production-grade, deterministic, contract-stable. Proposed: Research-grade, experimental, evolving |
| **Dependency Profile** | ⚠️ Conflict | Current: Minimal ML (depth models only). Proposed: Heavy ML (SAM 3, video processing, tokenization models) |

**Conclusion:** V-JEPA 2 integration represents a **mission pivot** from "rendering toolkit" to "ML training infrastructure." This is not incremental evolution.

---

## 2. Technical Feasibility Analysis

### 2.1 Component Mapping

**Existing Components with Clean Mapping:**

| Proposed | Existing | Mapping Quality |
|----------|----------|-----------------|
| Manifest hashing | `spatial_ai/ingest/manifest_schema.py` | ✅ Strong (deterministic hashing already implemented) |
| Provenance tracking | `spatial_ai/ingest/provenance.py` | ✅ Strong (EXIF + content hash + lineage tracking) |
| Atomic writes | `io_atomic.py` (if exists) | ⚠️ Unknown (not found in current scan) |
| Video ingest | Video grading pipelines (FFmpeg-based) | ⚠️ Weak (designed for grading, not frame extraction) |

**Proposed Components with No Existing Equivalent:**

| Component | Gap Analysis |
|-----------|--------------|
| SAM 3 perception backend | ❌ Not present. SAM 2 exists in `lux_depth_v3/segmentation_backend.py` but is rendering-focused, not training-focused (ADR-023 isolation conflict) |
| Token-mask schedule generation | ❌ Not present. Requires new spatiotemporal tokenization logic |
| Motion descriptor extraction | ❌ Not present. Requires optical flow or tracking infrastructure |
| Tier A/B action conditioning | ❌ Not present. Requires event inference and robot telemetry ingestion |
| CUDA IPC transport layer | ❌ Not present. Requires multiprocessing + GPU memory sharing |
| Stage graph for world model | ⚠️ Partial. `stage_graph/` exists but is rendering-focused |

**Verdict:** ~30% component reuse, ~70% net-new infrastructure.

### 2.2 Architectural Conflicts

**CONFLICT 1: ADR-023 Pipeline Isolation Violation**

ADR-023 mandates complete isolation between `lux_depth_v3` (rendering) and `spatial_ai` (training):
- No shared decode logic
- No cross-pipeline imports
- CI enforcement via `scripts/security/verify_pipeline_isolation.py`

**Problem:** V-JEPA 2 proposal requires SAM 3 for perception. Current SAM 2 backend lives in `lux_depth_v3/segmentation_backend.py` (rendering pipeline). Options:

1. **Duplicate SAM implementation** → Violates DRY, increases maintenance burden
2. **Create shared SAM module** → Violates ADR-023 isolation boundary
3. **Build separate SAM 3 backend** → Adds dependency complexity, requires new backend architecture

All options introduce architectural friction.

**CONFLICT 2: Video Processing Paradigm Mismatch**

Current video workflows:
- **Purpose:** Professional grading (color correction, LUT application)
- **Pipeline:** FFmpeg filter graphs, frame-by-frame processing, output to graded video
- **Optimization:** Perceptual quality, encoding efficiency

V-JEPA 2 requirements:
- **Purpose:** Training data preparation (clip extraction, tokenization, mask schedule generation)
- **Pipeline:** Frame extraction → tensor conversion → spatiotemporal chunking → disk serialization
- **Optimization:** I/O throughput, GPU utilization, deterministic hashing

**Verdict:** Different optimization targets, different quality gates, different output formats. Shared infrastructure would create coupling and compromise both use cases.

**CONFLICT 3: Dependency Governance Tension**

Current dependency posture (per ADR-024, security policy):
- Minimal ML dependencies (depth models only, pinned to commit SHAs)
- Banned dependencies enforced via `requirements/constraints.txt` hard-blocks
- Supply chain risk assessment required for new deps

V-JEPA 2 requirements (proposed):
- SAM 3 (segmentation model, likely 1-2GB checkpoint)
- Video tokenization models (unknown size, unknown license)
- Motion estimation backends (optical flow, tracking)
- Action conditioning models (Tier A/B, unknown deps)
- Multiprocessing libraries (CUDA IPC, shared memory)

**Estimated new dependencies:** 6-10 packages, 3-5GB checkpoints, multiple new license types.

**Verdict:** Significant dependency expansion that conflicts with current governance posture. Requires full supply chain audit and tier classification before approval.

### 2.3 Maintenance Burden Projection

**Current Repository Complexity:**

| Subsystem | LoC (est.) | Maintenance Level |
|-----------|------------|-------------------|
| Lux Depth V3 (rendering) | ~11,800 | High (production-grade, stable) |
| Spatial AI Phase I (ingest) | ~2,500 | Medium (in PR review, hardening) |
| PBR generation | ~3,000 | Medium (recently stabilized) |
| Video grading | ~4,000 | Medium (FFmpeg wrappers) |
| **Total** | **~21,300** | **- ** |

**V-JEPA 2 Proposal (10 milestones):**

| Milestone | Est. LoC | Complexity |
|-----------|----------|------------|
| M0: Contract lockdown | 500 | Medium (schema design) |
| M1: Durable writes | 300 | Low (filesystem utilities) |
| M2: Deterministic hashing | 400 | Medium (hash chain logic) |
| M3: Video ingest | 800 | High (clip extraction, frame indexing) |
| M4: SAM 3 perception | 1,200 | High (model integration, masklet tracking) |
| M5: Tokenization | 1,000 | High (spatiotemporal tokenization, mask schedules) |
| M6: Motion summaries | 900 | High (optical flow, trajectory extraction) |
| M7: Tier B streams | 700 | Medium (telemetry ingestion, schema) |
| M8: Transport layer | 1,200 | High (CUDA IPC, multiprocessing safety) |
| M9: CLI integration | 600 | Medium (command wrappers, docs) |
| **Total** | **~7,600** | **- ** |

**Projected Repository Growth:** +36% LoC (21,300 → 28,900)

**Maintenance Implications:**
- **Testing burden:** +50-70 tests (unit + integration + determinism checks)
- **CI time increase:** +10-15 minutes (video processing tests are slow)
- **Documentation debt:** +5-7 new docs (architecture, user guides, troubleshooting)
- **Dependency audit frequency:** Quarterly → monthly (more deps = more CVEs)

**Verdict:** Substantial increase in complexity and maintenance surface area.

---

## 3. Risk Assessment

### 3.1 Highest Risks

**RISK 1: Scope Creep Death Spiral (CRITICAL)**

**Probability:** High
**Impact:** Catastrophic

**Scenario:**
1. V-JEPA 2 integration begins as "just 10 milestones"
2. Milestone 4 (SAM 3) reveals integration complexity → adds 3 sub-milestones
3. Milestone 8 (transport layer) uncovers CUDA IPC edge cases → adds debugging tools, fallback modes
4. User requests "just one more feature" (e.g., optical flow backend selection) → adds backend registry, protocol abstraction
5. Original 10 milestones balloon to 20+ over 6 months
6. Spatial AI Phase I stalls due to resource contention
7. Neither track reaches production stability

**Mitigation:**
- Rigid scope freeze after milestone definition (no "just one more" additions)
- Time-box each milestone (2 weeks max)
- **Better mitigation:** Don't start (NO-GO decision)

**RISK 2: ADR-023 Isolation Compromise (HIGH)**

**Probability:** Medium
**Impact:** High

**Scenario:**
1. Developer needs SAM functionality for both rendering and world model training
2. "Temporarily" shares SAM backend code to avoid duplication
3. Isolation CI check bypassed via `# type: ignore` or conditional imports
4. Cross-contamination occurs: rendering gets training-focused config, training gets tone-mapped inputs
5. Silent quality degradation in both pipelines
6. Rollback requires untangling months of coupled commits

**Mitigation:**
- Strict CI enforcement (already exists)
- Separate SAM 3 backend in `spatial_ai/` (duplication accepted)
- **Better mitigation:** Separate repository (no shared code surface)

**RISK 3: Dependency Supply Chain Exposure (HIGH)**

**Probability:** High
**Impact:** Medium-High

**Scenario:**
1. V-JEPA 2 integration adds 8 new dependencies (SAM 3, tokenization models, etc.)
2. One dependency (`video-tokenizer-lib`) becomes unmaintained, CVE discovered
3. No alternative exists, migration cost is high
4. Repository stuck on vulnerable dependency
5. Security audit blocks production deployment

**Mitigation:**
- Full supply chain audit before adding each dependency (per ADR-024 pattern)
- Pin all deps to commit SHAs (per existing HuggingFace revision policy)
- Maintain vendor forks for critical deps
- **Better mitigation:** Limit dependency surface (separate repo reduces main repo exposure)

**RISK 4: Testing Infrastructure Collapse (MEDIUM)**

**Probability:** Medium
**Impact:** Medium

**Scenario:**
1. Video processing tests require large fixture files (100MB+ video clips)
2. CI test suite time increases from 8 minutes to 25 minutes
3. Developers start skipping CI locally ("it takes too long")
4. Regressions slip through due to incomplete local testing
5. CI becomes the only quality gate, slowing PR velocity

**Mitigation:**
- Synthetic fixtures only (generated on-the-fly, deterministic)
- Separate `pytest -m world_model` marker (opt-in for full video tests)
- Pre-commit runs fast subset, CI runs full suite
- **Better mitigation:** Separate repo allows independent CI optimization

### 3.2 Long-Term Maintenance Implications

**Projected 2-Year Costs:**

| Cost Category | Current (rendering-focused) | With V-JEPA 2 |
|---------------|----------------------------|---------------|
| **Active Maintainers Required** | 1-2 (part-time) | 2-3 (full-time) |
| **Monthly Dependency CVE Scans** | 20-30 packages | 40-50 packages |
| **Quarterly Security Audits** | 4 hours | 10 hours |
| **New Contributor Onboarding Time** | 2-3 days | 5-7 days (complexity barrier) |
| **Breaking Change Migration Cost** | Low (stable contracts) | High (evolving ML ecosystem) |

**Organizational Capacity Reality Check:**

Current evidence suggests **single-maintainer operation** (based on commit patterns, PR review cadence). V-JEPA 2 integration pushes complexity beyond single-maintainer sustainable threshold.

**Verdict:** Repository will accumulate technical debt faster than it can be retired.

### 3.3 Compliance and Licensing Risks

**Current License Posture:**
- Depth Anything V3 (commercial variant): Commercial-friendly
- DA3 1.1: CC BY-NC 4.0 (research only, tier-restricted)
- Depth Pro: Apple AMLR (research only, tier-restricted)
- Clear tier boundaries enforced in code

**V-JEPA 2 License Unknowns:**
- SAM 3: License TBD (SAM 2 is Apache 2.0, but SAM 3 may differ)
- Video tokenization models: Unknown (depends on chosen implementation)
- Action conditioning models: Unknown
- Motion estimation backends: Varies (some GPL, some BSD, some proprietary)

**AB 2013 / EU AI Act Considerations:**

Current use case (luxury real estate rendering):
- **Risk Level:** Low (creative tooling, not automated decision-making)
- **Compliance Burden:** Minimal

V-JEPA 2 use case (world model training):
- **Risk Level:** Medium-High (depends on downstream model use)
- **Compliance Burden:** TBD (training data provenance tracking already exists, but model deployment constraints unknown)

**Verdict:** Adding world model training infrastructure introduces regulatory ambiguity that requires legal review.

---

## 4. Prioritization Recommendation

### 4.1 GO / NO-GO Decision

**DECISION: NO-GO**

**Binding Rationale:**

1. **Mission misalignment:** V-JEPA 2 world model training is not core to luxury real estate rendering or architectural visualization.

2. **Organizational capacity:** Single-maintainer operation cannot sustain two parallel expansion tracks (Spatial AI Phase I + V-JEPA 2).

3. **Architectural health:** Repository is mid-transition (PR #946 in review). Adding second major expansion jeopardizes stability of both efforts.

4. **Opportunity cost:** V-JEPA 2 resources directly compete with completing and validating Spatial AI Phase I, which is already aligned and in-flight.

5. **Maintenance burden:** Projected +36% LoC growth, +60% dependency surface area, +2-3x testing complexity is unsustainable.

### 4.2 Recommended Alternatives

**ALTERNATIVE 1: Separate Repository (STRONGLY RECOMMENDED)**

**Proposal:** Create `transformation-portal-world-model` as sibling repository.

**Structure:**
```
transformation-portal/           # Existing (rendering + spatial data prep)
transformation-portal-world-model/  # New (V-JEPA 2 training data export)
```

**Integration Pattern:**
- `transformation-portal` exports clean spatial data (linear tensors, provenance manifests)
- `transformation-portal-world-model` consumes those exports as inputs
- Clean separation of concerns, independent evolution, minimal coupling

**Benefits:**
✅ **Mission clarity:** Each repo has single, focused purpose
✅ **Dependency isolation:** Heavy ML deps stay in world-model repo
✅ **Independent versioning:** Breaking changes don't cascade
✅ **Reduced blast radius:** Experimental work doesn't destabilize production tooling
✅ **Organizational scalability:** Can recruit world-model specialist maintainers separately

**Costs:**
⚠️ **Coordination overhead:** Must define stable export format (one-time cost)
⚠️ **Duplication:** Some utilities duplicated (manifest parsing, hashing) - ~200 LoC

**Architect Assessment:** Benefits far outweigh costs. This is the correct architecture.

**ALTERNATIVE 2: Defer to Phase III (After Phase I Validated)**

**Proposal:** Complete Spatial AI Foundation Phase I, validate in production, then re-evaluate V-JEPA 2 as Phase III.

**Phasing:**
1. **Phase I** (current, in PR #946): Linear ingest, provenance, manifests
2. **Phase II** (future): SAM 2 segmentation, MaterialGAN, 3DGS reconstruction (per ADR-027)
3. **Phase III** (speculative): V-JEPA 2 world model training data export (if validated need exists)

**Gates for Phase III Consideration:**
- Phase I reaches production stability (6+ months clean operation)
- Organizational capacity increases (2+ full-time maintainers)
- User demand validation (5+ users requesting world model training capabilities)
- Dependency ecosystem stabilizes (SAM 3, tokenization models reach 1.0 releases)

**Timeline:** Phase III earliest consideration date: Q3 2026 (6 months from now)

**Benefits:**
✅ **Risk mitigation:** Validate Phase I before expanding further
✅ **Resource focus:** Complete one thing well before starting next
✅ **User validation:** Ensure demand exists before building

**Costs:**
⚠️ **Delayed capability:** V-JEPA 2 not available for 6-12 months
⚠️ **Potential rework:** If architecture changes in Phase I/II, V-JEPA 2 design may need revision

**Architect Assessment:** Prudent fallback if separate repository is rejected. Still requires organizational capacity increase.

**ALTERNATIVE 3: Minimal Viable Integration (CONDITIONAL APPROVAL)**

**THIS ALTERNATIVE IS NOT RECOMMENDED.** Included for completeness only.

If organizational leadership insists on in-repo integration despite architectural concerns, the **minimum viable scope** would be:

**Approved Milestones (3 only):**
- M0: Contract lockdown (schemas only, no code)
- M1: Durable writes (filesystem utilities, reusable)
- M2: Deterministic hashing (extends existing manifest system)

**Deferred Milestones (7, blocked until Phase I complete):**
- M3-M9: All video-specific work deferred

**Conditions for Approval:**
1. ✅ Architect pre-approval required for every PR (per governance policy)
2. ✅ ADR-023 isolation compliance enforced (no exceptions)
3. ✅ Zero new ML dependencies in M0-M2
4. ✅ Test coverage ≥85% before merge
5. ✅ Documentation complete (architecture + user guide) before merge

**Timeline:** 2-3 weeks for M0-M2 (foundational only)

**Re-evaluation Gate:** After Spatial AI Phase I reaches production (6+ months), re-assess M3-M9 viability.

**Architect Assessment:** Compromise position that preserves core architectural health while allowing limited progress. Still not optimal.

---

## 5. Concrete Next Steps

### 5.1 If NO-GO Decision Accepted (Recommended)

**Action Plan:**

**Week 1: Communicate Decision**
1. Document this assessment and share with stakeholders
2. Explain rationale: mission misalignment, capacity constraints, maintenance burden
3. Recommend Alternative 1 (separate repository) as path forward

**Week 2-4: Support Separate Repository Creation (if desired)**
1. Define clean export format from `transformation-portal` (manifest schema, tensor layout)
2. Create `transformation-portal-world-model` repository template
3. Document integration contract (how world-model repo consumes spatial AI exports)
4. Transfer V-JEPA 2 roadmap to new repo

**Ongoing: Focus on Spatial AI Phase I**
1. Complete PR #946 review and merge
2. Validate linear ingest pipeline in production
3. Build Phase II capabilities (SAM 2, MaterialGAN, 3DGS) per ADR-027

**No new work required in Transformation Portal repository for V-JEPA 2.**

### 5.2 If Alternative 2 Chosen (Defer to Phase III)

**Action Plan:**

**Week 1: Update Roadmap**
1. Move V-JEPA 2 milestones to `docs/spatial_ai/ROADMAP_PHASE_III.md` (speculative)
2. Mark as "Future Consideration, Pending Phase I/II Validation"
3. Document re-evaluation gates and timeline

**Week 2-26: Focus on Phase I/II Execution**
1. Complete Spatial AI Phase I (PR #946 + hardening)
2. Execute Phase II milestones (SAM 2, MaterialGAN, 3DGS) per ADR-027
3. Validate user demand for world model training capabilities

**Month 6: Phase III Re-Evaluation**
1. Assess organizational capacity (maintainer count, velocity)
2. Validate user demand (survey, feature requests)
3. Re-run architectural assessment with updated context
4. Make binding GO/NO-GO decision at that time

**No immediate work required for V-JEPA 2.**

### 5.3 If Alternative 3 Chosen (Minimal Viable, Not Recommended)

**Action Plan (Conditional):**

**PR #1: M0 Contract Lockdown (Week 1-2)**
- Create schemas only: `VideoInputMetadata`, `TokenizationSpec`, `TokenMaskSchedule`
- Location: `src/transformation_portal/spatial_ai/world_model/contracts.py`
- No executable code, pure data contracts
- ADR required: `ADR-0XX-world-model-contracts.md`
- Architect pre-approval: Mandatory before PR creation

**PR #2: M1 Durable Writes (Week 3)**
- Extend `io_atomic.py` (if exists) or create new `atomic_write.py`
- Add directory fsync for durability guarantees
- Reusable utility, not world-model-specific
- Test coverage ≥90% (filesystem failure modes)
- Architect pre-approval: Mandatory

**PR #3: M2 Deterministic Hashing (Week 4)**
- Extend `spatial_ai/ingest/manifest_schema.py` with semantic manifest hash
- Add reproducibility harness (hash chain verification)
- Integrates cleanly with existing provenance system
- Test coverage ≥85%
- Architect pre-approval: Mandatory

**STOP POINT:** M3-M9 blocked until Spatial AI Phase I reaches production (6+ months).

**Re-Evaluation (Month 6):**
- Assess Phase I stability
- Re-run risk assessment for M3-M9
- Make binding decision on continuation vs abandonment

---

## 6. Draft ADR (If Separate Repository Approved)

### ADR-0XX: V-JEPA 2 World Model Integration via Separate Repository

**Status:** Proposed
**Date:** 2026-02-15
**Authority:** Transformation Portal Architect

---

**Context:**

V-JEPA 2 world model training data export capabilities were proposed for integration into the Transformation Portal repository. Architectural assessment identified mission misalignment, capacity constraints, and maintenance burden as critical blockers.

**Decision:**

V-JEPA 2 integration will proceed via **separate repository** (`transformation-portal-world-model`), consuming Transformation Portal's spatial AI exports as inputs.

**Rationale:**

1. **Mission Clarity:** Transformation Portal remains focused on luxury real estate rendering and spatial data preparation. World model training is orthogonal concern.

2. **Dependency Isolation:** Heavy ML dependencies (SAM 3, tokenization models, action conditioning) stay isolated in world-model repo, not polluting rendering toolkit.

3. **Independent Evolution:** Breaking changes in world model research don't destabilize production rendering workflows.

4. **Organizational Scalability:** World model repo can recruit specialist maintainers independently.

**Integration Contract:**

Transformation Portal exports:
- Linear tensors (float32, gamma=1.0)
- Provenance manifests (content hash, EXIF, lineage)
- Spatial catalog (spatiotemporal index)

World Model repo consumes:
- Manifest-driven data loading
- No direct filesystem coupling
- Versioned schema for forward compatibility

**Consequences:**

✅ Clean separation of concerns
✅ Reduced coupling and blast radius
✅ Independent versioning and release cycles
⚠️ ~200 LoC duplication (manifest parsing, hashing utilities)
⚠️ Coordination overhead for export format changes (mitigated by versioned schema)

**Enforcement:**

- Export format defined in `docs/spatial_ai/EXPORT_CONTRACT.md`
- Breaking changes require major version bump
- World model repo CI validates against pinned export schema version

**Approval:**

- Architect: ✅ Approved
- Implementation: Create `transformation-portal-world-model` repository template
- Timeline: 2-4 weeks for export contract definition and repo creation

---

## 7. Final Architect Position

**As the senior technical authority for the Transformation Portal repository, I render the following binding decision:**

### Verdict: NO-GO for In-Repository Integration

V-JEPA 2 world model training data export capabilities **shall not be integrated** into the Transformation Portal repository at this time.

### Recommended Path

Create **separate repository** (`transformation-portal-world-model`) consuming Transformation Portal's spatial AI exports as inputs.

### Justification

1. **Mission Alignment:** V-JEPA 2 world model training is research ML infrastructure, not core to luxury real estate rendering or architectural visualization.

2. **Organizational Capacity:** Single-maintainer operation cannot sustain two parallel major expansions (Spatial AI Phase I + V-JEPA 2).

3. **Architectural Health:** Repository is mid-transition with PR #946 in review. Adding second expansion jeopardizes both efforts.

4. **Maintenance Sustainability:** Projected +36% LoC, +60% dependency surface, +2-3x testing complexity exceeds sustainable threshold.

5. **Risk Mitigation:** Separate repository reduces coupling, isolates dependencies, and limits blast radius of experimental work.

### Implementation Authority

This decision is **binding** per `docs/architecture/agent_governance.md`. Deviations require:
- Explicit superseding ADR
- Architectural review with updated context
- Clear migration plan
- Demonstrated organizational capacity increase (2+ full-time maintainers)

### Next Steps

1. **Communicate decision** to stakeholders (Week 1)
2. **Support separate repository creation** if desired (Week 2-4)
3. **Focus on Spatial AI Phase I completion** (ongoing priority)
4. **Re-evaluate in 6 months** if organizational capacity increases

### Exceptions

None. This decision is firm.

---

**Approval:**
Transformation Portal Architect
Date: 2026-02-15

**Review Date:** 2026-08-15 (6 months)
**Supersedes:** None
**Related:** ADR-023 (Pipeline Isolation), ADR-027 (Spatial AI Phase II), PR #946 (Phase I Implementation)

---

**END OF ASSESSMENT**
