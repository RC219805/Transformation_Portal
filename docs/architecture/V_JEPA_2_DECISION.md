# V-JEPA 2 Integration Decision — Executive Summary

**Date:** 2026-02-15
**Reviewer:** Transformation Portal Architect
**Decision:** **NO-GO**

---

## The Question

Should we integrate V-JEPA 2 video world model training data export capabilities (10 milestones) into the Transformation Portal repository?

## The Answer

**No.** Create a separate `transformation-portal-world-model` repository instead.

---

## Why NO-GO?

### 1. Mission Misalignment
- **Current Mission:** Luxury real estate rendering + architectural visualization
- **V-JEPA 2 Mission:** ML research infrastructure for training video world models
- **Assessment:** Orthogonal concerns, different user personas, incompatible goals

### 2. Organizational Capacity
- **Current State:** Single-maintainer operation, Spatial AI Phase I in PR review
- **V-JEPA 2 Demand:** 10 milestones, +36% LoC, +60% dependencies, 2-3x testing complexity
- **Assessment:** Exceeds sustainable threshold, will accumulate debt faster than retirement

### 3. Architectural Health
- **Current State:** Mid-transition (PR #946 hardening spatial AI foundation)
- **Risk:** Adding second major expansion jeopardizes stability of both efforts
- **Assessment:** Complete Phase I before considering Phase III expansions

### 4. Maintenance Burden
- **Projected Growth:** 21,300 → 28,900 LoC (+36%)
- **Dependency Growth:** 20-30 → 40-50 packages (+67%)
- **Testing Growth:** +50-70 tests, +10-15 min CI time
- **Assessment:** Unsustainable for single-maintainer operation

### 5. Opportunity Cost
- **Resource Competition:** V-JEPA 2 work directly competes with Phase I completion
- **Validation Gap:** Phase I not yet production-validated, adding Phase III is premature
- **Assessment:** Finish what you started before starting new work

---

## What Should Happen Instead?

### Recommended: Separate Repository

**Create:** `transformation-portal-world-model` (sibling repo)

**Integration Pattern:**
```
transformation-portal/
├── Exports: Linear tensors, provenance manifests, spatial catalog
└── Mission: Rendering + spatial data preparation

transformation-portal-world-model/
├── Imports: Transformation Portal exports
└── Mission: V-JEPA 2 training data export
```

**Benefits:**
- ✅ Clean mission separation
- ✅ Independent dependency management
- ✅ Reduced coupling and blast radius
- ✅ Independent versioning and evolution
- ✅ Can recruit specialist maintainers

**Costs:**
- ⚠️ ~200 LoC duplication (manifest parsing, hashing)
- ⚠️ Coordination overhead for export format changes

**Verdict:** Benefits far outweigh costs. This is the correct architecture.

---

## Alternative: Defer to Phase III (Fallback)

If separate repository is rejected:

**Timeline:** Re-evaluate in Q3 2026 (6 months)

**Gates:**
- Phase I reaches production stability (6+ months clean operation)
- Organizational capacity increases (2+ full-time maintainers)
- User demand validated (5+ users requesting capability)
- Dependency ecosystem stabilizes

**Action Until Then:** Focus on Phase I completion, build Phase II (SAM 2, MaterialGAN, 3DGS)

---

## What's Blocked?

**All 10 V-JEPA 2 milestones:**
- M0: Contract lockdown
- M1: Durable writes
- M2: Deterministic hashing
- M3: Video ingest
- M4: SAM 3 perception
- M5: Tokenization
- M6: Motion summaries
- M7: Tier B streams
- M8: Transport layer
- M9: CLI integration

**Reason:** Mission misalignment + capacity constraints + architectural health

---

## What's Next?

### Week 1: Communicate Decision
- Share this assessment with stakeholders
- Explain rationale clearly
- Recommend separate repository path

### Week 2-4: Support Separate Repo (If Desired)
- Define export contract (manifest schema, tensor layout)
- Create `transformation-portal-world-model` template
- Document integration pattern

### Ongoing: Focus on Phase I
- Complete PR #946 review and merge
- Validate linear ingest in production
- Build Phase II capabilities per ADR-027

**No V-JEPA 2 work in Transformation Portal repository.**

---

## Authority and Exceptions

**Authority:** This decision is **binding** per `docs/architecture/agent_governance.md`.

**Deviations Require:**
- Explicit superseding ADR
- Architectural review with updated context
- Clear migration plan
- Demonstrated organizational capacity increase (2+ full-time maintainers)

**Exceptions:** None. This decision is firm.

**Review Date:** 2026-08-15 (6 months from now)

---

## Key Takeaway

**V-JEPA 2 is valuable research, but it belongs in a separate repository.**

The Transformation Portal should remain focused on its core mission: professional luxury real estate rendering and architectural visualization, with clean spatial data preparation as a supporting capability.

World model training infrastructure is a different product with different users, different dependencies, and different evolution patterns. Mixing them creates coupling, complexity, and maintenance burden that exceeds sustainable capacity.

**Separate repositories. Clean boundaries. Independent evolution.**

---

## References

- **Full Assessment:** `docs/architecture/V_JEPA_2_INTEGRATION_ASSESSMENT.md`
- **Governance Policy:** `docs/architecture/agent_governance.md`
- **Related ADRs:** ADR-023 (Pipeline Isolation), ADR-027 (Spatial AI Phase II)
- **Related PRs:** PR #946 (Spatial AI Phase I)

---

**Architect Signature:**
Transformation Portal Architect
Date: 2026-02-15
