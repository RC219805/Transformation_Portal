# V-JEPA 2 Integration — Quick Reference

**Decision:** NO-GO (create separate repository instead)
**Date:** 2026-02-15
**Authority:** Transformation Portal Architect

---

## TL;DR

| Question | Answer |
|----------|--------|
| **Can we integrate V-JEPA 2?** | No, not in this repository |
| **Why not?** | Mission misalignment + capacity constraints + architectural health |
| **What instead?** | Create `transformation-portal-world-model` separate repository |
| **When revisit?** | August 2026 (6 months), if organizational capacity increases |
| **Is this negotiable?** | No, binding decision per governance policy |

---

## Decision Matrix

| Factor | Current State | V-JEPA 2 Requirement | Verdict |
|--------|---------------|----------------------|---------|
| **Mission** | Rendering toolkit | ML training infrastructure | ❌ Orthogonal |
| **Users** | ArchViz professionals | ML researchers | ❌ Different |
| **LoC** | 21,300 | +7,600 (+36%) | ⚠️ Unsustainable |
| **Dependencies** | 20-30 packages | +20 packages (+67%) | ⚠️ Exceeds threshold |
| **Maintainers** | 1 (part-time) | Needs 2-3 (full-time) | ❌ Insufficient |
| **Phase I Status** | In PR review | N/A | ⚠️ Not validated |

---

## Recommended Path

```
┌─────────────────────────────────┐
│  transformation-portal          │  ← Main Repo (Current)
│  ├─ Rendering (Lux Depth V3)    │
│  ├─ PBR Generation              │
│  └─ Spatial AI Foundation       │
│     └─ Exports: Tensors +       │
│        Manifests + Catalog      │
└────────────┬────────────────────┘
             │ Export Contract
             ↓
┌─────────────────────────────────┐
│  transformation-portal-         │  ← New Repo (Recommended)
│  world-model                    │
│  ├─ V-JEPA 2 Integration        │
│  ├─ Token-Mask Schedules        │
│  ├─ Motion Descriptors          │
│  └─ Action Conditioning         │
└─────────────────────────────────┘
```

**Benefits:**
- Clean mission boundaries
- Independent evolution
- Isolated dependencies
- Reduced blast radius

---

## Timeline

| Week | Action |
|------|--------|
| **1** | Communicate NO-GO decision to stakeholders |
| **2-4** | (Optional) Support separate repo creation |
| **Ongoing** | Focus on Spatial AI Phase I completion |
| **Month 6** | Re-evaluate if capacity increases |

---

## What's Blocked

**All 10 V-JEPA 2 milestones** are blocked in Transformation Portal:
1. ❌ M0: Contract lockdown
2. ❌ M1: Durable writes
3. ❌ M2: Deterministic hashing
4. ❌ M3: Video ingest
5. ❌ M4: SAM 3 perception
6. ❌ M5: Tokenization
7. ❌ M6: Motion summaries
8. ❌ M7: Tier B streams
9. ❌ M8: Transport layer
10. ❌ M9: CLI integration

**Unblocked in separate repository:** All milestones can proceed independently

---

## Escalation Path

**This decision is binding.** Exceptions require:

1. ✅ Explicit superseding ADR
2. ✅ Full architectural review with updated context
3. ✅ Demonstrated organizational capacity increase (2+ maintainers)
4. ✅ Clear migration plan and risk mitigation

**Contact:** Transformation Portal Architect
**Via:** `docs/architecture/agent_governance.md` escalation protocol

---

## Key Metrics

| Metric | Current | With V-JEPA 2 In-Repo | With Separate Repo |
|--------|---------|----------------------|-------------------|
| **LoC** | 21,300 | 28,900 (+36%) | 21,300 (stable) |
| **Dependencies** | 20-30 | 40-50 (+67%) | 20-30 (stable) |
| **CI Time** | 8 min | 18-23 min (+125%) | 8 min (stable) |
| **Maintainers Needed** | 1-2 | 2-3 | 1-2 (main) + 1 (world-model) |
| **Coupling Risk** | Low | High | Low |

---

## References

- **Full Assessment:** `docs/architecture/V_JEPA_2_INTEGRATION_ASSESSMENT.md` (27KB, comprehensive)
- **Executive Summary:** `docs/architecture/V_JEPA_2_DECISION_SUMMARY.md` (5KB, detailed)
- **This Card:** `docs/architecture/V_JEPA_2_QUICKREF.md` (2KB, at-a-glance)

---

**Decision Authority:**
Transformation Portal Architect
2026-02-15

**Review Date:** 2026-08-15
