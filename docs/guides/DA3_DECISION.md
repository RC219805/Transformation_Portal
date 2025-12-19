# DA3 Decision Document - Adopt/Defer/Reject

**Date**: 2025-12-19  
**Decision Status**: 🚧 **PENDING A/B TEST RESULTS**  
**Baseline**: v1.0-validation-baseline (DA2-Large-hf, 84.8% pass)  
**Candidate**: DA3-Large-1.1  
**Decision Authority**: Transformation Portal Architect

---

## Decision Framework

This document provides an **explicit, data-driven recommendation** on whether to:

1. ✅ **ADOPT** DA3 as the production depth model
2. ⚠️ **DEFER** DA3 integration pending improvements
3. ❌ **REJECT** DA3 and maintain DA2 baseline

**Decision Criteria** (Non-negotiable thresholds):
- **Structure Performance**: ≥60% pass rate (vs 25% DA2)
- **Overall Quality**: ≥95% lenient pass (vs 84.8% DA2)
- **Texture Regression**: ≤2% degradation (maintain ≥95.4%)

**No rationalization permitted**: If thresholds are not met, DA3 is rejected.

---

## Decision Matrix

### Scenario A: Full Adoption (✅ ADOPT)

**Conditions**:
- ✅ Structure scenes: ≥60% pass (5/8 images)
- ✅ Overall lenient: ≥95% (44/46 images)
- ✅ Texture scenes: ≥95.4% (36/38 images, ≤1 regression)

**Recommendation**: **ADOPT DA3-Large-1.1 as default production model**

**Implementation**:
1. Update `lux_depth_v2/config.py` default to DA3-Large-1.1
2. Update `lux_depth_v3/` to production status
3. Archive `depth_tools.py` (DA2) as legacy
4. Update CI/CD to use DA3 for validation
5. Document migration guide for users

**Timeline**: 2 weeks (regression testing, documentation, deployment)

**Risk**: 🟢 LOW - Validation proves superior performance

---

### Scenario B: Conditional Adoption (⚠️ DEFER)

**Conditions**:
- ⚠️ Structure scenes: 45-59% pass (improvement but below threshold)
- ✅ Overall lenient: ≥95%
- ✅ Texture scenes: ≥95.4%

**Recommendation**: **DEFER - DA3 shows promise but needs refinement**

**Deferral Actions**:
1. Investigate structure scene failures (edge detection tuning?)
2. Test alternative DA3 variants (METRIC_LARGE, INDOOR)
3. Explore input size optimization (518 → 768px)
4. Re-run validation after adjustments
5. Set 30-day deadline for threshold achievement

**Fallback**: If 30-day deadline missed, proceed to Scenario C (reject)

**Risk**: 🟡 MEDIUM - Delays production improvements, uncertainty remains

---

### Scenario C: Rejection (❌ REJECT)

**Conditions**:
- ❌ Structure scenes: <45% pass (insufficient improvement)
- OR ❌ Overall lenient: <95%
- OR ❌ Texture regression: >2% (unacceptable degradation)

**Recommendation**: **REJECT DA3 - Maintain DA2 baseline**

**Rejection Rationale**:
- DA3 fails to justify 40 dev hours investment
- No measurable improvement on critical structure scenes
- Risk of texture regression unacceptable
- License complexity (CC-BY-NC) not justified by performance

**Alternative Path**:
1. Maintain DA2-Large-hf as production model
2. Explore Depth Anything V4 (when released)
3. Investigate alternative depth models (MiDaS, DPT, ZoeDepth)
4. Focus on post-processing improvements (Materials V3, edge refinement)

**Archive Actions**:
- Move `lux_depth_v3/` to `archive/da3_integration_abandoned/`
- Document rejection in `docs/decisions/ADR-001-DA3-REJECTION.md`
- Update roadmap to reflect DA2 continuation

**Risk**: 🟢 LOW - Baseline performance already validated

---

## Current Status: PENDING VALIDATION

**A/B Test Status**: 🚧 **IN PROGRESS**

**Results Summary** (will be updated):

| Criterion | Threshold | DA2 Baseline | DA3 Result | Status |
|-----------|-----------|--------------|------------|--------|
| Structure Pass Rate | ≥60% | 25.0% (2/8) | **TBD** | 🚧 |
| Overall Lenient Pass | ≥95% | 84.8% (39/46) | **TBD** | 🚧 |
| Texture Regression | ≤2% | 97.4% (37/38) | **TBD** | 🚧 |

**Decision**: **PENDING** - Awaiting validation completion

---

## Decision Tree

```
START: Run DA3 validation against baseline
│
├─> Structure pass ≥60%? ──NO──> ❌ REJECT
│   └─> YES
│       │
│       ├─> Overall pass ≥95%? ──NO──> ❌ REJECT
│       │   └─> YES
│       │       │
│       │       └─> Texture regression ≤2%? ──NO──> ❌ REJECT
│       │           └─> YES
│       │               │
│       │               └─> ✅ ADOPT
│       │
│       └─> Structure pass 45-59%? ──YES──> ⚠️ DEFER (conditional)
│           └─> NO ──> ❌ REJECT
```

---

## Supporting Evidence (TBD)

### Quantitative Metrics

**Structure Scene Improvement**:
- DA2 pass rate: 25.0% (2/8)
- DA3 pass rate: **TBD**
- Improvement: **TBD** percentage points

**Edge F1 Distribution**:
- DA2 median: 0.327
- DA3 median: **TBD**
- Change: **TBD**

**Statistical Significance**:
- McNemar's test p-value: **TBD**
- Effect size (Cohen's h): **TBD**

### Qualitative Assessment

**Strengths** (observed during validation):
- **TBD**

**Weaknesses**:
- **TBD**

**Unexpected Findings**:
- **TBD**

---

## Stakeholder Impact Analysis

### Impact of ADOPT Decision

**Positive Impacts**:
- ✅ Improved structure scene quality → better architectural renders
- ✅ Advanced capabilities (metric depth, multi-view) unlock new workflows
- ✅ Future-proofed architecture for emerging use cases
- ✅ Competitive advantage with premium depth features

**Negative Impacts**:
- ⚠️ License complexity (CC-BY-NC for NESTED variants)
- ⚠️ Migration effort for existing users (~2 weeks)
- ⚠️ Increased VRAM requirements (2-10GB vs 4-8GB)
- ⚠️ Learning curve for new API and configuration

**Net Business Value**: **TBD** (depends on validation results)

### Impact of DEFER Decision

**Positive Impacts**:
- ✅ Avoids premature commitment to underperforming model
- ✅ Allows time for DA3 team to release improvements
- ✅ Maintains stable production baseline

**Negative Impacts**:
- ❌ Delayed improvements for structure scenes
- ❌ Competitive disadvantage (competitors may adopt DA3 first)
- ❌ Sunk cost of 40 dev hours (partially wasted)

### Impact of REJECT Decision

**Positive Impacts**:
- ✅ Clear closure, no technical debt accumulation
- ✅ Resources freed for alternative improvements
- ✅ Simplicity maintained (no license complexity)

**Negative Impacts**:
- ❌ Structure scene problem remains unsolved (25% pass)
- ❌ No access to advanced features (metric depth, multi-view)
- ❌ 40 dev hours fully wasted
- ❌ Exploration of alternatives required

---

## Risk Assessment

### Adoption Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| License non-compliance | HIGH | License validation module prevents misuse |
| Performance regression | MEDIUM | Comprehensive A/B validation before deployment |
| User migration friction | MEDIUM | Backward-compatible API, migration guide |
| Increased VRAM requirements | LOW | Provide multiple model variants (SMALL, BASE, LARGE) |

### Deferral Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Indefinite delays | HIGH | 30-day hard deadline for re-validation |
| Opportunity cost | MEDIUM | Parallel exploration of alternatives |
| Team morale | LOW | Transparent communication of decision rationale |

### Rejection Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Competitive gap | MEDIUM | Accelerate alternative improvements (Materials V3) |
| Structure scene problem persists | HIGH | Invest in post-processing edge refinement |
| Sunk cost | LOW | Document learnings for future integrations |

---

## Timeline and Next Steps

### Immediate Actions (Phase 2, Current)

1. ✅ **Documentation Consolidation**: 15 docs → 4 docs (COMPLETE)
2. 🚧 **Code Organization**: Commit untracked DA3 files
3. 🚧 **A/B Validation**: Run DA3 against 46-image baseline
4. 🚧 **Metrics Analysis**: Compare to decision thresholds
5. 🚧 **Final Decision**: Update this document with recommendation

### Post-Decision Actions (Scenario-Dependent)

**If ADOPT**:
- Week 1: Update production configs, run regression tests
- Week 2: Deploy to staging, user acceptance testing
- Week 3: Production rollout, monitor for issues
- Week 4: Retrospective, documentation finalization

**If DEFER**:
- Week 1: Investigate structure scene failures, test variants
- Week 2-3: Implement refinements, re-run validation
- Week 4: Re-evaluate decision (adopt or reject)

**If REJECT**:
- Week 1: Archive DA3 code, update roadmap
- Week 2: Document rejection rationale (ADR)
- Week 3: Initiate alternative depth model evaluation
- Week 4: Begin Materials V3 acceleration

---

## Decision Authority

**Primary**: Transformation Portal Architect (system-level implications)  
**Consulted**: Transformation Portal Specialist (implementation feasibility)  
**Informed**: Product Owner, End Users

**Approval Requirements**:
- ✅ Validation results meet thresholds (data-driven)
- ✅ Architect sign-off on system impact
- ✅ Specialist sign-off on implementation plan

**No override permitted**: Thresholds are non-negotiable guardrails.

---

## Final Recommendation

**Status**: 🚧 **PENDING A/B TEST COMPLETION**

**Preliminary Assessment**:
- DA3 integration is **complete and production-ready** (lux_depth_v3/)
- License validation prevents non-commercial model misuse
- 40 dev hours invested, architecture sound
- **Decision solely depends on validation performance**

**Expected Timeline**:
- Validation completion: +90 minutes
- Metrics analysis: +30 minutes
- Final decision: +15 minutes
- **Total**: ~2-3 hours from now

**This document will be updated with FINAL RECOMMENDATION upon validation completion.**

---

**Document Version**: 1.0 (Pre-validation)  
**Last Updated**: 2025-12-19 19:58 UTC  
**Next Update**: Upon validation completion
