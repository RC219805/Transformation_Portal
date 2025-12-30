# Golden Path Consolidation - Documentation Index

**Strategic Architecture Review - December 20, 2025**

> 🎯 **PRIMARY PRODUCTION WORKFLOW**: `lux_depth_v2`
> All other pipelines are advanced/experimental. See [Golden Path Definition](#-golden-path-definition) below.

---

## 📋 Quick Navigation

**Start Here** → 📊 [ARCHITECT_RESPONSE_SUMMARY.md](ARCHITECT_RESPONSE_SUMMARY.md) (3 minutes)

**For Executives** → 👔 [EXECUTIVE_SUMMARY_GOLDEN_PATH.md](EXECUTIVE_SUMMARY_GOLDEN_PATH.md) (90 seconds)

**For Implementation** → ⚡ [GOLDEN_PATH_CONSOLIDATION_CHECKLIST.md](GOLDEN_PATH_CONSOLIDATION_CHECKLIST.md) (day-by-day plan)

**For Policy** → 📜 [FEATURE_FREEZE_POLICY.md](FEATURE_FREEZE_POLICY.md) (freeze rules)

**For Deep Dive** → 🔬 [STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md](STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md) (full analysis)

---

## 📂 Document Overview

| Document | Size | Type | Purpose | Audience |
|----------|------|------|---------|----------|
| **ARCHITECT_RESPONSE_SUMMARY.md** | 3.5K | 📊 Quick Reference | Quick answer to strategic questions | All stakeholders |
| **EXECUTIVE_SUMMARY_GOLDEN_PATH.md** | 5.1K | 👔 Executive Brief | 90-second overview for decision-makers | Leadership, PMs |
| **GOLDEN_PATH_CONSOLIDATION_CHECKLIST.md** | 6.5K | ⚡ Implementation | Day-by-day implementation tasks | Developers |
| **FEATURE_FREEZE_POLICY.md** | 7.2K | 📜 Policy | Freeze rules and enforcement | Contributors |
| **STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md** | 14K | 🔬 Deep Analysis | Complete architectural analysis | Architects, Tech Leads |

**Total**: 36.3K (5 documents)

---

## 🎯 Decision Flow

```
┌─────────────────────────────────────────────────┐
│ Need quick yes/no on consolidation?            │
│ → ARCHITECT_RESPONSE_SUMMARY.md                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Need to explain to leadership?                 │
│ → EXECUTIVE_SUMMARY_GOLDEN_PATH.md             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Ready to implement?                             │
│ → GOLDEN_PATH_CONSOLIDATION_CHECKLIST.md       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Need to communicate freeze to team?            │
│ → FEATURE_FREEZE_POLICY.md                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Need to understand full rationale?             │
│ → STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20  │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Golden Path Definition

**PRIMARY PRODUCTION WORKFLOW**: `lux_depth_v2`

```bash
# Quick Start
pip install -e .
lux-depth-v2 --input-dir renders/ --output-dir out/ --preset interior_luxury
docker-compose -f deployment/docker-compose.production.yml up -d
```

**Why lux_depth_v2?**
- ✅ Security hardened (CVE-2024-27763 mitigated)
- ✅ Production validated (127-400 images/hour)
- ✅ Docker deployment ready
- ✅ Actively maintained (Dec 20, 2025)

**What's NOT golden path?**
- Async pipeline → Performance layer (advanced)
- Unified/context-aware → Orchestration (advanced)
- Training → Research infrastructure (not production)
- Lux Depth V3 → DEFERRED (DA3 evaluation pending)

---

## 🚀 Implementation Timeline

**Week 1 (Dec 20-27)**:
- Day 1-2: Golden Path declaration (README restructure)
- Day 3-4: Training collapse, edge refinement flag
- Day 5-6: Validation & testing
- Day 7: Feature freeze communication

**Weeks 2-3 (Dec 28 - Jan 10)**:
- Edge refinement validation complete
- Performance baseline established
- Documentation polish
- Success metrics measured

**End Date**: January 10, 2026 (freeze lifts)

> ⚠️ **Freeze Scope Reminder**: New pipelines, presets, or stages require architect approval during freeze period. See [FEATURE_FREEZE_POLICY.md](FEATURE_FREEZE_POLICY.md) for details.

---

## ✅ Key Decisions

### 1. Consolidation Needed?
**YES** - Cognitive load is real, navigation degraded

### 2. Golden Path Defined?
**YES** - lux_depth_v2 (security, deployment, validation)

### 3. Priority Ranking?
1. Declare Golden Path (Week 1)
2. Freeze Feature Surface (Weeks 1-3)
3. Promote Edge Refinement (Week 2)
4. Collapse Training Docs (Week 2-3)

### 4. Implementation Plan?
**YES** - 1-week plan with clear success metrics

---

## 📊 Expected Outcomes

### Before
- ⏱️ 15-30 min to answer "How do I process images?"
- 🔀 6+ decision branches presented
- 📜 Scroll 150+ lines to find lux-depth-v2
- 🎓 Training prominently displayed (line 536)

### After
- ⏱️ <2 min to answer "How do I process images?"
- 🔀 1 golden path + "Advanced" section
- 📜 <10 lines to find lux-depth-v2
- 🎓 Training gated behind "Advanced" label

---

## 🛡️ What's Protected

During implementation:
- ✅ All code preserved (no deletions)
- ✅ All features functional (no deprecations)
- ✅ All docs preserved (only reorganized)
- ✅ All workflows intact (no breaking changes)
- ✅ CI/CD unchanged (tests, security, dependencies)

---

## 🔗 Related Documents

### Pre-Review Context
- `.github/copilot-instructions.md` - Repository structure
- `README.md` - Current state (1,082 lines)
- `docs/COMPREHENSIVE_CODEBASE_UPDATE_2025.md` - Recent improvements

### Post-Implementation
- `docs/GOLDEN_PATH_DECISION.md` - ADR (to be created)
- `docs/training/TRAINING_GUIDE.md` - Consolidated training docs (to be created)
- `.github/ISSUE_TEMPLATE/feature_freeze_check.md` - Freeze template (to be created)

---

## 📞 Contact

**Questions about strategic assessment?**
→ Review STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md Appendix A (alternatives) and Appendix B (decision criteria)

**Questions about implementation?**
→ See GOLDEN_PATH_CONSOLIDATION_CHECKLIST.md "Risk Mitigation" section

**Questions about freeze policy?**
→ See FEATURE_FREEZE_POLICY.md FAQ section

**Need exception approval?**
→ File GitHub issue with label `feature-freeze-exception`

---

## 📌 Status

- **Assessment**: ✅ Complete (Dec 20, 2025)
- **Approval**: ✅ Consolidation approved (95% confidence)
- **Implementation**: 🟡 Pending (starts Dec 20, 2025)
- **Freeze Active**: 🟡 Dec 20 - Jan 10, 2026

---

**Owner**: Transformation Portal Architect
**Last Updated**: December 20, 2025
**Next Review**: January 8, 2026 (assess freeze extension)
