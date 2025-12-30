# Governance Cleanup - Complete

**Date**: 2025-12-15
**Architect**: Transformation Portal Architect
**Status**: ✅ COMPLETE

---

## Summary

Successfully resolved governance crisis in PR-W1 water detection baseline management. Established clean, reproducible state with disciplined baseline versioning and update policy.

---

## Commits Made

1. **afef3dd** - `fix(water): portable path resolution (ground truth relative paths)`
   - Ground truth root relative to ground_truth.json location
   - CI workflow portable across working directories

2. **4222491** - `feat(water): add PR-W4 two-stage gating telemetry`
   - WaterCandidateReport: confidence tracking (raw, after suppressors, final)
   - MaterialsV3Config: two-stage thresholds + saturation boost

3. **e70dab4** - `refactor(water): implement baseline governance discipline`
   - Renamed baseline_ci_v0.json → baseline_ci_audit_v0.json (immutable)
   - Created baseline_ci_current_v1.json (current enforced baseline)
   - Updated CI to reference canonical baseline

4. **b60ea3e** - `docs(water): document baseline governance scheme in README`
   - Documented baseline versioning scheme
   - Baseline update policy with holdout validation requirement
   - Known limitations (pool_0008 miss at 83.3% recall)

5. **8bf7181** - `docs(architecture): add ADR-001 for baseline governance policy`
   - Architectural Decision Record for two-tier baseline system
   - Prevents overfitting, requires validation before updates

---

## Current State

### Baselines
- **baseline_ci_audit_v0.json**: Immutable historical (83.3% pool recall)
- **baseline_ci_current_v1.json**: Current enforced (83.3% pool recall)

### Safe Thresholds (Reverted from Experiments)
- `glass_edge_alignment_threshold: 0.15` (conservative)
- `glass_grid_score_threshold: 0.25` (conservative)
- `glass_penalty: 0.6` (conservative)

### Known Limitations
- **Pool recall**: 83.3% (pool_0008 missed - low-sat pool with tile grid)
- **Accepted**: Until holdout validation framework exists (PR-W1.2)

### CI Integration
- References `baseline_ci_current_v1.json` for regression checks
- Clean working tree (no uncommitted experiments)

---

## Deleted Files

**Experimental baselines**: baseline_ci_v1.json, baseline_ci_v1_clean.json, test_v1_*.json
**Premature docs**: GLASS_SUPPRESSOR_*_COMPLETE.md, PATH_RESOLUTION_*_COMPLETE.md, PR_W1.3_*.md

**Retained analysis docs** (untracked):
- BASELINE_THRESHOLD_ANALYSIS.md (accurate diagnosis)
- GLASS_SUPPRESSOR_MULTISCALE_FIX.md (design, not implementation)
- GLASS_SUPPRESSOR_CLAIMS_VALIDATION.md (honest assessment)

---

## Next Steps

### PR-W1.2: Holdout Validation Framework
1. Create diverse holdout set (10-20 real negatives, edge cases)
2. Validate glass suppressor multi-scale logic
3. ROC analysis for optimal thresholds
4. Regenerate baseline v2 with validated thresholds

### PR-W1.3: ADE20K Integration (AFTER PR-W1.2)
- Semantic segmentation for water/pool detection
- Hybrid heuristic + semantic approach

---

## Governance Lessons

✅ **Baseline versioning**: Immutable audit + mutable current
✅ **Commit discipline**: Path fixes separate from experiments
✅ **Honest documentation**: Known limitations documented
❌ **Anti-patterns avoided**: Test set tuning, premature completion claims

---

**Repository Health**: ✅ **RESTORED**
**Governance**: ✅ **DISCIPLINED AND DOCUMENTED**
