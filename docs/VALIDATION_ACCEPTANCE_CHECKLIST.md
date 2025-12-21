# Validation Acceptance Checklist

**Purpose**: Gate validation quality and ensure reproducibility  
**Owner**: Architecture Team  
**Timeline**: Days 9-12 of Phase 3

---

## Pre-Execution Verification

### Dataset Integrity ✅ / ❌
- [ ] 10 images gathered (4 interior, 4 exterior, 2 aerial)
- [ ] SHA256 checksums generated
- [ ] Checksums verified before execution
- [ ] Dataset locked (no changes allowed during validation)

**Evidence Required**: `VALIDATION_CHECKSUMS.txt`

---

### Environment Validation ✅ / ❌
- [ ] lux-depth-v2 CLI version verified
- [ ] Test environment isolated (no background load)
- [ ] Sufficient disk space (>50GB for 40 runs)
- [ ] Performance baseline re-confirmed

**Evidence Required**: Environment snapshot (Python version, GPU/MPS, disk space)

---

### Harness Verification ✅ / ❌
- [ ] validation_harness.py tested on single image
- [ ] Output directory structure confirmed
- [ ] Incremental result saving working
- [ ] Timeout handling tested

**Evidence Required**: Dry-run execution log

---

## Execution Verification

### Quantitative Metrics ✅ / ❌

**Edge F1 Score**:
- [ ] Computed for all 40 runs
- [ ] Baseline established (Canny edge detection)
- [ ] Improvement ≥ +5% OR degradation documented
- [ ] Outliers investigated

**Threshold**: ≥ +5% improvement (recommended)  
**Actual**: _________ %  
**Decision**: PASS / FAIL / CONDITIONAL

---

**PSNR (Peak Signal-to-Noise Ratio)**:
- [ ] Computed for all 40 runs
- [ ] Degradation ≤ -1 dB OR documented
- [ ] Outliers investigated
- [ ] Per-scene-type breakdown

**Threshold**: ≤ -1 dB degradation (acceptable)  
**Actual**: _________ dB  
**Decision**: PASS / FAIL / CONDITIONAL

---

**SSIM (Structural Similarity)**:
- [ ] Computed for all 40 runs
- [ ] Degradation ≤ -0.02 OR documented
- [ ] Outliers investigated
- [ ] Per-material-type breakdown

**Threshold**: ≤ -0.02 degradation (acceptable)  
**Actual**: _________  
**Decision**: PASS / FAIL / CONDITIONAL

---

### Qualitative Assessment ✅ / ❌

**Review Authority**: Architectural Quality Reviewer  
**Reviewer Name**: _____________________  
**Review Date**: _____________________

**Visual Artifact Detection**:
- [ ] Side-by-side comparison (baseline vs refined)
- [ ] Edge halos checked
- [ ] Depth bleeding checked
- [ ] Material breakup checked
- [ ] Unnatural sharpening checked
- [ ] Color shifts checked

**Artifact Rate**: _____ / 40 images (____ %)  
**Threshold**: ≤ 10% (4 images max)  
**Decision**: PASS / FAIL

**Reviewer Sign-Off**: _____________________  
**Date**: _____________________

---

**Client-Readiness Test**:
- [ ] "Would this ship?" rating for all images
- [ ] "Would this require rework?" assessment
- [ ] "Is this better than baseline?" comparison

**Rating Distribution**:
- Better: _____ images
- Neutral: _____ images
- Worse: _____ images
- Artifact: _____ images

**Threshold**: 90%+ Better or Neutral  
**Decision**: PASS / FAIL

**Reviewer Sign-Off**: _____________________  
**Date**: _____________________

---

**Sentinel Case Identification**:
- [ ] 2+ images flagged as regression tests
- [ ] Known failure modes documented
- [ ] Edge cases captured for future CI

**Sentinel Cases**:
1. ________________________________
2. ________________________________

---

## Synthesis & Decision

### Threshold Review ✅ / ❌

| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Edge F1 | ≥ +5% | _____ | ⬜ PASS / ⬜ FAIL |
| PSNR | ≤ -1 dB | _____ | ⬜ PASS / ⬜ FAIL |
| SSIM | ≤ -0.02 | _____ | ⬜ PASS / ⬜ FAIL |
| Visual | ≤ 10% artifacts | _____ | ⬜ PASS / ⬜ FAIL |

**Overall Metrics**: ⬜ PASS / ⬜ CONDITIONAL / ⬜ FAIL

---

### Decision Matrix

**Scenario 1: All Metrics PASS**
- [ ] Metrics meet all thresholds
- [ ] No visual artifacts detected
- [ ] Client-readiness confirmed

**Recommendation**: ✅ **Enable edge refinement by default**

---

**Scenario 2: Metrics PASS, Visual CONDITIONAL**
- [ ] Metrics meet thresholds
- [ ] Minor artifacts in <10% of images
- [ ] Artifacts are scene-specific (documented)

**Recommendation**: ⚠️ **Enable with opt-out for specific scene types**

---

**Scenario 3: Metrics CONDITIONAL, Visual PASS**
- [ ] Metrics marginally below threshold (<10% miss)
- [ ] No visual artifacts
- [ ] User experience positive

**Recommendation**: ⚠️ **Enable with monitoring, document metric variance**

---

**Scenario 4: Any Component FAIL**
- [ ] Metrics significantly below threshold (>10% miss)
- [ ] OR significant visual artifacts (>10% of images)
- [ ] OR client-readiness fails

**Recommendation**: ❌ **Keep opt-in, document failure modes**

---

### Rationale Documentation ✅ / ❌

**Threshold Acceptance Rationale** (Centralized Location):

**Document**: `docs/EDGE_REFINEMENT_VALIDATION_REPORT.md` § Threshold Acceptance Rationale

**Required Content**:
- [ ] Quantitative evidence cited (metrics + actual values)
- [ ] Qualitative assessment summarized (visual review findings)
- [ ] Edge cases documented (specific images + failure modes)
- [ ] Trade-offs acknowledged (where thresholds marginally missed)
- [ ] Future mitigation planned (if conditional acceptance)

**Why These Thresholds Were Accepted** (must answer):
1. **Edge F1 (≥ +5%)**: Why this improvement level is acceptable?
2. **PSNR (≤ -1 dB)**: Why this degradation level is acceptable?
3. **SSIM (≤ -0.02)**: Why this degradation level is acceptable?
4. **Visual (≤ 10%)**: Why this artifact rate is acceptable?

**Rationale Author**: _____________________  
**Date**: _____________________  
**Approval**: _____________________

**Note**: This section is the single source of truth for threshold decisions. Future disputes reference this section only.

---

## Post-Decision Verification

### Validation Report ✅ / ❌
- [ ] Executive summary (1 page)
- [ ] Methodology (reproducible)
- [ ] Quantitative results (metrics + graphs)
- [ ] Qualitative assessment (visual review)
- [ ] Recommendation (default OR opt-in)
- [ ] Rationale (why this decision)

**Deliverable**: `docs/EDGE_REFINEMENT_VALIDATION_REPORT.md`

---

### Artifact Preservation ✅ / ❌
- [ ] All 40 test outputs archived
- [ ] Metrics computation scripts preserved
- [ ] Visual comparison images saved
- [ ] Decision memo filed

**Archive**: `validation_results/` (permanent)

---

### Freeze Lift Preparation ✅ / ❌
- [ ] Recommendation documented
- [ ] README updated with decision
- [ ] CHANGELOG.md entry prepared
- [ ] GitHub Discussion updated

**Blocker Check**: No items preventing freeze lift on Jan 10

---

## Sign-Off

### Technical Review
**Reviewer**: _____________________  
**Date**: _____________________  
**Status**: ⬜ APPROVED / ⬜ CONDITIONAL / ⬜ REJECTED  
**Notes**:

---

### Architecture Review
**Reviewer**: _____________________  
**Date**: _____________________  
**Status**: ⬜ APPROVED / ⬜ CONDITIONAL / ⬜ REJECTED  
**Notes**:

---

### Final Approval
**Authority**: _____________________  
**Date**: _____________________  
**Decision**: ⬜ ENABLE BY DEFAULT / ⬜ KEEP OPT-IN  
**Effective**: January 10, 2026 (freeze lift)

---

## Audit Trail

**Validation Execution Date**: _____________________  
**Dataset Version**: VALIDATION_CHECKSUMS.txt (SHA256)  
**Harness Version**: validation_harness.py (commit hash: _______)  
**Environment**: Python _____, lux-depth-v2 _____, GPU/MPS _____  

**Artifacts**:
- Validation results: validation_results/validation_results.json
- Summary report: validation_results/validation_summary.json
- Validation report: docs/EDGE_REFINEMENT_VALIDATION_REPORT.md
- Freeze lift memo: docs/FREEZE_LIFT_DECISION_MEMO.md

---

**Status**: PENDING EXECUTION  
**Next Review**: After validation execution (Days 9-12)
