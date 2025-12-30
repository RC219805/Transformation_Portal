# Freeze-Lift Decision Memo - Template

**Decision Type**: Risk Acceptance with Bounded Exposure
**Authority**: Architecture Team
**Effective Date**: January 10, 2026 (freeze lift)
**Status**: TEMPLATE - Pending Validation Execution

---

## Executive Summary

**Decision**: [ENABLE BY DEFAULT / KEEP OPT-IN]

**Risk Posture**: This decision represents a **risk acceptance** with clearly bounded exposure, not a claim of perfection.

**Key Finding**: Edge refinement [MEETS / DOES NOT MEET] validation thresholds with [ACCEPTABLE / UNACCEPTABLE] edge case handling.

---

## Background

### Feature Description
Edge refinement applies guided filtering and bilateral smoothing to depth maps, improving edge fidelity in architectural rendering outputs.

### Validation Scope
- **Dataset**: 10 representative images (4 interior, 4 exterior, 2 aerial)
- **Test Matrix**: 40 runs (baseline, subtle, balanced, aggressive presets)
- **Timeline**: Days 4-14 of Phase 3 (December 21 - January 10, 2026)

### Decision Authority
**Primary**: Architectural Quality Reviewer
**Final Approval**: Architecture Team
**Accountability**: This decision is signed by named individuals, not "the team"

---

## Quantitative Evidence

### Metrics Summary

| Metric | Threshold | Actual | Status | Notes |
|--------|-----------|--------|--------|-------|
| Edge F1 | ≥ +5% | _____ % | ⬜ PASS / ⬜ FAIL | Edge detection improvement |
| PSNR | ≤ -1 dB | _____ dB | ⬜ PASS / ⬜ FAIL | Signal degradation |
| SSIM | ≤ -0.02 | _____ | ⬜ PASS / ⬜ FAIL | Structural similarity |
| Visual | ≤ 10% artifacts | _____ % | ⬜ PASS / ⬜ FAIL | Manual inspection |

**Overall Metrics**: ⬜ PASS / ⬜ CONDITIONAL / ⬜ FAIL

---

## Qualitative Evidence

### Visual Review Findings

**Artifact Types Observed**:
- [LIST SPECIFIC ARTIFACT TYPES, E.G., "EDGE HALOS IN 2/40 IMAGES"]
- [OR "NO ARTIFACTS DETECTED"]

**Client-Readiness Assessment**:
- Would ship: _____ / 40 images (____ %)
- Requires rework: _____ / 40 images (____ %)

**Sentinel Case Performance**:
1. Glass-on-glass (interior_bedroom_01_SENTINEL): [PASS / FAIL]
2. Low-contrast facade (exterior_facade_01_SENTINEL): [PASS / FAIL]

---

## Threshold Acceptance Rationale

**Why These Results Are Acceptable**:

### Edge F1 (≥ +5%)
**Rationale**:
[EXPLAIN WHY THIS LEVEL OF EDGE IMPROVEMENT JUSTIFIES THE CHANGE]
[OR WHY DEGRADATION IS ACCEPTABLE GIVEN OTHER FACTORS]

### PSNR (≤ -1 dB)
**Rationale**:
[EXPLAIN WHY THIS SIGNAL DEGRADATION IS WITHIN ACCEPTABLE BOUNDS]
[CITE VISUAL QUALITY ASSESSMENT IF METRICS MARGINALLY FAIL]

### SSIM (≤ -0.02)
**Rationale**:
[EXPLAIN WHY STRUCTURAL SIMILARITY CHANGES ARE ACCEPTABLE]
[REFERENCE CLIENT-READINESS TEST IF APPLICABLE]

### Visual Artifacts (≤ 10%)
**Rationale**:
[EXPLAIN WHY OBSERVED ARTIFACT RATE IS ACCEPTABLE]
[CITE SCENE-SPECIFIC NATURE IF APPLICABLE]

---

## Risk Assessment

### Identified Risks

**Technical Risk**: 🟢 / 🟡 / 🔴
- **Description**: [E.G., "EDGE HALOS IN LOW-CONTRAST SCENES"]
- **Exposure**: [E.G., "AFFECTS ~5% OF TYPICAL WORKLOAD"]
- **Mitigation**: [E.G., "OPT-OUT AVAILABLE VIA --NO-EDGE-REFINEMENT"]

**Client Impact Risk**: 🟢 / 🟡 / 🔴
- **Description**: [E.G., "MINOR VISUAL ARTIFACTS IN EDGE CASES"]
- **Exposure**: [E.G., "CLIENT-FACING DELIVERABLES MAY REQUIRE MANUAL REVIEW"]
- **Mitigation**: [E.G., "BASELINE MODE ALWAYS AVAILABLE"]

**Regression Risk**: 🟢 / 🟡 / 🔴
- **Description**: [E.G., "FUTURE CHANGES MAY DEGRADE EDGE QUALITY"]
- **Exposure**: [E.G., "NO AUTOMATED CI CHECKS YET"]
- **Mitigation**: [E.G., "SENTINEL IMAGES DESIGNATED FOR FUTURE TESTING"]

---

## Bounded Exposure

### Exposure Limits

**What We Accept**:
- [E.G., "EDGE REFINEMENT MAY INTRODUCE MINOR ARTIFACTS IN <10% OF IMAGES"]
- [E.G., "PERFORMANCE OVERHEAD OF +5-10% IN EDGE-HEAVY SCENES"]
- [E.G., "POTENTIAL FOR SCENE-SPECIFIC DEGRADATION IN LOW-CONTRAST SCENARIOS"]

**What We Do NOT Accept**:
- [E.G., "WIDESPREAD VISUAL DEGRADATION (>10% ARTIFACT RATE)"]
- [E.G., "CLIENT-FACING FAILURES REQUIRING SYSTEMATIC REWORK"]
- [E.G., "STRUCTURAL QUALITY LOSS (SSIM DEGRADATION > -0.05)"]

**Rollback Capability**: ✅
- Environment variable: `LUX_EMERGENCY_DISABLE_EDGE=1`
- CLI flag: `--no-edge-refinement`
- Config hot-patch: Available

---

## Decision Recommendation

### Recommended Action

**Primary Recommendation**: [ENABLE BY DEFAULT / KEEP OPT-IN]

**Rationale**:
[SYNTHESIZE QUANTITATIVE + QUALITATIVE EVIDENCE]
[EXPLAIN WHY THIS DECISION BALANCES QUALITY, RISK, AND CLIENT IMPACT]

**Conditional Factors** (if applicable):
- [E.G., "ENABLE FOR INTERIOR SCENES, OPT-IN FOR EXTERIORS"]
- [E.G., "ENABLE WITH 3-MONTH MONITORING PERIOD"]
- [E.G., "ENABLE WITH DOCUMENTATION OF KNOWN EDGE CASES"]

---

## Approval & Accountability

### Review Sign-Off

**Quantitative Review**:
- Reviewer: _____________________
- Date: _____________________
- Recommendation: ⬜ APPROVE / ⬜ CONDITIONAL / ⬜ REJECT

**Qualitative Review** (Architectural Quality Reviewer):
- Reviewer: _____________________
- Date: _____________________
- Recommendation: ⬜ APPROVE / ⬜ CONDITIONAL / ⬜ REJECT

**Architecture Team Approval**:
- Approver: _____________________
- Date: _____________________
- Decision: ⬜ ENABLE BY DEFAULT / ⬜ KEEP OPT-IN
- Effective: January 10, 2026

---

## Post-Decision Actions

### Immediate (Freeze Lift)
- [ ] Update README with decision
- [ ] Update CHANGELOG.md
- [ ] Post GitHub Discussion (freeze lift announcement)
- [ ] Tag release (if enabling by default)

### 30-Day Monitoring (If Enabled)
- [ ] Track artifact reports
- [ ] Monitor client feedback
- [ ] Measure opt-out rate
- [ ] Review sentinel case performance

### 90-Day Review (If Enabled)
- [ ] Assess production impact
- [ ] Evaluate rollback necessity
- [ ] Update sentinel cases
- [ ] Propose CI integration

---

## Audit Trail

**Validation Execution**: [DATE]
**Dataset Version**: VALIDATION_CHECKSUMS.txt (SHA256)
**Harness Version**: validation_harness.py ([COMMIT HASH])
**Environment**: Python [VERSION], lux-depth-v2 [VERSION]

**Evidence Artifacts**:
- Validation results: `validation_results/validation_results.json`
- Summary report: `validation_results/validation_summary.json`
- Visual comparisons: `validation_results/{baseline,balanced}/*/`
- Validation report: `docs/EDGE_REFINEMENT_VALIDATION_REPORT.md`

---

## Notes

**This is a risk acceptance decision, not a quality guarantee.**

- Metrics meet defined thresholds with known limitations
- Visual quality assessed by named reviewers
- Rollback capability preserved
- Bounded exposure documented
- Future monitoring planned

**Challenge Scenario**: If questioned post-deployment:
> "Who decided this was acceptable?"

**Answer**: Named reviewers (see Approval & Accountability section), based on quantitative metrics + qualitative assessment + bounded exposure analysis documented in this memo.

---

**Status**: TEMPLATE - To be completed after validation execution
**Next Update**: After Days 9-12 (threshold review complete)
