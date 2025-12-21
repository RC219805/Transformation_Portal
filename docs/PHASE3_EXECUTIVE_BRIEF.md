# Edge Refinement Validation - Executive Brief

**Phase**: 3 (Quality Validation & Production Hardening)  
**Timeline**: December 21 - January 10, 2026 (21 days)  
**Status**: Foundation Ready, Execution Pending

---

## Objective

Prove edge refinement quality lift or neutrality with audit-grade evidence before enabling by default.

**Success Condition**: A reader with zero context can answer:
> "Why was this allowed into production?"

---

## Execution Roadmap (14-Day Critical Path)

### Days 0-2: Foundation (Automatable) ✅
**Status**: Complete

- [x] Validation harness created (`validation_harness.py`)
- [x] Directory structure defined
- [x] Dataset manifest locked
- [x] Metrics framework planned
- [x] Acceptance criteria established

**Deliverables**:
- Automated test runner (40 validation runs)
- Dataset integrity verification (SHA256 checksums)
- Incremental result tracking (JSON)

---

### Days 2-5: Data Reality Check (Human-Gated)
**Status**: Pending

**Critical Path**:
1. Gather 10 representative architectural images
   - 4 interior (bedrooms, kitchens, bathrooms, great rooms)
   - 4 exterior (pools, facades, courtyards, gardens)
   - 2 aerial (roofs, twilight)

2. Include **sentinel cases** (reusable regression tests)
   - Known failure modes: glass-on-glass, low-contrast facades
   - Edge-heavy scenes: foliage, railings, complex materials

3. Generate checksums → Lock dataset
   ```bash
   cd validation_images/
   shasum -a 256 *.tiff > ../VALIDATION_CHECKSUMS.txt
   ```

**Risk**: Insufficient diversity → validation not representative  
**Mitigation**: Use public datasets + synthetic renders if needed

---

### Days 4-8: Quantitative Proof (Automatable)
**Status**: Harness ready, awaiting dataset

**Execution**:
```bash
python validation_harness.py \
  --dataset-dir validation_images/ \
  --output-dir validation_results/ \
  --checksum-file VALIDATION_CHECKSUMS.txt
```

**Outputs** (40 test runs):
- Baseline vs subtle vs balanced vs aggressive
- Processing time per configuration
- Output quality (master16, upscaled, marketing, preview)

**Metrics** (automated computation):
- Edge F1 score (Canny edge detection baseline)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

**Key Insight**: This is audit evidence, not experimentation. Treat results as permanent record.

---

### Days 7-10: Human Judgment Gate (Non-Automatable)
**Status**: Pending

**Critical Review**:
- Visual artifact detection
  - Edge halos, depth bleeding, material breakup
  - Unnatural sharpening, color shifts
  - Structural deformation

- Client-readiness test
  - "Would this ship to a paying client?"
  - "Would this require rework?"
  - "Is this better than baseline?"

**Hard Truth**: No metric will save you here. Senior judgment matters.

**Process**:
1. Side-by-side comparison (baseline vs refined)
2. Rating scale: Better / Neutral / Worse / Artifact
3. Document **why** ratings were assigned (prevents future re-litigation)

---

### Days 9-12: Synthesis (Threshold Review)
**Status**: Template ready, awaiting data

**Acceptance Thresholds**:
- Edge F1: ≥ +5% improvement vs baseline
- PSNR: ≤ -1 dB degradation
- SSIM: ≤ -0.02 degradation
- Visual: No artifacts in 90%+ of images

**Decision Matrix**:

| Metrics | Visual | Recommendation |
|---------|--------|----------------|
| ✅ Pass | ✅ Pass | Enable by default |
| ✅ Pass | ❌ Fail | Keep opt-in, document artifacts |
| ❌ Fail | ✅ Pass | Keep opt-in, cite metrics |
| ❌ Fail | ❌ Fail | Opt-in only, flag for investigation |

**Refinement Opportunity**: Document **why** thresholds were accepted, not just that they passed.

---

### Days 11-14: Closure (Executive Summary)
**Status**: Template prepared

**Validation Report** (single narrative):
- Executive summary (1 page)
- Methodology (reproducible steps)
- Quantitative results (metrics + graphs)
- Qualitative assessment (visual review)
- Recommendation (default OR opt-in)
- Rationale (why this decision)

**Freeze Lift Decision**:
- Clear recommendation
- Signed-off rationale
- Audit trail (all artifacts preserved)

---

## Risk Posture

| Area | Status | Notes |
|------|--------|-------|
| Technical Risk | 🟢 Low | Harness ready, rollback documented |
| Governance Risk | 🟢 Low | CI enforced, regression tests active |
| Reputational Risk | 🟡 Medium | Until visual review complete |
| Process Drift | 🟢 Controlled | Phase 3 scope locked |

---

## Non-Negotiable Constraints

### Do Not Compress
**Qualitative review window** (Days 7-10)
- Most common failure mode at this maturity level
- No automation can replace senior judgment
- Client trust depends on this gate

### Keep Artifact Count Small
**Fewer, stronger documents > more noise**
- Validation report (1 file)
- Metrics summary (1 file)
- Freeze lift memo (1 file)

### Freeze-Lift Tone
**Recommendation, not celebration**
- Tone matters if this goes external
- Focus: evidence-based decision, not feature promotion
- Language: "validated" not "proven", "acceptable" not "optimal"

---

## Success Criteria

### Minimum (Required)
- [ ] 40 validation runs executed
- [ ] Metrics computed (Edge F1, PSNR, SSIM)
- [ ] Visual review completed (all images)
- [ ] Recommendation made (default OR opt-in)
- [ ] Validation report published

### Target (Desired)
- [ ] All metrics meet thresholds
- [ ] No visual artifacts detected
- [ ] Edge refinement enabled by default
- [ ] Sentinel cases identified for future regression tests

### Stretch (Optional)
- [ ] Automated metrics computation integrated to CI
- [ ] Visual comparison tool for manual review
- [ ] External stakeholder summary published

---

## Current Status

**Phase 2**: ✅ Complete (all gaps closed)  
**Phase 3 Foundation**: ✅ Complete (automation ready)  
**Phase 3 Execution**: 📋 Pending (awaiting dataset)

**Blocking Items**: 9 additional test images needed

**Next Actions** (48 hours):
1. Source validation dataset (10 images)
2. Generate checksums
3. Execute validation harness
4. Begin metric computation

---

## Governance

**Feature Freeze**: Active until Jan 10, 2026  
**CI Enforcement**: Automated label checks  
**Rollback Capability**: 3 methods documented  
**Dataset Integrity**: SHA256 verification

**Approval Authority**: Architecture Team  
**Final Sign-Off**: Required before freeze lift

---

## Bottom Line

**You are entering Phase 3 from a position of strength.**

- Plan is coherent
- Risk-aware
- Appropriately conservative
- Automation foundation complete

**Proceed — but preserve the discipline that got you here.**

---

**Document Owner**: Architecture Team  
**Last Updated**: December 21, 2025  
**Next Review**: December 28, 2025 (Week 2 completion)
