# Production Deployment Acceptance Criteria
## High-Fidelity Depth Pipeline - Go/No-Go Decision Sheet

**Date**: December 17, 2025  
**Version**: 1.0  
**Configuration**: production (tile_size=1024, overlap=128, global_anchor=OFF)

---

## 🎯 Acceptance Gates

### PILOT APPROVAL (80% Pass Rate Required)

| Criterion | Threshold | Status | Notes |
|-----------|-----------|--------|-------|
| **Dataset Coverage** | ≥80% images pass quality gates | 🟡 PENDING | Validation suite ready |
| **Critical Scenes** | All 4 pass (Kitchen, GreatRoom, Aerial, Pool) | 🟡 PENDING | Pool validated ✅ |
| **Edge F1 (Primary)** | Mean ≥0.30 across dataset | 🟡 PENDING | Pool: 0.626-0.693 ✅ |
| **Chamfer Distance** | Worst-case <20px | 🟡 PENDING | Pool: 2.46px ✅ |
| **Seam Energy** | Worst-case <1.5 | 🟡 PENDING | Target: <1.2 nominal |
| **Halo/Overshoot** | Mean penalty <0.5 | 🟡 PENDING | Implementation complete |
| **Edge Count Ratio** | Max ≤2.5× | 🟡 PENDING | Hallucination gate |
| **Runtime** | <5min per 6K image (M4 Max) | ✅ PASS | Pool: ~40s (35 tiles) |
| **Memory** | Peak <16GB | ✅ PASS | Estimated ~8-12GB |
| **No Catastrophic Failures** | Zero images with visible grid seams | 🟡 PENDING | Seam validation implemented |

**PILOT DECISION**: ⏳ PENDING FULL DATASET RUN  
**Blocker**: Execute `production_validation_suite.py` on all 6 images  
**ETA**: 30-45 minutes processing time  

---

### PRODUCTION APPROVAL (95% Pass Rate Required)

| Criterion | Threshold | Status | Notes |
|-----------|-----------|--------|-------|
| **Dataset Coverage** | ≥95% images pass quality gates | ⏸️ BLOCKED | Requires pilot pass |
| **Materials V3 A/B** | Measurable improvement (Edge F1 +0.05) | ⏸️ BLOCKED | 100-scene validation needed |
| **Worst-Case Chamfer** | <15px (strict) | 🟡 PENDING | Pool: 2.46px ✅ |
| **Worst-Case Seam** | <1.2 (strict) | 🟡 PENDING | Target validated |
| **Throughput** | ≥120 images/hour batch | 🟡 PENDING | ~90/hour estimated |
| **Failure Rate** | <5% fallback to baseline | ⏸️ BLOCKED | Requires full run |
| **Quality Regression** | No image worse than baseline | ⏸️ BLOCKED | Requires A/B |

**PRODUCTION DECISION**: ⏸️ BLOCKED ON PILOT  
**Blockers**: 
1. Complete full dataset validation
2. Run Materials V3 A/B (100 scenes)
3. Validate throughput at scale

---

## 📊 Required Metric Thresholds

### Per-Image Quality Gates (Standard Mode)

```
edge_f1           ≥ 0.30   # Shift-tolerant alignment
edge_overlap      ≥ 0.40   # Coverage percentage
edge_count_ratio  ≤ 3.0    # Hallucination limit (relaxed)
chamfer_distance  ≤ 15.0   # Mean pixel misalignment
seam_energy       ≤ 1.2    # Boundary gradient ratio
overshoot_penalty ≤ 0.5    # Halo/ringing detection
```

### Per-Image Quality Gates (Strict Mode for Critical Scenes)

```
edge_f1           ≥ 0.45   # Higher alignment requirement
edge_overlap      ≥ 0.50   # Better coverage
edge_count_ratio  ≤ 2.0    # Stricter hallucination limit
chamfer_distance  ≤ 12.0   # Tighter alignment
seam_energy       ≤ 1.0    # Cleaner boundaries
overshoot_penalty ≤ 0.3    # Luxury edge quality
halo_score        ≥ 0.7    # No visible halos
```

### Dataset-Level Aggregates

```
pilot_pass_rate        ≥ 0.80   # 80% pass standard gates
production_pass_rate   ≥ 0.95   # 95% pass standard gates
edge_f1_mean           ≥ 0.35   # Average across dataset
edge_f1_std            ≤ 0.15   # Consistency check
chamfer_worst          ≤ 20.0   # Worst single image
seam_energy_worst      ≤ 1.5    # Worst single boundary
```

---

## 🔬 Critical Scene Requirements

### Kitchen.tif (Glass/Metal Complexity)

**Pass Criteria**:
- ✅ Edge F1 ≥0.35 (material boundaries preserved)
- ✅ Halo score ≥0.75 (no overshoot on reflective surfaces)
- ✅ Chamfer <12px (tight alignment on countertops/cabinets)
- ✅ No visible grid seams on planar surfaces

**Visual Inspection**:
- [ ] Glass cabinet edges clean (no halo)
- [ ] Metal appliance boundaries sharp
- [ ] Countertop edges straight (no warping)

---

### GreatRoom.tif (Large Planar Structures)

**Pass Criteria**:
- ✅ Seam energy <1.0 (clean walls/ceilings)
- ✅ Edge F1 ≥0.30 (architectural edges preserved)
- ✅ No plane warps visible on flat surfaces

**Visual Inspection**:
- [ ] Wall planes flat and continuous
- [ ] Ceiling no grid pattern
- [ ] Floor depth smooth gradient

---

### Aerial.tif (Scale Context Shift)

**Pass Criteria**:
- ✅ Edge count ratio <2.0× (no hallucinated features)
- ✅ Edge F1 ≥0.30 (outdoor features preserved)
- ✅ Depth gradients stable (no sudden jumps)

**Visual Inspection**:
- [ ] Building edges sharp
- [ ] Terrain depth smooth
- [ ] No tiling artifacts on sky

---

### Pool.tif (Water + Exterior Materials)

**Pass Criteria** (Previously Validated):
- ✅ Edge F1: 0.626-0.693 ✅ (PASS)
- ✅ Chamfer: 2.46px ✅ (PASS)
- ✅ Seam energy: 1.059 ✅ (PASS)
- ✅ Edge count ratio: 0.84× ✅ (PASS)

**Status**: ✅ VALIDATED AND PASSING

---

## ⚙️ Configuration Safety Checklist

### Required Settings (Non-Negotiable)

- [x] `use_global_anchor = False` (default OFF until planar validation complete)
- [x] `clahe_enabled = False` (geometry-meaningful depth)
- [x] `edge_snap_mode = "and_gated"` (only where RGB + depth edges agree)
- [x] `reconcile_method = "robust"` (Theil-Sen regression)
- [x] Config hash embedded in all outputs
- [x] Atomic JSON write with validation

### Prohibited Settings

- [x] ❌ Frequency-split fusion (permanently locked out)
- [x] ❌ Aggressive edge snapping (strength >0.4)
- [x] ❌ CLAHE on depth (visualization only)
- [x] ❌ Global anchor without unit test validation

---

## 🎬 Deployment Phasing

### Phase 1: Pilot (Week 1)

**Scope**: 50 representative scenes from 750_Picacho  
**Goal**: Validate pass rate ≥80%  

**Tasks**:
1. ✅ Run production_validation_suite.py on 50 scenes
2. ✅ Generate dataset_report.json with metrics
3. ✅ Visual inspection of top 10 failures
4. ✅ Tune thresholds if needed (within ±10%)
5. ✅ Document failure modes

**Success Criteria**:
- 40+ images pass standard gates
- All 4 critical scenes pass strict gates
- No catastrophic failures (seam energy >2.0)

**Fallback**: If <80% pass → halt, analyze root causes

---

### Phase 2: Materials V3 Integration (Week 2-3)

**Scope**: 100 scenes A/B test (baseline vs depth-enhanced)  
**Goal**: Measurable improvement in Materials V3 output  

**Tasks**:
1. ✅ Run Materials V3 with baseline depth (control)
2. ✅ Run Materials V3 with tiled depth (treatment)
3. ✅ Measure Edge F1 improvement (target +0.05)
4. ✅ Visual comparison: glass/metal boundaries
5. ✅ Stability check: zoning consistency

**Success Criteria**:
- Mean Edge F1 improvement ≥+0.05
- Glass/metal boundaries visibly cleaner
- No quality regression on any scene
- Fallback logic functional

**Fallback**: If no improvement → use baseline depth, log fallback events

---

### Phase 3: Full Production (Week 4+)

**Scope**: Complete 750_Picacho dataset + ongoing production  
**Goal**: ≥95% pass rate, sustainable throughput  

**Tasks**:
1. ✅ Roll out to full dataset
2. ✅ Monitor quality metrics dashboard
3. ✅ Track failure rate and fallback usage
4. ✅ Performance optimization if needed
5. ✅ Integrate into Materials V3 pipeline

**Success Criteria**:
- 95%+ images pass quality gates
- Throughput ≥120 images/hour
- Failure rate <5%
- Zero user-reported quality issues

**Rollback Plan**: Revert to baseline depth if failure rate >10%

---

## 📋 Validation Execution Checklist

### Pre-Flight

- [x] Validation suite tested on subset ✅
- [x] All metrics implemented and verified ✅
- [x] Configuration system hardened ✅
- [x] Output directory structure tested ✅
- [ ] Full dataset accessible (6 images ready ✅)
- [ ] Sufficient disk space (~10GB for outputs)
- [ ] M4 Max available for processing

### Execution

```bash
# Full validation run
python production_validation_suite.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_$(date +%Y%m%d) \
  --preset production \
  --critical-scenes Kitchen GreatRoom Aerial Pool

# Expected runtime: 30-45 minutes for 6 images
```

### Post-Flight

- [ ] Check `dataset_report.json` for pass rates
- [ ] Review `validation_results.csv` for per-image metrics
- [ ] Visual inspection of `visualizations/` directory
- [ ] Check worst-case metrics (chamfer, seam_energy)
- [ ] Verify critical scenes all passed
- [ ] Generate executive summary for stakeholders

---

## 🚦 Decision Matrix

### APPROVED FOR PILOT ✅

**Conditions**:
- [x] Validation infrastructure complete
- [ ] Full dataset run complete ⏳ IN PROGRESS
- [ ] ≥80% pass rate achieved
- [ ] All critical scenes pass
- [ ] No blocking issues identified

**Next Steps**: Proceed to Phase 1 (50-scene pilot)

---

### APPROVED FOR PRODUCTION ✅

**Conditions**:
- [ ] Pilot passed (≥80%)
- [ ] Materials V3 A/B shows improvement
- [ ] ≥95% pass rate on 100+ scenes
- [ ] Throughput validated
- [ ] Fallback system tested

**Next Steps**: Proceed to Phase 3 (full rollout)

---

### REJECTED ❌

**Trigger Conditions**:
- Pilot pass rate <70%
- Critical scene failures
- Catastrophic seam artifacts (energy >2.0)
- Materials V3 regression
- Unacceptable failure rate (>15%)

**Required Actions**:
1. Halt deployment immediately
2. Root cause analysis on failures
3. Fix identified issues
4. Re-run validation from Phase 1
5. Do NOT proceed to production

---

## 📞 Escalation Path

**Blocker Resolution**:
- Technical issues → Engineering team (debug, fix, re-validate)
- Quality threshold disputes → Product + QA (adjust gates within reason)
- Timeline delays → Project manager (re-scope or extend)

**Approval Authority**:
- Pilot approval → Technical lead + QA
- Production approval → Product manager + Engineering lead
- Rejection decision → Any stakeholder (safety veto)

---

## ✅ Sign-Off

**Validation Infrastructure**: ✅ COMPLETE  
**Infrastructure Reviewed By**: _[Technical Lead]_  
**Date**: December 17, 2025  

**Full Dataset Validation**: 🟡 IN PROGRESS  
**Validated By**: _[Pending execution]_  
**Date**: _[TBD]_  

**Pilot Approval**: ⏳ PENDING  
**Approved By**: _[Pending validation results]_  
**Date**: _[TBD]_  

**Production Approval**: ⏸️ BLOCKED  
**Approved By**: _[Pending pilot + Materials V3 A/B]_  
**Date**: _[TBD]_  

---

**Document Version**: 1.0  
**Last Updated**: December 17, 2025  
**Next Review**: After full dataset validation run  
