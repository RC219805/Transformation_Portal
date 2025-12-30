# DA3 Evaluation Decision Record

**Date**: 2025-12-19
**Decision**: **DEFER** Depth Anything 3 (DA3) for production
**Status**: Final
**Decider**: Transformation Portal Architect

---

## Context

Evaluated DA3-Large-1.1 vs DA2-Large-hf (baseline) for production depth estimation in luxury real estate architectural rendering pipeline.

**Evaluation method**: Controlled A/B validation against frozen v1.0 baseline (46 images, texture + structure scenes)

---

## Decision

**DEFER DA3-Large-1.1 for current production cycle.**

**Production model**: DA2-Large-hf (Depth-Anything-V2-Large-hf)

---

## Rationale

### 1. Validation Results

| Metric | DA3-Large-1.1 | DA2-Large-hf | Decision |
|--------|---------------|--------------|----------|
| **Overall Lenient Pass** | 13.0% (6/46) | **84.8%** (39/46) | ❌ FAIL |
| **Texture Scenes** | ~16% | **97.4%** (37/38) | ❌ FAIL |
| **Structure Scenes** | ~0% | 25.0% (2/8) | ❌ FAIL |
| **Edge F1 (median)** | 0.09-0.22 | **0.26-0.53** | ❌ FAIL |

**Conclusion**: DA3 fails production quality gates (71.8% regression vs baseline)

### 2. Root Cause: Metric Incompatibility (NOT Model Quality)

**DA3's strengths** (validated by academic benchmarks):
- State-of-the-art metric depth accuracy (AbsRel, RMSE, δ₁)
- Superior geometric generalization
- Transformer architecture with depth-ray representations
- 23-25% improvement over VGGT on standard benchmarks

**Production requirement** (architectural rendering):
- **Edge-preserving depth** for structure/texture boundaries
- Sharp architectural edges (buildings, windows, facades)
- Edge F1, chamfer distance, structural fidelity

**Mismatch**: DA3 optimizes global geometric metrics, not edge fidelity. Different evaluation targets → different outcomes.

### 3. Engineering Trade-offs

**To make DA3 work would require**:
- Ground-truth depth collection (COLMAP reconstruction: 5-9 hours)
- Fine-tuning with edge-aware losses (training: 8-16 hours)
- Custom quality gate calibration (2-4 hours)
- Validation infrastructure (2-3 hours)
- **Total**: 17-32 hours minimum

**For uncertain outcome**: No guarantee DA3 matches edge fidelity after fine-tuning

**Current proven solution (DA2)**:
- 84.8% pass rate (validated)
- 97.4% texture scenes (near-perfect)
- Zero additional cost
- **Production ready NOW**

**Decision velocity principle**: Ship proven solution, defer speculative improvements

---

## What This Decision Does NOT Mean

### ❌ DA3 is "Bad"
DA3 is state-of-the-art for metric depth estimation in academic benchmarks. Our validation simply shows it's **incompatible with current edge-quality gates**.

### ❌ DA3 is "Rejected Forever"
**DEFER** means "evaluate later when conditions change":
- We have ground-truth depth datasets
- Business requires metric depth (3D reconstruction, pose estimation)
- We can invest 2-3 weeks in fine-tuning + calibration
- Validation framework includes standard depth metrics (AbsRel, δ₁, RMSE)

### ❌ Validation Framework Failed
**Validation worked perfectly** - it proved DA3 doesn't meet current architectural edge-quality requirements. That's success, not failure.

---

## Lessons Learned

### 1. Validation-First Methodology Works
- Froze baseline before comparing (v1.0-validation-baseline)
- Controlled A/B comparison (same images, same metrics)
- Objective decision criteria (pass/fail thresholds)
- **Result**: Definitive answer in 11 hours (vs weeks of speculation)

### 2. Benchmark Performance ≠ Production Readiness
- DA3's superior AbsRel/RMSE/δ metrics don't address our edge fidelity bottleneck
- Production requirements (edge quality) differ from research metrics (global geometry)
- Model selection must align with business requirements, not just benchmark rankings

### 3. Different Models Optimize Different Targets
- DA2: Trained for sharp edges, architectural details (convolutional backbone)
- DA3: Trained for metric depth accuracy (transformer, depth-ray representations)
- Neither is "better" universally - depends on evaluation target

### 4. Engineering Efficiency Over Exploration
- Decision velocity: 11 hours to definitive answer
- Avoided 17-32 hour speculative fine-tuning
- Clear next step: structure input-size sweep (proven approach, 6 hours, high ROI)

---

## Future Evaluation Criteria

DA3 should be reconsidered when:

1. **Ground truth available**: LiDAR scans, multi-view stereo, or annotated depth
2. **Business needs metric depth**: 3D reconstruction, pose estimation, spatial measurements
3. **Time available**: 2-3 week fine-tuning + calibration cycle acceptable
4. **Validation expanded**: Standard depth metrics (AbsRel, δ₁, RMSE) added to gates
5. **Custom fine-tuning**: Edge-aware losses, architectural domain adaptation

**Not before**: All 5 conditions met

---

## Production Recommendation

### Immediate (Current Sprint)
- **Model**: DA2-Large-hf
- **Quality**: 84.8% validated
- **Status**: Production ready

### Next Optimization (Follow-up Sprint)
- **Goal**: Improve structure scenes (25% → 60%+)
- **Approach**: Input-size sweep (518px → 1022px for structure scenes)
- **Effort**: 6 hours
- **Risk**: Low (validated approach)
- **ROI**: High (direct bottleneck fix)

### Future Investigation
- **DA3 fine-tuning**: When conditions permit (see criteria above)
- **Materials V3**: After structure improvement validated
- **Multi-model ensemble**: DA2 (edges) + DA3 (metric) fusion

---

## References

**Validation artifacts**:
- Baseline: `validation_v1_baseline_pack/` (tag: `v1.0-validation-baseline`)
- DA3 results: `outputs/da3_gate_fix_test/`
- Decision report: `outputs/da3_gate_fix_test/DA3_DECISION_REPORT.md`

**Session summaries**:
- `SESSION_END_SUMMARY_2025-12-19.md`
- `PHASE1_BASELINE_FREEZE_COMPLETE.md`
- `STRATEGIC_PRIORITY_DECISION.md`

**Academic references**:
- DA3 benchmark performance: Outperforms VGGT by 23-25% on geometry tasks
- Transformer depth estimation: Depth-ray representations, teacher-student training
- Standard metrics: AbsRel, RMSE, δ₁ (established in depth estimation literature)

---

## Sign-off

**Decision**: DEFER DA3, SHIP with DA2
**Confidence**: High
**Risk**: Low
**Timeline**: Production ready immediately

**Next action**: Structure input-size sweep (6 hours, proven ROI)

---

*Decision is final unless material new evidence emerges (e.g., ground truth becomes available, business requirements change).*

---

**Approved**: Transformation Portal Architect
**Date**: 2025-12-19
