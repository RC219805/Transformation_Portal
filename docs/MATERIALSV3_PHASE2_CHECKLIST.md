# MaterialsV3 Phase 2: E2E Validation - Execution Checklist

**Date**: December 21, 2025  
**Phase**: 2 of 4 (E2E Validation & Workflow Integration)  
**Priority**: HIGH  
**Duration**: 3-4 hours active work, 2-3 days calendar time

---

## Phase 2 Objectives

1. ✅ Validate all MaterialsV3 presets end-to-end
2. ✅ Build preset compatibility matrix
3. ✅ Set up automated regression testing
4. ✅ Execute full stress test suite
5. ✅ Establish performance baselines

---

## Prerequisites (From Phase 1)

- [x] **COMPLETE**: Exception handling verified
- [x] **COMPLETE**: Edge case tests passing (13/14)
- [x] **COMPLETE**: Stress tests created
- [ ] **CRITICAL**: Tests integrated into CI/CD ([Guide](docs/MATERIALSV3_CI_INTEGRATION_GUIDE.md))
- [ ] **CRITICAL**: Phase 1 artifacts committed to version control
- [ ] **MEDIUM**: Skipped test resolved or documented

**Status**: **85% Ready** (pending CI/CD integration)

---

## Phase 2 Task Breakdown

### Pre-Flight Checks (1-2 hours)

#### Task 0.1: CI/CD Integration 🔴 CRITICAL
**Priority**: CRITICAL  
**Duration**: 60-90 minutes  
**Owner**: Architect / DevOps  
**Blockers**: None

**Actions**:
- [ ] Review [CI Integration Guide](docs/MATERIALSV3_CI_INTEGRATION_GUIDE.md)
- [ ] Choose integration approach (Quick vs Comprehensive)
- [ ] Create feature branch: `feat/materials-v3-ci-integration`
- [ ] Edit `.github/workflows/ci-consolidated.yml`
  - [ ] Add "Run MaterialsV3 Edge Case Tests" step
  - [ ] Add "Verify Phase 1 Safety" step
- [ ] Create `.github/workflows/materials-v3-stress-nightly.yml`
- [ ] Test CI changes locally (if possible)
- [ ] Commit and push: `git push origin feat/materials-v3-ci-integration`
- [ ] Create PR and verify CI runs
- [ ] Merge PR after verification

**Success Criteria**:
- ✅ Edge case tests run in CI (<5 minutes)
- ✅ Test failures block PR merge
- ✅ Nightly stress test workflow created

**Commands**:
```bash
cd /Users/rc/Transformation_Portal
git checkout -b feat/materials-v3-ci-integration
# ... edit workflows ...
git add .github/workflows/
git commit -m "feat(ci): integrate MaterialsV3 tests into CI/CD pipeline"
git push origin feat/materials-v3-ci-integration
# Create PR via GitHub UI
```

---

#### Task 0.2: Commit Phase 1 Artifacts 🔴 CRITICAL
**Priority**: CRITICAL  
**Duration**: 15 minutes  
**Owner**: Architect  
**Blockers**: None

**Actions**:
- [ ] Review untracked Phase 1 files
- [ ] Commit test files
- [ ] Commit documentation
- [ ] Commit verification script
- [ ] Push to main branch

**Success Criteria**:
- ✅ All Phase 1 deliverables in version control
- ✅ Clean `git status` output

**Commands**:
```bash
cd /Users/rc/Transformation_Portal
git add tests/test_materials_v3_edge_cases.py
git add tests/test_materials_v3_stress.py
git add verify_phase1_safety.py
git add PHASE1_CRITICAL_SAFETY_COMPLETE.md
git add MATERIALSV3_PHASE1_TO_PHASE2_TRANSITION.md
git add docs/MATERIALSV3_CI_INTEGRATION_GUIDE.md
git commit -m "feat(materialsv3): Phase 1 complete - edge cases, stress tests, CI integration"
git push origin main
```

---

#### Task 0.3: Investigate Skipped Test 🟡 MEDIUM
**Priority**: MEDIUM  
**Duration**: 30 minutes  
**Owner**: Specialist  
**Blockers**: None (non-blocking for Phase 2)

**Actions**:
- [ ] Review `test_missing_depth_map_continues` skip reason
- [ ] Option A: Mock depth adapter to enable test
- [ ] Option B: Make test environment-conditional
- [ ] Option C: Document acceptable skip with comment
- [ ] Update test or add skip documentation

**Success Criteria**:
- ✅ Test passing OR
- ✅ Skip reason documented in test file

**Commands**:
```bash
# Review test
vim tests/test_materials_v3_edge_cases.py +82

# Run test in isolation
pytest tests/test_materials_v3_edge_cases.py::TestMaterialsV3EdgeCases::test_missing_depth_map_continues -v

# Document skip if acceptable
# Add comment explaining why skip is acceptable
```

---

### Phase 2 Execution (3-4 hours)

#### Task 2.1: Full Workflow Testing 🔵 CRITICAL
**Priority**: CRITICAL  
**Duration**: 90 minutes  
**Owner**: Specialist  
**Dependencies**: Pre-flight checks complete

**Objective**: Test all 4 MaterialsV3 presets end-to-end

**Test Matrix**:

| Preset | Test Image | Expected Behavior | Status |
|--------|-----------|-------------------|--------|
| INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS | interior_glass.jpg | Glass enhancement | ⬜ |
| INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_METAL | interior_metal.jpg | Metal highlights | ⬜ |
| INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_WOOD | interior_wood.jpg | Wood grain | ⬜ |
| INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_FABRIC | interior_fabric.jpg | Fabric texture | ⬜ |

**Actions**:
- [ ] Prepare test images (4 images, representative of each material)
- [ ] Test Preset 1: GLASS
  - [ ] Run: `lux-depth-v2 --preset INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS --input-dir test_images/glass/ --output-dir output/glass/`
  - [ ] Verify output quality
  - [ ] Check logs for errors
  - [ ] Capture metadata: `output/glass/processing_report.json`
- [ ] Test Preset 2: METAL
  - [ ] Run: `lux-depth-v2 --preset INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_METAL --input-dir test_images/metal/ --output-dir output/metal/`
  - [ ] Verify metal highlights enhanced
  - [ ] Check fallback rate (should be 0%)
- [ ] Test Preset 3: WOOD
  - [ ] Run: `lux-depth-v2 --preset INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_WOOD --input-dir test_images/wood/ --output-dir output/wood/`
  - [ ] Verify wood grain preserved
  - [ ] Check processing time
- [ ] Test Preset 4: FABRIC
  - [ ] Run: `lux-depth-v2 --preset INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_FABRIC --input-dir test_images/fabric/ --output-dir output/fabric/`
  - [ ] Verify fabric texture enhancement
  - [ ] Compare to Materials V2 output
- [ ] Test Fallback Mechanism
  - [ ] Run with corrupted image
  - [ ] Verify graceful fallback to Materials V2
  - [ ] Check metadata contains `fallback: true`

**Success Criteria**:
- ✅ All 4 presets produce valid outputs
- ✅ No crashes or unhandled exceptions
- ✅ Fallback rate <1% for valid images
- ✅ Metadata includes `materials_v3` section
- ✅ Processing time reasonable (<30s per image on CPU)

**Deliverable**: Test results summary (markdown or JSON)

---

#### Task 2.2: Preset Compatibility Matrix 🔵 HIGH
**Priority**: HIGH  
**Duration**: 60 minutes  
**Owner**: Architect + Specialist  
**Dependencies**: Task 2.1 complete

**Objective**: Document MaterialsV3 behavior and compatibility

**Actions**:
- [ ] Create compatibility matrix spreadsheet/markdown
- [ ] Document MaterialsV3 features per preset
- [ ] Test preset combinations (if applicable)
- [ ] Identify known limitations
- [ ] Define supported vs unsupported configurations

**Matrix Structure**:

| Preset | MaterialsV3 Feature | Segmentation Backend | Min Resolution | Max Resolution | Known Issues |
|--------|---------------------|---------------------|----------------|----------------|--------------|
| GLASS | Glass response | ONNX/SegFormer/Heuristic | 256x256 | 8192x8192 | None |
| METAL | Metal highlights | ONNX/SegFormer/Heuristic | 256x256 | 8192x8192 | None |
| WOOD | Wood grain | ONNX/SegFormer/Heuristic | 256x256 | 8192x8192 | None |
| FABRIC | Fabric texture | ONNX/SegFormer/Heuristic | 256x256 | 8192x8192 | None |

**Success Criteria**:
- ✅ Compatibility matrix complete
- ✅ Known limitations documented
- ✅ Supported configurations clearly defined

**Deliverable**: `MATERIALSV3_COMPATIBILITY_MATRIX.md`

---

#### Task 2.3: Regression Testing Infrastructure 🔵 CRITICAL
**Priority**: CRITICAL  
**Duration**: 60 minutes  
**Owner**: Architect / DevOps  
**Dependencies**: Task 2.1 complete

**Objective**: Set up automated regression testing

**Actions**:
- [ ] Create baseline outputs directory: `tests/fixtures/materials_v3_baseline/`
- [ ] Capture baseline outputs for all 4 presets
- [ ] Create regression test script: `tests/test_materials_v3_regression.py`
- [ ] Integrate regression tests into CI
- [ ] Define acceptable drift thresholds (e.g., PSNR, SSIM)

**Regression Test Structure**:
```python
# tests/test_materials_v3_regression.py
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    # ... implementation ...

def test_glass_preset_regression():
    """Verify glass preset output hasn't regressed."""
    baseline = Image.open("tests/fixtures/materials_v3_baseline/glass_output.png")
    current = Image.open("output/glass/test_image.png")
    
    psnr = calculate_psnr(baseline, current)
    assert psnr > 30, f"PSNR regression: {psnr} < 30"
```

**Success Criteria**:
- ✅ Baseline outputs captured
- ✅ Regression tests written
- ✅ Tests integrated into CI
- ✅ Acceptable drift thresholds defined

**Deliverable**: Regression test suite

---

#### Task 2.4: Stress Test Execution 🔵 CRITICAL
**Priority**: CRITICAL  
**Duration**: 30 minutes  
**Owner**: Specialist  
**Dependencies**: None (can run in parallel)

**Objective**: Execute full stress test suite and establish performance baseline

**Actions**:
- [ ] Run stress tests locally:
  ```bash
  pytest tests/test_materials_v3_stress.py -v -m slow --tb=short
  ```
- [ ] Capture performance metrics:
  - [ ] 1000-iteration stability: Pass/Fail, fallback rate, avg time
  - [ ] Batch processing (100 images): Pass/Fail, throughput
  - [ ] Concurrent execution: Pass/Fail, no race conditions
  - [ ] Memory stability: Pass/Fail, max memory growth
- [ ] Document baseline performance
- [ ] Identify performance bottlenecks (if any)
- [ ] Export metrics to JSON:
  ```json
  {
    "test_1000_iteration_stability": {
      "status": "PASSED",
      "fallback_rate": 0.0,
      "avg_time_per_iteration": 0.45,
      "total_time": 450.0
    },
    "test_batch_processing_100_images": {
      "status": "PASSED",
      "throughput": 2.2,
      "total_time": 45.5
    }
  }
  ```

**Success Criteria**:
- ✅ All 7 stress tests passing
- ✅ Fallback rate <1% for synthetic images
- ✅ No memory leaks detected
- ✅ Performance baseline documented

**Deliverable**: Stress test results JSON + summary report

---

### Phase 2 Completion (30 minutes)

#### Task 2.5: Phase 2 Completion Report 📝 MEDIUM
**Priority**: MEDIUM  
**Duration**: 30 minutes  
**Owner**: Architect  
**Dependencies**: Tasks 2.1-2.4 complete

**Actions**:
- [ ] Compile Phase 2 results
- [ ] Document findings and issues
- [ ] Update MaterialsV3 roadmap status
- [ ] Create Phase 3 kickoff document
- [ ] Update star rating (4.5 → 4.75)

**Deliverable**: `PHASE2_FINAL_STATUS.md`

---

## Success Criteria (Phase 2 Complete)

### Must-Have (Critical)
- [ ] All 4 MaterialsV3 presets validated end-to-end
- [ ] Fallback mechanism tested and working
- [ ] Regression tests automated
- [ ] Stress tests passing (7/7)
- [ ] No crashes or unhandled exceptions
- [ ] Performance baseline established

### Nice-to-Have (Optional)
- [ ] Compatibility matrix complete and reviewed
- [ ] Performance optimization opportunities identified
- [ ] Phase 3 kickoff document ready

**Phase 2 Completion Threshold**: All critical items + regression tests passing

---

## Risk Management

### Risks Identified
| Risk | Severity | Mitigation |
|------|----------|------------|
| Preset incompatibilities | Medium | Fallback to V2, document limitations |
| Stress tests reveal issues | High | Fix before Phase 3 |
| Regression drift | Medium | Adjust thresholds, investigate |
| CI integration delays | Low | Parallel E2E testing |

### Contingency Plans
- **If stress tests fail**: Investigate, fix, re-run (add 1-2 hours)
- **If presets incompatible**: Document, disable affected presets, proceed
- **If regression tests fail**: Review baselines, adjust thresholds

---

## Timeline

| Day | Tasks | Duration | Owner |
|-----|-------|----------|-------|
| **Day 1** | Pre-flight checks (0.1-0.3) | 2 hours | Architect |
| **Day 2** | Full workflow testing (2.1) | 90 min | Specialist |
| **Day 2** | Stress test execution (2.4) | 30 min | Specialist |
| **Day 2-3** | Compatibility matrix (2.2) | 60 min | Architect + Specialist |
| **Day 3** | Regression testing (2.3) | 60 min | Architect |
| **Day 3** | Completion report (2.5) | 30 min | Architect |

**Total**: 2-3 days calendar time, 4-5 hours active work

---

## Phase 2 → Phase 3 Handoff

**Phase 3 Prerequisites**:
- [x] Phase 2 all critical tasks complete
- [ ] Regression tests passing in CI
- [ ] Performance baseline documented
- [ ] Known limitations documented

**Phase 3 Objectives**:
- Canary deployment (5% traffic)
- Production monitoring setup
- Real-world validation

**Estimated Phase 3 Start**: Day 4 (after Phase 2 complete)

---

## Document Version

**Version**: 1.0  
**Last Updated**: December 21, 2025  
**Status**: READY FOR EXECUTION  
**Next Review**: After Phase 2 completion
