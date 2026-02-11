# APEX Research Implementation Roadmap

**Date:** 2026-02-10
**Authority:** Transformation Portal Architect
**Related:** ADR-025 (APEX Research Workflow Architecture)

---

## Overview

This roadmap defines the phased implementation strategy for APEX Research, broken into 4 independent PRs to minimize risk and enable incremental validation.

**Design Principle:** Each PR must be independently deployable and reversible. No PR should block commercial APEX workflows.

---

## Phase 1: Core Infrastructure (PR #1)

**Goal:** Establish APEX Research preset configuration and license enforcement foundation.

**Estimated Effort:** 2-3 days
**Risk:** LOW (minimal code changes, mostly configuration)

### Tasks

#### Configuration Files
- [ ] Create `config/presets/apex_research.yaml` (stable preset)
- [ ] Create `config/presets/apex_research_canary.yaml` (canary preset)
- [ ] Create `config/presets/apex_research_experimental.yaml` (experimental preset)
- [ ] Validate YAML syntax and schema compliance

#### Compliance Enforcement
- [ ] Implement `src/transformation_portal/compliance/validate_apex_research.py`
  - Preset license marker validation
  - Required compliance flags check
  - Model license verification
- [ ] Add `tier` field to `PerformanceCapsule` (backward compatible default)
- [ ] Add `license_mode` field to `PerformanceCapsule` (default: "commercial")

#### CI Integration
- [ ] Create `.github/workflows/apex_research_compliance.yml`
  - Validate preset license markers
  - Verify required compliance flags
  - Run compliance unit tests
- [ ] Update `.github/workflows/ci.yml` to include compliance checks

#### Unit Tests
- [ ] Create `tests/compliance/test_apex_research_enforcement.py`
  - Test `apex_research` preset requires `non_commercial_ok=True`
  - Test Depth Pro requires `accept_apple_depth_pro_research_license=True`
  - Test commercial APEX unaffected by research tier changes
  - Test preset validation script catches missing markers
- [ ] Add contract tests for `PerformanceCapsule` new fields

#### Documentation
- [ ] Update `README.md` with APEX Research section
  - Quality tier comparison table
  - License restrictions summary
  - Quick start example
- [ ] Create `docs/architecture/license_compliance_guide.md`
  - Legal requirements explanation
  - Technical enforcement overview
  - Violation consequences
- [ ] Update `config/presets/README.md` with tier taxonomy

### Acceptance Criteria

✅ **All `apex_research*.yaml` presets pass CI validation**
✅ **License enforcement tests pass (3/3 layers)**
✅ **Loading research preset without `non_commercial_ok=True` raises `LicenseRestrictionError`**
✅ **Commercial APEX workflows work identically (backward compatibility)**
✅ **Documentation clearly explains research tier restrictions**

### Dependencies

**None** (standalone infrastructure changes)

### Rollback Strategy

- Delete preset files
- Remove compliance validation script
- Remove CI workflow
- Revert `PerformanceCapsule` changes (backward compatible, safe)

---

## Phase 2: SAM vit_h Integration (PR #2)

**Goal:** Integrate Segment Anything (SAM vit_h) as research-tier segmentation backend.

**Estimated Effort:** 3-4 days
**Risk:** MEDIUM (new model integration, checkpoint management)

### Tasks

#### Backend Implementation
- [ ] Extend `SegmentationBackend` protocol to support SAM variants
- [ ] Implement `SAMVitHBackend` class
  - Wrap existing `SAMSegmenter` from `src/transformation_portal/lux_depth_v3/sam_segmenter.py`
  - Implement protocol methods: `load()`, `segment()`, `info`
  - Add device auto-detection (MPS > CUDA > CPU)
- [ ] Add backend registry entry for `sam_vit_h`

#### Checkpoint Management
- [ ] Create checkpoint download utility
  - URL: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
  - Size: 2.4GB
  - SHA256 validation (obtain from SAM repo)
- [ ] Add checkpoint validation to preset loading
- [ ] Document checkpoint setup in README

#### Configuration Integration
- [ ] Update `apex_research.yaml` to use `segmentation_backend: sam_vit_h`
- [ ] Add SAM-specific configuration options:
  - `confidence_threshold: 0.85`
  - `points_per_side: 32`
  - `pred_iou_thresh: 0.88`

#### Performance Profiling
- [ ] Benchmark SAM vit_h vs EfficientSAM on synthetic fixtures
  - Inference time (4K input)
  - Memory usage
  - Segmentation quality (IoU metrics)
- [ ] Document performance characteristics in preset comments

#### Unit Tests
- [ ] Create `tests/materials/test_sam_vit_h_backend.py`
  - Protocol compliance tests
  - Checkpoint validation tests
  - Inference shape/dtype contract tests
  - Device placement tests (CPU/MPS/CUDA)
  - Mock backend for fast CI tests
- [ ] Integration test with full pipeline (synthetic 4K image)

#### Documentation
- [ ] Add SAM vit_h setup instructions to README
- [ ] Document checkpoint download and validation process
- [ ] Update preset documentation with segmentation backend options

### Acceptance Criteria

✅ **SAM vit_h backend selectable via `apex_research.yaml` preset**
✅ **Checkpoint SHA256 validation enforced (mismatch blocks execution)**
✅ **Segmentation quality ≥10% IoU improvement over EfficientSAM (on benchmark)**
✅ **Backend works on MPS (Apple Silicon), CUDA, and CPU devices**
✅ **Unit tests pass without downloading checkpoint (mocked)**

### Dependencies

**Phase 1** must be complete (preset infrastructure exists)

### Rollback Strategy

- Remove `SAMVitHBackend` class
- Revert `apex_research.yaml` to use `efficientsam_v2`
- Delete SAM checkpoint from `checkpoints/` directory
- No impact on commercial workflows (SAM integration is opt-in)

---

## Phase 3: Quality Benchmarking (PR #3)

**Goal:** Implement reproducible benchmark suite validating APEX Research superiority over APEX Commercial.

**Estimated Effort:** 4-5 days
**Risk:** MEDIUM (requires synthetic fixture creation, metric design)

### Tasks

#### Benchmark Fixtures
- [ ] Create `tests/fixtures/apex_research_benchmark/` directory structure
  - `architectural_exteriors/` (3-5 synthetic 4K images)
  - `architectural_interiors/` (3-5 synthetic 4K images)
  - `ground_truth/` (reference depth maps, segmentation masks)
- [ ] Generate or source high-quality synthetic images
  - Modern glass facade (edge detail challenge)
  - Historic stone detail (texture preservation)
  - Mixed materials balcony (multi-material boundaries)
  - Luxury kitchen HDR 16-bit (high dynamic range)
- [ ] Create ground truth depth maps (synthetic or LiDAR-derived)

#### Metrics Implementation
- [ ] Create `src/transformation_portal/metrics/apex_research_quality.py`
  - Implement `APEXResearchQualityMetrics` dataclass
  - Depth quality: MAE, RMSE, edge sharpness, boundary precision
  - Segmentation quality: material IoU, confidence scores
  - PBR quality: normal detail score, roughness variance, AO realism
  - Composite score calculation
  - `is_research_grade()` validation method
- [ ] Implement metric computation functions
  - Depth MAE/RMSE vs ground truth
  - Edge sharpness (Sobel gradient analysis)
  - Material IoU (overlap with ground truth masks)
  - Normal map detail (high-frequency FFT power)

#### Benchmark Script
- [ ] Create `scripts/apex_research_benchmark.py`
  - Run APEX Commercial workflow on benchmark fixtures
  - Run APEX Research workflow on benchmark fixtures
  - Compute quality metrics for both
  - Validate research tier ≥10% improvement in ≥3/4 metrics
  - Generate comparison report (JSON + markdown)
- [ ] Add CLI interface with argparse
  - `--input`: benchmark fixtures directory
  - `--output`: results output directory
  - `--ground-truth`: reference data directory

#### CI Integration
- [ ] Create `.github/workflows/apex_research_benchmark.yml`
  - Run benchmark on synthetic fixtures (no large checkpoints in CI initially)
  - Validate quality improvement threshold
  - Upload benchmark artifacts (comparison report)
- [ ] Add benchmark validation gate
  - Fail PR if research tier < 10% improvement
  - Log detailed comparison metrics

#### Unit Tests
- [ ] Create `tests/benchmark/test_apex_research_quality.py`
  - Test metric computation functions (unit tests)
  - Test `is_research_grade()` validation logic
  - Test benchmark script end-to-end (mocked pipelines)
- [ ] Property-based tests for metric stability

#### Documentation
- [ ] Create `docs/apex/apex_research_benchmarking.md`
  - Benchmark methodology
  - Metric definitions
  - Quality validation criteria
  - How to run benchmarks locally
- [ ] Update README with benchmark results (once validated)

### Acceptance Criteria

✅ **Benchmark runs in CI without manual intervention**
✅ **APEX Research demonstrates ≥10% improvement in ≥3/4 metrics**
✅ **Benchmark results reproducible (deterministic)**
✅ **Synthetic fixtures adequate quality (representative of real workflows)**
✅ **Comparison report generated and uploaded as artifact**

### Dependencies

**Phase 1** (presets) and **Phase 2** (SAM vit_h) must be complete

### Rollback Strategy

- Delete benchmark fixtures (large files, can bloat repo)
- Remove benchmark workflow from CI
- Delete metrics module (no dependencies on it)
- Benchmark is validation-only, no impact on runtime code

---

## Phase 4: Enhanced PBR Research Preset (PR #4) — OPTIONAL

**Goal:** Fine-tune PBR parameters specifically for research tier quality (higher AO samples, enhanced normal strength).

**Estimated Effort:** 2-3 days
**Risk:** LOW (configuration tuning, no new code)

### Tasks

#### PBR Preset Creation
- [ ] Create `RESEARCH_PREMIUM` PBR preset in `src/transformation_portal/lux_depth_v3/pbr_presets.py`
  - Normal strength: 1.5 → 2.0
  - AO samples: 128 → 256
  - AO radius: 5.0 → 6.0
  - Roughness modulation: 0.4 → 0.5
- [ ] Update `apex_research.yaml` to use `RESEARCH_PREMIUM` preset

#### A/B Testing
- [ ] Compare research PBR vs standard PBR on benchmark fixtures
  - Normal detail score improvement
  - AO realism improvement
  - Performance overhead (time/memory)
- [ ] Document quality vs performance trade-off

#### Experimental Tuning
- [ ] Add aggressive PBR tuning to `apex_research_experimental.yaml`
  - Multi-scale normal synthesis
  - Material-specific roughness
  - SSAO experimental method

#### Unit Tests
- [ ] Test `RESEARCH_PREMIUM` preset loads correctly
- [ ] Validate PBR parameter ranges (no invalid values)

#### Documentation
- [ ] Document PBR tuning rationale in preset comments
- [ ] Add performance expectations (slower but higher quality)

### Acceptance Criteria

✅ **`RESEARCH_PREMIUM` PBR preset demonstrates ≥5% improvement in normal detail score**
✅ **Performance overhead acceptable (<1.5x slower than standard PBR)**
✅ **No regression in other quality metrics**

### Dependencies

**Phase 3** (benchmarking) must be complete (to validate improvement)

### Rollback Strategy

- Revert `RESEARCH_PREMIUM` preset
- Restore standard PBR parameters in `apex_research.yaml`
- No impact on commercial workflows

---

## Cross-Phase Considerations

### Dependency Pinning

All phases must ensure:
- Model checkpoints pinned via SHA256 validation
- HuggingFace model revisions pinned (not `main` branch)
- Python dependency versions constrained in `requirements/`

### Performance Monitoring

Track performance metrics across all phases:
- Inference time (depth, segmentation, PBR)
- Memory usage (peak, VRAM)
- Output quality (APEX gates, benchmark scores)

### Backward Compatibility

Every PR must maintain:
- Commercial APEX workflows unchanged
- Existing presets work identically
- No breaking changes to public APIs

### Documentation Hygiene

Every PR must include:
- ADR cross-references (if applicable)
- README updates (user-facing changes)
- Docstring updates (code-level changes)
- Migration guide (if behavior changes)

---

## Risk Mitigation

### Risk: Research Tools Underperform

**Mitigation:**
- Phase 3 benchmark validation catches this early
- Can revert to commercial tools without breaking workflows
- Document performance characteristics transparently

### Risk: License Enforcement Bypassed

**Mitigation:**
- Multi-layer enforcement (config + registry + runtime + CI)
- Comprehensive unit tests for all bypass scenarios
- CI validation blocks PRs with missing license markers

### Risk: Large Checkpoint Files Bloat Repository

**Mitigation:**
- Checkpoints live in `checkpoints/` directory (gitignored)
- Documentation provides download instructions
- CI uses mocked backends (no large downloads)
- Optional: Use Git LFS for fixtures (if needed)

### Risk: Benchmark Fixtures Inadequate Quality

**Mitigation:**
- Start with synthetic fixtures (easy to generate)
- Validate synthetic quality with human review
- Optional: Add real-world validation set (not in repo)

---

## Success Metrics

### Phase 1 Success
- ✅ Zero license enforcement bypass tests fail
- ✅ All presets pass CI validation
- ✅ Documentation clarity validated by peer review

### Phase 2 Success
- ✅ SAM vit_h ≥10% IoU improvement over EfficientSAM
- ✅ Checkpoint validation catches SHA256 mismatches
- ✅ Backend works on all supported devices (MPS/CUDA/CPU)

### Phase 3 Success
- ✅ Research tier ≥10% improvement in ≥3/4 metrics
- ✅ Benchmark reproducible across CI runs
- ✅ Comparison report clearly communicates quality gains

### Phase 4 Success (Optional)
- ✅ Research PBR ≥5% improvement in normal detail
- ✅ Performance overhead acceptable (<1.5x slower)

---

## Timeline

**Conservative Estimate:** 12-15 working days across all phases

| Phase | Duration | Dependencies | Status |
|-------|----------|--------------|--------|
| Phase 1: Core Infrastructure | 2-3 days | None | 🟡 Planned |
| Phase 2: SAM vit_h Integration | 3-4 days | Phase 1 | 🟡 Planned |
| Phase 3: Quality Benchmarking | 4-5 days | Phase 1, 2 | 🟡 Planned |
| Phase 4: Enhanced PBR (Optional) | 2-3 days | Phase 3 | 🟡 Planned |

**Aggressive Estimate:** 8-10 working days (if phases parallelized where safe)

**Note:** Phases 1 and 2 can be parallelized if SAM integration doesn't depend on preset infrastructure.

---

## Approval Gates

### Phase 1 Approval
- [ ] Architect: Review preset configuration structure
- [ ] CI: All compliance tests pass
- [ ] Specialist: Implementation complete

### Phase 2 Approval
- [ ] Architect: Review SAM integration approach
- [ ] CI: All backend tests pass
- [ ] Benchmark: SAM vit_h quality validated

### Phase 3 Approval
- [ ] Architect: Review benchmark methodology
- [ ] CI: Benchmark validation passes
- [ ] Quality threshold met (≥10% improvement)

### Phase 4 Approval (Optional)
- [ ] Architect: Review PBR tuning rationale
- [ ] Benchmark: PBR quality improvement validated

---

## Post-Implementation

### Maintenance Plan
- **Quarterly:** Re-run benchmarks to validate quality stability
- **Per Model Release:** Update checkpoints, revalidate benchmarks
- **Per ADR Update:** Ensure implementation matches governance

### Future Enhancements
- Multi-backend depth fusion (experimental → canary → stable)
- LLaVA quality validation (experimental → canary → stable)
- Real-world benchmark validation set (optional)
- APEX Research v2: incorporate next-gen research models

---

**Document History**
- **2026-02-10:** Initial implementation roadmap created
