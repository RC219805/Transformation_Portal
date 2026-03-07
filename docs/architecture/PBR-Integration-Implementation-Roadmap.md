# PBR Integration - Implementation Roadmap

**Status:** Proposed Architecture
**Owner:** Transformation Portal Architect
**Date:** 2026-01-30
**Target Completion:** 6 weeks (Phases 1-3)

---

## Executive Summary

**Problem:** PBR module is production-ready but repository has critical fragmentation:
- 44 depth-related files across 3 modules
- 5 duplicate DeviceType enums
- 2 duplicate DepthConfig classes
- 2 duplicate DepthEstimator classes

**Solution:** Consolidate all depth processing into single canonical module with integrated PBR support.

**Impact:**
- ✅ File reduction: 44 → 25 files (~45%)
- ✅ Single source of truth for all depth-related classes
- ✅ Clean public API surface
- ✅ PBR generation as optional, configurable feature
- ✅ 6-month backward compatibility window
- ✅ No breaking changes until v2.0.0

---

## Phase 1: Foundation (Weeks 1-2)

### Week 1: Core Infrastructure

#### Deliverables
- [ ] Create `src/transformation_portal/depth_canonical/` directory structure
- [ ] Implement `depth_canonical/config.py`
  - [ ] Canonical `DeviceType` enum
  - [ ] Unified `ModelVariant` enum (DA2 + DA3)
  - [ ] `PBRConfig` dataclass (frozen)
  - [ ] `ProcessingConfig` dataclass
  - [ ] `DepthConfig` dataclass with preset loading
- [ ] Implement `depth_canonical/device.py`
  - [ ] Device detection logic
  - [ ] CPU/CUDA/MPS/CoreML support
- [ ] Migrate PBR module
  - [ ] Copy `lux_depth_v3/pbr.py` → `depth_canonical/processing/pbr.py`
  - [ ] Copy `lux_depth_v3/pbr_writer.py` → `depth_canonical/io/pbr_writer.py`
  - [ ] No changes to implementation (already tested)

#### Tests
- [ ] `tests/test_depth_canonical_config.py`
  - [ ] Test DepthConfig initialization
  - [ ] Test preset loading from YAML
  - [ ] Test PBRConfig frozen dataclass
  - [ ] Test config validation
- [ ] `tests/test_depth_canonical_device.py`
  - [ ] Test device detection
  - [ ] Test device fallback logic
- [ ] `tests/test_depth_canonical_pbr.py`
  - [ ] Port all 13 tests from `test_pbr.py`
  - [ ] Verify no regressions

#### Acceptance Criteria
- ✅ All config classes have single canonical definition
- ✅ PBR module passes all 13 existing tests
- ✅ YAML preset loading works
- ✅ No external dependencies on old modules

---

### Week 2: Model and Processing Migration

#### Deliverables
- [ ] Implement `depth_canonical/models/base.py`
  - [ ] `DepthEstimator` abstract base class
  - [ ] Shared inference interface
- [ ] Migrate DA3 model
  - [ ] Copy `lux_depth_v3/da3_model_backend.py` → `depth_canonical/models/depth_anything_v3.py`
  - [ ] Copy `lux_depth_v3/inference.py` → `depth_canonical/processing/inference.py`
  - [ ] Adapt imports to new structure
- [ ] Migrate DA2 model
  - [ ] Copy `depth/models/depth_anything_v2.py` → `depth_canonical/models/depth_anything_v2.py`
  - [ ] Adapt to canonical interface
- [ ] Migrate postprocessing
  - [ ] Copy `lux_depth_v3/postprocessing.py` → `depth_canonical/processing/postprocessing.py`
- [ ] Migrate processors
  - [ ] Copy `depth/processors/zone_tone_mapping.py` → `depth_canonical/processing/zone_mapping.py`
  - [ ] Copy `depth/processors/depth_aware_denoise.py` → `depth_canonical/processing/denoise.py`
  - [ ] Copy `depth/processors/atmospheric_effects.py` → `depth_canonical/processing/atmospheric.py`
- [ ] Implement I/O layer
  - [ ] Copy `lux_depth_v3/depth_writer.py` → `depth_canonical/io/depth_writer.py`
  - [ ] Copy `depth/utils/cache.py` → `depth_canonical/io/cache.py`
  - [ ] Unified caching strategy

#### Tests
- [ ] `tests/test_depth_canonical_models.py`
  - [ ] Test DA3 inference
  - [ ] Test DA2 inference
  - [ ] Test model loading
  - [ ] Test device assignment
- [ ] `tests/test_depth_canonical_processing.py`
  - [ ] Test postprocessing pipeline
  - [ ] Test zone mapping
  - [ ] Test denoising
  - [ ] Test atmospheric effects
- [ ] `tests/test_depth_canonical_io.py`
  - [ ] Test depth writer atomic operations
  - [ ] Test PBR writer atomic operations
  - [ ] Test cache hit/miss
  - [ ] Test cache eviction

#### Acceptance Criteria
- ✅ DA2 and DA3 models work through unified interface
- ✅ All processors ported and tested
- ✅ Atomic write operations verified
- ✅ Cache provides 10-20x speedup on repeated images

---

## Phase 2: Pipeline Integration (Weeks 3-4)

### Week 3: Unified Pipeline Orchestrator

#### Deliverables
- [ ] Implement `depth_canonical/pipeline.py`
  - [ ] `DepthPipeline` class
  - [ ] `process_image()` method
  - [ ] `batch_process()` method
  - [ ] PBR generation integration (optional flag)
  - [ ] Progress tracking with tqdm
  - [ ] Error handling and logging
- [ ] Implement `depth_canonical/__init__.py`
  - [ ] Public API exports
  - [ ] Version info
  - [ ] Documentation strings
- [ ] Implement security layer
  - [ ] Copy `lux_depth_v3/security.py` → `depth_canonical/security/validation.py`
  - [ ] Path sanitization
  - [ ] Input validation

#### Tests
- [ ] `tests/test_depth_canonical_pipeline.py`
  - [ ] Test single image processing
  - [ ] Test batch processing
  - [ ] Test PBR enabled/disabled
  - [ ] Test caching in pipeline
  - [ ] Test error handling
  - [ ] Test progress tracking
- [ ] `tests/test_depth_canonical_security.py`
  - [ ] Test path traversal prevention
  - [ ] Test filename sanitization
  - [ ] Test atomic write failures

#### Acceptance Criteria
- ✅ Pipeline processes images end-to-end
- ✅ PBR generation is optional and configurable
- ✅ Batch processing handles errors gracefully
- ✅ Security validation passes all tests

---

### Week 4: CLI and Existing Pipeline Updates

#### Deliverables
- [ ] Create `cli/depth_process.py`
  - [ ] Typer CLI application
  - [ ] Single image mode
  - [ ] Batch directory mode
  - [ ] Preset selection
  - [ ] PBR flags and configuration
  - [ ] Progress output
- [ ] Update existing pipelines
  - [ ] `pipelines/lux_render_pipeline.py` → use `depth_canonical`
  - [ ] `pipelines/unified_luxury_pipeline.py` → use `depth_canonical`
  - [ ] `luxury_tiff_batch_processor.py` → use `depth_canonical`
- [ ] Create YAML presets
  - [ ] `config/architectural_interior_pbr.yaml`
  - [ ] `config/architectural_exterior_pbr.yaml`
  - [ ] `config/luxury_estate_pbr.yaml`
  - [ ] Update existing presets to include PBR section

#### Tests
- [ ] `tests/test_depth_canonical_cli.py`
  - [ ] Test CLI argument parsing
  - [ ] Test single image mode
  - [ ] Test batch mode
  - [ ] Test preset loading
  - [ ] Test PBR flag handling
- [ ] `tests/integration/test_pipeline_integration.py`
  - [ ] Test Lux Render Pipeline integration
  - [ ] Test Unified Luxury Pipeline integration
  - [ ] Test TIFF batch processor integration
  - [ ] Test end-to-end workflows

#### Acceptance Criteria
- ✅ CLI tool is functional and user-friendly
- ✅ Existing pipelines use canonical module
- ✅ YAML presets work correctly
- ✅ Integration tests pass

---

## Phase 3: Deprecation and Migration (Weeks 5-6)

### Week 5: Deprecation Shims

#### Deliverables
- [ ] Add deprecation warnings to old modules
  - [ ] `depth/__init__.py` with DeprecationWarning
  - [ ] `lux_depth_v3/__init__.py` with DeprecationWarning
  - [ ] `depth_intelligence/__init__.py` with DeprecationWarning
- [ ] Create compatibility shims
  - [ ] Re-export canonical classes from old modules
  - [ ] Map old class names to new ones
  - [ ] Ensure backward compatibility
- [ ] Update documentation
  - [ ] `docs/migration/DEPTH_MODULE_MIGRATION.md`
  - [ ] `README.md` with deprecation notices
  - [ ] `docs/architecture/ARCHITECTURE.md` with new structure
  - [ ] `docs/depth_pipeline/DEPTH_PIPELINE_README.md` rewrite
- [ ] Create migration guide
  - [ ] Before/after code examples
  - [ ] Preset migration instructions
  - [ ] Common pitfalls
  - [ ] FAQ

#### Tests
- [ ] `tests/test_deprecation_warnings.py`
  - [ ] Verify DeprecationWarning is raised
  - [ ] Test shim compatibility
  - [ ] Test old imports still work
- [ ] Update existing tests
  - [ ] Suppress deprecation warnings in legacy tests
  - [ ] Add migration tests

#### Acceptance Criteria
- ✅ Deprecation warnings appear when using old modules
- ✅ Old imports still work via shims
- ✅ Migration guide is comprehensive
- ✅ Documentation is updated

---

### Week 6: Validation and CI Enforcement

#### Deliverables
- [ ] Performance benchmarking
  - [ ] Benchmark depth estimation (DA2/DA3)
  - [ ] Benchmark PBR generation
  - [ ] Benchmark end-to-end pipeline
  - [ ] Compare against current implementation
  - [ ] Document results
- [ ] CI/CD updates
  - [ ] Add `.github/workflows/depth_canonical_tests.yml`
  - [ ] Add pre-commit hook for banned imports
  - [ ] Add import linting script
  - [ ] Update existing workflows
- [ ] Security audit
  - [ ] Review all file I/O operations
  - [ ] Review path handling
  - [ ] Review configuration validation
  - [ ] CodeQL scan
- [ ] Final integration testing
  - [ ] Run full test suite
  - [ ] Test on production data
  - [ ] Performance regression testing
  - [ ] Memory profiling

#### Tests
- [ ] `tests/performance/test_depth_canonical_performance.py`
  - [ ] Benchmark depth estimation
  - [ ] Benchmark PBR generation
  - [ ] Benchmark caching
  - [ ] Benchmark batch processing
- [ ] `tests/security/test_depth_canonical_security.py`
  - [ ] Path traversal tests
  - [ ] Atomic write failure tests
  - [ ] Config injection tests

#### Acceptance Criteria
- ✅ Performance meets or exceeds targets
- ✅ CI gates enforce architecture rules
- ✅ Security audit passes
- ✅ All tests pass (unit + integration + performance)
- ✅ Memory usage is acceptable

---

---

## Phase 4+: Post-v2.0 Optimization Backlog

**Timeline:** Months 7-12 (after v2.0.0 release)
**Trigger:** Production telemetry and user feedback from v2.0 deployment

### Performance Optimizations (Priority 1)

#### P1.1: Parallel PBR Processing
- **Goal:** 2-3x speedup for multi-core systems
- **Deliverable:** `batch_generate_pbr_parallel()` with ProcessPoolExecutor
- **Risk:** Memory usage (N cores × 2GB per image)
- **Acceptance:** >50% throughput improvement without OOM

#### P1.2: GPU-Accelerated Convolutions
- **Goal:** 5-10x speedup for CUDA GPUs
- **Deliverable:** CuPy-based normal/AO generation with NumPy fallback
- **Risk:** New dependency, GPU assumptions
- **Acceptance:** Optional feature, graceful CPU fallback

#### P1.3: Disk Cache for PBR Maps
- **Goal:** Near-instant regeneration
- **Deliverable:** SHA256-based disk cache with eviction policy
- **Risk:** Cache poisoning, disk space growth
- **Acceptance:** Security audit passes

### Model Enhancements (Priority 2)

#### P2.1: ONNX Inference Backend
- **Goal:** 20-30% speedup, better cross-platform compatibility
- **Deliverable:** `ONNXInferenceEngine` with accuracy validation
- **Risk:** Model conversion accuracy
- **Acceptance:** MSE < 0.01 vs. PyTorch

#### P2.2: Streaming Depth Estimation
- **Goal:** Real-time video depth processing
- **Deliverable:** Temporal coherence for 30fps 4K video
- **Risk:** Temporal jitter
- **Acceptance:** <33ms per frame

### Advanced Features (Priority 3)

#### P3.1: Adaptive Preset Selection
- **Goal:** Automatic preset selection based on scene classification
- **Deliverable:** Scene classifier with 90%+ accuracy
- **Risk:** Misclassification degrades results
- **Acceptance:** User study validation

#### P3.2: Progressive PBR Generation
- **Goal:** Low-res preview with progressive refinement
- **Deliverable:** Multi-resolution PBR generation (256px → 4K)
- **Risk:** UI complexity
- **Acceptance:** User testing shows improved perceived performance

---

## Deprecation Timeline and Contingency Plan

### Standard Timeline

| Milestone | Date (est.) | Actions |
|-----------|-------------|---------|
| **v1.8 (Phase 3 Complete)** | Q1 2026 | Deprecation warnings enabled, shims active |
| **v1.9** | Q2 2026 | Final reminder warnings, migration tooling validated |
| **v2.0** | Q3 2026 | Old modules removed, `depth_canonical` only |

**Default Deprecation Period:** 6 months (v1.8 → v2.0)

### Contingency: Extended Support

**Extension Criteria (evaluated at Month 5):**

1. **User Adoption:** >20% of users still on deprecated APIs
   - **Metric:** Import warnings logged in telemetry (opt-in)
   - **Threshold:** If >20% of active installs import old modules

2. **Critical Bugs:** Major migration tooling issues discovered
   - **Examples:** Config migration failures, shim incompatibilities
   - **Threshold:** >5 high-severity migration bugs reported

3. **Client Requests:** Major clients/partners request additional time
   - **Threshold:** >3 enterprise users with active support contracts

**Extension Process:**

1. **Decision Point:** Month 5 (1 month before v2.0 release)
2. **Owner:** Product Owner + Architect (joint decision)
3. **Announcement:** 30 days before scheduled v2.0 release
4. **Maximum Extension:** +3 months (total 9 months deprecation)
5. **Hard Deadline:** No extensions beyond v2.1 (Month 9)

**Extended Timeline (if activated):**

| Milestone | Date (est.) | Actions |
|-----------|-------------|---------|
| **v2.0** | Q3 2026 | Deprecation warnings upgraded to FutureWarning (louder) |
| **v2.0.1-2.0.3** | Q3-Q4 2026 | Bug fixes only, no new features |
| **v2.1** | Q4 2026 | **Final removal** of `depth/` and `lux_depth_v3/` |

**Communication Plan (if extension activated):**

- GitHub issue announcement with rationale
- Update README.md with new timeline
- Email notification to known enterprise users
- Deprecation warning messages updated with new deadline

**Guardrails:**

- ❌ No new features in deprecated modules (security fixes only)
- ❌ No extending beyond v2.1 under any circumstances
- ✅ Migration tooling improvements allowed during extension
- ✅ Documentation updates to clarify new timeline

---

## Post-Merge Monitoring

### Week 1-4 Monitoring (Immediately Post-v1.8)

**Metrics:**
- Import warning frequency (from telemetry, if enabled)
- GitHub issues related to migration
- CI test pass rate for `depth_canonical` tests
- Performance regression vs. baseline (±5% acceptable)

**Action Thresholds:**
- **>10 migration issues:** Hotfix release within 48 hours
- **>20% performance regression:** Immediate investigation, rollback if needed
- **CI failure rate >5%:** Block new merges until fixed

### Month 2-5 Monitoring (Pre-v2.0)

**Metrics:**
- Deprecation warning suppression rate (users actively silencing warnings)
- Documentation page views (`DEPTH_MODULE_MIGRATION.md`)
- Support ticket volume (migration-related)
- Community forum posts about migration

**Action Thresholds:**
- **High suppression rate (>50%):** Indicates warnings are too noisy, refine messaging
- **Low documentation views (<100):** Marketing/communication issue, increase visibility
- **High support volume (>15 tickets/month):** Improve migration guide, add FAQ

### Post-v2.0 Monitoring

**Metrics:**
- Adoption rate of `depth_canonical` (100% expected)
- Performance metrics vs. roadmap targets
- User satisfaction (GitHub stars, community feedback)
- Bug reports (critical vs. minor)
  - [ ] Delete `src/transformation_portal/depth_intelligence/`
- [ ] Remove compatibility shims
- [ ] Update all imports across codebase
- [ ] Update documentation to remove deprecation notices
- [ ] Release v2.0.0 with breaking changes
- [ ] Announce deprecation removals

### Acceptance Criteria
- ✅ No deprecated modules remain
- ✅ All imports use canonical module
- ✅ Documentation updated
- ✅ Release notes comprehensive

---

## Risk Management

### Critical Risks

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| Import breakage in external tools | Deprecation shims + 6-month window | Architect | Planned |
| Performance regression | Benchmark before/after + profiling | Specialist | Planned |
| ML model compatibility | Keep wrappers unchanged + test matrix | Specialist | Planned |
| Config migration failures | Validation + preset tests | Specialist | Planned |
| CI/CD disruption | Parallel CI + feature flags | Architect | Planned |

### Medium Risks

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| Cache key collisions | SHA256 hashing + config fingerprint | Specialist | Planned |
| Preset backward compatibility | Version field in YAML | Architect | Planned |
| Documentation drift | Update docs in same PR as code | Specialist | Planned |

---

## Success Metrics

### Phase 1-3 (6 weeks)
- ✅ File count reduction: 44 → 25 files
- ✅ Test coverage: >90% for canonical module
- ✅ Zero breaking changes
- ✅ Performance: Same or better than current
- ✅ Security: Pass all validation tests

### Post-Launch (v1.x)
- ✅ Deprecation warning adoption rate >80%
- ✅ Migration guide usage (tracked via docs analytics)
- ✅ Production stability (no depth-related regressions)
- ✅ User satisfaction (feedback collection)

### v2.0.0
- ✅ 100% migration to canonical module
- ✅ Deprecated modules removed
- ✅ Documentation up-to-date
- ✅ Community support for migration

---

## Resource Requirements

### Development Time
- **Architect:** 10 hours (reviews, decisions, ADR maintenance)
- **Specialist:** 80 hours (implementation, testing, documentation)
- **Total:** 90 hours over 6 weeks

### Infrastructure
- CI/CD capacity: +10% for additional test jobs
- Storage: Minimal (code reorganization)
- Compute: Benchmark runs on standard CI runners

### External Dependencies
- No new external dependencies
- No model weight migrations
- No breaking changes to PyPI dependencies

---

## Communication Plan

### Internal (Team)
- **Week 0:** Present ADR to team, collect feedback
- **Week 2:** Progress update, demo config system
- **Week 4:** Progress update, demo CLI tool
- **Week 6:** Final review, launch decision

### External (Users)
- **Week 6:** Announcement in README and release notes
- **Week 6:** Blog post with migration guide
- **Ongoing:** Support on GitHub Issues
- **v2.0.0:** Advance notice (3 months before removal)

---

## Approval Checklist

- [ ] Architect approval of ADR-001
- [ ] Security implications assessed
- [ ] CI enforcement strategy approved
- [ ] Migration plan approved
- [ ] Backward compatibility strategy approved
- [ ] Performance benchmarks established
- [ ] Resource allocation confirmed
- [ ] Team agreement on timeline

**Next Steps:**
1. Architect reviews ADR-001 and visual architecture
2. Architect provides explicit approval or requests changes
3. Specialist begins Phase 1 implementation
4. Weekly progress reviews with Architect

---

## References

- **ADR:** `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
- **Visual Architecture:** `docs/architecture/PBR-Integration-Visual-Architecture.md`
- **Current PBR Module:** `src/transformation_portal/lux_depth_v3/pbr.py`
- **Current PBR Tests:** `tests/test_pbr.py` (13/13 passing)
- **Governance Policy:** `docs/architecture/agent_governance.md`

---

**Status:** Awaiting Architect Approval
**Last Updated:** 2026-01-30
