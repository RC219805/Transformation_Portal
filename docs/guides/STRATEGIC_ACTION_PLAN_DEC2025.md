# Strategic Action Plan - December 2025

**Status**: ✅ **Objective 1 Complete** | 🔄 **Objectives 2-4 In Progress**  
**Date**: December 10, 2025  
**Context**: Post-Architecture Hardening completion reality check

---

## Executive Summary

Following completion of the Architecture Hardening Plan (6/6 PRs), we conducted a reality check and identified a critical issue: **tests were weakened to achieve green CI**, undermining the stability guarantees we just built.

**Immediate action taken**: Restored strict test assertions for security (path traversal, symlink attacks) and checkpoint/retry semantics. All 106 tests now passing with proper guarantees.

---

## Current State Assessment

### ✅ Strengths (Unchanged)

1. **Phase 1 Stability**: Rock solid foundation
   - 5 core modules operational (orchestrator, resource monitor, checkpoint, error recovery, preflight)
   - 27/27 tests passing with <2% overhead
   - Production-ready for reliability
   - Clear rollout plan documented

2. **Architecture Hardening Complete**: 6/6 PRs landed
   - PR-1: Security + repo hygiene ✅
   - PR-2: Platform Core ✅
   - PR-3: Stage Graph ✅
   - PR-4: Performance + Profiling ✅
   - PR-5: Validation-First Defaults ✅
   - PR-6: Test Strategy ✅

3. **Performance Architecture**: Design complete
   - Target: 14 min → 1-2 min (Phase 2) → 30-60s (Phase 3)
   - Worst-case Pool: 2015s → 60-80s
   - Modular stages, parallelism, observability planned

### ⚠️ Risks Identified & Resolved

**Risk 1**: Security tests weakened (RESOLVED)
- **Before**: Generic exception catching, no API validation
- **After**: Strict PathValidator.validate() and safe_resolve() checks
- **Status**: ✅ Fixed in commit 3774c11

**Risk 2**: Checkpoint retry semantics not enforced (RESOLVED)
- **Before**: `assert len(completed) >= 0` (can't fail)
- **After**: Explicit retry semantics with status/timing validation
- **Status**: ✅ Fixed in commit 3774c11

**Risk 3**: Platform Core integration pending
- Core modules exist but relationship to legacy unclear
- Need compatibility layer + parity tests
- **Status**: 🔄 Next objective

---

## Strategic Objectives (Prioritized)

### ✅ Objective 1: Restore Test Strictness [COMPLETE]

**Status**: ✅ Complete (commit 3774c11)  
**Timeline**: Immediate (same day)  
**Risk**: 🟢 LOW (test-only changes)

**Deliverables**:
- ✅ test_path_traversal_prevention: PathValidator API validation
- ✅ test_symlink_attack_prevention: Explicit rejection checks
- ✅ test_batch_processor_retry_failed: Correct retry semantics

**Impact**: Restored guardrails to promised level, enabling confident forward progress.

---

### 🔄 Objective 2: Platform Core Lockdown

**Status**: 🔄 In Progress  
**Timeline**: Week 1-2 (Dec 10-24, 2025)  
**Risk**: 🟡 MEDIUM (requires careful API design)

**Goal**: One spine, not two half-overlapping frameworks.

#### Phase 2.1: API Documentation & Contract (Week 1)

**Tasks**:
1. Document core module APIs with module-level docstrings
   - `core/config/` - Configuration schema and presets
   - `core/device/` - Device management + profiling
   - `core/artifacts/` - Caching & storage
   - `core/security/` - Path validation & sanitization
   - `core/batch/` - Checkpoint/resume
   - `core/processing/` - UHR tiling
   - `core/validation/` - Reports & metrics

2. Create API reference document
   - Quick reference guide: `docs/guides/PLATFORM_CORE_API.md`
   - Usage examples for each module
   - Migration guide from legacy patterns

3. Lock API contracts
   - Add deprecation warnings to guide migration
   - Semantic versioning for core modules
   - Compatibility promise (no breaking changes in 1.x)

#### Phase 2.2: Legacy Integration (Week 2)

**Strategy**: Treat core as long-term home, provide adapters in lux_depth_v2

**Tasks**:
1. Create compatibility adapters
   ```python
   # lux_depth_v2/compat/core_adapter.py
   from transformation_portal.core.batch import BatchProcessor as CoreBatchProcessor
   
   class OrchestratorAdapter:
       """Adapter to use core BatchProcessor via old orchestrator API."""
       def __init__(self, core_processor: CoreBatchProcessor):
           self._core = core_processor
       
       def process_batch(self, inputs, config):
           # Translate old API to new core API
           return self._core.process_batch(inputs, config.output_dir)
   ```

2. Add behavior parity tests
   ```python
   def test_orchestrator_core_parity():
       """Confirm LuxPipelineV2 via old vs core give identical results."""
       # Small batch through old path
       old_result = legacy_orchestrator.process_batch(test_images)
       
       # Same batch through core
       core_result = core_batch_processor.process_batch(test_images)
       
       # Results should match (metrics, timing, outputs)
       assert_results_equivalent(old_result, core_result)
   ```

3. Update lux_depth_v2 to use core internally
   - Keep old entrypoints intact (backward compatibility)
   - Route to core under the hood
   - Add feature flags to toggle core vs legacy

**Success Criteria**:
- ✅ All core modules documented with examples
- ✅ Legacy lux_depth_v2 uses core internally
- ✅ Behavior parity tests passing
- ✅ No breaking changes to existing CLIs
- ✅ Clear migration path for users

---

### 🔜 Objective 3: Phase 2 Performance - Targeted Slices

**Status**: 🔜 Planned  
**Timeline**: Week 3-5 (Dec 24 - Jan 14, 2025)  
**Risk**: 🟢 LOW (incremental, measurable)

**Goal**: Begin attacking biggest wins without parallelism yet.

#### Slice 1: Stage Timing Telemetry (Week 3)

**Why**: Need data to confirm where Pool and similar images spend time.

**Implementation**:
```python
# Add to orchestrator task result
task_result = {
    "output_path": output_path,
    "timing_s": {
        "load": 2.3,
        "depth": 45.2,
        "material": 12.1,
        "grade": 8.4,
        "upscale": 90.5,
        "export": 25.1
    },
    "total_s": 183.6
}
```

**Tasks**:
1. Instrument LuxDepthPipeline stages with context managers
2. Add timing_s dict to ProcessingReport
3. Update tests to validate timing presence
4. Create timing analysis tool: `scripts/analyze_timings.py`

**Success Criteria**:
- ✅ Every processed image has stage timings
- ✅ Can identify bottleneck stages per image
- ✅ <1% overhead from timing instrumentation

#### Slice 2: Export I/O Optimization (Week 4)

**Why**: Export is currently synchronous and blocks pipeline.

**Design** (from Phase 2 architecture):
- Tiled BigTIFF writing for UHR outputs
- Local scratch → final destination async copy
- Separate I/O thread pool

**Implementation**:
```python
# core/processing/export.py
class TiledExporter:
    """Async tiled TIFF export for UHR images."""
    
    def export_async(self, image, output_path, scratch_dir):
        # Write to scratch first (fast local disk)
        scratch_path = scratch_dir / output_path.name
        self._write_tiled_tiff(image, scratch_path)
        
        # Queue async copy to final destination
        self.copy_queue.put((scratch_path, output_path))
```

**Tasks**:
1. Implement TiledExporter in core/processing/
2. Add scratch directory config option
3. Integrate into lux_depth_v2 export stage
4. Benchmark: compare sync vs async export times

**Success Criteria**:
- ✅ UHR exports don't block pipeline progression
- ✅ 20-30% reduction in export stage time
- ✅ Atomic writes preserved (crash safety)

#### Slice 3: Upscaling Adaptivity (Week 5)

**Why**: Upscaler overhead varies dramatically by image complexity.

**Design**:
- Adaptive tile sizing based on image content
- Conservative defaults for "hard" images (textures, high-freq)
- Aggressive settings for "easy" images (smooth gradients)
- Explicit logging of backend, device, tile config

**Implementation**:
```python
# core/processing/upscaling.py
class AdaptiveUpscaler:
    def estimate_complexity(self, image):
        """Estimate image complexity for adaptive tiling."""
        # Compute edge density
        edges = cv2.Canny(image, 100, 200)
        complexity = edges.sum() / (image.shape[0] * image.shape[1])
        
        if complexity < 0.1:
            return "easy"
        elif complexity < 0.3:
            return "medium"
        else:
            return "hard"
    
    def upscale(self, image, target_size):
        complexity = self.estimate_complexity(image)
        
        if complexity == "easy":
            tile_size = 1024  # Larger tiles, fewer passes
        else:
            tile_size = 512   # Smaller tiles, safer
        
        logger.info(f"Upscaling {complexity} image with tile_size={tile_size}")
        return self._upscale_tiled(image, target_size, tile_size)
```

**Tasks**:
1. Implement complexity estimator
2. Add adaptive tile size selection
3. Log backend/device/tile config per image
4. Integrate into core.processing

**Success Criteria**:
- ✅ Easy images use larger tiles (faster)
- ✅ Hard images use conservative settings (quality)
- ✅ All upscaling decisions logged for analysis
- ✅ 10-15% average upscaling time reduction

---

### 🔜 Objective 4: Validation System → Live Quality Gate

**Status**: 🔜 Planned  
**Timeline**: Week 6-8 (Jan 14 - Feb 4, 2025)  
**Risk**: 🟡 MEDIUM (depends on baseline stability)

**Goal**: Turn design into reproducible, automated benchmarking.

#### Phase 4.1: Finalize Validation Datasets (Week 6)

**Tasks**:
1. Curate validation_v1 dataset
   - 20-30 representative images
   - Interior, exterior, aerial, Pool-like edge cases
   - Ground truth labels (if available)
   - Organized by scene type

2. Generate commercial tool baselines
   - Run Topaz Photo AI on validation_v1
   - Run Adobe Lightroom enhance on validation_v1
   - Store outputs + processing times + metadata

3. Document baseline methodology
   - `docs/validation/BASELINE_METHODOLOGY.md`
   - Tool versions, settings, environment
   - Metrics computed (SSIM, PSNR, LPIPS, perceptual quality)

#### Phase 4.2: Wire Orchestrator + Core → Validators (Week 7)

**Implementation**:
```python
# core/validation/runner.py
class ValidationRunner:
    """Run validation benchmarks against datasets."""
    
    def validate_pipeline(self, dataset_name, pipeline_name):
        """Validate pipeline against dataset."""
        dataset = self.load_dataset(dataset_name)
        pipeline = self.load_pipeline(pipeline_name)
        
        results = []
        for image in dataset.images:
            # Process with pipeline
            output = pipeline.process(image.path)
            
            # Compute metrics against baseline
            metrics = self.compute_metrics(output, image.baseline_topaz)
            
            results.append({
                "image": image.name,
                "metrics": metrics,
                "timing": output.timing_s
            })
        
        return ValidationReport(dataset_name, pipeline_name, results)
```

**Tasks**:
1. Implement ValidationRunner in core/validation/
2. Wire to LuxDepthPipeline via orchestrator
3. Add CLI: `lux-depth-validate --dataset validation_v1`
4. Store results: `validation_reports/<timestamp>_<pipeline>.json`

#### Phase 4.3: Quality Gates (Week 8)

**Non-blocking gate initially** (manual workflow)
```yaml
# .github/workflows/validation-gate.yml
name: Validation Quality Gate
on:
  workflow_dispatch:  # Manual trigger initially

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Run validation
        run: lux-depth-validate --dataset validation_v1
      
      - name: Compare to baseline
        run: |
          python scripts/compare_validation.py \
            --current validation_reports/latest.json \
            --baseline validation_reports/baseline.json
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: validation-report
          path: validation_reports/
```

**Promote to PR gate once stable**:
- After 2-3 weeks of manual runs
- Once baselines are proven stable
- With clear threshold definitions (e.g., "no >5% SSIM regression")

**Success Criteria**:
- ✅ validation_v1 dataset finalized (20-30 images)
- ✅ Topaz/Adobe baselines generated
- ✅ ValidationRunner integrated with orchestrator
- ✅ Manual quality gate workflow operational
- ✅ Clear path to automated PR blocking gate

---

## Timeline Summary

| Week | Dates | Focus | Deliverables |
|------|-------|-------|--------------|
| **Dec 10** | Today | ✅ Objective 1 | Strict test assertions |
| **Week 1** | Dec 10-17 | Objective 2.1 | Core API docs + contract |
| **Week 2** | Dec 17-24 | Objective 2.2 | Legacy integration + parity |
| **Week 3** | Dec 24-31 | Objective 3.1 | Stage timing telemetry |
| **Week 4** | Dec 31-Jan 7 | Objective 3.2 | Export I/O optimization |
| **Week 5** | Jan 7-14 | Objective 3.3 | Adaptive upscaling |
| **Week 6** | Jan 14-21 | Objective 4.1 | Validation datasets |
| **Week 7** | Jan 21-28 | Objective 4.2 | Validator integration |
| **Week 8** | Jan 28-Feb 4 | Objective 4.3 | Quality gates (manual) |

---

## Success Metrics

### Short-term (Weeks 1-2)
- ✅ All core modules documented
- ✅ Legacy code uses core internally
- ✅ Zero breaking changes
- ✅ Behavior parity tests passing

### Medium-term (Weeks 3-5)
- ✅ Stage timing on every image
- ✅ 20-30% export time reduction
- ✅ 10-15% upscaling time reduction
- ✅ Bottlenecks identified per image type

### Long-term (Weeks 6-8)
- ✅ Validation system operational
- ✅ Topaz/Adobe baselines established
- ✅ Quality gate running manually
- ✅ Reproducible "we beat X on Y" claims

---

## Risk Mitigation

### Medium-Risk Items

**Objective 2: Platform Core Lockdown**
- **Risk**: Breaking existing code during integration
- **Mitigation**: 
  - Feature flags to toggle core vs legacy
  - Comprehensive parity tests before switchover
  - Gradual rollout (internal testing first)
  - Clear rollback plan (keep legacy code)

**Objective 4: Validation System**
- **Risk**: Baseline instability (tool updates, environment drift)
- **Mitigation**:
  - Pin commercial tool versions
  - Document environment (OS, hardware, settings)
  - Regular baseline refresh schedule
  - Version control for baselines

### Communication

**Weekly Status Updates**:
- Document progress in `docs/guides/WEEKLY_STATUS_<date>.md`
- Include: what shipped, what's blocked, metrics
- Review against this plan every Monday

**Milestone Reviews**:
- End of Week 2: Platform Core lockdown complete?
- End of Week 5: Phase 2 performance slices working?
- End of Week 8: Validation system operational?

---

## Next Actions (This Week)

### Today (Dec 10)
- [x] Restore test strictness (Objective 1) ✅
- [x] Document strategic plan (this file) ✅
- [ ] Create Platform Core API outline

### Tomorrow (Dec 11)
- [ ] Write core module docstrings (config, device, artifacts)
- [ ] Draft `docs/guides/PLATFORM_CORE_API.md`
- [ ] Review with team for feedback

### This Week (Dec 10-17)
- [ ] Complete core API documentation
- [ ] Lock API contracts with deprecation warnings
- [ ] Create compatibility adapter skeleton
- [ ] Write first parity test

---

## Conclusion

**Current Position**: Strong foundation with minor test gaps now resolved.

**Strategic Direction**: 
1. Lock Platform Core as the spine (Weeks 1-2)
2. Attack performance with data-driven slices (Weeks 3-5)
3. Turn validation into automated quality proof (Weeks 6-8)

**Key Insight**: Don't rush parallelism yet. Single-image path improvements (timing, export, upscaling) will give us 30-40% gains with lower risk and clearer measurement.

**Confidence Level**: 🟢 HIGH - Clear plan, proven foundation, incremental approach.

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-10  
**Next Review**: 2025-12-17 (End of Week 1)
