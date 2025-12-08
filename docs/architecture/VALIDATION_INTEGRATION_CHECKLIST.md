# Validation System Integration Checklist

**Purpose**: Step-by-step guide for integrating the validation system into Lux Depth V2  
**Audience**: Implementation engineers  
**Status**: Ready for use  
**Date**: 2025-12-08  

---

## Overview

This checklist ensures systematic integration of the 5-priority validation system. Each section includes prerequisites, implementation steps, verification tests, and rollback procedures.

---

## Phase 1: Benchmark Framework (Week 1-2)

### Prerequisites
- [ ] Read `VALIDATION_ARCHITECTURE.md` sections on Priority 1
- [ ] Confirm Git LFS installed and configured
- [ ] Verify access to commercial baseline tools (Topaz/Adobe) OR open-source alternatives
- [ ] Python 3.11+ environment with dependencies installed

### Implementation Steps

#### 1.1 Dataset Acquisition
- [ ] Create directory structure:
  ```bash
  mkdir -p data/benchmark_datasets/validation_v1/{sources,baselines}
  ```
- [ ] Curate 20 representative images per category specification
- [ ] Verify image quality (4K resolution, proper exposure, no compression artifacts)
- [ ] Document provenance (source, license, date acquired) in `data/benchmark_datasets/validation_v1/README.md`
- [ ] Compute SHA256 checksums:
  ```bash
  find data/benchmark_datasets/validation_v1/sources -type f -exec sha256sum {} \; > checksums.txt
  ```

#### 1.2 Baseline Generation
- [ ] Run Topaz Gigapixel 7.2.1 (or alternative):
  ```bash
  for img in data/benchmark_datasets/validation_v1/sources/*.tif; do
    topaz gigapixel --scale 4 --model standard-v6 "$img" \
      "data/benchmark_datasets/validation_v1/baselines/topaz_$(basename $img)"
  done
  ```
- [ ] Run Adobe Super Resolution OR Real-ESRGAN fallback:
  ```bash
  python -m lux_depth_v2.tools.realesrgan_baseline \
    --input-dir data/benchmark_datasets/validation_v1/sources \
    --output-dir data/benchmark_datasets/validation_v1/baselines/realesrgan \
    --model RealESRGAN_x4plus
  ```
- [ ] Verify baseline outputs (correct resolution, no processing failures)
- [ ] Document exact tool versions and settings in `baselines/README.md`

#### 1.3 Module Implementation
- [ ] Create module structure:
  ```bash
  mkdir -p lux_depth_v2/validation/benchmark/{templates,configs}
  touch lux_depth_v2/validation/benchmark/{__init__,dataset_registry,baseline_runner,metric_engine,category_scorer,report_generator}.py
  ```
- [ ] Implement `dataset_registry.py` per API contract in IMPLEMENTATION_PLAN.md
- [ ] Implement `baseline_runner.py` with caching logic
- [ ] Extend `metric_engine.py` to wrap existing `metrics.py` functions
- [ ] Implement `category_scorer.py` with weighted aggregation
- [ ] Implement `report_generator.py` with Jinja2 templates

#### 1.4 Configuration Files
- [ ] Create `lux_depth_v2/validation/benchmark/configs/validation_v1.yaml`:
  ```yaml
  version: "1.0.0"
  name: "validation_v1"
  root_dir: "data/benchmark_datasets/validation_v1"
  
  categories:
    - name: interior_luxury
      images: [interior_01.tif, interior_02.tif, ...]
      weight: 0.30
      description: "Living rooms, bedrooms, kitchens"
    
    - name: exterior_showcase
      images: [exterior_01.tif, ...]
      weight: 0.25
      description: "Facades, courtyards, pools"
    
    # ... (rest of categories)
  
  metrics:
    lpips: {weight: 0.40, net: alex, device: auto}
    nima: {weight: 0.30, device: auto}
    ssim: {weight: 0.20}
    psnr: {weight: 0.10}
  ```

#### 1.5 Git LFS Setup
- [ ] Create `.gitattributes` entries:
  ```bash
  echo "data/benchmark_datasets/**/*.tif filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
  echo "data/benchmark_datasets/**/*.tiff filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
  ```
- [ ] Track files with LFS:
  ```bash
  git lfs track "data/benchmark_datasets/**/*.tif"
  git lfs track "data/benchmark_datasets/**/*.tiff"
  ```
- [ ] Commit LFS configuration:
  ```bash
  git add .gitattributes data/benchmark_datasets/
  git commit -m "feat(validation): Add benchmark dataset with Git LFS"
  ```

### Verification Tests

#### 1.6 Unit Tests
- [ ] Create `tests/validation/test_dataset_registry.py`:
  ```python
  def test_loads_validation_v1():
      registry = DatasetRegistry(Path("lux_depth_v2/validation/benchmark/configs"))
      dataset = registry.load_dataset("validation_v1")
      assert dataset.version == "1.0.0"
      assert len(dataset.categories) == 5
      assert sum(c.weight for c in dataset.categories) == pytest.approx(1.0)
  
  def test_validates_checksums():
      registry = DatasetRegistry(...)
      dataset = registry.load_dataset("validation_v1")
      assert dataset.validate()  # All checksums match
  ```
- [ ] Run tests:
  ```bash
  pytest tests/validation/test_dataset_registry.py -v
  ```

#### 1.7 Integration Test
- [ ] Run manual benchmark (single image):
  ```bash
  python -m lux_depth_v2.validation.benchmark \
    --dataset validation_v1 \
    --images interior_01.tif \
    --baselines topaz_gigapixel \
    --output test_benchmark_output
  ```
- [ ] Verify report generation:
  ```bash
  ls test_benchmark_output/
  # Expected: index.html, summary.md, category_scores.json
  ```
- [ ] Inspect HTML report (open in browser)
- [ ] Validate metric values (LPIPS in 0-1 range, SSIM in 0-1, PSNR >20dB, NIMA in 1-10)

### Rollback Procedure
If Phase 1 fails:
1. Remove validation module: `git rm -rf lux_depth_v2/validation/benchmark`
2. Remove dataset: `git rm -rf data/benchmark_datasets`
3. Revert `.gitattributes`: `git checkout .gitattributes`
4. Force push if already pushed to remote (coordination required)

---

## Phase 2: CI/CD Quality Gates (Week 3-4)

### Prerequisites
- [ ] Phase 1 complete and verified
- [ ] GitHub Actions GPU runner available (or self-hosted configured)
- [ ] 5-10 golden images curated and stored in `tests/golden_images/sources/`

### Implementation Steps

#### 2.1 Golden Image Curation
- [ ] Select images exposing failure modes:
  - Edge aliasing (diagonal lines, architectural edges)
  - Texture loss (fine fabrics, wood grain)
  - Halo artifacts (high-contrast boundaries)
  - Color banding (smooth gradients, skies)
  - Material realism (glass reflections, metal specularity)
- [ ] Store in `tests/golden_images/sources/`
- [ ] Generate reference outputs from current `main`:
  ```bash
  git checkout main
  for img in tests/golden_images/sources/*.tif; do
    python -m lux_depth_v2 \
      --input "$img" \
      --preset photo_realistic \
      --output tests/golden_images/references/
  done
  ```

#### 2.2 Registry Configuration
- [ ] Create `tests/golden_images/registry.yaml`:
  ```yaml
  golden_images:
    - name: edge_aliasing
      source: edge_aliasing.tif
      preset: photo_realistic
      failure_mode: "Edge aliasing in diagonal lines"
      thresholds:
        lpips_max_delta: 0.02
        ssim_min_delta: -0.01
        psnr_min_delta: -0.5
        nima_min_delta: -0.1
    
    # ... (rest of golden images)
  ```

#### 2.3 Module Implementation
- [ ] Create `lux_depth_v2/validation/golden/`:
  ```bash
  mkdir -p lux_depth_v2/validation/golden
  touch lux_depth_v2/validation/golden/{__init__,registry,validator,regression_checker,trend_tracker}.py
  ```
- [ ] Implement modules per API contracts in IMPLEMENTATION_PLAN.md
- [ ] Add CLI entry point (`__main__.py`)

#### 2.4 GitHub Actions Workflow
- [ ] Create `.github/workflows/quality-gate-validation.yml` per spec in IMPLEMENTATION_PLAN.md
- [ ] Test workflow locally with `act` (if available):
  ```bash
  act pull_request -W .github/workflows/quality-gate-validation.yml
  ```
- [ ] Push workflow to branch and open test PR
- [ ] Verify workflow runs and posts comment

#### 2.5 Trend Tracking Setup
- [ ] Create trend storage directory: `mkdir -p .github/quality_trends/`
- [ ] Add CSV header: `echo "date,commit,golden_name,lpips,ssim,psnr,nima" > .github/quality_trends/metrics.csv`
- [ ] Update workflow to append metrics after each run

### Verification Tests

#### 2.6 Unit Tests
- [ ] Test registry loading: `tests/validation/test_golden_registry.py`
- [ ] Test validator metrics computation: `tests/validation/test_golden_validator.py`
- [ ] Test regression checker logic: `tests/validation/test_regression_checker.py`

#### 2.7 Integration Test
- [ ] Create intentional regression:
  ```python
  # In pipeline.py, temporarily reduce sharpness
  sharpness_strength = 0.0  # Was 0.5
  ```
- [ ] Open PR with this change
- [ ] Verify workflow blocks merge with regression message
- [ ] Revert change, verify PR now passes

### Rollback Procedure
If Phase 2 fails:
1. Disable workflow: `git mv .github/workflows/quality-gate-validation.yml .github/workflows/quality-gate-validation.yml.disabled`
2. Remove golden module: `git rm -rf lux_depth_v2/validation/golden`
3. Remove golden images: `git rm -rf tests/golden_images`
4. Merge without quality gate (manual review)

---

## Phase 3: Performance Profiler (Week 5-6)

### Prerequisites
- [ ] Phase 1-2 complete
- [ ] `psutil` installed for memory monitoring
- [ ] GPU monitoring libraries available (`torch.cuda`, `torch.mps`)

### Implementation Steps

#### 3.1 Module Implementation
- [ ] Create `lux_depth_v2/profiling/`:
  ```bash
  mkdir -p lux_depth_v2/profiling/reports
  touch lux_depth_v2/profiling/{__init__,stage_profiler,gpu_monitor,bottleneck_analyzer,optimization_advisor}.py
  ```
- [ ] Implement `stage_profiler.py` per API contract
- [ ] Implement GPU monitoring (CUDA/MPS/CPU fallback)
- [ ] Implement bottleneck analysis heuristics
- [ ] Implement optimization advisor

#### 3.2 Pipeline Integration
- [ ] Modify `pipeline.py` to accept optional `profiler` parameter
- [ ] Wrap stages with profiler context managers:
  ```python
  with (profiler.stage("load") if profiler else nullcontext()):
      img = load_image(image_path)
  ```
- [ ] Add `--profile` flag to CLI
- [ ] Add `--profile-output` flag for report path

#### 3.3 Telemetry Integration
- [ ] Extend `telemetry.py` to include profiling data
- [ ] Add Prometheus metric definitions for stage timings
- [ ] Update service endpoint to expose profiling metrics

### Verification Tests

#### 3.4 Unit Tests
- [ ] Test stage profiler timing accuracy: `tests/profiling/test_stage_profiler.py`
- [ ] Test GPU monitor fallback (CPU mode): `tests/profiling/test_gpu_monitor.py`
- [ ] Test bottleneck analyzer heuristics: `tests/profiling/test_bottleneck_analyzer.py`

#### 3.5 Integration Test
- [ ] Run profiling on test image:
  ```bash
  python -m lux_depth_v2 \
    --input tests/data/test_image.tif \
    --preset photo_realistic \
    --profile \
    --profile-output profiling_report.json
  ```
- [ ] Inspect report:
  ```bash
  cat profiling_report.json | jq '.stages[] | {name, duration_s, percent_of_total}'
  ```
- [ ] Verify bottleneck identified correctly (manual validation against known slow stage)

#### 3.6 Benchmark Integration
- [ ] Add profiling to benchmark runs (optional flag)
- [ ] Generate performance comparison report (Lux Depth V2 vs baselines)

### Rollback Procedure
If Phase 3 fails:
1. Remove profiling module: `git rm -rf lux_depth_v2/profiling`
2. Revert pipeline.py changes: `git checkout pipeline.py`
3. Remove CLI flags: revert CLI changes
4. Continue with non-profiled pipeline

---

## Phase 4: Segmentation Validator (Week 7-8)

### Prerequisites
- [ ] Phase 1-3 complete
- [ ] Segmentation backends operational (ONNX, SegFormer, Heuristic)

### Implementation Steps

#### 4.1 Module Implementation
- [ ] Create `lux_depth_v2/validation/segmentation/`:
  ```bash
  mkdir -p lux_depth_v2/validation/segmentation
  touch lux_depth_v2/validation/segmentation/{__init__,consistency_evaluator,impact_analyzer,surface_metrics,annotation_tool}.py
  ```
- [ ] Implement consistency evaluator (10-run IoU test)
- [ ] Implement impact analyzer (ON vs OFF comparison)
- [ ] Implement surface-specific metrics computation
- [ ] Implement annotation tool (optional, lightweight UI)

#### 4.2 Validation Suite
- [ ] Select test images for segmentation validation (5-10 images with diverse materials)
- [ ] Run consistency evaluation:
  ```bash
  python -m lux_depth_v2.validation.segmentation.consistency_evaluator \
    --image tests/data/glass_metal_wood.tif \
    --backend auto \
    --num-runs 10 \
    --output consistency_report.json
  ```
- [ ] Run impact analysis:
  ```bash
  python -m lux_depth_v2.validation.segmentation.impact_analyzer \
    --image tests/data/glass_metal_wood.tif \
    --preset interior_luxury \
    --output impact_report.json
  ```

#### 4.3 Benchmark Integration
- [ ] Add segmentation validation to benchmark framework
- [ ] Include per-surface metrics in benchmark reports
- [ ] Document segmentation impact in summary

### Verification Tests

#### 4.4 Unit Tests
- [ ] Test consistency evaluator: `tests/validation/test_consistency_evaluator.py`
- [ ] Test impact analyzer: `tests/validation/test_impact_analyzer.py`
- [ ] Test surface metrics: `tests/validation/test_surface_metrics.py`

#### 4.5 Integration Test
- [ ] Run full validation on test set:
  ```bash
  python -m lux_depth_v2.validation.segmentation \
    --test-set tests/data/segmentation_test_set/ \
    --output validation_results/
  ```
- [ ] Verify consistency scores >0.95 (or document exceptions)
- [ ] Verify impact analysis shows quality improvement (LPIPS delta negative)

### Rollback Procedure
If Phase 4 fails:
1. Remove segmentation validation module: `git rm -rf lux_depth_v2/validation/segmentation`
2. Revert benchmark integration changes
3. Document segmentation as experimental (not validated)

---

## Phase 5: Production Observability (Week 9-10)

### Prerequisites
- [ ] Phase 1-4 complete
- [ ] FastAPI service operational (`lux_depth_v2/service.py`)
- [ ] Prometheus client library installed
- [ ] Grafana instance available (or plan to deploy)

### Implementation Steps

#### 5.1 Module Implementation
- [ ] Create `lux_depth_v2/observability/`:
  ```bash
  mkdir -p lux_depth_v2/observability/dashboards
  touch lux_depth_v2/observability/{__init__,prometheus_metrics,request_tracer,error_tracker}.py
  ```
- [ ] Implement Prometheus metric definitions per spec
- [ ] Implement request tracer with audit trail
- [ ] Implement error categorization

#### 5.2 Service Integration
- [ ] Add `/metrics` endpoint to `service.py`
- [ ] Add middleware for request tracking
- [ ] Add error handler with categorization
- [ ] Add tracing configuration options

#### 5.3 Grafana Dashboard
- [ ] Create dashboard JSON: `lux_depth_v2/observability/dashboards/lux_depth_v2_dashboard.json`
- [ ] Include panels for:
  - Request throughput (requests/min)
  - Latency percentiles (p50, p95, p99)
  - Error rate by category
  - GPU memory utilization
  - Processing time by stage
- [ ] Import dashboard to Grafana instance

#### 5.4 Alerting Configuration
- [ ] Configure Prometheus alert rules:
  ```yaml
  # prometheus_alerts.yml
  groups:
    - name: lux_depth_v2
      rules:
        - alert: HighErrorRate
          expr: rate(lux_errors_total[5m]) > 0.05
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High error rate detected"
        
        - alert: HighLatency
          expr: histogram_quantile(0.95, lux_request_duration_seconds) > 60
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "p95 latency exceeds 60s"
  ```
- [ ] Configure alert destination (Slack, email, PagerDuty)

### Verification Tests

#### 5.5 Unit Tests
- [ ] Test Prometheus metrics export: `tests/observability/test_prometheus_metrics.py`
- [ ] Test request tracer: `tests/observability/test_request_tracer.py`
- [ ] Test error categorization: `tests/observability/test_error_tracker.py`

#### 5.6 Integration Test
- [ ] Start service with observability:
  ```bash
  python -m lux_depth_v2.service \
    --output-dir /tmp/lux_output \
    --service \
    --enable-tracing \
    --trace-dir /tmp/lux_traces
  ```
- [ ] Send test requests:
  ```bash
  curl -X POST http://localhost:8088/process \
    -F "file=@tests/data/test_image.tif" \
    -F "preset=photo_realistic"
  ```
- [ ] Verify metrics endpoint:
  ```bash
  curl http://localhost:8088/metrics | grep lux_requests_total
  ```
- [ ] Verify trace logs:
  ```bash
  ls /tmp/lux_traces/*.json
  cat /tmp/lux_traces/latest.json | jq
  ```

#### 5.7 Load Test
- [ ] Run load test (100 requests over 5 minutes):
  ```bash
  python scripts/load_test.py \
    --endpoint http://localhost:8088/process \
    --requests 100 \
    --duration 300
  ```
- [ ] Monitor Grafana dashboard during load test
- [ ] Verify metrics accuracy (request count matches actual)
- [ ] Verify no errors under normal load

### Rollback Procedure
If Phase 5 fails:
1. Remove observability module: `git rm -rf lux_depth_v2/observability`
2. Revert service.py changes: `git checkout service.py`
3. Disable tracing in production config
4. Continue with basic logging (existing)

---

## Post-Integration Verification

### System-Wide Checks
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check test coverage: `pytest --cov=lux_depth_v2 tests/`
- [ ] Verify no regressions in existing functionality
- [ ] Run linting: `make lint`
- [ ] Build documentation: `make docs`

### Performance Baseline
- [ ] Run profiling on standard test set (before optimization)
- [ ] Document baseline metrics:
  - Total processing time
  - Per-stage breakdown
  - Memory usage
  - GPU utilization
- [ ] Store baseline report for future comparison

### Documentation Updates
- [ ] Update `lux_depth_v2/README.md` with validation system overview
- [ ] Add validation examples to `lux_depth_v2/docs/`
- [ ] Update `docs/QUICK_START.md` with profiling and benchmarking examples
- [ ] Create operator guide for production monitoring

### Deployment Checklist
- [ ] Tag release: `git tag v2.1.0-validation`
- [ ] Push to origin: `git push origin main --tags`
- [ ] Update CHANGELOG.md with validation system features
- [ ] Announce validation system availability to users
- [ ] Schedule first benchmark run (weekly recurring)

---

## Rollback Plan (Complete System)

If validation system causes critical issues:

1. **Immediate Mitigation** (< 1 hour):
   - Disable quality gate workflow: Rename `.github/workflows/quality-gate-validation.yml` to `.disabled`
   - Revert main branch to pre-validation commit: `git revert <validation-merge-commit>`
   - Notify team of rollback

2. **Root Cause Analysis** (< 24 hours):
   - Identify failing component (benchmark, profiler, segmentation, observability)
   - Review logs and error messages
   - Reproduce issue in isolated environment

3. **Selective Re-enable** (< 1 week):
   - Re-enable working components individually
   - Keep problematic component disabled
   - Schedule fix for broken component

---

## Support and Escalation

### Issues During Integration
- **Build Failures**: Check dependencies, Python version, Git LFS setup
- **Test Failures**: Review test logs, verify input data integrity
- **Performance Degradation**: Disable profiling (overhead), check GPU availability
- **CI/CD Failures**: Check GitHub Actions logs, verify runner configuration

### Escalation Path
1. **Level 1**: Implementation engineer debugs (1 hour)
2. **Level 2**: Senior engineer reviews architecture (4 hours)
3. **Level 3**: Architect approves rollback or workaround (1 day)

### Contact Points
- **Architecture Questions**: Transformation Portal Architect
- **Implementation Support**: Transformation Portal Specialist
- **CI/CD Issues**: DevOps team (GitHub Actions maintainer)
- **Production Issues**: On-call engineer (observability team)

---

## Success Metrics (Post-Integration)

### Week 4 (Foundation + CI/CD)
- [ ] Benchmark runs successfully on full dataset (20 images)
- [ ] Quality gate blocks at least one intentional regression (test)
- [ ] Zero false positives in golden image validation (30-day period)

### Week 6 (Performance)
- [ ] Profiling identifies dominant stage (upscaling confirmed)
- [ ] At least one optimization implemented and measured (>10% improvement)
- [ ] Profiling overhead <5% when enabled

### Week 8 (Segmentation)
- [ ] Consistency scores >0.95 for all backends (or exceptions documented)
- [ ] Impact analysis shows measurable improvement (LPIPS delta negative)
- [ ] Per-surface metrics available in benchmark reports

### Week 10 (Observability)
- [ ] Service runs 7 days without monitoring-related incidents
- [ ] Grafana dashboard shows real-time metrics
- [ ] Request traces enable reproduction of production issues

---

## Appendix: Command Reference

### Benchmark Commands
```bash
# Run full benchmark
python -m lux_depth_v2.validation.benchmark \
  --dataset validation_v1 \
  --output benchmark_results/

# Run single category
python -m lux_depth_v2.validation.benchmark \
  --dataset validation_v1 \
  --category interior_luxury \
  --output benchmark_results_interior/
```

### Golden Image Commands
```bash
# Validate golden images
python -m lux_depth_v2.validation.golden.validator \
  --registry tests/golden_images/registry.yaml \
  --output golden_validation_report.md

# Update references (after merge to main)
python -m lux_depth_v2.validation.golden.validator \
  --generate-references \
  --output tests/golden_images/references/
```

### Profiling Commands
```bash
# Profile single image
python -m lux_depth_v2 \
  --input test.tif \
  --preset photo_realistic \
  --profile \
  --profile-output profiling_report.json

# Profile batch
python -m lux_depth_v2 \
  --input-dir input/ \
  --preset photo_realistic \
  --profile \
  --profile-output profiling_batch.json
```

### Segmentation Validation Commands
```bash
# Consistency evaluation
python -m lux_depth_v2.validation.segmentation.consistency_evaluator \
  --image test.tif \
  --num-runs 10

# Impact analysis
python -m lux_depth_v2.validation.segmentation.impact_analyzer \
  --image test.tif \
  --preset interior_luxury
```

### Observability Commands
```bash
# Start service with tracing
python -m lux_depth_v2.service \
  --service \
  --enable-tracing \
  --trace-dir ./traces

# Query traces
python -m lux_depth_v2.observability.query_traces \
  --trace-dir ./traces \
  --preset interior_luxury \
  --success-only
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-08  
**Status**: Ready for Implementation
