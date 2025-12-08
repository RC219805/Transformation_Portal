# Validation Architecture: From Framework to Quality Breakthrough

**Status**: Design Document  
**Author**: Transformation Portal Architect  
**Date**: 2025-12-08  
**Context**: Strategic plan to establish commercial proof for Lux Depth V2

---

## Executive Summary

This document defines the architecture for transforming Lux Depth V2 from a production-ready framework into a validated quality breakthrough with commercial proof. The system integrates five interconnected priorities into a cohesive validation infrastructure that provides repeatable, measurable evidence of quality superiority.

**Key Deliverables**:
1. **Benchmark Framework**: Extensible, category-based validation system
2. **CI/CD Quality Gates**: Regression prevention with golden image validation
3. **Performance Optimization**: GPU-accelerated pipeline with profiling instrumentation
4. **Material Segmentation Validation**: Evaluation loop with impact measurement
5. **Production Observability**: Service monitoring with traceability

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VALIDATION & QUALITY SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ├── BENCHMARK FRAMEWORK (Priority 1)
         │   ├── Dataset Registry (curated, versioned, categorized)
         │   ├── Baseline Repository (Topaz/Adobe reference outputs)
         │   ├── Metric Computation Engine (LPIPS, NIMA, SSIM/PSNR)
         │   └── Report Generator (HTML/MD, category rollups, weighted scores)
         │
         ├── CI/CD QUALITY GATES (Priority 2)
         │   ├── Golden Image Registry (failure mode coverage)
         │   ├── Regression Detector (delta thresholds per metric)
         │   ├── GitHub Actions Integration (blocking merge on regression)
         │   └── Trend Dashboard (quality metrics over time)
         │
         ├── PERFORMANCE PROFILER (Priority 3)
         │   ├── Stage Timing Instrumentation (I/O, tiling, inference, save)
         │   ├── GPU Utilization Tracker (CUDA/MPS memory, compute %)
         │   ├── Bottleneck Analyzer (identify dominant stages)
         │   └── Optimization Recommendations (actionable insights)
         │
         ├── SEGMENTATION VALIDATOR (Priority 4)
         │   ├── Consistency Evaluator (mask stability across runs)
         │   ├── Impact Measurement (segmentation ON vs OFF quality delta)
         │   ├── Surface-Specific Metrics (per-material LPIPS/NIMA)
         │   └── Ground Truth Annotator (lightweight labeling tool)
         │
         └── PRODUCTION TELEMETRY (Priority 5)
             ├── FastAPI Metrics Endpoint (/metrics for Prometheus)
             ├── Request Tracing (config hash, model versions, latency)
             ├── Error Rate Monitoring (failure modes, stack traces)
             └── Reproducibility Logs (full audit trail for every output)
```

---

## Priority 1: Repeatable Benchmark Framework

### Architecture

**Module**: `lux_depth_v2/validation/benchmark/`

**Components**:
```
benchmark/
├── __init__.py
├── dataset_registry.py      # Dataset management with versioning
├── baseline_runner.py        # Execute baseline tools (Topaz, Adobe, etc.)
├── metric_engine.py          # Compute LPIPS, NIMA, SSIM/PSNR
├── category_scorer.py        # Per-category aggregation with weights
├── report_generator.py       # HTML/MD output with visualizations
└── config.yaml               # Benchmark configuration
```

**Dataset Registry Schema**:
```yaml
# config.yaml
datasets:
  validation_v1:
    version: "1.0.0"
    images: 20
    categories:
      - name: interior_luxury
        images: [interior_01.tif, interior_02.tif, ...]
        weight: 0.30
      - name: exterior_showcase
        images: [exterior_01.tif, ...]
        weight: 0.25
      - name: glossy_surfaces
        images: [glass_01.tif, metal_01.tif, ...]
        weight: 0.20
      - name: fine_patterns
        images: [texture_01.tif, ...]
        weight: 0.15
      - name: mixed_lighting
        images: [hdr_01.tif, ...]
        weight: 0.10

baselines:
  topaz_gigapixel:
    version: "7.2.1"
    command: "topaz gigapixel --scale 4 --model standard-v6 {input} {output}"
  adobe_super_resolution:
    version: "Photoshop 2024"
    method: manual  # Requires human operator for Adobe tools

metrics:
  lpips:
    weight: 0.40
    net: alex
    device: auto
  nima:
    weight: 0.30
    device: auto
  ssim:
    weight: 0.20
  psnr:
    weight: 0.10
```

**Metric Engine Integration**:
- Leverage existing `lux_depth_v2/validation/metrics.py`
- Add category-aware scoring in `category_scorer.py`
- Support configurable weights for different use cases (fidelity vs aesthetics)

**Report Generator Output**:
```
benchmark_report_20251208_014530/
├── index.html                    # Interactive dashboard
├── summary.md                    # Text summary for CI/CD
├── category_scores.json          # Machine-readable results
├── per_image_metrics.csv         # Detailed per-image breakdown
└── visual_comparisons/           # Side-by-side image comparisons
    ├── interior_01_comparison.html
    └── ...
```

**Key Architectural Decisions**:

1. **Versioned Datasets**: All datasets versioned in Git LFS or external storage (S3/GCS) with SHA256 checksums
2. **Baseline Immutability**: Baseline outputs stored with exact tool versions; never overwritten
3. **Extensible Metrics**: Plugin architecture for adding new metrics (e.g., FID, CLIP score)
4. **Category Flexibility**: Support arbitrary category definitions via YAML config

**Integration Points**:
- CLI: `lux-depth-v2-benchmark --dataset validation_v1 --output benchmark_results/`
- API: `from lux_depth_v2.validation.benchmark import run_benchmark`
- CI/CD: Automated benchmark runs on release branches

---

## Priority 2: CI/CD Quality Gates

### Architecture

**GitHub Actions Workflow**: `.github/workflows/quality-gate-validation.yml`

**Golden Image Strategy**:
```
tests/golden_images/
├── registry.yaml                 # Golden image metadata
├── sources/                      # Input images exposing failure modes
│   ├── edge_aliasing.tif
│   ├── texture_loss.tif
│   ├── halo_artifact.tif
│   ├── color_banding.tif
│   └── glass_reflection.tif
└── references/                   # Known-good outputs from current main
    ├── edge_aliasing_photo_realistic.tif
    ├── texture_loss_interior_luxury.tif
    └── ...
```

**Registry Schema**:
```yaml
# tests/golden_images/registry.yaml
golden_images:
  - name: edge_aliasing
    source: edge_aliasing.tif
    preset: photo_realistic
    failure_mode: Edge aliasing in diagonal lines
    thresholds:
      lpips_max_delta: 0.02      # Max allowed regression
      ssim_min_delta: -0.01      # Max allowed drop
      psnr_min_delta: -0.5       # Max allowed drop in dB
      nima_min_delta: -0.1       # Max allowed aesthetic drop
  
  - name: texture_loss
    source: texture_loss.tif
    preset: interior_luxury
    failure_mode: Fine texture detail loss in fabrics
    thresholds:
      lpips_max_delta: 0.03
      ssim_min_delta: -0.02
```

**Regression Detection Workflow**:
```yaml
# .github/workflows/quality-gate-validation.yml
name: Quality Gate - Visual Validation

on:
  pull_request:
    paths:
      - 'lux_depth_v2/**'
      - 'tests/golden_images/**'
  workflow_dispatch:

jobs:
  golden_image_validation:
    runs-on: ubuntu-latest-gpu  # GPU runner for realistic performance
    
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true  # Pull golden images from LFS
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[validation]"
      
      - name: Run golden image validation
        id: validate
        run: |
          python -m lux_depth_v2.validation.golden_validator \
            --registry tests/golden_images/registry.yaml \
            --baseline-branch origin/main \
            --output-report golden_validation_report.md
      
      - name: Upload validation report
        uses: actions/upload-artifact@v4
        with:
          name: golden-validation-report
          path: golden_validation_report.md
      
      - name: Check for regressions
        run: |
          python -m lux_depth_v2.validation.regression_checker \
            --report golden_validation_report.md \
            --fail-on-regression
      
      - name: Post PR comment
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('golden_validation_report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🎨 Golden Image Validation\n\n${report}`
            });
```

**Trend Dashboard Integration**:
- Store metrics in GitHub Actions artifacts or external time-series DB (InfluxDB, Prometheus)
- Generate trend charts showing quality evolution over time
- Alert on sustained degradation patterns

**Key Architectural Decisions**:

1. **Separate Validation from Unit Tests**: Golden image validation runs in dedicated workflow to avoid slowing down fast tests
2. **Baseline Comparison**: Always compare against `origin/main`, not just previous commit
3. **Configurable Thresholds**: Per-image thresholds allow different sensitivity for different failure modes
4. **Automatic Reference Updates**: On merge to main, update reference outputs for future PRs

**Integration Points**:
- Blocking PR merge if any golden image regresses beyond threshold
- Weekly scheduled runs to catch drift in dependencies or models
- Manual trigger for testing specific presets or configurations

---

## Priority 3: Performance Optimization

### Architecture

**Profiling Module**: `lux_depth_v2/profiling/`

```
profiling/
├── __init__.py
├── stage_profiler.py         # Fine-grained stage timing
├── gpu_monitor.py            # CUDA/MPS utilization tracking
├── bottleneck_analyzer.py    # Identify optimization targets
└── optimization_advisor.py   # Generate actionable recommendations
```

**Stage Profiler Design**:
```python
# stage_profiler.py
from dataclasses import dataclass
from typing import Dict, List
from contextlib import contextmanager
import time

@dataclass
class StageProfile:
    name: str
    duration_s: float
    cpu_percent: float
    gpu_percent: float
    gpu_mem_mb: float
    io_wait_s: float

class PipelineProfiler:
    """Fine-grained profiling for Lux Depth V2 pipeline."""
    
    def __init__(self, enable_gpu_sync: bool = True):
        self.enable_gpu_sync = enable_gpu_sync
        self.stages: List[StageProfile] = []
    
    @contextmanager
    def stage(self, name: str):
        """Profile a pipeline stage."""
        # Pre-stage sync
        if self.enable_gpu_sync:
            self._sync_device()
        
        start = time.perf_counter()
        start_gpu_mem = self._get_gpu_memory()
        
        try:
            yield
        finally:
            # Post-stage sync
            if self.enable_gpu_sync:
                self._sync_device()
            
            duration = time.perf_counter() - start
            end_gpu_mem = self._get_gpu_memory()
            
            self.stages.append(StageProfile(
                name=name,
                duration_s=duration,
                cpu_percent=self._get_cpu_percent(),
                gpu_percent=self._get_gpu_percent(),
                gpu_mem_mb=end_gpu_mem,
                io_wait_s=self._get_io_wait()
            ))
    
    def generate_report(self) -> Dict:
        """Generate performance report with bottleneck analysis."""
        total_time = sum(s.duration_s for s in self.stages)
        
        return {
            "total_duration_s": total_time,
            "stages": [
                {
                    "name": s.name,
                    "duration_s": s.duration_s,
                    "percent_of_total": (s.duration_s / total_time * 100),
                    "gpu_utilization": s.gpu_percent,
                    "gpu_memory_mb": s.gpu_mem_mb,
                }
                for s in self.stages
            ],
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
```

**GPU Monitoring Strategy**:
- **CUDA**: Use `torch.cuda` APIs for memory and utilization
- **MPS** (Apple Silicon): Use `torch.mps` APIs where available
- **Fallback**: CPU-only mode with graceful degradation

**Bottleneck Analysis Heuristics**:
1. **I/O Bound**: If I/O wait time > 30% of total, suggest faster storage or parallel loading
2. **Upscaling Bound**: If upscaling > 60% of total, prioritize GPU optimization and tile size tuning
3. **Memory Bound**: If GPU memory near capacity, suggest reduced batch size or FP16 precision
4. **CPU Bound**: If CPU utilization high but GPU idle, suggest async data loading

**Integration with Existing Telemetry**:
- Extend `lux_depth_v2/telemetry.py` with profiling hooks
- Add `--profile` flag to CLI for detailed performance analysis
- Generate Prometheus metrics for production monitoring

**Key Architectural Decisions**:

1. **Optional GPU Sync**: Configurable device synchronization for accurate timing (overhead acceptable in profiling mode)
2. **Stage Granularity**: Profile at meaningful boundaries (load, segment, tile, upscale, save) not per-operation
3. **Production-Safe**: Profiler can be enabled in production with minimal overhead when `enable_gpu_sync=False`

**Integration Points**:
- CLI: `lux-depth-v2 --profile --output profiling_report.json`
- Service: `/metrics` endpoint includes per-stage latency histograms
- Benchmark: Profile runs included in benchmark reports for regression detection

---

## Priority 4: Material Segmentation Validation

### Architecture

**Segmentation Validation Module**: `lux_depth_v2/validation/segmentation/`

```
segmentation/
├── __init__.py
├── consistency_evaluator.py  # Measure mask stability across runs
├── impact_analyzer.py        # Segmentation ON vs OFF quality delta
├── surface_metrics.py        # Per-material LPIPS/NIMA computation
└── annotation_tool.py        # Lightweight ground truth labeling
```

**Consistency Evaluation Strategy**:
```python
# consistency_evaluator.py
from typing import Dict, List
import numpy as np

def evaluate_consistency(
    image_path: str,
    num_runs: int = 10,
    backend: str = "auto"
) -> Dict[str, float]:
    """Measure segmentation mask stability across multiple runs.
    
    Returns:
        Consistency metrics:
        - iou_mean: Average IoU between consecutive runs
        - iou_std: Standard deviation of IoU (lower is better)
        - label_agreement: Fraction of pixels with consistent labels
    """
    masks = []
    for _ in range(num_runs):
        mask = run_segmentation(image_path, backend=backend)
        masks.append(mask)
    
    # Pairwise IoU
    ious = []
    for i in range(len(masks) - 1):
        iou = compute_iou(masks[i], masks[i+1])
        ious.append(iou)
    
    return {
        "iou_mean": np.mean(ious),
        "iou_std": np.std(ious),
        "label_agreement": compute_label_agreement(masks)
    }
```

**Impact Measurement Architecture**:
```python
# impact_analyzer.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class SegmentationImpact:
    """Quality impact of material segmentation."""
    
    # Overall metrics
    lpips_without_seg: float
    lpips_with_seg: float
    lpips_improvement: float  # Negative = better
    
    nima_without_seg: float
    nima_with_seg: float
    nima_improvement: float  # Positive = better
    
    # Per-surface metrics
    surface_impacts: Dict[str, Dict[str, float]]  # surface -> {metric: value}
    
    # Segmentation quality
    segmentation_consistency: float
    segmentation_coverage: float  # Fraction of image segmented

def measure_segmentation_impact(
    image_path: str,
    reference: np.ndarray,
    preset: str
) -> SegmentationImpact:
    """Run pipeline with and without segmentation, measure quality delta."""
    
    # Baseline: segmentation disabled
    output_no_seg = run_pipeline(
        image_path,
        preset=preset,
        segmentation_backend="none"
    )
    
    # With segmentation
    output_with_seg = run_pipeline(
        image_path,
        preset=preset,
        segmentation_backend="auto"
    )
    
    # Compute metrics
    lpips_no_seg = compute_lpips(output_no_seg, reference)
    lpips_with_seg = compute_lpips(output_with_seg, reference)
    
    nima_no_seg = compute_nima(output_no_seg)
    nima_with_seg = compute_nima(output_with_seg)
    
    # Per-surface analysis
    surface_impacts = analyze_per_surface_impact(
        output_no_seg, output_with_seg, reference,
        segmentation_mask=get_segmentation_mask(image_path)
    )
    
    return SegmentationImpact(
        lpips_without_seg=lpips_no_seg,
        lpips_with_seg=lpips_with_seg,
        lpips_improvement=lpips_with_seg - lpips_no_seg,
        nima_without_seg=nima_no_seg,
        nima_with_seg=nima_with_seg,
        nima_improvement=nima_with_seg - nima_no_seg,
        surface_impacts=surface_impacts,
        segmentation_consistency=evaluate_segmentation_consistency(image_path),
        segmentation_coverage=compute_coverage(get_segmentation_mask(image_path))
    )
```

**Ground Truth Annotation Tool**:
- Lightweight web UI for manual mask correction
- Export annotated masks to `tests/segmentation_ground_truth/`
- Use for training/fine-tuning segmentation backends
- Not required for validation (consistency is proxy for quality)

**Key Architectural Decisions**:

1. **Consistency as Proxy**: Instead of perfect ground truth, use stability across runs as quality signal
2. **Ablation Testing**: Measure segmentation impact by comparing ON vs OFF
3. **Surface-Specific Metrics**: Separate quality analysis for each material type
4. **Optional Ground Truth**: Ground truth annotations improve validation but aren't required

**Integration Points**:
- Benchmark: Include segmentation impact in benchmark reports
- CI/CD: Gate on segmentation consistency (no random mask flipping)
- Documentation: Surface-specific quality improvements reported to users

---

## Priority 5: Production Observability

### Architecture

**Service Monitoring Extension**: `lux_depth_v2/observability/`

```
observability/
├── __init__.py
├── prometheus_metrics.py     # Prometheus metric definitions
├── request_tracer.py         # Full request traceability
├── error_tracker.py          # Error categorization and alerting
└── dashboard_templates/      # Grafana dashboard JSON
    └── lux_depth_v2_dashboard.json
```

**FastAPI Metrics Endpoint**:
```python
# service.py (extension)
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter(
    "lux_requests_total",
    "Total requests processed",
    ["preset", "status"]
)

REQUEST_DURATION = Histogram(
    "lux_request_duration_seconds",
    "Request processing duration",
    ["preset", "stage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

GPU_MEMORY = Gauge(
    "lux_gpu_memory_bytes",
    "Current GPU memory usage"
)

ERROR_RATE = Counter(
    "lux_errors_total",
    "Total errors",
    ["error_type", "stage"]
)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track request metrics."""
    preset = request.query_params.get("preset", "unknown")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        status = "success" if response.status_code == 200 else "error"
        REQUEST_COUNT.labels(preset=preset, status=status).inc()
        return response
    except Exception as e:
        REQUEST_COUNT.labels(preset=preset, status="error").inc()
        ERROR_RATE.labels(error_type=type(e).__name__, stage="handler").inc()
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(preset=preset, stage="total").observe(duration)
```

**Request Tracing Architecture**:
```python
# request_tracer.py
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from datetime import datetime

@dataclass
class RequestTrace:
    """Full audit trail for a single request."""
    
    trace_id: str  # UUID
    timestamp: str  # ISO 8601
    
    # Input
    input_image_hash: str  # SHA256
    input_width: int
    input_height: int
    
    # Configuration
    config_hash: str  # Hash of full pipeline config
    preset: str
    segmentation_backend: str
    upscaler_backend: str
    
    # Model versions
    model_versions: dict  # {model_name: version/hash}
    
    # Performance
    total_duration_s: float
    stage_timings: dict  # {stage_name: duration_s}
    peak_memory_mb: float
    
    # Output
    output_image_hash: str  # SHA256
    output_width: int
    output_height: int
    
    # Quality metrics (if computed)
    quality_metrics: dict  # {metric_name: value}
    
    # Status
    success: bool
    error: str | None

class RequestTracer:
    """Traceability system for production requests."""
    
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
    
    def create_trace(self, input_path: Path, config: dict) -> RequestTrace:
        """Initialize trace for new request."""
        trace_id = self._generate_trace_id()
        
        return RequestTrace(
            trace_id=trace_id,
            timestamp=datetime.utcnow().isoformat(),
            input_image_hash=self._hash_file(input_path),
            config_hash=self._hash_config(config),
            preset=config.get("preset", "unknown"),
            # ... populate other fields
        )
    
    def save_trace(self, trace: RequestTrace) -> None:
        """Persist trace to disk."""
        trace_file = self.trace_dir / f"{trace.trace_id}.json"
        with open(trace_file, 'w') as f:
            json.dump(asdict(trace), f, indent=2)
    
    def query_traces(
        self,
        preset: str | None = None,
        date_range: tuple | None = None,
        success_only: bool = False
    ) -> list[RequestTrace]:
        """Query historical traces."""
        # Implementation: scan trace files, filter by criteria
        pass
```

**Error Categorization**:
```python
# error_tracker.py
from enum import Enum

class ErrorCategory(Enum):
    INPUT_VALIDATION = "input_validation"
    MODEL_INFERENCE = "model_inference"
    GPU_OOM = "gpu_oom"
    SEGMENTATION_FAILURE = "segmentation_failure"
    UPSCALING_FAILURE = "upscaling_failure"
    OUTPUT_WRITE = "output_write"
    UNKNOWN = "unknown"

def categorize_error(exception: Exception, stage: str) -> ErrorCategory:
    """Categorize error for structured logging and alerting."""
    if "CUDA out of memory" in str(exception):
        return ErrorCategory.GPU_OOM
    elif stage == "segmentation":
        return ErrorCategory.SEGMENTATION_FAILURE
    # ... more categorization logic
    return ErrorCategory.UNKNOWN
```

**Grafana Dashboard**:
- Request throughput (requests/min)
- Latency percentiles (p50, p95, p99)
- Error rate by category
- GPU memory utilization
- Queue depth (if using async queue)
- Processing time breakdown by stage

**Key Architectural Decisions**:

1. **Prometheus-Compatible**: Use standard Prometheus client library for broad compatibility
2. **Full Traceability**: Every output traceable to exact config + model versions
3. **Structured Errors**: Categorize errors for targeted alerting and debugging
4. **Query Interface**: Historical trace queries for debugging production issues

**Integration Points**:
- Service: `/metrics` endpoint for Prometheus scraping
- CLI: Optional `--trace-dir` flag to enable tracing in batch mode
- Monitoring: Grafana dashboards for ops team
- Debugging: Trace query tool for reproducing production issues locally

---

## Cross-Priority Integration

### Unified Configuration

**Central Config**: `lux_depth_v2/validation_config.yaml`

```yaml
# Unified configuration for all validation systems

benchmark:
  dataset: validation_v1
  baselines:
    - topaz_gigapixel
    - adobe_super_resolution
  metrics:
    lpips: {weight: 0.40, net: alex}
    nima: {weight: 0.30}
    ssim: {weight: 0.20}
    psnr: {weight: 0.10}
  output_dir: benchmark_results

quality_gates:
  golden_images_dir: tests/golden_images
  baseline_branch: origin/main
  fail_on_regression: true
  thresholds:
    lpips_max_delta: 0.02
    ssim_min_delta: -0.01
    psnr_min_delta: -0.5
    nima_min_delta: -0.1

profiling:
  enable_gpu_sync: true
  stage_granularity: coarse  # coarse|fine
  export_format: json  # json|prometheus
  output_dir: profiling_results

segmentation_validation:
  consistency_runs: 10
  backends: [auto, onnx, heuristic]
  impact_analysis: true
  ground_truth_dir: tests/segmentation_ground_truth

observability:
  enable_tracing: true
  trace_dir: traces
  prometheus_port: 9090
  error_alerting:
    slack_webhook: ${SLACK_WEBHOOK_URL}
    email_recipients: [ops@example.com]
```

### Data Flow

```
INPUT IMAGE
    │
    ├─→ [BENCHMARK] ──→ Baseline Comparison ──→ Report Generator
    │                                                 │
    │                                                 ├─→ HTML Dashboard
    │                                                 ├─→ JSON Results
    │                                                 └─→ CI/CD Summary
    │
    ├─→ [QUALITY GATE] ──→ Golden Image Validation ──→ Regression Checker
    │                                                       │
    │                                                       ├─→ PASS (merge)
    │                                                       └─→ FAIL (block + alert)
    │
    ├─→ [PROFILER] ──→ Stage Timing ──→ Bottleneck Analyzer ──→ Optimization Advice
    │                                                               │
    │                                                               └─→ Performance Report
    │
    ├─→ [SEGMENTATION] ──→ Consistency Check + Impact Analysis ──→ Surface Metrics
    │                                                                   │
    │                                                                   └─→ Validation Report
    │
    └─→ [PIPELINE] ──→ Processing ──→ Output
                         │
                         ├─→ [TRACER] ──→ Audit Log
                         ├─→ [METRICS] ──→ Prometheus
                         └─→ [TELEMETRY] ──→ Analytics DB
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Establish core infrastructure for validation

**Deliverables**:
1. Benchmark framework architecture (`validation/benchmark/` module structure)
2. Dataset registry with version control (Git LFS setup)
3. Metric engine integration (extend existing `metrics.py`)
4. Golden image registry schema (YAML + initial 5 images)

**Acceptance Criteria**:
- Can run benchmark manually with CLI
- Metrics computed correctly (verified against known values)
- Golden images stored in Git LFS with checksums

**Critical Path**: Dataset acquisition and baseline generation

---

### Phase 2: CI/CD Integration (Week 3-4)

**Goal**: Automate quality gates in merge process

**Deliverables**:
1. GitHub Actions workflow for golden image validation
2. Regression detection logic with configurable thresholds
3. PR comment integration (post validation results)
4. Trend dashboard (initial version, CSV-based)

**Acceptance Criteria**:
- PR blocked if golden image regresses beyond threshold
- Validation report posted as PR comment
- Trend data collected and visualized

**Critical Path**: GitHub Actions GPU runner access

---

### Phase 3: Performance Optimization (Week 5-6)

**Goal**: Identify and eliminate bottlenecks

**Deliverables**:
1. Stage profiler with GPU monitoring
2. Bottleneck analysis heuristics
3. Optimization advisor (actionable recommendations)
4. Performance regression detection in CI/CD

**Acceptance Criteria**:
- Profiling report shows stage-by-stage breakdown
- Bottleneck analyzer identifies dominant stage (validated manually)
- At least one optimization implemented and measured

**Critical Path**: GPU profiling API integration (CUDA/MPS)

---

### Phase 4: Segmentation Validation (Week 7-8)

**Goal**: Validate segmentation quality and impact

**Deliverables**:
1. Consistency evaluator (10-run stability test)
2. Impact analyzer (ON vs OFF comparison)
3. Surface-specific metrics computation
4. Annotation tool (lightweight ground truth labeling)

**Acceptance Criteria**:
- Consistency score > 0.95 (IoU) for stable backends
- Impact analysis shows measurable quality improvement with segmentation
- Per-surface metrics available for all major material types

**Critical Path**: Segmentation backend stability

---

### Phase 5: Production Observability (Week 9-10)

**Goal**: Enable operational monitoring and traceability

**Deliverables**:
1. Prometheus metrics endpoint in FastAPI service
2. Request tracer with full audit trail
3. Error categorization and alerting
4. Grafana dashboard for ops team

**Acceptance Criteria**:
- Service exposes `/metrics` endpoint (Prometheus format)
- Every request logged with full traceability
- Error alerts trigger on production failures
- Grafana dashboard shows real-time metrics

**Critical Path**: Prometheus + Grafana infrastructure setup

---

## Operational Considerations

### Resource Requirements

**Compute**:
- **Benchmark Runs**: 1x GPU instance (A100/4090-class), 4-8 hours per full benchmark
- **CI/CD Validation**: GitHub Actions GPU runner (1x per PR), ~15 min per validation
- **Profiling**: Same as production (CPU/GPU as configured)
- **Production Monitoring**: Minimal overhead (<5% latency increase)

**Storage**:
- **Datasets**: ~50 GB (20 images @ 4K TIFF + baseline outputs)
- **Golden Images**: ~2 GB (5-10 images + references)
- **Traces**: ~10 MB/day (typical production load)
- **Metrics**: ~100 MB/month (Prometheus time-series)

**Personnel**:
- **Initial Setup**: 1x senior engineer, 10 weeks part-time
- **Ongoing Maintenance**: 0.5x engineer (weekly benchmark reviews, threshold tuning)

### Security Considerations

**Dataset Protection**:
- Private datasets stored in private Git LFS or S3 with access controls
- No PII or client-identifiable content in public benchmarks
- Baseline tool outputs do not include proprietary client images

**Service Hardening**:
- Rate limiting on `/metrics` endpoint to prevent DoS
- Request tracing excludes sensitive metadata (GPS coordinates, EXIF data)
- Error logs sanitized to remove file paths and user data

**CI/CD Security**:
- GPU runner isolated from production systems
- Golden images checksummed to prevent tampering
- Validation results signed to prevent spoofing

### Maintenance & Evolution

**Dataset Evolution**:
- Quarterly review of dataset composition (add new categories, remove stale images)
- Version bumps trigger full re-benchmark of all baselines
- Deprecated datasets archived but not deleted (historical comparison)

**Threshold Tuning**:
- Monthly review of false positive/negative rates in quality gates
- Thresholds adjusted based on ops team feedback
- Per-preset thresholds allow different sensitivity

**Metric Addition**:
- Plugin architecture allows new metrics without refactoring
- New metrics beta-tested in benchmark before adding to quality gates
- Weights rebalanced when new metrics added

---

## Success Metrics

### Immediate Validation (Phase 1-2)

**Objective Evidence**:
1. ✅ Benchmark report shows Lux Depth V2 wins on ≥60% of images vs Topaz
2. ✅ LPIPS improves by ≥10% on glossy surface category
3. ✅ NIMA score matches or exceeds Adobe on ≥70% of images
4. ✅ Zero regressions in golden image validation for 30 days

**Commercial Proof**:
- Publishable benchmark results with reproducible methodology
- Side-by-side comparisons for marketing materials
- Client demos backed by quantitative evidence

### Performance Targets (Phase 3)

**Objective Evidence**:
1. ✅ Upscaling time reduced by ≥30% (GPU optimization)
2. ✅ End-to-end latency <30s for 4K image @ 4x upscale (A100 GPU)
3. ✅ Memory usage <8GB VRAM for typical workloads
4. ✅ Profiling overhead <5% when enabled

**Operational Impact**:
- Higher throughput enables larger batch jobs
- Faster iteration for interactive workflows
- Lower cloud costs (shorter GPU rental time)

### Segmentation Quality (Phase 4)

**Objective Evidence**:
1. ✅ Segmentation consistency IoU >0.95 across 10 runs
2. ✅ LPIPS improves by ≥5% with segmentation ON vs OFF
3. ✅ Per-surface metrics show targeted improvements (glass +10%, wood +7%)
4. ✅ Zero segmentation-related artifacts in golden images

**System Primitive Status**:
- Segmentation trusted enough to enable by default
- Surface-specific enhancements demonstrably improve output quality
- Ablation testing proves segmentation value

### Production Readiness (Phase 5)

**Objective Evidence**:
1. ✅ 99.9% uptime over 30-day period
2. ✅ p95 latency <45s for typical requests
3. ✅ Error rate <1% with categorized failure modes
4. ✅ Full traceability for 100% of requests

**Operational Maturity**:
- Ops team can debug production issues without developer involvement
- Capacity planning informed by real production metrics
- Proactive alerting prevents outages

---

## Risk Mitigation

### Technical Risks

**Risk**: Baseline tools (Topaz, Adobe) unavailable or require licensing  
**Mitigation**: Use open-source alternatives (Real-ESRGAN, waifu2x) as secondary baselines; focus on internal consistency metrics if external baselines unavailable

**Risk**: GPU runner availability limited in GitHub Actions  
**Mitigation**: Use self-hosted runners or defer to scheduled weekly runs instead of per-PR validation

**Risk**: Segmentation consistency low (<0.90 IoU)  
**Mitigation**: Identify root cause (randomness in model, preprocessing variance); disable segmentation in presets until resolved

**Risk**: Metric disagreement (LPIPS improves, SSIM regresses)  
**Mitigation**: Weighted scoring handles trade-offs; manual review for edge cases; adjust weights based on use case priority

### Operational Risks

**Risk**: False positives in quality gates block valid PRs  
**Mitigation**: Configurable thresholds + manual override capability; weekly threshold review meetings

**Risk**: Benchmark results not reproducible across machines  
**Mitigation**: Containerized benchmark environment (Docker); pinned dependencies; checksummed datasets

**Risk**: Monitoring data volume exceeds budget  
**Mitigation**: Sampling (trace 10% of requests); retention policies (7 days detailed, 90 days aggregated)

---

## Conclusion

This validation architecture transforms Lux Depth V2 from "production-ready framework" to "validated quality breakthrough" through five integrated priorities:

1. **Benchmark Framework** provides repeatable, category-based proof of superiority
2. **Quality Gates** prevent regression and ensure sustained quality leadership
3. **Performance Profiler** enables targeted optimization with measurable impact
4. **Segmentation Validator** elevates material processing from effect to system primitive
5. **Production Observability** ensures operational excellence and traceability

The 10-week roadmap delivers incremental value at each phase while building toward comprehensive validation. Success metrics are objective, measurable, and aligned with commercial proof requirements.

**Next Steps**:
1. Approve architecture and roadmap
2. Acquire/curate validation dataset (20 representative images)
3. Generate baseline outputs (Topaz + Adobe/alternatives)
4. Begin Phase 1 implementation (Week 1-2: Benchmark Foundation)

**Critical Dependencies**:
- GPU compute access (GitHub Actions runner or self-hosted)
- Dataset curation (representative luxury real estate imagery)
- Baseline tool access (Topaz license or open-source alternatives)

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-08  
**Next Review**: After Phase 2 completion (Week 4)
