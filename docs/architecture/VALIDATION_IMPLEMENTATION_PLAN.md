# Validation System Implementation Plan

**Status**: Implementation Specification  
**Author**: Transformation Portal Architect  
**Date**: 2025-12-08  
**Parent**: VALIDATION_ARCHITECTURE.md

---

## Overview

This document provides concrete implementation specifications for the 5-priority validation system. Each section includes module structure, API contracts, integration points, and acceptance tests.

---

## Priority 1: Benchmark Framework

### Module Structure

```
lux_depth_v2/validation/benchmark/
├── __init__.py
├── dataset_registry.py      # Dataset versioning and management
├── baseline_runner.py        # Execute baseline tools
├── metric_engine.py          # Compute quality metrics
├── category_scorer.py        # Category-based aggregation
├── report_generator.py       # HTML/MD report generation
├── templates/                # Report templates
│   ├── index.html.j2
│   └── summary.md.j2
└── configs/                  # Benchmark configurations
    └── validation_v1.yaml
```

### API Contract

```python
# dataset_registry.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class ImageCategory:
    name: str
    images: List[Path]
    weight: float
    description: str

@dataclass
class BenchmarkDataset:
    version: str
    name: str
    categories: List[ImageCategory]
    root_dir: Path
    
    def validate(self) -> bool:
        """Verify all images exist and checksums match."""
        pass
    
    def get_image_category(self, image_path: Path) -> str:
        """Return category for given image."""
        pass

class DatasetRegistry:
    """Manage versioned benchmark datasets."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.datasets: Dict[str, BenchmarkDataset] = {}
    
    def load_dataset(self, name: str, version: str = "latest") -> BenchmarkDataset:
        """Load dataset by name and version."""
        pass
    
    def register_dataset(self, dataset: BenchmarkDataset) -> None:
        """Register new dataset version."""
        pass
```

```python
# baseline_runner.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess

@dataclass
class BaselineConfig:
    name: str
    version: str
    command_template: str  # e.g., "topaz gigapixel --scale 4 {input} {output}"
    method: str  # "cli" or "manual"
    
class BaselineRunner:
    """Execute baseline tools and cache results."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run_baseline(
        self,
        baseline: BaselineConfig,
        input_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """Run baseline tool on input image.
        
        Returns:
            Path to output image (from cache if available)
        """
        # Check cache
        cache_key = self._compute_cache_key(baseline, input_path)
        cached_output = self.cache_dir / f"{cache_key}.tif"
        
        if cached_output.exists():
            return cached_output
        
        # Run baseline
        if baseline.method == "cli":
            self._run_cli(baseline, input_path, cached_output)
        elif baseline.method == "manual":
            raise RuntimeError(f"Manual baseline {baseline.name} requires human operator")
        
        return cached_output
    
    def _run_cli(self, baseline: BaselineConfig, input_path: Path, output_path: Path):
        """Execute CLI-based baseline."""
        cmd = baseline.command_template.format(
            input=str(input_path),
            output=str(output_path)
        )
        subprocess.run(cmd, shell=True, check=True)
```

```python
# metric_engine.py
from typing import Dict
import numpy as np
from ..metrics import compute_lpips, compute_nima, compute_ssim, compute_psnr

class MetricEngine:
    """Compute quality metrics for benchmark."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
    
    def compute_all(
        self,
        test_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute all configured metrics."""
        metrics = {}
        
        # No-reference metrics
        metrics["nima"] = compute_nima(test_image, device=self.device)
        
        # Reference-based metrics
        if reference_image is not None:
            metrics["lpips"] = compute_lpips(test_image, reference_image, device=self.device)
            metrics["ssim"] = compute_ssim(test_image, reference_image)
            metrics["psnr"] = compute_psnr(test_image, reference_image)
        
        return metrics
```

```python
# category_scorer.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CategoryScore:
    category: str
    images: List[str]
    metrics: Dict[str, float]  # metric -> avg score
    wins: int  # Number of images where this method wins
    losses: int
    ties: int

@dataclass
class BenchmarkScore:
    method_name: str
    overall_score: float
    category_scores: List[CategoryScore]
    weighted_score: float

class CategoryScorer:
    """Aggregate metrics by category with configurable weights."""
    
    def __init__(self, metric_weights: Dict[str, float]):
        self.metric_weights = metric_weights
    
    def compute_scores(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],  # image -> method -> metrics
        categories: Dict[str, List[str]]  # category -> image list
    ) -> Dict[str, BenchmarkScore]:
        """Compute category-based scores for each method."""
        pass
    
    def compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted score from metric values."""
        score = 0.0
        for metric, value in metrics.items():
            weight = self.metric_weights.get(metric, 0.0)
            
            # Normalize to 0-1 scale (metric-specific)
            if metric == "lpips":
                normalized = 1.0 - min(value, 1.0)  # Lower is better
            elif metric == "nima":
                normalized = value / 10.0  # Scale 1-10 to 0-1
            elif metric == "ssim":
                normalized = value  # Already 0-1
            elif metric == "psnr":
                normalized = min(value / 50.0, 1.0)  # Cap at 50 dB
            else:
                normalized = value
            
            score += weight * normalized
        
        return score
```

```python
# report_generator.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import jinja2

@dataclass
class BenchmarkReport:
    dataset_name: str
    dataset_version: str
    timestamp: str
    methods: List[str]
    scores: Dict[str, BenchmarkScore]
    per_image_results: Dict[str, Dict[str, Dict[str, float]]]

class ReportGenerator:
    """Generate HTML and Markdown benchmark reports."""
    
    def __init__(self, template_dir: Path):
        self.template_dir = template_dir
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir)
        )
    
    def generate_html(self, report: BenchmarkReport, output_path: Path) -> None:
        """Generate interactive HTML dashboard."""
        template = self.jinja_env.get_template("index.html.j2")
        html = template.render(report=report)
        output_path.write_text(html)
    
    def generate_markdown(self, report: BenchmarkReport, output_path: Path) -> None:
        """Generate text-based summary for CI/CD."""
        template = self.jinja_env.get_template("summary.md.j2")
        md = template.render(report=report)
        output_path.write_text(md)
    
    def generate_comparison_images(
        self,
        report: BenchmarkReport,
        output_dir: Path
    ) -> None:
        """Generate side-by-side comparison images."""
        pass
```

### CLI Interface

```python
# lux_depth_v2/validation/benchmark/__main__.py
import typer
from pathlib import Path
from . import DatasetRegistry, BaselineRunner, MetricEngine, CategoryScorer, ReportGenerator

app = typer.Typer()

@app.command()
def run_benchmark(
    dataset: str = typer.Option("validation_v1", help="Dataset name"),
    output_dir: Path = typer.Option(Path("benchmark_results"), help="Output directory"),
    baselines: List[str] = typer.Option(["topaz_gigapixel"], help="Baseline methods to compare"),
    device: str = typer.Option("auto", help="Device for metric computation"),
):
    """Run full benchmark comparing Lux Depth V2 against baselines."""
    
    # Load dataset
    registry = DatasetRegistry(Path("lux_depth_v2/validation/benchmark/configs"))
    dataset_obj = registry.load_dataset(dataset)
    
    # Initialize components
    baseline_runner = BaselineRunner(cache_dir=output_dir / "baseline_cache")
    metric_engine = MetricEngine(device=device)
    
    # Run benchmark
    results = {}
    for category in dataset_obj.categories:
        for image_path in category.images:
            typer.echo(f"Processing {image_path.name}...")
            
            # Run baselines
            baseline_outputs = {}
            for baseline_name in baselines:
                baseline_config = load_baseline_config(baseline_name)
                baseline_output = baseline_runner.run_baseline(baseline_config, image_path)
                baseline_outputs[baseline_name] = baseline_output
            
            # Run Lux Depth V2
            lux_output = run_lux_pipeline(image_path)
            
            # Compute metrics
            reference = load_image(image_path)
            lux_metrics = metric_engine.compute_all(lux_output, reference)
            
            results[image_path.name] = {"lux_depth_v2": lux_metrics}
            for baseline_name, baseline_output in baseline_outputs.items():
                baseline_metrics = metric_engine.compute_all(baseline_output, reference)
                results[image_path.name][baseline_name] = baseline_metrics
    
    # Generate report
    scorer = CategoryScorer(metric_weights={"lpips": 0.4, "nima": 0.3, "ssim": 0.2, "psnr": 0.1})
    scores = scorer.compute_scores(results, dataset_obj.categories)
    
    report_gen = ReportGenerator(template_dir=Path("lux_depth_v2/validation/benchmark/templates"))
    report = BenchmarkReport(...)
    report_gen.generate_html(report, output_dir / "index.html")
    report_gen.generate_markdown(report, output_dir / "summary.md")
    
    typer.echo(f"Benchmark complete. Report: {output_dir / 'index.html'}")

if __name__ == "__main__":
    app()
```

### Acceptance Tests

```python
# tests/validation/test_benchmark.py
import pytest
from pathlib import Path
from lux_depth_v2.validation.benchmark import DatasetRegistry, BaselineRunner, MetricEngine

def test_dataset_registry_loads_config():
    """Test dataset registry loads and validates configuration."""
    registry = DatasetRegistry(Path("lux_depth_v2/validation/benchmark/configs"))
    dataset = registry.load_dataset("validation_v1")
    
    assert dataset.version == "1.0.0"
    assert len(dataset.categories) > 0
    assert sum(c.weight for c in dataset.categories) == pytest.approx(1.0)

def test_baseline_runner_caches_results():
    """Test baseline runner uses cache for repeated runs."""
    runner = BaselineRunner(cache_dir=Path("/tmp/baseline_cache"))
    
    # First run
    output1 = runner.run_baseline(baseline_config, test_image)
    
    # Second run should use cache
    output2 = runner.run_baseline(baseline_config, test_image)
    
    assert output1 == output2
    assert output1.exists()

def test_metric_engine_computes_all_metrics():
    """Test metric engine computes all configured metrics."""
    engine = MetricEngine(device="cpu")
    
    test_img = np.random.rand(256, 256, 3).astype(np.float32)
    ref_img = test_img + 0.01  # Slight perturbation
    
    metrics = engine.compute_all(test_img, ref_img)
    
    assert "lpips" in metrics
    assert "nima" in metrics
    assert "ssim" in metrics
    assert "psnr" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
```

---

## Priority 2: CI/CD Quality Gates

### Module Structure

```
lux_depth_v2/validation/golden/
├── __init__.py
├── registry.py              # Golden image management
├── validator.py             # Run validation against golden images
├── regression_checker.py    # Detect quality regressions
└── trend_tracker.py         # Track metrics over time

tests/golden_images/
├── registry.yaml            # Golden image metadata
├── sources/                 # Input images
└── references/              # Reference outputs from main branch
```

### API Contract

```python
# registry.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class GoldenImageThresholds:
    lpips_max_delta: float = 0.02
    ssim_min_delta: float = -0.01
    psnr_min_delta: float = -0.5
    nima_min_delta: float = -0.1

@dataclass
class GoldenImage:
    name: str
    source_path: Path
    reference_path: Path
    preset: str
    failure_mode: str
    thresholds: GoldenImageThresholds
    metadata: Dict[str, str]

class GoldenImageRegistry:
    """Manage golden images for quality gate validation."""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.golden_images: Dict[str, GoldenImage] = {}
    
    def load_registry(self) -> None:
        """Load golden images from YAML registry."""
        pass
    
    def add_golden_image(self, golden: GoldenImage) -> None:
        """Add new golden image to registry."""
        pass
    
    def update_references(self, baseline_branch: str = "main") -> None:
        """Update reference outputs from baseline branch."""
        pass
```

```python
# validator.py
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ValidationResult:
    golden_name: str
    preset: str
    metrics_current: Dict[str, float]
    metrics_baseline: Dict[str, float]
    deltas: Dict[str, float]
    passed: bool
    failed_metrics: List[str]

class GoldenValidator:
    """Validate current outputs against golden image baselines."""
    
    def __init__(self, registry: GoldenImageRegistry):
        self.registry = registry
    
    def validate_all(self, output_dir: Path) -> List[ValidationResult]:
        """Run validation for all golden images."""
        results = []
        
        for golden in self.registry.golden_images.values():
            result = self.validate_single(golden, output_dir)
            results.append(result)
        
        return results
    
    def validate_single(self, golden: GoldenImage, output_dir: Path) -> ValidationResult:
        """Validate single golden image."""
        # Run pipeline on source
        current_output = self._run_pipeline(golden.source_path, golden.preset)
        
        # Load baseline reference
        baseline_output = self._load_image(golden.reference_path)
        
        # Compute metrics
        metrics_current = self._compute_metrics(current_output, golden.source_path)
        metrics_baseline = self._compute_metrics(baseline_output, golden.source_path)
        
        # Check thresholds
        deltas = {k: metrics_current[k] - metrics_baseline[k] for k in metrics_current}
        failed_metrics = self._check_thresholds(deltas, golden.thresholds)
        
        return ValidationResult(
            golden_name=golden.name,
            preset=golden.preset,
            metrics_current=metrics_current,
            metrics_baseline=metrics_baseline,
            deltas=deltas,
            passed=len(failed_metrics) == 0,
            failed_metrics=failed_metrics
        )
```

```python
# regression_checker.py
from typing import List

class RegressionChecker:
    """Detect quality regressions from validation results."""
    
    def check_regressions(
        self,
        results: List[ValidationResult],
        fail_on_regression: bool = True
    ) -> bool:
        """Check if any validation result indicates regression.
        
        Returns:
            True if no regressions, False otherwise
        """
        regressions = [r for r in results if not r.passed]
        
        if regressions:
            self._log_regressions(regressions)
            if fail_on_regression:
                return False
        
        return True
    
    def _log_regressions(self, regressions: List[ValidationResult]) -> None:
        """Log detailed regression information."""
        for result in regressions:
            print(f"❌ Regression in {result.golden_name}:")
            for metric in result.failed_metrics:
                delta = result.deltas[metric]
                print(f"  {metric}: {delta:+.4f} (threshold exceeded)")
```

### GitHub Actions Workflow

```yaml
# .github/workflows/quality-gate-validation.yml
name: Quality Gate - Golden Image Validation

on:
  pull_request:
    paths:
      - 'lux_depth_v2/**'
      - 'tests/golden_images/**'
  workflow_dispatch:

jobs:
  golden_validation:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@v4
        with:
          lfs: true
      
      - name: Checkout baseline (main)
        uses: actions/checkout@v4
        with:
          ref: main
          path: baseline
          lfs: true
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -e ".[validation]"
      
      - name: Generate baseline references
        run: |
          cd baseline
          python -m lux_depth_v2.validation.golden.validator \
            --generate-references \
            --output tests/golden_images/references
      
      - name: Run golden image validation
        id: validate
        run: |
          python -m lux_depth_v2.validation.golden.validator \
            --registry tests/golden_images/registry.yaml \
            --baseline-dir baseline/tests/golden_images/references \
            --output golden_validation_report.md
      
      - name: Check for regressions
        run: |
          python -m lux_depth_v2.validation.golden.regression_checker \
            --report golden_validation_report.md \
            --fail-on-regression
      
      - name: Upload validation artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: golden-validation-artifacts
          path: |
            golden_validation_report.md
            golden_validation_outputs/
      
      - name: Post PR comment
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('golden_validation_report.md', 'utf8');
            
            const header = '## 🎨 Golden Image Validation';
            const body = context.payload.pull_request?.number 
              ? await github.rest.issues.listComments({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  issue_number: context.payload.pull_request.number
                })
              : { data: [] };
            
            const existingComment = body.data.find(c => c.body.startsWith(header));
            
            const commentBody = `${header}\n\n${report}`;
            
            if (existingComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: existingComment.id,
                body: commentBody
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.payload.pull_request.number,
                body: commentBody
              });
            }
```

### Acceptance Tests

```python
# tests/validation/test_golden_validator.py
import pytest
from pathlib import Path
from lux_depth_v2.validation.golden import GoldenImageRegistry, GoldenValidator

def test_golden_registry_loads_config():
    """Test golden image registry loads configuration."""
    registry = GoldenImageRegistry(Path("tests/golden_images/registry.yaml"))
    registry.load_registry()
    
    assert len(registry.golden_images) > 0
    assert all(g.source_path.exists() for g in registry.golden_images.values())

def test_validator_detects_regression():
    """Test validator detects quality regression."""
    registry = GoldenImageRegistry(Path("tests/golden_images/registry.yaml"))
    registry.load_registry()
    
    validator = GoldenValidator(registry)
    results = validator.validate_all(output_dir=Path("/tmp/validation_output"))
    
    # All should pass if no changes
    assert all(r.passed for r in results)

def test_regression_checker_fails_on_regression():
    """Test regression checker fails when threshold exceeded."""
    results = [
        ValidationResult(
            golden_name="test",
            preset="photo_realistic",
            metrics_current={"lpips": 0.15},
            metrics_baseline={"lpips": 0.10},
            deltas={"lpips": 0.05},  # Exceeds 0.02 threshold
            passed=False,
            failed_metrics=["lpips"]
        )
    ]
    
    checker = RegressionChecker()
    assert not checker.check_regressions(results, fail_on_regression=True)
```

---

## Priority 3: Performance Profiler

### Module Structure

```
lux_depth_v2/profiling/
├── __init__.py
├── stage_profiler.py         # Fine-grained stage timing
├── gpu_monitor.py            # GPU utilization tracking
├── bottleneck_analyzer.py    # Identify optimization targets
├── optimization_advisor.py   # Generate recommendations
└── reports/                  # Profiling report templates
```

### API Contract

```python
# stage_profiler.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from contextlib import contextmanager
import time

@dataclass
class StageProfile:
    name: str
    duration_s: float
    percent_of_total: float
    cpu_percent: float
    gpu_percent: float
    gpu_mem_mb: float
    io_wait_s: float

class PipelineProfiler:
    """Fine-grained profiling for Lux Depth V2 pipeline."""
    
    def __init__(self, enable_gpu_sync: bool = True):
        self.enable_gpu_sync = enable_gpu_sync
        self.stages: List[StageProfile] = []
        self.start_time: Optional[float] = None
    
    def start(self) -> None:
        """Start profiling session."""
        self.start_time = time.perf_counter()
        self.stages = []
    
    @contextmanager
    def stage(self, name: str):
        """Profile a pipeline stage."""
        if self.enable_gpu_sync:
            self._sync_device()
        
        start = time.perf_counter()
        start_gpu_mem = self._get_gpu_memory()
        
        try:
            yield
        finally:
            if self.enable_gpu_sync:
                self._sync_device()
            
            duration = time.perf_counter() - start
            end_gpu_mem = self._get_gpu_memory()
            
            self.stages.append(StageProfile(
                name=name,
                duration_s=duration,
                percent_of_total=0.0,  # Calculated in finalize()
                cpu_percent=self._get_cpu_percent(),
                gpu_percent=self._get_gpu_percent(),
                gpu_mem_mb=end_gpu_mem,
                io_wait_s=self._estimate_io_wait(duration)
            ))
    
    def finalize(self) -> Dict:
        """Finalize profiling and generate report."""
        total_time = sum(s.duration_s for s in self.stages)
        
        for stage in self.stages:
            stage.percent_of_total = (stage.duration_s / total_time * 100) if total_time > 0 else 0.0
        
        return {
            "total_duration_s": total_time,
            "stages": [asdict(s) for s in self.stages],
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Find stages taking >30% of total time
        for stage in self.stages:
            if stage.percent_of_total > 30.0:
                bottlenecks.append({
                    "stage": stage.name,
                    "issue": "Dominant stage",
                    "percent": stage.percent_of_total,
                    "severity": "high" if stage.percent_of_total > 50 else "medium"
                })
        
        # Check for I/O bottlenecks
        total_io = sum(s.io_wait_s for s in self.stages)
        total_time = sum(s.duration_s for s in self.stages)
        if total_io / total_time > 0.3:
            bottlenecks.append({
                "stage": "I/O",
                "issue": "High I/O wait time",
                "percent": (total_io / total_time * 100),
                "severity": "medium"
            })
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        for bn in bottlenecks:
            if bn["stage"] == "upscaling":
                recommendations.append("Consider GPU acceleration for upscaling stage")
                recommendations.append("Try reducing tile size to fit in GPU memory")
            elif bn["stage"] == "I/O":
                recommendations.append("Use faster storage (NVMe SSD)")
                recommendations.append("Enable parallel image loading")
            elif "segmentation" in bn["stage"]:
                recommendations.append("Try lighter segmentation backend (heuristic)")
                recommendations.append("Reduce segmentation input resolution")
        
        return recommendations
```

### Integration with Pipeline

```python
# Modified pipeline.py to support profiling
from .profiling import PipelineProfiler

def process_image(
    image_path: Path,
    config: PipelineConfig,
    profiler: Optional[PipelineProfiler] = None
) -> np.ndarray:
    """Process image with optional profiling."""
    
    if profiler:
        profiler.start()
    
    # Load image
    with (profiler.stage("load") if profiler else nullcontext()):
        img = load_image(image_path)
    
    # Segmentation
    with (profiler.stage("segmentation") if profiler else nullcontext()):
        masks = segment_materials(img, config)
    
    # Upscaling
    with (profiler.stage("upscaling") if profiler else nullcontext()):
        upscaled = upscale_image(img, config)
    
    # Post-processing
    with (profiler.stage("post_processing") if profiler else nullcontext()):
        output = apply_post_processing(upscaled, masks, config)
    
    # Save
    with (profiler.stage("save") if profiler else nullcontext()):
        save_image(output, output_path)
    
    if profiler:
        return profiler.finalize()
    
    return output
```

---

## Priority 4: Material Segmentation Validation

### Module Structure

```
lux_depth_v2/validation/segmentation/
├── __init__.py
├── consistency_evaluator.py  # Mask stability across runs
├── impact_analyzer.py        # ON vs OFF comparison
├── surface_metrics.py        # Per-material metrics
└── annotation_tool.py        # Ground truth labeling
```

### API Contract

```python
# consistency_evaluator.py
from typing import Dict, List
import numpy as np

def evaluate_consistency(
    image_path: Path,
    num_runs: int = 10,
    backend: str = "auto",
    config: Optional[SegmentationConfig] = None
) -> Dict[str, float]:
    """Measure segmentation mask stability across multiple runs.
    
    Returns:
        Consistency metrics:
        - iou_mean: Average IoU between consecutive runs
        - iou_std: Standard deviation of IoU
        - label_agreement: Fraction of pixels with consistent labels
        - entropy_mean: Average spatial entropy of masks
    """
    segmenter = create_material_segmenter(backend, config)
    
    masks = []
    for run_idx in range(num_runs):
        img = load_image(image_path)
        mask = segmenter.segment(img)
        masks.append(mask)
    
    # Compute pairwise IoU
    ious = []
    for i in range(len(masks) - 1):
        iou = compute_mask_iou(masks[i], masks[i+1])
        ious.append(iou)
    
    # Label agreement: mode label at each pixel
    label_agreement = compute_label_agreement(masks)
    
    # Spatial entropy: measure mask complexity
    entropies = [compute_spatial_entropy(m) for m in masks]
    
    return {
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "label_agreement": label_agreement,
        "entropy_mean": float(np.mean(entropies))
    }
```

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
    lpips_improvement: float
    
    nima_without_seg: float
    nima_with_seg: float
    nima_improvement: float
    
    # Per-surface metrics
    surface_impacts: Dict[str, Dict[str, float]]
    
    # Segmentation quality
    consistency_score: float
    coverage_percent: float

def measure_segmentation_impact(
    image_path: Path,
    preset: str,
    device: str = "auto"
) -> SegmentationImpact:
    """Measure quality impact of segmentation."""
    
    from ..metrics import compute_lpips, compute_nima
    
    # Baseline: no segmentation
    config_no_seg = PipelineConfig(preset=preset)
    config_no_seg.segmentation.backend = "none"
    output_no_seg = process_image(image_path, config_no_seg)
    
    # With segmentation
    config_with_seg = PipelineConfig(preset=preset)
    output_with_seg = process_image(image_path, config_with_seg)
    
    # Load reference
    reference = load_image(image_path)
    
    # Compute overall metrics
    lpips_no_seg = compute_lpips(output_no_seg, reference, device=device)
    lpips_with_seg = compute_lpips(output_with_seg, reference, device=device)
    
    nima_no_seg = compute_nima(output_no_seg, device=device)
    nima_with_seg = compute_nima(output_with_seg, device=device)
    
    # Per-surface analysis
    segmenter = create_material_segmenter("auto", config_with_seg.segmentation)
    masks = segmenter.segment(load_image(image_path))
    
    surface_impacts = {}
    for surface_name in masks.keys():
        mask = masks[surface_name]
        surface_lpips_no = compute_masked_lpips(output_no_seg, reference, mask, device)
        surface_lpips_with = compute_masked_lpips(output_with_seg, reference, mask, device)
        
        surface_impacts[surface_name] = {
            "lpips_improvement": surface_lpips_with - surface_lpips_no,
            "coverage_percent": (mask.sum() / mask.size * 100)
        }
    
    # Consistency
    consistency = evaluate_consistency(image_path, num_runs=5, backend=config_with_seg.segmentation.backend)
    
    return SegmentationImpact(
        lpips_without_seg=lpips_no_seg,
        lpips_with_seg=lpips_with_seg,
        lpips_improvement=lpips_with_seg - lpips_no_seg,
        nima_without_seg=nima_no_seg,
        nima_with_seg=nima_with_seg,
        nima_improvement=nima_with_seg - nima_no_seg,
        surface_impacts=surface_impacts,
        consistency_score=consistency["iou_mean"],
        coverage_percent=sum(si["coverage_percent"] for si in surface_impacts.values())
    )
```

---

## Priority 5: Production Observability

### Module Structure

```
lux_depth_v2/observability/
├── __init__.py
├── prometheus_metrics.py     # Prometheus metric definitions
├── request_tracer.py         # Request traceability
├── error_tracker.py          # Error categorization
└── dashboards/
    └── grafana_dashboard.json
```

### API Contract

```python
# prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
REQUEST_COUNT = Counter(
    "lux_requests_total",
    "Total requests processed",
    ["preset", "status", "upscaler", "device"]
)

REQUEST_DURATION = Histogram(
    "lux_request_duration_seconds",
    "Request processing duration",
    ["preset", "stage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

GPU_MEMORY = Gauge(
    "lux_gpu_memory_bytes",
    "Current GPU memory usage",
    ["device"]
)

ERROR_COUNT = Counter(
    "lux_errors_total",
    "Total errors",
    ["error_type", "stage", "preset"]
)

PIPELINE_INFO = Info(
    "lux_pipeline_info",
    "Pipeline version and configuration"
)

# Initialize with current version
PIPELINE_INFO.info({
    "version": "2.0.0",
    "build": "20251208",
    "default_preset": "photo_realistic"
})
```

```python
# service.py integration
from fastapi import FastAPI, Request
from prometheus_client import generate_latest
from .observability.prometheus_metrics import REQUEST_COUNT, REQUEST_DURATION, ERROR_COUNT

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    preset = request.query_params.get("preset", "unknown")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        status = "success" if response.status_code == 200 else "error"
        REQUEST_COUNT.labels(preset=preset, status=status, upscaler="auto", device="auto").inc()
        return response
    except Exception as e:
        REQUEST_COUNT.labels(preset=preset, status="error", upscaler="auto", device="auto").inc()
        ERROR_COUNT.labels(error_type=type(e).__name__, stage="handler", preset=preset).inc()
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(preset=preset, stage="total").observe(duration)
```

---

## Execution Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Create benchmark module structure
- [ ] Implement dataset registry with YAML config
- [ ] Extend metrics.py with category scorer
- [ ] Set up Git LFS for datasets
- [ ] Create 5 initial golden images
- [ ] Write acceptance tests for Priority 1

### Phase 2: CI/CD Integration (Week 3-4)
- [ ] Implement golden image validator
- [ ] Create regression checker
- [ ] Build GitHub Actions workflow
- [ ] Add PR comment integration
- [ ] Set up trend tracking (CSV-based)
- [ ] Test workflow on dummy PRs

### Phase 3: Performance (Week 5-6)
- [ ] Implement stage profiler
- [ ] Add GPU monitoring (CUDA/MPS)
- [ ] Build bottleneck analyzer
- [ ] Create optimization advisor
- [ ] Integrate profiler into pipeline
- [ ] Profile test suite and identify bottleneck

### Phase 4: Segmentation (Week 7-8)
- [ ] Implement consistency evaluator
- [ ] Build impact analyzer
- [ ] Add surface-specific metrics
- [ ] Create annotation tool (optional)
- [ ] Run validation on all presets
- [ ] Document segmentation quality

### Phase 5: Observability (Week 9-10)
- [ ] Add Prometheus metrics to service
- [ ] Implement request tracer
- [ ] Build error categorization
- [ ] Create Grafana dashboard
- [ ] Test monitoring under load
- [ ] Write ops documentation

---

## Files Changed Summary

New files created:
- `docs/architecture/VALIDATION_ARCHITECTURE.md` - System architecture design
- `docs/architecture/VALIDATION_IMPLEMENTATION_PLAN.md` - Implementation specifications

---

SUCCEEDED
