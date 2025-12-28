# Phase 2 Performance Benchmarking

This directory contains benchmarking tools for measuring Phase 2 feature overhead and production throughput validation.

## Directory Structure

```
bench/
├── README.md                           # This file
├── __init__.py                         # Python package init
├── bench_phase2.py                     # Phase 2 initialization benchmark
├── baselines/                          # Performance baselines for regression detection
│   └── throughput_baseline.json        # Throughput validation baselines
├── config/                             # Performance budget configurations
│   └── performance_budgets.yaml        # Per-operation performance targets
└── results/                            # Benchmark output files
    └── phase2_benchmark_results.json   # Phase 2 benchmark results
```

## Quick Start

### Phase 2 Initialization Benchmark

```bash
# Run benchmark (requires test images in input_images/750_Picacho/Source_JPEGS/)
python bench/bench_phase2.py

# View results
cat docs/PHASE2_PERFORMANCE.md
```

### Throughput Validation (NEW - PR #608)

```bash
# Run throughput tests
pytest tests/test_performance_throughput.py -v

# Validate against baseline
python scripts/validate_throughput.py \
  --baseline bench/baselines/throughput_baseline.json \
  --current results.json \
  --quality standard
```

## What's Measured

### Phase 2 Initialization (bench_phase2.py)
- **CLIP Classification Overhead**: Time to classify scene type/subtype (~0.15-0.36s)
- **Pipeline Initialization**: Time to load models and configure pipeline (~1s)
- **Memory Usage**: Peak memory consumption per preset (~850MB)
- **Quality Tier Comparison**: STANDARD vs MAX vs APEX initialization costs

### Throughput Validation (test_performance_throughput.py)
- **End-to-End Throughput**: Images/hour for full pipeline processing
- **Per-Image Latency**: Average time to process single image
- **Memory Consumption**: Peak memory during batch processing
- **Scaling Behavior**: Linear scaling validation and memory leak detection

## Baselines and Budgets

### Throughput Baselines (`baselines/throughput_baseline.json`)

Defines minimum acceptable performance thresholds:

- **Standard Quality (CPU)**: 50 images/hour, 2000MB memory
- **Max Quality (CPU)**: 30 images/hour, 3000MB memory  
- **Max Quality (GPU)**: 100 images/hour, 3000MB memory

Production targets (aspirational):
- **CPU Standard**: 127 images/hour
- **GPU Max**: 400 images/hour

### Performance Budgets (`config/performance_budgets.yaml`)

Per-operation performance targets:

- **Depth Estimation**: < 2.0s, < 1500MB
- **Material Segmentation**: < 1.0s, < 800MB
- **4x Upscaling**: < 10.0s, < 2000MB
- **End-to-End Pipeline**: See throughput baselines above

## Output

- **JSON**: `bench/results/phase2_benchmark_results.json` - Machine-readable results
- **Markdown**: `docs/PHASE2_PERFORMANCE.md` - Human-readable report
- **Artifacts**: CI uploads results to GitHub Actions artifacts

## CI Integration

### Phase 2 Benchmark
- **Fast**: Only measures initialization overhead (~10-15s total)
- **No Heavy Processing**: Skips actual image processing to avoid timeouts
- **Deterministic**: Single run per configuration (no warmup, no averaging)

### Throughput Validation (NEW)
- **Automated**: Runs on every PR and push to main
- **Regression Detection**: Fails CI if throughput < baseline
- **PR Comments**: Posts throughput metrics as PR comment
- **Artifact Storage**: Uploads results for trend analysis

## Related Documentation

- `docs/THROUGHPUT_VALIDATION.md` - Detailed throughput system documentation
- `.github/workflows/PERFORMANCE_READINESS_ASSESSMENT.md` - Full performance roadmap
- `.github/workflows/ci-consolidated.yml` - CI integration (test-throughput job)

## Future Enhancements

- End-to-end processing benchmarks (requires dedicated hardware)
- EfficientSAM vs SegFormer backend comparison
- Lighting detection overhead measurement
- GPU comparison matrix (CUDA vs MPS vs CPU)
- Baseline versioning and historical trend tracking
- Latency percentile validation (P50/P95/P99)

