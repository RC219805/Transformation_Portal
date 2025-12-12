# Phase 2 Performance Benchmarking

This directory contains benchmarking tools for measuring Phase 2 feature overhead.

## Quick Start

```bash
# Run benchmark (requires test images in input_images/750_Picacho/Source_JPEGS/)
python bench/bench_phase2.py

# View results
cat docs/PHASE2_PERFORMANCE.md
```

## What's Measured

- **CLIP Classification Overhead**: Time to classify scene type/subtype (~0.15-0.36s)
- **Pipeline Initialization**: Time to load models and configure pipeline (~1s)
- **Memory Usage**: Peak memory consumption per preset (~850MB)
- **Quality Tier Comparison**: STANDARD vs MAX vs APEX initialization costs

## Output

- **JSON**: `bench/results/phase2_benchmark_results.json` - Machine-readable results
- **Markdown**: `docs/PHASE2_PERFORMANCE.md` - Human-readable report

## CI Integration

This benchmark is designed to be CI-friendly:
- **Fast**: Only measures initialization overhead (~10-15s total)
- **No Heavy Processing**: Skips actual image processing to avoid timeouts
- **Deterministic**: Single run per configuration (no warmup, no averaging)

## Future Enhancements

- End-to-end processing benchmarks (requires dedicated hardware)
- EfficientSAM vs SegFormer backend comparison
- Lighting detection overhead measurement
- GPU comparison matrix (CUDA vs MPS vs CPU)
