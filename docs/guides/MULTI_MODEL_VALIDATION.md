# Multi-Model Depth Validation Framework

## Overview

Comprehensive A/B testing framework for comparing Depth Anything V2 model variants across:
- **Relative depth models**: Large, Giant (1.3B params)
- **Metric depth models**: Indoor (0-20m), Outdoor (0-80m)
- **Input size sweeps**: 518, 768, 896, 1022 (patch-aligned for ViT/14)
- **Statistical rigor**: Paired t-tests, McNemar's test, effect sizes, confidence intervals

## Quick Start

### 1. Run Multi-Model Comparison

```bash
python scripts/run_multi_model_comparison.py \
    --input-dir data/validation_full \
    --labels data/validation_full/labels.csv \
    --models DA2_Large DA2_Metric_Indoor DA2_Metric_Outdoor \
    --sweep-sizes 518 768 896 1022 \
    --output-root outputs/model_comparison
```

### 2. Analyze Results Statistically

```bash
python scripts/analyze_model_comparison.py \
    --comparison-dir outputs/model_comparison/run_20251218_* \
    --baseline-model DA2_Large \
    --confidence-level 0.95
```

### 3. View Results

```bash
# Summary CSVs
cat outputs/model_comparison/run_*/comparison_overall.csv
cat outputs/model_comparison/run_*/best_per_model.csv

# Statistical analysis
cat outputs/model_comparison/run_*/analysis/statistical_summary.csv
```

## Supported Models

### Relative Depth (Affine-Invariant)

| Model Key | HF ID | Type | VRAM | Description |
|-----------|-------|------|------|-------------|
| `DA2_Large` | `depth-anything/Depth-Anything-V2-Large-hf` | relative | 12GB | Baseline high-quality relative depth |
| `DA2_Giant` | `depth-anything/Depth-Anything-V2-Giant-hf` | relative | 24GB | Maximum capacity (1.3B params) |

### Metric Depth (Absolute in Meters)

| Model Key | HF ID | Type | VRAM | Range | Use Case |
|-----------|-------|------|------|-------|----------|
| `DA2_Metric_Indoor` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | metric | 12GB | 0-20m | Interior scenes |
| `DA2_Metric_Outdoor` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | metric | 12GB | 0-80m | Exterior/aerial |

## Architecture

### Multi-Model Runner (`run_multi_model_comparison.py`)

**What it does:**
- Runs each model across all input sizes (518, 768, 896, 1022)
- Calls `production_depth_validation_fixed.py` with model override
- Collects per-image metrics JSON files
- Generates cross-model comparison CSVs and JSON summary

**Key outputs:**
```
outputs/model_comparison/run_TIMESTAMP_SHA/
├── model_DA2_Large/
│   ├── input_518/
│   │   ├── *_metrics.json (per-image)
│   │   └── validation_report.json
│   ├── input_768/
│   └── input_1022/
├── model_DA2_Metric_Indoor/
├── comparison_overall.csv
├── comparison_by_model.csv
├── best_per_model.csv
└── model_comparison_summary.json
```

### Statistical Analyzer (`analyze_model_comparison.py`)

**What it does:**
- Loads metrics from all models
- Performs paired t-tests on continuous metrics (edge_f1, chamfer_px)
- Performs McNemar's test on binary outcomes (lenient/strict pass)
- Computes effect sizes (Cohen's d) and confidence intervals
- Stratifies results by scene type

**Key outputs:**
```
outputs/model_comparison/run_*/analysis/
├── statistical_comparison.json (full results)
└── statistical_summary.csv (effect sizes + p-values)
```

## Usage Examples

### Test All Default Models

```bash
python scripts/run_multi_model_comparison.py \
    --input-dir data/validation_full \
    --labels data/validation_full/labels.csv \
    --models all \
    --sweep-sizes 518 768 896 1022
```

### Test Only Metric Models (Indoor vs Outdoor)

```bash
python scripts/run_multi_model_comparison.py \
    --input-dir data/validation_full \
    --labels data/validation_full/labels.csv \
    --models DA2_Metric_Indoor DA2_Metric_Outdoor \
    --sweep-sizes 518 768
```

### Test Giant Model (Requires High VRAM)

```bash
python scripts/run_multi_model_comparison.py \
    --input-dir data/validation_full \
    --labels data/validation_full/labels.csv \
    --models DA2_Large DA2_Giant \
    --sweep-sizes 518 768 \
    --skip-vram-check  # Use with caution
```

## Statistical Interpretation

### Continuous Metrics (Paired t-test)

Example output:
```
[✓] edge_f1: Δ=0.0823, p=0.0012, d=0.45
```

- **Δ (mean difference)**: Treatment improved edge_f1 by 0.0823 on average
- **p-value**: Probability this difference is random (p<0.05 = significant)
- **d (Cohen's d)**: Effect size (0.2=small, 0.5=medium, 0.8=large)
- **✓**: Statistically significant at α=0.05

### Binary Metrics (McNemar's test)

Example output:
```
[✓] lenient_pass: Δ=8, p=0.0234
```

- **Δ**: Treatment passed 8 more images than baseline
- **p-value**: Probability this difference is random
- **✓**: Significant change in pass rate

### Stratified Analysis

Breaks down results by scene type (texture vs structure):

```csv
scene_type,n,baseline_lenient_rate,treatment_lenient_rate,lenient_rate_delta,edge_f1_delta
texture_dominated,32,0.875,0.938,0.063,0.012
structure_dominated,18,0.333,0.556,0.222,0.095
```

**Interpretation**: Treatment model improved structure scenes more than texture (Δ=0.222 vs 0.063).

## Integration with CI/CD

### GitHub Actions Workflow (Example)

```yaml
name: Multi-Model Validation

on:
  pull_request:
    paths:
      - 'high_fidelity_depth/**'
      - 'lux_depth_v2/**'

jobs:
  model-comparison:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install scipy scikit-learn pandas
      
      - name: Run multi-model validation
        run: |
          python scripts/run_multi_model_comparison.py \
            --input-dir data/validation_quick \
            --labels data/validation_quick/labels.csv \
            --models DA2_Large DA2_Metric_Indoor \
            --sweep-sizes 518 768
      
      - name: Analyze results
        run: |
          python scripts/analyze_model_comparison.py \
            --comparison-dir outputs/model_comparison/run_* \
            --baseline-model DA2_Large
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-comparison
          path: outputs/model_comparison/
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Quick validation (7-10 images)
--input-dir data/validation_quick

# Full validation (50+ images)
--input-dir data/validation_full
```

### 2. Use Stratified Datasets

Ensure `labels.csv` has balanced representation:
```csv
filename,scene_type,notes
pool_aerial.jpg,texture_dominated,Ocean shimmer
interior_kitchen.jpg,structure_dominated,Strong geometry
```

### 3. Pre-Cache Models

```bash
python scripts/download_depth_models.py \
    --models depth-anything/Depth-Anything-V2-Large-hf \
             depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
```

### 4. Monitor VRAM

```python
import torch
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### 5. Version Control Outputs

All outputs are timestamped and include git SHA:
```
outputs/model_comparison/run_20251218_143022_a7b3c4d/
```

## Troubleshooting

### Error: "Insufficient paired data"

**Cause**: Not enough images completed validation in both baseline and treatment.

**Fix**: Check that both models ran on the same input set. Review individual model logs.

### Error: "VRAM insufficient"

**Cause**: Model requires more VRAM than available.

**Fix**: 
- Use smaller model (Large instead of Giant)
- Reduce batch size in validator
- Use `--skip-vram-check` (CPU fallback, slow)

### Error: "validation_report.json not found"

**Cause**: Underlying validation script failed.

**Fix**: Check `production_depth_validation_fixed.py` logs in model output directory.

## Advanced Usage

### Custom Model Registry

Edit `run_multi_model_comparison.py` to add models:

```python
MODEL_REGISTRY = {
    "My_Custom_Model": {
        "hf_id": "myorg/my-depth-model",
        "type": "relative",
        "description": "Custom fine-tuned model",
        "vram_gb": 16,
        "default_input_size": 768,
    },
}
```

### Ensemble Depth Estimation

After comparison, combine models:

```python
# Average relative and metric depth (after alignment)
depth_ensemble = 0.5 * depth_relative + 0.5 * depth_metric_aligned
```

## References

- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)
- [HuggingFace Depth Estimation](https://huggingface.co/tasks/depth-estimation)
- [Statistical Testing Guide](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Cohen's d Effect Size](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)

## Support

For issues or questions:
1. Check `outputs/model_comparison/run_*/model_comparison_summary.json`
2. Review individual model logs in `model_*/input_*/`
3. Consult `docs/guides/` for validation framework details

---

**Last Updated**: 2025-12-18  
**Maintained By**: Transformation Portal Team
