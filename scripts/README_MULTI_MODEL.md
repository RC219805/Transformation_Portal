# Multi-Model Depth Validation Scripts

Quick reference for the multi-model comparison framework.

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `preflight_model_comparison.py` | Environment validation | **Before** running validation |
| `run_model_comparison_suite.sh` | Complete workflow automation | Primary entry point |
| `run_multi_model_comparison.py` | Multi-model runner (advanced) | Custom model/size selection |
| `analyze_model_comparison.py` | Statistical analysis (advanced) | Re-analyze existing runs |

## Quick Start

### 1. Pre-Flight Check

```bash
python scripts/preflight_model_comparison.py --mode quick
```

Validates:
- Dataset availability
- Python dependencies
- GPU/VRAM
- HuggingFace models
- Disk space

### 2. Run Validation

**Quick Mode** (2 models, 2 sizes, ~15-30 min):
```bash
./scripts/run_model_comparison_suite.sh quick
```

**Full Mode** (3 models, 4 sizes, ~2-4 hours):
```bash
./scripts/run_model_comparison_suite.sh full
```

### 3. View Results

```bash
# Latest run directory
LATEST=$(ls -td outputs/model_comparison/run_* | head -1)

# Summary tables
cat $LATEST/best_per_model.csv | column -t -s,
cat $LATEST/analysis/statistical_summary.csv | column -t -s,

# Full JSON
cat $LATEST/model_comparison_summary.json | jq .
```

## Advanced Usage

### Custom Model Selection

```bash
python scripts/run_multi_model_comparison.py \
    --input-dir data/validation_full \
    --labels data/validation_full/labels.csv \
    --models DA2_Large DA2_Metric_Indoor \
    --sweep-sizes 518 768 896
```

### Re-Analyze Existing Run

```bash
python scripts/analyze_model_comparison.py \
    --comparison-dir outputs/model_comparison/run_20251218_* \
    --baseline-model DA2_Large \
    --confidence-level 0.95
```

## Model Keys

| Key | HF Model ID | Type | VRAM |
|-----|-------------|------|------|
| `DA2_Large` | `depth-anything/Depth-Anything-V2-Large-hf` | relative | 12GB |
| `DA2_Metric_Indoor` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | metric | 12GB |
| `DA2_Metric_Outdoor` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | metric | 12GB |
| `DA2_Giant` | `depth-anything/Depth-Anything-V2-Giant-hf` | relative | 24GB |

## Troubleshooting

### "No module named 'scipy'"

```bash
pip install scipy scikit-learn pandas
```

### "VRAM insufficient"

Use smaller model or CPU mode:
```bash
python scripts/run_multi_model_comparison.py \
    --models DA2_Large \
    --skip-vram-check
```

### "Dataset not found"

Ensure dataset exists:
```bash
ls data/validation_quick/
ls data/validation_full/
```

Create labels.csv if missing:
```csv
filename,scene_type
image1.jpg,texture_dominated
image2.jpg,structure_dominated
```

## Output Structure

```
outputs/model_comparison/run_TIMESTAMP_SHA/
├── model_DA2_Large/
│   ├── input_518/
│   │   ├── *_metrics.json
│   │   └── validation_report.json
│   └── input_768/
├── model_DA2_Metric_Indoor/
├── comparison_overall.csv       ← All results
├── best_per_model.csv           ← Optimal configs
├── model_comparison_summary.json ← Full data
└── analysis/
    ├── statistical_comparison.json
    └── statistical_summary.csv  ← Significance tests
```

## Documentation

- Full guide: `docs/guides/MULTI_MODEL_VALIDATION.md`
- Session summary: `docs/SESSION_COMPLETE_MULTI_MODEL_FRAMEWORK_20251218.md`

## Support

1. Check preflight: `python scripts/preflight_model_comparison.py`
2. Review logs in model output directories
3. Consult full documentation in `docs/guides/`
