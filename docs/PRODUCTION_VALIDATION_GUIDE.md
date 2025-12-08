# Production Validation Guide for Lux Depth V2

## Overview

The Lux Depth V2 validation framework provides production-grade quality assessment capabilities for verifying image enhancement quality, comparing against baselines (Topaz, Adobe, etc.), and ensuring consistent output quality across production runs.

## Key Features

### 1. Dual Validation Modes

#### Synthetic Reference Mode
- Create controlled test pairs by degrading high-quality originals
- Degraded image becomes the test input
- Original becomes ground truth for comparison
- Enables precise quantitative measurement of enhancement quality

#### Real-World Mode
- Validate production images without reference
- Uses no-reference metrics (NIMA aesthetic scoring)
- Combined with perceptual quality heuristics
- Suitable for actual client deliverables

### 2. Multi-Category Metrics

#### Fidelity Metrics (Reference-Based)
- **SSIM** (Structural Similarity): Measures structural preservation (0-1, higher is better)
- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy (20-50 dB, higher is better)

#### Perceptual Metrics (Reference-Based)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Deep learning-based perceptual similarity (0-1, lower is better)
- Uses pre-trained AlexNet or VGG features

#### Aesthetic Metrics (No-Reference)
- **NIMA** (Neural Image Assessment): Aesthetic quality scoring (1-10, higher is better)
- Fallback to heuristic-based scoring if NIMA model unavailable

### 3. Baseline Comparison
- Compare against industry tools (Topaz, Adobe, etc.)
- Side-by-side metric comparison
- Win/loss/tie statistics
- Per-image and aggregate comparisons

### 4. Reproducibility Stamping
All validation runs include:
- Git commit hash
- Configuration hash (deterministic)
- Device information (GPU model, driver version)
- Model versions (depth model, upscaler backend)
- Tiling settings used
- Per-stage timing information
- Timestamp (UTC)

## Usage Examples

### Basic Validation

```python
from lux_depth_v2.validation import QualityValidator
from pathlib import Path

# Initialize validator
validator = QualityValidator(device="cuda")

# Validate batch of processed images
test_images = list(Path("output/").glob("*_upscaled16.tif"))
report = validator.validate_batch(
    test_images=test_images,
    output_dir=Path("validation_results/"),
    mode="real",  # or "synthetic" if references available
    metrics_list=["ssim", "psnr", "lpips", "nima"],
)

# Save report
report.save(Path("validation_results/report.json"))

print(f"Composite Quality Score: {report.composite_score:.3f}")
```

### Synthetic Reference Creation

```python
from lux_depth_v2.validation import QualityValidator

validator = QualityValidator()

# Create synthetic test pair from high-quality original
degraded_path, reference_path = validator.create_synthetic_reference(
    original=Path("originals/mansion_8k.tif"),
    output_dir=Path("test_pairs/"),
    degradations=["downsample", "blur", "noise", "compress"]
)

print(f"Degraded test input: {degraded_path}")
print(f"Ground truth reference: {reference_path}")

# Now process degraded_path with your pipeline
# Then validate against reference_path
```

### Baseline Comparison

```python
from lux_depth_v2.validation import QualityValidator

validator = QualityValidator()

# Compare our output vs Topaz
comparison = validator.compare_baselines(
    ours=Path("output/mansion_upscaled16.tif"),
    baseline=Path("topaz_output/mansion_upscaled.tif"),
    reference=Path("test_pairs/mansion_reference.tif"),  # Optional
    metrics_list=["ssim", "psnr", "lpips", "nima"]
)

comparison.save(Path("comparisons/vs_topaz.json"))

print(f"Our scores: {comparison.our_scores}")
print(f"Baseline scores: {comparison.baseline_scores}")
print(f"Winner: {'Ours' if comparison.our_wins > 0 else 'Baseline' if comparison.baseline_wins > 0 else 'Tie'}")
```

### Batch Baseline Comparison

```python
from lux_depth_v2.validation import QualityValidator

validator = QualityValidator()

# Compare entire batch against baseline directory
test_images = list(Path("output/").glob("*_upscaled16.tif"))
report = validator.validate_batch(
    test_images=test_images,
    output_dir=Path("validation_results/"),
    baseline_dir=Path("topaz_output/"),
    mode="synthetic",
    metrics_list=["ssim", "psnr", "lpips", "nima"]
)

baseline_comp = report.baseline_comparison
print(f"Our wins: {baseline_comp['our_wins']}")
print(f"Baseline wins: {baseline_comp['baseline_wins']}")
print(f"Win rate: {baseline_comp['win_rate']:.1%}")
```

## Metric Interpretation

### SSIM (Structural Similarity)
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 0.95-1.0: Excellent structural preservation
  - 0.90-0.95: Good structural preservation
  - 0.85-0.90: Acceptable
  - < 0.85: Poor structural fidelity

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: Typically 20-50 dB
- **Interpretation**:
  - > 40 dB: Excellent pixel accuracy
  - 35-40 dB: Good accuracy
  - 30-35 dB: Acceptable
  - < 30 dB: Poor accuracy

### LPIPS (Learned Perceptual Similarity)
- **Range**: 0.0 to 1.0 (lower is better)
- **Interpretation**:
  - 0.0-0.1: Perceptually very similar
  - 0.1-0.2: Perceptually similar
  - 0.2-0.4: Noticeable differences
  - > 0.4: Significant perceptual differences

### NIMA (Aesthetic Quality)
- **Range**: 1.0 to 10.0
- **Interpretation**:
  - 7.0-10.0: High aesthetic quality
  - 5.0-7.0: Moderate aesthetic quality
  - 3.0-5.0: Low aesthetic quality
  - < 3.0: Poor aesthetic quality

### Composite Score
- **Range**: 0.0 to 1.0
- Weighted combination of all metrics (normalized)
- **Default weights**:
  - SSIM: 25%
  - PSNR: 15%
  - LPIPS: 35% (most perceptually relevant)
  - NIMA: 25%

## Reproducibility Stamping

Every validation run and processing report includes comprehensive reproducibility metadata:

```json
{
  "reproducibility": {
    "git_commit": "b35b72433132a678c52a21a033573a36917f0192",
    "config_hash": "a3f5c8e12d9b4a6c",
    "device": "cuda:0",
    "gpu_name": "NVIDIA RTX 4090",
    "cuda_version": "12.1",
    "python_version": "3.11.5",
    "torch_version": "2.1.0",
    "upscaler_backend": "torch",
    "model_path": "weights/RealESRGAN_x4plus.pth",
    "model_sha256": "abc123...",
    "post_tile": 2048,
    "post_overlap": 64,
    "upscale_tile": 512,
    "upscale_tile_pad": 16,
    "timestamp": "2025-12-08 00:45:00 UTC",
    "preset": "interior_luxury"
  }
}
```

## Production Presets

All production presets now enforce safety defaults:

```python
# Automatic in production presets:
validate_ai = True  # AI safety checks enabled
post_tile = 2048    # UHR tiling enabled (324MP+ support)
post_overlap = 64   # Increased overlap for quality
```

### Production Preset Characteristics

#### `interior_luxury`
- Material Response: 0.90 (strong)
- Post-tiling: Enabled (2048px)
- AI Validation: Mandatory
- Best for: High-end residential interiors

#### `exterior_showcase`
- Material Response: 0.80 (moderate-strong)
- Post-tiling: Enabled (2048px)
- AI Validation: Mandatory
- Best for: Architectural exteriors, landscapes

#### `archival_quality`
- Material Response: 0.60 (conservative)
- Post-tiling: Enabled (2048px)
- AI Validation: Mandatory
- Best for: Museum-grade, archival work

## Regression Testing

Use validation framework for regression testing:

```python
# Golden image workflow
validator = QualityValidator()

# 1. Establish baseline (one-time)
golden_images = list(Path("golden_set/").glob("*.tif"))
baseline_report = validator.validate_batch(
    test_images=golden_images,
    output_dir=Path("golden_baseline/"),
    mode="real"
)
baseline_report.save(Path("golden_baseline/baseline_report.json"))

# 2. Regression test after changes
current_report = validator.validate_batch(
    test_images=golden_images,
    output_dir=Path("current_test/"),
    mode="real"
)

# 3. Compare scores
print(f"Baseline composite: {baseline_report.composite_score:.3f}")
print(f"Current composite: {current_report.composite_score:.3f}")

# Assert no quality regression
assert current_report.composite_score >= baseline_report.composite_score - 0.05, \
    "Quality regression detected!"
```

## Dependency Safety

The validation framework respects security constraints:

### ✅ Safe Dependencies (Used)
- `opencv-python` - Image I/O
- `numpy` - Numerical operations
- `scipy` - Filtering (optional)
- `scikit-image` - SSIM computation (optional)
- `lpips` - Perceptual similarity (optional)
- `torch` - Deep learning backend

### ❌ Vulnerable Packages (Avoided)
- `basicsr` - CVE-2024-27763 (command injection)
- `realesrgan` - Depends on vulnerable basicsr
- `gfpgan` - Depends on vulnerable basicsr

**Note**: Always use `requirements-repo.txt` for safe dependencies.

## CLI Integration (Future)

Planned CLI integration:

```bash
# Validate batch
lux-depth-v2-validate --input output/ --mode real --output validation_results/

# Compare vs baseline
lux-depth-v2-validate --input output/ --baseline topaz/ --output comparison/

# Create synthetic pairs
lux-depth-v2-validate --create-pairs originals/ --output test_pairs/
```

## Service Endpoint Integration (Future)

Planned FastAPI service endpoint:

```bash
POST /validate
{
  "test_images": ["output/img1.tif", "output/img2.tif"],
  "mode": "real",
  "metrics": ["ssim", "psnr", "lpips", "nima"]
}

Response:
{
  "composite_score": 0.87,
  "metrics_scores": {
    "ssim": 0.95,
    "psnr": 38.2,
    "lpips": 0.12,
    "nima": 7.8
  },
  ...
}
```

## Best Practices

### 1. Validation Strategy
- Use **synthetic mode** for controlled testing during development
- Use **real mode** for production validation of client deliverables
- Maintain **golden image set** for regression testing

### 2. Metric Selection
- Include both **fidelity** (SSIM, PSNR) and **perceptual** (LPIPS, NIMA) metrics
- Weight LPIPS heavily (35%) as most correlated with human perception
- Use NIMA for aesthetic quality assessment

### 3. Baseline Comparisons
- Collect baseline outputs from industry tools (Topaz, Adobe)
- Run side-by-side comparisons on representative test set
- Track win rates over time as pipeline improves

### 4. Reproducibility
- Always include git commit hash in reports
- Use deterministic config hashing
- Track model versions and checksums
- Document hardware configuration

### 5. Continuous Validation
- Integrate validation into CI/CD pipeline
- Set quality thresholds for automated pass/fail
- Generate validation reports for each production run

## Troubleshooting

### LPIPS Import Error
```python
# Install lpips package
pip install lpips
```

### NIMA Model Unavailable
- Fallback to heuristic aesthetic scoring automatically
- Consider training/downloading NIMA model for production use

### Memory Issues with Large Batches
```python
# Process in smaller batches
for batch in batched(test_images, batch_size=10):
    report = validator.validate_batch(batch, ...)
```

### GPU Out of Memory
```python
# Use CPU for metric computation
validator = QualityValidator(device="cpu")
```

## See Also

- `lux_depth_v2/validation/quality_validator.py` - Main validator implementation
- `lux_depth_v2/validation/metrics.py` - Metric implementations
- `lux_depth_v2/validation/degradation.py` - Synthetic degradation pipeline
- `docs/QUALITY_BREAKTHROUGH_CLAIMS.md` - Template for defensible quality claims
