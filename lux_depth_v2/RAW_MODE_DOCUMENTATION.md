# RAW Mode Documentation

## Overview

**RAW mode** (`--mode raw`) provides a **true pixel-perfect baseline** for forensic analysis and validation. Unlike `baseline` mode which applies minimal processing, RAW mode bypasses ALL processing stages and provides a pure decode→write path.

## Purpose

RAW mode is designed to:

1. **Eliminate contamination**: Ensure zero processing artifacts for Phase B comparisons
2. **Guarantee determinism**: CPU-only execution with pixel-perfect reproducibility
3. **Provide verifiable baseline**: SHA256 hash verification confirms exact pixel preservation
4. **Enable forensic analysis**: Clean reference for detecting processing-induced changes

## Technical Implementation

### Key Features

- ✅ **Zero processing**: Direct numpy passthrough, no torch conversion
- ✅ **CPU-only execution**: Forces `device="cpu"` for deterministic behavior
- ✅ **No model loading**: Skips segmentation, materials, upscaling
- ✅ **No preset contamination**: `_preset_applied=False`, `_config_locked=False`
- ✅ **Pixel hash verification**: Automatic SHA256 verification of input→output
- ✅ **Minimal stages**: Only `io/read_input`, `io/read_depth`, `export_master`, `verify_raw`

### Processing Bypasses

RAW mode disables:

- ✗ Torch grading pipeline (sat/exp/con/temp)
- ✗ Material segmentation and response
- ✗ Upscaling (forced to 1x)
- ✗ Clarity and sharpening
- ✗ Detail transfer
- ✗ Post-processing tiling
- ✗ AI validation
- ✗ All exports except master16

## Usage

### Basic Command

```bash
lux-depth-v2 --mode raw \
  --input input.tiff \
  --output-dir output/
```

### Batch Processing

```bash
lux-depth-v2 --mode raw \
  --input-dir renders/ \
  --output-dir baseline_raw/
```

### Verification

The execution report includes pixel verification:

```json
{
  "pixel_verification": {
    "input_hash": "e67529bd94b2bb1197dba1e0b69397e8e685868b0ba23934db421bf2f64e385c",
    "output_hash": "e67529bd94b2bb1197dba1e0b69397e8e685868b0ba23934db421bf2f64e385c",
    "deterministic": true
  }
}
```

### Determinism Test

Verify reproducibility:

```bash
# Run 1
lux-depth-v2 --mode raw --input image.tiff --output-dir run1/

# Run 2
lux-depth-v2 --mode raw --input image.tiff --output-dir run2/

# Compare
diff run1/image_master16.tif run2/image_master16.tif
# Should produce NO output (identical files)
```

## Configuration

### Force Overrides (Automatic)

When `--mode raw` is specified, the following overrides are applied:

```python
device = "cpu"           # Force CPU
precision = "fp32"       # No half precision
upscale = 1              # No upscaling
upscaler_backend = "none"
enable_material = False
segmentation.backend = "none"
save_upscaled = False
save_marketing_png = False
save_preview_jpg = False
```

### What Gets Preserved

- ✅ RGB channels (alpha channel dropped if present)
- ✅ 16-bit precision (uint16)
- ✅ Image dimensions
- ✅ Pixel values (exact)

## Execution Report

### Minimal Stages

```json
{
  "mode": "raw",
  "stages_executed": [
    "io/read_input",
    "io/read_depth",
    "export_master",
    "verify_raw"
  ],
  "config": {
    "preset": "raw",
    "_preset_applied": false,
    "_config_locked": false,
    "device": "cpu",
    "precision": "fp32",
    "enable_material": false,
    "upscale": 1
  }
}
```

### Timing

RAW mode is fast due to minimal processing:

- Input: 4320×7680 (33MP)
- Timing: ~1.0-1.5 seconds
- Stages:
  - `io/read_input`: ~0.18s
  - `io/read_depth`: ~0.05s
  - `export_master`: ~0.27s
  - `verify_raw`: ~0.55s

## Comparison: RAW vs BASELINE vs Normal

| Feature | RAW Mode | Baseline Mode | Normal Mode |
|---------|----------|---------------|-------------|
| Grading | ✗ No | ✓ Minimal (identity) | ✓ Full |
| Material | ✗ No | ✗ No | ✓ Yes |
| Upscaling | ✗ No (1x) | ✗ No (1x) | ✓ Yes (4x) |
| Segmentation | ✗ No | ✗ No | ✓ Yes |
| Device | CPU only | Auto (GPU/CPU) | Auto (GPU/CPU) |
| Preset applied | ✗ No | ✓ Yes (locked) | ✓ Yes |
| Hash verification | ✓ Yes | ✗ No | ✗ No |
| Processing | None | Minimal | Full |

## Use Cases

### 1. Phase B Baseline (Primary)

Generate clean baseline for contamination-free comparisons:

```bash
lux-depth-v2 --mode raw \
  --input-dir phase_b_inputs/ \
  --output-dir phase_b_baseline_raw/
```

### 2. Forensic Analysis

Detect processing-induced changes:

```bash
# Generate raw baseline
lux-depth-v2 --mode raw --input original.tiff --output-dir raw/

# Generate processed version
lux-depth-v2 --preset interior_luxury --input original.tiff --output-dir processed/

# Compare
compare raw/original_master16.tif processed/original_master16.tif diff.png
```

### 3. Determinism Validation

Verify reproducibility across runs:

```bash
for i in {1..5}; do
  lux-depth-v2 --mode raw --input test.tiff --output-dir run_$i/
done

# All outputs should be identical
sha256sum run_*/test_master16.tif
```

### 4. Pipeline Testing

Establish ground truth for regression testing:

```bash
# Generate reference
lux-depth-v2 --mode raw --input test_suite/*.tiff --output-dir reference/

# After code changes, verify
lux-depth-v2 --mode raw --input test_suite/*.tiff --output-dir verification/

# Should be identical
diff -r reference/ verification/
```

## Limitations

### Known Differences

1. **Alpha channel**: Dropped if present (RGBA → RGB)
2. **Colorspace**: No colorspace transforms applied
3. **Metadata**: EXIF/IPTC may be lost (use tifffile)

### What RAW Mode Is NOT

- ✗ **Not a file copy**: Still decodes and re-encodes TIFF
- ✗ **Not format conversion**: Only TIFF master16 output
- ✗ **Not a backup**: Use `cp` for file-level copies

## Troubleshooting

### Hash Mismatch

If `deterministic: false`, check:

1. **Input format**: TIFF with 3-4 channels supported
2. **File corruption**: Verify input integrity
3. **Compression**: Different compression doesn't affect pixels
4. **Precision**: Both input/output are uint16

### Slow Performance

RAW mode should be fast (~1-2s for 33MP). If slow:

1. Check disk I/O (use SSD)
2. Verify CPU not throttled
3. Check tifffile version

### Verification Failed

If `pixel_verification` shows error:

1. Ensure tifffile installed: `pip install tifffile`
2. Check file permissions
3. Verify output exists

## Advanced: Hash Computation

The pixel hash is computed as:

```python
import hashlib
import tifffile

# Read image (uint16 RGB)
img_array = tifffile.imread("image.tiff")

# Extract RGB channels only (drop alpha if present)
rgb_array = img_array[..., :3]

# Compute SHA256 of raw bytes
pixel_hash = hashlib.sha256(rgb_array.tobytes()).hexdigest()
```

## Validation Criteria

A successful RAW mode execution must show:

```json
{
  "mode": "raw",
  "preset": "raw",
  "_preset_applied": false,
  "_config_locked": false,
  "device": "cpu",
  "enable_material": false,
  "segmentation": {"backend": "none"},
  "upscale": 1,
  "timing_s": "<2.0",
  "stages_executed": ["io/read_input", "io/read_depth", "export_master", "verify_raw"],
  "pixel_verification": {
    "deterministic": true
  }
}
```

## References

- **Implementation**: `lux_depth_v2/config.py` (Preset.RAW)
- **CLI**: `lux_depth_v2/cli.py` (`--mode raw`)
- **Pipeline**: `lux_depth_v2/pipeline.py` (raw bypass logic)
- **Hash verification**: `lux_depth_v2/io_utils.py` (`compute_pixel_hash`)

---

**Last Updated**: 2025-12-22  
**Status**: Production-ready  
**Security**: No model loading, CPU-only, safe
