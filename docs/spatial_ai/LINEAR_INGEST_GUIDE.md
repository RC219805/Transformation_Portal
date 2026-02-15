# Linear Ingest User Guide

**Quick Start Guide for Spatial AI Training Data Preparation**

---

## Overview

This guide shows you how to convert RAW camera files and high-bit-depth images into validated, linear-light tensors ready for spatial AI training.

**What you'll learn:**
1. Installing dependencies
2. Processing single images
3. Batch processing datasets
4. Creating dataset manifests
5. Validating data quality
6. Troubleshooting common errors

---

## Installation

### Base Installation

```bash
# Install transformation-portal with RAW support
pip install -e ".[raw]"

# Or install core + RAW separately
pip install transformation-portal
pip install rawpy
```

### Optional: OpenEXR Support (HDR Output)

```bash
# macOS (Homebrew)
brew install openexr
pip install OpenEXR Imath

# Ubuntu/Debian
sudo apt-get install libopenexr-dev
pip install OpenEXR Imath
```

### Verify Installation

```python
from transformation_portal.spatial_ai.ingest import LinearDecoder

print("✅ Linear ingest ready")
```

---

## Quick Start: Single Image

### Example 1: Convert RAW to Linear Tensor

```python
from transformation_portal.spatial_ai.ingest import decode

# Simplest usage
result = decode("IMG_1234.CR2", gamma=1.0)

# Access linear RGB tensor (numpy array)
tensor = result.linear_rgb
print(f"Shape: {tensor.shape}")        # e.g., (5464, 8192, 3)
print(f"Dtype: {tensor.dtype}")        # float32
print(f"Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
```

### Example 2: With Provenance and EXR Output

```python
from pathlib import Path
from transformation_portal.spatial_ai.ingest import LinearDecoder

# Create decoder with strict mode
decoder = LinearDecoder(
    gamma=1.0,
    bit_depth=32,
    strict_ingest=True,  # Reject 8-bit inputs
)

# Decode with artifacts
result = decoder.decode(
    input_path="IMG_1234.CR2",
    output_dir=Path("./processed"),
    emit_exr=True,           # Save linear EXR file
    emit_provenance=True,    # Save provenance JSON sidecar
)

# Check outputs
print(f"Linear EXR: {result.output_exr_path}")
print(f"Provenance: {result.provenance_path}")
print(f"Content hash: {result.content_hash[:16]}...")
```

### Example 3: Process TIFF/PNG

```python
# Works identically for TIFF/PNG
result = decode(
    "architectural_render_16bit.tiff",
    gamma=1.0,
    strict_ingest=True,
    emit_provenance=True,
)

print(f"Input format: {result.input_format}")  # TIFF
print(f"Gamma: {result.gamma}")                # 1.0
```

---

## Batch Processing

### Example 4: Process All RAW Files in Directory

```python
from pathlib import Path
from transformation_portal.spatial_ai.ingest import LinearDecoder
from tqdm import tqdm  # Progress bar

# Setup
input_dir = Path("./raw_images")
output_dir = Path("./processed")
output_dir.mkdir(exist_ok=True)

decoder = LinearDecoder(gamma=1.0, strict_ingest=True)

# Find all RAW files
raw_files = list(input_dir.glob("*.CR2")) + \
            list(input_dir.glob("*.NEF")) + \
            list(input_dir.glob("*.ARW"))

print(f"Found {len(raw_files)} RAW files")

# Process batch
results = []
for raw_path in tqdm(raw_files, desc="Decoding"):
    try:
        result = decoder.decode(
            input_path=raw_path,
            output_dir=output_dir,
            emit_exr=True,
            emit_provenance=True,
        )
        results.append(result)
    except Exception as e:
        print(f"⚠️  Failed: {raw_path.name}: {e}")

print(f"✅ Successfully processed {len(results)} images")
```

### Example 5: Filter by Camera Model

```python
from transformation_portal.spatial_ai.ingest import ProvenanceCapture

# Process and filter
canon_results = []

for raw_path in raw_files:
    result = decoder.decode(raw_path, emit_provenance=True)

    # Load provenance
    prov = ProvenanceCapture().load_sidecar(result.provenance_path)

    # Filter by camera
    if prov["camera"].get("make") == "Canon":
        canon_results.append(result)

print(f"Found {len(canon_results)} Canon images")
```

---

## Dataset Manifests

### Example 6: Create Training Dataset Manifest

```python
from pathlib import Path
from transformation_portal.spatial_ai.ingest import (
    DatasetManifestBuilder,
    ImageManifestEntry,
    LinearDecoder,
)

# Setup
dataset_root = Path("./luxury_estate_dataset")
images_dir = dataset_root / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Create manifest builder
builder = DatasetManifestBuilder(
    name="luxury_estate_training_v1",
    description="High-fidelity architectural renders with linear light",
    version="1.0.0",
    tags=["linear_sRGB", "luxury", "architectural", "training"],
)

# Process images and add to manifest
decoder = LinearDecoder(gamma=1.0, strict_ingest=True)

for raw_path in dataset_raw_files:
    # Decode
    result = decoder.decode(
        input_path=raw_path,
        output_dir=images_dir,
        emit_exr=True,
        emit_provenance=True,
    )

    # Create manifest entry
    entry = ImageManifestEntry(
        file_path=str(result.output_exr_path.relative_to(dataset_root)),
        provenance_path=str(result.provenance_path.relative_to(dataset_root)),
        content_hash=result.content_hash,
        input_format=result.input_format,
        dimensions=result.linear_rgb.shape,
        value_range=(
            float(result.linear_rgb.min()),
            float(result.linear_rgb.max()),
        ),
        has_hdr=result.linear_rgb.max() > 1.0,
        tags=["interior"],  # Add custom tags
    )

    builder.add_image(entry)

# Build and write manifest
manifest = builder.build()
manifest.write(dataset_root / "manifest.json")

print(f"✅ Dataset manifest created: {manifest.total_images} images")
```

### Example 7: Load and Query Manifest

```python
from transformation_portal.spatial_ai.ingest import ManifestSchema

# Load manifest
manifest = ManifestSchema.from_directory(dataset_root)

print(f"Dataset: {manifest.dataset_name}")
print(f"Total images: {manifest.total_images}")
print(f"Gamma: {manifest.gamma}")
print(f"Color space: {manifest.color_space}")

# Query images
interior_images = manifest.get_images(tag="interior")
print(f"Interior images: {len(interior_images)}")

# Get specific image
img = manifest.get_image_by_path("images/IMG_1234_linear.exr")
if img:
    print(f"Hash: {img.content_hash}")
    print(f"Dimensions: {img.dimensions}")
    print(f"HDR: {img.has_hdr}")
```

---

## Validation and Quality Control

### Example 8: Validate Linear Output

```python
from transformation_portal.spatial_ai.ingest import (
    LinearDecoder,
    validate_linear_output,
)

# Decode
decoder = LinearDecoder(gamma=1.0, strict_ingest=True)
result = decoder.decode("test.tiff")

# Explicit validation (already done internally)
validate_linear_output(
    result.linear_rgb,
    gamma=result.gamma,
    input_path=result.input_path,
)

print("✅ All validation checks passed")
```

### Example 9: Check for HDR Clipping

```python
import numpy as np

def check_hdr_clipping(tensor, threshold=0.95):
    """Check if HDR values are being clipped."""
    max_value = np.max(tensor)

    if max_value <= 1.0:
        print("⚠️  No HDR values detected (max ≤ 1.0)")
        print("   This might indicate tone-mapping or clipping")
    elif max_value > threshold:
        print(f"✅ HDR values preserved (max = {max_value:.3f})")
    else:
        print(f"⚠️  Possible HDR clipping (max = {max_value:.3f})")

result = decode("bright_scene.CR2")
check_hdr_clipping(result.linear_rgb)
```

### Example 10: Verify Determinism

```python
# Decode same file twice
result1 = decode("test.CR2")
result2 = decode("test.CR2")

# Hashes should match
assert result1.content_hash == result2.content_hash
print("✅ Deterministic decode verified")
```

---

## Error Handling

### Example 11: Handle 8-bit Rejection

```python
from transformation_portal.spatial_ai.ingest import (
    BitDepthViolationError,
    decode,
)

try:
    # This will fail if test.jpg is 8-bit
    result = decode("test.jpg", gamma=1.0, strict_ingest=True)
except BitDepthViolationError as e:
    print(f"❌ Bit depth error: {e}")
    print("\nRemediation:")
    print("  1. Convert to 16-bit TIFF/PNG")
    print("  2. Use RAW files instead")
    print("  3. Set strict_ingest=False (not recommended)")
```

### Example 12: Graceful Degradation

```python
from transformation_portal.spatial_ai.ingest import LinearDecoder

def safe_decode(path, strict=True, fallback=False):
    """Decode with optional fallback to non-strict mode."""
    decoder = LinearDecoder(gamma=1.0, strict_ingest=strict)

    try:
        return decoder.decode(path)
    except BitDepthViolationError:
        if fallback:
            print(f"⚠️  Falling back to non-strict mode for {path.name}")
            decoder_relaxed = LinearDecoder(gamma=1.0, strict_ingest=False)
            return decoder_relaxed.decode(path)
        else:
            raise

# Process with fallback
result = safe_decode("uncertain_bitdepth.png", strict=True, fallback=True)
```

---

## Advanced Usage

### Example 13: Custom Provenance Notes

```python
from transformation_portal.spatial_ai.ingest import ProvenanceCapture

# Capture provenance with custom notes
capture = ProvenanceCapture()

prov = capture.capture(
    source_path=input_path,
    tensor=result.linear_rgb,
    gamma=1.0,
    bit_depth=32,
    notes="Captured during golden hour, exterior courtyard, ISO 100",
)

capture.write_sidecar(prov, output_path)
```

### Example 14: Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from transformation_portal.spatial_ai.ingest import LinearDecoder

def process_image(raw_path):
    """Worker function for parallel processing."""
    decoder = LinearDecoder(gamma=1.0, strict_ingest=True)
    try:
        result = decoder.decode(
            raw_path,
            output_dir=Path("./processed"),
            emit_exr=True,
            emit_provenance=True,
        )
        return (raw_path.name, "success", result.content_hash)
    except Exception as e:
        return (raw_path.name, "failed", str(e))

# Parallel batch processing
raw_files = list(Path("./raw").glob("*.CR2"))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, raw_files))

# Report
successes = [r for r in results if r[1] == "success"]
failures = [r for r in results if r[1] == "failed"]

print(f"✅ Processed: {len(successes)}")
print(f"❌ Failed: {len(failures)}")
```

---

## Troubleshooting

### Common Errors

#### 1. `BitDepthViolationError: 8-bit input rejected`

**Cause:** Input image is 8-bit, which loses 99.6% of color resolution.

**Solution:**
```python
# Option 1: Use 16-bit source
result = decode("high_quality.tiff", strict_ingest=True)

# Option 2: Disable strict mode (not recommended for training)
result = decode("low_quality.png", strict_ingest=False)

# Option 3: Convert 8-bit to 16-bit TIFF externally
# (but can't recover lost information)
```

#### 2. `UnsupportedFormatError: JPEG not supported`

**Cause:** JPEG is 8-bit lossy format with baked gamma.

**Solution:**
```bash
# Convert JPEG to 16-bit TIFF (external tool)
# Note: Can't recover lost information from JPEG
convert input.jpg -depth 16 output.tiff
```

#### 3. `LinearityViolationError: gamma must be 1.0`

**Cause:** Attempting to use non-linear gamma (e.g., 2.2).

**Solution:**
```python
# For training data, gamma MUST be 1.0 (linear light)
decoder = LinearDecoder(gamma=1.0)  # Non-negotiable

# For rendering, use lux_depth_v3.raw_loader instead
```

#### 4. `ImportError: No module named 'rawpy'`

**Cause:** RAW support not installed.

**Solution:**
```bash
pip install rawpy
# Or install with RAW extra:
pip install -e ".[raw]"
```

#### 5. `SchemaVersionError: unsupported version '2.0.0'`

**Cause:** Manifest created with newer schema version.

**Solution:**
```bash
# Option 1: Upgrade ingest pipeline
pip install --upgrade transformation-portal

# Option 2: Regenerate manifest with current version
python regenerate_manifest.py
```

---

## Performance Tips

### 1. Use RAW for Best Quality

```python
# ✅ BEST: RAW files preserve full sensor data
result = decode("IMG_1234.CR2")  # 12-14 bit → float32

# ⚠️  GOOD: 16-bit TIFF (if properly converted)
result = decode("render_16bit.tiff")

# ❌ AVOID: 8-bit PNG (loses 99.6% of color depth)
result = decode("export_8bit.png", strict_ingest=False)
```

### 2. Batch Processing with Progress

```python
from tqdm import tqdm

results = []
for path in tqdm(raw_files, desc="Processing"):
    result = decode(path)
    results.append(result)
```

### 3. Skip EXR Export for Speed

```python
# Fast: Skip EXR export if only need tensor
result = decode("test.CR2", emit_exr=False)

# Use tensor directly
train_batch = np.stack([r.linear_rgb for r in results])
```

### 4. Parallel Processing

See Example 14 above for parallel batch processing.

---

## Best Practices

### ✅ DO

1. **Use strict mode for training data:**
   ```python
   decoder = LinearDecoder(gamma=1.0, strict_ingest=True)
   ```

2. **Verify linear output:**
   ```python
   assert result.gamma == 1.0
   assert result.linear_rgb.dtype == np.float32
   ```

3. **Track provenance:**
   ```python
   result = decode(path, emit_provenance=True)
   ```

4. **Validate manifests on load:**
   ```python
   manifest = ManifestSchema.from_file(path)
   manifest.validate()
   ```

5. **Check for HDR preservation:**
   ```python
   assert result.linear_rgb.max() > 1.0  # HDR preserved
   ```

### ❌ DON'T

1. **Don't use 8-bit inputs for training:**
   ```python
   # BAD: 8-bit destroys detail
   decode("low_quality.jpg", strict_ingest=False)
   ```

2. **Don't mix rendering and training decoders:**
   ```python
   # BAD: Cross-contamination risk
   from lux_depth_v3.raw_loader import load_raw  # Wrong decoder!
   ```

3. **Don't skip validation:**
   ```python
   # BAD: Silent corruption risk
   tensor = np.array(Image.open("test.tiff"))  # No validation
   ```

4. **Don't ignore schema versions:**
   ```python
   # BAD: Version drift causes silent errors
   manifest["schema_version"] = "custom"  # Don't do this
   ```

---

## Next Steps

### For Training Workflows

1. ✅ Process dataset with linear ingest
2. ✅ Create manifest with provenance
3. ✅ Validate all images (gamma=1.0, float32, no clipping)
4. → **Load tensors in training loop** (see training guide)
5. → **Monitor for distribution shifts** (value range, HDR ratio)

### For Pipeline Integration

1. ✅ Install with RAW support: `pip install -e ".[raw]"`
2. ✅ Test on sample images
3. ✅ Run validation suite: `pytest tests/spatial_ai/ingest/`
4. → **Integrate into data pipeline** (see architecture doc)

### For Quality Assurance

1. ✅ Verify determinism (same input → same hash)
2. ✅ Check HDR preservation (max >1.0 for bright scenes)
3. ✅ Validate provenance completeness
4. → **Audit dataset statistics** (value distribution, bit depth)

---

## Additional Resources

- **Architecture Doc:** `docs/spatial_ai/LINEAR_INGEST_ARCHITECTURE.md`
- **API Reference:** Module docstrings in `src/transformation_portal/spatial_ai/ingest/`
- **ADR-023:** Spatial AI Ingest Isolation Boundary
- **Tests:** `tests/spatial_ai/ingest/` (73 examples)

---

## Support

**Questions or Issues?**

1. Check docstrings: `help(LinearDecoder)`
2. Review tests: `tests/spatial_ai/ingest/test_*.py`
3. See architecture doc for deep dive
4. Consult ADR-023 for design rationale

---

**Document Version:** 1.0.0
**Last Updated:** 2025-02-14
**Maintained By:** Transformation Portal Architect
