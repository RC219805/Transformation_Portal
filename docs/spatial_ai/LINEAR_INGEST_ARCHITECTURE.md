# Linear Ingest Architecture

**Status:** Production Ready
**Version:** 1.0.0 (Issue #890 Phase I)
**Date:** 2025-02-14
**Owner:** Transformation Portal Architect

---

## Executive Summary

The Linear Ingest pipeline provides a **production-ready, validated, provenance-tracked** system for converting RAW camera files and high-bit-depth images into float32 linear light tensors suitable for spatial AI training.

**Key Properties:**
- ✅ **Deterministic:** Same input → same output
- ✅ **Validated:** Hard failure on constraint violations (no silent corruption)
- ✅ **Traceable:** Full provenance from source file to tensor
- ✅ **Versioned:** Schema evolution with compatibility checks
- ✅ **Isolated:** Complete separation from rendering pipelines (ADR-023)

**Success Metrics:**
- 73 passing tests, 74% test coverage
- Zero tolerance for 8-bit collapse (strict mode)
- 100% linear light enforcement (gamma=1.0, no exceptions)
- Full EXIF metadata extraction
- Versioned manifest schema (v1.0.0)

---

## Architecture Principles

### 1. **Fail-Fast Philosophy**

Silent corruption is **unacceptable** for training data. The pipeline enforces hard constraints:

```python
# ✅ CORRECT: Explicit validation with clear errors
validate_bit_depth(input_path, array, min_bits=16, strict=True)
# Raises: BitDepthViolationError with remediation guidance

# ❌ WRONG: Silent 8-bit conversion
img_array.astype(np.uint8)  # Destroys 99.6% of color depth
```

**Enforced Constraints:**
- Gamma must be 1.0 (linear light, not perceptual)
- Dtype must be float32 (not float16, not uint16)
- No NaN, no Inf, no negative values (invalid for light)
- HDR values >1.0 are REQUIRED (not clipped)

### 2. **Complete Isolation from Rendering**

Per ADR-023, rendering and training pipelines have **incompatible requirements**:

| Requirement         | Rendering (lux_depth_v3) | Training (spatial_ai) |
|---------------------|--------------------------|------------------------|
| Gamma               | 2.2 (sRGB perceptual)   | 1.0 (linear physics)  |
| Bit depth           | 8-bit (display-ready)   | 32-bit float (HDR)    |
| Color space         | sRGB gamut              | linear sRGB / ACEScg  |
| Dynamic range       | Clipped [0, 1]          | HDR [0, ∞)            |
| Tone curve          | Baked perceptual curve  | None (raw sensor data)|

**Enforcement:**
```python
# CI lint rule prevents cross-contamination
assert "lux_depth_v3.raw_loader" not in spatial_ai_imports
assert "spatial_ai.ingest" not in lux_depth_v3_imports
```

### 3. **Full Provenance Tracking**

Every tensor has a complete audit trail:

```json
{
  "camera": {
    "make": "Canon", "model": "EOS R5",
    "iso": 100, "aperture": 2.8, "shutter_speed": "1/250",
    "focal_length": 85.0, "lens_model": "RF 85mm F1.2L"
  },
  "ingest": {
    "timestamp": "2025-02-14T12:00:00Z",
    "source_file_hash_sha256": "abc123...",
    "loader_version": "1.0.0"
  },
  "transform": {
    "gamma": 1.0, "bit_depth": 32, "dtype": "float32",
    "color_space": "linear_sRGB",
    "demosaic_method": "AHD",  // For RAW files
    "white_balance_method": "camera"
  },
  "output": {
    "content_hash_sha256": "def456...",
    "value_range_min": 0.0, "value_range_max": 2.3,
    "has_hdr_values": true
  }
}
```

### 4. **Versioned Schema Evolution**

Manifests include schema version for forward/backward compatibility:

```json
{
  "schema_version": "1.0.0",
  "dataset": { ... },
  "images": [ ... ]
}
```

**Version Compatibility Matrix:**

| Schema Version | Status     | Notes                              |
|----------------|------------|------------------------------------|
| 1.0.0          | Current    | Issue #890 Phase I baseline        |
| 1.0.1          | Supported  | Minor clarifications (if needed)   |
| 2.0.0          | Future     | Phase II: ACEScg, multi-exposure   |

Unsupported versions **fail loudly** on load:
```python
SchemaVersionError: manifest.json has version '2.0.0',
but only ['1.0.0', '1.0.1'] are supported.
```

---

## Component Architecture

### Module Structure

```
src/transformation_portal/spatial_ai/ingest/
├── __init__.py                 # Public API exports
├── linear_decoder.py           # Core RAW/TIFF → linear float32
├── provenance.py               # EXIF + metadata capture
├── manifest_schema.py          # Versioned dataset manifests
├── validators.py               # Hard-constraint enforcement
└── exceptions.py               # Clear, actionable error messages

tests/spatial_ai/ingest/
├── test_linear_decoder.py      # LinearDecoder + integration
├── test_provenance.py          # Provenance capture
├── test_manifest_schema.py     # Manifest validation
└── test_validators.py          # Validator guardrails
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Linear Ingest Pipeline                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LinearDecoder (linear_decoder.py)                          │
│  • Format detection (TIFF, PNG, EXR, RAW)                   │
│  • Linear decode (gamma=1.0, no tone curve)                 │
│  • HDR preservation (values >1.0 allowed)                   │
│  • Integration with validators + provenance                 │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Validators      │  │  Provenance      │  │  Manifest        │
│  (validators.py) │  │  (provenance.py) │  │  (manifest_      │
│                  │  │                  │  │   schema.py)     │
│  • Bit depth     │  │  • EXIF extract  │  │  • Pydantic      │
│  • Dtype         │  │  • File hashing  │  │    validation    │
│  • Gamma         │  │  • Metadata       │  │  • Versioning    │
│  • Range (NaN)   │  │    assembly      │  │  • Inventory     │
│  • Schema ver    │  │  • Sidecar JSON  │  │    management    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Exceptions (exceptions.py)                                 │
│  • BitDepthViolationError                                   │
│  • LinearityViolationError                                  │
│  • RangeViolationError                                      │
│  • SchemaVersionError                                       │
│  • ProvenanceError, ManifestError, UnsupportedFormatError   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Ingest Pipeline Flow

```
┌─────────────┐
│  RAW File   │  (CR2, NEF, ARW, DNG)
│  or         │  OR
│  TIFF/PNG   │  (16-bit/32-bit)
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 1. Format Detection                     │
│    • Extension → format string          │
│    • Validation (unsupported → error)   │
└─────┬───────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 2. Decode to Linear RGB                 │
│    • RAW: rawpy + LibRaw                │
│      - Linear demosaic (AHD default;    │
│        configurable via demosaic param) │
│      - Camera white balance             │
│      - No tone curve, gamma=1.0         │
│    • TIFF/PNG: PIL/tifffile             │
│      - 16-bit → float32 / 65535.0       │
│    • EXR: OpenEXR (if available)        │
│      - Direct float32 load              │
└─────┬───────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 3. Validation                           │
│    • Bit depth check (strict mode)      │
│    • Dtype = float32                    │
│    • Gamma = 1.0                        │
│    • Range check (no NaN/Inf/neg)       │
│    • Shape = (H, W, 3)                  │
└─────┬───────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 4. Provenance Capture                   │
│    • EXIF extraction (camera metadata)  │
│    • File hash (SHA-256)                │
│    • Ingest metadata (timestamp, etc.)  │
│    • Transform record (gamma, method)   │
│    • Output hash + value range          │
└─────┬───────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 5. Artifact Emission                    │
│    • Linear RGB tensor (float32)        │
│    • Provenance JSON sidecar (optional) │
│    • Linear EXR file (optional)         │
└─────┬───────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ 6. Manifest Registration                │
│    • Add to dataset manifest            │
│    • Schema validation (v1.0.0)         │
│    • Image inventory + tags             │
└─────────────────────────────────────────┘
```

---

## Supported Formats

### RAW Formats (via rawpy/LibRaw)

| Format | Extension | Camera Brands              | Bit Depth     | Status  |
|--------|-----------|----------------------------|---------------|---------|
| CR2    | .cr2      | Canon                      | 12-14 bit     | ✅ Full  |
| NEF    | .nef      | Nikon                      | 12-14 bit     | ✅ Full  |
| ARW    | .arw      | Sony                       | 12-14 bit     | ✅ Full  |
| DNG    | .dng      | Adobe Universal, Leica, etc| 12-16 bit     | ✅ Full  |

**RAW Decode Settings (Linear Mode):**
```python
rawpy.postprocess(
    gamma=(1, 1),                # Linear light (no gamma correction)
    no_auto_bright=True,         # No auto exposure
    output_color=rawpy.ColorSpace.sRGB,  # Linear sRGB (Phase I)
    output_bps=16,               # Max precision before float32
    use_camera_wb=True,          # Camera white balance from EXIF
    demosaic_algorithm=resolve_demosaic_algorithm(self.demosaic),  # AHD default
)
```

The demosaic algorithm is configurable via the `demosaic` parameter on
`LinearDecoder` and the `decode_contract` `IngestOptions.demosaic` field.
Any name exposed by the installed `rawpy.DemosaicAlgorithm` enum is
accepted (e.g. `AHD`, `AMAZE`, `DCB`, `LMMSE`, `VNG`, `PPG`); unknown
names fail closed with `ValueError`. The `legacy_linear_srgb` contract no
longer hard-restricts to `AHD`.

### Processed Formats

| Format | Extension   | Bit Depth      | Color Space      | Status      |
|--------|-------------|----------------|------------------|-------------|
| TIFF   | .tif, .tiff | 16-bit, 32-bit | Assumed linear   | ✅ Full      |
| PNG    | .png        | 16-bit         | Assumed linear   | ✅ Full      |
| EXR    | .exr        | 32-bit float   | Linear (native)  | ✅ Full*     |

\* EXR requires OpenEXR package for full support. Pillow fallback available but slower.

### Rejected Formats

| Format | Extension   | Reason                                    |
|--------|-------------|-------------------------------------------|
| JPEG   | .jpg, .jpeg | 8-bit lossy, baked gamma, chroma subsampling |
| WebP   | .webp       | Lossy compression, perceptual encoding    |
| HEIC   | .heic       | Lossy, proprietary, complex color handling|

**8-bit JPEG rejection is INTENTIONAL:**
```python
# 8-bit quantization loses 99.6% of color resolution:
# 16-bit: 65,536 levels per channel
# 8-bit:  256 levels per channel
# Loss: (65536 - 256) / 65536 = 99.6%
```

---

## Validation Guarantees

### Contract: Linear Light

All output tensors satisfy the **SpatialCaptureV1** contract:

```python
# Post-conditions (enforced by validators):
assert tensor.dtype == np.float32            # Precision
assert gamma == 1.0                          # Linear light
assert np.all(tensor >= 0.0)                 # Non-negative (physical light)
assert not np.any(np.isnan(tensor))          # No NaN
assert not np.any(np.isinf(tensor))          # No Inf
assert tensor.shape[2] == 3                  # RGB channels
# HDR values >1.0 are ALLOWED (not clipped)
```

### Failure Modes

| Violation              | Exception                   | Remediation                          |
|------------------------|-----------------------------|--------------------------------------|
| 8-bit input (strict)   | `BitDepthViolationError`    | Use 16-bit TIFF/PNG or RAW          |
| Non-linear gamma       | `LinearityViolationError`   | Gamma must be 1.0 (no exceptions)   |
| Wrong dtype            | `LinearityViolationError`   | Output must be float32              |
| NaN/Inf values         | `RangeViolationError`       | Check input corruption              |
| Unsupported format     | `UnsupportedFormatError`    | Convert to TIFF/PNG/EXR/RAW         |
| Schema version mismatch| `SchemaVersionError`        | Regenerate manifest or upgrade pipeline |

**Error Message Example:**
```
BitDepthViolationError: test.png is uint8 (8-bit), but linear ingest requires ≥16-bit inputs.

Remediation:
  1. Use 16-bit TIFF/PNG or 32-bit EXR inputs for training data
  2. Convert RAW files to 16-bit TIFF with linear gamma
  3. Set strict_ingest=False to allow lossy 8-bit normalization (NOT recommended)

Context: 8-bit quantization destroys shadow/highlight detail needed for accurate training.
```

---

## Performance Characteristics

### Throughput Benchmarks

**Test System:** MacBook Pro M1 Max, 64GB RAM

| Format  | Resolution | Decode Time | Throughput    |
|---------|------------|-------------|---------------|
| CR2     | 8192×5464  | 1.2s        | ~37 MP/s      |
| NEF     | 6048×4024  | 0.9s        | ~27 MP/s      |
| TIFF 16 | 4096×3072  | 0.15s       | ~84 MP/s      |
| PNG 16  | 4096×3072  | 0.18s       | ~70 MP/s      |
| EXR 32  | 4096×3072  | 0.12s       | ~105 MP/s     |

**Notes:**
- RAW decode includes demosaic (compute-intensive)
- TIFF/PNG decode is I/O bound
- EXR decode via OpenEXR (optimized C++ library)

### Memory Footprint

| Stage                  | Memory per 24MP Image     |
|------------------------|---------------------------|
| RAW decode (uint16)    | ~144 MB (24M × 3 × 2)    |
| Linear float32 tensor  | ~288 MB (24M × 3 × 4)    |
| Provenance JSON        | ~5 KB                    |
| Peak (decode + tensor) | ~432 MB                  |

---

## Security Considerations

### Input Validation

**Threat Model:** Untrusted image files may contain:
- Malformed headers (buffer overflow attacks)
- Path traversal in EXIF (e.g., `../../../etc/passwd`)
- Billion laughs XML (ZIP bombs in metadata)

**Mitigations:**
1. **Sandboxed decoding:** PIL, rawpy, OpenEXR are battle-tested libraries
2. **Path sanitization:** All output paths validated before write
3. **EXIF sanitization:** Metadata extracted read-only (no execution)
4. **File size limits:** (Future) Reject files >500MB

### Hash Integrity

All files and tensors are SHA-256 hashed:
```python
source_file_hash_sha256: "abc123..."  # Input file integrity
content_hash_sha256: "def456..."      # Output tensor integrity
```

Use case: Detect corrupted transfers, verify reproducibility.

---

## Future Roadmap

### Phase II (Planned)

- [ ] **ACEScg color space:** Wider gamut for luxury materials
- [ ] **Multi-exposure HDR merge:** Bracket merge with alignment
- [ ] **Lens correction:** Distortion/vignette/CA correction
- [ ] **Metadata augmentation:** Add manual tags, scene annotations
- [ ] **Distributed ingest:** Multi-GPU/multi-node processing

### Phase III (Exploration)

- [ ] **ML-based demosaic:** Neural demosaic for quality improvement
- [ ] **RAW denoising:** Preserve detail while reducing sensor noise
- [ ] **Scene classification:** Auto-tag by scene type (interior/exterior/aerial)

---

## References

- **ADR-023:** Spatial AI Ingest Isolation Boundary
- **ADR-026:** APEX Research Ultra (Linear Light Contract)
- **Issue #890:** Spatial AI Foundation — Phase I: High-Fidelity Data + Linear Ingest
- **rawpy docs:** https://letmaik.github.io/rawpy/
- **LibRaw:** https://www.libraw.org/
- **EXR spec:** https://www.openexr.com/

---

## Appendix A: Example Code

### Basic Usage

```python
from transformation_portal.spatial_ai.ingest import LinearDecoder

# Initialize decoder with strict mode
decoder = LinearDecoder(
    gamma=1.0,
    bit_depth=32,
    strict_ingest=True,  # Reject 8-bit inputs
)

# Decode RAW to linear tensor
result = decoder.decode(
    input_path="IMG_1234.CR2",
    output_dir="./processed",
    emit_exr=True,           # Save linear EXR
    emit_provenance=True,    # Save provenance JSON
)

# Access results
print(f"Shape: {result.linear_rgb.shape}")
print(f"Range: [{result.linear_rgb.min():.3f}, {result.linear_rgb.max():.3f}]")
print(f"HDR: {result.linear_rgb.max() > 1.0}")
print(f"Provenance: {result.provenance_path}")
```

### Dataset Manifest

```python
from transformation_portal.spatial_ai.ingest import (
    DatasetManifestBuilder,
    ImageManifestEntry,
)

# Build manifest
builder = DatasetManifestBuilder(
    name="luxury_estate_training_v1",
    description="High-fidelity linear renders for spatial AI training",
    tags=["linear_sRGB", "luxury", "architectural"],
)

# Add images
for img_path in dataset_images:
    result = decoder.decode(img_path)

    builder.add_image(ImageManifestEntry(
        file_path=str(img_path.relative_to(dataset_root)),
        provenance_path=str(result.provenance_path.relative_to(dataset_root)),
        content_hash=result.content_hash,
        input_format=result.input_format,
        dimensions=result.linear_rgb.shape,
        value_range=(float(result.linear_rgb.min()), float(result.linear_rgb.max())),
        has_hdr=result.linear_rgb.max() > 1.0,
        camera_make=result.provenance_data.camera.make,
        camera_model=result.provenance_data.camera.model,
    ))

# Write manifest
manifest = builder.build()
manifest.write(dataset_root / "manifest.json")
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-02-14
**Maintained By:** Transformation Portal Architect
