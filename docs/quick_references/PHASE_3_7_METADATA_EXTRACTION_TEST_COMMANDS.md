# Phase 3.7 Metadata Extraction Test Commands

Quick reference for testing Phase 3.7 metadata extraction capabilities on images.

## Prerequisites

Ensure the following dependencies are installed:

```bash
# Required: exiftool
brew install exiftool        # macOS
apt-get install libimage-exiftool-perl  # Linux

# Check system readiness
python scripts/test_metadata_extraction.py check-system
```

## Test Commands for `/Users/rc/Projects/Transformation_Portal/input_images`

### 1. Check System Readiness

```bash
python scripts/test_metadata_extraction.py check-system
```

Expected output includes:
- ✅ exiftool found (with version)
- ✅ pydantic found
- ✅ ingest module available

### 2. Single Image Extraction

Extract metadata from a single image:

```bash
# Basic extraction (output to <image>_provenance.json)
python scripts/test_metadata_extraction.py extract /Users/rc/Projects/Transformation_Portal/input_images/your_image.tif

# Custom output path
python scripts/test_metadata_extraction.py extract /Users/rc/Projects/Transformation_Portal/input_images/your_image.tif -o /tmp/test_provenance.json

# With durable write (fsync)
python scripts/test_metadata_extraction.py extract /Users/rc/Projects/Transformation_Portal/input_images/your_image.tif --fsync
```

### 3. Batch Extraction (Entire Directory)

Extract metadata from all images in the input_images directory:

```bash
# Default: sidecars saved to input_images/provenance_sidecars/
python scripts/test_metadata_extraction.py extract-batch /Users/rc/Projects/Transformation_Portal/input_images/

# Custom output directory
python scripts/test_metadata_extraction.py extract-batch /Users/rc/Projects/Transformation_Portal/input_images/ -o /tmp/metadata_sidecars/

# Non-recursive (only top-level images)
python scripts/test_metadata_extraction.py extract-batch /Users/rc/Projects/Transformation_Portal/input_images/ --no-recursive

# Stop on first error
python scripts/test_metadata_extraction.py extract-batch /Users/rc/Projects/Transformation_Portal/input_images/ --fail-fast
```

### 4. Validate Sidecar Files

Validate provenance sidecars for schema compliance:

```bash
# Basic validation
python scripts/test_metadata_extraction.py validate /Users/rc/Projects/Transformation_Portal/input_images/provenance_sidecars/your_image_provenance.json

# Verbose output (show sidecar summary)
python scripts/test_metadata_extraction.py validate /Users/rc/Projects/Transformation_Portal/input_images/provenance_sidecars/your_image_provenance.json -v

# Non-strict mode (legacy compatibility)
python scripts/test_metadata_extraction.py validate /Users/rc/Projects/Transformation_Portal/input_images/provenance_sidecars/your_image_provenance.json --no-strict
```

### 5. Summarize Extraction Results

Get aggregate statistics from extracted sidecars:

```bash
python scripts/test_metadata_extraction.py summarize /Users/rc/Projects/Transformation_Portal/input_images/provenance_sidecars/
```

Output includes:
- Total file sizes
- EXIF tag counts
- Camera distribution
- Image dimensions
- GPS data coverage

## Direct Python API Usage

For programmatic access to metadata extraction:

```python
from pathlib import Path
from transformation_portal.ingest import capture_provenance, write_sidecar, load_sidecar

# Extract metadata from single image
image_path = Path("/Users/rc/Projects/Transformation_Portal/input_images/your_image.tif")

sidecar = capture_provenance(
    input_path=image_path,
    cli_args=["--preset", "luxury"],
    config_dict={"model": "da3", "device": "mps"},
)

# Write sidecar JSON
output_path = Path("/tmp/your_image_provenance.json")
write_sidecar(sidecar, output_path, fsync=True)

# Access metadata
print(f"Camera: {sidecar.exif.camera_make} {sidecar.exif.camera_model}")
print(f"File SHA256: {sidecar.file_integrity.sha256[:16]}...")
print(f"Total EXIF tags: {len(sidecar.exif.all_tags)}")

# Load existing sidecar
loaded = load_sidecar(output_path, schema_type="provenance")
print(f"Schema version: {loaded.schema_version}")
```

## Using Existing CI Validation Script

Run schema validation on existing provenance sidecars:

```bash
# Validate test fixtures
python scripts/validate_ingest_contract.py --test-dir /Users/rc/Projects/Transformation_Portal/input_images/provenance_sidecars/ --strict
```

## Exit Codes

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | All validations passed |
| 1 | Schema validation failed | Missing required field |
| 2 | 8-bit conversion detected | uint8 instead of uint16 |
| 3 | Gamma correction detected | Non-linear histogram |
| 4 | Schema drift detected | Unknown field added |
| 5 | Other failure | exiftool not found |

## Supported Image Formats

### RAW Formats
- Canon: `.cr2`, `.cr3`
- Nikon: `.nef`, `.nrw`
- Sony: `.arw`, `.srf`
- Adobe: `.dng`
- Fujifilm: `.raf`
- Olympus: `.orf`
- Panasonic: `.rw2`
- Pentax: `.pef`
- Samsung: `.srw`

### Standard Formats
- TIFF: `.tif`, `.tiff`
- JPEG: `.jpg`, `.jpeg`
- PNG: `.png`
- HEIC: `.heic`, `.heif`

## Provenance Sidecar Schema (v1.0.0)

The provenance sidecar captures:

| Section | Contents | Deterministic |
|---------|----------|---------------|
| `file_integrity` | SHA256, size, path, MIME type | ✅ Yes |
| `exif` | Complete EXIF via exiftool | ✅ Yes |
| `toolchain` | exiftool, rawpy, Python versions | ❌ No (env-dependent) |
| `host` | Hostname, OS, architecture | ❌ No (env-dependent) |
| `timestamps` | Ingest start/end times | ❌ No (time-dependent) |
| `pipeline_config` | CLI args, preset, config SHA256 | ✅ Yes (if config same) |
| `git_commit` | Git SHA at ingest time | ❌ No (repo-dependent) |
| `run_id` | UUID v4 | ❌ No (unique per run) |

## Phase 3.7 Governance Verification

For governance export verification (Phase 3.7):

```bash
# Generate governance export
python tools/regulatory_export.py \
    --manifest-dir /path/to/manifests \
    --out-json /tmp/regulatory_export.json \
    --governance-export /tmp/governance_export.json

# Verify governance export integrity
python tools/regulatory_export.py \
    --verify-governance-export /tmp/governance_export.json
```

## See Also

- [Ingest Contract v1.0.0](../docs/apex/ingest_contract.md) - Full contract documentation
- [Schema Version Policy](../docs/compliance/SCHEMA_VERSION_POLICY.md) - Versioning rules
- [ADR-036](../docs/architecture/ADR-036-accountability-governance-invariants.md) - Governance invariants
