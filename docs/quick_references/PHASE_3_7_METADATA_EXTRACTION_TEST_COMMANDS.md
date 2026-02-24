# Phase 3.7 Metadata Extraction Test Commands

Quick reference for testing Phase 3.7 metadata extraction capabilities on images.

## Prerequisites

Ensure the following dependencies are installed:

```bash
# Required: Install the package first
pip install -e .

# Required: exiftool
brew install exiftool        # macOS
apt-get install libimage-exiftool-perl  # Linux

# Check system readiness
python scripts/test_metadata_extraction.py check-system
```

## Test Commands for `<project_root>/input_images`

### Global machine-mode flags

Use these flags before the subcommand when you need deterministic machine output:

```bash
# Emit machine JSON to stdout
python scripts/test_metadata_extraction.py --json <command> ...

# Pretty-print machine JSON
python scripts/test_metadata_extraction.py --json --json-pretty <command> ...

# Write machine JSON to file (stdout remains clean)
python scripts/test_metadata_extraction.py --json --json-output /tmp/metadata_result.json <command> ...
```

### 1. Check System Readiness

```bash
python scripts/test_metadata_extraction.py check-system

# Machine JSON output
python scripts/test_metadata_extraction.py --json check-system
```

Expected output includes:
- ✅ exiftool found (with version)
- ✅ pydantic found
- ✅ ingest module available

### 2. Single Image Extraction

Extract metadata from a single image:

```bash
# Basic extraction (output to <image>.provenance.json)
python scripts/test_metadata_extraction.py extract /path/to/input_images/your_image.tif

# Custom output path
python scripts/test_metadata_extraction.py extract /path/to/input_images/your_image.tif -o /tmp/test_provenance.json

# With durable write (fsync)
python scripts/test_metadata_extraction.py extract /path/to/input_images/your_image.tif --fsync

# Debug mode (full tracebacks on errors)
python scripts/test_metadata_extraction.py --debug extract /path/to/input_images/your_image.tif

# Machine JSON output
python scripts/test_metadata_extraction.py --json extract /path/to/input_images/your_image.tif
```

### 3. Batch Extraction (Entire Directory)

Extract metadata from all images in the input_images directory:

```bash
# Default: sidecars saved to input_images/provenance_sidecars/
python scripts/test_metadata_extraction.py extract-batch /path/to/input_images/

# Custom output directory
python scripts/test_metadata_extraction.py extract-batch /path/to/input_images/ -o /tmp/metadata_sidecars/

# Non-recursive (only top-level images)
python scripts/test_metadata_extraction.py extract-batch /path/to/input_images/ --no-recursive

# Stop on first error
python scripts/test_metadata_extraction.py extract-batch /path/to/input_images/ --fail-fast

# Verbose output (show per-file status)
python scripts/test_metadata_extraction.py extract-batch /path/to/input_images/ -v

# Machine JSON output
python scripts/test_metadata_extraction.py --json extract-batch /path/to/input_images/
```

### 4. Validate Sidecar Files

Validate provenance sidecars for schema compliance:

```bash
# Basic validation
python scripts/test_metadata_extraction.py validate /path/to/input_images/provenance_sidecars/your_image_provenance.json

# Verbose output (show sidecar summary)
python scripts/test_metadata_extraction.py validate /path/to/input_images/provenance_sidecars/your_image_provenance.json -v

# Non-strict mode (legacy compatibility)
python scripts/test_metadata_extraction.py validate /path/to/input_images/provenance_sidecars/your_image_provenance.json --no-strict

# Machine JSON output
python scripts/test_metadata_extraction.py --json validate /path/to/input_images/provenance_sidecars/your_image_provenance.json
```

### 5. Summarize Extraction Results

Get aggregate statistics from extracted sidecars:

```bash
python scripts/test_metadata_extraction.py summarize /path/to/input_images/provenance_sidecars/

# Machine JSON output
python scripts/test_metadata_extraction.py --json summarize /path/to/input_images/provenance_sidecars/
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
from transformation_portal.ingest import (
    capture_provenance,
    write_sidecar,
    load_sidecar,
    validate_schema,
    # Exit codes (contract-aligned for CI compatibility)
    EXIT_SUCCESS,
    EXIT_SCHEMA_VALIDATION_FAILED,
    EXIT_8BIT_CONVERSION,
    EXIT_GAMMA_VIOLATION,
    EXIT_SCHEMA_DRIFT,
    EXIT_OTHER_FAILURE,
    # Exit code classification functions
    classify_validation_exit_code,
    classify_validation_errors,
)

# Extract metadata from single image
image_path = Path("/path/to/input_images/your_image.tif")

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

# Validate and classify errors
errors = validate_schema(output_path, schema_type="provenance")
if errors:
    exit_code = classify_validation_errors(errors)
    print(f"Validation failed with exit code: {exit_code}")
```

## Using Existing CI Validation Script

Run schema validation on existing provenance sidecars:

```bash
# Validate test fixtures
python scripts/validate_ingest_contract.py --test-dir /path/to/input_images/provenance_sidecars/ --strict
```

## Exit Codes

Exit codes are defined in `transformation_portal.ingest.validator` and exported from `transformation_portal.ingest` for programmatic use.

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

| Section | Contents | Determinism Basis |
|---------|----------|-------------------|
| `file_integrity` | SHA256, size, path, MIME type | ✅ Byte-stable for identical file input |
| `exif` | Complete EXIF via exiftool | ✅ Stable when source image metadata is unchanged |
| `toolchain` | exiftool, rawpy, Python versions | ❌ Environment-dependent |
| `host` | Hostname, OS, architecture | ❌ Environment-dependent |
| `timestamps` | Ingest start/end times | ❌ Time-dependent |
| `pipeline_config` | CLI args, preset, config SHA256 | ✅ Stable for identical args/config |
| `git_commit` | Git SHA at ingest time | ❌ Repository-state-dependent |
| `run_id` | UUID v4 | ❌ Unique per run |

Determinism in this table means output values remain stable when inputs and execution context are held constant.

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

- [Ingest Contract v1.0.0](../apex/ingest_contract.md) - Full contract documentation
- [Schema Version Policy](../compliance/SCHEMA_VERSION_POLICY.md) - Versioning rules
- [ADR-036](../architecture/ADR-036-accountability-governance-invariants.md) - Governance invariants
