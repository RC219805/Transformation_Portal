# Provenance & Metadata Capture

## Overview

The Transformation Portal implements **audit-grade provenance capture** for all RAW and TIFF inputs. This ensures complete traceability, reproducibility, and dataset governance for luxury real estate rendering workflows.

## Phase 4 Hardening Roadmap

For the **forward-looking**, spec-first deterministic rollout plan (schema governance, canonicalization, contract versioning, provenance binding, and CI determinism gates), see `docs/historical/architecture/PHASE4_CAPTURE_PROVENANCE_FLAWLESS_ROADMAP.md`. That roadmap describes planned behavior (including potential shifts to warnings-first defaults with `--strict` for hard failures) and may not match the current hard failure policy documented below.

## Scope

Provenance capture provides:

- **Complete metadata extraction**: All EXIF tags and groups via exiftool
- **Toolchain versioning**: Exact versions of all tools in the processing chain
- **Ingest context**: CLI args, git SHA, config fingerprint, timestamps, environment
- **Deterministic output**: Same input → same sidecar (except explicit nondeterministic fields)
- **Versioned schema**: Forward-compatible schema with validation at write time
- **Hard failure policy**: Missing/malformed metadata causes immediate error

## Sidecar Schema

### Version: 1.0.0

Provenance metadata is written to a colocated JSON sidecar file with the following structure:

```json
{
  "schema_version": "1.0.0",
  "input": {
    "file_path": "/path/to/input.tif",
    "file_sha256": "abc123...",
    "file_size_bytes": 12345678,
    "file_mtime_utc": "2026-02-10T11:37:29+00:00"
  },
  "exif": {
    "File:FileType": "TIFF",
    "File:FileSize": "12345678",
    "EXIF:Make": "Canon",
    "EXIF:Model": "EOS 5D Mark IV",
    "EXIF:DateTime": "2026:02:10 12:00:00",
    "EXIF:ISO": "200",
    "EXIF:FNumber": "8.0",
    "EXIF:ExposureTime": "1/125",
    "...": "..."
  },
  "toolchain": {
    "python_version": "3.11.8 (main, Feb 10 2026, 12:00:00)",
    "exiftool_version": "12.76",
    "rawpy_version": "0.18.1",
    "libraw_version": "0.21.2",
    "imagemagick_version": "ImageMagick 7.1.0-62"
  },
  "ingest_context": {
    "git_commit_sha": "abc123def456...",
    "config_fingerprint": "sha256:f1e2d3...",
    "ingest_timestamp_utc": "2026-02-10T11:37:29.141234+00:00",
    "host_os": "Linux-6.5.0-1015-azure",
    "host_machine": "x86_64",
    "cli_args": ["--preset", "max_quality", "--depth-device", "mps"],
    "working_directory": "/workspace/project"
  }
}
```

## Field Descriptions

### `schema_version` (required)

The provenance schema version. Currently `"1.0.0"`. Used for forward/backward compatibility.

### `input` (required)

Metadata about the input file itself:

- `file_path` (string, required): Absolute or relative path to the input file
- `file_sha256` (string, required): SHA256 hash of the input file (hex digest)
- `file_size_bytes` (integer, required): File size in bytes
- `file_mtime_utc` (string, required): File modification time in UTC (ISO 8601 format)

### `exif` (required)

Complete EXIF and file-level metadata extracted via exiftool. This is the raw JSON output from `exiftool -G -a -s -j`, capturing:

- All EXIF tags (camera make/model, exposure settings, GPS, etc.)
- File metadata (format, size, type)
- Maker notes (camera-specific metadata)
- IPTC/XMP tags (if present)

**Note**: EXIF may be an empty object `{}` for files without EXIF data, but the field must always be present.

### `toolchain` (required)

Versions of all tools in the processing toolchain:

- `python_version` (string, required): Python interpreter version
- `exiftool_version` (string, required): exiftool version
- `rawpy_version` (string or null): rawpy library version (if installed)
- `libraw_version` (string or null): LibRaw version (embedded in rawpy)
- `imagemagick_version` (string or null): ImageMagick version (if available)

### `ingest_context` (required)

Context about the ingestion process:

- `git_commit_sha` (string or null): Git commit SHA of the repository (if in a git repo)
- `config_fingerprint` (string, required): SHA256 hash of the pipeline configuration
- `ingest_timestamp_utc` (string, required): Timestamp when provenance was captured (ISO 8601 UTC)
- `host_os` (string, required): Operating system (via `platform.platform()`)
- `host_machine` (string, required): Machine architecture (e.g., "x86_64", "arm64")
- `cli_args` (array or null): Command-line arguments used (if available)
- `working_directory` (string or null): Working directory when processing started

## Determinism Guarantees

The provenance system is designed to be **deterministic**: processing the same input file with the same configuration should produce **identical provenance sidecars**, except for explicitly nondeterministic fields.

### Deterministic Fields

These fields will be identical across runs:

- `input.file_sha256` - Hash of the input file
- `input.file_size_bytes` - File size
- `input.file_path` - Path to the file (if absolute)
- `exif` - EXIF metadata (assuming file hasn't changed)
- `toolchain.*_version` - Tool versions (assuming same environment)
- `ingest_context.config_fingerprint` - Config hash (assuming same config)

### Nondeterministic Fields

These fields may differ across runs (but are separated for clarity):

- `ingest_context.ingest_timestamp_utc` - Time of capture
- `ingest_context.git_commit_sha` - May change if code is updated
- `input.file_mtime_utc` - May change if file is touched

### Implementation Details

Determinism is enforced through:

1. **Stable key ordering**: All JSON output uses `sort_keys=True`
2. **Normalized types**: NumPy types converted to Python natives
3. **Consistent formatting**: ISO 8601 for timestamps, hex for hashes
4. **Explicit nondeterministic sections**: Timestamps isolated in `ingest_context`

## Hard Failure Policy

The provenance system follows a **hard failure policy** to ensure audit integrity:

### Required Dependencies

- **exiftool must be installed**: Processing will fail with a clear error if exiftool is not available
  - Ubuntu/Debian: `apt-get install libimage-exiftool-perl`
  - macOS: `brew install exiftool`

### Required Fields

The following fields are **required** and will cause a hard failure if missing or malformed:

- `input.file_path` - Must be non-empty
- `input.file_sha256` - Must be non-empty, valid SHA256 hex digest
- `input.file_size_bytes` - Must be positive integer
- `input.file_mtime_utc` - Must be non-empty, valid ISO 8601 timestamp
- `exif` - Must be present (can be empty object)
- `toolchain.python_version` - Must be non-empty
- `toolchain.exiftool_version` - Must be non-empty (ensures exiftool available)
- `ingest_context.config_fingerprint` - Must be non-empty
- `ingest_context.ingest_timestamp_utc` - Must be non-empty
- `ingest_context.host_os` - Must be non-empty

### Error Handling

Provenance errors result in immediate failure with explicit error messages:

```python
# Example error messages
ExiftoolNotFoundError:
  "exiftool not found in PATH. Install with: apt-get install libimage-exiftool-perl"

MissingRequiredFieldError:
  "input.file_sha256 is required"
  "toolchain.exiftool_version is required (exiftool must be available)"

SchemaValidationError:
  "Schema version mismatch: expected 1.0.0, got 2.0.0"
  "Unsupported provenance schema version: 99.0.0"
```

## File Naming and Location

Provenance sidecars are colocated with the pipeline manifests:

```
output/
  manifests/
    image_name_combined.json          # Pipeline manifest
    image_name_provenance.json        # Provenance sidecar (NEW)
  depth/
    image_name_depth.png
    image_name_metadata.json
```

Naming convention:
- Input: `750Picacho_GreatRoom_UltraQuality.tif`
- Manifest: `750Picacho_GreatRoom_UltraQuality_combined.json`
- Provenance: `750Picacho_GreatRoom_UltraQuality_provenance.json`

## Usage

### Automatic Capture

Provenance is captured automatically during pipeline execution. No special configuration required.

```python
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.config import EnhanceConfig

config = EnhanceConfig.from_preset("max_quality")
orchestrator = EnhanceOrchestrator(config=config, output_dir="output")

# Process image - provenance sidecar written automatically
result = orchestrator.enhance_image(image_input)
# -> Creates: output/manifests/image_name_provenance.json
```

### Manual Capture

For custom workflows, you can capture provenance manually:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3.provenance import capture_provenance

# Capture provenance for a single file
provenance = capture_provenance(
    image_path=Path("input.tif"),
    config_fingerprint="sha256:abc123...",
    cli_args=["--preset", "max_quality"],
    repo_root=Path.cwd(),
)

# Write sidecar
provenance.write_sidecar(Path("output/input_provenance.json"))
```

### Reading Provenance

```python
from transformation_portal.lux_depth_v3.provenance import ProvenanceMetadata

# Load provenance sidecar
provenance = ProvenanceMetadata.load_sidecar(Path("output/input_provenance.json"))

# Access metadata
print(f"Input file: {provenance.input.file_path}")
print(f"SHA256: {provenance.input.file_sha256}")
print(f"EXIF Make: {provenance.exif.get('EXIF:Make')}")
print(f"Python: {provenance.toolchain['python_version']}")
print(f"Git SHA: {provenance.ingest_context.git_commit_sha}")
```

## Testing

Comprehensive tests validate provenance capture:

```bash
# Run all provenance tests (unit + integration)
pytest tests/test_provenance.py -v

# Run only unit tests (fast, no ML deps)
pytest tests/test_provenance.py -v -m "not integration"

# Run integration tests with real TIFF fixtures
pytest tests/test_provenance_integration.py -v
```

Test coverage includes:
- ✅ EXIF metadata extraction
- ✅ Toolchain version capture
- ✅ Schema validation (required fields enforced)
- ✅ JSON determinism (stable key ordering)
- ✅ Sidecar file operations (atomic writes)
- ✅ Hard failure on missing exiftool
- ✅ Integration with orchestrator
- ✅ Real TIFF fixture processing

## Contract & Governance

This implementation follows the **APEX Performance Contract** (docs/apex/APEX_CONTRACT.md) requirements:

- **Deterministic**: Same input → same sidecar (modulo explicit nondeterministic fields)
- **Versioned**: Schema v1.0.0 with forward compatibility
- **Validated**: Hard failure on schema drift or missing required fields
- **Complete**: No metadata is silently dropped or inferred
- **Audit-grade**: Suitable for dataset governance, replay, and compliance

## Migration and Compatibility

### Adding New Fields

To add new fields to the provenance schema:

1. Increment schema version (e.g., `1.0.0` → `1.1.0`)
2. Update `ProvenanceMetadata` dataclass
3. Update validation logic in `validate_required_fields()`
4. Update tests
5. Document changes in this file

### Backward Compatibility

The `from_dict()` method validates schema versions:

```python
schema_version = data.get("schema_version")
if schema_version != PROVENANCE_SCHEMA_VERSION:
    raise SchemaValidationError(
        f"Unsupported provenance schema version: {schema_version}"
    )
```

This ensures that old code cannot accidentally read new schemas without validation.

## Related Documentation

- [APEX Performance Contract](../apex/APEX_CONTRACT.md)
- [Manifest Management](./manifest_management.md)
- [Architecture Overview](./architecture.md)

## References

- Parent Issue: RC219805/Transformation_Portal#890
- Implementation PR: [Link to be added]
- exiftool: https://exiftool.org/
- LibRaw: https://www.libraw.org/
