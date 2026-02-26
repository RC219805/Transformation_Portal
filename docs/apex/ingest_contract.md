# Ingest Contract Documentation (v1.0.1)

**Status:** Official Contract
**Effective Date:** 2026-02-10
**Schema Version:** 1.0.1

---

## Purpose

This document defines the **formal contract** for the Phase I linear ingest pipeline provenance and metadata capture system.

The ingest contract ensures:
- **Audit-grade traceability**: Every RAW/TIFF file ingested has a complete, immutable provenance record
- **Deterministic file-derived fields**: File hash, size, EXIF metadata are stable across runs
- **Schema enforcement**: Hard-fail on missing required fields, type mismatches, or schema drift
- **Quality firewall**: Block 8-bit conversions, gamma corrections, and dtype/range violations

---

## Contract Boundary

### Inputs (What Ingest Consumes)

| Input | Format | Source | Notes |
|-------|--------|--------|-------|
| **RAW Files** | CR2, NEF, ARW, DNG, etc. | Camera captures | Via rawpy (LibRaw) |
| **TIFF Files** | .tif, .tiff | Rendered outputs | Via PIL/tifffile |
| **Pipeline Config** | Python dict | CLI or orchestrator | Fingerprinted via SHA256 |
| **CLI Arguments** | List[str] | Command line | Captured for reproducibility |

### Outputs (What Ingest Produces)

| Output | Format | Consumer | Binding Contract |
|--------|--------|----------|------------------|
| **ProvenanceSidecar** | JSON (v1.0.1) | Audit trail | Complete EXIF + toolchain + env metadata |
| **IngestManifest** | JSON (v1.0.1) | Pipeline orchestrator | Summary status + sidecar pointer |
| **Validation Report** | Exit code | CI/CD | 0=pass, 1-5=specific failure modes |

---

## Schema Definitions

### ProvenanceSidecar (v1.0.1)

Complete, lossless provenance record for every ingested file.

**Required Fields:**

```json
{
  "schema_version": "1.0.1",  // Literal type (only "1.0.1" accepted)
  "file_integrity": {
    "sha256": "...",            // 64 hex chars (lowercase)
    "size_bytes": 1024000,      // Integer
    "path": "/input/IMG_1234.CR2",
    "mime_type": "image/x-canon-cr2"  // Optional
  },
  "exif": {
    "all_tags": { /* complete exiftool JSON */ },
    "camera_make": "Canon",     // Optional convenience fields
    "camera_model": "EOS 5D Mark IV",
    "iso": 400,
    "aperture": 2.8,
    "focal_length": 50.0,
    "width": 6720,
    "height": 4480,
    "bit_depth": 14,
    // ... many more optional fields
  },
  "toolchain": [
    {
      "name": "exiftool",
      "version": "12.50",
      "path": null  // Not captured for security
    },
    {
      "name": "rawpy",
      "version": "0.18.1"
    },
    {
      "name": "python",
      "version": "3.11.7",
      "path": "/usr/bin/python"
    }
  ],
  "host": {
    "hostname": "render-node-01",
    "os": "Linux",
    "os_version": "5.10.0-21-amd64",
    "python_version": "3.11.7",
    "arch": "x86_64"
  },
  "timestamps": {
    "ingest_start": "2026-02-10T12:00:00+00:00",  // ISO 8601 with TZ
    "ingest_end": "2026-02-10T12:05:32+00:00",
    "exiftool_extract_duration_sec": 2.3
  },
  "pipeline_config": {
    "config_sha256": "...",     // 64 hex chars
    "cli_args": ["--preset", "luxury"],
    "preset": "luxury",
    "custom_params": { /* any overrides */ }
  },
  "git_commit": "1a2b3c4d...",  // 40 hex chars (optional)
  "run_id": "550e8400-e29b-41d4-a716-446655440000"  // UUID v4
}
```

**Determinism Guarantee:**

**File-derived fields** are deterministic (stable across runs):
- `file_integrity`: SHA256, size, path (content-addressed)
- `exif`: Complete EXIF metadata via exiftool (file-intrinsic)
- `pipeline_config.config_sha256`: Config fingerprint (input-derived)

**Run metadata fields** vary across runs (non-deterministic by design):
- `run_id`: UUID v4 per ingest operation
- `timestamps`: Ingest start/end times
- `host`: Hostname, OS version, Python version
- `toolchain`: Versions of exiftool, ImageMagick, etc. (environment-dependent)
- `git_commit`: Git SHA at ingest time (repo state)

Determinism assertions in CI and contract validation are evaluated as **post-normalization** comparisons using the governed normalization profile `ingest_v1`. This profile removes run-metadata volatility while preserving file-derived contract fields.

This split enables:
- **Content verification**: Use file_integrity SHA256 to validate input
- **Provenance audit**: Full run context captured for compliance
- **Diff-friendly**: File-derived fields stable; run metadata expected to vary

### Machine Contract vs Evidence Artifact

`tp.meta.machine.v1` remains the automation wire contract for routing and orchestration:
- route by `exit_code` and typed `error` payloads
- validate shape via JSON Schema
- do not depend on message text

Machine-mode contract validation does **not** guarantee byte-identical serialization across all runtimes.
For cryptographic attestations, use the separate evidence artifact flow:
- projection profile: `tp.projection.machine_to_evidence.v1`
- canonicalization profile: `tp.canonical.json.v1`
- evidence schema: `tp.meta.evidence.v1`

The evidence flow hashes a projected envelope that removes volatile telemetry fields (for example `elapsed_seconds` and tool version strings), enabling reproducible third-party verification without changing the machine-mode contract.
Serialization intentionally differs by layer: machine wire output uses `ensure_ascii=True` for transport-facing payloads, while evidence canonicalization uses `ensure_ascii=False` under `tp.canonical.json.v1`.

### Phase 3.4 Detached Attestation Boundary

Phase 3.4 introduces detached attestations under schema `tp.attestation.detached.v1` at:
- `docs/schemas/attestation/tp.attestation.detached.v1/attestation.schema.json`

Governance invariants for this boundary:
- Attestation subject binds to `subject.evidence_sha256` from `tp.meta.evidence.v1`, not raw evidence JSON bytes.
- `subject.file_sha256` and `subject.bundle_root_sha256` are optional secondary anchors when present.
- Builder and CLI flows keep evidence immutable and detached (no mutation of `tp.meta.evidence.v1` payloads).
- Evidence recompute checks are on by default: `sha256(canonicalize_json(projected_envelope))` must match stored `evidence_sha256`.

Operational model:
- Signing can run offline with custody-held keys; detached attestation payloads are later distributed with evidence artifacts.
- Verifiers first validate attestation schema surface and evidence hash binding, then perform optional signature backend checks.

**Immutability:**

All Pydantic models are frozen (`frozen=True`). Once created, sidecar objects cannot be modified.

---

### IngestManifest (v1.0.1)

Lighter-weight summary for pipeline orchestration.

**Required Fields:**

```json
{
  "schema_version": "1.0.1",
  "input_file": {
    "sha256": "...",
    "size_bytes": 1024000,
    "path": "/input/IMG_1234.CR2"
  },
  "output_file": null,  // Optional (if ingest modifies file)
  "status": "success",  // "success", "error", or "skipped"
  "error_message": null,  // Set if status == "error"
  "provenance_sidecar_path": "/output/IMG_1234_provenance.json",
  "ingest_duration_sec": 5.5
}
```

**Status Values:**

- `"success"`: Ingest completed without errors
- `"error"`: Ingest failed (see `error_message`)
- `"skipped"`: File skipped (e.g., already processed)

---

## Validation Rules

### Schema Version Enforcement

**Contract:** Schema version must be exactly `"1.0.1"`.

**Enforcement:**
- Literal type in Pydantic model (`Literal["1.0.1"]`)
- CI validation script checks all sidecar files
- Hard-fail on version mismatch

**Breaking change policy:**
- Schema v2.0.0+ requires ADR approval
- Migration guide required in `docs/`
- Deprecated schemas supported for ≥ 1 release cycle

---

### Required Fields

**Contract:** All non-Optional fields must be present and non-null.

**Enforcement:**
- Pydantic validation at construction time
- CI validation script on all test artifacts
- Exit code 1 on missing required field

**Example error:**
```
Missing required field: exif
```

---

### Schema Drift Detection

**Contract:** Unknown fields trigger hard-fail in strict mode.

**Enforcement:**
- `validate_schema(data, strict_mode=True)` checks for unknown fields
- CI runs in strict mode by default
- Exit code 4 on schema drift

**Example error:**
```
Unknown fields detected (schema drift): unknown_field
```

---

### Type Validation

**Contract:** Field types must match schema exactly.

**Common type constraints:**
- `sha256`: 64 lowercase hex characters
- `git_commit`: 40 lowercase hex characters
- `timestamps`: ISO 8601 with timezone
- `size_bytes`: Positive integer
- `status`: One of {"success", "error", "skipped"}

**EXIF normalization semantics (v1.0.1):**
- `exif.focal_length` accepts canonical numeric input and EXIF-style strings such as `"4.5 mm"` (normalized to `4.5`).
- `exif.bit_depth` accepts canonical integer input and EXIF-style triplet strings such as `"8 8 8"` (normalized to `8`).
- Normalization happens before strict typing in the schema model; malformed values are still rejected.

**Example error:**
```
Type mismatch at file_integrity.size_bytes: value is not a valid integer
```

---

## Quality Firewall Rules

### 8-Bit Conversion Detection

**Contract:** 16-bit ingests must not be downsampled to 8-bit.

**Enforcement:**
```python
validate_no_8bit_conversion(image_data, expected_dtype="uint16")
```

**Checks:**
1. dtype is uint16 (not uint8)
2. Max pixel value > 255 (not 8-bit range)

**Exit code:** 2 on 8-bit conversion violation

---

### Gamma Correction Detection

**Contract:** Linear ingest must not apply gamma correction.

**Enforcement:**
```python
validate_linear_gamma(image_data, tolerance=0.05)
```

**Heuristic:** Linear images have ≥20% of pixels in shadow range (0.0-0.3). Gamma-corrected images skew toward midtones.

**Exit code:** 3 on gamma violation

**Note:** This is a heuristic check. Some linear images may fail if they are high-key (bright scenes). Use with caution.

---

## CI Integration

### Workflow: `ingest_contract_validation.yml`

**Triggers:**
- Pull request to main/develop
- Push to main/develop
- Manual dispatch

**Jobs:**

1. **validate-ingest-contract**
   - Scans `tests/fixtures/ingest` for sidecar files
   - Validates schema compliance (strict mode)
   - Runs pytest suite `tests/ingest/`
   - Hard-fail on any violations

2. **schema-drift-detection** (PR only)
   - Detects changes to `src/transformation_portal/ingest/schemas.py`
   - Warns if schema modified without version bump
   - Verifies schema version consistency across models

---

### Validation Script: `scripts/validate_ingest_contract.py`

**Usage:**
```bash
python scripts/validate_ingest_contract.py \
  --test-dir tests/fixtures/ingest \
  --strict
```

**Exit Codes:**

| Code | Meaning | Example |
|------|---------|---------|
| 0 | All validations passed | ✅ |
| 1 | Schema validation failed | Missing required field |
| 2 | 8-bit conversion detected | uint8 instead of uint16 |
| 3 | Gamma correction detected | Non-linear histogram |
| 4 | Schema drift detected | Unknown field added |
| 5 | Other validation failure | Unexpected error |

---

## Usage Examples

### Capture Provenance

```python
from pathlib import Path
from transformation_portal.ingest import capture_provenance, write_sidecar

# Capture provenance from RAW file
sidecar = capture_provenance(
    input_path=Path("IMG_1234.CR2"),
    cli_args=["--preset", "luxury"],
    config_dict={"model": "da3", "device": "mps"},
)

# Write deterministic sidecar
write_sidecar(
    sidecar=sidecar,
    output_path=Path("IMG_1234_provenance.json"),
    fsync=True,  # Durable write
)
```

### Validate Sidecar

```python
from pathlib import Path
from transformation_portal.ingest import validate_ingest_contract

# Validate existing sidecar
validate_ingest_contract(
    sidecar_path=Path("IMG_1234_provenance.json"),
    strict_mode=True,  # Fail on unknown fields
)
# Raises SchemaValidationError if invalid
```

### Load and Inspect Sidecar

```python
from pathlib import Path
from transformation_portal.ingest import load_sidecar

# Load sidecar
sidecar = load_sidecar(
    sidecar_path=Path("IMG_1234_provenance.json"),
    schema_type="provenance",
)

# Inspect metadata
print(f"Camera: {sidecar.exif.camera_make} {sidecar.exif.camera_model}")
print(f"ISO: {sidecar.exif.iso}")
print(f"Git commit: {sidecar.git_commit[:8]}")
print(f"File SHA256: {sidecar.file_integrity.sha256[:8]}...")
```

---

## Determinism Guarantees

The provenance sidecar is split semantically into **file-derived** and **run metadata** sections.

### File-Derived Fields (Deterministic)

These fields are **stable across runs** for the same input file:

- `file_integrity.sha256`: Content-addressed hash (bitwise deterministic)
- `file_integrity.size_bytes`: File size
- `file_integrity.path`: Input path (relative or absolute)
- `exif.all_tags`: Complete EXIF metadata from exiftool (file-intrinsic)
- `pipeline_config.config_sha256`: Config fingerprint (if config unchanged)

**Use case:** Content verification, duplicate detection, cache invalidation.

### Run Metadata Fields (Non-Deterministic)

These fields **vary across runs** (environment and time-dependent):

- `run_id`: UUID v4 per ingest run (distinguishes operations)
- `timestamps.ingest_start`, `timestamps.ingest_end`: Wall clock times
- `host.hostname`, `host.os_version`: Execution environment
- `host.python_version`, `host.arch`: Runtime versions
- `toolchain[].version`: Installed tool versions (e.g., exiftool 12.50 vs 12.76)
- `git_commit`: Git SHA at ingest time (repo state)

**Use case:** Audit trail, reproducibility analysis, environment debugging.

### JSON Serialization

`to_json_deterministic()` method guarantees:
- Sorted keys (alphabetical)
- Stable 2-space indentation
- No trailing whitespace
- UTF-8 encoding (no ASCII escapes)

**Example:**

```python
sidecar1 = capture_provenance(input_file)
sidecar2 = capture_provenance(input_file)  # Different run

# File-derived fields are identical
assert sidecar1.file_integrity.sha256 == sidecar2.file_integrity.sha256
assert sidecar1.exif.all_tags == sidecar2.exif.all_tags

# Run metadata differs
assert sidecar1.run_id != sidecar2.run_id
assert sidecar1.timestamps.ingest_start != sidecar2.timestamps.ingest_start
```

### Future: Determinism Mode

For testing and reproducibility, a future enhancement could add:
- `DETERMINISM_MODE=true`: Inject fixed timestamps, hostname, versions
- Enables byte-identical outputs for contract tests
- Not currently implemented (P3 roadmap item)

---

---

## Emergency Override

**Contract:** Ingest validation can be overridden in emergencies.

**Procedure:**
1. Add `INGEST-OVERRIDE: <reason>` to PR description
2. Requires maintainer approval
3. Logged in audit trail

**Requirements:**
- Must include justification
- Must document impact in PR
- Must file follow-up issue to fix

---

## Prohibited Behaviors

Ingest contract **must never**:

1. ❌ Silently ignore validation errors (log + fail instead)
2. ❌ Modify schema without version bump
3. ❌ Skip required fields
4. ❌ Accept unknown fields in strict mode
5. ❌ Convert 16-bit to 8-bit without explicit flag

---

## Dependencies

### Required

- **exiftool** (libimage-exiftool-perl on Ubuntu)
  - Used for complete metadata extraction
  - Must be in PATH
  - Version ≥ 12.0 recommended

- **pydantic** ≥ 2.0
  - Schema validation and serialization
  - Type enforcement
  - Immutable models

### Optional

- **rawpy** (for RAW file support)
  - Install with: `pip install rawpy`
  - Or: `pip install -e ".[raw]"`

- **numpy** (for 8-bit/gamma detection)
  - Already in core dependencies

---

## Migration Guide

This is the initial release (v1.0.1). No migration required.

**Future schema changes:**

When updating schema to v2.0.0:
1. Create new models with Literal["2.0.0"]
2. Keep v1.0.1 models for backward compat
3. Add migration function: `migrate_v1_to_v2()`
4. Update validation to support both versions
5. Document breaking changes in CHANGELOG

---

## Related Governance Documents

- **[APEX Performance Contract](APEX_CONTRACT.md)** - Performance observability contract
- **[Phase I Architecture](phase1/APEX_ARCHITECTURE_NOTES.md)** - Linear ingest design
- **[Schema Module](../../src/transformation_portal/ingest/schemas.py)** - Implementation
- **[CI Workflow](../../.github/workflows/ingest_contract_validation.yml)** - Enforcement

---

## Contract Authority

This contract is **binding** for all ingest operations.

Changes to this contract require:
- ADR approval (if breaking changes)
- Consensus from 2+ maintainers
- Version bump
- Migration guide

**Contract Authority:** Transformation Portal Governance
**Effective Date:** 2026-02-10
**Next Review:** 2026-05-10 (quarterly)
