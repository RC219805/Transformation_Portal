# Phase 4 Portal Integration Design

This document outlines the recommended approach for integrating Phase 4
chain verification into the Transformation Portal's archive gate system.

> **Note**: This integration requires Architect escalation per governance,
> as it modifies the public `ARCHIVE_GATE_ALLOWED_COMMANDS` contract.

## Overview

Phase 4 provides deterministic provenance capture and verification for
image archives. The integration would expose Phase 4F chain verification
through the existing archive-gate-a pipeline.

## Proposed Changes

### 1. Add `phase4-verify` Command to Archive Gate A

```python
# In app.py ARCHIVE_GATE_ALLOWED_COMMANDS
ARCHIVE_GATE_ALLOWED_COMMANDS = {
    "archive-gate-a": {
        "fixity-scan",
        "fixity-verify",
        "manifest-build",
        "rights-apply",
        "phase4-verify",  # NEW
    },
    # ... other gates unchanged
}
```

### 2. Add Phase 4 Schema Path Constants

```python
# New constants in app.py
PHASE4_SCHEMA_DIR = REPO_ROOT / "schemas" / "phase4"
PHASE4_METADATA_SCHEMA = PHASE4_SCHEMA_DIR / "metadata.schema.json"
PHASE4_METADATA_MANIFEST_SCHEMA = PHASE4_SCHEMA_DIR / "metadata_manifest.schema.json"
PHASE4_PROVENANCE_MANIFEST_SCHEMA = PHASE4_SCHEMA_DIR / "provenance_manifest.schema.json"
PHASE4_PROVENANCE_MERKLE_SCHEMA = PHASE4_SCHEMA_DIR / "provenance_merkle.schema.json"
PHASE4_VERIFICATION_REPORT_SCHEMA = PHASE4_SCHEMA_DIR / "verification_report.schema.json"

PHASE4_REQUIRED_SCHEMAS = [
    PHASE4_METADATA_SCHEMA,
    PHASE4_METADATA_MANIFEST_SCHEMA,
    PHASE4_PROVENANCE_MANIFEST_SCHEMA,
    PHASE4_PROVENANCE_MERKLE_SCHEMA,
    PHASE4_VERIFICATION_REPORT_SCHEMA,
]
```

### 3. Add Readiness Check for Phase 4

In `_archive_gate_readiness()`:

```python
if command == "phase4-verify":
    notes.append("Phase 4 chain verification requires all Phase 4C/4D/4E artifacts.")
    
    # Check schema availability
    for schema_path in PHASE4_REQUIRED_SCHEMAS:
        if not schema_path.is_file():
            issues.append(
                _readiness_issue(
                    "missing_phase4_schema",
                    severity="blocked",
                    message=f"Phase 4 schema missing: {schema_path.name}",
                )
            )
    
    if require_dispatch_inputs:
        # Validate required artifact paths
        for field, keys, desc in [
            ("capture_metadata", ("capture_metadata", "captureMetadata"), "Phase 4C artifact"),
            ("metadata_manifest", ("metadata_manifest", "metadataManifest"), "Phase 4D artifact"),
            ("provenance_manifest", ("provenance_manifest", "provenanceManifest"), "Phase 4E provenance artifact"),
            ("provenance_merkle", ("provenance_merkle", "provenanceMerkle"), "Phase 4E Merkle artifact"),
        ]:
            _, issue = _validate_existing_path(
                _pick(args or {}, *keys, default=""),
                field=field,
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason=f"{field}_required",
                missing_message=f"Provide an existing {desc} before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
```

### 4. Add Argv Builder for Phase 4 Verify

In `_archive_gate_argv()`:

```python
elif command == "phase4-verify":
    capture_metadata = _path_arg(
        args,
        "capture_metadata",
        "captureMetadata",
        default=str(Path(output_dir) / "capture_metadata.tp.meta.capture.v1.json"),
        allowed_roots=ALLOWED_OUTPUT_ROOTS,
    )
    metadata_manifest = _path_arg(
        args,
        "metadata_manifest",
        "metadataManifest",
        default=str(Path(output_dir) / "metadata_manifest.tp.meta.capture_manifest.v1.json"),
        allowed_roots=ALLOWED_OUTPUT_ROOTS,
    )
    provenance_manifest = _path_arg(
        args,
        "provenance_manifest",
        "provenanceManifest",
        default=str(Path(output_dir) / "provenance_manifest.tp.meta.provenance.v1.json"),
        allowed_roots=ALLOWED_OUTPUT_ROOTS,
    )
    provenance_merkle = _path_arg(
        args,
        "provenance_merkle",
        "provenanceMerkle",
        default=str(Path(output_dir) / "provenance_merkle.tp.meta.provenance_merkle.v1.json"),
        allowed_roots=ALLOWED_OUTPUT_ROOTS,
    )
    out_report = _path_arg(
        args,
        "out_report",
        "outReport",
        default=str(Path(output_dir) / "verification_report.tp.meta.verification_report.v1.json"),
        allowed_roots=ALLOWED_OUTPUT_ROOTS,
    )

    argv = [
        sys.executable,
        str(REPO_ROOT / "tools" / "verify_phase4_chain.py"),
        "--capture-metadata", capture_metadata,
        "--metadata-manifest", metadata_manifest,
        "--provenance-manifest", provenance_manifest,
        "--provenance-merkle", provenance_merkle,
        "--out-report", out_report,
    ]

    strict_input_order = _pick(args, "strict_input_order", "strictInputOrder")
    if strict_input_order is not None:
        if _as_bool(strict_input_order):
            argv.append("--strict-input-order")
        else:
            argv.append("--no-strict-input-order")
```

## API Request/Response

### Job Submission

```json
POST /v1/jobs
{
  "pipeline": "archive-gate-a",
  "input_dir": "/data/archive/incoming",
  "output_dir": "/data/archive/output",
  "args": {
    "archive_command": "phase4-verify",
    "capture_metadata": "/data/archive/output/capture_metadata.json",
    "metadata_manifest": "/data/archive/output/metadata_manifest.json",
    "provenance_manifest": "/data/archive/output/provenance_manifest.json",
    "provenance_merkle": "/data/archive/output/provenance_merkle.json",
    "out_report": "/data/archive/output/verification_report.json",
    "strict_input_order": true
  }
}
```

### Job Result

On success:
```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "schema": "tp.meta.machine.v1",
    "command": "phase4-verify",
    "success": true,
    "exit_code": 0,
    "data": {
      "verification_status": "passed",
      "computed": {
        "metadata_entry_count": 42,
        "provenance_leaf_count": 42,
        "provenance_merkle_root": "abc123..."
      }
    }
  }
}
```

On failure:
```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "schema": "tp.meta.machine.v1",
    "command": "phase4-verify",
    "success": false,
    "exit_code": 34,
    "error": {
      "code": "METADATA_HASH_MISMATCH",
      "message": "metadata_sha256 mismatch for images/photo_01.dng"
    }
  }
}
```

## Exit Codes

The Phase 4F verifier uses a dedicated exit code range (31-37):

| Exit Code | Label | Description |
|-----------|-------|-------------|
| 0 | SUCCESS | Verification passed |
| 31 | MALFORMED_INPUT | Input files cannot be parsed |
| 32 | SCHEMA_VALIDATION_FAILURE | Artifacts fail schema validation |
| 33 | ALIGNMENT_FAILURE | Path/version alignment errors |
| 34 | METADATA_HASH_MISMATCH | Recomputed metadata hash differs |
| 35 | PROVENANCE_ENTRY_HASH_MISMATCH | Recomputed provenance hash differs |
| 36 | MERKLE_MISMATCH | Merkle root or leaf count mismatch |
| 37 | REPORT_WRITE_FAILURE | Failed to write output report |

## Security Considerations

1. **Path Validation**: All input paths are validated against `ALLOWED_OUTPUT_ROOTS`
   to prevent path traversal attacks.

2. **Input Size Limits**: Large artifact files should respect the existing
   `TP_MAX_REQUEST_BYTES` limit for uploads via the job API.

3. **Timeout Enforcement**: The subprocess should inherit the existing
   archive gate timeout configuration.

## Testing Requirements

Before implementation:

1. Add contract tests for the new command in `tests/test_app_orchestrator_contract_http.py`
2. Add readiness check tests for Phase 4 schema validation
3. Add integration tests that verify end-to-end job dispatch
4. Update portal smoke tests if UI surfaces the new command

## Escalation Checklist

- [ ] Architect approval for `ARCHIVE_GATE_ALLOWED_COMMANDS` modification
- [ ] Security review of path validation approach
- [ ] CI/CD impact assessment
- [ ] Documentation update for API consumers
- [ ] Changelog entry for the new command
