# Machine-Mode JSON Contract (tp.meta.machine.v1)

**Status:** Official Contract
**Effective Date:** 2026-02-25
**Schema Version:** tp.meta.machine.v1
**Supersedes:** Human-readable text output for automation workflows

---

## Purpose

This document defines the **formal contract** for machine-readable JSON output mode across all CLI commands that support the `--json` flag.

The machine-mode contract ensures:
- **Deterministic structure**: Keys, ordering, and shape are stable across runs
- **Typed error handling**: Structured error information for automation
- **Exit code semantics**: Clear success/failure signaling for CI/CD pipelines
- **Version stability**: Schema changes require explicit version bumps

---

## Canonical JSON Schemas

The authoritative schema artifacts for this contract are versioned in-repo at:

- `docs/schemas/machine_mode/tp.meta.machine.v1/`

Canonical entrypoint:

- `docs/schemas/machine_mode/tp.meta.machine.v1/machine_mode.schema.json`

The schema entrypoint is enforced in CI and tests. Payloads that do not match this schema are contract violations.

---

## Contract Boundary

### What This Contract Covers

| Aspect | Guarantee | Notes |
|--------|-----------|-------|
| **Envelope Structure** | Keys, ordering, types are stable | Top-level envelope fields are versioned |
| **Exit Code Semantics** | Meaning of exit codes is stable | Source of truth is ingest layer exit codes |
| **Data Payload Keys** | Per-command keys are stable | Order and shape guaranteed deterministic |
| **Error Structure** | Typed errors with exit codes | Structured for programmatic parsing |

### What This Contract Does NOT Cover

| Aspect | Why Not Guaranteed | Guidance |
|--------|-------------------|----------|
| **Byte-identical outputs** | Some values are inherently variable | Use schema validation, not byte comparison |
| **Timing values** | `elapsed_seconds` varies by machine/load | Use as monitoring data, not assertion |
| **Environment versions** | Tool versions change with upgrades | Parse version strings, don't assume specific values |
| **Error messages** | Message text may be refined | Parse `error.type` and `error.exit_code`, not `error.message` |

---

## Envelope Structure

All machine-mode JSON outputs use a **common envelope** with these top-level fields:

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "<command-name>",
  "success": true|false,
  "exit_code": 0-255,
  "data": { /* command-specific payload */ },
  "error": null|{ /* typed error object */ }
}
```

### Envelope Fields

| Field | Type | Required | Semantics |
|-------|------|----------|-----------|
| `schema` | string | Yes | Always `"tp.meta.machine.v1"` for this version |
| `command` | string | Yes | Command name (e.g., `"extract"`, `"validate"`, `"check-system"`, `"summarize"`) |
| `success` | boolean | Yes | `true` if operation succeeded, `false` otherwise |
| `exit_code` | integer | Yes | Process exit code (0 = success, 1-255 = failure) |
| `data` | object | Yes | Command-specific result payload (see per-command schemas) |
| `error` | object\|null | Yes | Typed error for command-level failures, or `null` if no command error |

### Envelope Field Ordering

Envelope keys are serialized with `json.dumps(..., sort_keys=True)`, so ordering is lexicographic:

1. `command`
2. `data`
3. `error`
4. `exit_code`
5. `schema`
6. `success`

Consumers must parse by key name, not positional order. Additive keys may change relative key positions in future versions.

---

## Exit Code Semantics

Exit codes follow the ingest layer's `IngestExitCode` enum:

| Exit Code | Name | Meaning | When to Use |
|-----------|------|---------|-------------|
| 0 | SUCCESS | Operation completed successfully | All required checks passed |
| 1 | SCHEMA_VALIDATION_FAILED | Schema validation failed | Required fields missing, type mismatch, etc. |
| 2 | BIT_DEPTH_VIOLATION | 8-bit conversion detected | Quality firewall: image is 8-bit when 16-bit expected |
| 3 | GAMMA_VIOLATION | Gamma correction detected | Quality firewall: gamma curve applied when linear expected |
| 4 | SCHEMA_DRIFT | Schema drift detected | Unknown fields or structure changes detected |
| 5 | OTHER_FAILURE | Other failure | File not found, tool missing, I/O error, etc. |

**Exit code is the primary control signal for automation.** Your parser should route by `exit_code`, not by parsing `success` or error messages.

---

## Typed Error Structure

When `error` is not `null` at the command level (for command setup/orchestration failures), it has this shape:

```json
{
  "type": "OtherIngestFailure",
  "message": "Input directory does not exist: /missing",
  "exit_code": {
    "name": "OTHER_FAILURE",
    "value": 5
  },
  "priority": 10
}
```

### Error Fields

| Field | Type | Semantics |
|-------|------|-----------|
| `type` | string | Error class name (e.g., `"SchemaValidationFailure"`, `"OtherIngestFailure"`) |
| `message` | string | Human-readable description (may change; parse `type` not `message`) |
| `exit_code` | object | Structured exit code with `name` and `value` |
| `priority` | integer | Error priority for aggregation (higher = more severe) |

**For domain errors within data**, errors appear in `data.error` or `data.errors` (plural for batch operations), using the same structure.

---

## Per-Command Data Payloads

### `check-system`

System readiness check for required dependencies.

**Example:**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "check-system",
  "success": true,
  "exit_code": 0,
  "data": {
    "all_required_ok": true,
    "errors": [],
    "exiftool_available": true,
    "exiftool_version": "12.50",
    "git_available": true,
    "git_version": "2.39.0",
    "ingest_module_available": true,
    "libraw_version": "0.21.1",
    "pydantic_available": true,
    "pydantic_version": "2.5.3",
    "rawpy_available": true,
    "rawpy_version": "0.18.1"
  },
  "error": null
}
```

**Data Field Ordering (guaranteed):**

1. `all_required_ok`
2. `errors`
3. `exiftool_available`
4. `exiftool_version`
5. `git_available`
6. `git_version`
7. `ingest_module_available`
8. `libraw_version`
9. `pydantic_available`
10. `pydantic_version`
11. `rawpy_available`
12. `rawpy_version`

**Variable Fields:**
- Tool versions (`exiftool_version`, `git_version`, etc.) vary by environment
- `errors` array may contain diagnostic strings (for debugging, not parsing)

---

### `extract`

Single-image metadata extraction with provenance sidecar generation.

**Example (success):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "extract",
  "success": true,
  "exit_code": 0,
  "data": {
    "input_path": "input_images/IMG_1234.CR2",
    "success": true,
    "output_path": "output/IMG_1234.provenance.json",
    "elapsed_seconds": 1.234,
    "preset": "luxury",
    "error": null
  },
  "error": null
}
```

**Example (failure):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "extract",
  "success": false,
  "exit_code": 5,
  "data": {
    "input_path": "input_images/missing.CR2",
    "success": false,
    "output_path": null,
    "elapsed_seconds": 0.001,
    "preset": "luxury",
    "error": {
      "type": "OtherIngestFailure",
      "message": "Input file does not exist: input_images/missing.CR2",
      "exit_code": {"name": "OTHER_FAILURE", "value": 5},
      "priority": 10
    }
  },
  "error": null
}
```

**Data Field Ordering (guaranteed):**

1. `elapsed_seconds`
2. `error`
3. `input_path`
4. `output_path`
5. `preset`
6. `success`

**Variable Fields:**
- `elapsed_seconds`: Varies by machine, load, image size
- `error.message`: Text may change; parse `error.type` and `error.exit_code`

---

### `validate`

Schema validation of an existing provenance sidecar.

**Example (success):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "validate",
  "success": true,
  "exit_code": 0,
  "data": {
    "sidecar_path": "output/IMG_1234.provenance.json",
    "strict": true,
    "success": true,
    "errors": [],
    "dominant_error": null
  },
  "error": null
}
```

**Example (failure - schema drift):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "validate",
  "success": false,
  "exit_code": 4,
  "data": {
    "sidecar_path": "output/IMG_1234.provenance.json",
    "strict": true,
    "success": false,
    "errors": [
      {
        "type": "SchemaDriftFailure",
        "message": "Unknown field detected: extra_field",
        "exit_code": {"name": "SCHEMA_DRIFT", "value": 4},
        "priority": 30
      }
    ],
    "dominant_error": {
      "type": "SchemaDriftFailure",
      "message": "Unknown field detected: extra_field",
      "exit_code": {"name": "SCHEMA_DRIFT", "value": 4},
      "priority": 30
    }
  },
  "error": null
}
```

**Data Field Ordering (guaranteed):**

1. `dominant_error`
2. `errors`
3. `sidecar_path`
4. `strict`
5. `success`

**Variable Fields:**
- `errors` array: May contain multiple typed errors
- `dominant_error`: Highest-priority error (for routing)

---

### `extract-batch`

Batch metadata extraction for multiple images.

**Example (partial success):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "extract-batch",
  "success": false,
  "exit_code": 5,
  "data": {
    "input_root": "input_images",
    "output_dir": "output",
    "fail_fast": false,
    "preserve_structure": false,
    "success": false,
    "items": [
      {
        "path": "input_images/IMG_1234.CR2",
        "success": true,
        "output_path": "output/IMG_1234.provenance.json",
        "elapsed_seconds": 1.234,
        "error": null
      },
      {
        "path": "/input/IMG_5678.CR2",
        "success": false,
        "output_path": null,
        "elapsed_seconds": 0.001,
        "error": {
          "type": "OtherIngestFailure",
          "message": "exiftool not found",
          "exit_code": {"name": "OTHER_FAILURE", "value": 5},
          "priority": 10
        }
      }
    ],
    "summary_counts": {
      "total": 2,
      "success": 1,
      "failure": 1,
      "by_exit_code": {
        "OTHER_FAILURE": 1
      }
    },
    "dominant_error": {
      "type": "OtherIngestFailure",
      "message": "exiftool not found",
      "exit_code": {"name": "OTHER_FAILURE", "value": 5},
      "priority": 10
    }
  },
  "error": null
}
```

**Data Field Ordering (guaranteed):**

1. `dominant_error`
2. `fail_fast`
3. `input_root`
4. `items`
5. `output_dir`
6. `preserve_structure`
7. `success`
8. `summary_counts`

**Batch Item Field Ordering (guaranteed):**

1. `elapsed_seconds`
2. `error`
3. `output_path`
4. `path`
5. `success`

**Summary Counts Field Ordering (guaranteed):**

1. `by_exit_code` (sorted by exit code enum value, then alphabetically)
2. `failure`
3. `success`
4. `total`

**Variable Fields:**
- `items` array: Length varies by input set
- `elapsed_seconds`: Varies per image
- `summary_counts.by_exit_code`: Keys depend on which exit codes occurred

---

### `summarize`

Aggregate summary of sidecar validation results in a directory.

**Example (success):**

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "summarize",
  "success": true,
  "exit_code": 0,
  "data": {
    "sidecar_dir": "/output/provenance_sidecars",
    "total_sidecars": 12,
    "valid": 12,
    "invalid": 0,
    "errors": []
  },
  "error": null
}
```

**Data Field Ordering (guaranteed):**

1. `errors`
2. `invalid`
3. `sidecar_dir`
4. `total_sidecars`
5. `valid`

**Variable Fields:**
- `total_sidecars`: Depends on files present in the directory
- `errors`: Contains parse/load failures for unreadable sidecars

---

## Determinism Semantics

### What IS Deterministic

✅ **Structure and ordering:**
- Envelope field order
- Data field order
- Array field order within objects
- Exit code enum ordering in `by_exit_code`

✅ **Keys and types:**
- Field names never change within a schema version
- Field types are stable

✅ **Exit code values:**
- Exit code integers are stable for each named code

### What IS NOT Deterministic

❌ **Timing values:**
- `elapsed_seconds` varies by machine, load, concurrency

❌ **Environment versions:**
- Tool versions (`exiftool_version`, `git_version`) change with upgrades

❌ **Error messages:**
- `error.message` text may be refined for clarity
- Parse `error.type` and `error.exit_code.name`, not message text

❌ **UUIDs and timestamps:**
- If present in data (e.g., `run_id`), these are non-deterministic by design

**Automation guidance:** Use **schema validation** (structure + types) and **exit code routing** (semantic meaning), not byte-level diffing.

---

## CLI Usage

### Basic Usage

```bash
# Emit machine JSON to stdout
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2

# Pretty-printed JSON to stdout
.venv/bin/python scripts/test_metadata_extraction.py --json --json-pretty extract input_images/image.CR2

# Write JSON to file, keep stdout clean
.venv/bin/python scripts/test_metadata_extraction.py --json --json-output result.json extract input_images/image.CR2

# Pretty JSON to file
.venv/bin/python scripts/test_metadata_extraction.py --json --json-pretty --json-output result.json extract input_images/image.CR2
```

### Exit Code Handling

```bash
# Exit code reflects operation status
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2
echo $?  # 0 = success, 1-5 = specific failure modes

# Route by exit code in CI
if .venv/bin/python scripts/test_metadata_extraction.py --json validate sidecar.json; then
  echo "Validation passed"
else
  exit_code=$?
  case $exit_code in
    1) echo "Schema validation failed" ;;
    2) echo "8-bit conversion detected" ;;
    3) echo "Gamma correction detected" ;;
    4) echo "Schema drift detected" ;;
    5) echo "Other failure (file not found, tool missing, etc.)" ;;
  esac
  exit $exit_code
fi
```

### Flags and Requirements

| Flag | Requires | Effect |
|------|----------|--------|
| `--json` | (none) | Emit machine JSON to stdout |
| `--json-pretty` | `--json` | Pretty-print JSON (2-space indent) |
| `--json-output <path>` | `--json` | Write JSON to file, keep stdout clean |

**Error:** Using `--json-pretty` or `--json-output` without `--json` exits with code 2 and an error message on stderr.

---

## Reference Parser (Python)

A minimal reference parser for consuming machine-mode JSON:

```python
"""Reference parser for tp.meta.machine.v1 JSON output."""

import json
import sys
from pathlib import Path
from typing import Any, Dict


def parse_machine_json(json_str: str) -> Dict[str, Any]:
    """Parse and validate machine JSON envelope."""
    payload = json.loads(json_str)

    # Validate schema version
    schema = payload.get("schema")
    if schema != "tp.meta.machine.v1":
        raise ValueError(f"Unsupported schema: {schema}")

    return payload


def route_by_command(payload: Dict[str, Any]) -> None:
    """Route by command and handle typed results."""
    command = payload["command"]
    exit_code = payload["exit_code"]
    success = payload["success"]
    data = payload["data"]

    if command == "check-system":
        if success:
            print(f"✅ System check passed (all_required_ok={data['all_required_ok']})")
        else:
            print(f"❌ System check failed: {data['errors']}")

    elif command == "extract":
        if success:
            print(f"✅ Extracted: {data['input_path']} → {data['output_path']}")
            print(f"   Elapsed: {data['elapsed_seconds']:.3f}s")
        else:
            error = data["error"]
            print(f"❌ Extract failed: {data['input_path']}")
            print(f"   Error type: {error['type']}")
            print(f"   Exit code: {error['exit_code']['name']} ({error['exit_code']['value']})")

    elif command == "validate":
        if success:
            print(f"✅ Validation passed: {data['sidecar_path']}")
        else:
            print(f"❌ Validation failed: {data['sidecar_path']}")
            dominant = data["dominant_error"]
            print(f"   Dominant error: {dominant['type']} ({dominant['exit_code']['name']})")
            print(f"   Total errors: {len(data['errors'])}")

    elif command == "extract-batch":
        counts = data["summary_counts"]
        print(f"Batch result: {counts['success']}/{counts['total']} succeeded")
        if not success:
            dominant = data["dominant_error"]
            print(f"   Dominant error: {dominant['type']}")
            print(f"   Failed by exit code: {counts['by_exit_code']}")

    elif command == "summarize":
        print(f"Sidecars: {data['valid']}/{data['total_sidecars']} valid")
        if not success:
            print(f"   Errors: {data['errors']}")

    else:
        print(f"Unknown command: {command}")

    sys.exit(exit_code)


def main() -> None:
    """Parse machine JSON from stdin or file argument."""
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        json_str = json_path.read_text(encoding="utf-8")
    else:
        json_str = sys.stdin.read()

    payload = parse_machine_json(json_str)
    route_by_command(payload)


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Parse from file
.venv/bin/python tools/parse_machine_json.py result.json

# Parse from stdin
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2 | .venv/bin/python tools/parse_machine_json.py

# Exit code forwarding
.venv/bin/python scripts/test_metadata_extraction.py --json validate sidecar.json | .venv/bin/python tools/parse_machine_json.py
echo $?  # Parser exits with same code as CLI
```

---

## Reference Parser (jq + bash)

For bash automation without Python:

### Extract Command

```bash
#!/bin/bash
set -euo pipefail

result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract "$1")
exit_code=$?

# Validate schema
schema=$(echo "$result" | jq -r '.schema')
if [[ "$schema" != "tp.meta.machine.v1" ]]; then
  echo "ERROR: Unsupported schema: $schema" >&2
  exit 99
fi

# Route by exit code
if [[ $exit_code -eq 0 ]]; then
  input_path=$(echo "$result" | jq -r '.data.input_path')
  output_path=$(echo "$result" | jq -r '.data.output_path')
  echo "✅ Extracted: $input_path → $output_path"
  exit 0
else
  error_type=$(echo "$result" | jq -r '.data.error.type')
  error_name=$(echo "$result" | jq -r '.data.error.exit_code.name')
  echo "❌ Extract failed: $error_type ($error_name)" >&2
  exit $exit_code
fi
```

### Validate Command

```bash
#!/bin/bash
set -euo pipefail

result=$(.venv/bin/python scripts/test_metadata_extraction.py --json validate "$1")
exit_code=$?

success=$(echo "$result" | jq -r '.success')
sidecar=$(echo "$result" | jq -r '.data.sidecar_path')

if [[ "$success" == "true" ]]; then
  echo "✅ Validation passed: $sidecar"
  exit 0
else
  dominant_type=$(echo "$result" | jq -r '.data.dominant_error.type')
  dominant_name=$(echo "$result" | jq -r '.data.dominant_error.exit_code.name')
  error_count=$(echo "$result" | jq '.data.errors | length')
  echo "❌ Validation failed: $sidecar" >&2
  echo "   Dominant error: $dominant_type ($dominant_name)" >&2
  echo "   Total errors: $error_count" >&2
  exit $exit_code
fi
```

### Batch Summary

```bash
#!/bin/bash
set -euo pipefail

result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract-batch "$1")
exit_code=$?

total=$(echo "$result" | jq '.data.summary_counts.total')
success=$(echo "$result" | jq '.data.summary_counts.success')
failure=$(echo "$result" | jq '.data.summary_counts.failure')

echo "Batch result: $success/$total succeeded, $failure failed"

if [[ $exit_code -ne 0 ]]; then
  echo "Exit code breakdown:" >&2
  echo "$result" | jq -r '.data.summary_counts.by_exit_code | to_entries[] | "  \(.key): \(.value)"' >&2
fi

exit $exit_code
```

---

## Observability and Debugging

### Human Debugging vs Machine Output

| Mode | Primary Output | Diagnostics | When to Use |
|------|----------------|-------------|-------------|
| **Human mode** (default) | Stdout (text) | Inline + stderr | Interactive debugging, ad-hoc analysis |
| **Machine mode** (`--json`) | Stdout/file (JSON) | Stderr only (when requested) | Automation, CI/CD, programmatic parsing |

### Diagnostic Output Policy

**In machine mode:**
- Stdout (or `--json-output` file) contains **only machine JSON**
- Stderr is **reserved for diagnostics** (optional, not guaranteed)
- Progress bars, debug logs, warnings → **stderr or disabled**

**To get diagnostics in machine mode:**

```bash
# Separate JSON (stdout) from diagnostics (stderr)
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2 > result.json 2> diagnostics.log

# Interactive debugging: see diagnostics, capture JSON
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2 2>&1 | tee full_output.log | jq .
```

**DO NOT parse stderr in automation.** It may contain:
- Progress indicators (may be removed)
- Debug logs (format may change)
- Warnings (non-fatal, informational)

**Only parse JSON from stdout (or `--json-output` file).**

---

## Contract Versioning

### Schema Version Bumping Criteria

A new schema version (e.g., `tp.meta.machine.v2`) is required when:

✅ **Envelope changes:**
- Adding/removing/renaming envelope fields
- Changing envelope field types
- Changing envelope field ordering semantics

✅ **Data payload breaking changes:**
- Removing a data field
- Renaming a data field
- Changing a field type (e.g., string → integer)
- Changing field ordering guarantees

✅ **Exit code semantics changes:**
- Changing exit code integer values
- Changing exit code meanings

### Non-Breaking Changes (No Version Bump)

❌ **Additive changes within `data`:**
- Adding new optional fields to `data` (consumers should ignore unknown keys)

❌ **Error message refinements:**
- Improving `error.message` text (consumers should parse `error.type`)

❌ **Performance optimizations:**
- Changing internal implementation without output shape changes

---

## Migration Plan (Future Versions)

When `tp.meta.machine.v2` is introduced:

1. **Parallel support:** Both v1 and v2 available via `--json-schema v1|v2` flag (default: v2)
2. **Deprecation period:** v1 supported for ≥6 months after v2 release
3. **EOL warning:** v1 emits warning on stderr: `WARN: tp.meta.machine.v1 is deprecated, use --json-schema v2`
4. **EOL date:** v1 removed in a major version bump (e.g., 2.0.0)

---

## Support and Feedback

**Questions or issues?**

1. Check this contract documentation first
2. Review examples in `tests/ingest/test_metadata_cli_machine_mode.py`
3. Open a GitHub issue with:
   - Schema version (`tp.meta.machine.v1`)
   - Command (`extract`, `validate`, etc.)
   - Expected vs actual behavior
   - Minimal reproducible example

**Reporting contract violations:**

If you find a case where the implementation violates this contract:

1. File a GitHub issue with `[contract-violation]` label
2. Include the command that violates the contract
3. Include the actual JSON output vs. contract spec
4. This is a **blocking bug** and will be prioritized

---

## Related Documentation

- **Ingest Contract:** [docs/apex/ingest_contract.md](../apex/ingest_contract.md) - Provenance sidecar schema
- **Machine JSON Schemas:** [docs/schemas/machine_mode/tp.meta.machine.v1/machine_mode.schema.json](../schemas/machine_mode/tp.meta.machine.v1/machine_mode.schema.json) - Canonical schema entrypoint
- **Exit Codes:** [src/transformation_portal/ingest/errors.py](../../src/transformation_portal/ingest/errors.py) - Exit code enum definitions
- **Machine Output Implementation:** [src/transformation_portal/ingest/machine_output.py](../../src/transformation_portal/ingest/machine_output.py) - Serializer source code
- **Contract Tests:** [tests/ingest/test_metadata_cli_machine_mode.py](../../tests/ingest/test_metadata_cli_machine_mode.py) - Enforcement tests
- **JSON Schema Tests:** [tests/ingest/test_machine_mode_jsonschema.py](../../tests/ingest/test_machine_mode_jsonschema.py) - Schema contract validation

---

## Changelog

### tp.meta.machine.v1 (2026-02-25)

**Initial release** (PR #1024)

- Envelope structure with `schema`, `command`, `success`, `exit_code`, `data`, `error`
- Per-command data payloads: `check-system`, `extract`, `validate`, `extract-batch`, `summarize`
- Typed error structure with exit code objects
- Deterministic JSON serialization (`sort_keys=True`)
- Exit code semantics from `IngestExitCode` enum
- Golden master tests for contract enforcement
