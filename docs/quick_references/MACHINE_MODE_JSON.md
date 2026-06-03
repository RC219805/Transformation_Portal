# Machine-Mode JSON Quick Reference

**For automation engineers consuming metadata CLI output.**

---

## Basic Usage

```bash
# Emit JSON to stdout
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2

# Write JSON to file
.venv/bin/python scripts/test_metadata_extraction.py --json --json-output result.json extract input_images/image.CR2

# Pretty-printed JSON
.venv/bin/python scripts/test_metadata_extraction.py --json --json-pretty extract input_images/image.CR2
```

---

## Exit Codes (Route by This!)

| Code | Name | Meaning |
|------|------|---------|
| 0 | SUCCESS | All checks passed |
| 1 | SCHEMA_VALIDATION_FAILED | Required fields missing, type mismatch |
| 2 | BIT_DEPTH_VIOLATION | 8-bit conversion detected (quality gate) |
| 3 | GAMMA_CORRECTION_DETECTED | Gamma curve applied (quality gate) |
| 4 | SCHEMA_DRIFT | Unknown fields detected |
| 5 | OTHER_FAILURE | File not found, tool missing, I/O error |

**Use exit code as primary control signal, not error messages.**

---

## JSON Envelope (Always Present)

```json
{
  "schema": "tp.meta.machine.v1",
  "command": "extract|validate|extract-batch|check-system",
  "success": true|false,
  "exit_code": 0-5,
  "data": { /* command-specific */ },
  "error": null|{ /* typed error */ }
}
```

**Fields are always in this order** (sorted keys).

---

## Commands

### `extract` - Single Image

```bash
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/IMG_1234.CR2
```

**Success:**
```json
{
  "data": {
    "input_path": "input_images/IMG_1234.CR2",
    "output_path": "output/IMG_1234.provenance.json",
    "elapsed_seconds": 1.234,
    "preset": "luxury",
    "success": true,
    "error": null
  }
}
```

**Failure:**
```json
{
  "data": {
    "input_path": "input_images/missing.CR2",
    "success": false,
    "error": {
      "type": "OtherIngestFailure",
      "exit_code": {"name": "OTHER_FAILURE", "value": 5}
    }
  }
}
```

### `validate` - Schema Check

```bash
.venv/bin/python scripts/test_metadata_extraction.py --json validate output/sidecar.json
```

**Success:**
```json
{
  "data": {
    "sidecar_path": "output/sidecar.json",
    "strict": true,
    "success": true,
    "errors": [],
    "dominant_error": null
  }
}
```

**Failure:**
```json
{
  "data": {
    "success": false,
    "errors": [
      {"type": "SchemaValidationFailure", "message": "..."}
    ],
    "dominant_error": {"type": "SchemaValidationFailure", ...}
  }
}
```

### `extract-batch` - Multiple Images

```bash
.venv/bin/python scripts/test_metadata_extraction.py --json extract-batch input_images --output output
```

**Result:**
```json
{
  "data": {
    "input_root": "input_images",
    "output_dir": "output",
    "success": false,
    "items": [
      {"path": "input_images/a.CR2", "success": true, ...},
      {"path": "input_images/b.CR2", "success": false, "error": {...}}
    ],
    "summary_counts": {
      "total": 2,
      "success": 1,
      "failure": 1,
      "by_exit_code": {"OTHER_FAILURE": 1}
    },
    "dominant_error": {...}
  }
}
```

### `check-system` - Readiness

```bash
.venv/bin/python scripts/test_metadata_extraction.py --json check-system
```

**Result:**
```json
{
  "data": {
    "all_required_ok": true,
    "exiftool_available": true,
    "exiftool_version": "12.50",
    "git_available": true,
    ...
  }
}
```

---

## Reference Parser (Python)

```bash
# Parse from file or stdin
.venv/bin/python tools/parse_machine_json.py result.json
.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2 | .venv/bin/python tools/parse_machine_json.py

# Exit code forwarded
echo $?
```

---

## Bash + jq Examples

```bash
# Extract with exit code routing
result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2)
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
  output=$(echo "$result" | jq -r '.data.output_path')
  echo "✅ Extracted: $output"
else
  error_type=$(echo "$result" | jq -r '.data.error.type')
  echo "❌ Failed: $error_type"
  exit $exit_code
fi
```

```bash
# Validate with error details
result=$(.venv/bin/python scripts/test_metadata_extraction.py --json validate output/sidecar.json)

if [[ $(echo "$result" | jq -r '.success') == "true" ]]; then
  echo "✅ Valid"
else
  echo "$result" | jq -r '.data.errors[] | "- \(.type): \(.message)"'
  exit $(echo "$result" | jq -r '.exit_code')
fi
```

```bash
# Batch summary
result=$(.venv/bin/python scripts/test_metadata_extraction.py --json extract-batch input_images --output output)

total=$(echo "$result" | jq '.data.summary_counts.total')
success=$(echo "$result" | jq '.data.summary_counts.success')
echo "Result: $success/$total succeeded"
```

**More examples:** `tools/parse_machine_json_examples.sh`

---

## Determinism Notes

### ✅ Deterministic (Stable)
- Envelope structure
- Field ordering
- Exit code values
- Key names and types

### ❌ Non-Deterministic (Variable)
- `elapsed_seconds` (varies by machine/load)
- Tool versions (e.g., `exiftool_version`)
- Error messages (parse `error.type`, not `error.message`)

**Use schema validation, not byte-exact diffing.**

---

## Full Documentation

- **Contract Spec:** `docs/api/MACHINE_MODE_CONTRACT.md`
- **Reference Parser:** `tools/parse_machine_json.py`
- **Bash Examples:** `tools/parse_machine_json_examples.sh`
- **Tests:** `tests/ingest/test_metadata_cli_machine_mode.py`

---

## Troubleshooting

**Problem:** `--json-pretty requires --json`
**Fix:** Always use `--json` flag before `--json-pretty` or `--json-output`

**Problem:** Exit code always 0
**Fix:** Don't use `set -e` before capturing JSON; capture exit code explicitly:
```bash
result=$(.venv/bin/python scripts/test_metadata_extraction.py --json ...)
exit_code=$?  # Capture before any other commands
```

**Problem:** Unexpected JSON shape
**Fix:** Validate `schema` field equals `tp.meta.machine.v1` first

**Problem:** Byte-exact outputs expected
**Fix:** Don't compare bytes; use schema validation and exit code routing

---

## Schema Versioning

**Current:** `tp.meta.machine.v1`

**Version bumps when:**
- Envelope field changes (add/remove/rename)
- Data field changes (remove/rename/type change)
- Exit code semantics changes

**Non-breaking (no bump):**
- Adding optional fields to `data`
- Improving error messages
- Performance optimizations

---

## Support

**Questions?** Check `docs/api/MACHINE_MODE_CONTRACT.md` first.

**Contract violations?** File GitHub issue with `[contract-violation]` label.

**Feature requests?** File GitHub issue describing automation use case.
