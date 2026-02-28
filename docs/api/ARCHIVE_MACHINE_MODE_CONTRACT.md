# Archive Machine-Mode JSON Contract (`tp.archive.machine.v1`)

**Status:** Official Contract
**Effective Date:** 2026-02-28
**Schema Version:** `tp.archive.machine.v1`
**Compatibility:** Additive to existing `tp.meta.machine.v1`; no modifications to ingest machine contract.

## Purpose
This contract defines deterministic machine-readable output for archive-governance workflows driven by `tools/archive_governance.py`.

## Canonical Schemas
Schema root:
- `docs/schemas/machine_mode/tp.archive.machine.v1/`

Entrypoint:
- `docs/schemas/machine_mode/tp.archive.machine.v1/machine_mode.schema.json`

## Envelope
All commands emit this envelope:

```json
{
  "schema": "tp.archive.machine.v1",
  "command": "fixity-scan",
  "success": true,
  "exit_code": 0,
  "data": {},
  "error": null
}
```

Top-level key ordering is deterministic through canonical serialization. Consumers must parse by key name, not key position.

## Command Set
- `fixity-scan`
- `fixity-verify`
- `manifest-build`
- `rights-apply`
- `bag-build`
- `bag-validate`
- `premis-export`
- `dedup-plan`
- `mets-export`
- `prov-export`
- `stac-export`
- `sealed-eval-run`

`sealed-eval-run` supports `--subset-root` and enforces a read-only subset mount contract by default (override with `--allow-writable-subset` for local development). The harness writes a deterministic audit package under `audit_package/` with `audit_manifest.json`.

## Serialization Profiles
`archive_governance.py` supports:
- `canonical_v1`: UTF-8 JSON, sorted keys, `allow_nan=false`
- `jcs`: RFC 8785 canonical JSON via `transformation_portal.determinism.jcs`

## Error Contract
On failure, `error` is a typed object with:
- `type`
- `message`
- `exit_code` (`name`, `value`)
- `priority`

Failures should route by `exit_code` and `error.type`, not message text.

## Non-Deterministic Fields
The contract allows runtime variance in non-semantic command text fields such as human-oriented tool `stdout/stderr` snapshots. Deterministic artifacts are surfaced via explicit output paths in `data`.

## Versioning Policy
- Additive changes in `data` payloads are allowed within v1.
- Breaking structural changes require version bump to `tp.archive.machine.v2`.
