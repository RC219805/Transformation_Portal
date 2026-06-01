# Schema Contracts

This directory contains repo-owned machine-readable contracts and versioned
runtime profiles that code loads directly from the repository root. It is a
live contract surface, not historical material.

## Placement

- `schemas/phase4/*.schema.json` are the authoritative JSON Schema contracts
  for Phase 4 capture metadata, manifests, provenance, Merkle roots, and
  verification reports. ADR-035 and the Phase 4 tools intentionally reference
  this root-level location.
- `schemas/profiles/*.json` are versioned runtime projection profiles. They
  are not JSON Schema contracts and must not use the `.schema.json` suffix.
- Published schema contracts, schema docs, and externally linked schema copies
  that are not runtime defaults belong under `docs/schemas/`.

Do not add another top-level schema tree. Use root `schemas/` only when the
runtime, validation tools, or tests intentionally load the file from the repo
root. Use `docs/schemas/` for published documentation contracts.

## Naming

- JSON Schema files use `*.schema.json`, declare Draft 2020-12 with `$schema`,
  and keep stable `$id` values that match their repo path.
- Runtime profile files use a fully qualified versioned contract id as the
  filename, for example `tp.projection.machine_to_evidence.v1.json`.
- Runtime profile payloads include a `schema` field matching the filename stem
  and a `source_schema` field for the input contract they project.

## Validation

Run the focused topology and contract checks after moving, adding, or editing
schema files:

```bash
.venv/bin/python -m pytest tests/test_schema_topology_contract.py
.venv/bin/python tools/validate_evalsuite_contract_schemas.py
```
