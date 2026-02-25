# Phase 4D Metadata Hash Canonicalization Spec

## Status

- Normative
- `canonicalization_spec_version`: `phase4d-canon-v1`
- Applies to:
  - `tp.meta.capture.v1` (object hashing preimage)
  - `tp.meta.capture_manifest.v1` (manifest bytes)

## Scope

This spec defines byte-level canonicalization for Phase 4D object hashing and
manifest serialization.

Any change that can alter emitted bytes or hashes requires contract versioning
per ADR-035.

## Object Hash Preimage (`metadata_sha256`)

For each metadata object:

1. Input object MUST validate against `schemas/phase4/metadata.schema.json`.
2. Serialize the single object with:
   - UTF-8
   - `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`
3. No trailing newline in the object preimage.
4. Hash bytes with SHA-256 and encode as lowercase hex.

Definition:

```text
metadata_sha256 = SHA256(canonical_object_bytes).hexdigest()
```

## Manifest Bytes (`tp.meta.capture_manifest.v1`)

Manifest payload:

- `metadata_manifest_contract_version = "tp.meta.capture_manifest.v1"`
- `metadata_contract_version = "tp.meta.capture.v1"`
- `entries` sorted lexicographically by `relative_path`

Each entry:

- `relative_path`
- `file_sha256`
- `metadata_sha256`

Manifest serialization:

- UTF-8
- `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`
- Exactly one trailing LF (`\n`)

## Data Rules

- `relative_path` uniqueness is required.
- Paths are normalized upstream by Phase 4C; this spec assumes already
  normalized paths in input objects.
- No NaN/Infinity values are allowed (`allow_nan=False` is mandatory).
- Non-ASCII is allowed and preserved (`ensure_ascii=False`).
- Strings are consumed as already-canonicalized values from Phase 4C.
- Unicode normalization is not performed in Phase 4D; byte differences are
  authoritative at this layer.
- Lists are order-sensitive. Phase 4D does not reorder arrays; list ordering
  MUST already be deterministic in upstream Phase 4C output.
- Float representation is CPython JSON encoder output under the serialization
  settings above. Determinism is enforced by upstream Phase 4C rounding rules
  and by cross-runtime parity tests.
- Input payloads are expected to be JSON-native types only
  (`dict/list/str/int/float/bool/null`).

## Reference Implementation

- Object hashing + manifest serialization:
  - `tp/phase4/hash_capture_metadata.py`
- CLI builder:
  - `tools/build_metadata_manifest.py`
- Cross-runtime parity checker:
  - `tools/check_phase4d_manifest_cross_runtime.py`
