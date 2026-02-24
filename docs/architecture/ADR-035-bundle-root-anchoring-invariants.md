# ADR-035 Bundle Root Anchoring Invariants

## Status
Proposed

## Date
2026-02-24

## Context

Phase 3.3 introduced canonical evidence bundle manifests and strict digest
verification across roots, hash manifest, hash summary, signature, and detached
timestamp artifacts.

Phase 3.4 adds bundle-level anchoring using a deterministic
`bundle_root_sha256`. This root is intended for external timestamping, signing,
and ledger anchoring.

The repository needs a stable governance contract that freezes:

- the projection used to compute the bundle root,
- fields explicitly excluded from the preimage,
- byte-level serialization rules,
- root-field co-presence semantics,
- exit code behavior,
- backward compatibility with Phase 3.3 manifests.

## Decision

### D1. Canonical Projection Is Fixed

The bundle root preimage projection includes exactly:

- `bundle_version`
- `hash_algorithm`
- `roots_sha256`
- `hash_manifest_sha256`
- `hash_summary_sha256`
- `signature_sha256`
- `timestamp_target`
- `timestamp_sha256`
- `merkle_leaf_count`
- `phase3_version`
- `phase3_1_version`
- `phase3_2_version`
- `bundle_tool_name`
- `bundle_tool_version`

### D2. Excluded Fields Are Fixed

The bundle root preimage excludes:

- all manifest path fields (`*_path`),
- the entire optional `notarization` block,
- any unspecified manifest fields.

### D3. Preimage Serialization Is Fixed

The canonical preimage bytes are JSON serialized with:

- UTF-8
- sorted keys (`sort_keys=True`)
- compact separators (`separators=(",", ":")`)
- trailing LF (`\n`) required

The bundle root is:

`bundle_root_sha256 = sha256(preimage_v1)`

### D4. Root Field Co-Presence Is Mandatory

If any bundle-root metadata field is present, all are required:

- `bundle_root_algorithm`
- `bundle_root_preimage_version`
- `bundle_root_sha256`

### D5. Backward Compatibility Is Required

Verifiers must accept:

- Phase 3.3 manifests with no bundle root fields,
- Phase 3.4 manifests with bundle root fields,
- Phase 3.4 manifests with optional notarization blocks.

### D6. Exit Code Contract Is Frozen

`tools/compute_bundle_root.py`:

- `0`: success
- `21`: malformed manifest / invalid arguments
- `22`: write failure
- `23`: root mismatch against existing manifest root

`tools/verify_evidence_bundle_manifest.py`:

- `0`: success
- `11`: verification mismatch / failure
- `12`: malformed input

## Enforcement

Enforcement occurs through:

- strict schema constraints in
  `docs/archive/schemas/evidence_bundle_manifest.schema.json`,
- shared projection/validation logic in `tools/bundle_root_common.py`,
- verifier behavior in `tools/verify_evidence_bundle_manifest.py`,
- determinism and parity tests in `tests/test_bundle_root.py`,
- CI gate in `.github/workflows/determinism-gate.yml`,
- cross-runtime parity check in `tools/check_bundle_root_cross_runtime.py`.

## Consequences

Positive:

- stable external anchoring semantics,
- deterministic roots across supported Python runtimes,
- reduced risk of projection drift.

Trade-off:

- projection changes require a deliberate versioned preimage evolution and ADR
  update.

## References

- `docs/archive/ARCHIVE_MANIFEST_PHASE3_EVIDENCE_BUNDLE.md`
- `docs/archive/schemas/evidence_bundle_manifest.schema.json`
- `tools/bundle_root_common.py`
- `tools/compute_bundle_root.py`
- `tools/verify_evidence_bundle_manifest.py`
- `tests/test_bundle_root.py`
