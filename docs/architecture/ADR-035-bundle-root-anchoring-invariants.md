# ADR-035 Bundle Root Anchoring Invariants

## Status
Proposed

## Date
2026-02-24

## Executive Summary

Phase 3.4 introduced a canonical bundle-root digest for evidence bundles. This
ADR freezes the root projection, preimage serialization, field exclusion rules,
and exit code behavior so all tools and CI environments compute a stable
cross-runtime anchor for external timestamping, signatures, and ledger binding.

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

### D7. Phase 4 Contract Surfaces Are Versioned and Additive

Phase 4 introduces additive, versioned contracts for capture provenance:

- `tp.meta.capture.v1` at `schemas/phase4/metadata.schema.json`
- `tp.meta.capture_manifest.v1` at `schemas/phase4/metadata_manifest.schema.json`
- `tp.meta.provenance.v1` at `schemas/phase4/provenance_manifest.schema.json`
- `tp.meta.provenance_merkle.v1` at `schemas/phase4/provenance_merkle.schema.json`

The authoritative machine-readable Phase 4 contract location is
`schemas/phase4/`. Documentation references under `docs/` are informative and
must not be treated as canonical schema sources.

These additions MUST NOT mutate Phase 3 artifact schemas or silently extend
Evidence Bundle v1. Any incompatible contract change requires an explicit
version bump and corresponding ADR update.

### D8. Provenance Entry Hash Semantics Must Mirror Phase 3

`provenance_entry_sha256` representation and concatenation conventions are part
of the contract surface. Phase 4 implementations MUST mirror the established
Phase 3 convention exactly. Changing digest representation (binary vs hex
text), concatenation rules, or odd-leaf handling semantics requires a version
bump and ADR update.

### D9. Phase 4B Canonicalization Governance Is Frozen

Phase 4B introduces canonicalization semantics governance before extractor
implementation:

- canonicalization config file: `tools/capture_metadata_config.json`
- config fingerprint surface:
  `capture_metadata_config_fingerprint_sha256`
- deterministic fingerprint tool: `tools/capture_metadata_fingerprint.py`
- golden fingerprint fixture:
  `tests/golden/phase4/config_fingerprint.txt`

The canonicalization config is version-bound to `tp.meta.capture.v1`.
Any semantic change (for example tag whitelist, datetime precedence, rounding
rules, warning taxonomy, or path normalization policy) requires all of:

1. config file update,
2. fingerprint change,
3. golden fixture update,
4. explicit PR note documenting the semantic change.

Phase 4C extractor implementations MUST embed
`capture_metadata_config_fingerprint_sha256` in emitted metadata objects.

### D10. Phase 4C Deterministic Extraction Surface Is Isolated

Phase 4C introduces one new deterministic artifact surface only:

- `artifacts/capture_metadata.tp.meta.capture.v1.json`

Phase 4C implementations MUST:

- consume `tools/capture_metadata_config.json`,
- embed `capture_metadata_config_fingerprint_sha256` in each output record,
- validate each record against `schemas/phase4/metadata.schema.json` before write,
- serialize output canonically (`sort_keys=True`, compact separators,
  `ensure_ascii=False`, `allow_nan=False`, UTF-8).

Phase 4C implementations MUST NOT introduce:

- metadata/provenance binding or Merkle roots,
- evidence bundle schema mutations,
- machine-contract mutations,
- changes to Phase 2/3 integrity surfaces.

### D11. Phase 4D Object-Hash and Metadata-Manifest Semantics Are Frozen

Phase 4D introduces one new deterministic artifact surface only:

- `artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json`

Normative references:

- Canonicalization spec:
  `docs/contracts/phase4d_metadata_hash_canonicalization.md`
- Exit-code contract:
  `docs/contracts/exit_codes.md`

Phase 4D object-level hash semantics are fixed:

- `metadata_sha256` is computed over the canonical JSON bytes of each single
  metadata object (`sort_keys=True`, compact separators, `ensure_ascii=False`,
  `allow_nan=False`, UTF-8, no trailing newline in object preimage).
- Hashing is object-scoped only; array-level formatting does not affect
  `metadata_sha256`.

Phase 4D builder invariants are fixed:

- input must be an array of `tp.meta.capture.v1` objects,
- each object must validate against `schemas/phase4/metadata.schema.json`,
- `relative_path` values must be unique,
- strict mode requires input already sorted by `relative_path`,
- output manifest entries are sorted by `relative_path`,
- optional fingerprint enforcement requires each record extractor fingerprint to
  match the current `tools/capture_metadata_config.json` fingerprint.

Phase 4D manifest serialization is fixed:

- canonical JSON (`sort_keys=True`, compact separators, `ensure_ascii=False`,
  `allow_nan=False`, UTF-8),
- single trailing LF required.

Phase 4D determinism verification is fixed:

- cross-runtime parity MUST hold across supported interpreters for both
  `metadata_sha256` lists and serialized manifest bytes (CI gate:
  `tools/check_phase4d_manifest_cross_runtime.py`).

Phase 4D implementations MUST NOT introduce provenance binding, Merkle roots,
evidence bundle mutations, or machine-contract mutations.

### D12. Phase 4E Provenance Binding and Merkle Semantics Are Frozen

Phase 4E introduces two deterministic artifact surfaces only:

- `artifacts/provenance_manifest.tp.meta.provenance.v1.json`
- `artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json`

Normative references:

- schema: `schemas/phase4/provenance_manifest.schema.json`
- schema: `schemas/phase4/provenance_merkle.schema.json`
- exit-code contract: `docs/contracts/exit_codes.md`

Phase 4E provenance entry hash semantics are fixed:

- `F = bytes.fromhex(file_sha256)` (32 bytes)
- `M = bytes.fromhex(metadata_sha256)` (32 bytes)
- `V = b"tp.meta.capture.v1"`
- `provenance_entry_sha256 = SHA256(F || M || V).hexdigest()`

No delimiters, no hex-string concatenation, no JSON serialization, no runtime
metadata, and no host/environment inputs are permitted in this preimage.

Phase 4E builder invariants are fixed:

- capture metadata and metadata manifest inputs must each validate against their
  Phase 4 schemas,
- each input must have unique `relative_path` values,
- strict mode requires each input already sorted by `relative_path`,
- path sets must match 1:1 across capture metadata and metadata manifest,
- `file_sha256` must match for each aligned path,
- `metadata_sha256` must be recomputed from canonical object bytes and match the
  metadata manifest value,
- optional fingerprint enforcement requires each capture record extractor
  fingerprint to match the current canonicalization config fingerprint.

Phase 4E manifest/merkle serialization is fixed:

- canonical JSON (`sort_keys=True`, compact separators, `ensure_ascii=False`,
  `allow_nan=False`, UTF-8),
- single trailing LF required for each artifact.

Phase 4E Merkle semantics are fixed:

- leaves are `bytes.fromhex(provenance_entry_sha256)` in sorted
  `relative_path` order,
- internal hashing, pair concatenation, and odd-leaf handling MUST reuse the
  existing Phase 3 `_merkle_root` implementation unchanged,
- empty-set policy for `tp.meta.provenance_merkle.v1` is strict non-empty
  (`leaf_count >= 1`).

Phase 4E implementations MUST NOT modify Phase 4C object format, Phase 4D
object-hash semantics, Evidence Bundle contracts, Machine contracts, or Phase 3
Merkle algorithms.

## Alternatives Considered

- Implicit preimage definition via current implementation only:
  rejected because auditors and external validators need an explicit byte-level
  contract, not inferred behavior.
- Including `*_path` and notarization metadata in the root preimage:
  rejected because relocation and re-anchoring would change the root despite
  unchanged artifact content.
- Runtime-specific projections:
  rejected because bundle roots must remain stable across supported Python
  runtimes and execution environments.

## Implementation Plan

- Keep canonical projection and serialization logic centralized in
  `tools/bundle_root_common.py`.
- Enforce optional root-field co-presence and notarization shape constraints in
  schema and verifier logic.
- Validate Phase 4 contract schemas and enforce ownership governance through
  CODEOWNERS and CI schema checks.
- Enforce Phase 4B fingerprint stability via deterministic test gates.
- Maintain CI determinism checks, including cross-runtime parity validation.
- Preserve verifier compatibility for both Phase 3.3 (no root fields) and
  Phase 3.4 (with optional root/notarization).

## Success Metrics

- Bundle-root recomputation remains bit-identical for deterministic fixtures
  under Python 3.11 and Python 3.12.
- CI determinism gate fails on projection drift, serialization drift, or root
  mismatch.
- Existing Phase 3.3 manifests continue to verify without schema or runtime
  regressions.

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
- `schemas/phase4/metadata.schema.json`
- `schemas/phase4/metadata_manifest.schema.json`
- `schemas/phase4/provenance_manifest.schema.json`
- `schemas/phase4/provenance_merkle.schema.json`
- `tools/capture_metadata_config.json`
- `tools/capture_metadata_fingerprint.py`
- `tools/extract_capture_metadata.py`
- `tools/build_metadata_manifest.py`
- `tp/phase4/hash_capture_metadata.py`
- `tests/test_capture_metadata_fingerprint.py`
- `tests/test_extract_capture_metadata.py`
- `tests/test_build_metadata_manifest.py`
- `tests/golden/phase4/config_fingerprint.txt`
- `tests/golden/phase4/expected_capture_metadata.tp.meta.capture.v1.json`
- `tests/golden/phase4/expected_metadata_manifest.tp.meta.capture_manifest.v1.json`
- `tools/bundle_root_common.py`
- `tools/compute_bundle_root.py`
- `tools/verify_evidence_bundle_manifest.py`
- `tests/test_bundle_root.py`
