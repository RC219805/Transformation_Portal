# Archive Manifest Phase 3.3 - Evidence Bundle Canonicalization

## Status
Proposed (Phase 3.3 - Evidence Bundle Layer)

## Context

Phase 3 and later layers currently emit artifacts across integrity,
attestation, and timestamp layers:

- `hash_manifest.csv.gz` (deterministic hash manifest)
- `hash_summary.json` (deterministic hash manifest summary)
- `merkle_roots.json` (deterministic integrity)
- `merkle_roots.sig.json` (detached attestation)
- `merkle_roots.sig.tsr` or `merkle_roots.tsr` (detached timestamp)

These artifacts are independently verifiable but not yet tied together by a
single canonical manifest contract.

Phase 3.3 defines that contract.

---

# Design Principles

1. Bundle metadata is detached and additive.
2. Existing artifacts remain immutable inputs.
3. The bundle contract is explicit, versioned, and schema-validated.
4. The canonical signed object is a JSON manifest, not a container byte stream.
5. The contract must support third-party offline verification.

---

# Bundle Contract

## Canonical bundle manifest filename

```text
evidence_bundle_manifest.json
```

## Required manifest fields

- `bundle_version`
- `hash_algorithm`
- `roots_path`
- `roots_sha256`
- `hash_manifest_path`
- `hash_manifest_sha256`
- `hash_summary_path`
- `hash_summary_sha256`
- `signature_path`
- `signature_sha256`
- `timestamp_target`
- `timestamp_path`
- `timestamp_sha256`
- `global_leaf_count`
- `phase3_version`
- `phase3_1_version`
- `phase3_2_version`
- `bundle_tool_name`
- `bundle_tool_version`

## Field semantics

- `bundle_version`: version for this bundle manifest contract.
- `hash_algorithm`: digest algorithm used for manifest digests (`sha256`).
- `roots_path`: relative path of deterministic roots artifact.
- `roots_sha256`: digest of exact `merkle_roots.json` bytes.
- `hash_manifest_path`: relative path of deterministic hash manifest artifact.
- `hash_manifest_sha256`: digest of exact `hash_manifest.csv.gz` bytes.
- `hash_summary_path`: relative path of deterministic hash summary artifact.
- `hash_summary_sha256`: digest of exact `hash_summary.json` bytes.
- `signature_path`: relative path of detached signature artifact.
- `signature_sha256`: digest of exact `merkle_roots.sig.json` bytes.
- `timestamp_target`: whether timestamp was taken over `roots` or `signature`.
- `timestamp_path`: relative path of detached RFC 3161 response artifact.
- `timestamp_sha256`: digest of exact `.tsr` bytes.
- `global_leaf_count`: global leaf count from deterministic roots artifact.
- `phase3_version`, `phase3_1_version`, `phase3_2_version`: contract versions
  used to produce and validate bundle members.
- `bundle_tool_name`: identifier of bundle-manifest emitting tool.
- `bundle_tool_version`: version of bundle-manifest emitting tool.

## Recommended timestamp target

For Phase 3.3 evidence bundles, timestamping the signature artifact is
recommended (`timestamp_target = "signature"`), since identity and time are
bound in the same detached chain. Signature artifacts are mandatory in this
bundle contract.

---

# Canonical Serialization

`evidence_bundle_manifest.json` MUST be serialized with:

- UTF-8 encoding
- sorted keys
- two-space indentation
- separators fixed to `(",", ": ")`
- trailing newline

This keeps emitted manifests deterministic across hosts and Python versions.

---

# Example Manifest

```json
{
  "bundle_version": "1",
  "hash_algorithm": "sha256",
  "phase3_1_version": "1",
  "phase3_2_version": "1",
  "phase3_version": "1",
  "bundle_tool_name": "phase3_bundle_builder",
  "bundle_tool_version": "1.0.0",
  "global_leaf_count": 48291,
  "hash_manifest_path": "hash_manifest.csv.gz",
  "hash_manifest_sha256": "6cd77312a9ee3c11d73dbf17d02fd5f1e6e8cf70f92f9d0377ce4d79073e34ee",
  "hash_summary_path": "hash_summary.json",
  "hash_summary_sha256": "48ad5e285629f854ea5f93f3f0562ec3e13793a7a8ccf8a07c1f0bdd1818f26c",
  "roots_path": "merkle_roots.json",
  "roots_sha256": "2f44bcaee8cf9fc5fe91f8c9f8ce87b17cf5f6e11323191b37a89f2df5a37a99",
  "signature_path": "merkle_roots.sig.json",
  "signature_sha256": "2cc31d95be9b4ed8d8410fcf94fdc7a4a1034fd1433f20f8db1208149ca52e29",
  "timestamp_path": "merkle_roots.sig.tsr",
  "timestamp_sha256": "5b1fca7f20f6cca96c39f96d6fb2f7758867a2334df8459f39bc4826ce16d276",
  "timestamp_target": "signature"
}
```

---

# Verification Contract

Phase 3.3 bundle verification MUST:

1. Load and schema-validate `evidence_bundle_manifest.json`.
2. Recompute SHA-256 for each referenced artifact path:
   - `merkle_roots.json`
   - `hash_manifest.csv.gz`
   - `hash_summary.json`
   - `merkle_roots.sig.json`
   - detached timestamp response (`.tsr`)
3. Compare recomputed digests against manifest digest fields.
4. Verify timestamp target consistency:
   - `timestamp_target = "signature"` implies `timestamp_path = "merkle_roots.sig.tsr"`
   - `timestamp_target = "roots"` implies `timestamp_path = "merkle_roots.tsr"`
5. Verify `global_leaf_count` matches `merkle_roots.json` global leaf count.
6. Verify `bundle_tool_name` and `bundle_tool_version` are present and non-empty.
7. Fail non-zero on any mismatch.

---

# Container Policy

Bundle transport MAY use zip/tar directories for convenience.

The container bytes MUST NOT be treated as the signed object.

Only `evidence_bundle_manifest.json` is the canonical representation of bundle
membership. If signing is applied, sign the canonical manifest bytes.

---

# Optional Detached Manifest Signature

A future detached signature over `evidence_bundle_manifest.json` MAY be added as:

```text
evidence_bundle_manifest.sig.json
```

If present, it follows the same detached-envelope model used for
`merkle_roots.sig.json`.

---

# Non-Goals

Phase 3.3 does not include:

- TSA certificate chain archival
- long-term revocation evidence packaging
- multi-signature policy orchestration
- legal/compliance certification semantics

---

# Schema Reference

- `docs/archive/schemas/evidence_bundle_manifest.schema.json`
