# Archive Manifest Phase 3.4 - Bundle Root Anchoring and Notarization

## Status
Proposed (Phase 3.4 - Bundle Root Layer)

## Context

Phase 3.3 introduced a portable canonical bundle manifest that binds these
artifacts by SHA-256:

- `hash_manifest.csv.gz`
- `hash_summary.json`
- `merkle_roots.json`
- `merkle_roots.sig.json`
- `merkle_roots.sig.tsr` or `merkle_roots.tsr`

Phase 3.4 extends that contract with an externally anchorable
`bundle_root_sha256` and an optional notarization layer. The extension is
additive and does not weaken Phase 3.3 verification invariants.

---

# Design Principles

1. Bundle metadata remains detached and additive.
2. Existing artifacts remain immutable inputs.
3. The bundle contract stays strict and schema-validated.
4. Bundle root digest is deterministic across hosts and Python versions.
5. Notarization artifacts are optional and re-anchorable.

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
- `merkle_leaf_count`
- `phase3_version`
- `phase3_1_version`
- `phase3_2_version`
- `bundle_tool_name`
- `bundle_tool_version`

## Optional Phase 3.4 fields

- `bundle_root_algorithm` (`"sha256"`)
- `bundle_root_preimage_version` (`"1"`)
- `bundle_root_sha256` (64 lowercase hex)
- `notarization`

All three `bundle_root_*` fields MUST appear together when present.

### Optional notarization object

`notarization` is an optional object with one or both providers:

- `rfc3161`:
  - `timestamp_path`
  - `timestamp_sha256`
- `sigstore`:
  - `bundle_path`
  - `bundle_sha256`

Unknown fields are rejected.

## Field semantics

- `bundle_version`: bundle manifest contract version (`"1"`).
- `hash_algorithm`: digest algorithm for manifest digests (`"sha256"`).
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
- `merkle_leaf_count`: `merkle_roots.json.global.leaf_count`.
- `phase3_version`, `phase3_1_version`, `phase3_2_version`: non-empty
  contract version strings.
- `bundle_tool_name`: identifier of bundle-manifest emitting tool.
- `bundle_tool_version`: version of bundle-manifest emitting tool.
- `bundle_root_algorithm`: bundle root digest algorithm (`"sha256"`).
- `bundle_root_preimage_version`: canonical preimage format version (`"1"`).
- `bundle_root_sha256`: SHA-256 digest over canonical root preimage bytes.

---

# Canonical Serialization

`evidence_bundle_manifest.json` MUST be serialized with:

- UTF-8 encoding
- sorted keys
- two-space indentation
- separators fixed to `(",", ": ")`
- trailing newline

---

# Bundle Root Canonical Preimage (Normative)

## Projection fields

`bundle_root_sha256` is computed from a canonical projection using exactly these
fields:

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

Path fields (`*_path`) and notarization fields are excluded from the root
preimage.

## Canonical preimage serialization

Projection bytes (`preimage_v1`) MUST be serialized as canonical JSON with:

- UTF-8
- `sort_keys = true`
- `separators = (",", ":")`
- LF newline appended (`\n`)

## Bundle root digest

```text
bundle_root_sha256 = sha256(preimage_v1)
```

---

# Verification Contract

Bundle verification MUST:

1. Load and strict-validate `evidence_bundle_manifest.json`.
2. Recompute SHA-256 for `merkle_roots.json`, `hash_manifest.csv.gz`,
   `hash_summary.json`, `merkle_roots.sig.json`, and timestamp `.tsr` artifact.
3. Verify `timestamp_target` to `timestamp_path` coupling.
4. Verify `merkle_leaf_count` matches `merkle_roots.json.global.leaf_count`.
5. Verify `bundle_tool_name` and `bundle_tool_version` are non-empty.
6. If `bundle_root_sha256` is present, recompute canonical root and compare.
7. If `notarization` is present, recompute SHA-256 of notarization artifacts and
   compare against notarization digests.
8. Fail non-zero on any mismatch.

---

# CLI Interface

Generate canonical bundle manifest (Phase 3.3):

```bash
python tools/generate_evidence_bundle_manifest.py \
  --roots /path/to/merkle_roots.json \
  --hash-manifest /path/to/hash_manifest.csv.gz \
  --hash-summary /path/to/hash_summary.json \
  --signature /path/to/merkle_roots.sig.json \
  --timestamp-target signature \
  --timestamp /path/to/merkle_roots.sig.tsr \
  --phase3-version 1 \
  --phase3-1-version 1 \
  --phase3-2-version 1 \
  --bundle-tool-name phase3_bundle_builder \
  --bundle-tool-version 1.0.0 \
  --out /path/to/evidence_bundle_manifest.json
```

Compute bundle root only (no file mutation):

```bash
python tools/compute_bundle_root.py \
  --bundle-manifest /path/to/evidence_bundle_manifest.json
```

Compute and write root fields into manifest:

```bash
python tools/compute_bundle_root.py \
  --bundle-manifest /path/to/evidence_bundle_manifest.json \
  --write
```

Verify canonical bundle manifest:

```bash
python tools/verify_evidence_bundle_manifest.py \
  --bundle-manifest /path/to/evidence_bundle_manifest.json \
  --bundle-dir /path/to/bundle_dir
```

---

# Exit Codes

Generate CLI (`generate_evidence_bundle_manifest.py`):

- `0` = manifest generated successfully
- `10` = bundle generation failure

Compute root CLI (`compute_bundle_root.py`):

- `0` = bundle root computed/written successfully
- `21` = malformed manifest or invalid arguments
- `22` = manifest write failure
- `23` = existing `bundle_root_sha256` mismatches computed root

Verify CLI (`verify_evidence_bundle_manifest.py`):

- `0` = bundle manifest and referenced artifacts verified
- `11` = verification mismatch/failure
- `12` = malformed manifest input

---

# Container Policy

Bundle transport MAY use zip/tar directories for convenience.

Container bytes MUST NOT be treated as the signed object.

`evidence_bundle_manifest.json` remains the canonical representation of bundle
membership.

---

# Non-Goals

Phase 3.4 does not include:

- public ledger integration in core repo
- TSA certificate chain archival
- long-term revocation evidence packaging
- multi-signature policy orchestration
- legal/compliance certification semantics

---

# Schema Reference

- `docs/archive/schemas/evidence_bundle_manifest.schema.json`
