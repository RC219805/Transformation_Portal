# Phase 4 Roadmap: Spec-First, Deterministic, Versioned, Test-Complete

## Objective

Tighten Phase 4 into a deterministic, schema-governed, versioned rollout that prevents common failure modes:

- tool drift
- float drift
- timezone ambiguity
- contract drift
- backward-compatibility breakage

`schemas/phase4/metadata.schema.json` is the source of truth for capture metadata shape. Everything else is engineered around it.

## Non-Negotiable Invariants

1. Deterministic bytes for all Phase 4 artifacts given identical inputs (file bytes + relative paths + pinned toolchain).
2. No runtime timestamps (or any host-dependent fields) inside deterministic artifacts.
3. Schema-governed output: every metadata object validates against `schemas/phase4/metadata.schema.json`.
4. Toolchain determinism: extractor + runtime pinned (recommended: containerized).
5. Explicit versioning when any contract surface changes (no silent extension of v1 artifacts).
6. Stable ordering everywhere: discovery -> extraction -> serialization -> hashing -> merkle.

## 4A - Contract Freeze

### 4A.1 Land schema as authoritative contract

Deliverable:
- `schemas/phase4/metadata.schema.json` exactly as drafted (Draft 2020-12, strict required fields, `additionalProperties: false`).

Acceptance criteria:
- CI validates schema parse correctness and schema meta-validation.

### 4A.2 Add schema governance

Deliverables:
- `CODEOWNERS` entry for `schemas/phase4/metadata.schema.json`.
- ADR for capture metadata contract immutability and versioning policy.

ADR requirements:
- `tp.meta.capture.v1` is immutable once released.
- Any contract change requires explicit version bump (recommended: `tp.meta.capture.v2`) or an explicitly documented compatibility rule.

## 4B - Canonicalization Spec

### 4B.1 Create canonicalization spec doc

Deliverable:
- `docs/contracts/metadata_canonicalization.md`

Required deterministic rules:

Path normalization (`relative_path`):
- forward slashes only
- strip leading `./`
- reject `..` segments
- reject backslashes (`\`)
- treat paths as case-sensitive; duplicates after normalization are hard errors

String normalization (all schema string fields):
- Unicode NFC normalization
- trim ASCII leading/trailing whitespace
- reject embedded NUL bytes

Time policy (`capture_datetime_utc`), strict precedence:
1. If `GPSDateStamp` + `GPSTimeStamp` parse successfully, emit UTC `YYYY-MM-DDTHH:MM:SSZ`.
2. Else if `DateTimeOriginal` exists and an offset exists:
   - prefer `OffsetTimeOriginal`
   - else accept unambiguous `TimeZoneOffset`
   - convert to UTC and format with trailing `Z`
3. Else emit `null` and warning code `WARN_DATETIME_NO_TZ`.

Numeric rounding (half-even):
- `gps_latitude`, `gps_longitude`: 8 decimals
- `focal_length_mm`: 3 decimals
- `aperture_fnumber`: 3 decimals
- `shutter_speed_seconds`: 6 decimals
- `exposure_compensation_ev`: 3 decimals

Orientation mapping:
- 1 -> `Horizontal`
- 2 -> `MirrorHorizontal`
- 3 -> `Rotate180`
- 4 -> `MirrorVertical`
- 5 -> `MirrorHorizontalRotate270CW`
- 6 -> `Rotate90CW`
- 7 -> `MirrorHorizontalRotate90CW`
- 8 -> `Rotate270CW`
- missing or unparseable -> `null`

Warnings policy (`extraction_warnings`):
- warnings are stable machine codes, not free-form prose
- example codes: `WARN_DATETIME_NO_TZ`, `WARN_GPS_PARSE_FAIL`, `WARN_TAG_CONFLICT`
- warnings are unique
- warnings are lexicographically sorted

## 4C - Extractor Implementation

### 4C.1 Lock extractor engine

Recommendation:
- use `exiftool` pinned to an exact version for deterministic extraction.

Deliverables:
- `tools/extract_canonical_metadata.py`
- `tools/exiftool_taglist.txt` (explicit whitelist)
- `tools/capture_metadata_config.json` (canonical config for fingerprinting)

### 4C.2 Deterministic invocation strategy

Requirements:
1. Discover files internally (not shell glob order):
   - walk ingest root
   - extension filter from config
   - normalize to `relative_path`
   - lexicographic sort
2. Pass exact sorted file list to exiftool:
   - write sorted `filelist.txt`
   - call `exiftool -@ filelist.txt ...`
3. Parse exiftool JSON output and build schema-compliant objects in fixed key order before canonical serialization.

Typical flags:
- `-json`
- `-n`
- explicit whitelist only

### 4C.3 Extractor fingerprint

`extractor.config_fingerprint_sha256` is computed from canonical JSON config containing:
- metadata contract version
- exact exiftool args
- exact tag whitelist
- canonicalization precision and datetime rules version
- orientation mapping table

Acceptance criteria:
- fingerprint is stable across hosts when toolchain is pinned
- any change in taglist or canonicalization rules changes fingerprint

## 4D - Artifact Surfaces

### 4D.1 Deterministic metadata payload

Deliverable:
- `artifacts/capture_metadata.tp.meta.capture.v1.json`

Rules:
- JSON array sorted by `relative_path`
- every object validates against `metadata.schema.json`
- canonical serialization

### 4D.2 Deterministic metadata manifest (recommended)

Deliverable:
- `artifacts/metadata_manifest.tp.meta.capture.v1.json`

Per-file fields:
- `relative_path`
- `file_sha256`
- `metadata_sha256`

### 4D.3 Canonical JSON for hashing

Canonical encoding rules:
- UTF-8
- `sort_keys=true`
- `separators=(",", ":")`
- `ensure_ascii=false`
- `allow_nan=false`

Acceptance criteria:
- metadata object hash is computed over canonical bytes of the object only
- object hash is independent of array indentation or pretty-print style

## 4E - Provenance Binding and Merkle

### 4E.1 Define provenance contract

Deliverable:
- `tp.meta.provenance.v1`

### 4E.2 Provenance entry hash

Definition:
- `provenance_entry_sha256 = SHA256(file_sha256_bytes || metadata_sha256_bytes || UTF8(metadata_contract_version))`

Note:
- if Phase 3 uses a different concatenation convention, mirror Phase 3 exactly and lock it in ADR.

### 4E.3 Provenance manifest

Deliverable:
- `artifacts/provenance_manifest.tp.meta.provenance.v1.json`

Per-file fields:
- `relative_path`
- `file_sha256`
- `metadata_sha256`
- `provenance_entry_sha256`

Ordering:
- sorted by `relative_path`

### 4E.4 Provenance merkle root

Deliverable:
- `artifacts/provenance_merkle.tp.merkle.v1.json`

Requirements:
- reuse Phase 3 merkle implementation exactly (odd-leaf handling and concatenation rules)
- leaves are sorted `provenance_entry_sha256`

Acceptance criteria:
- root recomputation matches golden fixtures in CI
- verifier rejects deviations

## 4F - Evidence Bundle Integration

### 4F.1 Introduce evidence bundle v2

Deliverable:
- new contract version (example: `tp.meta.evidence_bundle.v2`)

Rule:
- do not silently extend strict v1 schema

### 4F.2 Bind capture provenance in bundle v2

Required bundle fields:
- `capture_metadata_sha256`
- `metadata_manifest_sha256`
- `provenance_manifest_sha256`
- `provenance_merkle_root`
- `metadata_contract_version`
- `provenance_contract_version`
- extractor identity (or referenced equivalent)

Acceptance criteria:
- verifier can validate provenance inclusion via integrity chain

## 4G - Machine Mode Integration

### 4G.1 Introduce machine-mode v2

Deliverable:
- `tp.meta.machine.v2` with `capture_provenance` block

Compatibility:
- keep v1 available during migration

Acceptance criteria:
- consumers can opt into v2 without breaking v1 tooling

## 4H - CI Determinism Gates

### 4H.1 Golden fixture suite

Deliverables:
- `tests/fixtures/raw_sample.dng` (optionally `.tif`)
- `tests/golden/phase4/expected_capture_metadata.json`
- `tests/golden/phase4/expected_hashes.json` with:
  - expected `metadata_sha256`
  - expected `provenance_entry_sha256`
  - expected `provenance_merkle_root`

Acceptance criteria:
- CI checks exact byte equality and exact hashes with zero tolerance

### 4H.2 Cross-platform determinism matrix

Recommended CI matrix:
- `ubuntu-latest`
- `macos-latest` (if practical)

### 4H.3 Containerized toolchain

Deliverable:
- `Dockerfile.tp-tools` with pinned:
  - exiftool version
  - python version
  - dependency lock

Acceptance criteria:
- CI artifact generation runs in containerized pinned environment

### 4H.4 Strict mode gating

Deliverable:
- `--strict` CLI flag

Behavior:
- strict mode fails on any warning
- default mode permits warnings while preserving deterministic output

## 4I - Operational UX

Required CLI flags:
- `--enable-capture-provenance`
- `--exiftool-path` (or container default)
- `--strict`
- `--machine-contract v1|v2`
- `--evidence-bundle v1|v2`

Acceptance criteria:
- Phase 3 behavior remains possible without forced migration
- Phase 4 outputs are reproducible and verifiable

## 4J - Definition of Done

Phase 4 is complete only when all are true:

1. `capture_metadata.tp.meta.capture.v1.json` is deterministic and schema-valid.
2. `metadata_sha256` is stable per file.
3. `provenance_manifest.tp.meta.provenance.v1.json` is deterministic.
4. `provenance_merkle_root` is stable and verified.
5. Evidence bundle v2 binds capture provenance artifacts and roots.
6. CI enforces pinned toolchain and golden fixtures.
7. Verification CLI recomputes outputs and fails fast on drift.
8. ADRs exist for canonicalization rules, merkle rules (or exact Phase 3 reuse), and versioning policy.
