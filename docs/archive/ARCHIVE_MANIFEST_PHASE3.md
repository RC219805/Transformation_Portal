# Archive Manifest Phase 3: Hash-First Integrity

Phase 3 extends the deterministic, schema-governed Phase 2 manifest surface with content hashing,
verification, and Merkle aggregation.

## Scope

Phase 3 outputs:

- `hash_manifest.csv.gz`
- `hash_summary.json`
- `merkle_roots.json`

Phase 3 input:

- Phase 2 `archive_index_normalized.csv.gz` (or equivalent CSV with required identity columns).

## Integrity Contract

### Identity key (stable across host mounts)

Each row is identified by:

1. `origin_drive`
2. `partition`
3. `relpath`

Absolute host paths MUST NOT be used as identity keys.

### What is hashed

- Hash input is raw file bytes from the resolved file at `archive_root / relpath`.
- Metadata is not hashed.
- `filesize_bytes` is measured from `stat().st_size` of the regular file bytes.

### Symlink policy

- Symlinks are not followed and are emitted as:
  - `hash_status = "skipped"`
  - `sha256 = ""`
  - stable `error` reason (`"symlink_skipped"`)

### Missing/unreadable policy

- Missing path: `hash_status = "missing"`, `sha256 = ""`.
- Unreadable file: `hash_status = "unreadable"`, `sha256 = ""`.
- Invalid path materialization (for example parent traversal): `hash_status = "skipped"`.
- Non-strict mode emits rows for all files.
- `--strict` exits non-zero if any row is not `ok`.
- `--strict-identity` exits non-zero if duplicate `(origin_drive, partition, relpath)` keys are present.

### Hash algorithm

- Default and required algorithm in Phase 3: `sha256`.
- Algorithm name is explicit in JSON outputs.
- `hash_manifest.csv.gz` embeds a deterministic metadata preamble line:
  - `# hash_algorithm=sha256`

### Deterministic ordering

- Canonical row order: `(origin_drive, partition, relpath)` lexical ascending.
- CSV columns are fixed order and schema-locked.
- JSON is serialized with sorted keys.
- `hash_manifest.csv.gz` is written with deterministic gzip metadata (`mtime=0`, empty filename).

## Output Contracts

### `hash_manifest.csv.gz`

#### Manifest Metadata Preamble

- `hash_manifest.csv.gz` MAY contain one or more leading metadata lines.
- Metadata lines MUST:
  - begin with `#`
  - use ASCII `key=value` format
  - precede the CSV header row
- Parsers MUST ignore all leading lines beginning with `#`.

Column order is fixed and schema-governed:

1. `origin_drive`
2. `partition`
3. `relpath`
4. `filesize_bytes`
5. `sha256`
6. `hash_status`
7. `error`

`hash_status` domain:

- `ok`
- `missing`
- `unreadable`
- `skipped`

### `hash_summary.json`

Contains:

- `hash_algorithm`
- `hash_manifest_schema_version`
- `rows_total`
- status counts: `hashed_ok`, `missing`, `unreadable`, `skipped`
- `total_bytes_hashed`

### `merkle_roots.json`

Contains deterministic Merkle roots at:

- partition granularity: `(origin_drive, partition)`
- global scope

Includes method metadata:

- `leaf_format_version`
- `tree_method_version`
- ordering rules

## Merkle Canonicalization

Leaf preimage is UTF-8 bytes:

`origin_drive + "\\0" + partition + "\\0" + relpath + "\\0" + hash_status + "\\0" + sha256_hex`

Leaf hash:

- `leaf_hash = sha256(leaf_preimage)`

Tree method:

- binary tree over sorted leaf hashes
- odd node handling: duplicate last leaf at each layer
- partition and global roots are lower-case hex digests

## Verification Contract

Verification compares archive bytes against `hash_manifest.csv.gz`:

- `--verify-all`: rehash all rows
- `--verify-sample N`: deterministic sample of first `N` canonical rows

Verification report (`verification_report.json`) contains deterministic mismatch details and exits non-zero
on any mismatch.

## CLI Usage

Generate Phase 3 outputs from Phase 2 index:

```bash
python tools/archive_hash_manifest.py \
  --archive-index /path/to/archive_index_normalized.csv.gz \
  --archive-root /path/to/All\ Archive \
  --out-dir /path/to/out_phase3 \
  --workers 4 \
  --strict \
  --validate-schemas
```

Verify archive state against manifest:

```bash
python tools/verify_hash_manifest.py \
  --hash-manifest /path/to/out_phase3/hash_manifest.csv.gz \
  --archive-root /path/to/All\ Archive \
  --verify-all \
  --report-path /path/to/out_phase3/verification_report.json
```

Fast deterministic spot-check:

```bash
python tools/verify_hash_manifest.py \
  --hash-manifest /path/to/out_phase3/hash_manifest.csv.gz \
  --archive-root /path/to/All\ Archive \
  --verify-sample 250
```

## Schema References

- `docs/archive/schemas/hash_manifest.schema.json`
- `docs/archive/schemas/hash_summary.schema.json`
- `docs/archive/schemas/merkle_roots.schema.json`
