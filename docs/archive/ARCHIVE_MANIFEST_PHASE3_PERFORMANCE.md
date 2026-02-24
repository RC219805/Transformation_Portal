# Archive Manifest Phase 3 Performance Notes

Phase 3 hashing is designed for deterministic output first, with bounded throughput telemetry from the CLI.

## Runtime telemetry

`tools/archive_hash_manifest.py` prints:

- elapsed wall time
- MiB/s hashed (only `hash_status=ok` bytes)
- files per second (`hash_status=ok` rows)

## Scaling guidance

- Small to medium archives: run single-worker (`--workers 1`) for simplest execution.
- Larger archives: use `--workers N` with deterministic ordering preserved by canonical post-sort.
- Strict integrity gate: add `--strict` to fail non-zero if any file is missing/unreadable/skipped.

## Determinism guarantees

- Canonical row order: `(origin_drive, partition, relpath)`
- Deterministic `.csv.gz` header (`mtime=0`, no original filename)
- Deterministic JSON key ordering (`sort_keys=True`)

## Resume behavior

Current implementation is full-pass (no resume checkpoints). For very large archives, run per partition and merge by
canonical key order if operationally required.
