# Phase 3 L1 Cache Invariants

> **Living document** — update whenever cache behavior, locking, or storage
> layout changes. Reviewers: check the **Code Review Checklist** at the bottom
> before approving any PR that touches `ArtifactStore`.

---

## 1. Numbered Invariants

### Invariant #1 — Content Addressing

Every cache entry is keyed by a **64-character lowercase hex SHA256** of its
deterministic inputs (data fingerprints + config snapshot). The same inputs
always produce the same cache key.

*Code reference:* `SAFE_CACHE_KEY` regex in `artifact_store.py`.

### Invariant #2 — Transactional Commit (Issue #929)

Artifact + provenance commits have **all-or-nothing visibility**. The
`.committed` marker file is the single source of truth:

1. `store()` writes artifact (`.npz`) and provenance (`.json`) via atomic
   temp→fsync→rename.
2. After both renames succeed, a `.committed` marker is atomically created
   via temp→rename.
3. **Readers (`exists`, `load`, `load_provenance`) require the marker.**
   Entries without a marker are treated as non-existent.

**Failure mode (resolved):** If the process crashes between step 1/2 and
step 3, the entry is uncommitted. Readers ignore it; the **scavenger**
(`scavenge()`) cleans it up.

*Code reference:* `store()`, `exists()`, `load()`, `load_provenance()`,
`_committed_path()` in `artifact_store.py`.

### Invariant #3 — Atomic Writes

All file writes use the **temp → fsync → rename** pattern. No partial files
are ever visible to readers under the final filename.

*Code reference:* `store()`, `_save_stats_atomic()` in `artifact_store.py`.

### Invariant #4 — Lock Ordering

If both a per-key lock and the stats lock are needed in the same operation,
**always acquire the per-key lock first**, then the stats lock. Reversing
this order risks AB/BA deadlock.

```
per-key lock(s) → stats lock    ✅ correct
stats lock → per-key lock(s)    ❌ deadlock hazard
```

*Code reference:* `store()` (per-key lock wraps the entire operation;
`_record_cache_miss()` internally acquires stats lock).

### Invariant #5 — No Indefinite Waits

All lock acquisitions respect the configured `lock_timeout_seconds`
(default 30 s) and raise `CacheLockTimeout` on expiry. No unbounded
blocking anywhere in the lock path.

*Code reference:* `_acquire_lock()`, `_acquire_stats_lock()` in
`artifact_store.py`.

### Invariant #6 — Security: No Pickle Deserialization

Artifacts are loaded with `allow_pickle=False`. Object-dtype arrays are
**rejected at store time** to prevent pickle-based code execution attacks.

*Code reference:* `store()` dtype check, `load()` `allow_pickle=False`.

---

## 2. Storage Layout

```
.cache/spatial_ai/
├── artifacts/
│   ├── ab/
│   │   ├── ab3f5e...npz          # Artifact data (NumPy compressed)
│   │   ├── ab3f5e...json         # Provenance metadata (JSON)
│   │   └── ab3f5e...committed    # Commit marker (Issue #929)
│   └── ...
├── locks/
│   ├── ab3f5e...lock             # Per-key lock file
│   └── ...
├── stats.lock                    # Global stats lock
└── stats.json                    # Cache statistics
```

### File lifecycle

| Step | File created | Visible to readers? |
|------|-------------|-------------------|
| 1 | `<key>.npz` (via atomic rename) | No — marker absent |
| 2 | `<key>.json` (via atomic rename) | No — marker absent |
| 3 | `<key>.committed` (via atomic rename) | **Yes** — entry is committed |

Eviction reverses the order: marker is removed **first**, then artifact and
provenance.

---

## 3. Scavenger (Orphan Cleanup)

`ArtifactStore.scavenge(max_temp_age_seconds=300.0)` performs best-effort
cleanup:

| What | Condition | Action |
|------|-----------|--------|
| Orphaned `.npz` | No matching `.committed` | Removed |
| Orphaned `.json` | No matching `.committed` | Removed |
| Stale temp files | `tmp*` prefix, older than threshold | Removed |

The scavenger is **safe to run concurrently** with `store()`/`load()`
because `store()` holds an exclusive per-key lock while writing, and the
scavenger only targets entries without a `.committed` marker.

*Code reference:* `scavenge()` in `artifact_store.py`.

---

## 4. Failure Modes

| Failure | Effect | Mitigation |
|---------|--------|------------|
| Crash after artifact rename, before marker | Uncommitted entry on disk | Readers skip it; scavenger cleans it up |
| Crash during temp file write | Temp file on disk | Scavenger removes stale temps (> threshold age) |
| Disk full during marker creation | Artifact + provenance written, no marker | Same as crash — readers skip, scavenger cleans |
| Lock holder crash (stale lock) | Other processes time out | `CacheLockTimeout` raised after configured timeout |
| Corrupted `.npz` | `load()` raises `ValueError` | Caller handles; `evict()` + re-compute if needed |

---

## 5. Evolution Guidelines

- **Adding new file types to a cache entry:** extend the commit marker
  pattern — all files must be written before the marker is created.
- **Changing storage layout:** provide a migration path or versioned layout
  (e.g., a layout version file in the cache root).
- **Adding new lock scopes:** document the ordering relative to existing
  locks to prevent deadlock.

---

## 6. Code Review Checklist

Before approving any PR that touches `ArtifactStore`:

- [ ] Lock ordering preserved (per-key → stats, never reversed)?
- [ ] All lock acquisitions respect timeout (no indefinite waits)?
- [ ] Atomic write pattern used for all new file outputs?
- [ ] Commit marker created **after** all files are in place?
- [ ] Readers check for `.committed` marker before trusting entry?
- [ ] `evict()` removes marker **before** artifact/provenance?
- [ ] Object-dtype arrays still rejected at store time?
- [ ] Tests cover the new/changed behavior (including fault injection)?
- [ ] Scavenger updated if new file types are added?
