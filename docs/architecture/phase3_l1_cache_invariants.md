# Phase 3 L1 Cache Invariants

**Status**: Living document (Phase 3 L1 Stabilization)
**Last Updated**: 2026-02-13
**Related**: Issue #923, PR #924, PR #926, ADR-029

## Overview

This document captures the correctness invariants, safety properties, and operational constraints of the ArtifactStore implementation as hardened during Phase 3 L1 stabilization.

**Purpose**: Preserve institutional memory beyond PR archaeology. These invariants must be maintained by future changes to the cache subsystem.

---

## Core Invariants

### 1. Content Addressing (Determinism)

**Invariant**: Same inputs + same config → same cache key → same artifact (bitwise identical)

**Enforcement**:
- Cache keys are SHA256 hashes of serialized inputs + config
- Keys are 64-character lowercase hex strings (validated by `SAFE_CACHE_KEY` regex)
- No external mutable state influences cache key computation

**Why it matters**: Cache hits must be semantically equivalent to re-execution. Violations break pipeline determinism.

---

### 2. Atomic Writes (No Partial Artifacts)

**Invariant**: Artifacts are either fully written or not present at all (no partial/corrupted states visible)

**Enforcement**:
- temp file → fsync → atomic rename pattern
- Provenance and artifact written to separate temp files, both renamed atomically
- Write failures leave no artifact (temp files cleaned up)

**Implementation** (artifact_store.py:500-545):
```python
with tempfile.NamedTemporaryFile(...) as tmp_artifact:
    np.savez_compressed(tmp_artifact, **artifact)
    tmp_artifact.flush()
    os.fsync(tmp_artifact.fileno())

tmp_artifact_path.replace(artifact_path)  # atomic rename
```

**Why it matters**: Prevents readers from loading incomplete data during concurrent writes.

---

### 3. Multi-Process Safety (Per-Key Locking)

**Invariant**: Concurrent operations on the same cache key are serialized via exclusive file locks

**Enforcement** (PR #924):
- Per-key lock files in `locks/` subdirectory
- Exclusive locks for writes (store, evict)
- Shared locks for reads (load, load_provenance)
- Lock acquisition uses `fcntl.flock` with timeout

**Implementation** (artifact_store.py:226-280):
```python
with self._acquire_lock(cache_key, exclusive=True):
    # write operations (store, evict)

with self._acquire_lock(cache_key, exclusive=False):
    # read operations (load, load_provenance)
```

**Timeout behavior**:
- Default: 30 seconds (`DEFAULT_LOCK_TIMEOUT`)
- Configurable via `lock_timeout_seconds` parameter
- Raises `CacheLockTimeout` on failure

**Why it matters**: Prevents corruption/lost-update scenarios when multiple processes access the same artifact.

---

### 4. Stats Integrity (Global Locking)

**Invariant**: Cache statistics (`stats.json`) remain consistent under concurrent access to different cache keys

**Enforcement** (PR #926):
- Global `stats.lock` serializes all stats access
- Read-modify-write pattern inside lock (reload from disk → increment → save atomically)
- Same timeout as per-key locks

**Implementation** (artifact_store.py:722-766):
```python
def _record_cache_hit(self):
    with self._acquire_stats_lock():
        # Reload from disk (get latest counts)
        current_stats = json.load(...)
        # Increment
        current_stats["hits"] += 1
        # Save atomically
        self._save_stats_atomic()
```

**Why it matters**: Per-key locks don't serialize stats access across different keys. Without global lock, concurrent updates lose increments.

---

### 5. Lock Ordering (Deadlock Prevention)

**Invariant**: If both per-key lock and stats lock are required, ALWAYS acquire per-key lock(s) first, then stats lock

**Enforcement**:
- Documented in module docstring (artifact_store.py:12-15)
- All current operations respect this order
- No operation exists that acquires stats → per-key

**Proof sketch**:
- All operations acquire per-key → stats (or just one type)
- No operation acquires stats → per-key
- Therefore: partial order respected; no AB/BA cycle possible ∎

**Why it matters**: Violating this ordering creates deadlock potential in future features (e.g., "snapshot stats + sweep keys").

---

## Security Properties

### 6. Path Traversal Prevention

**Invariant**: Cache keys cannot escape the cache directory via path traversal

**Enforcement**:
- Cache key format validated: must be 64-char hex (SHA256)
- Explicit checks for `/`, `\`, `..` characters
- Two-level directory hierarchy (prefix-based sharding)

**Implementation** (artifact_store.py:70-72):
```python
SAFE_CACHE_KEY = re.compile(r"^[a-f0-9]{64}$")

if not SAFE_CACHE_KEY.match(cache_key):
    raise ValueError(...)
if "/" in cache_key or "\\" in cache_key or ".." in cache_key:
    raise ValueError(...)
```

**Why it matters**: Prevents malicious/malformed keys from accessing arbitrary filesystem paths.

---

### 7. No Pickle Deserialization

**Invariant**: Artifacts use NumPy `.npz` format only (no Python pickle)

**Enforcement**:
- `np.savez_compressed()` for writes
- `np.load(..., allow_pickle=False)` for reads
- Explicit rejection of object arrays

**Why it matters**: Pickle deserialization is a code execution vector. NumPy `.npz` with `allow_pickle=False` is safe.

---

## Operational Constraints

### 8. POSIX Dependency

**Constraint**: Per-key locking and stats locking require `fcntl.flock` (POSIX systems only)

**Platforms**:
- ✅ Linux, macOS, BSD
- ❌ Windows (no `fcntl` support)

**Handling**:
- Import guarded: `try: import fcntl` with `_HAVE_FCNTL` flag
- Runtime check in lock methods raises `RuntimeError` on non-POSIX
- Module imports successfully on all platforms (fails at runtime only)

**Network filesystem caveats**:
- NFS: `flock` behavior depends on mount options (`nolock` breaks correctness)
- CIFS/SMB: advisory locks may not work reliably
- **Recommendation**: Use local filesystem for cache directory

---

### 9. Advisory Lock Semantics

**Constraint**: Correctness depends on all access going through `ArtifactStore` contract

**Implication**:
- Direct filesystem manipulation bypasses locks
- External processes must not modify cache files
- Cache eviction/cleanup must use `ArtifactStore.evict()`

**Why it matters**: Advisory locks don't enforce OS-level access control. Cooperation required.

---

### 10. Crash Semantics

**Behavior**: `flock` releases automatically on process exit/crash

**Implications**:
- ✅ Dead processes don't leave locks held indefinitely
- ✅ Timeout covers "wedged but alive" processes
- ⚠️ Crash during write may leave temp files (cleaned up on next `store()`)

---

## Performance Characteristics

### 11. Stats Are "Chatty But Correct"

**Current behavior**: Every cache hit/miss does:
1. Acquire global stats lock
2. Reload `stats.json` from disk
3. Increment counter
4. Write `stats.json` atomically (temp → fsync → rename)

**Cost**: ~1-5ms per hit/miss (dominated by fsync)

**Rationale**: L1 prioritizes correctness over cleverness. Stats must be accurate under multi-process concurrency.

**Future optimizations** (deferred to L2+):
- Periodic flush (buffer stats in-memory, flush every N seconds)
- Best-effort telemetry (tolerate lost increments for throughput)
- SQLite stats table (better concurrency than JSON file)

---

### 12. Lock Contention Behavior

**Per-key lock contention**:
- Multiple processes accessing **same key**: serialized by lock (bounded wait up to timeout)
- Multiple processes accessing **different keys**: no contention (true parallelism)

**Stats lock contention**:
- All cache operations contend on global `stats.lock`
- Exponential backoff minimizes CPU spin
- Timeout prevents indefinite hangs

**Operational guidance**:
- Per-key contention is rare (content-addressed keys naturally distribute)
- Stats contention scales with cache operation rate (acceptable for L1; optimize in L2 if needed)

---

## Testing Requirements

### 13. Multi-Process Test Coverage

**Required tests** (test_artifact_store_multiprocess.py):
1. Concurrent writes to same key (last-writer-wins, no corruption)
2. Reader during writer (shared/exclusive lock interaction)
3. Concurrent writes to different keys (no global bottleneck)
4. Lock timeout behavior (wedged process simulation)
5. Concurrent reads to same key (shared lock allows parallelism)
6. Eviction with concurrent access (exclusive lock blocks readers)
7. Stats integrity under concurrent different-key access (global lock correctness)

**Why it matters**: Thread-based tests don't validate `fcntl.flock` behavior. Must use `multiprocessing.Process`.

---

## Evolution Guidelines

### What Changes Are Safe

✅ **Safe changes** (preserve invariants):
- Add new cache operations that respect lock ordering
- Optimize stats persistence (as long as atomicity + consistency maintained)
- Add eviction policies (as long as `evict()` acquires exclusive lock)
- Extend provenance metadata schema (backward-compatible additions only)

❌ **Unsafe changes** (violate invariants):
- Change cache key computation without version migration
- Remove atomic write pattern (temp → fsync → rename)
- Acquire stats lock before per-key lock (deadlock hazard)
- Allow pickle deserialization
- Skip lock acquisition for "read-only" operations (breaks multi-process safety)

### Code Review Checklist

When reviewing cache subsystem changes:

1. **Lock ordering**: Does this acquire stats → per-key? (❌ deadlock hazard)
2. **Atomicity**: Does this write directly to cache files? (❌ needs temp → rename)
3. **Timeout**: Does this block indefinitely? (❌ needs timeout or non-blocking retry)
4. **Validation**: Does this trust external cache key input? (❌ needs `SAFE_CACHE_KEY` check)
5. **Stats consistency**: Does this update stats without global lock? (❌ lost-update hazard)
6. **Test coverage**: Does this add multi-process tests? (✅ required for concurrency changes)

---

## References

- **ADR-029**: Content-Addressed Artifact Store with Provenance (Phase 3 L1 Foundation)
- **Issue #923**: Phase 3 L1 Stabilization (pre-L2 hardening window)
- **PR #924**: Per-key file locking (multi-process artifact safety)
- **PR #926**: Stats hardening + multiprocess test reliability
- **Code**: `src/transformation_portal/spatial_ai/orchestration/graph/artifact_store.py`
- **Tests**: `tests/spatial_ai/orchestration/graph/test_artifact_store*.py`

---

## Appendix: Failure Modes Addressed

| Failure Mode | Root Cause | Fix (PR) | Invariant |
|--------------|------------|----------|-----------|
| Partial artifact reads | Concurrent write during read | Atomic writes (ADR-029) | #2 |
| Lost cache updates | Concurrent write to same key | Per-key exclusive locks (#924) | #3 |
| Stats corruption | Concurrent updates to stats.json | Global stats lock (#926) | #4 |
| Infinite lock waits | Wedged process holding lock | Lock timeout (#924, #926) | #3, #4 |
| Test flakiness | `Queue.empty()` race in tests | Exact-count queue reads (#926) | #13 |
| Temp file pollution | Failed atomic rename | Cleanup on exception (#926) | #2, #4 |
| Deadlock potential | Inconsistent lock ordering | Lock ordering invariant (#926) | #5 |

---

**Document Status**: Draft for post-PR-#926-merge review
**Owner**: RC219805
**Reviewers**: TBD (after merge)
