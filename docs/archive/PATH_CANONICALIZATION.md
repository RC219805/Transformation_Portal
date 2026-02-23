# Path Canonicalization in archive_manifest_reports.py

## Overview

The `archive_manifest_reports.py` tool implements deterministic path canonicalization to ensure consistent grouping of archive assets across different mounting contexts and operating systems.

## Canonicalization Steps

The tool applies the following transformations in order:

1. **Backslash → Forward Slash Normalization**
   - `\` → `/`
   - Ensures cross-platform compatibility

2. **Repeated Slash Collapse**
   - `/+` → `/` (regex)
   - Normalizes double-slashes from network paths

3. **Trailing Slash Removal**
   - `.str.rstrip("/")`
   - Removes directory markers

4. **Root Marker Split**
   - Split on `root_marker` (default: `"All Archive/"`)
   - Uses first occurrence only (n=1)
   - Falls back to full path if marker not found

5. **Leading Slash Removal** ✨
   - `.str.lstrip("/")`
   - **CRITICAL**: Prevents empty `origin_drive` from absolute paths
   - Stabilizes cross-environment behavior

6. **Empty Path Placeholder**
   - Replace `""` with `"."` after lstrip
   - Prevents downstream empty-string issues

## Edge Cases Handled

### 1. Absolute Paths Without Root Marker

**Problem:**
```python
SourceFile = "/vault/All Archive/DriveA/Part1/file.CR2"
root_marker = "NONEXISTENT/"  # Won't match
```

**Before fix:**
```python
relpath = "/vault/All Archive/DriveA/Part1/file.CR2"  # Still has leading /
dir_rel = "/vault/All Archive/DriveA/Part1"
parts = dir_rel.split("/", n=2)
origin_drive = ""  # ❌ Empty! (first element before /)
partition = "vault"  # ❌ Wrong! (should be meaningful)
```

**After fix (with lstrip):**
```python
relpath = "vault/All Archive/DriveA/Part1/file.CR2"  # Leading / removed
dir_rel = "vault/All Archive/DriveA/Part1"
parts = dir_rel.split("/", n=2)
origin_drive = "vault"  # ✅ Stable
partition = "All Archive"  # ✅ Deterministic
```

**Impact:**
- Prevents incorrect custody provenance
- Prevents hotspot pollution into fake partitions
- Ensures cross-run consistency regardless of mount structure

### 2. UNC Paths (Windows Network Shares)

**Input:**
```python
SourceFile = r"\\fileserver\archive_vault\DriveA\Part1\IMG_0001.CR2"
```

**Normalization sequence:**
```python
# Step 1: Backslash → forward slash
"//fileserver/archive_vault/DriveA/Part1/IMG_0001.CR2"

# Step 2: Collapse repeated slashes
"/fileserver/archive_vault/DriveA/Part1/IMG_0001.CR2"

# Step 3: Strip trailing slashes (none in this case)
"/fileserver/archive_vault/DriveA/Part1/IMG_0001.CR2"

# Step 4: Root marker split (assume no match, fallback to full path)
relpath = "/fileserver/archive_vault/DriveA/Part1/IMG_0001.CR2"

# Step 5: Strip leading slashes ✨
relpath = "fileserver/archive_vault/DriveA/Part1/IMG_0001.CR2"
```

**Result:**
```python
origin_drive = "fileserver"  # ✅ Stable
partition = "archive_vault"  # ✅ Deterministic
```

### 3. Root Slash Only

**Input:**
```python
SourceFile = "/"
```

**Normalization:**
```python
# After lstrip("/")
relpath = ""  # Empty!

# Placeholder replacement
relpath = "."  # ✅ Prevents empty string
```

**Result:**
```python
dir_rel = ""  # No directory component
origin_drive = ""  # Expected (no drive info)
partition = ""  # Expected (no partition info)
```

### 4. Multiple Occurrences of Root Marker

**Input:**
```python
SourceFile = "/vault/All Archive/DriveA/All Archive/nested/file.CR2"
root_marker = "All Archive/"
```

**Behavior:**
```python
# Split uses n=1, so only FIRST occurrence is used
rel = sf.str.split("All Archive/", n=1, expand=True)
# Result: ["...", "DriveA/All Archive/nested/file.CR2"]
```

**Result:**
```python
relpath = "DriveA/All Archive/nested/file.CR2"  # Second "All Archive/" preserved
origin_drive = "DriveA"  # ✅ Correct
partition = "All Archive"  # ✅ Deterministic
```

### 5. Case Sensitivity

**Policy Decision:** Root marker split is **case-sensitive**.

**Rationale:**
- Archive governance requires explicit matching
- Case-insensitive normalization would lose fidelity
- Warning emitted if marker coverage < 50%

**Alternative Paths:**
- If ExifTool output varies in casing, you'll get fallback behavior (full path)
- Operators should ensure consistent casing in their ExifTool invocations
- For case-insensitive filesystems (macOS), this is a governance concern, not a code concern

## Integrity Contract

These canonicalization rules are part of the **6-point integrity contract** documented in the tool's docstring:

### Path Canonicalization (Point 1)
- ✅ Backslash → forward slash normalization
- ✅ Repeated slash collapse
- ✅ Trailing slash removal
- ✅ Leading slash removal (prevents empty origin_drive)
- ✅ Empty relpath placeholder
- ✅ `origin_drive` and `partition` derived from `dir_rel` (not `relpath`)

## Testing

The following test cases validate edge-case behavior:

1. **`test_absolute_paths_without_root_marker_get_leading_slash_stripped`**
   - Verifies absolute paths → stable origin_drive
   - Tests `/vault/...` and `/Volumes/...` patterns

2. **`test_unc_path_normalization_produces_stable_origin_drive`**
   - Verifies UNC paths → deterministic parsing
   - Tests `\\server\share\...` patterns

3. **`test_edge_case_root_slash_only_gets_placeholder`**
   - Verifies `"/"` → `"."` placeholder behavior

4. **`test_multiple_occurrences_of_root_marker_uses_first_only`**
   - Verifies n=1 split behavior with repeated markers

## Cross-Environment Behavior

### Environment-Derived Drive Names

If `root_marker` doesn't match, you get mount-structure-derived drive names:

| Mount Context | `origin_drive` | `partition` |
|---------------|----------------|-------------|
| `/vault/DriveA/...` | `vault` | `DriveA` |
| `/Volumes/RAID/DriveA/...` | `Volumes` | `RAID` |
| `/mnt/archive/DriveA/...` | `mnt` | `archive` |

**Implication:** Reports are **system-context dependent** without a matching `root_marker`.

**Mitigation:**
- Ensure `root_marker` matches ExifTool output context
- Monitor marker coverage warning (emitted if < 50%)
- Consider `--strict-root-marker` mode in future (fail if coverage too low)

## Future Enhancements

Potential improvements (not currently implemented):

1. **Strict Root Marker Mode**
   - `--strict-root-marker` flag
   - Exit non-zero if marker coverage < threshold
   - Fail-closed behavior for contract-grade guarantees

2. **Case-Insensitive Root Marker**
   - Optional `.str.lower()` normalization
   - Trade-off: loses original case fidelity
   - Policy-dependent, not currently implemented

3. **UNC Prefix Detection**
   - Optionally detect and preserve UNC semantics
   - Prefix UNC-derived paths with marker (e.g., `__UNC__/server/share/...`)
   - Out of scope for Phase 2

## References

- **Comment ID:** 3946806396 (PR #1000)
- **Author:** @RC219805
- **Implementation:** Lines 332-339 in `tools/archive_manifest_reports.py`
- **Tests:** Lines 353-end in `tests/test_archive_manifest_reports.py`
