# HashMode Quick Reference Guide

## Overview

The `HashMode` enum controls when input file hashes are computed for manifest integrity verification in the lux_depth_v3 enhancement pipeline.

## Available Modes

### ALWAYS (Maximum Security)
```python
from lux_depth_v3.enhance import EnhanceConfig, HashMode

config = EnhanceConfig(hash_mode=HashMode.ALWAYS)
```

**Behavior**:
- Computes SHA256 hash for every input image
- Stored in manifest for verification
- Validates input integrity on every run

**Use When**:
- Production environments
- Untrusted input sources
- Compliance/audit requirements
- Maximum data integrity needed

**Performance**: ~10-50ms per image overhead

---

### IF_MANIFEST_EXISTS (Smart Resume) **[DEFAULT]**
```python
config = EnhanceConfig(hash_mode=HashMode.IF_MANIFEST_EXISTS)
# or
config = EnhanceConfig()  # Uses default
```

**Behavior**:
- Computes hash only when manifest exists (resume scenarios)
- Skips hashing for new images
- Validates cached outputs against original inputs

**Use When**:
- Resuming interrupted batches
- Incremental processing
- Balancing security and performance

**Performance**: Hash computed only for cache validation

---

### NEVER (Performance Mode) ⚠️
```python
config = EnhanceConfig(hash_mode=HashMode.NEVER)
```

**⚠️ Security Warning**: Runtime warning displayed:
```
WARNING: Hash mode set to 'never' - manifests will not include input file hashes.
This provides no protection against input tampering and prevents cache validation.
Use only in trusted environments where performance is critical.
```

**Behavior**:
- No hash computation
- `InputMetadata.image_sha256` will be `None`
- No input integrity verification

**Use When**:
- Trusted environment (local machine, controlled inputs)
- Performance is critical
- Processing thousands of images
- YOU ACCEPT THE RISK ⚠️

**Performance**: No hash overhead

---

## Error Handling

### Fail-Fast Behavior

When hash computation is **required** but **fails**, the system fails immediately:

```python
# If hash_mode=ALWAYS and file is unreadable:
IOError: Hash computation failed for /path/to/image.jpg (mode=always).
Cannot create verifiable manifest. Error: [Errno 13] Permission denied
```

This prevents creation of unverifiable manifests.

### When Hashes Are Optional

When `hash_mode=NEVER`, hash computation is **skipped** (not failed):

```python
# hash_mode=NEVER → input_sha256 is None (expected)
manifest.input.image_sha256  # None (valid)
```

---

## Cache Validation Logic

### Resume with Existing Manifest

```python
# Scenario: Re-running enhancement on same inputs

# hash_mode=ALWAYS:
# ✓ Computes current hash
# ✓ Compares with manifest hash
# ✓ Skips if match, regenerates if mismatch

# hash_mode=IF_MANIFEST_EXISTS:
# ✓ Computes current hash (manifest exists)
# ✓ Compares with manifest hash
# ✓ Skips if match, regenerates if mismatch

# hash_mode=NEVER:
# ✓ Skips hash validation
# ✓ Assumes inputs unchanged (risky!)
# ✓ Reuses cached outputs
```

### Mode Switching Scenarios

#### From NEVER → ALWAYS (Security Upgrade)
```python
# Old run: hash_mode=NEVER (manifest has no hash)
# New run: hash_mode=ALWAYS

# Behavior: Forces regeneration
# Reason: Cannot verify if input changed
```

#### From ALWAYS → NEVER (Performance Downgrade)
```python
# Old run: hash_mode=ALWAYS (manifest has hash)
# New run: hash_mode=NEVER

# Behavior: Skips hash validation
# Reason: Performance mode explicitly requested
```

---

## Code Examples

### Basic Usage

```python
from pathlib import Path
from lux_depth_v3.enhance import EnhanceOrchestrator, EnhanceConfig, HashMode

# Production setup (maximum security)
config = EnhanceConfig(
    hash_mode=HashMode.ALWAYS,
    model_variant=ModelVariant.METRIC_LARGE,
    v2_preset="production_ultra"
)

orchestrator = EnhanceOrchestrator(
    config=config,
    output_root=Path("output")
)

# Process batch
results = orchestrator.enhance_batch(
    input_dir=Path("renders"),
    image_extensions=[".jpg", ".png"]
)
```

### Performance-Critical Scenario

```python
# Large batch, trusted inputs
config = EnhanceConfig(
    hash_mode=HashMode.NEVER,  # ⚠️ Warning displayed
    force_depth=False,  # Allow resume
    force_v2=False
)

# Warning appears in logs:
# WARNING: Hash mode set to 'never' - manifests will not include input file hashes...
```

### Smart Resume (Default)

```python
# Incremental processing with validation
config = EnhanceConfig()  # hash_mode=IF_MANIFEST_EXISTS by default

orchestrator = EnhanceOrchestrator(config, output_root=Path("output"))

# First run: No manifests exist → no hashing overhead
results1 = orchestrator.enhance_batch(Path("batch1"))

# Second run (resume): Manifests exist → hashes computed for validation
results2 = orchestrator.enhance_batch(Path("batch1"))  # Validates cache
```

---

## Manifest Schema Impact

### With Hashing (ALWAYS or IF_MANIFEST_EXISTS)

```json
{
  "schema": "lux-depth-v3.enhance.v1",
  "input": {
    "image_path": "/path/to/image.jpg",
    "image_sha256": "a7b2c3d4e5f6...",  // ✓ Hash present
    "exif_normalized": true,  // ✓ Always true (file is always normalized)
    "normalized_path": "/path/to/tmp_inputs/image_normalized.png"  // ✓ Always set
  },
  ...
}
```

### Without Hashing (NEVER)

```json
{
  "schema": "lux-depth-v3.enhance.v1",
  "input": {
    "image_path": "/path/to/image.jpg",
    "image_sha256": null,  // ⚠️ No hash
    "exif_normalized": true,  // ✓ Always true (file is always normalized)
    "normalized_path": "/path/to/tmp_inputs/image_normalized.png"  // ✓ Always set
  },
  ...
}
```

---

## Best Practices

### ✅ DO

- Use `ALWAYS` in production environments
- Use `IF_MANIFEST_EXISTS` for development/testing
- Use `NEVER` only in trusted environments with performance requirements
- Monitor warnings in logs
- Document security trade-offs in deployment docs

### ❌ DON'T

- Use `NEVER` with untrusted inputs
- Ignore runtime warnings
- Mix modes in production (confusing security posture)
- Assume `NEVER` provides any integrity verification

---

## Performance Benchmarks

### Hash Computation Overhead

| File Size | Hash Time (SHA256) |
|-----------|-------------------|
| 1 MB      | ~5-10 ms         |
| 5 MB      | ~25-40 ms        |
| 10 MB     | ~50-80 ms        |
| 50 MB     | ~250-400 ms      |

### Batch Processing Impact

| Batch Size | ALWAYS   | IF_MANIFEST_EXISTS | NEVER   |
|-----------|----------|-------------------|---------|
| 100 imgs  | +2-5s    | +1-2s (resume)    | +0s     |
| 1000 imgs | +20-50s  | +10-20s (resume)  | +0s     |

*Estimates assume 5MB average file size*

---

## Troubleshooting

### "Hash computation failed" Error

```
IOError: Hash computation failed for /path/to/image.jpg (mode=always).
Cannot create verifiable manifest. Error: [Errno 2] No such file or directory
```

**Cause**: Image was deleted/moved between discovery and processing

**Solution**:
1. Check if file exists
2. Verify file permissions
3. Ensure file isn't being modified concurrently

### "Old manifest lacks hash" Warning

```
WARNING: Old manifest lacks hash and hash_mode=ALWAYS - regenerating for security
```

**Cause**: Switching from `NEVER` to `ALWAYS` mode

**Solution**: Expected behavior. System regenerates to establish secure baseline.

### Warning Spam in Logs

```
WARNING: Hash mode set to 'never' - manifests will not include input file hashes...
```

**Cause**: Using `hash_mode=NEVER`

**Solution**:
- Accept the warning (security reminder)
- Or switch to `IF_MANIFEST_EXISTS` if warnings are problematic

---

## Migration from Legacy Code

### If You're Not Using hash_mode

**Before** (implicit ALWAYS behavior):
```python
config = EnhanceConfig()
# Hash was always computed
```

**After** (explicit default):
```python
config = EnhanceConfig()
# hash_mode=IF_MANIFEST_EXISTS (default)
# Smart resume: hashes only when needed
```

**Action Required**: None. Default is sensible for most use cases.

### If You Want Old Behavior (Always Hash)

```python
config = EnhanceConfig(hash_mode=HashMode.ALWAYS)
```

---

## Related Documentation

- **Security**: `lux_depth_v3/enhance/security.py` (HashMode enum definition)
- **Implementation**: `lux_depth_v3/enhance/orchestrator.py` (_compute_or_skip_hash)
- **Manifest**: `lux_depth_v3/enhance/manifest.py` (InputMetadata schema)
- **Full Summary**: `PR_651_FIXES_SUMMARY.md`

---

## Questions?

- Check inline code comments for implementation details
- Review test cases in `lux_depth_v3/tests/` for usage examples
- See `HARDENING_ROADMAP.md` for future enhancements

**Last Updated**: 2024-12-19 (PR #651 Fixes)
