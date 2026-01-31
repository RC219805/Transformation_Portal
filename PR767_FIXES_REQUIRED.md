# PR #767 Critical Correctness Fixes Required

## Summary

PR #767 has **3 correctness bugs** that will cause production issues:
1. **P1**: HashMode.IF_MANIFEST_EXISTS never stores baseline hash → stale cache hits
2. **P2**: Cached PNG depth double-normalized → flat PBR maps
3. **P2**: `generate_pbr` gate in `should_skip_v2()` → incorrect skip logic

---

## Fix 1: HashMode.IF_MANIFEST_EXISTS Baseline Hash (P1 Bug)

### Problem
Current code never persists a hash on first run, so future runs can't detect file changes.

### Location
`src/transformation_portal/lux_depth_v3/orchestrator.py`, lines 193-226

### Current Code
```python
def _compute_or_skip_hash(
    self,
    image_path: Path,
    manifest_exists: bool = False,
    saved_hash: Optional[str] = None
) -> Optional[str]:
    if self.config.hash_mode == HashMode.NEVER:
        return None

    # IF_MANIFEST_EXISTS: only compute hash if manifest exists with a saved hash
    if self.config.hash_mode == HashMode.IF_MANIFEST_EXISTS:
        if not manifest_exists or not saved_hash:
            # No manifest or no saved hash - skip hashing on first run
            return None

    # ALWAYS or IF_MANIFEST_EXISTS (with existing manifest)
    try:
        return compute_file_sha256(image_path)
    except Exception as e:
        logger.error(f"Hash computation failed for {image_path}: {e}")
        raise IOError(f"Hash computation failed: {e}") from e
```

### Fixed Code
```python
def _compute_or_skip_hash(
    self,
    image_path: Path,
    *,
    manifest_exists: bool = False,
    saved_hash: Optional[str] = None,
    for_manifest_write: bool = False
) -> Optional[str]:
    """Compute file hash respecting HashMode configuration.

    Args:
        image_path: Path to the image file
        manifest_exists: Whether a manifest exists
        saved_hash: Previously saved hash from manifest
        for_manifest_write: If True, compute hash for baseline (even on first run)

    Returns:
        SHA256 hash string, or None if hash not computed
    """
    if self.config.hash_mode == HashMode.NEVER:
        return None

    try:
        if self.config.hash_mode == HashMode.ALWAYS:
            return compute_file_sha256(image_path)

        # HashMode.IF_MANIFEST_EXISTS
        if for_manifest_write:
            # Establish baseline (even on first run) so future runs can detect changes
            return compute_file_sha256(image_path)

        # For skip checks: only compute if we have a baseline to compare against
        if manifest_exists and saved_hash:
            return compute_file_sha256(image_path)

        return None
    except Exception as e:
        logger.error(f"Hash computation failed for {image_path}: {e}")
        raise IOError(f"Hash computation failed: {e}") from e
```

### Call Site Changes

#### In `should_skip_depth()` (line ~247)
```python
# BEFORE
current_hash = self._compute_or_skip_hash(
    image_input.path,
    manifest_exists=True,
    saved_hash=saved_hash
)

# AFTER
current_hash = self._compute_or_skip_hash(
    image_input.path,
    manifest_exists=True,
    saved_hash=saved_hash,
    for_manifest_write=False  # Skip check, not manifest write
)
```

#### In `enhance_image()` manifest write section (line ~637)
```python
# BEFORE
input_sha = self._compute_or_skip_hash(
    image_input.path,
    manifest_exists=manifest_exists,
    saved_hash=saved_hash
)

# AFTER
input_sha = self._compute_or_skip_hash(
    image_input.path,
    manifest_exists=manifest_exists,
    saved_hash=saved_hash,
    for_manifest_write=True  # Always establish baseline for future comparisons
)
```

---

## Fix 2: Cached PNG Depth Double-Normalization (P2 Bug)

### Problem
If `read_depth_u16_png()` returns float32 in [0,1], dividing by 65535 again crushes range to ~[0, 0.00002], producing flat PBR maps.

### Location
`src/transformation_portal/lux_depth_v3/orchestrator.py`, `_load_cached_depth()` method (line ~730)

### Current Code
```python
# Fall back to quantized depth image
if depth_path.exists():
    try:
        from .depth_writer import read_depth_u16_png
        depth_data = read_depth_u16_png(depth_path)
        # Convert 16-bit to float [0, 1] range
        depth_data = depth_data.astype(np.float32) / 65535.0
        logger.debug(f"Loaded quantized depth from: {depth_path}")
        return depth_data
    except Exception as e:
        logger.warning(f"Failed to load depth image: {e}")
```

### Fixed Code
```python
# Fall back to quantized depth image
if depth_path.exists():
    try:
        from .depth_writer import read_depth_u16_png
        depth_data = read_depth_u16_png(depth_path)

        # Robust normalization - handle both uint16 and pre-normalized float
        depth_data = np.asarray(depth_data)
        if depth_data.dtype == np.uint16:
            # Reader returned uint16 - normalize once
            depth_data = depth_data.astype(np.float32) / 65535.0
        else:
            # Reader returned float - ensure correct range
            depth_data = depth_data.astype(np.float32, copy=False)
            # If reader returned unnormalized values, normalize
            maxv = float(np.nanmax(depth_data)) if depth_data.size else 0.0
            if maxv > 1.5:
                depth_data /= 65535.0

        logger.debug(f"Loaded quantized depth from: {depth_path}")
        return depth_data
    except Exception as e:
        logger.warning(f"Failed to load depth image: {e}")
```

---

## Fix 3: Remove generate_pbr Gate from should_skip_v2() (P2 Bug)

### Problem
`should_skip_v2()` returns `True` when `generate_pbr=False`, incorrectly conflating PBR generation (optional) with V2 enhancement subprocess (separate stage).

### Location
`src/transformation_portal/lux_depth_v3/orchestrator.py`, `should_skip_v2()` method (line ~313)

### Current Code
```python
def should_skip_v2(
    self, v2_report_path: Optional[Path], manifest_path: Path,
    image_input: ImageInput, depth_was_skipped: bool
) -> bool:
    if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
        return False

    try:
        manifest = CombinedManifest.load(manifest_path)

        # If PBR not enabled, nothing to skip (or not needed)
        pbr_enabled = getattr(self.config, 'generate_pbr', False)
        if not pbr_enabled:
            return True  # No PBR needed, so skip is allowed  # <-- WRONG

        # Config Fingerprint Check ...
```

### Fixed Code
```python
def should_skip_v2(
    self, v2_report_path: Optional[Path], manifest_path: Path,
    image_input: ImageInput, depth_was_skipped: bool
) -> bool:
    """Determine whether to skip V2 (enhancement) stage.

    V2 skip logic is independent of PBR generation (which is Stage A optional).
    This method checks V2 report existence, config fingerprint, and depth freshness.
    """
    if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
        return False

    try:
        manifest = CombinedManifest.load(manifest_path)

        # Config Fingerprint Check - use stored fingerprint directly
        if not manifest.config_fingerprint:
            logger.debug("No config fingerprint in manifest - regenerating V2")
            return False

        # Compare V2/PBR config using stored fingerprint's SHA256
        current_fp = self.compute_config_fingerprint()
        stored_fp = manifest.config_fingerprint

        if current_fp.v2_only().to_sha256() != stored_fp.v2_only().to_sha256():
            logger.info("V2/PBR config changed - regenerating")
            return False

        # Consistency Check - if depth was recomputed, V2 must also rerun
        if not depth_was_skipped:
            logger.info("Depth was regenerated - V2 must rerun")
            return False

        # V2 Metadata Check
        if not manifest.v2 or manifest.v2.status != "ok":
            return False

        # Defensive output existence check
        if self.verify_outputs:
            # Verify V2 report exists
            if v2_report_path and not v2_report_path.exists():
                logger.debug(f"V2 report missing: {v2_report_path}")
                return False

            # Verify PBR outputs if they exist in manifest
            if manifest.pbr_assets:
                for label, filepath in manifest.pbr_assets.items():
                    if isinstance(filepath, str) and filepath and label.endswith('_path'):
                        if not os.path.exists(filepath):
                            logger.debug(f"PBR output missing: {filepath}")
                            return False

        return True
    except Exception as e:
        logger.debug(f"V2 skip check failed: {e}")
        return False
```

---

## Additional Improvements (Low-Risk)

### 1. Replace silent exceptions with debug logging

**Location**: Multiple places with `except Exception: pass`

**Change**:
```python
# BEFORE
except Exception:
    pass

# AFTER
except Exception as exc:
    logger.debug(
        "Failed to load existing manifest at %s: %s. Will continue.",
        manifest_path,
        exc,
    )
```

### 2. Remove redundant depth_path.exists() check

**Location**: `should_skip_depth()` after defensive output check

The file existence is already checked at function entry and in the defensive block.

---

## Required Test Additions

Add these 3 tests to `tests/test_orchestrator_improvements.py`:

### Test 1: IF_MANIFEST_EXISTS stores baseline hash
```python
def test_if_manifest_exists_stores_baseline_hash(self, tmp_path):
    """Test that IF_MANIFEST_EXISTS writes hash on first run."""
    config = EnhanceConfig(hash_mode=HashMode.IF_MANIFEST_EXISTS)
    # ... create test image ...
    orchestrator = EnhanceOrchestrator(config, tmp_path)

    # First run
    result = orchestrator.enhance_image(image_input, input_root)

    # Load manifest and verify hash was stored
    manifest = CombinedManifest.load(manifest_path)
    assert manifest.input is not None
    assert manifest.input.image_sha256 is not None
    assert len(manifest.input.image_sha256) == 64  # SHA256 hex length
```

### Test 2: Cached depth PNG fallback doesn't double-normalize
```python
@patch('transformation_portal.lux_depth_v3.depth_writer.read_depth_u16_png')
def test_cached_depth_no_double_normalization(self, mock_read):
    """Test that cached PNG depth isn't double-normalized."""
    # Mock reader to return already-normalized float32
    mock_depth = np.random.rand(512, 512).astype(np.float32)  # [0, 1]
    mock_read.return_value = mock_depth

    orchestrator = EnhanceOrchestrator(config, output_root)
    depth = orchestrator._load_cached_depth(depth_path, float_depth_path)

    # Should preserve range, not divide by 65535 again
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0
    assert depth.max() > 0.5  # Not crushed to near-zero
```

### Test 3: V2 skip logic independent of generate_pbr
```python
def test_v2_skip_independent_of_generate_pbr(self):
    """Test that V2 skip doesn't depend on generate_pbr flag."""
    config = EnhanceConfig(generate_pbr=False)  # PBR disabled
    orchestrator = EnhanceOrchestrator(config, output_root)

    # Create valid V2 report and matching config fingerprint
    # ... setup ...

    # should_skip_v2 should evaluate based on V2 config/report, not PBR
    skip = orchestrator.should_skip_v2(
        v2_report_path, manifest_path, image_input, depth_was_skipped=True
    )

    # Should be able to skip based on V2 validity, not PBR flag
    assert skip is True  # or False based on actual V2 state
```

---

## Recommended PR Comment

```
Applied fixes for three correctness issues flagged in automated review:

1. **HashMode.IF_MANIFEST_EXISTS now stores baseline hash** on first run so future runs can detect content changes (prevents stale cache hits)

2. **Cached PNG depth fallback no longer double-normalizes** depth values, preventing flat PBR maps when reader returns pre-normalized floats

3. **Removed generate_pbr gate from should_skip_v2()** so V2 cache invalidation is based on V2 config/report rather than PBR settings (separate concerns)

Added 3 unit tests covering these edge cases.

Commits:
- fix(orchestrator): establish baseline hash for IF_MANIFEST_EXISTS mode
- fix(orchestrator): prevent double-normalization in cached depth loading
- fix(orchestrator): decouple V2 skip logic from PBR generation flag
- test(orchestrator): add tests for hash baseline, depth normalization, and V2 skip independence
```

---

## Files Modified
- `src/transformation_portal/lux_depth_v3/orchestrator.py` (~80 lines changed)
- `tests/test_orchestrator_improvements.py` (~60 lines added)
