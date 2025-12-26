# Config Audit Fixes - December 25, 2025

## Summary

Implemented two critical fixes identified in the custom agent config audit to ensure APEX preset correctness and cache invalidation reliability.

## Issues Fixed

### 1. Phantom Config Attributes (High Priority)

**Problem**: APEX preset was setting `upscale_tile_size` and `upscale_tile_overlap` as dynamic attributes on `PipelineConfig`, which:
- Looked configured but didn't drive actual behavior
- Created silent configuration that appeared to work but was ignored
- Violated the contract between config and upscaler

**Root Cause**: Lines 697-698 in `config.py` set attributes that didn't exist as dataclass fields:
```python
self.upscale_tile_size = 2048  # Phantom attribute
self.upscale_tile_overlap = 128  # Phantom attribute
```

**Fix Applied**:
1. Added `_ensure_phase2()` helper method to create `Phase2Config` if missing
2. Modified APEX preset to write to actual `Phase2Config` fields:
```python
ph2 = self._ensure_phase2()
ph2.tile_based_upscaling = True
ph2.upscale_tile_size = 2048
ph2.upscale_overlap = 128
```
3. Updated `TorchUpscaler` to read from `Phase2Config` with backward-compatible fallback

**Impact**: APEX preset now correctly configures tiled upscaling (2048px tiles, 128px overlap) for memory-efficient processing.

### 2. Incomplete Fingerprint (Cache Invalidation Risk)

**Problem**: `_cfg_fingerprint()` method omitted depth contract parameters, causing cache invalidation failures when:
- Depth model changed (`depth.auto_model`)
- Depth tiling changed (`depth.auto_tile_size`, `depth.auto_overlap`)
- Strict depth mode changed (`strict_depth`)
- Cache settings changed (`depth.enable_cache`)

**Root Cause**: Fingerprint only included segmentation and depth zones mode, not full depth configuration.

**Fix Applied**: Extended fingerprint to include all depth contract inputs:
```python
"strict_depth": self.strict_depth,
"depth_mode": self.depth.mode.value,
"depth_auto_model": self.depth.auto_model,
"depth_auto_tile_size": self.depth.auto_tile_size,
"depth_auto_overlap": self.depth.auto_overlap,
"depth_cache_enabled": self.depth.enable_cache,
```

**Impact**: Cache now correctly invalidates when depth configuration changes, preventing stale output artifacts.

## Tests Added

### Config Tests (`test_config.py`)

**TestApexPresetConfiguration** class with 4 tests:

1. **test_apex_sets_phase2_upscale_tiling**: Verifies APEX creates `Phase2Config` with correct tiling parameters
2. **test_cfg_fingerprint_includes_depth_auto_model**: Confirms fingerprint changes when depth model changes
3. **test_cfg_fingerprint_includes_depth_tiling**: Confirms fingerprint changes when tiling parameters change
4. **test_cfg_fingerprint_includes_strict_depth**: Confirms fingerprint changes when strict_depth changes

### Upscaler Tests (`test_tiled_upscaling.py`)

**test_phase2_config_integration**: Validates three scenarios:
1. APEX preset → Phase2Config with tiling enabled (2048px, 128px overlap)
2. Phase2Config with `tile_based_upscaling=False` → tiling disabled
3. Legacy fallback (no Phase2Config) → reads from direct attributes

## Verification

All tests pass:
- 59 config tests (including 4 new audit tests)
- 5 upscaling tests (including 1 new integration test)
- No regressions in existing functionality

## Files Modified

1. **lux_depth_v2/config.py**:
   - Added `_ensure_phase2()` helper method
   - Fixed APEX preset to write to Phase2Config
   - Extended `_cfg_fingerprint()` with depth contract fields

2. **lux_depth_v2/upscaling.py**:
   - Updated `TorchUpscaler.__init__()` to read from Phase2Config with backward-compatible fallback

3. **lux_depth_v2/tests/test_config.py**:
   - Added `TestApexPresetConfiguration` class with 4 audit-recommended tests

4. **lux_depth_v2/tests/test_tiled_upscaling.py**:
   - Added `test_phase2_config_integration()` for Phase2 integration validation
   - Fixed `test_blend_mask_creation()` tensor indexing bug

## Depth Contract Consistency

Audit confirmed depth contract rules are correct and internally consistent:

- **CI_BASELINE** → `DepthMode.OPTIONAL` (allows testing without depth)
- **Presets with "APEX"** → `DepthMode.REQUIRED` (fail fast without depth)
- **All others** (including production_standard, production_ultra) → `DepthMode.AUTO` (generate if missing)

Rule engine correctly enforces these contracts without scattered conditionals.

## Backward Compatibility

Both fixes maintain backward compatibility:

1. **Phase2Config**: `_ensure_phase2()` creates config on-demand; legacy code without Phase2 continues to work
2. **Upscaler**: Reads Phase2Config first, falls back to legacy attributes if Phase2Config is missing
3. **Fingerprint**: Only adds new fields; existing cache entries remain valid unless depth config actually changes

## Audit Credits

Fixes implement recommendations from custom agent consistency audit focusing on:
- Depth contract correctness
- Preset intent vs. actual behavior
- Silent configuration footguns
- Cache invalidation reliability
