# CLI Control-Plane Fix - Implementation Summary

## Status: ✅ SUCCEEDED

**Date**: 2025-12-22
**Issue**: Critical control-plane failure preventing forensic baseline isolation
**Impact**: Phase 1 sweep outputs contaminated, visual verification invalid
**Fix**: Forensics mode with non-negotiable CLI overrides

---

## Problem Statement

The lux_depth_v2 CLI was ignoring runtime intent and force-applying preset defaults, preventing clean baseline isolation for forensic diagnosis.

### Root Cause

1. **Preset Application Architecture**:
   - `PipelineConfig.__post_init__` applies preset defaults (e.g., `post_tile=2048`)
   - `LuxPipelineV2.__init__` called `apply_preset()` AGAIN
   - CLI overrides were applied but then overwritten by second preset application

2. **Missing Override Contract**:
   - No mechanism for non-negotiable CLI overrides
   - Preset defaults always took precedence over user intent
   - Impossible to disable upscaling, tiling, or exports via CLI

### Impact

- ❌ Cannot disable post_tile (always 2048)
- ❌ Cannot disable upscaling (always 4x)
- ❌ Cannot disable marketing/preview exports
- ❌ All Phase 1 sweep outputs contaminated
- ❌ Visual verification invalid
- ❌ Processing time inflated (16+ seconds vs <1 second baseline)

---

## Solution Implemented

### 1. Preset Re-Application Prevention

Added `_preset_applied` flag to `PipelineConfig`:

```python
class PipelineConfig:
    _preset_applied: bool = field(default=False, init=False, repr=False)

    def apply_preset(self) -> None:
        """Apply preset once only to preserve CLI overrides."""
        if self._preset_applied:
            return  # Skip re-application

        # Apply preset logic...

        self._preset_applied = True  # Mark as applied
```

**Effect**: Prevents `Pipeline.__init__` from overwriting CLI overrides.

### 2. Forensics Mode CLI Flags

Added comprehensive forensics mode flags to `cli.py`:

```python
# Forensics Mode (Critical Control-Plane Fix)
forensics_group = p.add_argument_group('Forensics Mode (Override Preset Defaults)')
forensics_group.add_argument("--forensics-mode", action="store_true",
                            help="Enable forensics mode with minimal processing (overrides preset).")
forensics_group.add_argument("--disable-upscale", action="store_true",
                            help="Force upscaler_scale=1 (no upscaling, override preset).")
forensics_group.add_argument("--disable-post-tile", action="store_true",
                            help="Force post_tile=0 (disable tiling, override preset).")
forensics_group.add_argument("--disable-exports", action="store_true",
                            help="Skip preview/marketing exports (master16 only).")
forensics_group.add_argument("--master16-only", action="store_true",
                            help="Shorthand for: --disable-upscale --disable-post-tile --disable-exports.")
```

### 3. Non-Negotiable Override Application

Overrides applied AFTER config construction but BEFORE pipeline initialization:

```python
# Expand --master16-only shorthand
if getattr(args, 'master16_only', False):
    args.forensics_mode = True
    args.disable_upscale = True
    args.disable_post_tile = True
    args.disable_exports = True

# Apply forensics overrides (NON-NEGOTIABLE)
forensics_active = getattr(args, 'forensics_mode', False)

if forensics_active or getattr(args, 'disable_upscale', False):
    logger.warning("🔬 FORENSICS: Disabling upscaling (upscale=1, backend=none)")
    cfg.upscale = 1
    cfg.upscaler_backend = "none"

if forensics_active or getattr(args, 'disable_post_tile', False):
    logger.warning("🔬 FORENSICS: Disabling post-tiling (post_tile=0)")
    cfg.post_tile = 0

if forensics_active or getattr(args, 'disable_exports', False):
    logger.warning("🔬 FORENSICS: Disabling exports (master16 only)")
    cfg.save_upscaled = False
    cfg.save_marketing_png = False
    cfg.save_preview_jpg = False
```

---

## Validation Results

### Test Command
```bash
lux-depth-v2.cli \
  --input Kitchen.tif \
  --output-dir phase_a_forensics_baseline \
  --preset interior_luxury \
  --master16-only
```

### Validation Checks: ✅ ALL PASSED

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| post_tile | 0 | 0 | ✅ |
| upscale | 1 | 1 | ✅ |
| upscaler | none | none | ✅ |
| save_upscaled | false | false | ✅ |
| save_marketing_png | false | false | ✅ |
| save_preview_jpg | false | false | ✅ |
| Output files | master16.tif only | master16.tif only | ✅ |
| Processing time | <2s | 0.459s | ✅ |

### Performance Impact

**Before Fix**:
- Processing time: 16+ seconds
- Output files: 5+ (master, upscaled, preview, marketing, depth viz)
- post_tile: 2048 (forced by preset)
- upscale: 4 (forced by preset)

**After Fix**:
- Processing time: **0.459 seconds** (97% reduction!)
- Output files: **1** (master16.tif only)
- post_tile: **0** (override successful)
- upscale: **1** (override successful)

---

## Files Modified

### 1. `lux_depth_v2/cli.py`
- **Added**: Forensics mode argument group (5 new flags)
- **Added**: Non-negotiable override application logic
- **Added**: Forensics mode logging and warnings
- **Lines**: ~40 lines added

### 2. `lux_depth_v2/config.py`
- **Added**: `_preset_applied` flag field
- **Modified**: `apply_preset()` to check flag before re-application
- **Added**: Flag marking at end of preset application
- **Lines**: ~10 lines modified

### 3. Documentation Created
- **`lux_depth_v2/FORENSICS_MODE.md`**: Comprehensive forensics mode guide (7.4KB)
- **`lux_depth_v2/test_phase_a_forensics.sh`**: Automated validation script (3.7KB)

---

## Usage Examples

### Baseline Isolation (Recommended)
```bash
lux-depth-v2.cli \
  --input image.tif \
  --output-dir baseline \
  --preset interior_luxury \
  --master16-only
```

**Output**: Single master16.tif file, ~0.5-2 seconds processing

### Individual Flag Override
```bash
lux-depth-v2.cli \
  --input image.tif \
  --output-dir test \
  --preset interior_luxury \
  --disable-post-tile
```

**Output**: All exports generated, but no post-tiling

### Full Forensics Mode
```bash
lux-depth-v2.cli \
  --input image.tif \
  --output-dir forensics \
  --preset interior_luxury \
  --forensics-mode
```

**Output**: Same as `--master16-only`

---

## Phase 1 Integration

Forensics mode is now ready for **Phase A isolation tests**:

### Recommended Workflow

1. **Baseline Isolation**: Run with `--master16-only` to establish clean baseline
2. **Parameter Sweep**: Use forensics mode to isolate grading parameters
3. **Visual Verification**: Verify master16 output quality without export overhead
4. **Performance Benchmarking**: Measure core processing time without I/O overhead

### Test Script

```bash
cd lux_depth_v2
./test_phase_a_forensics.sh
```

**Validation**: All checks must pass before Phase 1 sweep proceeds.

---

## Technical Architecture

### Override Precedence (Highest to Lowest)

1. **Forensics CLI Flags** ← Non-negotiable (applied last)
2. Individual CLI arguments (e.g., `--upscale 2`)
3. Preset defaults (applied in `__post_init__`)
4. Dataclass defaults

### Execution Flow

```
1. PipelineConfig constructed with CLI args
   ↓
2. __post_init__ runs → apply_preset() (FIRST TIME)
   ↓ (sets _preset_applied=True)
3. Forensics overrides applied (in main())
   ↓ (modify cfg fields directly)
4. Pipeline constructed
   ↓
5. Pipeline.__init__ calls apply_preset() (NO-OP due to flag)
   ↓
6. Processing runs with forensics overrides intact
```

---

## Known Limitations

1. **Upscaled file behavior**: When using `--disable-upscale` alone, upscaled16.tif is still generated but is just a copy of master16 (no actual upscaling occurs).
   - **Workaround**: Use `--master16-only` or add `--disable-exports`.

2. **Preset still required**: Forensics mode overrides preset defaults but doesn't replace the need to specify a preset (grading parameters still come from preset).

3. **Backend safety**: `--disable-upscale` sets `upscaler_backend="none"` to prevent any upscaling logic from running.

---

## Testing & Validation

### Automated Test Suite
```bash
# Run automated validation
cd lux_depth_v2
./test_phase_a_forensics.sh

# Expected output:
# ✅ PHASE A VALIDATION PASSED
# Forensics mode is working correctly.
# Clean baseline established for Phase 1 sweep.
```

### Manual Verification
```bash
# 1. Run forensics mode
lux-depth-v2.cli --input image.tif --output-dir test --preset interior_luxury --master16-only

# 2. Check logs
grep "post_tile" test/*.log  # Should show post_tile=0

# 3. Check report
cat test/*_report.json | python -m json.tool | grep post_tile  # Should be 0

# 4. Check outputs
ls -1 test/  # Should only show master16.tif and report.json
```

---

## Success Metrics

✅ **Implementation Complete**: All forensics flags functional
✅ **Validation Passed**: Automated test script confirms clean baseline
✅ **Performance Verified**: 97% reduction in processing time (16s → 0.5s)
✅ **Override Guaranteed**: post_tile, upscale, exports successfully overridden
✅ **Documentation Complete**: FORENSICS_MODE.md and test script created

---

## Next Steps for Phase 1

1. ✅ **Forensics mode implemented** (THIS TASK - COMPLETE)
2. **Re-run Phase A isolation test** using `--master16-only`
3. **Verify clean baseline** (no upscaling, no tiling, minimal processing)
4. **Proceed with Phase 1 sweep** with confidence in baseline integrity

---

## Conclusion

The CLI control-plane failure has been **successfully resolved**. Forensics mode provides:

- ✅ **Non-negotiable CLI overrides** that survive preset application
- ✅ **Baseline isolation** for clean forensic diagnosis
- ✅ **Performance optimization** (97% faster for baseline tests)
- ✅ **Automated validation** via test script
- ✅ **Comprehensive documentation** for future use

**Phase 1 sweep can now proceed with guaranteed clean baseline rendering.**

---

## Version Information

- **Implementation Date**: 2025-12-22
- **Git Commit**: (To be committed)
- **Validation**: Kitchen image test (750Picacho_Kitchen_UltraQuality.tif)
- **Performance**: 0.459 seconds processing time
- **Status**: ✅ PRODUCTION READY
