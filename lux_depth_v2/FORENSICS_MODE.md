# Forensics Mode - CLI Control-Plane Fix

## Critical Issue Resolved

**Problem**: CLI was ignoring runtime intent and force-applying preset defaults, preventing clean baseline forensic diagnosis.

**Root Cause**:
1. Preset application in `PipelineConfig.__post_init__` sets defaults (e.g., `post_tile=2048`)
2. Pipeline's `__init__` called `apply_preset()` AGAIN, overwriting CLI overrides
3. No mechanism for non-negotiable CLI overrides

**Solution**:
- Added `_preset_applied` flag to prevent re-application
- Added forensics mode CLI flags with guaranteed override capability
- Overrides applied AFTER preset load and preserved through pipeline construction

## Forensics Mode Flags

### `--master16-only` (Recommended)
Shorthand for complete baseline isolation. Equivalent to:
```bash
--disable-upscale --disable-post-tile --disable-exports
```

**Usage**:
```bash
lux-depth-v2.cli \
  --input image.tif \
  --output-dir forensics_baseline \
  --preset interior_luxury \
  --master16-only
```

**Output**: Single `master16.tif` file only (no preview, marketing, or upscaled variants)

**Processing Time**: ~0.5-2 seconds (vs 16+ seconds with full pipeline)

### Individual Flags

#### `--forensics-mode`
Enables forensics logging and minimal processing mode. Sets all three disable flags.

#### `--disable-upscale`
Forces `upscale=1` and `upscaler_backend="none"`.
- **Effect**: No upscaling, master resolution = input resolution
- **Override**: Preset default (usually `upscale=4`)

#### `--disable-post-tile`
Forces `post_tile=0`.
- **Effect**: Disables memory-safe tiling for post-processing
- **Override**: Preset default (usually `post_tile=2048` for UHR support)
- **Use When**: Need untiled processing for diagnostic accuracy

#### `--disable-exports`
Forces `save_upscaled=False`, `save_marketing_png=False`, `save_preview_jpg=False`.
- **Effect**: Only master16.tif is generated
- **Override**: Preset defaults (usually all enabled)

## Validation Test Cases

### Test 1: Baseline Isolation (Kitchen)
```bash
python -m lux_depth_v2.cli \
  --input projects/750_picacho_lane/Kitchen_Only_Test/750Picacho_Kitchen_UltraQuality.tif \
  --output-dir test_forensics_baseline \
  --preset interior_luxury \
  --master16-only
```

**Expected Results**:
- ✅ CLI logs show forensics warnings
- ✅ Pipeline init shows `post_tile=0`
- ✅ Only `master16.tif` generated (no preview, marketing, upscaled)
- ✅ Processing time: <2 seconds
- ✅ report.json shows `post_tile: 0`, `upscale: 1`, `upscaler: "none"`

### Test 2: Individual Flag Override
```bash
python -m lux_depth_v2.cli \
  --input image.tif \
  --output-dir test_individual \
  --preset interior_luxury \
  --disable-post-tile
```

**Expected Results**:
- ✅ `post_tile=0` in pipeline and report
- ✅ All exports still generated (upscaled, preview, marketing)
- ✅ Upscaling still active (default scale=4)

### Test 3: Forensics Mode (Full)
```bash
python -m lux_depth_v2.cli \
  --input image.tif \
  --output-dir test_forensics_full \
  --preset interior_luxury \
  --forensics-mode
```

**Expected Results**:
- ✅ All forensics warnings logged
- ✅ Minimal processing (no upscale, no tiling, no exports)
- ✅ Same behavior as `--master16-only`

## Override Precedence

**Guaranteed Order** (highest to lowest priority):
1. **Forensics CLI Flags** (non-negotiable, applied last)
2. Individual CLI arguments (e.g., `--upscale 2`)
3. Preset defaults (applied in `__post_init__`)
4. Dataclass defaults

## Implementation Details

### Config Architecture
```python
# 1. Config constructed with CLI args
cfg = PipelineConfig(
    preset=Preset.INTERIOR_LUXURY,
    upscale=4,  # From CLI or default
    ...
)

# 2. __post_init__ runs, applies preset (first time only)
# Sets: post_tile=2048, upscale=4, material_strength=0.9, etc.
# Marks: _preset_applied=True

# 3. Forensics overrides applied (after config construction)
if args.master16_only:
    cfg.upscale = 1
    cfg.upscaler_backend = "none"
    cfg.post_tile = 0
    cfg.save_upscaled = False
    cfg.save_marketing_png = False
    cfg.save_preview_jpg = False

# 4. Pipeline constructed
pipe = LuxPipelineV2(cfg)
# Pipeline.__init__ calls cfg.apply_preset() but it's now a no-op
# due to _preset_applied=True flag, preserving forensics overrides
```

### Prevention Mechanism
```python
def apply_preset(self) -> None:
    """Apply preset configuration.

    Only applies once to preserve forensics mode overrides.
    """
    if self._preset_applied:
        return  # Skip re-application

    # Apply preset logic...

    self._preset_applied = True  # Mark as applied
```

## Phase 1 Sweep Integration

Use forensics mode for **Phase A isolation tests**:

```bash
# Clean baseline - Kitchen only
lux-depth-v2.cli \
  --input Kitchen.tif \
  --output-dir phase_a_baseline \
  --preset interior_luxury \
  --master16-only

# Verify isolation:
# - Check logs: post_tile=0, upscale=1
# - Check outputs: Only master16.tif
# - Check timing: <2 seconds
# - Check report: Confirm minimal processing
```

**Use forensics mode when:**
- Establishing clean baselines
- Isolating pipeline stages for debugging
- Measuring core processing performance
- Validating depth/material response without overhead
- A/B testing grading parameters without export overhead

**Do NOT use forensics mode for:**
- Production client deliverables
- Final quality assessment (need full pipeline)
- Performance benchmarking of full pipeline
- Visual verification of exports/marketing assets

## Success Metrics

Forensics mode is working correctly when:
- ✅ Pipeline logs show `post_tile=0` (not preset default)
- ✅ report.json shows `upscale: 1, upscaler: "none", post_tile: 0`
- ✅ Only master16.tif is generated (when using `--master16-only`)
- ✅ Processing time is minimal (~0.5-2s for typical images)
- ✅ No upscaling or tiling overhead in stage timings

## Known Limitations

1. **Upscaled file still generated**: When using `--disable-upscale` alone (without `--disable-exports`), an upscaled16.tif is still generated but it's just a copy of master16 with no actual upscaling.
   - **Workaround**: Use `--master16-only` or `--disable-exports` to prevent this.

2. **Preview/marketing still generated**: When using `--disable-upscale` or `--disable-post-tile` alone, exports are still generated.
   - **Workaround**: Add `--disable-exports` flag.

3. **Preset must still be specified**: Forensics mode overrides preset defaults but doesn't replace the need to specify a preset.
   - **Reason**: Grading parameters (saturation, clarity, etc.) still come from preset.

## Troubleshooting

### Issue: Pipeline still shows `post_tile=2048`
**Cause**: Running older code before fix
**Solution**: Ensure you're running latest code with `_preset_applied` flag

### Issue: Upscaled file still generated
**Cause**: Using `--disable-upscale` without `--disable-exports`
**Solution**: Use `--master16-only` or add `--disable-exports`

### Issue: Processing still slow (>5 seconds)
**Cause**: Not using forensics mode correctly
**Solution**: Use `--master16-only` and verify logs show forensics warnings

## Version History

- **2025-12-22**: Initial implementation (CLI control-plane fix)
  - Added `--forensics-mode`, `--disable-upscale`, `--disable-post-tile`, `--disable-exports`, `--master16-only`
  - Added `_preset_applied` flag to prevent preset re-application
  - Verified with Kitchen test: 0.469s processing time, clean baseline output
