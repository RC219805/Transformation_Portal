# Test Execution Commands

## Quick Validation Test

```bash
# Test that sweep_runner imports correctly
python -c "from exploration.sweep_runner import SweepRunner; print('✓ Import successful')"

# Test help
python exploration/sweep_runner.py --help

# Test baseline mode (will actually run - takes 10-15 min)
# python exploration/sweep_runner.py --mode baseline
```

## Single Parameter Sweep Test

```bash
# Test a single parameter sweep (color_tone is fastest to test)
# This will run 3 parameter values × 6 images = 18 processing runs
# Estimated time: 30-45 minutes

python exploration/sweep_runner.py \
  --mode single \
  --category color_tone \
  --parameter saturation_protection
```

## Expected Output Structure

After running baseline:
```
sweep_runs/
└── baseline/
    ├── params.json
    ├── outputs/
    │   ├── 750Picacho_Kitchen_UltraQuality_master16.tif
    │   ├── 750Picacho_Pool_UltraQuality_master16.tif
    │   ├── 750Picacho_Aerial_UltraQuality_master16.tif
    │   ├── 750Picacho_PrimaryBathroom_UltraQuality_master16.tif
    │   ├── 750Picacho_GreatRoom_UltraQuality_master16.tif
    │   ├── 750Picacho_PrimaryBedroom_UltraQuality_master16.tif
    │   └── TEST_SUMMARY.json
    └── notes.md
```

After running single parameter sweep:
```
sweep_runs/
├── baseline/
└── color_tone_saturation_protection_delta0/
    ├── params.json          # {"parameter": "saturation_protection", "value": 1.0}
    ├── metrics.json         # {"processing_time_seconds": 123.45, ...}
    ├── outputs/
    │   └── [same 6 images as baseline]
    └── notes.md
├── color_tone_saturation_protection_delta1/
    └── [same structure, value: 0.8]
└── color_tone_saturation_protection_delta2/
    └── [same structure, value: 0.85]
```

## Verification Steps

1. **Check outputs exist:**
   ```bash
   ls -lh sweep_runs/baseline/outputs/*.tif
   ```

2. **Check metrics were captured:**
   ```bash
   cat sweep_runs/baseline/outputs/TEST_SUMMARY.json | jq .
   ```

3. **Check parameter sweep outputs:**
   ```bash
   find sweep_runs/color_tone_* -name "metrics.json" -exec cat {} \;
   ```

4. **Verify parameter values in each run:**
   ```bash
   find sweep_runs/color_tone_* -name "params.json" -exec cat {} \;
   ```

## Success Criteria

- ✅ All 6 input images processed
- ✅ 6 master16.tif files generated per run
- ✅ TEST_SUMMARY.json exists with metrics
- ✅ params.json has correct parameter values
- ✅ metrics.json has sweep metadata
- ✅ No error messages in console
- ✅ Processing completes in reasonable time

## Troubleshooting

### If baseline fails:
```bash
# Check input images exist
ls -lh projects/750_picacho_lane/Final_Production_UltraQuality/*.tif

# Check dependencies
pip install -r lux_depth_v2/requirements-repo.txt

# Test manually
python lux_depth_v2/test_750_picacho.py --dry-run
```

### If parameter sweep fails:
```bash
# Check sweep_runner can create override files
python -c "from exploration.sweep_runner import SweepRunner; r = SweepRunner(); f = r._apply_parameter_override('test', 'test', 1); print(f); f.unlink()"

# Check test script can read overrides
echo '{"category":"test","parameter":"test","value":1}' > /tmp/test.json
SWEEP_OVERRIDE_FILE=/tmp/test.json python lux_depth_v2/test_750_picacho.py --dry-run
```

## Performance Expectations

- **Baseline (6 images):** 10-15 minutes
- **Single parameter (3 values × 6 images):** 30-45 minutes
- **Full Phase 1 (9 parameters):** 4.5-6.75 hours
- **Per-image processing:** ~1-2 minutes on M4 Max

## Next Steps After Testing

1. Review visual quality of outputs
2. Compare metrics between delta values
3. Update notes.md with observations
4. Identify best-performing parameters
5. Proceed to Phase 2 combined sweeps
