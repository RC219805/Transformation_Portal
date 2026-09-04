# V2 Integration Test Script

## Overview

`test_v2_integration.sh` is a comprehensive test harness for validating the lux-depth-v3 pipeline with V2 enhancement enabled. It tests the full pipeline orchestration, V2 script integration, output generation, and metadata tracking.

## Purpose

This script validates:
- ✅ V2 script invocation and execution
- ✅ V2 output generation (enhanced images + JSON reports)
- ✅ V2 timing metadata in manifests
- ✅ Pipeline orchestration with V2 stage enabled
- ✅ Error handling and graceful degradation

## Usage

### Basic Test
```bash
./scripts/test_v2_integration.sh
```

### Verbose Output
```bash
./scripts/test_v2_integration.sh --verbose
```

### Clean Run (Remove Previous Outputs)
```bash
./scripts/test_v2_integration.sh --clean
```

### Help
```bash
./scripts/test_v2_integration.sh --help
```

## Test Images

The script tests with a representative mix of formats:
- **750Picacho_Kitchen.jpg** - JPEG interior (3.1 MB)
- **750Picacho_Pool.jpg** - JPEG exterior (3.3 MB)
- **750Picacho_PrimaryBedroom_Ultimate.tif** - Large TIFF (137 MB)

## Test Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Quality | standard | Fast execution for testing |
| Depth Backend | da3 | Depth Anything V3 (validated) |
| V2 Preset | default | Uses placeholder passthrough |
| Output Directory | `output/v2_integration_test` | Isolated test outputs |

## Validation Checks

### 1. V2 Script Execution
- Verifies V2 script was invoked
- Counts invocation occurrences
- Detects V2 errors in logs

### 2. V2 Outputs
- Checks V2 directory exists
- Counts enhanced images
- Counts JSON report files
- Validates file sizes

### 3. V2 Reports
- Validates JSON syntax
- Checks report structure
- Verifies all reports are parseable

### 4. V2 Timing Metadata
- Checks manifests for V2 timing data
- Validates timing is recorded
- Reports V2 stage duration

### 5. Output Structure
- Validates expected directories:
  - `depth/` - Depth maps
  - `pbr/` - PBR material maps
  - `v2/` - V2 enhanced images ← **Critical**
  - `manifests/` - Pipeline metadata

## Expected Output Structure

```
output/v2_integration_test/
├── depth/                      # Depth maps from DA3
│   ├── <input-key>_depth.png
│   ├── <input-key>_depth.npy   # When float depth is requested
│   └── <input-key>_depth_metadata.json
├── pbr/                        # Only when --pbr on
│   ├── <input-key>_normal.png
│   ├── <input-key>_roughness.png
│   └── <input-key>_ao.png
├── v2/                         # V2 enhanced images ← NEW
│   ├── <input-key>_v2_enhanced.{png,tif}
│   └── <input-key>_report.json
├── manifests/                  # Combined manifests with V2 timing
│   ├── <input-key>_combined.json
│   ├── batch_<batch-id>.json
│   └── execution_evidence_<batch-id>.json
├── run_card_<batch-id>.json    # Reproducibility card when enabled
├── run_card_<batch-id>.self.json # Run-card self-integrity sidecar
└── test.log                    # Pipeline execution logs
```

`<input-key>` is the orchestrator's `<stem>_<ext>_<hash8>` identity. Read the
exact paths from the batch results and combined manifests; do not reconstruct
them from the source filename.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Test failed or validation error |
| 2 | Setup/prerequisite error |

## Prerequisites

1. **lux-depth-v3 installed:**
   ```bash
   pip install -e '.[ml]'
   ```

2. **V2 enhancement script:**
   ```bash
   scripts/enhance_image.py
   ```

3. **Test images available:**
   - `input_images/750_picacho/source_jpegs/750Picacho_Kitchen.jpg`
   - `input_images/750_picacho/source_jpegs/750Picacho_Pool.jpg`
   - `input_images/750Picacho_PrimaryBedroom_Ultimate.tif`

## Troubleshooting

### "lux-depth-v3 command not found"
```bash
pip install -e '.[ml]'
# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### "V2 enhancement script not found"
Verify the script exists:
```bash
ls -l scripts/enhance_image.py
```

### "Missing test images"
Ensure test images are present:
```bash
ls -lh input_images/750_picacho/source_jpegs/
ls -lh input_images/750Picacho_PrimaryBedroom_Ultimate.tif
```

### Test failed with validation errors
Run with verbose mode to see detailed logs:
```bash
./scripts/test_v2_integration.sh --verbose
```

Check the log file:
```bash
cat output/v2_integration_test/test.log
```

## Performance

Expected execution time: **< 2 minutes**
- 3 images (2 JPEG + 1 large TIFF)
- Standard quality preset
- V2 passthrough (minimal overhead)
- DA3 depth estimation

## Integration Points

### V2 Script Contract

The test validates the V2 script is called with correct arguments:

```bash
python3 scripts/enhance_image.py \
    --depth-dir "output/v2_integration_test/depth" \
    --output-dir "output/v2_integration_test/v2" \
    --preset "default" \
    input_images/750_picacho/source_jpegs/750Picacho_Kitchen.jpg
```

### Expected V2 Outputs

For each input image, V2 script produces:
1. **Enhanced image:** `v2/750Picacho_Kitchen.jpg`
2. **JSON report:** `v2/750Picacho_Kitchen_report.json`

### Manifest Integration

Pipeline manifests include V2 metadata:
```json
{
  "stages": {
    "depth": { "seconds": 10.5 },
    "v2": { "seconds": 0.1, "status": "success" },
    "pbr": { "seconds": 2.3 }
  }
}
```

## Success Criteria

All checks must pass:
- ✅ Pipeline completes successfully
- ✅ V2 script invoked for each image
- ✅ V2 directory contains enhanced images
- ✅ V2 reports are valid JSON
- ✅ No V2 errors in logs
- ✅ Expected output structure present

## Next Steps

After successful V2 integration test:
1. **Replace placeholder:** Implement real V2 enhancement logic
2. **Add presets:** Create enhancement presets (subtle, balanced, dramatic)
3. **Performance tuning:** Optimize V2 processing for batch workflows
4. **Quality validation:** Visual inspection and automated quality checks

## Related Documentation

- **V2 Enhancement Script:** `scripts/enhance_image.py`
- **Pipeline Documentation:** `docs/pipelines/lux_depth_v3.md`
- **Test Strategy:** `docs/testing/integration_tests.md`

## Notes

- Script uses `--overwrite` flag for idempotent runs
- Logs are saved to `output/v2_integration_test/test.log`
- V2 script currently uses passthrough implementation (copies input to output)
- Test is designed for fast feedback (< 2 minutes)
