# Test Outputs Directory

This directory contains outputs from Lux Depth V2 pipeline test runs.

## Structure

```
test_outputs/
├── 750_picacho/          # 750 Picacho project test outputs
│   ├── *_master16.tif    # 16-bit TIFF masters (pre-upscale)
│   ├── *_upscaled16.tif  # 16-bit TIFF final (upscaled)
│   ├── *_marketing.png   # 8-bit PNG for fast review
│   ├── *_preview.jpg     # JPEG thumbnails
│   ├── *_report.json     # Processing metadata
│   ├── TEST_SUMMARY.json # Batch test summary (JSON)
│   └── TEST_SUMMARY.txt  # Batch test summary (human-readable)
└── README.md             # This file
```

## Running Tests

### Quick Test
```bash
cd /home/runner/work/Transformation_Portal/Transformation_Portal
python lux_depth_v2/test_750_picacho.py
```

### Custom Output Directory
```bash
python lux_depth_v2/test_750_picacho.py \
  --output-dir lux_depth_v2/test_outputs/custom_test/
```

## Expected Files per Test

For each input TIFF file, you will get:
- 1 × master16.tif (16-bit TIFF, 30-50 MB)
- 1 × upscaled16.tif (16-bit TIFF, 80-120 MB)
- 1 × marketing.png (8-bit PNG, 5-10 MB)
- 1 × preview.jpg (JPEG thumbnail, 500KB-1MB)
- 1 × report.json (Processing metadata, <10KB)

Plus 2 summary files for the entire batch:
- TEST_SUMMARY.json (Machine-readable)
- TEST_SUMMARY.txt (Human-readable)

## Validation

Check TEST_SUMMARY.txt for:
- Processing time per file
- Total batch processing time
- 16-bit verification status
- Material detection accuracy

## Cleanup

To remove test outputs:
```bash
rm -rf lux_depth_v2/test_outputs/750_picacho/
```

To remove all test outputs:
```bash
rm -rf lux_depth_v2/test_outputs/*/
```

---

**Note**: This directory is created automatically when running tests and can be safely deleted between test runs.
