# Tools Directory - Phase 1 Enhancement Utilities

**Status:** ✅ Production Ready  
**Version:** 1.0.0

Standalone tools for image processing workflow optimization, quality assurance, and analysis.

## Quick Reference

| Tool | Purpose | Key Feature |
|------|---------|-------------|
| `comparison_tool.py` | Compare outputs | PSNR/SSIM metrics |
| `hdr_visualizer.py` | HDR analysis | Before/after histograms |
| `time_predictor.py` | Time estimation | Historical learning |
| `qa_validator.py` | Pre-flight QA | Go/No-Go decisions |

## Installation

```bash
# Core dependencies
pip install numpy Pillow tifffile

# Visualization (recommended)
pip install matplotlib scikit-image scipy
```

## Quick Start Examples

```bash
# 1. Validate inputs
python tools/qa_validator.py input_images/*.tif --output qa_report.md

# 2. Predict processing time
python tools/time_predictor.py input_images/*.tif --output predictions.json

# 3. Compare outputs
python tools/comparison_tool.py --dir1 baseline/ --dir2 enhanced/ --output comparisons/

# 4. Analyze HDR processing
python tools/hdr_visualizer.py --before hdr_input.tif --after tone_mapped.tif --name Kitchen
```

## Documentation

- **Full Documentation:** `docs/PHASE1_ENHANCEMENTS.md`
- **Utils Directory:** `utils/README.md`
- **AD Pipeline:** `tools/README_AD_PIPELINE.md`

## Support

For detailed usage, see `docs/PHASE1_ENHANCEMENTS.md` or run `python tools/<tool>.py --help`
