# Deprecated Files

This directory contains older versions of scripts that have been superseded by newer implementations.

## Why Are These Here?

These files are kept for historical reference and backward compatibility but are no longer actively maintained.

## Deprecated Files

### AD Editorial Post Pipeline

**Current Version:** `../ad_editorial_post_pipeline.py`

**Deprecated Versions:**
- `ad_editorial_post_pipeline_v2.py` - Version 2.0 (superseded by main version)
- `ad_editorial_post_pipeline_v3.py` - Version 3.0 (merged into main version)
- `test_ad_pipeline.py` - Old tests (superseded by `test_ad_pipeline_v3.py`)

**Migration:** 
Use `../ad_editorial_post_pipeline.py` which incorporates the best features from all versions:
- Proper sRGB gamma conversion
- Fine-grained progress tracking
- Comprehensive config validation
- Downsampled auto-upright (30x speedup)
- Style registry pattern
- Stage-based architecture

## When Will These Be Removed?

These files will be removed in v0.2.0 (estimated Q1 2026). If you depend on any of these files, please migrate to the current versions before then.

## Need Help?

If you're unsure which version to use or need help migrating, see:
- [Migration Guide](../../docs/REFACTORING_2025.md)
- [Tools README](../README.md)
