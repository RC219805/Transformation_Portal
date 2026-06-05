# Deprecated Files

This directory contains older AD editorial snapshots that have been superseded
by the maintained implementation in `tools/ad_editorial_post_pipeline.py`.

## Why Are These Here?

These files are kept for historical audit only and are no longer actively
maintained.

## Deprecated Files

### AD Editorial Post Pipeline

**Current maintained entrypoint:** `../ad_editorial_post_pipeline.py`

**Deprecated Versions:**
- `ad_editorial_post_pipeline_v2.py` - Version 2.0 (superseded by main version)
- `ad_editorial_post_pipeline_v3.py` - Version 3.0 (merged into main version)
- `test_ad_pipeline.py` - Version 2 tests
- `test_ad_pipeline_v3.py` - Version 3 tests

**Migration:**
Use `../ad_editorial_post_pipeline.py` for current AD editorial processing.
These snapshots remain co-located with their tests only so historical behavior
can be audited without advertising deprecated entrypoints as active tools.

## Removal Policy

No removal date is promised here. Removing these snapshots requires an explicit
contract audit confirming that no docs, tests, or historical evidence still need
the files.

## Need Help?

If you're unsure which version to use or need help migrating, see:
- [AD Editorial Post-Production Guide](../../docs/guides/AD_EDITORIAL_POST_PIPELINE.md)
- [Migration Guide](../../docs/guides/REFACTORING_2025.md)
- [Tools README](../README.md)
