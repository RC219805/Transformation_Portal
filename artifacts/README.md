# Artifacts Directory

This directory stores pipeline outputs and test artifacts that should NOT be committed to git.

## Structure

- `outputs/` - Pipeline execution results (EXR, TIFF, MP4)
- `phase2_benchmark_results.json` - Performance metrics
- `750_picacho_*.json` - Project metadata

## Symlink Compatibility Note

⚠️ **Windows/CI Portability**

Some legacy scripts may reference `phase2_task1_outputs/` via symlink.
Modern scripts should prefer `artifacts/outputs` directly.

Symlinks may break on:
- Windows (unless Developer Mode enabled)
- Some zip/unzip tools
- Certain CI runners

**Migration Path**: Update scripts to use `artifacts/outputs` instead of `phase2_task1_outputs`.

## .gitignore Coverage

All artifact patterns are excluded via `.gitignore`:
```
artifacts/**/*.exr
artifacts/**/*.tif
artifacts/**/*.tiff
artifacts/**/*.mp4
artifacts/**/*.mov
```

Defense-in-depth: CI enforcement also validates no large binaries leak into git.
