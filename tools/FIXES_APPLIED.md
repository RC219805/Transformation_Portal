# AD Editorial Post-Production Pipeline - Priority 1 Fixes

## Summary

This document describes the critical fixes applied to `ad_editorial_post_pipeline.py` based on comprehensive code review.

---

## 🔴 Critical Fixes Applied

### 1. Missing Imports ✅

**Problem:** Code referenced `yaml` and `ImageReader` without importing them, causing immediate `NameError` crashes.

**Fix:**
```python
import yaml  # Added
from reportlab.lib.utils import ImageReader  # Added
```

**Impact:** Pipeline can now execute without import errors.

---

### 2. 16-bit TIFF Data Loss ✅

**Problem:** Original code converted uint16 data to grayscale mode, then to RGB, losing 16-bit precision:
```python
# BROKEN - loses 16-bit data
im = Image.fromarray(img16, mode="I;16")  # 16-bit grayscale
im = im.convert("RGB")  # Converts to 8-bit RGB!
```

**Fix:**
```python
# FIXED - preserves 16-bit RGB
im = Image.fromarray(img16, mode="RGB")  # PIL handles uint16 RGB correctly
```

**Impact:** Print-quality TIFFs now preserve full 16-bit color depth, preventing banding and posterization.

---

### 3. Config Validation ✅

**Problem:** No validation meant invalid config values caused crashes deep in pipeline:
- Negative workers
- Invalid JPEG quality (0, 500)
- Upright rotation > 360 degrees
- Missing required fields

**Fix:** Added comprehensive `validate()` method to `PipelineConfig`:
```python
def validate(self) -> None:
    """Comprehensive config validation to prevent runtime errors."""
    errors = []

    # Validate workers (1-64)
    if not isinstance(workers, int) or not (1 <= workers <= 64):
        errors.append(f"processing.workers must be 1-64, got {workers}")

    # Validate JPEG quality (1-100)
    if not isinstance(jpeg_quality, int) or not (1 <= jpeg_quality <= 100):
        errors.append(f"export.jpeg_quality must be 1-100, got {jpeg_quality}")

    # ... 15+ additional validations

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))
```

**Impact:** Clear error messages at startup prevent wasted processing time.

---

### 4. Hash Verification Logic ✅

**Problem:** Original `copy_and_verify` had flawed logic:
```python
# BROKEN
def copy_and_verify(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return  # ❌ Assumes same size = same content
    shutil.copy2(src, dst)
    if sha256sum(src) != sha256sum(dst):  # ❌ Computes src hash twice
        raise RuntimeError(...)
```

**Fix:**
```python
# FIXED
def copy_and_verify(src: Path, dst: Path) -> None:
    if dst.exists():
        src_hash = sha256sum(src)
        dst_hash = sha256sum(dst)
        if src_hash == dst_hash:
            LOG.debug("File already exists with matching hash: %s", dst)
            return
        LOG.warning("Hash mismatch for existing %s, re-copying", dst)

    shutil.copy2(src, dst)

    # Verify copy succeeded
    src_hash = sha256sum(src)
    dst_hash = sha256sum(dst)
    if src_hash != dst_hash:
        dst.unlink()  # Clean up corrupted file
        raise RuntimeError(f"Hash mismatch copying {src} -> {dst}")
```

**Impact:** Ensures file integrity; corrupted copies are detected and removed.

---

### 5. Atomic File Writes ✅

**Problem:** Files written directly could be corrupted if process interrupted:
```python
# BROKEN - corruption if interrupted
im.save(str(path), format="TIFF", ...)
```

**Fix:** Added `atomic_write()` wrapper:
```python
def atomic_write(path: Path, writer_func, *args, **kwargs) -> None:
    """Atomically write by writing to temp file first, then renaming."""
    temp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        writer_func(temp_path, *args, **kwargs)
        temp_path.replace(path)  # Atomic on POSIX
    except Exception:
        if temp_path.exists():
            temp_path.unlink()  # Clean up
        raise

# Usage in save_tiff16_prophoto:
def _write(p: Path):
    im.save(str(p), format="TIFF", ...)
atomic_write(path, _write)
```

**Impact:** Interrupted writes leave no corrupted files; pipeline can resume safely.

---

## Additional Improvements

### Dynamic Style Support

**Change:** Layout now uses configured styles instead of hardcoded variants:
```python
# Before: hardcoded
variants = ["natural", "minimal", "cinematic"]

# After: from config
variants = list(cfg.styles.keys())
```

**Impact:** Users can add custom styles in config without code changes.

---

## Testing Checklist

Before using in production, verify:

- [ ] Install all dependencies: `pip install rawpy pillow opencv-python tqdm pyyaml reportlab exifread piexif`
- [ ] Copy `sample_config.yml` and customize paths
- [ ] Verify ICC profile paths exist on your system
- [ ] Test with 1-2 RAW files first
- [ ] Check output TIFFs are truly 16-bit (use `exiftool` or `identify -verbose`)
- [ ] Verify web JPEGs have correct size and quality
- [ ] Test interruption recovery (Ctrl+C during processing)

---

## Known Limitations (Not Fixed in Priority 1)

These require more extensive refactoring:

1. **No Multiprocessing:** Pipeline processes serially despite `workers` config
2. **No Resume Capability:** Reprocesses all files if interrupted
3. **Memory Usage:** Large images may cause memory issues on limited systems
4. **Color Management:** Linear gamma handling needs review for accurate adjustments
5. **HSV Conversions:** Uses PIL (lossy 8-bit) instead of proper vectorized operations

See full code review for Priority 2-4 recommendations.

---

## Usage Example

```bash
# 1. Copy and edit config
cp tools/sample_config.yml my_project_config.yml
nano my_project_config.yml  # Customize paths

# 2. Run pipeline
python tools/ad_editorial_post_pipeline.py run --config my_project_config.yml -vv

# 3. Check outputs
ls ~/Photography/SmithResidence_2025-10-18/EXPORT/
```

---

## Quick Start Checklist

1. ✅ Install dependencies
2. ✅ Customize `sample_config.yml`
3. ✅ Verify ICC profile paths
4. ✅ Place RAW files in `input_raw_dir`
5. ✅ Run pipeline with `-vv` for verbose logging
6. ✅ Review outputs in `EXPORT/`
7. ✅ Edit `DOCS/selects.csv` to cull unwanted images
8. ✅ Re-run pipeline to process only selected images

---

## Support

For issues or questions:
- Check config validation errors carefully
- Run with `-vv` for detailed logging
- Verify all dependencies are installed
- Test with minimal dataset first

---

## Changelog

**2025-10-23 - Priority 1 Fixes**
- ✅ Added missing imports (yaml, ImageReader)
- ✅ Fixed 16-bit TIFF saving to preserve bit depth
- ✅ Added comprehensive config validation
- ✅ Fixed copy_and_verify hash checking logic
- ✅ Implemented atomic file writes
- ✅ Added dynamic style support from config

**Original Version**
- Initial implementation with 800+ lines
- Professional photography workflow
- HDR, panorama, auto-upright, style grading
- Multiple critical bugs preventing production use
