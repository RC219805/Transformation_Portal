# Quality Control & Best Practices

## 750 Picacho Lane Project - Best Practices Summary

### Source File Management

**Canon ical Truth: 6 Source Files (JPEGs)**
```
750Picacho_Aerial.jpg
750Picacho_GreatRoom.jpg
750Picacho_Kitchen.jpg
750Picacho_Pool.jpg
750Picacho_PrimaryBathroom.jpg
750Picacho_PrimaryBedroom.jpg
```

**Location:** `/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/JPEGs`

### Standard Processing Workflow

#### Option 1: Quality-Controlled Pipeline (Recommended)
```bash
cd /Users/rc/Transformation_Portal
python3 quality_control_pipeline.py
```

#### Option 2: Direct Pipeline
```bash
cd /Users/rc/Transformation_Portal
python3 unified_luxury_pipeline.py \
  --input /Users/rc/Desktop/Cache/750_LightFiction_Final_Views/JPEGs \
  --output /Users/rc/Desktop/Cache/750_LightFiction_Final_Views/Final_Production \
  --formats jpeg png tiff \
  --preset luxury_estate
```

### Output Structure
```
Final_Production/
├── 750Picacho_Aerial_luxury.jpg        (Web-optimized)
├── 750Picacho_Aerial_luxury.png        (Lossless RGB)
├── 750Picacho_Aerial_luxury.tif        (16-bit master)
├── 750Picacho_GreatRoom_luxury.jpg
├── ... (repeat for all 6 views × 3 formats = 18 files)
```

### Quality Verification Checklist

**Before Processing:**
- [ ] Verify 6 source files present in JPEGs folder
- [ ] No duplicate files (esp. Aerial)
- [ ] Source files readable and uncorrupted

**During Processing:**
- [ ] Monitor pipeline progress
- [ ] Check for error messages
- [ ] Verify depth maps generate correctly

**After Processing:**
- [ ] Count output files: should be exactly 18 (6 × 3 formats)
- [ ] Open sample TIFF files to verify 16-bit quality
- [ ] Compare TIFF to JPEG - TIFF should be equal or better quality
- [ ] Check `quality_control_report.json` for any warnings

### Common Issues & Solutions

#### Issue: TIFF Files Degraded (Lower Quality than JPEG)
**Root Cause:** PIL/Pillow saving TIFFs as 8-bit or using inappropriate color modes

**Solution:** Use `tifffile` library
```python
import tifffile
import numpy as np
from PIL import Image

# Load image
img = Image.open("source.jpg")
img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1

# Convert to 16-bit
img_16bit = (img_array * 65535).astype(np.uint16)

# Save with tifffile
tifffile.imwrite("output.tif", img_16bit, compression='lzw')
```

#### Issue: Duplicate Aerial Files
**Root Cause:** Multiple versions in source directory

**Solution:** Clean source directory before processing
```bash
# Verify exactly 6 files
cd /Users/rc/Desktop/Cache/750_LightFiction_Final_Views/JPEGs
ls -1 | wc -l  # Should output: 6
```

#### Issue: Multiple Processing Scripts
**Root Cause:** Ad-hoc scripts created for troubleshooting

**Solution:** Use ONE canonical script
- Primary: `unified_luxury_pipeline.py`
- QC Wrapper: `quality_control_pipeline.py`
- Archive old scripts to `deprecated/`

### Codebase Quality Standards

#### 1. **Markdown File Organization**
- **Maximum 10 markdown files in root**
- Move session notes to `docs/sessions/`
- Move technical guides to `docs/guides/`
- Keep only: README.md, START_HERE.md, MIGRATION_GUIDE.md, DEPRECATION_POLICY.md, CONTRIBUTING.md

#### 2. **Python Code Quality**
```bash
# Before committing:
# 1. Fix flake8 critical errors
flake8 . --select=E9,F63,F7,F82

# 2. Address pylint issues
pylint your_file.py

# 3. Run tests
pytest tests/ -v

# 4. Format code
black your_file.py  # or autopep8
```

#### 3. **Import Handling**
```python
# ❌ BAD: Undefined names at runtime
from some_module import *

# ✅ GOOD: Explicit imports
from some_module import specific_function, SpecificClass
```

#### 4. **Dependency Management**
```python
# ❌ BAD: Import without availability check
import optional_library  # Crashes if not installed

# ✅ GOOD: Graceful degradation
try:
    import optional_library
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    print("Warning: optional_library not available, some features disabled")
```

### Git Workflow Best Practices

#### Before Pushing to Main:
```bash
# 1. Check status
git status

# 2. Run quality checks
make lint  # or: flake8 + pylint
make test-fast

# 3. Clean up markdown files if needed
# (Keep root markdown count ≤ 10)

# 4. Commit with descriptive message
git add -A
git commit -m "feat: Add quality control pipeline for 750 Picacho

- Implemented QualityControlPipeline class
- Added source file verification
- Added output quality verification
- Fixed TIFF degradation issue with tifffile library"

# 5. Push
git push origin main
```

### Performance Optimization

#### Use CoreML for Depth Processing (Apple Silicon)
```python
# Check for CoreML availability
import platform
if platform.processor() == 'arm' and platform.system() == 'Darwin':
    # Download CoreML depth model
    # Model will use Apple Neural Engine (ANE) for 3-5x speedup
    use_coreml = True
```

#### Batch Processing Best Practices
```python
# ❌ BAD: Process one at a time
for image in images:
    result = process(image)
    save(result)

# ✅ GOOD: Batch with progress tracking
from tqdm import tqdm

results = []
for image in tqdm(images, desc="Processing"):
    result = process(image)
    results.append(result)

# Save all at once
for result, output_path in zip(results, output_paths):
    save(result, output_path)
```

### Troubleshooting Guide

#### CI/CD Failures

**Test Failure: "Too many markdown files in root"**
```bash
# Count markdown files
ls *.md | wc -l

# If > 10, move extras to docs/
mkdir -p docs/sessions
mv SESSION_*.md docs/sessions/
mv SUMMARY_*.md docs/sessions/
```

**Flake8 Error: "undefined name 'iio'"**
```python
# ❌ Missing import
iio.imwrite(path, data)

# ✅ Add import at top of file
import imageio as iio
```

**Pylint Warning: "Trailing whitespace"**
```bash
# Fix automatically
autopep8 --in-place --select=W291,W293 your_file.py

# Or configure editor to remove trailing whitespace on save
```

### Decision Matrix: Which Tool to Use

| Task | Tool | Reason |
|------|------|--------|
| Save 16-bit TIFF | `tifffile.imwrite()` | Preserves bit depth, avoids PIL degradation |
| Load any image | `PIL.Image.open()` | Universal, handles metadata well |
| Depth estimation | `DepthAnything V2 + CoreML` | Fastest on Apple Silicon |
| Batch processing | `unified_luxury_pipeline.py` | Production-ready, quality-controlled |
| Quality verification | `quality_control_pipeline.py` | Comprehensive checks |
| One-off testing | Direct Python script | Fast iteration |

### Next Steps for Quality Improvement

1. **Consolidate Processing Scripts**
   - Archive: `process_750_picacho.py`, `process_750picacho_proper_16bit.py`, etc.
   - Keep only: `unified_luxury_pipeline.py`, `quality_control_pipeline.py`

2. **Implement Pre-commit Hooks**
   ```bash
   # Create .pre-commit-config.yaml
   # Auto-format, lint, and test before commits
   ```

3. **Add Integration Tests**
   - Test full pipeline on sample images
   - Verify TIFF quality automatically
   - Check for regressions

4. **Document Architecture**
   - Create `docs/ARCHITECTURE.md`
   - Explain pipeline stages
   - Show data flow diagrams

5. **Performance Profiling**
   - Benchmark each stage
   - Identify bottlenecks
   - Optimize critical paths

