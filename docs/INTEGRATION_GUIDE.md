# Integration Guide: Enhanced Format Utilities

This guide shows how to integrate the new enhancements into your Transformation Portal codebase.

## 📦 What You've Got

Three new files in `/mnt/user-data/outputs/`:

1. **`format_utils_enhancements.py`** - New enhanced functionality
2. **`test_format_utils_enhancements.py`** - Comprehensive tests
3. **`INTEGRATION_GUIDE.md`** - This file

## 🎯 What's New

### Option 2: Enhanced Format Detection
- ✅ Magic number detection (detect format from file content)
- ✅ MIME type detection
- ✅ Image integrity validation
- ✅ Comprehensive metadata extraction

### Option 3: Format Conversion Utilities
- ✅ Single file conversion with quality control
- ✅ Batch directory conversion
- ✅ Smart conversion (auto-optimize quality)
- ✅ Format recommendations by use case

### Option 4: Improved 16-bit TIFF Handling
- ✅ Save 16-bit TIFFs with metadata
- ✅ Load TIFFs preserving bit depth
- ✅ Convert TIFFs without losing quality
- ✅ Compression optimization
- ✅ Better error messages and fallbacks

## 🚀 Quick Start Integration

### Step 1: Copy Files to Your Project

```bash
# Navigate to your project
cd /Users/rc/Transformation_Portal/Transformation_Portal/Transformation_Portal

# Copy the enhancements module
cp /mnt/user-data/outputs/format_utils_enhancements.py ./

# Copy the tests
cp /mnt/user-data/outputs/test_format_utils_enhancements.py ./tests/
```

### Step 2: Update Your format_utils.py

You have two options:

#### Option A: Import from Enhancements (Recommended)

Add to the top of your existing `format_utils.py`:

```python
# At the top of format_utils.py
from format_utils_enhancements import (
    # Option 2: Enhanced detection
    detect_format_from_content,
    get_mime_type,
    validate_image_integrity,
    get_image_metadata,
    # Option 3: Conversion
    convert_image_format,
    batch_convert_directory,
    smart_convert,
    get_optimal_format_for_use_case,
    # Option 4: TIFF
    check_tifffile_available,
    save_tiff_16bit,
    load_tiff_preserve_depth,
    convert_tiff_preserve_depth,
)

# Now these functions are available in format_utils
```

#### Option B: Merge Directly

Copy the functions from `format_utils_enhancements.py` directly into your `format_utils.py`.

### Step 3: Install Optional Dependencies (if needed)

```bash
# For 16-bit TIFF support
pip install tifffile imagecodecs

# Or use the extras
pip install -e ".[tiff]"
```

### Step 4: Run the Tests

```bash
# Run new tests
pytest tests/test_format_utils_enhancements.py -v

# Run all format tests
pytest tests/test_format_utils.py tests/test_format_utils_enhancements.py -v
```

## 📖 Usage Examples

### Example 1: Enhanced Format Detection

```python
from format_utils import detect_format_from_content, validate_image_integrity

# Detect format from content (not extension)
format_type = detect_format_from_content('mystery_file')
print(f"Detected: {format_type}")  # Output: "JPEG"

# Validate image before processing
is_valid, error = validate_image_integrity('photo.jpg')
if not is_valid:
    print(f"Invalid image: {error}")
else:
    # Process the image...
    pass
```

### Example 2: Batch Format Conversion

```python
from format_utils import batch_convert_directory

# Convert all TIFFs to high-quality JPEGs
stats = batch_convert_directory(
    input_dir='./raw_tiffs',
    output_dir='./web_jpgs',
    target_format='.jpg',
    quality=95,
    recursive=True,
    preserve_metadata=True
)

print(f"Converted {stats['success']}/{stats['total']} images")
```

### Example 3: Smart Conversion

```python
from format_utils import smart_convert

# Automatically choose best settings
smart_convert('photo.tiff', 'photo.jpg')  # Auto quality=95
smart_convert('logo.png', 'logo.webp')    # Preserves alpha
smart_convert('render.bmp', 'render.tiff') # Best quality
```

### Example 4: 16-bit TIFF Processing

```python
from format_utils import load_tiff_preserve_depth, save_tiff_16bit
import numpy as np

# Load 16-bit TIFF preserving quality
array, bit_depth = load_tiff_preserve_depth('photo_16bit.tiff')
print(f"Loaded {bit_depth}-bit image: {array.shape}")

# Process the image (your custom processing)
processed = your_processing_function(array)

# Save back as 16-bit with compression
save_tiff_16bit(
    processed,
    'output_16bit.tiff',
    compression='lzw',
    metadata={'software': 'Transformation Portal'}
)
```

### Example 5: Complete Validation + Conversion Workflow

```python
from format_utils import (
    validate_image_integrity,
    get_image_metadata,
    convert_image_format,
    get_optimal_format_for_use_case
)
from pathlib import Path

def process_image(input_path: Path, output_dir: Path):
    """Complete image processing workflow with validation."""
    
    # 1. Validate integrity
    is_valid, error = validate_image_integrity(input_path)
    if not is_valid:
        print(f"Skipping {input_path}: {error}")
        return False
    
    # 2. Get metadata
    meta = get_image_metadata(input_path)
    print(f"Processing {meta['format']} image: {meta['size']}")
    
    # 3. Determine optimal output format
    output_format = get_optimal_format_for_use_case(
        use_case='web',
        has_alpha=meta['has_alpha'],
        requires_16bit=(meta['bit_depth'] == 16)
    )
    
    # 4. Convert with appropriate settings
    output_path = output_dir / input_path.with_suffix(output_format).name
    success = convert_image_format(
        input_path,
        output_path,
        quality=95,
        preserve_metadata=True
    )
    
    return success

# Use it
process_image(Path('input/photo.tiff'), Path('output/'))
```

## 🔧 Integration with Existing Pipelines

### Luxury TIFF Batch Processor

```python
# In luxury_tiff_batch_processor.py
from format_utils import validate_image_integrity, load_tiff_preserve_depth

def process_tiff(input_path):
    # Add validation before processing
    is_valid, error = validate_image_integrity(input_path)
    if not is_valid:
        logging.warning(f"Skipping invalid TIFF: {error}")
        return None
    
    # Load preserving 16-bit depth
    array, bit_depth = load_tiff_preserve_depth(input_path)
    if bit_depth == 16:
        logging.info(f"Processing 16-bit TIFF: {input_path}")
    
    # Continue with existing processing...
    return process_image_array(array)
```

### Lux Render Pipeline

```python
# In lux_render_pipeline.py
from format_utils import detect_format_from_content, smart_convert

def load_render(input_path):
    # Detect format reliably
    format_type = detect_format_from_content(input_path)
    if not format_type:
        raise ValueError(f"Cannot identify image format: {input_path}")
    
    # Use smart conversion if format needs changing
    if format_type not in ['PNG', 'TIFF']:
        temp_path = Path(tempfile.mktemp(suffix='.png'))
        smart_convert(input_path, temp_path)
        input_path = temp_path
    
    # Continue with existing pipeline...
```

### Material Response System

```python
# In material_response.py  
from format_utils import get_image_metadata, convert_tiff_preserve_depth

def enhance_materials(input_path, output_path):
    # Get metadata to determine processing path
    meta = get_image_metadata(input_path)
    
    if meta['bit_depth'] == 16:
        # Use 16-bit preserving workflow
        array, _ = load_tiff_preserve_depth(input_path)
        enhanced = enhance_materials_16bit(array)
        save_tiff_16bit(enhanced, output_path, compression='lzw')
    else:
        # Standard 8-bit workflow
        enhanced = enhance_materials_8bit(input_path)
        # ...
```

## 📝 Updating Documentation

### Update SUPPORTED_FILE_FORMATS.md

Add a new section about the enhanced features:

```markdown
## Enhanced Format Features

### Intelligent Format Detection

The system now includes magic number detection that identifies formats from file content, not just extensions:

\`\`\`python
from format_utils import detect_format_from_content

# Detects JPEG even with wrong extension
format_type = detect_format_from_content('image.txt')  # Returns 'JPEG'
\`\`\`

### Batch Format Conversion

Convert entire directories with one command:

\`\`\`bash
python -c "from format_utils import batch_convert_directory; \
    batch_convert_directory('./tiffs', './jpgs', '.jpg', quality=95)"
\`\`\`

### 16-bit TIFF Preservation

The system now fully preserves 16-bit depth throughout the pipeline when `tifffile` is installed.
```

### Update README.md

Add to the features section:

```markdown
## ✨ Enhanced Features

### Smart Format Handling
- 🔍 **Magic Number Detection**: Identify formats from content, not just extensions
- ✅ **Integrity Validation**: Automatically detect corrupted images
- 📊 **Metadata Extraction**: Extract comprehensive EXIF, bit depth, and format info

### Format Conversion
- 🔄 **Batch Conversion**: Convert entire directories with quality control
- 🎯 **Smart Conversion**: Auto-optimize settings based on content
- 💎 **16-bit Preservation**: Maintain 16-bit depth through conversion pipeline

### Installation

For full 16-bit TIFF support:
\`\`\`bash
pip install -e ".[tiff]"  # Includes tifffile + imagecodecs
\`\`\```
```

## 🧪 Testing Strategy

### Run Tests Incrementally

```bash
# 1. Test new functions in isolation
pytest tests/test_format_utils_enhancements.py::TestEnhancedFormatDetection -v

# 2. Test conversions
pytest tests/test_format_utils_enhancements.py::TestFormatConversion -v

# 3. Test TIFF handling (may skip if tifffile not installed)
pytest tests/test_format_utils_enhancements.py::TestTIFFHandling -v

# 4. Run integration tests
pytest tests/test_format_utils_enhancements.py::TestIntegration -v

# 5. Run all format tests together
pytest tests/test_format_utils.py tests/test_format_utils_enhancements.py -v
```

### Add to CI Pipeline

Update `.github/workflows/` to include the new tests:

```yaml
- name: Test format utilities
  run: |
    pytest tests/test_format_utils.py -v
    pytest tests/test_format_utils_enhancements.py -v
```

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Copy files to project
2. ✅ Run tests locally
3. ✅ Commit to your feature branch

### Short-term (This Week)
4. 📝 Update documentation
5. 🔄 Integrate with existing pipelines
6. 🧪 Add integration tests with real workflows

### Medium-term (This Sprint)
7. 📊 Add performance benchmarks
8. 🎨 Create CLI commands for conversion
9. 📚 Write user guide with examples

## 💡 Usage Patterns

### Pattern 1: Safe Image Loading

```python
def safe_load_image(path):
    """Load image with validation and format detection."""
    is_valid, error = validate_image_integrity(path)
    if not is_valid:
        raise ValueError(f"Invalid image: {error}")
    
    format_type = detect_format_from_content(path)
    meta = get_image_metadata(path)
    
    if meta['bit_depth'] == 16:
        array, _ = load_tiff_preserve_depth(path)
        return array, meta
    else:
        img = Image.open(path)
        return np.array(img), meta
```

### Pattern 2: Batch Processing with Progress

```python
from tqdm import tqdm
from pathlib import Path

def batch_process_with_validation(input_dir, output_dir):
    """Process all images with validation."""
    input_files = list(Path(input_dir).glob('**/*'))
    image_files = [f for f in input_files if f.suffix.lower() in 
                   {'.jpg', '.png', '.tiff', '.tif'}]
    
    for img_path in tqdm(image_files, desc="Processing"):
        # Validate
        is_valid, error = validate_image_integrity(img_path)
        if not is_valid:
            print(f"Skipping {img_path.name}: {error}")
            continue
        
        # Process
        output_path = output_dir / img_path.name
        smart_convert(img_path, output_path)
```

### Pattern 3: Format-Aware Pipeline

```python
def process_any_format(input_path, output_path):
    """Process image regardless of input format."""
    # Detect and validate
    format_type = detect_format_from_content(input_path)
    meta = get_image_metadata(input_path)
    
    # Choose processing path based on format
    if meta['bit_depth'] == 16:
        return process_16bit(input_path, output_path)
    elif meta['has_alpha']:
        return process_with_alpha(input_path, output_path)
    else:
        return process_standard(input_path, output_path)
```

## 🐛 Troubleshooting

### Issue: tifffile not found

```bash
# Install tifffile
pip install tifffile imagecodecs

# Or use extras
pip install -e ".[tiff]"

# Verify
python -c "from format_utils import check_tifffile_available; \
    print(f'tifffile available: {check_tifffile_available()}')"
```

### Issue: Tests failing on TIFF tests

Some tests require tifffile. They will automatically skip if not available:

```bash
# Run with skip messages
pytest tests/test_format_utils_enhancements.py -v -rs
```

### Issue: Conversion quality issues

Adjust quality settings for your use case:

```python
# Web/preview: quality=85-90
convert_image_format(src, dst, quality=85)

# Print/delivery: quality=95-100
convert_image_format(src, dst, quality=98)

# Lossless formats (PNG, TIFF): quality ignored
convert_image_format(src, dst)  # Quality doesn't matter
```

## 📈 Performance Considerations

### Memory Usage

```python
# For large images, process in chunks
def process_large_image(path):
    meta = get_image_metadata(path)
    
    if meta['size'][0] * meta['size'][1] > 10_000_000:  # 10MP
        # Use tiled processing
        return process_in_tiles(path)
    else:
        # Load entire image
        return process_whole(path)
```

### Batch Performance

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_batch_convert(input_dir, output_dir, target_format):
    """Convert directory using multiple threads."""
    files = list(Path(input_dir).glob('*'))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(convert_image_format, f, 
                          output_dir / f.with_suffix(target_format).name)
            for f in files
        ]
        
        # Wait for all
        results = [f.result() for f in futures]
    
    return sum(results)  # Count successes
```

## 🎓 Learning Resources

### Example Scripts

Create example scripts in `examples/`:

```python
# examples/batch_convert_example.py
"""Example: Batch convert directory of TIFFs to web-optimized JPEGs."""

from format_utils import batch_convert_directory

stats = batch_convert_directory(
    input_dir='./raw_photos',
    output_dir='./web_photos',
    target_format='.jpg',
    quality=90,
    recursive=True
)

print(f"""
Conversion Complete:
- Total files: {stats['total']}
- Successfully converted: {stats['success']}
- Failed: {stats['failed']}
- Skipped (already target format): {stats['skipped']}
""")
```

### Interactive Notebook

Create `notebooks/format_utilities_demo.ipynb` with examples.

## ✅ Checklist

Before committing your changes:

- [ ] Files copied to project
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Integration examples added
- [ ] CI pipeline updated
- [ ] Committed to feature branch
- [ ] PR description includes new features

## 🚀 Commit Message

```bash
git add format_utils_enhancements.py tests/test_format_utils_enhancements.py
git commit -m "Add enhanced format utilities with detection, conversion, and 16-bit TIFF support

- Option 2: Enhanced format detection (magic numbers, MIME types, integrity validation)
- Option 3: Format conversion utilities (single, batch, smart convert)
- Option 4: Improved 16-bit TIFF handling (metadata, compression, quality preservation)
- Add comprehensive test suite with 40+ test cases
- Backward compatible with existing format_utils.py"

git push origin copilot/enhance-image-file-types
```

## 📞 Support

If you encounter issues:

1. Check tests: `pytest tests/test_format_utils_enhancements.py -v`
2. Verify dependencies: `pip list | grep -E 'Pillow|tifffile|numpy'`
3. Review error messages in test output

---

**Ready to integrate?** Start with Step 1 and work through the checklist! 🎉
