# Quick Reference: Enhanced Format Utilities

## 🚀 Most Common Functions

### Format Detection

```python
# Detect format from file content (not extension)
format_type = detect_format_from_content('mystery.file')
# Returns: 'JPEG', 'PNG', 'TIFF', etc.

# Validate image isn't corrupted
is_valid, error = validate_image_integrity('photo.jpg')
# Returns: (True, None) or (False, "error message")

# Get full metadata
meta = get_image_metadata('image.jpg')
# Returns: {'format': 'JPEG', 'size': (4000, 3000), 'bit_depth': 8, ...}
```

### Format Conversion

```python
# Convert single image
convert_image_format('input.tiff', 'output.jpg', quality=95)

# Batch convert directory
stats = batch_convert_directory(
    './raw',
    './processed', 
    '.jpg',
    quality=90
)
print(f"Converted {stats['success']}/{stats['total']}")

# Smart convert (auto-optimizes)
smart_convert('input.bmp', 'output.png')
```

### 16-bit TIFF

```python
# Load preserving bit depth
array, bit_depth = load_tiff_preserve_depth('photo.tiff')
print(f"{bit_depth}-bit image")

# Save as 16-bit
save_tiff_16bit(array, 'output.tiff', compression='lzw')

# Convert preserving depth
convert_tiff_preserve_depth('input.tiff', 'output.tiff')
```

---

## 📋 Function Reference Table

| Function | Purpose | Returns |
|----------|---------|---------|
| `detect_format_from_content(path)` | Identify format from file | Format string |
| `validate_image_integrity(path)` | Check if image valid | (bool, error) |
| `get_image_metadata(path)` | Extract all metadata | Dict |
| `get_mime_type(path)` | Get MIME type | String |
| `convert_image_format(in, out, quality)` | Convert single file | Bool |
| `batch_convert_directory(dir, out, fmt)` | Convert directory | Stats dict |
| `smart_convert(in, out)` | Auto-optimize convert | Bool |
| `save_tiff_16bit(array, path, comp)` | Save 16-bit TIFF | Bool |
| `load_tiff_preserve_depth(path)` | Load preserving depth | (array, depth) |
| `check_tifffile_available()` | Check for 16-bit support | Bool |

---

## ⚡ Quick Examples

### Example 1: Safe Processing

```python
# Validate before expensive processing
if not validate_image_integrity(input_file)[0]:
    print("Skipping corrupted file")
    return

# Now safe to process
result = process_image(input_file)
```

### Example 2: Batch Job

```python
# Convert entire project to web format
stats = batch_convert_directory(
    './renders',
    './web',
    '.webp',
    quality=90,
    recursive=True
)
```

### Example 3: 16-bit Workflow

```python
# Load, enhance, save as 16-bit
array, depth = load_tiff_preserve_depth('luxury.tiff')
enhanced = enhance_image(array)
save_tiff_16bit(enhanced, 'output.tiff', compression='lzw')
```

---

## 🔍 When to Use What

| Task | Use This Function |
|------|-------------------|
| Check if file is really a JPEG | `detect_format_from_content()` |
| Validate before AI processing | `validate_image_integrity()` |
| Convert 100 TIFFs to JPEGs | `batch_convert_directory()` |
| Check bit depth of TIFF | `get_image_metadata()` |
| Preserve 16-bit quality | `load_tiff_preserve_depth()` |
| Auto-choose best format | `get_optimal_format_for_use_case()` |

---

## 💡 Tips

1. **Always validate** before expensive operations
2. **Check bit depth** for luxury format workflows
3. **Use batch convert** for entire directories
4. **Preserve metadata** when converting formats
5. **Install tifffile** for 16-bit support

---

## 🔧 Installation

```bash
# Basic (required)
pip install Pillow numpy

# For 16-bit TIFF support (recommended)
pip install tifffile imagecodecs

# Or use extras
pip install -e ".[tiff]"
```

---

## 📖 Full Documentation

See [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) for:
- Complete usage examples
- Integration patterns
- Performance tips
- Troubleshooting

---

## ✅ Checklist

Before using in production:

- [ ] Tests pass: `pytest tests/test_format_utils_enhancements.py -v`
- [ ] tifffile installed (if using 16-bit)
- [ ] Imported in format_utils.py
- [ ] Tried example conversions
- [ ] Read integration guide

---

**Print this page and keep it handy!** 📄
