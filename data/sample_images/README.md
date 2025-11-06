# Sample Images

This directory contains sample images for **Transformation Portal** development and testing.

> ⚠️ **Note**: Sample images are **not included in Git** to keep repository size small. Download them using the provided script.

---

## 📥 Download Samples

### Quick Start

```bash
# Download minimal test fixtures (required for unit tests)
python scripts/download_samples.py

# Download all samples including demos
python scripts/download_samples.py --all

# List available samples
python scripts/download_samples.py --list
```

---

## 📦 Sample Categories

### Minimal (< 50KB total)
**Required for unit tests** - Tiny synthetic images

- `test_image_small.jpg` (1KB) - 100x100px test image
- `test_depth.jpg` (5KB) - 256x256px grayscale depth map

### Demo (~10MB total)
**For README examples** - Downscaled renders

- `demo_coastal_interior.jpg` (5MB) - Coastal interior render at 2K
- `demo_pool_aerial.jpg` (8MB) - Pool aerial enhancement at 2K

### Full (~50MB total)
**For pipeline testing** - Complete sample dataset

- `sample_render_4k.tiff` (25MB) - 4K architectural render (16-bit TIFF)
- `sample_depth.npy` (2MB) - Pre-computed Depth Anything V2 depth map
- Additional sample images for batch processing tests

---

## 🗂️ Directory Structure

After downloading samples:

```
data/sample_images/
├── README.md                        # This file
├── demo_coastal_interior.jpg        # Demo render
├── demo_pool_aerial.jpg             # Demo aerial
├── sample_render_4k.tiff            # Full resolution sample
└── depth_maps/
    └── sample_depth.npy             # Pre-computed depth map
```

---

## 🏠 Local Development Files

For your own development work:

- Use `input_images/` directory for your local development files
- This directory is excluded from Git (privacy + size)
- Place TIFF renders, RAW files, client imagery there

```bash
# Your workflow
cp ~/Downloads/my_render.tiff input_images/
python lux_render_pipeline.py input_images/my_render.tiff
```

---

## 🔗 Hosting

Sample images are hosted on **GitHub Releases**:

https://github.com/RC219805/Transformation_Portal/releases/tag/samples-v1.0.0

> **TODO**: Upload samples to GitHub Release when ready

---

## 📊 Size Reference

| Category | Total Size | Files | Purpose |
|----------|------------|-------|---------|
| Minimal | < 50KB | 2 | Unit tests (CI/CD) |
| Demo | ~10MB | 2 | README examples |
| Full | ~50MB | 5+ | Pipeline testing |

---

## 🚫 What's NOT Included

The following are excluded from Git and this download:

- **Client production files** (privacy concerns)
- **High-resolution renders** > 4K (use your own)
- **Video files** (always external)
- **ML model weights** (use `scripts/download_depth_models.py`)

---

## ✅ Verification

After downloading, verify checksums:

```bash
# The download script verifies SHA256 automatically
# If verification fails, re-run with --force

python scripts/download_samples.py --force
```

---

## 🔐 Privacy

All sample images are:

- ✅ Non-confidential demonstration renders
- ✅ No client proprietary information
- ✅ GPS and IPTC metadata stripped
- ✅ Publicly shareable

**Never commit client files to this repository.**

---

## 📚 Documentation

- **Binary file guidelines**: `BINARY_FILE_BEST_PRACTICES.md`
- **Download script**: `scripts/download_samples.py`
- **Pipeline operations**: `docs/PIPELINE_OPERATIONS_GUIDE.md`

---

## 🆘 Troubleshooting

### Download fails

```bash
# Check internet connection
curl -I https://github.com

# Try with force flag
python scripts/download_samples.py --force

# Check for proxy/firewall issues
echo $HTTP_PROXY
```

### Checksum mismatch

```bash
# Re-download the file
python scripts/download_samples.py --force

# If persistent, report issue:
# https://github.com/RC219805/Transformation_Portal/issues
```

### Samples not available

Some samples may show "TODO: upload to GitHub Release" - this means they haven't been uploaded yet. The script will skip them gracefully.

---

**Last Updated**: 2025-11-06  
**Sample Version**: v1.0.0 (planned)  
**Status**: 📋 Ready for sample upload
