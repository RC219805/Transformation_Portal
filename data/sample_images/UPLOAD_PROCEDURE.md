# Sample Image Upload Procedure

## Overview

Sample images are hosted on GitHub Releases to avoid bloating the Git repository with binary files. This document describes the upload process.

## Prerequisites

1. **GitHub CLI** installed and authenticated:
   ```bash
   # Install (macOS)
   brew install gh

   # Authenticate
   gh auth login
   ```

2. **Write permissions** to the repository

3. **Sample images** prepared and validated:
   - Downscaled to reasonable sizes (< 10MB per image ideal)
   - Representative of common use cases
   - Validated with processing pipelines

## Upload Procedure

### Step 1: Prepare Sample Images

Organize sample images in `data/sample_images/`:

```
data/sample_images/
├── test_image_small.jpg        # 100x100, < 1KB (unit tests)
├── test_depth.jpg              # 256x256, < 5KB (unit tests)
├── demo_coastal_interior.jpg   # 2048x1365, ~5MB (demo)
├── demo_pool_aerial.jpg        # 2048x1536, ~8MB (demo)
└── sample_render_4k.tiff       # 3840x2160, 16-bit TIFF (~25MB)
```

### Step 2: Generate SHA256 Hashes

```bash
python scripts/utilities/upload_samples_to_release.py \
    --samples-dir data/sample_images/ \
    --generate-hashes-only
```

This creates `data/sample_images/sample_manifest.json` with SHA256 hashes and sizes.

### Step 3: Create GitHub Release (One-time)

```bash
python scripts/utilities/upload_samples_to_release.py \
    --samples-dir data/sample_images/ \
    --release-tag samples-v1.0.0 \
    --repo RC219805/Transformation_Portal \
    --create-release
```

### Step 4: Upload Samples

```bash
python scripts/utilities/upload_samples_to_release.py \
    --samples-dir data/sample_images/ \
    --release-tag samples-v1.0.0 \
    --repo RC219805/Transformation_Portal
```

### Step 5: Update Registry

```bash
python scripts/utilities/upload_samples_to_release.py \
    --samples-dir data/sample_images/ \
    --release-tag samples-v1.0.0 \
    --repo RC219805/Transformation_Portal \
    --update-registry
```

This generates `data/sample_images/registry_updates.py` with code to copy into `scripts/download_samples.py`.

### Step 6: Update `download_samples.py`

Open `data/sample_images/registry_updates.py` and copy the generated code into `SAMPLE_REGISTRY` in `scripts/download_samples.py`:

```python
SAMPLE_REGISTRY: Dict[str, Dict] = {
    # ... existing entries ...

    # Copy generated entries here:
    "demo_coastal_interior.jpg": {
        "url": "https://github.com/RC219805/Transformation_Portal/releases/download/samples-v1.0.0/demo_coastal_interior.jpg",
        "sha256": "abc123...",
        "size": "5.2 MB",
        "path": "data/sample_images/demo_coastal_interior.jpg",
        "category": "demo",
        "description": "Coastal interior render (downscaled to 2K for demo)",
    },
}
```

### Step 7: Validate Downloads

```bash
python scripts/download_samples.py --all
```

Verify that all samples download successfully with matching SHA256 hashes.

### Step 8: Commit Changes

```bash
git add scripts/download_samples.py
git commit -m "feat(samples): Upload samples to GitHub Releases

- Added 6 sample images to samples-v1.0.0 release
- Updated SAMPLE_REGISTRY with verified URLs and SHA256 hashes
- All samples now downloadable via download_samples.py script"

git push
```

## Sample Image Guidelines

### Minimal (Test Fixtures)
- **Purpose**: Unit tests, CI validation
- **Size**: < 50KB total
- **Format**: JPEG (sufficient for tests)
- **Examples**: test_image_small.jpg (100x100px)

### Demo (README Examples)
- **Purpose**: Documentation, quick demos
- **Size**: ~10MB total
- **Format**: JPEG, downscaled to 2K
- **Examples**: demo_coastal_interior.jpg (2048x1365px)

### Full (Pipeline Testing)
- **Purpose**: Comprehensive validation, benchmarking
- **Size**: ~50MB total
- **Format**: 16-bit TIFF (preserve quality)
- **Examples**: sample_render_4k.tiff (3840x2160px, 16-bit)

## Troubleshooting

### "gh: command not found"
Install GitHub CLI: `brew install gh` (macOS) or see https://cli.github.com/

### "authentication failed"
Run `gh auth login` and follow prompts.

### "release not found"
Create release first with `--create-release` flag.

### SHA256 mismatch on download
- Re-upload file (may have been corrupted)
- Verify SHA256 hash in manifest matches uploaded file
- Check file wasn't modified after upload

## References

- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [Download Samples Script](../scripts/download_samples.py)
- [Upload Automation Script](../scripts/utilities/upload_samples_to_release.py)
