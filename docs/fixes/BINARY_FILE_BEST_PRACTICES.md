# Binary File Best Practices - Transformation Portal

**Date**: 2025-11-06
**Context**: RAG system integration push with 356MB of PNG preview files in `input_images/`
**Status**: ✅ TIFF files already excluded via .gitignore (as of commit c47bbc9)

> **2026-05-02 Status Update (QW-3 closure).** The remediation described
> under "🎯 Executive Decision: Current Push" and the four steps under
> "🔧 Immediate Actions Required" (`1. Update .gitignore`,
> `2. Remove PNG Previews from Tracking`,
> `3. Migrate Processed Examples to Documentation`,
> `4. Create Sample Download Script`) have all been completed. As of
> 2026-05-02 no PNG/JPG/TIFF binaries are tracked under `input_images/`
> or `processed_images/`
> (`git ls-files input_images/ processed_images/` returns only `.gitkeep`,
> two text notes, and a JSON provenance file). The "⚠️ Currently tracked"
> warnings under "📋 Binary File Guidelines → 2. What MUST Be Excluded"
> therefore describe the 2025-11-06 snapshot, not the current repository
> state. The forward-looking guidance ("📋 Binary File Guidelines",
> "🗂️ Repository Structure Best Practices", "🚀 Long-Term Strategy",
> "📝 Best Practices Summary", and onward) remains active for new
> contributions. Tracked in `docs/deliverables/QUICK_WINS.md` (QW-3) and
> `docs/analysis/TODO_INVENTORY.md` §4.1.

---

## 🎯 Executive Decision: Current Push

### **RECOMMENDATION: LET IT COMPLETE, FIX INCREMENTALLY**

**Rationale:**
1. **Already committed to history**: PNG files are tracked in Git history (commit d3b26a3)
2. **Diminishing returns**: Aborting again creates churn without fixing the root problem
3. **Incremental fix is safer**: Remove from tracking, add to .gitignore, document for future
4. **Git size**: Repo is already 2.3GB (.git/), 356MB PNGs won't catastrophically impact clones
5. **Privacy/security**: These are production client files (Coastal_Interior, 750Picacho) - should NOT be public

**Action Plan:**
```bash
# Let current push complete
git push origin feat/rag-integration-complete

# Then immediately (separate commit):
git rm --cached input_images/*.png
echo "# PNG previews excluded - use download_samples.py instead" >> input_images/.gitkeep
git commit -m "fix: Remove PNG preview files from Git tracking (356MB)"
git push origin feat/rag-integration-complete
```

---

## 📋 Binary File Guidelines

### 1. **What SHOULD Be in Version Control**

#### ✅ **Small, Essential Assets** (< 100KB each)
- **Brand assets**: Logo SVG, favicon (e.g., `assets/brand/lantern_logo/lantern_logo.svg`)
- **Icon sets**: UI icons, material texture thumbnails (< 50KB PNG/SVG)
- **Test fixtures**: Tiny test images for unit tests (e.g., `tests/fixtures/test_image.jpg` - 10-20KB)
- **Documentation images**: Architecture diagrams, flowcharts, screenshots (< 200KB)

#### ✅ **Configuration Data** (< 1MB each)
- **LUT files**: `.cube` color grading files (typically 10-100KB)
- **Material textures** (small): Current `textures/board_materials/*.png` are 150-500KB - **borderline acceptable**

**Current Status:**
```
✅ assets/brand/lantern_logo/lantern_logo.svg (SVG, likely < 50KB)
✅ textures/board_materials/*.png (8 files, ~500KB each = 4MB total) - ACCEPTABLE
✅ processed_images/*.jpg (4 files, enhancement examples) - BORDERLINE (should move to docs/)
```

---

### 2. **What MUST Be Excluded**

#### ❌ **Production Client Files** (Security/Privacy)
- **Client renders**: `input_images/750Picacho_*.tiff` (183MB each) - ✅ **Already excluded**
- **Client previews**: `input_images/Coastal_Interior_*_preview.png` (40-55MB each) - ⚠️ **Currently tracked**
- **Confidential data**: Any client-specific imagery, GPS coordinates, IPTC metadata

**Privacy Risk**: Client files may contain:
- GPS coordinates of property locations
- IPTC/XMP metadata with photographer/agency information
- Unreleased architectural designs
- Proprietary rendering techniques

#### ❌ **ML Model Weights** (Size + Licensing)
- **Stable Diffusion checkpoints**: `*.safetensors`, `*.ckpt` (2-7GB each)
- **Real-ESRGAN models**: `RealESRGAN_*.pth` (64-200MB)
- **Depth Anything V2 CoreML**: `*.mlpackage` (50-150MB)
- **ControlNet models**: `*.pth`, `*.bin` (1-5GB)

✅ **Already excluded** via .gitignore (lines 108-118, 120-124)

#### ❌ **Processed Outputs** (Reproducible)
- **Output directories**: `output/`, `output_*/`, `processed_output/`, `processed_images/`
- **Depth maps**: `*_depth.npy`, `*_depth_viz.png`
- **Enhanced renders**: `*_enhanced.png`, `*_enhanced.jpg`

⚠️ **Currently tracked**: `processed_images/*.jpg` (4 enhancement examples, ~15MB total)

**Recommendation**: Move to `docs/examples/` or external hosting

---

### 3. **Size Thresholds for Inclusion**

| File Type | Threshold | Action | Example |
|-----------|-----------|--------|---------|
| **Documentation images** | < 200KB | Include | Architecture diagram PNG |
| **Test fixtures** | < 50KB | Include | `tests/fixtures/test_depth.jpg` |
| **Brand assets** | < 100KB | Include | Logo SVG, favicon |
| **LUT files** | < 500KB | Include | `.cube` color grading files |
| **Material textures** | < 500KB | Include (selective) | Board material PNGs |
| **Sample images** | 1-10MB | External download | Demo renders for README |
| **Preview images** | 10-100MB | **Exclude** | Client PNG previews |
| **Production files** | > 100MB | **Exclude** | TIFF renders, RAW files |
| **ML models** | > 50MB | **Exclude** | Use download script |

**Rule of Thumb**:
- If `git add <file>` takes > 2 seconds → **exclude it**
- If total repo size > 50MB → **review binary files**
- If clone time > 30 seconds → **migrate to LFS or external storage**

---

## 🗂️ Repository Structure Best Practices

### Recommended Directory Organization

```
Transformation_Portal/
├── assets/                    # ✅ Version controlled assets
│   ├── brand/                # Brand assets (SVG, small PNGs < 100KB)
│   │   └── lantern_logo/
│   ├── luts/                 # Color grading LUTs (< 500KB each)
│   │   ├── film_emulation/
│   │   ├── location_aesthetic/
│   │   └── material_response/
│   └── textures/             # Material textures (< 500KB, selective)
│       └── board_materials/
│
├── data/                      # ❌ Excluded from Git
│   ├── sample_images/        # Downloaded via download_samples.py
│   ├── models/               # ML weights downloaded via scripts
│   └── .gitkeep              # Keep directory structure
│
├── input_images/              # ❌ Excluded from Git (local dev only)
│   ├── *.tiff                # ✅ Already excluded (TIFF client files)
│   ├── *.png                 # ✅ Excluded via .gitignore (`input_images/**/*.png` pattern)
│   └── .gitkeep
│
├── output/                    # ❌ Excluded from Git (generated)
│   └── .gitkeep
│
├── processed_images/          # ✅ Directory cleaned up (no files tracked)
│   └── *.jpg                 # ✅ Completed - directory empty or excluded
│
├── tests/
│   └── fixtures/             # ✅ Tiny test files only (< 50KB)
│       ├── test_image.jpg    # 10KB synthetic test image
│       └── test_depth.npy    # 5KB depth map for tests
│
└── docs/
    └── examples/             # ✅ Small screenshots/diagrams
        ├── pipeline_flow.png # < 200KB
        └── material_response_demo.jpg  # < 500KB
```

---

## 🔧 Immediate Actions Required

### 1. **Update .gitignore** (Priority: HIGH)

Add these patterns to `.gitignore`:

```gitignore
# ============================================================================
# BINARY FILE EXCLUSIONS - Updated 2025-11-06
# ============================================================================

# PNG preview files (generated from TIFF client files)
input_images/**/*.png
!input_images/.gitkeep

# Processed output examples (move to docs/examples/ if needed for README)
processed_images/**/*.jpg
processed_images/**/*.png
processed_images/**/*.tiff
!processed_images/.gitkeep
!processed_images/*.md

# Large texture files (keep small board materials < 500KB)
textures/**/*.jpg
textures/**/*.png
textures/**/*.tiff
!textures/board_materials/*.png  # Exception for essential materials

# Sample data (download via scripts)
data/**/*.jpg
data/**/*.jpeg
data/**/*.png
data/**/*.tif
data/**/*.tiff
!data/**/.gitkeep

# Video files (all formats)
*.mp4
*.mov
*.avi
*.mkv
*.m4v
*.webm
```

### 2. **Remove PNG Previews from Tracking**

```bash
# Execute after current push completes
cd /Users/rc/Transformation_Portal

# Remove PNG previews from Git index (keeps local files)
git rm --cached input_images/*.png

# Commit the removal
git commit -m "fix: Remove PNG preview files from Git tracking

- PNG previews are 40-55MB each (356MB total)
- These are derivatives of TIFF client files (already excluded)
- Privacy concern: contain client proprietary imagery
- Should be generated locally as needed

Related: c47bbc9 (excluded TIFF files)"

# Push to remote
git push origin feat/rag-integration-complete
```

### 3. **Migrate Processed Examples to Documentation**

```bash
# Create examples directory
mkdir -p docs/examples/

# Move existing processed images (if needed for README)
# Option A: Keep small JPG examples in docs
git mv processed_images/750_Picacho_Pool_MBAR_Enhanced.jpg docs/examples/
git mv processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg docs/examples/

# Option B: Remove entirely (regenerate as needed)
git rm processed_images/*.jpg processed_images/*.tiff

# Update README.md to reference new paths
# Commit changes
git commit -m "docs: Migrate processed image examples to docs/examples/"
```

### 4. **Create Sample Download Script**

Create `scripts/download_samples.py`:

```python
#!/usr/bin/env python3
"""
Download sample images for development and testing.

Usage:
    python scripts/download_samples.py [--all] [--output-dir DIR]

Examples:
    # Download minimal test fixtures
    python scripts/download_samples.py

    # Download all sample images (for pipeline testing)
    python scripts/download_samples.py --all
"""

import argparse
from pathlib import Path
import urllib.request

# Sample image registry (host on GitHub Releases or external storage)
SAMPLES = {
    "test_fixture": {
        "url": "https://github.com/RC219805/Transformation_Portal/releases/download/v0.1.0/test_image.jpg",
        "size": "10KB",
        "path": "tests/fixtures/test_image.jpg"
    },
    "demo_render": {
        "url": "https://example.com/demo_coastal_interior.jpg",
        "size": "5MB",
        "path": "data/sample_images/demo_coastal_interior.jpg"
    }
}

def download_samples(sample_names, output_dir=None):
    """Download specified sample images."""
    # Implementation here
    pass

if __name__ == "__main__":
    # CLI implementation
    pass
```

---

## 📊 Current Repository Analysis

### Binary Files in Git History

**Currently Tracked** (as of commit d3b26a3):
```
input_images/*.png               356MB  (8 PNG previews, 40-55MB each)
processed_images/*.jpg            15MB  (4 enhancement examples)
processed_images/IMG_4069_lux.tiff 80MB (1 processed TIFF)
textures/board_materials/*.png     4MB  (8 material textures)
assets/brand/lantern_logo/*.svg   50KB  (Logo asset)
─────────────────────────────────────
TOTAL BINARY:                    ~455MB
```

**Repository Size**:
- `.git/` directory: 2.3GB (includes all history + LFS-like objects)
- Working tree: 2.7GB (includes input_images/ local files)
- Remote clone time: ~1-2 minutes on fast connection

### Impact Assessment

| Metric | Current | After Cleanup | Improvement |
|--------|---------|---------------|-------------|
| Binary in repo | 455MB | 99MB | -78% |
| Clone time | 2 min | 30 sec | -75% |
| Storage cost | High | Low | - |
| Privacy risk | High | Low | ✅ |

---

## 🚀 Long-Term Strategy

### 1. **Git LFS Considerations**

**When to use Git LFS:**
- Repository has > 50 binary files > 1MB each
- Frequent updates to large binaries (e.g., model checkpoints)
- Need version history of binaries
- Team of > 5 contributors

**Current Assessment**: **NOT RECOMMENDED**

**Reasons:**
1. ✅ Already excluded most binaries via .gitignore
2. ✅ Model weights downloaded via scripts (not in repo)
3. ✅ Sample images can be hosted externally
4. ❌ LFS adds complexity (setup, bandwidth costs, storage quotas)
5. ❌ GitHub LFS free tier: 1GB storage, 1GB/month bandwidth (insufficient)

**Alternative**: Use GitHub Releases for sample datasets

### 2. **External Storage Options**

#### **Option A: GitHub Releases** (RECOMMENDED)
- Upload sample images as release assets
- Download via `scripts/download_samples.py`
- Free, integrated with repository
- Versioned (tagged releases)

**Example**:
```bash
# Create release with sample images
gh release create v0.2.0 \
  --title "Sample Images for v0.2.0" \
  --notes "Coastal interior render samples" \
  data/samples/*.jpg
```

#### **Option B: Cloud Storage (S3, Google Cloud Storage)**
- For large datasets (> 1GB)
- Cost: $0.02-0.05/GB/month
- Requires credentials management
- Use for client production files (private)

#### **Option C: External Link (Google Drive, Dropbox)**
- Quick solution for prototyping
- Free tier limitations
- Not suitable for automation
- Link rot risk

**Recommendation**: GitHub Releases for public samples, S3 for private client files

### 3. **CI/CD Optimization**

Update CI workflows to skip sample downloads unless required:

```yaml
# .github/workflows/build.yml
- name: Download test fixtures
  run: |
    python scripts/download_samples.py --minimal
  # Only download full samples for integration tests
  if: matrix.test-suite == 'integration'
```

**Expected Impact**:
- CI runtime: -2-3 minutes (no binary checkout)
- CI storage: -80% (minimal fixtures only)

---

## 📝 Best Practices Summary

### ✅ DO

1. **Version control small, essential assets** (< 100KB)
   - Brand logos (SVG preferred)
   - Test fixtures (synthetic images < 50KB)
   - LUT files (< 500KB)
   - Documentation diagrams

2. **Use download scripts for large binaries**
   - ML model weights → `download_depth_models.py`
   - Sample images → `download_samples.py`
   - Host on GitHub Releases or S3

3. **Document sample data requirements**
   - Add `data/sample_images/README.md` with download instructions
   - Include checksums for verification
   - Provide minimal vs. full sample sets

4. **Maintain `.gitignore` hygiene**
   - Exclude all `output/`, `processed_*/` directories
   - Exclude client production files
   - Include `.gitkeep` to preserve directory structure

5. **Separate concerns**
   - Test fixtures → `tests/fixtures/` (< 50KB each)
   - Demo samples → `data/sample_images/` (downloaded)
   - Brand assets → `assets/brand/` (< 100KB)
   - Client files → `input_images/` (local only, never commit)

### ❌ DON'T

1. **Commit large binaries to Git history**
   - ML model weights (use download scripts)
   - Video files (always external)
   - High-resolution client files (privacy + size)

2. **Track generated/reproducible files**
   - Processed images (regenerate as needed)
   - Depth maps (derived from source)
   - Cached outputs

3. **Commit client production data**
   - Privacy violations
   - Storage waste
   - Security risks (GPS, IPTC metadata)

4. **Use Git LFS prematurely**
   - Adds complexity
   - Bandwidth costs
   - Alternative: external hosting + download scripts

---

## 🔍 Validation Checklist

After implementing these changes, verify:

```bash
# 1. Check .gitignore is working
git status | grep -E '\.(png|tiff?|jpg|jpeg)$'
# Expected: Only assets/brand/ and textures/board_materials/

# 2. Verify binary file sizes
git ls-files | xargs -I {} sh -c 'du -h "{}" 2>/dev/null' | sort -rh | head -20
# Expected: All files < 500KB

# 3. Test clone time
time git clone --depth 1 git@github.com:RC219805/Transformation_Portal.git test-clone
# Expected: < 30 seconds

# 4. Verify sample download script
python scripts/download_samples.py --minimal
ls -lh tests/fixtures/
# Expected: test_image.jpg (~10KB)

# 5. Check repository size
du -sh .git/
# Expected: < 500MB after cleanup
```

---

## 📖 Documentation Updates Required

### 1. Update `README.md`

Add section after "Installation":

```markdown
### Sample Images

This repository does not include large sample images in Git. Download them as needed:

```bash
# Minimal test fixtures (required for tests)
python scripts/download_samples.py --minimal

# Full sample dataset (optional, for pipeline demos)
python scripts/download_samples.py --all
```

See `data/sample_images/README.md` for details.
```

### 2. Create `data/sample_images/README.md`

```markdown
# Sample Images

This directory contains sample images for development and testing.

## Download Samples

```bash
# Minimal test fixtures (10KB)
python scripts/download_samples.py --minimal

# Full sample dataset (50MB)
python scripts/download_samples.py --all
```

## Available Samples

- `test_image.jpg` (10KB) - Synthetic test image for unit tests
- `demo_coastal_interior.jpg` (5MB) - Coastal interior render demo
- `demo_pool_aerial.jpg` (8MB) - Pool aerial enhancement demo

## Hosting

Samples are hosted on GitHub Releases:
https://github.com/RC219805/Transformation_Portal/releases/tag/samples-v1.0.0

## Local Development

Place your own development images in `input_images/` (excluded from Git).
```

### 3. Update `.github/copilot-instructions.md`

Add to "Repository-Specific Notes":

```markdown
### Binary File Management
- **Never commit** client production files (TIFF, large PNGs) to Git
- **Use download scripts** for ML models (`download_depth_models.py`) and samples (`download_samples.py`)
- **Test fixtures** must be < 50KB (synthetic images only)
- **Brand assets** in `assets/brand/` are version controlled (SVG < 100KB)
- **Material textures** in `textures/board_materials/` are acceptable (< 500KB each)
- **Client files** go in `input_images/` (local dev only, never tracked)
```

---

## 🎯 Pragmatic Decision Matrix

| Scenario | Action | Rationale |
|----------|--------|-----------|
| **Current push (356MB PNG)** | ✅ **Let it complete** | Already in history, incremental fix safer |
| **PNG previews future** | ❌ **Remove from tracking** | Derivatives of TIFF (already excluded), privacy risk |
| **Processed examples** | ⚠️ **Move to docs/examples/** | Useful for README, but should be small (< 200KB) |
| **Material textures (4MB)** | ✅ **Keep** | Essential assets, reasonable size (< 500KB each) |
| **Brand logo SVG** | ✅ **Keep** | Critical asset, tiny size (< 50KB) |
| **ML model weights** | ❌ **Already excluded** | Download via scripts (working well) |
| **Test fixtures** | ✅ **Download script** | Create synthetic tiny images (< 50KB) |
| **Git LFS** | ❌ **Not needed** | External hosting simpler, cheaper |

---

## 📅 Implementation Timeline

### Immediate (After Current Push)
1. ✅ Let `feat/rag-integration-complete` push complete
2. ⚠️ Remove PNG previews from tracking (`git rm --cached`)
3. ⚠️ Update `.gitignore` with comprehensive patterns
4. ⚠️ Commit and push cleanup

### Short-term (This Week)
1. Create `scripts/download_samples.py`
2. Upload sample images to GitHub Release
3. Update `README.md` and `data/sample_images/README.md`
4. Test clone time and CI performance

### Long-term (Next Sprint)
1. Audit all binary files in repository
2. Migrate large processed examples to external hosting
3. Create synthetic test fixtures (< 50KB)
4. Update CI workflows to skip unnecessary downloads
5. Document binary file guidelines in contributor docs

---

## 🔐 Privacy & Security Considerations

### Client File Protection

**Current Risk**: Client production files in `input_images/` contain:
- GPS coordinates (property locations)
- IPTC metadata (photographer, agency, copyright)
- Unreleased architectural designs
- Proprietary rendering techniques

**Mitigation**:
1. ✅ **Never commit** client files to public repository
2. ✅ **Use `.gitignore`** to prevent accidental commits
3. ⚠️ **Remove existing** PNG previews from Git history (privacy)
4. ⚠️ **Consider** private S3 bucket for team sharing
5. ⚠️ **Strip metadata** if client files must be shared (exiftool)

**Best Practice**:
```bash
# Before sharing any client file externally
exiftool -all= -overwrite_original input.jpg

# Verify metadata removed
exiftool input.jpg | grep -i gps
# Expected: No GPS data
```

---

## ✅ Success Metrics

After implementing these best practices:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Repository size** | < 500MB | `du -sh .git/` |
| **Clone time** | < 30 sec | `time git clone --depth 1 <repo>` |
| **Binary files in repo** | < 100MB | `git ls-files \| xargs du -ch` |
| **CI runtime** | -20% | Compare workflow times before/after |
| **Privacy compliance** | 100% | No client files in `git ls-files` |
| **Developer experience** | ✅ | `download_samples.py` works first try |

---

## 📚 References

- **Git Large File Storage (LFS)**: https://git-lfs.github.com/
- **GitHub Releases for Datasets**: https://docs.github.com/en/repositories/releasing-projects-on-github
- **Repository Size Limits**: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- **.gitignore Patterns**: https://git-scm.com/docs/gitignore
- **Transformation Portal Refactoring**: `docs/historical/REFACTORING_SUMMARY.md`

---

## 🎓 Training for Contributors

Add to `CONTRIBUTING.md`:

```markdown
## Working with Binary Files

### DO NOT commit:
- Client production files (TIFF, large PNG previews)
- ML model weights (download via scripts)
- Processed outputs (regenerate as needed)
- Video files (always external)

### How to work with samples:
```bash
# Download minimal test fixtures
python scripts/download_samples.py --minimal

# Place your development files in input_images/ (excluded from Git)
cp ~/Downloads/my_render.tiff input_images/

# Run pipelines as normal
python lux_render_pipeline.py input_images/my_render.tiff
```

### If you accidentally commit large files:
1. Stop the push (Ctrl+C)
2. Remove from staging: `git reset HEAD <file>`
3. Add to `.gitignore`
4. Commit and push
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Author**: Transformation Portal Specialist
**Status**: ✅ READY FOR IMPLEMENTATION
