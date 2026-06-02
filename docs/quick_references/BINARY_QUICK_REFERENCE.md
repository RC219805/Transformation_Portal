# Binary File Best Practices - Quick Reference

**TL;DR**: Let the push complete. Fix incrementally. 356MB PNG previews should be excluded.

---

## 🎯 Decision for Current Push

**✅ LET IT COMPLETE** - Then remove PNG previews from tracking in next commit

---

## 📏 Size Thresholds

| Type | Max Size | Action |
|------|----------|--------|
| **Brand assets** | 100KB | ✅ Include |
| **Test fixtures** | 50KB | ✅ Include |
| **LUT files** | 500KB | ✅ Include |
| **Documentation** | 200KB | ✅ Include |
| **Material textures** | 500KB | ⚠️ Selective |
| **Sample images** | 1-10MB | ❌ External download |
| **Preview files** | 10-100MB | ❌ **Exclude** |
| **Production files** | > 100MB | ❌ **Exclude** |
| **ML models** | > 50MB | ❌ **Exclude** |

---

## 🗂️ Directory Policy

```
✅ TRACK IN GIT:
  assets/brand/               # Logo SVG (< 100KB)
  assets/luts/                # Color grading LUTs (< 500KB)
  assets/textures/board_materials/   # Essential materials (< 500KB each)
  tests/fixtures/             # Tiny synthetic images (< 50KB)
  docs/examples/              # Screenshots (< 200KB)

❌ EXCLUDE FROM GIT:
  input_images/               # Client production files
  output/                     # Generated outputs
  processed_images/           # Reproducible results
  data/sample_images/         # Downloaded via scripts
  models/                     # ML weights (download scripts)
```

---

## 🔧 Quick Fix Commands

```bash
# After current push completes:

# 1. Remove PNG previews from tracking
git rm --cached input_images/*.png

# 2. Update .gitignore
cat .gitignore.additions >> .gitignore
rm .gitignore.additions

# 3. Commit and push
git commit -m "fix: Remove PNG preview files from Git tracking (356MB)"
git push origin feat/rag-integration-complete
```

---

## 📦 What's Currently Tracked

```
input_images/*.png              356MB  ⚠️ REMOVE (privacy + size)
processed_images/*.jpg           15MB  ⚠️ Consider removing
processed_images/*.tiff          80MB  ⚠️ Consider removing
assets/textures/board_materials/*.png    4MB  KEEP (essential assets)
assets/brand/lantern_logo.svg   50KB  ✅ KEEP (brand asset)
```

---

## ⚠️ Privacy Concerns

**Client files contain**:
- GPS coordinates (property locations)
- IPTC metadata (photographer, agency)
- Unreleased architectural designs
- Proprietary rendering techniques

**Action**: Remove from public repository immediately after push

---

## 📚 Full Documentation

- **Comprehensive guide**: `BINARY_FILE_BEST_PRACTICES.md`
- **Step-by-step plan**: `BINARY_CLEANUP_ACTION_PLAN.md`
- **Download script**: `scripts/download_samples.py`
- **Updated patterns**: `.gitignore.additions`

---

## ✅ Success Criteria

- [ ] PNG previews removed from tracking
- [ ] .gitignore updated
- [ ] Local files still exist
- [ ] No client files in `git ls-files`
- [ ] Documentation updated

---

**Status**: ✅ Ready to execute after current push
**Time Required**: ~15 minutes
**Impact**: -356MB in future clones, improved privacy
