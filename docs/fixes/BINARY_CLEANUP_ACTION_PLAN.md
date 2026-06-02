# Binary File Cleanup - Action Plan

**Date**: 2025-11-06
**Branch**: feat/rag-integration-complete
**Issue**: 356MB PNG preview files currently being pushed

---

## 🎯 IMMEDIATE DECISION: LET PUSH COMPLETE ✅

**Action**: Allow current `git push origin feat/rag-integration-complete` to finish

**Why This is the Right Call**:
1. ✅ Files are already in Git history (commit d3b26a3)
2. ✅ Aborting creates churn without fixing root problem
3. ✅ Incremental cleanup is safer and more predictable
4. ✅ Repo is already 2.3GB - 356MB won't catastrophically impact clones
5. ⚠️ Privacy concern exists but can be addressed immediately after

**Next Steps**: Execute cleanup immediately after push completes

---

## 📝 Step-by-Step Cleanup (Execute After Push)

### Step 1: Remove PNG Previews from Tracking

```bash
cd /Users/rc/Transformation_Portal

# Ensure you're on the correct branch
git branch --show-current
# Expected: feat/rag-integration-complete

# Remove PNG previews from Git index (keeps local files intact)
git rm --cached input_images/*.png

# Verify removal (should show "deleted" in staged changes)
git status

# Expected output:
#   deleted:    input_images/750Picacho_Ready.png
#   deleted:    input_images/Coastal_Interior_2_preview.png
#   deleted:    input_images/Coastal_Interior_3_preview.png
#   (... 8 PNG files total)
```

### Step 2: Update .gitignore

```bash
# Append the new patterns to .gitignore
cat .gitignore.additions >> .gitignore

# Verify the additions
tail -70 .gitignore

# Remove the temporary file
rm .gitignore.additions
```

### Step 3: Commit and Push Cleanup

```bash
# Commit the cleanup
git add .gitignore
git commit -m "fix: Remove PNG preview files from Git tracking (356MB)

- PNG previews are 40-55MB each (356MB total)
- These are derivatives of TIFF client files (already excluded via c47bbc9)
- Privacy concern: contain client proprietary imagery
- Should be generated locally as needed

Files removed from tracking:
- input_images/750Picacho_Ready.png (13MB)
- input_images/Coastal_Interior_*_preview.png (40-55MB each, 7 files)

Updated .gitignore to prevent future commits of:
- PNG previews in input_images/
- Large processed outputs
- Video files
- RAW camera files

Related commits:
- c47bbc9: Excluded TIFF client files
- d3b26a3: Added RAG system (where PNGs were initially committed)

See BINARY_FILE_BEST_PRACTICES.md for comprehensive guidelines."

# Push the cleanup
git push origin feat/rag-integration-complete
```

### Step 4: Verify Cleanup

```bash
# Check what's still tracked in input_images/
git ls-files input_images/

# Expected: Nothing (all images excluded)
# Or: Only input_images/.gitkeep if you add it

# Verify local files still exist
ls -lh input_images/*.png

# Expected: All 8 PNG files still present locally
```

---

## 🧹 Optional: Additional Cleanup

### Option A: Move Processed Examples to docs/

If the processed examples are needed for README:

```bash
# Create examples directory
mkdir -p docs/examples/

# Move relevant processed images (selective)
git mv processed_images/750_Picacho_Pool_MBAR_Enhanced.jpg docs/examples/enhancement_example_pool.jpg
git mv processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg docs/examples/enhancement_example_aerial.jpg

# Update README.md to reference new paths
# (Manual edit required)

# Commit
git add README.md
git commit -m "docs: Move processed image examples to docs/examples/

- Reduced clutter in processed_images/
- Better organization for documentation assets
- Examples now < 200KB each (resized for web)"

git push origin feat/rag-integration-complete
```

### Option B: Remove Processed Examples Entirely

If examples can be regenerated:

```bash
# Remove from tracking (but keep markdown docs)
git rm processed_images/*.jpg processed_images/*.tiff

# Commit
git commit -m "chore: Remove processed image examples from Git

- These are reproducible outputs
- Can be regenerated via pipelines as needed
- Reduces repository size by ~95MB

Markdown documentation retained:
- processed_images/MBAR_Enhancement_Report.md
- processed_images/Pool_MBAR_Enhancement_Report.md"

git push origin feat/rag-integration-complete
```

---

## 🚀 Future Prevention

### Add .gitkeep Files

```bash
# Create .gitkeep files to preserve directory structure
touch input_images/.gitkeep
touch data/sample_images/.gitkeep
touch processed_images/.gitkeep
touch output/.gitkeep

# Add to Git
git add .gitkeep
git commit -m "chore: Add .gitkeep files to preserve directory structure

- Ensures directories exist after clone
- Prevents errors when pipelines expect output directories"

git push origin feat/rag-integration-complete
```

### Test the Download Script

```bash
# Make download script executable
chmod +x scripts/download_samples.py

# List available samples
python scripts/download_samples.py --list

# Test download (will show TODOs for unavailable URLs)
python scripts/download_samples.py

# Expected: Creates tests/fixtures/ with placeholder test images
```

---

## 📊 Validation Checklist

After all cleanup steps:

```bash
# ✅ 1. Verify .gitignore is working
git status
# Expected: No PNG/TIFF files shown in input_images/

# ✅ 2. Check tracked binary files
git ls-files | grep -E '\.(png|tiff?|jpg|jpeg)$'
# Expected: Only assets/brand/ and assets/textures/board_materials/ (if kept)

# ✅ 3. Verify local files still exist
ls -lh input_images/*.png
# Expected: All 8 PNG files present

ls -lh input_images/*.tiff
# Expected: All TIFF files present

# ✅ 4. Check repository size reduction
du -sh .git/
# Expected: Unchanged initially (cleanup doesn't rewrite history)
# Future clones will be smaller after Git gc

# ✅ 5. Test clone (optional)
cd /tmp
time git clone --depth 1 git@github.com:RC219805/Transformation_Portal.git test-clone
cd test-clone
ls input_images/
# Expected: Only .gitkeep (no large files)
```

---

## 🔐 Privacy Verification

```bash
# Verify no client files are tracked
git ls-files | grep -i "picacho\|coastal"

# Expected: Only markdown docs, no .tiff or .png client files

# If any client files found, remove them:
git rm --cached <file>
git commit -m "fix: Remove client file from tracking (privacy)"
```

---

## 📅 Timeline

| Step | Duration | Status |
|------|----------|--------|
| Current push completes | ~5-10 min | ⏳ In progress |
| Step 1: Remove PNGs | 2 min | ⏭️ Ready |
| Step 2: Update .gitignore | 1 min | ⏭️ Ready |
| Step 3: Commit & push | 2 min | ⏭️ Ready |
| Step 4: Validation | 3 min | ⏭️ Ready |
| **Total** | **15 min** | |

---

## 🎓 Key Learnings

### What Went Wrong
1. PNG previews (derivatives) committed alongside RAG system changes
2. No .gitignore patterns for PNG previews (TIFF exclusion was incomplete)
3. Large binaries slipped through review in feature branch

### What Went Right
1. ✅ TIFF files excluded via c47bbc9 (good instinct)
2. ✅ ML models already excluded (working well)
3. ✅ Caught before merge to main (easier to fix)

### How to Prevent
1. ✅ Comprehensive .gitignore (now includes PNG, video, RAW)
2. ✅ Pre-commit hook (future: warn on files > 1MB)
3. ✅ Documentation (BINARY_FILE_BEST_PRACTICES.md)
4. ✅ Download scripts (scripts/download_samples.py template)

---

## 📞 Support

If issues arise during cleanup:

```bash
# Abort cleanup and return to pre-cleanup state
git reset --hard HEAD~1
git push origin feat/rag-integration-complete --force

# Then review this action plan again
```

**Need help?** See `BINARY_FILE_BEST_PRACTICES.md` for comprehensive guidance.

---

## ✅ Success Criteria

After cleanup is complete:

- [ ] PNG previews removed from Git tracking (`git ls-files input_images/` shows nothing)
- [ ] Local PNG files still exist (`ls input_images/*.png` shows 8 files)
- [ ] .gitignore updated with comprehensive patterns
- [ ] Cleanup commit pushed to `feat/rag-integration-complete`
- [ ] Validation checklist passed (all ✅)
- [ ] No client files in `git ls-files` output
- [ ] `scripts/download_samples.py` is executable and lists samples

**When all checkboxes are complete**: Cleanup is done! 🎉

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Status**: ✅ READY TO EXECUTE
