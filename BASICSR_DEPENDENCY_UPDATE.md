# BasicSR Dependency Update - Summary

**Date:** November 13, 2025  
**Branch:** `basicsr-explicit`  
**Base Branch:** `origin/copilot/fine-tune-dependency-management`

---

## ✅ Changes Completed

### What Was Done

Added **explicit BasicSR dependency** to the layered requirements system to replace implicit transitive dependency via Real-ESRGAN.

### Files Modified

1. **requirements/ml.in** - Source file
   - Added: `basicsr>=1.4.2,<2` (before realesrgan)
   - Comment: "Explicit dependency (required by realesrgan)"

2. **requirements/ml.txt** - Compiled ML dependencies
   - Pinned: `basicsr==1.4.2`
   - Includes all BasicSR transitive dependencies

3. **requirements/all.txt** - Global compiled dependencies
   - Updated with BasicSR and its full dependency tree

4. **requirements/base.txt, ci.txt, dev.txt**
   - Recompiled for consistency with updated global resolution

---

## Why This Change?

### Problem
- BasicSR was **implicitly installed** as a transitive dependency of Real-ESRGAN
- No explicit version control or pinning
- Could lead to version conflicts or unexpected updates
- Harder to debug dependency issues

### Solution  
- **Explicit declaration** in `requirements/ml.in`
- Version constraint: `>=1.4.2,<2`
- Compiled to pinned version: `basicsr==1.4.2`

### Benefits
✅ **Version Control** - Explicit version constraints prevent surprises  
✅ **Dependency Clarity** - Clear relationship between BasicSR and Real-ESRGAN  
✅ **Better Resolution** - pip-compile can optimize dependency tree  
✅ **Easier Debugging** - Issues traced to specific versions  
✅ **Documentation** - Dependencies are self-documenting  

---

## Technical Details

### BasicSR 1.4.2 Dependencies (from compiled ml.txt)

```
basicsr==1.4.2
├── addict (via basicsr)
├── future (via basicsr)  
├── lmdb (via basicsr)
├── opencv-python (via basicsr, realesrgan, others)
├── pillow (via basicsr, others)
├── pyyaml (via basicsr, others)
├── requests (via basicsr)
├── scikit-image (via basicsr)
├── scipy (via basicsr, others)
├── tb-nightly (via basicsr)
├── torch (via basicsr, others)
├── torchvision (via basicsr, others)
├── tqdm (via basicsr, others)
└── yapf (via basicsr)
```

### Compilation Process

```bash
cd requirements/
make compile
```

This runs:
1. `pip-compile all.in -o all.txt` (global resolution)
2. `pip-compile -c all.txt ml.in -o ml.txt` (ML deps constrained by global)
3. Recompiles base, ci, dev with updated constraints

---

## Requirements File Structure

```
requirements/
├── *.in files - Source requirements with version ranges
│   ├── all.in - References all other .in files
│   ├── base.in - Core runtime dependencies
│   ├── ml.in - ML/AI dependencies (✨ BasicSR added here)
│   ├── dev.in - Development tools
│   └── ci.in - CI/testing dependencies
│
└── *.txt files - Compiled pinned versions (auto-generated)
    ├── all.txt - All dependencies with exact pins
    ├── base.txt - Core with exact pins
    ├── ml.txt - ML with exact pins (basicsr==1.4.2)
    ├── dev.txt - Dev tools with exact pins
    └── ci.txt - CI with exact pins
```

---

## Installation

### For ML Features (includes BasicSR)
```bash
pip install -r requirements/base.txt -r requirements/ml.txt
```

### For Everything
```bash
pip install -r requirements/all.txt
```

### Legacy Support
The root-level `requirements.txt` now points to the layered system:
```bash
pip install -r requirements.txt  # Uses requirements/base.txt
```

---

## Verification

### Check BasicSR is installed:
```bash
python -c "import basicsr; print(basicsr.__version__)"
# Expected: 1.4.2
```

### Check dependency tree:
```bash
pip show basicsr
# Should show version 1.4.2 and its dependencies
```

### Verify in requirements:
```bash
grep basicsr requirements/ml.txt
# basicsr==1.4.2
```

---

## Git Commit

**Commit:** `fe592c5`  
**Branch:** `basicsr-explicit`

**Commit Message:**
```
Add explicit BasicSR dependency to ML requirements

- Add basicsr>=1.4.2,<2 to requirements/ml.in
- Previously relied on transitive dependency via realesrgan
- Explicit version ensures compatibility and prevents conflicts
- Compiled all requirements files with pip-compile

Files changed:
- requirements/ml.in: Added basicsr>=1.4.2,<2
- requirements/*.txt: Recompiled with basicsr==1.4.2

Benefits:
- Explicit version control for Real-ESRGAN dependency
- Prevents unexpected BasicSR version changes
- Better dependency resolution and debugging
```

**Files Changed:** 6 files, +1115/-184 lines

---

## Next Steps

1. ✅ BasicSR explicitly added to `requirements/ml.in`
2. ✅ All requirements compiled successfully
3. ✅ Changes committed to `basicsr-explicit` branch
4. ⏳ **Ready for:** PR to merge into `copilot/fine-tune-dependency-management`
5. ⏳ **Then:** Merge dependency management branch into `main`

---

## Related Documentation

- `requirements/README.md` - Layered dependency system guide
- `docs/LAYERED_DEPENDENCIES_IMPLEMENTATION.md` - Implementation details
- Real-ESRGAN docs: https://github.com/xinntao/Real-ESRGAN
- BasicSR docs: https://github.com/XPixelGroup/BasicSR

---

**Status:** ✅ **Complete and ready for review**
