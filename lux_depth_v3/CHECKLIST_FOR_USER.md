# Integration Testing Checklist for User

**Status**: ✅ All preparation complete - Ready for your action
**Date**: 2026-01-02
**Estimated Time**: 20-30 minutes

---

## Quick Start (Recommended)

Run the interactive guide that will walk you through everything:

```bash
./QUICK_START_TESTING.sh
```

This will handle steps 1-5 below automatically with prompts.

---

## Manual Steps (If you prefer step-by-step)

### ☐ Step 1: Install Dependencies (5-15 min)

```bash
./INSTALL_DEPENDENCIES.sh
```

**What this does**:
- Detects your hardware (GPU/CPU)
- Installs PyTorch with correct variant
- Installs Depth Anything V3
- Installs NumPy, Pillow, pytest
- (Optional) Installs lux_depth_v2 if available

**Expected output**: "✓ Dependencies installed"

---

### ☐ Step 2: Verify Installation (30 sec)

```bash
python3 -c "import torch; import depth_anything_3; import numpy; import PIL; import pytest; print('✓ All OK')"
```

**Expected output**: `✓ All OK`

**If this fails**: Re-run INSTALL_DEPENDENCIES.sh or see TESTING_STATUS_REPORT.md

---

### ☐ Step 3: Run Integration Tests (2-5 min)

```bash
pytest tests/ -v
```

**Expected output**: All tests PASS

**If tests fail**:
- Check error messages carefully
- See TESTING_STATUS_REPORT.md Troubleshooting section
- Try: `pytest tests/ -v --tb=short` for shorter error messages

---

### ☐ Step 4: Generate Test Images (30 sec)

```bash
cd test_images
python3 generate_test_image.py
cd ..
```

**Expected output**: `✓ Created test_gradient.ppm`

**Alternative**: Copy your own images to `test_images/` directory

---

### ☐ Step 5: Run End-to-End Pipeline Test (1-3 min)

```bash
lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --model metric-large \
  --verbose
```

**Expected outputs**:
- `test_output/depth/` - Depth maps (U16 PNG)
- `test_output/v2/` - Enhanced images (if V2 installed)
- `test_output/manifests/` - Metadata (JSON)

**If this fails**:
- Check that test images exist: `ls test_images/`
- Try smaller model: `--model small`
- Check verbose output for specific errors

---

### ☐ Step 6: Validate Outputs (2-5 min)

```bash
# List outputs
ls -lh test_output/depth/
ls -lh test_output/v2/
ls -lh test_output/manifests/

# View depth map (macOS)
open test_output/depth/*.png

# View enhanced image (if V2 installed)
open test_output/v2/*.png

# Inspect manifest
cat test_output/manifests/*.json | python3 -m json.tool | less
```

**What to check**:
- ✅ Depth maps show reasonable depth gradients
- ✅ Enhanced images are visually acceptable
- ✅ Manifests contain complete metadata

---

## Success Criteria

All items should be checked:

- [ ] Dependencies installed successfully
- [ ] Verification command outputs "✓ All OK"
- [ ] All pytest tests PASS
- [ ] E2E test produces depth maps
- [ ] E2E test produces manifests
- [ ] Outputs are visually acceptable

**When all checked**: Integration testing is COMPLETE! 🎉

---

## If Something Goes Wrong

### Installation Issues

**Problem**: PyTorch won't install
**Solution**: Try CPU-only version
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: DA3 installation fails
**Solution**: Ensure PyTorch installed first
```bash
python3 -c "import torch; print(torch.__version__)"
pip install depth-anything-v3
```

### Test Issues

**Problem**: Import errors in tests
**Solution**: Verify all dependencies
```bash
python3 -c "import torch, depth_anything_3, numpy, PIL, pytest"
```

**Problem**: Tests timeout
**Solution**: Tests may be downloading models - wait or pre-download:
```bash
python3 -c "from depth_anything_3 import DepthAnything; DepthAnything.from_pretrained('metric-large')"
```

### E2E Issues

**Problem**: No test images
**Solution**: Generate them
```bash
cd test_images && python3 generate_test_image.py && cd ..
```

**Problem**: V2 outputs missing
**Solution**: V2 integration is optional - core V3 depth maps should still be generated

**Problem**: Out of memory
**Solution**: Use smaller model or images
```bash
lux-depth-v3 enhance --model small --input-dir test_images/ --output-dir test_output/
```

---

## Documentation Reference

- **Quick Start**: `QUICK_START_TESTING.sh` (interactive)
- **Comprehensive Guide**: `TESTING_STATUS_REPORT.md`
- **Full Summary**: `INTEGRATION_TESTING_COMPLETE.md`
- **Test Images**: `test_images/README_TEST_IMAGES.md`
- **Main Docs**: `README.md`

---

## Need Help?

1. Read the error message carefully (most include solutions)
2. Check `TESTING_STATUS_REPORT.md` Troubleshooting section
3. Review `INTEGRATION_TESTING_COMPLETE.md` for detailed guidance
4. Verify environment: `python3 --version`, `pip list | grep torch`

---

## After Testing Complete

Once all steps pass, you're ready for:

1. **Production Testing**: Use real project images
2. **Performance Benchmarking**: Measure throughput
3. **Documentation Review**: Update based on findings
4. **Deployment**: Integrate into production pipeline

See `INTEGRATION_TESTING_COMPLETE.md` for next milestones.

---

**Current Status**: Waiting for Step 1 (dependency installation)
**Blocking Issues**: None
**Ready to start**: YES ✅

Run `./QUICK_START_TESTING.sh` to begin!
