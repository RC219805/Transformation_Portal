# MPS Fix Validation Checklist

## Pre-Deployment Checklist

### ✅ Code Changes
- [x] **high_fidelity_depth/depth_estimator.py**: OpenCV INTER_CUBIC replaces PIL BICUBIC
- [x] **lux_depth_v2/torch_ops.py**: MPS device detection + bicubic→bilinear fallback
- [x] **lux_depth_v2/torch_ops.py**: Memory estimation + CPU fallback for >2.5GB
- [x] **lux_depth_v2/pipeline.py**: Automatic tiled upscaling for large images

### ✅ Documentation
- [x] **lux_depth_v2/MPS_COMPATIBILITY.md**: Comprehensive guide created
- [x] **lux_depth_v2/SECURITY.md**: Section 9 added (MPS compatibility)
- [x] **test_mps_compatibility.py**: Automated test suite created
- [x] **MPS_FIX_SUMMARY.md**: Implementation summary created
- [x] **MPS_QUICK_REF.md**: Quick reference created

### ✅ Testing (Manual Verification Required)

#### Test 1: Small Image (No Tiling)
```bash
# Expected: Uses MPS bilinear, no tiling
# Input: 1800x3000, Output: 7200x12000 (1.2 GB)
lux-depth-v2 --input-dir test_small/ --output-dir out/ --preset interior_luxury
```
**Success Criteria:**
- ✅ No "upsample_bicubic2d" errors
- ✅ No "Invalid buffer size" errors
- ✅ Log: "Base upsample: GPU bilinear"
- ✅ Memory peak <2 GB

#### Test 2: Large Image (Automatic Tiling)
```bash
# Expected: Uses tiled upscaling
# Input: 3600x6000, Output: 14400x24000 (3.86 GB)
lux-depth-v2 --input-dir test_large/ --output-dir out/ --preset interior_luxury
```
**Success Criteria:**
- ✅ Log: "Using tiled upscaling: 3600x6000 → 14400x24000 (3.86 GB buffer)"
- ✅ No buffer size errors
- ✅ Memory peak <2 GB
- ✅ Output image has no tile seams

#### Test 3: Global Anchor (Depth Generation)
```bash
# Expected: Uses OpenCV for global anchor upsampling
# Input: 3600x6000, no depth provided
lux-depth-v2 --input-dir test_no_depth/ --output-dir out/ --preset interior_luxury
```
**Success Criteria:**
- ✅ Log: "Computing global anchor at 768×1280..."
- ✅ Log: "✓ Global anchor computed: (3600, 6000)"
- ✅ No "upsample_bicubic2d" errors
- ✅ Depth map generated successfully

#### Test 4: Automated Test Suite
```bash
python3 test_mps_compatibility.py --device mps
```
**Success Criteria:**
- ✅ Test 1: Resize MPS Fallback - PASS
- ✅ Test 2: Large Tensor CPU Fallback - PASS
- ✅ Test 3: Tiled Upscaling - PASS
- ✅ Test 4: Global Anchor OpenCV - PASS
- ✅ Total: 4/4 tests passed

### ✅ Quality Validation

#### Visual Inspection
- [ ] Compare bicubic (CPU) vs bilinear (MPS) on 5 test images
- [ ] Measure edge sharpness degradation: Expected ~5-10%
- [ ] Verify tiled outputs have no seam artifacts
- [ ] Check color accuracy (no shifts)

#### Performance Benchmarks
- [ ] Measure throughput: Expected 127-400 images/hour on M4 Max
- [ ] Verify MPS 3-5x faster than CPU
- [ ] Confirm tiling overhead 15-20% (acceptable)

### ✅ Edge Cases

#### Fallback Scenarios
- [ ] CPU device: All operations work (no MPS code triggered)
- [ ] CUDA device: No MPS code triggered
- [ ] Very large image (>6000px): CPU fallback or tiling works
- [ ] Environment variable `PYTORCH_MPS_DISABLE=1`: Falls back to CPU

#### Error Handling
- [ ] Missing depth: Auto-generates with global anchor (OpenCV)
- [ ] Invalid preset: Graceful error message
- [ ] Out of memory (even with tiling): Informative error

## Post-Deployment Monitoring

### Week 1 (Critical)
- [ ] Monitor production logs for "upsample_bicubic2d" errors (should be zero)
- [ ] Monitor memory usage (should be <2 GB peak on MPS)
- [ ] Collect user feedback on quality (bilinear vs bicubic)
- [ ] Track throughput metrics (compare to pre-fix baseline)

### Week 2-4 (Ongoing)
- [ ] Review PyTorch release notes for MPS bicubic support
- [ ] Analyze quality metrics (automated SSIM/PSNR if available)
- [ ] Identify any unexpected edge cases

## Rollback Plan

If critical issues arise:

1. **Revert commits:**
   ```bash
   git revert <commit-hash>  # Revert MPS fixes
   git push
   ```

2. **Temporary workaround:**
   ```bash
   export PYTORCH_MPS_DISABLE=1  # Force CPU mode
   ```

3. **Alternative:**
   ```bash
   lux-depth-v2 --device cpu  # Explicit CPU flag
   ```

## Sign-Off

- [x] **Implementation Complete**: All code changes verified
- [x] **Documentation Complete**: All docs created and reviewed
- [x] **Testing Plan Ready**: Manual test cases defined
- [ ] **Quality Validation**: Pending manual visual inspection
- [ ] **Production Deployment**: Pending final approval

**Ready for Production**: ⏳ Pending manual testing

**Next Steps:**
1. Run manual Test 1-4 on actual Apple Silicon device
2. Visual quality inspection (bicubic vs bilinear comparison)
3. Performance benchmarking (throughput measurement)
4. Final sign-off

---
**Date**: 2026-01-14
**Validator**: _________________
**Approved**: ⬜ Yes  ⬜ No  ⬜ Needs Revision
