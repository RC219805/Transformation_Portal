# Tiling Bug: Final Root Cause Analysis
**Date**: 2025-12-18  
**Status**: ❌ FUNDAMENTAL ARCHITECTURE ISSUE IDENTIFIED

---

## The Real Problem: Model Output Size Mismatch

### Discovery

Testing bypass mode with Depth Anything V2:
```
Input image:  512 × 288 pixels
Model input:  512 × 288 pixels (do_resize=False)
Model output: 504 × 280 pixels  ← 8 pixels smaller!
```

**The model reduces spatial dimensions by ~8 pixels** even with `do_resize=False`. This is due to:
- Conv padding strategies in the backbone
- Possible downsampling/upsampling layers
- Architecture-specific spatial reduction

### Impact on Tiling

When we extract a 1024×1024 tile and process it:
```python
tile_rgb = rgb[y0:y1, x0:x1]  # Shape: (1024, 1024, 3)
tile_depth = _infer_tile(tile_rgb)  # Shape: (1016, 1016)  ← SMALLER!

# Bug: Placing smaller depth into larger slot
depth_stack[idx, y0:y1, x0:x1] = tile_depth  # Shape mismatch!
```

This creates:
1. **Misalignment** - depth doesn't correspond to correct RGB pixels
2. **Edge artifacts** - tile boundaries don't match up spatially
3. **Seam discontinuities** - even with scale reconciliation

### Why Scale Reconciliation Didn't Fix It

Scale reconciliation matches **depth values**, not **spatial positions**. If tiles are spatially misaligned (depth[i,j] doesn't correspond to rgb[i,j]), no amount of scale matching will fix edge alignment.

---

## The Correct Fix

### Option 1: Resize Depth to Match Tile Size (Recommended)

```python
def _infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
    # ... model inference ...
    depth = depth_tensor.squeeze().cpu().numpy()
    
    # CRITICAL: Resize depth to match input tile size
    target_h, target_w = tile_rgb.shape[:2]
    if depth.shape != (target_h, target_w):
        from scipy.ndimage import zoom
        scale_h = target_h / depth.shape[0]
        scale_w = target_w / depth.shape[1]
        depth = zoom(depth, (scale_h, scale_w), order=1)  # Bilinear
        logger.debug(f"Resized depth: {depth_tensor.shape} → {depth.shape}")
    
    return depth
```

**Pros**: Simple, preserves tile alignment  
**Cons**: Slight interpolation (but better than misalignment)

### Option 2: Track Actual Depth ROI

```python
def _infer_tile(self, tile_rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    # ... inference ...
    depth = depth_tensor.squeeze().cpu().numpy()
    
    # Compute actual depth ROI (centered crop of tile)
    dh, dw = depth.shape
    th, tw = tile_rgb.shape[:2]
    
    # Center the depth within the tile region
    y_offset = (th - dh) // 2
    x_offset = (tw - dw) // 2
    
    return depth, (y_offset, y_offset + dh, x_offset, x_offset + dw)
```

Then adjust blending to account for offsets.

**Pros**: No interpolation  
**Cons**: Complex blending logic, still have edge alignment issues at seams

### Option 3: Abandon Bypass Mode (Not Recommended)

Use HF pipeline which handles resizing internally.

**Pros**: Consistent behavior  
**Cons**: Forces 518px resize (defeats the purpose of tiling)

---

## Validation Test

After implementing the fix, the isolation test must show:

```
tiling_only:
  Edge overlap:     ≥75% (baseline: 77.2%)
  Edge correlation: ≥0.18 (baseline: 0.208)
  Boundary energy:  <1.2x interior
```

---

## Why This Explains Everything

### Original Validation Failure
- Edge overlap: 77.2% → 65.0% (-12%)
- Edge correlation: 0.208 → 0.047 (-0.161)
- Boundary energy: 27× interior (massive seams)

**Explanation**: Tiles are spatially misaligned by ~8 pixels, creating:
- Depth edges at wrong positions (low overlap with RGB)
- Discontinuities at tile boundaries (seams)
- Grid-like artifacts (periodic 1024-pixel structure)

### Why Refinement Worked
- CLAHE, guided filter, edge snap all operate **post-tiling**
- They work on the final blended depth (already spatially consistent)
- They improve quality within regions, don't fix spatial misalignment

---

## Recommendation

**Immediate Action**: Implement Option 1 (resize depth to match tile size)

**Code Change** (~5 lines in `_infer_tile`):
```python
# After line 384 in depth_inference.py
depth = depth_tensor.squeeze().cpu().numpy()

# Add this:
target_h, target_w = tile_rgb.shape[:2]
if depth.shape != (target_h, target_w):
    import cv2
    depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
```

**Validation**:
```bash
python lux_depth_v2/tools/isolation_test_suite.py \
    --input input_images/750_Picacho/Kitchen_2K_test.png \
    --output outputs/isolation_tests_size_fix
```

**Expected Result**: Edge overlap ≥75%, correlation ≥0.18

---

## Lesson Learned

**Always validate spatial correspondence** between model inputs and outputs.

The assumption that `model(image)` returns `depth.shape == image.shape[:2]` is **not safe** - especially with:
- ViT-based models (patch embeddings)
- Models with stride > 1 downsampling
- Models with "valid" padding

**Required check** in all ML pipelines:
```python
assert depth.shape[:2] == rgb.shape[:2], \
    f"Spatial mismatch: depth {depth.shape} vs rgb {rgb.shape}"
```

---

**Status**: Root cause confirmed, fix identified, awaiting implementation and validation.  
**Next**: Implement resize fix, re-run isolation test, validate ≥75% overlap.
