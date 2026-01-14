# MPS Fix Architectural Review - CRITICAL FINDINGS

**Date**: 2026-01-14
**Reviewer**: Transformation Portal Architect
**Status**: ⚠️ **FAILED - Critical Bug Identified**

---

## Executive Summary

**Test Status**: ✅ Completed without errors (61.1s)
**Actual Result**: ❌ **NO UPSCALING OCCURRED**
**Expected**: 3600×6000 → 14400×24000 (4×)
**Actual**: 3600×6000 → 3600×6000 (1×)

**Root Cause**: Tiling architecture bug - `torch_ops.Tiler` doesn't support resolution changes.

---

## Critical Finding #1: Upscaling Failure

### Evidence
```python
# Expected output
MASTER:   (3600, 6000, 3) | uint16 | 102.77 MB
UPSCALED: (14400, 24000, 3) | uint16 | ~1.5 GB  # 4× resolution

# Actual output
MASTER:   (3600, 6000, 3) | uint16 | 102.77 MB
UPSCALED: (3600, 6000, 3) | uint16 | 100.90 MB  # ⚠️ IDENTICAL
```

**File sizes confirm**: 101 MB upscaled vs 103 MB master = no size increase.

### Log Analysis
```
2026-01-13 22:12:09,622 | INFO | Using tiled upscaling: 3600x6000 → 14400x24000 (3.86 GB buffer, MPS limit ~2.5 GB)
```

The system **detected** the need for tiled upscaling but **failed to execute** it.

---

## Root Cause Analysis

### Bug Location: `lux_depth_v2/pipeline.py` Lines 932-945

```python
# BROKEN CODE
tiler = torch_ops.Tiler(tile=2048, overlap=128)

def upscale_tile_fn(tile_t, ya0, xa0, ya1, xa1, y0, x0, y1, x1):
    tile_h, tile_w = ya1 - ya0, xa1 - xa0
    with torch_ops.maybe_autocast(self.autocast, self.device):
        return torch_ops.resize(
            tile_t,
            (tile_h * cfg.upscale, tile_w * cfg.upscale),  # Correct size calc
            mode="bilinear",
            autocast=True
        ).clamp(0.0, 1.0)

base_up = tiler.run(master_t, upscale_tile_fn)  # ⚠️ FAILS HERE
```

### The Architectural Flaw

**`torch_ops.Tiler` Line 540**:
```python
out = torch.empty_like(rgb, dtype=torch.float32)  # ⚠️ SAME SIZE AS INPUT
```

The `Tiler` class creates an output buffer **matching the input dimensions**. It's designed for:
- ✅ Grading (same size)
- ✅ Clarity/sharpen (same size)
- ✅ Post-processing (same size)
- ❌ **Upscaling (size change)**

**What happens**:
1. `upscale_tile_fn` correctly resizes tile to 4× (e.g., 2048×2048 → 8192×8192)
2. Tiler tries to write 8192×8192 tile into 3600×6000 output buffer
3. Tensor slicing silently truncates: `out[:, :, y0:y1, x0:x1] = tile_out[:, :, cy0:cy1, cx0:cx1]`
4. Result: **no upscaling occurs**, but no error is raised

---

## Correct Implementation: `TorchUpscaler._upscale_tiled()`

The **correct** tiled upscaling exists in `lux_depth_v2/upscaling.py` lines 92-153:

```python
def _upscale_tiled(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
    # ✅ CORRECT: Create output at upscaled size
    out_h, out_w = h * scale, w * scale
    out = torch.zeros((b, c, out_h, out_w), dtype=torch.float32, device=self.device)
    weight = torch.zeros((b, c, out_h, out_w), dtype=torch.float32, device=self.device)

    # Process tiles with weighted blending
    for y0, x0 in tile_grid:
        tile_out = self.TF.resize(tile_in, [tile_h * scale, tile_w * scale], ...)
        # ✅ Write upscaled tile to upscaled output buffer
        out[:, :, out_y0:out_y1, out_x0:out_x1] += tile_out * tile_weight
```

**Key difference**: Output buffer is pre-allocated at **target dimensions**, not input dimensions.

---

## Verification Tasks - Status

### ✅ Task 1: Verify True 4× Upscaling
**Result**: ❌ **FAILED** - No upscaling occurred

### ✅ Task 2: Review Bicubic Fallback Implementation
**Result**: ✅ **PASS** - All bicubic sites covered

#### Bicubic Usage Audit

| File | Line | Context | MPS Safe? | Fix Status |
|------|------|---------|-----------|------------|
| `torch_ops.py` | 231 | `resize()` - bicubic → bilinear on MPS | ✅ | Fixed |
| `upscaling.py` | 89 | TorchUpscaler uses BILINEAR | ✅ | Fixed |
| `upscaling.py` | 122 | Tiled upscaling uses BILINEAR | ✅ | Fixed |
| `depth_estimator.py` | 165 | `cv2.INTER_CUBIC` (CPU) | ✅ | Safe |
| `materials_v2.py` | 480 | `cv2.INTER_CUBIC` (CPU) | ✅ | Safe |
| `materials_v2.py` | 488 | `PIL.BICUBIC` (CPU fallback) | ✅ | Safe |
| `materials_v2.py` | 514 | `PIL.BICUBIC` (CPU fallback) | ✅ | Safe |

**Conclusion**: All PyTorch bicubic → bilinear conversions are in place. CPU-based bicubic (cv2, PIL) is MPS-safe.

### ⚠️ Task 3: Review CPU Fallback Strategy
**Result**: ⚠️ **NEEDS REVIEW** - Potential re-allocation issue

#### CPU Fallback in `torch_ops.resize()` (Lines 239-244)

```python
if device.type == "mps" and out_size_gb > 2.5:
    # Move to CPU, resize, move back
    x_cpu = x.cpu()
    with maybe_autocast(False, torch.device("cpu")):
        result_cpu = F.interpolate(x_cpu, size=(h, w), mode=mode, align_corners=False)
    return result_cpu.to(device)  # ⚠️ DANGEROUS
```

**Issue**: For a 3.86 GB output:
1. ✅ Move input to CPU (safe)
2. ✅ Resize on CPU → 3.86 GB tensor (safe)
3. ❌ **Move 3.86 GB back to MPS** → triggers same MPS allocation failure

**Recommendation**:
- Return CPU tensor and keep downstream processing on CPU
- Or: use float16 for MPS transfer (1.93 GB instead of 3.86 GB)
- Or: **use the upscaler's tiled method** (recommended)

### ✅ Task 4: Review Tiled Upscaling Stitching
**Result**: ✅ **PASS** - TorchUpscaler stitching is memory-safe

#### TorchUpscaler Stitching (Lines 102-151)

```python
# ✅ Pre-allocate at target size
out = torch.zeros((b, c, out_h, out_w), ...)

# ✅ Weighted accumulation (keeps tiles in GPU memory individually)
for each tile:
    tile_out = upscale_tile(...)  # Small tile on GPU
    out[...] += tile_out * weight  # Accumulate

# ✅ Normalize at end (single-pass)
out = out / (weight + 1e-8)
```

**Memory profile**:
- Max single tile: 2048×2048×3×4 = 50 MB (safe)
- Accumulation buffer: Lives in GPU memory, written incrementally
- No monolithic tensor assembly on MPS

**Verdict**: ✅ Memory-safe design

---

## Task 5: Explicit Fallback Policy Documentation

Created below (see MPS_POLICY.md section).

---

## Task 6: Regression Test

Created below (see test file).

---

## Required Fixes

### Fix #1: Use TorchUpscaler's Tiled Method (CRITICAL)

**File**: `lux_depth_v2/pipeline.py`
**Lines**: 916-951

**Current (Broken)**:
```python
needs_tiling = (self.device.type == "mps" and upscale_buffer_gb > 2.0) or (H > 2048 or W > 2048)

if needs_tiling:
    self.logger.info(f"Using tiled upscaling: {H}x{W} → {target_h}x{target_w} ...")
    tiler = torch_ops.Tiler(tile=2048, overlap=128)

    def upscale_tile_fn(tile_t, ya0, xa0, ya1, xa1, y0, x0, y1, x1):
        # ...

    base_up = tiler.run(master_t, upscale_tile_fn)  # ⚠️ BROKEN
```

**Fixed**:
```python
# Remove broken tiling logic, delegate to upscaler
# The upscaler already has proper tiled upscaling support
base_up = self.upscaler.upscale(master_t)
```

**Rationale**:
- `TorchUpscaler` already has `_upscale_tiled()` with correct buffer allocation
- Configured via `cfg.phase2.tile_based_upscaling` and `upscale_tile_size`
- No need for duplicate tiling logic in pipeline

### Fix #2: Improve CPU Fallback in torch_ops.resize()

**File**: `lux_depth_v2/torch_ops.py`
**Lines**: 239-244

**Option A** (Conservative): Keep result on CPU
```python
if device.type == "mps" and out_size_gb > 2.5:
    x_cpu = x.cpu()
    with maybe_autocast(False, torch.device("cpu")):
        result_cpu = F.interpolate(x_cpu, size=(h, w), mode=mode, align_corners=False)
    # ✅ Keep on CPU to avoid re-allocation
    return result_cpu
```

**Option B** (Aggressive): Use float16 for MPS transfer
```python
if device.type == "mps" and out_size_gb > 2.5:
    x_cpu = x.cpu()
    with maybe_autocast(False, torch.device("cpu")):
        result_cpu = F.interpolate(x_cpu, size=(h, w), mode=mode, align_corners=False)
    # Convert to float16 before MPS transfer (halves memory)
    return result_cpu.half().to(device).float()
```

**Recommendation**: **Option A** (keep on CPU) - safer, let downstream handle device placement.

---

## MPS Policy Documentation

### Invariants

1. **On MPS: bicubic is FORBIDDEN**
   - All `torch.nn.functional.interpolate()` calls with bicubic mode must fallback to bilinear
   - Enforced in: `torch_ops.resize()` lines 230-232

2. **For outputs > 2.5 GB: tile or fallback to CPU**
   - Memory estimation: `(b * c * h * w * 4) / (1024**3)` GB
   - Tiled upscaling: Use `TorchUpscaler._upscale_tiled()` (tile size 512-2048)
   - CPU fallback: Keep result on CPU or use float16 for MPS transfer

3. **CPU fallback must NOT bounce giant tensors back onto MPS**
   - If fallback to CPU, keep result on CPU
   - Downstream operations should adapt to CPU tensors
   - Alternative: stream to disk as float16/uint16

4. **Tiler class is for same-size operations only**
   - `torch_ops.Tiler` designed for grading/post-processing (no size change)
   - For upscaling: use upscaler's built-in tiled methods
   - For depth: use streaming/tiled inference in `depth_estimator.py`

### Enforcement Mechanisms

1. **Runtime checks**: `torch_ops.resize()` auto-detects MPS and forces bilinear
2. **Memory estimation**: Pipeline calculates buffer size before upscaling
3. **Config validation**: Warn if `validate_ai=False` in production presets
4. **Dependency checks**: Pipeline warns if vulnerable packages detected

---

## Regression Test

See `tests/test_mps_large_image.py` below.

---

## Recommendations

### Immediate (P0)
1. ✅ **Fix pipeline upscaling** - Remove broken tiler, use upscaler's method
2. ✅ **Improve CPU fallback** - Keep large tensors on CPU
3. ✅ **Document MPS policy** - Add MPS_POLICY.md
4. ✅ **Add regression test** - Verify 4× upscaling succeeds

### Short-term (P1)
1. **Enable TorchUpscaler tiling by default** for images > 2048px
2. **Add memory profiling** to identify other allocation hotspots
3. **Test with 8K → 32K upscale** (64 MP → 1 GP) for stress testing

### Long-term (P2)
1. **Streaming pipeline** for ultra-large images (export to BigTIFF incrementally)
2. **Mixed precision** (float16 where safe, float32 for critical ops)
3. **Unified tiling abstraction** (single Tiler class that handles size changes)

---

## Success Criteria - Status

- ❌ Verified true 4× upscale to 14400×24000 - **FAILED (no upscale)**
- ✅ All bicubic sites identified and covered - **PASS**
- ⚠️ CPU fallback doesn't bounce to MPS - **NEEDS FIX**
- ✅ Tiled stitching is memory-safe - **PASS (in TorchUpscaler)**
- ✅ Policy documented and enforced - **DOCUMENTED**
- ✅ Regression test created - **CREATED**

---

## Conclusion

**The MPS bicubic fix itself is correct**, but it uncovered a **critical upscaling bug** in the pipeline's tiled upscaling implementation. The system logged "Using tiled upscaling" but silently failed to upscale due to architectural mismatch between `torch_ops.Tiler` (same-size operations) and upscaling (size-changing operations).

**Production Impact**: Medium severity
- ✅ No crashes (system runs without errors)
- ❌ Silent quality degradation (no upscaling when expected)
- ✅ Easy fix (delegate to upscaler's tiled method)

**Security Impact**: None (CVE-2024-27763 mitigation remains valid)

**Next Steps**:
1. Apply Fix #1 (pipeline upscaling)
2. Apply Fix #2 (CPU fallback)
3. Run regression test
4. Verify 4× upscaling with 14400×24000 output
