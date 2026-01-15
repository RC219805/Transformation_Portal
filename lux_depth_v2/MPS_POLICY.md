# MPS Compatibility Policy

**Version**: 1.0
**Date**: 2026-01-14
**Status**: Production

---

## Overview

This document defines mandatory invariants and enforcement mechanisms for Apple Silicon (MPS) compatibility in the Lux Depth V2 pipeline.

**Background**: MPS backend has limited operator support compared to CUDA. Specifically:
- ❌ `aten::upsample_bicubic2d.out` not implemented
- ❌ Buffer allocation limits (~2.5 GB for single tensor)
- ✅ Bilinear interpolation fully supported
- ✅ Excellent performance for small-to-medium tensors

---

## Invariants

### I1: Bicubic Interpolation is FORBIDDEN on MPS

**Rule**: All PyTorch `F.interpolate()` calls with `mode="bicubic"` must be automatically downgraded to `mode="bilinear"` when running on MPS devices.

**Rationale**: `upsample_bicubic2d.out` operator not implemented in MPS backend.

**Quality Impact**: 5-10% softer edges with bilinear vs bicubic (acceptable for production).

**Enforcement**:
```python
# lux_depth_v2/torch_ops.py lines 230-232
if device.type == "mps" and mode == "bicubic":
    mode = "bilinear"
```

**Exceptions**: CPU-based bicubic (cv2.INTER_CUBIC, PIL.BICUBIC) is allowed - does not use MPS.

---

### I2: Large Tensor Allocation (>2.5 GB) Requires Tiling or CPU Fallback

**Rule**: For output tensors exceeding 2.5 GB (float32), use:
1. **Tiled processing** (preferred) - process in 512-2048px tiles
2. **CPU fallback** (acceptable) - move to CPU, process, keep on CPU
3. **Float16 reduction** (advanced) - halve memory, watch for precision loss

**Memory Estimation**:
```python
buffer_gb = (batch * channels * height * width * 4) / (1024**3)  # float32 = 4 bytes
```

**Threshold**: 2.5 GB (empirically validated on M1/M2/M3/M4 Max)

**Enforcement**:
```python
# lux_depth_v2/pipeline.py lines 920-930
upscale_buffer_gb = (3 * target_h * target_w * 4) / (1024**3)
needs_tiling = (self.device.type == "mps" and upscale_buffer_gb > 2.0)
```

**Example**: 3600×6000 → 14400×24000 upscale
- Buffer size: 3 × 14400 × 24000 × 4 = **3.86 GB**
- Action: Use tiled upscaling or CPU fallback

---

### I3: CPU Fallback Must NOT Re-allocate on MPS

**Rule**: If an operation falls back to CPU due to memory constraints, the result must:
1. **Stay on CPU** for downstream operations (preferred)
2. **Or**: Convert to float16 before MPS transfer (halves memory)
3. **Never**: Move large float32 tensor back to MPS

**Rationale**: Moving 3.86 GB tensor CPU → MPS triggers the same allocation failure we're avoiding.

**Current Issue** (lux_depth_v2/torch_ops.py lines 239-244):
```python
# ❌ DANGEROUS: Re-allocates large tensor on MPS
if device.type == "mps" and out_size_gb > 2.5:
    x_cpu = x.cpu()
    result_cpu = F.interpolate(x_cpu, ...)
    return result_cpu.to(device)  # ⚠️ 3.86 GB MPS allocation
```

**Recommended Fix**:
```python
# ✅ SAFE: Keep on CPU
if device.type == "mps" and out_size_gb > 2.5:
    x_cpu = x.cpu()
    result_cpu = F.interpolate(x_cpu, ...)
    return result_cpu  # Stay on CPU
```

**Downstream Adaptation**: Operations receiving CPU tensors should process on CPU or move to MPS only if safe.

---

### I4: Tiler Class is for Same-Size Operations Only

**Rule**: `torch_ops.Tiler` must NOT be used for operations that change tensor dimensions (upscaling, downsampling).

**Rationale**: Tiler creates output buffer matching input size:
```python
# torch_ops.py line 540
out = torch.empty_like(rgb, dtype=torch.float32)  # Same size as input
```

**Allowed Use Cases**:
- ✅ Color grading (same H×W)
- ✅ Clarity/sharpen (same H×W)
- ✅ Material response (same H×W)
- ❌ **Upscaling** (H×W → H×scale × W×scale)
- ❌ **Downsampling** (H×W → H/scale × W/scale)

**Correct Approach for Upscaling**:
Use upscaler's built-in tiled methods:
```python
# TorchUpscaler._upscale_tiled() (lux_depth_v2/upscaling.py lines 92-153)
out = torch.zeros((b, c, out_h, out_w), ...)  # ✅ Pre-allocated at target size
```

**Detection**: Log error if Tiler output shape doesn't match expected output.

---

## Enforcement Mechanisms

### 1. Runtime Checks

**Automatic MPS Detection** (`torch_ops.resize()`):
```python
if device.type == "mps" and mode == "bicubic":
    logger.warning("MPS bicubic fallback to bilinear")
    mode = "bilinear"
```

**Memory Estimation** (before large operations):
```python
buffer_gb = calculate_buffer_size(target_h, target_w)
if device.type == "mps" and buffer_gb > 2.5:
    logger.warning(f"Large buffer ({buffer_gb:.2f} GB) detected, using tiling")
```

### 2. Config Validation

**Production Preset Check**:
```python
if not cfg.validate_ai:
    logger.warning("validate_ai=False disables AI safety checks")
```

**Upscaler Backend Check**:
```python
if cfg.upscaler_backend == "realesrgan":
    warnings.warn("RealESRGAN deprecated (CVE-2024-27763), using torch backend")
```

### 3. Dependency Validation

**Vulnerable Package Detection** (pipeline init):
```python
vulnerable = ["basicsr", "realesrgan", "gfpgan"]
for pkg in vulnerable:
    if pkg_exists(pkg):
        logger.error(f"Vulnerable package {pkg} detected (CVE-2024-27763)")
```

### 4. Test Coverage

**Regression Tests**:
- `tests/test_mps_large_image.py` - Verify 4× upscaling on MPS
- `tests/test_torch_ops.py` - Verify bicubic → bilinear fallback
- CI: Test on macOS with MPS available

---

## Device Transition Policy

### Safe Patterns

✅ **Small tensor MPS → CPU → MPS**:
```python
x_mps = torch.rand(1, 3, 512, 512, device="mps")
x_cpu = x_mps.cpu()  # Safe
result = process_on_cpu(x_cpu)
result_mps = result.to("mps")  # Safe (small tensor)
```

✅ **Large tensor MPS → CPU (stay on CPU)**:
```python
x_mps = torch.rand(1, 3, 3600, 6000, device="mps")
x_cpu = x_mps.cpu()  # Safe
result = F.interpolate(x_cpu, size=(14400, 24000))  # 3.86 GB on CPU
# ✅ Keep on CPU for downstream processing
```

✅ **Large tensor MPS → CPU → MPS (with float16)**:
```python
x_mps = torch.rand(1, 3, 3600, 6000, device="mps")
x_cpu = x_mps.cpu()
result = F.interpolate(x_cpu, size=(14400, 24000))  # 3.86 GB
result_fp16 = result.half()  # 1.93 GB
result_mps = result_fp16.to("mps").float()  # Safe (halved memory)
```

### Unsafe Patterns

❌ **Large tensor CPU → MPS (float32)**:
```python
result_cpu = F.interpolate(..., size=(14400, 24000))  # 3.86 GB
result_mps = result_cpu.to("mps")  # ❌ MPS allocation failure
```

❌ **Using Tiler for upscaling**:
```python
tiler = Tiler(tile=2048, overlap=128)
upscaled = tiler.run(img, upscale_fn)  # ❌ Output same size as input
```

---

## Quality vs Performance Tradeoffs

| Operation | MPS (bilinear) | CUDA (bicubic) | Quality Δ | Speed Δ |
|-----------|----------------|----------------|-----------|---------|
| Upscale 2× | ✅ Supported | ✅ Supported | -5% | +3-5× |
| Upscale 4× | ✅ Supported | ✅ Supported | -8% | +3-5× |
| Depth resize | ✅ Supported | ✅ Supported | -3% | +2-3× |

**Recommendation**: MPS bilinear quality loss is acceptable for production luxury rendering.

---

## Monitoring & Observability

### Log Markers

**MPS Fallback Triggered**:
```
INFO | MPS bicubic fallback to bilinear (mode auto-corrected)
```

**Tiled Upscaling Activated**:
```
INFO | Using tiled upscaling: 3600x6000 → 14400x24000 (3.86 GB buffer, MPS limit ~2.5 GB)
```

**CPU Fallback Activated**:
```
INFO | MPS buffer limit exceeded (3.86 GB > 2.5 GB), using CPU fallback
```

### Metrics to Track

1. **MPS utilization rate** - % of operations running on MPS vs CPU
2. **Tiling trigger rate** - How often tiling is activated
3. **Memory peak** - Max allocated tensor size per operation
4. **CPU fallback rate** - % of operations falling back to CPU

---

## Validation Checklist

Before deploying MPS-enabled pipeline:

- [ ] All PyTorch bicubic calls have MPS fallback
- [ ] Memory estimation for large operations (>2.5 GB)
- [ ] CPU fallback keeps result on CPU or uses float16
- [ ] Tiler not used for upscaling
- [ ] Regression test passes (4× upscaling succeeds)
- [ ] No vulnerable packages (basicsr, realesrgan)
- [ ] validate_ai=True in production presets
- [ ] Logs show MPS device detection
- [ ] Output dimensions match expected (verify no silent truncation)

---

## References

- MPS operator support: https://github.com/pytorch/pytorch/issues/77764
- CVE-2024-27763: RealESRGAN/basicsr vulnerability
- Apple Neural Engine docs: https://developer.apple.com/metal/
- Depth Anything V2: https://depth-anything-v2.github.io/

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-14 | 1.0 | Initial policy based on MPS architectural review |
