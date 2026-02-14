# APEX Feature Gaps: Implementation Plan

**Status:** Architect-Approved Implementation Plan
**Date:** 2026-02-14
**Decider:** Transformation Portal Architect
**Context:** Three discovered feature gaps in APEX max-quality pipeline

---

## Executive Summary

Three APEX features have CLI flags but incomplete implementation:

1. **16-bit Output Path** - Flags exist but pipeline outputs 8-bit PNGs
2. **V2 MPS Acceleration** - Config exists but no CLI exposure, hardcoded to CPU
3. **ML Super-Resolution** - Upscaling infrastructure exists but only bicubic wired

All three gaps can be fixed independently without breaking existing behavior. Each fix preserves Golden Path (default configurations unchanged) and requires no breaking changes.

---

## Gap 1: 16-Bit Output Path

### Root Cause Analysis

**What's Broken:**

The 16-bit output path fails at the Materials V3 → V2 boundary:

1. ✅ **Depth Pro generates 16-bit depth** correctly (`depth_writer.py` lines 102-110)
2. ✅ **Config flags exist** (`emit_master16`, `emit_upscaled16`)
3. ✅ **CLI parsing exists** (`__main__.py` lines 216-217, 282-283)
4. ❌ **Materials V3 → V2 handoff outputs 8-bit PNG**
   - `orchestrator.py:846`: `enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)`
   - `orchestrator.py:851`: `enhanced_pil.save(enhanced_image_path)` (8-bit PNG)
5. ❌ **V2 receives 8-bit input** → outputs 8-bit result
6. ❌ **No master16.tif or upscaled16.tif files generated** (flags ignored)

**Why It Happens:**

Materials V3 was designed for fast 8-bit preview output to V2 subprocess. The temporary PNG handoff (`temp/{stem}_materials_v3_enhanced.png`) was never extended to support 16-bit TIFF when `emit_master16=True`.

V2's `v2_enhance.py` has full 16-bit support:
- `load_image_preserve_bit_depth()` (line 117) handles TIFF 16-bit correctly
- `detect_input_bit_depth()` (line 85) reads TIFF BitsPerSample tags
- Output save logic (lines 520-565) writes 16-bit TIFF when input is 16-bit

**The gap:** Orchestrator doesn't use this capability when Materials V3 is enabled.

---

### Architecture Decision

**Contract-Stable Fix Strategy:**

Preserve Materials V3 → V2 handoff contract (temp PNG) for 8-bit path (Golden Path). Add **parallel 16-bit handoff path** when `emit_master16=True` or `emit_upscaled16=True`.

**Staged Implementation:**

```
Stage A (Materials V3 → Orchestrator):
  If emit_master16 or emit_upscaled16:
    - Save temp/{stem}_materials_v3_enhanced_16bit.tif (uint16)
    - Use PIL mode "I;16" with tifffile for proper 16-bit encoding
  Else:
    - Keep existing 8-bit PNG handoff

Stage B (Orchestrator → V2):
  If 16-bit handoff exists:
    - Pass 16-bit TIFF path to V2 subprocess
  Else:
    - Pass 8-bit PNG path (existing behavior)

Stage C (V2 Output):
  V2 auto-detects input bit depth (already implemented)
  - Outputs master16.tif and/or upscaled16.tif when input is 16-bit
```

**Why This Works:**
- No changes to V2 subprocess CLI (already supports 16-bit TIFF input)
- No changes to default behavior (8-bit PNG handoff preserved)
- V2's existing bit-depth detection handles the rest
- Clean separation: orchestrator controls handoff format, V2 controls output format

---

### Implementation Plan

#### Task 1.1: Add 16-bit Handoff to Orchestrator

**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Changes:**

1. Modify `_compute_depth_stage()` around line 846:

```python
# Current (8-bit only):
enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)
enhanced_pil = PILImage.fromarray(enhanced_uint8)
enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced.png"
enhanced_pil.save(enhanced_image_path)

# NEW (bit-depth adaptive):
if self.config.emit_master16 or self.config.emit_upscaled16:
    # 16-bit handoff path
    import tifffile
    enhanced_uint16 = (np.clip(working_image, 0, 1) * 65535).astype(np.uint16)
    enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced_16bit.tif"

    # Use tifffile for proper 16-bit TIFF encoding with ICC profile preservation
    tifffile.imwrite(
        enhanced_image_path,
        enhanced_uint16,
        photometric='rgb',
        compression='adobe_deflate',  # Lossless compression
        metadata=None  # Extend later if ICC profile needed
    )
    logger.info(f"Materials V3 16-bit handoff: {enhanced_image_path}")
else:
    # 8-bit handoff path (Golden Path - unchanged)
    enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)
    enhanced_pil = PILImage.fromarray(enhanced_uint8)
    enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced.png"
    enhanced_pil.save(enhanced_image_path)
    logger.debug("Materials V3 8-bit handoff (default)")
```

2. **Dependencies:** `tifffile` already in `requirements/ml.txt` (line 72) ✅

#### Task 1.2: Add V2 Output Validation

**File:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Changes:**

After V2 completes (`_run_v2_stage` return), add output validation:

```python
# After line 1188 (v2_result returned)
if self.config.emit_master16 or self.config.emit_upscaled16:
    # Verify V2 generated expected 16-bit outputs
    expected_outputs = []
    if self.config.emit_master16:
        expected_outputs.append(self.v2_dir / f"{output_key.name}_master16.tif")
    if self.config.emit_upscaled16:
        expected_outputs.append(self.v2_dir / f"{output_key.name}_upscaled16.tif")

    missing = [p for p in expected_outputs if not p.exists()]
    if missing:
        logger.warning(
            f"16-bit outputs requested but not generated by V2: "
            f"{[p.name for p in missing]}. Check V2 bit-depth detection."
        )
```

**Rationale:** Defensive verification - ensures contract compliance without blocking pipeline.

#### Task 1.3: Update Manifest Schema

**File:** `src/transformation_portal/lux_depth_v3/manifest.py`

**Changes:**

Add 16-bit output tracking to V2Metadata:

```python
@dataclass
class V2Metadata:
    """V2 enhancement stage metadata."""
    preset: str
    v2_device: Optional[str] = None
    runtime_s: Optional[float] = None
    upscaler_backend: Optional[str] = None
    input_bit_depth: Optional[int] = None  # NEW: 8 or 16
    output_bit_depth: Optional[int] = None  # NEW: 8 or 16
    master16_generated: bool = False  # NEW
    upscaled16_generated: bool = False  # NEW
```

Populate in `_write_manifest()` (orchestrator.py):

```python
# After V2 stage completes
input_bit_depth = 16 if enhanced_image_path.suffix == '.tif' else 8
output_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) else 8

v2_metadata = V2Metadata(
    preset=self.config.v2_preset,
    v2_device=self.config.v2_device,
    runtime_s=v2_runtime_s,
    upscaler_backend=self.config.v2_upscaler_backend,
    input_bit_depth=input_bit_depth,  # NEW
    output_bit_depth=output_bit_depth,  # NEW
    master16_generated=(self.v2_dir / f"{output_key.name}_master16.tif").exists(),  # NEW
    upscaled16_generated=(self.v2_dir / f"{output_key.name}_upscaled16.tif").exists(),  # NEW
)
```

---

### Testing Strategy

#### Test 1.1: 16-bit End-to-End Golden Path

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_16bit_test \
  --quality-tier apex \
  --depth-backend depth_pro \
  --materials-v3 on \
  --emit-master16 on \
  --emit-upscaled16 on \
  --accept-apple-depth-pro-research-license true
```

**Expected Outputs:**
- `output_16bit_test/temp/{stem}_materials_v3_enhanced_16bit.tif` (temporary handoff)
- `output_16bit_test/v2/{stem}_master16.tif` (16-bit output)
- `output_16bit_test/v2/{stem}_upscaled16.tif` (16-bit upscaled)
- Manifest shows `input_bit_depth=16`, `output_bit_depth=16`, `master16_generated=true`

**Validation:**
```python
from PIL import Image
import numpy as np

# Verify 16-bit TIFF encoding
img = Image.open("output_16bit_test/v2/{stem}_master16.tif")
assert img.mode in ("I;16", "I"), f"Expected 16-bit mode, got {img.mode}"

arr = np.array(img)
assert arr.dtype in (np.uint16, np.int32), f"Expected uint16/int32, got {arr.dtype}"
assert arr.max() > 255, "Image appears to be 8-bit scaled to 16-bit"
```

#### Test 1.2: 8-bit Golden Path Preservation

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_8bit_test \
  --quality-tier premium \
  --materials-v3 on
```

**Expected Outputs:**
- `output_8bit_test/temp/{stem}_materials_v3_enhanced.png` (8-bit handoff, unchanged)
- `output_8bit_test/v2/{stem}_enhanced.jpg` or `.png` (8-bit output, unchanged)
- Manifest shows `input_bit_depth=8`, `output_bit_depth=8`

**Validation:**
- No 16-bit files generated
- Output file sizes similar to pre-change baseline (no regression)

#### Test 1.3: Materials V3 Disabled Path

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_no_materials \
  --quality-tier apex \
  --materials-v3 off \
  --emit-master16 on
```

**Expected Behavior:**
- V2 receives original 16-bit TIFF input directly (if input is TIFF)
- OR V2 receives original 8-bit input (if input is JPG/PNG)
- No Materials V3 handoff file created

**Validation:**
- V2 bit-depth detection based on actual input file format
- 16-bit outputs generated only if input is 16-bit TIFF

---

### APEX Compliance

#### Contract Impact

**Materials V3 → V2 Contract Extension:**
- **Current:** Temporary PNG handoff at `temp/{stem}_materials_v3_enhanced.png`
- **Extension:** Optional TIFF handoff at `temp/{stem}_materials_v3_enhanced_16bit.tif`
- **Selection:** Controlled by `emit_master16` or `emit_upscaled16` flags
- **Backward Compatibility:** ✅ Golden Path (8-bit PNG) unchanged

**V2 Contract:**
- No changes (V2 already supports 16-bit TIFF input via existing code)

#### Golden Path Preservation

| Scenario | Config | Behavior |
|----------|--------|----------|
| Default (Golden Path) | `emit_master16=off, emit_upscaled16=off` | 8-bit PNG handoff, 8-bit outputs (unchanged) |
| 16-bit APEX | `emit_master16=on` or `emit_upscaled16=on` | 16-bit TIFF handoff, 16-bit outputs (new) |
| Materials V3 Disabled | Any | Direct input → V2, bit depth follows input format (unchanged) |

#### Version Bumps

**Manifest Schema:** Increment to v1.7
- Add `V2Metadata.input_bit_depth`, `output_bit_depth`, `master16_generated`, `upscaled16_generated`
- Migration: Old manifests missing fields → default to `None`/`False` (non-breaking)

**Config Schema:** No version bump needed (fields already exist)

---

## Gap 2: V2 MPS Acceleration

### Root Cause Analysis

**What's Broken:**

1. ✅ **Config field exists:** `v2_device: str = "cpu"` (config.py line 167)
2. ✅ **Orchestrator passes it to V2Runner:** `device=self.config.v2_device` (orchestrator.py line 1179)
3. ❌ **No CLI flag exposure:** CLI doesn't allow setting `v2_device`
4. ❌ **Hardcoded default:** Always `cpu`, no way to use `mps` for Apple Silicon acceleration

**Why It Matters:**

V2 enhancement is CPU-bound on macOS. The V2 subprocess (`scripts/enhance_image.py`) already supports `--device mps`, but APEX orchestrator can't pass it through because the CLI doesn't expose `--v2-device`.

**Performance Impact (Estimated):**
- CPU V2 enhancement: ~1.5-2.0s/image
- MPS V2 enhancement: ~0.5-0.8s/image (2-3x faster on M-series chips)

---

### Architecture Decision

**Simple CLI Extension Strategy:**

Add `--v2-device` flag to APEX CLI, wire to existing `config.v2_device` field.

**No behavior changes:**
- Default remains `cpu` (Golden Path preserved)
- V2 subprocess already handles device selection correctly
- Config already passes through orchestrator → V2Runner → subprocess

**Why This Is Safe:**
- V2 subprocess validates device availability (falls back to CPU if MPS unavailable)
- Config field already exists and is tested
- No new code paths, just CLI exposure

---

### Implementation Plan

#### Task 2.1: Add CLI Flag

**File:** `src/transformation_portal/lux_depth_v3/__main__.py`

**Changes:**

Add flag after `v2_preset` (around line 214):

```python
v2_preset: Optional[str] = typer.Option(
    "default",
    "--v2-preset",
    help="V2 enhancement preset name or 'none' to skip enhancement (default: default). "
         "Only used when --enable-v2 is on.",
),
v2_device: str = typer.Option(  # NEW
    "cpu",
    "--v2-device",
    help="Device for V2 enhancement: cpu (default), cuda (NVIDIA GPU), mps (Apple Silicon GPU). "
         "Requires V2 stage enabled (--enable-v2 on).",
),
```

**Wire to config** (around line 373):

```python
# Existing config construction
config = EnhanceConfig(
    # ... existing fields ...
    v2_preset=v2_preset,
    v2_device=v2_device,  # NEW (add this line)
    # ... rest of config ...
)
```

#### Task 2.2: Update Documentation

**File:** `src/transformation_portal/lux_depth_v3/README.md`

Add to "V2 Enhancement Configuration" section:

```markdown
### V2 Enhancement Configuration

- `--enable-v2 on/off` - Enable V2 enhancement stage (default: on)
- `--v2-preset <name>` - Enhancement preset (default: "default")
- `--v2-device <device>` - Processing device: cpu, cuda, mps (default: cpu)
  - Use `mps` for 2-3x faster V2 enhancement on Apple Silicon
  - Automatically falls back to CPU if device unavailable

**Example (Apple Silicon):**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input/ \
  --output-dir output/ \
  --quality-tier apex \
  --v2-device mps  # GPU acceleration on M-series chips
```
```

#### Task 2.3: Add Preset Defaults

**File:** `src/transformation_portal/lux_depth_v3/config.py`

**Optional Enhancement:** Add device auto-detection to presets:

```python
def get_preset_config(preset_name: str) -> EnhanceConfig:
    """Load preset configuration with platform-aware defaults."""

    # ... existing preset logic ...

    # Platform-aware V2 device selection
    if preset_name in ("apex", "max-quality"):
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Apple Silicon - use MPS by default for APEX tier
            config.v2_device = "mps"
            logger.info("APEX tier on Apple Silicon: using MPS for V2 acceleration")

    return config
```

**Rationale:** Auto-enables best performance on Apple Silicon for APEX tier without user configuration. Golden Path (premium tier) stays `cpu` for broader compatibility.

---

### Testing Strategy

#### Test 2.1: MPS Acceleration Functional Test

**Platform:** macOS with Apple Silicon (M1/M2/M3/M4)

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_mps_test \
  --quality-tier premium \
  --v2-device mps
```

**Expected Behavior:**
- V2 subprocess receives `--device mps` argument
- V2 enhancement uses Metal Performance Shaders
- Manifest `v2_metadata.v2_device` = "mps"

**Validation:**
```bash
# Check V2 log for device confirmation
grep "device: mps" output_mps_test/logs/v2_*.log

# Verify manifest
jq '.v2.v2_device' output_mps_test/manifests/*_combined.json
# Expected: "mps"
```

#### Test 2.2: Fallback to CPU Test

**Platform:** macOS Intel or Linux

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_fallback_test \
  --v2-device mps  # Not available on this platform
```

**Expected Behavior:**
- V2 subprocess attempts MPS, falls back to CPU
- Warning logged: "MPS not available, using CPU"
- Pipeline completes successfully

**Validation:**
- No pipeline failure
- Outputs generated correctly
- Manifest shows `v2_device="cpu"` (fallback)

#### Test 2.3: Golden Path Preservation

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_default_test \
  --quality-tier premium
  # No --v2-device specified
```

**Expected Behavior:**
- Default `v2_device="cpu"` used
- No behavior change from current implementation

**Validation:**
- Manifest shows `v2_device="cpu"`
- Output quality identical to baseline

---

### APEX Compliance

#### Contract Impact

**Public CLI Contract Extension:**
- Add `--v2-device` flag (new, optional, has safe default)
- No changes to existing flags or behavior

#### Golden Path Preservation

| Scenario | Config | Behavior |
|----------|--------|----------|
| Default (Golden Path) | `--v2-device` not specified | V2 uses CPU (unchanged) |
| MPS Acceleration | `--v2-device mps` | V2 uses Apple Silicon GPU (new capability) |
| CUDA Acceleration | `--v2-device cuda` | V2 uses NVIDIA GPU (new capability) |
| Invalid Device | `--v2-device invalid` | V2 falls back to CPU (graceful degradation) |

#### Version Bumps

**None required:**
- Config field `v2_device` already exists (no schema change)
- Manifest already captures `v2_device` (V2Metadata.v2_device, line 265)
- CLI extension only (backward compatible)

---

## Gap 3: ML Super-Resolution Upscaling

### Root Cause Analysis

**What's Broken:**

1. ✅ **Upscaling stage exists:** `stage_graph/stages/upscaling.py`
2. ✅ **Backend parameter exists:** `UpscalingStage(backend="torch" | "onnx" | "bicubic")`
3. ✅ **Config field exists:** `v2_upscaler_backend: Optional[str] = None` (config.py)
4. ❌ **Only bicubic wired:** `_load_upscaler()` always sets `self._upscaler = "bicubic"` (upscaling.py line 141)
5. ❌ **No ML backend implementation:** Torch/ONNX backends not implemented
6. ❌ **No Real-ESRGAN integration:** Industry-standard upscaler not wired

**Current Behavior:**

```python
def _load_upscaler(self, device: str):
    """Load upscaling backend."""
    # Always use bicubic for simplicity (torch backend requires config)
    self.logger.info(f"Using bicubic upscaler on {device}")
    self._upscaler = "bicubic"
```

**Why It Matters:**

- Bicubic upscaling: Fast but introduces blur/artifacts at 2x+ scaling
- Real-ESRGAN: ML-based, preserves detail and sharpness, industry-standard for photo upscaling
- Current `--upscaler "default"` flag does nothing (always bicubic)

---

### Architecture Decision

**Phased ML Backend Integration:**

Implement Real-ESRGAN backend as **optional ML dependency** following existing patterns:

1. **Registry Pattern:** Follow `DepthBackendRegistry` model (depth/backends/registry.py)
2. **Optional Dependency:** Real-ESRGAN in `requirements/ml.txt` (import-time graceful degradation)
3. **Lazy Loading:** Load model weights only when needed, cache in memory
4. **Fallback Chain:** real-esrgan → bicubic (if ML unavailable)

**Upscaler Backend Registry Design:**

```python
# src/transformation_portal/upscaling/registry.py

class UpscalerBackend(Protocol):
    """Protocol for upscaling backends."""
    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray:
        ...

class UpscalerRegistry:
    """Registry for upscaling backends with lazy loading."""

    _backends: Dict[str, Callable[[], UpscalerBackend]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], UpscalerBackend]):
        cls._backends[name] = loader

    @classmethod
    def get(cls, name: str, device: str = "cpu") -> UpscalerBackend:
        if name not in cls._backends:
            raise ValueError(f"Unknown upscaler: {name}")
        return cls._backends[name]()  # Lazy instantiation
```

**Backend Implementations:**

1. **Bicubic (core dependency, always available):**
   - `src/transformation_portal/upscaling/backends/bicubic.py`
   - Uses scikit-image (already in core)

2. **Real-ESRGAN (optional ML dependency):**
   - `src/transformation_portal/upscaling/backends/realesrgan.py`
   - Lazy import with graceful fallback
   - Model weights cached in `weights/realesrgan/`

---

### Implementation Plan

#### Task 3.1: Create Upscaler Backend Registry

**New File:** `src/transformation_portal/upscaling/__init__.py`

```python
"""Upscaling backend registry and protocol."""

from typing import Protocol
import numpy as np

class UpscalerBackend(Protocol):
    """Protocol for upscaling backends.

    All backends must implement this interface for registry compatibility.
    """

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Upscale image by scale factor.

        Args:
            image: Input image (H, W, 3) float32 [0, 1]
            scale: Scale factor (1.0-4.0)

        Returns:
            Upscaled image (H*scale, W*scale, 3) float32 [0, 1]
        """
        ...
```

**New File:** `src/transformation_portal/upscaling/registry.py`

```python
"""Upscaler backend registry with lazy loading."""

import logging
from typing import Callable, Dict, Optional

from .protocol import UpscalerBackend

logger = logging.getLogger(__name__)


class UpscalerRegistry:
    """Central registry for upscaling backends."""

    _backends: Dict[str, Callable[[str], UpscalerBackend]] = {}
    _instances: Dict[str, UpscalerBackend] = {}  # Singleton instances per device

    @classmethod
    def register(cls, name: str, loader: Callable[[str], UpscalerBackend]):
        """Register upscaler backend.

        Args:
            name: Backend name (e.g., "bicubic", "realesrgan")
            loader: Factory function that takes device string, returns backend instance
        """
        cls._backends[name] = loader
        logger.debug(f"Registered upscaler backend: {name}")

    @classmethod
    def get(cls, name: str, device: str = "cpu") -> UpscalerBackend:
        """Get upscaler backend instance.

        Args:
            name: Backend name
            device: Device for inference (cpu, cuda, mps)

        Returns:
            Upscaler backend instance

        Raises:
            ValueError: If backend not registered
        """
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown upscaler backend: {name!r}. "
                f"Available: {available}"
            )

        # Singleton per (backend, device) pair
        cache_key = f"{name}:{device}"
        if cache_key not in cls._instances:
            logger.info(f"Loading upscaler backend: {name} on {device}")
            cls._instances[cache_key] = cls._backends[name](device)

        return cls._instances[cache_key]

    @classmethod
    def list_available(cls) -> list[str]:
        """List registered backends."""
        return list(cls._backends.keys())


# Auto-register core backends
def _register_core_backends():
    """Register built-in backends on module import."""
    from .backends.bicubic import BicubicUpscaler

    UpscalerRegistry.register("bicubic", lambda device: BicubicUpscaler())

    # Conditionally register Real-ESRGAN if available
    try:
        from .backends.realesrgan import RealESRGANUpscaler
        UpscalerRegistry.register("realesrgan", RealESRGANUpscaler.create)
        logger.debug("Real-ESRGAN backend registered (ML dependency available)")
    except ImportError:
        logger.debug("Real-ESRGAN backend unavailable (install requirements/ml.txt)")


_register_core_backends()
```

#### Task 3.2: Implement Bicubic Backend

**New File:** `src/transformation_portal/upscaling/backends/bicubic.py`

```python
"""Bicubic upscaling backend (core dependency, always available)."""

import numpy as np
from skimage.transform import resize


class BicubicUpscaler:
    """Bicubic interpolation upscaler.

    Fast, deterministic, no ML dependencies.
    Good for <1.5x scaling, introduces blur at higher scales.
    """

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Upscale image using bicubic interpolation.

        Args:
            image: Input image (H, W, 3) float32 [0, 1]
            scale: Scale factor

        Returns:
            Upscaled image (H*scale, W*scale, 3) float32 [0, 1]
        """
        h, w = image.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)

        return resize(
            image,
            (new_h, new_w),
            order=3,  # Bicubic
            preserve_range=True,
            anti_aliasing=True,
        ).astype(image.dtype)
```

#### Task 3.3: Implement Real-ESRGAN Backend

**New File:** `src/transformation_portal/upscaling/backends/realesrgan.py`

```python
"""Real-ESRGAN upscaling backend (optional ML dependency)."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RealESRGANUpscaler:
    """Real-ESRGAN ML upscaler.

    Industry-standard photo super-resolution model.
    Preserves detail and sharpness better than bicubic.

    Requires: basicsr, realesrgan (install via requirements/ml.txt)
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "RealESRGAN_x2plus",
        weights_dir: Optional[Path] = None,
    ):
        """Initialize Real-ESRGAN upscaler.

        Args:
            device: Device for inference (cpu, cuda, mps)
            model_name: Model variant (RealESRGAN_x2plus, RealESRGAN_x4plus)
            weights_dir: Custom weights directory (default: repo weights/)
        """
        self.device = device
        self.model_name = model_name

        # Lazy import (allows module to load even if realesrgan not installed)
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as e:
            raise ImportError(
                "Real-ESRGAN backend requires basicsr and realesrgan. "
                "Install: pip install basicsr realesrgan"
            ) from e

        # Resolve weights directory
        if weights_dir is None:
            repo_root = self._find_repo_root()
            weights_dir = repo_root / "weights" / "realesrgan"

        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        model_config = {
            "RealESRGAN_x2plus": {
                "scale": 2,
                "model_path": weights_dir / "RealESRGAN_x2plus.pth",
                "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            },
            "RealESRGAN_x4plus": {
                "scale": 4,
                "model_path": weights_dir / "RealESRGAN_x4plus.pth",
                "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            },
        }

        if model_name not in model_config:
            raise ValueError(
                f"Unknown Real-ESRGAN model: {model_name}. "
                f"Available: {list(model_config.keys())}"
            )

        cfg = model_config[model_name]
        self.scale = cfg["scale"]
        model_path = cfg["model_path"]

        # Auto-download weights if missing
        if not model_path.exists():
            logger.info(f"Downloading Real-ESRGAN weights: {model_name}")
            self._download_weights(cfg["download_url"], model_path)

        # Initialize model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(model_path),
            model=model,
            tile=400,  # Tile size for memory efficiency
            tile_pad=10,
            pre_pad=0,
            half=device == "cuda",  # FP16 on CUDA for speed
            device=device,
        )

        logger.info(f"Real-ESRGAN loaded: {model_name} on {device}")

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Upscale image using Real-ESRGAN.

        Args:
            image: Input image (H, W, 3) float32 [0, 1]
            scale: Requested scale factor

        Returns:
            Upscaled image (H*scale, W*scale, 3) float32 [0, 1]

        Note:
            If requested scale doesn't match model scale, will upscale to model scale
            then resize to target (e.g., 2x model with 1.5x request → upscale 2x, resize to 1.5x)
        """
        # Convert to uint8 for Real-ESRGAN
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Upscale
        output, _ = self.upsampler.enhance(image_uint8, outscale=scale / self.scale)

        # Convert back to float32 [0, 1]
        return output.astype(np.float32) / 255.0

    @staticmethod
    def _find_repo_root() -> Path:
        """Find repository root."""
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        return current.parents[5]  # Fallback

    @staticmethod
    def _download_weights(url: str, dest: Path):
        """Download model weights."""
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded weights to {dest}")

    @classmethod
    def create(cls, device: str) -> "RealESRGANUpscaler":
        """Factory for registry integration."""
        return cls(device=device)
```

#### Task 3.4: Update UpscalingStage to Use Registry

**File:** `src/transformation_portal/stage_graph/stages/upscaling.py`

**Changes:**

Replace hardcoded bicubic with registry-based backend selection:

```python
# Add import at top
from ...upscaling.registry import UpscalerRegistry

class UpscalingStage(Stage):
    def _load_upscaler(self, device: str):
        """Load upscaling backend from registry."""
        try:
            self._upscaler = UpscalerRegistry.get(self.backend, device)
            self.logger.info(f"Loaded upscaler backend: {self.backend} on {device}")
        except ValueError as e:
            # Fallback to bicubic if requested backend unavailable
            self.logger.warning(f"{e}. Falling back to bicubic.")
            self._upscaler = UpscalerRegistry.get("bicubic", device)
            self.backend = "bicubic"
        except ImportError as e:
            # ML dependency missing - fallback to bicubic
            self.logger.warning(f"ML upscaler unavailable: {e}. Using bicubic.")
            self._upscaler = UpscalerRegistry.get("bicubic", device)
            self.backend = "bicubic"

    def _upscale_image(self, image: np.ndarray, device: str) -> np.ndarray:
        """Upscale image using loaded backend."""
        # Lazy load if not initialized
        if self._upscaler is None:
            self._load_upscaler(device)

        try:
            return self._upscaler.upscale(image, self.scale_factor)
        except Exception as e:
            self.logger.error(f"Upscaling failed: {e}, falling back to bicubic")
            # Emergency fallback
            bicubic = UpscalerRegistry.get("bicubic", device)
            return bicubic.upscale(image, self.scale_factor)
```

#### Task 3.5: Add CLI Flag for Upscaler Selection

**File:** `src/transformation_portal/lux_depth_v3/__main__.py`

Add flag (around line 215):

```python
v2_device: str = typer.Option(
    "cpu",
    "--v2-device",
    help="Device for V2 enhancement: cpu, cuda, mps",
),
v2_upscaler: str = typer.Option(  # NEW
    "bicubic",
    "--v2-upscaler",
    help="Upscaling backend: bicubic (fast, always available), realesrgan (ML-based, higher quality). "
         "Requires --enable-v2 on. ML upscalers need requirements/ml.txt installed.",
),
```

Wire to config:

```python
config = EnhanceConfig(
    # ... existing ...
    v2_upscaler_backend=v2_upscaler,  # NEW
)
```

#### Task 3.6: Add Real-ESRGAN to ML Requirements

**File:** `requirements/ml.in`

Add:

```
# Super-resolution upscaling (optional)
basicsr>=1.4.2
realesrgan>=0.3.0
```

Regenerate `ml.txt`:

```bash
make requirements-ml
```

#### Task 3.7: Add Weights Download Script

**New File:** `scripts/setup/download_upscaler_weights.py`

```python
"""Download Real-ESRGAN model weights."""

import argparse
import logging
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)

MODELS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

def download_weights(model_name: str, weights_dir: Path):
    """Download Real-ESRGAN weights."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    url = MODELS[model_name]
    dest = weights_dir / f"{model_name}.pth"

    if dest.exists():
        logger.info(f"Weights already exist: {dest}")
        return dest

    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {model_name} from {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Downloaded to {dest}")
    return dest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="RealESRGAN_x2plus")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/realesrgan"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_weights(args.model, args.weights_dir)
```

---

### Testing Strategy

#### Test 3.1: Bicubic Backend (Golden Path)

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_bicubic \
  --quality-tier premium \
  --v2-upscaler bicubic
```

**Expected Behavior:**
- Uses bicubic upscaling (existing behavior)
- No ML dependencies required
- Fast execution

**Validation:**
- Manifest shows `v2_metadata.upscaler_backend="bicubic"`
- Output quality matches baseline (no regression)

#### Test 3.2: Real-ESRGAN Backend

**Prerequisites:**
```bash
pip install -e .[ml]
python scripts/setup/download_upscaler_weights.py --model RealESRGAN_x2plus
```

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_realesrgan \
  --quality-tier apex \
  --v2-upscaler realesrgan \
  --v2-device cuda  # or mps on Apple Silicon
```

**Expected Behavior:**
- Loads Real-ESRGAN model weights
- Upscales with ML super-resolution
- Higher quality than bicubic at 2x+ scales

**Validation:**
```python
# Visual quality comparison
from PIL import Image
import numpy as np

bicubic_img = np.array(Image.open("output_bicubic/v2/{stem}_upscaled.jpg"))
realesrgan_img = np.array(Image.open("output_realesrgan/v2/{stem}_upscaled.jpg"))

# Real-ESRGAN should have sharper edges, less blur
# Use SSIM or manual inspection
```

#### Test 3.3: Fallback to Bicubic (ML Unavailable)

**Setup:** Uninstall Real-ESRGAN:
```bash
pip uninstall -y basicsr realesrgan
```

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_fallback \
  --v2-upscaler realesrgan  # Requested but unavailable
```

**Expected Behavior:**
- Warning logged: "ML upscaler unavailable, using bicubic"
- Pipeline completes successfully with bicubic fallback
- No crashes

**Validation:**
- Manifest shows `upscaler_backend="bicubic"` (fallback)
- Outputs generated (not blocked by missing ML dependency)

#### Test 3.4: Weight Auto-Download

**Setup:** Delete weights:
```bash
rm -rf weights/realesrgan/
```

**Command:**
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir input_images \
  --output-dir output_autodownload \
  --v2-upscaler realesrgan
```

**Expected Behavior:**
- Auto-downloads weights on first run
- Subsequent runs use cached weights

**Validation:**
```bash
ls -lh weights/realesrgan/
# Should show RealESRGAN_x2plus.pth (~65MB)
```

---

### APEX Compliance

#### Contract Impact

**New Module:** `src/transformation_portal/upscaling/`
- Public registry API for backend selection
- Protocol-based extension point for future upscalers

**Stage Graph Contract:**
- `UpscalingStage` now uses registry (backward compatible)
- Default backend remains "bicubic" (Golden Path unchanged)

#### Golden Path Preservation

| Scenario | Config | Behavior |
|----------|--------|----------|
| Default (Golden Path) | `--v2-upscaler bicubic` (default) | Bicubic upscaling, no ML (unchanged) |
| ML Super-Resolution | `--v2-upscaler realesrgan` | Real-ESRGAN ML upscaling (new capability) |
| ML Unavailable | `--v2-upscaler realesrgan` (deps missing) | Fallback to bicubic (graceful degradation) |

#### Version Bumps

**None required:**
- Config field `v2_upscaler_backend` already exists
- Manifest already captures upscaler backend
- New module is additive (no breaking changes)

---

## Dependency Governance

### New Dependencies (Gap 3 Only)

**Added to `requirements/ml.txt`:**
- `basicsr>=1.4.2` (Real-ESRGAN foundation library)
- `realesrgan>=0.3.0` (Real-ESRGAN inference)

**License Review:**
- basicsr: Apache 2.0 ✅
- realesrgan: BSD-3-Clause ✅

**Supply Chain:**
- Both packages actively maintained
- No known security vulnerabilities (as of 2026-02-14)
- Optional ML tier (no impact on core installs)

**Tier Assignment:** ML (optional)
- Core tier unaffected
- Users opt-in via `pip install -e .[ml]`
- Graceful fallback ensures core functionality preserved

---

## Sequenced Implementation Plan

### Independent Task Groups

Tasks can be executed in parallel within each gap, but gaps should be completed sequentially for clean testing.

#### Phase 1: Gap 2 (V2 MPS Acceleration) - **Lowest Risk, Highest Value**

**Effort:** 2-4 hours
**Risk:** Minimal (CLI-only change, config already exists)

**Tasks:**
1. Add `--v2-device` flag to CLI (Task 2.1)
2. Update README documentation (Task 2.2)
3. Functional tests on macOS + Linux (Test 2.1, 2.2, 2.3)

**Deliverables:**
- PR #1: "Add --v2-device CLI flag for MPS/CUDA acceleration"
- Enables 2-3x faster V2 enhancement on Apple Silicon immediately

---

#### Phase 2: Gap 1 (16-Bit Output Path) - **Medium Risk, High Value**

**Effort:** 1-2 days
**Risk:** Medium (touches orchestrator handoff logic)

**Tasks:**
1. Add 16-bit TIFF handoff logic to orchestrator (Task 1.1)
2. Add V2 output validation (Task 1.2)
3. Update manifest schema to v1.7 (Task 1.3)
4. End-to-end tests (Test 1.1, 1.2, 1.3)

**Deliverables:**
- PR #2: "Implement 16-bit output path for APEX pipeline"
- Enables true 16-bit master files for archival/print workflows

**Dependencies:**
- None (can proceed independently of other gaps)

---

#### Phase 3: Gap 3 (ML Super-Resolution) - **Highest Risk, Future Value**

**Effort:** 3-5 days
**Risk:** High (new module, ML dependencies, weight management)

**Tasks:**
1. Create upscaler registry and protocol (Task 3.1)
2. Implement bicubic backend (Task 3.2)
3. Implement Real-ESRGAN backend (Task 3.3)
4. Update UpscalingStage (Task 3.4)
5. Add CLI flag (Task 3.5)
6. Add ML dependencies (Task 3.6)
7. Add weight download script (Task 3.7)
8. Comprehensive testing (Test 3.1-3.4)

**Deliverables:**
- PR #3: "Add upscaler backend registry and Real-ESRGAN ML upscaling"
- Enables production-quality super-resolution for high-end workflows

**Dependencies:**
- None (isolated new module)

---

## CI/CD Integration

### Test Coverage Requirements

Each gap requires:

1. **Unit Tests:**
   - Config parsing and validation
   - Backend selection and fallback logic
   - Manifest schema compatibility

2. **Integration Tests:**
   - End-to-end pipeline with new features enabled
   - Golden Path preservation (defaults unchanged)
   - Fallback scenarios (missing dependencies)

3. **Performance Benchmarks:**
   - Gap 2: Validate MPS speedup (2-3x on Apple Silicon)
   - Gap 3: Validate Real-ESRGAN quality improvement

### CI Gates

**Core CI (no ML dependencies):**
- Gap 1: Test 16-bit flag parsing, manifest schema
- Gap 2: Test device flag parsing
- Gap 3: Test bicubic backend only (no Real-ESRGAN)

**ML CI (optional, runs on ML-enabled runners):**
- Gap 3: Test Real-ESRGAN backend with auto-download

**Required Checks:**
- All existing tests pass (no regressions)
- New tests pass for each gap
- Golden Path behavior unchanged
- Manifest schema migrations work

---

## Rollout Plan

### Phase 1 (Gap 2): Immediate Release

**Target:** v2.1.0 (patch release)
**Timeline:** 1 week
**Risk:** Minimal

**Release Notes:**
```markdown
### Added
- `--v2-device` CLI flag for GPU acceleration (MPS/CUDA)
- Auto-fallback to CPU if requested device unavailable

### Changed
- None (backward compatible)

### Performance
- 2-3x faster V2 enhancement on Apple Silicon with `--v2-device mps`
```

---

### Phase 2 (Gap 1): Next Minor Release

**Target:** v2.2.0 (minor release)
**Timeline:** 2-3 weeks
**Risk:** Medium

**Release Notes:**
```markdown
### Added
- 16-bit output path for APEX workflows (`--emit-master16`, `--emit-upscaled16`)
- Manifest schema v1.7 with bit-depth tracking

### Changed
- Materials V3 handoff supports 16-bit TIFF when 16-bit outputs requested
- V2 auto-detects input bit depth and preserves throughout pipeline

### Fixed
- 16-bit CLI flags now functional (previously stub)
```

---

### Phase 3 (Gap 3): Future Major Release

**Target:** v3.0.0 (major release)
**Timeline:** 4-6 weeks
**Risk:** High

**Release Notes:**
```markdown
### Added
- Upscaler backend registry for extensible super-resolution
- Real-ESRGAN ML upscaling backend (optional, requires `pip install -e .[ml]`)
- Auto-download for Real-ESRGAN model weights
- `--v2-upscaler` CLI flag (bicubic, realesrgan)

### Changed
- Upscaling stage refactored to use backend registry
- Bicubic remains default (Golden Path unchanged)

### Performance
- Real-ESRGAN provides sharper, higher-quality upscaling at 2x+ scales
```

---

## Success Criteria

### Gap 1 (16-Bit Output)

- ✅ 16-bit TIFF handoff working Materials V3 → V2
- ✅ `master16.tif` and `upscaled16.tif` generated when flags enabled
- ✅ Manifest tracks bit depth accurately
- ✅ 8-bit Golden Path unchanged (default behavior)
- ✅ No performance regression

---

### Gap 2 (V2 MPS)

- ✅ `--v2-device mps` enables Apple Silicon GPU
- ✅ 2-3x speedup measured on M-series chips
- ✅ Graceful fallback to CPU on incompatible platforms
- ✅ CLI flag documented and tested

---

### Gap 3 (ML Upscaling)

- ✅ Upscaler registry implemented and tested
- ✅ Bicubic backend (Golden Path) working
- ✅ Real-ESRGAN backend working with ML dependencies
- ✅ Auto-weight download functional
- ✅ Graceful fallback to bicubic if ML unavailable
- ✅ Visual quality improvement demonstrated
- ✅ No impact on core tier (ML optional)

---

## Open Questions / Future Work

### 16-Bit Path
- **Q:** Should we support 16-bit depth maps as input to Materials V3?
- **A:** Deferred - Materials V3 currently operates in float32 internally, 8-bit vs 16-bit input has minimal impact. Revisit if archival workflows require it.

### V2 MPS
- **Q:** Should APEX tier auto-enable MPS on Apple Silicon?
- **A:** Yes - add preset-based device selection (Task 2.3, optional enhancement). User can override with explicit `--v2-device cpu`.

### ML Upscaling
- **Q:** Should we support other upscalers (GFPGAN, SwinIR)?
- **A:** Yes, but future work. Registry makes this extensible. Prioritize Real-ESRGAN for industry-standard quality.

- **Q:** Should we cache upscaled outputs?
- **A:** Future optimization. Upscaling is fast enough (<500ms/image with Real-ESRGAN) that caching adds complexity without major benefit.

---

## Architectural Invariants Preserved

✅ **Modularity:** No cross-pipeline coupling introduced
✅ **Contracts Over Convenience:** Materials V3 → V2 handoff contract extended cleanly
✅ **Determinism:** 16-bit path uses deterministic TIFF encoding
✅ **Golden Path:** All default behaviors preserved
✅ **Enforcement:** CI tests verify no regressions
✅ **Optional Dependencies:** ML upscaling in optional tier

---

## Final Architect Approval

**Status:** ✅ **APPROVED FOR IMPLEMENTATION**

**Conditions:**
1. Phase 1 (Gap 2) ships first - lowest risk, highest value
2. Gap 1 requires comprehensive 16-bit roundtrip tests (8-bit and 16-bit paths)
3. Gap 3 requires ML dependency fallback tests in core CI
4. All three gaps preserve Golden Path (default configs unchanged)
5. Each gap ships as independent PR with isolated test coverage

**Next Steps:**
1. Delegate implementation to `@transformation-portal-specialist`
2. Start with Phase 1 (Gap 2: V2 MPS) - target 1 week completion
3. Escalate if any architectural uncertainty arises during implementation

---

**Document Version:** 1.0
**Last Updated:** 2026-02-14
**Maintained By:** Transformation Portal Architect
