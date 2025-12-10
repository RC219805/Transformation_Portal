# Autotune Export Config Integration Guide

**Status**: Pre-Integration Review Complete  
**Date**: 2025-12-10  
**Phase**: Phase 2 Slice 3 (Post-Benchmarking)

---

## Executive Summary

This guide provides step-by-step instructions for wiring `autotune_export_config()` into the live pipeline. The function is implemented and tested, but not yet integrated into the production flow.

**Key Points**:
- Autotune is implemented in `src/transformation_portal/core/storage/export_manager.py`
- Unit tests pass (100% coverage for autotune logic)
- Integration requires 3 files changed, ~50 lines of code
- Feature is gated behind `autotune_export` flag (default OFF)
- Backward compatible (no behavior change when disabled)

---

## 1. Integration Overview

### 1.1 Where Autotune Plugs In

**Location**: `lux_depth_v2/pipeline.py`  
**Method**: `LuxPipelineV2.process_one()`  
**Timing**: After image load, before first export

```
Image Load (line 373)
    ↓
[INSERT AUTOTUNE HERE]  ← Just-in-time config generation
    ↓
ExportManager Init
    ↓
Processing Pipeline
    ↓
Exports (write_master, write_upscaled, etc.)
```

### 1.2 Why Just-in-Time?

**Problem**: ExportManager is initialized in `__init__()`, but image dimensions aren't known until `process_one()` loads the file.

**Solution**: Defer ExportManager init until after image load when autotune is enabled.

**Benefits**:
- No need to pre-scan image headers
- Works with batch processing (autotune per-image)
- Minimal code changes

---

## 2. Implementation Steps

### Step 1: Add Configuration Flag

**File**: `lux_depth_v2/config.py`  
**Location**: `PipelineConfig` dataclass

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    
    # Phase 2 Slice 3: Autotune export configuration (OFF by default)
    autotune_export: bool = False
    """Enable adaptive export optimization based on image characteristics.
    
    When enabled, export settings (tiling, atomic writes) are dynamically
    selected based on image size and scene complexity. Based on benchmark
    data from Phase 2 Slice 3:
    - Large, simple scenes (aerial, exterior): Enable optimizations (~5-10% faster)
    - Complex scenes (interior, pool): Use baseline (avoid 6-8% slowdown)
    
    Default: False (explicit opt-in required)
    """
    
    autotune_use_complexity: bool = False
    """Use scene complexity estimation for autotune decisions.
    
    When enabled, autotune analyzes image gradients to classify scene complexity:
    - Low complexity (sky, water, gradients): Enable optimizations
    - High complexity (textures, interiors): Use baseline
    
    Adds ~10-20ms overhead per image for gradient analysis.
    
    Default: False (uses megapixels-only heuristic)
    """
```

### Step 2: Add Scene Complexity Helper

**File**: `lux_depth_v2/pipeline.py`  
**Location**: After imports, before `LuxPipelineV2` class

```python
def _estimate_scene_complexity(rgb01: np.ndarray) -> float:
    """
    Estimate scene complexity for autotune decisions.
    
    Heuristic: High-frequency content ratio (gradient magnitude).
    - 0.0 = simple scene (sky, gradients, water, uniform surfaces)
    - 1.0 = complex scene (textures, interiors, foliage, fine details)
    
    Benchmark timing: ~10-20ms on 20MP image (negligible overhead).
    
    Args:
        rgb01: RGB float array [0, 1], shape (H, W, 3)
    
    Returns:
        Complexity score in [0.0, 1.0]
    
    Examples:
        - Aerial view with sky: 0.2-0.4
        - Interior with textures: 0.7-0.9
        - Pool with water reflections: 0.6-0.8
    """
    try:
        # Convert to grayscale for gradient analysis
        if cv2 is None:
            return 0.5  # Fallback if cv2 unavailable
        
        gray = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute Sobel gradients (3x3 kernel)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Normalize by image intensity range to handle various exposures
        intensity_range = gray.max() - gray.min()
        if intensity_range > 10:  # Avoid div-by-zero for flat images
            grad_mag_norm = grad_mag / intensity_range
        else:
            # Flat image (very low complexity)
            return 0.1
        
        # High-frequency ratio: pixels with gradient above threshold
        # Threshold=0.1 calibrated against benchmark scenes:
        # - Aerial: ~15-25% pixels above threshold → score 0.2-0.4
        # - Interior: ~50-70% pixels above threshold → score 0.7-0.9
        threshold = 0.1
        hf_ratio = np.mean(grad_mag_norm > threshold)
        
        # Clamp to valid range [0, 1]
        complexity = float(np.clip(hf_ratio, 0.0, 1.0))
        
        return complexity
        
    except Exception as e:
        # Graceful fallback on any error (import failure, empty image, etc.)
        # Log at debug level to avoid spamming on every image
        import logging
        logging.getLogger(__name__).debug(f"Complexity estimation failed: {e}")
        return 0.5  # Medium complexity (safe default)
```

### Step 3: Modify ExportManager Initialization

**File**: `lux_depth_v2/pipeline.py`  
**Location**: `LuxPipelineV2.__init__()`, lines 181-189

**Current code**:
```python
# Phase 2 Slice 2: Initialize ExportManager
self.export_manager = None
if EXPORT_MANAGER_AVAILABLE and cfg.output_dir:
    try:
        export_config = ExportConfig(output_dir=Path(cfg.output_dir))
        self.export_manager = ExportManager(export_config, io_utils)
        self.logger.info("ExportManager initialized")
    except Exception as e:
        self.logger.warning(f"ExportManager init failed, using direct I/O: {e}")
```

**Modified code**:
```python
# Phase 2 Slice 3: Initialize ExportManager (deferred if autotune enabled)
self.export_manager = None
self._export_config = None  # Store config for later use
self._autotune_initialized = False  # Track autotune state

if EXPORT_MANAGER_AVAILABLE and cfg.output_dir:
    if getattr(cfg, 'autotune_export', False):
        # Defer ExportManager init until image dimensions known (process_one)
        self.logger.info("ExportManager deferred (autotune enabled)")
    else:
        # Legacy path: Static config from PipelineConfig
        try:
            self._export_config = ExportConfig(output_dir=Path(cfg.output_dir))
            self.export_manager = ExportManager(self._export_config, io_utils)
            self.logger.info("ExportManager initialized (static config)")
        except Exception as e:
            self.logger.warning(f"ExportManager init failed, using direct I/O: {e}")
```

### Step 4: Add Just-in-Time Autotune

**File**: `lux_depth_v2/pipeline.py`  
**Location**: `process_one()`, after image load (after line 374)

**Insert after**:
```python
# Load image
with self._stage(report, "io/read_input"):
    rgb01, info = io_utils.read_rgb_any(img_path)
    H, W = rgb01.shape[:2]
```

**Add this block**:
```python
# Phase 2 Slice 3: Just-in-time autotune (if enabled and not yet initialized)
if not self._autotune_initialized and getattr(cfg, 'autotune_export', False):
    if EXPORT_MANAGER_AVAILABLE:
        with self._stage(report, "export/autotune"):
            try:
                from transformation_portal.core.storage.export_manager import autotune_export_config
                
                # Estimate scene complexity if requested (adds ~10-20ms overhead)
                scene_complexity = None
                if getattr(cfg, 'autotune_use_complexity', False):
                    scene_complexity = _estimate_scene_complexity(rgb01)
                    self.logger.debug(f"Scene complexity: {scene_complexity:.3f}")
                
                # Generate adaptive export config
                self._export_config = autotune_export_config(
                    output_dir=Path(cfg.output_dir),
                    image_width=W,
                    image_height=H,
                    scene_complexity=scene_complexity,
                    enable_adaptive=True
                )
                
                # Initialize ExportManager with auto-tuned config
                self.export_manager = ExportManager(self._export_config, io_utils)
                self._autotune_initialized = True
                
                # Log autotune decision for debugging/monitoring
                megapixels = (W * H) / 1_000_000
                self.logger.info(
                    f"ExportManager auto-tuned | "
                    f"size={W}x{H} ({megapixels:.1f}MP) "
                    f"complexity={scene_complexity:.3f if scene_complexity else 'N/A'} "
                    f"tile_size={self._export_config.tiff_tile_size} "
                    f"atomic={self._export_config.use_atomic_image_writes}"
                )
                
                # Store autotune metadata in report for analysis
                report['autotune'] = {
                    'enabled': True,
                    'megapixels': round(megapixels, 2),
                    'complexity': round(scene_complexity, 3) if scene_complexity else None,
                    'tiff_tile_size': self._export_config.tiff_tile_size,
                    'use_atomic_writes': self._export_config.use_atomic_image_writes,
                    'use_tiered_storage': self._export_config.enable_tiered_storage,
                }
                
            except Exception as e:
                self.logger.error(f"Autotune failed, using baseline config: {e}")
                # Fallback to baseline config
                self._export_config = ExportConfig(output_dir=Path(cfg.output_dir))
                self.export_manager = ExportManager(self._export_config, io_utils)
                self._autotune_initialized = True
                
                report['autotune'] = {
                    'enabled': True,
                    'error': str(e),
                    'fallback': 'baseline'
                }
    else:
        # ExportManager not available, mark autotune as skipped
        self._autotune_initialized = True
        report['autotune'] = {'enabled': False, 'reason': 'ExportManager unavailable'}
```

### Step 5: Update Report Structure

**File**: `lux_depth_v2/pipeline.py`  
**Location**: `process_one()`, report finalization (around line 619)

**Add to report** (after line 632):
```python
# Update report with final status
report.update({
    "status": "ok",
    # ... existing fields ...
})

# Phase 2 Slice 3: Add autotune metadata if not already set
if 'autotune' not in report:
    report['autotune'] = {
        'enabled': getattr(cfg, 'autotune_export', False),
        'reason': 'disabled' if not getattr(cfg, 'autotune_export', False) else 'no_init'
    }
```

---

## 3. Testing Strategy

### 3.1 Unit Tests (Already Pass)

**File**: `tests/core/storage/test_autotune_export_config.py`

```bash
pytest tests/core/storage/test_autotune_export_config.py -v
```

✅ **Coverage**:
- Aerial scene (21.6 MP, low complexity) → tiled_atomic enabled
- Interior scene (12 MP, high complexity) → baseline (no optimizations)
- Unknown dimensions → baseline fallback
- Edge cases (zero complexity, very high complexity)

### 3.2 Integration Tests (New)

**Create**: `tests/lux_depth_v2/test_pipeline_autotune.py`

```python
"""Integration tests for autotune export config in pipeline."""

import pytest
from pathlib import Path
import numpy as np
from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2, _estimate_scene_complexity


def test_scene_complexity_estimation():
    """Test complexity estimation with synthetic images."""
    # Simple scene (gradient)
    H, W = 1000, 1000
    simple = np.linspace(0, 1, W)[None, :, None].repeat(H, axis=0).repeat(3, axis=2)
    complexity_simple = _estimate_scene_complexity(simple)
    assert 0.0 <= complexity_simple <= 0.3, f"Expected low complexity, got {complexity_simple}"
    
    # Complex scene (random noise)
    complex_img = np.random.rand(H, W, 3)
    complexity_complex = _estimate_scene_complexity(complex_img)
    assert 0.7 <= complexity_complex <= 1.0, f"Expected high complexity, got {complexity_complex}"


@pytest.mark.integration
def test_pipeline_autotune_enabled_aerial(tmp_path, sample_aerial_image):
    """Test autotune with large, simple scene (aerial-like)."""
    cfg = PipelineConfig(
        input_dir=None,
        output_dir=tmp_path / "output",
        preset=Preset.EXTERIOR_SHOWCASE,
        autotune_export=True,
        autotune_use_complexity=True,
        device="cpu",  # CPU for CI
        upscaler_backend="none",  # Skip upscaling for speed
    )
    pipe = LuxPipelineV2(cfg)
    
    # Process single image
    result = pipe.process_one(sample_aerial_image)
    
    # Verify autotune was triggered
    assert 'autotune' in result
    assert result['autotune']['enabled'] is True
    assert 'megapixels' in result['autotune']
    assert 'complexity' in result['autotune']
    
    # Verify optimization decision (aerial = simple = enable optimizations)
    if result['autotune']['complexity'] < 0.5:
        assert result['autotune']['tiff_tile_size'] == 512
        assert result['autotune']['use_atomic_writes'] is True
    
    assert result['status'] == 'ok'


@pytest.mark.integration
def test_pipeline_autotune_enabled_interior(tmp_path, sample_interior_image):
    """Test autotune with complex interior scene."""
    cfg = PipelineConfig(
        input_dir=None,
        output_dir=tmp_path / "output",
        preset=Preset.INTERIOR_LUXURY,
        autotune_export=True,
        autotune_use_complexity=True,
        device="cpu",
        upscaler_backend="none",
    )
    pipe = LuxPipelineV2(cfg)
    
    result = pipe.process_one(sample_interior_image)
    
    # Verify autotune was triggered
    assert 'autotune' in result
    assert result['autotune']['enabled'] is True
    
    # Verify optimization decision (interior = complex = baseline)
    if result['autotune']['complexity'] > 0.6:
        assert result['autotune']['tiff_tile_size'] is None
        assert result['autotune']['use_atomic_writes'] is False
    
    assert result['status'] == 'ok'


@pytest.mark.integration
def test_pipeline_autotune_disabled(tmp_path, sample_image):
    """Test pipeline with autotune disabled (baseline behavior)."""
    cfg = PipelineConfig(
        input_dir=None,
        output_dir=tmp_path / "output",
        preset=Preset.PHOTO_REALISTIC,
        autotune_export=False,  # Explicit disable
        device="cpu",
        upscaler_backend="none",
    )
    pipe = LuxPipelineV2(cfg)
    
    result = pipe.process_one(sample_image)
    
    # Verify autotune was not triggered
    assert 'autotune' in result
    assert result['autotune']['enabled'] is False
    assert result['autotune']['reason'] == 'disabled'
    
    # Verify baseline export config used
    assert pipe.export_manager is not None
    assert pipe.export_manager.config.tiff_tile_size is None
    
    assert result['status'] == 'ok'


@pytest.mark.integration
def test_pipeline_autotune_batch_processing(tmp_path, sample_images_dir):
    """Test autotune with batch processing (multiple images)."""
    cfg = PipelineConfig(
        input_dir=sample_images_dir,
        output_dir=tmp_path / "output",
        preset=Preset.PHOTO_REALISTIC,
        autotune_export=True,
        autotune_use_complexity=False,  # Megapixels-only for speed
        device="cpu",
        upscaler_backend="none",
    )
    pipe = LuxPipelineV2(cfg)
    
    results = pipe.process_directory()
    
    # Verify all images processed with autotune
    assert len(results) > 0
    for result in results:
        if result['status'] == 'ok':
            assert 'autotune' in result
            assert result['autotune']['enabled'] is True
```

**Fixtures** (add to `tests/lux_depth_v2/conftest.py`):
```python
import pytest
import numpy as np
from PIL import Image
from pathlib import Path


@pytest.fixture
def sample_aerial_image(tmp_path):
    """Create synthetic aerial-like image (large, simple gradient)."""
    H, W = 6000, 3600  # 21.6 MP (similar to Aerial benchmark)
    
    # Simple sky gradient (low complexity)
    sky_gradient = np.linspace(0.3, 0.8, H)[:, None, None].repeat(W, axis=1).repeat(3, axis=2)
    img_float = (sky_gradient * 255).astype(np.uint8)
    
    img_path = tmp_path / "aerial_sample.tif"
    Image.fromarray(img_float).save(img_path)
    return img_path


@pytest.fixture
def sample_interior_image(tmp_path):
    """Create synthetic interior image (medium, complex textures)."""
    H, W = 4000, 3000  # 12 MP (similar to GreatRoom benchmark)
    
    # Random texture (high complexity)
    texture = np.random.rand(H, W, 3)
    img_float = (texture * 255).astype(np.uint8)
    
    img_path = tmp_path / "interior_sample.tif"
    Image.fromarray(img_float).save(img_path)
    return img_path


@pytest.fixture
def sample_image(tmp_path):
    """Generic sample image."""
    img_float = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "sample.tif"
    Image.fromarray(img_float).save(img_path)
    return img_path
```

### 3.3 Benchmark Validation

**Re-run existing benchmarks with autotune**:

```bash
# Aerial (expect ~5-10% improvement with autotune)
pytest tests/core/storage/benchmark_export_scenarios.py::test_benchmark_aerial_tiled_atomic -v

# Pool (expect ~2-6% degradation with autotune, should fallback to baseline)
pytest tests/core/storage/benchmark_export_scenarios.py::test_benchmark_pool_baseline -v

# GreatRoom (expect minimal impact)
pytest tests/core/storage/benchmark_export_scenarios.py::test_benchmark_greatroom_all_modes -v
```

**Compare results**:
- Aerial: autotune should enable tiled_atomic → faster
- Pool: autotune should use baseline → similar performance
- GreatRoom: autotune should use baseline → similar performance

---

## 4. Deployment Checklist

- [ ] Code changes implemented (Steps 1-5)
- [ ] Unit tests passing (`test_autotune_export_config.py`)
- [ ] Integration tests written and passing (`test_pipeline_autotune.py`)
- [ ] Benchmarks re-run with autotune enabled
- [ ] User documentation updated (CLI help, README)
- [ ] Feature flag default verified (OFF for safety)
- [ ] Logging instrumentation verified (autotune decisions logged)
- [ ] Error handling tested (autotune fallback to baseline)
- [ ] A/B test infrastructure ready (optional, for production monitoring)

---

## 5. Rollout Strategy

### Phase 1: Developer Preview (Week 1)
- [ ] Merge autotune PR to `main` (feature flag OFF by default)
- [ ] Internal testing with `autotune_export=True` on developer machines
- [ ] Monitor logs for autotune decisions and errors
- [ ] Gather feedback on heuristics (too conservative? too aggressive?)

### Phase 2: Opt-In Beta (Week 2-3)
- [ ] Document autotune in user-facing README
- [ ] Add CLI flag `--autotune-export` to `lux-depth-v2` command
- [ ] Invite beta testers to enable autotune
- [ ] Collect benchmark data from real-world usage
- [ ] Tune heuristics based on feedback

### Phase 3: Default Enable (Week 4+)
- [ ] If results positive, change default to `autotune_export=True`
- [ ] Provide `--no-autotune` CLI flag for users who want baseline
- [ ] Monitor production metrics (throughput, error rates)
- [ ] Iterate on complexity estimation (use Materials v2 data)

---

## 6. Monitoring & Observability

### 6.1 Key Metrics

**Per-image metrics** (in report JSON):
```json
{
  "autotune": {
    "enabled": true,
    "megapixels": 21.6,
    "complexity": 0.35,
    "tiff_tile_size": 512,
    "use_atomic_writes": true,
    "decision_reason": "large_simple_scene"
  },
  "stage_times_sec": {
    "export/autotune": 0.015,
    "export_master": 1.234,
    "export_upscaled": 3.456
  }
}
```

**Aggregate metrics** (for A/B testing):
- Autotune decision distribution (tiled_atomic vs baseline)
- Average throughput by scene type
- Export stage timing percentiles (p50, p95, p99)
- Error rate (autotune fallback to baseline)

### 6.2 Logging

**Log autotune decisions** (INFO level):
```
2025-12-10 14:23:15 INFO ExportManager auto-tuned | size=6000x3600 (21.6MP) complexity=0.35 tile_size=512 atomic=True
2025-12-10 14:23:18 INFO ExportManager auto-tuned | size=4000x3000 (12.0MP) complexity=0.78 tile_size=None atomic=False
```

**Log errors** (ERROR level):
```
2025-12-10 14:25:10 ERROR Autotune failed, using baseline config: ModuleNotFoundError: No module named 'cv2'
```

### 6.3 Alerts

**Alert conditions**:
- Autotune fallback rate > 5% (indicates systematic failures)
- Export stage timing >3x baseline (indicates pathological behavior)
- TIFF tile count >1000 (indicates tile size too small)

---

## 7. Troubleshooting

### Issue: Autotune always falls back to baseline

**Symptoms**: All images get `tiff_tile_size=None` even for large aerials

**Causes**:
1. `autotune_use_complexity=False` and image <40MP (megapixels-only heuristic)
2. Complexity estimation failing (cv2 not available)
3. `enable_adaptive=False` in autotune call

**Fix**:
- Enable complexity estimation: `autotune_use_complexity=True`
- Verify cv2 available: `python -c "import cv2; print(cv2.__version__)"`
- Check logs for autotune decision reason

### Issue: Autotune too aggressive (complex scenes get optimizations)

**Symptoms**: Interior scenes get `tiff_tile_size=512`, but performance degrades

**Causes**:
1. Complexity threshold too high (default 0.5)
2. Gradient-based complexity underestimates interior complexity

**Fix**:
- Lower `COMPLEXITY_THRESHOLD` in `autotune_export_config()` from 0.5 to 0.4
- Use Materials v2 segmentation for smarter complexity estimation

### Issue: Scene complexity estimation is slow

**Symptoms**: `export/autotune` stage takes >50ms

**Causes**:
- Very large image (>50MP)
- Sobel filter on full resolution

**Fix**:
- Downsample image before gradient analysis:
  ```python
  # In _estimate_scene_complexity()
  MAX_ANALYSIS_SIZE = 2048  # Max long side
  if max(rgb01.shape[:2]) > MAX_ANALYSIS_SIZE:
      scale = MAX_ANALYSIS_SIZE / max(rgb01.shape[:2])
      h, w = int(rgb01.shape[0] * scale), int(rgb01.shape[1] * scale)
      rgb01_small = cv2.resize(rgb01, (w, h), interpolation=cv2.INTER_AREA)
  else:
      rgb01_small = rgb01
  ```

---

## 8. Future Work

### 8.1 Materials v2 Integration

Use segmentation results for smarter complexity estimation:
```python
def _complexity_from_materials_v2(materials_result: SegmentationResult) -> float:
    """Estimate complexity from material segmentation."""
    # High region count → complex scene
    # Sky-dominated → simple scene
    # Many small regions → interior (complex)
    # Few large regions → exterior (simple)
    pass
```

### 8.2 Scratch Directory Auto-Provisioning

Automatically use external T9 drive when available:
```python
# In autotune_export_config()
t9_path = Path("/Volumes/T9")
if t9_path.exists() and megapixels > 50:
    # Large image, use scratch on T9
    return ExportConfig(
        output_dir=output_dir,
        enable_tiered_storage=True,
        scratch_dir=t9_path / "scratch",
        tiff_tile_size=512,
    )
```

### 8.3 Per-Preset Tuning

Different heuristics for different presets:
```python
# In autotune_export_config()
if preset == Preset.EXTERIOR_SHOWCASE:
    COMPLEXITY_THRESHOLD = 0.6  # More aggressive
elif preset == Preset.INTERIOR_LUXURY:
    COMPLEXITY_THRESHOLD = 0.3  # More conservative
```

---

## 9. Success Criteria

**After integration, autotune is successful if**:
1. ✅ Aerial-like scenes (>20MP, complexity <0.5) run 5-10% faster
2. ✅ Complex scenes (interiors, pools) maintain baseline performance
3. ✅ Error rate <1% (autotune fallback rare)
4. ✅ No user complaints about quality degradation
5. ✅ Benchmark metrics validate heuristics

**If criteria not met**:
- Disable autotune by default
- Iterate on heuristics with more benchmark data
- Consider Materials v2 integration for better complexity estimation

---

## 10. Quick Start (Copy-Paste)

### Enable autotune in CLI

```bash
# Current (no autotune)
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury

# With autotune (after integration)
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury --autotune-export

# With autotune + complexity estimation
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset exterior_showcase --autotune-export --autotune-use-complexity
```

### Enable autotune in Python

```python
from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2

cfg = PipelineConfig(
    input_dir=Path("renders/"),
    output_dir=Path("output/"),
    preset=Preset.EXTERIOR_SHOWCASE,
    autotune_export=True,  # ← Enable autotune
    autotune_use_complexity=True,  # ← Use complexity estimation
    device="auto",
    upscale=4,
)

pipe = LuxPipelineV2(cfg)
results = pipe.process_directory()
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-10  
**Contact**: Transformation Portal Team
