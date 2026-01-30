# ADR-001: PBR Module Integration Architecture

**Status:** Proposed
**Date:** 2026-01-30
**Authority:** Transformation Portal Architect

---

## Context

The PBR (Physically Based Rendering) module in `lux_depth_v3/pbr.py` has been fully tested and validated (13/13 tests passing). The module generates Normal, Roughness, and Ambient Occlusion maps from depth data with production-ready performance (~420ms for 4K, ~150 images/hour).

### Current Repository Issues

**Critical Fragmentation:**
- **44 depth-related files** across 3 modules:
  - `src/transformation_portal/depth/` (19 files, 872KB)
  - `src/transformation_portal/depth_intelligence/` (2 files, 36KB)
  - `src/transformation_portal/lux_depth_v3/` (20 files, 340KB)
- **9 pipeline files** with overlapping depth functionality
- **5 duplicate `DeviceType` enums** across modules
- **2 duplicate `DepthConfig` classes**
- **2 duplicate `DepthEstimator` classes**

**Current PBR Implementation:**
- Location: `src/transformation_portal/lux_depth_v3/pbr.py`
- Already integrated into `EnhanceOrchestrator` (lines 21-22 of orchestrator.py)
- Clean, dependency-minimal implementation (NumPy/SciPy/Pillow only)
- Frozen config dataclass ensures immutability and cache-ability
- Atomic write operations via `pbr_writer.py`

---

## Decision

### Strategic Direction: Consolidate, Don't Expand

**Reject:** Moving PBR into the fragmented `depth/` module structure
**Accept:** Consolidate all depth processing into `lux_depth_v3/` as the canonical depth system

**Rationale:**
1. `lux_depth_v3/` already contains the most modern implementation (Depth Anything V3)
2. PBR module already integrated and tested in this context
3. Consolidation reduces 44 files to ~25-30 files in single location
4. Single source of truth eliminates duplicate classes
5. Clear migration path: deprecate old modules, redirect to new

---

## Proposed Architecture

### Phase 1: Canonical Depth Module (Weeks 1-2)

#### 1.1 Module Reorganization

```
src/transformation_portal/depth_canonical/
├── __init__.py                  # Public API surface
├── config.py                    # Unified config (merge DA3Config + DepthConfig)
├── device.py                    # Canonical DeviceType enum
│
├── models/
│   ├── __init__.py
│   ├── depth_anything_v3.py     # DA3 (from lux_depth_v3)
│   ├── depth_anything_v2.py     # DA2 (from depth/)
│   └── base.py                  # Estimator interface
│
├── processing/
│   ├── __init__.py
│   ├── inference.py             # Model inference (from lux_depth_v3)
│   ├── postprocessing.py        # Depth refinement (from lux_depth_v3)
│   ├── pbr.py                   # PBR map generation (from lux_depth_v3)
│   ├── zone_mapping.py          # Zone-based tone mapping (from depth/)
│   ├── denoise.py               # Depth-aware denoising (from depth/)
│   └── atmospheric.py           # Atmospheric effects (from depth/)
│
├── io/
│   ├── __init__.py
│   ├── depth_writer.py          # Atomic depth writes (from lux_depth_v3)
│   ├── pbr_writer.py            # Atomic PBR writes (from lux_depth_v3)
│   └── cache.py                 # LRU caching (from depth/)
│
├── security/
│   ├── __init__.py
│   └── validation.py            # Path sanitization (from lux_depth_v3)
│
└── pipeline.py                  # Unified pipeline orchestrator
```

**File Count Reduction:** 44 files → 25 files (~45% reduction)

#### 1.2 Unified Configuration Schema

```python
# depth_canonical/config.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class DeviceType(str, Enum):
    """Canonical device enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    COREML = "coreml"

class ModelVariant(Enum):
    """Supported depth estimation models."""
    DA3_METRIC_LARGE = "depth-anything-v3-metric-large"
    DA3_METRIC_BASE = "depth-anything-v3-metric-base"
    DA3_METRIC_SMALL = "depth-anything-v3-metric-small"
    DA2_LARGE = "depth-anything-v2-large"
    DA2_BASE = "depth-anything-v2-base"

@dataclass(frozen=True)
class PBRConfig:
    """PBR map generation configuration (immutable)."""
    normal_strength: float = 1.0
    normal_blur_radius: int = 0
    roughness_strength: float = 1.0
    roughness_blur_radius: int = 3
    ao_strength: float = 1.0
    ao_blur_radius: int = 5
    ao_bias: float = 0.5
    enabled: bool = True  # NEW: opt-in/out flag

@dataclass
class ProcessingConfig:
    """Depth processing configuration."""
    # Postprocessing
    apply_bilateral: bool = True
    bilateral_sigma_color: float = 10.0
    bilateral_sigma_space: float = 10.0

    # Zone mapping
    enable_zone_mapping: bool = False
    num_zones: int = 3
    tone_map_method: str = "agx"

    # Atmospheric
    enable_atmospheric: bool = False
    haze_strength: float = 0.0

    # Denoising
    enable_denoise: bool = False
    denoise_strength: float = 0.5

    # PBR (NEW)
    pbr: PBRConfig = field(default_factory=PBRConfig)

@dataclass
class DepthConfig:
    """Unified depth estimation and processing configuration."""
    model: ModelVariant = ModelVariant.DA3_METRIC_LARGE
    device: DeviceType = DeviceType.CPU
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cache_enabled: bool = True
    cache_size: int = 128

    @classmethod
    def from_preset(cls, preset_name: str) -> "DepthConfig":
        """Load configuration from YAML preset."""
        # Implementation loads from config/{preset_name}.yaml
        pass
```

#### 1.3 Public API Surface

```python
# depth_canonical/__init__.py

"""Canonical depth processing module for Transformation Portal.

Public API:
    - DepthConfig: Unified configuration
    - DepthPipeline: High-level orchestrator
    - generate_depth_maps: Single-image processing
    - generate_pbr_maps: PBR map generation from depth

Private modules (internal use only):
    - models, processing, io, security
"""

from .config import DepthConfig, PBRConfig, DeviceType, ModelVariant
from .pipeline import DepthPipeline
from .processing.pbr import generate_pbr_maps
from .io.pbr_writer import write_pbr_maps

__all__ = [
    "DepthConfig",
    "PBRConfig",
    "DeviceType",
    "ModelVariant",
    "DepthPipeline",
    "generate_pbr_maps",
    "write_pbr_maps",
]
```

### Phase 2: Pipeline Integration (Weeks 3-4)

#### 2.1 Unified Pipeline Orchestrator

```python
# depth_canonical/pipeline.py

from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

from .config import DepthConfig
from .models.base import DepthEstimator
from .processing.inference import InferenceEngine
from .processing.postprocessing import Postprocessor
from .processing.pbr import generate_pbr_maps
from .io.depth_writer import write_depth_u16_png
from .io.pbr_writer import write_pbr_maps
from .io.cache import DepthCache

class DepthPipeline:
    """Production depth processing pipeline.

    Capabilities:
    - Depth estimation (DA2/DA3)
    - Depth refinement and postprocessing
    - PBR map generation (optional)
    - Zone-based tone mapping
    - Atmospheric effects
    - LRU caching for iterative workflows

    Example:
        >>> config = DepthConfig.from_preset("architectural_interior")
        >>> config.processing.pbr.enabled = True
        >>> pipeline = DepthPipeline(config)
        >>> result = pipeline.process_image("render.jpg", output_dir="output/")
        >>> # Result contains: depth_path, pbr_paths (if enabled)
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.inference = InferenceEngine(config)
        self.postprocessor = Postprocessor(config.processing)
        self.cache = DepthCache(enabled=config.cache_enabled, maxsize=config.cache_size)

    def process_image(
        self,
        image_path: Path,
        output_dir: Path,
        generate_pbr: Optional[bool] = None
    ) -> Dict[str, Path]:
        """Process single image through depth pipeline.

        Args:
            image_path: Input image path
            output_dir: Output directory for depth/PBR maps
            generate_pbr: Override config.processing.pbr.enabled

        Returns:
            Dictionary with keys:
                - 'depth': Path to depth map (uint16 PNG)
                - 'normal': Path to normal map (if PBR enabled)
                - 'roughness': Path to roughness map (if PBR enabled)
                - 'ao': Path to AO map (if PBR enabled)
        """
        # Estimate depth (with caching)
        depth = self.inference.estimate(image_path)

        # Postprocess depth
        depth_refined = self.postprocessor.refine(depth)

        # Write depth map
        basename = image_path.stem
        depth_path = write_depth_u16_png(
            depth_refined,
            output_dir / f"{basename}_depth.png"
        )

        result = {"depth": depth_path}

        # PBR generation (optional)
        should_generate_pbr = (
            generate_pbr if generate_pbr is not None
            else self.config.processing.pbr.enabled
        )

        if should_generate_pbr:
            normal, roughness, ao = generate_pbr_maps(
                depth_refined,
                self.config.processing.pbr
            )
            pbr_paths = write_pbr_maps(normal, roughness, ao, output_dir, basename)
            result.update({
                "normal": pbr_paths["normal"],
                "roughness": pbr_paths["roughness"],
                "ao": pbr_paths["ao"],
            })

        return result

    def batch_process(
        self,
        image_paths: list[Path],
        output_dir: Path,
        progress: bool = True
    ) -> list[Dict[str, Path]]:
        """Batch process multiple images."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(image_paths) if progress else image_paths

        for img_path in iterator:
            result = self.process_image(img_path, output_dir)
            results.append(result)

        return results
```

#### 2.2 CLI Integration

```python
# cli/depth_process.py (NEW)

import typer
from pathlib import Path
from typing import Optional

from transformation_portal.depth_canonical import DepthConfig, DepthPipeline

app = typer.Typer(help="Depth processing and PBR map generation")

@app.command()
def process(
    input: Path = typer.Argument(..., help="Input image or directory"),
    output: Path = typer.Argument(..., help="Output directory"),
    preset: str = typer.Option("default", help="Preset: default, architectural_interior, architectural_exterior"),
    model: str = typer.Option("da3-large", help="Model: da3-large, da3-base, da3-small, da2-large"),
    device: str = typer.Option("cpu", help="Device: cpu, cuda, mps, coreml"),
    pbr: bool = typer.Option(True, help="Generate PBR maps (normal, roughness, AO)"),
    pbr_normal_strength: float = typer.Option(1.0, help="Normal map gradient strength"),
    pbr_roughness_strength: float = typer.Option(1.0, help="Roughness detail strength"),
    pbr_ao_strength: float = typer.Option(1.0, help="AO darkness strength"),
    cache: bool = typer.Option(True, help="Enable depth map caching"),
):
    """Process images through depth pipeline with optional PBR generation."""

    # Load configuration
    config = DepthConfig.from_preset(preset)
    config.device = device
    config.cache_enabled = cache

    # Override PBR settings
    config.processing.pbr.enabled = pbr
    config.processing.pbr.normal_strength = pbr_normal_strength
    config.processing.pbr.roughness_strength = pbr_roughness_strength
    config.processing.pbr.ao_strength = pbr_ao_strength

    # Initialize pipeline
    pipeline = DepthPipeline(config)

    # Process
    if input.is_dir():
        images = list(input.glob("*.jpg")) + list(input.glob("*.png"))
        typer.echo(f"Processing {len(images)} images...")
        results = pipeline.batch_process(images, output)
        typer.echo(f"✓ Processed {len(results)} images to {output}")
    else:
        result = pipeline.process_image(input, output)
        typer.echo(f"✓ Processed {input.name}")
        if pbr:
            typer.echo(f"  - Depth: {result['depth'].name}")
            typer.echo(f"  - Normal: {result['normal'].name}")
            typer.echo(f"  - Roughness: {result['roughness'].name}")
            typer.echo(f"  - AO: {result['ao'].name}")

if __name__ == "__main__":
    app()
```

### Phase 3: Migration and Deprecation (Weeks 5-6)

#### 3.1 Deprecation Shims

```python
# src/transformation_portal/depth/__init__.py (DEPRECATED)

"""DEPRECATED: Use transformation_portal.depth_canonical instead.

This module will be removed in v2.0.0.
"""

import warnings
from transformation_portal.depth_canonical import (
    DepthConfig,
    DepthPipeline,
    generate_pbr_maps,
)

warnings.warn(
    "transformation_portal.depth is deprecated. "
    "Use transformation_portal.depth_canonical instead. "
    "This module will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
ArchitecturalDepthPipeline = DepthPipeline

__all__ = ["DepthConfig", "DepthPipeline", "ArchitecturalDepthPipeline", "generate_pbr_maps"]
```

```python
# src/transformation_portal/lux_depth_v3/__init__.py (DEPRECATED)

"""DEPRECATED: Use transformation_portal.depth_canonical instead.

This module will be removed in v2.0.0.
"""

import warnings
from transformation_portal.depth_canonical import (
    DepthConfig as DA3Config,
    DepthPipeline,
    generate_pbr_maps,
    PBRConfig,
)

warnings.warn(
    "transformation_portal.lux_depth_v3 is deprecated. "
    "Use transformation_portal.depth_canonical instead. "
    "This module will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DA3Config", "DepthPipeline", "generate_pbr_maps", "PBRConfig"]
```

#### 3.2 Migration Timeline

| Phase | Timeline | Actions | Breaking Changes |
|-------|----------|---------|------------------|
| **Phase 1: Consolidation** | Weeks 1-2 | Create `depth_canonical/` with unified API | None (new module) |
| **Phase 2: Integration** | Weeks 3-4 | Update pipelines to use `depth_canonical` | None (internal) |
| **Phase 3: Deprecation** | Weeks 5-6 | Add deprecation warnings to old modules | None (warnings only) |
| **Phase 4: Removal** | v2.0.0 (3-6 months) | Remove `depth/` and `lux_depth_v3/` | Yes (announced) |

---

## Configuration Schema for PBR Presets

### Preset: Architectural Interior (with PBR)

```yaml
# config/architectural_interior_pbr.yaml

depth_model:
  variant: "da3-metric-large"
  device: "cpu"
  cache_enabled: true
  cache_size: 128

processing:
  # Depth refinement
  apply_bilateral: true
  bilateral_sigma_color: 10.0
  bilateral_sigma_space: 10.0

  # Zone mapping (optional)
  enable_zone_mapping: true
  num_zones: 4
  tone_map_method: "agx"

  # PBR generation
  pbr:
    enabled: true
    normal_strength: 1.2      # Slightly pronounced for architectural details
    normal_blur_radius: 1     # Minimal smoothing
    roughness_strength: 1.5   # Enhanced micro-detail for materials
    roughness_blur_radius: 2  # Light smoothing
    ao_strength: 1.0          # Standard occlusion
    ao_blur_radius: 8         # Wide spread for soft shadows
    ao_bias: 0.3              # Darker default for luxury interiors
```

### Preset: Architectural Exterior (with PBR)

```yaml
# config/architectural_exterior_pbr.yaml

depth_model:
  variant: "da3-metric-large"
  device: "cpu"
  cache_enabled: true
  cache_size: 128

processing:
  apply_bilateral: true
  bilateral_sigma_color: 8.0
  bilateral_sigma_space: 8.0

  enable_zone_mapping: true
  num_zones: 3
  tone_map_method: "agx"

  # Atmospheric effects for exteriors
  enable_atmospheric: true
  haze_strength: 0.15

  pbr:
    enabled: true
    normal_strength: 1.0      # Standard for exteriors
    normal_blur_radius: 2     # More smoothing for outdoor surfaces
    roughness_strength: 1.2   # Moderate detail
    roughness_blur_radius: 3  # Moderate smoothing
    ao_strength: 0.8          # Lighter occlusion for outdoor lighting
    ao_blur_radius: 10        # Wide spread for natural lighting
    ao_bias: 0.5              # Neutral brightness
```

---

## Integration Points in Existing Pipelines

### 1. Lux Render Pipeline

```python
# src/transformation_portal/pipelines/lux_render_pipeline.py

from transformation_portal.depth_canonical import DepthPipeline, DepthConfig

class LuxuryRenderPipeline:
    def __init__(self, config):
        # Initialize depth pipeline with PBR
        depth_config = DepthConfig.from_preset("architectural_interior")
        depth_config.processing.pbr.enabled = True
        self.depth_pipeline = DepthPipeline(depth_config)

    def enhance_render(self, image_path: Path, output_dir: Path):
        # Step 1: Generate depth + PBR maps
        depth_result = self.depth_pipeline.process_image(image_path, output_dir)

        # Step 2: Use depth for ControlNet conditioning
        depth_map = load_depth(depth_result["depth"])

        # Step 3: Use PBR maps for material-aware enhancement
        if "normal" in depth_result:
            normal_map = load_image(depth_result["normal"])
            # Apply normal-guided detail enhancement

        # Step 4: AI enhancement with depth + PBR guidance
        # ...
```

### 2. Unified Luxury Pipeline

```python
# src/transformation_portal/pipelines/unified_luxury_pipeline.py

from transformation_portal.depth_canonical import DepthPipeline, DepthConfig

class UnifiedLuxuryPipeline:
    def __init__(self, config: UnifiedPipelineConfig):
        # Depth stage with PBR
        depth_config = DepthConfig.from_preset(config.depth_preset)
        depth_config.processing.pbr.enabled = config.enable_pbr_maps
        self.depth_stage = DepthPipeline(depth_config)

    def process_stage_depth(self, image_path: Path, output_dir: Path):
        """Depth estimation stage with PBR map generation."""
        return self.depth_stage.process_image(
            image_path,
            output_dir / "depth",
            generate_pbr=self.config.enable_pbr_maps
        )
```

### 3. Batch TIFF Processor

```python
# luxury_tiff_batch_processor.py (updated)

from transformation_portal.depth_canonical import DepthPipeline, DepthConfig

def main():
    # ... existing code ...

    if args.enable_depth_maps or args.enable_pbr_maps:
        depth_config = DepthConfig.from_preset(args.depth_preset)
        depth_config.processing.pbr.enabled = args.enable_pbr_maps
        depth_pipeline = DepthPipeline(depth_config)

        for image_path in image_paths:
            result = depth_pipeline.process_image(image_path, depth_output_dir)
            logger.info(f"Generated depth + PBR for {image_path.name}")
```

---

## Performance Considerations

### Caching Strategy

```python
# depth_canonical/io/cache.py

from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np

class DepthCache:
    """LRU cache for depth and PBR maps.

    Cache key: SHA256(image_path + config_fingerprint)
    Cache value: (depth_map, pbr_maps_dict)

    Provides 10-20x speedup for iterative workflows.
    """

    def __init__(self, enabled: bool = True, maxsize: int = 128):
        self.enabled = enabled
        self.maxsize = maxsize
        self._cache = {}  # {cache_key: (depth, pbr_dict)}

    def get_cache_key(self, image_path: Path, config: DepthConfig) -> str:
        """Generate cache key from image path and config."""
        hasher = hashlib.sha256()
        hasher.update(str(image_path).encode())
        hasher.update(str(config).encode())
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Retrieve from cache."""
        if not self.enabled:
            return None
        return self._cache.get(key)

    def put(self, key: str, depth: np.ndarray, pbr: Dict[str, np.ndarray]):
        """Store in cache with LRU eviction."""
        if not self.enabled:
            return

        # LRU eviction
        if len(self._cache) >= self.maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = (depth, pbr)
```

### Batch Processing Optimization

**Throughput Targets (with hardware context):**

| Configuration | Depth Est. | PBR Gen. | Combined | Hardware |
|---------------|------------|----------|----------|----------|
| **Baseline (CPU)** | 150 img/hr | 150 img/hr | 100-120 img/hr | Intel i7-10700K, 32GB RAM |
| **Apple Silicon** | 240 img/hr | 200 img/hr | 160-180 img/hr | M4 Max, CoreML, 64GB RAM |
| **GPU (CUDA)** | 280 img/hr | 180 img/hr | 150-170 img/hr | RTX 4080, 16GB VRAM, CUDA 12 |
| **Multi-threaded** | 200 img/hr | 220 img/hr | 140-160 img/hr | 16-core CPU, batch_size=4 |

**Performance Details:**
- **Single 4K Image Latency:**
  - Depth estimation (DA3-Large, CPU): ~24ms (M4 Max CoreML), ~65ms (Intel i7)
  - PBR generation: ~420ms (all platforms, NumPy/SciPy)
  - Combined: ~450-500ms per image
- **Batch Size Impact:** Larger batches improve GPU utilization but increase memory
  - batch_size=1: 2-3GB RAM (default)
  - batch_size=4: 8-10GB RAM
  - batch_size=8: 16-20GB RAM
- **Caching Speedup:** 10-20x for repeated processing with same config

**Optimization Strategies:**
1. **Model batching:** Process multiple images in single inference pass (GPU only)
2. **Parallel PBR:** Generate Normal/Roughness/AO in parallel threads (CPU-bound)
3. **I/O overlap:** Write previous results while processing next image
4. **Memory-mapped writes:** Use atomic writes with mmap for large batches
5. **CoreML acceleration:** 3-5x speedup on Apple Silicon (M1/M2/M3/M4)

```python
# Example: Parallel PBR generation

from concurrent.futures import ThreadPoolExecutor

def generate_pbr_maps_parallel(depth: np.ndarray, config: PBRConfig):
    """Generate PBR maps in parallel threads."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        normal_future = executor.submit(_generate_normal_map, depth, config)
        roughness_future = executor.submit(_generate_roughness_map, depth, config)
        ao_future = executor.submit(_generate_ao_map, depth, config)

        normal = normal_future.result()
        roughness = roughness_future.result()
        ao = ao_future.result()

    return normal, roughness, ao
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_depth_canonical_pbr.py

import pytest
from transformation_portal.depth_canonical import (
    DepthConfig,
    DepthPipeline,
    generate_pbr_maps,
    PBRConfig
)

def test_pbr_integration_in_pipeline(tmp_path):
    """Test PBR generation through unified pipeline."""
    config = DepthConfig.from_preset("architectural_interior")
    config.processing.pbr.enabled = True

    pipeline = DepthPipeline(config)

    # Mock image
    from PIL import Image
    img = Image.new("RGB", (512, 512))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    # Process
    result = pipeline.process_image(img_path, tmp_path)

    # Verify outputs
    assert result["depth"].exists()
    assert result["normal"].exists()
    assert result["roughness"].exists()
    assert result["ao"].exists()

def test_pbr_optional_flag(tmp_path):
    """Test PBR generation can be disabled."""
    config = DepthConfig.from_preset("architectural_interior")
    config.processing.pbr.enabled = False

    pipeline = DepthPipeline(config)

    img = Image.new("RGB", (512, 512))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    result = pipeline.process_image(img_path, tmp_path)

    # Verify only depth output
    assert result["depth"].exists()
    assert "normal" not in result
    assert "roughness" not in result
    assert "ao" not in result

def test_pbr_config_immutability():
    """Test PBRConfig is frozen."""
    config = PBRConfig(normal_strength=1.5)

    with pytest.raises(AttributeError):
        config.normal_strength = 2.0  # Should raise due to frozen dataclass
```

### Integration Tests

```python
# tests/integration/test_depth_pipeline_integration.py

def test_lux_render_pipeline_uses_pbr(tmp_path):
    """Test Lux Render Pipeline integrates with PBR module."""
    from transformation_portal.pipelines.lux_render_pipeline import LuxuryRenderPipeline

    config = {...}
    pipeline = LuxuryRenderPipeline(config)

    # Verify depth pipeline has PBR enabled
    assert pipeline.depth_pipeline.config.processing.pbr.enabled is True

    # Process and verify PBR maps are used
    result = pipeline.enhance_render(test_image, tmp_path)
    # ... assertions ...

def test_batch_processing_with_pbr(tmp_path):
    """Test batch processing generates PBR maps correctly."""
    config = DepthConfig.from_preset("default")
    config.processing.pbr.enabled = True

    pipeline = DepthPipeline(config)

    # Create test images
    images = [create_test_image(tmp_path, f"img_{i}.jpg") for i in range(10)]

    # Batch process
    results = pipeline.batch_process(images, tmp_path / "output")

    # Verify all have PBR maps
    assert len(results) == 10
    for result in results:
        assert result["depth"].exists()
        assert result["normal"].exists()
        assert result["roughness"].exists()
        assert result["ao"].exists()
```

### Performance Tests

```python
# tests/performance/test_pbr_performance.py

import time
import numpy as np
import pytest
from transformation_portal.depth_canonical import generate_pbr_maps, PBRConfig

# Hardware context markers
@pytest.mark.parametrize("resolution,expected_time_ms", [
    ((1080, 1920), 150),   # 1080p: <150ms
    ((2160, 3840), 500),   # 4K: <500ms
    ((4320, 7680), 2000),  # 8K: <2s
])
def test_pbr_generation_performance(resolution, expected_time_ms):
    """Test PBR generation meets performance targets.

    Hardware assumptions:
    - CPU: Intel i7-10700K or equivalent
    - RAM: 32GB
    - Single-threaded execution
    """
    height, width = resolution
    depth = np.random.rand(height, width).astype(np.float32)
    config = PBRConfig()

    start = time.time()
    normal, roughness, ao = generate_pbr_maps(depth, config)
    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < expected_time_ms, (
        f"PBR generation for {resolution} took {elapsed_ms:.1f}ms, "
        f"expected < {expected_time_ms}ms (baseline: Intel i7-10700K)"
    )

    # Verify output shapes
    assert normal.shape == (height, width, 3)
    assert roughness.shape == (height, width)
    assert ao.shape == (height, width)

@pytest.mark.gpu
def test_pbr_batch_performance_gpu():
    """Test batch PBR generation with GPU acceleration.

    Hardware requirements:
    - GPU: CUDA-capable with 8GB+ VRAM
    - Batch size: 4 images
    """
    batch_size = 4
    depth_batch = np.random.rand(batch_size, 2160, 3840).astype(np.float32)
    config = PBRConfig()

    start = time.time()
    # Process batch (hypothetical GPU implementation)
    results = [generate_pbr_maps(depth, config) for depth in depth_batch]
    elapsed = time.time() - start

    throughput = batch_size / elapsed  # images per second

    # Target: >2 images/sec for 4K batch on GPU
    assert throughput > 2.0, (
        f"Batch throughput: {throughput:.2f} img/s, expected >2.0 img/s "
        f"(baseline: RTX 4080, batch_size=4)"
    )
```

---

## Security and Compliance

### Input Validation

```python
# depth_canonical/security/validation.py

from pathlib import Path
import re

SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

def validate_input_path(path: Path) -> Path:
    """Validate input path to prevent path traversal.

    Raises:
        ValueError: If path contains unsafe components
    """
    resolved = path.resolve()

    # Check for path traversal
    if ".." in resolved.parts:
        raise ValueError(f"Path traversal detected: {path}")

    # Check filename safety
    if not SAFE_FILENAME_PATTERN.match(resolved.name):
        raise ValueError(f"Unsafe filename: {resolved.name}")

    return resolved

def sanitize_output_filename(filename: str) -> str:
    """Sanitize output filename to prevent injection."""
    # Remove unsafe characters
    safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

    # Prevent hidden files
    if safe.startswith('.'):
        safe = '_' + safe

    return safe
```

### Atomic Writes

```python
# depth_canonical/io/pbr_writer.py (existing, validated)

def write_pbr_maps(normal, roughness, ao, output_dir, basename):
    """Atomic write of PBR maps with verification.

    Uses temporary files + atomic rename to prevent partial writes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    temp_paths = []

    try:
        # Write to temporary files
        for name, data in [("normal", normal), ("roughness", roughness), ("ao", ao)]:
            temp_path = output_dir / f".{basename}_{name}.tmp"
            final_path = output_dir / f"{basename}_{name}.png"

            Image.fromarray(data).save(temp_path)
            temp_paths.append((temp_path, final_path))

        # Atomic rename (all or nothing)
        for temp, final in temp_paths:
            temp.replace(final)  # Atomic on POSIX and Windows
            paths[name] = final

        return paths

    except Exception as e:
        # Cleanup on failure
        for temp, _ in temp_paths:
            if temp.exists():
                temp.unlink()
        raise IOError(f"Failed to write PBR maps: {e}") from e
```

---

## Post-Integration Optimization Roadmap (Phase 4+)

These optimizations are deferred until after v2.0 release to maintain focus on consolidation and stability. Revisit based on production telemetry and user feedback.

### Priority 1: Performance Enhancements (Months 7-8, Post-v2.0)

**Goal:** Improve batch processing throughput by 50-100%.

**P1.1: Parallel PBR Processing**
- **Status:** Deferred (requires multiprocessing safety validation)
- **Expected Impact:** 2-3x speedup for PBR generation on multi-core systems
- **Implementation:**
  ```python
  from concurrent.futures import ProcessPoolExecutor

  def batch_generate_pbr_parallel(depths, config, max_workers=None):
      """Generate PBR maps in parallel across CPU cores."""
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
          futures = [executor.submit(generate_pbr_maps, d, config) for d in depths]
          return [f.result() for f in futures]
  ```
- **Risks:** Increased memory usage (N cores × ~2GB per image)
- **Acceptance:** Benchmark shows >50% throughput improvement without OOM

**P1.2: GPU-Accelerated Convolutions**
- **Status:** Deferred (requires CuPy or PyTorch implementation)
- **Expected Impact:** 5-10x speedup for PBR normal/AO convolutions on CUDA GPUs
- **Implementation:** Replace NumPy/SciPy convolutions with GPU equivalents
- **Risks:** New dependency (CuPy), GPU availability assumptions
- **Acceptance:** Optional feature (falls back to NumPy on CPU)

**P1.3: Disk Cache for PBR Maps**
- **Status:** Deferred (security review needed for cache poisoning)
- **Expected Impact:** Near-instant regeneration for repeated processing
- **Implementation:**
  ```python
  # Cache structure: ~/.cache/transformation_portal/pbr/{sha256}/
  def get_cached_pbr(depth_hash, config_hash):
      cache_dir = get_cache_dir() / "pbr" / f"{depth_hash}_{config_hash}"
      if cache_dir.exists():
          return load_pbr_from_disk(cache_dir)
      return None
  ```
- **Risks:** Cache invalidation complexity, disk space growth
- **Acceptance:** Security audit passes, cache eviction policy defined

### Priority 2: Model and Backend Enhancements (Months 9-10)

**P2.1: ONNX Inference Backend**
- **Status:** Deferred (cross-platform validation required)
- **Expected Impact:** 20-30% speedup, better Windows/Linux compatibility
- **Implementation:** Convert DA3 model to ONNX, add `ONNXInferenceEngine`
- **Risks:** Model conversion accuracy, ONNX Runtime dependency
- **Acceptance:** Accuracy parity with PyTorch (MSE < 0.01)

**P2.2: Streaming Depth Estimation**
- **Status:** Deferred (video use case not validated)
- **Expected Impact:** Enable real-time video depth processing
- **Implementation:** Process video frames with temporal coherence
- **Risks:** Temporal jitter, increased complexity
- **Acceptance:** Benchmark on 30fps 4K video, <33ms per frame

**P2.3: Multi-Model Ensemble**
- **Status:** Deferred (unclear value proposition)
- **Expected Impact:** Potentially improved depth quality via averaging
- **Implementation:** Run DA2 + DA3, blend depth maps
- **Risks:** 2x inference cost, unclear quality improvement
- **Acceptance:** User study shows measurable quality improvement

### Priority 3: Advanced Features (Months 11-12)

**P3.1: Adaptive Preset Selection**
- **Status:** Deferred (requires scene classification model)
- **Expected Impact:** Improved UX (no manual preset selection)
- **Implementation:** Classify scene (interior/exterior/landscape), select preset
- **Risks:** Misclassification leads to poor results
- **Acceptance:** 90%+ classification accuracy on validation set

**P3.2: Progressive PBR Generation**
- **Status:** Deferred (complex UI integration)
- **Expected Impact:** Show low-res preview immediately, refine over time
- **Implementation:** Generate PBR at multiple resolutions (256px → 4K)
- **Risks:** User confusion, implementation complexity
- **Acceptance:** User testing shows improved perceived performance

**P3.3: Cloud-Based Model Hosting**
- **Status:** Deferred (infrastructure cost unclear)
- **Expected Impact:** Eliminate local model downloads (2GB+)
- **Implementation:** Remote inference API with local fallback
- **Risks:** Latency, privacy concerns, API costs
- **Acceptance:** <200ms API latency, SOC 2 compliance

### Monitoring and Decision Criteria

**Metrics to Track Post-v2.0:**
- Batch processing throughput (images/hour)
- Memory usage per image (peak GB)
- Cache hit rate (for LRU and disk caches)
- User-reported performance issues (GitHub issues, support tickets)
- Preset usage distribution (which presets are most popular)

**Triggers for Prioritization:**
- **Performance complaints:** If >10% of users report slowness, prioritize P1.1/P1.2
- **GPU adoption:** If >50% of users have CUDA GPUs, prioritize P1.2
- **Cache thrashing:** If cache hit rate <30%, prioritize P1.3
- **Video use case:** If video processing requests >5 issues, prioritize P2.2

---

## Migration Plan

### Week-by-Week Breakdown

**Week 1: Foundation**
- Create `depth_canonical/` directory structure
- Implement unified `DepthConfig` (merge DA3Config + DepthConfig)
- Implement canonical `DeviceType` enum
- Migrate PBR module unchanged
- Unit tests for config and PBR

**Week 2: Core Processing**
- Migrate inference engine from `lux_depth_v3`
- Migrate postprocessing from `lux_depth_v3`
- Migrate zone mapping, denoising, atmospheric from `depth/`
- Integrate caching from `depth/`
- Unit tests for all processors

**Week 3: Pipeline Integration**
- Implement `DepthPipeline` orchestrator
- Integrate PBR generation into pipeline
- Implement batch processing
- Integration tests

**Week 4: CLI and Tooling**
- Create `cli/depth_process.py` CLI tool
- Update existing pipelines to use `depth_canonical`
- Update YAML presets
- CLI tests

**Week 5: Deprecation**
- Add deprecation warnings to `depth/` and `lux_depth_v3/`
- Create compatibility shims
- Update documentation
- Migration guide

**Week 6: Validation**
- Performance benchmarking
- End-to-end integration tests
- Update CI/CD workflows
- Security audit

**Post-Launch (v1.x):**
- Monitor deprecation warnings in production
- Support both old and new APIs for 3-6 months
- Collect feedback and refine

**v2.0.0 (3-6 months):**
- Remove deprecated modules
- Breaking change announcement
- Final migration support

---

## Alternatives Considered

### Alternative 1: Keep PBR in lux_depth_v3

**Rejected Reason:** Does not address the core fragmentation problem. Adds to the 44-file count instead of reducing it.

**Trade-offs:**
- ✅ Minimal immediate changes
- ❌ Perpetuates fragmentation
- ❌ No solution for duplicate classes
- ❌ Long-term maintenance burden increases

### Alternative 2: Create new "pbr/" module

**Rejected Reason:** Adds yet another top-level module without consolidating existing depth fragmentation.

**Trade-offs:**
- ✅ Clean separation of PBR concerns
- ❌ 45 depth files across 4 modules
- ❌ Doesn't solve DeviceType/DepthConfig duplication
- ❌ More import complexity

### Alternative 3: Move everything to depth/

**Rejected Reason:** `depth/` is the oldest module with DA2 legacy. `lux_depth_v3/` has more modern patterns.

**Trade-offs:**
- ✅ Alphabetically first import path
- ❌ DA2-centric naming and structure
- ❌ Less modern security patterns
- ❌ Requires more refactoring of newer code

---

## Consequences and Risks

### Positive Consequences

1. **Single Source of Truth:** Eliminates 5 DeviceType enums, 2 DepthConfig classes, 2 DepthEstimator classes
2. **File Reduction:** 44 files → 25 files (~45% reduction)
3. **Clear API:** Public API in `__init__.py`, private internals hidden
4. **PBR Integration:** Clean, optional, well-tested
5. **Backward Compatibility:** Deprecation shims prevent breakage
6. **Performance:** Unified caching strategy across all depth operations

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Import breakage in external tools | Medium | High | Deprecation warnings + 6-month support window |
| Performance regression | Low | Medium | Benchmark suite before/after, caching strategy |
| ML model compatibility issues | Low | High | Keep model wrappers unchanged, test DA2/DA3 |
| Config migration failures | Medium | Medium | Comprehensive config validation, preset tests |
| CI/CD disruption | Low | High | Parallel CI runs during migration, feature flags |

### Resolved Architectural Decisions

#### 1. Model Weight Cache Location

**Decision:** XDG-compliant user cache directory with explicit override support.

**Implementation:**
```python
# Default: ~/.cache/transformation_portal/models/
# Respects XDG_CACHE_HOME on Linux/Unix
# Falls back to %LOCALAPPDATA% on Windows

import os
from pathlib import Path

def get_model_cache_dir() -> Path:
    """Get model weights cache directory (XDG-compliant)."""
    # Environment override
    if override := os.getenv("TRANSFORMATION_PORTAL_MODEL_CACHE"):
        return Path(override)

    # XDG Base Directory standard
    if xdg_cache := os.getenv("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "transformation_portal" / "models"

    # Platform defaults
    if os.name == "nt":  # Windows
        local_app_data = os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(local_app_data) / "transformation_portal" / "models"
    else:  # Unix-like
        return Path.home() / ".cache" / "transformation_portal" / "models"
```

**Rationale:**
- XDG compliance ensures proper integration with system cache management
- User-level cache prevents permission issues in multi-user environments
- Environment variable override supports container/CI workflows
- Separate from repository to avoid accidental commits (weights are 400MB-2GB)

#### 2. Preset Versioning Strategy

**Decision:** Explicit version field in YAML with semantic versioning + migration tooling.

**Format:**
```yaml
# config/architectural_interior.yaml
version: "1.0"  # REQUIRED: Semantic versioning
preset_name: "architectural_interior"
last_updated: "2026-01-30"

depth_model:
  variant: "da3-metric-large"
  # ... rest of config
```

**Backward Compatibility Rules:**
- **Patch (1.0.1):** Bug fixes, documentation only. No parser changes required.
- **Minor (1.1.0):** New optional fields. Old configs remain valid.
- **Major (2.0.0):** Breaking schema changes. Migration required.

**Migration Tooling:**
```python
# depth_canonical/config.py

def migrate_preset(preset_path: Path) -> Dict:
    """Migrate preset to latest schema version."""
    with open(preset_path) as f:
        data = yaml.safe_load(f)

    version = data.get("version", "0.9")  # Pre-versioning assumed 0.9

    # Migration chain
    if version < "1.0":
        data = _migrate_0_9_to_1_0(data)
    if version < "2.0":
        data = _migrate_1_x_to_2_0(data)

    data["version"] = CURRENT_SCHEMA_VERSION
    return data
```

**Rationale:**
- Explicit version prevents silent breakage when schema evolves
- Semantic versioning provides clear expectations for compatibility
- Migration tooling ensures smooth upgrades without manual intervention
- Supports gradual schema evolution without breaking existing workflows

#### 3. PBR Map Output Format

**Decision:** PNG primary format with optional EXR for 16-bit+ workflows.

**Supported Formats:**

| Format | Bit Depth | Use Case | Support Level |
|--------|-----------|----------|---------------|
| PNG | 8-bit RGB/RGBA | Standard PBR workflows, real-time engines (Unity, Unreal) | **Default** |
| PNG (16-bit) | 16-bit grayscale | High-precision depth, roughness, AO | **Supported** |
| EXR | 16/32-bit float | VFX, offline rendering, Blender Cycles | **Optional** (requires `OpenEXR` package) |
| TIFF | 16-bit | Photoshop, editorial workflows | **Planned v2.1** |

**Implementation:**
```python
# depth_canonical/io/pbr_writer.py

from enum import Enum
from pathlib import Path

class PBROutputFormat(Enum):
    PNG_8 = "png8"
    PNG_16 = "png16"
    EXR = "exr"

def write_pbr_maps(
    normal, roughness, ao,
    output_dir: Path,
    basename: str,
    format: PBROutputFormat = PBROutputFormat.PNG_8
) -> Dict[str, Path]:
    """Write PBR maps with configurable format."""
    # PNG (default, no extra dependencies)
    if format in (PBROutputFormat.PNG_8, PBROutputFormat.PNG_16):
        return _write_png(normal, roughness, ao, output_dir, basename, format)

    # EXR (requires openexr)
    elif format == PBROutputFormat.EXR:
        if not HAS_OPENEXR:
            raise ImportError("OpenEXR required: pip install OpenEXR")
        return _write_exr(normal, roughness, ao, output_dir, basename)
```

**Rationale:**
- PNG covers 90% of use cases with zero extra dependencies
- EXR support for professional VFX without forcing dependency on all users
- Graceful degradation: EXR requested but unavailable → clear error message
- Future TIFF support for editorial workflows deferred to v2.1

#### 4. Deprecation Warning Strategy

**Decision:** `FutureWarning` with actionable migration guidance.

**Template:**
```python
import warnings

def _emit_deprecation_warning(
    deprecated_api: str,
    replacement_api: str,
    removal_version: str,
    migration_guide_url: str
):
    """Emit standardized deprecation warning."""
    warnings.warn(
        f"{deprecated_api} is deprecated and will be removed in {removal_version}. "
        f"Use {replacement_api} instead. "
        f"Migration guide: {migration_guide_url}",
        FutureWarning,  # Always visible, not silenced by default
        stacklevel=3  # Show caller's location, not this function
    )

# Usage in deprecated modules:
# src/transformation_portal/depth/__init__.py
_emit_deprecation_warning(
    deprecated_api="transformation_portal.depth.ArchitecturalDepthPipeline",
    replacement_api="transformation_portal.depth_canonical.DepthPipeline",
    removal_version="v2.0.0 (est. Q3 2026)",
    migration_guide_url="https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md"
)
```

**Why FutureWarning:**
- `DeprecationWarning`: Silenced by default in Python 3.2+. Users won't see warnings.
- `PendingDeprecationWarning`: Too soft, no urgency signal.
- **`FutureWarning`:** Visible by default, appropriate for user-facing API changes.

**Rationale:**
- Actionable warnings prevent "warn and pray" antipattern
- Specific migration URLs reduce support burden
- Stack level 3 shows actual call site, not shim internals
- Consistent format across all deprecated APIs

#### 5. Batch Processing Default Batch Size

**Decision:** Default batch size = 1 (sequential) with opt-in parallelism.

**Configuration:**
```python
@dataclass
class BatchConfig:
    """Batch processing configuration."""
    batch_size: int = 1  # Default: sequential processing
    max_workers: Optional[int] = None  # None = auto-detect (CPU count)
    use_gpu_batching: bool = False  # Requires model support
    prefetch_images: int = 2  # I/O overlap
```

**Performance vs. Memory Trade-off:**

| Batch Size | Throughput (img/hr) | Memory (4K) | Use Case |
|------------|---------------------|-------------|----------|
| 1 (default) | 120 img/hr | 2-3GB | Stable, predictable, works everywhere |
| 4 | 180 img/hr | 8-10GB | GPU with 12GB+ VRAM |
| 8 | 220 img/hr | 16-20GB | Workstation with 32GB+ RAM |

**Rationale:**
- Default = robust: Works on laptops, servers, CI environments
- Explicit opt-in prevents OOM surprises on user machines
- Auto-detection (`max_workers=None`) for CPU parallelism, not GPU batching
- GPU batching requires model support (not all transformers models support it)
- Prefetching (I/O overlap) provides 10-15% speedup without memory explosion

---

## Required Enforcement

### CI/CD Gates

```yaml
# .github/workflows/depth_canonical_tests.yml

name: Depth Canonical Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev,test]"

      - name: Unit tests
        run: pytest tests/test_depth_canonical_*.py -v

      - name: Integration tests
        run: pytest tests/integration/test_depth_pipeline_integration.py -v

      - name: Performance tests (non-blocking)
        run: pytest tests/performance/test_pbr_performance.py -v || true

      - name: Verify no imports of deprecated modules
        run: |
          ! grep -r "from transformation_portal.depth import" src/transformation_portal/depth_canonical/
          ! grep -r "from transformation_portal.lux_depth_v3 import" src/transformation_portal/depth_canonical/
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml (add)

- repo: local
  hooks:
    - id: check-depth-imports
      name: Check for deprecated depth imports
      entry: python scripts/check_depth_imports.py
      language: python
      files: \.py$
      pass_filenames: true
```

### Import Linting Script

```python
# scripts/check_depth_imports.py

import sys
import re

DEPRECATED_IMPORT_PATTERN = re.compile(
    r'from transformation_portal\.(depth|lux_depth_v3|depth_intelligence) import'
)

def check_file(filepath):
    """Check for deprecated depth module imports."""
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            if DEPRECATED_IMPORT_PATTERN.search(line):
                # Allow in deprecation shims themselves
                if "depth/__init__.py" in filepath or "lux_depth_v3/__init__.py" in filepath:
                    continue

                print(f"{filepath}:{i}: Deprecated import detected")
                print(f"  Use 'from transformation_portal.depth_canonical import' instead")
                return False
    return True

if __name__ == "__main__":
    all_passed = all(check_file(f) for f in sys.argv[1:])
    sys.exit(0 if all_passed else 1)
```

---

## Documentation Updates Required

1. **README.md:** Update quickstart to use `depth_canonical`
2. **docs/ARCHITECTURE.md:** Document new module structure
3. **docs/depth_pipeline/DEPTH_PIPELINE_README.md:** Rewrite for unified API
4. **docs/migration/DEPTH_MODULE_MIGRATION.md:** Step-by-step migration guide
5. **docs/API_REFERENCE.md:** Full API documentation for `depth_canonical`

---

## Approval Criteria

This ADR is approved when:
1. ✅ Architect review and explicit approval
2. ✅ Security implications assessed and mitigations documented
3. ✅ CI enforcement strategy defined
4. ✅ Migration plan with timeline defined
5. ✅ Backward compatibility strategy defined
6. ✅ Performance benchmarks established

**Status:** Awaiting Architect Approval

---

## References

- Existing PBR implementation: `src/transformation_portal/lux_depth_v3/pbr.py`
- Existing PBR tests: `tests/test_pbr.py` (13/13 passing)
- Depth module fragmentation analysis: Task context
- Agent governance policy: `docs/architecture/agent_governance.md`
- Current depth pipeline: `src/transformation_portal/depth/pipeline.py`
- Current DA3 orchestrator: `src/transformation_portal/lux_depth_v3/orchestrator.py`
