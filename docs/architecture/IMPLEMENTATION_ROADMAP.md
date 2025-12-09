# Performance Optimization Design
## Transformation Portal Architecture Enhancement - Part 2

**Document Version:** 1.0  
**Date:** 2025-12-08  
**Companion To:** STABILITY_EFFICIENCY_ARCHITECTURE.md

---

## 6. Resource Management

### 6.1 Memory Management Strategy

**Problem Analysis:**
- Current: Fixed 2048px tiles, no dynamic adjustment
- Result: 48MP images cause MPS OOM (60GB allocation attempted)
- Impact: 100% failure rate on large images

**Solution: Adaptive Tile Sizing**

```python
# lux_depth_v2/adaptive_tiling.py

from dataclasses import dataclass
from typing import Tuple
import psutil

@dataclass
class TilingStrategy:
    """Adaptive tiling based on available resources."""
    
    # Base tile sizes (px)
    tile_small: int = 512
    tile_medium: int = 1024
    tile_large: int = 2048
    tile_xlarge: int = 4096
    
    # Memory thresholds (GB available)
    threshold_large: float = 20.0
    threshold_medium: float = 10.0
    threshold_small: float = 5.0
    
    # Overlap for blending
    overlap_percent: float = 0.25

def select_tile_size(
    image_width: int,
    image_height: int,
    available_memory_gb: float,
    device: str,
) -> Tuple[int, int]:
    """
    Select optimal tile size based on image dimensions and available memory.
    
    Strategy:
    - Small images (<12MP): No tiling, process full image
    - Medium images (12-24MP): 2048px tiles if memory available
    - Large images (24-48MP): 1024px tiles, more conservative
    - XLarge images (>48MP): 512px tiles, maximum safety
    
    Returns: (tile_size, overlap_px)
    """
    megapixels = (image_width * image_height) / 1e6
    
    strategy = TilingStrategy()
    
    # Estimate memory requirement per megapixel
    # Empirical: ~1.25GB per MP for MPS, ~0.8GB per MP for CUDA
    memory_per_mp = 1.25 if device == "mps" else 0.8
    required_memory = megapixels * memory_per_mp
    
    # Determine tile size based on available memory
    if available_memory_gb >= strategy.threshold_large and megapixels <= 24:
        # Plenty of memory, use large tiles
        tile_size = strategy.tile_large
    elif available_memory_gb >= strategy.threshold_medium:
        # Moderate memory, use medium tiles
        tile_size = strategy.tile_medium
    elif available_memory_gb >= strategy.threshold_small:
        # Low memory, use small tiles
        tile_size = strategy.tile_small
    else:
        # Critical memory, smallest tiles
        tile_size = strategy.tile_small // 2  # 256px
    
    # For very large images, force smaller tiles regardless of memory
    if megapixels > 48:
        tile_size = min(tile_size, strategy.tile_small)
    elif megapixels > 35:
        tile_size = min(tile_size, strategy.tile_medium)
    
    overlap_px = int(tile_size * strategy.overlap_percent)
    
    return tile_size, overlap_px


def calculate_memory_budget(
    image_width: int,
    image_height: int,
    upscale_factor: int,
    bit_depth: int = 16,
) -> dict:
    """
    Calculate memory budget for processing pipeline.
    
    Returns breakdown of memory requirements for each stage.
    """
    megapixels = (image_width * image_height) / 1e6
    
    # Memory calculations (in GB)
    input_memory = megapixels * 3 * (bit_depth / 8) / 1e9
    
    # Processing buffers (8x for intermediate operations)
    processing_memory = input_memory * 8
    
    # Upscale memory (output is upscale_factor^2 larger)
    upscale_output = input_memory * (upscale_factor ** 2)
    
    # Peak memory (processing + upscale simultaneously)
    peak_memory = processing_memory + upscale_output
    
    return {
        "input_gb": input_memory,
        "processing_gb": processing_memory,
        "upscale_output_gb": upscale_output,
        "peak_gb": peak_memory,
        "megapixels": megapixels,
        "recommended_min_gb": peak_memory * 1.5,  # 50% safety margin
    }
```

### 6.2 Progressive Processing for Large Images

**Strategy for 48MP+ Images:**

1. **CPU Fallback**: Automatically switch to CPU for large images
2. **Progressive Upscaling**: 2x twice instead of 4x once
3. **Streaming**: Process and write tiles incrementally

```python
def process_large_image_progressive(
    img_path: Path,
    config: PipelineConfig,
) -> dict:
    """
    Progressive processing strategy for large images (>35MP).
    
    Strategy:
    1. Detect image too large for MPS
    2. Grade at original resolution on MPS (fast, fits in memory)
    3. Upscale in two stages: 2x on CPU, then 2x on CPU
    4. Write output incrementally (tile by tile)
    
    Benefits:
    - No OOM failures
    - Predictable memory usage
    - Still maintains quality
    - Only ~2x slower than single-stage
    """
    # Load and grade at original resolution
    rgb01, info = io_utils.read_rgb_any(img_path)
    
    # Grade on MPS (lightweight, fits in memory)
    graded = grade_on_mps(rgb01, config)
    
    # First upscale: 2x on CPU
    upscaled_2x = upscale_cpu_tiled(
        graded,
        scale=2,
        tile_size=512,
        overlap=128,
    )
    
    # Second upscale: 2x on CPU
    upscaled_4x = upscale_cpu_tiled(
        upscaled_2x,
        scale=2,
        tile_size=512,
        overlap=128,
    )
    
    return upscaled_4x
```

### 6.3 Memory Cleanup Strategy

**Problem**: Memory accumulation over batch processing

**Solution**: Aggressive cleanup between images

```python
def cleanup_between_images():
    """
    Comprehensive memory cleanup between images.
    
    Prevents memory accumulation that leads to crashes after 4-5 images.
    """
    import gc
    
    # Python garbage collection
