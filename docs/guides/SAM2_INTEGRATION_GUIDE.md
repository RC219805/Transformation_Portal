# SAM2 Integration Guide

**Status:** Phase 3 Complete (2026-02-17)
**Backend:** Direct checkpoint loading (not HuggingFace Hub)
**License:** Apache 2.0 (commercial-friendly)

## Overview

SAM2 (Segment Anything 2) backend provides universal object/material segmentation for luxury real estate and architectural visualization pipelines.

## Installation

### 1. Install SAM2 package

```bash
pip install sam2
```

### 2. Download model checkpoints

```bash
# Download large model (~856 MB, best quality)
python scripts/download_sam2_checkpoint.py --model large

# Or download base model (~200 MB, faster)
python scripts/download_sam2_checkpoint.py --model base
```

Checkpoints are stored in: `checkpoints/sam2_hiera_{base_plus,large}.pt`

## Usage

### Auto Mode (Whole Image Segmentation)

```bash
transformation_portal spatial-ai segment \
  --preset experimental/sam2_segmentation \
  --input scene.tiff \
  --output output/ \
  --mode auto
```

### Prompted Mode (Points/Bounding Boxes)

```python
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
import numpy as np

# Initialize backend
backend = SAM2Backend(
    model_size="large",
    checkpoint_path="checkpoints/sam2_hiera_large.pt",
    device="cuda"  # or "cpu", "mps"
)

# Prepare input image (linear RGB, float32)
image = np.array(...).astype(np.float32)  # Shape: (H, W, 3), range [0, 1]

# Point prompts
seg_input = SegmentationInput(
    image=image,
    gamma=1.0,  # Enforced (linear RGB)
    mode="points",
    prompts={
        "points": [[100, 200], [300, 400]],  # (x, y) coordinates
        "labels": [1, 1]  # 1=foreground, 0=background
    }
)

# Segment
result = backend.segment(seg_input)
print(f"Found {len(result.masks)} masks")
```

## Architecture

### Checkpoint Loading (NOT HuggingFace)

SAM2Backend uses **direct torch.load()** of official checkpoints from Meta AI Research.

- ✅ Uses `build_sam2()` from sam2 package
- ✅ Loads `.pt` files directly
- ❌ Does NOT use HuggingFace Hub
- ❌ Does NOT use `transformers.AutoModel`

### Model Variants

| Variant | Checkpoint | Config | Size | Speed | Quality |
|---------|-----------|--------|------|-------|---------|
| base    | `sam2_hiera_base_plus.pt` | `sam2_hiera_b+.yaml` | ~200 MB | Fast | Good |
| large   | `sam2_hiera_large.pt` | `sam2_hiera_l.yaml` | ~856 MB | Slow | Best |

### Device Selection

- **MPS** (Apple Silicon): Auto-detected, prioritized
- **CUDA** (NVIDIA GPU): Auto-detected
- **CPU**: Fallback, slower

## Integration Status

### ✅ Phase 3 Complete
- [x] Direct checkpoint loading
- [x] Auto mode (SAM2AutomaticMaskGenerator)
- [x] Prompted mode (SAM2ImagePredictor)
- [x] Device selection (MPS/CUDA/CPU)
- [x] Checkpoint download script
- [x] Basic integration tests
- [x] Updated experimental preset

### ⏳ Phase 4 (Future)
- [ ] Video tracking mode (temporal consistency)
- [ ] Full test suite with fixtures
- [ ] Performance benchmarks
- [ ] Material classification integration
- [ ] Promotion to stable preset

## Contract

### Input (SpatialCaptureV1)
- **gamma:** Must be 1.0 (linear RGB enforced)
- **dtype:** float32
- **shape:** (H, W, 3)
- **range:** [0.0, 1.0] (or HDR, clipped internally to sRGB for SAM2)

### Output
- **masks:** List of boolean masks (H, W)
- **scores:** Confidence scores [0, 1]
- **metadata:** Bounding boxes, areas, stability scores

## Troubleshooting

### "SAM2 checkpoint not found"

**Solution:** Download checkpoint:
```bash
python scripts/download_sam2_checkpoint.py --model large
```

### "ImportError: No module named 'sam2'"

**Solution:** Install SAM2 package:
```bash
pip install sam2
```

### Slow inference on CPU

**Solution:** Use GPU (CUDA or MPS):
```python
backend = SAM2Backend(model_size="base", device="cuda")  # or "mps"
```

### Out of memory errors

**Solutions:**
1. Use smaller model: `model_size="base"`
2. Process smaller image crops
3. Enable memory_efficient mode in preset

## References

- SAM2 Paper: https://ai.meta.com/research/publications/segment-anything-2/
- Official Repository: https://github.com/facebookresearch/segment-anything-2
- License: Apache 2.0
- Checkpoint Release: July 2024

## ADR Compliance

- **ADR-027:** Isolated from lux_depth_v3 (no cross-pipeline imports) ✅
- **ADR-032:** Experimental preset allows checkpoint paths (not HF revisions) ✅
- **Contract:** SpatialCaptureV1 with gamma=1.0 enforcement ✅
