# SAM2 Integration Guide

**Status:** Phase 4A Complete (2026-02-17)
**Backend:** Direct checkpoint loading (not HuggingFace Hub)
**License:** Apache 2.0 (commercial-friendly)
**Video Mode:** ✅ Operational (temporal tracking)

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

### Video Mode (Temporal Tracking)

**NEW in Phase 4A:** Track objects across video frames with temporal consistency.

```python
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
from pathlib import Path

# Initialize backend
backend = SAM2Backend(
    model_size="large",
    checkpoint_path="checkpoints/sam2_hiera_large.pt",
    device="cuda"  # or "mps" for Apple Silicon
)

# Option 1: Video file (MP4/MOV) - requires 'decord' package
# Note: decord may not be available on macOS
seg_input = SegmentationInput(
    image=None,  # Not used for video mode
    gamma=1.0,
    mode="video",
    video_path="scene.mp4",
    prompts={
        "frame_idx": 0,  # Which frame to add prompts to
        "object_id": 1,  # Unique ID for this object
        "points": [[100, 150]],  # Point on object in initial frame
        "labels": [1]  # 1=positive, 0=negative
    }
)

# Option 2: Frame directory (JPEG images) - works everywhere
# Convert video to frames: ffmpeg -i video.mp4 frames/%05d.jpg
seg_input = SegmentationInput(
    image=None,
    gamma=1.0,
    mode="video",
    video_path="/path/to/frames/",  # Directory with 00000.jpg, 00001.jpg, ...
    prompts={
        "frame_idx": 0,
        "object_id": 1,
        "points": [[100, 150]],
        "labels": [1]
    }
)

# Run video segmentation
result = backend.segment(seg_input)

# Result contains masks for ALL frames
print(f"Tracked {len(result.masks)} frames")
print(f"Mask shape: {result.masks.shape}")  # (N, H, W) where N=num_frames
print(f"Temporal IDs: {result.temporal_ids}")  # Same ID for tracked object

# Each frame's mask
for i, mask in enumerate(result.masks):
    coverage = mask.sum() / mask.size * 100
    print(f"Frame {i}: {coverage:.1f}% masked")
```

**Video mode features:**
- Temporal consistency across frames
- Single prompt tracks object through entire video
- Works with MP4/MOV files (if `decord` available) or JPEG frame directories
- Returns masks for all frames in video
- Uses `temporal_ids` to identify same object across time

**Convert video to frames (recommended for portability):**
```bash
# Using ffmpeg
mkdir frames
ffmpeg -i video.mp4 frames/%05d.jpg

# Using OpenCV (Python)
import cv2
from pathlib import Path

cap = cv2.VideoCapture("video.mp4")
frames_dir = Path("frames")
frames_dir.mkdir(exist_ok=True)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
    idx += 1
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

### ✅ Phase 3 Complete (2026-02-17)
- [x] Direct checkpoint loading
- [x] Auto mode (SAM2AutomaticMaskGenerator)
- [x] Prompted mode (SAM2ImagePredictor)
- [x] Device selection (MPS/CUDA/CPU)
- [x] Checkpoint download script
- [x] Basic integration tests
- [x] Updated experimental preset

### ✅ Phase 4A Complete (2026-02-17)
- [x] Video tracking mode (SAM2VideoPredictor)
- [x] Temporal consistency across frames
- [x] Point/bbox prompts for initial frame
- [x] Object tracking with stable IDs
- [x] Contract validation passing

### ⏳ Phase 4B-D (Future)
- [ ] Full test suite with real image fixtures
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
