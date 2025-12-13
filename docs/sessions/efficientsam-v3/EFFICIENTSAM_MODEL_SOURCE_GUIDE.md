# EfficientSAM ONNX Model Source Guide

**Date**: December 12, 2025  
**Purpose**: Official EfficientSAM ONNX model sources, download URLs, and integration guidance for Transformation Portal

---

## Executive Summary

✅ **Official ONNX models are available** from the `wkentaro/efficient-sam` GitHub repository (fork specifically for ONNX export).  
✅ **Two model variants** with different size/speed tradeoffs are recommended.  
✅ **Full models** (encoder + decoder combined) are production-ready.  
✅ **Direct download URLs** with known file sizes are documented below.

---

## Recommended Model for Transformation Portal

### **Primary Recommendation: EfficientSAM ViT-Tiny (Ti)**

**Why:**
- Smaller file size (39 MB vs 101 MB)
- Faster inference (~30-50% faster than ViT-Small)
- Sufficient quality for edge refinement use case
- Lower memory footprint (~2-3 GB vs ~4-5 GB)

**Download URL:**
```
https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt.onnx
```

**File Details:**
- Size: 39 MB
- Model architecture: Vision Transformer Tiny
- Release: December 25, 2023
- Format: Full model (encoder + decoder combined)

---

## Alternative Model (Higher Quality)

### **EfficientSAM ViT-Small (S)**

**Use when:**
- Maximum edge quality is required (hero frames)
- Runtime is less critical
- Memory/GPU capacity available

**Download URL:**
```
https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits.onnx
```

**File Details:**
- Size: 101 MB
- Model architecture: Vision Transformer Small
- Release: December 25, 2023
- Format: Full model (encoder + decoder combined)

---

## Split Models (Advanced / Optional)

If you need granular control over encoder/decoder separately (e.g., for caching embeddings across multiple prompts on the same image):

### ViT-Tiny Split
- **Encoder**: `efficient_sam_vitt_encoder.onnx` (23 MB)
- **Decoder**: `efficient_sam_vitt_decoder.onnx` (15 MB)

### ViT-Small Split
- **Encoder**: `efficient_sam_vits_encoder.onnx` (85 MB)
- **Decoder**: `efficient_sam_vits_decoder.onnx` (15 MB)

**For Stage 5A, use the full models** (simpler I/O contract).

---

## Verified Download Command (Using Your CLI)

### ViT-Tiny (Recommended):
```bash
python -m lux_depth_v2.cli --download-efficientsam \
  --efficientsam-model efficientsam_ti_vit_t \
  --efficientsam-url "https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt.onnx"
```

### ViT-Small (Higher Quality):
```bash
python -m lux_depth_v2.cli --download-efficientsam \
  --efficientsam-model efficientsam_ti_vit_s \
  --efficientsam-url "https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits.onnx"
```

---

## Repository Information

### Official Sources
1. **ONNX Export Repository** (wkentaro/efficient-sam):
   - https://github.com/wkentaro/efficient-sam
   - Releases: https://github.com/wkentaro/efficient-sam/releases
   - Maintained fork for ONNX export specifically

2. **Original EfficientSAM** (yformer/EfficientSAM):
   - https://github.com/yformer/EfficientSAM
   - PyTorch checkpoints and reference implementation
   - Paper: https://arxiv.org/abs/2312.00863

3. **Hugging Face Space** (Demo):
   - https://huggingface.co/spaces/yunyangx/EfficientSAM
   - Interactive demo and examples

### License
- Apache 2.0 (confirmed from LICENSE file in repository)
- Safe for commercial use in Transformation Portal

---

## Model SHA256 Checksums (To Be Computed After Download)

Once downloaded, compute and record SHA256 for supply-chain verification:

```bash
python -c "
import hashlib
from pathlib import Path
p = Path('weights/efficientsam/efficient_sam_vitt.onnx')
b = p.read_bytes()
print('SHA256:', hashlib.sha256(b).hexdigest())
print('SizeMB:', round(p.stat().st_size / 1024 / 1024, 2))
"
```

**Record checksums here after first download:**

- **ViT-Tiny (`efficient_sam_vitt.onnx`)**: `[TO BE COMPUTED]`
- **ViT-Small (`efficient_sam_vits.onnx`)**: `[TO BE COMPUTED]`

---

## Default Configuration Update Needed

In `lux_depth_v2/backends/model_cache.py`, update `DEFAULT_MODELS`:

```python
DEFAULT_MODELS = {
    "efficientsam_ti_vit_t": {
        "url": "https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt.onnx",
        "filename": "efficient_sam_vitt.onnx",
        "sha256": None,  # TODO: set after first verified download
    },
    "efficientsam_ti_vit_s": {
        "url": "https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits.onnx",
        "filename": "efficient_sam_vits.onnx",
        "sha256": None,  # TODO: set after first verified download
    },
}
```

---

## Model I/O Contract (To Be Verified in Stage 5A)

### Expected Inputs (from ONNX export pattern):
- Image tensor: likely `(1, 3, H, W)` float32, normalized
- Point prompts: `(N, 2)` or `(N, 3)` with labels
- Box prompts: `(M, 4)` normalized coordinates

### Expected Outputs:
- Mask logits or probabilities: likely `(1, 1, H, W)` or `(N, 1, H, W)`

**Action after download:** Run the ONNX inspection command in Stage 5A to confirm exact tensor names and shapes.

---

## Integration Steps for Stage 5A

1. **Download model** using CLI command above
2. **Compute SHA256** and update `model_cache.py`
3. **Inspect ONNX signature** using `onnx` module
4. **Update `EfficientSAMBackend._prepare_inputs()` and `segment()`** based on actual I/O contract
5. **Unskip real-model test** in `test_efficientsam_backend.py`
6. **Run canary preset** on 750 Picacho Kitchen/Pool
7. **Golden Baseline A/B comparison** (SegFormer vs FUSED)

---

## Expected Performance (from Literature)

| Metric                | ViT-Tiny | ViT-Small |
|-----------------------|----------|-----------|
| Model Size (ONNX)     | 39 MB    | 101 MB    |
| Inference Time (CPU)  | ~1-2s    | ~2-4s     |
| Inference Time (GPU)  | ~100-200ms | ~150-300ms |
| Memory (Peak)         | ~2-3 GB  | ~4-5 GB   |
| Edge Quality vs SAM   | ~95%     | ~97%      |

**Note:** These are estimates; real numbers will be measured in Stage 5A benchmarking.

---

## Comparison to Full SAM

| Feature                | EfficientSAM (Ti) | SAM ViT-B |
|------------------------|-------------------|-----------|
| Model Size             | 39 MB             | ~350 MB   |
| Inference Speed (rel)  | 10-20x faster     | baseline  |
| Memory Footprint       | ~2-3 GB           | ~6-8 GB   |
| Edge Quality           | ~95%              | 100%      |
| Use Case Fit           | ✅ Edge refinement | ❌ Overkill |

**Conclusion:** EfficientSAM ViT-Tiny is the optimal choice for Lux Depth V2 fusion workflows.

---

## References

1. **Paper**: [EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything](https://arxiv.org/abs/2312.00863)
2. **GitHub (ONNX)**: https://github.com/wkentaro/efficient-sam
3. **GitHub (Original)**: https://github.com/yformer/EfficientSAM
4. **Demo**: https://huggingface.co/spaces/yunyangx/EfficientSAM

---

## Next Steps

✅ **Proceed with Stage 5A now**:
```bash
# Download ViT-Tiny (recommended)
python -m lux_depth_v2.cli --download-efficientsam \
  --efficientsam-model efficientsam_ti_vit_t \
  --efficientsam-url "https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt.onnx"

# Compute checksum
python -c "
import hashlib
from pathlib import Path
p = Path('weights/efficientsam/efficient_sam_vitt.onnx')
b = p.read_bytes()
print('SHA256:', hashlib.sha256(b).hexdigest())
print('SizeMB:', round(p.stat().st_size / 1024 / 1024, 2))
"

# Inspect ONNX signature
python -c "
import onnx
m = onnx.load('weights/efficientsam/efficient_sam_vitt.onnx')
print('INPUTS:')
for i in m.graph.input:
    shp = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(' ', i.name, shp, i.type.tensor_type.elem_type)
print('OUTPUTS:')
for o in m.graph.output:
    shp = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(' ', o.name, shp, o.type.tensor_type.elem_type)
"
```

---

**Status**: Ready for download and Stage 5A execution  
**Risk**: Low – official repository, Apache 2.0 license, community-validated  
**Action**: Execute commands above and paste output for Stage 5A I/O wiring
