# Depth Map Generation Instructions

**Status:** Ready to enable
**Required for:** Depth-aware tone mapping and atmospheric effects in APEX V2
**Impact:** Unlocks 100% of luxury_estate preset features

---

## Current State

**Issue:** Depth maps were not generated during APEX V2 batch processing.

**Evidence:**
- `depth_maps_apex/` directory is empty
- Depth-aware features were skipped (tone mapping, atmospheric effects)
- Only 75% of luxury_estate preset features were applied

**Root Cause:**
Depth generation section in `process_source_tiffs_apex.sh` (lines 112-142) is **commented out**.

---

## Prerequisites

### Check ML Dependencies

Depth estimation requires PyTorch and depth-pro:

```bash
# Check if ML dependencies are installed
pip list | grep -E "torch|depth|transformers"

# If missing, install:
pip install -r requirements/ml.txt
```

**Note:** This adds ~10GB of dependencies (PyTorch, transformers, depth-pro).

### Verify Depth Estimation Script

```bash
# Test depth estimation on one image
python scripts/run_depth_estimation.py \
    --input input_images/source_tiffs/V2_750Picacho_Kitchen.tiff \
    --output test_depth.png \
    --backend depth_pro \
    --device mps  # or 'cuda' for NVIDIA, 'cpu' for CPU-only
```

**Expected output:**
- `test_depth.png` created
- Size: matches input dimensions
- Format: 16-bit PNG (grayscale depth map)

---

## Option 1: Enable in Batch Script (Recommended)

### Edit `process_source_tiffs_apex.sh`

**Uncomment lines 124-137:**

```bash
# BEFORE:
# if [[ "${EXISTING_DEPTH_COUNT}" -lt "${INPUT_COUNT}" ]]; then
#     ...commented depth generation...
# fi

# AFTER:
if [[ "${EXISTING_DEPTH_COUNT}" -lt "${INPUT_COUNT}" ]]; then
    log_info "Generating depth maps (missing or incomplete)..."
    echo

    for input_file in "${INPUT_DIR}"/*.{tif,tiff}; do
        [[ -e "${input_file}" ]] || continue
        filename=$(basename "${input_file%.*}")
        depth_output="${DEPTH_DIR}/${filename}_depth.png"

        if [[ ! -f "${depth_output}" ]]; then
            log_info "Generating depth: ${filename}"
            python scripts/run_depth_estimation.py \
                --input "${input_file}" \
                --output "${depth_output}" \
                --backend depth_pro \
                --device "${DEVICE}" \
                || log_warn "Depth generation failed for ${filename} (will proceed without depth)"
        fi
    done
else
    log_info "Using existing depth maps (${EXISTING_DEPTH_COUNT} found)"
    echo
fi
```

### Run Batch Script

```bash
./process_source_tiffs_apex.sh
```

**Expected behavior:**
1. Checks for existing depth maps
2. Generates missing depth maps
3. Logs progress for each image
4. Continues with enhancement (uses depth maps if available)

---

## Option 2: Generate Depth Maps Manually

### Single Image

```bash
python scripts/run_depth_estimation.py \
    --input input_images/source_tiffs/V2_750Picacho_Kitchen.tiff \
    --output depth_maps_apex/V2_750Picacho_Kitchen_depth.png \
    --backend depth_pro \
    --device mps
```

### Batch Generation (All TIFFs)

```bash
#!/bin/bash
# generate_depth_maps.sh

INPUT_DIR="input_images/source_tiffs"
DEPTH_DIR="depth_maps_apex"
DEVICE="mps"  # or 'cuda', 'cpu'

mkdir -p "${DEPTH_DIR}"

for tiff in "${INPUT_DIR}"/*.tiff; do
    [[ -e "${tiff}" ]] || continue

    stem=$(basename "${tiff}" .tiff)
    depth_out="${DEPTH_DIR}/${stem}_depth.png"

    if [[ -f "${depth_out}" ]]; then
        echo "Skipping ${stem} (depth map exists)"
        continue
    fi

    echo "Generating depth: ${stem}"
    python scripts/run_depth_estimation.py \
        --input "${tiff}" \
        --output "${depth_out}" \
        --backend depth_pro \
        --device "${DEVICE}"
done

echo "Depth map generation complete!"
echo "Generated: $(ls -1 ${DEPTH_DIR}/*.png | wc -l) depth maps"
```

**Run:**
```bash
chmod +x generate_depth_maps.sh
./generate_depth_maps.sh
```

---

## Option 3: Skip Depth (Fallback)

If ML dependencies cannot be installed, enhancement will proceed **without depth awareness**:

```bash
# Enhancement without depth maps
python scripts/enhance_image.py \
    input.tiff \
    --output-dir output/ \
    --preset luxury_estate
    # No --depth-dir flag
```

**Impact:**
- ✅ Basic enhancements still work (clarity, color)
- ❌ Depth-aware tone mapping skipped
- ❌ Atmospheric effects skipped
- ⚠️ Only ~75% of luxury_estate preset features applied

---

## Depth Map Details

### Format

**Output format:** 16-bit PNG (single channel, grayscale)

**Value range:**
- 0 = near (foreground, closest objects)
- 65535 = far (background, farthest objects)
- Normalized internally to [0.0, 1.0] for processing (after p01-p99 clipping)
- Higher normalized depth values represent greater distance

**Normalized internally to [0.0, 1.0] for processing.**

### Storage Requirements

**Per depth map:**
- Size: ~10-20 MB (16-bit PNG, compressed)
- Dimensions: Match input image

**For 6 source TIFFs (6000×3375 each):**
- Total: ~60-120 MB

### Backends

**Available backends:**

1. **`depth_pro`** (Recommended)
   - Accuracy: Excellent
   - Speed: Fast (~2-3s per image on Apple Silicon)
   - Requirements: PyTorch, depth-pro

2. **`depth_anything_v2`** (Alternative)
   - Accuracy: Good
   - Speed: Moderate
   - Requirements: PyTorch, transformers

3. **`midas`** (Legacy)
   - Accuracy: Fair
   - Speed: Slow
   - Requirements: PyTorch, timm

**Select backend:**
```bash
python scripts/run_depth_estimation.py \
    --backend depth_pro  # or depth_anything_v2, midas
    ...
```

---

## Troubleshooting

### Error: "No module named 'depth_pro'"

**Solution:** Install ML dependencies:
```bash
pip install -r requirements/ml.txt
```

### Error: "CUDA out of memory"

**Solution:** Use CPU or reduce batch size:
```bash
python scripts/run_depth_estimation.py \
    --device cpu \
    ...
```

### Error: "MPS backend not available"

**Solution:** Use CPU (macOS without Metal support):
```bash
python scripts/run_depth_estimation.py \
    --device cpu \
    ...
```

### Depth maps look incorrect

**Check:**
1. Input image format (should be RGB, not grayscale)
2. Depth backend (try different backend)
3. File format (should be .png, not .jpg)

**Verify depth map:**
```python
from PIL import Image
import numpy as np

depth = Image.open("depth_maps_apex/image_depth.png")
print(f"Mode: {depth.mode}")  # Should be 'I' or 'I;16'
arr = np.array(depth)
print(f"dtype: {arr.dtype}")  # Should be uint16
print(f"Range: [{arr.min()}, {arr.max()}]")  # Should be [0, 65535]
```

---

## Performance

### Depth Generation Times (Apple M1 Max, MPS)

| Image Size | Backend      | Time   |
|-----------|--------------|--------|
| 6000×3375 | depth_pro    | ~2.5s  |
| 6000×3375 | depth_any_v2 | ~4.0s  |
| 6000×3375 | midas        | ~8.0s  |

### Enhancement Times (With Depth)

| Stage          | Time (no depth) | Time (with depth) |
|----------------|-----------------|-------------------|
| Load           | ~0.1s          | ~0.1s             |
| Depth load     | —              | ~0.01s            |
| Enhancement    | ~0.8s          | ~0.9s             |
| Save (16-bit)  | ~0.1s          | ~0.1s             |
| **Total**      | **~1.0s**      | **~1.1s**         |

**Impact:** Depth awareness adds ~10% to enhancement time.

---

## Verification

### Check Depth Maps Were Used

**Look for in enhancement logs:**
```
Loaded depth map: depth_maps_apex/V2_750Picacho_Kitchen_depth.png
```

**Check JSON report:**
```json
{
  "depth_map": "depth_maps_apex/V2_750Picacho_Kitchen_depth.png",
  "stage_metadata": {
    "has_depth": true  // Should be true
  }
}
```

### Visual Verification

**Open depth map in image viewer:**
- Should show spatial depth (foreground bright, background dark)
- Should align with scene geometry

**Compare enhanced output:**
- With depth: Foreground objects have subtle brightness boost
- Without depth: Flat tone mapping across entire image

---

## Next Steps

1. **Generate depth maps** (using Option 1 or 2 above)
2. **Re-run APEX V2 batch** with depth maps present
3. **Verify depth awareness** in JSON reports
4. **Compare output quality** (with depth vs. without)

---

## References

- **Depth-Pro:** https://github.com/apple/ml-depth-pro
- **Depth Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2
- **MiDaS:** https://github.com/isl-org/MiDaS
- **APEX V2 Architecture:** `docs/architecture/V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`
