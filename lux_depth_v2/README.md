# Lux Depth Pipeline V2 (Production-Oriented)

This is a modular, GPU-accelerated rewrite of the V1 “Gold Standard Depth-Aware 16-bit Lux” pipeline, designed for:
- **Real-time / service mode** (persistent models, low latency)
- **GPU-accelerated post** (Torch-based grading + clarity/sharpen/detail transfer)
- **Advanced automatic material segmentation** (pluggable backends: ONNX / SegFormer / Heuristic fallback)
- **Safe AI detail transfer** (color/luma drift guardrails)

## ⚠️ Security Notice

**When integrating within Transformation Portal, use `requirements-repo.txt` instead of `requirements.txt`**

This avoids CVE-2024-27763 (basicsr command injection vulnerability). See [SECURITY.md](SECURITY.md) for details.

## Quickstart (Batch)

```bash
# Install core deps (within Transformation Portal repository)
pip install -r requirements-repo.txt

# Run on a folder (torch backend is default and secure)
python -m lux_depth_v2.cli \
  --input-dir /data/images \
  --depth-dir /data/depth \
  --output-dir /data/out \
  --preset interior_luxury \
  --device cuda \
  --upscaler-backend torch
```

Outputs:
- `*_master16.tif` (16-bit, graded, pre-upscale)
- `*_upscaled16.tif` (16-bit, final)
- `*_marketing.png` (8-bit, for fast review)
- `*_preview.jpg` (small preview)
- `*_report.json` (per-image report)

## Service Mode (FastAPI)

```bash
pip install fastapi uvicorn[standard]
python -m lux_depth_v2.cli   --output-dir /data/out   --service --host 0.0.0.0 --port 8088
```

Endpoints:
- `GET /health`
- `POST /v2/process` (multipart form: `image` required, `depth` optional)

The service writes outputs into `output_dir` and returns a JSON report containing paths.

## Material Segmentation Backends

### 1) ONNX (Recommended for production)
Set:
- `--seg-backend onnx`
- `--seg-onnx-model /models/material_seg.onnx`
- optionally `--seg-onnx-labels /models/material_labels.json`

**Expected I/O**
- Input: `1x3xHxW` float32 **RGB** in `[0,1]`
- Output: either:
  - `1xCxHxW` logits/probabilities (softmax applied in code), OR
  - `1xHxW` class ids

Label mapping JSON can be either:
```json
{"0":"wood","1":"metal","2":"glass","3":"stone"}
```
or
```json
{"wood":0,"metal":1,"glass":2,"stone":3}
```

### 2) SegFormer (Practical “advanced” proxy)
Set:
- `--seg-backend segformer`
- `--seg-segformer-model <local_dir_or_hf_model_id>`
- if you want automatic downloads: `--seg-allow-downloads`

This uses a SegFormer ADE20K scene parser and maps semantic labels to material buckets (glass/wood/metal/stone/sky/foliage).
It’s not true material segmentation, but is often surprisingly effective for real-estate.

### 3) Heuristic (fallback)
Set:
- `--seg-backend heuristic`

No dependencies, fast, but least accurate.

## Notes / Production Guidance

- For maximum stability: keep `validate_ai=True` (default). AI detail injection is skipped automatically if the upscaler drifts too far in color/luma.
- If working with extremely large files, enable post tiling:
  - `cfg.post_tile` and `cfg.post_overlap` (or modify config defaults).
- For best results, prefer a consistent depth convention where **near = low, far = high** after normalization.

