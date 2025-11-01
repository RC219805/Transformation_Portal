#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_predict_coreml.py — run DepthAnything V2 Small F16 (Core ML) on all images in a folder
and write results to the depth output folder, matching the pattern:
    *_depth16.png   (16-bit normalized depth)
    *_depth8_vis.png  (8-bit visualization)
"""

import os, glob, numpy as np
from PIL import Image
import coremltools as ct

# --------------------------------------------------------------------------
# CONFIG
MODEL_PATH = "/Users/rc/Desktop/my_project/DepthAnythingV2SmallF16.mlpackage"
IN_DIR     = "/Users/rc/Desktop/my_project/images/750_Picacho"
OUT_DIR    = "/Users/rc/Desktop/my_project/outputs/depth/750_Picacho"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Load model (auto-select CPU / GPU / ANE)
model = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.ALL)
print("Loaded model:", os.path.basename(MODEL_PATH))

# Gather input files
exts = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp",
        ".TIF", ".TIFF", ".JPG", ".JPEG", ".PNG", ".WEBP")
paths = sorted([p for p in glob.glob(os.path.join(IN_DIR, "*")) if p.endswith(exts)])
if not paths:
    raise SystemExit(f"No input images found in {IN_DIR}")

# --------------------------------------------------------------------------
for i, src in enumerate(paths, 1):
    base = os.path.splitext(os.path.basename(src))[0]
    print(f"[{i}/{len(paths)}] {base}")

    # 1️⃣ Load + preprocess
    img = Image.open(src).convert("RGB").resize((518, 518))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW order
    inputs = {"image": arr[np.newaxis, ...]}  # add batch dim if needed

    # 2️⃣ Predict
    result = model.predict(inputs)
    out_key = list(result.keys())[0]
    depth = np.asarray(result[out_key][0], dtype=np.float32)

    # 3️⃣ Normalize 0..1
    depth = (depth - depth.min()) / (depth.ptp() + 1e-8)

    # 4️⃣ Save 16-bit + 8-bit files
    d16 = (depth * 65535.0 + 0.5).astype(np.uint16)
    d8  = (depth * 255.0 + 0.5).astype(np.uint8)

    out16 = os.path.join(OUT_DIR, f"{base}_depth16.png")
    out8  = os.path.join(OUT_DIR, f"{base}_depth8_vis.png")

    Image.fromarray(d16, mode="I;16").save(out16)
    Image.fromarray(d8,  mode="L").save(out8)

    print("   ✓ saved:", os.path.basename(out16), "and", os.path.basename(out8))

print("\nAll done →", OUT_DIR)