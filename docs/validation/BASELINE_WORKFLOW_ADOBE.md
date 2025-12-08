# Baseline Workflow - Adobe (validation_v1)

This document describes how to generate Adobe baselines for the `validation_v1` dataset on the baseline host (Richards-Mac-Studio).

Paths:
- Inputs: `data/benchmark_datasets/validation_v1/input/*.tif`
- Outputs:
  - `data/benchmark_datasets/validation_v1/baselines/adobe_sr/`
  - `data/benchmark_datasets/validation_v1/baselines/adobe_neutral/`

Versions (freeze for Baseline v1):
- Lightroom Classic: 15.0.1
- Camera Raw: 18.0
- Photoshop: 27.1.0

---

## 1) Adobe Super Resolution — `adobe_sr`

Workflow (Lightroom Classic):
1. Import all TIFFs from `.../input/`
2. Right-click → **Enhance…**
3. Enable **Super Resolution**
4. Export as:
   - 16-bit TIFF
   - ProPhoto RGB (or original wide gamut)
   - No lossy compression
5. Write to: `.../baselines/adobe_sr/`

Naming convention:
- `0001_living_room_adobe_sr.tif`

---

## 2) Adobe Neutral Enhance — `adobe_neutral`

Create a Camera Raw preset `Lux_Validation_Neutral`:
- Sharpening: moderate (e.g. 25)
- Clarity: 0
- Texture: 0
- Noise Reduction: 0
- Lens corrections: OFF (baseline) unless explicitly required
- No creative grading

Workflow (Photoshop + ACR):
1. Open TIFFs (or batch) via Camera Raw
2. Apply `Lux_Validation_Neutral`
3. Save as:
   - 16-bit TIFF
   - ProPhoto RGB
   - No lossy compression
4. Write to: `.../baselines/adobe_neutral/`

Naming convention:
- `0001_living_room_adobe_neutral.tif`

---

## Post-Generation

After exporting baselines, run:

```bash
python scripts/validation/generate_manifest.py
```

This stamps SHA256 hashes (inputs + baselines) into:
- `data/benchmark_datasets/validation_v1/baselines/manifest.json`
