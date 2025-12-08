# Baseline Workflow - Topaz (validation_v1)

This document describes how to generate Topaz baselines for the `validation_v1` dataset on the baseline host (Richards-Mac-Studio).

Paths:
- Inputs: `data/benchmark_datasets/validation_v1/input/*.tif`
- Outputs: `data/benchmark_datasets/validation_v1/baselines/<baseline_id>/`

> You are intentionally creating **non-creative**, reproducible, fidelity-first baselines.

---

## 1) Topaz Photo AI — `topaz_photo`

- App: Topaz Photo AI Pro
- Preset: `Lux_Validation_Photo`
- Output: 16-bit TIFF, ProPhoto (or original), no lossy compression
- Output dir: `.../baselines/topaz_photo/`

Naming convention:
- `0001_living_room_topaz_photo.tif`

---

## 2) Topaz Gigapixel — `topaz_gigapixel`

- App: Topaz Gigapixel Pro
- Preset: `Lux_Validation_Giga4x`
  - 4× upscale
  - Standard mode
  - Artifact suppression ON
- Output dir: `.../baselines/topaz_gigapixel/`

Naming convention:
- `0001_living_room_topaz_gigapixel.tif`

---

## 3) Topaz Video AI — `topaz_video` (optional for v1)

- App: Topaz Video AI Pro
- Preset: `Lux_Validation_Video_SR`
- Output dir: `.../baselines/topaz_video/`

Use only if your `validation_v1` includes a video-derived frame subset.

---

## Post-Generation

After exporting baselines, run:

```bash
python scripts/validation/generate_manifest.py
```

This stamps SHA256 hashes (inputs + baselines) into:
- `data/benchmark_datasets/validation_v1/baselines/manifest.json`
