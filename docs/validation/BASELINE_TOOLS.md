# Baseline Tools - Versioned Reference (validation_v1)

This document records the exact versions, presets, and host details used to generate baselines for `validation_v1`.
All baselines should be created on the same host to ensure reproducibility.

## Host (Baseline Host)

- Device label: Richards-Mac-Studio
- Baseline pack: validation_v1
- OS: record via `sw_vers` during baseline generation
- GPU: Apple Silicon integrated GPU (record exact chip via `system_profiler SPHardwareDataType`)

---

## Topaz Applications

### Topaz Photo AI Pro
- Version: 4.0.4
- Preset: `Lux_Validation_Photo`
- Output: 16-bit TIFF, ProPhoto RGB (or original wide-gamut), no lossy compression
- Role: Single-image enhancement + upscale baseline

### Topaz Gigapixel Pro
- Version: 8.4.4
- Preset: `Lux_Validation_Giga4x`
- Output: 4× scale, Standard mode, artifact suppression ON
- Role: Detail/fidelity-focused upscale baseline

### Topaz Video AI Pro (optional for validation_v1)
- Version: 7.1.5
- Preset: `Lux_Validation_Video_SR`
- Output: 4× scale SR, per-frame TIFFs
- Role: Video SR baseline (reserved for future dataset packs)

---

## Adobe Tools

### Lightroom Classic / Camera Raw (Super Resolution)
- Lightroom Classic: 15.0.1
- Camera Raw: 18.0
- Feature: Super Resolution (4×)
- Output: 16-bit TIFF, ProPhoto RGB (or original wide-gamut), no lossy compression
- Role: Adobe SR fidelity baseline

### Photoshop / Camera Raw (Neutral Enhance)
- Photoshop: 27.1.0
- Camera Raw: 18.0
- Preset: `Lux_Validation_Neutral`
- Output: 16-bit TIFF, ProPhoto RGB, no lossy compression
- Role: Neutral, non-hallucinating enhancement baseline

---

## Change Control (Non-Negotiable)

Any change to tool versions, presets, or export settings requires:

- New dataset/baseline pack (e.g. `validation_v2`)
- New baseline directories under `data/benchmark_datasets/`
- New `manifest.json` with updated versions + SHA256 hashes

Baselines must never be silently overwritten in-place.
