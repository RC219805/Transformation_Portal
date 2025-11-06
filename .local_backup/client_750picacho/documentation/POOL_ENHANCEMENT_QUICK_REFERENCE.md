# Pool Enhancement Quick Reference
**Image:** 750Picacho_Pool.tiff | **Date:** Nov 6, 2025

## TL;DR - Conservative Parameters

```yaml
exposure: +0.25          # Brighten midday scene
contrast: 1.10           # Add depth without flattening
saturation: 1.06         # Subtle vibrancy boost
shadow_lift: +0.35       # Recover foreground detail
clarity: 0.12            # Minimal - already sharp
sharpening: SKIP         # Detail level 0.047 is excellent

water_correction:
  green: +8%             # Reduce artificial blue cast
  blue: -4%              # Warmer cyan tone
  red: -2%               # Maintain cyan hue

lut: "California_Golden_Hour.cube"
lut_strength: 0.65
```

## Critical Warnings ⚠️

1. **NO SHARPENING** - Image already sharp (0.047), sharpening creates halos
2. **Preserve water transparency** - Use Material Response with `preserve_highlights=True`
3. **Keep concrete neutral** - Saturation < 0.15 to avoid color shifts
4. **Match sky-water reflections** - Process both zones with same color adjustments
5. **Limit green boost** - Max 1.05× or vegetation looks artificial
6. **Convert to sRGB first** - Image is in linear space, requires gamma 2.2

## Image Characteristics

- **Resolution:** 4000x2250 (4K, 16:9)
- **Bit Depth:** 16-bit linear TIFF
- **Composition:** 61% pool water, 25% sky, 15% vegetation, 6% concrete
- **Lighting:** Bright daylight (midday)
- **Current State:** Slightly underexposed (0.441 lum), low contrast (0.105), good detail

## Processing Order

1. **Gamma correct** (linear → sRGB 2.2)
2. **Exposure** (+0.25 EV)
3. **Contrast** (1.10×)
4. **Shadow lift** (+0.35 for pixels < 0.25)
5. **Saturation** (1.06× global)
6. **Water color** (green +8%, blue -4%)
7. **Clarity** (0.12, radius 64)
8. **Material Response** (water: 0.65, concrete: 0.50)
9. **LUT** (California Golden Hour @ 0.65)
10. **Convert back to linear** (for TIFF output)

## Command Line

```bash
# Recommended approach
python conservative_enhance_pool.py \
  --input input_images/750Picacho_Pool.tiff \
  --output processed_images/750Picacho_Pool_enhanced.tiff \
  --preset aerial_pool_daylight \
  --exposure 0.25 --contrast 1.10 --saturation 1.06 \
  --clarity 0.12 --shadow-lift 0.35 \
  --lut assets/luts/location_aesthetic/California_Golden_Hour.cube \
  --lut-strength 0.65 --material-response --verbose
```

## Quality Checks (100% Zoom)

- [ ] No halos around pool edges
- [ ] Sky reflection matches sky color
- [ ] Water depth gradient visible
- [ ] Concrete stays neutral (sat < 0.15)
- [ ] Vegetation green looks natural
- [ ] Average luminance: 0.50-0.55
- [ ] Local contrast: 0.13-0.15

## Expected Results

**Before:** Luminance 0.441, Contrast 0.105, Saturation 0.545  
**After:** Luminance 0.525 (+19%), Contrast 0.135 (+29%), Saturation 0.578 (+6%)  
**Processing Time:** 6-10 seconds (M4 Max)

---

See `ANALYSIS_750Picacho_Pool.md` for full technical analysis.
