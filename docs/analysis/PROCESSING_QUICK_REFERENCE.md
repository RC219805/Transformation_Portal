# Transformation Portal - Processing Quick Reference

**Last Updated:** November 5, 2025  
**Version:** 1.0 (Great Room lessons incorporated)

---

## 🎯 Image Type Classification

### Dark Interior (Like Great Room)
**Characteristics:**
- Brightness < 0.4
- Low saturation (< 0.15)
- Significant shadow regions

**Recommended Script:** `conservative_enhance_greatroom_final.py`

**Key Parameters:**
```python
EXPOSURE_LIFT = 0.20-0.30      # Aggressive lift needed
SHADOW_RECOVERY = 20-30        # Target deep shadows
SATURATION_LIFT = 1.08-1.12    # Boost flat rendering
```

### Bright Interior (Like Kitchen)
**Characteristics:**
- Brightness > 0.5
- Moderate saturation
- Well-exposed overall

**Recommended Script:** `conservative_enhance_kitchen.py`

**Key Parameters:**
```python
EXPOSURE_LIFT = 0.0-0.10       # Minimal or none
SATURATION_LIFT = 1.10-1.15    # Primary enhancement
BRIGHTNESS_PRESERVE = True     # Maintain within 0.5%
```

### Exterior/Aerial
**Characteristics:**
- Sky visible
- High dynamic range
- Material variation

**Recommended Script:** `enhance_pool_aerial.py` or custom

**Key Parameters:**
```python
SKY_PROTECTION = True          # Critical!
MATERIAL_RESPONSE = True       # Surface enhancement
HDR_TONE_MAPPING = "agx"       # Filmic response
```

---

## 🛠️ Universal Processing Checklist

### Step 1: Analyze Original
```python
# Always run first!
import tifffile, numpy as np

img = tifffile.imread("input.tif")
print(f"Brightness: {img.mean():.4f}")
print(f"Saturation: {(img.max(axis=2) - img.min(axis=2)).mean():.4f}")

# Check sky if present
sky_pixels = img[img.mean(axis=2) > 0.8]
if len(sky_pixels) > 0:
    sky_rgb = sky_pixels.mean(axis=0)
    b_r_ratio = sky_rgb[2] / sky_rgb[0]
    print(f"Sky B/R: {b_r_ratio:.3f} {'✓' if 0.98 <= b_r_ratio <= 1.02 else '⚠️'}")
```

### Step 2: Choose Strategy
```python
if brightness < 0.4:
    strategy = "EXPOSURE_LIFT"
    focus = "Shadow recovery + global lift"
elif brightness < 0.6:
    strategy = "BALANCED"
    focus = "Material enhancement + moderate lift"
else:
    strategy = "SATURATION_BOOST"
    focus = "Color enhancement + sharpening"
```

### Step 3: Configure Parameters
```python
# Adjust based on image type (see above)
EXPOSURE_LIFT = ?
SHADOW_RECOVERY = ?
SATURATION_LIFT = ?
CLARITY_ZONES = {
    'shadows': 0.06,    # Always gentle
    'midtones': 0.12,   # Primary zone
    'highlights': 0.08  # Moderate
}
```

### Step 4: Process
```bash
python your_chosen_script.py
# Monitor output metrics during processing
```

### Step 5: Validate
```python
# Check metrics:
✓ Brightness: appropriate for scene type
✓ Sky B/R: 0.98-1.02 (if sky present)
✓ Clipping: < 0.5%
✓ Visual: side-by-side comparison
```

---

## ⚠️ Common Pitfalls & Solutions

### Problem: Cyan Sky Artifacts
**Cause:** RGB channel manipulation without sky masking  
**Solution:** Always include sky neutrality protection
```python
# Step 4 in pipeline
sky_mask = (brightness > 200/255)
if sky_mask.sum() > 100:
    target_gray = sky_rgb.mean()
    # Neutralize to gray with smooth mask
```

### Problem: Too Dark After Processing
**Cause:** Overly conservative approach  
**Solution:** "Conservative" ≠ dark, lift appropriately
```python
# Don't be afraid to lift dark images!
EXPOSURE_LIFT = 0.22  # Not 0.05
```

### Problem: Noisy Shadows
**Cause:** Excessive clarity in shadow zones  
**Solution:** Reduce shadow clarity
```python
CLARITY_ZONES['shadows'] = 0.06  # Not 0.15
```

### Problem: Loss of Material Detail
**Cause:** Uniform enhancement without zones  
**Solution:** Zone-based processing
```python
# Different strengths for different luminance zones
shadows: 6%, midtones: 12%, highlights: 8%
```

### Problem: Shifted Hue
**Cause:** RGB multiplication for saturation  
**Solution:** Use HSV transformation
```python
# Convert to HSV, boost S channel, convert back
```

---

## 📊 Target Metrics

### Brightness
- **Dark interior:** Target +15-25% lift
- **Bright interior:** Maintain ±5%
- **Exterior:** Target dynamic range preservation

### Sky Neutrality (if present)
- **B/R ratio:** 0.98-1.02 (neutral)
- **If outside range:** Apply neutralization

### Clipping
- **Target:** < 0.5% of pixels
- **Acceptable:** < 1.0% for high-contrast scenes

### Saturation
- **Dark interior:** +20-40% boost acceptable
- **Bright interior:** +10-15% boost
- **Exterior:** +5-10% boost (material-dependent)

---

## 🎨 Color Adjustments

### Warmth Preservation
```python
# Interior lighting: preserve warm tones
WARMTH_RED = 1.01-1.02    # Slight boost
WARMTH_BLUE = 0.98-0.99   # Slight reduction
```

### Sky Correction (Only if needed!)
```python
# Check first: is it actually a problem?
if sky_br_ratio < 0.98 or sky_br_ratio > 1.02:
    # Apply correction
else:
    # Skip - already neutral!
```

### Material-Specific
```python
Wood:    Warmth +2%, saturation +10%
Metal:   Neutral, clarity +15%
Glass:   Protect highlights, +5% clarity
Fabric:  Texture boost, +8% saturation
Stone:   Contrast +10%, neutral warmth
```

---

## 🔧 Script Templates

### Minimal Enhancement (Preserve Quality)
```python
EXPOSURE_LIFT = 0.05
SATURATION_LIFT = 1.05
CLARITY = 0.08
SHARPNESS = 0.10
SKY_PROTECTION = True
```

### Moderate Enhancement (Balanced)
```python
EXPOSURE_LIFT = 0.15
SATURATION_LIFT = 1.10
CLARITY = 0.12
SHARPNESS = 0.14
SKY_PROTECTION = True
```

### Aggressive Enhancement (Dark Images)
```python
EXPOSURE_LIFT = 0.25
SHADOW_RECOVERY = 30
SATURATION_LIFT = 1.12
CLARITY = 0.15
SHARPNESS = 0.16
SKY_PROTECTION = True
```

---

## 📁 Output Standards

### Always Generate
1. **16-bit TIFF master** (LZW compression)
2. **High-quality JPG preview** (95% quality)
3. **Side-by-side comparison** (validation)

### File Naming
```
{basename}_Final.tiff         → Master
{basename}_Final.jpg          → Preview
{basename}_Comparison.jpg     → Validation
```

### Directory Structure
```
processed_images/
├── Conservative/              → Standard enhancements
├── Aggressive/               → High-impact processing
├── Material_Response/        → Material-specific
└── Comparisons/              → Side-by-side validations
```

---

## 🚀 Quick Commands

### Process Standard Image
```bash
python conservative_enhance_greatroom_final.py
```

### View Results
```bash
open processed_images/Conservative/*_Comparison.jpg
```

### Check Metrics
```bash
# Brightness
python -c "from PIL import Image; import numpy as np; img=np.array(Image.open('output.jpg')); print(f'{img.mean()/255:.4f}')"

# Sky B/R
python -c "from PIL import Image; import numpy as np; img=np.array(Image.open('output.jpg'))/255; sky=img[img.mean(axis=2)>0.8]; print(f'{sky[:,2].mean()/sky[:,0].mean():.3f}' if len(sky)>0 else 'No sky')"
```

### Batch Process
```bash
for img in input_images/*.tif; do
    python process_script.py "$img"
done
```

---

## 💡 Pro Tips

1. **Always analyze first** - Don't assume issues exist
2. **Sky protection is mandatory** - Include even if no visible sky
3. **Zone-based is better** - Different areas need different treatment
4. **HSV for saturation** - Preserves hue better than RGB
5. **16-bit output** - Always, even if input is 8-bit
6. **Compare side-by-side** - Visual validation is critical
7. **Iterate if needed** - First attempt may need refinement
8. **Document parameters** - Know what worked for each image type

---

## 📚 Reference Documents

### Detailed Guides
- **GREATROOM_MASTER_SUMMARY.md** - Complete Great Room journey
- **GREATROOM_FINAL_APPROACH.md** - Technical deep dive
- **KITCHEN_QUICK_START.md** - Kitchen processing guide

### Scripts
- **conservative_enhance_greatroom_final.py** - Dark interiors
- **conservative_enhance_kitchen.py** - Bright interiors
- **enhance_pool_aerial.py** - Exteriors/aerials

### Analysis Tools
- **analyze_image.py** - Pre-processing analysis (create if needed)
- **compare_outputs.py** - Side-by-side generation (create if needed)

---

## ⭐ Golden Rules

1. **Analyze → Choose → Process → Validate**
2. **Match strategy to image characteristics**
3. **Protect sky neutrality always**
4. **Zone-based > uniform enhancement**
5. **Quality preservation > aggressive processing**
6. **16-bit output is standard**
7. **Compare before finalizing**
8. **Document what works**

---

**Status:** ✅ **Production Ready Reference**  
**Based on:** 750 Picacho Great Room comprehensive analysis  
**Confidence:** 95% - Validated approach  

*Keep this document handy for all future processing!*
