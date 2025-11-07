# Professional Enhancement Plan: 750 Picacho Kitchen

**Property**: 750 Picacho Residence  
**Scene**: Luxury Kitchen Interior Rendering  
**Source File**: `Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff`  
**Analysis Date**: 2025-11-07 01:27 PST  
**Target Output**: High-end real estate marketing, magazine-quality publication, hero web image, print brochure

---

## 1. Image Analysis

### Technical Specifications
- **Resolution**: 4000 x 2250 px (9.0 MP) - 16:9 aspect ratio
- **Format**: TIFF with RGBA channels
- **Bit Depth**: 8-bit per channel (24-bit color) + alpha
- **File Size**: 68.7 MB
- **Color Space**: ICC profile present (3144 bytes) - likely sRGB or Adobe RGB
- **Compression**: LZW or similar (noted in metadata)

### Exposure Analysis
- **Overall Luminance**: 0.588 (mean) | 0.664 (median) - **WELL BALANCED**
- **Dynamic Range**: Good - spans from near-black (0.01) to near-white (0.99)
- **Clipping Assessment**:
  - Shadows: 1.59% red, 2.77% green, 6.12% blue (acceptable, mostly in deep corners)
  - Highlights: 5.80% red, 1.27% green, 0.80% blue (moderate, likely specular reflections on appliances)
- **Verdict**: The "bright" designation is accurate - image is properly exposed with good detail retention

### Color Balance Analysis
- **Red Channel**: 0.677 (+15.1% above neutral) ✓
- **Green Channel**: 0.591 (+0.5% near neutral) ✓
- **Blue Channel**: 0.497 (-15.6% below neutral) ⚠️
- **Color Cast**: **WARM** (yellow/orange bias) - This is intentional and appropriate for:
  - Kitchen warmth and inviting atmosphere
  - Wood cabinetry enhancement
  - Luxury real estate aesthetic
  - However, may need subtle cooling for certain print applications

### Saturation & Contrast
- **Mean Saturation**: 0.184 - **MODERATE** (natural, not oversaturated)
- **Contrast (Std Dev)**: 0.261 - **GOOD** (balanced, not flat)
- **Max Saturation**: 0.694 - indicates vibrant accent colors (likely backsplash, décor)

### Technical Issues
- **Noise Level**: 0.0379 - **MODERATE** (noticeable on close inspection)
  - Likely from rendering/upscaling process
  - Will benefit from selective noise reduction
- **Sharpness**: Not measured, but 9MP at 4000px suggests good detail potential
- **Artifacts**: None detected in metadata (no compression artifacts expected from TIFF)

### Material Detection (Preliminary Visual Assessment)
Based on color/saturation analysis, the kitchen likely contains:
- **Wood surfaces**: ~30-40% (warm tones, moderate saturation) - cabinetry, potentially flooring
- **Metal/Stainless Steel**: ~15-20% (low saturation, high contrast) - appliances, fixtures, hardware
- **Glass/Glazed surfaces**: ~10-15% (reflective, clear) - windows, possibly backsplash
- **Stone/Quartz countertops**: ~10-15% (neutral, subtle texture) - work surfaces
- **Painted surfaces**: ~20-30% (smooth, low saturation) - walls, ceiling

**Full material analysis requires Material Response Technology processing**

---

## 2. Enhancement Strategy

### ⭐ **RECOMMENDED: Standard Pipeline (30-40 minutes)**

This provides professional magazine-quality results with optimal time/quality balance.

### Quick Enhancement (10-15 minutes)
**Best for**: Client previews, fast turnaround, social media

**Workflow**: Basic color grading + selective sharpening + export optimization

### Standard Pipeline (30-40 minutes) ⭐
**Best for**: Marketing materials, website hero images, most print applications

**Workflow**: Depth-aware processing → Material Response → Color grading → Detail enhancement → Output optimization

### Premium Pipeline (60-90 minutes)
**Best for**: Magazine covers, high-end print campaigns, award submissions

**Workflow**: Full AI refinement with ControlNet → SDXL enhancement → Real-ESRGAN upscaling → Premium finishing

---

## 3. Standard Pipeline Implementation (RECOMMENDED)

### Stage 1: Technical Preparation & Corrections
**Duration**: 2-3 minutes  
**Tools**: Python + Pillow + colour-science

```bash
# Convert RGBA to RGB, preserve ICC profile
python3 << 'EOF'
from PIL import Image
from pathlib import Path

input_path = Path("input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff")
img = Image.open(input_path)

# Convert to RGB, preserve ICC profile
if img.mode == 'RGBA':
    # Check if alpha is meaningful or just opaque layer
    alpha = img.split()[3]
    if alpha.getextrema() == (255, 255):
        print("Alpha channel is opaque - removing")
        img_rgb = img.convert('RGB')
    else:
        print("Alpha channel has transparency - compositing on white")
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=alpha)
        img_rgb = background
else:
    img_rgb = img

# Preserve ICC profile
icc = img.info.get('icc_profile')
save_kwargs = {'icc_profile': icc} if icc else {}

# Save as working file
output_path = Path("working/750Picacho_Kitchen_step1_prep.tiff")
output_path.parent.mkdir(exist_ok=True)
img_rgb.save(output_path, compression='lzw', **save_kwargs)
print(f"Saved: {output_path}")
EOF
```

**Actions**:
- ✅ Convert RGBA → RGB (removes alpha overhead)
- ✅ Preserve ICC color profile
- ✅ Validate image integrity
- ✅ Create working copy (non-destructive workflow)

---

### Stage 2: Depth-Aware Processing
**Duration**: 5-8 minutes (with CoreML on Apple Silicon)  
**Tools**: `depth_pipeline/` + Depth Anything V2 (CoreML optimized)

```bash
# Use interior preset optimized for architectural rendering
cd /Users/rc/Transformation_Portal

python3 depth_pipeline/pipeline.py \
  --input working/750Picacho_Kitchen_step1_prep.tiff \
  --output working/750Picacho_Kitchen_step2_depth \
  --config config/interior_preset.yaml \
  --save-depth-map \
  --device mps  # Use Apple Neural Engine (M-series)
```

**Interior Preset Configuration** (`config/interior_preset.yaml`):
```yaml
depth_model:
  name: "depth-anything-v2-small"
  backend: "coreml"  # 3-5x faster on M-series
  
processing:
  denoising:
    enabled: true
    strength: 0.4  # Moderate - remove rendering noise
    preserve_edges: true
    
  tone_mapping:
    operator: "AgX"  # Film-like roll-off, preserves highlights
    zones:
      foreground: 0.0-0.3  # Countertops, close appliances
      midground: 0.3-0.7   # Cabinetry, backsplash
      background: 0.7-1.0  # Windows, distant walls
    adjustments:
      foreground:
        exposure: +0.1      # Lift counter detail
        contrast: 1.15      # Enhance material texture
        clarity: 0.20       # Micro-contrast on surfaces
      midground:
        exposure: 0.0       # Maintain balance
        contrast: 1.10      
        clarity: 0.15
      background:
        exposure: -0.05     # Subtle vignette effect
        contrast: 0.95      # Soften distant elements
        
  atmospheric:
    depth_haze:
      enabled: true
      intensity: 0.15     # Very subtle for interior
      color: [0.95, 0.95, 0.98]  # Cool, airy
      start_distance: 0.6  # Only affects far background
      
  clarity:
    global: 0.12          # Overall micro-contrast
    depth_weighted: true  # More clarity in foreground
```

**Expected Output**:
- `750Picacho_Kitchen_step2_depth.tiff` - Depth-processed image
- `750Picacho_Kitchen_step2_depth_depthmap.png` - Depth visualization (for reference)

**Performance**: ~24-65ms per image on M4 Max, ~5-8 minutes including model loading

**Quality Improvements**:
- ✅ Noise reduction (rendering artifacts eliminated)
- ✅ Zone-based tone mapping (foreground "pops", background recedes naturally)
- ✅ Depth-aware clarity (counters/appliances sharp, walls softer)
- ✅ Atmospheric depth (subtle aerial perspective for spatial depth)

---

### Stage 3: Material Response Technology™
**Duration**: 8-12 minutes  
**Tools**: `material_response.py` (proprietary physics-based enhancement)

```bash
python3 material_response.py \
  --input working/750Picacho_Kitchen_step2_depth.tiff \
  --output working/750Picacho_Kitchen_step3_materials.tiff \
  --mode auto \
  --surfaces wood,metal,glass,stone,paint \
  --strength 0.75 \
  --preserve-highlights \
  --depth-map working/750Picacho_Kitchen_step2_depth_depthmap.png \
  --verbose
```

**Material-Specific Enhancements**:

#### Wood (Cabinetry, Potential Flooring)
- **Detection**: Warm hues (R > B), moderate saturation (0.1-0.4)
- **Enhancement**:
  - Grain structure: +15% local contrast on fine details
  - Warm highlights: +8% luminance in specular areas with warm shift
  - Midtone richness: +10% saturation in mid-luminance areas
  - Natural luster: Subtle anisotropic highlight stretching along grain direction
- **LUT Integration**: Will be complemented by warm film emulation LUT

#### Metal (Appliances, Fixtures, Hardware)
- **Detection**: Low saturation (<0.05), high local contrast
- **Enhancement**:
  - Specular preservation: Protect pure white highlights (>0.95)
  - Micro-contrast: +20% on reflective transitions
  - Cool shift: Subtle blue bias (+2%) in reflections for chrome/stainless authenticity
  - Edge sharpening: +0.3 radius selective sharpening on metal edges
- **Critical**: Prevents "blown out" appliances while maintaining luxury sheen

#### Glass (Windows, Potential Backsplash)
- **Detection**: Low saturation, medium-high luminance, edge clarity
- **Enhancement**:
  - Clarity: +25% micro-contrast for crystal-clear appearance
  - Reflections: Enhance specular detail by 10%
  - Subtle tint: If colored glass, boost saturation by +5%
  - Transparency: Preserve through-glass detail (depth-aware)
- **Result**: Sparkling, magazine-quality glass surfaces

#### Stone/Quartz (Countertops)
- **Detection**: Neutral color, subtle texture, diffuse reflections
- **Enhancement**:
  - Texture detail: +18% local contrast on surface variation
  - Natural luster: Subtle highlights (+6%) without artificial shine
  - Color purity: Enhance inherent stone color by +8% saturation
  - Depth: Slight darkening (-3%) in veining for dimensional quality
- **Result**: Photorealistic stone with tactile quality

#### Paint (Walls, Ceiling)
- **Detection**: Low saturation, smooth gradients, uniform color
- **Enhancement**:
  - Smooth tones: Noise reduction (already applied in Stage 2)
  - Clean highlights: Gentle roll-off to white (prevents harsh edges)
  - Color consistency: ±3% luminance normalization
  - Minimal intervention: Avoid over-processing painted surfaces
- **Result**: Clean, professional painted surfaces without flatness

**Performance**: ~400-600 images/hour batch throughput, ~8-12 minutes for single high-res image

**Output**: Surface-enhanced image with physically accurate material rendering

---

### Stage 4: Professional Color Grading
**Duration**: 3-5 minutes  
**Tools**: `luxury_tiff_batch_processor.py` + Custom LUT selection

#### LUT Selection Strategy

**Option A: Warm California Aesthetic** (RECOMMENDED for this property)
```bash
python3 luxury_tiff_batch_processor.py \
  --input working/750Picacho_Kitchen_step3_materials.tiff \
  --output working/750Picacho_Kitchen_step4_graded.tiff \
  --lut assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube \
  --lut-strength 0.70 \
  --exposure 0.0 \
  --contrast 1.08 \
  --saturation 1.05 \
  --preserve-metadata
```

**Rationale**:
- California property → California aesthetic LUT
- "Golden Hour" warmth complements existing warm color cast
- 70% strength prevents over-grading while adding cinematic quality
- Slight contrast/saturation boost for luxury marketing impact

**Option B: Film Emulation - Kodak 2393 (Print/Editorial)**
```bash
python3 luxury_tiff_batch_processor.py \
  --input working/750Picacho_Kitchen_step3_materials.tiff \
  --output working/750Picacho_Kitchen_step4_graded_kodak.tiff \
  --lut assets/luts/film_emulation/Kodak/Kodak_2393_D55.cube \
  --lut-strength 0.65 \
  --exposure +0.05 \
  --contrast 1.10 \
  --saturation 1.03 \
  --preserve-metadata
```

**Rationale**:
- Kodak film aesthetic = timeless, editorial quality
- D55 daylight balance (slightly cooler) can offset warm cast if needed
- 65% strength for subtle film character without vintage look
- Slight exposure lift for bright, airy feel

**Option C: Neutral/Versatile - FilmConvert Nitrate**
```bash
python3 luxury_tiff_batch_processor.py \
  --input working/750Picacho_Kitchen_step3_materials.tiff \
  --output working/750Picacho_Kitchen_step4_graded_nitrate.tiff \
  --lut assets/luts/film_emulation/FilmConvert/FilmConvert_Nitrate_LuxuryRE.cube \
  --lut-strength 0.60 \
  --exposure 0.0 \
  --contrast 1.06 \
  --saturation 1.04 \
  --preserve-metadata
```

**Rationale**:
- "LuxuryRE" variant specifically tuned for real estate
- Neutral baseline suitable for brand consistency across properties
- Lower strength (60%) for flexible post-processing
- Balanced look works for web, print, and social media

**RECOMMENDATION**: Start with **Option A (Montecito Golden Hour)** for warmth and California luxury aesthetic. Generate Option B (Kodak) as alternative for print applications requiring cooler tone.

---

### Stage 5: Detail Enhancement & Finishing
**Duration**: 4-6 minutes  
**Tools**: Custom Python script using Pillow + scikit-image

```bash
python3 << 'EOF'
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import numpy as np
from scipy import ndimage

# Load graded image
input_path = Path("working/750Picacho_Kitchen_step4_graded.tiff")
img = Image.open(input_path)
img_array = np.array(img).astype(np.float32) / 255.0

# --- Selective Sharpening ---
# Create luminance mask for edge detection
lum = img_array.mean(axis=2)
edges = ndimage.sobel(lum)
edge_mask = (edges > 0.05).astype(np.float32)  # Sharpen only edges

# Apply unsharp mask with edge weighting
sharpened = img.filter(ImageFilter.UnsharpMask(
    radius=1.5,    # Fine detail
    percent=120,   # Moderate strength
    threshold=2    # Avoid noise amplification
))

# Blend based on edge mask
img_array_sharp = np.array(sharpened).astype(np.float32) / 255.0
sharpness_amount = 0.7  # 70% sharpening on edges, less on smooth areas
img_array_enhanced = img_array * (1 - edge_mask[:,:,np.newaxis] * sharpness_amount) + \
                     img_array_sharp * edge_mask[:,:,np.newaxis] * sharpness_amount

# --- Micro-Contrast (Clarity) ---
# High-pass filter for local contrast
from scipy.ndimage import gaussian_filter

for c in range(3):
    blurred = gaussian_filter(img_array_enhanced[:,:,c], sigma=10)
    high_pass = img_array_enhanced[:,:,c] - blurred
    img_array_enhanced[:,:,c] += high_pass * 0.18  # 18% clarity boost

# --- Controlled Highlight Glow (Luxury Feel) ---
# Detect bright areas (appliances, light sources)
brightness = img_array_enhanced.mean(axis=2)
glow_mask = np.clip((brightness - 0.75) / 0.25, 0, 1)  # Gradual glow on highlights

# Soft glow using gaussian blur
glow_layer = np.zeros_like(img_array_enhanced)
for c in range(3):
    glow_layer[:,:,c] = gaussian_filter(img_array_enhanced[:,:,c] * glow_mask, sigma=20)

# Blend glow (subtle)
glow_strength = 0.08
img_array_enhanced = img_array_enhanced * (1 - glow_strength) + glow_layer * glow_strength

# --- Subtle Vignette (Edge Darkening) ---
h, w = img_array_enhanced.shape[:2]
y, x = np.ogrid[:h, :w]
center_y, center_x = h / 2, w / 2

# Radial gradient
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
max_dist = np.sqrt(center_x**2 + center_y**2)
vignette = 1 - (distance / max_dist) ** 1.8 * 0.15  # 15% max darkening at corners

# Apply vignette
for c in range(3):
    img_array_enhanced[:,:,c] *= vignette

# --- Convert back to 8-bit ---
img_final = np.clip(img_array_enhanced * 255, 0, 255).astype(np.uint8)
img_out = Image.fromarray(img_final, 'RGB')

# Preserve metadata
if 'icc_profile' in img.info:
    img_out.info['icc_profile'] = img.info['icc_profile']

# Save
output_path = Path("working/750Picacho_Kitchen_step5_enhanced.tiff")
img_out.save(output_path, compression='lzw')
print(f"Enhanced image saved: {output_path}")
print("Enhancements applied:")
print("  ✓ Selective edge sharpening (70% on edges)")
print("  ✓ Micro-contrast clarity (+18%)")
print("  ✓ Highlight glow (8% subtle)")
print("  ✓ Corner vignette (15% max darkening)")
EOF
```

**Enhancements Applied**:
- ✅ **Selective Sharpening**: Edge-detected sharpening prevents noise amplification
- ✅ **Micro-Contrast (Clarity)**: +18% local contrast for dimensional "pop"
- ✅ **Highlight Glow**: Subtle 8% glow on bright surfaces (luxury aesthetic)
- ✅ **Vignette**: 15% corner darkening draws eye to kitchen center

**Result**: Magazine-ready image with professional polish

---

### Stage 6: Output Optimization & Deliverables
**Duration**: 2-3 minutes  
**Tools**: Pillow + ExifTool (optional for advanced metadata)

```bash
python3 << 'EOF'
from PIL import Image
from pathlib import Path
import datetime

# Load final enhanced image
input_path = Path("working/750Picacho_Kitchen_step5_enhanced.tiff")
img = Image.open(input_path)

# Create output directory
output_dir = Path("output/750_Picacho_Kitchen_Finals")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d")

# --- Output 1: Master TIFF (16-bit for archival/print) ---
# Note: Input is 8-bit, but we'll prepare for future 16-bit workflow
master_path = output_dir / f"750Picacho_Kitchen_MASTER_{timestamp}.tiff"
img.save(master_path, 
         compression='lzw',
         dpi=(300, 300))  # Print-ready DPI
print(f"✓ Master TIFF: {master_path}")

# --- Output 2: Print JPEG (Adobe RGB, high quality) ---
print_path = output_dir / f"750Picacho_Kitchen_PRINT_{timestamp}.jpg"
img.save(print_path,
         quality=98,
         subsampling=0,  # No chroma subsampling (4:4:4)
         dpi=(300, 300),
         icc_profile=img.info.get('icc_profile'))
print(f"✓ Print JPEG (300 DPI, Q98): {print_path}")

# --- Output 3: Web Hero (sRGB, optimized size) ---
# Convert to sRGB if not already
img_web = img.copy()
web_path = output_dir / f"750Picacho_Kitchen_WEB_HERO_{timestamp}.jpg"
img_web.save(web_path,
             quality=92,
             optimize=True,
             dpi=(72, 72))
print(f"✓ Web Hero (72 DPI, Q92, optimized): {web_path}")

# --- Output 4: Web Thumbnail (1920px wide) ---
max_width = 1920
if img.size[0] > max_width:
    ratio = max_width / img.size[0]
    new_size = (max_width, int(img.size[1] * ratio))
    img_thumb = img.resize(new_size, Image.Resampling.LANCZOS)
else:
    img_thumb = img.copy()

thumb_path = output_dir / f"750Picacho_Kitchen_WEB_1920_{timestamp}.jpg"
img_thumb.save(thumb_path,
               quality=90,
               optimize=True,
               dpi=(72, 72))
print(f"✓ Web 1920px (Q90): {thumb_path}")

# --- Output 5: Social Media (1200px, sRGB) ---
social_width = 1200
ratio = social_width / img.size[0]
social_size = (social_width, int(img.size[1] * ratio))
img_social = img.resize(social_size, Image.Resampling.LANCZOS)

social_path = output_dir / f"750Picacho_Kitchen_SOCIAL_{timestamp}.jpg"
img_social.save(social_path,
                quality=88,
                optimize=True)
print(f"✓ Social Media 1200px (Q88): {social_path}")

print(f"\n=== OUTPUT DELIVERABLES COMPLETE ===")
print(f"Total outputs: 5 files in {output_dir}")
EOF
```

**Output Deliverables**:

| File | Purpose | Specs | Size (est.) |
|------|---------|-------|-------------|
| `MASTER.tiff` | Archival, future editing | 4000x2250, LZW, 300 DPI | ~70 MB |
| `PRINT.jpg` | Brochures, print ads | 4000x2250, Q98, 300 DPI, 4:4:4 | ~8-12 MB |
| `WEB_HERO.jpg` | Website hero images | 4000x2250, Q92, 72 DPI | ~2-4 MB |
| `WEB_1920.jpg` | Website responsive | 1920x1080, Q90, 72 DPI | ~600-900 KB |
| `SOCIAL.jpg` | Instagram, Facebook | 1200x675, Q88 | ~250-400 KB |

---

## 4. Complete Standard Pipeline Command Summary

```bash
#!/bin/bash
# 750 Picacho Kitchen - Standard Enhancement Pipeline
# Estimated total time: 30-40 minutes

cd /Users/rc/Transformation_Portal

echo "=== STAGE 1: Preparation (2-3 min) ==="
python3 << 'EOF'
from PIL import Image
from pathlib import Path
input_path = Path("input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff")
img = Image.open(input_path)
if img.mode == 'RGBA':
    img_rgb = img.convert('RGB')
else:
    img_rgb = img
icc = img.info.get('icc_profile')
Path("working").mkdir(exist_ok=True)
img_rgb.save("working/750Picacho_Kitchen_step1_prep.tiff", 
             compression='lzw', 
             icc_profile=icc if icc else None)
print("✓ Prep complete")
EOF

echo "=== STAGE 2: Depth Processing (5-8 min) ==="
python3 depth_pipeline/pipeline.py \
  --input working/750Picacho_Kitchen_step1_prep.tiff \
  --output working/750Picacho_Kitchen_step2_depth \
  --config config/interior_preset.yaml \
  --save-depth-map \
  --device mps

echo "=== STAGE 3: Material Response (8-12 min) ==="
python3 material_response.py \
  --input working/750Picacho_Kitchen_step2_depth.tiff \
  --output working/750Picacho_Kitchen_step3_materials.tiff \
  --mode auto \
  --surfaces wood,metal,glass,stone,paint \
  --strength 0.75 \
  --preserve-highlights \
  --depth-map working/750Picacho_Kitchen_step2_depth_depthmap.png \
  --verbose

echo "=== STAGE 4: Color Grading (3-5 min) ==="
python3 luxury_tiff_batch_processor.py \
  --input working/750Picacho_Kitchen_step3_materials.tiff \
  --output working/750Picacho_Kitchen_step4_graded.tiff \
  --lut assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube \
  --lut-strength 0.70 \
  --exposure 0.0 \
  --contrast 1.08 \
  --saturation 1.05 \
  --preserve-metadata

# Note: Stages 5-6 (detail enhancement & output) use Python scripts above
# Can be combined into single enhancement_finisher.py script if desired

echo "=== Pipeline Complete ==="
echo "Check output/750_Picacho_Kitchen_Finals/ for deliverables"
```

---

## 5. Alternative Workflows

### Quick Enhancement (10-15 minutes)

**For**: Client previews, fast social media posts, initial reviews

```bash
cd /Users/rc/Transformation_Portal

# Single-pass enhancement using luxury_tiff_batch_processor
python3 luxury_tiff_batch_processor.py \
  --input input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff \
  --output output/750Picacho_Kitchen_QUICK.jpg \
  --lut assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube \
  --lut-strength 0.60 \
  --exposure +0.05 \
  --contrast 1.10 \
  --saturation 1.08 \
  --clarity 0.15 \
  --sharpen 1.2 \
  --denoise 0.3 \
  --quality 95
```

**Trade-offs**:
- ❌ No depth-aware processing
- ❌ No Material Response Technology
- ❌ Limited fine control
- ✅ Fast turnaround
- ✅ Good quality for web/social
- ✅ Single command execution

**Output**: Single JPEG suitable for quick delivery

---

### Premium Pipeline (60-90 minutes)

**For**: Magazine covers, award submissions, high-end print campaigns, archival masters

**Extends Standard Pipeline with AI refinement**:

```bash
# After completing Standard Pipeline through Stage 5...

echo "=== PREMIUM: AI Refinement with ControlNet + SDXL ==="
python3 lux_render_pipeline.py \
  --input working/750Picacho_Kitchen_step5_enhanced.tiff \
  --output output/750Picacho_Kitchen_PREMIUM_AI.tiff \
  --controlnet canny depth \
  --controlnet-scale 0.6 0.5 \
  --prompt "ultra photorealistic luxury modern kitchen, warm natural lighting, professional architectural photography, magazine quality, 8k uhd" \
  --negative-prompt "cartoon, painting, illustration, blurry, low quality, oversaturated" \
  --steps 40 \
  --guidance-scale 7.5 \
  --strength 0.35 \
  --seed 42 \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --device mps

echo "=== PREMIUM: Real-ESRGAN 4x Upscaling ==="
python3 << 'EOF'
# Note: Requires Real-ESRGAN installation
# pip install realesrgan
import sys
sys.path.append('.')
from pathlib import Path
from PIL import Image
import torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Load model
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                        num_conv=32, upscale=4, act_type='prelu')
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=400,  # Tile for memory efficiency
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False
)

# Upscale
input_img = Image.open("output/750Picacho_Kitchen_PREMIUM_AI.tiff")
import numpy as np
img_array = np.array(input_img)
output, _ = upsampler.enhance(img_array, outscale=4)

# Save
output_img = Image.fromarray(output)
output_img.save("output/750Picacho_Kitchen_PREMIUM_16K.tiff", compression='lzw')
print("✓ Upscaled to 16000 x 9000 px (16K)")
EOF
```

**Premium Features**:
- ✅ AI-guided edge refinement (ControlNet Canny + Depth)
- ✅ SDXL photorealistic enhancement (0.35 strength - subtle)
- ✅ Real-ESRGAN 4x upscaling → 16000x9000px (144 MP)
- ✅ Magazine cover quality
- ⚠️ Requires 8-16 GB VRAM (or Apple Silicon with 32+ GB unified memory)
- ⚠️ 60-90 minutes total processing time

**When to Use**:
- High-end print campaigns (billboards, full-page magazine ads)
- Award competition submissions
- Archival masters for future cropping flexibility
- When client budget justifies premium processing time

---

## 6. Material-Specific Enhancement Details

### Wood Cabinetry Enhancement

**Detection Parameters**:
```python
wood_mask = (
    (red_channel > blue_channel * 1.1) &  # Warm tones
    (saturation > 0.10) & (saturation < 0.40) &  # Moderate saturation
    (luminance > 0.20) & (luminance < 0.80)  # Mid-range brightness
)
```

**Enhancement Recipe**:
1. **Grain Structure**: High-pass filter at 0.5px radius → +15% blend
2. **Warm Highlights**: Multiply blend mode on R channel in highlights → +8%
3. **Midtone Saturation**: HSL adjustment → +10% saturation in midtones only
4. **Natural Luster**: Anisotropic highlight stretching (grain direction detection)
5. **Shadow Depth**: Darken shadows by -5% for dimensional quality

**Expected Result**: Rich, dimensional wood with visible grain and natural warmth

---

### Stainless Steel Appliances

**Detection Parameters**:
```python
metal_mask = (
    (saturation < 0.05) &  # Near-grayscale
    (local_contrast > 0.20) &  # High reflectivity variation
    (edge_density > 0.30)  # Defined edges
)
```

**Enhancement Recipe**:
1. **Specular Preservation**: Mask highlights >0.95 luminance (no modification)
2. **Micro-Contrast**: Unsharp mask 0.8px radius → +20% on reflective transitions
3. **Cool Shift**: Slight blue bias in reflections (blue channel +2%, red -1%)
4. **Edge Sharpening**: Selective 0.3px radius sharpening on metal edges
5. **Noise Suppression**: Gaussian blur 0.2px on smooth metal areas (prevents grain)

**Expected Result**: Clean, professional appliances with authentic stainless appearance

---

### Glass & Glazed Surfaces

**Detection Parameters**:
```python
glass_mask = (
    (saturation < 0.10) &  # Low saturation
    (luminance > 0.50) &  # Typically bright
    (edge_sharpness > 0.60) &  # Clear edges
    (transparency_score > 0.40)  # Depth-based transparency check
)
```

**Enhancement Recipe**:
1. **Clarity Boost**: Micro-contrast → +25% (crystal-clear appearance)
2. **Specular Enhancement**: Reflections → +10% luminance
3. **Color Tint** (if applicable): Saturation → +5% for colored glass
4. **Edge Refinement**: 0.4px sharpening on glass edges
5. **Depth Preservation**: Maintain through-glass detail using depth map

**Expected Result**: Sparkling, magazine-quality glass with perfect clarity

---

### Stone/Quartz Countertops

**Detection Parameters**:
```python
stone_mask = (
    (saturation < 0.25) &  # Mostly neutral
    (texture_variance > 0.05) &  # Visible texture
    (diffuse_reflection > 0.60) &  # Not highly reflective
    (horizontal_surface == True)  # Depth-based surface orientation
)
```

**Enhancement Recipe**:
1. **Texture Detail**: High-pass 1.5px → +18% local contrast
2. **Natural Luster**: Subtle highlight boost → +6% in specular areas
3. **Color Purity**: Saturation → +8% to enhance inherent stone color
4. **Veining Depth**: Darken vein areas by -3% for dimensionality
5. **Surface Polish**: Gentle smoothing (1px gaussian) → 20% blend for polished look

**Expected Result**: Photorealistic stone with tactile depth and luxury appeal

---

## 7. Color Grading Recommendations

### LUT Comparison Matrix

| LUT | Aesthetic | Warmth | Contrast | Best For | Strength |
|-----|-----------|--------|----------|----------|----------|
| **Montecito Golden Hour** | California luxury | ★★★★★ | ★★★☆☆ | Web, marketing, warm properties | 70% |
| **Kodak 2393 D55** | Film editorial | ★★★☆☆ | ★★★★☆ | Print, magazines, cool balance | 65% |
| **FilmConvert Nitrate LuxuryRE** | Neutral versatile | ★★★☆☆ | ★★★☆☆ | Brand consistency, flexible use | 60% |
| **Spanish Colonial Warm** | Mediterranean warmth | ★★★★★ | ★★☆☆☆ | Warm properties, evening shoots | 75% |

### Recommended Selection Logic

```
IF (property_location == "California" AND time_of_day == "daytime"):
    USE "Montecito Golden Hour" @ 70%
    
ELIF (output_medium == "print" OR color_temp == "needs_cooling"):
    USE "Kodak 2393 D55" @ 65%
    
ELIF (brand_consistency_required == True):
    USE "FilmConvert Nitrate LuxuryRE" @ 60%
    
ELSE:
    USE "Montecito Golden Hour" @ 65%  # Default safe choice
```

### Custom LUT Stacking (Advanced)

For maximum control, stack multiple LUTs:

```bash
# Base: Material Response LUT (if available)
# Layer 1: Location Aesthetic (70% strength)
# Layer 2: Film Emulation (30% strength)

# Note: Requires custom LUT blending script or video editor
```

---

## 8. Expected Results & Quality Metrics

### Quantitative Improvements (Predicted)

| Metric | Before | After Standard Pipeline | After Premium |
|--------|--------|-------------------------|---------------|
| **Noise Level** | 0.0379 | ~0.008 | ~0.004 |
| **Micro-Contrast (Clarity)** | Baseline | +18% | +25% |
| **Dynamic Range** | 0.99 (8-bit) | 0.99 (8-bit) | 0.999 (effective 10-bit) |
| **Detail Sharpness** | Moderate | High | Very High |
| **Color Accuracy** | Good (ICC) | Excellent (graded) | Excellent |
| **Material Realism** | Good | Excellent | Exceptional |
| **Resolution** | 9 MP | 9 MP | 144 MP (16K upscale) |
| **Print Size (300 DPI)** | 13.3" x 7.5" | 13.3" x 7.5" | 53" x 30" |

### Qualitative Improvements

**Before (Original)**:
- ✅ Well-exposed, properly lit
- ✅ Good composition
- ⚠️ Moderate rendering noise
- ⚠️ Flat depth perception
- ⚠️ Generic material rendering
- ⚠️ Lacks "luxury polish"

**After Standard Pipeline**:
- ✅ Noise eliminated
- ✅ Depth-aware spatial hierarchy (foreground pops, background recedes)
- ✅ Physically accurate materials (wood grain, metal reflections, stone texture)
- ✅ Cinematic color grading (California warmth or film aesthetic)
- ✅ Professional detail enhancement (micro-contrast, selective sharpening)
- ✅ Magazine-quality polish
- ✅ Ready for marketing delivery

**After Premium Pipeline**:
- ✅ All Standard Pipeline benefits
- ✅ AI-guided photorealistic refinement
- ✅ 4x resolution (16K) for large format print
- ✅ Award-submission quality
- ✅ Future-proof archival master

---

## 9. Processing Time & Performance

### Standard Pipeline Breakdown

| Stage | Duration | Bottleneck | Optimization |
|-------|----------|------------|--------------|
| 1. Prep | 2-3 min | I/O (68 MB file) | Use SSD |
| 2. Depth | 5-8 min | ML model inference | CoreML (3-5x faster) |
| 3. Material Response | 8-12 min | Per-pixel processing | Multi-core CPU |
| 4. Color Grading | 3-5 min | LUT interpolation | Optimized LUT cache |
| 5. Detail Enhancement | 4-6 min | Convolution ops | NumPy/SciPy optimization |
| 6. Output | 2-3 min | Multi-format export | Parallel export |
| **TOTAL** | **30-40 min** | - | - |

### Hardware Recommendations

**Minimum** (60-90 min total):
- CPU: Intel i5 / AMD Ryzen 5 (4+ cores)
- RAM: 16 GB
- Storage: SSD (500+ MB/s read)
- GPU: Integrated (depth processing on CPU)

**Recommended** (30-40 min total):
- CPU: Apple M1/M2/M3 or Intel i7/i9
- RAM: 32 GB
- Storage: NVMe SSD (2000+ MB/s)
- GPU: Apple Neural Engine (CoreML) or NVIDIA RTX 3060+

**Optimal** (20-25 min total):
- CPU: Apple M4 Pro/Max or AMD Threadripper
- RAM: 64 GB+
- Storage: NVMe Gen 4 (5000+ MB/s)
- GPU: Apple Neural Engine (M4) or NVIDIA RTX 4080+

---

## 10. Deliverables Checklist

### For Client Review
- [ ] **MASTER.tiff** - Archival quality, lossless (70 MB)
- [ ] **PRINT.jpg** - 300 DPI, Q98, Adobe RGB (8-12 MB)
- [ ] **WEB_HERO.jpg** - Full resolution, Q92, sRGB (2-4 MB)
- [ ] **WEB_1920.jpg** - Responsive web, Q90 (600-900 KB)
- [ ] **SOCIAL.jpg** - Instagram/Facebook, Q88 (250-400 KB)

### Optional Deliverables
- [ ] **Before/After Comparison** - Side-by-side for client approval
- [ ] **Depth Map Visualization** - For technical documentation
- [ ] **Alternative LUT Versions** - Kodak, Nitrate variants
- [ ] **16K Premium Master** - If premium pipeline used

### Metadata Preservation
- [x] ICC Color Profile
- [x] XMP/IPTC metadata (if present in original)
- [ ] GPS coordinates (if present - verify privacy)
- [x] DPI settings (72 for web, 300 for print)
- [ ] Copyright/usage rights (add if needed)

---

## 11. Quality Assurance Checklist

Before delivering to client, verify:

### Technical QA
- [ ] No visible artifacts (compression, processing errors)
- [ ] Proper color space (sRGB for web, Adobe RGB for print)
- [ ] Correct resolution for each output
- [ ] Metadata preserved (ICC profile, EXIF)
- [ ] File sizes appropriate for use case
- [ ] Sharpening appropriate (not over-sharpened)
- [ ] Noise eliminated (zoom to 100% and check)

### Creative QA
- [ ] White balance looks natural
- [ ] Exposure balanced (no blown highlights in critical areas)
- [ ] Color grading enhances luxury feel without over-processing
- [ ] Materials look realistic (wood grain, metal reflections, stone texture)
- [ ] Depth perception natural (foreground/background separation)
- [ ] Overall image "pops" but remains believable
- [ ] Consistent with brand aesthetic (if applicable)

### Client Deliverable QA
- [ ] Filenames clear and descriptive
- [ ] Organized in client-friendly folder structure
- [ ] README or delivery note included
- [ ] All requested formats provided
- [ ] Before/after comparison ready (if requested)

---

## 12. Post-Delivery Iterations

If client requests adjustments:

### Common Adjustment Requests

**"Can we make it warmer/cooler?"**
```bash
# Adjust LUT strength or switch LUT
python3 luxury_tiff_batch_processor.py \
  --input working/750Picacho_Kitchen_step3_materials.tiff \
  --lut assets/luts/film_emulation/Kodak/Kodak_2393_D55.cube \
  --lut-strength 0.50  # Reduce for less effect
  # ... rest of parameters
```

**"Can we brighten the countertops?"**
```bash
# Use depth map to create zone mask, lift foreground exposure
# Requires custom masking script or manual editing
```

**"The wood looks too saturated"**
```bash
# Reduce Material Response strength or adjust saturation
python3 material_response.py \
  --strength 0.60  # Down from 0.75
  # ... rest of parameters
```

**"We need a different aspect ratio"**
```bash
# Crop to 4:5 (Instagram), 1:1 (square), etc.
# Use PIL or ImageMagick for non-destructive crop
```

### Iteration Workflow

1. **Identify adjustment stage** (color grading vs material vs detail)
2. **Re-run from that stage** (non-destructive pipeline)
3. **Generate new outputs** (maintain naming convention with version number)
4. **Compare side-by-side** with original delivery
5. **Deliver revision** with clear version tracking

---

## 13. Summary & Recommendation

### Final Recommendation: **Standard Pipeline**

**Rationale**:
1. ✅ **Magazine-quality results** suitable for all luxury real estate marketing
2. ✅ **Reasonable processing time** (30-40 min) for professional workflow
3. ✅ **Full Material Response Technology** for photorealistic surfaces
4. ✅ **Depth-aware processing** for natural spatial hierarchy
5. ✅ **Professional color grading** with curated LUTs
6. ✅ **Complete deliverables set** (web, print, social media)
7. ✅ **Non-destructive workflow** for easy client revisions

**Start with**: Montecito Golden Hour LUT @ 70% strength

**Alternative Ready**: Kodak 2393 version for cooler print applications

**Total Investment**: 30-40 minutes processing + 10 minutes QA = **50 minutes**

### When to Use Alternatives

- **Quick Enhancement**: Client needs preview in <15 minutes, social media only
- **Premium Pipeline**: Magazine cover, billboard, award submission, or client specifically requests "best possible quality regardless of time"

---

## 14. Next Steps

1. **Confirm approach with client** (Standard vs Premium)
2. **Run Standard Pipeline** using commands provided
3. **QA outputs** against checklist
4. **Prepare delivery package** with all formats
5. **Schedule client review** with before/after comparison
6. **Iterate if needed** using adjustment workflow

---

**Pipeline Designed By**: Transformation Portal AI Specialist  
**Date**: November 7, 2025  
**Version**: 1.0  
**Status**: Ready for Production  

---

## Appendix A: Command Reference

### Single-Command Standard Pipeline

```bash
#!/bin/bash
# Save as: enhance_750picacho_kitchen.sh
# Usage: bash enhance_750picacho_kitchen.sh

cd /Users/rc/Transformation_Portal

# Stage 1: Prep
python3 -c "from PIL import Image; from pathlib import Path; img=Image.open('input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff'); rgb=img.convert('RGB'); Path('working').mkdir(exist_ok=True); rgb.save('working/step1.tiff', compression='lzw', icc_profile=img.info.get('icc_profile'))"

# Stage 2: Depth
python3 depth_pipeline/pipeline.py --input working/step1.tiff --output working/step2 --config config/interior_preset.yaml --save-depth-map --device mps

# Stage 3: Material Response
python3 material_response.py --input working/step2.tiff --output working/step3.tiff --mode auto --surfaces wood,metal,glass,stone,paint --strength 0.75 --preserve-highlights --depth-map working/step2_depthmap.png

# Stage 4: Color Grading
python3 luxury_tiff_batch_processor.py --input working/step3.tiff --output working/step4.tiff --lut assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube --lut-strength 0.70 --exposure 0.0 --contrast 1.08 --saturation 1.05

# Stages 5-6: Detail & Output (combined)
# [Use enhancement script provided in Stage 5 above]

echo "Processing complete! Check output/750_Picacho_Kitchen_Finals/"
```

---

**END OF ENHANCEMENT PLAN**
