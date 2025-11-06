# 4K Photorealistic Rendering Workflow
## 750 Picacho Aerial - Optimized for Apple M4 Max

**Generated:** 2025-11-05
**Source:** 750Picacho_Aerial.tiff (4000×2400, 32-bit float, 146MB)
**Target:** 4K photorealistic marketing deliverable
**Hardware:** Apple M4 Max (MPS acceleration available)

---

## 🎯 Workflow Options Summary

### Option 1: Fast Track - Material Enhancement Only
**Time:** ~15 seconds | **Quality:** Good | **GPU:** Not required

Best for: Quick previews, batch processing, time-sensitive deliverables

### Option 2: AI-Enhanced - Lux Render Pipeline  
**Time:** ~90-120 seconds | **Quality:** Excellent | **GPU:** MPS (M4 Max)

Best for: High-end marketing materials, photorealistic enhancement

### Option 3: Premium - Multi-Stage AI Pipeline
**Time:** ~3-5 minutes | **Quality:** Best | **GPU:** MPS (M4 Max)

Best for: Hero images, portfolio pieces, maximum photorealism

---

## 📋 OPTION 1: Fast Track Material Enhancement

### Stage 1: Format Conversion
```bash
cd /Users/rc/Transformation_Portal

python3 << 'EOCONV'
import numpy as np
import tifffile

# Convert 32-bit float to 16-bit integer
img = tifffile.imread('input_images/750Picacho_Aerial.tiff')
img_16bit = (np.clip(img[:,:,:3], 0, 1) * 65535).astype(np.uint16)
tifffile.imwrite('input_images/750Picacho_16bit.tiff', img_16bit, photometric='rgb')
print("✓ Converted to 16-bit TIFF")
EOCONV
```

### Stage 2: Realize V8 Enhancement
```bash
python3 << 'EOREALZE'
from realize_v8_unified import enhance
from PIL import Image

# Load image
img = Image.open('input_images/750Picacho_16bit.tiff')

# Apply enhancement with "vivid" preset
result = enhance(
    img,
    preset='vivid',          # Options: signature, vivid, natural, moody
    exposure=0.15,           # +15% brightness for aerial
    contrast=1.12,           # Enhanced depth
    saturation=1.08,         # Vibrant colors
    clarity=0.25,            # Sharpness boost
    warmth=5                 # Slight warm cast
)

# Save result
result.save('processed_images/750Picacho_Enhanced.tiff', compression='tiff_lzw')
print("✓ Enhancement complete")
EOREALZE
```

**Output:** `processed_images/750Picacho_Enhanced.tiff` (4000×2400)
**Time:** ~10-15 seconds
**Next Steps:** Apply LUTs in DaVinci Resolve or export to PNG

---

## 📋 OPTION 2: AI-Enhanced Lux Render Pipeline

### Prerequisites Check
```bash
# Verify GPU acceleration
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Verify dependencies
python3 -c "from diffusers import ControlNetModel; print('✓ Diffusers ready')"
python3 -c "from realesrgan import RealESRGANer; print('✓ RealESRGAN ready')"
```

### Full Pipeline Command
```bash
cd /Users/rc/Transformation_Portal

# Create 8-bit PNG for AI processing (PIL compatibility)
python3 << 'EOCONV'
from PIL import Image
import numpy as np
import tifffile

img = tifffile.imread('input_images/750Picacho_Aerial.tiff')
img_8bit = (np.clip(img[:,:,:3], 0, 1) * 255).astype(np.uint8)
Image.fromarray(img_8bit).save('input_images/750Picacho_8bit.png', quality=100)
print("✓ Created PNG for AI processing")
EOCONV

# Run Lux Render Pipeline with optimized settings
python lux_render_pipeline.py \
  --input input_images/750Picacho_8bit.png \
  --out processed_images/LuxRender/ \
  --prompt "luxury coastal estate aerial photography, pristine landscaping, dramatic architecture, golden hour lighting, photorealistic detail, professional real estate photography" \
  --neg "blurry, low quality, artifacts, oversaturated, cartoon, unrealistic, low resolution, noise" \
  --width 1024 --height 768 \
  --steps 35 \
  --strength 0.35 \
  --gs 7.5 \
  --seed 42 \
  --upscale 4x \
  --material-response \
  --texture-boost 0.25 \
  --brand_text "750 Picacho Lane | Montecito Estate"
```

### Parameter Explanation
- `--strength 0.35`: Moderate AI influence (preserves original structure)
- `--steps 35`: High quality (more = better, slower)
- `--gs 7.5`: Guidance scale (how closely to follow prompt)
- `--upscale 4x`: Real-ESRGAN 4x upscaling → 16000×9600
- `--material-response`: Surface-aware enhancement
- `--texture-boost 0.25`: Enhance material details

**Expected Outputs:**
- `750Picacho_8bit_enhanced.png` - AI-enhanced 4K
- `750Picacho_8bit_enhanced_4x.png` - 16K upscaled version
- Processing time: ~90-120 seconds on M4 Max

---

## 📋 OPTION 3: Premium Multi-Stage Pipeline

### Complete 5-Stage Workflow

#### Stage 1: HDR Enhancement
```bash
python3 << 'EOSTAGE1'
from realize_v8_unified import enhance
from PIL import Image
import tifffile
import numpy as np

# Load and enhance with HDR preset
img_array = tifffile.imread('input_images/750Picacho_Aerial.tiff')
img_8bit = (np.clip(img_array[:,:,:3], 0, 1) * 255).astype(np.uint8)
img = Image.fromarray(img_8bit)

result = enhance(
    img,
    preset='vivid',
    exposure=0.12,
    contrast=1.15,
    saturation=1.05,
    clarity=0.20
)

result.save('processed_images/Stage1_HDR.png', quality=100)
print("✓ Stage 1: HDR Enhancement complete")
EOSTAGE1
```

#### Stage 2: Material-Aware Enhancement
```bash
# Apply board material enhancement
python board_material_aerial_enhancer.py \
  processed_images/Stage1_HDR.png \
  processed_images/Stage2_Material.png \
  --k-means-clusters 8 \
  --texture-blend 0.35 \
  --aerial-perspective

echo "✓ Stage 2: Material enhancement complete"
```

#### Stage 3: AI Refinement (ControlNet + SDXL)
```bash
python lux_render_pipeline.py \
  --input processed_images/Stage2_Material.png \
  --out processed_images/Stage3_AI/ \
  --prompt "luxury estate aerial view, dramatic architecture, pristine pool, lush landscaping, architectural photography, ultra detailed, 8k, photorealistic" \
  --neg "blurry, artifacts, cartoon, painting, illustration" \
  --width 1024 --height 768 \
  --steps 40 \
  --strength 0.30 \
  --gs 8.0 \
  --seed 42 \
  --material-response \
  --texture-boost 0.28

echo "✓ Stage 3: AI refinement complete"
```

#### Stage 4: Super-Resolution Upscaling
```bash
python3 << 'EOUPSCALE'
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

# Initialize Real-ESRGAN 4x model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=False  # Use FP32 on M4 Max MPS
)

# Upscale
img = cv2.imread('processed_images/Stage3_AI/750Picacho_8bit_enhanced.png')
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite('processed_images/Stage4_Upscaled_16K.png', output)

print("✓ Stage 4: 16K upscaling complete")
print(f"Final resolution: {output.shape[1]}×{output.shape[0]}")
EOUPSCALE
```

#### Stage 5: Final Polish & Export
```bash
python3 << 'EOPOLISH'
from PIL import Image, ImageEnhance, ImageFilter

# Load upscaled image
img = Image.open('processed_images/Stage4_Upscaled_16K.png')

# Final touches
img = ImageEnhance.Sharpness(img).enhance(1.15)  # Subtle sharpening
img = ImageEnhance.Color(img).enhance(1.03)      # Color pop

# Export 4K version (downsampled from 16K for maximum quality)
img_4k = img.resize((3840, 2160), Image.Resampling.LANCZOS)
img_4k.save('processed_images/750Picacho_FINAL_4K.png', quality=100, optimize=True)

# Also save 16K master
img.save('processed_images/750Picacho_FINAL_16K.png', quality=100)

print("✓ Stage 5: Final polish complete")
print("✓ 4K deliverable: processed_images/750Picacho_FINAL_4K.png")
print("✓ 16K master: processed_images/750Picacho_FINAL_16K.png")
EOPOLISH
```

**Total Time:** ~3-5 minutes
**Final Outputs:**
- 4K: 3840×2160 (cinema standard)
- 16K: 16000×9600 (archival master)

---

## 🎨 Recommended: Option 2 (AI-Enhanced)

For your 750 Picacho aerial rendering, **Option 2** provides the best balance:

### Why Option 2?
✅ **Photorealistic AI enhancement** via Stable Diffusion + ControlNet  
✅ **Preserves architectural accuracy** with low strength (0.35)  
✅ **4x upscaling** via Real-ESRGAN for crisp detail  
✅ **Material Response** for surface realism  
✅ **Fast processing** (~2 minutes on M4 Max)  
✅ **Brand overlay** capability  

### Execution
```bash
# One-command solution
cd /Users/rc/Transformation_Portal

# Convert to PNG
python3 -c "from PIL import Image; import numpy as np; import tifffile; img = tifffile.imread('input_images/750Picacho_Aerial.tiff'); Image.fromarray((np.clip(img[:,:,:3], 0, 1) * 255).astype(np.uint8)).save('input_images/750Picacho_8bit.png')"

# Run Lux Render Pipeline
python lux_render_pipeline.py \
  --input input_images/750Picacho_8bit.png \
  --out processed_images/Final/ \
  --prompt "luxury montecito estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour, ultra detailed, professional real estate" \
  --neg "blurry, artifacts, cartoon, oversaturated" \
  --width 1024 --height 768 --steps 35 --strength 0.35 --gs 7.5 \
  --upscale 4x --material-response --texture-boost 0.25 \
  --brand_text "750 Picacho Lane"
```

**Timeline:**
- Conversion: 5 seconds
- AI Processing: 60-80 seconds
- Upscaling: 30-40 seconds
- **Total: ~2 minutes**

---

## 🔍 Quality Validation Checklist

After processing, verify:

1. **Architectural Accuracy** - Building lines/proportions preserved
2. **Material Realism** - Pool water, roofing, landscaping look natural
3. **Lighting Consistency** - No unrealistic shadows or highlights
4. **Detail Preservation** - No blur or artifacts in critical areas
5. **Color Balance** - Warm, inviting, but not oversaturated
6. **Edge Sharpness** - Building edges crisp, not oversharpened
7. **Sky Quality** - Natural gradient, no banding
8. **Resolution** - Zoom to 100% and check detail at pixel level

---

## 📊 Comparison Matrix

| Metric | Option 1 | Option 2 | Option 3 |
|--------|----------|----------|----------|
| **Time** | 15s | 2min | 5min |
| **Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GPU Required** | No | Yes (MPS) | Yes (MPS) |
| **Max Resolution** | 4K | 16K | 16K |
| **AI Enhancement** | No | Yes | Yes |
| **Material Response** | No | Yes | Yes |
| **Upscaling** | No | Real-ESRGAN 4x | Real-ESRGAN 4x |
| **Best For** | Previews | Marketing | Portfolio |

---

## 💡 Pro Tips

1. **Prompt Engineering Matters** - Be specific about "architectural photography" and "photorealistic"
2. **Lower Strength = More Accuracy** - Use 0.30-0.40 for architectural renders
3. **Seed Consistency** - Use same seed (--seed 42) for variations
4. **Batch Processing** - Process multiple views with same settings
5. **Save Intermediate Stages** - Keep Stage 1, 2, 3 outputs for iterations
6. **Monitor GPU Memory** - M4 Max handles 1024×768 comfortably
7. **Use Material Response** - Critical for realistic surfaces in aerials

---

## 🚀 Ready to Execute?

**Recommended command:**
```bash
cd /Users/rc/Transformation_Portal && \
python3 -c "from PIL import Image; import numpy as np; import tifffile; img = tifffile.imread('input_images/750Picacho_Aerial.tiff'); Image.fromarray((np.clip(img[:,:,:3], 0, 1) * 255).astype(np.uint8)).save('input_images/750Picacho_Ready.png')" && \
python lux_render_pipeline.py --input input_images/750Picacho_Ready.png --out processed_images/Photorealistic/ --prompt "luxury montecito coastal estate aerial photography, dramatic architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed 8k, professional architectural photography" --neg "blurry, artifacts, cartoon, painting, oversaturated, unrealistic" --width 1024 --height 768 --steps 35 --strength 0.35 --gs 7.5 --seed 42 --upscale 4x --material-response --texture-boost 0.25 --brand_text "750 Picacho Lane | Montecito"
```

Say "execute option 2" to begin!
