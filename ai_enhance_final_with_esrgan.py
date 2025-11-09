#!/usr/bin/env python3
"""AI Enhancement with Real-ESRGAN 4x Upscaling - FULLY FUNCTIONAL"""
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from controlnet_aux import CannyDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image, ImageEnhance

# Real-ESRGAN imports (now working!)
from realesrgan import RealESRGANer

print("=" * 70)
print("AI ENHANCEMENT + REAL-ESRGAN 4X UPSCALING")
print("750 Picacho Aerial - Full Quality Pipeline")
print("=" * 70)

INPUT = "input_images/750Picacho_Ready.png"
OUT_DIR = Path("processed_images/Photorealistic_4x")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "luxury montecito coastal estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed, professional architectural photography, photorealistic, 8k"
NEG_PROMPT = "blurry, artifacts, cartoon, painting, oversaturated, unrealistic, low quality, distorted"

# SD settings
WIDTH, HEIGHT = 768, 512
STEPS, GUIDANCE, STRENGTH, SEED = 35, 7.5, 0.35, 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Device: {device}\n")

# Load image
print("[1/7] Loading image...")
img = Image.open(INPUT).convert("RGB")
orig_size = img.size
print(f"  Original: {orig_size} → Processing: {WIDTH}×{HEIGHT}")
img_resized = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

# Canny edges
print("\n[2/7] Generating Canny edges...")
canny = CannyDetector()(img_resized, 100, 200)
canny.save(OUT_DIR / "canny.png")
print("  ✓ Edge detection complete")

# Load ControlNet
print("\n[3/7] Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
).to(device)
print("  ✓ Ready")

# Load pipeline
print("\n[4/7] Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("  ✓ Ready")

# Generate
print(f"\n[5/7] AI Enhancement ({STEPS} steps, ~90 seconds)...")
gen = torch.Generator(device=device).manual_seed(SEED)
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEG_PROMPT,
    image=img_resized,
    control_image=canny,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    strength=STRENGTH,
    generator=gen
).images[0]

result.save(OUT_DIR / "ai_enhanced.png", quality=100)
print("  ✓ AI enhancement complete")

# Material Response
print("\n[6/7] Material Response finishing...")
result = ImageEnhance.Sharpness(result).enhance(1.3)
result = ImageEnhance.Color(result).enhance(1.10)
result = ImageEnhance.Contrast(result).enhance(1.15)

# Initialize Real-ESRGAN 4x upscaler
print("\n[7/7] Real-ESRGAN 4x Upscaling...")
print("  Initializing upsampler...")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)
print("  ✓ Upsampler ready")

# Convert PIL to cv2 format
result_cv2 = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

# Upscale with Real-ESRGAN
print("  Upscaling 768×512 → 3072×2048 (4x)...")
upscaled_cv2, _ = upsampler.enhance(result_cv2, outscale=4)
print(f"  ✓ Upscaled to: {upscaled_cv2.shape[1]}×{upscaled_cv2.shape[0]}")

# Convert back to PIL
upscaled = Image.fromarray(cv2.cvtColor(upscaled_cv2, cv2.COLOR_BGR2RGB))

# Resize to original dimensions (4000×2400)
print(f"  Resizing to original dimensions: {orig_size[0]}×{orig_size[1]}...")
final = upscaled.resize(orig_size, Image.Resampling.LANCZOS)

# Save results
final.save(OUT_DIR / "750Picacho_FINAL_4K_ESRGAN.png", quality=100, optimize=True)
upscaled.save(OUT_DIR / "750Picacho_4x_upscaled.png", quality=100)

print("\n" + "=" * 70)
print("✅ COMPLETE!")
print("=" * 70)
print("\nDeliverables:")
print(f"  • {OUT_DIR / '750Picacho_FINAL_4K_ESRGAN.png'} ⭐ (4K - Real-ESRGAN)")
print(f"  • {OUT_DIR / '750Picacho_4x_upscaled.png'} (3072×2048)")
print(f"  • {OUT_DIR / 'ai_enhanced.png'} (768×512 - SD output)")
print("\nQuality: Real-ESRGAN 4x AI upscaling (maximum detail)")
print("=" * 70)
