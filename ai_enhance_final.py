#!/usr/bin/env python3
"""AI Enhancement - Standard Dimensions"""
import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
from pathlib import Path

print("=" * 70)
print("AI PHOTOREALISTIC ENHANCEMENT - 750 Picacho Aerial")
print("=" * 70)

INPUT = "input_images/750Picacho_Ready.png"
OUT_DIR = Path("processed_images/Photorealistic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "luxury montecito coastal estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed, professional architectural photography, photorealistic, 8k"
NEG_PROMPT = "blurry, artifacts, cartoon, painting, oversaturated, unrealistic, low quality, distorted"

# Standard SD 1.5 dimensions
WIDTH, HEIGHT = 768, 512  # Matches aspect ratio better
STEPS, GUIDANCE, STRENGTH, SEED = 35, 7.5, 0.35, 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Device: {device}\n")

# Load image
print("[1/6] Loading image...")
img = Image.open(INPUT).convert("RGB")
orig_size = img.size
print(f"  Original: {orig_size} → Processing: {WIDTH}×{HEIGHT}")
img_resized = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

# Canny edges
print("\n[2/6] Generating Canny edges...")
canny = CannyDetector()(img_resized, 100, 200)
canny.save(OUT_DIR / "canny.png")
print("  ✓ Edge detection complete")

# Load ControlNet
print("\n[3/6] Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
).to(device)
print("  ✓ Ready")

# Load pipeline
print("\n[4/6] Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("  ✓ Ready")

# Generate
print(f"\n[5/6] AI Enhancement ({STEPS} steps, ~90 seconds)...")
print(f"  Prompt: {PROMPT[:60]}...")

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
print("\n[6/6] Material Response finishing...")
result = ImageEnhance.Sharpness(result).enhance(1.3)
result = ImageEnhance.Color(result).enhance(1.10)
result = ImageEnhance.Contrast(result).enhance(1.15)
result.save(OUT_DIR / "final_processed.png", quality=100)

# Upscale to 4K
result_4k = result.resize(orig_size, Image.Resampling.LANCZOS)
final_path = OUT_DIR / "750Picacho_FINAL_4K.png"
result_4k.save(final_path, quality=100, optimize=True)

print("\n" + "=" * 70)
print("✓ SUCCESS!")
print("=" * 70)
print(f"\nDeliverable: {final_path}")
print(f"Size: {orig_size[0]}×{orig_size[1]} (4K)")
print("\n" + "=" * 70)
