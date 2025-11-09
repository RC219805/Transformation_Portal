#!/usr/bin/env python3
"""AI Enhancement for 750 Picacho - Fixed dimensions"""
from pathlib import Path

import torch
from controlnet_aux import CannyDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image, ImageEnhance

print("=" * 60)
print("AI PHOTOREALISTIC ENHANCEMENT - 750 Picacho")
print("=" * 60)

INPUT_IMAGE = "input_images/750Picacho_Ready.png"
OUTPUT_DIR = Path("processed_images/Photorealistic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "luxury montecito coastal estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed 8k, professional architectural photography, photorealistic"
NEGATIVE_PROMPT = "blurry, artifacts, cartoon, painting, oversaturated, unrealistic, low quality, distorted, deformed"

# Fixed dimensions (must be multiples of 8)
WIDTH = 1024
HEIGHT = 616  # Maintains aspect ratio of 4000:2400, multiple of 8

STEPS = 35
GUIDANCE_SCALE = 7.5
STRENGTH = 0.35
SEED = 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load and resize image
print("\n[1/6] Loading image...")
image = Image.open(INPUT_IMAGE).convert("RGB")
original_size = image.size
print(f"  Original: {original_size}")
image_resized = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
print(f"  Processing: {WIDTH}×{HEIGHT}")

# Generate Canny
print("\n[2/6] Generating Canny edges...")
canny_detector = CannyDetector()
canny_image = canny_detector(image_resized, low_threshold=100, high_threshold=200)
canny_output = OUTPUT_DIR / "750Picacho_canny.png"
canny_image.save(canny_output)
print(f"  ✓ Saved: {canny_output}")

# Load models
print("\n[3/6] Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32
).to(device)
print("  ✓ ControlNet ready")

print("\n[4/6] Loading Stable Diffusion...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("  ✓ Pipeline ready")

# Generate
print("\n[5/6] Running AI enhancement (60-90 seconds)...")
print(f"  Steps: {STEPS}, Strength: {STRENGTH}, Guidance: {GUIDANCE_SCALE}")

generator = torch.Generator(device=device).manual_seed(SEED)
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=image_resized,
    control_image=canny_image,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    strength=STRENGTH,
    generator=generator,
).images[0]

ai_output = OUTPUT_DIR / "750Picacho_AI.png"
result.save(ai_output, quality=100)
print(f"\n  ✓ AI complete: {ai_output}")

# Material Response finishing
print("\n[6/6] Material Response finishing...")
result = ImageEnhance.Sharpness(result).enhance(1.25)
result = ImageEnhance.Color(result).enhance(1.08)
result = ImageEnhance.Contrast(result).enhance(1.12)

final_output = OUTPUT_DIR / "750Picacho_FINAL.png"
result.save(final_output, quality=100)

# Upscale to 4K
result_4k = result.resize(original_size, Image.Resampling.LANCZOS)
upscaled_output = OUTPUT_DIR / "750Picacho_FINAL_4K.png"
result_4k.save(upscaled_output, quality=100)

print("\n" + "=" * 60)
print("✓ COMPLETE!")
print("=" * 60)
print("\nOutputs:")
print(f"  • {canny_output.name}")
print(f"  • {ai_output.name}")
print(f"  • {final_output.name}")
print(f"  • {upscaled_output.name} ⭐ DELIVERABLE")
print("=" * 60)
