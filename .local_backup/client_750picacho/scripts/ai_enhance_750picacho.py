#!/usr/bin/env python3
"""
AI Enhancement for 750 Picacho Aerial
Using Stable Diffusion + ControlNet with Material Response
"""
from pathlib import Path

import numpy as np
import torch
from controlnet_aux import CannyDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image, ImageEnhance

print("=" * 60)
print("AI PHOTOREALISTIC ENHANCEMENT")
print("750 Picacho Lane - Aerial Rendering")
print("=" * 60)

# Configuration
INPUT_IMAGE = "input_images/750Picacho_Ready.png"
OUTPUT_DIR = Path("processed_images/Photorealistic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "luxury montecito coastal estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed 8k, professional architectural photography, photorealistic"
NEGATIVE_PROMPT = "blurry, artifacts, cartoon, painting, oversaturated, unrealistic, low quality, distorted, deformed"

# Model settings
WIDTH = 1024
HEIGHT = 768
STEPS = 35
GUIDANCE_SCALE = 7.5
STRENGTH = 0.35
SEED = 42

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load image
print("\n[1/6] Loading image...")
image = Image.open(INPUT_IMAGE).convert("RGB")
original_size = image.size
print(f"  Original size: {original_size}")

# Resize to processing dimensions
image_resized = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
print(f"  Processing size: {WIDTH}×{HEIGHT}")

# Generate Canny edge map
print("\n[2/6] Generating ControlNet conditioning (Canny edges)...")
canny_detector = CannyDetector()
canny_image = canny_detector(image_resized, low_threshold=100, high_threshold=200)
canny_output = OUTPUT_DIR / "750Picacho_canny_edges.png"
canny_image.save(canny_output)
print(f"  ✓ Saved: {canny_output}")

# Load ControlNet
print("\n[3/6] Loading ControlNet model...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32
).to(device)
print("  ✓ ControlNet loaded")

# Load pipeline
print("\n[4/6] Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("  ✓ Pipeline ready")

# Generate
print("\n[5/6] Running AI enhancement...")
print(f"  Prompt: {PROMPT[:80]}...")
print(f"  Steps: {STEPS}, Strength: {STRENGTH}, Guidance: {GUIDANCE_SCALE}")
print("  This will take ~60-90 seconds...")

generator = torch.Generator(device=device).manual_seed(SEED)

with torch.no_grad():
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

# Save AI result
ai_output = OUTPUT_DIR / "750Picacho_AI_Enhanced.png"
result.save(ai_output, quality=100)
print(f"\n  ✓ AI enhancement complete: {ai_output}")

# Material Response post-processing
print("\n[6/6] Applying Material Response finishing...")

# Enhance details
enhancer = ImageEnhance.Sharpness(result)
result = enhancer.enhance(1.25)  # Clarity boost

# Slight saturation boost for vibrancy
enhancer = ImageEnhance.Color(result)
result = enhancer.enhance(1.08)

# Subtle contrast for depth
enhancer = ImageEnhance.Contrast(result)
result = enhancer.enhance(1.12)

# Save final result
final_output = OUTPUT_DIR / "750Picacho_FINAL.png"
result.save(final_output, quality=100, optimize=True)
print("  ✓ Material Response applied")

# Upscale back to original resolution
print("\n  Upscaling to original resolution...")
result_upscaled = result.resize(original_size, Image.Resampling.LANCZOS)
upscaled_output = OUTPUT_DIR / "750Picacho_FINAL_4K.png"
result_upscaled.save(upscaled_output, quality=100)

print("\n" + "=" * 60)
print("✓ PROCESSING COMPLETE!")
print("=" * 60)
print("\nOutputs:")
print(f"  • Canny edges: {canny_output}")
print(f"  • AI enhanced: {ai_output}")
print(f"  • Final (processed): {final_output}")
print(f"  • Final (4K original size): {upscaled_output}")
print(f"\nRecommended for delivery: {upscaled_output.name}")
print("=" * 60)
