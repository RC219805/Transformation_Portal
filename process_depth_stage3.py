#!/usr/bin/env python3
"""Stage 3: Depth-aware enhancement using ControlNet depth processor"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from controlnet_aux import ZoeDetector
import os

print("Loading depth processor...")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")

print("Processing image with depth awareness...")
input_path = "processed_images/750Picacho_Stage2_MaterialEnhanced.png"
image = Image.open(input_path).convert("RGB")

# Generate depth map
print("Estimating depth...")
depth_map = zoe(image)
depth_map.save("processed_images/750Picacho_Stage3_depth_viz.png")
print("✓ Saved depth visualization")

# Apply depth-aware enhancements
print("Applying depth-guided enhancements...")

# Convert to arrays
img_array = np.array(image).astype(float) / 255.0
depth_array = np.array(depth_map.convert('L')).astype(float) / 255.0

# Zone-based processing (5 depth zones)
zones = []
for i in range(5):
    threshold_low = i * 0.2
    threshold_high = (i + 1) * 0.2
    zone_mask = (depth_array >= threshold_low) & (depth_array < threshold_high)
    zones.append(zone_mask)

# Apply zone-specific adjustments
enhanced = img_array.copy()
contrasts = [1.25, 1.15, 1.05, 0.95, 0.85]
saturations = [1.15, 1.10, 1.00, 0.92, 0.85]

for zone_idx, (zone_mask, contrast, saturation) in enumerate(zip(zones, contrasts, saturations)):
    # Apply contrast
    enhanced[zone_mask] = np.clip((enhanced[zone_mask] - 0.5) * contrast + 0.5, 0, 1)
    
    # Apply saturation
    gray = np.dot(enhanced[zone_mask], [0.299, 0.587, 0.114])[:, np.newaxis]
    enhanced[zone_mask] = np.clip(gray + (enhanced[zone_mask] - gray) * saturation, 0, 1)

# Atmospheric haze on distant areas
haze_strength = 0.15
haze_color = np.array([0.85, 0.88, 0.92])  # Cool blue-white haze
far_mask = depth_array < 0.3
enhanced[far_mask] = enhanced[far_mask] * (1 - haze_strength) + haze_color * haze_strength

# Convert back to image
result = Image.fromarray((enhanced * 255).astype(np.uint8))

# Apply clarity enhancement
print("Applying clarity boost...")
result = ImageEnhance.Sharpness(result).enhance(1.65)

# Save result
output_path = "processed_images/750Picacho_Stage3_DepthEnhanced.png"
result.save(output_path, quality=100)
print(f"✓ Saved enhanced image: {output_path}")
print("✓ Stage 3 complete")
