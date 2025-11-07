"""Apply MBAR board materials to the 750 Picacho Lane aerial photograph."""
from pathlib import Path
import numpy as np
from PIL import Image
from board_material_aerial_enhancer import enhance_aerial

# Input: Aerial TIFF from input_images
input_path = Path("/workspaces/800-Picacho-Lane-LUTs/input_images/RC-office750Picacho_Aerial.tiff")
output_path = Path("/workspaces/800-Picacho-Lane-LUTs/processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg")

print(f"Processing: {input_path.name}")
print(f"Output: {output_path.name}")
print("Resolution: 4K (4096px width)")
print("Materials: MBAR-approved palette (8 materials)")
print()

# Load and prepare image
image = Image.open(input_path).convert("RGB")
image_array = np.asarray(image, dtype=np.float32) / 255.0

# pylint: disable=duplicate-code  # Similar enhance_aerial call in enhance_pool_aerial.py
# Enhance with MBAR board materials
# Note: enhance_aerial expects a numpy array, not a Path
result_array = enhance_aerial(
    image_array,
    str(output_path),  # Convert Path to str for out_path parameter
    k=8,  # 8 color clusters for material assignment
    textures=None,  # Use default textures
)
# pylint: enable=duplicate-code

print(f"✅ Enhanced aerial saved to: {output_path}")
print(f"✅ File size: {output_path.stat().st_size / (1024**2):.2f} MB")
