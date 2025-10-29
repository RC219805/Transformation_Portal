"""Apply MBAR board materials to the 750 Picacho Lane aerial photograph."""
from pathlib import Path
from board_material_aerial_enhancer import enhance_aerial

# Input: Aerial TIFF from input_images
input_path = Path("/workspaces/800-Picacho-Lane-LUTs/input_images/RC-office750Picacho_Aerial.tiff")
output_path = Path("/workspaces/800-Picacho-Lane-LUTs/processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg")

print(f"Processing: {input_path.name}")
print(f"Output: {output_path.name}")
print("Resolution: 4K (4096px width)")
print("Materials: MBAR-approved palette (8 materials)")
print()

# pylint: disable=duplicate-code  # Similar enhance_aerial call in enhance_pool_aerial.py
# Enhance with MBAR board materials
result = enhance_aerial(
    input_path,
    output_path,
    analysis_max_dim=1280,  # Fast clustering on downsampled image
    k=8,                     # 8 color clusters for material assignment
    seed=22,                 # Reproducible results
    target_width=4096,       # 4K deliverable
)
# pylint: enable=duplicate-code

print(f"✅ Enhanced aerial saved to: {result}")
print(f"✅ File size: {result.stat().st_size / (1024**2):.2f} MB")
