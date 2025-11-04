"""Example usage of board_material_aerial_enhancer.

This script demonstrates how to apply MBAR board materials to an aerial photograph
of the Montecito estate.
"""
from pathlib import Path
from board_material_aerial_enhancer import enhance_aerial

# Example: Enhance aerial with default settings
input_image = Path("input_images/montecito_aerial.jpg")
output_image = Path("processed_images/montecito_aerial_mbar_enhanced.jpg")

if input_image.exists():
    print(f"Enhancing {input_image}...")
    result = enhance_aerial(
        input_image,
        output_image,
        analysis_max_dim=1280,  # Faster clustering on downsampled image
        k=8,  # 8 color clusters for material assignment
        seed=22,  # Reproducible results
        target_width=4096,  # 4K output
    )
    print(f"Enhanced aerial saved to: {result}")
else:
    print(f"Input file not found: {input_image}")
    print("Please provide an aerial photograph to enhance.")

# Example: Custom texture paths
custom_textures = {
    "plaster": Path("textures/custom/plaster.png"),
    "stone": Path("textures/custom/stone.png"),
    # ... other materials
}

# Uncomment to use custom textures:
# result = enhance_aerial(
#     input_image,
#     output_image,
#     textures=custom_textures,
# )
