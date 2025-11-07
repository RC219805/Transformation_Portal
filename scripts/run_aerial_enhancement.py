"""Apply MBAR board materials to the 750 Picacho Lane aerial photograph."""
from pathlib import Path
import numpy as np
from PIL import Image
from board_material_aerial_enhancer import enhance_aerial


def main():
    """Main entry point for the aerial enhancement script."""
    # Input: Aerial TIFF from input_images
    input_path = Path("input_images/RC-office750Picacho_Aerial.tiff")
    output_path = Path("processed_images/750_Picacho_Aerial_MBAR_Enhanced.jpg")

    print(f"Processing: {input_path.name}")
    print(f"Output: {output_path.name}")
    print("Materials: MBAR-approved palette (8 materials)")
    print()

    # Load image and convert to numpy array
    image = Image.open(input_path).convert("RGB")
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Enhance with MBAR board materials
    enhance_aerial(
        image_array,
        str(output_path),
        k=8,                     # 8 color clusters for material assignment
    )

    # File was saved by enhance_aerial
    print(f"✅ Enhanced aerial saved to: {output_path}")
    if output_path.exists():
        print(f"✅ File size: {output_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
