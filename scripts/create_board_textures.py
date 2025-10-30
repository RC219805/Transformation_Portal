"""Generate placeholder textures for MBAR board materials.

Creates procedurally-generated texture plates with colors representative
of the approved Montecito Board of Architectural Review palette.
"""
from pathlib import Path
import numpy as np
from PIL import Image

# MBAR-approved material colors (RGB 0-255)
MATERIAL_COLORS = {
    "plaster_marmorino_westwood_beige": (240, 230, 215),
    "stone_bokara_coastal": (195, 175, 155),
    "cladding_sculptform_warm": (200, 170, 140),
    "screens_grey_gum": (145, 140, 135),
    "equitone_lt85": (95, 95, 100),
    "bison_weathered_ipe": (165, 160, 165),
    "dark_bronze_anodized": (75, 60, 50),
    "louvretec_powder_white": (245, 245, 245),
}


def create_texture(base_color: tuple[int, int, int], size: int = 512) -> Image.Image:
    """Create a subtle procedural texture with color variation.

    Args:
        base_color: Base RGB color (0-255).
        size: Texture dimension in pixels.

    Returns:
        PIL Image with subtle noise and variation.
    """
    # Create base color array
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = base_color[0]
    img[..., 1] = base_color[1]
    img[..., 2] = base_color[2]

    # Add subtle noise for texture variation
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 8, (size, size, 3))
    img = img + noise

    # Add subtle gradient for depth
    y_gradient = np.linspace(-5, 5, size)
    x_gradient = np.linspace(-5, 5, size)
    gradient = y_gradient[:, None] + x_gradient[None, :]
    img = img + gradient[..., None]

    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def main():
    """Generate all MBAR material textures."""
    output_dir = Path(__file__).parent / "textures" / "board_materials"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, color in MATERIAL_COLORS.items():
        output_path = output_dir / f"{name}.png"
        texture = create_texture(color, size=512)
        texture.save(output_path)
        print(f"Created: {output_path}")

    print(f"\nAll {len(MATERIAL_COLORS)} textures created in {output_dir}")


if __name__ == "__main__":
    main()
