#!/usr/bin/env python3
"""Generate placeholder textures for MBAR board materials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "textures" / "board_materials"

MATERIAL_COLORS: dict[str, Tuple[int, int, int]] = {
    "plaster_marmorino_westwood_beige": (240, 230, 215),
    "stone_bokara_coastal": (195, 175, 155),
    "cladding_sculptform_warm": (200, 170, 140),
    "screens_grey_gum": (145, 140, 135),
    "equitone_lt85": (95, 95, 100),
    "bison_weathered_ipe": (165, 160, 165),
    "dark_bronze_anodized": (75, 60, 50),
    "louvretec_powder_white": (245, 245, 245),
}


def _import_pil():
    """Lazy import PIL to keep --help and --list startup cheap."""
    from PIL import Image

    return Image


def create_texture(
    base_color: Tuple[int, int, int],
    *,
    size: int = 512,
    noise_intensity: float = 8.0,
    seed: int = 42,
):
    """Create a deterministic subtle material texture."""
    Image = _import_pil()
    img = np.full((size, size, 3), base_color, dtype=np.float32)

    rng = np.random.default_rng(seed)
    img += rng.normal(0, noise_intensity, (size, size, 3))

    gradient = np.linspace(-5, 5, size)
    gradient_2d = gradient[:, None] + gradient[None, :]
    img += gradient_2d[..., None]

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MBAR-approved material textures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated textures",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        choices=[256, 512, 1024, 2048, 4096],
        help="Texture size in pixels",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=8.0,
        help="Noise intensity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        choices=list(MATERIAL_COLORS.keys()),
        help="Specific materials to generate",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available materials and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate all requested MBAR material textures."""
    args = parse_args(argv)

    if args.list:
        print("Available MBAR Materials:")
        for name, color in MATERIAL_COLORS.items():
            print(f"  - {name:<40} RGB{color}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    materials_to_generate = {name: MATERIAL_COLORS[name] for name in args.materials} if args.materials else MATERIAL_COLORS

    print("=" * 70)
    print("MBAR TEXTURE GENERATOR")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Texture size: {args.size}x{args.size}")
    print(f"Noise intensity: {args.noise}")
    print(f"Materials: {len(materials_to_generate)}/{len(MATERIAL_COLORS)}")
    print("=" * 70)

    for index, (name, color) in enumerate(materials_to_generate.items(), 1):
        output_path = args.output_dir / f"{name}.png"
        texture = create_texture(color, size=args.size, noise_intensity=args.noise, seed=args.seed)
        texture.save(output_path, optimize=True)
        size_kb = output_path.stat().st_size / 1024
        print(f"[{index}/{len(materials_to_generate)}] {name:<40} ({size_kb:.1f} KB)")

    print("=" * 70)
    print(f"Generated {len(materials_to_generate)} textures in {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
