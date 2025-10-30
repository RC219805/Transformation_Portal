"""Custom rendering refinement pipeline for the 750px review set.

This module builds on the QA snapshot provided for the five renders in
``/Users/rc/input_renderings_750``.  Each render receives a targeted
processing recipe that mirrors the requested highlight recovery, color
temperature harmonisation, micro-contrast, and grounding tweaks.

The adjustments intentionally avoid heavyweight third-party dependencies so
the script can run on a vanilla Python environment with Pillow and NumPy
installed.  The processing steps are conservative—highlight recovery and
shadow lifts are performed in floating point before returning to 8-bit RGB,
and the local contrast/contact shadow passes rely on light-weight blurs to
stay deterministic.

Usage (defaults match the brief):

    python process_renderings_750.py \
        --input /Users/rc/input_renderings_750 \
        --output /Users/rc/output_renderings_750

"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}
CONVERTIBLE_IMAGE_SUFFIXES = {".tif", ".tiff", ".webp", ".bmp", ".tga", ".psd", ".exr"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _iter_render_candidates(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file():
            yield path


def _require_module(module_name: str):
    """Return the imported module, raising a helpful error when missing."""

    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"The optional dependency '{module_name}' is required to convert that asset."
        )
    return importlib.import_module(module_name)


def _convert_exr_to_jpg(exr_path: Path, jpg_path: Path) -> None:
    """Convert an OpenEXR image to JPEG with simple tone mapping."""

    OpenEXR = _require_module("OpenEXR")
    Imath = _require_module("Imath")
    exr_file = OpenEXR.InputFile(str(exr_path))
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [
        np.frombuffer(exr_file.channel(channel, pixel_type), dtype=np.float32)
        for channel in ("R", "G", "B")
    ]
    exr_file.close()

    rgb = [channel.reshape(height, width) for channel in channels]
    rgb = np.stack(rgb, axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    Image.fromarray(rgb, mode="RGB").save(jpg_path, format="JPEG", quality=95)


def _convert_with_pillow(source: Path, destination: Path) -> None:
    """Convert ``source`` to RGB JPEG using Pillow."""

    with Image.open(source) as img:
        if img.mode in {"RGBA", "LA", "P"}:
            base = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            base.paste(img, mask=img.split()[-1])
            img = base
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(destination, format="JPEG", quality=95)


def convert_renderings_to_jpeg(input_dir: Path, output_dir: Path | None = None) -> Path:
    """Convert unsupported renderings to JPEG for downstream pipelines."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "converted_for_api"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in _iter_render_candidates(input_dir):
        suffix = path.suffix.lower()
        destination = output_dir / (path.stem + ".jpg")

        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            copied_destination = output_dir / path.name
            if copied_destination.exists() and copied_destination.stat().st_mtime >= path.stat().st_mtime:
                continue
            shutil.copy2(path, copied_destination)
            continue

        if suffix not in CONVERTIBLE_IMAGE_SUFFIXES:
            continue

        if destination.exists() and path.stat().st_mtime <= destination.stat().st_mtime:
            continue

        if suffix == ".exr":
            _convert_exr_to_jpg(path, destination)
        else:
            _convert_with_pillow(path, destination)

    return output_dir


def ensure_supported_renderings(input_dir: Path) -> Path:
    """Return a directory containing only API-compatible renderings."""

    paths = list(_iter_render_candidates(input_dir))
    convertible = [p for p in paths if p.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES]
    if not convertible:
        return input_dir

    converted_dir = convert_renderings_to_jpeg(input_dir)
    return converted_dir


def _load_rgb(path: Path) -> np.ndarray:
    """Return the image located at *path* as an ``np.float32`` RGB array."""

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    """Persist an ``np.float32`` RGB array to *path* as an 8-bit image."""

    clipped = np.clip(arr, 0.0, 1.0)
    out = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _luminance(arr: np.ndarray) -> np.ndarray:
    """Return the Rec.709 luminance for *arr*."""

    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _apply_highlight_compression(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    return np.power(arr, 1.0 + strength)


def _apply_shadow_lift(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return arr
    return np.power(arr, 1.0 / (1.0 + strength))


def _apply_split_tone(arr: np.ndarray, *, highlight_cool: float, shadow_warm: float) -> None:
    if highlight_cool <= 0 and shadow_warm <= 0:
        return

    lum = _luminance(arr)
    if highlight_cool > 0:
        hi_mask = np.clip((lum - 0.55) / 0.4, 0.0, 1.0)[..., None]
        arr[..., 0] *= 1.0 - highlight_cool * hi_mask
        arr[..., 2] *= 1.0 + highlight_cool * hi_mask
    if shadow_warm > 0:
        lo_mask = np.clip((0.4 - lum) / 0.4, 0.0, 1.0)[..., None]
        arr[..., 0] *= 1.0 + shadow_warm * lo_mask
        arr[..., 2] *= 1.0 - shadow_warm * lo_mask


def _apply_temperature(arr: np.ndarray, shift: float) -> np.ndarray:
    if shift == 0:
        return arr

    red_scale = 1.0 + shift
    blue_scale = 1.0 - shift
    arr[..., 0] *= red_scale
    arr[..., 2] *= blue_scale
    return arr


def _apply_contact_shadows(arr: np.ndarray, strength: float, radius: int) -> np.ndarray:
    if strength <= 0:
        return arr

    lum = _luminance(arr)
    pil_lum = Image.fromarray((np.clip(lum, 0.0, 1.0) * 255).astype(np.uint8))
    blurred = np.asarray(pil_lum.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0
    ao = np.clip(blurred - lum, 0.0, 1.0)[..., None]
    arr *= 1.0 - strength * ao
    return arr


def _apply_haze(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    lum = _luminance(arr)[..., None]
    haze = np.clip(1.0 - lum, 0.0, 1.0)
    return np.clip(arr * (1.0 - amount) + haze * amount, 0.0, 1.0)


def _apply_saturation(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Color(pil_img).enhance(factor)


def _apply_contrast(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Contrast(pil_img).enhance(factor)


def _apply_clarity(pil_img: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return pil_img
    percent = int(150 * amount)
    return pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))


def _apply_sharpness(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Sharpness(pil_img).enhance(factor)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderRecipe:
    highlight_compression: float = 0.0
    shadow_lift: float = 0.0
    temperature_shift: float = 0.0
    highlight_cool: float = 0.0
    shadow_warm: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0
    clarity: float = 0.0
    sharpness: float = 1.0
    contact_shadow: float = 0.0
    contact_radius: int = 5
    haze: float = 0.0


BASE_RECIPES: Dict[str, RenderRecipe] = {
    "aerial": RenderRecipe(
        highlight_compression=0.35,
        temperature_shift=0.06,
        saturation=1.02,
        clarity=0.1,
        contact_shadow=0.06,
        haze=0.05,
    ),
    "greatroom": RenderRecipe(
        highlight_compression=0.15,
        shadow_lift=0.05,
        temperature_shift=-0.03,
        highlight_cool=0.08,
        shadow_warm=0.04,
        saturation=1.1,
        contrast=1.05,
        clarity=0.18,
        sharpness=1.05,
        contact_shadow=0.08,
    ),
    "kitchen": RenderRecipe(
        highlight_compression=0.12,
        shadow_lift=0.04,
        temperature_shift=-0.05,
        highlight_cool=0.07,
        shadow_warm=0.05,
        saturation=1.08,
        contrast=1.07,
        clarity=0.22,
        sharpness=1.18,
        contact_shadow=0.12,
    ),
    "pool": RenderRecipe(
        highlight_compression=0.08,
        temperature_shift=0.1,
        saturation=1.05,
        contrast=1.03,
        clarity=0.12,
        sharpness=1.04,
        contact_shadow=0.1,
        contact_radius=6,
    ),
    "primarybedroom": RenderRecipe(
        highlight_compression=0.1,
        shadow_lift=0.2,
        temperature_shift=-0.04,
        highlight_cool=0.1,
        shadow_warm=0.03,
        saturation=1.06,
        contrast=1.04,
        clarity=0.16,
        sharpness=1.06,
        contact_shadow=0.09,
        contact_radius=7,
    ),
}


def _match_recipe(path: Path) -> RenderRecipe:
    stem = path.stem.lower()
    for key, recipe in BASE_RECIPES.items():
        if key in stem:
            return recipe
    raise KeyError(f"Unable to match render '{path.name}' to a configured recipe")


# ---------------------------------------------------------------------------
# Processing entry point
# ---------------------------------------------------------------------------


def process_render(path: Path, output_path: Path, recipe: RenderRecipe | None = None) -> None:
    if recipe is None:
        recipe = _match_recipe(path)
    arr = _load_rgb(path)

    arr = _apply_highlight_compression(arr, recipe.highlight_compression)
    arr = _apply_shadow_lift(arr, recipe.shadow_lift)
    _apply_temperature(arr, recipe.temperature_shift)
    _apply_split_tone(arr, highlight_cool=recipe.highlight_cool, shadow_warm=recipe.shadow_warm)
    _apply_contact_shadows(arr, recipe.contact_shadow, recipe.contact_radius)
    arr = _apply_haze(arr, recipe.haze)

    pil_img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    pil_img = _apply_saturation(pil_img, recipe.saturation)
    pil_img = _apply_contrast(pil_img, recipe.contrast)
    pil_img = _apply_clarity(pil_img, recipe.clarity)
    pil_img = _apply_sharpness(pil_img, recipe.sharpness)

    final = np.asarray(pil_img, dtype=np.float32) / 255.0
    _save_rgb(final, output_path)


def process_directory(input_dir: Path, output_dir: Path) -> None:
    normalized_input = ensure_supported_renderings(input_dir)

    for path in sorted(normalized_input.glob("*")):
        if path.suffix.lower() not in (SUPPORTED_IMAGE_SUFFIXES | CONVERTIBLE_IMAGE_SUFFIXES):
            continue
        try:
            recipe = _match_recipe(path)
        except KeyError as exc:  # pragma: no cover - defensive guard
            print(f"Skipping {path.name}: {exc}")
            continue
        out_name = f"{path.stem}_graded{path.suffix.lower()}"
        output_path = output_dir / out_name
        process_render(path, output_path, recipe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the 750px rendering review set")
    parser.add_argument("--input", type=Path, default=Path("/Users/rc/input_renderings_750"), help="Input directory")
    parser.add_argument("--output", type=Path, default=Path("/Users/rc/output_renderings_750"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_directory(args.input, args.output)


if __name__ == "__main__":
    main()
