"""Deterministic RGB image proxies for advisory VLM inference."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ProxyFormat = Literal["png", "jpeg"]


@dataclass(frozen=True)
class VLMImageProxy:
    """Metadata for a deterministic VLM input proxy."""

    source_path: Path
    proxy_path: Path
    source_sha256: str
    proxy_sha256: str
    width: int
    height: int
    mode: str
    format: str
    max_side_px: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["proxy_path"] = str(self.proxy_path)
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "image"


def _normalize_format(format_value: str) -> ProxyFormat:
    normalized = str(format_value).strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpeg"
    if normalized == "png":
        return "png"
    raise ValueError("format must be 'png' or 'jpeg'")


def _proxy_suffix(format_value: ProxyFormat) -> str:
    return ".jpg" if format_value == "jpeg" else ".png"


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert a decoded image to 8-bit RGB without clipping 16-bit integer data.

    Pillow's ``convert("RGB")`` clips ``I;16`` samples at 255, so any 16-bit
    grayscale master (TIFF or PNG) became a white proxy. ``I;16`` modes are
    unsigned with a 65535 full scale, which maps exactly onto 0..255.
    """
    if image.mode.startswith("I;16"):
        samples = np.asarray(image).astype(np.uint32)
        scaled = ((samples * 255 + 32767) // 65535).astype(np.uint8)
        return Image.fromarray(scaled).convert("RGB")
    return image.convert("RGB")


def build_vlm_image_proxy(
    source_path: Path,
    output_dir: Path,
    *,
    max_side_px: int = 1600,
    format: ProxyFormat = "png",
    jpeg_quality: int = 92,
    output_name: str | None = None,
) -> VLMImageProxy:
    """Build a deterministic 8-bit RGB proxy for VLM inference.

    Args:
        source_path: Pillow-readable source image.
        output_dir: Destination directory for the generated proxy.
        max_side_px: Longest proxy side in pixels.
        format: Proxy container format, ``png`` by default.
        jpeg_quality: JPEG quality when ``format='jpeg'``.
        output_name: Optional explicit filename for callers with a fixed output
            contract. Omit for hash-derived deterministic filenames.

    Returns:
        Proxy metadata including source and proxy hashes.

    Raises:
        FileNotFoundError: Source image is missing.
        ValueError: Image format or sizing arguments are invalid.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"VLM source image not found: {source}")
    if not source.is_file():
        raise ValueError(f"VLM source image is not a file: {source}")
    if max_side_px < 1:
        raise ValueError("max_side_px must be greater than zero")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    proxy_format = _normalize_format(format)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source)
    if output_name is not None:
        candidate_name = Path(output_name).name
        if not candidate_name:
            raise ValueError("output_name must not be empty")
        proxy_path = output_dir / candidate_name
    else:
        proxy_path = output_dir / f"{_safe_stem(source)}_{source_sha256[:12]}_proxy{_proxy_suffix(proxy_format)}"

    try:
        with Image.open(source) as image:
            rgb = _to_rgb(ImageOps.exif_transpose(image))
            if max(rgb.size) > max_side_px:
                rgb.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)
            save_kwargs: dict[str, object] = {}
            if proxy_format == "png":
                save_kwargs.update({"format": "PNG", "compress_level": 6})
            else:
                save_kwargs.update(
                    {
                        "format": "JPEG",
                        "quality": int(jpeg_quality),
                        "subsampling": 0,
                        "progressive": False,
                    }
                )
            rgb.save(proxy_path, **save_kwargs)
            width, height = rgb.size
            mode = rgb.mode
    except UnidentifiedImageError as exc:
        raise ValueError(f"VLM source image is not Pillow-readable: {source}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to build VLM image proxy for {source}: {exc}") from exc

    return VLMImageProxy(
        source_path=source,
        proxy_path=proxy_path,
        source_sha256=source_sha256,
        proxy_sha256=_sha256_file(proxy_path),
        width=width,
        height=height,
        mode=mode,
        format=proxy_format,
        max_side_px=max_side_px,
    )
