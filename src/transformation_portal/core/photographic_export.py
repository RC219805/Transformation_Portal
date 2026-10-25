"""Shared photographic export primitives, independent of optional RAW runtimes.

Pixels remain finite float RGB until final delivery quantization. Color-space
conversion and ICC validation belong to the caller's explicit color authority.
The editorial CLI and managed delivery use these same resize/sharpen/JPEG
operations; importing them never loads RAW, PDF, model, or external runtimes.
"""

from __future__ import annotations

import io
from typing import Mapping
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def _rgb(samples: np.ndarray) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float32)
    if result.ndim != 3 or result.shape[-1] != 3 or min(result.shape[:2]) < 1 or not np.isfinite(result).all():
        raise ValueError("Expected finite non-empty HWC RGB samples")
    return result


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Editorial float-channel Lanczos reduction, preserving sub-8-bit detail."""
    pixels = _rgb(img)
    if type(long_edge) is not int or not 1 <= long_edge <= 32768:
        raise ValueError("long_edge must be an integer in [1,32768]")
    height, width = pixels.shape[:2]
    if max(height, width) <= long_edge:
        return pixels
    scale = long_edge / max(height, width)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = np.stack(
        [np.asarray(Image.fromarray(pixels[..., channel]).resize(size, Image.Resampling.LANCZOS)) for channel in range(3)],
        axis=-1,
    )
    return np.clip(resized, 0, 1).astype(np.float32)


def unsharp_mask(img: np.ndarray, amount: float = 0.2, radius: float = 1.2, threshold: float = 0.0) -> np.ndarray:
    """Existing editorial reflect-border Gaussian unsharp operation."""
    pixels = _rgb(img)
    if not all(np.isfinite(value) for value in (amount, radius, threshold)) or radius <= 0 or threshold < 0:
        raise ValueError("Sharpen controls must be finite, with positive radius and nonnegative threshold")
    if amount <= 0:
        return pixels
    low = gaussian_filter(pixels, sigma=(radius, radius, 0), mode="reflect")
    high = pixels - low
    return np.clip(pixels + amount * np.where(np.abs(high) > threshold, high, 0.0), 0, 1)


def export_metadata(metadata: Mapping[str, str] | None) -> dict[str, str]:
    """Bounded explicit editorial metadata; no copied private capture/GPS tags."""
    if metadata is None:
        return {}
    allowed = {"title", "description", "creator", "copyright", "credit", "keywords", "location"}
    if not isinstance(metadata, Mapping) or set(metadata) - allowed:
        raise ValueError("Unsupported photographic metadata field")
    result = {}
    for key, value in metadata.items():
        if not isinstance(value, str) or len(value) > 2048 or any(ord(char) < 32 for char in value):
            raise ValueError("Photographic metadata must contain bounded printable strings")
        result[key] = value
    return result


def jpeg_bytes(
    encoded_srgb: np.ndarray, *, icc_profile: bytes, quality: int = 96, metadata: Mapping[str, str] | None = None
) -> bytes:
    """Quantize explicitly bounded encoded-sRGB pixels once, with caller-validated ICC.

    The caller must already have converted primaries and transfer. This function
    deliberately refuses extended samples rather than clipping HDR silently.
    """
    pixels = _rgb(encoded_srgb)
    if np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("JPEG requires an explicit bounded SDR render before quantization")
    if type(quality) is not int or not 1 <= quality <= 100 or not isinstance(icc_profile, bytes) or not icc_profile:
        raise ValueError("JPEG requires quality in [1,100] and an explicit validated ICC profile")
    fields = export_metadata(metadata)
    exif = Image.Exif()
    for key, tag in (("description", 270), ("creator", 315), ("copyright", 33432)):
        if fields.get(key):
            exif[tag] = fields[key]
    if fields.get("title"):
        exif[40091] = fields["title"].encode("utf-16le") + b"\0\0"
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    buffer = io.BytesIO()
    options = {"exif": exif.tobytes()} if len(exif) else {}
    if fields:
        tags = {
            "title": "dc:title",
            "description": "dc:description",
            "creator": "dc:creator",
            "copyright": "dc:rights",
            "credit": "photoshop:Credit",
            "keywords": "dc:subject",
            "location": "Iptc4xmpCore:Location",
        }
        entries = "".join(f"<{tags[key]}>{escape(value)}</{tags[key]}>" for key, value in sorted(fields.items()))
        options["xmp"] = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
            'xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/">'
            + entries
            + "</rdf:Description></rdf:RDF></x:xmpmeta>"
        ).encode("utf-8")
    image.save(buffer, format="JPEG", quality=quality, subsampling=0, icc_profile=icc_profile, optimize=True, **options)
    return buffer.getvalue()
