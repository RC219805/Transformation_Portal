"""Fidelity-preserving ingest, model proxies, finishing and delivery for V4.

Color preparation preserves source precision and converts supported RGB/gray
ICC encodings into extended linear sRGB. Profile descriptions never authorize
color transforms; untagged assumptions require an explicit input policy.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np
from PIL import Image

from transformation_portal.core.depth_artifact import DepthArtifact
from transformation_portal.core.image_artifact import ImageMaster, ImageProxy, ProxyTransform, metadata_payload
from transformation_portal.lux_depth_v4.color_preparation import (
    MAX_ICC_BYTES,
    InputColorError,
    PreparedColor,
    apply_input_color,
    linear_to_srgb,
    output_srgb_icc,
    prepare_input_color,
    srgb_to_linear,
)

RAW_SUFFIXES = frozenset({".arw", ".cr2", ".cr3", ".dng", ".nef", ".nrw", ".orf", ".raf", ".raw", ".rw2"})
RawDecoder = Callable[[bytes, str], tuple[np.ndarray, Mapping[str, Any]]]


def _orient(array: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 1:
        return array
    if orientation == 2:
        return np.flip(array, axis=1)
    if orientation == 3:
        return np.rot90(array, 2)
    if orientation == 4:
        return np.flip(array, axis=0)
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, -1)
    if orientation == 7:
        return np.flip(np.swapaxes(array, 0, 1), axis=(0, 1))
    if orientation == 8:
        return np.rot90(array, 1)
    raise ValueError("EXIF orientation must be an integer from 1 to 8")


def _resolve_color(
    *, input_color: str, image_format: str, profile: bytes | None, declared_color: str | None, exif_color: Any
) -> tuple[str, str]:
    prepared = prepare_input_color(
        input_color=input_color,
        image_format=image_format,
        profile=profile,
        declared_color=declared_color,
        exif_color=exif_color,
    )
    return prepared.source_color, prepared.resolution


def _tiff_color_metadata(page: Any, *, input_color: str) -> tuple[bytes | None, str | None, Any]:
    icc_tag = page.tags.get(34675)
    # These TIFF fields carry source calibration, even without an ICC. Until
    # that representation has a supported transform, it is not an untagged
    # image eligible for an sRGB assumption. Explicit corrections stay explicit.
    if (
        icc_tag is None
        and input_color in {"auto", "auto_assume_srgb"}
        and any(tag in page.tags for tag in (301, 318, 319, 532))
    ):
        raise InputColorError(
            "input_color_unsupported_metadata",
            "Unsupported TIFF calibration metadata without an ICC profile. Export a color-managed copy with an embedded "
            "RGB or grayscale ICC profile, or select a known source color explicitly.",
        )
    if icc_tag is not None and icc_tag.valuebytecount > MAX_ICC_BYTES:
        raise InputColorError("input_color_unsupported_icc", "ICC profile exceeds the supported 4 MiB limit.")
    profile = bytes(icc_tag.value) if icc_tag is not None else None
    try:
        description = json.loads(page.description) if page.description else {}
    except (ValueError, TypeError):
        description = {}
    declared_color = description.get("color_space") if isinstance(description, dict) else None
    exif_tag = page.tags.get(34665)
    exif = exif_tag.value if exif_tag is not None else {}
    if not isinstance(exif, Mapping):
        raise InputColorError("input_color_unsupported_metadata", "Invalid TIFF EXIF color metadata.")
    exif_color = exif.get("ColorSpace", exif.get(40961))
    return profile, declared_color, exif_color


def _jpeg_icc_profile(image: Image.Image, *, input_color: str) -> bytes | None:
    """Assemble bounded APP2 ICC fragments without Pillow's untagged fallback.

    Pillow discards an incomplete fragment set and only assembles profiles seen
    before SOF. Inspect all APP markers exposed by header parsing so malformed
    ICCs never authorize an assumption, and valid profiles before SOS survive.
    """
    fragments: list[bytes] = []
    total_bytes = 0
    for marker, payload in getattr(image, "applist", ()):
        if marker != "APP2" or not payload.startswith(b"ICC_PROFILE"):
            continue
        total_bytes += max(0, len(payload) - 14)
        if total_bytes > MAX_ICC_BYTES or len(fragments) >= 255:
            raise InputColorError("input_color_unsupported_icc", "JPEG ICC profile exceeds supported fragment bounds.")
        fragments.append(payload)
    if not fragments:
        return image.info.get("icc_profile")
    valid = all(len(fragment) >= 14 and fragment.startswith(b"ICC_PROFILE\0") for fragment in fragments)
    if valid:
        count = fragments[0][13]
        valid = (
            count == len(fragments)
            and all(fragment[13] == count for fragment in fragments)
            and {fragment[12] for fragment in fragments} == set(range(1, count + 1))
        )
    if not valid:
        if input_color in {"srgb", "linear_srgb"}:
            # Preserve the existing deliberate correction and retained profile
            # bytes, if Pillow could assemble any, without inventing a profile.
            return image.info.get("icc_profile")
        raise InputColorError(
            "input_color_unsupported_icc",
            "Invalid JPEG ICC fragments. Re-export the original with a complete embedded profile, "
            "or select a known source color explicitly.",
        )
    return b"".join(fragment[14:] for fragment in sorted(fragments, key=lambda fragment: fragment[12]))


def _standard_color_metadata(
    image: Image.Image, *, input_color: str, exif: Image.Exif | None = None
) -> tuple[bytes | None, str | None, Any]:
    if exif is None:
        exif = image.getexif()
    exif_color = exif.get(40961)
    if 34665 in exif:
        exif_color = exif.get_ifd(34665).get(40961, exif_color)
    declared_color = "srgb" if image.format == "PNG" and "srgb" in image.info else None
    profile = _jpeg_icc_profile(image, input_color=input_color) if image.format == "JPEG" else image.info.get("icc_profile")
    return profile, declared_color, exif_color


def _png_color_metadata(source: BinaryIO, image: Image.Image, *, input_color: str) -> tuple[Image.Exif, dict[str, Any]]:
    """Inspect bounded color chunks, including late EXIF, without raster reads.

    Compressed IDAT payloads are skipped. Color chunks have independent CRC,
    length, duplication and placement checks so malformed metadata cannot be
    mistaken for an untagged image by either preview or execution.
    """
    source.seek(8)
    seen: set[bytes] = set()
    metadata: dict[str, Any] = {}
    total = 0
    seen_pixels = False
    color_chunks = {b"IHDR", b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"cICP", b"eXIf", b"tRNS", b"mDCV", b"cLLI"}
    lengths = {b"IHDR": 13, b"sRGB": 1, b"gAMA": 4, b"cHRM": 32, b"cICP": 4, b"mDCV": 24, b"cLLI": 8}
    for _ in range(8192):
        header = source.read(8)
        if len(header) != 8:
            raise ValueError("Incomplete PNG metadata")
        size, kind = struct.unpack(">I4s", header)
        if size > 0x7FFFFFFF:
            raise ValueError("Invalid PNG chunk length")
        if kind == b"IEND":
            if size != 0 or len(source.read(4)) != 4 or not seen_pixels:
                raise ValueError("Incomplete PNG metadata")
            break
        if kind == b"IDAT":
            seen_pixels = True
        if kind in color_chunks:
            if kind in seen or (seen_pixels and kind != b"eXIf"):
                raise InputColorError("input_color_unsupported_metadata", "Duplicate or misplaced PNG color metadata.")
            if size > MAX_ICC_BYTES or total + size > MAX_ICC_BYTES or (kind in lengths and size != lengths[kind]):
                raise InputColorError("input_color_unsupported_metadata", "PNG color metadata exceeds supported bounds.")
            seen.add(kind)
            total += size
            data, checksum = source.read(size), source.read(4)
            if len(data) != size or len(checksum) != 4 or zlib.crc32(kind + data) != struct.unpack(">I", checksum)[0]:
                raise InputColorError("input_color_unsupported_metadata", "Invalid PNG color metadata checksum.")
            if kind == b"IHDR":
                metadata["bit_depth"] = data[8]
                metadata["color_type"] = data[9]
            elif kind == b"sRGB":
                metadata["srgb"] = data[0]
            elif kind == b"gAMA":
                metadata["gamma"] = struct.unpack(">I", data)[0] / 100000.0
            elif kind == b"cHRM":
                metadata["chromaticity"] = tuple(value / 100000.0 for value in struct.unpack(">8I", data))
            elif kind == b"cICP":
                metadata["cicp"] = data
            elif kind == b"eXIf":
                image.info["exif"] = b"Exif\x00\x00" + data
            elif kind == b"iCCP" and not image.info.get("icc_profile"):
                raise InputColorError("input_color_unsupported_icc", "Invalid PNG ICC profile metadata.")
            elif kind in {b"mDCV", b"cLLI"} and input_color in {"auto", "auto_assume_srgb"}:
                raise InputColorError("input_color_unsupported_metadata", "Unsupported PNG HDR color metadata.")
        else:
            source.seek(size, io.SEEK_CUR)
            if len(source.read(4)) != 4:
                raise ValueError("Incomplete PNG metadata")
    else:
        raise ValueError("PNG metadata exceeds the preflight chunk budget")
    if metadata.get("color_type") not in {0, 2, 4, 6} or metadata.get("bit_depth") not in {8, 16}:
        raise ValueError("V4 PNG ingest requires 8-bit or 16-bit RGB/gray samples")
    if metadata["bit_depth"] == 16:
        _png_precision_runtime()
    return Image.Image.getexif(image), metadata


def _png_precision_runtime() -> Any:
    try:
        import imagecodecs
    except ImportError as exc:
        raise ValueError("16-bit PNG ingest requires the governed imagecodecs PNG runtime") from exc
    if not imagecodecs.PNG.available:
        raise ValueError("16-bit PNG ingest requires the governed imagecodecs PNG runtime")
    return imagecodecs


def _validate_color_channels(prepared: PreparedColor, *, grayscale: bool) -> None:
    """Reject profile/sample channel mismatches using headers alone."""
    if prepared.action == "explicit" or prepared.source_profile is None:
        return
    profile_gray = prepared.source_profile[16:20] == b"GRAY"
    if profile_gray != grayscale:
        raise InputColorError("input_color_conflict", "ICC channel layout does not match the photographic samples")


def validate_input_color_metadata(
    source: BinaryIO, *, source_name: str, input_color: str = "auto", max_pixels: int = 100_000_000
) -> dict[str, Any]:
    """Resolve ingest color from confined, bounded headers without pixel decode."""
    if Path(source_name).suffix.lower() in RAW_SUFFIXES:
        return prepare_input_color(
            input_color=input_color,
            image_format="RAW",
            profile=None,
            declared_color=None,
            exif_color=None,
        ).evidence
    signature = source.read(4)
    source.seek(0)
    png_metadata = None
    if signature in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        import tifffile

        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ValueError("V4 photographic TIFF requires a complete image page")
            _validate_tiff_page(tif, page)
            height, width = int(page.imagelength), int(page.imagewidth)
            profile, declared_color, exif_color = _tiff_color_metadata(page, input_color=input_color)
            grayscale = page.photometric == tifffile.PHOTOMETRIC.MINISBLACK
            image_format = "TIFF"
    else:
        with Image.open(source) as image:
            height, width = image.height, image.width
            image_format = str(image.format)
            if image_format not in {"JPEG", "PNG"}:
                raise ValueError("V4 standard ingest supports RGB/gray JPEG, PNG and TIFF")
            if image_format == "PNG":
                exif, png_metadata = _png_color_metadata(source, image, input_color=input_color)
                grayscale = png_metadata["color_type"] in {0, 4}
            else:
                if image.mode not in {"RGB", "L"}:
                    raise ValueError("V4 standard ingest supports RGB/gray JPEG, PNG and TIFF")
                exif, grayscale = image.getexif(), image.mode == "L"
            profile, declared_color, exif_color = _standard_color_metadata(image, input_color=input_color, exif=exif)
    if height <= 0 or width <= 0 or height * width > max_pixels:
        raise ValueError("Decoded image dimensions exceed max_pixels")
    prepared = prepare_input_color(
        input_color=input_color,
        image_format=image_format,
        profile=profile,
        declared_color=declared_color,
        exif_color=exif_color,
        png_metadata=png_metadata,
    )
    _validate_color_channels(prepared, grayscale=grayscale)
    return prepared.evidence


def _validate_tiff_page(tif: Any, page: Any) -> None:
    """Share TIFF header geometry/precision policy between preview and decode."""
    import tifffile

    if len(tif.pages) != 1 or len(tif.series) != 1:
        raise ValueError("V4 photographic TIFF ingest requires one image page")
    if page.samplesperpixel not in {1, 2, 3, 4}:
        raise ValueError("Unsupported TIFF channel count")
    if page.photometric not in {tifffile.PHOTOMETRIC.RGB, tifffile.PHOTOMETRIC.MINISBLACK}:
        raise ValueError("Unsupported TIFF photometric interpretation")
    if (page.photometric == tifffile.PHOTOMETRIC.RGB) != (page.samplesperpixel in {3, 4}):
        raise ValueError("TIFF sample layout does not match its photometric interpretation")
    if page.planarconfig not in {None, 1}:
        raise ValueError("Planar TIFF input requires an explicit external conversion")
    expected_shape: tuple[int, ...] = (int(page.imagelength), int(page.imagewidth))
    if page.samplesperpixel != 1:
        expected_shape += (int(page.samplesperpixel),)
    if tuple(page.shape) != expected_shape:
        raise ValueError("Volumetric or non-photographic TIFF sample geometry is unsupported")
    if page.dtype not in {np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")}:
        raise ValueError("Photographic samples must be uint8, uint16 or float32")
    if page.bitspersample != page.dtype.itemsize * 8:
        raise ValueError("TIFF BitsPerSample must match supported 8, 16 or 32-bit sample precision")
    if page.extrasamples and any(int(sample) != 2 for sample in page.extrasamples):
        raise ValueError("Only unassociated (straight) TIFF alpha is supported")


def decode_master(
    source_bytes: bytes,
    *,
    source_name: str,
    input_color: str = "auto",
    raw_decoder: RawDecoder | None = None,
    max_pixels: int = 100_000_000,
) -> ImageMaster:
    """Decode one immutable byte snapshot into oriented high-precision pixels.

    RAW is delegated to a governed decoder supplied by execution preparation. That
    callback must return oriented linear-sRGB pixels plus its ingest evidence.
    """
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("Source snapshot must be non-empty immutable bytes")
    if input_color not in {"auto", "auto_assume_srgb", "srgb", "linear_srgb"}:
        raise ValueError("input_color must be auto, auto_assume_srgb, srgb, or linear_srgb")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")

    def check_dimensions(height: int, width: int) -> None:
        if height <= 0 or width <= 0 or height * width > max_pixels:
            raise ValueError("Decoded image dimensions exceed max_pixels")

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if Path(source_name).suffix.lower() in RAW_SUFFIXES:
        if raw_decoder is None:
            raise ValueError("RAW input requires a governed decoder")
        prepared = prepare_input_color(
            input_color=input_color,
            image_format="RAW",
            profile=None,
            declared_color=None,
            exif_color=None,
        )
        pixels, raw_metadata = raw_decoder(source_bytes, source_name)
        if pixels.ndim != 3:
            raise ValueError("Governed RAW decoder must return HxWx3 pixels")
        check_dimensions(*pixels.shape[:2])
        if str(raw_metadata.get("color_space", "")).lower() != "linear_srgb":
            raise ValueError("Governed RAW decoder must declare linear_srgb")
        metadata = dict(raw_metadata)
        metadata.update(
            {
                "source_format": "RAW",
                "input_color": "linear_srgb",
                "color_resolution": "governed_raw_decoder",
                "color_preparation": prepared.evidence,
            }
        )
        return ImageMaster(pixels, source_sha256, int(raw_metadata.get("source_bit_depth", 16)), metadata=metadata)

    profile = None
    declared_color = None
    exif_color = None
    orientation = 1
    png_metadata = None
    is_tiff = source_bytes[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}
    if is_tiff:
        import tifffile

        with tifffile.TiffFile(io.BytesIO(source_bytes)) as tif:
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ValueError("V4 photographic TIFF requires a complete image page")
            check_dimensions(int(page.imagelength), int(page.imagewidth))
            _validate_tiff_page(tif, page)
            orientation_tag = page.tags.get(274)
            orientation = int(orientation_tag.value) if orientation_tag else 1
            profile, declared_color, exif_color = _tiff_color_metadata(page, input_color=input_color)
            image_format = "TIFF"
            prepared = prepare_input_color(
                input_color=input_color,
                image_format=image_format,
                profile=profile,
                declared_color=declared_color,
                exif_color=exif_color,
            )
            _validate_color_channels(prepared, grayscale=page.photometric == tifffile.PHOTOMETRIC.MINISBLACK)
            pixels = page.asarray()
    else:
        source = io.BytesIO(source_bytes)
        with Image.open(source) as image:
            check_dimensions(image.height, image.width)
            image_format = str(image.format)
            if image_format not in {"JPEG", "PNG"}:
                raise ValueError("V4 standard ingest supports RGB/gray JPEG, PNG and TIFF")
            if image_format == "PNG":
                exif, png_metadata = _png_color_metadata(source, image, input_color=input_color)
                grayscale = png_metadata["color_type"] in {0, 4}
            else:
                if image.mode not in {"RGB", "L"}:
                    raise ValueError("V4 standard ingest supports RGB/gray JPEG, PNG and TIFF")
                exif, grayscale = image.getexif(), image.mode == "L"
            orientation = int(exif.get(274, 1))
            profile, declared_color, exif_color = _standard_color_metadata(image, input_color=input_color, exif=exif)
            prepared = prepare_input_color(
                input_color=input_color,
                image_format=image_format,
                profile=profile,
                declared_color=declared_color,
                exif_color=exif_color,
                png_metadata=png_metadata,
            )
            _validate_color_channels(prepared, grayscale=grayscale)
            if png_metadata is not None and png_metadata["bit_depth"] == 16:
                pixels = _png_precision_runtime().png_decode(source_bytes)
                if pixels.dtype != np.uint16 or pixels.shape[:2] != (image.height, image.width):
                    raise ValueError("PNG decoder did not preserve 16-bit sample precision and geometry")
            elif image_format == "PNG" and "transparency" in image.info:
                if grayscale:
                    gray = np.asarray(image)
                    transparency_alpha = np.where(gray == image.info["transparency"], 0, 255).astype(np.uint8)
                    pixels = np.stack((gray, transparency_alpha), axis=-1)
                else:
                    pixels = np.asarray(image.convert("RGBA"))
            else:
                pixels = np.asarray(image)

    pixels = _orient(pixels, orientation)
    if pixels.dtype not in {np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")}:
        raise ValueError("Photographic samples must be uint8, uint16 or float32")
    bits = pixels.dtype.itemsize * 8
    divisor = float(np.iinfo(pixels.dtype).max) if np.issubdtype(pixels.dtype, np.integer) else 1.0
    values = pixels.astype(np.float32) / divisor
    if values.ndim == 2:
        samples, alpha = values, None
    elif values.ndim == 3 and values.shape[2] in {2, 3, 4}:
        alpha = values[..., -1] if values.shape[2] in {2, 4} else None
        samples = values[..., 0] if values.shape[2] == 2 else values[..., :3]
    else:
        raise ValueError("Unsupported photographic sample shape")
    linear = apply_input_color(samples, prepared)
    return ImageMaster(
        linear,
        source_sha256,
        bits,
        alpha=alpha,
        source_icc=profile,
        metadata={
            "source_format": image_format,
            "input_color": prepared.source_color,
            "color_resolution": prepared.resolution,
            "color_preparation": prepared.evidence,
            "source_orientation": orientation,
            "orientation_normalized": True,
        },
    )


def _resize_float(array: np.ndarray, shape: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    if tuple(array.shape[:2]) == shape:
        return np.asarray(array, dtype=np.float32).copy()
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    size = (shape[1], shape[0])
    if array.ndim == 2:
        return np.asarray(Image.fromarray(np.asarray(array, dtype=np.float32)).resize(size, method), dtype=np.float32)
    return np.stack([_resize_float(array[..., channel], shape, nearest=nearest) for channel in range(array.shape[2])], axis=2)


def create_proxy(master: ImageMaster, target_size: int = 518) -> ImageProxy:
    """Render a bounded sRGB proxy; downsample, then pad without cropping."""
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size < 14:
        raise ValueError("Proxy target_size must be an integer >=14")
    height, width = master.shape
    scale = min(1.0, target_size / max(height, width))
    resized_shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    resized = _resize_float(master.pixels, resized_shape)
    encoded = np.rint(np.clip(linear_to_srgb(resized), 0, 1) * 255).astype(np.uint8)
    padded_shape = (((resized_shape[0] + 13) // 14) * 14, ((resized_shape[1] + 13) // 14) * 14)
    padded = np.pad(
        encoded, ((0, padded_shape[0] - resized_shape[0]), (0, padded_shape[1] - resized_shape[1]), (0, 0)), mode="edge"
    )
    return ImageProxy(padded, ProxyTransform(master.shape, resized_shape, padded_shape), master.content_hash())


def restore_depth(depth: np.ndarray, transform: ProxyTransform) -> np.ndarray:
    """Return model depth to master geometry using the recorded proxy mapping."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("Restored depth must start from finite non-empty HW model output")
    on_proxy = _resize_float(values, transform.padded_shape)
    unpadded = on_proxy[: transform.resized_shape[0], : transform.resized_shape[1]]
    return _resize_float(unpadded, transform.original_shape)


def restore_mask(mask: np.ndarray, transform: ProxyTransform) -> np.ndarray:
    """Restore a discrete model mask without introducing interpolated labels."""
    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("Restored masks must be finite non-empty HW arrays")
    on_proxy = _resize_float(values, transform.padded_shape, nearest=True)
    unpadded = on_proxy[: transform.resized_shape[0], : transform.resized_shape[1]]
    return _resize_float(unpadded, transform.original_shape, nearest=True)


def restore_depth_with_validity(
    depth: np.ndarray, valid_mask: np.ndarray, transform: ProxyTransform
) -> tuple[np.ndarray, np.ndarray]:
    """Remap valid depth without blending invalid samples into neighboring pixels.

    Nearest-neighbor validity retains explicit holes. Bilinear values are
    normalized by valid support, and invalid destinations contain zero only as
    a sentinel; consumers must carry the returned mask with the derivative.
    """
    values = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid_mask)
    if (
        values.ndim != 2
        or min(values.shape) <= 0
        or valid.dtype != np.bool_
        or valid.shape != values.shape
        or not np.isfinite(values[valid]).all()
    ):
        raise ValueError("Depth restoration requires finite valid HW samples and matching boolean validity")
    if valid.all():
        # Preserve the established interpolation exactly for all-valid output.
        return restore_depth(values, transform), np.ones(transform.original_shape, dtype=bool)
    support = restore_depth(valid.astype(np.float32), transform)
    numerator = restore_depth(np.where(valid, values, 0), transform)
    aligned_valid = restore_mask(valid, transform).astype(bool) & (support > 0)
    aligned = np.zeros(transform.original_shape, dtype=np.float32)
    np.divide(numerator, support, out=aligned, where=aligned_valid)
    return aligned, aligned_valid


def enhance_master(
    master: ImageMaster,
    depth: np.ndarray | DepthArtifact | None = None,
    *,
    strength: float = 0.25,
    clarity: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> ImageMaster:
    """Apply bounded depth exposure in linear light and optional float clarity."""
    if not np.isfinite([strength, clarity]).all() or not 0 <= strength <= 1 or not 0 <= clarity <= 1:
        raise ValueError("Enhancement strength and clarity must be finite in [0,1]")
    if valid_mask is not None:
        if depth is None or isinstance(depth, DepthArtifact):
            raise ValueError("Explicit validity requires a depth array without competing artifact validity")
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != master.shape:
            raise ValueError("Finishing validity must be a boolean mask matching master geometry")
    pixels = master.pixels.copy()
    depth_applied = False
    if depth is not None and strength:
        valid = (
            depth.valid_mask
            if isinstance(depth, DepthArtifact)
            else valid_mask if valid_mask is not None else np.ones(master.shape, dtype=bool)
        )
        relative = depth.relative_depth() if isinstance(depth, DepthArtifact) else np.asarray(depth, dtype=np.float32)
        if relative.shape != master.shape or not np.isfinite(relative[valid]).all():
            raise ValueError("Finishing depth must match the master geometry and be finite")
        if np.any((relative[valid] < 0) | (relative[valid] > 1)):
            raise ValueError("Finishing depth must be near=0/far=1 relative depth")
        if valid.any() and np.ptp(relative[valid]) > 1e-6:
            from scipy.ndimage import gaussian_filter

            # Conservative lowpass avoids transferring model striping into
            # smooth photographic regions. Invalid pixels receive unity gain.
            weight = gaussian_filter(valid.astype(np.float32), sigma=2.0)
            smooth = gaussian_filter(np.where(valid, relative, 0), sigma=2.0) / np.maximum(weight, 1e-6)
            center = float(np.percentile(relative[valid], 75))
            exposure = np.clip(center - smooth, -1, 1) * float(strength)
            gain = np.where(valid, np.exp2(exposure), 1.0)
            pixels *= gain[..., None]
            depth_applied = True
    if clarity:
        from scipy.ndimage import gaussian_filter

        encoded = linear_to_srgb(pixels)
        blurred = gaussian_filter(encoded, sigma=(1.0, 1.0, 0.0))
        pixels = srgb_to_linear(encoded + float(clarity) * 0.2 * (encoded - blurred))
    metadata = metadata_payload(master.metadata)
    metadata["finishing"] = {
        "version": "v4.1",
        "strength": float(strength),
        "clarity": float(clarity),
        "depth_applied": depth_applied,
    }
    return ImageMaster(pixels, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc)


def write_delivery(master: ImageMaster, path: Path) -> dict[str, Any]:
    """Encode one photographic 16-bit sRGB TIFF, quantizing only at delivery."""
    import tifffile

    from transformation_portal.lux_depth_v3.io_atomic import atomic_temp_file

    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("V4 photographic delivery path must end in .tif or .tiff")
    encoded = linear_to_srgb(master.pixels)
    low_fraction = float(np.mean(encoded < 0))
    high_fraction = float(np.mean(encoded > 1))
    samples = np.rint(np.clip(encoded, 0, 1) * 65535).astype(np.uint16)
    if master.alpha is not None:
        alpha = np.rint(master.alpha * 65535).astype(np.uint16)
        samples = np.concatenate([samples, alpha[..., None]], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_temp_file(path, suffix=".tif", create_file=False) as temporary:
        tifffile.imwrite(
            temporary,
            samples,
            photometric="rgb",
            extrasamples="unassalpha" if master.alpha is not None else None,
            metadata={"color_space": "srgb", "image_artifact_schema": "tp.image.master.v1"},
            iccprofile=output_srgb_icc(),
            extratags=[(274, "H", 1, 1, False)],
        )
    return {
        "path": str(path),
        "output_bit_depth": 16,
        "source_bit_depth": master.source_bit_depth,
        "color_space": "srgb",
        "alpha_preserved": master.alpha is not None,
        "clipped_low_sample_fraction": low_fraction,
        "clipped_high_sample_fraction": high_fraction,
        "master_content_hash": master.content_hash(),
        "output_icc_sha256": hashlib.sha256(output_srgb_icc()).hexdigest(),
    }


def apply_materials(
    master: ImageMaster,
    material_masks: Mapping[str, np.ndarray] | None = None,
    material_confidences: Mapping[str, float] | None = None,
    *,
    min_confidence: float = 0.6,
    min_coverage_px: int = 500,
) -> tuple[ImageMaster, dict[str, Any]]:
    """Apply existing pixel operations only to explicitly supplied evidence.

    This adapter does not infer segmentation or call supplied scores calibrated.
    Unknown, uncovered, and uncertain materials abstain. Malformed authority fails
    instead of silently disappearing. Timing telemetry is excluded from identity.
    """
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.materials_v3 import MaterialsV3Engine
    from transformation_portal.lux_depth_v3.materials_v3_taxonomy import DEFAULT_MATERIAL_METADATA
    from transformation_portal.lux_depth_v3.pixel_ops_registry import OP_REGISTRY

    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be finite in [0,1]")
    if isinstance(min_coverage_px, bool) or not isinstance(min_coverage_px, int) or min_coverage_px <= 0:
        raise ValueError("min_coverage_px must be a positive integer")
    if material_masks is not None and not isinstance(material_masks, Mapping):
        raise ValueError("Supplied material masks must be a mapping")
    if material_confidences is not None and not isinstance(material_confidences, Mapping):
        raise ValueError("Supplied material confidences must be a mapping")
    masks = material_masks or {}
    confidences = material_confidences or {}
    if any(not isinstance(name, str) or not name.strip() for name in masks):
        raise ValueError("Material names must be non-empty strings")
    if any(name not in masks for name in confidences):
        raise ValueError("Material confidence cannot authorize an absent mask")
    report: dict[str, Any] = {
        "schema": "tp.lux.materials.application.v1",
        "evidence_source": "caller_supplied_mask_and_confidence",
        "segmentation_inferred": False,
        "status": "abstained",
        "reason": "no_supplied_material_masks" if not masks else "no_eligible_materials",
        "materials": {},
    }
    eligible = {}
    for name in sorted(masks):
        mask = np.asarray(masks[name])
        if mask.shape != master.shape or mask.dtype.kind not in "buif":
            raise ValueError(f"Material mask {name!r} must be a numeric HW array matching the master")
        mask = mask.astype(np.float32)
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError(f"Material mask {name!r} must be finite in [0,1]")
        confidence = confidences.get(name)
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not np.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"Material confidence {name!r} must be finite in [0,1]")
        threshold = max(float(min_confidence), float(DEFAULT_MATERIAL_METADATA.get(name, {}).get("threshold", 0)))
        coverage = int(np.count_nonzero(mask > 0.5))
        entry: dict[str, Any] = {
            "status": "abstained",
            "confidence": float(confidence) if confidence is not None else None,
            "threshold": threshold,
            "coverage_px": coverage,
            "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        }
        if name not in OP_REGISTRY or not any(op.implemented for op in OP_REGISTRY[name].values()):
            entry["reason"] = "unsupported_material"
        elif confidence is None:
            entry["reason"] = "missing_confidence"
        elif confidence < threshold:
            entry["reason"] = "below_confidence_threshold"
        elif coverage < min_coverage_px:
            entry["reason"] = "below_coverage_threshold"
        else:
            eligible[name] = (mask, float(confidence))
            entry["reason"] = "pixel_ops_not_recommended"
        report["materials"][name] = entry
    if not eligible:
        return master, report

    encoded = linear_to_srgb(master.pixels)
    working = np.clip(encoded, 0, 1).astype(np.float32)
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        min_mean_conf=float(min_confidence),
        min_coverage_px=min_coverage_px,
    )
    result = MaterialsV3Engine(config).process(working, {"materials": eligible})
    applied = result.get("materials_v3_pixel_ops", {}).get("applied", [])
    for item in applied:
        name = item["material"]
        report["materials"][name].update(
            {"status": "applied", "reason": "supplied_evidence_accepted", "ops": list(item["ops"])}
        )
    if not applied:
        return master, report
    modified = np.asarray(result["enhanced_image"], dtype=np.float32)
    if modified.shape != master.pixels.shape or not np.isfinite(modified).all():
        raise ValueError("Material operations returned invalid photographic pixels")
    delta = modified - working
    # Preserve exact untouched master samples and retain unbounded headroom.
    changed = np.any(delta != 0, axis=2)
    pixels = master.pixels.copy()
    pixels[changed] = srgb_to_linear((encoded + delta)[changed])
    report.update({"status": "applied", "reason": "supplied_evidence_accepted", "pixels_changed": bool(changed.any())})
    metadata = metadata_payload(master.metadata)
    metadata["material_application"] = report
    return (
        ImageMaster(pixels, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc),
        report,
    )


def generate_preview_maps(depth: DepthArtifact) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate compatibility depth-gradient previews, not physical materials."""
    from transformation_portal.lux_depth_v3.pbr import generate_pbr_maps

    if not isinstance(depth, DepthArtifact):
        raise ValueError("Preview maps require a semantic DepthArtifact")
    normal, roughness, ao = generate_pbr_maps(depth.relative_depth())
    # Unknown depth must not appear as evaluated geometry/material evidence.
    normal[~depth.valid_mask] = [128, 128, 255]
    roughness[~depth.valid_mask] = 0
    ao[~depth.valid_mask] = 255
    return {"normal": normal, "roughness": roughness, "ao": ao}, {
        "schema": "tp.lux.preview_maps.v1",
        "kind": "depth_derived_preview",
        "physical_material_estimate": False,
        "depth_content_hash": depth.content_hash(),
        "valid_fraction": float(np.mean(depth.valid_mask)),
        "method": "legacy_minmax_sobel_laplacian_gradient",
    }
