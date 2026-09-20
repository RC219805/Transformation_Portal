"""Fidelity-preserving ingest, model proxies, finishing and delivery for V4.

Color support is intentionally bounded: encoded sRGB and linear sRGB. An
explicit input-color correction is required for unknown TIFFs or unsupported
profiles; profile descriptions are never trusted as color-transform authority.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageCms

from transformation_portal.core.depth_artifact import DepthArtifact
from transformation_portal.core.image_artifact import ImageMaster, ImageProxy, ProxyTransform, metadata_payload

RAW_SUFFIXES = frozenset({".arw", ".cr2", ".cr3", ".dng", ".nef", ".nrw", ".orf", ".raf", ".raw", ".rw2"})
RawDecoder = Callable[[bytes, str], tuple[np.ndarray, Mapping[str, Any]]]


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Exact sRGB inverse transfer; preserve finite negative/over-range values."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((np.maximum(values, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """sRGB display transfer without clipping or integer quantization."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055).astype(np.float32)


def _normalize_icc(profile: bytes) -> bytes:
    if len(profile) < 128:
        return profile
    normalized = bytearray(profile)
    # LittleCMS embeds creation time. Fix it for deterministic delivery; the
    # optional profile ID is cleared because the header bytes have changed.
    normalized[24:36] = bytes.fromhex("07d000010001000000000000")
    normalized[84:100] = bytes(16)
    return bytes(normalized)


def output_srgb_icc() -> bytes:
    """Return the deterministic built-in sRGB profile for delivery pixels."""
    return _normalize_icc(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


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
    if input_color not in {"auto", "srgb", "linear_srgb"}:
        raise ValueError("input_color must be auto, srgb, or linear_srgb")
    if input_color != "auto":
        return input_color, "explicit_input_color"
    if profile is not None:
        if _normalize_icc(profile) != output_srgb_icc():
            raise ValueError("Unsupported ICC profile; provide an explicit input_color correction")
        if declared_color not in {None, "srgb"}:
            raise ValueError("ICC and declared input color disagree")
        return "srgb", "recognized_srgb_icc"
    if exif_color not in {None, 1, 65535}:
        raise ValueError("Unsupported EXIF color space; provide an explicit input_color correction")
    if declared_color is not None:
        if declared_color not in {"srgb", "linear_srgb"}:
            raise ValueError("Unsupported declared color space; provide an explicit input_color correction")
        if declared_color == "linear_srgb" and exif_color == 1:
            raise ValueError("EXIF sRGB and declared linear input color disagree")
        return declared_color, "declared_input_color"
    if exif_color == 1:
        return "srgb", "exif_srgb"
    if image_format == "JPEG" and exif_color is None:
        return "srgb", "untagged_jpeg_srgb_assumption"
    raise ValueError("Ambiguous input color; provide input_color='srgb' or 'linear_srgb'")


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
    if input_color not in {"auto", "srgb", "linear_srgb"}:
        raise ValueError("input_color must be auto, srgb, or linear_srgb")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")

    def check_dimensions(height: int, width: int) -> None:
        if height <= 0 or width <= 0 or height * width > max_pixels:
            raise ValueError("Decoded image dimensions exceed max_pixels")

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if Path(source_name).suffix.lower() in RAW_SUFFIXES:
        if raw_decoder is None:
            raise ValueError("RAW input requires a governed decoder")
        if input_color not in {"auto", "linear_srgb"}:
            raise ValueError("RAW decoder output is linear_srgb; encoded input_color is invalid")
        pixels, raw_metadata = raw_decoder(source_bytes, source_name)
        if pixels.ndim != 3:
            raise ValueError("Governed RAW decoder must return HxWx3 pixels")
        check_dimensions(*pixels.shape[:2])
        if str(raw_metadata.get("color_space", "")).lower() != "linear_srgb":
            raise ValueError("Governed RAW decoder must declare linear_srgb")
        metadata = dict(raw_metadata)
        metadata.update({"source_format": "RAW", "input_color": "linear_srgb", "color_resolution": "governed_raw_decoder"})
        return ImageMaster(pixels, source_sha256, int(raw_metadata.get("source_bit_depth", 16)), metadata=metadata)

    profile = None
    declared_color = None
    exif_color = None
    orientation = 1
    is_tiff = source_bytes[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}
    if is_tiff:
        import tifffile

        with tifffile.TiffFile(io.BytesIO(source_bytes)) as tif:
            if len(tif.pages) != 1 or len(tif.series) != 1:
                raise ValueError("V4 photographic TIFF ingest requires one image page")
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ValueError("V4 photographic TIFF requires a complete image page")
            check_dimensions(int(page.imagelength), int(page.imagewidth))
            if page.samplesperpixel not in {1, 2, 3, 4}:
                raise ValueError("Unsupported TIFF channel count")
            if page.photometric not in {tifffile.PHOTOMETRIC.RGB, tifffile.PHOTOMETRIC.MINISBLACK}:
                raise ValueError("Unsupported TIFF photometric interpretation")
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
            pixels = page.asarray()
            if page.extrasamples and any(int(sample) != 2 for sample in page.extrasamples):
                raise ValueError("Only unassociated (straight) TIFF alpha is supported")
            orientation_tag = page.tags.get(274)
            orientation = int(orientation_tag.value) if orientation_tag else 1
            icc_tag = page.tags.get(34675)
            profile = bytes(icc_tag.value) if icc_tag is not None else None
            try:
                description = json.loads(page.description) if page.description else {}
            except (ValueError, TypeError):
                description = {}
            if isinstance(description, dict):
                declared_color = description.get("color_space")
            image_format = "TIFF"
    else:
        with Image.open(io.BytesIO(source_bytes)) as image:
            check_dimensions(image.height, image.width)
            image_format = str(image.format)
            if image_format not in {"JPEG", "PNG"} or image.mode not in {"RGB", "RGBA", "L", "LA"}:
                raise ValueError("V4 standard ingest supports RGB/gray JPEG, PNG and TIFF")
            if image_format == "PNG" and source_bytes[24] == 16:
                raise ValueError("16-bit PNG requires lossless conversion to TIFF before V4 ingest")
            exif = image.getexif()
            orientation = int(exif.get(274, 1))
            exif_color = exif.get(40961)
            if 34665 in exif:
                exif_color = exif.get_ifd(34665).get(40961, exif_color)
            profile = image.info.get("icc_profile")
            if image_format == "PNG" and "srgb" in image.info:
                declared_color = "srgb"
            pixels = np.asarray(image)

    pixels = _orient(pixels, orientation)
    if pixels.dtype not in {np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")}:
        raise ValueError("Photographic samples must be uint8, uint16 or float32")
    bits = pixels.dtype.itemsize * 8
    divisor = float(np.iinfo(pixels.dtype).max) if np.issubdtype(pixels.dtype, np.integer) else 1.0
    values = pixels.astype(np.float32) / divisor
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    if values.ndim != 3 or values.shape[2] not in {2, 3, 4}:
        raise ValueError("Unsupported photographic sample shape")
    alpha = values[..., -1] if values.shape[2] in {2, 4} else None
    rgb = np.repeat(values[..., :1], 3, axis=2) if values.shape[2] == 2 else values[..., :3]
    color, resolution = _resolve_color(
        input_color=input_color,
        image_format=image_format,
        profile=profile,
        declared_color=declared_color,
        exif_color=exif_color,
    )
    linear = srgb_to_linear(rgb) if color == "srgb" else rgb
    return ImageMaster(
        linear,
        source_sha256,
        bits,
        alpha=alpha,
        source_icc=profile,
        metadata={
            "source_format": image_format,
            "input_color": color,
            "color_resolution": resolution,
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
