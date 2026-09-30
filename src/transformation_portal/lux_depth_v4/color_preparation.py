"""Bounded source-color resolution and precision-preserving linear-sRGB preparation.

Profiles authorize transforms through their bytes, never their display names.
The existing analytic sRGB path remains exact; other RGB/gray ICC profiles use
the governed imagecodecs LittleCMS float engine without quantizing or clipping.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import ImageCms

MAX_ICC_BYTES = 4 * 1024**2
_MAX_ICC_TAGS = 1024
_TILE_PIXELS = 262144
_SRGB_CHROMATICITY = (0.3127, 0.3290, 0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600)
_P3_CHROMATICITY = (0.3127, 0.3290, 0.6800, 0.3200, 0.2650, 0.6900, 0.1500, 0.0600)


class InputColorError(ValueError):
    """A deterministic color-policy rejection safe to surface before execution."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Exact historical sRGB inverse transfer, including extended finite values."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((np.maximum(values, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Exact historical sRGB display transfer without clipping."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - 0.055).astype(np.float32)


def normalize_icc(profile: bytes) -> bytes:
    """Normalize generated profile timestamps and IDs; source bytes stay intact."""
    if len(profile) < 128:
        return profile
    normalized = bytearray(profile)
    normalized[24:36] = bytes.fromhex("07d000010001000000000000")
    normalized[84:100] = bytes(16)
    return bytes(normalized)


def output_srgb_icc() -> bytes:
    """Preserve the established deterministic delivery profile exactly."""
    return normalize_icc(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def _unsupported_profile() -> InputColorError:
    return InputColorError(
        "input_color_unsupported_icc",
        "Unsupported ICC profile. Convert the photographs to sRGB with a profile-aware editor and upload the converted "
        "copies. If Auto still rejects the exported sRGB profile, select sRGB only for those converted copies. "
        "An input color override does not convert pixels.",
    )


def _metadata_error(message: str) -> InputColorError:
    return InputColorError("input_color_unsupported_metadata", message)


def _cms() -> Any:
    try:
        import imagecodecs
    except ImportError as exc:
        raise _metadata_error("Color preparation requires the governed imagecodecs CMS runtime.") from exc
    if not imagecodecs.CMS.available:
        raise _metadata_error("Color preparation requires the governed imagecodecs CMS runtime.")
    return imagecodecs


def _profile_tags(profile: bytes) -> dict[bytes, memoryview]:
    """Validate ICC ranges before passing bounded bytes to a native parser."""
    if not isinstance(profile, bytes) or not 132 <= len(profile) <= MAX_ICC_BYTES:
        raise _unsupported_profile()
    size = struct.unpack_from(">I", profile)[0]
    count = struct.unpack_from(">I", profile, 128)[0]
    table_end = 132 + count * 12
    if (
        size != len(profile)
        or profile[36:40] != b"acsp"
        or profile[8] not in {2, 4}
        or profile[12:16] not in {b"mntr", b"scnr", b"spac"}
        or profile[16:20] not in {b"RGB ", b"GRAY"}
        or profile[20:24] not in {b"XYZ ", b"Lab "}
        or not 1 <= count <= _MAX_ICC_TAGS
        or table_end > size
    ):
        raise _unsupported_profile()
    tags: dict[bytes, memoryview] = {}
    view = memoryview(profile)
    ranges: list[tuple[int, int]] = []
    for index in range(count):
        signature, offset, length = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if signature in tags or offset % 4 or offset < table_end or length < 8 or offset + length > size:
            raise _unsupported_profile()
        interval = (offset, offset + length)
        for other in ranges:
            if interval != other and max(interval[0], other[0]) < min(interval[1], other[1]):
                raise _unsupported_profile()
        ranges.append(interval)
        tags[signature] = view[offset : offset + length]
    return tags


def _validated_profile(profile: bytes) -> tuple[str, bytes]:
    _profile_tags(profile)
    cms = _cms()
    try:
        cms.cms_profile_validate(profile)
    except (ValueError, RuntimeError) as exc:
        raise _unsupported_profile() from exc
    return ("icc_gray" if profile[16:20] == b"GRAY" else "icc_rgb"), profile


def _linear_profile() -> bytes:
    try:
        return normalize_icc(_cms().cms_profile("linearrgb"))
    except (ValueError, RuntimeError, AttributeError) as exc:
        raise _metadata_error(
            "Color preparation requires an imagecodecs CMS runtime with linear RGB profile support."
        ) from exc


def _rgb_profile(chromaticity: tuple[float, ...], *, gamma: float | None = None) -> bytes:
    """Build calibrated PNG RGB metadata, with exact sRGB TRCs when requested."""
    cms = _cms()
    profile = cms.cms_profile("rgb", whitepoint=chromaticity[:2], primaries=chromaticity[2:], gamma=gamma or 1.0)
    if gamma is not None:
        return normalize_icc(profile)
    # imagecodecs exposes gamma/tables, but not parametric curves. Reuse the
    # native engine's exact sRGB curve bytes in this newly generated RGB matrix
    # profile instead of approximating the piecewise transfer with a LUT.
    tags = _profile_tags(profile)
    curve = _profile_tags(cms.cms_profile("srgb"))[b"rTRC"]
    for signature in (b"rTRC", b"gTRC", b"bTRC"):
        tags[signature] = curve
    header = bytearray(profile[:128])
    table = bytearray(struct.pack(">I", len(tags)))
    data = bytearray()
    offset = 132 + 12 * len(tags)
    for signature, value in tags.items():
        table.extend(struct.pack(">4sII", signature, offset + len(data), len(value)))
        data.extend(value)
        data.extend(b"\0" * (-len(value) % 4))
    result = header + table + data
    struct.pack_into(">I", result, 0, len(result))
    return normalize_icc(bytes(result))


def _png_color(metadata: Mapping[str, Any], *, icc_authoritative: bool = False) -> tuple[str | None, bytes | None]:
    """Resolve recognized PNG declarations; incomplete metadata never authorizes guessing."""
    srgb = metadata.get("srgb")
    cicp = metadata.get("cicp")
    gamma = metadata.get("gamma")
    chromaticity = metadata.get("chromaticity")
    if srgb is not None and (type(srgb) is not int or srgb not in range(4)):
        raise _metadata_error("Invalid PNG sRGB rendering intent.")
    if gamma is not None and (
        isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not np.isfinite(gamma) or not 0 < gamma <= 10
    ):
        raise _metadata_error("Invalid PNG gamma metadata.")
    if chromaticity is not None:
        if not isinstance(chromaticity, (list, tuple)) or len(chromaticity) != 8:
            raise _metadata_error("Invalid PNG chromaticity metadata.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in chromaticity
        ):
            raise _metadata_error("Invalid PNG chromaticity metadata.")
        chromaticity = tuple(float(value) for value in chromaticity)
        if any(not 0 <= value <= 1 for value in chromaticity) or any(
            chromaticity[index] + chromaticity[index + 1] > 1 for index in range(0, 8, 2)
        ):
            raise _metadata_error("Invalid PNG chromaticity metadata.")
    # PNG's gamma/chromaticity chunks are fallback encodings when an ICC exists.
    # Validate their syntax, but do not apply them a second time or reinterpret
    # ICC pixels through a legacy fallback gamma approximation.
    if icc_authoritative:
        gamma, chromaticity = None, None
    if cicp is not None:
        if (
            not isinstance(cicp, (bytes, tuple, list))
            or len(cicp) != 4
            or any(type(value) is not int for value in cicp)
            or tuple(cicp) not in {(1, 13, 0, 1), (12, 13, 0, 1)}
        ):
            raise _metadata_error("Unsupported PNG cICP color encoding; export a photograph with a supported RGB ICC profile.")
        p3 = tuple(cicp)[0] == 12
        if srgb is not None and p3:
            raise InputColorError("input_color_conflict", "PNG cICP and sRGB metadata disagree")
        expected = _P3_CHROMATICITY if p3 else _SRGB_CHROMATICITY
        if chromaticity is not None and not np.allclose(chromaticity, expected, atol=0.00002, rtol=0):
            raise InputColorError("input_color_conflict", "PNG cICP and chromaticity metadata disagree")
        if gamma is not None and abs(gamma - 0.45455) > 0.00002:
            raise InputColorError("input_color_conflict", "PNG cICP and gamma metadata disagree")
        return ("png_cicp_display_p3", _rgb_profile(_P3_CHROMATICITY)) if p3 else ("png_cicp_srgb", None)
    if srgb is not None:
        if (gamma is not None and abs(gamma - 0.45455) > 0.00002) or (
            chromaticity is not None and not np.allclose(chromaticity, _SRGB_CHROMATICITY, atol=0.00002, rtol=0)
        ):
            raise InputColorError("input_color_conflict", "PNG sRGB and calibrated color metadata disagree")
        return "png_srgb", None
    if gamma is None and chromaticity is None:
        return None, None
    if gamma is None or chromaticity is None:
        raise _metadata_error(
            "Incomplete PNG gamma/chromaticity metadata; supply a known source encoding or a color-managed export."
        )
    if not any(np.allclose(chromaticity, known, atol=0.00002, rtol=0) for known in (_SRGB_CHROMATICITY, _P3_CHROMATICITY)):
        raise _metadata_error("Unsupported PNG chromaticity; export a photograph with a supported RGB ICC profile.")
    return "png_calibrated_rgb", _rgb_profile(chromaticity, gamma=1.0 / gamma)


@dataclass(frozen=True)
class PreparedColor:
    source_color: str
    resolution: str
    action: str
    source_profile: bytes | None = None
    transform_profile: bytes | None = None
    target_profile: bytes | None = None
    engine_version: str = ""
    requested_input_color: str = "auto"
    source_metadata: Mapping[str, Any] | None = None

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "schema": "tp.color.preparation.v1",
            "source_metadata": {
                **{key: value for key, value in (self.source_metadata or {}).items() if key != "png"},
                "png": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in (self.source_metadata or {}).get("png", {}).items()
                },
            },
            "requested_input_color": self.requested_input_color,
            "action": self.action,
            "source_color": self.source_color,
            "output_color": "linear_srgb",
            "output_precision": "float32",
            "resolution": self.resolution,
            "assumed_srgb": self.action == "assume",
            "source_icc_sha256": hashlib.sha256(self.source_profile).hexdigest() if self.source_profile is not None else None,
            "target_icc_sha256": hashlib.sha256(self.target_profile).hexdigest() if self.target_profile is not None else None,
            "transform_icc_sha256": (
                hashlib.sha256(self.transform_profile).hexdigest() if self.transform_profile is not None else None
            ),
            "engine": "imagecodecs_lcms" if self.transform_profile is not None else "numpy",
            "engine_version": self.engine_version if self.transform_profile is not None else np.__version__,
            "intent": "relative_colorimetric" if self.transform_profile is not None else None,
            "alpha_policy": "preserve_separately",
        }


def _icc_preparation(profile: bytes, *, source_profile: bytes | None, resolution: str) -> PreparedColor:
    source_color, profile = _validated_profile(profile)
    cms = _cms()
    target = _linear_profile()
    # Fail closed if the active native engine clips extended linear RGB. This
    # tiny fixed probe exercises gamut expansion independently of source pixels.
    primaries = np.eye(3, dtype=np.float32)[None, ...]
    extended = _cms_transform(primaries, cms.cms_profile("adobergb"), target, "icc_rgb")
    if extended.dtype != np.float32 or not np.isfinite(extended).all() or extended.min() >= -0.1 or extended.max() <= 1.1:
        raise _metadata_error("Color preparation requires an unclipped float32 CMS transform.")
    # Transform creation can reject a structurally valid but unusable profile.
    # This bounded header-time probe never reads photographic pixels.
    probe = np.zeros((1, 1) if source_color == "icc_gray" else (1, 1, 3), np.float32)
    try:
        _cms_transform(probe, profile, target, source_color)
    except (ValueError, RuntimeError) as exc:
        raise _unsupported_profile() from exc
    return PreparedColor(
        source_color,
        resolution,
        "convert",
        source_profile,
        profile,
        target,
        f"imagecodecs {cms.__version__}; {cms.cms_version()}",
    )


def _prepare_input_color(
    *,
    input_color: str,
    image_format: str,
    profile: bytes | None,
    declared_color: str | None,
    exif_color: Any,
    png_metadata: Mapping[str, Any] | None = None,
) -> PreparedColor:
    """Resolve a bounded transform or explicit/recorded assumption before pixel reads."""
    if input_color not in {"auto", "auto_assume_srgb", "srgb", "linear_srgb"}:
        raise ValueError("input_color must be auto, auto_assume_srgb, srgb, or linear_srgb")
    if image_format == "RAW":
        if input_color not in {"auto", "auto_assume_srgb", "linear_srgb"}:
            raise InputColorError("input_color_conflict", "RAW decoder output is linear_srgb; encoded input_color is invalid")
        return PreparedColor("linear_srgb", "governed_raw_decoder", "identity")
    if profile is not None and (not isinstance(profile, bytes) or len(profile) > MAX_ICC_BYTES):
        raise _unsupported_profile()
    # Preserve the existing deliberate operator override. The additive Auto
    # assumption mode never enters this branch or ignores malformed metadata.
    if input_color in {"srgb", "linear_srgb"}:
        return PreparedColor(input_color, "explicit_input_color", "explicit", profile)
    if type(exif_color) not in {int, type(None)} or exif_color not in {None, 1, 65535}:
        raise _metadata_error("Unsupported EXIF color space; provide an explicit input_color correction")
    if profile is not None:
        if declared_color not in {None, "srgb"}:
            raise InputColorError("input_color_conflict", "ICC and declared input color disagree")
        if normalize_icc(profile) == output_srgb_icc():
            prepared = PreparedColor("srgb", "recognized_srgb_icc", "convert", profile)
        else:
            prepared = _icc_preparation(profile, source_profile=profile, resolution="embedded_icc_to_linear_srgb")
        png_resolution, png_profile = _png_color(png_metadata or {}, icc_authoritative=True)
        if declared_color == "srgb" or exif_color == 1 or png_resolution in {"png_srgb", "png_cicp_srgb"}:
            # A declaration may not turn a non-sRGB ICC into sRGB. Compare the
            # transform itself against sRGB, never the profile description.
            samples = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.5, 0.75]]], np.float32)
            if prepared.source_color not in {"icc_rgb", "srgb"} or not np.allclose(
                apply_input_color(samples, prepared),
                srgb_to_linear(samples),
                atol=0.0001,
                rtol=0,
            ):
                raise InputColorError("input_color_conflict", "ICC and declared input color disagree")
        if png_profile is not None:
            png_prepared = _icc_preparation(png_profile, source_profile=None, resolution=png_resolution or "png_color")
            samples = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.5, 0.75]]], np.float32)
            if prepared.source_color != "icc_rgb" or not np.allclose(
                apply_input_color(samples, prepared),
                apply_input_color(samples, png_prepared),
                atol=0.0003,
                rtol=0,
            ):
                raise InputColorError("input_color_conflict", "ICC and PNG cICP color metadata disagree")
        return prepared
    png_resolution, png_profile = _png_color(png_metadata or {})
    if png_resolution:
        if declared_color not in {None, "srgb"}:
            raise InputColorError("input_color_conflict", "PNG and declared input color disagree")
        if png_profile is not None:
            prepared = _icc_preparation(png_profile, source_profile=None, resolution=png_resolution)
            if exif_color == 1 or declared_color == "srgb":
                raise InputColorError("input_color_conflict", "PNG calibrated color and sRGB metadata disagree")
            return prepared
        return PreparedColor("srgb", png_resolution, "convert")
    if declared_color is not None:
        if declared_color not in {"srgb", "linear_srgb"}:
            raise _metadata_error("Unsupported declared color space; provide an explicit input_color correction")
        if declared_color == "linear_srgb" and exif_color == 1:
            raise InputColorError("input_color_conflict", "EXIF sRGB and declared linear input color disagree")
        return PreparedColor(
            declared_color, "declared_input_color", "identity" if declared_color == "linear_srgb" else "convert"
        )
    if exif_color == 1:
        return PreparedColor("srgb", "exif_srgb", "convert")
    if exif_color is None and image_format == "JPEG":
        return PreparedColor("srgb", "untagged_jpeg_srgb_assumption", "assume")
    if input_color == "auto_assume_srgb" and exif_color is None and image_format in {"JPEG", "PNG", "TIFF"}:
        return PreparedColor("srgb", "untagged_input_srgb_assumption", "assume")
    raise InputColorError(
        "input_color_ambiguous",
        "Ambiguous input color. Auto found no usable color-space metadata. Choose Auto with sRGB assumption only if that "
        "assumption is acceptable, select a known source color, or re-export from the original with an embedded profile. "
        "Assumptions are recorded and do not recover the original profile.",
    )


def prepare_input_color(
    *,
    input_color: str,
    image_format: str,
    profile: bytes | None,
    declared_color: str | None,
    exif_color: Any,
    png_metadata: Mapping[str, Any] | None = None,
) -> PreparedColor:
    """Return immutable color preparation with the exact requested policy bound."""
    prepared = _prepare_input_color(
        input_color=input_color,
        image_format=image_format,
        profile=profile,
        declared_color=declared_color,
        exif_color=exif_color,
        png_metadata=png_metadata,
    )
    png = {}
    for key in ("srgb", "gamma", "chromaticity", "cicp"):
        value = (png_metadata or {}).get(key)
        if value is not None:
            if isinstance(value, (tuple, list, bytes)):
                value = tuple(value)
                if len(value) > 8 or any(type(item) not in {int, float} or not np.isfinite(item) for item in value):
                    raise _metadata_error("Invalid PNG color metadata.")
            elif type(value) not in {int, float} or not np.isfinite(value):
                raise _metadata_error("Invalid PNG color metadata.")
            png[key] = value
    source_metadata = MappingProxyType(
        {
            "image_format": image_format,
            "declared_color": (
                declared_color
                if declared_color is None or (isinstance(declared_color, str) and declared_color in {"srgb", "linear_srgb"})
                else "unsupported"
            ),
            "exif_color": exif_color if type(exif_color) in {int, type(None)} else "unsupported",
            "png": MappingProxyType(png),
        }
    )
    return replace(prepared, requested_input_color=input_color, source_metadata=source_metadata)


def _cms_transform(samples: np.ndarray, profile: bytes, target: bytes, source_color: str) -> np.ndarray:
    cms = _cms()
    return cms.cms_transform(
        samples,
        profile,
        target,
        colorspace="gray" if source_color == "icc_gray" else "rgb",
        planar=False,
        outcolorspace="rgb",
        outplanar=False,
        outdtype=np.float32,
        intent=cms.CMS.INTENT.RELATIVE_COLORIMETRIC,
        flags=cms.CMS.FLAGS.NOOPTIMIZE | cms.CMS.FLAGS.NOCACHE,
    )


def apply_input_color(samples: np.ndarray, prepared: PreparedColor) -> np.ndarray:
    """Normalize finite float32 RGB/gray samples; callers retain separate alpha."""
    if samples.dtype != np.float32 or samples.ndim not in {2, 3} or not np.isfinite(samples).all():
        raise ValueError("Color preparation requires finite float32 RGB or grayscale samples")
    if min(samples.shape[:2]) <= 0 or (samples.ndim == 3 and samples.shape[2] != 3):
        raise ValueError("Color preparation requires HxW grayscale or HxWx3 RGB samples")
    if prepared.transform_profile is None:
        rgb = np.repeat(samples[..., None], 3, axis=2) if samples.ndim == 2 else samples
        return srgb_to_linear(rgb) if prepared.source_color == "srgb" else rgb
    if samples.ndim == 2 and prepared.source_color == "icc_rgb" and prepared.source_profile is None:
        # Calibrated PNG declarations describe neutral RGB even for a grayscale
        # raster; embedded ICC profiles must still match their actual channels.
        samples = np.repeat(samples[..., None], 3, axis=2)
    if (prepared.source_color == "icc_gray") != (samples.ndim == 2):
        raise InputColorError("input_color_conflict", "ICC channel layout does not match the photographic samples")
    if prepared.target_profile is None:
        raise ValueError("Prepared ICC transform has no target profile")
    output = np.empty((*samples.shape[:2], 3), np.float32)
    rows = max(1, _TILE_PIXELS // samples.shape[1])
    for start in range(0, samples.shape[0], rows):
        converted = _cms_transform(
            samples[start : start + rows], prepared.transform_profile, prepared.target_profile, prepared.source_color
        )
        if (
            converted.shape != output[start : start + rows].shape
            or converted.dtype != np.float32
            or not np.isfinite(converted).all()
        ):
            raise ValueError("Color transform returned invalid photographic samples")
        output[start : start + rows] = converted
    return output
