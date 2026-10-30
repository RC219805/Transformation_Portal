"""V2 Depth-Aware Enhancement Implementation.

Consumes depth maps from V3 stage and applies perceptual finishing
for luxury real estate marketing output.

Architecture:
- Reuses existing EnhancementStage for core enhancement logic
- Depth-aware tone mapping for spatial hierarchy
- Material-aware processing using Materials V3 taxonomy
- NO ML DEPENDENCIES (image processing only)

Performance:
- Enhancement stage: ~0.02s/image (isolated computation)
- End-to-end pipeline: ~1.8s/image (with depth estimation, I/O, orchestration)
- Target: <2s/image end-to-end ✅

Dependencies:
    - numpy, scipy (core)
    - PIL (image I/O)
    - EnhancementStage (reuse existing)

Reference: V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import struct
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from ..core.image_artifact import ImageMaster
from ..stage_graph.stage import StageContext, StageStatus
from ..stage_graph.stages.enhancement import EnhancementStage
from .v2_presets import V2_COLOR_CONTRACT, V2_PROCESSING_VERSION, V2EnhancementConfig, processing_version_for_geometry

logger = logging.getLogger(__name__)


class V2EnhancementError(Exception):
    """Raised when V2 enhancement operation fails."""

    pass


def load_native_materials_binding(path: Path, *, source_path: str | None = None) -> dict[str, Any]:
    """Read bounded private execution authority without following symlinks."""
    from .geometry_handoff import snapshot_geometry_artifact
    from .native_materials import MAX_NATIVE_MATERIALS_BINDING_BYTES, revalidate_native_materials, validate_native_materials

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V2EnhancementError("Native Materials binding contains duplicate JSON fields")
            result[key] = value
        return result

    raw = snapshot_geometry_artifact(Path(path), MAX_NATIVE_MATERIALS_BINDING_BYTES)
    binding = json.loads(raw, object_pairs_hook=unique_object)
    validate_native_materials(binding)
    if source_path is None:
        if len(binding["sources"]) != 1:
            raise V2EnhancementError("Native Materials child requires an explicit source selector for multiple sources")
        source_path = binding["sources"][0]["path"]
    revalidate_native_materials(binding, source_path=source_path)
    return binding


def _legacy_container_snapshot(source: bytes, image: Image.Image) -> bytes:
    """Losslessly normalize supported WebP/BMP samples without inventing color.

    The PNG carrier is private to the photographic decoder. Original profile
    and EXIF evidence remain authorizing; untagged samples still require the
    caller's recorded assumption or explicit encoding assertion.
    """
    profile = image.info.get("icc_profile")
    declared_srgb = False
    if image.format == "WEBP":
        if source[:4] != b"RIFF" or source[8:12] != b"WEBP" or len(source) != int.from_bytes(source[4:8], "little") + 8:
            raise V2EnhancementError("Invalid WebP container length")
        offset, seen, total = 12, set(), 0
        for _ in range(8192):
            if offset == len(source):
                break
            if offset + 8 > len(source):
                raise V2EnhancementError("Incomplete WebP metadata")
            kind, size = source[offset : offset + 4], int.from_bytes(source[offset + 4 : offset + 8], "little")
            end = offset + 8 + size
            if end + size % 2 > len(source):
                raise V2EnhancementError("Incomplete WebP chunk")
            if kind in {b"ICCP", b"EXIF"}:
                total += size
                if kind in seen or total > 4 * 1024**2:
                    raise V2EnhancementError("Duplicate or oversized WebP color metadata")
                seen.add(kind)
                if kind == b"ICCP":
                    profile = source[offset + 8 : end]
                    if not profile:
                        raise V2EnhancementError("Empty WebP ICC metadata")
            offset = end + size % 2
        else:
            raise V2EnhancementError("WebP exceeds the metadata chunk budget")
    elif image.format == "BMP":
        header_size = int.from_bytes(source[14:18], "little")
        if header_size >= 108:
            if header_size not in {108, 124} or len(source) < 14 + header_size:
                raise V2EnhancementError("Unsupported BMP color header")
            color_space = int.from_bytes(source[70:74], "little")
            if color_space == 0x73524742:  # LCS_sRGB
                declared_srgb = True
            elif color_space == 0x4D424544 and header_size == 124:  # PROFILE_EMBEDDED
                start = 14 + int.from_bytes(source[126:130], "little")
                size = int.from_bytes(source[130:134], "little")
                if start < 14 + header_size or not 0 < size <= 4 * 1024**2 or start + size > len(source):
                    raise V2EnhancementError("Invalid BMP embedded ICC extent")
                profile = source[start : start + size]
            else:
                raise V2EnhancementError("Unsupported calibrated, linked, or unknown BMP color metadata")
    if image.mode not in {"RGB", "RGBA", "L", "P", "1"}:
        raise V2EnhancementError("Unsupported WebP/BMP photographic sample layout")
    if image.mode in {"P", "1"}:
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    metadata = PngImagePlugin.PngInfo()
    if declared_srgb:
        metadata.add(b"sRGB", b"\0")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", icc_profile=profile, exif=image.getexif().tobytes(), pnginfo=metadata)
    return buffer.getvalue()


def decode_v2_source(source: bytes, *, source_name: str, input_color: str = "auto") -> ImageMaster:
    """Use photographic color authority for every supported legacy V3 format."""
    from transformation_portal.core.image_artifact import ImageMaster, metadata_payload
    from transformation_portal.lux_depth_v4.photography import decode_master

    original = source
    source_format = None
    if source[:4] not in {b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"}:
        with Image.open(io.BytesIO(source)) as image:
            source_format = image.format
            if image.format in {"WEBP", "BMP"}:
                source = _legacy_container_snapshot(source, image)
            elif image.format == "PNG" and image.mode == "P":
                source = _expand_palette_png_snapshot(source)
    master = decode_master(source, source_name=source_name, input_color=input_color)
    if source is original:
        return master
    metadata = metadata_payload(master.metadata)
    metadata["source_format"] = source_format
    metadata["source_normalization"] = {
        "schema": "tp.v3.lossless_color_carrier.v1",
        "source_format": source_format,
        "carrier_format": "PNG",
        "carrier_sha256": hashlib.sha256(source).hexdigest(),
        "pixel_quantization": False,
        "color_metadata_preserved": True,
    }
    return ImageMaster(
        master.pixels, hashlib.sha256(original).hexdigest(), master.source_bit_depth, master.alpha, metadata, master.source_icc
    )


def bounded_encoded_srgb(master: ImageMaster) -> np.ndarray:
    """V3 cannot retain float source precision or out-of-gamut/HDR samples."""
    from transformation_portal.lux_depth_v4.photography import linear_to_srgb

    if master.source_bit_depth > 16:
        raise V2EnhancementError("V3 finishing supports at most 16-bit masters; use Unified/V5/V6 to retain float precision")
    pixels = linear_to_srgb(master.pixels)
    if not np.isfinite(pixels).all() or np.any(pixels < -1e-6) or np.any(pixels > 1 + 1e-6):
        raise V2EnhancementError(
            "V3 finishing requires bounded encoded sRGB; use Unified/V5/V6 for extended/HDR or wide-gamut masters"
        )
    return np.clip(pixels, 0, 1)


def prepare_materials_srgb_proxy(
    pixels: np.ndarray, *, source: bytes | None, source_name: str, input_color: str, raw_contract: dict | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Color-manage the existing Materials proxy while retaining its geometry.

    Depth/model preprocessing is unchanged. Its RGB proxy still has the source
    transfer/primaries, so apply the source's validated transform to that proxy
    before Materials operates. Source-wide validation prevents a clipped proxy
    from concealing extended/wide-gamut photographic samples.
    """
    from transformation_portal.core.image_artifact import metadata_payload
    from transformation_portal.lux_depth_v4.color_preparation import apply_input_color, prepare_input_color
    from transformation_portal.lux_depth_v4.photography import linear_to_srgb

    if pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise V2EnhancementError("Materials requires a finite float32 RGB proxy")
    if raw_contract is not None:
        if (
            raw_contract.get("contract") != "legacy_linear_srgb"
            or raw_contract.get("preview_escape_enabled") is not False
            or raw_contract.get("mode") not in {"auto", "force_rawpy"}
            or input_color not in {"auto", "auto_assume_srgb", "linear_srgb"}
        ):
            raise V2EnhancementError(
                "Materials RAW finishing requires the established linear-sRGB ingest; preview is depth-only"
            )
        linear = pixels
        evidence = {
            "schema": "tp.v3.materials_color.v1",
            "input_policy": input_color,
            "source_name": source_name,
            "source_contract": raw_contract,
            "source_precision_policy": "Existing clipped legacy RAW proxy; no precision or HDR recovery",
        }
    else:
        if source is None:
            raise V2EnhancementError("Materials color preparation requires its exact photographic source")
        master = decode_v2_source(source, source_name=source_name, input_color=input_color)
        bounded_encoded_srgb(master)
        color = metadata_payload(master.metadata["color_preparation"])
        origin = color["source_metadata"]
        prepared = prepare_input_color(
            input_color=input_color,
            image_format=origin["image_format"],
            profile=master.source_icc,
            declared_color=origin["declared_color"],
            exif_color=origin["exif_color"],
            png_metadata=origin["png"],
        )
        samples = pixels
        if prepared.source_color == "icc_gray":
            if not np.array_equal(pixels[..., 0], pixels[..., 1]) or not np.array_equal(pixels[..., 0], pixels[..., 2]):
                raise V2EnhancementError("Grayscale source profile does not match the Materials RGB proxy")
            samples = pixels[..., 0]
        linear = apply_input_color(samples, prepared)
        evidence = {
            "schema": "tp.v3.materials_color.v1",
            "source_sha256": master.source_sha256,
            "source_format": master.metadata["source_format"],
            "preparation": color,
            "source_precision_policy": "Existing V3 proxy precision; no precision recovery",
        }
    encoded = linear_to_srgb(linear)
    if not np.isfinite(encoded).all() or np.any(encoded < -1e-6) or np.any(encoded > 1 + 1e-6):
        raise V2EnhancementError("Materials proxy is outside bounded sRGB; use Unified/V5/V6 for extended/HDR sources")
    return np.clip(encoded, 0, 1).astype(np.float32), {
        **evidence,
        "output_color": "bounded_encoded_srgb",
        "color_contract": V2_COLOR_CONTRACT,
        "processing_version": V2_PROCESSING_VERSION,
        "geometry": list(pixels.shape[:2]),
        "depth_preprocessing_changed": False,
        "segmentation_preprocessing_changed": False,
    }


def _expand_palette_png_snapshot(source: bytes) -> bytes:
    """Expand palette entries without dropping or laundering source color evidence."""
    color_chunks = {b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"cICP", b"eXIf", b"mDCV", b"cLLI"}
    retained = []
    offset, seen_pixels, retained_size = 8, False, 0
    for _ in range(8192):
        if offset + 12 > len(source):
            raise V2EnhancementError("Incomplete palette PNG metadata")
        size, kind = struct.unpack_from(">I4s", source, offset)
        end = offset + size + 12
        if end > len(source):
            raise V2EnhancementError("Incomplete palette PNG chunk")
        if kind == b"IDAT":
            seen_pixels = True
        if kind in color_chunks:
            if seen_pixels and kind != b"eXIf":
                raise V2EnhancementError("Misplaced palette PNG color metadata")
            retained_size += size
            if retained_size > 4 * 1024**2:
                raise V2EnhancementError("Palette PNG color metadata exceeds supported bounds")
            retained.append(source[offset:end])
        offset = end
        if kind == b"IEND":
            break
    else:
        raise V2EnhancementError("Palette PNG exceeds the metadata chunk budget")
    with Image.open(io.BytesIO(source)) as palette:
        expanded = palette.convert("RGBA" if "transparency" in palette.info else "RGB")
        buffer = io.BytesIO()
        expanded.save(buffer, format="PNG", icc_profile=None, exif=b"")
    encoded = buffer.getvalue()
    # Original ICC/transfer chunks, including their original CRCs and duplicate
    # evidence, pass unchanged through the common photographic validator.
    return encoded[:33] + b"".join(retained) + encoded[33:]


def _normalize_icc_profile_payload(value: Any) -> bytes | None:
    """Normalize common PIL/TIFF ICC payload forms to bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value or None
    if isinstance(value, bytearray):
        payload = bytes(value)
        return payload or None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(isinstance(item, int) for item in value):
            try:
                payload = bytes(value)
            except ValueError:
                return None
            return payload or None
        chunks: list[bytes] = []
        for item in value:
            chunk = _normalize_icc_profile_payload(item)
            if chunk is None:
                return None
            chunks.append(chunk)
        payload = b"".join(chunks)
        return payload or None
    return None


def _extract_icc_profile(pil_image: Image.Image) -> bytes | None:
    """Extract ICC profile from PIL metadata, falling back to TIFF tag 34675."""
    icc_profile = _normalize_icc_profile_payload(pil_image.info.get("icc_profile"))
    if icc_profile is not None:
        return icc_profile

    tag_v2 = getattr(pil_image, "tag_v2", None)
    if tag_v2 is None:
        return None
    try:
        return _normalize_icc_profile_payload(tag_v2.get(34675))
    except Exception:
        return None


_DERIVED_STEM_SUFFIXES = (
    "_materials_v3_enhanced",
    "_v2_enhanced",
    "_materials_v3",
    "_enhanced",
    "_pbr",
)

_KNOWN_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".npy",
    ".exr",
)

_RAW_SOURCE_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dng",
    ".iiq",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
}


def emitted_v2_suffix_for_bit_depth(bit_depth: int) -> str:
    """Return the canonical emitted suffix for a V2 artifact."""
    return ".tif" if int(bit_depth) >= 16 else ".png"


def resolve_v2_emitted_artifact_path(
    output_path: Path,
    *,
    bit_depth: int,
    identity: str | None = None,
    materials_enabled: bool = True,
) -> Path:
    """Normalize a V2 emitted artifact path to the canonical basename and suffix."""
    candidate_path = Path(output_path)
    normalized_identity = canonical_asset_stem(identity) if identity else canonical_asset_stem(candidate_path.stem)
    if not normalized_identity:
        normalized_identity = "artifact"
    stage_label = "materials_v3_enhanced" if materials_enabled else "v2_enhanced"
    emitted_name = f"{normalized_identity}_{stage_label}{emitted_v2_suffix_for_bit_depth(bit_depth)}"
    return candidate_path.with_name(emitted_name)


def infer_v2_output_bit_depth(input_path: Path, *, allow_8bit_output: bool = False) -> int:
    """Infer the likely emitted V2 bit depth before enhancement runs."""
    if allow_8bit_output:
        return 8

    input_path = Path(input_path)
    if input_path.suffix.lower() in _RAW_SOURCE_EXTENSIONS:
        return 16

    try:
        with Image.open(input_path) as pil_image:
            return 16 if detect_input_bit_depth(pil_image) == 16 else 8
    except Exception as exc:  # pragma: no cover - best-effort inference fallback
        logger.debug("Could not inspect input bit depth for %s: %s", input_path, exc)

    if input_path.suffix.lower() in {".tif", ".tiff"}:
        return 16

    return 8


def _convert_alpha_to_target_dtype(alpha: np.ndarray, target_dtype: np.dtype[Any]) -> np.ndarray:
    """Scale an alpha plane into the selected output encoding range."""

    if alpha.dtype == target_dtype:
        return alpha
    if target_dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise V2EnhancementError(f"Unsupported alpha target dtype: {target_dtype}")
    if np.issubdtype(alpha.dtype, np.integer):
        normalized = alpha.astype(np.float32) / float(np.iinfo(alpha.dtype).max)
    elif np.issubdtype(alpha.dtype, np.floating):
        normalized = alpha.astype(np.float32)
    else:
        raise V2EnhancementError(f"Unsupported alpha dtype: {alpha.dtype}")
    target_max = float(np.iinfo(target_dtype).max)
    return (np.clip(normalized, 0.0, 1.0) * target_max + 0.5).astype(target_dtype)


def _coerce_to_stem_preserving_dots(input_path_or_stem: str) -> str:
    """Treat value as a stem by default; only strip extensions for path-like inputs."""
    raw = str(input_path_or_stem).strip()
    if not raw:
        return ""

    # Real paths should use pathlib stem extraction.
    if "/" in raw or "\\" in raw:
        return Path(raw).stem

    # If value looks like a filename with a known extension, strip extension only.
    lowered = raw.lower()
    for extension in _KNOWN_IMAGE_EXTENSIONS:
        if lowered.endswith(extension):
            return raw[: -len(extension)]

    # Already-stem values may legitimately include dots (e.g., image.v1_materials_v3_enhanced).
    return raw


def canonical_asset_stem(input_path_or_stem: str) -> str:
    """Normalize derived V2 stems back to canonical source asset stem."""
    stem = _coerce_to_stem_preserving_dots(input_path_or_stem)
    normalized = stem

    # Repeated stripping handles multi-derived names like *_materials_v3_enhanced.
    changed = True
    while changed:
        changed = False
        for suffix in _DERIVED_STEM_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
                break

    return normalized


def _relative_depth_candidate(depth_dir: Path, depth_path: Path) -> str:
    """Return a stable candidate path for ambiguity diagnostics."""
    try:
        return str(depth_path.relative_to(depth_dir))
    except ValueError:
        return str(depth_path)


def find_depth_map(depth_dir: Path, image_stem: str) -> Optional[Path]:
    """Find depth map for input image in depth directory.

    Searches for depth maps with common naming patterns:
    - {image_stem}_depth.png
    - {image_stem}_depth_u16.png
    - {image_stem}.png (in depth_dir)

    Args:
        depth_dir: Directory containing depth maps
        image_stem: Input image filename stem

    Returns:
        Path to depth map or None if not found

    Raises:
        V2EnhancementError: If more than one plausible depth map is found
    """
    if not depth_dir or not depth_dir.exists():
        return None

    canonical_stem = canonical_asset_stem(image_stem)
    candidate_stems = [canonical_stem]
    if image_stem not in candidate_stems:
        candidate_stems.append(image_stem)

    candidates: list[Path] = []
    seen_candidates: set[Path] = set()

    def add_candidate(depth_path: Path) -> None:
        if depth_path not in seen_candidates:
            seen_candidates.add(depth_path)
            candidates.append(depth_path)

    for stem in candidate_stems:
        patterns = [
            f"{stem}_depth.png",
            f"{stem}_depth_u16.png",
            f"{stem}_depth_f32.png",
            f"{stem}.png",
        ]

        # Fast path: direct children of depth_dir.
        for pattern in patterns:
            depth_path = depth_dir / pattern
            if depth_path.exists():
                add_candidate(depth_path)

        # Fallback: recursive search for nested output_key
        # directories. Sort for deterministic selection.
        for pattern in patterns:
            recursive_matches = sorted(depth_dir.glob(f"**/{pattern}"))
            for depth_path in recursive_matches:
                if depth_path.exists():
                    add_candidate(depth_path)

    if len(candidates) == 1:
        logger.debug(f"Found depth map: {candidates[0]}")
        return candidates[0]

    if len(candidates) > 1:
        candidate_list = [_relative_depth_candidate(depth_dir, candidate) for candidate in candidates]
        raise V2EnhancementError(
            "Ambiguous depth map matches for "
            f"'{image_stem}' (canonical='{canonical_stem}') in {depth_dir}: "
            f"{candidate_list}"
        )

    logger.warning(
        "No depth map found for '%s' (canonical='%s') in %s",
        image_stem,
        canonical_stem,
        depth_dir,
    )
    return None


def detect_input_bit_depth(pil_image: Image.Image) -> int:
    """Detect bit-depth of input image from TIFF tags.

    Args:
        pil_image: PIL Image object

    Returns:
        Bits per sample (8 or 16)
    """
    # Check TIFF tags for BitsPerSample
    if hasattr(pil_image, "tag_v2"):
        bits_per_sample = pil_image.tag_v2.get(258)  # BitsPerSample TIFF tag
        if bits_per_sample:
            # Can be tuple (R,G,B) or single value
            if isinstance(bits_per_sample, (tuple, list)):
                bits = bits_per_sample[0]
            else:
                bits = bits_per_sample

            if bits == 16:
                return 16
            elif bits == 8:
                return 8

    # Fallback: check PIL mode
    if pil_image.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return 16

    # Default to 8-bit
    return 8


def _apply_exif_orientation_to_array(image_array: np.ndarray, orientation: int | None) -> np.ndarray:
    """Apply EXIF orientation to a numpy image array using Pillow-compatible semantics."""
    if orientation in (None, 1):
        return image_array

    if orientation == 2:
        return np.flip(image_array, axis=1)
    if orientation == 3:
        return np.rot90(image_array, 2)
    if orientation == 4:
        return np.flip(image_array, axis=0)
    if orientation == 5:
        return np.rot90(np.flip(image_array, axis=1), 1)
    if orientation == 6:
        return np.rot90(image_array, -1)
    if orientation == 7:
        return np.rot90(np.flip(image_array, axis=1), -1)
    if orientation == 8:
        return np.rot90(image_array, 1)

    logger.debug("Ignoring unsupported EXIF orientation value: %s", orientation)
    return image_array


def _normalized_exif_after_orientation(pil_image: Image.Image, orientation_applied: bool) -> Optional[bytes]:
    """Return EXIF bytes that cannot trigger a second orientation transform."""
    try:
        exif_obj = pil_image.getexif()
    except (AttributeError, ValueError, OSError):
        return None

    if not exif_obj:
        return None

    if orientation_applied:
        try:
            if 0x0112 in exif_obj:
                del exif_obj[0x0112]
        except (KeyError, TypeError, ValueError):
            return None
        if not exif_obj:
            return None

    try:
        return exif_obj.tobytes()
    except (AttributeError, ValueError, OSError):
        return None


def load_image_preserve_bit_depth(input_path: Path, allow_8bit_output: bool = False) -> tuple[np.ndarray, int, dict]:
    """Load image preserving bit depth (8-bit or 16-bit).

    Uses tifffile for 16-bit TIFFs to avoid PIL's auto-conversion to 8-bit.

    Args:
        input_path: Path to input image
        allow_8bit_output: If False, raises error if 16-bit input must be downconverted

    Returns:
        Tuple of (image_array, bits_per_sample, metadata)
        - image_array: np.ndarray with dtype uint8 or uint16
        - bits_per_sample: 8 or 16
        - metadata: dict with ICC profile, EXIF, etc.

    Raises:
        V2EnhancementError: If Quality Firewall blocks downconversion
    """
    # First, open with PIL to check format and extract metadata
    with Image.open(input_path) as pil_image:
        # Extract metadata before any processing
        metadata = {
            "icc_profile": _extract_icc_profile(pil_image),
            "exif": pil_image.info.get("exif"),
            "format": pil_image.format,
            "mode": pil_image.mode,
            "size": pil_image.size,
            "load_backend": "pil",
            "exif_orientation": None,
            "exif_orientation_applied": False,
            "exif_preservation_mode": "none",
        }

        # Detect bit depth FIRST (before any loading)
        detected_input_bits = detect_input_bit_depth(pil_image)
        image_format = pil_image.format

        # Extract EXIF orientation for tifffile path (needed later)
        exif_orientation = None
        try:
            if hasattr(pil_image, "getexif"):
                exif = pil_image.getexif()
                exif_orientation = exif.get(0x0112)  # Orientation tag
        except Exception as e:
            logger.debug(f"Could not extract EXIF orientation: {e}")
        metadata["exif_orientation"] = exif_orientation

    # For 16-bit TIFFs, use tifffile to load correctly
    if detected_input_bits == 16 and image_format == "TIFF":
        try:
            import tifffile

            # Load with tifffile which preserves 16-bit data
            image_array = tifffile.imread(input_path)

            # tifffile returns (H, W, C) for RGB or (H, W) for grayscale
            logger.debug(f"Loaded 16-bit TIFF with tifffile: " f"shape={image_array.shape}, dtype={image_array.dtype}")

            # Handle EXIF orientation for tifffile path
            # tifffile doesn't apply EXIF orientation automatically
            orientation_applied = bool(exif_orientation and exif_orientation != 1)
            if exif_orientation and exif_orientation != 1:
                image_array = _apply_exif_orientation_to_array(image_array, int(exif_orientation))
                logger.info(f"Applied EXIF orientation {exif_orientation} to 16-bit TIFF")

            with Image.open(input_path) as pil_image_for_exif:
                metadata["exif"] = _normalized_exif_after_orientation(pil_image_for_exif, orientation_applied)
            metadata["load_backend"] = "tifffile"
            metadata["exif_orientation_applied"] = orientation_applied
            if orientation_applied:
                metadata["exif_preservation_mode"] = "normalized"
            elif metadata.get("exif"):
                metadata["exif_preservation_mode"] = "full"

            # Ensure RGB format (H, W, 3)
            if image_array.ndim == 2:
                # Grayscale -> RGB
                image_array = np.stack([image_array] * 3, axis=2)
            elif image_array.ndim == 3 and image_array.shape[2] == 4:
                # RGBA -> extract RGB and alpha separately
                # Note: alpha will be handled later in the pipeline
                logger.debug("16-bit RGBA detected - will preserve alpha channel")

            return image_array, detected_input_bits, metadata

        except Exception as e:
            # Check Quality Firewall before falling back to PIL
            if detected_input_bits == 16 and not allow_8bit_output:
                raise V2EnhancementError(
                    f"Cannot load 16-bit TIFF with tifffile (error: {e}). "
                    f"Fallback to PIL would downconvert to 8-bit, blocked by Quality Firewall. "
                    f"Install tifffile correctly or use --allow-8bit to permit downgrade."
                )

            logger.warning(f"Failed to load 16-bit TIFF with tifffile: {e}. " f"Falling back to PIL (will convert to 8-bit)")
            # Fall through to PIL loading

    # Standard PIL loading for 8-bit or fallback
    with Image.open(input_path) as opened_image:
        # Handle EXIF orientation
        orientation_applied = bool(exif_orientation and exif_orientation != 1)
        oriented_image: Image.Image = ImageOps.exif_transpose(opened_image)

        # After applying exif_transpose, normalize EXIF to avoid double-rotation:
        # - pixels have been rotated already
        # - EXIF Orientation must not request additional rotation in viewers
        exif_data = _normalized_exif_after_orientation(oriented_image, orientation_applied)
        metadata["exif"] = exif_data
        metadata["load_backend"] = "pil"
        metadata["exif_orientation_applied"] = orientation_applied
        if orientation_applied:
            metadata["exif_preservation_mode"] = "normalized"
        elif exif_data:
            metadata["exif_preservation_mode"] = "full"

        # Handle palette mode
        if oriented_image.mode == "P":
            oriented_image = oriented_image.convert("RGB")

        # Handle LA mode
        if oriented_image.mode == "LA":
            oriented_image = oriented_image.convert("RGBA")

        image_array = np.array(oriented_image)

    # Track actual loaded bits
    actual_bits = detected_input_bits
    if detected_input_bits == 16:
        # We got here via PIL fallback - input was downconverted
        logger.warning(
            f"16-bit input was downconverted to 8-bit by PIL. " f"Original BitsPerSample=16, loaded as {image_array.dtype}"
        )
        actual_bits = 8

    logger.debug(f"Loaded image with PIL: " f"shape={image_array.shape}, dtype={image_array.dtype}")

    return image_array, actual_bits, metadata


def load_depth_map(depth_path: Path) -> np.ndarray:
    """Load and normalize depth map.

    Preserves 16-bit precision by using fixed-scale normalization for uint16 depth.

    Args:
        depth_path: Path to depth map image

    Returns:
        Normalized depth map [0, 1] as float32 (H, W)

    Raises:
        V2EnhancementError: If depth map cannot be loaded
    """
    try:
        with Image.open(depth_path) as opened_depth_image:
            depth_image: Image.Image = opened_depth_image

            # Handle 16-bit depth maps explicitly to preserve precision
            if depth_image.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
                # Convert to uint16 array and normalize via fixed scale
                depth_map = np.array(depth_image, dtype=np.uint16).astype(np.float32) / 65535.0
            else:
                # Convert to grayscale if needed
                if depth_image.mode not in ("L", "I", "F"):
                    depth_image = depth_image.convert("L")

                depth_map = np.array(depth_image, dtype=np.float32)

                # Normalize to [0, 1] - handle all-zeros case
                depth_max = depth_map.max()
                if depth_max > 1.0:
                    depth_map = depth_map / depth_max
                elif depth_max == 0.0:
                    # All-zeros depth map: return as-is (will be handled gracefully by enhancement)
                    logger.warning(f"Depth map {depth_path} is all zeros - depth effects will be skipped")

        return depth_map

    except Exception as e:
        raise V2EnhancementError(f"Failed to load depth map from {depth_path}: {e}") from e


def enhance_image(
    input_path: Path,
    output_path: Path,
    depth_map_path: Optional[Path] = None,
    material_masks: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[V2EnhancementConfig] = None,
    device: str = "cpu",
    allow_8bit_output: bool = False,
    output_bit_depth: Optional[int] = None,
    input_color: str = "auto",
    geometry_recipe: str = "legacy_encoded_crop_stretch_v1",
    geometry_handoff_path: Optional[Path] = None,
    masks_file: Optional[Path] = None,
    native_materials_binding_path: Optional[Path] = None,
    native_materials_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply V2 depth-aware enhancement to input image.

    Main entry point for V2 enhancement. Applies perceptual finishing
    using depth-aware tone mapping, clarity enhancement, and material-specific
    processing.

    **BIT-DEPTH PRESERVATION GUARANTEE:**
    - Omitted output_bit_depth preserves input precision
    - Explicit output_bit_depth selects 8-bit PNG or 16-bit TIFF
    - Processing always done in float32 [0,1] to preserve precision

    Args:
        input_path: Path to input image
        output_path: Path to output enhanced image
        depth_map_path: Optional path to depth map from V3 stage
        material_masks: Optional material segmentation masks
        config: Enhancement configuration (uses default if None)
        device: Processing device (cpu/cuda/mps) - currently only cpu supported
        allow_8bit_output: Allow 16-bit → 8-bit downgrade (Quality Firewall bypass)
        output_bit_depth: Explicit target encoding depth (8 or 16)

    Returns:
        Dict containing processing metadata:
            - status: "success" or "error"
            - input: Input image path
            - output: Output image path
            - depth_map: Depth map path (if provided)
            - depth_consumed: Whether depth was actually consumed by enhancement stage
            - preset: Preset name
            - runtime_s: Processing time in seconds
            - metadata: Enhancement metadata from stage
            - timestamp: Processing timestamp
            - bit_depth: Bit-depth metadata (input/output/conversion)

    Raises:
        V2EnhancementError: If enhancement fails or bit-depth violation
        FileNotFoundError: If input image not found
    """
    start_time = time.perf_counter()
    native_materials = native_materials_binding_path is not None
    processing_version = processing_version_for_geometry(geometry_recipe, native_materials=native_materials)
    corrected_geometry = geometry_recipe == "normalized_crop_support_v1"
    if corrected_geometry and (geometry_handoff_path is None or depth_map_path is None):
        raise V2EnhancementError("Corrected V2 processing requires explicit geometry handoff and depth artifacts")
    if not corrected_geometry and (geometry_handoff_path is not None or masks_file is not None):
        raise V2EnhancementError("Geometry handoff artifacts require the corrected recipe")
    if corrected_geometry and material_masks is not None:
        raise V2EnhancementError("Corrected V2 masks must come from the verified handoff, not bare arrays")
    if native_materials_source is not None and not native_materials:
        raise V2EnhancementError("Native Materials source selection requires its frozen binding")
    if native_materials and masks_file is not None:
        raise V2EnhancementError("Native Materials cannot compose legacy material-mask adjustments")
    native_binding = (
        load_native_materials_binding(native_materials_binding_path, source_path=native_materials_source)
        if native_materials_binding_path is not None
        else None
    )
    if native_binding is not None and native_binding["input_color"] != input_color:
        raise V2EnhancementError("Native Materials color policy differs from its frozen binding")

    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Use default config if not provided
    if config is None:
        config = V2EnhancementConfig()
    if corrected_geometry:
        config = replace(config, version=processing_version)

    logger.info(f"V2 Enhancement: {input_path.name} with preset '{config.preset}'")

    if output_bit_depth is not None and (
        isinstance(output_bit_depth, bool) or not isinstance(output_bit_depth, int) or output_bit_depth not in {8, 16}
    ):
        raise V2EnhancementError("output_bit_depth must be 8 or 16")

    passthrough_requested = config.preset == "none" or (
        config.enhancement_strength == 0.0 and config.clarity_strength == 0.0 and config.material_strength == 0.0
    )
    # Without an explicit encoding contract, retain the historical exact-copy
    # passthrough. An explicit depth must flow through the canonical encoder so
    # the emitted bytes, suffix, and manifest all agree.
    if passthrough_requested and output_bit_depth is None and not corrected_geometry:
        logger.info("Preset 'none' - skipping enhancement (passthrough)")
        # True passthrough: preserve metadata and pixel data exactly (no re-encoding)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return {
            "status": "passthrough",
            "implementation": "v2_enhance",
            "artifact_contract": "passthrough_exact_copy",
            "is_canonical_emitted_artifact": False,
            "output_naming_policy": "caller_path_exact",
            "input": str(input_path),
            "output": str(output_path),
            "depth_map": str(depth_map_path) if depth_map_path else None,
            "depth_consumed": False,
            "preset": config.preset,
            "runtime_s": time.perf_counter() - start_time,
            "timestamp": time.time(),
            "message": "Preset 'none' - enhancement skipped",
            # Structured depth resolution semantics
            "depth": {
                "requested": depth_map_path is not None,
                "resolved_path": str(depth_map_path) if depth_map_path else None,
                "loaded": False,
                "supplied_to_stage": False,
                "consumed": False,
                "consumption_source": "passthrough",
                "stage_has_depth": None,
            },
        }
    if passthrough_requested:
        logger.info("Preset 'none' - skipping adjustments and applying requested output encoding")

    try:
        geometry = None
        native_receipt = None
        source_snapshot = None
        master = None
        # Load input image with bit-depth preservation
        allow_8bit_load = output_bit_depth == 8 or (output_bit_depth is None and allow_8bit_output)
        try:
            if corrected_geometry:
                from .geometry_handoff import load_geometry_handoff, snapshot_geometry_artifact

                if geometry_handoff_path is None or depth_map_path is None:
                    raise V2EnhancementError("Corrected V2 requires explicit geometry handoff and depth artifacts")
                # Color pixels, metadata, alpha and source identity all derive
                # from the same bounded byte snapshot in this versioned path.
                source_snapshot = snapshot_geometry_artifact(input_path, 2 * 1024**3)
                master = decode_v2_source(source_snapshot, source_name=input_path.name, input_color=input_color)
                if master.source_bit_depth not in {8, 16}:
                    raise V2EnhancementError("Corrected V2 supports bounded 8/16-bit SDR sources only")
                input_bits = master.source_bit_depth
                # Keep one photographic float working buffer. Source precision
                # controls encoding; an unused integer RGB proxy would only
                # duplicate this full-resolution allocation.
                with Image.open(io.BytesIO(source_snapshot)) as header:
                    orientation = header.getexif().get(274, 1)
                    metadata = {
                        "icc_profile": master.source_icc,
                        "exif": _normalized_exif_after_orientation(header, orientation != 1),
                        "exif_orientation": orientation,
                        "exif_orientation_applied": orientation != 1,
                        "exif_preservation_mode": "normalized" if orientation != 1 else "full",
                        "load_backend": "verified_byte_snapshot",
                    }
                geometry = load_geometry_handoff(
                    geometry_handoff_path,
                    input_bytes=source_snapshot,
                    depth_map_path=depth_map_path,
                    masks_file=masks_file,
                    recipe=geometry_recipe,
                )
                if master.shape != geometry.native_shape:
                    raise V2EnhancementError("Photographic source differs from the bound native geometry")
                if native_binding is not None:
                    from .native_materials import apply_native_materials

                    if geometry.source_sha256 != master.source_sha256 or geometry.v2_input_sha256 != master.source_sha256:
                        raise V2EnhancementError("Native Materials requires the exact original photographic source")
                    if geometry.masks:
                        raise V2EnhancementError("Native Materials forbids a legacy mask inventory")
                    master, native_receipt = apply_native_materials(
                        master, native_binding, source_path=native_materials_source, support=geometry.support
                    )
                image = bounded_encoded_srgb(master)
            else:
                image, input_bits, metadata = load_image_preserve_bit_depth(input_path, allow_8bit_load)
        except (OSError, V2EnhancementError) as load_error:
            # Pillow cannot open ordinary RGB float TIFFs. Diagnose the native
            # header without attempting a lower-precision fallback or masking
            # the existing integer quality-firewall error.
            with input_path.open("rb") as header:
                is_tiff = header.read(4) in {b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"}
            if is_tiff:
                import tifffile

                with tifffile.TiffFile(input_path) as native:
                    native_dtype = native.pages[0].dtype
                    if native_dtype is not None and native_dtype.kind == "f":
                        raise V2EnhancementError(
                            "V3 finishing supports at most 16-bit integer masters; use Unified/V5/V6 to retain float precision"
                        ) from load_error
            raise
        input_dtype = str(np.dtype(np.uint16 if input_bits == 16 else np.uint8)) if corrected_geometry else str(image.dtype)
        # Pixel processing has one explicit color domain. Preserve the source
        # precision through the existing V4/V5 photographic color authority;
        # an ICC is a transform, never a label copied onto unrelated pixels.
        from transformation_portal.core.image_artifact import metadata_payload
        from transformation_portal.lux_depth_v4.photography import output_srgb_icc

        if source_snapshot is None:
            source_snapshot = input_path.read_bytes()
        if master is None:
            master = decode_v2_source(source_snapshot, source_name=input_path.name, input_color=input_color)
        prepared_pixels = image if corrected_geometry else bounded_encoded_srgb(master)
        color_preparation = metadata_payload(master.metadata["color_preparation"])
        icc_profile = metadata.get("icc_profile")
        source_icc_profile = icc_profile
        exif_data = metadata.get("exif")
        load_backend = str(metadata.get("load_backend") or "pil")
        source_had_icc = bool(icc_profile)
        source_had_exif = bool(exif_data or metadata.get("exif_orientation"))
        load_exif_preservation_mode = str(metadata.get("exif_preservation_mode") or "none")
        exif_preservation_mode = load_exif_preservation_mode
        save_backend = "unknown"
        save_degraded = False
        save_degradation_reason = None
        icc_preserved = False
        icc_profile = output_srgb_icc()

        logger.debug(f"Loaded image: shape={image.shape}, dtype={image.dtype}, " f"bits_per_sample={input_bits}")

        # Quality Firewall: Enforce bit-depth preservation
        # 16-bit input MUST produce 16-bit output unless explicitly allowed
        # Decide target dtype up front to ensure consistency throughout pipeline
        if output_bit_depth is not None:
            target_bits = int(output_bit_depth)
            target_dtype = np.dtype(np.uint16 if target_bits == 16 else np.uint8)
        elif input_bits == 16 and allow_8bit_output:
            target_dtype = np.dtype(np.uint8)
            target_bits = 8
            logger.warning("Quality Firewall BYPASSED: 16-bit → 8-bit downgrade allowed " "by --allow-8bit flag")
        else:
            target_dtype = np.dtype(input_dtype) if corrected_geometry else np.dtype(image.dtype)
            target_bits = input_bits
            if input_bits == 16:
                logger.info("Quality Firewall ACTIVE: 16-bit input detected - " "will preserve 16-bit output")

        if input_bits == 16 and target_bits == 8:
            save_degraded = True
            save_degradation_reason = "explicit_output_bit_depth" if output_bit_depth is not None else "allow_8bit_output"

        # Handle RGBA inputs: extract RGB, enhance, restore alpha
        alpha_channel = None
        if image.ndim == 3 and image.shape[2] == 4:
            logger.debug("RGBA input detected - extracting alpha channel for preservation")
            alpha_channel = _convert_alpha_to_target_dtype(image[:, :, 3], target_dtype)
            image = image[:, :, :3]  # RGB only for enhancement
        if master.alpha is not None:
            alpha_channel = _convert_alpha_to_target_dtype(master.alpha, target_dtype)

        # Enforce RGB (H, W, 3) contract for EnhancementStage
        if image.ndim == 2:
            # Grayscale -> RGB
            image = np.stack([image, image, image], axis=2)
        elif image.ndim == 3 and image.shape[2] != 3:
            raise V2EnhancementError(f"Unexpected image shape: {image.shape} (expected (H,W,3) RGB)")
        image = prepared_pixels

        # Load depth map if provided
        # NOTE: depth_aware_tone_mapping and atmospheric_effects config flags are
        # currently reserved for future use. The current implementation always applies
        # depth-aware tone mapping when a depth map is present. To disable depth effects,
        # simply don't provide a depth map or use preset="none".
        depth_map = None
        if geometry is not None:
            geometry = geometry.with_image(image, master.alpha)
            depth_map = geometry.depth_map
            # Preserve the optional-mask contract: an empty verified inventory
            # is not a Materials run and must retain the plain V2 output label.
            material_masks = dict(geometry.masks) or None
        elif not passthrough_requested and depth_map_path and depth_map_path.exists():
            depth_map = load_depth_map(depth_map_path)
            logger.debug(f"Loaded depth map: {depth_map.shape}")
        elif not passthrough_requested and depth_map_path:
            logger.warning(f"Depth map path provided but not found: {depth_map_path}")

        stage_metadata: Dict[str, Any] = {}
        enhancement_metadata: Dict[str, Any] = {}
        enhanced_image: np.ndarray
        if passthrough_requested:
            enhanced_image = image
        else:
            # Apply enhancement using existing EnhancementStage
            enhancer = EnhancementStage(
                enhancement_strength=config.enhancement_strength,
                clarity_strength=config.clarity_strength,
                material_strength=config.material_strength,
                version=processing_version if corrected_geometry else config.version,
                output_dtype=target_dtype,  # Use consistent target dtype
                tone_low_tex_strength=getattr(config, "tone_low_tex_strength", 0.6),
                tone_depth_smoothing=getattr(config, "tone_depth_smoothing", True),
                **({"geometry_recipe": geometry_recipe} if corrected_geometry else {}),
            )

            # Create minimal context for stage execution
            context = StageContext(device=device)
            context.set_artifact("image", image)

            if depth_map is not None:
                context.set_artifact("depth_map", depth_map)

            if material_masks:
                context.set_artifact("material_masks", material_masks)
            if geometry is not None:
                context.set_artifact("geometry_handoff", geometry)
            if native_receipt is not None:
                context.set_artifact(
                    "native_materials", {"binding": native_binding, "receipt": native_receipt, "master": master}
                )

            # Execute enhancement
            logger.debug("Executing EnhancementStage...")
            result = enhancer.compute(context)

            if result.status != StageStatus.COMPLETED:
                error_msg = result.error or "Unknown error"
                raise V2EnhancementError(f"Enhancement failed: {error_msg}")

            # Extract enhanced image
            raw_enhanced_image = result.artifacts.get("enhanced_image")
            if raw_enhanced_image is None:
                raise V2EnhancementError("EnhancementStage did not produce 'enhanced_image' artifact")
            if not isinstance(raw_enhanced_image, np.ndarray):
                raise V2EnhancementError("EnhancementStage produced a non-array 'enhanced_image' artifact")
            enhanced_image = raw_enhanced_image
            stage_metadata = result.metadata if isinstance(result.metadata, dict) else {}
            raw_enhancement_metadata = result.artifacts.get("enhancement_metadata", {})
            if isinstance(raw_enhancement_metadata, dict):
                enhancement_metadata = raw_enhancement_metadata

        # Ensure output dtype matches target (handle potential mismatches)
        if enhanced_image.dtype != target_dtype:
            logger.debug(f"Converting enhanced image from {enhanced_image.dtype} to {target_dtype}")
            if np.issubdtype(enhanced_image.dtype, np.floating):
                maximum = float(np.iinfo(np.uint16 if target_bits == 16 else np.uint8).max)
                enhanced_image = np.rint(np.clip(enhanced_image, 0, 1) * maximum).astype(target_dtype)
            elif target_dtype == np.dtype(np.uint8) and enhanced_image.dtype == np.uint16:
                # 16-bit → 8-bit conversion with proper normalization
                enhanced_image = (enhanced_image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            elif target_dtype == np.dtype(np.uint16) and enhanced_image.dtype == np.uint8:
                # 8-bit → 16-bit conversion (unusual but handle it)
                enhanced_image = (enhanced_image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
            else:
                raise V2EnhancementError(f"Unsupported dtype conversion: {enhanced_image.dtype} → {target_dtype}")

        # Ensure output directory exists
        output_path = resolve_v2_emitted_artifact_path(
            output_path,
            bit_depth=target_bits,
            materials_enabled=material_masks is not None,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save enhanced image with metadata preservation and correct bit-depth
        if target_bits == 16 and enhanced_image.dtype == np.uint16:
            # Save 16-bit TIFF
            try:
                import tifffile

                # Restore alpha channel if present
                if alpha_channel is not None:
                    logger.debug("Restoring alpha channel to 16-bit RGBA image")
                    # Check dimensions
                    if corrected_geometry and alpha_channel.shape[:2] != enhanced_image.shape[:2]:
                        raise V2EnhancementError("Corrected geometry forbids resizing photographic alpha")
                    if alpha_channel.shape[:2] != enhanced_image.shape[:2]:
                        logger.warning(
                            f"Alpha dimension mismatch: alpha={alpha_channel.shape[:2]} "
                            f"enhanced={enhanced_image.shape[:2]}. Resizing alpha."
                        )
                        from PIL import Image as PILImage

                        alpha_pil = PILImage.fromarray(alpha_channel)
                        alpha_resized = alpha_pil.resize(
                            (enhanced_image.shape[1], enhanced_image.shape[0]), PILImage.Resampling.LANCZOS
                        )
                        alpha_channel = np.array(alpha_resized, dtype=np.uint16)

                    enhanced_image = np.dstack([enhanced_image, alpha_channel])

                # Build extratags for ICC profile preservation.
                extratags = []

                # Preserve ICC profile (TIFF tag 34675)
                if icc_profile:
                    extratags.append((34675, "B", len(icc_profile), icc_profile, False))
                    icc_preserved = source_icc_profile == icc_profile
                    logger.debug("Preserving ICC profile in 16-bit TIFF output")

                if exif_data:
                    logger.debug(
                        "EXIF data present in source but not written by the 16-bit tifffile save path. "
                        "ICC profile is preserved separately when available."
                    )

                # Save with tifffile (preserves 16-bit)
                tifffile.imwrite(
                    output_path,
                    enhanced_image,
                    photometric="rgb",
                    compression="lzw",  # Lossless compression
                    extratags=extratags if extratags else None,
                )
                save_backend = "tifffile"
                if source_had_exif:
                    exif_preservation_mode = "none"
                logger.info(f"Saved 16-bit TIFF: {output_path}")

            except Exception as e:
                raise V2EnhancementError(
                    f"Cannot save requested 16-bit output with tifffile (error: {e}); "
                    "publishing an 8-bit file under a 16-bit contract is forbidden"
                ) from e

        else:
            # Save 8-bit with PIL (standard path)
            # Restore alpha channel if present
            if alpha_channel is not None:
                logger.debug("Restoring alpha channel to enhanced RGB image")
                # Check dimensions
                if corrected_geometry and alpha_channel.shape[:2] != enhanced_image.shape[:2]:
                    raise V2EnhancementError("Corrected geometry forbids resizing photographic alpha")
                if alpha_channel.shape[:2] != enhanced_image.shape[:2]:
                    from PIL import Image as PILImage

                    logger.warning(
                        f"Alpha dimension mismatch: alpha={alpha_channel.shape[:2]} "
                        f"enhanced={enhanced_image.shape[:2]}. Resizing alpha."
                    )
                    alpha_pil = PILImage.fromarray(alpha_channel, mode="L")
                    alpha_resized = alpha_pil.resize(
                        (enhanced_image.shape[1], enhanced_image.shape[0]), PILImage.Resampling.LANCZOS
                    )
                    alpha_channel = np.array(alpha_resized)

                enhanced_image = np.dstack([enhanced_image, alpha_channel])

            output_image = Image.fromarray(enhanced_image)

            # Preserve ICC profile and EXIF if present
            save_kwargs: Dict[str, Any] = {}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
                icc_preserved = source_icc_profile == icc_profile
                logger.debug("Preserving ICC color profile")

            # Preserve EXIF data if present
            if exif_data:
                save_kwargs["exif"] = exif_data
                if load_exif_preservation_mode in {"normalized", "partial"}:
                    exif_preservation_mode = load_exif_preservation_mode
                else:
                    exif_preservation_mode = "full"
                logger.debug("Preserving EXIF metadata")
            elif source_had_exif:
                exif_preservation_mode = "none"

            output_image.save(output_path, **save_kwargs)
            save_backend = "pil"
            logger.info(f"Saved enhanced image: {output_path}")

        runtime_s = time.perf_counter() - start_time

        stage_has_depth = stage_metadata.get("has_depth") if isinstance(stage_metadata, dict) else None
        exif_orientation_normalized = bool(metadata.get("exif_orientation_applied"))

        # Determine depth consumption semantics
        if passthrough_requested:
            depth_consumed = False
            consumption_source = "passthrough"
        elif stage_has_depth is not None:
            depth_consumed = bool(stage_has_depth)
            consumption_source = "stage_metadata"
        elif depth_map is not None:
            depth_consumed = True
            consumption_source = "fallback_input_presence"
        else:
            depth_consumed = False
            consumption_source = "not_found" if depth_map_path else "not_requested"

        effective_exif_preserved = exif_preservation_mode == "full"
        if not source_had_icc and not source_had_exif:
            metadata_preservation_mode = "none"
        elif (not source_had_icc or icc_preserved) and (not source_had_exif or effective_exif_preserved):
            metadata_preservation_mode = "full"
        elif icc_preserved or exif_preservation_mode in {"full", "normalized", "partial"}:
            metadata_preservation_mode = "partial"
        else:
            metadata_preservation_mode = "none"

        # Build metadata report with bit-depth information
        return {
            "status": "success",
            "implementation": "v2_enhance",
            "processing_version": processing_version,
            **({"native_materials": native_receipt} if native_receipt is not None else {}),
            **(
                {
                    "geometry": {
                        "recipe": geometry_recipe,
                        "handoff_identity_sha256": geometry.identity_sha256,
                        "handoff": metadata_payload(geometry.handoff),
                        "source_sha256": geometry.source_sha256,
                        "v2_input_sha256": geometry.v2_input_sha256,
                        "shape": list(geometry.native_shape),
                    }
                }
                if geometry is not None
                else {}
            ),
            "color": {
                "contract": V2_COLOR_CONTRACT,
                "input_policy": input_color,
                "preparation": color_preparation,
                "working_color": "bounded_encoded_srgb",
                "output_color": "srgb",
                "quantization": "at_final_output",
                "source_icc_preserved": icc_preserved,
                "output_icc_sha256": hashlib.sha256(icc_profile).hexdigest(),
            },
            "artifact_contract": "canonical_v2_emitted_artifact",
            "is_canonical_emitted_artifact": True,
            "output_naming_policy": "canonical_v2_emitted_artifact",
            "input": str(input_path),
            "output": str(output_path),
            "depth_map": str(depth_map_path) if depth_map_path else None,
            "depth_consumed": depth_consumed,
            "preset": config.preset,
            "config": config.to_dict(),
            "runtime_s": runtime_s,
            "timestamp": time.time(),
            "stage_metadata": stage_metadata,
            "enhancement_metadata": enhancement_metadata,
            "io": {
                "load_backend": load_backend,
                "save_backend": save_backend,
                "metadata_preservation_mode": metadata_preservation_mode,
                "icc_preserved": icc_preserved,
                "exif_preservation_mode": exif_preservation_mode,
                "exif_orientation_normalized": exif_orientation_normalized,
                "source_exif_orientation": metadata.get("exif_orientation"),
                "save_degraded": save_degraded,
                "save_degradation_reason": save_degradation_reason,
            },
            # BIT-DEPTH METADATA (Quality Firewall contract)
            "bit_depth": {
                "input_bits_per_sample": input_bits,
                "output_bits_per_sample": target_bits,
                "input_dtype": input_dtype,
                "output_dtype": str(enhanced_image.dtype),
                "quality_firewall_active": input_bits == 16 and target_bits == 16,
                "bit_depth_preserved": input_bits == target_bits,
                "downgrade_allowed": bool(
                    input_bits == 16 and target_bits == 8 and (allow_8bit_output or output_bit_depth == 8)
                ),
            },
            # STRUCTURED DEPTH RESOLUTION SEMANTICS
            "depth": {
                "requested": depth_map_path is not None,
                "resolved_path": str(depth_map_path) if depth_map_path else None,
                "loaded": depth_map is not None,
                "supplied_to_stage": depth_map is not None,
                "consumed": depth_consumed,
                "consumption_source": consumption_source,
                "stage_has_depth": stage_has_depth,
            },
        }

    except V2EnhancementError:
        raise
    except Exception as e:
        runtime_s = time.perf_counter() - start_time
        logger.exception("V2 enhancement failed")
        raise V2EnhancementError(f"Enhancement failed after {runtime_s:.2f}s: {e}") from e
