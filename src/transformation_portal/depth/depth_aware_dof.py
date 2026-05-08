"""Depth-aware architectural DOF post-processing.

This module is intentionally separate from the legacy ``depth_tools`` batch
helpers because this workflow must preserve float depth and 16-bit TIFF data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw

from transformation_portal.ingest.canonical_json import dumps_json

from .tools import gaussian_blur_float

try:
    import tifffile

    _TIFFFILE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in stripped envs
    tifffile = None  # type: ignore[assignment]
    _TIFFFILE_AVAILABLE = False

DepthConvention = Literal["higher-is-farther", "lower-is-farther"]

_LOG = logging.getLogger(__name__)
_VALID_CONVENTIONS: set[DepthConvention] = {"higher-is-farther", "lower-is-farther"}
_TIFF_EXTENSIONS = {".tif", ".tiff"}
_HAZE_COLOR = np.array([0.94, 0.96, 0.99], dtype=np.float32)


@dataclass(frozen=True)
class DepthAwareDofOptions:
    """Options for one depth-aware DOF render."""

    source: Path
    depth_npy: Path
    out_dir: Path
    metadata: Optional[Path] = None
    protect_mask: Optional[Path] = None
    sky_mask: Optional[Path] = None
    edge_mask: Optional[Path] = None
    focus_depth: Optional[float] = None
    focus_roi: Optional[tuple[int, int, int, int]] = None
    depth_convention: Optional[DepthConvention] = None
    preview_long_edge: int = 2400
    near_strength: float = 0.34
    far_strength: float = 0.24
    haze_strength: float = 0.08
    edge_protection: float = 0.70
    focus_protection: float = 0.88


@dataclass(frozen=True)
class DepthAwareDofResult:
    """Artifact paths and selected parameters from one DOF render."""

    production_tiff: Path
    preview_jpeg: Path
    diagnostic_contact_sheet: Path
    summary_json: Path
    package_zip: Path
    focus_depth: float
    depth_convention: DepthConvention
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class _SourceImage:
    path: Path
    rgb01: np.ndarray
    bit_depth: int
    dtype: str


@dataclass(frozen=True)
class _DofIntermediates:
    focus_matte: np.ndarray
    near_matte: np.ndarray
    far_matte: np.ndarray
    edge_matte: np.ndarray
    protection_matte: np.ndarray
    blur_matte: np.ndarray


def run_depth_aware_dof(options: DepthAwareDofOptions) -> DepthAwareDofResult:
    """Run a single-image float-depth DOF composite."""

    opts = _normalize_options(options)
    metadata = _load_metadata(opts.metadata)
    convention = _resolve_depth_convention(metadata, opts.depth_convention)
    source = _load_source_image(opts.source)
    depth = _load_depth_npy(opts.depth_npy)
    height, width = source.rgb01.shape[:2]
    if depth.shape != (height, width):
        raise ValueError(
            "Depth/source dimension mismatch: "
            f"depth={depth.shape}, source={(height, width)} for {opts.depth_npy} and {opts.source}"
        )

    focus_depth = _select_focus_depth(depth, opts.focus_depth, opts.focus_roi)
    sky_mask = _load_optional_mask(opts.sky_mask, (height, width))
    explicit_edge_mask = _load_optional_mask(opts.edge_mask, (height, width))
    explicit_protect_mask = _load_optional_mask(opts.protect_mask, (height, width))

    composite, intermediates = _composite_depth_dof(
        source.rgb01,
        depth,
        convention=convention,
        focus_depth=focus_depth,
        sky_mask=sky_mask,
        edge_mask=explicit_edge_mask,
        protect_mask=explicit_protect_mask,
        options=opts,
    )

    opts.out_dir.mkdir(parents=True, exist_ok=True)
    stem = opts.source.stem
    production_tiff = opts.out_dir / f"{stem}_depth_aware_DOF_{source.bit_depth}bit.tiff"
    preview_jpeg = opts.out_dir / f"{stem}_depth_aware_DOF_preview_{opts.preview_long_edge}px.jpg"
    diagnostic_contact_sheet = opts.out_dir / f"{stem}_depth_dof_diagnostic_contact_sheet.jpg"
    summary_json = opts.out_dir / f"{stem}_depth_dof_summary.json"
    package_zip = opts.out_dir / f"{stem}_depth_dof_package.zip"

    _save_tiff_preserve_depth(production_tiff, composite, bit_depth=source.bit_depth)
    _save_preview(preview_jpeg, composite, long_edge=opts.preview_long_edge)
    _save_contact_sheet(
        diagnostic_contact_sheet,
        source.rgb01,
        depth,
        intermediates,
        focus_depth=focus_depth,
        convention=convention,
    )

    artifact_hashes = {
        "production_tiff": _sha256_file(production_tiff),
        "preview_jpeg": _sha256_file(preview_jpeg),
        "diagnostic_contact_sheet": _sha256_file(diagnostic_contact_sheet),
    }
    summary = _build_summary(
        opts,
        source,
        depth,
        metadata=metadata,
        convention=convention,
        focus_depth=focus_depth,
        artifacts={
            "production_tiff": production_tiff,
            "preview_jpeg": preview_jpeg,
            "diagnostic_contact_sheet": diagnostic_contact_sheet,
            "summary_json": summary_json,
            "package_zip": package_zip,
        },
        artifact_hashes=artifact_hashes,
    )
    summary_json.write_text(dumps_json(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    _write_package_zip(
        package_zip,
        {
            production_tiff.name: production_tiff,
            preview_jpeg.name: preview_jpeg,
            diagnostic_contact_sheet.name: diagnostic_contact_sheet,
            summary_json.name: summary_json,
        },
    )

    artifact_hashes = {
        **artifact_hashes,
        "summary_json": _sha256_file(summary_json),
        "package_zip": _sha256_file(package_zip),
    }
    return DepthAwareDofResult(
        production_tiff=production_tiff,
        preview_jpeg=preview_jpeg,
        diagnostic_contact_sheet=diagnostic_contact_sheet,
        summary_json=summary_json,
        package_zip=package_zip,
        focus_depth=focus_depth,
        depth_convention=convention,
        artifact_hashes=artifact_hashes,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-aware-dof",
        description="Single-image depth-aware DOF using float .npy depth and 16-bit TIFF preservation.",
    )
    parser.add_argument("--source", required=True, type=Path, help="Source RGB image, preferably 16-bit TIFF.")
    parser.add_argument("--depth-npy", required=True, type=Path, help="Float32 depth .npy aligned to the source image.")
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Output directory for TIFF, preview, diagnostics, JSON, ZIP."
    )
    parser.add_argument("--metadata", type=Path, default=None, help="Optional depth metadata JSON.")
    parser.add_argument("--protect-mask", type=Path, default=None, help="Optional grayscale architecture protection mask.")
    parser.add_argument("--sky-mask", type=Path, default=None, help="Optional grayscale sky/horizon mask.")
    parser.add_argument("--edge-mask", type=Path, default=None, help="Optional grayscale edge protection mask.")
    parser.add_argument("--focus-depth", type=float, default=None, help="Explicit focus depth in source depth units.")
    parser.add_argument(
        "--focus-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Focus ROI used to select median focus depth when --focus-depth is omitted.",
    )
    parser.add_argument(
        "--depth-convention",
        choices=sorted(_VALID_CONVENTIONS),
        default=None,
        help="Required when metadata does not provide stats.convention.",
    )
    parser.add_argument("--preview-long-edge", type=int, default=2400)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    options = DepthAwareDofOptions(
        source=args.source,
        depth_npy=args.depth_npy,
        out_dir=args.out_dir,
        metadata=args.metadata,
        protect_mask=args.protect_mask,
        sky_mask=args.sky_mask,
        edge_mask=args.edge_mask,
        focus_depth=args.focus_depth,
        focus_roi=tuple(args.focus_roi) if args.focus_roi is not None else None,
        depth_convention=args.depth_convention,
        preview_long_edge=args.preview_long_edge,
    )
    try:
        result = run_depth_aware_dof(options)
    except Exception as exc:  # pragma: no cover - shell behavior
        _LOG.error("depth-aware DOF failed: %s", exc)
        return 2

    print(
        dumps_json(
            {
                "production_tiff": str(result.production_tiff),
                "preview_jpeg": str(result.preview_jpeg),
                "diagnostic_contact_sheet": str(result.diagnostic_contact_sheet),
                "summary_json": str(result.summary_json),
                "package_zip": str(result.package_zip),
                "focus_depth": result.focus_depth,
                "depth_convention": result.depth_convention,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _normalize_options(options: DepthAwareDofOptions) -> DepthAwareDofOptions:
    if options.preview_long_edge <= 0:
        raise ValueError("preview_long_edge must be positive")
    for field_name in ("source", "depth_npy"):
        path = getattr(options, field_name)
        if not path.exists():
            raise FileNotFoundError(f"{field_name.replace('_', ' ')} not found: {path}")
        if not path.is_file():
            raise ValueError(f"{field_name.replace('_', ' ')} is not a file: {path}")
    optional_inputs = (
        ("metadata", options.metadata),
        ("protect mask", options.protect_mask),
        ("sky mask", options.sky_mask),
        ("edge mask", options.edge_mask),
    )
    for field_name, path in optional_inputs:
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{field_name} not found: {path}")
        if path is not None and not path.is_file():
            raise ValueError(f"{field_name} is not a file: {path}")
    if options.focus_depth is not None and not np.isfinite(options.focus_depth):
        raise ValueError("focus_depth must be finite")
    if options.depth_convention is not None and options.depth_convention not in _VALID_CONVENTIONS:
        raise ValueError(f"Unsupported depth convention: {options.depth_convention!r}")
    return DepthAwareDofOptions(
        source=Path(options.source),
        depth_npy=Path(options.depth_npy),
        out_dir=Path(options.out_dir),
        metadata=Path(options.metadata) if options.metadata is not None else None,
        protect_mask=Path(options.protect_mask) if options.protect_mask is not None else None,
        sky_mask=Path(options.sky_mask) if options.sky_mask is not None else None,
        edge_mask=Path(options.edge_mask) if options.edge_mask is not None else None,
        focus_depth=options.focus_depth,
        focus_roi=options.focus_roi,
        depth_convention=options.depth_convention,
        preview_long_edge=options.preview_long_edge,
        near_strength=float(options.near_strength),
        far_strength=float(options.far_strength),
        haze_strength=float(options.haze_strength),
        edge_protection=float(options.edge_protection),
        focus_protection=float(options.focus_protection),
    )


def _load_metadata(metadata_path: Optional[Path]) -> dict[str, Any]:
    if metadata_path is None:
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata JSON must be an object: {metadata_path}")
    return payload


def _resolve_depth_convention(metadata: Mapping[str, Any], explicit: Optional[str]) -> DepthConvention:
    if explicit is not None:
        return _normalize_depth_convention(explicit)
    raw = None
    stats = metadata.get("stats")
    if isinstance(stats, Mapping):
        raw = stats.get("convention")
    if raw is None:
        raw = metadata.get("convention")
    if raw is None:
        raise ValueError("Depth convention is required when metadata does not provide stats.convention")
    return _normalize_depth_convention(str(raw))


def _normalize_depth_convention(value: str) -> DepthConvention:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "higher-is-farther": "higher-is-farther",
        "higher-farther": "higher-is-farther",
        "farther-is-higher": "higher-is-farther",
        "lower-is-farther": "lower-is-farther",
        "lower-farther": "lower-is-farther",
        "farther-is-lower": "lower-is-farther",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported depth convention: {value!r}")
    return resolved  # type: ignore[return-value]


def _load_source_image(source_path: Path) -> _SourceImage:
    suffix = source_path.suffix.lower()
    if suffix in _TIFF_EXTENSIONS:
        if not _TIFFFILE_AVAILABLE:
            raise ImportError("tifffile is required to preserve TIFF bit depth")
        arr = np.asarray(tifffile.imread(source_path))
    else:
        arr = np.asarray(Image.open(source_path).convert("RGB"))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported source image shape: {arr.shape}")

    bit_depth: int
    if np.issubdtype(arr.dtype, np.uint16):
        bit_depth = 16
        rgb01 = arr.astype(np.float32) / 65535.0
    elif np.issubdtype(arr.dtype, np.uint8):
        bit_depth = 8
        rgb01 = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        bit_depth = 32
        rgb01 = np.asarray(arr, dtype=np.float32)
        if float(np.nanmax(rgb01)) > 1.0:
            rgb01 = rgb01 / max(float(np.nanmax(rgb01)), 1.0)
    else:
        raise ValueError(f"Unsupported source image dtype: {arr.dtype}")

    if not np.isfinite(rgb01).all():
        raise ValueError(f"Source image contains non-finite values: {source_path}")
    return _SourceImage(path=source_path, rgb01=np.clip(rgb01, 0.0, 1.0), bit_depth=bit_depth, dtype=str(arr.dtype))


def _load_depth_npy(depth_path: Path) -> np.ndarray:
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Depth .npy must be a 2D array, got shape {depth.shape}")
    if depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise ValueError(f"Depth .npy must be non-empty with positive height and width, got shape {depth.shape}")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"Depth .npy must use a floating dtype, got {depth.dtype}")
    depth32 = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(depth32).all():
        raise ValueError(f"Depth .npy contains non-finite values: {depth_path}")
    return depth32


def _load_optional_mask(mask_path: Optional[Path], target_shape: tuple[int, int]) -> np.ndarray:
    if mask_path is None:
        return np.zeros(target_shape, dtype=np.float32)
    with Image.open(mask_path) as mask_image:
        raw = np.asarray(mask_image)
    raw_dtype = raw.dtype
    arr = raw
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., 3]
        else:
            arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    if max_value > 1.0:
        if np.issubdtype(raw_dtype, np.integer):
            dtype_info = np.iinfo(raw_dtype)
            arr = arr / float(dtype_info.max)
        else:
            arr = arr / max_value
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if arr.shape != target_shape:
        arr = _resize_mask(arr, target_shape)
    return arr.astype(np.float32, copy=False)


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    resized = Image.fromarray(mask.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _select_focus_depth(depth: np.ndarray, explicit: Optional[float], roi: Optional[tuple[int, int, int, int]]) -> float:
    if explicit is not None:
        return float(explicit)
    height, width = depth.shape
    if roi is None:
        roi_width = max(1, int(width * 0.38))
        roi_height = max(1, int(height * 0.38))
        x0 = (width - roi_width) // 2
        y0 = (height - roi_height) // 2
        roi = (x0, y0, roi_width, roi_height)
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError(f"focus_roi width/height must be positive: {roi}")
    x0 = max(0, min(width, int(x)))
    y0 = max(0, min(height, int(y)))
    x1 = max(0, min(width, x0 + int(w)))
    y1 = max(0, min(height, y0 + int(h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"focus_roi does not intersect the image: {roi}")
    return float(np.median(depth[y0:y1, x0:x1]))


def _composite_depth_dof(
    img: np.ndarray,
    depth: np.ndarray,
    *,
    convention: DepthConvention,
    focus_depth: float,
    sky_mask: np.ndarray,
    edge_mask: np.ndarray,
    protect_mask: np.ndarray,
    options: DepthAwareDofOptions,
) -> tuple[np.ndarray, _DofIntermediates]:
    oriented_depth = depth if convention == "higher-is-farther" else -depth
    oriented_focus = focus_depth if convention == "higher-is-farther" else -focus_depth
    p01, p99 = np.percentile(oriented_depth, [1.0, 99.0])
    depth_span = max(float(p99 - p01), 1e-6)
    near_span = max(float(oriented_focus - p01), depth_span * 0.12, 1e-6)
    far_span = max(float(p99 - oriented_focus), depth_span * 0.12, 1e-6)
    focus_band = max(depth_span * 0.13, 1e-6)

    near_matte = _smoothstep(np.maximum(oriented_focus - oriented_depth, 0.0) / near_span)
    far_matte = _smoothstep(np.maximum(oriented_depth - oriented_focus, 0.0) / far_span)
    focus_matte = 1.0 - _smoothstep(np.abs(oriented_depth - oriented_focus) / focus_band)

    generated_edge = _generate_edge_matte(img) if not np.any(edge_mask) else edge_mask
    base_protection = np.maximum(protect_mask, focus_matte * options.focus_protection)
    edge_protection = np.clip(generated_edge * options.edge_protection, 0.0, 1.0)
    protection_matte = np.clip(np.maximum(base_protection, edge_protection), 0.0, 1.0)

    raw_blur_matte = np.maximum(near_matte * options.near_strength, far_matte * options.far_strength)
    raw_blur_matte *= 1.0 - protection_matte
    blur_matte = np.clip(gaussian_blur_float(raw_blur_matte.astype(np.float32), sigma=1.2), 0.0, 1.0)

    blur_small = gaussian_blur_float(img, sigma=1.2)
    blur_medium = gaussian_blur_float(img, sigma=3.2)
    blur_large = gaussian_blur_float(img, sigma=6.5)
    far_selector = far_matte[..., None]
    near_selector = near_matte[..., None]
    blur_layer = (
        blur_small * (1.0 - np.maximum(far_selector, near_selector) * 0.35)
        + blur_medium * np.maximum(far_selector, near_selector) * 0.35
    )
    blur_layer = blur_layer * (1.0 - far_selector * 0.28) + blur_large * (far_selector * 0.28)

    matte3 = blur_matte[..., None]
    out = img * (1.0 - matte3) + blur_layer * matte3

    haze_matte = np.clip(far_matte * (1.0 - protection_matte) + sky_mask * 0.18, 0.0, 1.0)[..., None]
    haze_weight = haze_matte * float(options.haze_strength)
    hazed = out * (1.0 - haze_weight) + _HAZE_COLOR[None, None, :] * haze_weight
    decontrasted = 0.5 + (hazed - 0.5) * (1.0 - haze_matte * 0.055)
    out = hazed * (1.0 - haze_matte * 0.45) + decontrasted * (haze_matte * 0.45)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    return out, _DofIntermediates(
        focus_matte=focus_matte.astype(np.float32),
        near_matte=near_matte.astype(np.float32),
        far_matte=far_matte.astype(np.float32),
        edge_matte=generated_edge.astype(np.float32),
        protection_matte=protection_matte.astype(np.float32),
        blur_matte=blur_matte.astype(np.float32),
    )


def _generate_edge_matte(img: np.ndarray) -> np.ndarray:
    luminance = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    gy, gx = np.gradient(luminance.astype(np.float32))
    magnitude = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.percentile(magnitude, 95.0))
    if p95 <= 1e-8:
        return np.zeros(luminance.shape, dtype=np.float32)
    edge = _smoothstep(magnitude / p95)
    return np.clip(gaussian_blur_float(edge.astype(np.float32), sigma=0.9), 0.0, 1.0)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(values.astype(np.float32), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _save_tiff_preserve_depth(path: Path, rgb01: np.ndarray, *, bit_depth: int) -> None:
    if not _TIFFFILE_AVAILABLE:
        raise ImportError("tifffile is required to write production TIFF output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 16:
        payload = np.rint(np.clip(rgb01, 0.0, 1.0) * 65535.0).astype(np.uint16)
    elif bit_depth == 32:
        payload = np.clip(rgb01, 0.0, 1.0).astype(np.float32)
    else:
        payload = np.rint(np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    tifffile.imwrite(path, payload, photometric="rgb", compression="deflate")


def _save_preview(path: Path, rgb01: np.ndarray, *, long_edge: int) -> None:
    image = _rgb01_to_pil(rgb01)
    width, height = image.size
    max_edge = max(width, height)
    if max_edge > long_edge:
        scale = long_edge / float(max_edge)
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    image.save(path, quality=94, optimize=True)


def _save_contact_sheet(
    path: Path,
    source: np.ndarray,
    depth: np.ndarray,
    intermediates: _DofIntermediates,
    *,
    focus_depth: float,
    convention: DepthConvention,
) -> None:
    panels = [
        ("source", _rgb01_to_pil(source)),
        ("depth", _mask_to_pil(_normalize_for_display(depth))),
        ("focus matte", _mask_to_pil(intermediates.focus_matte)),
        ("near blur", _mask_to_pil(intermediates.near_matte)),
        ("far blur", _mask_to_pil(intermediates.far_matte)),
        ("edge protect", _mask_to_pil(intermediates.edge_matte)),
        ("subject protect", _mask_to_pil(intermediates.protection_matte)),
        ("final blur matte", _mask_to_pil(intermediates.blur_matte)),
    ]
    thumb_w = 320
    thumb_h = 214
    label_h = 28
    cols = 4
    rows = 2
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"depth-aware DOF diagnostics | focus={focus_depth:.4g} | {convention}"
    draw.text((8, 6), title, fill=(20, 20, 20))
    y_offset = 30
    for index, (label, panel) in enumerate(panels):
        col = index % cols
        row = index // cols
        x = col * thumb_w
        y = y_offset + row * (thumb_h + label_h)
        thumb = panel.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + thumb_h + 7), label, fill=(20, 20, 20))
    sheet.save(path, quality=92, optimize=True)


def _rgb01_to_pil(rgb01: np.ndarray) -> Image.Image:
    payload = np.rint(np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(payload, mode="RGB")


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    payload = np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(payload, mode="L").convert("RGB")


def _normalize_for_display(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    p01, p99 = np.percentile(finite, [1.0, 99.0])
    if p99 <= p01:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - p01) / (p99 - p01), 0.0, 1.0).astype(np.float32)


def _build_summary(
    options: DepthAwareDofOptions,
    source: _SourceImage,
    depth: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    convention: DepthConvention,
    focus_depth: float,
    artifacts: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    return {
        "schema": "tp.depth_aware_dof.v1",
        "inputs": {
            "source": str(options.source),
            "depth_npy": str(options.depth_npy),
            "metadata": str(options.metadata) if options.metadata else None,
            "protect_mask": str(options.protect_mask) if options.protect_mask else None,
            "sky_mask": str(options.sky_mask) if options.sky_mask else None,
            "edge_mask": str(options.edge_mask) if options.edge_mask else None,
        },
        "source": {
            "shape": list(source.rgb01.shape),
            "dtype": source.dtype,
            "bit_depth": source.bit_depth,
        },
        "depth": {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "convention": convention,
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "max": float(np.max(finite)),
            "percentile_1": float(np.percentile(finite, 1.0)),
            "percentile_99": float(np.percentile(finite, 99.0)),
            "metadata_model": metadata.get("model"),
        },
        "parameters": {
            "focus_depth": float(focus_depth),
            "focus_roi": list(options.focus_roi) if options.focus_roi else None,
            "near_strength": float(options.near_strength),
            "far_strength": float(options.far_strength),
            "haze_strength": float(options.haze_strength),
            "edge_protection": float(options.edge_protection),
            "focus_protection": float(options.focus_protection),
            "preview_long_edge": int(options.preview_long_edge),
        },
        "outputs": {key: _artifact_summary(path, artifact_hashes.get(key)) for key, path in artifacts.items()},
    }


def _artifact_summary(path: Path, sha256: Optional[str]) -> dict[str, str]:
    payload = {"path": str(path)}
    if sha256 is not None:
        payload["sha256"] = sha256
    return payload


def _write_package_zip(package_zip: Path, members: Mapping[str, Path]) -> None:
    with zipfile.ZipFile(package_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, path in members.items():
            archive.write(path, arcname=archive_name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
