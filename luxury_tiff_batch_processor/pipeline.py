"""Core processing helpers shared between the CLI and integrations."""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image

from .adjustments import AdjustmentSettings, apply_adjustments
from .io_utils import (
    ProcessingContext,
    float_to_dtype_array,
    image_to_float,
    save_image,
)
from .profiles import DEFAULT_PROFILE_NAME, PROCESSING_PROFILES, ProcessingProfile


def _ensure_profile(profile: ProcessingProfile | None) -> ProcessingProfile:
    if profile is None:
        return PROCESSING_PROFILES[DEFAULT_PROFILE_NAME]
    return profile


try:  # Optional progress bar for batch runs
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

LOGGER = logging.getLogger("luxury_tiff_batch_processor")
WORKER_LOGGER = LOGGER.getChild("worker")


def _tqdm_progress(
    iterable: Iterable[object], *, total: Optional[int], description: Optional[str]
) -> Iterable[object]:
    """Wrap *iterable* with :mod:`tqdm` if available."""

    if _tqdm is None:  # pragma: no cover - defensive fallback
        return iterable
    return _tqdm(iterable, total=total, desc=description, unit="image")


_PROGRESS_WRAPPER = _tqdm_progress if _tqdm is not None else None


def _wrap_with_progress(
    iterable: Iterable[Path],
    *,
    total: Optional[int],
    description: str,
    enabled: bool,
) -> Iterable[Path]:
    """Return an iterable wrapped with a progress helper when available."""

    if not enabled:
        return iterable

    helper = _PROGRESS_WRAPPER
    if helper is None:
        LOGGER.debug(
            "Progress helper not available; install tqdm for progress reporting."
        )
        return iterable

    try:
        return helper(iterable, total=total, description=description)
    except Exception:  # pragma: no cover - defensive fallback
        LOGGER.exception("Progress helper failed; continuing without progress display.")
        return iterable


def collect_images(folder: Path, recursive: bool) -> Iterator[Path]:
    patterns: List[str] = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    if recursive:
        for pattern in patterns:
            yield from folder.rglob(pattern)
    else:
        for pattern in patterns:
            yield from folder.glob(pattern)


def ensure_output_path(
    input_root: Path,
    output_root: Path,
    source: Path,
    suffix: str,
    recursive: bool,
    *,
    create: bool = True,
) -> Path:
    relative = source.relative_to(input_root) if recursive else Path(source.name)
    destination = output_root / relative
    if create:
        destination.parent.mkdir(parents=True, exist_ok=True)
    new_name = destination.stem + suffix + destination.suffix
    return destination.with_name(new_name)


def resize_bilinear(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    height, width = arr.shape[:2]
    if width == new_width and height == new_height:
        return arr
    x = np.linspace(0, width - 1, new_width, dtype=np.float32)
    y = np.linspace(0, height - 1, new_height, dtype=np.float32)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x_weight = (x - x0).astype(np.float32).reshape(1, -1, 1)
    y_weight = (y - y0).astype(np.float32).reshape(-1, 1, 1)

    Ia = arr[np.ix_(y0, x0)]
    Ib = arr[np.ix_(y0, x1)]
    Ic = arr[np.ix_(y1, x0)]
    Id = arr[np.ix_(y1, x1)]

    top = Ia * (1.0 - x_weight) + Ib * x_weight
    bottom = Ic * (1.0 - x_weight) + Id * x_weight
    return (top * (1.0 - y_weight) + bottom * y_weight).astype(np.float32)


def resize_array(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    if arr.ndim == 2:
        expanded = arr[:, :, None]
        resized = resize_bilinear(expanded, new_width, new_height)
        return resized[:, :, 0]
    if arr.ndim == 3:
        return resize_bilinear(arr, new_width, new_height)
    raise ValueError("Unsupported array shape for resizing")


def resize_long_edge_array(arr: np.ndarray, target: int) -> np.ndarray:
    height, width = arr.shape[:2]
    long_edge = max(width, height)
    if long_edge <= target:
        return arr
    scale = target / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    LOGGER.debug("Resizing from %sx%s to %s", width, height, (new_width, new_height))
    return resize_array(arr, new_width, new_height)


def _coerce_resize_target(
    resize_long_edge: Optional[int], resize_target: Optional[int]
) -> Optional[int]:
    """Normalise legacy ``resize_target`` parameter usages."""

    if resize_long_edge is None:
        return resize_target
    if resize_target is None or resize_target == resize_long_edge:
        return resize_long_edge
    raise ValueError("Conflicting resize targets provided; choose one value")


def _process_image_worker(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
    profile: ProcessingProfile,
) -> bool:
    """Core implementation for processing a single image.

    Returns ``True`` when an output file was written. This helper is isolated so it
    can be safely used with :class:`concurrent.futures.ProcessPoolExecutor`.
    """

    WORKER_LOGGER.info("Processing %s -> %s", source, destination)
    if destination.exists() and not dry_run and not destination.is_file():
        if destination.is_dir():
            path_type = "directory"
        elif destination.is_symlink():
            path_type = "symlink"
        else:
            path_type = "non-file"
        raise ValueError(f"Destination path exists but is a {path_type}: {destination}")

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        metadata = None
        with contextlib.suppress(AttributeError):
            metadata = image.tag_v2
        icc_profile = None
        if isinstance(image.info, dict):
            icc_profile = image.info.get("icc_profile")
        float_result = image_to_float(image, return_format="object")
        arr = float_result.array
        dtype = float_result.dtype
        alpha = float_result.alpha
        base_channels = float_result.base_channels
        float_norm = float_result.float_normalisation
        effective_profile = _ensure_profile(profile)
        adjusted = apply_adjustments(arr, adjustments, profile=effective_profile)
        target = _coerce_resize_target(resize_long_edge, resize_target)
        if target is not None:
            adjusted = resize_long_edge_array(adjusted, target)
            if alpha is not None:
                alpha = resize_long_edge_array(alpha, target)
        target_dtype = effective_profile.target_dtype(dtype)
        arr_int = float_to_dtype_array(
            adjusted,
            target_dtype,
            alpha,
            base_channels,
            float_normalisation=float_norm,
        )
        if dry_run:
            WORKER_LOGGER.info("Dry run enabled, skipping save for %s", destination)
            return False
        with ProcessingContext(destination) as staged_path:
            save_image(staged_path, arr_int, target_dtype, metadata, icc_profile, compression)
    return True


def process_single_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
    profile: ProcessingProfile | None = None,
) -> None:
    """Public wrapper around :func:`_process_image_worker`."""

    _process_image_worker(
        source,
        destination,
        adjustments,
        compression=compression,
        resize_long_edge=resize_long_edge,
        resize_target=resize_target,
        dry_run=dry_run,
        profile=_ensure_profile(profile),
    )


# Backwards compatibility shim for older integrations expecting the previous helper name.
process_image = process_single_image


__all__ = [
    "_PROGRESS_WRAPPER",
    "_coerce_resize_target",
    "_wrap_with_progress",
    "collect_images",
    "ensure_output_path",
    "process_image",
    "process_single_image",
    "resize_array",
    "resize_bilinear",
    "resize_long_edge_array",
]
