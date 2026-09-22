"""Bounded paired color measurements, separate from photographic acceptance.

Inputs are caller-supplied extended linear sRGB samples on the same grid. Linear
errors retain headroom; Oklab measurements use only pairs within the SDR cube.
Neither reference provenance nor aesthetic quality is inferred from these arrays.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .color import linear_srgb_to_oklab

MAX_PIXELS = 100_000_000
MAX_TILE_PIXELS = 65536
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _validate(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError("Color evaluation requires nonempty float32 HWC RGB samples")
    if array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError("Color evaluation exceeds its pixel budget")
    if mask is not None and (np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != array.shape[:2]):
        raise ValueError("Evaluation mask must be boolean on the shared image grid")
    return array


def _tiles(shape: tuple[int, ...]) -> Iterator[tuple[slice, slice]]:
    height, width = shape[:2]
    tile_width = min(width, MAX_TILE_PIXELS)
    tile_height = max(1, MAX_TILE_PIXELS // tile_width)
    for top in range(0, height, tile_height):
        for left in range(0, width, tile_width):
            yield slice(top, min(height, top + tile_height)), slice(left, min(width, left + tile_width))


def numeric_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Count range/headroom without mistaking boundary samples for known clipping."""
    array = _validate(values, mask)
    mask = None if mask is None else np.asarray(mask)
    count = below = above = outside_pixels = at_zero = at_one = 0
    minimum, maximum = float("inf"), -float("inf")
    for ys, xs in _tiles(array.shape):
        tile = array[ys, xs]
        if not np.isfinite(tile).all():
            raise ValueError("Color evidence must contain only finite samples, including masked regions")
        pixels = tile.reshape(-1, 3) if mask is None else tile[mask[ys, xs]]
        if not len(pixels):
            continue
        count += len(pixels)
        below += int(np.count_nonzero(pixels < 0))
        above += int(np.count_nonzero(pixels > 1))
        outside_pixels += int(np.count_nonzero(np.any((pixels < 0) | (pixels > 1), axis=1)))
        at_zero += int(np.count_nonzero(pixels == 0))
        at_one += int(np.count_nonzero(pixels == 1))
        minimum, maximum = min(minimum, float(pixels.min())), max(maximum, float(pixels.max()))
    if not count:
        raise ValueError("Color evaluation requires at least one selected pixel")
    return {
        "selected_pixels": count,
        "selected_samples": count * 3,
        "minimum_linear_sample": minimum,
        "maximum_linear_sample": maximum,
        "below_zero_samples": below,
        "above_one_samples": above,
        "outside_sdr_pixels": outside_pixels,
        "outside_sdr_pixel_fraction": outside_pixels / count,
        "exact_zero_samples": at_zero,
        "exact_one_samples": at_one,
        "clipping_provenance": "unavailable_boundary_counts_do_not_prove_clipping",
    }


def evaluate_pairs(reference_rgb: np.ndarray, prediction_rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Measure array agreement without certifying a reference or a creative grade.

    Oklab distance is the unscaled Euclidean distance in the named Oklab 2021
    transform, not CIEDE2000. Hue error is undefined for near-neutral pairs and
    is measured only when both Oklab chroma magnitudes exceed 0.0001.
    """
    reference, prediction = _validate(reference_rgb, mask), _validate(prediction_rgb, mask)
    mask = None if mask is None else np.asarray(mask)
    if reference.shape != prediction.shape:
        raise ValueError("Paired color evidence must have identical image geometry")
    summaries = {"reference": numeric_summary(reference, mask), "prediction": numeric_summary(prediction, mask)}
    selected = summaries["reference"]["selected_pixels"]
    rgb_abs = rgb_squared = rgb_max = luma_abs = luma_squared = luma_max = 0.0
    lab_sum = lab_squared = lab_max = chroma_sum = chroma_max = 0.0
    hue_sum = hue_max = 0.0
    perceptual_count = hue_count = 0
    for ys, xs in _tiles(reference.shape):
        first, second = reference[ys, xs], prediction[ys, xs]
        first = first.reshape(-1, 3) if mask is None else first[mask[ys, xs]]
        second = second.reshape(-1, 3) if mask is None else second[mask[ys, xs]]
        if not len(first):
            continue
        first, second = first.astype(np.float64), second.astype(np.float64)
        difference = second - first
        absolute = np.abs(difference)
        rgb_abs += float(absolute.sum())
        rgb_squared += float(np.square(difference).sum())
        rgb_max = max(rgb_max, float(absolute.max()))
        luma = difference @ _LUMA
        luma_abs += float(np.abs(luma).sum())
        luma_squared += float(np.square(luma).sum())
        luma_max = max(luma_max, float(np.abs(luma).max()))
        bounded = np.all((first >= 0) & (first <= 1) & (second >= 0) & (second <= 1), axis=1)
        if not bounded.any():
            continue
        a, b = linear_srgb_to_oklab(first[bounded]), linear_srgb_to_oklab(second[bounded])
        distances = np.linalg.norm(b - a, axis=1)
        perceptual_count += len(distances)
        lab_sum += float(distances.sum())
        lab_squared += float(np.square(distances).sum())
        lab_max = max(lab_max, float(distances.max()))
        ca, cb = np.linalg.norm(a[:, 1:], axis=1), np.linalg.norm(b[:, 1:], axis=1)
        chroma_difference = np.abs(cb - ca)
        chroma_sum += float(chroma_difference.sum())
        chroma_max = max(chroma_max, float(chroma_difference.max()))
        chromatic = (ca > 0.0001) & (cb > 0.0001)
        if chromatic.any():
            angles = np.arctan2(b[chromatic, 2], b[chromatic, 1]) - np.arctan2(a[chromatic, 2], a[chromatic, 1])
            angular_error = np.degrees(np.abs(np.arctan2(np.sin(angles), np.cos(angles))))
            hue_count += len(angular_error)
            hue_sum += float(angular_error.sum())
            hue_max = max(hue_max, float(angular_error.max()))
    perceptual: dict[str, Any] = {
        "status": "measured" if perceptual_count else "unavailable",
        "reason": None if perceptual_count else "no_mutually_in_gamut_sdr_pairs",
        "space": "oklab_2021",
        "distance": "unscaled_euclidean_not_ciede2000",
        "valid_pixels": perceptual_count,
        "coverage": perceptual_count / selected,
    }
    if perceptual_count:
        perceptual.update(
            {
                "mean_distance": lab_sum / perceptual_count,
                "rms_distance": float(np.sqrt(lab_squared / perceptual_count)),
                "max_distance": lab_max,
                "mean_absolute_chroma_error": chroma_sum / perceptual_count,
                "max_absolute_chroma_error": chroma_max,
                "hue": {
                    "status": "measured" if hue_count else "unavailable",
                    "reason": None if hue_count else "no_chromatic_pairs",
                    "valid_pixels": hue_count,
                    "minimum_chroma": 0.0001,
                    "mean_absolute_degrees": hue_sum / hue_count if hue_count else None,
                    "max_absolute_degrees": hue_max if hue_count else None,
                },
            }
        )
    return {
        "schema": "tp.lux.paired_color_metrics.v1",
        "working_space": "extended_linear_srgb",
        "selected_pixels": selected,
        "range": summaries,
        "linear_rgb": {
            "mae": rgb_abs / (3 * selected),
            "rmse": float(np.sqrt(rgb_squared / (3 * selected))),
            "max_error": rgb_max,
        },
        "linear_luminance": {
            "mae": luma_abs / selected,
            "rmse": float(np.sqrt(luma_squared / selected)),
            "max_error": luma_max,
        },
        "oklab": perceptual,
        "reference_authority": "caller_supplied_arrays_not_independently_certified",
        "style_quality": "not_measured",
        "production_acceptance": "not_established",
        "max_scratch_tile_pixels": MAX_TILE_PIXELS,
    }
