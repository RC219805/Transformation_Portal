# pylint: disable=no-member  # False positives with package imports
from luxury_tiff_batch_processor.io_utils import (
    float_to_dtype_array,
    image_to_float,
    save_image,
)
from luxury_tiff_batch_processor.adjustments import (
    gaussian_blur,
    gaussian_kernel_cached,
)
import luxury_tiff_batch_processor as ltiff
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from PIL import Image

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_float_to_dtype_array_preserves_float_values():
    gradient = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    rgb = np.stack([gradient, gradient ** 2, np.sqrt(gradient)], axis=-1)
    result = float_to_dtype_array(rgb, np.float32, None)
    assert result.dtype == np.float32
    assert np.allclose(result, rgb)


def test_save_image_retains_float_tonal_range(tmp_path):
    width, height = 32, 16
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = (x + y) / 2.0
    rgb = np.stack([gradient, gradient ** 1.5, np.clip(gradient * 1.2, 0.0, 1.0)], axis=-1)

    float_data = float_to_dtype_array(rgb, np.float32, None)
    output_path = tmp_path / "float_image.tiff"
    save_image(output_path, float_data, np.dtype(np.float32), metadata=None, icc_profile=None, compression="deflate")

    if tifffile is not None:
        saved_array = tifffile.imread(output_path)
    else:
        with Image.open(output_path) as saved:
            saved_array = np.array(saved)

    flat = saved_array.reshape(-1)
    assert np.unique(flat).size > 10


def test_image_to_float_roundtrip_signed_int_image():
    gradient = np.linspace(-5000, 5000, 49, dtype=np.int32).reshape(7, 7)
    image = Image.fromarray(gradient)

    rgb_float, dtype, alpha, _ = image_to_float(image, return_format="tuple4")

    assert dtype == np.int32
    assert alpha is None
    assert rgb_float.dtype == np.float32
    assert np.all((rgb_float >= 0.0) & (rgb_float <= 1.0))

    restored = float_to_dtype_array(rgb_float, dtype, alpha)
    assert restored.dtype == np.int32
    max_diff = np.max(np.abs(restored[:, :, 0] - gradient))
    assert max_diff <= 256
    for channel in range(1, restored.shape[2]):
        assert np.array_equal(restored[:, :, channel], restored[:, :, 0])


def test_image_to_float_float_dynamic_range_restored():
    data = np.linspace(-2.0, 2.0, 36, dtype=np.float32).reshape(6, 6)
    image = Image.fromarray(data, mode="F")

    result = ltiff.image_to_float(image)
    assert isinstance(result, ltiff.ImageToFloatResult)
    arr = result.array
    dtype = result.dtype
    alpha = result.alpha
    base_channels = result.base_channels
    float_norm = result.float_normalisation

    rgb_only, dtype_only, alpha_only = ltiff.image_to_float(image, return_format="tuple3")
    np.testing.assert_allclose(rgb_only, arr)
    assert dtype_only == dtype
    if alpha is None:
        assert alpha_only is None
    else:
        assert alpha_only is not None
        np.testing.assert_allclose(alpha_only, alpha)

    assert np.issubdtype(dtype, np.floating)
    assert float_norm is not None

    restored = ltiff.float_to_dtype_array(
        arr,
        dtype,
        alpha,
        base_channels,
        float_normalisation=float_norm,
    )

    assert restored.dtype == np.float32
    assert restored.shape == data.shape
    assert np.allclose(restored, data, atol=1e-6)


def _reference_gaussian_blur(arr: np.ndarray, radius: int, sigma: Optional[float] = None) -> np.ndarray:
    kernel = gaussian_kernel_cached(radius, sigma)

    working = arr
    squeeze = False
    if working.ndim == 2:
        working = working[:, :, None]
        squeeze = True

    pad = kernel.size // 2
    padded = np.pad(working, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    vertical = np.empty_like(working, dtype=np.float32)
    for y in range(working.shape[0]):
        window = padded[y: y + kernel.size]
        vertical[y] = np.tensordot(kernel, window, axes=(0, 0))

    padded_h = np.pad(vertical, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    blurred = np.empty_like(working, dtype=np.float32)
    for x in range(working.shape[1]):
        window = padded_h[:, x: x + kernel.size]
        blurred[:, x] = np.tensordot(kernel, window, axes=(0, 1))

    if squeeze:
        return blurred[:, :, 0]
    return blurred


def test_gaussian_blur_matches_reference():
    rng = np.random.default_rng(42)
    data = rng.random((12, 10, 3), dtype=np.float32)
    radius = 3

    optimised = gaussian_blur(data, radius)
    reference = _reference_gaussian_blur(data, radius)

    assert np.allclose(optimised, reference, atol=1e-6)


def test_gaussian_blur_reuses_cached_kernel():
    ltiff.gaussian_kernel.cache_clear()
    rng = np.random.default_rng(123)
    data = rng.random((16, 12, 3), dtype=np.float32)
    radius = 2

    first = gaussian_blur(data, radius)
    first_info = ltiff.gaussian_kernel.cache_info()
    assert first_info.misses == 1

    second = gaussian_blur(data, radius)
    second_info = ltiff.gaussian_kernel.cache_info()
    assert second_info.hits >= 1

    assert np.allclose(first, second)
