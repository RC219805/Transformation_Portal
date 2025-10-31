"""Color adjustments, presets, and supporting image math."""
from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import Optional

import numpy as np

from .profiles import ProcessingProfile

LOGGER = logging.getLogger("luxury_tiff_batch_processor")


@dataclasses.dataclass
class AdjustmentSettings:
    """Holds the image adjustment parameters for a processing run."""

    exposure: float = 0.0
    white_balance_temp: Optional[float] = None
    white_balance_tint: float = 0.0
    shadow_lift: float = 0.0
    highlight_recovery: float = 0.0
    midtone_contrast: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    clarity: float = 0.0
    chroma_denoise: float = 0.0
    glow: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        def ensure_range(name: str, value: float, minimum: float, maximum: float) -> None:
            if not (minimum <= value <= maximum):
                raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")

        ensure_range("exposure", self.exposure, -5.0, 5.0)

        if self.white_balance_temp is not None:
            ensure_range("white_balance_temp", self.white_balance_temp, 1500.0, 20000.0)

        ensure_range("white_balance_tint", self.white_balance_tint, -150.0, 150.0)
        ensure_range("shadow_lift", self.shadow_lift, 0.0, 1.0)
        ensure_range("highlight_recovery", self.highlight_recovery, 0.0, 1.0)
        ensure_range("midtone_contrast", self.midtone_contrast, -1.0, 1.0)
        ensure_range("vibrance", self.vibrance, -1.0, 1.0)
        ensure_range("saturation", self.saturation, -1.0, 1.0)
        ensure_range("clarity", self.clarity, -1.0, 1.0)
        ensure_range("chroma_denoise", self.chroma_denoise, 0.0, 1.0)
        ensure_range("glow", self.glow, 0.0, 1.0)


LUXURY_PRESETS = {
    "signature": AdjustmentSettings(
        exposure=0.12,
        white_balance_temp=6500,
        white_balance_tint=4.0,
        shadow_lift=0.18,
        highlight_recovery=0.15,
        midtone_contrast=0.08,
        vibrance=0.18,
        saturation=0.06,
        clarity=0.16,
        chroma_denoise=0.08,
        glow=0.05,
    ),
    "architectural": AdjustmentSettings(
        exposure=0.08,
        white_balance_temp=6200,
        white_balance_tint=2.0,
        shadow_lift=0.12,
        highlight_recovery=0.1,
        midtone_contrast=0.05,
        vibrance=0.12,
        saturation=0.04,
        clarity=0.22,
        chroma_denoise=0.05,
        glow=0.02,
    ),
    "golden_hour_courtyard": AdjustmentSettings(
        exposure=0.08,
        white_balance_temp=5600,
        white_balance_tint=5.0,
        shadow_lift=0.24,
        highlight_recovery=0.18,
        midtone_contrast=0.10,
        vibrance=0.28,
        saturation=0.05,
        clarity=0.20,
        chroma_denoise=0.06,
        glow=0.12,
    ),
    "twilight": AdjustmentSettings(
        exposure=0.05,
        white_balance_temp=5400,
        white_balance_tint=8.0,
        shadow_lift=0.24,
        highlight_recovery=0.18,
        midtone_contrast=0.1,
        vibrance=0.24,
        saturation=0.08,
        clarity=0.12,
        chroma_denoise=0.1,
        glow=0.12,
    ),
}


def kelvin_to_rgb(temperature: float) -> np.ndarray:
    """Convert color temperature in Kelvin to RGB multipliers.

    Args:
        temperature: Color temperature in Kelvin (typically 1500-20000K).

    Returns:
        RGB multiplier array as float32 with values in [0, 1] range.
    """
    temp = temperature / 100.0
    if temp <= 0:
        temp = 0.1

    if temp <= 66:
        red = 1.0
        green = np.clip(0.39008157876901960784 * math.log(temp) - 0.63184144378862745098, 0, 1)
        blue = 0 if temp <= 19 else np.clip(0.54320678911019607843 * math.log(temp - 10) - 1.19625408914, 0, 1)
    else:
        red = np.clip(1.29293618606274509804 * (temp - 60) ** -0.1332047592, 0, 1)
        green = np.clip(1.12989086089529411765 * (temp - 60) ** -0.0755148492, 0, 1)
        blue = 1.0
    return np.array([red, green, blue], dtype=np.float32)


def apply_exposure(arr: np.ndarray, stops: float) -> np.ndarray:
    """Apply exposure adjustment to image array.

    Args:
        arr: Input RGB image array in float32 format.
        stops: Exposure adjustment in photographic stops (typically -5 to +5).

    Returns:
        Exposure-adjusted array with same shape as input.
    """
    if stops == 0:
        return arr
    factor = float(2.0 ** stops)
    LOGGER.debug("Applying exposure: %s stops (factor %.3f)", stops, factor)
    return arr * factor


def apply_white_balance(arr: np.ndarray, temperature: Optional[float], tint: float) -> np.ndarray:
    """Apply white balance adjustment using color temperature and tint.

    Args:
        arr: Input RGB image array in float32 format.
        temperature: Target color temperature in Kelvin (1500-20000K), or None to skip.
        tint: Green-magenta tint adjustment (-150 to +150, positive shifts magenta).

    Returns:
        White-balanced array with same shape as input.
    """
    result = arr
    if temperature is not None:
        ref = kelvin_to_rgb(6500.0)
        target = kelvin_to_rgb(temperature)
        scale = target / ref
        LOGGER.debug("Applying temperature: %sK scale=%s", temperature, scale)
        result = result * scale.reshape((1, 1, 3))
    if tint:
        tint_scale = np.array([1.0 + tint * 0.0015, 1.0, 1.0 - tint * 0.0015], dtype=np.float32)
        LOGGER.debug("Applying tint scale=%s", tint_scale)
        result = result * tint_scale.reshape((1, 1, 3))
    return result


def luminance(arr: np.ndarray) -> np.ndarray:
    """Calculate perceptual luminance from RGB values.

    Uses Rec. 709 luma coefficients for accurate perceptual brightness.

    Args:
        arr: Input RGB image array.

    Returns:
        2D luminance array with same height and width as input.
    """
    return arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722


def apply_shadow_lift(arr: np.ndarray, amount: float) -> np.ndarray:
    """Lift shadows while preserving highlights.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Shadow lift strength (0.0 to 1.0).

    Returns:
        Shadow-lifted array with same shape as input.
    """
    if amount <= 0:
        return arr
    gamma = 1.0 / (1.0 + amount * 3.0)
    lum = luminance(arr)
    lifted = np.power(np.clip(lum, 0.0, 1.0), gamma)
    LOGGER.debug("Shadow lift amount=%s gamma=%.3f", amount, gamma)
    return arr + (lifted - lum)[..., None]


def apply_highlight_recovery(arr: np.ndarray, amount: float) -> np.ndarray:
    """Recover detail in highlights by compressing bright regions.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Highlight recovery strength (0.0 to 1.0).

    Returns:
        Highlight-recovered array with same shape as input.
    """
    if amount <= 0:
        return arr
    gamma = 1.0 + amount * 2.0
    lum = luminance(arr)
    compressed = np.power(np.clip(lum, 0.0, 1.0), gamma)
    LOGGER.debug("Highlight recovery amount=%s gamma=%.3f", amount, gamma)
    return arr + (compressed - lum)[..., None]


def apply_midtone_contrast(arr: np.ndarray, amount: float) -> np.ndarray:
    """Adjust contrast in midtone regions.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Contrast adjustment (-1.0 to +1.0, negative reduces contrast).

    Returns:
        Contrast-adjusted array with same shape as input.
    """
    if amount == 0:
        return arr
    lum = luminance(arr)
    contrasted = 0.5 + (lum - 0.5) * (1.0 + amount)
    LOGGER.debug("Midtone contrast amount=%s", amount)
    return arr + (contrasted - lum)[..., None]


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Convert RGB color space to HSV (Hue, Saturation, Value).

    Args:
        arr: Input RGB image array in float32 format.

    Returns:
        HSV array with same shape, values in [0, 1] range.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    diff = maxc - minc

    hue = np.zeros_like(maxc)
    mask = diff != 0
    rc = np.zeros_like(maxc)
    gc = np.zeros_like(maxc)
    bc = np.zeros_like(maxc)

    # Use np.divide with an explicit mask so that we never evaluate the
    # division for pixels where ``diff`` is zero.  The previous approach
    # performed the division on the full array and then masked the result,
    # which triggered RuntimeWarnings due to invalid values generated by the
    # zero denominators.  Those warnings surfaced in the test suite and made
    # it harder to audit the processing pipeline.  By keeping the computation
    # within the masked region, we maintain identical results without the
    # noisy warnings.
    np.copyto(rc, maxc)
    rc -= r
    np.divide(rc, diff, out=rc, where=mask)
    np.copyto(gc, maxc)
    gc -= g
    np.divide(gc, diff, out=gc, where=mask)
    np.copyto(bc, maxc)
    bc -= b
    np.divide(bc, diff, out=bc, where=mask)

    hue[maxc == r] = (bc - gc)[maxc == r]
    hue[maxc == g] = 2.0 + (rc - bc)[maxc == g]
    hue[maxc == b] = 4.0 + (gc - rc)[maxc == b]
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc)
    non_zero = maxc != 0
    saturation[non_zero] = diff[non_zero] / maxc[non_zero]

    value = maxc
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert HSV color space to RGB.

    Args:
        arr: Input HSV image array in float32 format with values in [0, 1].

    Returns:
        RGB array with same shape as input.
    """
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    shape = h.shape + (3,)
    rgb = np.zeros(shape, dtype=np.float32)

    conditions = [
        (i_mod == 0, np.stack([v, t, p], axis=-1)),
        (i_mod == 1, np.stack([q, v, p], axis=-1)),
        (i_mod == 2, np.stack([p, v, t], axis=-1)),
        (i_mod == 3, np.stack([p, q, v], axis=-1)),
        (i_mod == 4, np.stack([t, p, v], axis=-1)),
        (i_mod == 5, np.stack([v, p, q], axis=-1)),
    ]
    for condition, value in conditions:
        rgb[condition] = value[condition]
    return rgb


def apply_vibrance(arr: np.ndarray, amount: float) -> np.ndarray:
    """Apply intelligent saturation boost that protects already saturated colors.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Vibrance strength (-1.0 to +1.0).

    Returns:
        Vibrance-adjusted array with same shape as input.
    """
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    saturation = hsv[..., 1]
    hsv[..., 1] = np.clip(
        saturation + amount * (1.0 - saturation) * np.sqrt(np.clip(saturation, 0.0, 1.0)),
        0.0,
        1.0,
    )
    LOGGER.debug("Vibrance amount=%s", amount)
    return hsv_to_rgb(hsv)


def apply_saturation(arr: np.ndarray, amount: float) -> np.ndarray:
    """Apply uniform saturation adjustment to all colors.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Saturation multiplier delta (-1.0 to +1.0).

    Returns:
        Saturation-adjusted array with same shape as input.
    """
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amount), 0.0, 1.0)
    LOGGER.debug("Saturation delta=%s", amount)
    return hsv_to_rgb(hsv)


@functools.lru_cache(maxsize=32)
def _gaussian_kernel_cached(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    if radius <= 0:
        return np.array([1.0], dtype=np.float32)
    sigma = sigma or max(radius / 3.0, 1e-6)
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    cached = kernel.astype(np.float32)
    cached.setflags(write=False)
    return cached


def gaussian_kernel(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    """Generate a 1D Gaussian convolution kernel with mutable copy.

    The internal cached kernel is stored in a read-only array. This function
    provides callers with a writable copy so that downstream code can safely
    adjust the kernel without corrupting the cached value.

    Args:
        radius: Kernel radius in pixels (half-width).
        sigma: Standard deviation for Gaussian distribution, defaults to radius/3.

    Returns:
        1D numpy array containing normalized Gaussian weights.
    """
    return _gaussian_kernel_cached(radius, sigma).copy()


def gaussian_kernel_cached(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    """Return cached immutable Gaussian kernel for efficient reuse.

    Args:
        radius: Kernel radius in pixels (half-width).
        sigma: Standard deviation for Gaussian distribution, defaults to radius/3.

    Returns:
        Read-only 1D numpy array containing normalized Gaussian weights.
    """
    return _gaussian_kernel_cached(radius, sigma)


# Preserve the cache management helpers for existing callers.
gaussian_kernel.cache_clear = _gaussian_kernel_cached.cache_clear  # type: ignore[attr-defined]
gaussian_kernel.cache_info = _gaussian_kernel_cached.cache_info  # type: ignore[attr-defined]


def separable_convolve(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply 1D convolution along specified axis with reflection padding.

    Args:
        arr: Input array to convolve.
        kernel: 1D convolution kernel.
        axis: Axis along which to apply convolution.

    Returns:
        Convolved array with same shape as input.
    """
    pad_width = [(0, 0)] * arr.ndim
    k = kernel.size // 2
    pad_width[axis] = (k, k)
    padded = np.pad(arr, pad_width, mode="reflect")
    convolved = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=padded
    )
    return convolved.astype(np.float32)


def gaussian_blur(arr: np.ndarray, radius: int, sigma: Optional[float] = None) -> np.ndarray:
    """Apply Gaussian blur using separable 2D convolution.

    Args:
        arr: Input image array.
        radius: Blur radius in pixels.
        sigma: Standard deviation for Gaussian kernel, defaults to radius/3.

    Returns:
        Blurred array with same shape as input.
    """
    kernel = gaussian_kernel_cached(radius, sigma)
    blurred = separable_convolve(arr, kernel, axis=0)
    blurred = separable_convolve(blurred, kernel, axis=1)
    return blurred


def apply_clarity(arr: np.ndarray, amount: float) -> np.ndarray:
    """Enhance local contrast using high-pass filtering.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Clarity strength (-1.0 to +1.0).

    Returns:
        Clarity-enhanced array, clipped to [0, 1] range.
    """
    if amount <= 0:
        return arr
    radius = max(1, int(round(1 + amount * 5)))
    blurred = gaussian_blur(arr, radius)
    high_pass = arr - blurred
    LOGGER.debug("Clarity amount=%s radius=%s", amount, radius)
    return np.clip(arr + high_pass * (0.6 + amount * 0.8), 0.0, 1.0)


def rgb_to_yuv(arr: np.ndarray) -> np.ndarray:
    """Convert RGB to YUV color space for chrominance processing.

    Args:
        arr: Input RGB image array.

    Returns:
        YUV array with Y in [0, 1] and U,V centered at 0.5.
    """
    matrix = np.array(
        [
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5000],
            [0.5000, -0.4542, -0.0458],
        ],
        dtype=np.float32,
    )
    yuv = arr @ matrix.T
    yuv[..., 1:] += 0.5
    return yuv


def yuv_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert YUV color space back to RGB.

    Args:
        arr: Input YUV array with U,V centered at 0.5.

    Returns:
        RGB array with same shape as input.
    """
    matrix = np.array(
        [
            [1.0, 0.0, 1.28033],
            [1.0, -0.21482, -0.38059],
            [1.0, 2.12798, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = arr.copy()
    rgb[..., 1:] -= 0.5
    rgb = rgb @ matrix.T
    return rgb


def apply_chroma_denoise(arr: np.ndarray, amount: float) -> np.ndarray:
    """Reduce color noise by blurring chrominance channels.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Denoising strength (0.0 to 1.0).

    Returns:
        Denoised array, clipped to [0, 1] range.
    """
    if amount <= 0:
        return arr
    yuv = rgb_to_yuv(arr)
    radius = max(1, int(round(1 + amount * 4)))
    for channel in (1, 2):
        channel_data = yuv[..., channel]
        blurred = gaussian_blur(channel_data[..., None], radius)[:, :, 0]
        yuv[..., channel] = channel_data * (1.0 - amount) + blurred * amount
    LOGGER.debug("Chroma denoise amount=%s radius=%s", amount, radius)
    return np.clip(yuv_to_rgb(yuv), 0.0, 1.0)


def apply_glow(arr: np.ndarray, amount: float) -> np.ndarray:
    """Apply diffusion glow effect for luxury aesthetic.

    Args:
        arr: Input RGB image array in float32 format.
        amount: Glow strength (0.0 to 1.0).

    Returns:
        Glowed array, clipped to [0, 1] range.
    """
    if amount <= 0:
        return arr
    radius = max(2, int(round(6 + amount * 20)))
    softened = gaussian_blur(arr, radius)
    LOGGER.debug("Glow amount=%s radius=%s", amount, radius)
    return np.clip(arr * (1.0 - amount) + softened * amount, 0.0, 1.0)


def apply_adjustments(
    arr: np.ndarray, adjustments: AdjustmentSettings, *, profile: ProcessingProfile | None = None
) -> np.ndarray:
    """Apply complete adjustment pipeline to image array.

    Applies adjustments in optimal order: white balance, exposure, tone curves,
    denoising, color adjustments, clarity, and finishing effects.

    Args:
        arr: Input RGB image array in float32 format, values in [0, 1].
        adjustments: Complete set of adjustment parameters.
        profile: Optional processing profile to modulate expensive operations.

    Returns:
        Adjusted image array, clipped to [0, 1] range.
    """
    arr = apply_white_balance(arr, adjustments.white_balance_temp, adjustments.white_balance_tint)
    arr = apply_exposure(arr, adjustments.exposure)
    arr = apply_shadow_lift(arr, adjustments.shadow_lift)
    arr = apply_highlight_recovery(arr, adjustments.highlight_recovery)
    arr = apply_midtone_contrast(arr, adjustments.midtone_contrast)
    arr = np.clip(arr, 0.0, 1.0)
    chroma_amount = adjustments.chroma_denoise
    glow_amount = adjustments.glow
    if profile is not None:
        chroma_amount = profile.resolve_chroma_denoise(chroma_amount)
        glow_amount = profile.resolve_glow(glow_amount)
    arr = apply_chroma_denoise(arr, chroma_amount)
    arr = apply_vibrance(arr, adjustments.vibrance)
    arr = apply_saturation(arr, adjustments.saturation)
    arr = apply_clarity(arr, adjustments.clarity)
    arr = apply_glow(arr, glow_amount)
    return np.clip(arr, 0.0, 1.0)


__all__ = [
    "AdjustmentSettings",
    "LUXURY_PRESETS",
    "apply_adjustments",
    "apply_chroma_denoise",
    "apply_clarity",
    "apply_exposure",
    "apply_glow",
    "apply_highlight_recovery",
    "apply_midtone_contrast",
    "apply_saturation",
    "apply_shadow_lift",
    "apply_vibrance",
    "apply_white_balance",
    "gaussian_blur",
    "gaussian_kernel",
    "gaussian_kernel_cached",
    "kelvin_to_rgb",
    "luminance",
    "rgb_to_hsv",
    "hsv_to_rgb",
]
