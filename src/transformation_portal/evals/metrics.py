"""Image quality metrics for evaluation and benchmarking.

This module provides standard image quality metrics:
- PSNR: Peak Signal-to-Noise Ratio (reconstruction fidelity)
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
- IoU: Intersection over Union (segmentation accuracy)

All metrics support both numpy arrays and torch tensors.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import lpips
    import torch

import numpy as np

logger = logging.getLogger(__name__)


# Type alias for image data
ImageLike = Union[np.ndarray, "torch.Tensor", Path]


def _torch_tensor_to_numpy(img: "torch.Tensor") -> np.ndarray:
    """Convert a torch tensor to numpy with a NumPy-bridge-free fallback."""
    tensor = img.detach().cpu()
    try:
        return tensor.numpy()
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        return np.asarray(tensor.tolist(), dtype=np.float32)


# Torch counterparts of the 8/16-bit unsigned image dtypes handled by ``_bit_depth_divisor``.
_TORCH_BIT_DEPTH_DIVISORS = {"torch.uint8": 255.0, "torch.uint16": 65535.0}


def _bit_depth_divisor(dtype: np.dtype) -> Optional[float]:
    """Return the full-scale value of an 8/16-bit unsigned image dtype, else ``None``.

    The range of uint8/uint16 image data is fixed by the dtype, never by the pixel content.
    Wider or signed integers are generic containers (``np.asarray`` of Python ints is int64),
    so, like floats, they keep the content convention applied by ``_range_divisors``.
    """
    if dtype.kind == "u" and dtype.itemsize <= 2:
        return float(np.iinfo(dtype).max)
    return None


def _raw_array(img: ImageLike) -> np.ndarray:
    """Load ``img`` as a numpy array without changing its value range (paths load as 8-bit RGB)."""
    if isinstance(img, Path):
        try:
            import cv2

            arr = cv2.imread(str(img))
            if arr is None:
                raise ValueError(f"Failed to load image: {img}")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image

            pil_img = Image.open(img).convert("RGB")
            return np.array(pil_img)

    if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):  # torch.Tensor
        return _torch_tensor_to_numpy(img)
    return np.asarray(img)


def _range_divisors(known: list[Optional[float]], peaks: list[Optional[float]]) -> list[float]:
    """Return the divisor that maps each member of an image pair into [0, 1].

    ``known`` holds the full-scale value fixed by a bit-depth dtype (``None`` when the
    dtype fixes nothing, e.g. floats); ``peaks`` holds the maximum of each such member.
    Data whose range the dtype does not fix follows the pinned convention that values
    above 1.0 are 0-255 data. The decision is made once for the whole pair, so both
    members are always scaled identically, and it is taken only when every one of them
    exceeds 1.0: dividing an image that lies in [0, 1] by 255 would shrink the error and
    inflate PSNR/SSIM, whereas leaving a 0-255 member unscaled can only make it score worse.
    """
    unknown_peaks = [peak for divisor, peak in zip(known, peaks) if divisor is None]
    content_divisor = 255.0 if unknown_peaks and all(peak is not None and peak > 1.0 for peak in unknown_peaks) else 1.0
    return [divisor if divisor is not None else content_divisor for divisor in known]


def _numpy_member(img: ImageLike) -> tuple[np.ndarray, Optional[float], Optional[float]]:
    """Return ``(raw array, dtype full-scale value, content peak)`` for one image."""
    arr = _raw_array(img)
    known = _bit_depth_divisor(arr.dtype)
    return arr, known, None if known is not None else float(arr.max())


def _scaled(arr: np.ndarray, divisor: float) -> np.ndarray:
    """Return ``arr`` as float32 divided by ``divisor``."""
    scaled = arr.astype(np.float32)
    return scaled / divisor if divisor != 1.0 else scaled


def _to_numpy(img: ImageLike) -> np.ndarray:
    """Convert image to numpy array.

    Args:
        img: Image as numpy array, torch tensor, or path

    Returns:
        Numpy array in [0, 1] range (uint8/uint16 are scaled by their dtype's full-scale value)
    """
    arr, known, peak = _numpy_member(img)
    (divisor,) = _range_divisors([known], [peak])
    return _scaled(arr, divisor)


def _to_numpy_pair(img1: ImageLike, img2: ImageLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert a prediction/target pair to numpy arrays in [0, 1], scaled consistently."""
    arr1, known1, peak1 = _numpy_member(img1)
    arr2, known2, peak2 = _numpy_member(img2)
    divisor1, divisor2 = _range_divisors([known1, known2], [peak1, peak2])
    return _scaled(arr1, divisor1), _scaled(arr2, divisor2)


def _to_torch_pair(img1: ImageLike, img2: ImageLike) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Convert a prediction/target pair to torch tensors in [0, 1], shape (C, H, W), scaled consistently."""
    import torch

    members: list[tuple[Union[np.ndarray, "torch.Tensor"], Optional[float], Optional[float]]] = []
    for img in (img1, img2):
        if isinstance(img, torch.Tensor):
            known = _TORCH_BIT_DEPTH_DIVISORS.get(str(img.dtype))
            tensor = img.float()
            members.append((tensor, known, None if known is not None else float(tensor.max())))
        else:
            members.append(_numpy_member(img))

    divisors = _range_divisors([known for _, known, _ in members], [peak for _, _, peak in members])
    tensors = []
    for (data, _, _), divisor in zip(members, divisors):
        if isinstance(data, torch.Tensor):
            tensor = data / divisor if divisor != 1.0 else data
            # Ensure (C, H, W) format
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
        else:
            arr = _scaled(data, divisor)
            # Convert (H, W, C) to (C, H, W)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).float()
        tensors.append(tensor)
    return tensors[0], tensors[1]


# ============================================================================
# PSNR - Peak Signal-to-Noise Ratio
# ============================================================================


def psnr(
    img1: ImageLike,
    img2: ImageLike,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Higher PSNR indicates better reconstruction quality.
    Typical values: 20-25 (poor), 25-30 (acceptable), 30-40 (good), >40 (excellent)

    Args:
        img1: First image (prediction)
        img2: Second image (ground truth)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)

    Returns:
        PSNR value in dB
    """
    arr1, arr2 = _to_numpy_pair(img1, img2)

    # Ensure same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Image shapes don't match: {arr1.shape} vs {arr2.shape}")

    mse = np.mean((arr1 - arr2) ** 2)

    if mse == 0:
        return float("inf")

    return float(20 * np.log10(max_val / np.sqrt(mse)))


def psnr_torch(
    img1: "torch.Tensor",
    img2: "torch.Tensor",
    max_val: float = 1.0,
) -> float:
    """Compute PSNR using torch (GPU-accelerated).

    Args:
        img1: First tensor
        img2: Second tensor
        max_val: Maximum value

    Returns:
        PSNR value
    """
    import torch

    mse = torch.mean((img1.float() - img2.float()) ** 2)

    if mse == 0:
        return 100.0  # Perfect match

    # Create max_val tensor on same device/dtype as mse to avoid device mismatch
    max_val_tensor = torch.as_tensor(max_val, device=mse.device, dtype=mse.dtype)
    return float(20 * torch.log10(max_val_tensor / torch.sqrt(mse)))


# ============================================================================
# SSIM - Structural Similarity Index
# ============================================================================


def ssim(
    img1: ImageLike,
    img2: ImageLike,
    win_size: int = 7,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    SSIM measures perceived quality considering luminance, contrast, and structure.
    Range: [-1, 1], where 1 = identical images.

    Args:
        img1: First image
        img2: Second image
        win_size: Window size for local statistics
        data_range: Data range (1.0 for normalized images)

    Returns:
        SSIM value
    """
    arr1, arr2 = _to_numpy_pair(img1, img2)

    # Try using skimage if available
    try:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(
                arr1,
                arr2,
                win_size=win_size,
                data_range=data_range,
                channel_axis=-1 if arr1.ndim == 3 else None,
            )
        )
    except ImportError:
        pass

    # Fallback without scikit-image: the same windowed SSIM (skimage's defaults).
    return _windowed_ssim(arr1, arr2, win_size=win_size, data_range=data_range)


def _windowed_ssim(arr1: np.ndarray, arr2: np.ndarray, *, win_size: int, data_range: float) -> float:
    """Mean windowed SSIM (Wang et al. 2004) with scikit-image's default parameters.

    Local statistics come from a ``win_size`` uniform window with the sample-covariance
    correction, the map is cropped by the window half-width and averaged; 3-D inputs are
    channel-last and the per-channel means are averaged. This matches
    ``skimage.metrics.structural_similarity(..., gaussian_weights=False)``, so the score does
    not depend on whether scikit-image is installed. A single global window is *not* SSIM:
    it scores a heavily blurred or row-shuffled image close to 1.0.
    """
    from scipy.ndimage import uniform_filter

    if arr1.shape != arr2.shape:
        raise ValueError(f"Image shapes don't match: {arr1.shape} vs {arr2.shape}")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError(f"win_size must be an odd integer >= 3, got {win_size}")

    channels_last = arr1.ndim == 3
    spatial_shape = arr1.shape[:-1] if channels_last else arr1.shape
    if arr1.ndim < 2 or min(spatial_shape) < win_size:
        raise ValueError(f"win_size {win_size} exceeds the image size {arr1.shape}; pass a smaller odd win_size")

    x_all = arr1.astype(np.float64)
    y_all = arr2.astype(np.float64)
    pairs = [(x_all[..., c], y_all[..., c]) for c in range(arr1.shape[-1])] if channels_last else [(x_all, y_all)]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window_pixels = win_size ** len(spatial_shape)
    cov_norm = window_pixels / (window_pixels - 1.0)
    pad = (win_size - 1) // 2
    crop = tuple(slice(pad, dim - pad) for dim in spatial_shape)

    channel_means = []
    for x, y in pairs:
        mu_x = uniform_filter(x, size=win_size)
        mu_y = uniform_filter(y, size=win_size)
        var_x = cov_norm * (uniform_filter(x * x, size=win_size) - mu_x * mu_x)
        var_y = cov_norm * (uniform_filter(y * y, size=win_size) - mu_y * mu_y)
        cov_xy = cov_norm * (uniform_filter(x * y, size=win_size) - mu_x * mu_y)
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
        channel_means.append(float(ssim_map[crop].mean()))
    return float(np.mean(channel_means))


# ============================================================================
# LPIPS - Learned Perceptual Image Patch Similarity
# ============================================================================


_lpips_model: Optional["lpips.LPIPS"] = None


def lpips_available() -> bool:
    """Return whether ``lpips_score`` can compute a real LPIPS distance (lpips and torch import).

    ``lpips_score`` returns 0.0 -- the "identical" distance -- when they are missing, so callers
    that aggregate LPIPS must check this first and report the metric as unavailable instead.
    """
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def lpips_score(
    img1: ImageLike,
    img2: ImageLike,
    net: str = "alex",
) -> float:
    """Compute LPIPS perceptual similarity.

    LPIPS uses deep features to measure perceptual similarity.
    Lower values indicate more similar images.
    Range: [0, 1+], where 0 = identical.

    Args:
        img1: First image
        img2: Second image
        net: Network backbone ("alex", "vgg", "squeeze")

    Returns:
        LPIPS distance (lower = more similar). When lpips/torch are not installed this
        returns 0.0, which is indistinguishable from "identical": check ``lpips_available()``
        before treating the value as a measurement.
    """
    global _lpips_model

    try:
        import lpips
        import torch
    except ImportError:
        logger.warning("LPIPS not available (install lpips package)")
        return 0.0

    # Lazy load model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net, verbose=False)
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()

    tensor1, tensor2 = _to_torch_pair(img1, img2)

    # LPIPS expects (N, C, H, W) and range [-1, 1]
    if tensor1.ndim == 3:
        tensor1 = tensor1.unsqueeze(0)
        tensor2 = tensor2.unsqueeze(0)

    # Convert [0, 1] to [-1, 1]
    tensor1 = tensor1 * 2 - 1
    tensor2 = tensor2 * 2 - 1

    if torch.cuda.is_available():
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()

    with torch.no_grad():
        distance = _lpips_model(tensor1, tensor2)

    return float(distance.item())


# ============================================================================
# IoU - Intersection over Union
# ============================================================================


def _binary_mask(mask: ImageLike, threshold: float) -> np.ndarray:
    """Binarise a segmentation mask.

    Bool and integer masks (``{0, 1}`` / ``{0, 255}`` masks, class-id maps, and mask files,
    which load as 8-bit) hold labels, not intensities: any non-zero value is foreground and
    they never go through image range normalisation, which would divide a ``{0, 1}`` uint8
    mask by 255 and empty it. Float masks are soft masks: they keep the image range
    convention and are compared against ``threshold``.
    """
    if not isinstance(mask, (np.ndarray, Path)) and hasattr(mask, "is_floating_point") and not mask.is_floating_point():
        # torch bool/integer tensor: binarise before a conversion that may fall back to float32
        return _torch_tensor_to_numpy(mask > 0) > 0
    arr = _raw_array(mask)
    if arr.dtype.kind in "biu":
        return arr > 0
    return _to_numpy(arr) > threshold


def segmentation_iou(
    mask1: ImageLike,
    mask2: ImageLike,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection over Union for segmentation masks.

    IoU measures overlap between predicted and ground truth masks.
    Range: [0, 1], where 1 = perfect overlap.

    Args:
        mask1: First mask (prediction)
        mask2: Second mask (ground truth)
        threshold: Binarization threshold for soft (float) masks; bool and integer
            masks count every non-zero value as foreground

    Returns:
        IoU score
    """
    mask1_bin = _binary_mask(mask1, threshold)
    mask2_bin = _binary_mask(mask2, threshold)

    # Handle multi-channel masks (take first channel or max)
    if mask1_bin.ndim == 3:
        mask1_bin = mask1_bin[..., 0] if mask1_bin.shape[-1] > 1 else mask1_bin.squeeze()
    if mask2_bin.ndim == 3:
        mask2_bin = mask2_bin[..., 0] if mask2_bin.shape[-1] > 1 else mask2_bin.squeeze()

    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def dice_coefficient(
    mask1: ImageLike,
    mask2: ImageLike,
    threshold: float = 0.5,
) -> float:
    """Compute Dice coefficient for segmentation masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Range: [0, 1], where 1 = perfect overlap.

    Args:
        mask1: First mask
        mask2: Second mask
        threshold: Binarization threshold for soft (float) masks; bool and integer
            masks count every non-zero value as foreground

    Returns:
        Dice coefficient
    """
    mask1_bin = _binary_mask(mask1, threshold)
    mask2_bin = _binary_mask(mask2, threshold)

    if mask1_bin.ndim == 3:
        mask1_bin = mask1_bin[..., 0]
    if mask2_bin.ndim == 3:
        mask2_bin = mask2_bin[..., 0]

    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()

    if total == 0:
        return 1.0

    return float(2 * intersection / total)


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a metric value to [0, 1] range.

    Args:
        value: Raw metric value
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value in [0, 1]; a NaN value scores 0.0 (fail closed),
        while +/-inf keep their ordered meaning (e.g. PSNR of identical
        images is +inf and maps to 1.0).
    """
    if math.isnan(value):
        # ``min(1.0, nan)`` is 1.0, so NaN would otherwise clamp to a perfect score.
        return 0.0
    if max_val == min_val:
        return 0.5

    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def psnr_to_score(psnr_val: float) -> float:
    """Convert PSNR to normalized score.

    Mapping: <20 dB -> 0, 20-40 dB -> 0-1, >40 dB -> 1
    """
    return normalize_score(psnr_val, 20.0, 40.0)


def lpips_to_score(lpips_val: float) -> float:
    """Convert LPIPS distance to similarity score.

    Lower LPIPS = higher similarity, so we invert.
    Mapping: 0 -> 1, 0.5 -> 0
    """
    return max(0.0, 1.0 - 2 * lpips_val)
