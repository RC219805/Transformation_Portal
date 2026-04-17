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


def _to_numpy(img: ImageLike) -> np.ndarray:
    """Convert image to numpy array.

    Args:
        img: Image as numpy array, torch tensor, or path

    Returns:
        Numpy array in [0, 1] range
    """
    if isinstance(img, Path):
        try:
            import cv2

            arr = cv2.imread(str(img))
            if arr is None:
                raise ValueError(f"Failed to load image: {img}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr.astype(np.float32) / 255.0
        except ImportError:
            from PIL import Image

            pil_img = Image.open(img).convert("RGB")
            return np.array(pil_img).astype(np.float32) / 255.0

    if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):  # torch.Tensor
        arr = _torch_tensor_to_numpy(img)
    else:
        arr = np.asarray(img)

    # Normalize to [0, 1] if needed
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0

    return arr.astype(np.float32)


def _to_torch(img: ImageLike) -> "torch.Tensor":
    """Convert image to torch tensor.

    Args:
        img: Image as numpy array, torch tensor, or path

    Returns:
        Torch tensor in [0, 1] range, shape (C, H, W)
    """
    import torch

    if isinstance(img, torch.Tensor):
        tensor = img.float()
        # Ensure [0, 1] range
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        # Ensure (C, H, W) format
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor

    arr = _to_numpy(img)

    # Convert (H, W, C) to (C, H, W)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr).float()


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
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

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
    arr1 = _to_numpy(img1)
    arr2 = _to_numpy(img2)

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

    # Fallback: simplified SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1_sq = np.var(arr1)
    sigma2_sq = np.var(arr2)
    sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_val)


# ============================================================================
# LPIPS - Learned Perceptual Image Patch Similarity
# ============================================================================


_lpips_model: Optional["lpips.LPIPS"] = None


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
        LPIPS distance (lower = more similar)
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

    tensor1 = _to_torch(img1)
    tensor2 = _to_torch(img2)

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
        threshold: Binarization threshold for soft masks

    Returns:
        IoU score
    """
    arr1 = _to_numpy(mask1)
    arr2 = _to_numpy(mask2)

    # Handle multi-channel masks (take first channel or max)
    if arr1.ndim == 3:
        arr1 = arr1[..., 0] if arr1.shape[-1] > 1 else arr1.squeeze()
    if arr2.ndim == 3:
        arr2 = arr2[..., 0] if arr2.shape[-1] > 1 else arr2.squeeze()

    # Binarize
    mask1_bin = arr1 > threshold
    mask2_bin = arr2 > threshold

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
        threshold: Binarization threshold

    Returns:
        Dice coefficient
    """
    arr1 = _to_numpy(mask1)
    arr2 = _to_numpy(mask2)

    if arr1.ndim == 3:
        arr1 = arr1[..., 0]
    if arr2.ndim == 3:
        arr2 = arr2[..., 0]

    mask1_bin = arr1 > threshold
    mask2_bin = arr2 > threshold

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
        Normalized value in [0, 1]
    """
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
