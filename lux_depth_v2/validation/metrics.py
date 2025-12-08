"""Image quality metrics for validation.

Provides reference-based and no-reference metrics:
- Fidelity: SSIM, PSNR, MSE
- Perceptual: LPIPS
- Aesthetic: NIMA (no-reference)
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def compute_ssim(img1: np.ndarray, img2: np.ndarray, multichannel: bool = True) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image, shape (H, W, C) or (H, W), float [0, 1]
        img2: Second image, same shape as img1
        multichannel: If True, compute SSIM across channels
    
    Returns:
        SSIM score, higher is better (max 1.0)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(img1, img2, multichannel=multichannel, data_range=1.0, channel_axis=-1 if multichannel else None))
    except ImportError:
        # Fallback: simplified SSIM implementation
        if img1.ndim == 3 and multichannel:
            # Average across channels
            return float(np.mean([compute_ssim(img1[..., i], img2[..., i], multichannel=False) for i in range(img1.shape[-1])]))
        
        # Single channel simplified SSIM
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(np.clip(ssim_val, -1, 1))


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image, shape (H, W, C) or (H, W), float [0, max_val]
        img2: Second image, same shape as img1
        max_val: Maximum possible pixel value
    
    Returns:
        PSNR in dB, higher is better (typical range: 20-50 dB)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_lpips(img1: np.ndarray, img2: np.ndarray, net: str = "alex", device: str = "cpu") -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1: First image, shape (H, W, C), float [0, 1]
        img2: Second image, same shape as img1
        net: Network to use ('alex' or 'vgg')
        device: Device for computation ('cpu', 'cuda', 'mps')
    
    Returns:
        LPIPS distance, lower is better (typical range: 0.0-1.0)
    """
    try:
        import lpips
        import torch
        
        # Initialize model (cached after first call)
        if not hasattr(compute_lpips, "_model"):
            compute_lpips._model = lpips.LPIPS(net=net).to(device)
            compute_lpips._model.eval()
        
        model = compute_lpips._model
        
        # Convert to torch tensors [N, C, H, W]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Normalize to [-1, 1] as expected by LPIPS
        t1 = t1 * 2.0 - 1.0
        t2 = t2 * 2.0 - 1.0
        
        with torch.no_grad():
            dist = model(t1, t2)
        
        return float(dist.item())
    
    except ImportError:
        # Fallback: use MSE as proxy for perceptual similarity
        mse = np.mean((img1 - img2) ** 2)
        return float(np.sqrt(mse))


def compute_nima(img: np.ndarray, device: str = "cpu") -> float:
    """Compute NIMA (Neural Image Assessment) aesthetic score.
    
    Args:
        img: Input image, shape (H, W, C), float [0, 1]
        device: Device for computation ('cpu', 'cuda', 'mps')
    
    Returns:
        NIMA score, higher is better (typical range: 1-10)
    """
    try:
        # NIMA implementation would require pre-trained model
        # For now, use heuristic-based aesthetic proxy
        return _heuristic_aesthetic_score(img)
    
    except Exception:
        return _heuristic_aesthetic_score(img)


def _heuristic_aesthetic_score(img: np.ndarray) -> float:
    """Heuristic-based aesthetic score as NIMA fallback.
    
    Considers:
    - Dynamic range
    - Color balance
    - Sharpness (gradient magnitude)
    - Contrast
    
    Returns:
        Score in range [1, 10]
    """
    # Dynamic range (0-1 scale)
    dr = float(img.max() - img.min())
    
    # Color balance (RGB variance)
    if img.ndim == 3 and img.shape[-1] == 3:
        rgb_mean = img.mean(axis=(0, 1))
        color_balance = 1.0 - float(np.std(rgb_mean))
    else:
        color_balance = 0.5
    
    # Sharpness (gradient magnitude)
    try:
        from scipy.ndimage import sobel
        if img.ndim == 3:
            gray = img.mean(axis=-1)
        else:
            gray = img
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        sharpness = float(np.sqrt(sx**2 + sy**2).mean())
    except ImportError:
        # Simple gradient approximation
        if img.ndim == 3:
            gray = img.mean(axis=-1)
        else:
            gray = img
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        sharpness = float(gx + gy)
    
    # Contrast (standard deviation)
    contrast = float(img.std())
    
    # Combine factors (weighted heuristic)
    score = (
        dr * 3.0 +
        color_balance * 2.0 +
        min(sharpness * 10, 3.0) +
        min(contrast * 5, 2.0)
    )
    
    # Scale to [1, 10]
    return float(np.clip(score, 1.0, 10.0))


def compute_all_metrics(
    img: np.ndarray,
    reference: Optional[np.ndarray] = None,
    device: str = "cpu"
) -> dict:
    """Compute all available metrics for an image.
    
    Args:
        img: Test image, shape (H, W, C), float [0, 1]
        reference: Optional reference image for reference-based metrics
        device: Device for computation
    
    Returns:
        Dictionary with metric scores
    """
    metrics = {}
    
    # No-reference metrics (always computed)
    metrics["nima"] = compute_nima(img, device=device)
    
    # Reference-based metrics (if reference provided)
    if reference is not None:
        metrics["ssim"] = compute_ssim(img, reference)
        metrics["psnr"] = compute_psnr(img, reference)
        metrics["lpips"] = compute_lpips(img, reference, device=device)
    
    return metrics
