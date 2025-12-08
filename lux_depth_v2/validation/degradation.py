"""Synthetic degradation pipeline for creating reference pairs.

Applies realistic degradations to high-quality images to create test inputs
while preserving the original as ground truth for validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def apply_downsample_degradation(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Downsample image by scale factor.
    
    Args:
        img: Input image, shape (H, W, C), float [0, 1]
        scale: Downsampling factor (2 or 4)
    
    Returns:
        Downsampled image
    """
    try:
        import cv2
        h, w = img.shape[:2]
        new_h, new_w = h // scale, w // scale
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Fallback: numpy-based simple downsampling
        return img[::scale, ::scale]


def apply_blur_degradation(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur to simulate out-of-focus or motion blur.
    
    Args:
        img: Input image, shape (H, W, C), float [0, 1]
        sigma: Gaussian kernel standard deviation
    
    Returns:
        Blurred image
    """
    try:
        from scipy.ndimage import gaussian_filter
        if img.ndim == 3:
            return np.stack([gaussian_filter(img[..., i], sigma=sigma) for i in range(img.shape[-1])], axis=-1)
        else:
            return gaussian_filter(img, sigma=sigma)
    except ImportError:
        # No blur if scipy unavailable
        return img.copy()


def apply_noise_degradation(img: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise.
    
    Args:
        img: Input image, shape (H, W, C), float [0, 1]
        noise_level: Standard deviation of noise
    
    Returns:
        Noisy image, clipped to [0, 1]
    """
    noise = np.random.normal(0, noise_level, img.shape).astype(img.dtype)
    return np.clip(img + noise, 0.0, 1.0)


def apply_jpeg_compression(img: np.ndarray, quality: int = 70) -> np.ndarray:
    """Apply JPEG compression artifacts.
    
    Args:
        img: Input image, shape (H, W, C), float [0, 1]
        quality: JPEG quality (1-100, lower = more artifacts)
    
    Returns:
        Compressed image
    """
    try:
        import cv2
        # Convert to uint8 for JPEG encoding
        img_u8 = (img * 255).astype(np.uint8)
        
        # Encode and decode to simulate compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img_u8, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        
        # Convert back to float [0, 1]
        return decoded.astype(np.float32) / 255.0
    except ImportError:
        # No compression if cv2 unavailable
        return img.copy()


def create_synthetic_pair(
    original: np.ndarray,
    degradations: Optional[List[str]] = None,
    downsample_scale: int = 4,
    blur_sigma: float = 1.5,
    noise_level: float = 0.02,
    jpeg_quality: int = 70
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic degraded/reference pair from a high-quality image.
    
    Args:
        original: High-quality input image, shape (H, W, C), float [0, 1]
        degradations: List of degradations to apply (order matters)
                      Options: ['downsample', 'blur', 'noise', 'compress']
        downsample_scale: Scale factor for downsampling
        blur_sigma: Gaussian blur sigma
        noise_level: Noise standard deviation
        jpeg_quality: JPEG quality for compression
    
    Returns:
        Tuple of (degraded_image, reference_image)
    """
    if degradations is None:
        degradations = ["downsample", "blur", "noise", "compress"]
    
    degraded = original.copy()
    
    for deg in degradations:
        if deg == "downsample":
            degraded = apply_downsample_degradation(degraded, scale=downsample_scale)
        elif deg == "blur":
            degraded = apply_blur_degradation(degraded, sigma=blur_sigma)
        elif deg == "noise":
            degraded = apply_noise_degradation(degraded, noise_level=noise_level)
        elif deg == "compress":
            degraded = apply_jpeg_compression(degraded, quality=jpeg_quality)
    
    return degraded, original


def save_synthetic_pair(
    degraded: np.ndarray,
    reference: np.ndarray,
    output_dir: Path,
    basename: str
) -> Tuple[Path, Path]:
    """Save degraded and reference images to disk.
    
    Args:
        degraded: Degraded image
        reference: Reference (ground truth) image
        output_dir: Output directory
        basename: Base filename (without extension)
    
    Returns:
        Tuple of (degraded_path, reference_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    degraded_path = output_dir / f"{basename}_degraded.png"
    reference_path = output_dir / f"{basename}_reference.tif"
    
    # Save degraded as PNG (8-bit)
    try:
        import cv2
        degraded_u8 = (degraded * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(degraded_path), cv2.cvtColor(degraded_u8, cv2.COLOR_RGB2BGR))
    except ImportError:
        try:
            from PIL import Image
            degraded_u8 = (degraded * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(degraded_u8).save(degraded_path)
        except ImportError:
            pass
    
    # Save reference as 16-bit TIFF
    try:
        import tifffile
        reference_u16 = (reference * 65535).clip(0, 65535).astype(np.uint16)
        tifffile.imwrite(reference_path, reference_u16)
    except ImportError:
        try:
            from PIL import Image
            reference_u16 = (reference * 65535).clip(0, 65535).astype(np.uint16)
            Image.fromarray(reference_u16).save(reference_path)
        except ImportError:
            pass
    
    return degraded_path, reference_path
