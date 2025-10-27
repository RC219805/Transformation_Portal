"""
Depth map utility functions.

Provides common operations for depth processing including:
- Normalization and conversion
- Edge detection
- Zone creation
- Visualization
- Statistics
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def normalize_depth(
    depth: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize depth map to [0, 1] range.

    Args:
        depth: Input depth map
        min_val: Minimum value for normalization (auto if None)
        max_val: Maximum value for normalization (auto if None)

    Returns:
        Normalized depth map [0, 1]
    """
    if min_val is None:
        min_val = depth.min()
    if max_val is None:
        max_val = depth.max()

    if max_val - min_val < 1e-8:
        logger.warning("Depth map has no variation, returning zeros")
        return np.zeros_like(depth)

    normalized = (depth - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1).astype(np.float32)


def depth_to_disparity(depth: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Convert depth to disparity.

    Disparity is inversely proportional to depth: d = 1 / z

    Args:
        depth: Depth map
        epsilon: Small value to avoid division by zero

    Returns:
        Disparity map
    """
    return 1.0 / (depth + epsilon)


def disparity_to_depth(disparity: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Convert disparity to depth.

    Args:
        disparity: Disparity map
        epsilon: Small value to avoid division by zero

    Returns:
        Depth map
    """
    return 1.0 / (disparity + epsilon)


def compute_depth_edges(
    depth: np.ndarray,
    method: str = "sobel",
    threshold: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Detect edges in depth map.

    Args:
        depth: Depth map
        method: Edge detection method ('sobel', 'canny', 'laplacian')
        threshold: Edge threshold (auto if None)
        normalize: Normalize output to [0, 1]

    Returns:
        Edge map
    """
    if method == "sobel":
        # Compute gradients
        grad_x = sobel(depth, axis=1)
        grad_y = sobel(depth, axis=0)

        # Gradient magnitude
        edges = np.sqrt(grad_x**2 + grad_y**2)

    elif method == "canny":
        # Convert to uint8 for Canny
        depth_uint8 = (normalize_depth(depth) * 255).astype(np.uint8)

        # Apply Canny
        low_threshold = threshold or 50
        high_threshold = threshold or 150
        edges = cv2.Canny(depth_uint8, low_threshold, high_threshold)
        edges = edges.astype(np.float32) / 255.0

    elif method == "laplacian":
        # Laplacian edge detection
        edges = cv2.Laplacian(depth, cv2.CV_32F)
        edges = np.abs(edges)

    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    # Normalize
    if normalize and edges.max() > 0:
        edges = edges / edges.max()

    # Apply threshold if specified
    if threshold is not None and method == "sobel":
        edges = (edges > threshold).astype(np.float32)

    return edges


def create_depth_zones(
    depth: np.ndarray,
    num_zones: int = 3,
    method: str = "percentile",
    smooth_sigma: float = 2.0,
) -> List[np.ndarray]:
    """
    Create depth zone masks with smooth transitions.

    Args:
        depth: Depth map [0, 1]
        num_zones: Number of zones
        method: Zoning method ('percentile', 'linear', 'log')
        smooth_sigma: Gaussian smoothing for zone boundaries

    Returns:
        List of zone masks (each HxW, [0, 1], sum to 1)
    """
    if method == "percentile":
        # Percentile-based quantization (robust to outliers)
        boundaries = np.percentile(
            depth,
            np.linspace(0, 100, num_zones + 1)
        )

    elif method == "linear":
        # Linear quantization
        boundaries = np.linspace(depth.min(), depth.max(), num_zones + 1)

    elif method == "log":
        # Logarithmic quantization (more zones in foreground)
        log_depth = np.log(depth + 1e-8)
        log_boundaries = np.linspace(log_depth.min(), log_depth.max(), num_zones + 1)
        boundaries = np.exp(log_boundaries)

    else:
        raise ValueError(f"Unknown zoning method: {method}")

    # Create hard zone masks
    zone_masks = []
    for i in range(num_zones):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == num_zones - 1:
            mask = (depth >= lower) & (depth <= upper)
        else:
            mask = (depth >= lower) & (depth < upper)

        zone_masks.append(mask.astype(np.float32))

    # Smooth boundaries
    if smooth_sigma > 0:
        zone_masks = [
            gaussian_filter(mask, sigma=smooth_sigma)
            for mask in zone_masks
        ]

    # Normalize so masks sum to 1
    mask_sum = np.sum(zone_masks, axis=0) + 1e-8
    zone_masks = [mask / mask_sum for mask in zone_masks]

    return zone_masks


def smooth_depth(
    depth: np.ndarray,
    method: str = "bilateral",
    sigma: float = 5.0,
    edge_preserve: float = 0.1,
) -> np.ndarray:
    """
    Smooth depth map while preserving edges.

    Args:
        depth: Input depth map
        method: Smoothing method ('gaussian', 'bilateral', 'median')
        sigma: Smoothing strength
        edge_preserve: Edge preservation strength (for bilateral)

    Returns:
        Smoothed depth map
    """
    if method == "gaussian":
        return gaussian_filter(depth, sigma=sigma)

    elif method == "bilateral":
        # Convert to uint8 for OpenCV
        depth_uint8 = (normalize_depth(depth) * 255).astype(np.uint8)

        # Apply bilateral filter
        d = int(sigma * 2)
        sigma_color = edge_preserve * 255
        sigma_space = sigma

        smoothed = cv2.bilateralFilter(
            depth_uint8,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

        return smoothed.astype(np.float32) / 255.0

    elif method == "median":
        # Median filter
        ksize = int(sigma * 2) + 1
        smoothed = cv2.medianBlur(depth, ksize)
        return smoothed

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def visualize_depth(
    depth: np.ndarray,
    colormap: str = "turbo",
    invert: bool = False,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create colorized visualization of depth map.

    Args:
        depth: Depth map [0, 1]
        colormap: Matplotlib colormap name
        invert: Invert depth (near=red, far=blue)
        save_path: Save visualization to file (optional)

    Returns:
        RGB visualization (HxWx3, uint8)
    """
    # Normalize
    depth_norm = normalize_depth(depth)

    # Invert if requested
    if invert:
        depth_norm = 1.0 - depth_norm

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_norm)

    # Convert to uint8 RGB
    rgb = (colored[..., :3] * 255).astype(np.uint8)

    # Save if requested
    if save_path:
        from PIL import Image
        img = Image.fromarray(rgb)
        img.save(save_path)
        logger.info(f"Saved depth visualization: {save_path}")

    return rgb


def depth_statistics(depth: np.ndarray) -> dict:
    """
    Compute depth map statistics.

    Args:
        depth: Depth map

    Returns:
        Dictionary of statistics
    """
    stats = {
        'min': float(depth.min()),
        'max': float(depth.max()),
        'mean': float(depth.mean()),
        'median': float(np.median(depth)),
        'std': float(depth.std()),
        'percentile_10': float(np.percentile(depth, 10)),
        'percentile_90': float(np.percentile(depth, 90)),
    }

    # Compute gradient statistics
    grad_x = np.abs(sobel(depth, axis=1))
    grad_y = np.abs(sobel(depth, axis=0))
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    stats.update({
        'gradient_mean': float(grad_magnitude.mean()),
        'gradient_max': float(grad_magnitude.max()),
        'edge_density': float((grad_magnitude > grad_magnitude.mean()).sum() / grad_magnitude.size),
    })

    return stats


def align_depth_to_image(
    depth: np.ndarray,
    target_shape: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize depth map to match image dimensions.

    Args:
        depth: Depth map
        target_shape: Target (height, width)
        interpolation: Interpolation method ('bilinear', 'bicubic', 'nearest')

    Returns:
        Resized depth map
    """
    if depth.shape[:2] == target_shape:
        return depth

    interp_map = {
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
    }

    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    resized = cv2.resize(
        depth,
        (target_shape[1], target_shape[0]),  # width, height
        interpolation=interp_flag
    )

    return resized


def inpaint_depth_holes(
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "telea",
) -> np.ndarray:
    """
    Fill holes in depth map using inpainting.

    Args:
        depth: Depth map with holes (0 or NaN values)
        mask: Hole mask (1=hole, 0=valid). Auto-detected if None.
        method: Inpainting method ('telea', 'ns')

    Returns:
        Inpainted depth map
    """
    # Auto-detect holes
    if mask is None:
        mask = ((depth == 0) | np.isnan(depth)).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Convert to uint8 for OpenCV inpainting
    depth_uint8 = (normalize_depth(depth) * 255).astype(np.uint8)

    # Inpaint
    if method == "telea":
        inpainted = cv2.inpaint(
            depth_uint8,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
    elif method == "ns":
        inpainted = cv2.inpaint(
            depth_uint8,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_NS
        )
    else:
        raise ValueError(f"Unknown inpainting method: {method}")

    return inpainted.astype(np.float32) / 255.0


def create_depth_of_field_map(
    depth: np.ndarray,
    focus_distance: float = 0.5,
    aperture: float = 2.8,
    focal_length: float = 50.0,
    sensor_size: float = 35.0,
) -> np.ndarray:
    """
    Create circle of confusion map for depth-of-field simulation.

    Args:
        depth: Depth map [0, 1]
        focus_distance: Focus plane (0=near, 1=far)
        aperture: Lens aperture (f-number)
        focal_length: Focal length (mm)
        sensor_size: Sensor size (mm)

    Returns:
        Circle of confusion map (blur radius in pixels)
    """
    # Compute circle of confusion
    # CoC = (A * f * |D - d|) / (D * (d - f))
    # Where:
    #   A = aperture diameter = focal_length / f_number
    #   f = focal_length
    #   D = focus distance
    #   d = object distance (depth)

    aperture_diameter = focal_length / aperture

    # Compute CoC
    numerator = aperture_diameter * focal_length * np.abs(depth - focus_distance)
    denominator = depth * (focus_distance - focal_length / 1000.0) + 1e-8

    coc = numerator / denominator

    # Normalize to pixel radius
    # Scale by sensor size
    coc_normalized = coc * (sensor_size / 1000.0)

    return coc_normalized
