"""Simplified differentiable Gaussian splatting rasterizer (Phase 6A).

Implements a PyTorch-native, MPS-compatible rasterizer for 3D Gaussian Splatting:
- Project 3D Gaussians to 2D screen space
- Differentiable alpha compositing
- Painter's algorithm (depth sort)
- Support for Apple Silicon (MPS), CUDA, and CPU

Simplifications for Phase 6A:
- Isotropic Gaussians (ignore rotation for now)
- Simple pinhole projection (no lens distortion)
- Back-to-front alpha compositing (no tile-based rendering)
- Fixed opacity (no learned alpha blending)

Performance targets:
- Memory: <8GB VRAM on MPS
- Throughput: ~10-30 FPS at 480p on M-series chips
- Gradient flow: stable (no NaN/inf)

Architecture:
- Pure PyTorch implementation (no CUDA kernels)
- Device-agnostic (MPS, CUDA, CPU)
- Differentiable end-to-end for optimization

References:
- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- Inria GraphDeco implementation (research license)
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 3x3 rotation matrices.

    Args:
        quaternions: (N, 4) quaternions as [w, x, y, z].

    Returns:
        Rotation matrices (N, 3, 3).
    """
    # Normalize to ensure the resulting rotation matrix is orthonormal even if
    # callers pass non-unit quaternions.
    norms = torch.norm(quaternions, dim=1, keepdim=True)
    safe_norms = torch.clamp(norms, min=1e-8)
    q_normalized = quaternions / safe_norms

    # Fallback for pathological near-zero quaternions: treat as identity.
    near_zero = (norms < 1e-8).squeeze(1)
    if near_zero.any():
        q_normalized = q_normalized.clone()
        q_normalized[near_zero] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quaternions.device, dtype=quaternions.dtype)

    # Extract quaternion components
    w, x, y, z = q_normalized[:, 0], q_normalized[:, 1], q_normalized[:, 2], q_normalized[:, 3]

    # Compute rotation matrix elements
    # fmt: off
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=1),
    ], dim=1)
    # fmt: on

    return R


def compute_3d_covariance(
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Compute 3D covariance matrices from scales and rotations.

    Covariance: Σ = R S S^T R^T where R is rotation and S is diagonal scale matrix.

    Args:
        scales: (N, 3) scale factors [sx, sy, sz].
        rotations: (N, 4) quaternions [w, x, y, z].

    Returns:
        Covariance matrices (N, 3, 3).
    """
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

    # Create diagonal scale matrix
    S = torch.diag_embed(scales)  # (N, 3, 3)

    # Covariance: Σ = R S S^T R^T
    RS = torch.bmm(R, S)  # (N, 3, 3)
    cov = torch.bmm(RS, RS.transpose(1, 2))  # (N, 3, 3)

    return cov


def project_gaussians_2d(
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    use_rotation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D screen space.

    Args:
        positions: (N, 3) 3D positions in world coordinates.
        scales: (N, 3) scale factors [sx, sy, sz].
        rotations: (N, 4) quaternions [w, x, y, z].
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) camera extrinsic matrix.
        image_size: (H, W) image dimensions.
        use_rotation: If False, use isotropic Gaussians (simpler, faster).

    Returns:
        Tuple of:
            - mean_2d: (N, 2) 2D centers in pixel coordinates [u, v].
            - cov_2d: (N, 2, 2) 2D covariance matrices.
            - depths: (N,) depth values (for sorting).
            - valid_mask: (N,) boolean mask for visible Gaussians.
    """
    N = positions.shape[0]
    device = positions.device

    # Transform to camera coordinates
    positions_hom = torch.cat([positions, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)
    positions_cam = torch.matmul(extrinsics, positions_hom.T).T[:, :3]  # (N, 3)

    # Extract depth
    depths = positions_cam[:, 2]  # (N,)

    # Visibility culling: keep only points in front of camera
    valid_mask = depths > 0.01  # At least 1cm in front

    # Project to 2D (pinhole projection)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Perspective projection
    u = fx * positions_cam[:, 0] / (positions_cam[:, 2] + 1e-8) + cx
    v = fy * positions_cam[:, 1] / (positions_cam[:, 2] + 1e-8) + cy
    mean_2d = torch.stack([u, v], dim=1)  # (N, 2)

    # Image bounds culling
    H, W = image_size
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_mask = valid_mask & in_bounds

    # Compute 2D covariance
    if use_rotation:
        # Full covariance projection (more accurate but slower)
        cov_3d = compute_3d_covariance(scales, rotations)  # (N, 3, 3)

        # Jacobian of perspective projection
        J = torch.zeros(N, 2, 3, device=device)
        z_inv = 1.0 / (positions_cam[:, 2] + 1e-8)
        J[:, 0, 0] = fx * z_inv
        J[:, 0, 2] = -fx * positions_cam[:, 0] * z_inv * z_inv
        J[:, 1, 1] = fy * z_inv
        J[:, 1, 2] = -fy * positions_cam[:, 1] * z_inv * z_inv

        # Transform 3D covariance to 2D: Σ_2d = J Σ_3d J^T
        # First: R_cam = R_world (rotation part of extrinsics)
        R_cam = extrinsics[:3, :3]  # (3, 3)
        cov_cam = torch.matmul(torch.matmul(R_cam, cov_3d), R_cam.T)  # (N, 3, 3)

        # Project to 2D
        cov_2d = torch.bmm(torch.bmm(J, cov_cam), J.transpose(1, 2))  # (N, 2, 2)
    else:
        # Simplified isotropic Gaussians (Phase 6A default)
        # Use average scale projected to screen space
        scale_avg = scales.mean(dim=1)  # (N,)
        z_inv = 1.0 / (depths + 1e-8)

        # Approximate 2D scale
        scale_2d_x = fx * scale_avg * z_inv
        scale_2d_y = fy * scale_avg * z_inv

        # Isotropic 2D covariance (diagonal)
        cov_2d = torch.zeros(N, 2, 2, device=device)
        cov_2d[:, 0, 0] = scale_2d_x * scale_2d_x
        cov_2d[:, 1, 1] = scale_2d_y * scale_2d_y

    # Add small constant to prevent singularity
    epsilon = 1e-4
    cov_2d[:, 0, 0] += epsilon
    cov_2d[:, 1, 1] += epsilon

    return mean_2d, cov_2d, depths, valid_mask


def evaluate_gaussian_2d(
    pixel_coords: torch.Tensor,
    mean_2d: torch.Tensor,
    cov_2d_inv: torch.Tensor,
) -> torch.Tensor:
    """Evaluate 2D Gaussian at pixel coordinates.

    Uses: G(x) = exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))

    Args:
        pixel_coords: (H, W, 2) pixel coordinates [u, v].
        mean_2d: (N, 2) Gaussian centers.
        cov_2d_inv: (N, 2, 2) inverse covariance matrices.

    Returns:
        Gaussian weights (N, H, W) in [0, 1].
    """
    H, W = pixel_coords.shape[:2]
    N = mean_2d.shape[0]
    device = pixel_coords.device

    # Flatten pixel coordinates: (P, 2) where P = H * W
    pixels_flat = pixel_coords.reshape(-1, 2)

    # Vectorized over all gaussians:
    # diff: (N, P, 2), tmp: (N, P, 2), mahal_sq: (N, P)
    diff = pixels_flat.unsqueeze(0) - mean_2d.unsqueeze(1)
    tmp = torch.einsum("npc,ncd->npd", diff, cov_2d_inv)
    mahal_sq = (tmp * diff).sum(dim=-1)

    return torch.exp(-0.5 * mahal_sq).reshape(N, H, W)


def _compute_splat_bounds(
    mean_uv: torch.Tensor,
    cov_2d: torch.Tensor,
    image_size: Tuple[int, int],
    radius_sigma: float = 3.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute an integer image window that bounds most of a gaussian footprint."""
    H, W = image_size

    sigma_u = torch.sqrt(torch.clamp(cov_2d[0, 0], min=1e-8))
    sigma_v = torch.sqrt(torch.clamp(cov_2d[1, 1], min=1e-8))

    radius_u = max(1, int(math.ceil(radius_sigma * float(sigma_u.item()))))
    radius_v = max(1, int(math.ceil(radius_sigma * float(sigma_v.item()))))

    center_u = float(mean_uv[0].item())
    center_v = float(mean_uv[1].item())

    u_min = max(0, int(math.floor(center_u)) - radius_u)
    u_max = min(W, int(math.floor(center_u)) + radius_u + 1)
    v_min = max(0, int(math.floor(center_v)) - radius_v)
    v_max = min(H, int(math.floor(center_v)) + radius_v + 1)

    if u_min >= u_max or v_min >= v_max:
        return None
    return u_min, u_max, v_min, v_max


def render_gaussians(
    positions: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    use_rotation: bool = False,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Simplified differentiable Gaussian splatting rasterizer.

    Approach:
    1. Project 3D Gaussians to 2D screen space
    2. Compute 2D Gaussian footprint for each splat
    3. Sort by depth (painter's algorithm)
    4. Alpha composite in back-to-front order

    Args:
        positions: (N, 3) 3D positions in world coordinates.
        colors: (N, 3) RGB colors in [0, 1].
        scales: (N, 3) scale factors [sx, sy, sz].
        rotations: (N, 4) quaternions [w, x, y, z].
        opacities: (N, 1) opacity values in [0, 1].
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) camera extrinsic matrix.
        image_size: (H, W) image dimensions.
        use_rotation: If False, use isotropic Gaussians (default for Phase 6A).
        device: Target device ("mps", "cuda", "cpu"). Auto-detected if None.

    Returns:
        Rendered image (H, W, 3) RGB in [0, 1].
    """
    if device is None:
        device = positions.device

    H, W = image_size

    # Project to 2D
    mean_2d, cov_2d, depths, valid_mask = project_gaussians_2d(
        positions, scales, rotations, intrinsics, extrinsics, image_size, use_rotation
    )

    # Filter visible Gaussians
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        # No visible Gaussians, return black image
        logger.warning("No visible Gaussians in view")
        return torch.zeros(H, W, 3, device=device)

    mean_2d = mean_2d[valid_mask]
    cov_2d = cov_2d[valid_mask]
    depths = depths[valid_mask]
    colors = colors[valid_mask]
    opacities = opacities[valid_mask]

    # Sort by depth (back to front for alpha compositing)
    depth_order = torch.argsort(depths, descending=True)
    mean_2d = mean_2d[depth_order]
    cov_2d = cov_2d[depth_order]
    colors = colors[depth_order]
    opacities = opacities[depth_order]

    # Compute inverse covariance (for Gaussian evaluation)
    try:
        cov_2d_inv = torch.inverse(cov_2d)  # (N_valid, 2, 2)
    except RuntimeError:
        # Handle singular matrices
        logger.warning("Singular covariance matrices detected, adding regularization")
        cov_2d = cov_2d + torch.eye(2, device=device) * 1e-3
        cov_2d_inv = torch.inverse(cov_2d)

    # Alpha composite (back to front)
    rendered = torch.zeros(H, W, 3, device=device)
    accumulated_alpha = torch.zeros(H, W, device=device)

    for i in range(mean_2d.shape[0]):
        bounds = _compute_splat_bounds(mean_2d[i], cov_2d[i], image_size)
        if bounds is None:
            continue
        u_min, u_max, v_min, v_max = bounds

        # Evaluate only inside the local gaussian window to avoid allocating
        # a full (N, H, W) tensor during compositing.
        u_coords = torch.arange(u_min, u_max, device=device, dtype=mean_2d.dtype)
        v_coords = torch.arange(v_min, v_max, device=device, dtype=mean_2d.dtype)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        local_pixels = torch.stack([u_grid, v_grid], dim=-1)  # (h, w, 2)

        diff = local_pixels - mean_2d[i].view(1, 1, 2)
        tmp = torch.einsum("...c,cd->...d", diff, cov_2d_inv[i])
        mahal_sq = (tmp * diff).sum(dim=-1)
        weight = torch.exp(-0.5 * mahal_sq)

        opacity = opacities[i, 0]
        alpha_i = weight * opacity

        rendered_patch = rendered[v_min:v_max, u_min:u_max, :]
        alpha_patch = accumulated_alpha[v_min:v_max, u_min:u_max]
        transmittance = 1.0 - alpha_patch

        rendered[v_min:v_max, u_min:u_max, :] = (
            rendered_patch + alpha_i.unsqueeze(-1) * transmittance.unsqueeze(-1) * colors[i]
        )
        accumulated_alpha[v_min:v_max, u_min:u_max] = alpha_patch + alpha_i * transmittance

    # Clamp to [0, 1]
    rendered = torch.clamp(rendered, 0.0, 1.0)

    return rendered


def render_gaussians_fast(
    positions: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    max_gaussians: int = 1000,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Fast rasterizer with Gaussian culling for optimization.

    Renders only the closest max_gaussians to reduce memory/computation.
    Useful during optimization when full quality is not needed.

    Args:
        Same as render_gaussians(), plus:
        max_gaussians: Maximum number of Gaussians to render (default: 1000).

    Returns:
        Rendered image (H, W, 3) RGB in [0, 1].
    """
    if device is None:
        device = positions.device

    N = positions.shape[0]

    # If below threshold, use full rendering
    if N <= max_gaussians:
        return render_gaussians(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, False, device
        )

    # Sort by distance to camera and keep closest
    positions_hom = torch.cat([positions, torch.ones(N, 1, device=device)], dim=1)
    positions_cam = torch.matmul(extrinsics, positions_hom.T).T[:, :3]
    distances = torch.norm(positions_cam, dim=1)

    closest_indices = torch.argsort(distances)[:max_gaussians]

    # Render subset
    return render_gaussians(
        positions[closest_indices],
        colors[closest_indices],
        scales[closest_indices],
        rotations[closest_indices],
        opacities[closest_indices],
        intrinsics,
        extrinsics,
        image_size,
        use_rotation=False,
        device=device,
    )


def compute_rgb_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute RGB reconstruction loss.

    Args:
        rendered: (H, W, 3) rendered image.
        target: (H, W, 3) target image.
        reduction: Loss reduction ("mean" or "sum").

    Returns:
        Loss scalar.
    """
    loss = F.mse_loss(rendered, target, reduction=reduction)
    return loss


def compute_depth_loss(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute depth consistency loss (optional).

    Args:
        rendered_depth: (H, W) rendered depth.
        target_depth: (H, W) target depth from prior.
        mask: Optional (H, W) valid depth mask.

    Returns:
        Loss scalar.
    """
    if mask is not None:
        diff = (rendered_depth - target_depth) * mask
        loss = (diff * diff).sum() / (mask.sum() + 1e-8)
    else:
        loss = F.mse_loss(rendered_depth, target_depth)

    return loss
