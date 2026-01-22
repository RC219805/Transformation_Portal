"""
Pose alignment utilities using RANSAC.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def align_poses_ransac(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    ransac_iterations: int = 1000,
    inlier_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align predicted poses to ground truth using RANSAC.

    This computes a similarity transformation (rotation + translation + scale)
    that best aligns the predicted camera trajectory to the ground truth.

    Args:
        pred_poses: Predicted poses (N, 4, 4)
        gt_poses: Ground truth poses (N, 4, 4)
        ransac_iterations: Number of RANSAC iterations
        inlier_threshold: Inlier distance threshold in meters

    Returns:
        rotation: Alignment rotation matrix (3, 3)
        translation: Alignment translation vector (3,)
    """
    if len(pred_poses) != len(gt_poses):
        raise ValueError("Number of poses must match")

    # Extract camera positions
    pred_positions = pred_poses[:, :3, 3]  # (N, 3)
    gt_positions = gt_poses[:, :3, 3]  # (N, 3)

    best_inliers = 0
    best_rotation = np.eye(3)
    best_translation = np.zeros(3)

    # RANSAC loop
    for _ in range(ransac_iterations):
        # Sample 3 random correspondences
        if len(pred_positions) < 3:
            # Not enough points, use all
            sample_indices = np.arange(len(pred_positions))
        else:
            sample_indices = np.random.choice(len(pred_positions), size=3, replace=False)

        sample_pred = pred_positions[sample_indices]
        sample_gt = gt_positions[sample_indices]

        # Compute transformation from sample
        R, t = compute_rigid_transform(sample_pred, sample_gt)

        # Transform all predicted positions
        pred_aligned = (R @ pred_positions.T).T + t

        # Count inliers
        distances = np.linalg.norm(pred_aligned - gt_positions, axis=1)
        inliers = distances < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update best transformation
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_rotation = R
            best_translation = t

    # Refine using all inliers
    if best_inliers > 3:
        pred_aligned = (best_rotation @ pred_positions.T).T + best_translation
        distances = np.linalg.norm(pred_aligned - gt_positions, axis=1)
        inliers = distances < inlier_threshold

        if np.sum(inliers) > 3:
            best_rotation, best_translation = compute_rigid_transform(pred_positions[inliers], gt_positions[inliers])

    inlier_ratio = best_inliers / len(pred_poses)
    logger.info(f"RANSAC alignment: {best_inliers}/{len(pred_poses)} inliers ({inlier_ratio:.1%})")

    return best_rotation, best_translation


def compute_rigid_transform(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transformation (rotation + translation) from source to target.

    Uses Kabsch algorithm (SVD-based optimal alignment).

    Args:
        source: Source points (N, 3)
        target: Target points (N, 3)

    Returns:
        rotation: Rotation matrix (3, 3)
        translation: Translation vector (3,)
    """
    # Center the point clouds
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = target_center - R @ source_center

    return R, t


def apply_alignment(poses: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Apply alignment transformation to poses.

    Args:
        poses: Input poses (N, 4, 4)
        rotation: Alignment rotation (3, 3)
        translation: Alignment translation (3,)

    Returns:
        Aligned poses (N, 4, 4)
    """
    aligned_poses = poses.copy()

    for i in range(len(poses)):
        # Extract position and rotation
        pos = poses[i, :3, 3]
        R_cam = poses[i, :3, :3]

        # Apply alignment
        pos_aligned = rotation @ pos + translation
        R_aligned = rotation @ R_cam

        # Update pose
        aligned_poses[i, :3, 3] = pos_aligned
        aligned_poses[i, :3, :3] = R_aligned

    return aligned_poses


def align_scale(pred_points: np.ndarray, gt_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute optimal scale alignment between point clouds.

    This is useful when depth predictions are scale-ambiguous
    (e.g., monocular depth estimation).

    Args:
        pred_points: Predicted point cloud (N, 3)
        gt_points: Ground truth point cloud (N, 3)

    Returns:
        scale: Optimal scale factor
        rotation: Alignment rotation (3, 3)
        translation: Alignment translation (3,)
    """
    # Center point clouds
    pred_center = np.mean(pred_points, axis=0)
    gt_center = np.mean(gt_points, axis=0)

    pred_centered = pred_points - pred_center
    gt_centered = gt_points - gt_center

    # Compute scale
    pred_scale = np.linalg.norm(pred_centered, axis=1).mean()
    gt_scale = np.linalg.norm(gt_centered, axis=1).mean()

    scale = gt_scale / pred_scale

    # Scale predictions
    pred_scaled = pred_centered * scale

    # Compute rotation and translation
    rotation, translation = compute_rigid_transform(pred_scaled, gt_centered)

    # Adjust translation for original centers
    translation = gt_center - rotation @ (pred_center * scale)

    logger.info(f"Computed scale alignment: scale={scale:.3f}")

    return scale, rotation, translation


def align_poses_umeyama(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Align poses using Umeyama algorithm (similarity transform with scale).

    Args:
        pred_poses: Predicted poses (N, 4, 4)
        gt_poses: Ground truth poses (N, 4, 4)

    Returns:
        scale: Scale factor
        rotation: Rotation matrix (3, 3)
        translation: Translation vector (3,)
    """
    # Extract positions
    pred_positions = pred_poses[:, :3, 3]
    gt_positions = gt_poses[:, :3, 3]

    return align_scale(pred_positions, gt_positions)


def compute_alignment_error(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """
    Compute alignment error after applying transformation.

    Args:
        pred_poses: Predicted poses (N, 4, 4)
        gt_poses: Ground truth poses (N, 4, 4)
        rotation: Alignment rotation (3, 3)
        translation: Alignment translation (3,)

    Returns:
        Mean alignment error in meters
    """
    # Apply alignment
    aligned_poses = apply_alignment(pred_poses, rotation, translation)

    # Compute position errors
    pred_positions = aligned_poses[:, :3, 3]
    gt_positions = gt_poses[:, :3, 3]

    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)

    return float(np.mean(errors))


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    """
    Compute inverse of a pose matrix.

    Args:
        pose: Pose matrix (4, 4)

    Returns:
        Inverse pose (4, 4)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    pose_inv = np.eye(4)
    pose_inv[:3, :3] = R_inv
    pose_inv[:3, 3] = t_inv

    return pose_inv


def compose_poses(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compose two pose transformations.

    Args:
        pose1: First pose (4, 4)
        pose2: Second pose (4, 4)

    Returns:
        Composed pose (4, 4)
    """
    return pose1 @ pose2
