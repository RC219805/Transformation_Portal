"""
Pose estimation metrics for DA3 benchmark.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_pose_error(
    pred_rotation: np.ndarray,
    gt_rotation: np.ndarray,
    pred_translation: np.ndarray,
    gt_translation: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute rotation and translation error between predicted and GT poses.

    Args:
        pred_rotation: Predicted rotation matrix (3, 3)
        gt_rotation: Ground truth rotation matrix (3, 3)
        pred_translation: Predicted translation vector (3,)
        gt_translation: Ground truth translation vector (3,)

    Returns:
        rotation_error: Angular error in degrees
        translation_error: Translation error in meters
    """
    # Compute relative rotation
    R_rel = pred_rotation.T @ gt_rotation

    # Compute rotation angle (geodesic distance on SO(3))
    trace = np.trace(R_rel)
    # Clamp to avoid numerical errors in arccos
    trace = np.clip(trace, -1.0, 3.0)
    rotation_error = np.rad2deg(np.arccos((trace - 1) / 2))

    # Compute translation error (Euclidean distance)
    translation_error = np.linalg.norm(pred_translation - gt_translation)

    return rotation_error, translation_error


def compute_auc(errors: np.ndarray, thresholds: List[float], max_threshold: float) -> float:
    """
    Compute Area Under Curve metric for pose estimation.

    The AUC metric measures the percentage of poses with error below
    varying thresholds, integrated over the threshold range.

    Args:
        errors: Array of angular errors in degrees (N,)
        thresholds: Threshold values for sampling the curve
        max_threshold: Maximum threshold (e.g., 3° or 30°)

    Returns:
        AUC score in range [0, 1], where 1 is perfect
    """
    if len(errors) == 0:
        return 0.0

    # Sort errors
    errors_sorted = np.sort(errors)

    # Compute recall at each threshold
    recalls = []
    for threshold in thresholds:
        recall = np.mean(errors_sorted <= threshold)
        recalls.append(recall)

    # Integrate using trapezoidal rule
    recalls = np.array(recalls)
    thresholds = np.array(thresholds)

    # Normalize thresholds to [0, 1]
    thresholds_norm = thresholds / max_threshold

    # Compute AUC
    auc = np.trapz(recalls, thresholds_norm)

    return float(auc)


def evaluate_pose_estimation(
    pred_poses: List[np.ndarray],
    gt_poses: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate pose estimation accuracy.

    Computes AUC@3° and AUC@30° metrics following the DA3 benchmark protocol.

    Args:
        pred_poses: List of predicted poses (4x4 matrices)
        gt_poses: List of ground truth poses (4x4 matrices)

    Returns:
        Dictionary with metrics:
        {
            "auc3": AUC@3° metric,
            "auc30": AUC@30° metric,
            "median_rotation_error": Median rotation error in degrees,
            "median_translation_error": Median translation error in meters,
            "mean_rotation_error": Mean rotation error in degrees,
            "mean_translation_error": Mean translation error in meters
        }
    """
    if len(pred_poses) != len(gt_poses):
        raise ValueError(f"Number of predicted poses ({len(pred_poses)}) doesn't match GT poses ({len(gt_poses)})")

    rotation_errors = []
    translation_errors = []

    for pred_pose, gt_pose in zip(pred_poses, gt_poses):
        # Extract rotation and translation
        pred_R = pred_pose[:3, :3]
        pred_t = pred_pose[:3, 3]
        gt_R = gt_pose[:3, :3]
        gt_t = gt_pose[:3, 3]

        # Compute errors
        rot_err, trans_err = compute_pose_error(pred_R, gt_R, pred_t, gt_t)
        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)

    # Define thresholds for AUC computation
    thresholds_3 = np.linspace(0, 3, 100)  # 0° to 3°
    thresholds_30 = np.linspace(0, 30, 100)  # 0° to 30°

    # Compute AUC metrics
    auc3 = compute_auc(rotation_errors, thresholds_3, max_threshold=3.0)
    auc30 = compute_auc(rotation_errors, thresholds_30, max_threshold=30.0)

    return {
        "auc3": auc3,
        "auc30": auc30,
        "median_rotation_error": float(np.median(rotation_errors)),
        "median_translation_error": float(np.median(translation_errors)),
        "mean_rotation_error": float(np.mean(rotation_errors)),
        "mean_translation_error": float(np.mean(translation_errors)),
        "num_poses": len(pred_poses),
    }


def compute_pose_metrics_batch(
    pred_poses: np.ndarray, gt_poses: np.ndarray, scene_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute pose metrics for multiple scenes.

    Args:
        pred_poses: Predicted poses (N, 4, 4)
        gt_poses: Ground truth poses (N, 4, 4)
        scene_names: Optional list of scene names for grouping

    Returns:
        Dictionary of metrics per scene (or overall if no scene names)
    """
    if scene_names is None:
        # Compute overall metrics
        return {"overall": evaluate_pose_estimation(list(pred_poses), list(gt_poses))}

    # Group by scene and compute per-scene metrics
    results = {}
    unique_scenes = set(scene_names)

    for scene in unique_scenes:
        # Get indices for this scene
        indices = [i for i, s in enumerate(scene_names) if s == scene]

        # Extract poses for this scene
        scene_pred = [pred_poses[i] for i in indices]
        scene_gt = [gt_poses[i] for i in indices]

        # Compute metrics
        results[scene] = evaluate_pose_estimation(scene_pred, scene_gt)

    # Compute overall metrics
    results["overall"] = evaluate_pose_estimation(list(pred_poses), list(gt_poses))

    return results
