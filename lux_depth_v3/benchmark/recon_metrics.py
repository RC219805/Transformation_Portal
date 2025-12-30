"""
3D reconstruction metrics for DA3 benchmark.
"""

import numpy as np
from typing import Dict, Union

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def compute_fscore(pred_points: np.ndarray, gt_points: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute F-score (harmonic mean of precision and recall).

    F-score measures geometric consistency between predicted and ground truth
    point clouds. Higher is better.

    Args:
        pred_points: Predicted point cloud (N, 3)
        gt_points: Ground truth point cloud (M, 3)
        threshold: Distance threshold in meters (default: 1cm)

    Returns:
        Dictionary with precision, recall, and F-score
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for F-score computation")

    # Convert to Open3D point clouds
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    # Compute precision: percentage of predicted points close to GT
    pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    precision = np.mean(pred_to_gt <= threshold)

    # Compute recall: percentage of GT points close to prediction
    gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    recall = np.mean(gt_to_pred <= threshold)

    # Compute F-score (harmonic mean)
    if precision + recall > 0:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
    }


def compute_chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> Dict[str, float]:
    """
    Compute Chamfer distance (accuracy and completeness).

    Chamfer distance measures the average distance between point clouds.
    Lower is better.

    Args:
        pred_points: Predicted point cloud (N, 3)
        gt_points: Ground truth point cloud (M, 3)

    Returns:
        Dictionary with accuracy, completeness, and overall distance
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for Chamfer distance computation")

    # Convert to Open3D point clouds
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    # Compute accuracy: average distance from predicted to GT
    pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    accuracy = np.mean(pred_to_gt)

    # Compute completeness: average distance from GT to predicted
    gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    completeness = np.mean(gt_to_pred)

    # Overall distance (average of accuracy and completeness)
    overall = (accuracy + completeness) / 2

    return {
        "accuracy": float(accuracy),
        "completeness": float(completeness),
        "overall": float(overall),
    }


def evaluate_reconstruction(
    pred_mesh_or_points: Union["o3d.geometry.TriangleMesh", np.ndarray],
    gt_mesh_or_points: Union["o3d.geometry.TriangleMesh", np.ndarray],
    dataset_name: str = "default",
) -> Dict[str, float]:
    """
    Evaluate 3D reconstruction quality.

    Computes both F-score and Chamfer distance metrics following the
    DA3 benchmark protocol.

    Args:
        pred_mesh_or_points: Predicted geometry (mesh or point cloud)
        gt_mesh_or_points: Ground truth geometry (mesh or point cloud)
        dataset_name: Dataset name for threshold adjustment

    Returns:
        Dictionary with all reconstruction metrics
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for reconstruction evaluation")

    # Convert meshes to point clouds if needed
    if isinstance(pred_mesh_or_points, o3d.geometry.TriangleMesh):
        # Sample points from mesh
        pred_pcd = pred_mesh_or_points.sample_points_uniformly(number_of_points=100000)
        pred_points = np.asarray(pred_pcd.points)
    else:
        pred_points = pred_mesh_or_points

    if isinstance(gt_mesh_or_points, o3d.geometry.TriangleMesh):
        gt_pcd = gt_mesh_or_points.sample_points_uniformly(number_of_points=100000)
        gt_points = np.asarray(gt_pcd.points)
    else:
        gt_points = gt_mesh_or_points

    # Determine F-score threshold based on dataset
    if "dtu" in dataset_name.lower():
        threshold = 0.01  # 10mm for DTU
    else:
        threshold = 0.01  # 1cm for other datasets

    # Compute F-score
    fscore_metrics = compute_fscore(pred_points, gt_points, threshold)

    # Compute Chamfer distance
    chamfer_metrics = compute_chamfer_distance(pred_points, gt_points)

    # Combine metrics
    return {
        **fscore_metrics,
        **chamfer_metrics,
        "threshold": threshold,
    }


def load_point_cloud(path: str) -> np.ndarray:
    """
    Load point cloud from file.

    Args:
        path: Path to point cloud file (.ply, .pcd, .xyz)

    Returns:
        Point cloud as numpy array (N, 3)
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for point cloud loading")

    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def load_mesh(path: str) -> "o3d.geometry.TriangleMesh":
    """
    Load triangle mesh from file.

    Args:
        path: Path to mesh file (.ply, .obj, .stl)

    Returns:
        Open3D triangle mesh
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for mesh loading")

    return o3d.io.read_triangle_mesh(path)


def save_point_cloud(points: np.ndarray, path: str) -> None:
    """
    Save point cloud to file.

    Args:
        points: Point cloud array (N, 3)
        path: Output path (.ply, .pcd, .xyz)
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for point cloud saving")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)


def save_mesh(mesh: "o3d.geometry.TriangleMesh", path: str) -> None:
    """
    Save triangle mesh to file.

    Args:
        mesh: Open3D triangle mesh
        path: Output path (.ply, .obj, .stl)
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for mesh saving")

    o3d.io.write_triangle_mesh(path, mesh)
