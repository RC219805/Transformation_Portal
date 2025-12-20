"""
DA3 Benchmark Integration Module

Provides comprehensive validation capabilities using official Depth Anything 3
Visual Geometry Benchmark:
- Pose Estimation (AUC@3°, AUC@30°)
- 3D Reconstruction (F-score, Chamfer distance)
- Multi-Dataset Support (ETH3D, 7Scenes, ScanNet++, HiRoom, DTU)
"""

from .evaluator import DA3BenchmarkEvaluator, BenchmarkConfig, EvaluationMode
from .dataset_loader import (
    DA3BenchmarkDataset,
    ETH3DDataset,
    SevenScenesDataset,
    ScanNetPPDataset,
    HiRoomDataset,
    DTUDataset,
    download_datasets,
)
from .pose_metrics import (
    compute_pose_error,
    compute_auc,
    evaluate_pose_estimation,
)
from .recon_metrics import (
    compute_fscore,
    compute_chamfer_distance,
    evaluate_reconstruction,
)
from .fusion import TSDFFusion, fuse_depth_maps
from .alignment import (
    align_poses_ransac,
    apply_alignment,
    align_scale,
    compute_rigid_transform,
)

__all__ = [
    # Evaluator
    "DA3BenchmarkEvaluator",
    "BenchmarkConfig",
    "EvaluationMode",
    # Datasets
    "DA3BenchmarkDataset",
    "ETH3DDataset",
    "SevenScenesDataset",
    "ScanNetPPDataset",
    "HiRoomDataset",
    "DTUDataset",
    "download_datasets",
    # Pose metrics
    "compute_pose_error",
    "compute_auc",
    "evaluate_pose_estimation",
    # Reconstruction metrics
    "compute_fscore",
    "compute_chamfer_distance",
    "evaluate_reconstruction",
    # Fusion
    "TSDFFusion",
    "fuse_depth_maps",
    # Alignment
    "align_poses_ransac",
    "apply_alignment",
    "align_scale",
    "compute_rigid_transform",
]
