"""
Benchmark configuration and constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class EvaluationMode(Enum):
    """Evaluation modes for DA3 benchmark."""

    POSE = "pose"  # Pose estimation only
    RECON_UNPOSED = "recon_unposed"  # Reconstruction with predicted poses
    RECON_POSED = "recon_posed"  # Reconstruction with GT poses


@dataclass
class BenchmarkConfig:
    """Configuration for DA3 benchmark evaluation."""

    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: ["eth3d", "7scenes", "scannetpp", "hiroom", "dtu", "dtu64"])

    # Evaluation modes
    modes: List[EvaluationMode] = field(
        default_factory=lambda: [EvaluationMode.POSE, EvaluationMode.RECON_UNPOSED, EvaluationMode.RECON_POSED]
    )

    # Frame limit per scene (-1 for unlimited)
    max_frames: int = 100

    # Specific scenes to evaluate (None for all)
    scenes: Optional[List[str]] = None

    # Inference settings
    num_fusion_workers: int = 4
    debug: bool = False

    # Paths
    data_root: Path = Path("workspace/benchmark_dataset")
    work_dir: Path = Path("workspace/evaluation")

    # TSDF fusion parameters
    voxel_length: float = 0.01  # 1cm voxels
    sdf_trunc: float = 0.04  # 4cm truncation

    # RANSAC parameters
    ransac_iterations: int = 1000
    ransac_inlier_threshold: float = 0.1

    # Reconstruction thresholds
    fscore_threshold: float = 0.01  # 1cm for most datasets
    dtu_fscore_threshold: float = 0.01  # 10mm for DTU

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_root = Path(self.data_root)
        self.work_dir = Path(self.work_dir)


# Dataset constants
HUGGINGFACE_REPO = "depth-anything/DA3-BENCH"

# Expected benchmark results for DA3-GIANT (for validation)
EXPECTED_RESULTS = {
    "eth3d": {
        "pose": {"auc3": 0.85, "auc30": 0.95},
        "recon_unposed": {"fscore": 0.78, "overall": 0.025},
        "recon_posed": {"fscore": 0.82, "overall": 0.020},
    },
    "7scenes": {
        "pose": {"auc3": 0.75, "auc30": 0.92},
        "recon_unposed": {"fscore": 0.72, "overall": 0.030},
        "recon_posed": {"fscore": 0.80, "overall": 0.022},
    },
    "scannetpp": {
        "pose": {"auc3": 0.80, "auc30": 0.93},
        "recon_unposed": {"fscore": 0.74, "overall": 0.028},
        "recon_posed": {"fscore": 0.81, "overall": 0.021},
    },
    "hiroom": {
        "pose": {"auc3": 0.82, "auc30": 0.94},
        "recon_unposed": {"fscore": 0.76, "overall": 0.026},
        "recon_posed": {"fscore": 0.83, "overall": 0.019},
    },
    "dtu": {
        "recon_posed": {"fscore": 0.88, "overall": 0.015},
    },
    "dtu64": {
        "pose": {"auc3": 0.78, "auc30": 0.91},
    },
}
