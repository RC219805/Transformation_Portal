"""
Dataset loaders for DA3 benchmark.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DA3BenchmarkDataset(ABC):
    """Base class for DA3 benchmark datasets."""

    HUGGINGFACE_REPO = "depth-anything/DA3-BENCH"

    def __init__(self, root_dir: Path):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing dataset
        """
        self.root_dir = Path(root_dir)
        self.scenes = self._get_scenes()

    @abstractmethod
    def _get_scenes(self) -> List[str]:
        """Get list of scene names."""
        pass

    @abstractmethod
    def load_scene(self, scene_name: str) -> Dict:
        """
        Load scene data.

        Returns:
            {
                "images": List of image paths,
                "depth_gt": Optional list of ground truth depth paths,
                "poses_gt": Optional ground truth poses (N, 4, 4),
                "intrinsics": Camera intrinsics (3, 3) or list of (3, 3),
                "metadata": Additional scene metadata
            }
        """
        pass

    def get_ground_truth(self, scene_name: str) -> Dict:
        """
        Load ground truth data for evaluation.

        Returns:
            {
                "depth": Optional depth maps,
                "poses": Optional camera poses,
                "point_cloud": Optional ground truth point cloud,
                "mesh": Optional ground truth mesh path
            }
        """
        scene_data = self.load_scene(scene_name)
        gt_data = {}

        if "depth_gt" in scene_data and scene_data["depth_gt"]:
            gt_data["depth"] = scene_data["depth_gt"]

        if "poses_gt" in scene_data and scene_data["poses_gt"] is not None:
            gt_data["poses"] = scene_data["poses_gt"]

        # Try to load ground truth mesh/point cloud
        mesh_path = self.root_dir / scene_name / "mesh_gt.ply"
        if mesh_path.exists():
            gt_data["mesh"] = mesh_path

        pc_path = self.root_dir / scene_name / "pointcloud_gt.ply"
        if pc_path.exists():
            gt_data["point_cloud"] = pc_path

        return gt_data


class ETH3DDataset(DA3BenchmarkDataset):
    """ETH3D benchmark dataset."""

    SCENES = [
        "courtyard",
        "electro",
        "kicker",
        "pipes",
        "relief",
        "delivery_area",
        "facade",
        "office",
        "playground",
        "relief_2",
        "terrains",
    ]

    # Images to filter due to unusual rotations
    FILTER_KEYS = {
        "courtyard": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "delivery_area": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "electro": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "facade": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "kicker": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "office": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "pipes": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "playground": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "relief": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "relief_2": ["DSC_0189", "DSC_0190", "DSC_0191"],
        "terrains": ["DSC_0189", "DSC_0190", "DSC_0191"],
    }

    def _get_scenes(self) -> List[str]:
        return self.SCENES

    def load_scene(self, scene_name: str) -> Dict:
        """Load ETH3D scene."""
        scene_dir = self.root_dir / scene_name

        # Load images
        images_dir = scene_dir / "images"
        image_paths = sorted(images_dir.glob("*.jpg"))

        # Filter images
        filter_keys = self.FILTER_KEYS.get(scene_name, [])
        image_paths = [p for p in image_paths if not any(key in p.stem for key in filter_keys)]

        # Load camera poses
        poses_file = scene_dir / "cameras.txt"
        poses = self._load_colmap_cameras(poses_file) if poses_file.exists() else None

        # Load intrinsics
        intrinsics_file = scene_dir / "intrinsics.txt"
        intrinsics = self._load_intrinsics(intrinsics_file) if intrinsics_file.exists() else None

        return {
            "images": image_paths,
            "depth_gt": None,  # ETH3D doesn't provide dense depth GT
            "poses_gt": poses,
            "intrinsics": intrinsics,
            "metadata": {"scene_name": scene_name, "dataset": "eth3d"},
        }

    def _load_colmap_cameras(self, path: Path) -> np.ndarray:
        """Load camera poses from COLMAP format."""
        poses = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    # COLMAP format: image_id qw qx qy qz tx ty tz
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])

                    # Convert quaternion to rotation matrix
                    R = self._quat_to_rotation(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz])

                    # Build 4x4 pose matrix
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    poses.append(pose)

        return np.array(poses) if poses else None

    def _quat_to_rotation(self, qw, qx, qy, qz) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        R = np.array(
            [
                [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
            ]
        )
        return R

    def _load_intrinsics(self, path: Path) -> np.ndarray:
        """Load camera intrinsics."""
        with open(path, "r") as f:
            lines = f.readlines()
            fx, fy, cx, cy = map(float, lines[0].strip().split())

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K


class SevenScenesDataset(DA3BenchmarkDataset):
    """7Scenes benchmark dataset."""

    SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

    def _get_scenes(self) -> List[str]:
        return self.SCENES

    def load_scene(self, scene_name: str) -> Dict:
        """Load 7Scenes scene."""
        scene_dir = self.root_dir / scene_name

        # Load images
        image_paths = sorted(scene_dir.glob("*.color.png"))

        # Load poses
        pose_paths = sorted(scene_dir.glob("*.pose.txt"))
        poses = [self._load_7scenes_pose(p) for p in pose_paths]
        poses = np.array(poses) if poses else None

        # Load intrinsics (typically fixed for 7Scenes)
        K = np.array([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]])

        return {
            "images": image_paths,
            "depth_gt": sorted(scene_dir.glob("*.depth.png")),
            "poses_gt": poses,
            "intrinsics": K,
            "metadata": {"scene_name": scene_name, "dataset": "7scenes"},
        }

    def _load_7scenes_pose(self, path: Path) -> np.ndarray:
        """Load 7Scenes pose (4x4 matrix)."""
        pose = np.loadtxt(path)
        return pose


class ScanNetPPDataset(DA3BenchmarkDataset):
    """ScanNet++ benchmark dataset (re-calibrated)."""

    def _get_scenes(self) -> List[str]:
        # 20 validation scenes
        return [f"scene{i:04d}" for i in range(20)]

    def load_scene(self, scene_name: str) -> Dict:
        """Load ScanNet++ scene."""
        scene_dir = self.root_dir / scene_name

        # Load metadata
        metadata_file = scene_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load images
        image_paths = sorted((scene_dir / "images").glob("*.jpg"))

        # Load poses
        poses_file = scene_dir / "poses.npy"
        poses = np.load(poses_file) if poses_file.exists() else None

        # Load intrinsics
        intrinsics_file = scene_dir / "intrinsics.npy"
        intrinsics = np.load(intrinsics_file) if intrinsics_file.exists() else None

        return {"images": image_paths, "depth_gt": None, "poses_gt": poses, "intrinsics": intrinsics, "metadata": metadata}


class HiRoomDataset(DA3BenchmarkDataset):
    """HiRoom benchmark dataset."""

    def _get_scenes(self) -> List[str]:
        # 24 validation scenes
        return [f"room{i:03d}" for i in range(24)]

    def load_scene(self, scene_name: str) -> Dict:
        """Load HiRoom scene."""
        scene_dir = self.root_dir / scene_name

        # Load images
        image_paths = sorted((scene_dir / "images").glob("*.png"))

        # Load poses
        poses_file = scene_dir / "poses.txt"
        poses = np.loadtxt(poses_file).reshape(-1, 4, 4) if poses_file.exists() else None

        # Load intrinsics
        intrinsics_file = scene_dir / "intrinsics.txt"
        K = np.loadtxt(intrinsics_file).reshape(3, 3) if intrinsics_file.exists() else None

        return {
            "images": image_paths,
            "depth_gt": sorted((scene_dir / "depth").glob("*.png")),
            "poses_gt": poses,
            "intrinsics": K,
            "metadata": {"scene_name": scene_name, "dataset": "hiroom"},
        }


class DTUDataset(DA3BenchmarkDataset):
    """DTU benchmark dataset."""

    def __init__(self, root_dir: Path, variant: str = "dtu"):
        """
        Initialize DTU dataset.

        Args:
            root_dir: Root directory
            variant: "dtu" (49 scenes, reconstruction) or "dtu64" (64 scenes, pose)
        """
        self.variant = variant
        super().__init__(root_dir)

    def _get_scenes(self) -> List[str]:
        if self.variant == "dtu":
            # DTU-49 reconstruction scenes
            return [f"scan{i}" for i in range(1, 50)]
        else:
            # DTU-64 pose estimation scenes
            return [f"scan{i}" for i in range(1, 65)]

    def load_scene(self, scene_name: str) -> Dict:
        """Load DTU scene."""
        scene_dir = self.root_dir / scene_name

        # Load images
        image_paths = sorted((scene_dir / "images").glob("*.png"))

        # Load poses
        poses_file = scene_dir / "cameras.npz"
        poses = None
        intrinsics = None

        if poses_file.exists():
            cameras = np.load(poses_file)
            poses = cameras["world_mat"]  # World matrices
            intrinsics = cameras["intrinsics"] if "intrinsics" in cameras else None

        return {
            "images": image_paths,
            "depth_gt": None,
            "poses_gt": poses,
            "intrinsics": intrinsics,
            "metadata": {"scene_name": scene_name, "dataset": self.variant},
        }


def download_datasets(datasets: List[str], root_dir: Path) -> None:
    """
    Download DA3 benchmark datasets from HuggingFace.

    Args:
        datasets: List of dataset names ("all", "eth3d", "7scenes", etc.)
        root_dir: Download destination directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required for dataset downloads. Install with: pip install huggingface_hub")

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if "all" in datasets:
        datasets = ["eth3d", "7scenes", "scannetpp", "hiroom", "dtu", "dtu64"]

    for dataset_name in datasets:
        logger.info(f"Downloading {dataset_name} dataset...")

        dataset_dir = root_dir / dataset_name

        try:
            snapshot_download(
                repo_id=DA3BenchmarkDataset.HUGGINGFACE_REPO,
                repo_type="dataset",
                allow_patterns=f"{dataset_name}/*",
                local_dir=dataset_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"✅ Downloaded {dataset_name} to {dataset_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")


def get_dataset(dataset_name: str, root_dir: Path) -> DA3BenchmarkDataset:
    """
    Get dataset instance by name.

    Args:
        dataset_name: Dataset name
        root_dir: Root directory containing datasets

    Returns:
        Dataset instance
    """
    dataset_map = {
        "eth3d": ETH3DDataset,
        "7scenes": SevenScenesDataset,
        "scannetpp": ScanNetPPDataset,
        "hiroom": HiRoomDataset,
        "dtu": lambda root: DTUDataset(root, variant="dtu"),
        "dtu64": lambda root: DTUDataset(root, variant="dtu64"),
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_root = root_dir / dataset_name
    return dataset_map[dataset_name](dataset_root)
