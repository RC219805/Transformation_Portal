"""
Main benchmark evaluator for DA3.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .config import BenchmarkConfig, EvaluationMode, EXPECTED_RESULTS
from .dataset_loader import get_dataset
from .pose_metrics import evaluate_pose_estimation
from .recon_metrics import evaluate_reconstruction, load_mesh, load_point_cloud
from .fusion import fuse_depth_maps, clean_mesh
from .alignment import align_poses_ransac, apply_alignment

try:
    from ..inference import DA3InferenceEngine
    from ..config import ModelVariant

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

logger = logging.getLogger(__name__)


class DA3BenchmarkEvaluator:
    """Main benchmark evaluator for DA3."""

    def __init__(
        self,
        model_variant: Optional["ModelVariant"] = None,
        config: Optional[BenchmarkConfig] = None,
        use_cli: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            model_variant: Model variant to use for inference
            config: Benchmark configuration
            use_cli: Use DA3 CLI for inference instead of Python API
        """
        self.config = config or BenchmarkConfig()
        self.use_cli = use_cli

        # Create work directory
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize inference engine
        if not use_cli:
            if not HAS_INFERENCE:
                raise ImportError("Inference engine not available. Use use_cli=True to use DA3 CLI instead.")

            if model_variant is None:
                raise ValueError("model_variant required when use_cli=False")

            from ..inference import DA3InferenceEngine

            self.engine = DA3InferenceEngine(
                model_variant=model_variant,
                batch_size=1,
                device="cuda" if self._has_cuda() else "cpu",
            )
        else:
            self.engine = None

        # Load datasets
        self.datasets = {}
        for dataset_name in self.config.datasets:
            try:
                self.datasets[dataset_name] = get_dataset(dataset_name, self.config.data_root)
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {e}")

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def run_inference(self, dataset_name: str, scene_name: str) -> Dict:
        """
        Run depth inference on a scene.

        Args:
            dataset_name: Dataset name
            scene_name: Scene name

        Returns:
            Dictionary with depth maps and metadata
        """
        dataset = self.datasets[dataset_name]
        scene_data = dataset.load_scene(scene_name)

        # Create output directory
        output_dir = self.config.work_dir / dataset_name / scene_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Limit frames if configured
        image_paths = scene_data["images"]
        if self.config.max_frames > 0:
            image_paths = image_paths[: self.config.max_frames]

        logger.info(f"Running inference on {dataset_name}/{scene_name} ({len(image_paths)} frames)")

        if self.use_cli:
            # Use CLI for inference
            depth_maps = self._run_inference_cli(image_paths, output_dir)
        else:
            # Use Python API for inference
            depth_maps = self._run_inference_api(image_paths, output_dir)

        # Save depth maps
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)

        for i, depth in enumerate(depth_maps):
            depth_path = depth_dir / f"depth_{i:06d}.npy"
            np.save(depth_path, depth)

        return {
            "depth_maps": depth_maps,
            "image_paths": image_paths,
            "output_dir": output_dir,
        }

    def _run_inference_cli(self, image_paths: List[Path], output_dir: Path) -> List[np.ndarray]:
        """Run inference using DA3 CLI."""
        import subprocess

        # Create temporary input list
        input_list = output_dir / "input_images.txt"
        with open(input_list, "w") as f:
            for path in image_paths:
                f.write(f"{path}\n")

        # Run CLI
        cmd = [
            "python",
            "-m",
            "lux_depth_v3.cli",
            "batch",
            "--input-list",
            str(input_list),
            "--output-dir",
            str(output_dir / "depth"),
            "--format",
            "npy",
        ]

        subprocess.run(cmd, check=True)

        # Load depth maps
        depth_maps = []
        for i in range(len(image_paths)):
            depth_path = output_dir / "depth" / f"depth_{i:06d}.npy"
            depth_maps.append(np.load(depth_path))

        return depth_maps

    def _run_inference_api(self, image_paths: List[Path], output_dir: Path) -> List[np.ndarray]:
        """Run inference using Python API."""
        from PIL import Image

        depth_maps = []

        for image_path in tqdm(image_paths, desc="Inference"):
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Run inference
            result = self.engine.predict(image)
            depth_maps.append(result["depth"])

        return depth_maps

    def evaluate_pose(self, dataset_name: str, scene_name: str) -> Dict:
        """
        Evaluate pose estimation for a scene.

        Args:
            dataset_name: Dataset name
            scene_name: Scene name

        Returns:
            Pose estimation metrics
        """
        logger.info(f"Evaluating pose: {dataset_name}/{scene_name}")

        # Load depth predictions
        output_dir = self.config.work_dir / dataset_name / scene_name
        depth_dir = output_dir / "depth"

        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth predictions not found: {depth_dir}. Run inference first.")

        # Load dataset
        dataset = self.datasets[dataset_name]
        scene_data = dataset.load_scene(scene_name)
        gt_poses = scene_data["poses_gt"]

        if gt_poses is None:
            raise ValueError(f"No ground truth poses for {dataset_name}/{scene_name}")

        # TODO: Implement pose estimation from depth
        # For now, this is a placeholder
        # Real implementation would use Structure-from-Motion (SfM)
        # or visual odometry with the depth maps

        logger.warning("Pose estimation not yet implemented - using placeholder")

        # Placeholder: return random poses for demonstration
        pred_poses = gt_poses + np.random.randn(*gt_poses.shape) * 0.1

        # Align poses using RANSAC
        R, t = align_poses_ransac(
            pred_poses,
            gt_poses,
            ransac_iterations=self.config.ransac_iterations,
            inlier_threshold=self.config.ransac_inlier_threshold,
        )

        # Apply alignment
        pred_poses_aligned = apply_alignment(pred_poses, R, t)

        # Compute metrics
        metrics = evaluate_pose_estimation(list(pred_poses_aligned), list(gt_poses))

        return metrics

    def evaluate_reconstruction(self, dataset_name: str, scene_name: str, use_gt_poses: bool = False) -> Dict:
        """
        Evaluate 3D reconstruction for a scene.

        Args:
            dataset_name: Dataset name
            scene_name: Scene name
            use_gt_poses: Use ground truth poses instead of estimated

        Returns:
            Reconstruction metrics
        """
        mode = "posed" if use_gt_poses else "unposed"
        logger.info(f"Evaluating reconstruction ({mode}): {dataset_name}/{scene_name}")

        # Load depth predictions
        output_dir = self.config.work_dir / dataset_name / scene_name
        depth_dir = output_dir / "depth"

        depth_files = sorted(depth_dir.glob("depth_*.npy"))
        if not depth_files:
            raise FileNotFoundError(f"No depth predictions found in {depth_dir}")

        depth_maps = [np.load(f) for f in depth_files]

        # Load dataset
        dataset = self.datasets[dataset_name]
        scene_data = dataset.load_scene(scene_name)

        # Load RGB images
        from PIL import Image

        image_paths = scene_data["images"][: len(depth_maps)]
        rgb_images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]

        # Get camera parameters
        if use_gt_poses:
            poses = scene_data["poses_gt"]
            if poses is None:
                raise ValueError(f"No GT poses for {dataset_name}/{scene_name}")
        else:
            # TODO: Use estimated poses from pose evaluation
            logger.warning("Using GT poses as placeholder for estimated poses")
            poses = scene_data["poses_gt"]

        intrinsics = scene_data["intrinsics"]
        if not isinstance(intrinsics, list):
            intrinsics = [intrinsics] * len(depth_maps)

        # Fuse depth maps
        mesh = fuse_depth_maps(
            depth_maps,
            rgb_images,
            intrinsics,
            poses,
            voxel_length=self.config.voxel_length,
            sdf_trunc=self.config.sdf_trunc,
        )

        # Clean mesh
        mesh = clean_mesh(mesh)

        # Save mesh
        mesh_path = output_dir / f"mesh_{mode}.ply"
        import open3d as o3d

        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        logger.info(f"Saved mesh to {mesh_path}")

        # Load ground truth
        gt_data = dataset.get_ground_truth(scene_name)

        if "mesh" in gt_data:
            gt_mesh = load_mesh(str(gt_data["mesh"]))
        elif "point_cloud" in gt_data:
            gt_points = load_point_cloud(str(gt_data["point_cloud"]))
            gt_mesh = gt_points
        else:
            raise ValueError(f"No GT geometry for {dataset_name}/{scene_name}")

        # Evaluate reconstruction
        metrics = evaluate_reconstruction(mesh, gt_mesh, dataset_name)

        return metrics

    def run_full_evaluation(self) -> Dict[str, Dict]:
        """
        Run full benchmark evaluation.

        Returns:
            Nested dictionary of results:
            {dataset_name: {mode: metrics}}
        """
        results = {}

        for dataset_name in self.config.datasets:
            if dataset_name not in self.datasets:
                logger.warning(f"Skipping {dataset_name} (not loaded)")
                continue

            dataset = self.datasets[dataset_name]
            dataset_results = {}

            # Get scenes to evaluate
            scenes = self.config.scenes or dataset.scenes

            for scene_name in scenes:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Evaluating {dataset_name}/{scene_name}")
                logger.info(f"{'=' * 60}\n")

                try:
                    # Run inference
                    self.run_inference(dataset_name, scene_name)

                    # Evaluate based on modes
                    scene_results = {}

                    for mode in self.config.modes:
                        if mode == EvaluationMode.POSE:
                            metrics = self.evaluate_pose(dataset_name, scene_name)
                            scene_results["pose"] = metrics

                        elif mode == EvaluationMode.RECON_UNPOSED:
                            metrics = self.evaluate_reconstruction(dataset_name, scene_name, use_gt_poses=False)
                            scene_results["recon_unposed"] = metrics

                        elif mode == EvaluationMode.RECON_POSED:
                            metrics = self.evaluate_reconstruction(dataset_name, scene_name, use_gt_poses=True)
                            scene_results["recon_posed"] = metrics

                    dataset_results[scene_name] = scene_results

                except Exception as e:
                    logger.error(f"Failed to evaluate {scene_name}: {e}")
                    if self.config.debug:
                        raise

            results[dataset_name] = dataset_results

        return results

    def print_results(self, results: Dict[str, Dict]) -> None:
        """Print formatted results table."""
        print("\n" + "=" * 80)
        print("DA3 BENCHMARK RESULTS")
        print("=" * 80 + "\n")

        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name.upper()}")
            print("-" * 60)

            # Aggregate scene results
            all_metrics = {}
            for scene_name, scene_results in dataset_results.items():
                for mode, metrics in scene_results.items():
                    if mode not in all_metrics:
                        all_metrics[mode] = []
                    all_metrics[mode].append(metrics)

            # Print aggregated metrics
            for mode, metrics_list in all_metrics.items():
                print(f"\n{mode}:")

                # Average metrics
                avg_metrics = {}
                for key in metrics_list[0].keys():
                    values = [m[key] for m in metrics_list if key in m]
                    if values and isinstance(values[0], (int, float)):
                        avg_metrics[key] = np.mean(values)

                for key, value in avg_metrics.items():
                    print(f"  {key}: {value:.4f}")

                # Compare with expected results
                if dataset_name in EXPECTED_RESULTS:
                    if mode in EXPECTED_RESULTS[dataset_name]:
                        expected = EXPECTED_RESULTS[dataset_name][mode]
                        print(f"\n  Expected:")
                        for key, exp_val in expected.items():
                            if key in avg_metrics:
                                act_val = avg_metrics[key]
                                diff = act_val - exp_val
                                status = "✅" if abs(diff) < 0.02 else "⚠️"
                                print(f"    {key}: {exp_val:.4f} (actual: {act_val:.4f}, diff: {diff:+.4f}) {status}")

        print("\n" + "=" * 80 + "\n")

    def save_results(self, results: Dict[str, Dict], output_path: Path = None) -> None:
        """Save results to JSON."""
        if output_path is None:
            output_path = self.config.work_dir / "benchmark_results.json"

        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    def load_results(self, results_path: Path = None) -> Dict[str, Dict]:
        """Load results from JSON."""
        if results_path is None:
            results_path = self.config.work_dir / "benchmark_results.json"

        results_path = Path(results_path)

        with open(results_path, "r") as f:
            results = json.load(f)

        return results
