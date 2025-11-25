"""
Perceptual Baseline Calibration System

Main interface for Phase 2: Establishes baseline quality metrics for source
images and provides empirical foundation for measuring enhancement trajectories
beyond conventional photorealistic limitations.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
import json
import time

import numpy as np
from torch import Tensor

from .image_loader import ImageLoader, ImageMetadata, ImageType
from .analyzer import PerceptualAnalyzer, AnalysisResult
from .tracker import EnhancementTracker
from .metrics import MetricType

logger = logging.getLogger(__name__)


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@dataclass
class BaselineConfig:
    """Configuration for baseline calibration."""
    # Image loading
    target_size: Optional[tuple] = None  # (H, W) or None for original
    normalize: bool = True
    preserve_aspect: bool = True

    # Enhancement tracking
    target_quality_multiplier: float = 1.3  # Target 30% improvement

    # Metric weights for overall quality
    metric_weights: Optional[Dict[MetricType, float]] = None

    # Output
    save_visualizations: bool = True
    save_reports: bool = True
    output_dir: Path = Path("outputs/phase2_baseline")

    @classmethod
    def default(cls) -> "BaselineConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def high_quality(cls) -> "BaselineConfig":
        """Configuration for high-quality analysis."""
        return cls(
            target_size=(2048, 2048),
            target_quality_multiplier=1.5,  # Target 50% improvement
            save_visualizations=True,
            save_reports=True
        )


class PerceptualBaseline:
    """
    Perceptual Baseline Calibration System

    Phase 2 entry point for establishing baseline quality metrics and
    tracking enhancement trajectories.

    Usage:
        # Initialize with substrate from Phase 1
        baseline = PerceptualBaseline(substrate)

        # Calibrate with six source images
        results = baseline.calibrate([
            "pool.jpg",
            "bedrooms.jpg",
            "bathroom.jpg",
            "aerial.jpg",
            "kitchen.jpg",
            "great_room.jpg"
        ])

        # Get baseline metrics
        metrics = baseline.get_baseline_metrics()

        # Track enhancements
        enhanced_result = baseline.analyze_enhanced("pool_enhanced.jpg", step=1)

        # Generate reports
        report = baseline.generate_report()
    """

    def __init__(
        self,
        substrate,
        config: Optional[BaselineConfig] = None
    ):
        """
        Initialize perceptual baseline system.

        Args:
            substrate: Computational substrate from Phase 1
            config: Baseline configuration
        """
        self.substrate = substrate
        self.config = config or BaselineConfig.default()

        # Initialize components
        self.image_loader = ImageLoader(
            substrate,
            target_size=self.config.target_size,
            normalize=self.config.normalize,
            preserve_aspect=self.config.preserve_aspect
        )

        self.analyzer = PerceptualAnalyzer(
            substrate,
            metric_weights=self.config.metric_weights
        )

        self.tracker = EnhancementTracker(
            target_quality_multiplier=self.config.target_quality_multiplier
        )

        # Storage
        self.baseline_results: List[AnalysisResult] = []
        self.baseline_images: Dict[str, Tensor] = {}
        self.baseline_metadatas: Dict[str, ImageMetadata] = {}
        self.calibrated = False

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("Initialized Perceptual Baseline - Phase 2")
        logger.info("=" * 70)

    def calibrate(
        self,
        image_paths: List[Union[str, Path]],
        image_types: Optional[List[ImageType]] = None
    ) -> List[AnalysisResult]:
        """
        Calibrate baseline with source images.

        This establishes the empirical foundation for measuring enhancements.

        Args:
            image_paths: List of paths to source images
            image_types: Optional list of image types

        Returns:
            List of baseline analysis results
        """
        logger.info(f"Starting baseline calibration with {len(image_paths)} images")
        start_time = time.time()

        # Load images
        logger.info("Loading and preprocessing images...")
        tensors, metadatas = self.image_loader.load_batch(image_paths, image_types)

        # Store baseline images
        for tensor, metadata in zip(tensors, metadatas):
            name = metadata.path.stem
            self.baseline_images[name] = tensor
            self.baseline_metadatas[name] = metadata

        # Analyze images
        logger.info("Performing perceptual analysis...")
        results = self.analyzer.analyze_batch(tensors, metadatas)

        # Store results
        self.baseline_results = results

        # Establish baseline in tracker
        self.tracker.establish_baseline(results)

        self.calibrated = True
        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info(f"Baseline calibration complete in {elapsed:.2f}s")
        logger.info("=" * 70)

        # Log baseline summary
        self._log_baseline_summary()

        # Generate outputs if requested
        if self.config.save_reports:
            self._save_calibration_report()

        if self.config.save_visualizations:
            self._save_visualizations()

        return results

    def analyze_enhanced(
        self,
        image_path: Union[str, Path],
        step: int,
        description: str = "",
        image_type: Optional[ImageType] = None
    ) -> AnalysisResult:
        """
        Analyze enhanced version of an image.

        Args:
            image_path: Path to enhanced image
            step: Enhancement step number
            description: Description of enhancement
            image_type: Optional image type

        Returns:
            Analysis result
        """
        if not self.calibrated:
            raise RuntimeError("Baseline not calibrated. Call calibrate() first.")

        # Load enhanced image
        tensor, metadata = self.image_loader.load(image_path, image_type)

        # Analyze
        result = self.analyzer.analyze(tensor, metadata)

        # Track in enhancement tracker
        self.tracker.track_enhancement(result, step, description)

        logger.info(
            f"Analyzed enhanced image: {metadata.path.name}, "
            f"quality={result.overall_quality:.3f}"
        )

        return result

    def compare_to_baseline(
        self,
        enhanced_path: Union[str, Path],
        baseline_name: str
    ) -> Dict[str, Any]:
        """
        Compare enhanced image to its baseline.

        Args:
            enhanced_path: Path to enhanced image
            baseline_name: Name of baseline image

        Returns:
            Comparison results
        """
        if baseline_name not in self.baseline_images:
            raise ValueError(f"No baseline found for '{baseline_name}'")

        # Load enhanced image
        enhanced_tensor, enhanced_metadata = self.image_loader.load(enhanced_path)

        # Get baseline
        baseline_tensor = self.baseline_images[baseline_name]
        baseline_metadata = self.baseline_metadatas[baseline_name]

        # Compare
        comparison = self.analyzer.compare(
            baseline_tensor, enhanced_tensor,
            baseline_metadata, enhanced_metadata
        )

        return comparison

    def get_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get baseline metrics for all images.

        Returns:
            Dictionary mapping image names to their metric scores
        """
        if not self.calibrated:
            raise RuntimeError("Baseline not calibrated")

        metrics = {}
        for result in self.baseline_results:
            name = result.image_path.stem
            metrics[name] = {
                "overall_quality": result.overall_quality,
                "sharpness": result.sharpness,
                "contrast": result.contrast,
                "colorfulness": result.colorfulness,
                "naturalness": result.naturalness,
            }

            # Add individual metric scores
            for metric_type, score in result.quality_scores.items():
                metrics[name][metric_type.value] = score.normalized_score

        return metrics

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of enhancement trajectories."""
        return self.tracker.get_summary()

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive baseline report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        lines = [
            "=" * 80,
            "PHASE 2: PERCEPTUAL BASELINE CALIBRATION REPORT",
            "=" * 80,
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Calibrated: {self.calibrated}",
            f"Number of Baseline Images: {len(self.baseline_results)}",
            "",
        ]

        if not self.calibrated:
            lines.append("Baseline not yet calibrated.")
            return "\n".join(lines)

        # Configuration
        lines.extend([
            "CONFIGURATION",
            "-" * 80,
            f"Target Size: {self.config.target_size or 'Original'}",
            f"Target Quality Multiplier: {self.config.target_quality_multiplier}x",
            f"Normalization: {self.config.normalize}",
            f"Preserve Aspect: {self.config.preserve_aspect}",
            "",
        ])

        # Baseline analysis report
        lines.append("BASELINE ANALYSIS")
        lines.append("-" * 80)
        analysis_report = self.analyzer.generate_report(self.baseline_results)
        lines.append(analysis_report)
        lines.append("")

        # Enhancement trajectory report (if available)
        if self.tracker.trajectories:
            lines.append("ENHANCEMENT TRAJECTORIES")
            lines.append("-" * 80)
            trajectory_report = self.tracker.generate_report()
            lines.append(trajectory_report)

        lines.append("=" * 80)
        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def export_baseline_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Export baseline data to JSON.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if not self.calibrated:
            raise RuntimeError("Baseline not calibrated")

        if output_path is None:
            output_path = self.config.output_dir / "baseline_data.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {
            "calibration_time": time.time(),
            "configuration": {
                "target_size": self.config.target_size,
                "target_quality_multiplier": self.config.target_quality_multiplier,
            },
            "baseline_metrics": self.get_baseline_metrics(),
            "images": []
        }

        for result in self.baseline_results:
            image_data = {
                "name": result.image_path.stem,
                "path": str(result.image_path),
                "type": result.image_metadata.image_type.value if result.image_metadata.image_type else None,
                "dimensions": {
                    "width": result.image_metadata.width,
                    "height": result.image_metadata.height
                },
                "statistics": {
                    "mean_intensity": result.image_metadata.mean_intensity,
                    "std_intensity": result.image_metadata.std_intensity,
                    "dynamic_range": result.image_metadata.dynamic_range
                },
                "quality": result.get_summary()
            }
            data["images"].append(image_data)

        # Convert numpy types to JSON-serializable types
        data = _convert_to_json_serializable(data)

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Baseline data exported to {output_path}")
        return output_path

    def plot_baseline_distribution(self, output_path: Optional[Path] = None):
        """
        Plot distribution of baseline quality metrics.

        Args:
            output_path: Optional path to save plot
        """
        if not self.calibrated:
            raise RuntimeError("Baseline not calibrated")

        try:
            from matplotlib import pyplot as plt

            metrics = self.get_baseline_metrics()
            image_names = list(metrics.keys())

            # Extract metric values
            qualities = [m["overall_quality"] for m in metrics.values()]
            sharpness = [m["sharpness"] for m in metrics.values()]
            contrast = [m["contrast"] for m in metrics.values()]
            colorfulness = [m["colorfulness"] for m in metrics.values()]

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Overall Quality
            axes[0, 0].bar(range(len(image_names)), qualities, color='steelblue')
            axes[0, 0].set_xticks(range(len(image_names)))
            axes[0, 0].set_xticklabels(image_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Overall Quality')
            axes[0, 0].set_title('Baseline Overall Quality')
            axes[0, 0].grid(axis='y', alpha=0.3)

            # Plot 2: Sharpness
            axes[0, 1].bar(range(len(image_names)), sharpness, color='green')
            axes[0, 1].set_xticks(range(len(image_names)))
            axes[0, 1].set_xticklabels(image_names, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Sharpness')
            axes[0, 1].set_title('Baseline Sharpness')
            axes[0, 1].grid(axis='y', alpha=0.3)

            # Plot 3: Contrast
            axes[1, 0].bar(range(len(image_names)), contrast, color='orange')
            axes[1, 0].set_xticks(range(len(image_names)))
            axes[1, 0].set_xticklabels(image_names, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Contrast')
            axes[1, 0].set_title('Baseline Contrast')
            axes[1, 0].grid(axis='y', alpha=0.3)

            # Plot 4: Colorfulness
            axes[1, 1].bar(range(len(image_names)), colorfulness, color='purple')
            axes[1, 1].set_xticks(range(len(image_names)))
            axes[1, 1].set_xticklabels(image_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Colorfulness')
            axes[1, 1].set_title('Baseline Colorfulness')
            axes[1, 1].grid(axis='y', alpha=0.3)

            plt.suptitle('Baseline Quality Metric Distribution')
            plt.tight_layout()

            if output_path:
                output_path = Path(output_path)
            else:
                output_path = self.config.output_dir / "baseline_distribution.png"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Baseline distribution plot saved to {output_path}")

        except ImportError:
            logger.warning("Matplotlib not available, skipping visualization")

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _log_baseline_summary(self):
        """Log baseline summary to console."""
        metrics = self.get_baseline_metrics()

        logger.info("\nBASELINE SUMMARY:")
        for name, metric_values in metrics.items():
            logger.info(
                f"  {name}: quality={metric_values['overall_quality']:.3f}, "
                f"sharpness={metric_values['sharpness']:.3f}, "
                f"contrast={metric_values['contrast']:.3f}"
            )

    def _save_calibration_report(self):
        """Save calibration report to file."""
        report_path = self.config.output_dir / "baseline_calibration_report.txt"
        self.generate_report(report_path)

    def _save_visualizations(self):
        """Save visualization outputs."""
        # Distribution plot
        self.plot_baseline_distribution()

        # Export baseline data
        self.export_baseline_data()

    def __repr__(self) -> str:
        return (
            f"PerceptualBaseline(calibrated={self.calibrated}, "
            f"images={len(self.baseline_results)}, "
            f"target_multiplier={self.config.target_quality_multiplier}x)"
        )
