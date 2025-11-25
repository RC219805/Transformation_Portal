"""
Enhancement Tracker for Quality Trajectory Monitoring

Tracks enhancement trajectories over time to measure improvements
beyond photorealistic baselines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import json

import numpy as np

from .analyzer import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in enhancement trajectory."""
    step: int
    timestamp: float
    overall_quality: float
    metric_scores: Dict[str, float]
    description: str = ""


@dataclass
class EnhancementTrajectory:
    """Complete enhancement trajectory for an image."""
    image_name: str
    baseline_quality: float
    points: List[TrajectoryPoint] = field(default_factory=list)
    target_quality: Optional[float] = None

    def add_point(self, point: TrajectoryPoint):
        """Add trajectory point."""
        self.points.append(point)

    def get_improvement(self) -> float:
        """Get total improvement from baseline."""
        if not self.points:
            return 0.0
        return self.points[-1].overall_quality - self.baseline_quality

    def get_progress(self) -> float:
        """Get progress toward target (0-1)."""
        if self.target_quality is None:
            return 0.0

        improvement = self.get_improvement()
        target_improvement = self.target_quality - self.baseline_quality

        if target_improvement <= 0:
            return 1.0

        return min(improvement / target_improvement, 1.0)

    def is_improving(self) -> bool:
        """Check if trajectory is improving."""
        if len(self.points) < 2:
            return False

        recent_points = self.points[-5:]  # Look at last 5 points
        return all(
            recent_points[i].overall_quality <= recent_points[i+1].overall_quality
            for i in range(len(recent_points) - 1)
        )


class EnhancementTracker:
    """
    Tracks enhancement trajectories for measuring improvements
    beyond conventional photorealistic limitations.
    """

    def __init__(self, target_quality_multiplier: float = 1.2):
        """
        Initialize enhancement tracker.

        Args:
            target_quality_multiplier: Multiplier for baseline to set target
                                      (e.g., 1.2 = 20% improvement)
        """
        self.target_quality_multiplier = target_quality_multiplier
        self.trajectories: Dict[str, EnhancementTrajectory] = {}
        self.baseline_established = False

        logger.info(f"Initialized EnhancementTracker (target: {target_quality_multiplier}x baseline)")

    def establish_baseline(self, results: List[AnalysisResult]):
        """
        Establish baseline from initial analysis results.

        Args:
            results: List of baseline analysis results
        """
        logger.info(f"Establishing baseline from {len(results)} images")

        for result in results:
            image_name = result.image_path.stem
            baseline_quality = result.overall_quality
            target_quality = baseline_quality * self.target_quality_multiplier

            trajectory = EnhancementTrajectory(
                image_name=image_name,
                baseline_quality=baseline_quality,
                target_quality=target_quality
            )

            # Add baseline as first point
            baseline_point = TrajectoryPoint(
                step=0,
                timestamp=result.timestamp,
                overall_quality=baseline_quality,
                metric_scores={
                    metric.value: score.normalized_score
                    for metric, score in result.quality_scores.items()
                },
                description="Baseline (original image)"
            )
            trajectory.add_point(baseline_point)

            self.trajectories[image_name] = trajectory

        self.baseline_established = True
        logger.info(f"Baseline established for {len(self.trajectories)} images")

    def track_enhancement(
        self,
        result: AnalysisResult,
        step: int,
        description: str = ""
    ):
        """
        Track enhancement step.

        Args:
            result: Analysis result after enhancement
            step: Enhancement step number
            description: Description of enhancement applied
        """
        if not self.baseline_established:
            raise RuntimeError("Baseline not established. Call establish_baseline() first.")

        image_name = result.image_path.stem

        if image_name not in self.trajectories:
            logger.warning(f"No baseline for {image_name}, creating new trajectory")
            self.trajectories[image_name] = EnhancementTrajectory(
                image_name=image_name,
                baseline_quality=result.overall_quality,
                target_quality=result.overall_quality * self.target_quality_multiplier
            )

        # Create trajectory point
        point = TrajectoryPoint(
            step=step,
            timestamp=result.timestamp,
            overall_quality=result.overall_quality,
            metric_scores={
                metric.value: score.normalized_score
                for metric, score in result.quality_scores.items()
            },
            description=description
        )

        self.trajectories[image_name].add_point(point)

        improvement = self.trajectories[image_name].get_improvement()
        progress = self.trajectories[image_name].get_progress()

        logger.info(
            f"Tracked {image_name} step {step}: "
            f"quality={result.overall_quality:.3f}, "
            f"improvement={improvement:+.3f}, "
            f"progress={progress:.1%}"
        )

    def get_trajectory(self, image_name: str) -> Optional[EnhancementTrajectory]:
        """Get trajectory for specific image."""
        return self.trajectories.get(image_name)

    def get_all_trajectories(self) -> Dict[str, EnhancementTrajectory]:
        """Get all trajectories."""
        return self.trajectories

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all trajectories."""
        if not self.trajectories:
            return {"status": "no_trajectories"}

        improvements = [t.get_improvement() for t in self.trajectories.values()]
        progresses = [t.get_progress() for t in self.trajectories.values()]

        return {
            "num_images": len(self.trajectories),
            "avg_improvement": np.mean(improvements),
            "min_improvement": np.min(improvements),
            "max_improvement": np.max(improvements),
            "avg_progress": np.mean(progresses),
            "images_improving": sum(1 for t in self.trajectories.values() if t.is_improving()),
            "images_at_target": sum(1 for p in progresses if p >= 1.0),
        }

    def plot_trajectories(
        self,
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """
        Plot enhancement trajectories.

        Args:
            output_path: Path to save plot
            show: Whether to display plot
        """
        if not self.trajectories:
            logger.warning("No trajectories to plot")
            return

        try:
            from matplotlib import pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib not available. "
                "Install with: pip install matplotlib or pip install -e '.[ml]'"
            ) from e

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Quality over steps
        for name, trajectory in self.trajectories.items():
            steps = [p.step for p in trajectory.points]
            qualities = [p.overall_quality for p in trajectory.points]

            ax1.plot(steps, qualities, marker='o', label=name, linewidth=2)

            # Add target line
            if trajectory.target_quality:
                ax1.axhline(
                    y=trajectory.target_quality,
                    linestyle='--',
                    alpha=0.3,
                    color='gray'
                )

        ax1.set_xlabel('Enhancement Step')
        ax1.set_ylabel('Overall Quality')
        ax1.set_title('Enhancement Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Improvement from baseline
        for name, trajectory in self.trajectories.items():
            steps = [p.step for p in trajectory.points]
            improvements = [
                p.overall_quality - trajectory.baseline_quality
                for p in trajectory.points
            ]

            ax2.plot(steps, improvements, marker='o', label=name, linewidth=2)

        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel('Enhancement Step')
        ax2.set_ylabel('Quality Improvement')
        ax2.set_title('Improvement from Baseline')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_metric_breakdown(
        self,
        image_name: str,
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """
        Plot metric breakdown for specific image.

        Args:
            image_name: Name of image
            output_path: Path to save plot
            show: Whether to display plot
        """
        trajectory = self.get_trajectory(image_name)
        if not trajectory:
            logger.warning(f"No trajectory found for {image_name}")
            return

        try:
            from matplotlib import pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib not available. "
                "Install with: pip install matplotlib or pip install -e '.[ml]'"
            ) from e

        # Get all metric types
        metric_names = list(trajectory.points[0].metric_scores.keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric_name in enumerate(metric_names[:6]):  # Plot up to 6 metrics
            steps = [p.step for p in trajectory.points]
            scores = [p.metric_scores[metric_name] for p in trajectory.points]

            axes[idx].plot(steps, scores, marker='o', linewidth=2, color='blue')
            axes[idx].set_title(f'{metric_name.upper()}')
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel('Normalized Score')
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f'Metric Breakdown: {image_name}')
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric breakdown plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def export_trajectories(self, output_path: Path):
        """
        Export trajectories to JSON.

        Args:
            output_path: Path to JSON file
        """
        data = {
            "target_quality_multiplier": self.target_quality_multiplier,
            "baseline_established": self.baseline_established,
            "trajectories": {}
        }

        for name, trajectory in self.trajectories.items():
            data["trajectories"][name] = {
                "baseline_quality": trajectory.baseline_quality,
                "target_quality": trajectory.target_quality,
                "current_quality": trajectory.points[-1].overall_quality if trajectory.points else 0,
                "improvement": trajectory.get_improvement(),
                "progress": trajectory.get_progress(),
                "points": [
                    {
                        "step": p.step,
                        "timestamp": p.timestamp,
                        "overall_quality": p.overall_quality,
                        "metric_scores": p.metric_scores,
                        "description": p.description
                    }
                    for p in trajectory.points
                ]
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Trajectories exported to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable trajectory report."""
        lines = [
            "=" * 80,
            "ENHANCEMENT TRAJECTORY REPORT",
            "=" * 80,
            f"Target Quality Multiplier: {self.target_quality_multiplier}x",
            f"Baseline Established: {self.baseline_established}",
            "",
        ]

        summary = self.get_summary()
        if "status" in summary:
            lines.append("No trajectories available.")
            return "\n".join(lines)

        lines.extend([
            "OVERALL SUMMARY",
            "-" * 80,
            f"Number of Images: {summary['num_images']}",
            f"Average Improvement: {summary['avg_improvement']:+.3f}",
            f"Min Improvement: {summary['min_improvement']:+.3f}",
            f"Max Improvement: {summary['max_improvement']:+.3f}",
            f"Average Progress: {summary['avg_progress']:.1%}",
            f"Images Improving: {summary['images_improving']}/{summary['num_images']}",
            f"Images at Target: {summary['images_at_target']}/{summary['num_images']}",
            "",
        ])

        lines.append("INDIVIDUAL TRAJECTORIES")
        lines.append("-" * 80)

        for name, trajectory in sorted(self.trajectories.items()):
            improvement = trajectory.get_improvement()
            progress = trajectory.get_progress()
            status = "✓ At target" if progress >= 1.0 else "↑ Improving" if trajectory.is_improving() else "→ Stable"

            lines.extend([
                f"\n{name}:",
                f"  Baseline: {trajectory.baseline_quality:.3f}",
                f"  Current: {trajectory.points[-1].overall_quality:.3f}",
                f"  Target: {trajectory.target_quality:.3f}",
                f"  Improvement: {improvement:+.3f}",
                f"  Progress: {progress:.1%}",
                f"  Status: {status}",
                f"  Steps: {len(trajectory.points) - 1}",
            ])

        lines.append("=" * 80)
        return "\n".join(lines)
