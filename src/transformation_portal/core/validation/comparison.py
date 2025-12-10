"""
Baseline comparison for validation.

Compares processing results against baseline metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ComparisonStatus(Enum):
    """Status of baseline comparison."""
    NO_BASELINE = "no_baseline"
    REGRESSION = "regression"
    STABLE = "stable"
    IMPROVEMENT = "improvement"


@dataclass
class ComparisonResult:
    """Result of baseline comparison."""
    status: ComparisonStatus
    delta: Dict[str, float]
    baseline: Dict[str, float]
    current: Dict[str, float]
    threshold: float = 0.05

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "delta": self.delta,
            "baseline": self.baseline,
            "current": self.current,
            "threshold": self.threshold
        }

    def __str__(self) -> str:
        """Format comparison result as string."""
        lines = [f"Status: {self.status.value}"]

        if self.status == ComparisonStatus.NO_BASELINE:
            lines.append("No baseline available for comparison")
        else:
            lines.append(f"Threshold: ±{self.threshold*100:.1f}%")
            lines.append("\nMetric Changes:")

            for metric, delta_val in self.delta.items():
                baseline_val = self.baseline.get(metric, 0)
                current_val = self.current.get(metric, 0)

                if delta_val > self.threshold:
                    symbol = "↑"
                    status = "improved"
                elif delta_val < -self.threshold:
                    symbol = "↓"
                    status = "regressed"
                else:
                    symbol = "="
                    status = "stable"

                lines.append(f"  {metric:.<15} {symbol} {delta_val:+.4f} ({status})")
                lines.append(f"    Baseline: {baseline_val:.4f}, Current: {current_val:.4f}")

        return "\n".join(lines)


class BaselineComparator:
    """
    Compare processing results against validation baseline.

    Tracks baseline metrics per preset and detects regressions.
    """

    def __init__(self, baseline_dir: Path, threshold: float = 0.05):
        """
        Initialize baseline comparator.

        Args:
            baseline_dir: Directory containing baseline metrics
            threshold: Threshold for detecting changes (default 5%)
        """
        self.baseline_dir = Path(baseline_dir)
        self.threshold = threshold
        self.baseline_metrics = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Dict[str, float]]:
        """Load baseline metrics from file."""
        baseline_path = self.baseline_dir / "baseline_metrics.json"

        if not baseline_path.exists():
            logger.warning(f"No baseline file found at {baseline_path}")
            return {}

        try:
            with open(baseline_path) as f:
                data = json.load(f)

            logger.info(f"Loaded baseline metrics for {len(data)} presets")
            return data

        except Exception as e:
            logger.error(f"Failed to load baseline metrics: {e}")
            return {}

    def compare(
        self,
        preset: str,
        metrics: Dict[str, float]
    ) -> ComparisonResult:
        """
        Compare metrics against baseline.

        Args:
            preset: Preset name
            metrics: Current metrics

        Returns:
            ComparisonResult
        """
        if preset not in self.baseline_metrics:
            logger.info(f"No baseline for preset '{preset}'")
            return ComparisonResult(
                status=ComparisonStatus.NO_BASELINE,
                delta={},
                baseline={},
                current=metrics,
                threshold=self.threshold
            )

        baseline = self.baseline_metrics[preset]

        # Calculate deltas
        delta = {}
        for key in metrics:
            if key in baseline:
                delta[key] = metrics[key] - baseline[key]

        # Determine status
        status = self._determine_status(delta)

        return ComparisonResult(
            status=status,
            delta=delta,
            baseline=baseline,
            current=metrics,
            threshold=self.threshold
        )

    def _determine_status(self, delta: Dict[str, float]) -> ComparisonStatus:
        """
        Determine comparison status from deltas.

        Args:
            delta: Metric deltas

        Returns:
            ComparisonStatus
        """
        if not delta:
            return ComparisonStatus.NO_BASELINE

        # Check for significant changes
        has_regression = any(d < -self.threshold for d in delta.values())
        has_improvement = any(d > self.threshold for d in delta.values())

        if has_regression:
            return ComparisonStatus.REGRESSION
        elif has_improvement:
            return ComparisonStatus.IMPROVEMENT
        else:
            return ComparisonStatus.STABLE

    def update_baseline(
        self,
        preset: str,
        metrics: Dict[str, float]
    ):
        """
        Update baseline metrics for a preset.

        Args:
            preset: Preset name
            metrics: New baseline metrics
        """
        self.baseline_metrics[preset] = metrics
        self._save_baseline()

        logger.info(f"Updated baseline for preset '{preset}'")

    def _save_baseline(self):
        """Save baseline metrics to file."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = self.baseline_dir / "baseline_metrics.json"

        try:
            with open(baseline_path, "w") as f:
                json.dump(self.baseline_metrics, f, indent=2)

            logger.debug(f"Saved baseline metrics to {baseline_path}")

        except Exception as e:
            logger.error(f"Failed to save baseline metrics: {e}")

    def list_presets(self) -> list[str]:
        """Get list of presets with baselines."""
        return list(self.baseline_metrics.keys())

    def get_baseline(self, preset: str) -> Optional[Dict[str, float]]:
        """
        Get baseline metrics for a preset.

        Args:
            preset: Preset name

        Returns:
            Baseline metrics or None
        """
        return self.baseline_metrics.get(preset)

    def has_baseline(self, preset: str) -> bool:
        """Check if baseline exists for preset."""
        return preset in self.baseline_metrics
