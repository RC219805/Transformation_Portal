"""Cost-aware reward shaping for RL optimization.

This module provides reward functions that balance quality metrics
with resource costs (latency, VRAM, scaling events).

Enables learning policies that optimize for:
- Quality (APEX score, PSNR, etc.)
- Efficiency (latency, throughput)
- Resource usage (VRAM, GPU time)
- Cluster stability (avoid scaling thrash)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostWeights:
    """Weights for cost-aware reward computation.

    Higher weights mean stronger penalty/reward for that component.
    Negative quality weight would be unusual (typically positive).
    """

    quality: float = 1.0  # Weight for quality metrics
    latency: float = 0.05  # Penalty per second of latency
    vram: float = 0.1  # Penalty per GB of VRAM used
    scaling: float = 0.2  # Penalty per scaling event
    iteration: float = 0.01  # Small penalty per iteration (encourage speed)


@dataclass
class SystemStats:
    """System resource statistics for reward computation.

    Attributes:
        latency_sec: Execution latency in seconds
        vram_bytes: Peak VRAM usage in bytes
        vram_gb: Peak VRAM usage in gigabytes
        scale_events: Number of cluster scaling events
        gpu_util: GPU utilization (0-1)
        cpu_util: CPU utilization (0-1)
        iteration_count: Number of iterations
    """

    latency_sec: float = 0.0
    vram_bytes: int = 0
    scale_events: int = 0
    gpu_util: float = 0.0
    cpu_util: float = 0.0
    iteration_count: int = 0

    @property
    def vram_gb(self) -> float:
        """VRAM in gigabytes."""
        return self.vram_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_sec": self.latency_sec,
            "vram_bytes": self.vram_bytes,
            "vram_gb": self.vram_gb,
            "scale_events": self.scale_events,
            "gpu_util": self.gpu_util,
            "cpu_util": self.cpu_util,
            "iteration_count": self.iteration_count,
        }


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    total: float
    quality_component: float
    latency_penalty: float
    vram_penalty: float
    scaling_penalty: float
    iteration_penalty: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "quality_component": self.quality_component,
            "latency_penalty": self.latency_penalty,
            "vram_penalty": self.vram_penalty,
            "scaling_penalty": self.scaling_penalty,
            "iteration_penalty": self.iteration_penalty,
        }


def compute_cost_reward(
    metrics: dict[str, float],
    system_stats: SystemStats,
    weights: CostWeights | None = None,
    baseline_score: float = 0.0,
) -> float:
    """Compute cost-aware reward from metrics and system stats.

    Reward = quality_gain - latency_penalty - vram_penalty - scaling_penalty

    Args:
        metrics: Quality metrics dict (must include "score")
        system_stats: System resource statistics
        weights: Cost weights
        baseline_score: Baseline score to compute improvement

    Returns:
        Cost-aware reward value

    Example:
        >>> metrics = {"score": 0.8, "psnr": 30.0}
        >>> stats = SystemStats(latency_sec=2.0, vram_bytes=4e9)
        >>> reward = compute_cost_reward(metrics, stats)
    """
    weights = weights or CostWeights()

    # Quality component (improvement over baseline)
    score = metrics.get("score", 0.0)
    quality = weights.quality * (score - baseline_score)

    # Penalties
    latency_penalty = weights.latency * system_stats.latency_sec
    vram_penalty = weights.vram * system_stats.vram_gb
    scaling_penalty = weights.scaling * system_stats.scale_events
    iteration_penalty = weights.iteration * system_stats.iteration_count

    return quality - latency_penalty - vram_penalty - scaling_penalty - iteration_penalty


def compute_cost_reward_detailed(
    metrics: dict[str, float],
    system_stats: SystemStats,
    weights: CostWeights | None = None,
    baseline_score: float = 0.0,
) -> RewardBreakdown:
    """Compute cost-aware reward with detailed breakdown.

    Args:
        metrics: Quality metrics
        system_stats: System stats
        weights: Cost weights
        baseline_score: Baseline for improvement

    Returns:
        RewardBreakdown with all components
    """
    weights = weights or CostWeights()

    score = metrics.get("score", 0.0)
    quality = weights.quality * (score - baseline_score)

    latency_penalty = weights.latency * system_stats.latency_sec
    vram_penalty = weights.vram * system_stats.vram_gb
    scaling_penalty = weights.scaling * system_stats.scale_events
    iteration_penalty = weights.iteration * system_stats.iteration_count

    total = quality - latency_penalty - vram_penalty - scaling_penalty - iteration_penalty

    return RewardBreakdown(
        total=total,
        quality_component=quality,
        latency_penalty=latency_penalty,
        vram_penalty=vram_penalty,
        scaling_penalty=scaling_penalty,
        iteration_penalty=iteration_penalty,
    )


class SystemStatsCollector:
    """Collector for system resource statistics.

    Tracks resource usage during pipeline execution.

    Example:
        >>> collector = SystemStatsCollector()
        >>> collector.start()
        >>> # ... run pipeline ...
        >>> stats = collector.stop()
    """

    def __init__(self) -> None:
        """Initialize collector."""
        self._start_time: float = 0.0
        self._scale_events: int = 0
        self._iteration_count: int = 0

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.time()
        self._reset_peak_memory()

    def _reset_peak_memory(self) -> None:
        """Reset CUDA peak memory tracking."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _get_peak_memory(self) -> int:
        """Get peak CUDA memory usage."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated()
        except Exception:
            pass
        return 0

    def _get_gpu_util(self) -> float:
        """Get current GPU utilization."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def record_scale_event(self) -> None:
        """Record a cluster scaling event."""
        self._scale_events += 1

    def record_iteration(self) -> None:
        """Record an iteration."""
        self._iteration_count += 1

    def stop(self) -> SystemStats:
        """Stop collecting and return stats.

        Returns:
            SystemStats with collected data
        """
        latency = time.time() - self._start_time
        vram = self._get_peak_memory()
        gpu_util = self._get_gpu_util()

        return SystemStats(
            latency_sec=latency,
            vram_bytes=vram,
            scale_events=self._scale_events,
            gpu_util=gpu_util,
            iteration_count=self._iteration_count,
        )

    def snapshot(self) -> SystemStats:
        """Get snapshot without stopping.

        Returns:
            Current SystemStats
        """
        latency = time.time() - self._start_time if self._start_time else 0.0
        vram = self._get_peak_memory()
        gpu_util = self._get_gpu_util()

        return SystemStats(
            latency_sec=latency,
            vram_bytes=vram,
            scale_events=self._scale_events,
            gpu_util=gpu_util,
            iteration_count=self._iteration_count,
        )


class CostAwareRewardShaper:
    """Reward shaper for cost-aware RL training.

    Wraps a pipeline runner to compute cost-aware rewards.

    Example:
        >>> shaper = CostAwareRewardShaper(run_fn, eval_fn, weights)
        >>> reward, metrics, stats = shaper.run_and_reward(pipeline)
    """

    def __init__(
        self,
        run_fn: Any,
        eval_fn: Any,
        weights: CostWeights | None = None,
    ) -> None:
        """Initialize reward shaper.

        Args:
            run_fn: Pipeline runner function
            eval_fn: Evaluation function
            weights: Cost weights
        """
        self.run_fn = run_fn
        self.eval_fn = eval_fn
        self.weights = weights or CostWeights()
        self.collector = SystemStatsCollector()

        self._baseline_score: float = 0.0

    def set_baseline(self, score: float) -> None:
        """Set baseline score for improvement calculation.

        Args:
            score: Baseline score
        """
        self._baseline_score = score

    def run_and_reward(
        self,
        pipeline: dict[str, Any],
    ) -> tuple[float, dict[str, float], SystemStats]:
        """Run pipeline and compute cost-aware reward.

        Args:
            pipeline: Pipeline configuration

        Returns:
            Tuple of (reward, metrics, system_stats)
        """
        self.collector.start()

        try:
            output = self.run_fn(pipeline)
            metrics = self.eval_fn(output)

        except Exception as e:
            logger.error("Pipeline run failed: %s", e)
            stats = self.collector.stop()
            return -1.0, {}, stats

        stats = self.collector.stop()

        reward = compute_cost_reward(
            metrics,
            stats,
            self.weights,
            self._baseline_score,
        )

        return reward, metrics, stats

    def record_scale_event(self) -> None:
        """Record a scaling event."""
        self.collector.record_scale_event()

    def record_iteration(self) -> None:
        """Record an iteration."""
        self.collector.record_iteration()


# Preset weight configurations
QUALITY_FIRST_WEIGHTS = CostWeights(
    quality=1.0,
    latency=0.01,
    vram=0.02,
    scaling=0.05,
)

BALANCED_WEIGHTS = CostWeights(
    quality=1.0,
    latency=0.05,
    vram=0.1,
    scaling=0.2,
)

EFFICIENCY_FIRST_WEIGHTS = CostWeights(
    quality=0.5,
    latency=0.2,
    vram=0.2,
    scaling=0.3,
)

COST_SENSITIVE_WEIGHTS = CostWeights(
    quality=0.3,
    latency=0.3,
    vram=0.3,
    scaling=0.5,
)
