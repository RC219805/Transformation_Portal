"""GPU-aware autoscaling for Ray clusters.

This module provides autoscaling policies that adjust GPU worker
counts based on utilization, queue backlog, and latency metrics.

Features:
- Utilization-driven scaling
- Backlog-aware scale-up
- Cooldown periods to prevent thrashing
- Budget caps and guardrails
- Integration with Ray autoscaler
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AutoscalePolicy:
    """Configuration for GPU autoscaling policy.

    Attributes:
        min_workers: Minimum GPU workers
        max_workers: Maximum GPU workers (budget cap)
        scale_up_threshold: GPU utilization threshold for scale-up
        scale_down_threshold: GPU utilization threshold for scale-down
        cooldown_sec: Seconds between scaling decisions
        target_gpu_util: Target GPU utilization
        max_pending_per_gpu: Maximum pending tasks per GPU before scale-up
        scale_up_increment: Workers to add per scale-up
        scale_down_increment: Workers to remove per scale-down
        enable_predictive: Enable predictive scaling
    """

    min_workers: int = 0
    max_workers: int = 8
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.2
    cooldown_sec: int = 120
    target_gpu_util: float = 0.75
    max_pending_per_gpu: int = 4
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    enable_predictive: bool = False


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions.

    Attributes:
        gpu_utilization: Current GPU utilization (0-1)
        pending_tasks: Number of pending tasks
        current_workers: Current worker count
        avg_task_latency: Average task latency (seconds)
        queue_depth: Depth of task queue
    """

    gpu_utilization: float = 0.0
    pending_tasks: int = 0
    current_workers: int = 0
    avg_task_latency: float = 0.0
    queue_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_utilization": self.gpu_utilization,
            "pending_tasks": self.pending_tasks,
            "current_workers": self.current_workers,
            "avg_task_latency": self.avg_task_latency,
            "queue_depth": self.queue_depth,
        }


@dataclass
class ScalingDecision:
    """Result of scaling decision.

    Attributes:
        action: Scaling action (none, scale_up, scale_down)
        current_workers: Current worker count
        target_workers: Target worker count
        reason: Reason for decision
        metrics: Metrics that informed decision
    """

    action: str  # "none", "scale_up", "scale_down"
    current_workers: int
    target_workers: int
    reason: str
    metrics: ScalingMetrics


@dataclass
class ScalingHistory:
    """History of scaling decisions."""

    decisions: list[tuple[float, ScalingDecision]] = field(default_factory=list)
    max_history: int = 100

    def add(self, decision: ScalingDecision) -> None:
        """Add decision to history."""
        self.decisions.append((time.time(), decision))
        if len(self.decisions) > self.max_history:
            self.decisions.pop(0)

    def recent_scale_ups(self, window_sec: int = 600) -> int:
        """Count recent scale-up decisions."""
        cutoff = time.time() - window_sec
        return sum(
            1
            for ts, dec in self.decisions
            if ts > cutoff and dec.action == "scale_up"
        )

    def recent_scale_downs(self, window_sec: int = 600) -> int:
        """Count recent scale-down decisions."""
        cutoff = time.time() - window_sec
        return sum(
            1
            for ts, dec in self.decisions
            if ts > cutoff and dec.action == "scale_down"
        )


class GPUUtilizationMonitor:
    """Monitor GPU utilization via NVML."""

    def __init__(self) -> None:
        """Initialize monitor."""
        self._nvml_initialized = False

    def _init_nvml(self) -> bool:
        """Initialize NVML."""
        if self._nvml_initialized:
            return True

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception as e:
            logger.debug("NVML not available: %s", e)
            return False

    def get_utilization(self, device_index: int = 0) -> float:
        """Get GPU utilization for device.

        Args:
            device_index: GPU device index

        Returns:
            Utilization as fraction (0-1)
        """
        if not self._init_nvml():
            return 0.0

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except Exception as e:
            logger.debug("Failed to get GPU utilization: %s", e)
            return 0.0

    def get_memory_utilization(self, device_index: int = 0) -> float:
        """Get GPU memory utilization.

        Args:
            device_index: GPU device index

        Returns:
            Memory utilization as fraction (0-1)
        """
        if not self._init_nvml():
            return 0.0

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / mem.total
        except Exception as e:
            logger.debug("Failed to get memory utilization: %s", e)
            return 0.0

    def get_device_count(self) -> int:
        """Get number of GPU devices."""
        if not self._init_nvml():
            return 0

        try:
            import pynvml

            return pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0


class GPUAutoscaler:
    """GPU-aware autoscaler for Ray clusters.

    Monitors GPU utilization and task backlog to make scaling decisions.
    Integrates with Ray autoscaler API for cluster management.

    Example:
        >>> autoscaler = GPUAutoscaler(ray, policy)
        >>> for iteration in range(100):
        ...     autoscaler.tick()
        ...     # training step
    """

    def __init__(
        self,
        ray_client: Any,
        policy: AutoscalePolicy | None = None,
    ) -> None:
        """Initialize autoscaler.

        Args:
            ray_client: Ray client/module
            policy: Autoscaling policy
        """
        self.ray = ray_client
        self.policy = policy or AutoscalePolicy()
        self.gpu_monitor = GPUUtilizationMonitor()
        self.history = ScalingHistory()

        self._last_scale_time: float = 0.0
        self._metrics_callback: Callable[[ScalingMetrics], None] | None = None

    def set_metrics_callback(
        self,
        callback: Callable[[ScalingMetrics], None],
    ) -> None:
        """Set callback for metrics reporting.

        Args:
            callback: Function to call with metrics
        """
        self._metrics_callback = callback

    def _get_metrics(self) -> ScalingMetrics:
        """Collect current metrics."""
        metrics = ScalingMetrics()

        # GPU utilization
        metrics.gpu_utilization = self.gpu_monitor.get_utilization()

        # Ray cluster metrics
        try:
            resources = self.ray.available_resources()
            cluster_resources = self.ray.cluster_resources()

            metrics.current_workers = int(cluster_resources.get("GPU", 0))

            # Estimate pending from difference
            available_gpus = resources.get("GPU", 0)
            metrics.pending_tasks = max(
                0, metrics.current_workers - int(available_gpus)
            )

        except Exception as e:
            logger.debug("Failed to get Ray metrics: %s", e)

        return metrics

    def _decide(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        current = metrics.current_workers
        target = current
        action = "none"
        reason = "No change needed"

        # Check scale-up conditions
        should_scale_up = False
        scale_up_reason = ""

        # High GPU utilization
        if metrics.gpu_utilization > self.policy.scale_up_threshold:
            should_scale_up = True
            scale_up_reason = f"High GPU util: {metrics.gpu_utilization:.2f}"

        # High pending tasks
        max_pending = current * self.policy.max_pending_per_gpu
        if metrics.pending_tasks > max_pending and current > 0:
            should_scale_up = True
            scale_up_reason = f"High pending: {metrics.pending_tasks} > {max_pending}"

        if should_scale_up and current < self.policy.max_workers:
            target = min(
                current + self.policy.scale_up_increment,
                self.policy.max_workers,
            )
            action = "scale_up"
            reason = scale_up_reason

        # Check scale-down conditions
        elif (
            metrics.gpu_utilization < self.policy.scale_down_threshold
            and metrics.pending_tasks == 0
            and current > self.policy.min_workers
        ):
            target = max(
                current - self.policy.scale_down_increment,
                self.policy.min_workers,
            )
            action = "scale_down"
            reason = f"Low util: {metrics.gpu_utilization:.2f}, no pending"

        return ScalingDecision(
            action=action,
            current_workers=current,
            target_workers=target,
            reason=reason,
            metrics=metrics,
        )

    def _apply_scaling(self, target_workers: int) -> bool:
        """Apply scaling decision.

        Args:
            target_workers: Target worker count

        Returns:
            True if scaling was applied
        """
        try:
            # Try Ray autoscaler SDK
            from ray.autoscaler.sdk import request_resources

            request_resources(num_gpus=target_workers)
            logger.info("Requested %d GPU workers via Ray autoscaler", target_workers)
            return True

        except ImportError:
            logger.debug("Ray autoscaler SDK not available")

        except Exception as e:
            logger.warning("Failed to apply scaling: %s", e)

        return False

    def tick(self) -> ScalingDecision | None:
        """Check and apply scaling if needed.

        Should be called periodically (e.g., each training iteration).

        Returns:
            ScalingDecision if a scaling action was taken, None otherwise
        """
        now = time.time()

        # Check cooldown
        if now - self._last_scale_time < self.policy.cooldown_sec:
            return None

        # Collect metrics
        metrics = self._get_metrics()

        # Report metrics
        if self._metrics_callback:
            self._metrics_callback(metrics)

        # Make decision
        decision = self._decide(metrics)

        # Apply if needed
        if decision.action != "none":
            if self._apply_scaling(decision.target_workers):
                self._last_scale_time = now
                self.history.add(decision)

                logger.info(
                    "Scaling %s: %d -> %d workers (%s)",
                    decision.action,
                    decision.current_workers,
                    decision.target_workers,
                    decision.reason,
                )

                return decision

        return None

    def get_status(self) -> dict[str, Any]:
        """Get current autoscaler status."""
        metrics = self._get_metrics()

        return {
            "metrics": metrics.to_dict(),
            "policy": {
                "min_workers": self.policy.min_workers,
                "max_workers": self.policy.max_workers,
                "target_gpu_util": self.policy.target_gpu_util,
            },
            "last_scale_time": self._last_scale_time,
            "recent_scale_ups": self.history.recent_scale_ups(),
            "recent_scale_downs": self.history.recent_scale_downs(),
        }


def create_autoscaler(
    ray_client: Any,
    min_workers: int = 0,
    max_workers: int = 8,
    **kwargs: Any,
) -> GPUAutoscaler:
    """Factory function to create autoscaler.

    Args:
        ray_client: Ray client/module
        min_workers: Minimum workers
        max_workers: Maximum workers
        **kwargs: Additional policy arguments

    Returns:
        GPUAutoscaler instance
    """
    policy = AutoscalePolicy(
        min_workers=min_workers,
        max_workers=max_workers,
        **kwargs,
    )
    return GPUAutoscaler(ray_client, policy)
