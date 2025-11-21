"""
Performance Monitor and Profiler

Real-time performance monitoring and profiling for tensor operations,
model inference, and system resource utilization.

Key Features:
- Real-time performance metrics collection
- Operation-level profiling
- Memory usage tracking
- Throughput analysis
- Performance bottleneck detection
- Export metrics for analysis
"""

import time
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from enum import Enum
import logging

import torch

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"  # Execution time
    THROUGHPUT = "throughput"  # Items per second
    MEMORY = "memory"  # Memory usage
    GPU_UTILIZATION = "gpu_utilization"  # GPU usage percentage
    BANDWIDTH = "bandwidth"  # Data transfer rate


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationProfile:
    """Profile data for a single operation."""
    operation_name: str
    total_calls: int = 0
    total_time_seconds: float = 0.0
    min_time_seconds: float = float('inf')
    max_time_seconds: float = 0.0
    avg_time_seconds: float = 0.0
    total_memory_allocated_mb: float = 0.0
    avg_memory_allocated_mb: float = 0.0

    def update(self, execution_time: float, memory_mb: float = 0.0):
        """Update profile with new measurement."""
        self.total_calls += 1
        self.total_time_seconds += execution_time
        self.min_time_seconds = min(self.min_time_seconds, execution_time)
        self.max_time_seconds = max(self.max_time_seconds, execution_time)
        self.avg_time_seconds = self.total_time_seconds / self.total_calls

        self.total_memory_allocated_mb += memory_mb
        self.avg_memory_allocated_mb = self.total_memory_allocated_mb / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation_name,
            "calls": self.total_calls,
            "total_time_s": round(self.total_time_seconds, 4),
            "avg_time_ms": round(self.avg_time_seconds * 1000, 3),
            "min_time_ms": round(self.min_time_seconds * 1000, 3),
            "max_time_ms": round(self.max_time_seconds * 1000, 3),
            "avg_memory_mb": round(self.avg_memory_allocated_mb, 2),
        }


class MetricsCollector:
    """
    Collector for performance metrics.

    Aggregates metrics from various sources and provides query interface.
    """

    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_metrics: Maximum number of metrics to store
        """
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetric] = []
        self.operation_profiles: Dict[str, OperationProfile] = {}

    def record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        unit: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a performance metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for filtering
        """
        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )

        self.metrics.append(metric)

        # Limit storage
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]

    def record_operation(
        self,
        operation_name: str,
        execution_time: float,
        memory_mb: float = 0.0
    ):
        """
        Record operation execution metrics.

        Args:
            operation_name: Name of operation
            execution_time: Execution time in seconds
            memory_mb: Memory allocated in MB
        """
        if operation_name not in self.operation_profiles:
            self.operation_profiles[operation_name] = OperationProfile(operation_name)

        self.operation_profiles[operation_name].update(execution_time, memory_mb)

        # Also record as individual metrics
        self.record_metric(
            f"{operation_name}_latency",
            MetricType.LATENCY,
            execution_time * 1000,  # Convert to ms
            "ms",
            tags={"operation": operation_name}
        )

        if memory_mb > 0:
            self.record_metric(
                f"{operation_name}_memory",
                MetricType.MEMORY,
                memory_mb,
                "MB",
                tags={"operation": operation_name}
            )

    def get_operation_profile(self, operation_name: str) -> Optional[OperationProfile]:
        """Get profile for specific operation."""
        return self.operation_profiles.get(operation_name)

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        """Get all operation profiles as list of dicts."""
        return [profile.to_dict() for profile in self.operation_profiles.values()]

    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        name_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """
        Query metrics with filters.

        Args:
            metric_type: Filter by metric type
            name_filter: Filter by name substring
            limit: Limit number of results

        Returns:
            List of matching metrics
        """
        results = self.metrics

        if metric_type:
            results = [m for m in results if m.metric_type == metric_type]

        if name_filter:
            results = [m for m in results if name_filter in m.name]

        if limit:
            results = results[-limit:]

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {"status": "no_metrics"}

        # Group by type
        by_type = defaultdict(list)
        for metric in self.metrics:
            by_type[metric.metric_type].append(metric.value)

        summary = {
            "total_metrics": len(self.metrics),
            "operations_profiled": len(self.operation_profiles),
            "by_type": {},
        }

        for metric_type, values in by_type.items():
            summary["by_type"][metric_type.value] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return summary

    def clear(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.operation_profiles.clear()


class PerformanceMonitor:
    """
    Performance monitor for real-time profiling.

    Provides decorators and context managers for profiling operations,
    with automatic metric collection.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        enable_memory_tracking: bool = True
    ):
        """
        Initialize performance monitor.

        Args:
            device: Target device for monitoring
            enable_memory_tracking: Track memory usage
        """
        self.device = device or torch.device("cpu")
        self.enable_memory_tracking = enable_memory_tracking
        self.collector = MetricsCollector()
        self._enabled = True

        logger.info(f"Performance monitor initialized for {self.device}")

    def enable(self):
        """Enable performance monitoring."""
        self._enabled = True

    def disable(self):
        """Disable performance monitoring."""
        self._enabled = False

    def profile_operation(self, operation_name: str):
        """
        Decorator for profiling operations.

        Usage:
            @monitor.profile_operation("my_operation")
            def my_operation(x):
                return x * 2
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                # Get memory before
                memory_before = self._get_memory_usage()

                # Time execution
                start_time = time.time()
                result = func(*args, **kwargs)
                self._synchronize()
                elapsed = time.time() - start_time

                # Get memory after
                memory_after = self._get_memory_usage()
                memory_delta = memory_after - memory_before

                # Record metrics
                self.collector.record_operation(
                    operation_name,
                    elapsed,
                    memory_delta
                )

                return result

            return wrapper
        return decorator

    def profile_context(self, context_name: str):
        """
        Context manager for profiling code blocks.

        Usage:
            with monitor.profile_context("data_loading"):
                data = load_data()
        """
        return _ProfileContext(self, context_name)

    def benchmark(
        self,
        operation: Callable,
        *args,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        operation_name: str = "benchmark",
        **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark an operation.

        Args:
            operation: Operation to benchmark
            *args: Positional arguments
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
            operation_name: Name for logging
            **kwargs: Keyword arguments

        Returns:
            Benchmark statistics
        """
        # Warmup
        for _ in range(warmup_iterations):
            operation(*args, **kwargs)

        self._synchronize()

        # Benchmark
        times = []
        memory_usage = []

        for _ in range(num_iterations):
            memory_before = self._get_memory_usage()

            start_time = time.time()
            operation(*args, **kwargs)
            self._synchronize()
            elapsed = time.time() - start_time

            memory_after = self._get_memory_usage()

            times.append(elapsed)
            memory_usage.append(memory_after - memory_before)

        # Calculate statistics
        stats = {
            "iterations": num_iterations,
            "avg_time_ms": sum(times) / len(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": self._std(times) * 1000,
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "throughput_per_sec": 1.0 / (sum(times) / len(times)),
        }

        # Record benchmark results
        self.collector.record_metric(
            f"{operation_name}_benchmark",
            MetricType.LATENCY,
            stats["avg_time_ms"],
            "ms",
            tags={"type": "benchmark"}
        )

        logger.info(
            f"Benchmark {operation_name}: "
            f"{stats['avg_time_ms']:.3f}ms avg, "
            f"{stats['throughput_per_sec']:.1f} ops/sec"
        )

        return stats

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking:
            return 0.0

        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        elif self.device.type == "mps":
            # MPS doesn't expose detailed memory stats
            return 0.0
        else:
            return 0.0

    def _synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_summary(self) -> str:
        """Get human-readable performance summary."""
        profiles = self.collector.get_all_profiles()

        if not profiles:
            return "No operations profiled yet"

        # Sort by total time
        profiles.sort(key=lambda p: p["total_time_s"], reverse=True)

        lines = [
            "=" * 90,
            "PERFORMANCE PROFILE SUMMARY",
            "=" * 90,
            f"{'Operation':<30} {'Calls':>8} {'Avg(ms)':>10} {'Total(s)':>10} {'Memory(MB)':>12}",
            "-" * 90,
        ]

        for profile in profiles[:20]:  # Top 20
            lines.append(
                f"{profile['operation']:<30} "
                f"{profile['calls']:>8} "
                f"{profile['avg_time_ms']:>10.3f} "
                f"{profile['total_time_s']:>10.3f} "
                f"{profile['avg_memory_mb']:>12.2f}"
            )

        lines.append("=" * 90)

        # Add summary stats
        summary = self.collector.get_summary()
        if "by_type" in summary:
            lines.append("\nMetrics Summary:")
            for metric_type, stats in summary["by_type"].items():
                lines.append(
                    f"  {metric_type}: {stats['count']} measurements, "
                    f"avg={stats['avg']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
                )

        return "\n".join(lines)

    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file.

        Args:
            filepath: Path to output file
        """
        import json

        data = {
            "device": str(self.device),
            "profiles": self.collector.get_all_profiles(),
            "summary": self.collector.get_summary(),
            "timestamp": time.time(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")

    def reset(self):
        """Reset all collected metrics."""
        self.collector.clear()

    def __repr__(self) -> str:
        return (
            f"PerformanceMonitor(device={self.device}, "
            f"enabled={self._enabled}, "
            f"operations={len(self.collector.operation_profiles)})"
        )


class _ProfileContext:
    """Context manager for profiling code blocks."""

    def __init__(self, monitor: PerformanceMonitor, context_name: str):
        self.monitor = monitor
        self.context_name = context_name
        self.start_time = 0.0
        self.memory_before = 0.0

    def __enter__(self):
        if not self.monitor._enabled:
            return self

        self.memory_before = self.monitor._get_memory_usage()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.monitor._enabled:
            return

        self.monitor._synchronize()
        elapsed = time.time() - self.start_time
        memory_after = self.monitor._get_memory_usage()
        memory_delta = memory_after - self.memory_before

        self.monitor.collector.record_operation(
            self.context_name,
            elapsed,
            memory_delta
        )
