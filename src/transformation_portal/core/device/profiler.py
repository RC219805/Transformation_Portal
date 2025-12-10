"""
Performance profiling utilities.

Provides consistent performance profiling across all pipelines with GPU support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class ProfileResult:
    """Performance profile result."""
    name: str
    duration_ms: float
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    gpu_memory_start_mb: Optional[float] = None
    gpu_memory_end_mb: Optional[float] = None
    gpu_duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Calculate memory change."""
        if self.memory_start_mb is not None and self.memory_end_mb is not None:
            return self.memory_end_mb - self.memory_start_mb
        return None
    
    @property
    def gpu_memory_delta_mb(self) -> Optional[float]:
        """Calculate GPU memory change."""
        if self.gpu_memory_start_mb is not None and self.gpu_memory_end_mb is not None:
            return self.gpu_memory_end_mb - self.gpu_memory_start_mb
        return None
    
    def __str__(self) -> str:
        """Format profile result as string."""
        lines = [f"{self.name}: {self.duration_ms:.1f}ms"]
        
        if self.gpu_duration_ms is not None:
            lines.append(f"  GPU: {self.gpu_duration_ms:.1f}ms")
        
        if self.memory_delta_mb is not None:
            lines.append(f"  Memory: {self.memory_delta_mb:+.1f}MB")
        
        if self.gpu_memory_delta_mb is not None:
            lines.append(f"  GPU Memory: {self.gpu_memory_delta_mb:+.1f}MB")
        
        if self.memory_peak_mb is not None:
            lines.append(f"  Peak: {self.memory_peak_mb:.1f}MB")
        
        return "\n".join(lines)


class PerformanceProfiler:
    """
    Performance profiler for pipeline operations.
    
    Tracks execution time and optionally memory usage.
    
    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile("operation"):
        ...     do_something()
        >>> 
        >>> results = profiler.get_results()
        >>> profiler.print_summary()
    """
    
    def __init__(self, enable_memory_tracking: bool = True, enable_gpu_profiling: bool = True):
        """
        Initialize profiler.
        
        Args:
            enable_memory_tracking: Track memory usage (requires psutil)
            enable_gpu_profiling: Track GPU metrics (requires torch with CUDA/MPS)
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_gpu_profiling = enable_gpu_profiling and TORCH_AVAILABLE
        self._results: list[ProfileResult] = []
        self._process = None
        
        if enable_memory_tracking:
            try:
                import psutil
                self._process = psutil.Process()
            except ImportError:
                logger.warning("psutil not available, memory tracking disabled")
                self.enable_memory_tracking = False
        
        if self.enable_gpu_profiling and not (torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())):
            logger.debug("No GPU available, GPU profiling disabled")
            self.enable_gpu_profiling = False
    
    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Profile a code block with CPU and GPU metrics.
        
        Args:
            name: Profile name
            **metadata: Additional metadata to store
            
        Yields:
            ProfileResult that will be populated after block completes
        """
        # Get starting memory
        memory_start = self._get_memory_mb() if self.enable_memory_tracking else None
        gpu_memory_start = self._get_gpu_memory_mb() if self.enable_gpu_profiling else None
        
        # Start GPU timing
        start_event = None
        end_event = None
        if self.enable_gpu_profiling and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        # Start CPU timer
        start_time = time.perf_counter()
        
        # Create result object (will be populated later)
        result = ProfileResult(
            name=name,
            duration_ms=0.0,
            memory_start_mb=memory_start,
            gpu_memory_start_mb=gpu_memory_start,
            metadata=metadata
        )
        
        try:
            yield result
        finally:
            # Stop CPU timer
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Stop GPU timing
            gpu_duration = None
            if self.enable_gpu_profiling and start_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                gpu_duration = start_event.elapsed_time(end_event)
            
            # Get ending memory
            memory_end = self._get_memory_mb() if self.enable_memory_tracking else None
            gpu_memory_end = self._get_gpu_memory_mb() if self.enable_gpu_profiling else None
            
            # Update result
            result.duration_ms = duration_ms
            result.memory_end_mb = memory_end
            result.gpu_memory_end_mb = gpu_memory_end
            result.gpu_duration_ms = gpu_duration
            
            # Store result
            self._results.append(result)
    
    def _get_memory_mb(self) -> Optional[float]:
        """Get current CPU memory usage in MB."""
        if self._process is None:
            return None
        
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception:
            return None
    
    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self.enable_gpu_profiling:
            return None
        
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have memory tracking yet
                return None
        except Exception:
            return None
        
        return None
    
    def get_results(self) -> list[ProfileResult]:
        """Get all profile results."""
        return self._results.copy()
    
    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get a specific result by name (returns last if multiple)."""
        for result in reversed(self._results):
            if result.name == name:
                return result
        return None
    
    def print_summary(self, top_n: Optional[int] = None):
        """
        Print performance summary.
        
        Args:
            top_n: Show only top N slowest operations
        """
        if not self._results:
            logger.info("No profile results")
            return
        
        # Sort by duration
        sorted_results = sorted(self._results, key=lambda r: r.duration_ms, reverse=True)
        
        if top_n is not None:
            sorted_results = sorted_results[:top_n]
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE PROFILE")
        logger.info("=" * 60)
        
        total_time = sum(r.duration_ms for r in self._results)
        
        for result in sorted_results:
            pct = (result.duration_ms / total_time * 100) if total_time > 0 else 0
            logger.info(f"{result.name:.<40} {result.duration_ms:>8.1f}ms ({pct:>5.1f}%)")
            
            if result.gpu_duration_ms is not None:
                logger.info(f"  {'GPU time':<38} {result.gpu_duration_ms:>8.1f}ms")
            
            if result.memory_delta_mb is not None:
                logger.info(f"  {'Memory change':<38} {result.memory_delta_mb:>+8.1f}MB")
            
            if result.gpu_memory_delta_mb is not None:
                logger.info(f"  {'GPU memory change':<38} {result.gpu_memory_delta_mb:>+8.1f}MB")
        
        logger.info("=" * 60)
        logger.info(f"{'Total':<40} {total_time:>8.1f}ms")
        logger.info("=" * 60)
    
    def clear(self):
        """Clear all results."""
        self._results.clear()


class GPUProfiler:
    """
    Lightweight GPU profiler with <5% overhead.
    
    Designed for production use with minimal performance impact.
    Uses CUDA events for accurate GPU timing.
    
    Example:
        >>> profiler = GPUProfiler(enabled=True)
        >>> with profiler.profile("model_inference"):
        ...     output = model(input_tensor)
        >>> 
        >>> report = profiler.report()
        >>> print(f"Total time: {report['total_ms']:.1f}ms")
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize GPU profiler.
        
        Args:
            enabled: Enable profiling (default False for production)
        """
        self.enabled = enabled
        self.events = []
        self._cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    @contextmanager
    def profile(self, name: str):
        """
        Profile a GPU operation with minimal overhead.
        
        Args:
            name: Operation name for reporting
            
        Yields:
            None
        """
        if not self.enabled:
            yield
            return
        
        # Start timing
        cpu_start = time.perf_counter()
        
        if self._cuda_available:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        try:
            yield
        finally:
            # Stop timing
            cpu_duration = (time.perf_counter() - cpu_start) * 1000
            
            if self._cuda_available:
                end_event.record()
                torch.cuda.synchronize()
                gpu_duration = start_event.elapsed_time(end_event)
                
                self.events.append({
                    "name": name,
                    "cpu_ms": cpu_duration,
                    "gpu_ms": gpu_duration
                })
            else:
                self.events.append({
                    "name": name,
                    "cpu_ms": cpu_duration
                })
    
    def report(self) -> dict:
        """
        Generate profiling report.
        
        Returns:
            Dictionary with total_ms and per-stage breakdown
        """
        if not self.events:
            return {"total_ms": 0.0, "stages": []}
        
        total_time = sum(e.get("gpu_ms", e["cpu_ms"]) for e in self.events)
        
        return {
            "total_ms": total_time,
            "stages": self.events.copy()
        }
    
    def print_report(self):
        """Print formatted profiling report."""
        report = self.report()
        
        if not report["stages"]:
            logger.info("No profiling data collected")
            return
        
        logger.info("=" * 60)
        logger.info("GPU PROFILING REPORT")
        logger.info("=" * 60)
        
        for event in report["stages"]:
            name = event["name"]
            if "gpu_ms" in event:
                logger.info(f"{name:.<40} GPU: {event['gpu_ms']:>7.1f}ms  CPU: {event['cpu_ms']:>7.1f}ms")
            else:
                logger.info(f"{name:.<40} CPU: {event['cpu_ms']:>7.1f}ms")
        
        logger.info("=" * 60)
        logger.info(f"{'Total':<40} {report['total_ms']:>7.1f}ms")
        logger.info("=" * 60)
    
    def clear(self):
        """Clear all profiling events."""
        self.events.clear()


class StageProfiler:
    """
    Zero-overhead stage timing profiler for pipeline instrumentation.
    
    Designed for Phase 2 performance optimization with <3% overhead.
    Returns timing in seconds with snake_case keys for consistency.
    
    Example:
        >>> profiler = StageProfiler(enabled=True)
        >>> with profiler.stage("load"):
        ...     image = load_image()
        >>> with profiler.stage("depth"):
        ...     depth = estimate_depth(image)
        >>> 
        >>> timings = profiler.summary_s()
        >>> print(f"Load: {timings['load']:.3f}s")
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize stage profiler.
        
        Args:
            enabled: Enable profiling (default True for production monitoring)
        """
        self.enabled = enabled
        self._stage_times: dict[str, float] = {}
    
    @contextmanager
    def stage(self, name: str):
        """
        Profile a pipeline stage with minimal overhead.
        
        Args:
            name: Stage name (use stable names like 'load', 'depth', 'upscale_infer')
            
        Yields:
            None
            
        Note:
            Accumulates time if the same stage is entered multiple times.
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._stage_times[name] = self._stage_times.get(name, 0.0) + elapsed
    
    def summary_s(self) -> dict[str, float]:
        """
        Get timing summary in seconds.
        
        Returns:
            Dictionary mapping stage names to elapsed time in seconds
        """
        return self._stage_times.copy()
    
    def clear(self):
        """Clear all stage timings."""
        self._stage_times.clear()
