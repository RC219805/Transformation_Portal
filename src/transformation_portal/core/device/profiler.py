"""
Performance profiling utilities.

Provides consistent performance profiling across all pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Performance profile result."""
    name: str
    duration_ms: float
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        """Calculate memory change."""
        if self.memory_start_mb is not None and self.memory_end_mb is not None:
            return self.memory_end_mb - self.memory_start_mb
        return None
    
    def __str__(self) -> str:
        """Format profile result as string."""
        lines = [f"{self.name}: {self.duration_ms:.1f}ms"]
        
        if self.memory_delta_mb is not None:
            lines.append(f"  Memory: {self.memory_delta_mb:+.1f}MB")
        
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
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize profiler.
        
        Args:
            enable_memory_tracking: Track memory usage (requires psutil)
        """
        self.enable_memory_tracking = enable_memory_tracking
        self._results: list[ProfileResult] = []
        self._process = None
        
        if enable_memory_tracking:
            try:
                import psutil
                self._process = psutil.Process()
            except ImportError:
                logger.warning("psutil not available, memory tracking disabled")
                self.enable_memory_tracking = False
    
    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Profile a code block.
        
        Args:
            name: Profile name
            **metadata: Additional metadata to store
            
        Yields:
            ProfileResult that will be populated after block completes
        """
        # Get starting memory
        memory_start = self._get_memory_mb() if self.enable_memory_tracking else None
        
        # Start timer
        start_time = time.perf_counter()
        
        # Create result object (will be populated later)
        result = ProfileResult(
            name=name,
            duration_ms=0.0,
            memory_start_mb=memory_start,
            metadata=metadata
        )
        
        try:
            yield result
        finally:
            # Stop timer
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Get ending memory
            memory_end = self._get_memory_mb() if self.enable_memory_tracking else None
            
            # Update result
            result.duration_ms = duration_ms
            result.memory_end_mb = memory_end
            
            # Store result
            self._results.append(result)
    
    def _get_memory_mb(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if self._process is None:
            return None
        
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception:
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
            
            if result.memory_delta_mb is not None:
                logger.info(f"  {'Memory change':<38} {result.memory_delta_mb:>+8.1f}MB")
        
        logger.info("=" * 60)
        logger.info(f"{'Total':<40} {total_time:>8.1f}ms")
        logger.info("=" * 60)
    
    def clear(self):
        """Clear all results."""
        self._results.clear()
