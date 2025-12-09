"""
Memory management utilities.

Provides memory tracking and estimation for pipeline operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    
    def __str__(self) -> str:
        """Format as string."""
        return (
            f"Memory: {self.used_mb:.1f}MB used / "
            f"{self.total_mb:.1f}MB total ({self.percent:.1f}%)"
        )


class MemoryManager:
    """
    Memory usage tracker and manager.
    
    Provides utilities for tracking memory usage and estimating
    memory requirements for pipeline operations.
    """
    
    def __init__(self):
        """Initialize memory manager."""
        self._process = None
        
        try:
            import psutil
            self._process = psutil.Process()
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not available, memory tracking disabled")
    
    def get_stats(self) -> Optional[MemoryStats]:
        """
        Get current memory statistics.
        
        Returns:
            MemoryStats or None if psutil not available
        """
        if self._process is None:
            return None
        
        try:
            vm = self._psutil.virtual_memory()
            
            return MemoryStats(
                total_mb=vm.total / (1024 * 1024),
                available_mb=vm.available / (1024 * 1024),
                used_mb=vm.used / (1024 * 1024),
                percent=vm.percent
            )
        except Exception as e:
            logger.debug(f"Failed to get memory stats: {e}")
            return None
    
    def get_process_memory_mb(self) -> Optional[float]:
        """
        Get current process memory usage in MB.
        
        Returns:
            Memory usage in MB or None if not available
        """
        if self._process is None:
            return None
        
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception as e:
            logger.debug(f"Failed to get process memory: {e}")
            return None
    
    def check_available(self, required_mb: float) -> bool:
        """
        Check if sufficient memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if sufficient memory available, False otherwise
        """
        stats = self.get_stats()
        if stats is None:
            # Cannot determine, assume available
            return True
        
        return stats.available_mb >= required_mb
    
    def log_stats(self):
        """Log current memory statistics."""
        stats = self.get_stats()
        if stats:
            logger.info(str(stats))
        
        process_mem = self.get_process_memory_mb()
        if process_mem:
            logger.info(f"Process Memory: {process_mem:.1f}MB")


def estimate_memory_usage(
    image_width: int,
    image_height: int,
    channels: int = 3,
    dtype_bytes: int = 4,
    processing_overhead: float = 3.0
) -> float:
    """
    Estimate memory usage for image processing.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        channels: Number of color channels (default: 3 for RGB)
        dtype_bytes: Bytes per value (4 for float32, 2 for float16)
        processing_overhead: Overhead multiplier for intermediate buffers (default: 3x)
        
    Returns:
        Estimated memory usage in MB
    """
    # Base image size
    pixels = image_width * image_height
    base_mb = (pixels * channels * dtype_bytes) / (1024 * 1024)
    
    # Apply overhead for intermediate buffers
    estimated_mb = base_mb * processing_overhead
    
    return estimated_mb


def estimate_batch_memory(
    image_width: int,
    image_height: int,
    batch_size: int,
    channels: int = 3,
    dtype_bytes: int = 4,
    processing_overhead: float = 3.0
) -> float:
    """
    Estimate memory usage for batch processing.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        batch_size: Number of images in batch
        channels: Number of color channels (default: 3 for RGB)
        dtype_bytes: Bytes per value (4 for float32, 2 for float16)
        processing_overhead: Overhead multiplier (default: 3x)
        
    Returns:
        Estimated memory usage in MB
    """
    single_image_mb = estimate_memory_usage(
        image_width, image_height, channels, dtype_bytes, processing_overhead
    )
    
    return single_image_mb * batch_size


def calculate_safe_batch_size(
    image_width: int,
    image_height: int,
    available_memory_gb: float,
    memory_reserve_gb: float = 2.0,
    channels: int = 3,
    dtype_bytes: int = 4,
    processing_overhead: float = 3.0
) -> int:
    """
    Calculate safe batch size given available memory.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        available_memory_gb: Available memory in GB
        memory_reserve_gb: Memory to reserve (default: 2GB)
        channels: Number of color channels
        dtype_bytes: Bytes per value
        processing_overhead: Processing overhead multiplier
        
    Returns:
        Safe batch size (minimum 1)
    """
    # Convert to MB
    usable_memory_mb = (available_memory_gb - memory_reserve_gb) * 1024
    
    if usable_memory_mb <= 0:
        return 1
    
    # Estimate per-image memory
    per_image_mb = estimate_memory_usage(
        image_width, image_height, channels, dtype_bytes, processing_overhead
    )
    
    # Calculate batch size
    batch_size = int(usable_memory_mb / per_image_mb)
    
    # Ensure at least 1
    return max(1, batch_size)
