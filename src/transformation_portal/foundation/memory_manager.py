"""
Memory Manager for Unified Memory Architecture

Intelligent memory allocation and management optimized for Apple Silicon's
unified memory architecture, where CPU and GPU share the same physical memory.

Key Features:
- Unified memory-aware allocation strategies
- Memory pool management for frequent allocations
- Automatic garbage collection and cache management
- Memory pressure monitoring and adaptation
- Allocation tracking and profiling
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
from collections import defaultdict, OrderedDict
import weakref

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _get_default_device() -> torch.device:
    """
    Get the default device based on availability.
    
    Returns:
        torch.device: The best available device (MPS > CUDA > CPU)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    IMMEDIATE = "immediate"  # Allocate immediately, no pooling
    POOLED = "pooled"  # Use memory pools for reuse
    LAZY = "lazy"  # Delay allocation until first use
    AGGRESSIVE_CACHE = "aggressive_cache"  # Cache aggressively, minimal cleanup
    CONSERVATIVE = "conservative"  # Frequent cleanup, minimal caching


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    strategy: AllocationStrategy = AllocationStrategy.POOLED
    max_memory_gb: float = 100.0  # Maximum memory to use
    high_watermark: float = 0.85  # Trigger cleanup above this threshold
    low_watermark: float = 0.70  # Target after cleanup
    pool_size_mb: int = 1024  # Size of memory pools
    enable_profiling: bool = False
    gc_interval_seconds: float = 60.0  # Garbage collection interval


@dataclass
class AllocationInfo:
    """Information about a tensor allocation."""
    tensor_id: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    timestamp: float
    tag: Optional[str] = None


class MemoryPool:
    """Memory pool for efficient tensor reuse."""

    def __init__(self, size_mb: int, device: torch.device):
        """
        Initialize memory pool.

        Args:
            size_mb: Pool size in megabytes
            device: Target device
        """
        self.size_bytes = size_mb * 1024 * 1024
        self.device = device
        self.available: OrderedDict[Tuple, List[Tensor]] = OrderedDict()
        self.allocated_bytes = 0

    def get(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[Tensor]:
        """
        Get tensor from pool if available.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Cached tensor or None
        """
        key = (shape, dtype)
        if key in self.available and self.available[key]:
            tensor = self.available[key].pop()
            return tensor
        return None

    def put(self, tensor: Tensor) -> bool:
        """
        Return tensor to pool.

        Args:
            tensor: Tensor to cache

        Returns:
            True if cached, False if pool is full
        """
        tensor_size = tensor.element_size() * tensor.numel()

        # Check if pool has space
        if self.allocated_bytes + tensor_size > self.size_bytes:
            return False

        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.available:
            self.available[key] = []

        self.available[key].append(tensor)
        self.allocated_bytes += tensor_size
        return True

    def clear(self):
        """Clear all cached tensors."""
        self.available.clear()
        self.allocated_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_keys": len(self.available),
            "total_tensors": sum(len(v) for v in self.available.values()),
            "allocated_mb": self.allocated_bytes / (1024 * 1024),
            "capacity_mb": self.size_bytes / (1024 * 1024),
            "utilization": self.allocated_bytes / self.size_bytes if self.size_bytes > 0 else 0
        }


class MemoryManager:
    """
    Intelligent memory manager for unified memory architectures.

    Optimized for Apple Silicon where CPU and GPU share physical memory,
    enabling efficient allocation strategies that leverage unified memory benefits.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
            device: Target device
        """
        self.config = config or MemoryConfig()
        self.device = device if device is not None else _get_default_device()

        # Memory pools for different allocation sizes
        self.pools: Dict[str, MemoryPool] = {}
        self._init_pools()

        # Allocation tracking
        self.allocations: Dict[int, AllocationInfo] = {}
        self.allocation_stats = defaultdict(lambda: {"count": 0, "total_bytes": 0})

        # Memory pressure monitoring
        self.last_gc_time = time.time()
        self.peak_memory_bytes = 0

        # Weak references to tracked tensors
        self.tracked_tensors = weakref.WeakValueDictionary()

        logger.info(f"Initialized MemoryManager with strategy={self.config.strategy.value}")

    def _init_pools(self):
        """Initialize memory pools for different tensor sizes."""
        # Create pools for common tensor sizes
        pool_configs = {
            "small": 256,    # 256MB for small tensors (< 10MB)
            "medium": 512,   # 512MB for medium tensors (10-100MB)
            "large": 1024,   # 1GB for large tensors (> 100MB)
        }

        for name, size_mb in pool_configs.items():
            self.pools[name] = MemoryPool(size_mb, self.device)

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        tag: Optional[str] = None,
        use_pool: bool = True
    ) -> Tensor:
        """
        Allocate tensor with intelligent memory management.

        Args:
            shape: Tensor shape
            dtype: Data type
            tag: Optional tag for tracking
            use_pool: Whether to use memory pooling

        Returns:
            Allocated tensor
        """
        # Calculate size
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = element_size * torch.prod(torch.tensor(shape)).item()
        size_mb = size_bytes / (1024 * 1024)

        # Try to get from pool if using pooled strategy
        tensor = None
        if use_pool and self.config.strategy in [AllocationStrategy.POOLED, AllocationStrategy.AGGRESSIVE_CACHE]:
            pool_name = self._get_pool_name(size_mb)
            tensor = self.pools[pool_name].get(shape, dtype)

        # Allocate new tensor if not found in pool
        if tensor is None:
            # Check memory pressure before allocation
            self._check_memory_pressure(size_bytes)

            tensor = torch.empty(shape, dtype=dtype, device=self.device)

        # Track allocation
        if self.config.enable_profiling:
            self._track_allocation(tensor, shape, dtype, size_bytes, tag)

        return tensor

    def deallocate(self, tensor: Tensor, return_to_pool: bool = True) -> bool:
        """
        Deallocate tensor and optionally return to pool.

        Args:
            tensor: Tensor to deallocate
            return_to_pool: Whether to return to pool for reuse

        Returns:
            True if returned to pool, False otherwise
        """
        tensor_id = id(tensor)

        # Remove from tracking
        if tensor_id in self.allocations:
            del self.allocations[tensor_id]

        # Return to pool if requested
        if return_to_pool and self.config.strategy in [AllocationStrategy.POOLED, AllocationStrategy.AGGRESSIVE_CACHE]:
            size_mb = (tensor.element_size() * tensor.numel()) / (1024 * 1024)
            pool_name = self._get_pool_name(size_mb)
            return self.pools[pool_name].put(tensor)

        return False

    def allocate_batch(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        tag: Optional[str] = None
    ) -> List[Tensor]:
        """
        Allocate batch of tensors efficiently.

        Args:
            batch_size: Number of tensors
            shape: Shape per tensor
            dtype: Data type
            tag: Optional tag for tracking

        Returns:
            List of allocated tensors
        """
        # For unified memory, it's more efficient to allocate a single
        # large tensor and split it
        batch_shape = (batch_size,) + shape
        batch_tensor = self.allocate(batch_shape, dtype, tag=f"{tag}_batch" if tag else None)

        # Split into individual tensors
        tensors = list(torch.split(batch_tensor, 1, dim=0))
        return [t.squeeze(0) for t in tensors]

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "device": str(self.device),
            "strategy": self.config.strategy.value,
            "max_memory_gb": self.config.max_memory_gb,
        }

        # Device-specific stats
        if self.device.type == "cuda":
            stats.update({
                "allocated_gb": torch.cuda.memory_allocated(self.device) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(self.device) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / (1024**3),
                "cached_gb": torch.cuda.memory_reserved(self.device) / (1024**3),
            })
        elif self.device.type == "mps":
            stats.update({
                "unified_memory": True,
                "allocated_gb": self.peak_memory_bytes / (1024**3),
            })

        # Pool stats
        pool_stats = {}
        for name, pool in self.pools.items():
            pool_stats[name] = pool.get_stats()
        stats["pools"] = pool_stats

        # Allocation stats
        if self.config.enable_profiling:
            stats["allocations"] = {
                "active_count": len(self.allocations),
                "total_allocated_mb": sum(a.size_bytes for a in self.allocations.values()) / (1024**2),
                "by_tag": dict(self.allocation_stats),
            }

        return stats

    def clear_cache(self):
        """Clear all cached memory."""
        # Clear pools
        for pool in self.pools.values():
            pool.clear()

        # Clear device cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            import gc
            gc.collect()

        logger.info("Memory cache cleared")

    def optimize_memory(self):
        """
        Optimize memory usage by cleaning up unused allocations.

        This performs garbage collection and pool cleanup based on current
        memory pressure.
        """
        current_time = time.time()

        # Check if GC interval has elapsed
        if current_time - self.last_gc_time < self.config.gc_interval_seconds:
            return

        logger.debug("Running memory optimization...")

        # Get current memory usage
        current_usage = self._get_memory_usage()
        usage_ratio = current_usage / (self.config.max_memory_gb * 1024**3)

        # If above high watermark, aggressive cleanup
        if usage_ratio > self.config.high_watermark:
            logger.info(f"Memory pressure high ({usage_ratio:.1%}), performing aggressive cleanup")
            self._aggressive_cleanup()

        # If above low watermark but below high, moderate cleanup
        elif usage_ratio > self.config.low_watermark:
            logger.debug(f"Memory pressure moderate ({usage_ratio:.1%}), performing moderate cleanup")
            self._moderate_cleanup()

        self.last_gc_time = current_time

    def _check_memory_pressure(self, requested_bytes: int):
        """Check memory pressure before allocation."""
        current_usage = self._get_memory_usage()
        projected_usage = current_usage + requested_bytes
        max_bytes = self.config.max_memory_gb * 1024**3

        if projected_usage > max_bytes * self.config.high_watermark:
            logger.warning("Memory pressure detected, running optimization before allocation")
            self.optimize_memory()

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        elif self.device.type == "mps":
            # Estimate from tracked allocations
            return sum(a.size_bytes for a in self.allocations.values())
        else:
            return 0

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Clear all pools
        for pool in self.pools.values():
            pool.clear()

        # Force garbage collection
        import gc
        gc.collect()

        # Clear device cache
        self.clear_cache()

    def _moderate_cleanup(self):
        """Perform moderate memory cleanup."""
        # Clear only the largest pool
        if "large" in self.pools:
            self.pools["large"].clear()

        # Force garbage collection
        import gc
        gc.collect()

    def _get_pool_name(self, size_mb: float) -> str:
        """Determine appropriate pool for tensor size."""
        if size_mb < 10:
            return "small"
        elif size_mb < 100:
            return "medium"
        else:
            return "large"

    def _track_allocation(
        self,
        tensor: Tensor,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        size_bytes: int,
        tag: Optional[str]
    ):
        """Track tensor allocation for profiling."""
        tensor_id = id(tensor)
        info = AllocationInfo(
            tensor_id=tensor_id,
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            timestamp=time.time(),
            tag=tag
        )

        self.allocations[tensor_id] = info
        self.tracked_tensors[tensor_id] = tensor

        # Update stats
        if tag:
            self.allocation_stats[tag]["count"] += 1
            self.allocation_stats[tag]["total_bytes"] += size_bytes

        # Update peak memory
        current_total = sum(a.size_bytes for a in self.allocations.values())
        self.peak_memory_bytes = max(self.peak_memory_bytes, current_total)

    def get_allocation_summary(self) -> str:
        """Get human-readable allocation summary."""
        stats = self.get_memory_stats()

        lines = [
            "=" * 70,
            "MEMORY ALLOCATION SUMMARY",
            "=" * 70,
            f"Device: {stats['device']}",
            f"Strategy: {stats['strategy']}",
            f"Max Memory: {stats['max_memory_gb']:.1f} GB",
        ]

        if "allocated_gb" in stats:
            lines.append(f"Allocated: {stats['allocated_gb']:.2f} GB")
            if "reserved_gb" in stats:
                lines.append(f"Reserved: {stats['reserved_gb']:.2f} GB")

        if "pools" in stats:
            lines.append("\nMemory Pools:")
            for name, pool_stats in stats["pools"].items():
                lines.append(
                    f"  {name}: {pool_stats['total_tensors']} tensors, "
                    f"{pool_stats['allocated_mb']:.1f}/{pool_stats['capacity_mb']:.1f} MB "
                    f"({pool_stats['utilization']:.1%} utilization)"
                )

        if "allocations" in stats:
            lines.append(f"\nActive Allocations: {stats['allocations']['active_count']}")
            lines.append(f"Total Allocated: {stats['allocations']['total_allocated_mb']:.1f} MB")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MemoryManager(device={self.device}, "
            f"strategy={self.config.strategy.value}, "
            f"max_memory={self.config.max_memory_gb:.1f}GB)"
        )
