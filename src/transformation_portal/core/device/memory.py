"""
Memory Management Utilities.

Handles VRAM garbage collection, cache clearing, and batch size estimation
to prevent CUDA Out-Of-Memory (OOM) errors.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    allocated_gb: float
    reserved_gb: float
    free_gb: float


class MemoryManager:
    """Static utility for memory hygiene."""

    @staticmethod
    def purge():
        """Aggressive memory cleanup."""
        # 1. Python Garbage Collector
        gc.collect()

        # 2. PyTorch CUDA Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # 3. MPS Cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def get_stats(device_index: int = 0) -> Optional[MemoryStats]:
        """Get current VRAM usage (CUDA only)."""
        if not torch.cuda.is_available():
            return None

        t = 1024**3
        allocated = torch.cuda.memory_allocated(device_index) / t
        reserved = torch.cuda.memory_reserved(device_index) / t

        props = torch.cuda.get_device_properties(device_index)
        total = props.total_memory / t

        return MemoryStats(allocated_gb=allocated, reserved_gb=reserved, free_gb=total - reserved)


def estimate_memory_usage(resolution: tuple[int, int], channels: int = 3, precision_bytes: int = 2) -> float:  # FP16
    """
    Estimate VRAM usage for a single image tensor in MB.
    Does NOT account for model weights or activation overhead.
    """
    h, w = resolution
    pixels = h * w

    # Base tensor size
    tensor_mb = (pixels * channels * precision_bytes) / (1024**2)

    # Heuristic for ML processing overhead (activations, gradients, etc.)
    # Usually 4x - 10x the raw tensor size depending on architecture
    overhead_factor = 6.0

    return tensor_mb * overhead_factor


def calculate_safe_batch_size(available_vram_gb: float, model_weights_gb: float, image_resolution: tuple[int, int]) -> int:
    """Calculate maximum batch size that fits in VRAM."""
    # Reserve buffer (system overhead, fragmentation)
    usable_vram = available_vram_gb * 0.85

    remaining_for_data = usable_vram - model_weights_gb
    if remaining_for_data <= 0:
        logger.warning("Model weights exceed estimated safe VRAM limit!")
        return 1

    per_image_gb = estimate_memory_usage(image_resolution) / 1024  # Convert MB to GB

    batch_size = int(remaining_for_data / per_image_gb)
    return max(1, batch_size)
