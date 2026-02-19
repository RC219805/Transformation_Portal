"""
Performance Profiler.

Context manager for measuring execution time and VRAM spikes.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from .memory import MemoryManager

logger = logging.getLogger(__name__)


def _get_torch():
    """Lazy import for torch (may be absent in core/CI environments)."""
    import torch

    return torch


@dataclass
class ProfileResult:
    name: str
    duration_seconds: float
    vram_used_gb: Optional[float] = None
    throughput_fps: Optional[float] = None


class PerformanceProfiler:
    """
    Context manager for profiling code blocks.

    Example:
        with PerformanceProfiler("Inference") as p:
            model(input)
        print(p.last_result)
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.last_result: Optional[ProfileResult] = None
        self._start_time = 0.0
        self._start_vram = 0.0

    def __enter__(self):
        torch = _get_torch()
        # Sync GPU for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_vram = torch.cuda.memory_allocated()

        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_vram = torch.cuda.memory_allocated()
            vram_delta = (end_vram - self._start_vram) / (1024**3)
        else:
            vram_delta = None

        duration = time.time() - self._start_time

        self.last_result = ProfileResult(name=self.name, duration_seconds=duration, vram_used_gb=vram_delta)

        logger.debug(
            f"Profile [{self.name}]: {duration:.3f}s"
            + (f", VRAM Delta: {vram_delta:+.2f}GB" if vram_delta is not None else "")
        )
