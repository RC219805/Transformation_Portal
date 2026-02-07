"""Timing instrumentation utilities for performance ledger.

Provides context managers and decorators for capturing phase-level timings
with zero-overhead when disabled.

Usage::

    from transformation_portal.metrics.timing import TimingContext, timing_context

    # Context manager
    with timing_context("inference") as timer:
        result = model.infer(image)
    print(f"Inference took {timer.elapsed_sec:.3f}s")

    # Accumulator pattern
    timings = {}
    with timing_context("load_decode", timings):
        image = load_image(path)
    with timing_context("inference", timings):
        depth = backend.compute(image)
    # timings = {"load_decode": 0.123, "inference": 4.567}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class TimingContext:
    """Simple timing context for capturing elapsed time with GPU synchronization.

    Attributes:
        phase_name: Name of the phase being timed
        elapsed_sec: Elapsed time in seconds (available after exit)
        timings_dict: Optional dict to accumulate timings into
        device: Optional device string for GPU synchronization ("mps", "cuda", "cpu")
    """

    def __init__(
        self,
        phase_name: str,
        timings_dict: Optional[Dict[str, float]] = None,
        device: Optional[str] = None
    ) -> None:
        self.phase_name = phase_name
        self.elapsed_sec: float = 0.0
        self.timings_dict = timings_dict
        self.device = device
        self._start_time: Optional[float] = None

    def __enter__(self) -> TimingContext:
        """Start timing."""
        self._sync_device()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and record elapsed time."""
        self._sync_device()  # CRITICAL: sync before reading timer
        if self._start_time is not None:
            self.elapsed_sec = time.perf_counter() - self._start_time

            # Accumulate into dict if provided
            if self.timings_dict is not None:
                self.timings_dict[self.phase_name] = self.elapsed_sec

    def _sync_device(self) -> None:
        """Synchronize GPU/MPS before timing.

        Uses unified torch.accelerator.synchronize() when available (PyTorch 2.4+),
        falls back to device-specific sync for older versions.
        """
        if self.device in {"mps", "cuda"}:
            try:
                import torch

                # Try unified API first (PyTorch 2.4+)
                if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
                    torch.accelerator.synchronize()
                    return

                # Fallback to device-specific sync (check attribute existence first)
                if self.device == "mps":
                    if (
                        hasattr(torch, "backends")
                        and hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        torch.mps.synchronize()
                elif self.device == "cuda":
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
            except (ImportError, AttributeError):
                pass  # torch not available or incomplete, fall back to CPU timing


@contextmanager
def timing_context(
    phase_name: str,
    timings_dict: Optional[Dict[str, float]] = None,
    device: Optional[str] = None
) -> TimingContext:
    """Context manager for timing a code block with GPU synchronization.

    Args:
        phase_name: Name of the phase being timed
        timings_dict: Optional dict to accumulate timings into
        device: Optional device string for GPU sync ("mps", "cuda", "cpu")

    Yields:
        TimingContext with elapsed_sec available after exit

    Example::

        timings = {}
        with timing_context("inference", timings, device="mps") as timer:
            result = expensive_operation()
        print(f"Took {timer.elapsed_sec:.3f}s")
        # timings = {"inference": 2.345}
    """
    ctx = TimingContext(phase_name, timings_dict, device)
    with ctx:
        yield ctx


def merge_timings(*timing_dicts: Dict[str, float]) -> Dict[str, float]:
    """Merge multiple timing dicts, summing values for duplicate keys.

    Args:
        *timing_dicts: Variable number of timing dictionaries

    Returns:
        Merged timing dictionary
    """
    result: Dict[str, float] = {}
    for d in timing_dicts:
        for key, value in d.items():
            result[key] = result.get(key, 0.0) + value
    return result


def compute_overhead(timings: Dict[str, float]) -> float:
    """Compute overhead as (total - sum_of_phases).

    Args:
        timings: Dict with "total" key and phase keys

    Returns:
        Overhead in seconds (can be negative due to timing granularity)
    """
    if "total" not in timings:
        raise ValueError("timings must include 'total' key")

    total = timings["total"]
    phase_sum = sum(v for k, v in timings.items() if k != "total")

    return total - phase_sum
