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

import atexit
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SHUTTING_DOWN = False


def _mark_shutting_down() -> None:
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True


atexit.register(_mark_shutting_down)


class TimingContext:
    """Simple timing context for capturing elapsed time with GPU synchronization.

    Attributes:
        phase_name: Name of the phase being timed
        elapsed_sec: Elapsed time in seconds (available after exit)
        timings_dict: Optional dict to accumulate timings into
        device: Optional device string for GPU synchronization ("mps", "cuda", "cpu")
    """

    def __init__(self, phase_name: str, timings_dict: Optional[Dict[str, float]] = None, device: Optional[str] = None) -> None:
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

        NOTE:
          * This is best-effort instrumentation only.
          * MPS sync is disabled by default because it can segfault on some
            macOS/PyTorch combinations.
          * Enable MPS sync explicitly with TP_TIMING_SYNC_MPS=1 if needed.

        Controls:
          * TP_DISABLE_DEVICE_SYNC=1 disables all device synchronization.
          * TP_TIMING_SYNC_MPS=1 enables MPS synchronization.
        """
        if _SHUTTING_DOWN:
            return

        if os.getenv("TP_DISABLE_DEVICE_SYNC", "").strip().lower() in {"1", "true", "yes"}:
            return

        if self.device not in {"mps", "cuda"}:
            return

        if self.device == "mps":
            if os.getenv("TP_TIMING_SYNC_MPS", "").strip().lower() not in {"1", "true", "yes"}:
                return

        try:
            import torch
        except ImportError:
            # torch not installed - skip GPU synchronization
            logger.debug("torch not available for GPU sync in timing context")
            return

        try:
            if self.device == "cuda":
                if hasattr(torch, "cuda") and torch.cuda.is_available() and hasattr(torch.cuda, "synchronize"):
                    torch.cuda.synchronize()
                return

            if (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "synchronize")
            ):
                torch.mps.synchronize()
        except RuntimeError as e:
            # GPU sync may fail if device not properly initialized or during shutdown
            logger.debug("GPU sync failed in timing context: %s", e)
            return


@contextmanager
def timing_context(
    phase_name: str, timings_dict: Optional[Dict[str, float]] = None, device: Optional[str] = None
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
