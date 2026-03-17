"""GPU Pool with deterministic leasing.

This module provides a GPU allocator that manages exclusive access
to GPU devices across the execution engine. It ensures:
- No GPU contention between nodes
- Deterministic device assignment
- Clean release on completion

The GPUPool complements the GPUSemaphore by providing a simpler
leasing model suitable for the process executor.

Example:
    >>> pool = GPUPool(devices=[0, 1])
    >>>
    >>> lease = pool.acquire()
    >>> print(f"Acquired device: {lease.device_id}")
    >>>
    >>> # Use device...
    >>> os.environ["CUDA_VISIBLE_DEVICES"] = str(lease.device_id)
    >>>
    >>> pool.release(lease)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPULease:
    """A lease for exclusive GPU access.

    Attributes:
        device_id: CUDA device ID
        lease_id: Unique identifier for this lease
        acquired_at: Timestamp when lease was acquired
    """

    device_id: int
    lease_id: int
    acquired_at: float

    @property
    def device_string(self) -> str:
        """PyTorch device string."""
        return f"cuda:{self.device_id}"


class GPUPoolError(RuntimeError):
    """Raised for GPU pool errors."""


class GPUPool:
    """GPU allocator with deterministic leasing.

    Manages a pool of GPU devices and provides exclusive leases.
    Thread-safe for concurrent acquisition.

    Example:
        >>> pool = GPUPool(devices=[0, 1])
        >>>
        >>> # Blocking acquire
        >>> lease = pool.acquire()
        >>> try:
        ...     run_on_gpu(lease.device_id)
        ... finally:
        ...     pool.release(lease)
        >>>
        >>> # Context manager
        >>> with pool.lease() as gpu:
        ...     run_on_gpu(gpu.device_id)
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        *,
        auto_detect: bool = True,
    ) -> None:
        """Initialize GPU pool.

        Args:
            devices: List of GPU device IDs to manage
            auto_detect: If True and devices not provided, detect GPUs
        """
        if devices is not None:
            self._devices = list(devices)
        elif auto_detect:
            self._devices = self._detect_devices()
        else:
            self._devices = []

        self._available = list(self._devices)
        self._leased: dict[int, GPULease] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._lease_counter = 0

        logger.info("GPUPool initialized with devices: %s", self._devices)

    @staticmethod
    def _detect_devices() -> List[int]:
        """Auto-detect available CUDA devices."""
        try:
            import torch

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except ImportError:
            pass

        import os

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
            except ValueError:
                pass

        return []

    @property
    def total_devices(self) -> int:
        """Total number of managed devices."""
        return len(self._devices)

    @property
    def available_count(self) -> int:
        """Number of currently available devices."""
        with self._lock:
            return len(self._available)

    @property
    def leased_count(self) -> int:
        """Number of currently leased devices."""
        with self._lock:
            return len(self._leased)

    def acquire(
        self,
        *,
        timeout: Optional[float] = None,
        block: bool = True,
    ) -> GPULease:
        """Acquire exclusive access to a GPU device.

        Args:
            timeout: Maximum time to wait in seconds
            block: If False, raise immediately if no GPU available

        Returns:
            GPULease with assigned device

        Raises:
            GPUPoolError: If no GPU available and not blocking/timeout
        """
        deadline = time.time() + timeout if timeout else None

        with self._condition:
            while not self._available:
                if not block:
                    raise GPUPoolError("No GPU available (non-blocking)")

                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise GPUPoolError(f"No GPU available (timeout={timeout}s)")
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()

            # Get first available device
            device_id = self._available.pop(0)
            self._lease_counter += 1

            lease = GPULease(
                device_id=device_id,
                lease_id=self._lease_counter,
                acquired_at=time.time(),
            )

            self._leased[lease.lease_id] = lease

            logger.debug(
                "Acquired GPU lease: device=%d, lease_id=%d",
                device_id,
                lease.lease_id,
            )

            return lease

    def try_acquire(self) -> Optional[GPULease]:
        """Try to acquire a GPU without blocking.

        Returns:
            GPULease if available, None otherwise
        """
        try:
            return self.acquire(block=False)
        except GPUPoolError:
            return None

    def release(self, lease: GPULease) -> None:
        """Release a GPU lease.

        Args:
            lease: Lease to release

        Raises:
            GPUPoolError: If lease is invalid
        """
        with self._condition:
            if lease.lease_id not in self._leased:
                raise GPUPoolError(f"Invalid lease: {lease.lease_id}")

            del self._leased[lease.lease_id]
            self._available.append(lease.device_id)

            logger.debug(
                "Released GPU lease: device=%d, lease_id=%d, duration=%.2fs",
                lease.device_id,
                lease.lease_id,
                time.time() - lease.acquired_at,
            )

            # Notify waiting threads
            self._condition.notify()

    def lease(
        self,
        *,
        timeout: Optional[float] = None,
    ):
        """Context manager for GPU lease.

        Args:
            timeout: Maximum time to wait for GPU

        Yields:
            GPULease

        Example:
            >>> with pool.lease() as gpu:
            ...     run_on_device(gpu.device_id)
        """
        from contextlib import contextmanager

        @contextmanager
        def _lease_context():
            gpu_lease = self.acquire(timeout=timeout)
            try:
                yield gpu_lease
            finally:
                self.release(gpu_lease)

        return _lease_context()

    def get_stats(self) -> dict:
        """Get pool statistics.

        Returns:
            Dictionary with pool state
        """
        with self._lock:
            return {
                "total_devices": len(self._devices),
                "available": len(self._available),
                "leased": len(self._leased),
                "total_leases_issued": self._lease_counter,
                "devices": list(self._devices),
            }
