"""GPU semaphore for multiprocess-safe GPU access.

This module provides a global GPU lease controller that:
- Enforces per-process GPU access
- Works with multiprocessing.spawn
- Prevents VRAM contention and fragmentation

Design:
    The semaphore uses a managed queue where each GPU device ID is a token.
    Processes acquire a token (blocking if none available), use the GPU,
    then release the token when done.

Example:
    >>> semaphore = GPUSemaphore(num_devices=2)
    >>> with semaphore.acquire() as slot:
    ...     model = load_model(device=f"cuda:{slot.device_id}")
    ...     result = model(inputs)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUSlot:
    """A leased GPU slot.

    Attributes:
        device_id: CUDA device ID (0, 1, 2, ...)
    """

    device_id: int

    @property
    def device_string(self) -> str:
        """Return PyTorch device string."""
        return f"cuda:{self.device_id}"


class GPUSemaphoreError(RuntimeError):
    """Raised for GPU semaphore errors."""


class GPUSemaphore:
    """Multiprocess-safe GPU semaphore.

    Manages exclusive access to GPU devices across multiple processes.
    Uses a managed queue where each GPU device ID is a token that
    processes can acquire and release.

    Example:
        >>> # Single GPU system
        >>> sem = GPUSemaphore(num_devices=1)
        >>>
        >>> def worker():
        ...     with sem.acquire() as slot:
        ...         # Exclusive GPU access
        ...         run_inference(device=slot.device_string)
        >>>
        >>> # Multi-GPU system with 2 parallel workers
        >>> sem = GPUSemaphore(num_devices=2)
        >>> with ThreadPoolExecutor(max_workers=2) as pool:
        ...     futures = [pool.submit(worker) for _ in range(4)]
    """

    def __init__(
        self,
        num_devices: Optional[int] = None,
        *,
        device_ids: Optional[list[int]] = None,
    ) -> None:
        """Initialize GPU semaphore.

        Args:
            num_devices: Number of GPU devices (0 to num_devices-1).
                        If None, auto-detects available GPUs.
            device_ids: Explicit list of device IDs to manage.
                       If provided, num_devices is ignored.

        Raises:
            GPUSemaphoreError: If no GPUs available and none specified
        """
        if device_ids is not None:
            self._device_ids = list(device_ids)
        elif num_devices is not None:
            self._device_ids = list(range(num_devices))
        else:
            self._device_ids = self._detect_devices()

        if not self._device_ids:
            raise GPUSemaphoreError(
                "No GPU devices available or specified. " "Use num_devices=1 to create a single-device semaphore."
            )

        self._manager = mp.Manager()
        self._queue = self._manager.Queue()

        # Populate queue with available device IDs
        for device_id in self._device_ids:
            self._queue.put(device_id)

        logger.info(
            "GPUSemaphore initialized with %d devices: %s",
            len(self._device_ids),
            self._device_ids,
        )

    @staticmethod
    def _detect_devices() -> list[int]:
        """Auto-detect available CUDA devices."""
        try:
            import torch

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except ImportError:
            pass

        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
            except ValueError as exc:
                logger.warning(
                    "Failed to parse CUDA_VISIBLE_DEVICES='%s': %s",
                    cuda_visible,
                    exc,
                )

        return []

    @property
    def num_devices(self) -> int:
        """Number of managed GPU devices."""
        return len(self._device_ids)

    @property
    def device_ids(self) -> list[int]:
        """List of managed device IDs."""
        return list(self._device_ids)

    @contextmanager
    def acquire(
        self,
        timeout: Optional[float] = None,
    ) -> Generator[GPUSlot, None, None]:
        """Acquire exclusive access to a GPU device.

        Blocks until a GPU slot is available. The slot is automatically
        released when the context manager exits.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Yields:
            GPUSlot with the assigned device ID

        Raises:
            GPUSemaphoreError: If timeout expires

        Example:
            >>> with semaphore.acquire() as slot:
            ...     torch.cuda.set_device(slot.device_id)
            ...     model = model.to(slot.device_string)
            ...     result = model(inputs)
        """
        try:
            device_id = self._queue.get(timeout=timeout)
        except Exception as exc:
            raise GPUSemaphoreError(f"Failed to acquire GPU slot (timeout={timeout}s): {exc}") from exc

        slot = GPUSlot(device_id=device_id)
        logger.debug("Acquired GPU slot: device_id=%d", device_id)

        try:
            yield slot
        finally:
            self._queue.put(device_id)
            logger.debug("Released GPU slot: device_id=%d", device_id)

    def try_acquire(self) -> Optional[GPUSlot]:
        """Try to acquire a GPU slot without blocking.

        Returns:
            GPUSlot if available, None otherwise

        Note:
            Unlike acquire(), this does not automatically release the slot.
            Call release() when done.
        """
        try:
            device_id = self._queue.get_nowait()
            return GPUSlot(device_id=device_id)
        except Exception:
            return None

    def release(self, slot: GPUSlot) -> None:
        """Release a manually acquired GPU slot.

        Args:
            slot: GPUSlot to release (from try_acquire)
        """
        self._queue.put(slot.device_id)
        logger.debug("Released GPU slot: device_id=%d", slot.device_id)
