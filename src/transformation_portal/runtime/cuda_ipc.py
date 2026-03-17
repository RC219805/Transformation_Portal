"""Zero-copy CUDA IPC utilities for efficient tensor sharing.

This module provides utilities for sharing tensors across processes
using CUDA IPC when possible, with safe fallback to CPU transfer.

Key features:
- Fast path: CUDA IPC handles (zero-copy)
- Fallback: CPU shared memory (safe but slower)
- Automatic degradation on unsupported systems

Note:
    True zero-copy CUDA IPC is fragile across PyTorch versions and
    requires specific CUDA driver support. This implementation
    provides a best-effort fast path with safe fallback.

Example:
    >>> tensor = torch.randn(1000, 1000, device="cuda")
    >>> payload = export_tensor(tensor)
    >>> # ... send payload to another process ...
    >>> reconstructed = import_tensor(payload)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CudaIPCError(RuntimeError):
    """Raised for CUDA IPC operation failures."""


@dataclass
class TensorPayload:
    """Serialized tensor payload for cross-process transfer.

    Attributes:
        shape: Tensor shape
        dtype: Tensor dtype as string
        device: Original device string
        data: Serialized data (IPC handle or numpy array)
        is_cuda_ipc: True if using CUDA IPC, False for CPU fallback
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    data: Any
    is_cuda_ipc: bool


def _try_cuda_ipc_export(tensor: "torch.Tensor") -> Optional[TensorPayload]:
    """Attempt CUDA IPC export (best-effort).

    Args:
        tensor: CUDA tensor to export

    Returns:
        TensorPayload if successful, None otherwise
    """
    try:
        import torch

        if not tensor.is_cuda:
            return None

        # Ensure tensor is contiguous
        tensor = tensor.contiguous()

        # Get storage for IPC handle
        storage = tensor.storage()

        # Try to get IPC handle (may fail on some systems)
        # Note: This is a simplified approach; full IPC requires
        # torch.multiprocessing.reductions for robust serialization
        ipc_handle = storage._share_cuda_()

        return TensorPayload(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            data=ipc_handle,
            is_cuda_ipc=True,
        )

    except Exception as exc:
        logger.debug("CUDA IPC export failed: %s", exc)
        return None


def _cpu_export(tensor: "torch.Tensor") -> TensorPayload:
    """Export tensor via CPU (safe fallback).

    Args:
        tensor: Tensor to export

    Returns:
        TensorPayload with numpy array data
    """
    import numpy as np

    # Move to CPU and convert to numpy
    arr = tensor.detach().cpu().numpy()

    return TensorPayload(
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        data=arr,
        is_cuda_ipc=False,
    )


def export_tensor(
    tensor: "torch.Tensor",
    *,
    prefer_cuda_ipc: bool = True,
) -> TensorPayload:
    """Export tensor for cross-process transfer.

    Attempts CUDA IPC for CUDA tensors (zero-copy), falls back
    to CPU transfer if IPC fails.

    Args:
        tensor: Tensor to export
        prefer_cuda_ipc: If True, try CUDA IPC first for CUDA tensors

    Returns:
        TensorPayload for import in another process

    Example:
        >>> tensor = torch.randn(1000, 1000, device="cuda")
        >>> payload = export_tensor(tensor)
        >>> print(f"Using CUDA IPC: {payload.is_cuda_ipc}")
    """
    import torch

    if prefer_cuda_ipc and tensor.is_cuda:
        ipc_payload = _try_cuda_ipc_export(tensor)
        if ipc_payload is not None:
            logger.debug("Exported tensor via CUDA IPC")
            return ipc_payload
        logger.debug("CUDA IPC export failed, falling back to CPU")

    return _cpu_export(tensor)


def _try_cuda_ipc_import(payload: TensorPayload) -> Optional["torch.Tensor"]:
    """Attempt CUDA IPC import (best-effort).

    Args:
        payload: TensorPayload with IPC handle

    Returns:
        Tensor if successful, None otherwise
    """
    try:
        import torch

        # Reconstruct from IPC handle
        storage = torch.cuda.storage._new_shared_cuda(*payload.data)
        tensor = torch.tensor([], dtype=eval(f"torch.{payload.dtype.split('.')[-1]}"))
        tensor.set_(storage, 0, payload.shape)

        return tensor

    except Exception as exc:
        logger.debug("CUDA IPC import failed: %s", exc)
        return None


def _cpu_import(
    payload: TensorPayload,
    target_device: Optional[str] = None,
) -> "torch.Tensor":
    """Import tensor from CPU data.

    Args:
        payload: TensorPayload with numpy array
        target_device: Device to place tensor on (default: original device)

    Returns:
        Reconstructed tensor
    """
    import torch

    tensor = torch.from_numpy(payload.data)

    # Restore dtype
    dtype_str = payload.dtype.split(".")[-1]
    target_dtype = getattr(torch, dtype_str, None)
    if target_dtype is not None:
        tensor = tensor.to(dtype=target_dtype)

    # Move to target device
    device = target_device or payload.device
    if device != "cpu" and "cuda" in device:
        try:
            tensor = tensor.to(device)
        except Exception as exc:
            logger.warning("Failed to move tensor to %s: %s", device, exc)

    return tensor


def import_tensor(
    payload: TensorPayload,
    *,
    target_device: Optional[str] = None,
) -> "torch.Tensor":
    """Import tensor from cross-process payload.

    Args:
        payload: TensorPayload from export_tensor
        target_device: Device to place tensor on (default: original device)

    Returns:
        Reconstructed tensor

    Example:
        >>> payload = receive_from_other_process()
        >>> tensor = import_tensor(payload)
        >>> tensor = import_tensor(payload, target_device="cuda:0")
    """
    if payload.is_cuda_ipc:
        tensor = _try_cuda_ipc_import(payload)
        if tensor is not None:
            # Move to target device if different
            if target_device and target_device != str(tensor.device):
                tensor = tensor.to(target_device)
            return tensor
        logger.debug("CUDA IPC import failed, falling back to CPU")

    return _cpu_import(payload, target_device)
