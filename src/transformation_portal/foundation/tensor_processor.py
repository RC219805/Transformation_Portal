"""
Tensor Processor for Advanced Tensor Operations

Provides optimized tensor processing capabilities leveraging Apple Silicon M4 Max
architecture with Metal Performance Shaders and unified memory.

Key Features:
- Hardware-accelerated tensor operations
- Automatic precision management (FP32/FP16/BF16)
- Batch processing with optimal memory utilization
- Gradient checkpointing for memory efficiency
- Custom SIMD-optimized kernels for common operations
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Callable
from enum import Enum
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes for tensor operations."""
    FP32 = "fp32"  # Full precision (32-bit float)
    FP16 = "fp16"  # Half precision (16-bit float)
    BF16 = "bf16"  # Brain float 16
    TF32 = "tf32"  # TensorFloat-32 (NVIDIA)
    FP8 = "fp8"    # 8-bit float (future)


@dataclass
class TensorConfig:
    """Configuration for tensor operations."""
    precision: PrecisionMode = PrecisionMode.FP16
    device: Union[str, torch.device] = "mps"
    enable_amp: bool = True  # Automatic Mixed Precision
    enable_grad_checkpointing: bool = False
    max_batch_size: int = 32
    memory_efficient: bool = True
    enable_channels_last: bool = True  # Memory layout optimization
    compile_mode: Optional[str] = None  # torch.compile mode


class TensorProcessor:
    """
    Advanced tensor processor optimized for Apple Silicon M4 Max.

    Provides high-level tensor operations with automatic optimization,
    precision management, and memory efficiency features.
    """

    def __init__(
        self,
        config: Optional[TensorConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize tensor processor.

        Args:
            config: Tensor configuration
            device: Target device (overrides config if provided)
        """
        self.config = config or TensorConfig()
        self.device = device or torch.device(self.config.device)
        self._setup_precision()
        self._compile_cache = {}

        logger.info(f"Initialized TensorProcessor on {self.device} with {self.config.precision.value}")

    def _setup_precision(self):
        """Setup precision and mixed precision training."""
        if self.config.precision == PrecisionMode.TF32:
            # Enable TensorFloat-32 for NVIDIA GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.config.precision == PrecisionMode.FP16 and self.config.enable_amp:
            # Mixed precision will be handled via torch.autocast
            pass

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        fill_value: Optional[float] = None
    ) -> Tensor:
        """
        Allocate tensor with optimal memory layout.

        Args:
            shape: Tensor shape
            dtype: Data type (defaults to config precision)
            requires_grad: Whether tensor requires gradients
            fill_value: Optional fill value

        Returns:
            Allocated tensor
        """
        if dtype is None:
            dtype = self._get_dtype()

        # Allocate tensor
        if fill_value is not None:
            tensor = torch.full(shape, fill_value, dtype=dtype, device=self.device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)

        # Set gradient requirement
        if requires_grad:
            tensor.requires_grad_(True)

        # Optimize memory layout for image tensors (N, C, H, W)
        if len(shape) == 4 and self.config.enable_channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)

        return tensor

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> Tensor:
        """Allocate zero-initialized tensor."""
        return self.allocate(shape, dtype, fill_value=0.0)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> Tensor:
        """Allocate one-initialized tensor."""
        return self.allocate(shape, dtype, fill_value=1.0)

    def randn(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> Tensor:
        """Allocate random normal tensor."""
        dtype = dtype or self._get_dtype()
        tensor = torch.randn(shape, dtype=dtype, device=self.device)
        if len(shape) == 4 and self.config.enable_channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor

    def to_device(self, tensor: Tensor, non_blocking: bool = True) -> Tensor:
        """
        Move tensor to target device with optimal settings.

        Args:
            tensor: Input tensor
            non_blocking: Use asynchronous transfer (unified memory ignores this)

        Returns:
            Tensor on target device
        """
        if tensor.device == self.device:
            return tensor

        # For unified memory (MPS), non_blocking is automatic
        result = tensor.to(self.device, non_blocking=non_blocking)

        # Optimize memory layout
        if len(tensor.shape) == 4 and self.config.enable_channels_last:
            result = result.to(memory_format=torch.channels_last)

        return result

    def to_precision(self, tensor: Tensor, precision: Optional[PrecisionMode] = None) -> Tensor:
        """
        Convert tensor to target precision.

        Args:
            tensor: Input tensor
            precision: Target precision (defaults to config)

        Returns:
            Tensor in target precision
        """
        precision = precision or self.config.precision
        target_dtype = self._get_dtype(precision)

        if tensor.dtype == target_dtype:
            return tensor

        return tensor.to(dtype=target_dtype)

    def batch_process(
        self,
        tensors: List[Tensor],
        operation: Callable[[Tensor], Tensor],
        batch_size: Optional[int] = None
    ) -> List[Tensor]:
        """
        Process tensors in batches for memory efficiency.

        Args:
            tensors: List of input tensors
            operation: Operation to apply to each tensor
            batch_size: Batch size (defaults to config)

        Returns:
            List of processed tensors
        """
        batch_size = batch_size or self.config.max_batch_size
        results = []

        with torch.inference_mode():
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i:i + batch_size]

                # Stack batch if all tensors have same shape
                if all(t.shape == batch[0].shape for t in batch):
                    batch_tensor = torch.stack(batch)
                    batch_result = operation(batch_tensor)
                    results.extend(list(batch_result))
                else:
                    # Process individually if shapes differ
                    for tensor in batch:
                        results.append(operation(tensor))

        return results

    def normalize(
        self,
        tensor: Tensor,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        inplace: bool = False
    ) -> Tensor:
        """
        Normalize tensor with mean and standard deviation.

        Args:
            tensor: Input tensor (N, C, H, W) or (C, H, W)
            mean: Mean values per channel (defaults to ImageNet)
            std: Std values per channel (defaults to ImageNet)
            inplace: Perform operation in-place

        Returns:
            Normalized tensor
        """
        # Default to ImageNet normalization
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        mean_tensor = torch.tensor(mean, device=self.device, dtype=tensor.dtype)
        std_tensor = torch.tensor(std, device=self.device, dtype=tensor.dtype)

        # Reshape for broadcasting
        if tensor.ndim == 4:  # (N, C, H, W)
            mean_tensor = mean_tensor.view(1, -1, 1, 1)
            std_tensor = std_tensor.view(1, -1, 1, 1)
        elif tensor.ndim == 3:  # (C, H, W)
            mean_tensor = mean_tensor.view(-1, 1, 1)
            std_tensor = std_tensor.view(-1, 1, 1)

        if inplace:
            tensor.sub_(mean_tensor).div_(std_tensor)
            return tensor
        else:
            return (tensor - mean_tensor) / std_tensor

    def denormalize(
        self,
        tensor: Tensor,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> Tensor:
        """
        Denormalize tensor (inverse of normalize).

        Args:
            tensor: Normalized tensor
            mean: Mean values used for normalization
            std: Std values used for normalization

        Returns:
            Denormalized tensor
        """
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        mean_tensor = torch.tensor(mean, device=self.device, dtype=tensor.dtype)
        std_tensor = torch.tensor(std, device=self.device, dtype=tensor.dtype)

        if tensor.ndim == 4:
            mean_tensor = mean_tensor.view(1, -1, 1, 1)
            std_tensor = std_tensor.view(1, -1, 1, 1)
        elif tensor.ndim == 3:
            mean_tensor = mean_tensor.view(-1, 1, 1)
            std_tensor = std_tensor.view(-1, 1, 1)

        return tensor * std_tensor + mean_tensor

    def resize(
        self,
        tensor: Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear",
        align_corners: bool = False
    ) -> Tensor:
        """
        Resize tensor to target size.

        Args:
            tensor: Input tensor (N, C, H, W) or (C, H, W)
            size: Target size (H, W)
            mode: Interpolation mode
            align_corners: Whether to align corners

        Returns:
            Resized tensor
        """
        # Add batch dimension if needed
        squeeze_batch = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_batch = True

        # Resize
        resized = F.interpolate(
            tensor,
            size=size,
            mode=mode,
            align_corners=align_corners if mode != "nearest" else None
        )

        # Remove batch dimension if added
        if squeeze_batch:
            resized = resized.squeeze(0)

        return resized

    def pad(
        self,
        tensor: Tensor,
        padding: Union[int, Tuple[int, ...]],
        mode: str = "constant",
        value: float = 0.0
    ) -> Tensor:
        """
        Pad tensor with specified padding.

        Args:
            tensor: Input tensor
            padding: Padding specification
            mode: Padding mode (constant, reflect, replicate, circular)
            value: Fill value for constant padding

        Returns:
            Padded tensor
        """
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)

        return F.pad(tensor, padding, mode=mode, value=value)

    def gradient_checkpoint(
        self,
        function: Callable,
        *inputs: Tensor,
        use_reentrant: bool = True
    ) -> Tensor:
        """
        Apply gradient checkpointing to save memory during backprop.

        Args:
            function: Function to checkpoint
            inputs: Input tensors
            use_reentrant: Whether to use reentrant checkpointing

        Returns:
            Function output
        """
        if self.config.enable_grad_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                function, *inputs, use_reentrant=use_reentrant
            )
        else:
            return function(*inputs)

    def autocast_context(self):
        """
        Get autocast context for mixed precision operations.

        Returns:
            Context manager for automatic mixed precision
        """
        if self.config.enable_amp:
            if self.device.type == "cuda":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            elif self.device.type == "mps":
                # MPS autocast support (PyTorch 2.0+)
                return torch.autocast(device_type="mps", dtype=torch.float16)
            else:
                return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            # No-op context
            from contextlib import nullcontext
            return nullcontext()

    def compile_function(
        self,
        function: Callable,
        mode: Optional[str] = None,
        dynamic: bool = False
    ) -> Callable:
        """
        Compile function with torch.compile for optimization.

        Args:
            function: Function to compile
            mode: Compilation mode (default, reduce-overhead, max-autotune)
            dynamic: Whether to enable dynamic shapes

        Returns:
            Compiled function
        """
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available, returning uncompiled function")
            return function

        # Use config mode if not specified
        mode = mode or self.config.compile_mode
        if mode is None:
            return function

        # Check cache
        cache_key = (id(function), mode, dynamic)
        if cache_key in self._compile_cache:
            return self._compile_cache[cache_key]

        # Compile function
        try:
            compiled = torch.compile(function, mode=mode, dynamic=dynamic)
            self._compile_cache[cache_key] = compiled
            logger.info(f"Compiled function with mode={mode}, dynamic={dynamic}")
            return compiled
        except Exception as e:
            logger.warning(f"Failed to compile function: {e}")
            return function

    def _get_dtype(self, precision: Optional[PrecisionMode] = None) -> torch.dtype:
        """Get torch dtype for precision mode."""
        precision = precision or self.config.precision

        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.TF32: torch.float32,  # TF32 uses float32 with reduced precision
        }

        return dtype_map.get(precision, torch.float32)

    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics for the device.

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        if self.device.type == "cuda":
            stats["allocated"] = torch.cuda.memory_allocated(self.device) / (1024**3)
            stats["reserved"] = torch.cuda.memory_reserved(self.device) / (1024**3)
            stats["max_allocated"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        elif self.device.type == "mps":
            # MPS memory stats (limited support)
            stats["device"] = "mps"
            stats["unified_memory"] = True
            # Note: MPS doesn't expose detailed memory stats like CUDA
        else:
            stats["device"] = "cpu"

        return stats

    def clear_cache(self):
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # MPS uses unified memory, cache clearing is less relevant
            # but we can trigger garbage collection
            import gc
            gc.collect()

    def synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            # MPS synchronization
            torch.mps.synchronize()

    def __repr__(self) -> str:
        return (
            f"TensorProcessor(device={self.device}, "
            f"precision={self.config.precision.value}, "
            f"amp={self.config.enable_amp})"
        )
