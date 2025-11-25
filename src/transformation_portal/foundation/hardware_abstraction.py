"""
Hardware Abstraction Layer

Provides a unified interface for tensor operations across different hardware
backends (MPS, CUDA, CoreML, CPU) with automatic optimization and fallback.

Key Features:
- Backend-agnostic tensor operations
- Automatic backend selection and fallback
- Operation compatibility checking
- Performance profiling per backend
- Seamless model migration between backends
"""

from enum import Enum
from typing import Optional, List, Callable, Any, Dict
from dataclasses import dataclass
import logging
import functools

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported hardware backends."""
    MPS = "mps"  # Metal Performance Shaders
    CUDA = "cuda"  # NVIDIA CUDA
    COREML = "coreml"  # Apple Neural Engine via CoreML
    CPU = "cpu"  # CPU fallback


@dataclass
class BackendCapability:
    """Capabilities of a specific backend."""
    backend: BackendType
    available: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    max_tensor_size_gb: float
    recommended_for_inference: bool
    recommended_for_training: bool
    special_features: List[str]


class BackendRegistry:
    """Registry of available backends and their capabilities."""

    def __init__(self):
        self.backends: Dict[BackendType, BackendCapability] = {}
        self._detect_backends()

    def _detect_backends(self):
        """Detect available backends and their capabilities."""
        # MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            self.backends[BackendType.MPS] = BackendCapability(
                backend=BackendType.MPS,
                available=True,
                supports_fp16=True,
                supports_bf16=True,
                supports_int8=True,
                max_tensor_size_gb=100.0,  # Unified memory allows large tensors
                recommended_for_inference=True,
                recommended_for_training=True,
                special_features=["unified_memory", "metal_simd", "neural_engine_access"]
            )
            logger.info("✓ MPS backend available")

        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            self.backends[BackendType.CUDA] = BackendCapability(
                backend=BackendType.CUDA,
                available=True,
                supports_fp16=True,
                supports_bf16=torch.cuda.is_bf16_supported(),
                supports_int8=True,
                max_tensor_size_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3),
                recommended_for_inference=True,
                recommended_for_training=True,
                special_features=["tensor_cores", "flash_attention", "cudnn"]
            )
            logger.info("✓ CUDA backend available")

        # CoreML (Apple Neural Engine)
        try:
            import coremltools
            self.backends[BackendType.COREML] = BackendCapability(
                backend=BackendType.COREML,
                available=True,
                supports_fp16=True,
                supports_bf16=False,
                supports_int8=True,
                max_tensor_size_gb=10.0,  # ANE has limitations
                recommended_for_inference=True,
                recommended_for_training=False,
                special_features=["neural_engine", "low_power", "optimized_inference"]
            )
            logger.info("✓ CoreML backend available")
        except ImportError:
            logger.debug("CoreML not available")

        # CPU (always available)
        self.backends[BackendType.CPU] = BackendCapability(
            backend=BackendType.CPU,
            available=True,
            supports_fp16=False,
            supports_bf16=True,
            supports_int8=True,
            max_tensor_size_gb=1000.0,  # Limited by system RAM
            recommended_for_inference=False,
            recommended_for_training=False,
            special_features=["universal_compatibility"]
        )

    def get_capability(self, backend: BackendType) -> Optional[BackendCapability]:
        """Get capability info for a backend."""
        return self.backends.get(backend)

    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends."""
        return [b for b, cap in self.backends.items() if cap.available]

    def get_optimal_backend(
        self,
        for_inference: bool = True,
        prefer_performance: bool = True
    ) -> BackendType:
        """
        Get optimal backend for the task.

        Args:
            for_inference: Whether for inference (vs training)
            prefer_performance: Prefer performance over power efficiency

        Returns:
            Optimal backend type
        """
        available = self.get_available_backends()

        if not available:
            return BackendType.CPU

        # Priority for inference with performance preference
        if for_inference and prefer_performance:
            priority = [BackendType.MPS, BackendType.CUDA, BackendType.COREML, BackendType.CPU]
        # Priority for inference with power efficiency
        elif for_inference and not prefer_performance:
            priority = [BackendType.COREML, BackendType.MPS, BackendType.CUDA, BackendType.CPU]
        # Priority for training
        else:
            priority = [BackendType.MPS, BackendType.CUDA, BackendType.CPU]

        for backend in priority:
            if backend in available:
                capability = self.backends[backend]
                if for_inference and capability.recommended_for_inference:
                    return backend
                elif not for_inference and capability.recommended_for_training:
                    return backend

        return available[0]


class HardwareAbstraction:
    """
    Hardware abstraction layer for unified tensor operations.

    Provides a consistent interface across different hardware backends,
    with automatic fallback and optimization.
    """

    def __init__(
        self,
        primary_backend: Optional[BackendType] = None,
        enable_auto_fallback: bool = True
    ):
        """
        Initialize hardware abstraction layer.

        Args:
            primary_backend: Primary backend to use (auto-detected if None)
            enable_auto_fallback: Enable automatic fallback on errors
        """
        self.registry = BackendRegistry()
        self.enable_auto_fallback = enable_auto_fallback

        # Determine primary backend
        if primary_backend is None:
            self.primary_backend = self.registry.get_optimal_backend()
        else:
            self.primary_backend = primary_backend

        # Create fallback chain
        self.fallback_chain = self._create_fallback_chain()

        # Get primary device
        self.primary_device = self._backend_to_device(self.primary_backend)

        logger.info(f"Hardware abstraction initialized with primary backend: {self.primary_backend.value}")
        logger.info(f"Fallback chain: {' -> '.join(b.value for b in self.fallback_chain)}")

    def _create_fallback_chain(self) -> List[BackendType]:
        """Create backend fallback chain."""
        available = self.registry.get_available_backends()
        chain = [self.primary_backend]

        # Add remaining backends in priority order
        priority = [BackendType.MPS, BackendType.CUDA, BackendType.COREML, BackendType.CPU]
        for backend in priority:
            if backend in available and backend not in chain:
                chain.append(backend)

        return chain

    def _backend_to_device(self, backend: BackendType) -> torch.device:
        """Convert backend type to torch device."""
        if backend == BackendType.MPS:
            return torch.device("mps")
        elif backend == BackendType.CUDA:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def execute_with_fallback(
        self,
        operation: Callable,
        *args,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic fallback on failure.

        Args:
            operation: Operation to execute
            *args: Positional arguments
            operation_name: Name for logging
            **kwargs: Keyword arguments

        Returns:
            Operation result

        Raises:
            RuntimeError: If operation fails on all backends
        """
        errors = []

        for backend in self.fallback_chain:
            try:
                device = self._backend_to_device(backend)

                # Move tensors to target device
                args_on_device = self._move_to_device(args, device)
                kwargs_on_device = self._move_to_device(kwargs, device)

                # Execute operation
                result = operation(*args_on_device, **kwargs_on_device)

                # Log if we used fallback
                if backend != self.primary_backend:
                    logger.warning(
                        f"{operation_name} failed on {self.primary_backend.value}, "
                        f"succeeded with fallback to {backend.value}"
                    )

                return result

            except Exception as e:
                errors.append((backend, str(e)))
                logger.debug(f"{operation_name} failed on {backend.value}: {e}")

                if not self.enable_auto_fallback:
                    raise

        # All backends failed
        error_msg = f"{operation_name} failed on all backends:\n"
        for backend, error in errors:
            error_msg += f"  {backend.value}: {error}\n"
        raise RuntimeError(error_msg)

    def _move_to_device(self, obj: Any, device: torch.device) -> Any:
        """Recursively move tensors/modules to device."""
        if isinstance(obj, Tensor):
            return obj.to(device)
        elif isinstance(obj, nn.Module):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            moved = [self._move_to_device(item, device) for item in obj]
            return type(obj)(moved)
        else:
            return obj

    def to_device(self, obj: Any, backend: Optional[BackendType] = None) -> Any:
        """
        Move object to target backend device.

        Args:
            obj: Tensor, module, or nested structure
            backend: Target backend (uses primary if None)

        Returns:
            Object on target device
        """
        backend = backend or self.primary_backend
        device = self._backend_to_device(backend)
        return self._move_to_device(obj, device)

    def get_device(self, backend: Optional[BackendType] = None) -> torch.device:
        """Get device for backend."""
        backend = backend or self.primary_backend
        return self._backend_to_device(backend)

    def supports_operation(
        self,
        operation_name: str,
        backend: Optional[BackendType] = None
    ) -> bool:
        """
        Check if backend supports specific operation.

        Args:
            operation_name: Name of operation to check
            backend: Backend to check (uses primary if None)

        Returns:
            True if operation is supported
        """
        backend = backend or self.primary_backend

        # Define operation support matrix
        unsupported_ops = {
            BackendType.MPS: [
                # Some ops that have limited MPS support
                "torch.sparse",
                "torch.linalg.svd",  # Limited support in older PyTorch
            ],
            BackendType.COREML: [
                # CoreML has many limitations for training ops
                "torch.autograd",
                "backward",
                "optimizer",
            ],
            BackendType.CPU: [],  # CPU supports all ops
            BackendType.CUDA: [],  # CUDA supports all ops
        }

        unsupported = unsupported_ops.get(backend, [])
        return not any(op in operation_name for op in unsupported)

    def benchmark_operation(
        self,
        operation: Callable,
        *args,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> Dict[BackendType, float]:
        """
        Benchmark operation across available backends.

        Args:
            operation: Operation to benchmark
            *args: Positional arguments
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            **kwargs: Keyword arguments

        Returns:
            Dictionary mapping backend to average execution time (seconds)
        """
        import time

        results = {}

        for backend in self.registry.get_available_backends():
            try:
                device = self._backend_to_device(backend)

                # Move data to device
                args_on_device = self._move_to_device(args, device)
                kwargs_on_device = self._move_to_device(kwargs, device)

                # Warmup
                for _ in range(warmup_iterations):
                    operation(*args_on_device, **kwargs_on_device)

                # Synchronize before benchmark
                if backend == BackendType.CUDA:
                    torch.cuda.synchronize()
                elif backend == BackendType.MPS:
                    torch.mps.synchronize()

                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    operation(*args_on_device, **kwargs_on_device)

                # Synchronize after benchmark
                if backend == BackendType.CUDA:
                    torch.cuda.synchronize()
                elif backend == BackendType.MPS:
                    torch.mps.synchronize()

                elapsed = time.time() - start_time
                avg_time = elapsed / num_iterations
                results[backend] = avg_time

                logger.info(f"Benchmark {backend.value}: {avg_time*1000:.3f}ms per iteration")

            except Exception as e:
                logger.warning(f"Benchmark failed on {backend.value}: {e}")
                results[backend] = float('inf')

        return results

    def with_fallback(self, operation_name: str = "operation"):
        """
        Decorator for automatic fallback support.

        Usage:
            @hal.with_fallback("my_operation")
            def my_operation(x):
                return x * 2
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_fallback(
                    func, *args,
                    operation_name=operation_name,
                    **kwargs
                )
            return wrapper
        return decorator

    def get_optimal_dtype(
        self,
        backend: Optional[BackendType] = None,
        prefer_speed: bool = True
    ) -> torch.dtype:
        """
        Get optimal data type for backend.

        Args:
            backend: Target backend
            prefer_speed: Prefer speed over precision

        Returns:
            Optimal torch dtype
        """
        backend = backend or self.primary_backend
        capability = self.registry.get_capability(backend)

        if capability is None:
            return torch.float32

        if prefer_speed:
            if capability.supports_fp16:
                return torch.float16
            elif capability.supports_bf16:
                return torch.bfloat16
        else:
            if capability.supports_bf16:
                return torch.bfloat16
            elif capability.supports_fp16:
                return torch.float16

        return torch.float32

    def __repr__(self) -> str:
        available = self.registry.get_available_backends()
        return (
            f"HardwareAbstraction(primary={self.primary_backend.value}, "
            f"available={[b.value for b in available]})"
        )
