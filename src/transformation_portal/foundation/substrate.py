"""
Computational Substrate - Phase 1 Foundation Architecture

The core computational substrate that integrates all foundation components
into a unified interface for optimal tensor processing on Apple Silicon M4 Max.

This is the primary entry point for Phase 1 foundation capabilities.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

import torch
from torch import Tensor

from .device_manager import DeviceManager, DeviceInfo
from .tensor_processor import TensorProcessor, TensorConfig, PrecisionMode
from .memory_manager import MemoryManager, MemoryConfig, AllocationStrategy
from .hardware_abstraction import HardwareAbstraction, BackendType
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class SubstrateConfig:
    """Configuration for computational substrate."""
    # Device configuration
    prefer_ane: bool = True
    memory_fraction: float = 0.85

    # Tensor processing
    precision: PrecisionMode = PrecisionMode.FP16
    enable_amp: bool = True
    enable_grad_checkpointing: bool = False
    enable_channels_last: bool = True
    compile_mode: Optional[str] = None

    # Memory management
    allocation_strategy: AllocationStrategy = AllocationStrategy.POOLED
    max_memory_gb: float = 100.0
    pool_size_mb: int = 1024

    # Hardware abstraction
    enable_auto_fallback: bool = True

    # Performance monitoring
    enable_profiling: bool = False

    @classmethod
    def for_m4_max(cls) -> "SubstrateConfig":
        """Create optimized configuration for M4 Max."""
        return cls(
            prefer_ane=True,
            memory_fraction=0.85,  # Use 85% of 128GB
            precision=PrecisionMode.FP16,  # Best for MPS
            enable_amp=True,
            enable_channels_last=True,
            allocation_strategy=AllocationStrategy.POOLED,
            max_memory_gb=108.0,  # 85% of 128GB
            pool_size_mb=2048,  # Larger pools for unified memory
            enable_auto_fallback=True,
            enable_profiling=False,
        )

    @classmethod
    def for_development(cls) -> "SubstrateConfig":
        """Create configuration for development with profiling."""
        config = cls.for_m4_max()
        config.enable_profiling = True
        config.compile_mode = None  # Disable compilation for debugging
        return config

    @classmethod
    def for_production(cls) -> "SubstrateConfig":
        """Create configuration for production deployment."""
        config = cls.for_m4_max()
        config.enable_profiling = False
        config.compile_mode = "reduce-overhead"  # Optimize for throughput
        return config


class ComputationalSubstrate:
    """
    Computational Substrate - Phase 1 Foundation Architecture

    The core computational layer that provides:
    - Optimal device detection and configuration for M4 Max
    - Advanced tensor processing with hardware acceleration
    - Intelligent memory management for unified memory architecture
    - Hardware abstraction with automatic fallback
    - Real-time performance monitoring and profiling

    This substrate serves as the foundation for all subsequent phases,
    ensuring optimal performance before any model loading occurs.

    Usage:
        # Initialize with default M4 Max optimizations
        substrate = ComputationalSubstrate()

        # Or with custom configuration
        config = SubstrateConfig.for_production()
        substrate = ComputationalSubstrate(config)

        # Allocate tensors with optimal memory patterns
        tensor = substrate.allocate_tensor((1024, 1024, 3))

        # Process with hardware acceleration
        result = substrate.process(tensor, operation_fn)

        # Get performance insights
        print(substrate.get_performance_summary())
    """

    def __init__(self, config: Optional[SubstrateConfig] = None):
        """
        Initialize computational substrate.

        Args:
            config: Substrate configuration (defaults to M4 Max optimized)
        """
        self.config = config or SubstrateConfig.for_m4_max()

        logger.info("=" * 70)
        logger.info("Initializing Computational Substrate - Phase 1")
        logger.info("=" * 70)

        # Initialize device manager
        self.device_manager = DeviceManager(
            prefer_ane=self.config.prefer_ane,
            memory_fraction=self.config.memory_fraction
        )
        self.device_info = self.device_manager.detect_devices()
        self.device = self.device_info.primary_device

        # Initialize tensor processor
        tensor_config = TensorConfig(
            precision=self.config.precision,
            device=self.device,
            enable_amp=self.config.enable_amp,
            enable_grad_checkpointing=self.config.enable_grad_checkpointing,
            max_batch_size=self.device_info.capabilities.recommended_batch_size,
            enable_channels_last=self.config.enable_channels_last,
            compile_mode=self.config.compile_mode,
        )
        self.tensor_processor = TensorProcessor(tensor_config, self.device)

        # Initialize memory manager
        memory_config = MemoryConfig(
            strategy=self.config.allocation_strategy,
            max_memory_gb=self.config.max_memory_gb,
            pool_size_mb=self.config.pool_size_mb,
            enable_profiling=self.config.enable_profiling,
        )
        self.memory_manager = MemoryManager(memory_config, self.device)

        # Initialize hardware abstraction
        backend_type = self._device_to_backend(self.device)
        self.hardware_abstraction = HardwareAbstraction(
            primary_backend=backend_type,
            enable_auto_fallback=self.config.enable_auto_fallback
        )

        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            device=self.device,
            enable_memory_tracking=self.config.enable_profiling
        )

        if not self.config.enable_profiling:
            self.performance_monitor.disable()

        logger.info("✓ Computational substrate initialized successfully")
        logger.info("=" * 70)

        # Run validation
        self._validate_substrate()

    def _device_to_backend(self, device: torch.device) -> BackendType:
        """Convert torch device to backend type."""
        if device.type == "mps":
            return BackendType.MPS
        elif device.type == "cuda":
            return BackendType.CUDA
        else:
            return BackendType.CPU

    def _validate_substrate(self):
        """Validate substrate initialization."""
        logger.info("Running substrate validation...")

        # Test tensor allocation
        test_tensor = self.allocate_tensor((100, 100), dtype=torch.float32)
        assert test_tensor.device == self.device, "Tensor not on correct device"

        # Test computation
        result = test_tensor * 2.0
        assert result.device == self.device, "Computation moved tensor"

        # Clean up
        del test_tensor, result
        self.memory_manager.clear_cache()

        logger.info("✓ Substrate validation passed")

    # ========================================================================
    # High-Level Interface
    # ========================================================================

    def allocate_tensor(
        self,
        shape: tuple,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        tag: Optional[str] = None
    ) -> Tensor:
        """
        Allocate tensor with optimal memory patterns.

        Args:
            shape: Tensor shape
            dtype: Data type (defaults to substrate precision)
            requires_grad: Whether tensor requires gradients
            tag: Optional tag for tracking

        Returns:
            Allocated tensor on optimal device
        """
        with self.performance_monitor.profile_context("allocate_tensor"):
            # Allocate through memory manager
            tensor = self.memory_manager.allocate(
                shape=shape,
                dtype=dtype or self.tensor_processor._get_dtype(),
                tag=tag,
                use_pool=True
            )

            # Configure gradients
            if requires_grad:
                tensor.requires_grad_(True)

            return tensor

    def process_batch(
        self,
        tensors: list,
        operation,
        batch_size: Optional[int] = None
    ) -> list:
        """
        Process batch of tensors with optimal batching.

        Args:
            tensors: List of input tensors
            operation: Operation to apply
            batch_size: Batch size (defaults to optimal)

        Returns:
            List of processed tensors
        """
        with self.performance_monitor.profile_context("process_batch"):
            return self.tensor_processor.batch_process(
                tensors,
                operation,
                batch_size or self.device_info.capabilities.recommended_batch_size
            )

    def to_device(self, tensor_or_module):
        """
        Move tensor or module to optimal device.

        Args:
            tensor_or_module: Tensor or PyTorch module

        Returns:
            Object on optimal device
        """
        return self.hardware_abstraction.to_device(tensor_or_module)

    def execute_with_fallback(self, operation, *args, **kwargs):
        """
        Execute operation with automatic hardware fallback.

        Args:
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        return self.hardware_abstraction.execute_with_fallback(
            operation, *args,
            operation_name=getattr(operation, "__name__", "operation"),
            **kwargs
        )

    # ========================================================================
    # Information and Monitoring
    # ========================================================================

    def get_device(self) -> torch.device:
        """Get the optimal device."""
        return self.device

    def get_device_info(self) -> DeviceInfo:
        """Get complete device information."""
        return self.device_info

    def get_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities."""
        cap = self.device_info.capabilities
        return {
            "device_name": cap.device_name,
            "device_type": cap.device_type.value,
            "total_memory_gb": cap.total_memory_gb,
            "available_memory_gb": cap.available_memory_gb,
            "performance_cores": cap.performance_cores,
            "efficiency_cores": cap.efficiency_cores,
            "gpu_cores": cap.gpu_cores,
            "unified_memory": cap.unified_memory,
            "neural_engine": cap.neural_engine_available,
            "fp16_support": cap.supports_fp16,
            "bf16_support": cap.supports_bf16,
            "recommended_batch_size": cap.recommended_batch_size,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return self.memory_manager.get_memory_stats()

    def get_performance_summary(self) -> str:
        """Get human-readable performance summary."""
        return self.performance_monitor.get_summary()

    def get_status(self) -> Dict[str, Any]:
        """Get complete substrate status."""
        return {
            "device": str(self.device),
            "capabilities": self.get_capabilities(),
            "memory": self.get_memory_stats(),
            "configuration": {
                "precision": self.config.precision.value,
                "amp_enabled": self.config.enable_amp,
                "allocation_strategy": self.config.allocation_strategy.value,
                "profiling_enabled": self.config.enable_profiling,
            }
        }

    # ========================================================================
    # Optimization and Maintenance
    # ========================================================================

    def optimize_memory(self):
        """Optimize memory usage and clear caches."""
        self.memory_manager.optimize_memory()

    def clear_cache(self):
        """Clear all caches."""
        self.memory_manager.clear_cache()
        self.tensor_processor.clear_cache()

    def enable_profiling(self):
        """Enable performance profiling."""
        self.config.enable_profiling = True
        self.performance_monitor.enable()

    def disable_profiling(self):
        """Disable performance profiling."""
        self.config.enable_profiling = False
        self.performance_monitor.disable()

    def export_metrics(self, filepath: str):
        """
        Export performance metrics to file.

        Args:
            filepath: Path to output JSON file
        """
        self.performance_monitor.export_metrics(filepath)

    def benchmark_operation(self, operation, *args, **kwargs) -> Dict[str, float]:
        """
        Benchmark an operation on the substrate.

        Args:
            operation: Operation to benchmark
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Benchmark statistics
        """
        return self.performance_monitor.benchmark(
            operation, *args,
            operation_name=getattr(operation, "__name__", "operation"),
            **kwargs
        )

    # ========================================================================
    # Context Managers
    # ========================================================================

    def autocast(self):
        """
        Get autocast context for mixed precision.

        Usage:
            with substrate.autocast():
                output = model(input)
        """
        return self.tensor_processor.autocast_context()

    def profile(self, context_name: str):
        """
        Profile context for code blocks.

        Usage:
            with substrate.profile("data_loading"):
                data = load_data()
        """
        return self.performance_monitor.profile_context(context_name)

    # ========================================================================
    # Advanced Features
    # ========================================================================

    def compile_model(self, model: torch.nn.Module, mode: Optional[str] = None):
        """
        Compile model with torch.compile for optimization.

        Args:
            model: PyTorch model
            mode: Compilation mode (default, reduce-overhead, max-autotune)

        Returns:
            Compiled model
        """
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available, returning uncompiled model")
            return model

        mode = mode or self.config.compile_mode
        if mode is None:
            return model

        logger.info(f"Compiling model with mode={mode}")
        return torch.compile(model, mode=mode)

    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type for current configuration."""
        return self.tensor_processor.get_dtype()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def __repr__(self) -> str:
        cap = self.device_info.capabilities
        return (
            f"ComputationalSubstrate(\n"
            f"  device={self.device},\n"
            f"  device_name='{cap.device_name}',\n"
            f"  memory={cap.available_memory_gb:.1f}GB,\n"
            f"  precision={self.config.precision.value},\n"
            f"  batch_size={cap.recommended_batch_size}\n"
            f")"
        )

    def __str__(self) -> str:
        """Human-readable substrate information."""
        lines = [
            "=" * 70,
            "COMPUTATIONAL SUBSTRATE - PHASE 1",
            "=" * 70,
        ]

        # Device info
        cap = self.device_info.capabilities
        lines.extend([
            f"Device: {cap.device_name}",
            f"Type: {cap.device_type.value.upper()}",
            f"Memory: {cap.available_memory_gb:.1f} GB available / {cap.total_memory_gb:.1f} GB total",
            f"Cores: {cap.performance_cores}P + {cap.efficiency_cores}E (CPU), {cap.gpu_cores} GPU",
            f"Neural Engine: {'Available' if cap.neural_engine_available else 'Not Available'}",
        ])

        # Configuration
        lines.extend([
            "",
            "Configuration:",
            f"  Precision: {self.config.precision.value}",
            f"  Mixed Precision: {self.config.enable_amp}",
            f"  Memory Strategy: {self.config.allocation_strategy.value}",
            f"  Batch Size: {cap.recommended_batch_size}",
            f"  Profiling: {self.config.enable_profiling}",
        ])

        lines.append("=" * 70)
        return "\n".join(lines)
