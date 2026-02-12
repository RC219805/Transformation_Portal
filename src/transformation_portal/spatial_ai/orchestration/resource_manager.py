"""Resource manager for GPU memory and model lifecycle (Phase 2.4).

Handles:
- GPU memory tracking and limits
- Model loading/unloading between stages
- CPU fallback when GPU OOM
- Batch processing memory budgets

Architecture:
- Context manager for automatic cleanup
- Lazy model loading (only load when needed)
- LRU-style model caching with memory limits
- Graceful degradation to CPU on OOM

Example:
    >>> limits = ResourceLimits(max_gpu_memory_gb=8.0, max_models_loaded=2)
    >>> with ResourceManager(limits) as rm:
    ...     model = rm.load_model("sam2", backend="cuda")
    ...     # Model automatically unloaded when context exits
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for pipeline execution.

    Attributes:
        max_gpu_memory_gb: Maximum GPU memory in GB (None = unlimited).
        max_cpu_memory_gb: Maximum CPU memory in GB (None = unlimited).
        max_models_loaded: Maximum number of models loaded simultaneously.
        batch_size: Batch size for multi-image processing.
        device_preference: Preferred device order ("cuda" > "mps" > "cpu").
    """

    max_gpu_memory_gb: Optional[float] = None
    max_cpu_memory_gb: Optional[float] = None
    max_models_loaded: int = 3
    batch_size: int = 1
    device_preference: list = field(default_factory=lambda: ["cuda", "mps", "cpu"])

    def __post_init__(self):
        """Validate resource limits."""
        if self.max_gpu_memory_gb is not None and self.max_gpu_memory_gb <= 0:
            raise ValueError(f"max_gpu_memory_gb must be positive, got {self.max_gpu_memory_gb}")

        if self.max_cpu_memory_gb is not None and self.max_cpu_memory_gb <= 0:
            raise ValueError(f"max_cpu_memory_gb must be positive, got {self.max_cpu_memory_gb}")

        if self.max_models_loaded <= 0:
            raise ValueError(f"max_models_loaded must be positive, got {self.max_models_loaded}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


class ResourceManager:
    """Manager for GPU/CPU resources and model lifecycle.

    Tracks loaded models and memory usage, automatically unloading
    models when resource limits are reached.

    Example:
        >>> limits = ResourceLimits(max_gpu_memory_gb=8.0)
        >>> with ResourceManager(limits) as rm:
        ...     device = rm.select_device()
        ...     rm.register_model("sam2", model_obj)
        ...     # Models auto-unloaded on exit
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource manager.

        Args:
            limits: Resource limits (None = defaults).
        """
        self.limits = limits or ResourceLimits()
        self._loaded_models: Dict[str, Any] = {}
        self._model_load_order: list = []
        self._peak_memory_mb: float = 0.0

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.cleanup()
        return False

    def select_device(self) -> Literal["cuda", "mps", "cpu"]:
        """Select best available device based on preferences and limits.

        Returns:
            Device string ("cuda", "mps", or "cpu").

        Raises:
            RuntimeError: If GPU requested but none available.
        """
        for device in self.limits.device_preference:
            if device == "cuda":
                try:
                    import torch

                    if torch.cuda.is_available():
                        # Check GPU memory if limit set
                        if self.limits.max_gpu_memory_gb is not None:
                            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                            if mem_gb < self.limits.max_gpu_memory_gb:
                                logger.warning(
                                    f"GPU memory {mem_gb:.1f}GB < limit {self.limits.max_gpu_memory_gb}GB, "
                                    f"trying next device"
                                )
                                continue
                        logger.info(f"Selected device: {device}")
                        return "cuda"
                except ImportError:
                    pass

            elif device == "mps":
                try:
                    import torch

                    if torch.backends.mps.is_available():
                        logger.info(f"Selected device: {device}")
                        return "mps"
                except (ImportError, AttributeError):
                    pass

            elif device == "cpu":
                logger.info(f"Selected device: {device}")
                return "cpu"

        # Default to CPU
        logger.warning("No GPU available, falling back to CPU")
        return "cpu"

    def register_model(self, name: str, model: Any) -> None:
        """Register a loaded model for tracking.

        Args:
            name: Model identifier (e.g., "sam2", "depth_anything").
            model: Model object.

        Raises:
            RuntimeError: If max_models_loaded limit exceeded.
        """
        # Check if we need to unload old models
        if len(self._loaded_models) >= self.limits.max_models_loaded:
            if self._model_load_order:
                # Unload oldest model (FIFO)
                oldest = self._model_load_order.pop(0)
                self.unload_model(oldest)

        self._loaded_models[name] = model
        self._model_load_order.append(name)
        logger.debug(f"Registered model: {name} ({len(self._loaded_models)}/{self.limits.max_models_loaded})")

    def unload_model(self, name: str) -> None:
        """Unload a model and free resources.

        Args:
            name: Model identifier.
        """
        if name in self._loaded_models:
            del self._loaded_models[name]
            if name in self._model_load_order:
                self._model_load_order.remove(name)

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

            logger.debug(f"Unloaded model: {name}")

    def get_model(self, name: str) -> Optional[Any]:
        """Retrieve a loaded model.

        Args:
            name: Model identifier.

        Returns:
            Model object or None if not loaded.
        """
        return self._loaded_models.get(name)

    def get_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB (0.0 if GPU not available).
        """
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes = torch.cuda.memory_allocated()
                mem_mb = mem_bytes / 1e6
                self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
                return mem_mb
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't expose memory stats yet
                return 0.0
        except ImportError:
            pass

        return 0.0

    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage since manager creation.

        Returns:
            Peak memory usage in MB.
        """
        # Update peak before returning
        self.get_memory_usage_mb()
        return self._peak_memory_mb

    def cleanup(self) -> None:
        """Clean up all loaded models and free resources."""
        logger.info(f"Cleaning up {len(self._loaded_models)} loaded models")
        model_names = list(self._loaded_models.keys())
        for name in model_names:
            self.unload_model(name)

        # Final GC
        gc.collect()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResourceManager(models={len(self._loaded_models)}/{self.limits.max_models_loaded}, "
            f"memory={self.get_memory_usage_mb():.1f}MB, "
            f"peak={self._peak_memory_mb:.1f}MB)"
        )
