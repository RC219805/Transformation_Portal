"""
Unified device detection for all pipelines.

Consolidates device detection logic from:
- src/transformation_portal/foundation/device_manager.py
- Multiple pipeline-specific implementations

Provides automatic device detection with Apple Silicon optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import logging
import platform

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported compute device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    COREML = "coreml"


@dataclass
class DeviceCapabilities:
    """Hardware capabilities for a device."""
    device_type: DeviceType
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    neural_engine_available: bool
    performance_cores: int
    efficiency_cores: int
    gpu_cores: int
    unified_memory: bool
    recommended_batch_size: int


@dataclass
class DeviceInfo:
    """Complete device information."""
    device: Any  # torch.device
    capabilities: DeviceCapabilities
    backend_priority: List[DeviceType]
    optimization_hints: Dict[str, Any]


class DeviceDetector:
    """
    Unified device detector for all pipelines.
    
    Automatically detects optimal compute device and provides
    hardware capabilities information.
    
    Example:
        >>> detector = DeviceDetector()
        >>> device_info = detector.detect()
        >>> print(f"Using device: {device_info.device}")
        >>> print(f"Memory: {device_info.capabilities.available_memory_gb:.1f} GB")
    """
    
    def __init__(self, memory_fraction: float = 0.85, prefer_neural_engine: bool = True):
        """
        Initialize device detector.
        
        Args:
            memory_fraction: Fraction of available memory to use (0.1-0.95)
            prefer_neural_engine: Prefer Apple Neural Engine when available
        """
        self.memory_fraction = max(0.1, min(0.95, memory_fraction))
        self.prefer_neural_engine = prefer_neural_engine
        self._cached_info: Optional[DeviceInfo] = None
    
    def detect(self, force_refresh: bool = False) -> DeviceInfo:
        """
        Detect optimal device and capabilities.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            DeviceInfo with complete hardware configuration
        """
        if self._cached_info is not None and not force_refresh:
            return self._cached_info
        
        # Try to import torch (optional dependency)
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available, using CPU-only mode")
            return self._create_cpu_only_info()
        
        # Check if torch has device class (minimal torch installs may not)
        if not hasattr(torch, 'device'):
            logger.warning("PyTorch minimal version detected, using CPU-only mode")
            return self._create_cpu_only_info()
        
        # Detect device type
        device_type = self._detect_device_type(torch)
        
        # Create torch device
        if device_type == DeviceType.MPS:
            device = torch.device("mps")
        elif device_type == DeviceType.CUDA:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Detect capabilities
        capabilities = self._detect_capabilities(device_type, torch)
        
        # Determine backend priority
        backend_priority = self._determine_backend_priority(device_type, capabilities)
        
        # Create optimization hints
        optimization_hints = self._create_optimization_hints(capabilities)
        
        # Cache and return
        self._cached_info = DeviceInfo(
            device=device,
            capabilities=capabilities,
            backend_priority=backend_priority,
            optimization_hints=optimization_hints
        )
        
        self._log_device_info(self._cached_info)
        return self._cached_info
    
    def _detect_device_type(self, torch) -> DeviceType:
        """Detect primary device type."""
        # Check for Apple Silicon MPS
        if (hasattr(torch, 'backends') and 
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()):
            logger.info("✓ Apple Silicon MPS detected")
            return DeviceType.MPS
        
        # Check for CUDA
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            logger.info("✓ CUDA device detected")
            return DeviceType.CUDA
        
        # Fallback to CPU
        logger.info("Using CPU (no GPU detected)")
        return DeviceType.CPU
    
    def _detect_capabilities(self, device_type: DeviceType, torch) -> DeviceCapabilities:
        """Detect hardware capabilities."""
        if device_type == DeviceType.MPS:
            return self._detect_mps_capabilities(torch)
        elif device_type == DeviceType.CUDA:
            return self._detect_cuda_capabilities(torch)
        else:
            return self._detect_cpu_capabilities()
    
    def _detect_mps_capabilities(self, torch) -> DeviceCapabilities:
        """Detect Apple Silicon MPS capabilities."""
        import subprocess
        
        # Get memory info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2
            )
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
        except Exception:
            total_memory_gb = 64.0  # Conservative default
        
        available_memory_gb = total_memory_gb * self.memory_fraction
        
        # Get CPU core counts
        try:
            perf_result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=2
            )
            perf_cores = int(perf_result.stdout.strip())
            
            eff_result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=2
            )
            eff_cores = int(eff_result.stdout.strip())
        except Exception:
            perf_cores, eff_cores = 8, 4  # Conservative defaults
        
        # Detect GPU cores (heuristic based on total cores)
        total_cores = perf_cores + eff_cores
        if total_cores >= 16:
            gpu_cores = 40  # M4 Max
        elif total_cores >= 12:
            gpu_cores = 30  # M3 Max / M4 Pro
        else:
            gpu_cores = 20  # M1/M2/M3 base
        
        # Calculate recommended batch size
        batch_size = max(1, int(available_memory_gb / 2.0))
        batch_size = min(batch_size, 64)
        
        return DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon (MPS)",
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            neural_engine_available=True,
            performance_cores=perf_cores,
            efficiency_cores=eff_cores,
            gpu_cores=gpu_cores,
            unified_memory=True,
            recommended_batch_size=batch_size
        )
    
    def _detect_cuda_capabilities(self, torch) -> DeviceCapabilities:
        """Detect CUDA capabilities."""
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory = total_memory * self.memory_fraction
        
        return DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name=device_name,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            supports_fp16=True,
            supports_bf16=torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False,
            supports_int8=True,
            neural_engine_available=False,
            performance_cores=0,
            efficiency_cores=0,
            gpu_cores=torch.cuda.get_device_properties(0).multi_processor_count,
            unified_memory=False,
            recommended_batch_size=max(1, int(available_memory / 2))
        )
    
    def _detect_cpu_capabilities(self) -> DeviceCapabilities:
        """Detect CPU capabilities."""
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count(logical=False) or 8
        except ImportError:
            total_memory = 16.0  # Conservative default
            cpu_count = 8
        
        available_memory = total_memory * self.memory_fraction
        
        # Rough estimate for performance/efficiency cores
        if platform.system() == "Darwin":
            perf_cores = max(4, cpu_count // 2)
            eff_cores = cpu_count - perf_cores
        else:
            perf_cores = cpu_count
            eff_cores = 0
        
        return DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name=platform.processor() or "CPU",
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            supports_fp16=False,
            supports_bf16=False,
            supports_int8=True,
            neural_engine_available=False,
            performance_cores=perf_cores,
            efficiency_cores=eff_cores,
            gpu_cores=0,
            unified_memory=False,
            recommended_batch_size=max(1, cpu_count // 2)
        )
    
    def _determine_backend_priority(
        self,
        device_type: DeviceType,
        capabilities: DeviceCapabilities
    ) -> List[DeviceType]:
        """Determine backend priority order."""
        if device_type == DeviceType.MPS:
            if self.prefer_neural_engine and capabilities.neural_engine_available:
                return [DeviceType.COREML, DeviceType.MPS, DeviceType.CPU]
            return [DeviceType.MPS, DeviceType.COREML, DeviceType.CPU]
        elif device_type == DeviceType.CUDA:
            return [DeviceType.CUDA, DeviceType.CPU]
        return [DeviceType.CPU]
    
    def _create_optimization_hints(self, capabilities: DeviceCapabilities) -> Dict[str, Any]:
        """Create optimization hints based on capabilities."""
        hints = {
            "device_type": capabilities.device_type.value,
            "precision": "fp16" if capabilities.supports_fp16 else "fp32",
            "enable_amp": capabilities.supports_fp16,
            "max_batch_size": capabilities.recommended_batch_size,
            "memory_limit_gb": capabilities.available_memory_gb,
            "num_workers": capabilities.performance_cores,
            "pin_memory": not capabilities.unified_memory,
        }
        
        if capabilities.device_type == DeviceType.MPS:
            hints.update({
                "enable_neural_engine": capabilities.neural_engine_available,
                "unified_memory": True,
            })
        elif capabilities.device_type == DeviceType.CUDA:
            hints.update({
                "enable_cudnn_benchmark": True,
                "enable_tf32": True,
            })
        
        return hints
    
    def _create_cpu_only_info(self) -> DeviceInfo:
        """Create CPU-only device info when torch is not available."""
        capabilities = self._detect_cpu_capabilities()
        
        return DeviceInfo(
            device="cpu",
            capabilities=capabilities,
            backend_priority=[DeviceType.CPU],
            optimization_hints=self._create_optimization_hints(capabilities)
        )
    
    def _log_device_info(self, info: DeviceInfo):
        """Log detected device information."""
        cap = info.capabilities
        logger.info("=" * 60)
        logger.info("DEVICE CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Device: {cap.device_name}")
        logger.info(f"Type: {cap.device_type.value.upper()}")
        logger.info(f"Memory: {cap.available_memory_gb:.1f} GB available")
        logger.info(f"Cores: {cap.performance_cores}P + {cap.efficiency_cores}E")
        logger.info(f"Batch Size: {cap.recommended_batch_size} (recommended)")
        logger.info("=" * 60)
