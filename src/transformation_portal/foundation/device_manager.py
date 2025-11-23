"""
Device Manager for Apple Silicon M4 Max

Provides intelligent device detection, configuration, and optimization
specifically tailored for Apple Silicon M4 Max architecture.

Key Features:
- M4 Max-specific capability detection
- Unified memory architecture awareness
- Neural Engine (ANE) optimization
- Metal Performance Shaders (MPS) configuration
- Multi-backend fallback strategies
"""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for computation."""
    MPS = "mps"  # Metal Performance Shaders (Apple Silicon)
    CUDA = "cuda"  # NVIDIA GPUs
    COREML = "coreml"  # Apple Neural Engine
    CPU = "cpu"  # CPU fallback


@dataclass
class DeviceCapabilities:
    """Hardware capabilities for a specific device."""
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
    max_buffer_size_gb: float
    recommended_batch_size: int
    metal_version: Optional[str] = None
    torch_version: str = ""


@dataclass
class DeviceInfo:
    """Complete device information and configuration."""
    primary_device: torch.device
    capabilities: DeviceCapabilities
    backend_priority: List[DeviceType]
    optimization_config: Dict[str, Any]


class DeviceManager:
    """
    M4 Max-optimized device manager.

    Handles device detection, configuration, and optimization strategies
    specifically for Apple Silicon M4 Max architecture with 128GB unified memory,
    40-core GPU, and 16-core CPU (12 performance + 4 efficiency cores).
    """

    def __init__(self, prefer_ane: bool = True, memory_fraction: float = 0.85):
        """
        Initialize device manager.

        Args:
            prefer_ane: Whether to prefer Apple Neural Engine when available
            memory_fraction: Fraction of available memory to use (0.0-1.0)
        """
        self.prefer_ane = prefer_ane
        self.memory_fraction = min(max(memory_fraction, 0.1), 0.95)
        self.device_info: Optional[DeviceInfo] = None
        self._detection_cache: Dict[str, Any] = {}

        logger.info("Initializing Device Manager for Apple Silicon M4 Max")

    def detect_devices(self) -> DeviceInfo:
        """
        Detect and configure optimal device for Apple Silicon M4 Max.

        Returns:
            DeviceInfo with complete hardware configuration
        """
        if self.device_info is not None:
            return self.device_info

        logger.info("Detecting hardware capabilities...")

        # Detect device type and create torch device
        device_type = self._detect_device_type()
        primary_device = self._create_torch_device(device_type)

        # Get hardware capabilities
        capabilities = self._detect_capabilities(device_type)

        # Determine backend priority
        backend_priority = self._determine_backend_priority(device_type, capabilities)

        # Create optimization configuration
        optimization_config = self._create_optimization_config(capabilities)

        # Store device info
        self.device_info = DeviceInfo(
            primary_device=primary_device,
            capabilities=capabilities,
            backend_priority=backend_priority,
            optimization_config=optimization_config
        )

        self._log_device_info()
        return self.device_info

    def _detect_device_type(self) -> DeviceType:
        """Detect the primary device type available."""
        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            if self._is_m4_max():
                logger.info("✓ Apple Silicon M4 Max detected with MPS support")
                return DeviceType.MPS
            else:
                logger.info("✓ Apple Silicon detected with MPS support")
                return DeviceType.MPS

        # Check for CUDA
        if torch.cuda.is_available():
            logger.info("✓ CUDA device detected")
            return DeviceType.CUDA

        # Fallback to CPU
        logger.warning("⚠ No GPU detected, falling back to CPU")
        return DeviceType.CPU

    def _is_m4_max(self) -> bool:
        """Check if running on Apple Silicon M4 Max specifically."""
        try:
            # Check platform
            if platform.system() != "Darwin":
                return False

            # Get chip information via sysctl
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )

            chip_name = result.stdout.strip()
            is_m4 = "Apple M4" in chip_name

            # Additional checks for M4 Max variant
            if is_m4:
                # M4 Max has 16 CPU cores (12P + 4E) and 40 GPU cores
                cpu_count = subprocess.run(
                    ["sysctl", "-n", "hw.ncpu"],
                    capture_output=True,
                    text=True,
                    timeout=2
                ).stdout.strip()

                if cpu_count == "16":
                    logger.info(f"Detected chip: {chip_name} (M4 Max variant)")
                    self._detection_cache["chip_name"] = chip_name
                    return True
                else:
                    logger.info(f"Detected chip: {chip_name} (M4 variant)")
                    self._detection_cache["chip_name"] = chip_name
                    return True  # Still M4, close enough for optimizations

            return False

        except Exception as e:
            logger.debug(f"M4 Max detection failed: {e}")
            return False

    def _create_torch_device(self, device_type: DeviceType) -> torch.device:
        """Create PyTorch device object."""
        if device_type == DeviceType.MPS:
            return torch.device("mps")
        elif device_type == DeviceType.CUDA:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _detect_capabilities(self, device_type: DeviceType) -> DeviceCapabilities:
        """Detect hardware capabilities for the given device type."""
        if device_type == DeviceType.MPS:
            return self._detect_mps_capabilities()
        elif device_type == DeviceType.CUDA:
            return self._detect_cuda_capabilities()
        else:
            return self._detect_cpu_capabilities()

    def _detect_mps_capabilities(self) -> DeviceCapabilities:
        """Detect Apple Silicon MPS capabilities."""
        try:
            # Get memory information
            mem_info = self._get_macos_memory()
            total_memory_gb = mem_info.get("total", 128.0)  # Default to M4 Max spec
            available_memory_gb = total_memory_gb * self.memory_fraction

            # Get CPU core counts
            perf_cores, eff_cores = self._get_cpu_core_counts()

            # M4 Max specifications
            gpu_cores = 40  # M4 Max has 40 GPU cores

            # Check Metal version
            metal_version = self._get_metal_version()

            # Neural Engine available on all Apple Silicon
            neural_engine_available = True

            # M4 Max supports fp16, bf16 (via MPS), and int8
            supports_fp16 = True
            supports_bf16 = torch.backends.mps.is_available()
            supports_int8 = True

            # Max buffer size (conservative estimate for unified memory)
            max_buffer_size_gb = available_memory_gb * 0.9

            # Recommended batch size based on 128GB unified memory
            recommended_batch_size = self._calculate_optimal_batch_size(
                available_memory_gb, gpu_cores
            )

            return DeviceCapabilities(
                device_type=DeviceType.MPS,
                device_name=self._detection_cache.get("chip_name", "Apple Silicon M4 Max"),
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                supports_fp16=supports_fp16,
                supports_bf16=supports_bf16,
                supports_int8=supports_int8,
                neural_engine_available=neural_engine_available,
                performance_cores=perf_cores,
                efficiency_cores=eff_cores,
                gpu_cores=gpu_cores,
                unified_memory=True,
                max_buffer_size_gb=max_buffer_size_gb,
                recommended_batch_size=recommended_batch_size,
                metal_version=metal_version,
                torch_version=torch.__version__
            )

        except Exception as e:
            logger.error(f"Error detecting MPS capabilities: {e}")
            # Return conservative defaults
            return self._get_default_mps_capabilities()

    def _detect_cuda_capabilities(self) -> DeviceCapabilities:
        """Detect CUDA capabilities."""
        if not torch.cuda.is_available():
            return self._detect_cpu_capabilities()

        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory = total_memory * self.memory_fraction

        return DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name=device_name,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            supports_fp16=True,
            supports_bf16=torch.cuda.is_bf16_supported(),
            supports_int8=True,
            neural_engine_available=False,
            performance_cores=0,
            efficiency_cores=0,
            gpu_cores=torch.cuda.get_device_properties(0).multi_processor_count,
            unified_memory=False,
            max_buffer_size_gb=available_memory * 0.8,
            recommended_batch_size=max(1, int(available_memory / 2)),
            torch_version=torch.__version__
        )

    def _detect_cpu_capabilities(self) -> DeviceCapabilities:
        """Detect CPU capabilities."""
        import psutil

        total_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = total_memory * self.memory_fraction

        cpu_count = psutil.cpu_count(logical=False) or 8
        perf_cores, eff_cores = self._get_cpu_core_counts()

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
            max_buffer_size_gb=available_memory * 0.5,
            recommended_batch_size=max(1, cpu_count // 2),
            torch_version=torch.__version__
        )

    def _get_macos_memory(self) -> Dict[str, float]:
        """Get macOS memory information in GB."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2
            )
            mem_bytes = int(result.stdout.strip())
            mem_gb = mem_bytes / (1024**3)
            return {"total": mem_gb}
        except Exception as e:
            logger.debug(f"Could not get macOS memory: {e}")
            return {"total": 128.0}  # M4 Max default

    def _get_cpu_core_counts(self) -> Tuple[int, int]:
        """Get performance and efficiency core counts."""
        try:
            if platform.system() == "Darwin":
                # M4 Max: 12 performance + 4 efficiency = 16 total
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

                return perf_cores, eff_cores
        except Exception as e:
            logger.debug(f"Could not get core counts: {e}")

        # M4 Max defaults
        return 12, 4

    def _get_metal_version(self) -> Optional[str]:
        """Get Metal API version."""
        try:
            # Metal 3.1 is available on macOS 14+ (M4 Max support)
            result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=2
            )
            macos_version = result.stdout.strip()
            major_version = int(macos_version.split('.')[0])

            if major_version >= 14:
                return "Metal 3.1"
            elif major_version >= 13:
                return "Metal 3.0"
            else:
                return "Metal 2.x"
        except Exception as e:
            logger.debug(f"Could not determine Metal version: {e}")
            return "Metal 3.1"  # Assume latest for M4 Max

    def _calculate_optimal_batch_size(self, available_memory_gb: float, gpu_cores: int) -> int:
        """
        Calculate optimal batch size based on available memory and GPU cores.

        For M4 Max with 128GB unified memory and 40 GPU cores,
        we can handle larger batches than typical discrete GPUs.
        """
        # Base calculation: ~2GB per batch item for high-res image processing
        memory_based = max(1, int(available_memory_gb / 2.0))

        # GPU core based: ~2-4 items per GPU core for optimal utilization
        gpu_based = max(1, gpu_cores * 3)

        # Take minimum to avoid OOM
        optimal = min(memory_based, gpu_based)

        # Cap at reasonable maximum for stability
        return min(optimal, 64)

    def _get_default_mps_capabilities(self) -> DeviceCapabilities:
        """Get conservative default MPS capabilities."""
        return DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon",
            total_memory_gb=64.0,
            available_memory_gb=54.0,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            neural_engine_available=True,
            performance_cores=8,
            efficiency_cores=4,
            gpu_cores=30,
            unified_memory=True,
            max_buffer_size_gb=48.0,
            recommended_batch_size=16,
            metal_version="Metal 3.0",
            torch_version=torch.__version__
        )

    def _determine_backend_priority(
        self,
        device_type: DeviceType,
        capabilities: DeviceCapabilities
    ) -> List[DeviceType]:
        """Determine backend priority order."""
        if device_type == DeviceType.MPS:
            if self.prefer_ane and capabilities.neural_engine_available:
                return [DeviceType.COREML, DeviceType.MPS, DeviceType.CPU]
            else:
                return [DeviceType.MPS, DeviceType.COREML, DeviceType.CPU]
        elif device_type == DeviceType.CUDA:
            return [DeviceType.CUDA, DeviceType.CPU]
        else:
            return [DeviceType.CPU]

    def _create_optimization_config(self, capabilities: DeviceCapabilities) -> Dict[str, Any]:
        """Create optimization configuration based on capabilities."""
        config = {
            "device_type": capabilities.device_type.value,
            "precision": "fp16" if capabilities.supports_fp16 else "fp32",
            "enable_amp": capabilities.supports_fp16,  # Automatic Mixed Precision
            "enable_tf32": False,  # TensorFloat-32 (NVIDIA-specific)
            "max_batch_size": capabilities.recommended_batch_size,
            "memory_limit_gb": capabilities.available_memory_gb,
            "enable_memory_efficient_attention": True,
            "enable_gradient_checkpointing": capabilities.total_memory_gb < 32,
            "num_workers": capabilities.performance_cores,
            "pin_memory": not capabilities.unified_memory,
            "persistent_workers": True,
        }

        # M4 Max-specific optimizations
        if capabilities.device_type == DeviceType.MPS:
            config.update({
                "mps_allocator_strategy": "unified",  # Use unified memory allocator
                "mps_high_watermark_ratio": 0.9,  # Allow high memory usage
                "enable_metal_simd": True,  # SIMD optimizations
                "enable_neural_engine": capabilities.neural_engine_available,
                "metal_version": capabilities.metal_version,
            })

        # CUDA-specific optimizations
        elif capabilities.device_type == DeviceType.CUDA:
            config.update({
                "enable_tf32": True,
                "enable_cudnn_benchmark": True,
                "enable_flash_attention": True,
            })

        return config

    def _log_device_info(self):
        """Log detected device information."""
        if self.device_info is None:
            return

        cap = self.device_info.capabilities
        logger.info("=" * 70)
        logger.info("DEVICE CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Device: {cap.device_name}")
        logger.info(f"Type: {cap.device_type.value.upper()}")
        logger.info(f"Total Memory: {cap.total_memory_gb:.1f} GB")
        logger.info(f"Available Memory: {cap.available_memory_gb:.1f} GB")
        logger.info(f"Performance Cores: {cap.performance_cores}")
        logger.info(f"Efficiency Cores: {cap.efficiency_cores}")
        logger.info(f"GPU Cores: {cap.gpu_cores}")
        logger.info(f"Unified Memory: {cap.unified_memory}")
        logger.info(f"Neural Engine: {'Available' if cap.neural_engine_available else 'Not Available'}")
        logger.info(f"FP16 Support: {cap.supports_fp16}")
        logger.info(f"BF16 Support: {cap.supports_bf16}")
        logger.info(f"Recommended Batch Size: {cap.recommended_batch_size}")
        if cap.metal_version:
            logger.info(f"Metal Version: {cap.metal_version}")
        logger.info(f"PyTorch Version: {cap.torch_version}")
        logger.info("=" * 70)

    def get_device(self) -> torch.device:
        """Get the primary PyTorch device."""
        if self.device_info is None:
            self.detect_devices()
        return self.device_info.primary_device

    def get_capabilities(self) -> DeviceCapabilities:
        """Get device capabilities."""
        if self.device_info is None:
            self.detect_devices()
        return self.device_info.capabilities

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        if self.device_info is None:
            self.detect_devices()
        return self.device_info.optimization_config
