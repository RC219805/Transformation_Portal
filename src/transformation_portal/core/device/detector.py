"""
Device Detector Module.

Identifies available compute hardware and determines optimal configuration.
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallbacks.
"""

import logging
import platform
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _get_torch():
    """Lazy import for torch (may be absent in core/CI environments)."""
    import torch

    return torch


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DeviceCapabilities:
    """Hardware capability profile."""

    fp16_supported: bool
    bf16_supported: bool
    vram_gb: float
    compute_capability: Optional[str] = None
    device_name: str = "Unknown"


@dataclass
class DeviceInfo:
    """Runtime device configuration."""

    device: "torch.device"
    type: DeviceType
    capabilities: DeviceCapabilities
    index: int = 0


class DeviceDetector:
    """
    Intelligent hardware discovery.

    Prioritizes: CUDA > MPS > CPU.
    """

    @staticmethod
    def get_optimal_device(force_cpu: bool = False) -> DeviceInfo:
        """Discover the best available compute device."""
        torch = _get_torch()

        if force_cpu:
            return DeviceDetector._get_cpu_info()

        # 1. Check NVIDIA CUDA
        if torch.cuda.is_available():
            return DeviceDetector._get_cuda_info()

        # 2. Check Apple Metal Performance Shaders (MPS)
        if torch.backends.mps.is_available():
            return DeviceDetector._get_mps_info()

        # 3. Fallback
        return DeviceDetector._get_cpu_info()

    @staticmethod
    def _get_cuda_info(index: int = 0) -> DeviceInfo:
        torch = _get_torch()
        props = torch.cuda.get_device_properties(index)
        vram_gb = props.total_memory / (1024**3)

        # FP16 is generally supported on all modern GPUs
        # BF16 requires Ampere (Compute Capability 8.0+)
        cc_major = props.major
        bf16_support = cc_major >= 8

        caps = DeviceCapabilities(
            fp16_supported=True,
            bf16_supported=bf16_support,
            vram_gb=vram_gb,
            compute_capability=f"{props.major}.{props.minor}",
            device_name=props.name,
        )

        logger.info(f"Detected CUDA Device: {props.name} ({vram_gb:.1f} GB VRAM)")
        return DeviceInfo(device=torch.device(f"cuda:{index}"), type=DeviceType.CUDA, capabilities=caps, index=index)

    @staticmethod
    def _get_mps_info() -> DeviceInfo:
        torch = _get_torch()
        # MPS doesn't expose memory queries directly via torch.cuda properties yet
        # We assume standard M1/M2/M3 unified memory behavior
        caps = DeviceCapabilities(
            fp16_supported=True,
            bf16_supported=False,  # MPS BF16 support is experimental/limited
            vram_gb=0.0,  # Unified memory - tricky to query from Python without psutil
            device_name="Apple Silicon (MPS)",
        )
        logger.info("Detected Apple Silicon (MPS) Device")
        return DeviceInfo(device=torch.device("mps"), type=DeviceType.MPS, capabilities=caps)

    @staticmethod
    def _get_cpu_info() -> DeviceInfo:
        torch = _get_torch()
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)

        caps = DeviceCapabilities(
            fp16_supported=False,  # CPU fp16 is usually slow/emulated
            bf16_supported=False,
            vram_gb=ram_gb,  # System RAM
            device_name=platform.processor() or "Generic CPU",
        )
        logger.info(f"Using CPU Fallback ({ram_gb:.1f} GB RAM)")
        return DeviceInfo(device=torch.device("cpu"), type=DeviceType.CPU, capabilities=caps)
