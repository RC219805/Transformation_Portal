"""
Policy engine for context-aware processing decisions.

Makes intelligent routing decisions based on:
- Device capabilities (CPU/CUDA/MPS/CoreML)
- Scene classification (interior/exterior/aerial)
- Quality requirements (draft/standard/production)
- Resource constraints (memory, time)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SceneType(str, Enum):
    """Scene classification for context-aware processing."""
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    AERIAL = "aerial"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class QualityPreset(str, Enum):
    """Quality preset levels."""
    DRAFT = "draft"           # Fast preview
    STANDARD = "standard"     # Good quality, reasonable speed
    HIGH = "high"             # High quality, slower
    PRODUCTION = "production"  # Maximum quality, slow


@dataclass
class DevicePolicy:
    """
    Device capability and routing policy.

    Determines which device to use for each stage based on
    availability and performance characteristics.
    """
    # Available devices
    has_cuda: bool = False
    has_mps: bool = False
    has_coreml: bool = False

    # Memory constraints (GB)
    available_memory_gb: float = 8.0

    # Preferences
    prefer_gpu: bool = True
    prefer_coreml_depth: bool = True  # CoreML faster for depth on M-series

    def select_device(self, stage_name: str) -> str:
        """
        Select optimal device for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Device string ("cuda", "mps", "cpu")
        """
        # Depth estimation: prefer CoreML on Apple Silicon
        if "depth" in stage_name.lower() and self.has_coreml and self.prefer_coreml_depth:
            return "coreml"

        # General GPU preference
        if self.prefer_gpu:
            if self.has_cuda:
                return "cuda"
            elif self.has_mps:
                return "mps"

        return "cpu"

    def can_use_batch(self, batch_size: int, image_size_mp: float) -> bool:
        """
        Check if batch processing is feasible.

        Args:
            batch_size: Batch size
            image_size_mp: Image size in megapixels

        Returns:
            True if batch can fit in memory
        """
        # Rough estimate: 4 bytes per pixel * 3 channels * 3x overhead for processing
        estimated_memory_gb = (batch_size * image_size_mp * 12 * 3) / 1024

        # Leave 2GB headroom
        return estimated_memory_gb < (self.available_memory_gb - 2.0)


@dataclass
class QualityPolicy:
    """
    Quality requirements and processing decisions.

    Maps quality presets to stage-specific parameters.
    """
    preset: QualityPreset = QualityPreset.STANDARD

    # Upscaling
    upscale_factor: float = 1.0
    upscale_backend: str = "torch"  # "torch", "onnx"

    # Enhancement strength
    enhancement_strength: float = 0.7
    clarity_strength: float = 0.5

    # Material processing
    enable_materials: bool = True
    material_strength: float = 0.6

    def apply_preset(self, preset: QualityPreset):
        """Apply quality preset."""
        self.preset = preset

        if preset == QualityPreset.DRAFT:
            self.upscale_factor = 1.0
            self.enhancement_strength = 0.3
            self.clarity_strength = 0.2
            self.enable_materials = False

        elif preset == QualityPreset.STANDARD:
            self.upscale_factor = 1.0
            self.enhancement_strength = 0.5
            self.clarity_strength = 0.4
            self.enable_materials = True
            self.material_strength = 0.5

        elif preset == QualityPreset.HIGH:
            self.upscale_factor = 2.0
            self.enhancement_strength = 0.7
            self.clarity_strength = 0.6
            self.enable_materials = True
            self.material_strength = 0.7

        elif preset == QualityPreset.PRODUCTION:
            self.upscale_factor = 2.0
            self.enhancement_strength = 0.8
            self.clarity_strength = 0.7
            self.enable_materials = True
            self.material_strength = 0.8


@dataclass
class CachingPolicy:
    """
    Caching behavior configuration.
    """
    enabled: bool = True
    cache_dir: Optional[Path] = None

    # Cache invalidation
    max_age_hours: Optional[float] = None
    max_size_gb: float = 10.0

    # Selective caching
    cache_depth_maps: bool = True
    cache_material_masks: bool = True
    cache_enhanced: bool = False  # Enhanced images usually final output

    def should_cache_stage(self, stage_name: str) -> bool:
        """Determine if a stage should use caching."""
        if not self.enabled:
            return False

        stage_lower = stage_name.lower()

        if "depth" in stage_lower:
            return self.cache_depth_maps
        elif "material" in stage_lower or "segmentation" in stage_lower:
            return self.cache_material_masks
        elif "enhance" in stage_lower or "upscale" in stage_lower:
            return self.cache_enhanced

        return True  # Default: cache everything


@dataclass
class ProcessingPolicy:
    """
    Complete processing policy combining all sub-policies.

    This is the main policy object passed to the pipeline.
    """
    device: DevicePolicy = None
    quality: QualityPolicy = None
    caching: CachingPolicy = None

    # Scene context
    scene_type: SceneType = SceneType.UNKNOWN

    # Parallel execution
    enable_parallel: bool = True
    max_workers: int = 4

    def __post_init__(self):
        """Initialize sub-policies with defaults."""
        if self.device is None:
            self.device = DevicePolicy()
        if self.quality is None:
            self.quality = QualityPolicy()
        if self.caching is None:
            self.caching = CachingPolicy()


class PolicyEngine:
    """
    Policy engine for intelligent processing decisions.

    Analyzes input and environment to create optimal processing policy.
    """

    def __init__(self):
        """Initialize policy engine."""
        self.logger = logging.getLogger(f"{__name__}.PolicyEngine")

    def create_policy(
        self,
        quality_preset: Optional[QualityPreset] = None,
        scene_type: Optional[SceneType] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ProcessingPolicy:
        """
        Create processing policy from inputs.

        Args:
            quality_preset: Desired quality level
            scene_type: Scene classification
            config: Additional configuration

        Returns:
            Complete processing policy
        """
        config = config or {}

        # Create base policy
        policy = ProcessingPolicy()

        # Apply quality preset
        if quality_preset:
            policy.quality.apply_preset(quality_preset)

        # Set scene type
        if scene_type:
            policy.scene_type = scene_type
            self._adjust_for_scene(policy, scene_type)

        # Detect device capabilities
        self._detect_devices(policy.device)

        # Apply config overrides
        self._apply_config(policy, config)

        return policy

    def _detect_devices(self, device_policy: DevicePolicy):
        """Detect available devices."""
        try:
            import torch
            device_policy.has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            device_policy.has_mps = (
                hasattr(torch, 'backends') and
                hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()
            )
        except (ImportError, AttributeError):
            pass

        # Check for CoreML
        try:
            import coremltools  # noqa: F401
            import platform
            device_policy.has_coreml = platform.system() == "Darwin"
        except ImportError:
            pass

        # Estimate available memory
        try:
            import psutil
            device_policy.available_memory_gb = (
                psutil.virtual_memory().available / (1024 ** 3)
            )
        except ImportError:
            pass

    def _adjust_for_scene(self, policy: ProcessingPolicy, scene_type: SceneType):
        """Adjust policy parameters based on scene type."""
        if scene_type == SceneType.AERIAL:
            # Aerial: more clarity, less material enhancement
            policy.quality.clarity_strength *= 1.2
            policy.quality.material_strength *= 0.8

        elif scene_type == SceneType.INTERIOR:
            # Interior: balanced, full material response
            policy.quality.material_strength = 0.8
            policy.quality.enable_materials = True

        elif scene_type == SceneType.EXTERIOR:
            # Exterior: emphasis on lighting and atmosphere
            policy.quality.enhancement_strength *= 1.1

    def _apply_config(self, policy: ProcessingPolicy, config: Dict[str, Any]):
        """Apply configuration overrides."""
        # Device overrides
        if "device" in config:
            if config["device"] == "cpu":
                policy.device.prefer_gpu = False

        # Caching overrides
        if "cache_dir" in config:
            policy.caching.cache_dir = Path(config["cache_dir"])

        if "cache_enabled" in config:
            policy.caching.enabled = config["cache_enabled"]

        # Quality overrides
        if "upscale_factor" in config:
            policy.quality.upscale_factor = config["upscale_factor"]

        if "enhancement_strength" in config:
            policy.quality.enhancement_strength = config["enhancement_strength"]

        # Parallel execution
        if "enable_parallel" in config:
            policy.enable_parallel = config["enable_parallel"]

        if "max_workers" in config:
            policy.max_workers = config["max_workers"]
