"""Configuration module for lux_depth_v3 pipeline.

STUB IMPLEMENTATION - Critical types to enable package imports.
Full implementation pending.
"""

from __future__ import annotations

# Check xxhash availability for default hasher selection
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .security import HashMode

_XXHASH_AVAILABLE = importlib.util.find_spec("xxhash") is not None


class ModelVariant(Enum):
    """Depth Anything V3 model variants.

    Note: DA3 models require custom library installation:
        git clone https://github.com/ByteDance/depth-anything-3
        cd depth-anything-3
        pip install -e .

    ⚠️  LICENSE: DA3NESTED-GIANT-LARGE-1.1 is CC BY-NC 4.0 (non-commercial use only).
    """

    METRIC_LARGE = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-large",
            "display_name": "Depth Anything V3 Metric Large (DA3 Nested Giant)",
            "huggingface_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        },
    )()
    METRIC_BASE = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-base",
            "display_name": "Depth Anything V3 Metric Base",
            "huggingface_id": "depth-anything/Depth-Anything-V3-Metric-Base-hf",
        },
    )()
    METRIC_SMALL = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-small",
            "display_name": "Depth Anything V3 Metric Small",
            "huggingface_id": "depth-anything/Depth-Anything-V3-Metric-Small-hf",
        },
    )()


class Preset(Enum):
    """Pipeline presets for different use cases."""

    ARCHITECTURAL_INTERIOR = "architectural_interior"
    ARCHITECTURAL_EXTERIOR = "architectural_exterior"
    LUXURY_ESTATE = "luxury_estate"
    DEFAULT = "default"


@dataclass
class DeviceConfig:
    """Device configuration for inference."""

    device: str = "cpu"
    dtype: str = "float32"
    use_fp16: bool = True  # Enable FP16 for MPS/CUDA (1.3-1.5x speedup, 2x memory reduction)
    use_coreml: bool = False  # Phase 3: CoreML ANE acceleration for Apple Silicon (5x speedup, opt-in)


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration for depth maps."""

    apply_metric_scaling: bool = True
    scale_factor: float = 1.0
    apply_median_filter: bool = False
    median_kernel_size: int = 3
    apply_bilateral_filter: bool = False
    bilateral_sigma_color: float = 0.0
    bilateral_sigma_space: float = 0.0
    preserve_edges: bool = True
    edge_threshold: float = 0.1
    fusion_mode: str = "weighted"
    refinement: Optional[Any] = None


@dataclass
class DA3Config:
    """Depth Anything V3 configuration."""

    model_variant: ModelVariant = ModelVariant.METRIC_LARGE
    device: DeviceConfig = field(default_factory=DeviceConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    @classmethod
    def from_preset(cls, preset: Preset) -> DA3Config:
        """Create configuration from preset.

        Presets provide different quality/performance tradeoffs:
        - ARCHITECTURAL_INTERIOR: High quality for interior architectural renders
        - ARCHITECTURAL_EXTERIOR: Balanced for exterior scenes
        - LUXURY_ESTATE: Premium quality for luxury real estate
        - DEFAULT: Standard balanced configuration
        """
        # Define preset-specific configurations
        if preset == Preset.ARCHITECTURAL_INTERIOR:
            return cls(
                model_variant=ModelVariant.METRIC_LARGE,
                postprocessing=PostprocessingConfig(
                    apply_metric_scaling=True,
                    scale_factor=1.0,
                    apply_bilateral_filter=True,
                    bilateral_sigma_color=0.05,
                    bilateral_sigma_space=5.0,
                    preserve_edges=True,
                    edge_threshold=0.05,
                ),
            )
        elif preset == Preset.ARCHITECTURAL_EXTERIOR:
            return cls(
                model_variant=ModelVariant.METRIC_BASE,
                postprocessing=PostprocessingConfig(
                    apply_metric_scaling=True,
                    scale_factor=1.0,
                    preserve_edges=True,
                    edge_threshold=0.1,
                ),
            )
        elif preset == Preset.LUXURY_ESTATE:
            return cls(
                model_variant=ModelVariant.METRIC_LARGE,
                postprocessing=PostprocessingConfig(
                    apply_metric_scaling=True,
                    scale_factor=1.0,
                    apply_bilateral_filter=True,
                    bilateral_sigma_color=0.03,
                    bilateral_sigma_space=7.0,
                    preserve_edges=True,
                    edge_threshold=0.03,
                ),
            )
        else:  # DEFAULT
            return cls()


@dataclass
class EnhanceConfig:
    """Configuration for the enhancement orchestrator."""

    # Depth configuration
    model_variant: Optional[ModelVariant] = None
    preset: Optional[Preset] = None
    depth_device: str = "cpu"
    depth_quantization: str = "none"

    # V2 configuration
    v2_preset: Optional[str] = "default"  # None = skip V2 stage entirely
    v2_device: str = "cpu"
    v2_upscaler_backend: str = "default"
    enable_v2: bool = True  # Master switch for V2 stage

    # Flags
    force_depth: bool = False
    force_v2: bool = False
    non_commercial_ok: bool = False
    verify_depth_writes: bool = True
    strict_inputs: bool = False  # Fail if depth artifacts found in input directory (validation mode)

    # License acceptance flags (for research-only models)
    accept_apple_depth_pro_research_license: bool = False  # Apple AMLR license for Depth Pro
    accept_research_tools_license: bool = False  # Umbrella flag for APEX Research Ultra (ADR-026)

    # Spatial AI Foundation (ADR-026 Phase I)
    spatial_ai_linear_ingest: bool = False  # Enable linear light preservation (float32, gamma=1.0)

    # Depth backend selection
    depth_backend: Optional[str] = None  # None = auto (DA3), "depth_pro", or "ensemble"
    depth_pro_checkpoint_path: Optional[str] = None  # Path to depth_pro.pt checkpoint

    # Fallback configuration
    depth_fallback: str = "fail"  # Options: "fail", "skip", "v2-auto"
    v2_timeout: int = 300
    allow_synthetic_fallback: bool = False  # Allow synthetic depth backend when no ML deps (test/CI only)

    # Hash mode
    hash_mode: HashMode = HashMode.IF_MANIFEST_EXISTS

    # Performance optimizations (Phase 1)
    enable_manifest_cache: bool = True  # LRU cache for manifest loading (15-20% I/O reduction)
    chunked_hashing: bool = True  # Chunked SHA-256 for large files (90% memory reduction)

    # Performance optimizations (Phase 2: Parallelization)
    enable_parallel_processing: bool = True  # Parallel I/O for batch workflows (3-5x throughput)
    max_parallel_workers: Optional[int] = None  # Auto-detect if None (default: cpu_count - 1)
    enable_depth_cache: bool = False  # Content-addressable depth cache (opt-in, requires storage)
    depth_cache_max_size_gb: float = 10.0  # Maximum cache size before LRU eviction

    # Performance optimizations (Phase 3: Advanced optimizations)
    use_coreml_backend: bool = False  # CoreML ANE for Apple Silicon (5x depth inference, requires conversion)
    enable_pbr_gpu_batching: bool = False  # GPU-accelerated PBR map batching (30% speedup, opt-in)
    use_msgpack_manifests: bool = False  # MessagePack binary format (60% smaller, 3x faster, less readable)
    # xxHash for output keys (5x faster than SHA-1, auto-enabled when available)
    use_xxhash: bool = field(default_factory=lambda: _XXHASH_AVAILABLE)

    # Float depth saving for high-precision PBR
    save_float_depth: bool = False

    # PBR map generation
    generate_pbr: bool = False
    pbr_normal_strength: float = 1.0
    pbr_normal_blur_radius: int = 0
    pbr_roughness_strength: float = 1.0
    pbr_roughness_blur_radius: int = 3
    pbr_ao_strength: float = 1.0
    pbr_ao_blur_radius: int = 5
    pbr_ao_bias: float = 0.5

    # Quality tier and Materials V3
    quality_tier: str = "standard"  # Options: standard, premium, apex

    # APEX depth validity gate (fail-closed quality policy)
    apex_depth_min_finite_pct: float = 0.999
    apex_depth_min_upper_iqr: float = 1e-4
    apex_depth_max_high_saturation_fraction: float = 0.02
    apex_depth_max_low_saturation_fraction: float = 0.02
    apex_depth_saturation_high_value: float = 0.999
    apex_depth_saturation_low_value: float = 0.001
    apex_depth_min_gradient_energy: float = 5e-4
    apex_depth_hist_bins: int = 64

    enable_materials_v3: bool = False  # Materials V3 surface-aware finishing
    apply_pixel_ops: bool = True  # Apply pixel operations in Materials V3 (requires enable_materials_v3=True)

    # Materials V3 configuration
    refinement_strategy: str = "canary"  # EfficientSAM refinement strategy (canary, disabled)
    min_coverage_px: int = 500  # Minimum material coverage in pixels
    min_mean_conf: float = 0.2  # Minimum mean confidence for material detection
    glass_response_enabled: bool = True  # Enable glass material response

    # Materials V3 Pixel Ops - Feathering Configuration (A3)
    mask_feather_sigma_default: float = 3.0  # Default Gaussian blur sigma for mask feathering
    mask_feather_sigma_overrides: Dict[str, float] = field(default_factory=dict)  # Material-specific overrides
    mask_feather_disabled_materials: list[str] = field(default_factory=list)  # Materials with feathering disabled

    # Materials V3 Phase B - Sky Bootstrap Configuration
    sky_top_region_fraction: float = 0.5  # Top fraction of image to consider for sky (default: upper 50%)
    sky_gradient_threshold: float = 0.05  # Maximum gradient magnitude for smooth sky regions
    sky_brightness_threshold: float = 0.4  # Minimum brightness threshold for sky pixels

    # Materials V3 segmentation backend (Phase 3)
    enable_material_segmentation: bool = False  # Enable automatic material segmentation
    material_segmentation_backend: str = "stub"  # Options: stub, efficientsam
    strict_backend: bool = False  # If True, raise on backend errors instead of falling back to stub

    # Emit flags (deliverables)
    emit_master16: bool = False  # Emit master 16-bit output
    emit_upscaled16: bool = False  # Emit upscaled 16-bit output
    emit_marketing: bool = False  # Emit marketing-ready output
    emit_report: bool = True  # Emit processing report
    emit_run_card: bool = True  # Emit run card for reproducibility

    @property
    def enable_pbr(self) -> bool:
        """Alias for generate_pbr (backward compatibility)."""
        return self.generate_pbr

    def to_pbr_config(self):
        """Convert EnhanceConfig to PBRConfig.

        Returns:
            PBRConfig instance with parameters from this config
        """
        from .pbr import PBRConfig

        return PBRConfig(
            normal_strength=self.pbr_normal_strength,
            normal_blur_radius=self.pbr_normal_blur_radius,
            roughness_strength=self.pbr_roughness_strength,
            roughness_blur_radius=self.pbr_roughness_blur_radius,
            ao_strength=self.pbr_ao_strength,
            ao_blur_radius=self.pbr_ao_blur_radius,
            ao_bias=self.pbr_ao_bias,
        )
