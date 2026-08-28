"""Lux Depth V3 configuration types and runtime defaults.

This module defines the typed configuration surface used by the CLI, portal
preview layer, orchestrator, and reproducibility metadata.
"""

from __future__ import annotations

# Check xxhash availability for default hasher selection
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ._backend_contract import normalize_backend_id, normalize_backend_sequence
from .security import HashMode

if TYPE_CHECKING:
    from .pbr import PBRConfig

_XXHASH_AVAILABLE = importlib.util.find_spec("xxhash") is not None


class ModelVariant(Enum):
    """Depth Anything V3 model variants.

    Note: DA3 models require custom library installation:
        git clone https://github.com/ByteDance-Seed/depth-anything-3
        cd depth-anything-3
        # macOS: ensure xformers is not required in default dependencies
        pip install -e .

    ⚠️  LICENSE: DA3NESTED-GIANT-LARGE-1.1 is
    CC BY-NC 4.0 (non-commercial use only).
    """

    METRIC_LARGE = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-large",
            "display_name": "Depth Anything V3 Research Default (DA3 Nested Giant Large)",
            "huggingface_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        },
    )()
    METRIC_BASE = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-base",
            "display_name": "Depth Anything V3 Base Compatibility Selector",
            "huggingface_id": "depth-anything/DA3-BASE",
        },
    )()
    METRIC_SMALL = type(
        "ModelVariantValue",
        (),
        {
            "name": "depth-anything-v3-metric-small",
            "display_name": "Depth Anything V3 Small Compatibility Selector",
            "huggingface_id": "depth-anything/DA3-SMALL",
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
    # Enable FP16 for MPS/CUDA
    # (1.3-1.5x speedup, 2x memory reduction)
    use_fp16: bool = True
    # Phase 3: CoreML ANE acceleration for
    # Apple Silicon (5x speedup, opt-in)
    use_coreml: bool = False


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
    model_key: Optional[str] = None
    raw_model_id: Optional[str] = None
    non_commercial_ok: bool = False
    # Authoritative single-resolution model contract (P0-1, issue #2065),
    # injected by DA3Backend so the inference engine consumes it instead of
    # re-resolving. Typed Any to keep this module import-light; concrete
    # type is model_resolution.ResolvedModel.
    resolved_model_contract: Optional[Any] = None
    device: DeviceConfig = field(default_factory=DeviceConfig)
    postprocessing: PostprocessingConfig = field(
        default_factory=PostprocessingConfig,
    )

    @classmethod
    def from_preset(cls, preset: Preset) -> DA3Config:
        """Create configuration from preset.

        Presets provide different quality/performance tradeoffs:

        * ``ARCHITECTURAL_INTERIOR``:
          High quality for interior architectural renders.
        * ``ARCHITECTURAL_EXTERIOR``: Balanced for exterior scenes.
        * ``LUXURY_ESTATE``: Premium quality for luxury real estate.
        * ``DEFAULT``: Standard balanced configuration.
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
    model_key: Optional[str] = None
    raw_model_id: Optional[str] = None
    preset: Optional[Preset] = None
    # Raw preset string from CLI/user input
    # (captured even when preset does not map
    # to Preset enum)
    preset_requested: Optional[str] = None
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
    # Fail if depth artifacts found in input
    # directory (validation mode)
    strict_inputs: bool = False
    # Preserve the Materials V3 intermediate TIFF/PNG in
    # <output>/temp/*_materials_v3_enhanced.* after V2 runs. Use this to
    # bisect a banding / color regression: compare the Materials V3
    # intermediate against the final V2 output without re-running the
    # pipeline with --enable-v2 off.
    keep_intermediates: bool = False

    # License acceptance flags (for research-only models)
    # Apple AMLR license for Depth Pro
    accept_apple_depth_pro_research_license: bool = False
    # Umbrella flag for APEX Research Ultra
    # (ADR-026)
    accept_research_tools_license: bool = False

    # Spatial AI Foundation (ADR-026 Phase I)
    # Enable linear light preservation
    # (float32, gamma=1.0)
    spatial_ai_linear_ingest: bool = False
    # RAW decode policy: auto, force_rawpy,
    # force_preview (debug env-gated)
    raw_ingest_mode: str = "auto"
    # RAW white-balance mode
    # (legacy_linear_srgb: "camera" only)
    raw_wb_mode: str = "camera"
    # RAW demosaic algorithm
    # (legacy_linear_srgb: "AHD" only)
    raw_demosaic: str = "AHD"

    # Depth backend selection
    # None = auto (DA3), "depth_pro",
    # or "ensemble"
    depth_backend: Optional[str] = None
    # Path to depth_pro.pt checkpoint
    depth_pro_checkpoint_path: Optional[str] = None
    # Optional Python executable for a
    # dedicated Depth Pro environment
    depth_pro_python_executable: Optional[str] = None
    # Optional Python executable for a
    # dedicated RAW ingest environment
    raw_python_executable: Optional[str] = None
    # Optional Python executable for a
    # dedicated DA3 / depth-anything-3 environment
    da3_python_executable: Optional[str] = None
    # Timeout for DA3 subprocess calls
    # (readiness + inference)
    da3_subprocess_timeout_seconds: int = 900

    # Fallback configuration. Operator-facing values: "fail", "skip", "v2-auto",
    # "apex-strict". The "apex-strict" sentinel canonicalizes to "fail" and
    # suppresses the APEX tier's auto-upgrade to "v2-auto" — use it when an
    # operator explicitly wants fail-closed depth on APEX.
    depth_fallback: str = "fail"
    v2_timeout: int = 300
    # Allow synthetic depth backend when no
    # ML deps (test/CI only)
    allow_synthetic_fallback: bool = False
    # Allow backend fallback after APEX
    # semantic-gate failures
    allow_semantic_fallback: bool = False
    depth_operational_fallback_chain: Tuple[str, ...] = ("da3", "da2")

    # Hash mode
    hash_mode: HashMode = HashMode.IF_MANIFEST_EXISTS

    # Performance optimizations (Phase 1)
    # LRU cache for manifest loading
    # (15-20% I/O reduction)
    enable_manifest_cache: bool = True
    # Chunked SHA-256 for large files
    # (90% memory reduction)
    chunked_hashing: bool = True

    # Performance optimizations (Phase 2:
    # Parallelization)
    # Parallel I/O for batch workflows
    # (3-5x throughput)
    enable_parallel_processing: bool = True
    # Auto-detect if None
    # (default: cpu_count - 1)
    max_parallel_workers: Optional[int] = None
    # Content-addressable depth cache
    # (opt-in, requires storage)
    enable_depth_cache: bool = False
    # Maximum cache size before LRU eviction
    depth_cache_max_size_gb: float = 10.0

    # Performance optimizations (Phase 3:
    # Advanced optimizations)
    # CoreML ANE for Apple Silicon
    # (5x depth inference, requires conversion)
    use_coreml_backend: bool = False
    # GPU-accelerated PBR map batching
    # (30% speedup, opt-in)
    enable_pbr_gpu_batching: bool = False
    # MessagePack binary format
    # (60% smaller, 3x faster, less readable)
    use_msgpack_manifests: bool = False
    # xxHash for output keys
    # (5x faster than SHA-1, auto-enabled
    # when available)
    use_xxhash: bool = field(
        default_factory=lambda: _XXHASH_AVAILABLE,
    )

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
    # Additional saturation tolerance applied only
    # when gate normalization scales metric depth
    # via percentile_1_99.
    apex_depth_scaled_saturation_margin: float = 0.0025
    # Warning-only grace band for isolated
    # low-end saturation when the structural
    # APEX depth checks otherwise pass.
    apex_depth_low_saturation_warning_band: float = 0.0075
    apex_depth_saturation_high_value: float = 0.999
    apex_depth_saturation_low_value: float = 0.001
    apex_depth_min_gradient_energy: float = 5e-4
    # Numeric epsilon used for threshold comparisons
    # to avoid edge-case float jitter.
    apex_depth_threshold_epsilon: float = 1e-6
    apex_depth_hist_bins: int = 64

    # Materials V3 surface-aware finishing
    enable_materials_v3: bool = False
    # Apply pixel operations in Materials V3
    # (requires enable_materials_v3=True)
    apply_pixel_ops: bool = True

    # Materials V3 configuration
    # EfficientSAM refinement strategy
    # (canary, disabled)
    refinement_strategy: str = "canary"
    # Minimum material coverage in pixels
    min_coverage_px: int = 500
    # Minimum mean confidence for material
    # detection
    min_mean_conf: float = 0.2
    # Enable glass material response
    glass_response_enabled: bool = True

    # Materials V3 Pixel Ops - Feathering
    # Configuration (A3)
    # Default Gaussian blur sigma for
    # mask feathering
    mask_feather_sigma_default: float = 3.0
    # Material-specific overrides
    mask_feather_sigma_overrides: Dict[str, float] = field(
        default_factory=dict,
    )
    # Materials with feathering disabled
    mask_feather_disabled_materials: list[str] = field(default_factory=list)

    # Materials V3 Pixel Ops - Seam-safe guard for large low-texture
    # materials (sky, water, smooth walls).
    # Gradient-energy threshold below which an ROI counts as "flat".
    # Measured on normalized [0, 1] luminance.
    pixel_ops_low_grad_threshold: float = 0.01
    # Minimum bbox fraction (relative to whole image) required for the guard
    # to engage. Small flat regions don't produce visible panels.
    pixel_ops_low_tex_min_bbox_frac: float = 0.05
    # Multiplier applied to the per-material feather sigma when the guard
    # fires. A 3 px feather becomes 24 px, pushing the mask transition into
    # a scale the eye cannot parse as a hard seam.
    pixel_ops_low_tex_feather_multiplier: float = 8.0
    # p99 ceiling on per-pixel |delta| (normalized [0, 1]) applied to the
    # pre/post-op difference on a large flat ROI. Any op that exceeds this
    # is soft-clamped proportionally so it cannot produce a visible
    # luminance step across the mask boundary.
    pixel_ops_low_tex_delta_ceiling: float = 0.04

    # Materials V3 Phase B - Sky Bootstrap
    # Top fraction of image to consider for
    # sky (default: upper 50%)
    sky_top_region_fraction: float = 0.5
    # Maximum gradient magnitude for smooth
    # sky regions
    sky_gradient_threshold: float = 0.05
    # Minimum brightness threshold for sky
    sky_brightness_threshold: float = 0.4

    # Materials V3 segmentation backend (Phase 3)
    # Enable automatic material segmentation
    enable_material_segmentation: bool = False
    # Options: stub, efficientsam, sam2, sam_vit_h
    material_segmentation_backend: str = "stub"
    # If True, raise on backend errors
    # instead of falling back to stub
    strict_backend: bool = False
    # Exact-result segmentation cache policy for real segmentation runs.
    # "off" disables cache lookup/write; "read_write" reuses validated masks
    # and records cache provenance in Materials V3 metadata.
    material_segmentation_cache_policy: str = "read_write"
    # SAM2 variant when backend="sam2":
    # base or large
    sam2_model_size: str = "base"
    # Optional SAM2 checkpoint override path
    sam2_checkpoint_path: Optional[str] = None
    # Optional SAM2 config override. None falls through to the pinned backend
    # default for the selected model size.
    sam2_model_config: Optional[str] = None
    # Optional SAM2 checkpoint integrity override. None falls through to the
    # pinned backend default for the selected model size.
    sam2_expected_sha256: Optional[str] = None
    # Explicit SAM2 tiling controls for large-image segmentation
    sam2_tiling_enabled: bool = False
    sam2_tile_size_px: int = 1536
    sam2_overlap_px: int = 256
    sam2_global_pass_longest_side: int = 1280
    sam2_max_concurrency: int = 1
    sam2_points_per_side: int = 32
    sam2_points_per_batch: int = 64
    sam2_pred_iou_thresh: float = 0.88
    sam2_stability_score_thresh: float = 0.85
    sam2_crop_n_layers: int = 1
    # SAM ViT-H backend (Phase 2, APEX Research tier)
    sam_vit_h_checkpoint_path: Optional[str] = None
    sam_vit_h_points_per_side: int = 32
    sam_vit_h_pred_iou_thresh: float = 0.88
    sam_vit_h_confidence_threshold: float = 0.85
    # Optional SHA-256 hex digest override for checkpoint integrity validation.
    # None falls through to SAMVitHBackend.EXPECTED_SHA256, the pinned canonical
    # SAM ViT-H release hash, so the runtime is fail-closed by default. Set this
    # explicitly only when loading an approved fine-tuned or custom checkpoint
    # whose digest differs from the canonical Meta release.
    sam_vit_h_expected_sha256: Optional[str] = None

    # Emit flags (deliverables)
    emit_master16: bool = False  # Emit master 16-bit output
    emit_upscaled16: bool = False  # Emit upscaled 16-bit output
    emit_marketing: bool = False  # Emit marketing-ready output
    emit_report: bool = True  # Emit processing report
    emit_run_card: bool = True  # Emit run card for reproducibility
    run_card_version: str = "v1"  # v1 legacy commitment or v2 transparency tree
    run_card_include_proofs: bool = False  # Opt-in per-artifact inclusion proofs for v2 run cards

    # Optional advisory VLM captioning sidecar. This is default-off and MUST NOT
    # participate in quality gates.
    vlm_captioning_enabled: bool = False
    vlm_captioning_backend: str = "fastvlm"
    vlm_captioning_model: str = "default"
    vlm_captioning_proxy_format: str = "png"
    vlm_captioning_max_side_px: int = 1600
    fastvlm_python_executable: Optional[str] = None
    fastvlm_mlx_vlm_dir: Optional[str] = None
    fastvlm_timeout_seconds: int = 180

    # Phase B1: optional scene-level reconstruction (off by default)
    enable_reconstruction: bool = False
    grouping_mode: str = "single"  # Options: single, parent_dir
    # Path to tp.scene_cameras.v1 sidecar JSON
    cameras_sidecar_path: Optional[str] = None
    reconstruction_iterations: int = 1000
    reconstruction_tier: str = "apex_research"
    # Emit per-scene debug bundle for
    # reconstruction triage
    emit_scene_debug_bundle: bool = False

    # Authoritative single-resolution invocation (P0-1, issue #2065).
    # Set by the CLI after the shared plan/run resolution pass; consumers
    # (ConfigResolver, depth backends) read the model contract from it via
    # resolved_invocation.authoritative_model_contract() instead of
    # re-resolving from compatibility fields. Typed as Any to keep this
    # module import-light; the concrete type is
    # lux_depth_v3.resolved_invocation.ResolvedInvocation.
    resolved_invocation: Optional[Any] = None

    def __post_init__(self) -> None:
        """Normalize backend identifiers and compatibility fields."""
        self.depth_backend = normalize_backend_id(
            self.depth_backend,
            warn=True,
            warning_context="EnhanceConfig.depth_backend",
        )
        self.depth_operational_fallback_chain = normalize_backend_sequence(
            self.depth_operational_fallback_chain or ("da3", "da2"),
            warn=True,
            warning_context="EnhanceConfig.depth_operational_fallback_chain",
        )
        self.apex_depth_scaled_saturation_margin = max(
            float(self.apex_depth_scaled_saturation_margin),
            0.0,
        )
        self.apex_depth_low_saturation_warning_band = max(
            float(self.apex_depth_low_saturation_warning_band),
            0.0,
        )
        self.apex_depth_threshold_epsilon = max(
            float(self.apex_depth_threshold_epsilon),
            0.0,
        )

        # Normalize and validate ``depth_fallback`` before any policy branch
        # reads it. Callers that bypass ``security.validate_depth_fallback``
        # (e.g. constructing ``EnhanceConfig`` directly with casing/whitespace
        # variants like "APEX-STRICT" or " apex-strict ") still get the same
        # sentinel handling and the same fail-fast on bad values.
        from .security import validate_depth_fallback

        if isinstance(self.depth_fallback, str):
            self.depth_fallback = validate_depth_fallback(self.depth_fallback) or "fail"
        else:
            # Non-string values are an outright contract violation; surface
            # before silently accepting them.
            raise ValueError(f"depth_fallback must be a string; got {type(self.depth_fallback).__name__}")

        # `apex-strict` is an operator-facing sentinel that opts out of the APEX
        # tier auto-upgrade below. Canonicalize to "fail" so all downstream
        # branches (orchestrator runtime, manifest serializer, validator) only
        # ever see the documented {fail, skip, v2-auto} value set.
        apex_strict_explicit = self.depth_fallback == "apex-strict"
        if apex_strict_explicit:
            self.depth_fallback = "fail"

        # APEX tier auto-upgrade: when DA3 + DA2 both fail their depth gates on
        # genuinely flat scenes (e.g. uniform sky / glare), recover via the V2
        # stage with independent depth instead of failing the batch. Skip the
        # upgrade when the operator explicitly chose `apex-strict`.
        if (
            not apex_strict_explicit
            and str(getattr(self, "quality_tier", "")).strip().lower() == "apex"
            and self.depth_fallback == "fail"
        ):
            self.depth_fallback = "v2-auto"

    @property
    def enable_pbr(self) -> bool:
        """Alias for generate_pbr (backward compatibility)."""
        return self.generate_pbr

    def to_pbr_config(self) -> PBRConfig:
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
