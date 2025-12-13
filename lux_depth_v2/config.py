from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

# Platform Core integration for unified configuration
try:
    from transformation_portal.core.config.schemas import DeviceConfig as CoreDeviceConfig
    from transformation_portal.core.config.schemas import PathsConfig as CorePathsConfig
    from transformation_portal.core.config.schemas import DeviceType, PrecisionType
    CORE_CONFIG_AVAILABLE = True
except ImportError:
    CORE_CONFIG_AVAILABLE = False
    CoreDeviceConfig = None
    CorePathsConfig = None
    DeviceType = None
    PrecisionType = None

if TYPE_CHECKING:
    from lux_depth_v2.materials_v2 import MaterialsV2Config


class Preset(str, Enum):
    """Curated looks (conservative defaults; tuned for photorealism)."""

    PHOTO_REALISTIC = "photo_realistic"
    INTERIOR_LUXURY = "interior_luxury"
    INTERIOR_LUXURY_MAX_QUALITY = "interior_luxury_max_quality"
    INTERIOR_LUXURY_APEX_QUALITY = "interior_luxury_apex_quality"
    INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM = "interior_luxury_apex_quality_efficientsam"  # Canary V3
    EXTERIOR_SHOWCASE = "exterior_showcase"
    EXTERIOR_POOL_APEX_QUALITY = "exterior_pool_apex_quality"
    EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM = "exterior_pool_apex_quality_efficientsam"  # Canary V3
    ARCHITECTURAL = "architectural"
    ARCHIVAL_QUALITY = "archival_quality"


class SegmentationBackend(str, Enum):
    """Material segmentation backend options (EfficientSAM V3)."""
    
    SEGFORMER = "segformer"
    EFFICIENTSAM = "efficientsam"
    FUSED = "fused"  # SegFormer + EfficientSAM edge refinement


class FusionMode(str, Enum):
    """Mask fusion modes for EfficientSAM V3."""
    
    NONE = "none"
    UNION = "union"
    INTERSECTION = "intersection"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class MaterialPropertySchema:
    """Material Property Schema for PHASE 1 Task 2.
    
    Physics-based material properties for enhanced rendering.
    Enables per-material enhancement strength and surface characteristics.
    """
    
    # Surface reflectance properties
    matte_gloss: float = 0.5  # 0=matte, 1=glossy
    specular_intensity: float = 0.5  # 0=diffuse, 1=specular
    roughness: float = 0.5  # 0=smooth, 1=rough (microfacet distribution)
    albedo: float = 0.5  # 0=dark, 1=bright (base reflectance)
    
    # Material-specific enhancement strength
    enhancement_strength: float = 1.0  # Multiplier for material response
    
    # Lighting interaction parameters
    highlight_response: float = 1.0  # How material responds to highlights
    shadow_response: float = 1.0  # How material responds to shadows
    midtone_response: float = 1.0  # How material responds to midtones
    
    # Advanced surface properties
    metalness: float = 0.0  # 0=dielectric, 1=metal (PBR)
    subsurface_scattering: float = 0.0  # 0=none, 1=full (e.g., skin, wax)
    
    @classmethod
    def wood(cls) -> 'MaterialPropertySchema':
        """Wood material preset."""
        return cls(
            matte_gloss=0.3,
            specular_intensity=0.4,
            roughness=0.6,
            albedo=0.5,
            enhancement_strength=1.0,
            highlight_response=0.8,
            shadow_response=1.1,
            midtone_response=1.0,
            metalness=0.0,
            subsurface_scattering=0.0
        )
    
    @classmethod
    def metal(cls) -> 'MaterialPropertySchema':
        """Metal material preset."""
        return cls(
            matte_gloss=0.9,
            specular_intensity=0.95,
            roughness=0.2,
            albedo=0.7,
            enhancement_strength=1.0,
            highlight_response=1.3,
            shadow_response=0.8,
            midtone_response=0.9,
            metalness=1.0,
            subsurface_scattering=0.0
        )
    
    @classmethod
    def glass(cls) -> 'MaterialPropertySchema':
        """Glass material preset."""
        return cls(
            matte_gloss=1.0,
            specular_intensity=0.85,
            roughness=0.05,
            albedo=0.8,
            enhancement_strength=0.8,
            highlight_response=1.4,
            shadow_response=0.6,
            midtone_response=0.9,
            metalness=0.0,
            subsurface_scattering=0.1
        )
    
    @classmethod
    def stone(cls) -> 'MaterialPropertySchema':
        """Stone material preset."""
        return cls(
            matte_gloss=0.2,
            specular_intensity=0.3,
            roughness=0.8,
            albedo=0.4,
            enhancement_strength=1.0,
            highlight_response=0.7,
            shadow_response=1.2,
            midtone_response=1.0,
            metalness=0.0,
            subsurface_scattering=0.0
        )
    
    @classmethod
    def fabric(cls) -> 'MaterialPropertySchema':
        """Fabric material preset."""
        return cls(
            matte_gloss=0.1,
            specular_intensity=0.2,
            roughness=0.9,
            albedo=0.5,
            enhancement_strength=0.9,
            highlight_response=0.6,
            shadow_response=1.1,
            midtone_response=1.0,
            metalness=0.0,
            subsurface_scattering=0.15
        )


@dataclass
class HybridDepthZoneConfig:
    """Hybrid Depth Zone Configuration for PHASE 1 Task 3.
    
    Combines percentile-based zones (relative) with metric-based zones (absolute).
    Enables scene-aware zone selection for optimal processing.
    """
    
    # Zone selection mode
    mode: str = "percentile"  # percentile|metric|hybrid|auto
    
    # Percentile-based zones (relative, scene-adaptive)
    fg_percentile: float = 0.35  # 0-35th percentile = foreground
    bg_percentile: float = 0.65  # 65-100th percentile = background
    
    # Metric-based zones (absolute, physically meaningful)
    close_range_m: float = 2.0  # 0-2m = close/foreground
    mid_range_m: float = 10.0  # 2-10m = midground
    far_range_m: float = 20.0  # 10-20m = background
    infinity_m: float = 1000.0  # 20m-1km+ = sky/infinity
    
    # Hybrid mode: Scene-aware zone selection
    auto_select_threshold: float = 0.7  # Confidence threshold for auto mode
    prefer_metric_outdoor: bool = True  # Use metric zones for outdoor scenes
    prefer_percentile_interior: bool = True  # Use percentile zones for interiors
    
    # Scene classification hints
    scene_type: Optional[str] = None  # interior|exterior|auto
    
    # Zone blending (smooth transitions)
    transition_blend_range: float = 0.08  # Blend range between zones
    
    def get_zones_for_scene(self, scene_type: Optional[str] = None) -> str:
        """Get optimal zone mode for scene type.
        
        Args:
            scene_type: interior|exterior|auto
            
        Returns:
            Zone mode to use (percentile|metric|hybrid)
        """
        if self.mode != "auto":
            return self.mode
        
        scene = scene_type or self.scene_type or "auto"
        
        if scene == "interior" and self.prefer_percentile_interior:
            return "percentile"
        elif scene == "exterior" and self.prefer_metric_outdoor:
            return "metric"
        else:
            return "hybrid"


@dataclass
class SegmentationConfig:
    """Material segmentation configuration."""

    backend: str = "auto"  # auto|onnx|segformer|efficientSAM|sam_clip|heuristic|none
    
    # EfficientSAM V3 backend selection
    backend_v3: SegmentationBackend = SegmentationBackend.SEGFORMER
    use_efficientsam_for_edges: bool = False  # Enable edge refinement via EfficientSAM
    
    # Fusion configuration (typed)
    fusion_mode: FusionMode = FusionMode.NONE
    fusion_min_iou: float = 0.30  # IoU gating threshold
    fusion_core_thresh: float = 0.70  # Core region threshold
    fusion_edge_low: float = 0.20  # Edge band lower threshold
    fusion_edge_high: float = 0.70  # Edge band upper threshold
    fusion_alpha_edge: float = 0.70  # Edge blending weight for EfficientSAM
    fusion_alpha_core: float = 0.30  # Core blending weight for EfficientSAM
    # For backend=onnx: path to ONNX model (expects NCHW float input 0..1, RGB)
    onnx_model_path: Optional[Path] = None
    onnx_labels_path: Optional[Path] = None  # optional JSON mapping class index->surface name
    # For backend=segformer: local HF directory or model id (if allow_downloads=True)
    # PRODUCTION DEFAULT: SegFormer-B5 (highest quality, ~339MB download)
    segformer_model: Optional[str] = "nvidia/segformer-b5-finetuned-ade-640-640"
    # Segformer model revision (commit hash) for security and reproducibility
    segformer_revision: Optional[str] = None
    # For backend=sam_clip: local SAM checkpoint path
    sam_checkpoint: Optional[Path] = None
    
    # PHASE 2: EfficientSAM backend configuration (STUB)
    # Phase 2: EfficientSAM integration (24-32h implementation)
    efficientSAM_model: Optional[str] = None  # Path to EfficientSAM checkpoint
    efficientSAM_variant: str = "s"  # s|ti|distilled (model size variant)
    efficientSAM_prompt_strategy: str = "grid"  # grid|edge_aware|adaptive

    input_long_side: int = 768  # segmentation input resolution (long side)
    soften_sigma_px: float = 2.0  # soften masks (in ORIGINAL px)
    min_confidence: float = 0.25  # suppress low-confidence masks
    allow_downloads: bool = True  # PRODUCTION: Enable downloads for SegFormer-B5


@dataclass
class OrchestratorConfig:
    """Process orchestrator configuration for Phase 1 stability."""
    enabled: bool = True
    max_workers: int = 1  # Sequential processing for GPU safety
    memory_budget_gb: Optional[float] = None  # None = no limit
    checkpoint_dir: str = ".checkpoints"
    max_retries: int = 3
    pre_flight_check: bool = True
    
    # Resource thresholds
    mps_memory_threshold_gb: float = 55.0  # 64GB - 9GB buffer
    disk_space_threshold_gb: float = 10.0
    
    # Retry strategy
    retry_backoff_base: float = 2.0
    retry_max_delay_s: float = 300.0


@dataclass
class LightingConfig:
    """Lighting condition detection and adaptation (PHASE 2 - STUB).
    
    Phase 2 Implementation (12-14h):
    - Implement lighting detection in lighting_detector.py
    - Enable adaptive tone mapping based on time of day
    - Enable adaptive color grading based on lighting condition
    """
    
    enabled: bool = False  # Feature gate (default: disabled)
    
    # Detection parameters
    use_sky_mask: bool = True  # Use material segmentation sky mask
    analyze_depth: bool = True  # Use depth map for lighting analysis
    
    # Adaptation parameters
    adapt_tone_mapping: bool = True  # Adjust tone mapping per lighting
    adapt_color_grading: bool = True  # Adjust color grading per lighting
    adaptation_strength: float = 0.7  # Blend factor [0, 1]
    
    # Time-of-day classification thresholds
    golden_hour_warmth_threshold: float = 0.5  # Warmth score to classify golden hour
    dawn_twilight_coolness_threshold: float = -0.3  # Coolness score for dawn/twilight
    
    # Color temperature adjustments per time of day (Kelvin offset)
    color_temp_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "dawn": -500.0,  # Cooler
        "golden_hour": +800.0,  # Warmer
        "twilight": -400.0,  # Cooler
        "midday": 0.0,  # Neutral
        "overcast": -200.0,  # Slightly cooler
    })


@dataclass
class Phase2Config:
    """Phase 2 performance optimization configuration."""
    
    # I/O Optimization
    async_io_enabled: bool = True
    tiff_compression: Optional[str] = 'lzw'  # 'lzw' | 'deflate' | None
    streaming_upscale: bool = True
    
    # Storage Management
    storage_internal_path: str = "."
    storage_external_t9: Optional[str] = None
    auto_migrate_large_files: bool = True
    migrate_threshold_gb: float = 2.0
    
    # Parallel Processing
    max_concurrent_workers: int = 2
    memory_budget_per_worker_gb: float = 25.0
    
    # Caching
    model_cache_enabled: bool = True
    depth_map_cache_enabled: bool = True
    cache_dir: str = '.cache'
    
    # Upscaling Optimization
    tile_based_upscaling: bool = True
    upscale_tile_size: int = 512
    upscale_overlap: int = 64
    progressive_upscaling: bool = True  # 2×2 instead of 4× for memory safety
    
    # Autotune Export Configuration (Phase 2 Slice 3)
    autotune_export: bool = False  # Default OFF
    autotune_use_complexity: bool = True


@dataclass
class ServiceConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8088
    workers: int = 1  # typical: 1 per GPU
    max_concurrency: int = 1  # serialize GPU inferences unless you know what you're doing


@dataclass
class PipelineConfig:
    """Primary pipeline settings.

    Notes:
      - Paths are optional for embedded use; CLI will populate them.
      - Values are in the spirit of V1 defaults; tuned for high-quality real estate imagery.
    """

    input_dir: Optional[Path] = None
    depth_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    preset: Preset = Preset.PHOTO_REALISTIC

    # Upscaling
    upscale: int = 4  # 2 or 4
    upscaler_backend: str = "realesrgan"  # realesrgan|onnx|none
    model_path: Optional[Path] = None     # .pth or .onnx depending on backend
    model_sha256: Optional[str] = None
    tile: int = 512
    tile_pad: int = 16
    half: bool = True

    # Device / precision
    device: str = "auto"   # auto|cuda|cpu
    precision: str = "fp16"  # fp16|fp32 (fp16 is only used on cuda; fp32 elsewhere)
    cudnn_benchmark: bool = True

    # Output
    save_master: bool = True
    save_upscaled: bool = True
    save_marketing_png: bool = True
    save_preview_jpg: bool = True
    preview_scale: float = 0.25
    
    # Marketing Export (M0+M1.1) - Benchmarked 2025-12-10
    marketing_png_compression: int = 1  # PNG compression level (0-9, default 1 for 84% speedup)

    skip_existing: bool = True
    overwrite: bool = False

    # Master switch for *any* filesystem output (images, reports, debug dumps)
    write_outputs: bool = True

    # Optional: per-stage timing accuracy on async backends (cuda/mps).
    # OFF by default because it introduces synchronization overhead.
    timing_sync_device: bool = False

    # Safety
    warn_float_gb: float = 6.0
    strict_depth: bool = False

    # Depth weight synthesis (if masks missing)
    fg_q: float = 0.35
    bg_q: float = 0.65
    transition: float = 0.08
    mask_soften_sigma: float = 4.0  # in ORIGINAL px

    # AI detail transfer (final-res)
    detail_sigma: float = 1.2
    detail_strength: float = 0.65
    detail_clip: float = 0.075
    detail_fg: float = 1.00
    detail_mid: float = 0.70
    detail_bg: float = 0.25

    # Clarity + sharpen (final-res, luma only)
    clarity_sigma: float = 2.2
    clarity_clip: float = 0.05
    clarity_fg: float = 0.18
    clarity_mid: float = 0.10
    clarity_bg: float = 0.05

    sharpen_sigma: float = 0.9
    sharpen_thresh: float = 0.004
    sharpen_fg: float = 0.08
    sharpen_mid: float = 0.05
    sharpen_bg: float = 0.03

    # Temperature / saturation / exposure / contrast grade (0..1 images)
    temp_fg: float = 0.010
    temp_mid: float = 0.004
    temp_bg: float = -0.001

    sat_fg: float = 1.030
    sat_mid: float = 1.015
    sat_bg: float = 1.000

    exp_fg: float = 1.010
    exp_mid: float = 1.000
    exp_bg: float = 0.995

    con_fg: float = 1.030
    con_mid: float = 1.020
    con_bg: float = 1.010

    soft_clip_knee: float = 0.92

    # Material response
    enable_material: bool = True
    material_strength: float = 0.75
    surfaces: Tuple[str, ...] = ("wood", "metal", "glass", "stone", "sky", "foliage")

    # Guard-rails for AI (MANDATORY in production)
    validate_ai: bool = True  # PRODUCTION: Must be True for safety
    ai_color_warn: float = 0.06
    ai_color_fail: float = 0.12
    ai_luma_warn: float = 0.06
    ai_luma_fail: float = 0.12

    # Tiling for post-processing (memory safety)
    # PRODUCTION DEFAULT: Enable tiling for UHR (324MP+) capability
    post_tile: int = 2048  # tile size at FINAL res (0 disables, 2048 for UHR support)
    post_overlap: int = 64  # pixels overlap at FINAL res (increased for quality)

    # Sub-configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    phase2: Optional[Phase2Config] = None  # Phase 2 optimizations (optional)
    
    # Materials v2 configuration (imported lazily to avoid circular dependency)
    materials_v2: Optional['MaterialsV2Config'] = None
    
    # PHASE 1 Task 2: Material Property Schema
    material_properties: Dict[str, MaterialPropertySchema] = field(default_factory=dict)
    
    # PHASE 1 Task 3: Hybrid Depth Zones
    depth_zones: HybridDepthZoneConfig = field(default_factory=HybridDepthZoneConfig)
    
    # PHASE 2: Lighting Condition Detection (STUB)
    lighting: LightingConfig = field(default_factory=LightingConfig)

    def apply_preset(self) -> None:
        """Mutate config in-place based on preset."""
        p = self.preset

        if p == Preset.PHOTO_REALISTIC:
            self.material_strength = 0.70
            self.temp_fg, self.temp_mid, self.temp_bg = 0.010, 0.003, -0.002
            self.sat_fg, self.sat_mid, self.sat_bg = 1.030, 1.015, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.025, 1.015, 1.010
            self.detail_strength = 0.65
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.18, 0.10, 0.05
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.08, 0.05, 0.03

        elif p == Preset.INTERIOR_LUXURY:
            self.material_strength = 0.90
            self.temp_fg, self.temp_mid, self.temp_bg = 0.013, 0.006, 0.000
            self.sat_fg, self.sat_mid, self.sat_bg = 1.045, 1.030, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.035, 1.030, 1.020
            self.detail_strength = 0.70
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.20, 0.12, 0.06
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.09, 0.06, 0.035
            # Production: Enable UHR tiling and enforce validation
            self.post_tile = 2048
            self.post_overlap = 64
            self.validate_ai = True

        elif p == Preset.INTERIOR_LUXURY_MAX_QUALITY:
            # MAXIMUM QUALITY: SegFormer-B5 + Materials V2 + Best Depth Visualization
            self.material_strength = 0.90
            self.temp_fg, self.temp_mid, self.temp_bg = 0.013, 0.006, 0.000
            self.sat_fg, self.sat_mid, self.sat_bg = 1.045, 1.030, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.035, 1.030, 1.020
            self.detail_strength = 0.70
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.20, 0.12, 0.06
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.09, 0.06, 0.035
            # Production: Enable UHR tiling and enforce validation
            self.post_tile = 2048
            self.post_overlap = 64
            self.validate_ai = True
            
            # MAX QUALITY Segmentation: SegFormer-B5 @ 1280px
            self.segmentation.backend = "segformer"
            self.segmentation.input_long_side = 1280
            self.segmentation.min_confidence = 0.25
            self.segmentation.allow_downloads = True
            
            # MAX QUALITY Materials V2: High thresholds + 2048px segmentation
            if self.materials_v2 is None:
                # Lazy import to avoid circular dependency
                from lux_depth_v2.materials_v2 import MaterialsV2Config
                self.materials_v2 = MaterialsV2Config()
            
            self.materials_v2.enabled = True
            self.materials_v2.confidence.confidence_threshold = 0.4
            self.materials_v2.confidence.material_thresholds = {
                "wood": 0.55,
                "metal": 0.55,
                "glass": 0.45,
                "fabric": 0.5,
                "stone": 0.55,
                "ceramic": 0.5,
                "water": 0.4,
                "polished": 0.45,
            }
            self.materials_v2.confidence.blend_range = 0.1
            self.materials_v2.confidence.blend_mode = "soft"
            self.materials_v2.confidence.fallback_strength = 0.2
            self.materials_v2.segmentation.max_segmentation_side = 2048
            self.materials_v2.segmentation.min_segmentation_side = 512
            self.materials_v2.segmentation.upsample_mode = "bicubic"
            self.materials_v2.segmentation.edge_feather_radius = 3
            self.materials_v2.segmentation.edge_feather_sigma = 1.0
            self.materials_v2.segmentation.require_high_quality = False
            self.materials_v2.segmentation.quality_threshold = 0.4

        elif p == Preset.INTERIOR_LUXURY_APEX_QUALITY:
            # ═══════════════════════════════════════════════════════════════
            # APEX QUALITY MODE - Absolute Maximum Quality
            # ═══════════════════════════════════════════════════════════════
            # Performance Impact: +40-60% slower, +50-100% VRAM, +200-300% disk
            # Quality Gain: 37-58% improvement over max_quality
            # Use Cases: Archival outputs, flagship portfolio, print materials
            # ═══════════════════════════════════════════════════════════════
            
            # Base grading (same as interior_luxury)
            self.material_strength = 0.90
            self.temp_fg, self.temp_mid, self.temp_bg = 0.013, 0.006, 0.000
            self.sat_fg, self.sat_mid, self.sat_bg = 1.045, 1.030, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.035, 1.030, 1.020
            
            # APEX: Enhanced detail transfer (+7% from max_quality)
            self.detail_strength = 0.75
            
            # APEX: Clarity/sharpening (already optimal)
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.20, 0.12, 0.06
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.09, 0.06, 0.035
            
            # APEX: Maximum Precision
            self.precision = "fp32"  # Maximum numerical precision
            self.half = False  # Disable fp16 even on CUDA
            
            # APEX: Post-Processing Quality
            self.post_tile = 2048  # UHR support with quality tiling
            self.post_overlap = 128  # +100% overlap for seamless blending
            self.validate_ai = True
            
            # APEX: Upscaling Quality
            self.tile = 1024  # +100% tile size for better quality
            self.tile_pad = 32  # +100% padding for edge quality
            self.upscale_tile_size = 2048  # Memory-efficient tiling for large images
            self.upscale_tile_overlap = 128  # Generous overlap for seamless blending
            
            # APEX: Export Quality
            self.marketing_png_compression = 0  # Lossless PNG
            
            # APEX: Maximum Segmentation Quality (PHASE 1: SegFormer-B5 Activated)
            self.segmentation.backend = "segformer"
            self.segmentation.segformer_model = "nvidia/segformer-b5-finetuned-ade-640-640"
            self.segmentation.input_long_side = 2048  # +60% resolution (vs 1280)
            self.segmentation.min_confidence = 0.15  # -40% threshold for better recall
            self.segmentation.soften_sigma_px = 2.0
            self.segmentation.allow_downloads = True
            
            # APEX: Maximum Materials V2 Quality (PHASE 1: SegFormer-B5 Activated)
            if self.materials_v2 is None:
                from lux_depth_v2.materials_v2 import MaterialsV2Config
                self.materials_v2 = MaterialsV2Config()
            
            self.materials_v2.enabled = True
            self.materials_v2.backend = "segformer"  # PHASE 1 FIX: Activate SegFormer-B5
            
            # APEX: Lower confidence thresholds for maximum coverage
            self.materials_v2.confidence.confidence_threshold = 0.3  # -25% (vs 0.4)
            self.materials_v2.confidence.material_thresholds = {
                "wood": 0.50,     # -9% for better wood coverage
                "metal": 0.50,    # -9% for better metal coverage
                "glass": 0.40,    # -11% (glass is hard to detect)
                "fabric": 0.45,   # -10% for better fabric coverage
                "stone": 0.50,    # -9% for better stone coverage
                "ceramic": 0.45,  # -10% for better ceramic coverage
                "water": 0.35,    # -12.5% (water is highly variable)
                "polished": 0.40, # -11% for polished surfaces
            }
            self.materials_v2.confidence.blend_range = 0.1
            self.materials_v2.confidence.blend_mode = "soft"
            self.materials_v2.confidence.fallback_strength = 0.2
            
            # APEX: Maximum segmentation resolution + quality enforcement
            self.materials_v2.segmentation.max_segmentation_side = 2048
            self.materials_v2.segmentation.min_segmentation_side = 512
            self.materials_v2.segmentation.upsample_mode = "bicubic"
            self.materials_v2.segmentation.edge_feather_radius = 3
            self.materials_v2.segmentation.edge_feather_sigma = 1.0
            self.materials_v2.segmentation.require_high_quality = True  # ENFORCE quality
            self.materials_v2.segmentation.quality_threshold = 0.55  # +37.5% (vs 0.4)
            
            # PHASE 1 Task 2: Material Property Schema (Physics-based properties)
            self.material_properties = {
                "wood": MaterialPropertySchema.wood(),
                "metal": MaterialPropertySchema.metal(),
                "glass": MaterialPropertySchema.glass(),
                "stone": MaterialPropertySchema.stone(),
                "fabric": MaterialPropertySchema.fabric(),
            }
            
            # PHASE 1 Task 3: Hybrid Depth Zones (Interior scene)
            self.depth_zones = HybridDepthZoneConfig(
                mode="auto",  # Automatic scene-aware selection
                fg_percentile=0.35,
                bg_percentile=0.65,
                close_range_m=2.0,
                mid_range_m=10.0,
                far_range_m=20.0,
                infinity_m=1000.0,
                scene_type="interior",  # Interior scene hint
                prefer_percentile_interior=True,
                transition_blend_range=0.08
            )

        elif p == Preset.EXTERIOR_SHOWCASE:
            self.material_strength = 0.80
            self.temp_fg, self.temp_mid, self.temp_bg = 0.006, 0.002, -0.004
            self.sat_fg, self.sat_mid, self.sat_bg = 1.055, 1.030, 1.010
            self.con_fg, self.con_mid, self.con_bg = 1.040, 1.030, 1.020
            self.detail_strength = 0.72
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.22, 0.13, 0.06
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.09, 0.06, 0.03
            # Production: Enable UHR tiling and enforce validation
            self.post_tile = 2048
            self.post_overlap = 64
            self.validate_ai = True

        elif p == Preset.EXTERIOR_POOL_APEX_QUALITY:
            # APEX quality for exterior pool/twilight scenes
            # Optimized for: water, sky, vegetation, stucco, stone
            self.material_strength = 0.95
            self.temp_fg, self.temp_mid, self.temp_bg = 0.005, 0.000, -0.008
            self.sat_fg, self.sat_mid, self.sat_bg = 1.065, 1.040, 1.020
            self.con_fg, self.con_mid, self.con_bg = 1.050, 1.035, 1.025
            self.detail_strength = 0.80
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.25, 0.16, 0.08
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.11, 0.08, 0.04
            
            # APEX: Production tiling and strict validation
            self.post_tile = 2048
            self.post_overlap = 128
            self.validate_ai = True
            self.ai_color_warn, self.ai_color_fail = 0.04, 0.08
            self.ai_luma_warn, self.ai_luma_fail = 0.04, 0.08
            
            # APEX: Maximum segmentation quality for SegFormer
            if self.segmentation is None:
                from lux_depth_v2.config import SegmentationConfig
                self.segmentation = SegmentationConfig()
            
            self.segmentation.backend = "segformer"
            self.segmentation.input_long_side = 2048  # Max resolution for pool scene
            self.segmentation.min_confidence = 0.15  # Maximum recall
            self.segmentation.soften_sigma_px = 2.5
            
            # APEX: Materials V2 with exterior-optimized settings
            if self.materials_v2 is None:
                from lux_depth_v2.materials_v2 import MaterialsV2Config
                self.materials_v2 = MaterialsV2Config()
            
            self.materials_v2.enabled = True
            self.materials_v2.backend = "segformer"
            
            # APEX: Exterior-specific thresholds (water, sky, vegetation critical)
            self.materials_v2.confidence.confidence_threshold = 0.30
            self.materials_v2.confidence.material_thresholds = {
                "wood": 0.50,
                "metal": 0.50,
                "glass": 0.40,
                "fabric": 0.45,
                "stone": 0.48,      # Critical for pool deck/columns
                "ceramic": 0.45,
                "water": 0.30,      # Critical for pool - lower threshold
                "polished": 0.38,   # For glossy surfaces
                "vegetation": 0.35, # Critical for landscaping
                "sky": 0.25,        # Critical for twilight gradient
            }
            self.materials_v2.confidence.blend_range = 0.12  # Smoother blending for sky/water
            self.materials_v2.confidence.blend_mode = "soft"
            self.materials_v2.confidence.fallback_strength = 0.25
            
            # APEX: Maximum segmentation resolution
            self.materials_v2.segmentation.max_segmentation_side = 2048
            self.materials_v2.segmentation.min_segmentation_side = 512
            self.materials_v2.segmentation.upsample_mode = "bicubic"
            self.materials_v2.segmentation.edge_feather_radius = 4  # Wider feather for exterior
            self.materials_v2.segmentation.edge_feather_sigma = 1.2
            self.materials_v2.segmentation.require_high_quality = True
            self.materials_v2.segmentation.quality_threshold = 0.55
            
            # PHASE 1 Task 2: Material Property Schema (Exterior scene)
            self.material_properties = {
                "wood": MaterialPropertySchema.wood(),
                "metal": MaterialPropertySchema.metal(),
                "glass": MaterialPropertySchema.glass(),
                "stone": MaterialPropertySchema.stone(),
                "water": MaterialPropertySchema(  # Custom water properties
                    matte_gloss=0.95,
                    specular_intensity=0.90,
                    roughness=0.15,
                    albedo=0.4,
                    enhancement_strength=1.2,
                    highlight_response=1.4,
                    shadow_response=0.9,
                    midtone_response=1.0,
                    metalness=0.0,
                    subsurface_scattering=0.3
                ),
                "vegetation": MaterialPropertySchema(  # Custom vegetation properties
                    matte_gloss=0.2,
                    specular_intensity=0.3,
                    roughness=0.8,
                    albedo=0.35,
                    enhancement_strength=1.1,
                    highlight_response=0.7,
                    shadow_response=1.2,
                    midtone_response=1.0,
                    metalness=0.0,
                    subsurface_scattering=0.4
                ),
            }
            
            # PHASE 1 Task 3: Hybrid Depth Zones (Exterior pool scene)
            self.depth_zones = HybridDepthZoneConfig(
                mode="auto",
                fg_percentile=0.30,  # Pool edge/foreground vegetation
                bg_percentile=0.70,  # Building/distant hills
                close_range_m=1.5,   # Immediate foreground
                mid_range_m=8.0,     # Pool + seating area
                far_range_m=25.0,    # Building facade
                infinity_m=5000.0,   # Distant mountains/sky
                scene_type="exterior",
                prefer_percentile_interior=False,
                transition_blend_range=0.10  # Wider for exterior depth
            )

        elif p == Preset.ARCHITECTURAL:
            self.material_strength = 0.75
            self.temp_fg, self.temp_mid, self.temp_bg = 0.008, 0.003, -0.003
            self.sat_fg, self.sat_mid, self.sat_bg = 1.020, 1.010, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.040, 1.030, 1.020
            self.detail_strength = 0.70
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.18, 0.10, 0.04
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.10, 0.07, 0.04

        elif p == Preset.ARCHIVAL_QUALITY:
            # hyper-conservative: minimal creative bias, maximal safety.
            self.material_strength = 0.60
            self.temp_fg, self.temp_mid, self.temp_bg = 0.004, 0.002, -0.002
            self.sat_fg, self.sat_mid, self.sat_bg = 1.010, 1.005, 1.000
            self.con_fg, self.con_mid, self.con_bg = 1.015, 1.010, 1.005
            self.detail_strength = 0.55
            self.clarity_fg, self.clarity_mid, self.clarity_bg = 0.14, 0.08, 0.03
            self.sharpen_fg, self.sharpen_mid, self.sharpen_bg = 0.06, 0.04, 0.02
            # Production: Enable UHR tiling and enforce validation
            self.post_tile = 2048
            self.post_overlap = 64
            self.validate_ai = True

        elif p in (Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM, 
                   Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM):
            # CANARY EfficientSAM V3 presets: APEX + FUSED segmentation
            # Inherits all settings from base APEX preset, then enables fusion
            
            # First, apply base APEX preset
            if p == Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM:
                base_preset = Preset.INTERIOR_LUXURY_APEX_QUALITY
            else:
                base_preset = Preset.EXTERIOR_POOL_APEX_QUALITY
            
            # Recursively apply base preset settings
            self.apply_preset(base_preset)
            
            # Now overlay EfficientSAM V3 fusion settings
            if self.segmentation is None:
                from lux_depth_v2.config import SegmentationConfig
                self.segmentation = SegmentationConfig()
            
            # Enable FUSED backend (SegFormer + EfficientSAM edge refinement)
            self.segmentation.backend_v3 = SegmentationBackend.FUSED
            self.segmentation.use_efficientsam_for_edges = True
            self.segmentation.fusion_mode = FusionMode.CONFIDENCE_WEIGHTED
            self.segmentation.fusion_min_iou = 0.30  # IoU gating threshold
            self.segmentation.fusion_alpha_edge = 0.70  # Prefer EfficientSAM on edges
            self.segmentation.fusion_alpha_core = 0.30  # Prefer SegFormer in core
            
            # Note: This preset will gracefully fall back to SegFormer-only if:
            # - EfficientSAM model not available
            # - onnxruntime not installed
            # - Fusion IoU gating fails

        # clamp some sanity
        self.upscale = 4 if int(self.upscale) not in (2, 4) else int(self.upscale)
        self.material_strength = float(max(0.0, min(1.25, self.material_strength)))
    
    # --- Platform Core Integration ---
    
    def get_device_config(self) -> Optional['CoreDeviceConfig']:
        """
        Get Platform Core DeviceConfig from legacy fields.
        
        Provides backward-compatible bridge to unified device configuration.
        Returns None if core module not available.
        """
        if not CORE_CONFIG_AVAILABLE:
            return None
        
        # Map legacy device string to DeviceType enum
        device_map = {
            "auto": DeviceType.AUTO,
            "cpu": DeviceType.CPU,
            "cuda": DeviceType.CUDA,
            "mps": DeviceType.MPS,
        }
        device_type = device_map.get(self.device.lower(), DeviceType.AUTO)
        
        # Map legacy precision string to PrecisionType enum
        precision_map = {
            "fp32": PrecisionType.FP32,
            "fp16": PrecisionType.FP16,
        }
        precision_type = precision_map.get(self.precision.lower(), PrecisionType.FP16)
        
        # Calculate memory fraction from warn_float_gb (assume 64GB total as baseline)
        memory_fraction = 0.85  # Default
        if hasattr(self, 'warn_float_gb'):
            # Conservative: if warning at 6GB, allow 85% of available memory
            memory_fraction = min(0.95, max(0.1, 1.0 - (self.warn_float_gb / 64.0)))
        
        return CoreDeviceConfig(
            device=device_type,
            precision=precision_type,
            enable_cudnn_benchmark=self.cudnn_benchmark,
            memory_fraction=memory_fraction,
            prefer_neural_engine=True  # Always prefer ANE on Apple Silicon
        )
    
    def get_paths_config(self) -> Optional['CorePathsConfig']:
        """
        Get Platform Core PathsConfig from legacy fields.
        
        Provides backward-compatible bridge to unified path configuration.
        Returns None if core module not available.
        """
        if not CORE_CONFIG_AVAILABLE:
            return None
        
        return CorePathsConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            cache_dir=Path('.cache'),
            checkpoint_dir=Path(self.orchestrator.checkpoint_dir) if self.orchestrator else Path('.checkpoints'),
            model_weights_dir=None  # Not used in lux_depth_v2
        )
