"""Configuration module for Depth Anything 3 integration.

Provides configuration schemas, model variants, and preset definitions for
the DA3 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Literal

import numpy as np
import torch

# Import RefViewStrategy for type hints (avoid circular import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lux_depth_v3.reference_view import RefViewStrategy


class ModelLicense(Enum):
    """Model license types."""
    APACHE_2_0 = "Apache-2.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"


@dataclass
class ModelInfo:
    """Model metadata and capabilities."""
    
    name: str
    params: str
    license: ModelLicense
    huggingface_id: str
    version: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None
    
    @property
    def is_commercial(self) -> bool:
        """Check if model allows commercial use."""
        return self.license == ModelLicense.APACHE_2_0
    
    @property
    def display_name(self) -> str:
        """Get display name with version."""
        if self.version:
            return f"{self.name}-{self.version}"
        return self.name


class ModelVariant(Enum):
    """Available DA3 model variants with metadata."""
    
    # Nested models (v1.1 - recommended)
    DA3_NESTED_GIANT_LARGE_V1_1 = ModelInfo(
        name="DA3NESTED-GIANT-LARGE",
        params="1.40B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        version="1.1",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": True,
            "sky_segmentation": True,
        }
    )
    
    # Nested models (v1.0 - deprecated)
    DA3_NESTED_GIANT_LARGE = ModelInfo(
        name="DA3NESTED-GIANT-LARGE",
        params="1.40B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3NESTED-GIANT-LARGE",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": True,
            "sky_segmentation": True,
        }
    )
    
    # Any-view models (v1.1)
    DA3_GIANT_V1_1 = ModelInfo(
        name="DA3-GIANT",
        params="1.15B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-GIANT-1.1",
        version="1.1",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    DA3_LARGE_V1_1 = ModelInfo(
        name="DA3-LARGE",
        params="0.35B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-LARGE-1.1",
        version="1.1",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    # Any-view models (v1.0 - deprecated)
    DA3_GIANT = ModelInfo(
        name="DA3-GIANT",
        params="1.15B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-GIANT",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    DA3_LARGE = ModelInfo(
        name="DA3-LARGE",
        params="0.35B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-LARGE",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    # Base/Small (no versioning needed - Apache 2.0)
    DA3_BASE = ModelInfo(
        name="DA3-BASE",
        params="0.12B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3-BASE",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    DA3_SMALL = ModelInfo(
        name="DA3-SMALL",
        params="0.08B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3-SMALL",
        capabilities={
            "relative_depth": True,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    # Metric/Mono models (Apache 2.0)
    DA3_METRIC_LARGE = ModelInfo(
        name="DA3METRIC-LARGE",
        params="0.35B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3METRIC-LARGE",
        capabilities={
            "relative_depth": True,
            "metric_depth": True,
            "sky_segmentation": True,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
        }
    )
    
    DA3_MONO_LARGE = ModelInfo(
        name="DA3MONO-LARGE",
        params="0.35B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3MONO-LARGE",
        capabilities={
            "relative_depth": True,
            "metric_depth": False,
            "sky_segmentation": False,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
        }
    )
    
    # Legacy string-based variants (kept for backward compatibility)
    NESTED_GIANT_LARGE = ModelInfo(
        name="DA3NESTED-GIANT-LARGE",
        params="1.40B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3NESTED-GIANT-LARGE",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": True,
            "sky_segmentation": True,
        }
    )
    
    GIANT = ModelInfo(
        name="DA3-GIANT",
        params="1.15B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-GIANT",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": True,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    LARGE = ModelInfo(
        name="DA3-LARGE",
        params="0.35B",
        license=ModelLicense.CC_BY_NC_4_0,
        huggingface_id="depth-anything/DA3-LARGE",
        version="1.0",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    BASE = ModelInfo(
        name="DA3-BASE",
        params="0.12B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3-BASE",
        capabilities={
            "relative_depth": True,
            "pose_estimation": True,
            "pose_conditioning": True,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    SMALL = ModelInfo(
        name="DA3-SMALL",
        params="0.08B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3-SMALL",
        capabilities={
            "relative_depth": True,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
            "metric_depth": False,
            "sky_segmentation": False,
        }
    )
    
    METRIC_LARGE = ModelInfo(
        name="DA3METRIC-LARGE",
        params="0.35B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3METRIC-LARGE",
        capabilities={
            "relative_depth": True,
            "metric_depth": True,
            "sky_segmentation": True,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
        }
    )
    
    MONO_LARGE = ModelInfo(
        name="DA3MONO-LARGE",
        params="0.35B",
        license=ModelLicense.APACHE_2_0,
        huggingface_id="depth-anything/DA3MONO-LARGE",
        capabilities={
            "relative_depth": True,
            "metric_depth": False,
            "sky_segmentation": False,
            "pose_estimation": False,
            "pose_conditioning": False,
            "gaussian_splatting": False,
        }
    )
    
    @property
    def info(self) -> ModelInfo:
        """Get model metadata."""
        return self.value
    
    @classmethod
    def get_recommended(cls) -> "ModelVariant":
        """Get recommended model (latest v1.1 nested)."""
        return cls.DA3_NESTED_GIANT_LARGE_V1_1
    
    @classmethod
    def get_commercial_alternative(cls, variant: "ModelVariant") -> Optional["ModelVariant"]:
        """Get commercial-friendly alternative for NC-licensed models."""
        if variant.info.is_commercial:
            return variant
        
        # Map NC models to Apache alternatives
        mapping = {
            cls.DA3_NESTED_GIANT_LARGE_V1_1: cls.DA3_METRIC_LARGE,
            cls.DA3_NESTED_GIANT_LARGE: cls.DA3_METRIC_LARGE,
            cls.NESTED_GIANT_LARGE: cls.DA3_METRIC_LARGE,
            cls.DA3_GIANT_V1_1: cls.DA3_BASE,
            cls.DA3_GIANT: cls.DA3_BASE,
            cls.GIANT: cls.DA3_BASE,
            cls.DA3_LARGE_V1_1: cls.DA3_BASE,
            cls.DA3_LARGE: cls.DA3_BASE,
            cls.LARGE: cls.DA3_BASE,
        }
        
        return mapping.get(variant)


class InferenceMode(str, Enum):
    """Inference mode for depth estimation."""
    
    MONOCULAR = "monocular"  # Single image depth estimation
    MULTI_VIEW = "multi_view"  # Multiple views with pose estimation
    METRIC = "metric"  # Monocular metric depth (absolute scale)


class Preset(str, Enum):
    """Curated presets for common use cases."""
    
    PHOTO_REALISTIC = "photo_realistic"  # High-quality monocular depth
    INTERIOR_LUXURY = "interior_luxury"  # Interior scenes with metric depth
    EXTERIOR_SHOWCASE = "exterior_showcase"  # Exterior architectural scenes
    ARCHITECTURAL_3D = "architectural_3d"  # Multi-view reconstruction
    METRIC_SCAN = "metric_scan"  # Metric depth for measurements


class ExportFormat(str, Enum):
    """Output export formats."""
    
    PNG = "png"  # 16-bit grayscale PNG
    NPZ = "npz"  # NumPy compressed array
    PLY = "ply"  # Point cloud (ASCII)
    GLB = "glb"  # GLTF binary (3D mesh)
    TIFF = "tiff"  # 32-bit float TIFF


@dataclass
class DA3CLIConfig:
    """Configuration for DA3 CLI integration."""
    
    use_cli: bool = False
    use_backend: bool = False
    backend_url: str = "http://localhost:8008"
    backend_port: int = 8008
    backend_host: str = "127.0.0.1"
    
    # CLI-specific export formats (supports hyphen-separated combinations)
    export_format: str = "mini_npz-glb"
    
    # Reference view strategy for multi-view
    ref_view_strategy: Literal[
        "first", "middle", "saddle_balanced", "saddle_sim_range"
    ] = "saddle_balanced"
    
    # Ray-based pose estimation
    use_ray_pose: bool = False
    
    # GLB export settings
    conf_thresh_percentile: float = 40.0
    num_max_points: int = 1_000_000
    show_cameras: bool = True
    
    # Feature visualization
    feat_vis_fps: int = 15
    export_feat: str = ""  # Comma-separated layer indices


@dataclass
class DA3APIConfig:
    """Configuration for DA3 Python API.
    
    Provides comprehensive configuration for all DA3 API features including
    pose estimation, Gaussian Splatting, and multi-format export.
    """
    
    # Model selection
    model_name: str = "da3-large"
    
    # Pose alignment
    align_to_input_ext_scale: bool = True
    infer_gs: bool = False
    use_ray_pose: bool = False
    ref_view_strategy: Literal[
        "first", "middle", "saddle_balanced", "saddle_sim_range"
    ] = "saddle_balanced"
    
    # Rendering (for gs_video)
    render_exts: Optional[np.ndarray] = None
    render_ixts: Optional[np.ndarray] = None
    render_hw: Optional[Tuple[int, int]] = None
    
    # Processing
    process_res: int = 504
    process_res_method: Literal["upper_bound_resize", "lower_bound_resize"] = "upper_bound_resize"
    
    # Export formats
    export_format: str = "mini_npz"  # Can combine: "mini_npz-glb-gs_ply"
    export_feat_layers: List[int] = field(default_factory=list)
    
    # GLB export
    conf_thresh_percentile: float = 40.0
    num_max_points: int = 1_000_000
    show_cameras: bool = True
    
    # Feature visualization
    feat_vis_fps: int = 15
    
    # Additional kwargs per format
    export_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_api_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for API call."""
        return {
            "align_to_input_ext_scale": self.align_to_input_ext_scale,
            "infer_gs": self.infer_gs,
            "use_ray_pose": self.use_ray_pose,
            "ref_view_strategy": self.ref_view_strategy,
            "render_exts": self.render_exts,
            "render_ixts": self.render_ixts,
            "render_hw": self.render_hw,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "export_format": self.export_format,
            "export_feat_layers": self.export_feat_layers,
            "conf_thresh_percentile": self.conf_thresh_percentile,
            "num_max_points": self.num_max_points,
            "show_cameras": self.show_cameras,
            "feat_vis_fps": self.feat_vis_fps,
            "export_kwargs": self.export_kwargs
        }


@dataclass
class DeviceConfig:
    """Device configuration for inference."""
    
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    precision: str = "fp16"  # "fp32", "fp16", "bf16"
    use_compile: bool = False  # torch.compile optimization (PyTorch 2.0+)
    
    def resolve_device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from precision string."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(self.precision, torch.float32)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    
    resize_mode: str = "bilinear"  # "bilinear", "bicubic", "lanczos"
    normalize: bool = True  # Apply ImageNet normalization
    target_size: Optional[Tuple[int, int]] = None  # (height, width) or None for auto
    maintain_aspect: bool = True  # Maintain aspect ratio during resize
    pad_to_multiple: int = 32  # Pad to multiple of N (for efficient inference)


@dataclass
class RefinementConfig:
    """Edge-aware refinement configuration for depth maps.
    
    Research-backed post-processing techniques to improve edge fidelity
    without sacrificing depth accuracy.
    """
    
    # Enable refinement pipeline
    enable_refinement: bool = False
    
    # Refinement stages (executed in order)
    stages: List[str] = field(default_factory=lambda: ["guided", "bilateral", "edge"])
    
    # Bilateral filtering (edge-preserving smoothing)
    enable_bilateral: bool = True
    bilateral_d: int = 9  # Diameter of pixel neighborhood
    bilateral_sigma_color: float = 75.0  # Filter sigma in depth value space
    bilateral_sigma_space: float = 75.0  # Filter sigma in pixel space
    
    # Guided filter (RGB-guided edge preservation)
    enable_guided: bool = True
    guided_radius: int = 8  # Filter radius
    guided_eps: float = 0.01  # Regularization (smaller = more edge-preserving)
    
    # Edge-guided enhancement
    enable_edge: bool = True
    edge_canny_low: float = 50.0  # Canny edge detection low threshold
    edge_canny_high: float = 150.0  # Canny edge detection high threshold
    edge_blend_sigma: float = 7.0  # Gaussian blur sigma for non-edge regions
    
    # Gradient consistency filtering
    enable_gradient: bool = False
    gradient_threshold: float = 0.1  # Gradient magnitude threshold for smoothing


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration."""
    
    # Metric scaling
    apply_metric_scaling: bool = False  # Scale to metric depth
    scale_factor: float = 1.0  # Manual scale factor
    
    # Filtering
    apply_median_filter: bool = False
    median_kernel_size: int = 5
    
    apply_bilateral_filter: bool = False
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # Edge preservation
    preserve_edges: bool = True
    edge_threshold: float = 0.1
    
    # Edge-aware refinement (new)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    
    # Multi-view fusion
    fusion_mode: str = "weighted"  # "weighted", "median", "mean"


@dataclass
class ValidationConfig:
    """Quality validation configuration."""
    
    enable_validation: bool = True
    
    # Quality metrics
    compute_rmse: bool = True
    compute_delta_thresholds: bool = True  # δ < 1.25, 1.25², 1.25³
    compute_edge_completeness: bool = True
    
    # Ground truth path (if available)
    ground_truth_path: Optional[Path] = None
    
    # Quality gates
    min_delta_1: float = 0.8  # Minimum δ < 1.25 threshold
    max_rmse: float = 0.5  # Maximum RMSE


@dataclass
class ExportConfig:
    """Output export configuration."""
    
    formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.PNG])
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Point cloud export
    point_cloud_downsample: int = 1  # Downsample factor
    point_cloud_max_points: int = 1_000_000  # Max points to export
    
    # Mesh export
    mesh_simplification: float = 0.0  # 0.0-1.0, 0=no simplification
    
    # Depth map export
    depth_format: str = "uint16"  # "uint16", "float32"
    depth_scale: float = 1000.0  # Scale factor for uint16 (mm per unit)


@dataclass
class DA3Config:
    """Main configuration for DA3 pipeline."""
    
    # Model configuration
    model_variant: ModelVariant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
    inference_mode: InferenceMode = InferenceMode.MONOCULAR
    
    # Device configuration
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    # CLI integration
    cli: DA3CLIConfig = field(default_factory=DA3CLIConfig)
    
    # Python API integration
    api: DA3APIConfig = field(default_factory=DA3APIConfig)
    
    # Pipeline stages
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Cache configuration
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "lux_depth_v3")
    enable_model_cache: bool = True
    
    # Batch processing
    batch_size: int = 1
    num_workers: int = 4
    
    @classmethod
    def from_preset(cls, preset: Preset) -> DA3Config:
        """Create configuration from preset."""
        configs = {
            Preset.PHOTO_REALISTIC: cls(
                model_variant=ModelVariant.DA3_MONO_LARGE,
                inference_mode=InferenceMode.MONOCULAR,
                postprocessing=PostprocessingConfig(
                    apply_bilateral_filter=True,
                    preserve_edges=True,
                ),
            ),
            Preset.INTERIOR_LUXURY: cls(
                model_variant=ModelVariant.DA3_METRIC_LARGE,
                inference_mode=InferenceMode.METRIC,
                preprocessing=PreprocessingConfig(
                    resize_mode="bicubic",
                    target_size=(1024, 1024),
                ),
                postprocessing=PostprocessingConfig(
                    apply_metric_scaling=True,
                    preserve_edges=True,
                ),
            ),
            Preset.EXTERIOR_SHOWCASE: cls(
                model_variant=ModelVariant.DA3_LARGE_V1_1,
                inference_mode=InferenceMode.MONOCULAR,
                preprocessing=PreprocessingConfig(
                    target_size=(1024, 1024),
                ),
            ),
            Preset.ARCHITECTURAL_3D: cls(
                model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
                inference_mode=InferenceMode.MULTI_VIEW,
                postprocessing=PostprocessingConfig(
                    fusion_mode="weighted",
                ),
            ),
            Preset.METRIC_SCAN: cls(
                model_variant=ModelVariant.DA3_METRIC_LARGE,
                inference_mode=InferenceMode.METRIC,
                postprocessing=PostprocessingConfig(
                    apply_metric_scaling=True,
                    apply_median_filter=True,
                ),
                validation=ValidationConfig(
                    compute_delta_thresholds=True,
                    min_delta_1=0.85,
                ),
            ),
        }
        return configs.get(preset, cls())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_variant": self.model_variant.info.huggingface_id,
            "inference_mode": self.inference_mode.value,
            "device": {
                "device": self.device.device,
                "precision": self.device.precision,
            },
            "preprocessing": self.preprocessing.__dict__,
            "postprocessing": self.postprocessing.__dict__,
            "validation": self.validation.__dict__,
            "export": {
                "formats": [f.value for f in self.export.formats],
                "output_dir": str(self.export.output_dir),
            },
        }
