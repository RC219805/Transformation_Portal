from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class Preset(str, Enum):
    """Curated looks (conservative defaults; tuned for photorealism)."""

    PHOTO_REALISTIC = "photo_realistic"
    INTERIOR_LUXURY = "interior_luxury"
    EXTERIOR_SHOWCASE = "exterior_showcase"
    ARCHITECTURAL = "architectural"
    ARCHIVAL_QUALITY = "archival_quality"


@dataclass
class SegmentationConfig:
    """Material segmentation configuration."""

    backend: str = "auto"  # auto|onnx|segformer|sam_clip|heuristic|none
    # For backend=onnx: path to ONNX model (expects NCHW float input 0..1, RGB)
    onnx_model_path: Optional[Path] = None
    onnx_labels_path: Optional[Path] = None  # optional JSON mapping class index->surface name
    # For backend=segformer: local HF directory or model id (if allow_downloads=True)
    segformer_model: Optional[str] = None
    # Segformer model revision (commit hash) for security and reproducibility
    segformer_revision: Optional[str] = None
    # For backend=sam_clip: local SAM checkpoint path
    sam_checkpoint: Optional[Path] = None

    input_long_side: int = 768  # segmentation input resolution (long side)
    soften_sigma_px: float = 2.0  # soften masks (in ORIGINAL px)
    min_confidence: float = 0.25  # suppress low-confidence masks
    allow_downloads: bool = False  # if True, may download pretrained weights


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

        # clamp some sanity
        self.upscale = 4 if int(self.upscale) not in (2, 4) else int(self.upscale)
        self.material_strength = float(max(0.0, min(1.25, self.material_strength)))
