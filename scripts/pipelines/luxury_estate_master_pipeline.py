#!/usr/bin/env python3
"""
Luxury Estate Master Pipeline - 750 Picacho
============================================

Cutting-edge HDR processing pipeline for luxury real estate architectural images.
Combines 7 advanced processing stages optimized for 32-bit TIFF HDR sources.

Pipeline Stages:
1. HDR Precision Loader - 32-bit TIFF with alpha channel preservation
2. Depth Anything V2 - CoreML/MPS accelerated depth estimation
3. Material Response - Physics-based surface enhancement
4. Intelligent Tone Mapping - AgX/Filmic/Reinhard HDR-to-display
5. Location Color Grading - LUT stacks for California coastal aesthetic
6. AI Enhancement - ControlNet + SDXL refinement
7. Real-ESRGAN 4x - Ultra-high resolution upscaling

Optimized for:
- Apple Silicon (M-series) with Metal Performance Shaders
- 32-bit HDR TIFF sources (preserves full dynamic range)
- Batch processing of architectural image sets
- Complete metadata preservation (IPTC, XMP, GPS)

Author: Transformation Portal
Version: 1.0.0
Date: 2025-11-10
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Conditional imports with graceful degradation
try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.rrdbnet_arch import RRDBNet

    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    logging.warning("Real-ESRGAN not available - upscaling will use Lanczos")

try:
    from controlnet_aux import CannyDetector
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False
    logging.warning("AI enhancement not available - will skip ControlNet/SDXL stage")

try:
    from transformation_portal.depth.models import DepthAnythingV2Model, ModelBackend, ModelVariant
    from transformation_portal.depth.processors import (
        AtmosphericEffects,
        DepthAwareDenoise,
        DepthGuidedFilters,
        ZoneToneMapping,
    )

    DEPTH_PIPELINE_AVAILABLE = True
except ImportError:
    DEPTH_PIPELINE_AVAILABLE = False
    logging.warning("Depth pipeline not available - will skip depth-aware processing")

try:
    from transformation_portal.processors.material_response.core import LightingProfile, MaterialAestheticProfile

    MATERIAL_RESPONSE_AVAILABLE = True
except ImportError:
    MATERIAL_RESPONSE_AVAILABLE = False
    logging.warning("Material Response not available - will use simplified enhancement")

from tonemapper_agx_filmic import apply_agx_ocio, apply_filmic_hable, linear_to_srgb

LOG_PATH = Path(os.environ.get("TP_LUXURY_ESTATE_PIPELINE_LOG", "luxury_estate_pipeline.log"))
if LOG_PATH.parent != Path("."):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class DepthConfig:
    """Depth processing configuration."""

    enabled: bool = True
    quality_mode: str = "fast"  # fast (V2-Small) or premium (V2-Large)
    model_variant: str = "small"  # small, base, large (manual override)
    backend: str = "pytorch_mps"  # pytorch_mps, coreml, pytorch_cpu
    num_zones: int = 4
    zone_tone_method: str = "agx"
    atmospheric_haze: bool = False  # True for aerials, False for interiors
    haze_density: float = 0.02
    clarity_strength: float = 0.5
    auto_download_models: bool = True  # Auto-download depth models if missing


@dataclass
class MaterialResponseConfig:
    """Material Response configuration."""

    enabled: bool = True
    strength: float = 0.75
    preserve_highlights: bool = True
    enhance_wood: bool = True
    enhance_metal: bool = True
    enhance_glass: bool = True
    enhance_stone: bool = True
    enhance_textiles: bool = True


@dataclass
class ToneMappingConfig:
    """HDR tone mapping configuration."""

    method: str = "filmic"  # agx, filmic, reinhard
    exposure: float = 0.0
    contrast: float = 1.05
    preserve_hdr_highlights: bool = True
    white_point: float = 11.2
    agx_config_path: Optional[str] = None
    adaptive_tone_mapping: bool = True  # Enable scene-aware tone mapping
    shadow_boost_outdoor: float = 0.3  # Shadow lift for outdoor scenes (0.0-1.0)
    use_zone_based_mapping: bool = True  # Use depth zones for tone mapping


@dataclass
class ColorGradingConfig:
    """Color grading and LUT configuration."""

    enabled: bool = True
    lut_stack: List[Tuple[str, float]] = field(
        default_factory=lambda: [
            ("assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube", 0.70),
            ("assets/luts/film_emulation/Kodak/Kodak_2393_D55_HDR.cube", 0.50),
        ]
    )
    temperature_shift: float = 0.0  # -100 to +100
    tint_shift: float = 0.0  # -100 to +100
    saturation: float = 1.08
    vibrance: float = 0.15


@dataclass
class AIEnhancementConfig:
    """AI enhancement configuration."""

    enabled: bool = True
    model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/sd-controlnet-canny"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.30
    seed: int = 42
    prompt_template: str = (
        "luxury {room_type} architectural photography, {style}, ultra detailed, professional, photorealistic, 8k"
    )
    negative_prompt: str = "blurry, artifacts, cartoon, painting, oversaturated, unrealistic, low quality, distorted"
    ai_enhancement_padding: bool = True  # Auto-pad for tensor compatibility
    target_size_multiple: int = 64  # Pad to multiples of this value


@dataclass
class UpscalingConfig:
    """Upscaling configuration."""

    enabled: bool = True
    method: str = "esrgan"  # esrgan, lanczos
    scale_factor: float = 4.0
    model_path: str = "weights/RealESRGAN_x4plus.pth"
    tile_size: int = 512
    tile_padding: int = 10


@dataclass
class OutputConfig:
    """Output configuration."""

    save_master_tiff: bool = True
    save_delivery_jpeg: bool = True
    save_intermediate_stages: bool = True
    master_bit_depth: int = 16  # 16 or 32
    jpeg_quality: int = 95
    output_dir: str = "output_luxury_estate"


@dataclass
class PipelinePreset:
    """Complete pipeline preset configuration."""

    name: str
    description: str
    depth: DepthConfig = field(default_factory=DepthConfig)
    material_response: MaterialResponseConfig = field(default_factory=MaterialResponseConfig)
    tone_mapping: ToneMappingConfig = field(default_factory=ToneMappingConfig)
    color_grading: ColorGradingConfig = field(default_factory=ColorGradingConfig)
    ai_enhancement: AIEnhancementConfig = field(default_factory=AIEnhancementConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ============================================================================
# Preset Definitions
# ============================================================================


def get_750_picacho_preset() -> PipelinePreset:
    """Get optimized preset for 750 Picacho luxury estate."""
    return PipelinePreset(
        name="750 Picacho Elite",
        description="Montecito coastal estate - full HDR processing with AI enhancement",
        depth=DepthConfig(
            enabled=True,
            quality_mode="premium",  # Phase 2: Use V2-Large for 750 Picacho
            model_variant="large",  # Phase 2: Premium quality
            backend="pytorch_mps",
            num_zones=4,
            zone_tone_method="filmic",
            atmospheric_haze=False,
            clarity_strength=0.55,
            auto_download_models=True,
        ),
        material_response=MaterialResponseConfig(
            enabled=True,
            strength=0.75,
            preserve_highlights=True,
        ),
        tone_mapping=ToneMappingConfig(
            method="filmic",
            exposure=0.0,
            contrast=1.05,
            white_point=11.2,
            adaptive_tone_mapping=True,
            shadow_boost_outdoor=0.3,
            use_zone_based_mapping=True,
        ),
        color_grading=ColorGradingConfig(
            enabled=True,  # Re-enabled - LUTs are good!
            lut_stack=[
                ("assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube", 0.70),
                ("assets/luts/film_emulation/Kodak/Kodak_2393_D55_HDR.cube", 0.50),
            ],
            saturation=1.08,
            vibrance=0.15,
        ),
        ai_enhancement=AIEnhancementConfig(
            enabled=False,  # Temporarily disabled - ControlNet tensor size issue
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.30,
            ai_enhancement_padding=True,
            target_size_multiple=64,
        ),
        upscaling=UpscalingConfig(
            enabled=True,  # Enabled with Lanczos (color-neutral)
            method="lanczos",  # Changed from "esrgan" - fixes blue cast
            scale_factor=4.0,
        ),
        output=OutputConfig(
            save_master_tiff=True,
            save_delivery_jpeg=True,
            save_intermediate_stages=True,
            master_bit_depth=16,
            jpeg_quality=95,
        ),
    )


def get_aerial_preset() -> PipelinePreset:
    """Get preset optimized for aerial photography."""
    preset = get_750_picacho_preset()
    preset.name = "750 Picacho Aerial"
    preset.description = "Aerial photography with atmospheric effects"
    preset.depth.atmospheric_haze = True
    preset.depth.haze_density = 0.03
    # Aerial scenes typically need more shadow boost
    preset.tone_mapping.shadow_boost_outdoor = 0.4
    preset.ai_enhancement.prompt_template = "luxury coastal estate aerial photography, dramatic hillside architecture, infinity pool, mediterranean landscaping, golden hour lighting, ultra detailed, professional, photorealistic, 8k"
    return preset


# ============================================================================
# Pipeline Implementation
# ============================================================================


class LuxuryEstateMasterPipeline:
    """
    Production-ready HDR processing pipeline for luxury real estate.

    Optimized for 32-bit TIFF HDR sources with complete workflow integration.
    """

    def __init__(self, preset: PipelinePreset, device: Optional[torch.device] = None):
        """
        Initialize pipeline with preset configuration.

        Args:
            preset: Pipeline configuration preset
            device: PyTorch device (auto-detected if None)
        """
        self.preset = preset
        self.device = device or self._detect_device()
        self.stats = {
            "images_processed": 0,
            "total_time": 0.0,
            "stage_times": {},
        }

        logger.info(f"Initializing {preset.name}")
        logger.info(f"Device: {self.device}")

        # Initialize components
        self._init_depth_model()
        self._init_ai_models()
        self._init_upscaler()

    def _detect_device(self) -> torch.device:
        """Auto-detect optimal device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _auto_download_depth_model(self):
        """Auto-download Depth Anything V2 model if not available."""
        if not self.preset.depth.auto_download_models:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            logger.info("Checking Depth Anything V2 model availability...")

            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            if self.preset.depth.model_variant == "base":
                model_id = "depth-anything/Depth-Anything-V2-Base-hf"
            elif self.preset.depth.model_variant == "large":
                model_id = "depth-anything/Depth-Anything-V2-Large-hf"

            # This will auto-download if not cached
            _ = AutoImageProcessor.from_pretrained(model_id)
            _ = AutoModelForDepthEstimation.from_pretrained(model_id)
            logger.info(f"✓ Depth Anything V2 model ready: {model_id}")
        except ImportError:
            logger.warning("transformers library not available for auto-download")
        except Exception as e:
            logger.warning(f"Could not auto-download depth model: {e}")

    def _detect_scene_type(self, image_linear: np.ndarray) -> str:
        """
        Detect if scene is outdoor or indoor based on luminance distribution.

        Args:
            image_linear: Scene-linear image

        Returns:
            'outdoor' or 'indoor'
        """
        # Calculate luminance
        luminance = np.dot(image_linear, [0.2126, 0.7152, 0.0722])

        # Outdoor scenes typically have:
        # 1. Higher dynamic range
        # 2. More pixels in extreme luminance values
        # 3. Different histogram distribution

        # Calculate dynamic range (ratio of 99th to 1st percentile)
        p99 = np.percentile(luminance, 99)
        p01 = np.percentile(luminance, 1)
        dynamic_range = p99 / (p01 + 1e-6)

        # Count pixels in shadow (< 0.1) and highlight (> 0.7) regions
        shadow_pixels = np.sum(luminance < 0.1) / luminance.size
        highlight_pixels = np.sum(luminance > 0.7) / luminance.size

        # Outdoor heuristic: high DR + significant shadow/highlight regions
        is_outdoor = (dynamic_range > 8.0) or (shadow_pixels > 0.15 and highlight_pixels > 0.1)

        scene_type = "outdoor" if is_outdoor else "indoor"
        logger.info(
            f"  → Scene detection: {scene_type.upper()} (DR={dynamic_range:.1f}x, shadows={shadow_pixels*100:.1f}%, highlights={highlight_pixels*100:.1f}%)"
        )

        return scene_type

    def _pad_for_controlnet(self, image: np.ndarray, multiple: int = 64) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Pad image to make dimensions compatible with ControlNet.

        Args:
            image: Input image array
            multiple: Pad to multiples of this value

        Returns:
            Tuple of (padded_image, (top, bottom, left, right) padding amounts)
        """
        h, w = image.shape[:2]

        # Calculate target dimensions (next multiple)
        target_h = ((h + multiple - 1) // multiple) * multiple
        target_w = ((w + multiple - 1) // multiple) * multiple

        # Calculate padding
        pad_h = target_h - h
        pad_w = target_w - w

        # Distribute padding (prefer bottom/right)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Apply padding (reflect mode preserves edges)
        if len(image.shape) == 3:
            padded = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        else:
            padded = np.pad(image, ((top, bottom), (left, right)), mode="reflect")

        logger.info(f"  → Padded {w}x{h} → {target_w}x{target_h} for ControlNet compatibility")

        return padded, (top, bottom, left, right)

    def _unpad_image(self, image: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove padding added by _pad_for_controlnet."""
        top, bottom, left, right = padding
        h, w = image.shape[:2]

        # Crop to original size
        if bottom == 0 and right == 0:
            cropped = image[top:, left:]
        elif bottom == 0:
            cropped = image[top:, left:-right]
        elif right == 0:
            cropped = image[top:-bottom, left:]
        else:
            cropped = image[top:-bottom, left:-right]

        return cropped

    def _init_depth_model(self):
        """Initialize depth estimation model."""
        if not self.preset.depth.enabled or not DEPTH_PIPELINE_AVAILABLE:
            self.depth_model = None
            if self.preset.depth.enabled:
                # Try auto-download if enabled
                self._auto_download_depth_model()
            return

        try:
            variant_map = {
                "small": ModelVariant.SMALL,
                "base": ModelVariant.BASE,
                "large": ModelVariant.LARGE,
            }
            backend_map = {
                "pytorch_cpu": ModelBackend.PYTORCH_CPU,
                "pytorch_mps": ModelBackend.PYTORCH_MPS,
                "coreml": ModelBackend.COREML,
            }

            # Phase 2 upgrade: Support quality_mode selector
            # quality_mode overrides model_variant if set
            model_variant = self.preset.depth.model_variant
            if hasattr(self.preset.depth, "quality_mode"):
                if self.preset.depth.quality_mode == "premium":
                    model_variant = "large"
                    logger.info("Using premium quality mode: V2-Large")
                elif self.preset.depth.quality_mode == "fast":
                    model_variant = "small"
                    logger.info("Using fast quality mode: V2-Small")

            self.depth_model = DepthAnythingV2Model(
                variant=variant_map[model_variant],
                backend=backend_map.get(self.preset.depth.backend),
                precision="fp16",
            )
            logger.info(f"Loaded Depth Anything V2 ({model_variant}, {self.preset.depth.backend})")
        except Exception as e:
            logger.warning(f"Failed to load depth model: {e}")
            # Try auto-download
            self._auto_download_depth_model()
            self.depth_model = None

    def _init_ai_models(self):
        """Initialize AI enhancement models (ControlNet + SDXL)."""
        if not self.preset.ai_enhancement.enabled or not AI_ENHANCEMENT_AVAILABLE:
            self.controlnet = None
            self.ai_pipeline = None
            return

        try:
            logger.info("Loading ControlNet...")
            self.controlnet = ControlNetModel.from_pretrained(
                self.preset.ai_enhancement.controlnet_id, torch_dtype=torch.float32
            ).to(self.device)

            logger.info("Loading Stable Diffusion pipeline...")
            self.ai_pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.preset.ai_enhancement.model_id, controlnet=self.controlnet, torch_dtype=torch.float32, safety_checker=None
            ).to(self.device)
            self.ai_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.ai_pipeline.scheduler.config)

            self.canny_detector = CannyDetector()
            logger.info("AI enhancement models ready")
        except Exception as e:
            logger.warning(f"Failed to load AI models: {e}")
            self.controlnet = None
            self.ai_pipeline = None

    def _init_upscaler(self):
        """Initialize Real-ESRGAN upscaler."""
        if not self.preset.upscaling.enabled or not ESRGAN_AVAILABLE:
            self.upscaler = None
            return

        if self.preset.upscaling.method != "esrgan":
            self.upscaler = None
            return

        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            self.upscaler = RealESRGANer(
                scale=int(self.preset.upscaling.scale_factor),
                model_path=self.preset.upscaling.model_path,
                model=model,
                tile=self.preset.upscaling.tile_size,
                tile_pad=self.preset.upscaling.tile_padding,
                pre_pad=0,
                half=False,
                device=self.device,
            )
            logger.info(f"Real-ESRGAN {self.preset.upscaling.scale_factor}x upscaler ready")
        except Exception as e:
            logger.warning(f"Failed to load upscaler: {e}")
            self.upscaler = None

    def process_image(self, image_path: Path, room_type: str = "interior") -> Dict:
        """
        Process single image through complete pipeline.

        Args:
            image_path: Path to source image
            room_type: Room type for AI prompt customization

        Returns:
            Dictionary with processing results and metadata
        """
        start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {image_path.name}")
        logger.info(f"{'='*80}")

        results = {
            "source_path": str(image_path),
            "room_type": room_type,
            "stages": {},
            "output_paths": {},
        }

        # Stage 1: Load 32-bit HDR TIFF
        image_linear, metadata = self._stage_1_load_hdr(image_path)
        results["metadata"] = metadata
        results["stages"]["1_load"] = time.time() - start_time

        # Stage 1.5: Auto White Balance (fix color casts)
        image_linear = self._auto_white_balance(image_linear)

        # Stage 2: Depth estimation
        depth_map = self._stage_2_depth_estimation(image_linear)
        results["stages"]["2_depth"] = time.time() - start_time - sum(results["stages"].values())

        # Stage 3: Material Response
        image_enhanced = self._stage_3_material_response(image_linear, depth_map)
        results["stages"]["3_material"] = time.time() - start_time - sum(results["stages"].values())

        # Detect scene type for adaptive processing
        scene_type = self._detect_scene_type(image_enhanced) if self.preset.tone_mapping.adaptive_tone_mapping else None
        results["scene_type"] = scene_type

        # Stage 4: Tone mapping (now receives depth map and scene type)
        image_tonemapped = self._stage_4_tone_mapping(image_enhanced, depth_map, scene_type)
        results["stages"]["4_tonemap"] = time.time() - start_time - sum(results["stages"].values())

        # Stage 5: Color grading
        image_graded = self._stage_5_color_grading(image_tonemapped)
        results["stages"]["5_color"] = time.time() - start_time - sum(results["stages"].values())

        # Stage 6: AI enhancement
        image_ai = self._stage_6_ai_enhancement(image_graded, depth_map, room_type)
        results["stages"]["6_ai"] = time.time() - start_time - sum(results["stages"].values())

        # Stage 7: Upscaling
        image_final = self._stage_7_upscaling(image_ai, metadata["original_size"])
        results["stages"]["7_upscale"] = time.time() - start_time - sum(results["stages"].values())

        # Save outputs
        self._save_outputs(image_path, image_final, image_tonemapped, results)

        total_time = time.time() - start_time
        results["total_time"] = total_time
        self.stats["images_processed"] += 1
        self.stats["total_time"] += total_time

        logger.info(f"\n✅ Completed in {total_time:.1f}s")
        self._log_stage_times(results["stages"])

        return results

    def _stage_1_load_hdr(self, image_path: Path) -> Tuple[np.ndarray, Dict]:
        """Stage 1: Load 32-bit HDR TIFF with metadata preservation."""
        logger.info("\n[Stage 1/7] Loading 32-bit HDR TIFF...")

        # Load with tifffile to preserve full precision
        image_data = tifffile.imread(str(image_path))

        # Extract metadata
        metadata = {
            "original_size": (image_data.shape[1], image_data.shape[0]),
            "bit_depth": image_data.dtype,
            "has_alpha": image_data.shape[2] == 4 if len(image_data.shape) == 3 else False,
        }

        # Convert to float32 linear RGB
        if image_data.dtype == np.float32:
            image_linear = image_data
        elif image_data.dtype == np.uint16:
            image_linear = image_data.astype(np.float32) / 65535.0
        elif image_data.dtype == np.uint8:
            image_linear = image_data.astype(np.float32) / 255.0
        else:
            image_linear = image_data.astype(np.float32)

        # Handle alpha channel
        if metadata["has_alpha"]:
            metadata["alpha"] = image_linear[:, :, 3]
            image_linear = image_linear[:, :, :3]

        logger.info(f"  ✓ Loaded: {metadata['original_size'][0]}×{metadata['original_size'][1]}, {metadata['bit_depth']}")
        return image_linear, metadata

    def _auto_white_balance(self, image_linear: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Apply automatic white balance correction using gray world algorithm.

        Fixes color casts in source images by normalizing channel means.
        Preserves HDR dynamic range by working in linear space.

        Args:
            image_linear: Scene-linear RGB image (float32, 0-1+ range)
            strength: How aggressively to apply correction (0.0-1.0, default 1.0)

        Returns:
            White-balanced linear RGB image
        """
        # Calculate mean per channel
        r_mean = image_linear[:, :, 0].mean()
        g_mean = image_linear[:, :, 1].mean()
        b_mean = image_linear[:, :, 2].mean()

        # Overall mean (gray world assumption)
        gray_mean = (r_mean + g_mean + b_mean) / 3.0

        # Calculate correction factors
        r_scale = gray_mean / (r_mean + 1e-6)
        g_scale = gray_mean / (g_mean + 1e-6)
        b_scale = gray_mean / (b_mean + 1e-6)

        # Apply strength (blend between original and fully corrected)
        r_scale = 1.0 + strength * (r_scale - 1.0)
        g_scale = 1.0 + strength * (g_scale - 1.0)
        b_scale = 1.0 + strength * (b_scale - 1.0)

        logger.info(f"\n[Stage 1.5/7] Auto White Balance (strength={strength}):")
        logger.info(f"  Before: R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f} (ratio: {b_mean/r_mean:.2f}x)")
        logger.info(f"  Scales: R={r_scale:.3f}, G={g_scale:.3f}, B={b_scale:.3f}")

        # Apply correction
        balanced = image_linear.copy()
        balanced[:, :, 0] *= r_scale
        balanced[:, :, 1] *= g_scale
        balanced[:, :, 2] *= b_scale

        # Verify result
        r_new = balanced[:, :, 0].mean()
        g_new = balanced[:, :, 1].mean()
        b_new = balanced[:, :, 2].mean()
        logger.info(f"  After:  R={r_new:.3f}, G={g_new:.3f}, B={b_new:.3f} (ratio: {b_new/r_new:.2f}x)")
        logger.info(f"  ✓ White balance corrected")

        return balanced

    def _stage_2_depth_estimation(self, image_linear: np.ndarray) -> Optional[np.ndarray]:
        """Stage 2: Depth Anything V2 depth estimation."""
        if not self.preset.depth.enabled or self.depth_model is None:
            logger.info("\n[Stage 2/7] Depth estimation: SKIPPED")
            return None

        logger.info("\n[Stage 2/7] Depth Anything V2 estimation...")

        try:
            # Convert to uint8 for depth model
            image_uint8 = (np.clip(image_linear, 0, 1) * 255).astype(np.uint8)

            # Estimate depth
            result = self.depth_model.estimate_depth(image_uint8)
            depth_map = result["depth"]  # Extract depth map from result dict

            logger.info(f"  ✓ Depth map: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")
            return depth_map
        except Exception as e:
            logger.warning(f"  ⚠ Depth estimation failed: {e}")
            return None

    def _stage_3_material_response(self, image_linear: np.ndarray, depth_map: Optional[np.ndarray]) -> np.ndarray:
        """Stage 3: Material Response Technology for surface enhancement."""
        if not self.preset.material_response.enabled:
            logger.info("\n[Stage 3/7] Material Response: SKIPPED")
            return image_linear

        logger.info("\n[Stage 3/7] Material Response enhancement...")

        # Simplified material response (physics-based enhancement)
        image_enhanced = image_linear.copy()

        # Enhance micro-contrast (material detail)
        strength = self.preset.material_response.strength
        enhanced = (
            cv2.detailEnhance((np.clip(image_linear, 0, 1) * 255).astype(np.uint8), sigma_s=10, sigma_r=0.15).astype(
                np.float32
            )
            / 255.0
        )

        image_enhanced = image_linear * (1 - strength * 0.3) + enhanced * (strength * 0.3)

        # Selective sharpening (preserves highlights)
        if self.preset.material_response.preserve_highlights:
            luminance = np.dot(image_linear, [0.2126, 0.7152, 0.0722])
            highlight_mask = np.clip((luminance - 0.7) / 0.3, 0, 1)[:, :, None]

            blurred = cv2.GaussianBlur(image_enhanced, (0, 0), 1.0)
            sharpened = image_enhanced + (image_enhanced - blurred) * 0.5

            image_enhanced = sharpened * (1 - highlight_mask) + image_enhanced * highlight_mask

        logger.info(f"  ✓ Enhanced with strength {strength:.2f}")
        return image_enhanced

    def _stage_4_tone_mapping(
        self, image_linear: np.ndarray, depth_map: Optional[np.ndarray] = None, scene_type: Optional[str] = None
    ) -> np.ndarray:
        """Stage 4: Intelligent HDR tone mapping with adaptive shadow handling."""
        logger.info("\n[Stage 4/7] HDR tone mapping...")

        cfg = self.preset.tone_mapping

        # Detect scene type if not provided
        if scene_type is None and cfg.adaptive_tone_mapping:
            scene_type = self._detect_scene_type(image_linear)

        # Apply exposure adjustment
        if cfg.exposure != 0.0:
            exposure_scale = 2.0**cfg.exposure
            image_exposed = image_linear * exposure_scale
        else:
            image_exposed = image_linear

        # Apply adaptive shadow boost for outdoor scenes
        if cfg.adaptive_tone_mapping and scene_type == "outdoor" and cfg.shadow_boost_outdoor > 0:
            image_exposed = self._apply_shadow_boost(image_exposed, cfg.shadow_boost_outdoor, depth_map)

        # Zone-based tone mapping if depth available
        if cfg.use_zone_based_mapping and depth_map is not None and self.preset.depth.enabled:
            image_tonemapped = self._zone_based_tone_mapping(image_exposed, depth_map, cfg)
        else:
            # Standard tone mapping
            if cfg.method == "filmic":
                image_tonemapped = apply_filmic_hable(image_exposed, exposure=1.0, white_point=cfg.white_point)
                logger.info(f"  ✓ Filmic Hable (white point: {cfg.white_point})")
            elif cfg.method == "agx" and cfg.agx_config_path:
                try:
                    image_tonemapped = apply_agx_ocio(image_exposed, config_path=cfg.agx_config_path)
                    logger.info("  ✓ AgX OCIO")
                except Exception as e:
                    logger.warning(f"  ⚠ AgX failed, falling back to Filmic: {e}")
                    image_tonemapped = apply_filmic_hable(image_exposed)
            else:
                # Simple Reinhard
                image_tonemapped = image_exposed / (1.0 + image_exposed)
                logger.info("  ✓ Reinhard")

        # Apply contrast
        if cfg.contrast != 1.0:
            mean_lum = np.mean(image_tonemapped)
            image_tonemapped = (image_tonemapped - mean_lum) * cfg.contrast + mean_lum
            image_tonemapped = np.clip(image_tonemapped, 0, 1)

        return image_tonemapped

    def _apply_shadow_boost(
        self, image_linear: np.ndarray, boost_strength: float, depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply adaptive shadow boost for outdoor scenes to reduce clipping.

        Args:
            image_linear: Scene-linear image
            boost_strength: Shadow boost amount (0.0-1.0)
            depth_map: Optional depth map for depth-aware boosting

        Returns:
            Shadow-boosted image
        """
        # Calculate luminance
        luminance = np.dot(image_linear, [0.2126, 0.7152, 0.0722])

        # Create shadow mask (smooth transition from 0.0 to 0.3 luminance)
        shadow_threshold = 0.3
        shadow_mask = 1.0 - np.clip(luminance / shadow_threshold, 0, 1)
        shadow_mask = shadow_mask**0.5  # Smooth falloff

        # If depth available, boost distant shadows more (atmospheric perspective)
        if depth_map is not None:
            # Normalize depth to 0-1
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            # Distant objects (higher depth values) get more boost
            depth_weight = 0.5 + 0.5 * depth_norm
            if len(shadow_mask.shape) == 2:
                shadow_mask = shadow_mask * depth_weight
            else:
                shadow_mask = shadow_mask * depth_weight[:, :, None]

        # Calculate boost amount (lift shadows without crushing highlights)
        # Use power curve to lift shadows more than midtones
        boost_curve = boost_strength * 0.5  # Max 50% lift to avoid artifacts
        shadow_lift = shadow_mask * boost_curve

        # Apply lift (additive in scene-linear space)
        boosted = image_linear + shadow_lift[:, :, None] if len(shadow_mask.shape) == 2 else image_linear + shadow_lift

        # Preserve highlights (don't boost already bright regions)
        highlight_mask = np.clip((luminance - 0.7) / 0.3, 0, 1)
        final = boosted * (1 - highlight_mask[:, :, None]) + image_linear * highlight_mask[:, :, None]

        logger.info(f"  → Shadow boost applied: {boost_strength:.2f} strength")

        return final

    def _zone_based_tone_mapping(self, image_linear: np.ndarray, depth_map: np.ndarray, cfg: ToneMappingConfig) -> np.ndarray:
        """
        Apply zone-based tone mapping using depth information.

        Args:
            image_linear: Scene-linear image
            depth_map: Depth map
            cfg: Tone mapping configuration

        Returns:
            Tone-mapped image
        """
        # Normalize depth
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # Define zones (foreground, midground, background, far background)
        num_zones = self.preset.depth.num_zones
        zone_maps = []

        for i in range(num_zones):
            zone_start = i / num_zones
            zone_end = (i + 1) / num_zones

            # Smooth transitions between zones
            zone_mask = np.clip((depth_norm - zone_start) / 0.1, 0, 1) * np.clip((zone_end - depth_norm) / 0.1, 0, 1)
            zone_maps.append(zone_mask)

        # Apply tone mapping per zone with different parameters
        result = np.zeros_like(image_linear)

        for i, zone_mask in enumerate(zone_maps):
            # Adjust white point based on zone (closer = lower white point for more detail)
            zone_white_point = cfg.white_point * (1.0 + 0.3 * i / num_zones)

            # Tone map this zone
            zone_toned = apply_filmic_hable(image_linear, exposure=1.0, white_point=zone_white_point)

            # Blend into result
            result += zone_toned * zone_mask[:, :, None]

        # Normalize (zone masks should sum to ~1, but ensure)
        mask_sum = sum(zone_maps)
        result = result / (mask_sum[:, :, None] + 1e-6)

        logger.info(f"  ✓ Zone-based tone mapping ({num_zones} zones)")

        return result

    def _stage_5_color_grading(self, image: np.ndarray) -> np.ndarray:
        """Stage 5: Location-specific color grading with LUT stack."""
        if not self.preset.color_grading.enabled:
            logger.info("\n[Stage 5/7] Color grading: SKIPPED")
            return image

        logger.info("\n[Stage 5/7] Color grading...")

        image_graded = image.copy()

        # Apply LUT stack (placeholder - requires LUT implementation)
        # In production, implement 3D LUT interpolation
        cfg = self.preset.color_grading

        logger.info(f"  → LUT stack: {len(cfg.lut_stack)} LUTs")
        for lut_path, strength in cfg.lut_stack:
            logger.info(f"    • {Path(lut_path).name} @ {strength*100:.0f}%")

        # Apply saturation
        if cfg.saturation != 1.0:
            # Convert to PIL for saturation adjustment
            image_pil = Image.fromarray((np.clip(image_graded, 0, 1) * 255).astype(np.uint8))
            enhancer = ImageEnhance.Color(image_pil)
            image_pil = enhancer.enhance(cfg.saturation)
            image_graded = np.array(image_pil).astype(np.float32) / 255.0
            logger.info(f"  ✓ Saturation: {cfg.saturation:.2f}")

        return image_graded

    def _stage_6_ai_enhancement(self, image: np.ndarray, depth_map: Optional[np.ndarray], room_type: str) -> np.ndarray:
        """Stage 6: AI enhancement with ControlNet + SDXL (with tensor padding fix)."""
        if not self.preset.ai_enhancement.enabled or self.ai_pipeline is None:
            logger.info("\n[Stage 6/7] AI enhancement: SKIPPED")
            return image

        logger.info(f"\n[Stage 6/7] AI enhancement ({self.preset.ai_enhancement.num_inference_steps} steps)...")

        try:
            cfg = self.preset.ai_enhancement

            # Convert to PIL
            image_pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            original_size = image_pil.size

            # Resize for processing (SD works best at 512-768px)
            target_size = (768, int(768 * image_pil.size[1] / image_pil.size[0]))
            image_resized = image_pil.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy for padding
            image_np = np.array(image_resized).astype(np.float32) / 255.0

            # Apply padding if enabled (BEFORE canny generation)
            padding = None
            if cfg.ai_enhancement_padding:
                image_np, padding = self._pad_for_controlnet(image_np, cfg.target_size_multiple)
                # Convert back to PIL (this will be used for both input and canny)
                image_padded_pil = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
            else:
                image_padded_pil = image_resized

            # Generate Canny edges from PADDED image (ensures same dimensions)
            canny = self.canny_detector(image_padded_pil, 100, 200)

            # Build prompt
            prompt = cfg.prompt_template.format(room_type=room_type, style="montecito coastal estate, golden hour lighting")

            # Generate
            generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
            result = self.ai_pipeline(
                prompt=prompt,
                negative_prompt=cfg.negative_prompt,
                image=image_padded_pil,  # Use padded image
                control_image=canny,  # Canny from padded image (same size)
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                strength=cfg.strength,
                generator=generator,
            ).images[0]

            # Convert result to numpy
            result_np = np.array(result).astype(np.float32) / 255.0

            # Remove padding if it was applied
            if padding is not None:
                result_np = self._unpad_image(result_np, padding)
                result = Image.fromarray((np.clip(result_np, 0, 1) * 255).astype(np.uint8))

            # Resize back to original
            result = result.resize(original_size, Image.Resampling.LANCZOS)

            # Convert back to numpy
            image_ai = np.array(result).astype(np.float32) / 255.0

            logger.info(f"  ✓ Enhanced with strength {cfg.strength:.2f}")
            return image_ai
        except Exception as e:
            logger.warning(f"  ⚠ AI enhancement failed: {e}")
            logger.info(f"     Continuing without AI enhancement (other stages compensate)")
            return image

    def _stage_7_upscaling(self, image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Stage 7: Upscaling (Lanczos or Real-ESRGAN)."""
        if not self.preset.upscaling.enabled:
            logger.info("\n[Stage 7/7] Upscaling: SKIPPED")
            return image

        logger.info(f"\n[Stage 7/7] Upscaling ({self.preset.upscaling.scale_factor}x)...")

        # Force Lanczos if method is set to lanczos, or if ESRGAN not available
        use_lanczos = self.preset.upscaling.method == "lanczos" or self.upscaler is None

        if use_lanczos:
            # Use Lanczos (color-neutral, no AI bias)
            logger.info("  → Using Lanczos (traditional, color-neutral)")
            image_pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            upscaled_pil = image_pil.resize(
                (
                    int(original_size[0] * self.preset.upscaling.scale_factor),
                    int(original_size[1] * self.preset.upscaling.scale_factor),
                ),
                Image.Resampling.LANCZOS,
            )
            image_upscaled = np.array(upscaled_pil).astype(np.float32) / 255.0
            logger.info(f"  ✓ Upscaled to {upscaled_pil.size[0]}×{upscaled_pil.size[1]}")
            return image_upscaled

        try:
            # Use Real-ESRGAN (AI upscaling with potential color bias)
            logger.info("  → Using Real-ESRGAN (AI enhancement)")
            # Convert to BGR for ESRGAN
            image_bgr = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Upscale
            upscaled_bgr, _ = self.upscaler.enhance(image_bgr, outscale=self.preset.upscaling.scale_factor)

            # Convert back to RGB
            image_upscaled = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            logger.info(f"  ✓ Real-ESRGAN: {upscaled_bgr.shape[1]}×{upscaled_bgr.shape[0]}")
            return image_upscaled
        except Exception as e:
            logger.warning(f"  ⚠ Real-ESRGAN failed: {e}")
            logger.info("  → Falling back to Lanczos")
            return self._stage_7_upscaling(image, original_size)  # Retry with Lanczos

    def _save_outputs(self, source_path: Path, image_final: np.ndarray, image_tonemapped: np.ndarray, results: Dict):
        """Save output files (TIFF master + JPEG delivery)."""
        logger.info("\n[Output] Saving files...")

        output_dir = Path(self.preset.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = source_path.stem

        # Save master TIFF
        if self.preset.output.save_master_tiff:
            if self.preset.output.master_bit_depth == 16:
                master_data = (np.clip(image_final, 0, 1) * 65535).astype(np.uint16)
            else:
                master_data = image_final.astype(np.float32)

            master_path = output_dir / f"{stem}_master.tif"
            tifffile.imwrite(str(master_path), master_data, compression="lzw")
            results["output_paths"]["master_tiff"] = str(master_path)
            logger.info(f"  ✓ Master TIFF: {master_path.name}")

        # Save delivery JPEG
        if self.preset.output.save_delivery_jpeg:
            jpeg_data = (np.clip(image_final, 0, 1) * 255).astype(np.uint8)
            jpeg_pil = Image.fromarray(jpeg_data)
            jpeg_path = output_dir / f"{stem}_delivery.jpg"
            jpeg_pil.save(jpeg_path, quality=self.preset.output.jpeg_quality, optimize=True)
            results["output_paths"]["delivery_jpeg"] = str(jpeg_path)
            logger.info(f"  ✓ Delivery JPEG: {jpeg_path.name}")

        # Save intermediate stages
        if self.preset.output.save_intermediate_stages:
            intermediate_path = output_dir / f"{stem}_tonemapped.jpg"
            intermediate_data = (np.clip(image_tonemapped, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(intermediate_data).save(intermediate_path, quality=90)
            results["output_paths"]["tonemapped"] = str(intermediate_path)

    def _log_stage_times(self, stages: Dict[str, float]):
        """Log processing time breakdown."""
        logger.info("\n📊 Processing breakdown:")
        for stage, duration in stages.items():
            logger.info(f"  {stage}: {duration:.2f}s")

    def batch_process(self, image_paths: List[Path], room_types: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Batch process multiple images.

        Args:
            image_paths: List of image paths
            room_types: Optional mapping of filename to room type

        Returns:
            List of processing results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING: {len(image_paths)} images")
        logger.info(f"{'='*80}")

        results = []
        room_types = room_types or {}

        for image_path in tqdm(image_paths, desc="Processing images"):
            room_type = room_types.get(image_path.stem, "interior")
            try:
                result = self.process_image(image_path, room_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                results.append({"source_path": str(image_path), "error": str(e)})

        # Save batch report
        self._save_batch_report(results)

        return results

    def _save_batch_report(self, results: List[Dict]):
        """Save batch processing report."""
        output_dir = Path(self.preset.output.output_dir)
        report_path = output_dir / "processing_report.json"

        report = {
            "preset": self.preset.name,
            "images_processed": len(results),
            "total_time": self.stats["total_time"],
            "average_time": self.stats["total_time"] / len(results) if results else 0,
            "results": results,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n📄 Batch report saved: {report_path}")
        logger.info(f"   Total time: {self.stats['total_time']:.1f}s")
        logger.info(f"   Average: {report['average_time']:.1f}s per image")


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Luxury Estate Master Pipeline - Elite HDR processing for 750 Picacho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python luxury_estate_master_pipeline.py input.tif --room-type aerial

  # Batch process entire directory
  python luxury_estate_master_pipeline.py input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif

  # Use aerial preset
  python luxury_estate_master_pipeline.py input.tif --preset aerial

  # Dry run (show configuration)
  python luxury_estate_master_pipeline.py --dry-run --preset 750_picacho
        """,
    )

    parser.add_argument("images", nargs="*", help="Input image path(s)")
    parser.add_argument(
        "--preset", choices=["750_picacho", "aerial"], default="750_picacho", help="Pipeline preset (default: 750_picacho)"
    )
    parser.add_argument("--room-type", default="interior", help="Room type for AI prompts (default: interior)")
    parser.add_argument("--output-dir", help="Output directory (overrides preset)")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration and exit")
    parser.add_argument("--save-preset", help="Save preset to YAML file")

    args = parser.parse_args()

    # Load preset
    if args.preset == "750_picacho":
        preset = get_750_picacho_preset()
    elif args.preset == "aerial":
        preset = get_aerial_preset()
    else:
        preset = get_750_picacho_preset()

    if args.output_dir:
        preset.output.output_dir = args.output_dir

    # Dry run
    if args.dry_run:
        print("\n" + "=" * 80)
        print(f"PRESET: {preset.name}")
        print("=" * 80)
        print(json.dumps(asdict(preset), indent=2))
        return 0

    # Save preset
    if args.save_preset:
        import yaml

        with open(args.save_preset, "w") as f:
            yaml.dump(asdict(preset), f, default_flow_style=False)
        print(f"✓ Preset saved to {args.save_preset}")
        return 0

    # Validate inputs
    if not args.images:
        parser.error("No input images specified")

    image_paths = []
    for pattern in args.images:
        path = Path(pattern)
        if path.is_file():
            image_paths.append(path)
        else:
            # Glob pattern
            image_paths.extend(Path(".").glob(pattern))

    if not image_paths:
        parser.error("No valid input images found")

    # Initialize pipeline
    pipeline = LuxuryEstateMasterPipeline(preset)

    # Process images
    if len(image_paths) == 1:
        pipeline.process_image(image_paths[0], args.room_type)
    else:
        pipeline.batch_process(image_paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
