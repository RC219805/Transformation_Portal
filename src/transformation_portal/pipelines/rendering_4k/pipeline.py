#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End 4K Rendering Enhancement Pipeline

Integrates the best features from the Transformation Portal ecosystem:
- Depth Anything V2 with CoreML/MPS acceleration
- Material Response Technology for surface realism
- Intelligent tone mapping (AgX, Filmic, Reinhard)
- AI-powered enhancement via ControlNet guidance
- Real-ESRGAN 4x upscaling to 4K resolution
- RAG-based quality feedback loop for iterative refinement
- Professional color grading with LUT stacks
- Complete metadata preservation

Designed for:
- Luxury real estate rendering
- Architectural visualization
- Editorial post-production

Optimized for:
- Apple Silicon (M-series) with Metal Performance Shaders
- NVIDIA CUDA GPUs
- CPU fallback for compatibility

Example:
    >>> from transformation_portal.pipelines.rendering_4k_pipeline import Rendering4KPipeline
    >>> pipeline = Rendering4KPipeline.from_preset("luxury_estate")
    >>> result = pipeline.process("input.jpg", output_dir="output/")
    >>> print(f"Quality Score: {result.quality_score}")

Author: Transformation Portal Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# Optional: PyYAML for configuration loading
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Optional: LPIPS for perceptual quality scoring
try:
    import torch  # noqa: F401 - used for availability check

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # noqa: F841 - placeholder for optional import

# Optional: ControlNet auxiliary processors
try:
    from controlnet_aux import CannyDetector

    HAS_CONTROLNET_AUX = True
except ImportError:
    HAS_CONTROLNET_AUX = False
    CannyDetector = None

# Optional: QualityFeedbackBridge for unified quality assessment
try:
    from ..quality_feedback_bridge import QualityFeedbackBridge, UnifiedQualityMetrics, create_rag_indexing_callback

    HAS_QUALITY_BRIDGE = True
except ImportError:
    HAS_QUALITY_BRIDGE = False
    QualityFeedbackBridge = None
    UnifiedQualityMetrics = None
    create_rag_indexing_callback = None

# Import internal utilities
from ...core.security.model_lock import resolve_model_lock_revision
from ...utils.image_utils import load_image, np_to_pil, pil_to_np
from .quality import GPUMemoryManager, QualityAssessor

# isort: off
from .stages import (
    apply_color_grading as apply_color_grading,
    apply_material_response as apply_material_response,
    apply_tone_mapping as apply_tone_mapping,
    apply_upscaling as apply_upscaling,
    estimate_depth_simple as estimate_depth_simple,
)

# isort: on
from .types import (
    STAGE_NAMES,
    AIEnhancementConfig,
    ColorGradingConfig,
    DepthConfig,
    DeviceType,
    MaterialResponseConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingResult,
    QualityFeedbackConfig,
    QualityLevel,
    QualityMetrics,
    StageMetrics,
    ToneMappingConfig,
    ToneMappingMethod,
    UpscalingConfig,
)

logger = logging.getLogger("transformation_portal.pipelines.rendering_4k_pipeline")


def _json_default(obj: object) -> object:
    """`json.dump` fallback serializer for common ML / pathlib types."""
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    if HAS_TORCH and torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()

    return str(obj)


# =============================================================================
# Main Pipeline Class
# =============================================================================


class Rendering4KPipeline:
    """
    End-to-End 4K Rendering Enhancement Pipeline.

    Combines depth estimation, tone mapping, material response, color grading,
    AI enhancement, and upscaling with a RAG-based quality feedback loop.

    Example:
        >>> pipeline = Rendering4KPipeline.from_preset("luxury_estate")
        >>> result = pipeline.process("input.jpg", output_dir="output/")
        >>> print(f"Quality: {result.quality_score:.2%}")
    """

    # Built-in presets
    PRESETS = {
        "default": PipelineConfig(
            name="default",
            description="Balanced settings for general use",
        ),
        "luxury_estate": PipelineConfig(
            name="luxury_estate",
            description="Optimized for luxury real estate interiors",
            material_response=MaterialResponseConfig(
                strength=0.75,
                texture_boost=0.3,
                micro_contrast=0.2,
            ),
            color_grading=ColorGradingConfig(
                saturation=1.08,
                vibrance=1.12,
                temperature_shift=(1.0, 0.98, 0.95),  # Warm
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for luxury workflows
                hybrid_mode=True,
                rag_indexing_enabled=True,
            ),
        ),
        "aerial_exterior": PipelineConfig(
            name="aerial_exterior",
            description="Optimized for aerial and exterior shots",
            depth=DepthConfig(
                num_zones=3,
            ),
            tone_mapping=ToneMappingConfig(
                method=ToneMappingMethod.FILMIC,
                contrast=1.1,
            ),
            color_grading=ColorGradingConfig(
                saturation=1.12,
                vibrance=1.15,
                temperature_shift=(1.05, 1.0, 0.95),  # Golden hour warmth
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for luxury workflows
                hybrid_mode=True,
            ),
        ),
        "editorial": PipelineConfig(
            name="editorial",
            description="High-end editorial/magazine quality",
            quality_level=QualityLevel.ULTRA,
            tone_mapping=ToneMappingConfig(
                method=ToneMappingMethod.ACES,
                contrast=1.05,
            ),
            material_response=MaterialResponseConfig(
                strength=0.8,
                texture_boost=0.35,
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,  # Enable LPIPS for editorial workflows
                hybrid_mode=True,
                rag_indexing_enabled=True,
            ),
        ),
        "750_picacho": PipelineConfig(
            name="750_picacho",
            description="Optimized preset for 750 Picacho Lane estate images",
            quality_level=QualityLevel.ULTRA,
            material_response=MaterialResponseConfig(
                strength=0.80,
                texture_boost=0.35,
                micro_contrast=0.25,
                surface_types=["quartzite", "oak", "metal", "glass", "stucco"],
            ),
            color_grading=ColorGradingConfig(
                saturation=1.10,
                vibrance=1.15,
                temperature_shift=(1.02, 0.99, 0.96),  # Warm Montecito tones
            ),
            quality_feedback=QualityFeedbackConfig(
                use_lpips=True,
                hybrid_mode=True,
                perceptual_percentile_target=95.0,
                material_fidelity_target=0.98,
                rag_indexing_enabled=True,
            ),
        ),
        "preview": PipelineConfig(
            name="preview",
            description="Fast preview with reduced quality",
            quality_level=QualityLevel.PREVIEW,
            depth=DepthConfig(enabled=False),
            material_response=MaterialResponseConfig(strength=0.5),
            upscaling=UpscalingConfig(enabled=False),
            quality_feedback=QualityFeedbackConfig(enabled=False),
            output=OutputConfig(
                master_tiff_16bit=False,
                save_intermediate=False,
                save_depth_visualization=False,
            ),
        ),
    }

    def __init__(
        self,
        config: PipelineConfig,
        *,
        strict_model_lock: Optional[bool] = None,
    ):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
            strict_model_lock: Enforce pinned revisions for remote model loads.
                If None, uses ``TP_STRICT_MODEL_LOCK`` environment variable.
        """
        self.config = config
        self.strict_model_lock = strict_model_lock
        self.quality_assessor = QualityAssessor(config.quality_feedback)
        # Use OrderedDict for true LRU cache behavior
        self._depth_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Initialize QualityFeedbackBridge if available and LPIPS requested
        self._quality_bridge: Optional[QualityFeedbackBridge] = None
        if HAS_QUALITY_BRIDGE and config.quality_feedback.use_lpips:
            rag_callback = None
            if config.quality_feedback.rag_indexing_enabled and config.quality_feedback.rag_index_path:
                rag_callback = create_rag_indexing_callback(config.quality_feedback.rag_index_path)
            self._quality_bridge = QualityFeedbackBridge(
                hybrid_mode=config.quality_feedback.hybrid_mode,
                lpips_network=config.quality_feedback.lpips_network,
                enable_material_fidelity=config.quality_feedback.enable_material_fidelity,
                rag_callback=rag_callback,
            )
            logger.info("QualityFeedbackBridge initialized for LPIPS scoring")

        # Track original input for quality comparison
        self._current_original: Optional[np.ndarray] = None
        self._current_image_id: str = ""

        # Detect compute device
        self.device = self._detect_device()

        # Initialize GPU memory manager
        self.memory_manager = GPUMemoryManager(self.device)

        # Initialize ML models (lazy loading)
        self._depth_model = None
        self._depth_model_initialized = False
        self._controlnet_pipe = None
        self._controlnet_initialized = False

        logger.info(f"Initialized Rendering4KPipeline: {config.name}")
        logger.info(f"Device: {self.device.value}")
        logger.info(f"Quality Level: {config.quality_level.value}")

        # Log GPU status if available
        if self.device != DeviceType.CPU:
            self.memory_manager.log_memory_status()

    @classmethod
    def from_preset(cls, preset_name: str) -> "Rendering4KPipeline":
        """
        Create pipeline from built-in preset.

        Args:
            preset_name: Name of preset (default, luxury_estate, aerial_exterior, editorial, preview)

        Returns:
            Initialized pipeline

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        config = deepcopy(cls.PRESETS[preset_name])
        return cls(config)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Rendering4KPipeline":
        """
        Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML config

        Returns:
            Initialized pipeline

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file does not exist
        """
        if not HAS_YAML or yaml is None:
            raise ImportError("PyYAML is required for loading YAML configs. " "Install with: pip install pyyaml")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open(encoding="utf-8") as f:
            # YAML_GOVERNANCE_EXEMPT: legacy render pipeline config, not a preset/governance entrypoint.
            data = yaml.safe_load(f)

        # Build config from YAML data
        config = cls._build_config_from_dict(data)

        return cls(config)

    @staticmethod
    def _build_config_from_dict(data: Dict) -> PipelineConfig:
        """Build PipelineConfig from dictionary with proper enum conversion."""
        # Parse nested configs (most use strings, no enum conversion needed)
        depth = DepthConfig(**data.get("depth", {}))

        # Parse tone mapping config with ToneMappingMethod enum conversion
        tone_mapping_data = data.get("tone_mapping", {})
        if "method" in tone_mapping_data and isinstance(tone_mapping_data["method"], str):
            try:
                tone_mapping_data["method"] = ToneMappingMethod(tone_mapping_data["method"])
            except ValueError:
                logger.warning(f"Invalid tone_mapping method '{tone_mapping_data['method']}', using 'agx'")
                tone_mapping_data["method"] = ToneMappingMethod.AGX
        tone_mapping = ToneMappingConfig(**tone_mapping_data)

        # Parse remaining configs (all use strings, no enum conversion needed)
        material_response = MaterialResponseConfig(**data.get("material_response", {}))
        color_grading = ColorGradingConfig(**data.get("color_grading", {}))
        ai_enhancement = AIEnhancementConfig(**data.get("ai_enhancement", {}))
        upscaling = UpscalingConfig(**data.get("upscaling", {}))
        quality_feedback = QualityFeedbackConfig(**data.get("quality_feedback", {}))
        output = OutputConfig(**data.get("output", {}))

        # Parse quality level with validation
        quality_level_value = data.get("quality_level", "high")
        try:
            quality_level = QualityLevel(quality_level_value)
        except ValueError:
            logger.warning(f"Invalid quality_level '{quality_level_value}', using 'high'")
            quality_level = QualityLevel.HIGH

        return PipelineConfig(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            quality_level=quality_level,
            depth=depth,
            tone_mapping=tone_mapping,
            material_response=material_response,
            color_grading=color_grading,
            ai_enhancement=ai_enhancement,
            upscaling=upscaling,
            quality_feedback=quality_feedback,
            output=output,
        )

    def _detect_device(self) -> DeviceType:
        """Detect best available compute device."""
        try:
            import torch

            # Check for MPS (Apple Silicon) support
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    return DeviceType.MPS
            # Check for CUDA support
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return DeviceType.CUDA
        except (ImportError, AttributeError):
            # torch is not installed or has unexpected structure; fall back to CPU processing
            pass
        return DeviceType.CPU

    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ProcessingResult:
        """
        Process single image through complete pipeline.

        Args:
            input_path: Path to input image
            output_dir: Output directory (optional)

        Returns:
            ProcessingResult with enhanced image and metadata
        """
        start_time = time.time()
        input_path = Path(input_path)
        stage_metrics: List[StageMetrics] = []

        # Store image ID and original for RAG provenance and LPIPS comparison
        self._current_image_id = input_path.stem
        self._current_original = None

        logger.info("=" * 70)
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"Preset: {self.config.name}")
        logger.info("=" * 70)

        # Stage 1: Input Validation
        stage_start = time.time()
        logger.info("[1/9] Input Validation")
        try:
            image_pil = load_image(input_path)
            image_np = pil_to_np(image_pil, to_float=True)
            # Store original for quality comparison
            self._current_original = image_np.copy()
            logger.info(f"  Size: {image_pil.size}, Shape: {image_np.shape}")
            stage_metrics.append(
                StageMetrics(
                    "input_validation",
                    (time.time() - stage_start) * 1000,
                    True,
                )
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            raise

        # Stage 2: Depth Estimation
        stage_start = time.time()
        logger.info("[2/9] Depth Estimation")
        depth_map = None
        if self.config.depth.enabled:
            depth_map = self._estimate_depth(image_np, input_path)
            logger.info(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "depth_estimation",
                (time.time() - stage_start) * 1000,
                True if depth_map is not None else False,
            )
        )

        # Stage 3: Tone Mapping
        stage_start = time.time()
        logger.info("[3/9] Tone Mapping")
        if self.config.tone_mapping.enabled:
            processed = apply_tone_mapping(image_np, self.config.tone_mapping)
            logger.info(f"  Method: {self.config.tone_mapping.method.value}")
        else:
            processed = image_np
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "tone_mapping",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 4: Material Response
        stage_start = time.time()
        logger.info("[4/9] Material Response")
        if self.config.material_response.enabled:
            processed = apply_material_response(processed, depth_map, self.config.material_response)
            logger.info(f"  Strength: {self.config.material_response.strength}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "material_response",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 5: Color Grading
        stage_start = time.time()
        logger.info("[5/9] Color Grading")
        if self.config.color_grading.enabled:
            processed = apply_color_grading(processed, self.config.color_grading)
            sat = self.config.color_grading.saturation
            vib = self.config.color_grading.vibrance
            logger.info(f"  Saturation: {sat}, Vibrance: {vib}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "color_grading",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Convert to PIL for AI enhancement and upscaling
        result_pil = np_to_pil(processed)

        # Stage 6: AI Enhancement (optional, requires ML deps)
        stage_start = time.time()
        logger.info("[6/9] AI Enhancement")
        if self.config.ai_enhancement.enabled:
            try:
                result_pil = self._apply_ai_enhancement(result_pil, depth_map)
                logger.info("  ✓ ControlNet enhancement complete")
                stage_metrics.append(
                    StageMetrics(
                        "ai_enhancement",
                        (time.time() - stage_start) * 1000,
                        True,
                    )
                )
            except Exception as e:
                logger.warning(f"  AI enhancement failed: {e}")
                stage_metrics.append(
                    StageMetrics(
                        "ai_enhancement",
                        (time.time() - stage_start) * 1000,
                        False,
                        notes=str(e),
                    )
                )
        else:
            logger.info("  Skipped (disabled)")
            stage_metrics.append(
                StageMetrics(
                    "ai_enhancement",
                    (time.time() - stage_start) * 1000,
                    False,
                )
            )

        # Stage 7: Upscaling to 4K
        stage_start = time.time()
        logger.info("[7/9] Upscaling")
        if self.config.upscaling.enabled:
            result_pil = apply_upscaling(result_pil, self.config.upscaling)
            logger.info(f"  Target: {self.config.upscaling.target_resolution}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "upscaling",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Stage 8: Quality Assessment & Feedback Loop
        stage_start = time.time()
        logger.info("[8/9] Quality Assessment")
        quality_metrics = None
        unified_metrics = None
        iterations = 1
        if self.config.quality_feedback.enabled:
            enhanced_np = pil_to_np(result_pil, to_float=True)

            # Use QualityFeedbackBridge if available (LPIPS-based scoring)
            if self._quality_bridge is not None:
                unified_metrics = self._quality_bridge.assess(
                    enhanced=enhanced_np,
                    original=self._current_original,
                    image_id=self._current_image_id,
                    pipeline_config_name=self.config.name,
                )
                # Translate unified metrics to QualityMetrics for backward compatibility
                quality_metrics = QualityMetrics(
                    sharpness=unified_metrics.heuristic.sharpness,
                    contrast=unified_metrics.heuristic.contrast,
                    colorfulness=unified_metrics.heuristic.colorfulness,
                    exposure_balance=unified_metrics.heuristic.exposure_balance,
                    noise_level=unified_metrics.heuristic.noise_level,
                    overall_score=unified_metrics.hybrid_score / 100.0,  # Normalize to 0-1
                    lpips_score=unified_metrics.perceptual.lpips_score,
                    lpips_percentile=unified_metrics.perceptual.lpips_percentile,
                    material_fidelity=unified_metrics.material_fidelity.overall_fidelity,
                    perceptual_quality=unified_metrics.perceptual_composite,
                )
                logger.info(f"  Hybrid Score: {unified_metrics.hybrid_score:.1f}/100")
                logger.info(f"  Perceptual: {unified_metrics.perceptual_composite:.1f}/100")
                logger.info(f"  Heuristic: {unified_metrics.heuristic_composite:.1f}/100")
                if unified_metrics.lpips_available:
                    logger.info(f"  LPIPS: {unified_metrics.perceptual.lpips_score:.4f}")
                    logger.info(f"  Material Fidelity: {unified_metrics.material_fidelity.overall_fidelity:.1%}")
                logger.info(f"  {unified_metrics.targets_summary}")
            else:
                # Fallback to heuristic-only QualityAssessor
                quality_metrics = self.quality_assessor.assess(enhanced_np)
                logger.info(f"  Overall Score: {quality_metrics.overall_score:.2%}")
                logger.info(f"  Sharpness: {quality_metrics.sharpness:.2%}")
                logger.info(f"  Contrast: {quality_metrics.contrast:.2%}")
                logger.info(f"  Colorfulness: {quality_metrics.colorfulness:.2%}")

            # Feedback loop for quality refinement
            auto_adjust = self.config.quality_feedback.auto_adjust
            threshold = self.config.quality_feedback.min_quality_threshold
            if auto_adjust and quality_metrics.overall_score < threshold:
                logger.info("  Quality below threshold, suggesting adjustments...")
                adjustments = self.quality_assessor.suggest_adjustments(quality_metrics)
                logger.info(f"  Suggested: {adjustments}")
        else:
            logger.info("  Skipped (disabled)")
        stage_metrics.append(
            StageMetrics(
                "quality_assessment",
                (time.time() - stage_start) * 1000,
                True if quality_metrics else False,
            )
        )

        # Stage 9: Output Generation
        stage_start = time.time()
        logger.info("[9/9] Output Generation")
        output_paths = {}
        if output_dir:
            output_paths = self._save_outputs(
                result_pil,
                depth_map,
                quality_metrics,
                input_path,
                Path(output_dir),
                unified_metrics=unified_metrics,
            )
            logger.info(f"  Saved {len(output_paths)} files")
        stage_metrics.append(
            StageMetrics(
                "output_generation",
                (time.time() - stage_start) * 1000,
                True,
            )
        )

        # Build result
        total_duration = (time.time() - start_time) * 1000

        logger.info("=" * 70)
        logger.info("✅ Processing Complete")
        logger.info(f"   Total Time: {total_duration:.0f}ms")
        if quality_metrics:
            logger.info(f"   Quality Score: {quality_metrics.overall_score:.2%}")
        logger.info("=" * 70)

        return ProcessingResult(
            image=result_pil,
            depth_map=depth_map,
            quality_metrics=quality_metrics,
            stage_metrics=stage_metrics,
            total_duration_ms=total_duration,
            iterations=iterations,
            output_paths=output_paths,
            config_used=self.config,
        )

    def _get_or_load_depth_model(self):
        """Lazy-load Depth Anything V2 model.

        Returns:
            Hugging Face depth estimation pipeline, or None if unavailable
        """
        if self._depth_model_initialized:
            return self._depth_model

        try:
            from transformers import pipeline as hf_pipeline

            model_map = {
                "small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf",
            }
            model_id = model_map.get(self.config.depth.model_variant, model_map["small"])
            model_revision = resolve_model_lock_revision(
                model_id,
                requested_revision=None,
                strict=self.strict_model_lock,
                context="Rendering4KPipeline(depth_estimation)",
            )
            device_id = 0 if self.device != DeviceType.CPU else -1

            logger.info(f"Loading Depth Anything V2 ({self.config.depth.model_variant})...")
            self._depth_model = hf_pipeline(
                "depth-estimation",
                model=model_id,
                revision=model_revision,
                device=device_id,
            )
            logger.info("✓ Depth Anything V2 loaded")
        except Exception as e:
            logger.warning(f"Depth Anything V2 unavailable: {e}. Using fallback.")
            self._depth_model = None

        self._depth_model_initialized = True
        return self._depth_model

    def _estimate_depth(
        self,
        image: np.ndarray,
        input_path: Path,
    ) -> np.ndarray:
        """
        Estimate depth map using Depth Anything V2 or fallback with caching.

        Args:
            image: RGB image as float32 array
            input_path: Path for cache key

        Returns:
            Depth map as float32 array [0, 1]
        """
        # Check cache
        cache_key = None
        if self.config.depth.cache_enabled:
            cache_key = self._compute_cache_key(image)
            if cache_key in self._depth_cache:
                logger.debug("  Using cached depth map")
                # Move to end to mark as recently used (LRU behavior)
                self._depth_cache.move_to_end(cache_key)
                return self._depth_cache[cache_key]

        # Try Depth Anything V2
        depth_model = self._get_or_load_depth_model()
        if depth_model is not None:
            try:
                image_pil = np_to_pil(image)
                result = depth_model(image_pil)
                # Validate result structure
                if "depth" not in result:
                    logger.warning("Depth model returned unexpected format, using fallback")
                    depth_map = estimate_depth_simple(image)
                else:
                    depth_map = np.array(result["depth"]).astype(np.float32)
                    # Normalize to [0, 1], handling edge case of constant depth
                    depth_range = depth_map.max() - depth_map.min()
                    if depth_range > 1e-8:
                        depth_map = (depth_map - depth_map.min()) / depth_range
                    else:
                        # Constant depth map - set to mid-range
                        depth_map = np.full_like(depth_map, 0.5)
            except Exception as e:
                logger.warning(f"Depth inference failed: {e}")
                depth_map = estimate_depth_simple(image)
        else:
            depth_map = estimate_depth_simple(image)

        # Cache result
        if self.config.depth.cache_enabled and cache_key is not None:
            if len(self._depth_cache) >= self.config.depth.cache_max_size:
                # Remove oldest (least recently used) entry
                self._depth_cache.popitem(last=False)
            self._depth_cache[cache_key] = depth_map

        return depth_map

    def _compute_cache_key(self, image: np.ndarray) -> str:
        """Compute cache key from image content (non-security, non-cryptographic)."""
        # Use SHA-256 for cache key (non-cryptographic; safe for content hashing)
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode("ascii"))
        digest.update(b"|")
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(b"|")

        contiguous_image = image if image.flags.c_contiguous else np.ascontiguousarray(image)
        byte_view = contiguous_image.view(np.uint8).reshape(-1)
        digest.update(memoryview(byte_view[:4096]))
        return digest.hexdigest()

    def _get_or_load_controlnet_pipe(self):
        """Lazy-load ControlNet pipeline for AI enhancement.

        Returns:
            StableDiffusionControlNetImg2ImgPipeline, or None if unavailable
        """
        if self._controlnet_initialized:
            return self._controlnet_pipe

        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

            logger.info("Loading ControlNet pipeline...")
            controlnets = []
            dtype = torch.float16 if self.device != DeviceType.CPU else torch.float32
            strict_model_lock = self.strict_model_lock

            if self.config.ai_enhancement.use_controlnet:
                canny_model_id = "lllyasviel/sd-controlnet-canny"
                canny_revision = resolve_model_lock_revision(
                    canny_model_id,
                    requested_revision=None,
                    strict=strict_model_lock,
                    context="Rendering4KPipeline(controlnet_canny)",
                )
                controlnets.append(
                    ControlNetModel.from_pretrained(  # nosec B615
                        canny_model_id,
                        revision=canny_revision,
                        torch_dtype=dtype,
                    )
                )
            if self.config.ai_enhancement.use_depth_guidance:
                depth_model_id = "lllyasviel/sd-controlnet-depth"
                depth_revision = resolve_model_lock_revision(
                    depth_model_id,
                    requested_revision=None,
                    strict=strict_model_lock,
                    context="Rendering4KPipeline(controlnet_depth)",
                )
                controlnets.append(
                    ControlNetModel.from_pretrained(  # nosec B615
                        depth_model_id,
                        revision=depth_revision,
                        torch_dtype=dtype,
                    )
                )

            if not controlnets:
                logger.warning("No ControlNet models configured")
                self._controlnet_pipe = None
                self._controlnet_initialized = True
                return None

            base_model_id = "runwayml/stable-diffusion-v1-5"
            base_model_revision = resolve_model_lock_revision(
                base_model_id,
                requested_revision=None,
                strict=strict_model_lock,
                context="Rendering4KPipeline(base_model)",
            )
            self._controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(  # nosec B615
                base_model_id,
                revision=base_model_revision,
                controlnet=controlnets if len(controlnets) > 1 else controlnets[0],
                torch_dtype=dtype,
                safety_checker=None,
            )
            self._controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(self._controlnet_pipe.scheduler.config)

            device_str = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}.get(self.device.value, "cpu")
            self._controlnet_pipe.to(device_str)

            if self.device == DeviceType.CUDA:
                self._controlnet_pipe.enable_attention_slicing()
                self._controlnet_pipe.enable_vae_slicing()

            logger.info("✓ ControlNet pipeline loaded")
        except Exception as e:
            logger.warning(f"ControlNet unavailable: {e}")
            self._controlnet_pipe = None

        self._controlnet_initialized = True
        return self._controlnet_pipe

    def _apply_ai_enhancement(
        self,
        image: Image.Image,
        depth_map: Optional[np.ndarray],
    ) -> Image.Image:
        """Apply ControlNet AI enhancement.

        Args:
            image: PIL Image to enhance
            depth_map: Optional depth map for depth-guided enhancement

        Returns:
            Enhanced PIL Image
        """
        pipe = self._get_or_load_controlnet_pipe()
        if pipe is None or not HAS_CONTROLNET_AUX:
            return image

        try:
            import torch

            control_images = []

            # Generate Canny edge map if ControlNet is enabled
            if self.config.ai_enhancement.use_controlnet and CannyDetector is not None:
                canny = CannyDetector()
                control_images.append(canny(image))

            # Use depth map if depth guidance is enabled
            if self.config.ai_enhancement.use_depth_guidance and depth_map is not None:
                # Convert depth map directly to RGB image for ControlNet
                # Use squeeze() to handle both (H, W) and (H, W, 1) shapes
                depth_uint8 = (depth_map.squeeze() * 255).astype(np.uint8)
                # Stack grayscale to RGB channels directly
                depth_rgb = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)
                depth_pil = Image.fromarray(depth_rgb, mode="RGB").resize(image.size)
                control_images.append(depth_pil)

            if not control_images:
                logger.warning("No control images available for ControlNet")
                return image

            # Use CPU generator for MPS as PyTorch's Generator doesn't support MPS directly
            device_for_gen = "cpu" if self.device == DeviceType.MPS else pipe.device
            generator = torch.Generator(device=device_for_gen).manual_seed(self.config.ai_enhancement.seed)

            result = pipe(
                prompt=self.config.ai_enhancement.prompt,
                negative_prompt=self.config.ai_enhancement.negative_prompt,
                image=image,
                control_image=(control_images if len(control_images) > 1 else control_images[0]),
                num_inference_steps=self.config.ai_enhancement.num_steps,
                guidance_scale=self.config.ai_enhancement.guidance_scale,
                strength=self.config.ai_enhancement.strength,
                generator=generator,
            ).images[0]

            return result
        except Exception as e:
            logger.error(f"ControlNet failed: {e}")
            return image

    def _save_outputs(
        self,
        image: Image.Image,
        depth_map: Optional[np.ndarray],
        quality_metrics: Optional[QualityMetrics],
        input_path: Path,
        output_dir: Path,
        unified_metrics: Optional[UnifiedQualityMetrics] = None,
    ) -> Dict[str, Path]:
        """Save all output files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        outputs = {}

        # Master TIFF (16-bit)
        if self.config.output.master_tiff_16bit and HAS_TIFFFILE:
            tiff_path = output_dir / f"{stem}_MASTER.tiff"
            img_np = pil_to_np(image, to_float=True)
            img_16bit = (np.clip(img_np, 0, 1) * 65535).astype(np.uint16)
            tifffile.imwrite(str(tiff_path), img_16bit, photometric="rgb")
            outputs["master_tiff"] = tiff_path
            logger.info(f"  Master TIFF: {tiff_path.name}")

        # Delivery JPEG
        if self.config.output.delivery_jpeg:
            jpeg_path = output_dir / f"{stem}_DELIVERY.jpg"
            image.save(
                jpeg_path,
                quality=self.config.output.jpeg_quality,
                progressive=self.config.output.jpeg_progressive,
                optimize=True,
            )
            outputs["delivery_jpeg"] = jpeg_path
            logger.info(f"  Delivery JPEG: {jpeg_path.name}")

        # Depth visualization
        if self.config.output.save_depth_visualization and depth_map is not None:
            depth_path = output_dir / f"{stem}_depth.png"
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_pil = Image.fromarray(depth_vis, mode="L")
            depth_pil.save(depth_path)
            outputs["depth_visualization"] = depth_path
            logger.info(f"  Depth Map: {depth_path.name}")

        # Quality report
        if self.config.output.save_quality_report and quality_metrics:
            report_path = output_dir / f"{stem}_quality_report.json"
            report = {
                "input": str(input_path),
                "preset": self.config.name,
                "quality_metrics": quality_metrics.to_dict(),
                "config": asdict(self.config),
            }
            # Include unified metrics if available (RAG-indexable)
            if unified_metrics is not None:
                report["unified_metrics"] = unified_metrics.to_dict()
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
            outputs["quality_report"] = report_path
            logger.info(f"  Quality Report: {report_path.name}")

        # Save unified metrics as separate RAG document if enabled
        if unified_metrics is not None and self.config.quality_feedback.rag_indexing_enabled:
            rag_path = output_dir / f"{stem}_unified_quality.json"
            with open(rag_path, "w") as f:
                json.dump(
                    unified_metrics.to_rag_document(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )
            outputs["unified_quality_doc"] = rag_path
            logger.info(f"  Unified Quality Doc: {rag_path.name}")

        return outputs

    def batch_process(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """
        Process multiple images in batch with GPU memory management.

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            show_progress: Show progress bar

        Returns:
            List of ProcessingResults
        """
        results = []

        # Use tqdm if available, otherwise simple iteration
        if show_progress and HAS_TQDM and tqdm is not None:
            iterator = tqdm(input_paths, desc="Processing")
        else:
            iterator = input_paths
            if show_progress and not HAS_TQDM:
                logger.info(f"Processing {len(input_paths)} images...")

        for i, path in enumerate(iterator):
            try:
                # Check GPU memory before processing (75% threshold is conservative
                # to allow headroom for spikes during inference)
                if not self.memory_manager.check_memory_threshold(0.75):
                    logger.warning("High GPU memory usage, clearing cache...")
                    self.memory_manager.clear_cache()
                    # Clear depth cache when over half full to prevent memory accumulation
                    # while preserving recent entries for potential cache hits
                    if len(self._depth_cache) > self.config.depth.cache_max_size // 2:
                        self._depth_cache.clear()

                if show_progress and not HAS_TQDM:
                    logger.info(f"Processing {i+1}/{len(input_paths)}: {Path(path).name}")
                result = self.process(path, output_dir)
                results.append(result)

                # Periodic cleanup every 5 images to prevent memory fragmentation
                # and ensure consistent performance across batch
                if (i + 1) % 5 == 0:
                    self.memory_manager.clear_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM on {path}, clearing cache and skipping")
                    self.memory_manager.clear_cache()
                    self.clear_cache()
                else:
                    logger.error(f"Failed to process {path}: {e}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")

        # Final cleanup
        self.memory_manager.clear_cache()

        # Print summary
        self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: List[ProcessingResult]):
        """Print batch processing summary."""
        if not results:
            logger.warning("No images processed successfully")
            return

        total_time = sum(r.total_duration_ms for r in results)
        avg_time = total_time / len(results)

        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        avg_quality = np.mean(quality_scores) if quality_scores else 0

        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Images processed: {len(results)}")
        logger.info(f"Total time: {total_time / 1000:.1f}s")
        logger.info(f"Average time per image: {avg_time:.0f}ms")
        logger.info(f"Average quality score: {avg_quality:.2%}")
        logger.info(f"Throughput: {len(results) / (total_time / 3600000):.0f} images/hour")
        logger.info("=" * 60)

    def clear_cache(self):
        """Clear depth cache."""
        self._depth_cache.clear()
        logger.info("Depth cache cleared")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point for the 4K rendering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-End 4K Rendering Enhancement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with default preset
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg -o output/

  # Use luxury estate preset
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg -o output/ --preset luxury_estate

  # Batch process directory
  python -m transformation_portal.pipelines.rendering_4k_pipeline -d inputs/ -o outputs/ --preset editorial

  # Custom config from YAML
  python -m transformation_portal.pipelines.rendering_4k_pipeline -i input.jpg --config custom.yaml
        """,
    )

    # Input/Output
    parser.add_argument("-i", "--input", type=Path, help="Input image path")
    parser.add_argument("-d", "--directory", type=Path, help="Batch process directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output_4k"),
        help="Output directory (default: output_4k)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg,*.png,*.tif,*.tiff",
        help="Glob pattern for batch (default: *.jpg,*.png,*.tif,*.tiff)",
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        choices=list(Rendering4KPipeline.PRESETS.keys()),
        default="default",
        help="Processing preset (default: default)",
    )
    parser.add_argument("--config", type=Path, help="Custom YAML config file")

    # Processing options
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation")
    parser.add_argument("--no-upscale", action="store_true", help="Disable 4K upscaling")
    parser.add_argument(
        "--no-quality-feedback",
        action="store_true",
        help="Disable quality feedback loop",
    )

    # Utility
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show config without processing")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if not args.input and not args.directory:
        parser.error("Must specify either --input or --directory")

    # Create pipeline
    if args.config:
        pipeline = Rendering4KPipeline.from_yaml(args.config)
    else:
        pipeline = Rendering4KPipeline.from_preset(args.preset)

    # Apply CLI overrides
    if args.no_depth:
        pipeline.config.depth.enabled = False
    if args.no_upscale:
        pipeline.config.upscaling.enabled = False
    if args.no_quality_feedback:
        pipeline.config.quality_feedback.enabled = False

    # Dry run
    if args.dry_run:
        logger.info("Configuration:")
        logger.info(json.dumps(asdict(pipeline.config), indent=2, default=str))
        return 0

    # Process
    try:
        if args.directory:
            # Batch processing
            patterns = args.pattern.split(",")
            input_paths = []
            for pattern in patterns:
                input_paths.extend(args.directory.glob(pattern.strip()))
            input_paths = sorted(set(input_paths))

            if not input_paths:
                logger.error(f"No files found matching pattern in {args.directory}")
                return 1

            logger.info(f"Found {len(input_paths)} images to process")
            pipeline.batch_process(input_paths, args.output)
        else:
            # Single image
            pipeline.process(args.input, args.output)

        logger.info("✅ Processing complete!")
        return 0

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
