#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline Orchestrator for Transformation Portal.

Core orchestrator that combines all processing pipelines into a single,
configurable workflow driven by YAML recipes. Supports stage-based processing
with dry-run mode, error recovery, and comprehensive reporting.

Architecture:
    The pipeline follows a stage-based architecture:
    1. depth_estimation - Depth Anything V2 depth map generation
    2. ai_enhancement - SDXL/ControlNet enhancement (optional)
    3. material_response - Physics-based surface enhancement
    4. color_grading - LUT application and color adjustments
    5. photo_finishing - ACES, bloom, vignette, grain
    6. branding - Logo/text overlay (optional)

Example:
    from transformation_portal.pipeline_unified import UnifiedPipeline

    # Load from recipe file
    pipeline = UnifiedPipeline.from_recipe("config/recipes/signature_estate.yaml")

    # Process single image
    result = pipeline.process_single("input.jpg")

    # Batch process with dry-run
    results = pipeline.process_batch("inputs/*.jpg", "outputs/", dry_run=True)
"""

from __future__ import annotations

import glob
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Optional: RAG-based quality feedback and 4K pipeline integration
try:
    from .pipelines.quality_feedback_bridge import (
        QualityFeedbackBridge,
        QualityTargets,
        UnifiedQualityMetrics,
    )
    HAS_QUALITY_BRIDGE = True
except ImportError:
    HAS_QUALITY_BRIDGE = False
    QualityFeedbackBridge = None
    QualityTargets = None
    UnifiedQualityMetrics = None

try:
    from .pipelines.rendering_4k_pipeline import (
        Rendering4KPipeline,
        PipelineConfig as Rendering4KConfig,
        ProcessingResult as Rendering4KResult,
        QualityMetrics,
    )
    HAS_4K_PIPELINE = True
except ImportError:
    HAS_4K_PIPELINE = False
    Rendering4KPipeline = None
    Rendering4KConfig = None
    Rendering4KResult = None
    QualityMetrics = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("unified_pipeline")


@dataclass
class ProcessingResult:
    """Result of processing a single image.

    Attributes:
        input_path: Path to the input image.
        output_path: Path to the output image (None if dry-run).
        success: Whether processing succeeded.
        error_message: Error message if failed.
        stages_executed: List of stages that were executed.
        stage_times: Dictionary of stage names to execution times.
        total_time: Total processing time in seconds.
        metadata: Additional result metadata.
        quality_metrics: Quality assessment metrics (when RAG feedback enabled).
        rag_document: RAG-indexable document (when RAG enabled).
    """
    input_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    error_message: Optional[str] = None
    stages_executed: List[str] = field(default_factory=list)
    stage_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[Dict[str, float]] = None
    rag_document: Optional[Dict[str, Any]] = None

    @property
    def quality_score(self) -> float:
        """Get overall quality score (0-1)."""
        if self.quality_metrics:
            return self.quality_metrics.get('overall_score', 0.0)
        return 0.0

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        quality = f", quality={self.quality_score:.2%}" if self.quality_metrics else ""
        return f"ProcessingResult({status} {self.input_path.name}, {self.total_time:.2f}s{quality})"


@dataclass
class BatchResult:
    """Result of batch processing.

    Attributes:
        results: List of individual ProcessingResults.
        total_time: Total batch processing time.
        successful_count: Number of successful processing operations.
        failed_count: Number of failed operations.
        dry_run: Whether this was a dry-run.
    """
    results: List[ProcessingResult] = field(default_factory=list)
    total_time: float = 0.0
    successful_count: int = 0
    failed_count: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "BATCH PROCESSING SUMMARY",
            "=" * 70,
            f"Total images: {len(self.results)}",
            f"Successful: {self.successful_count}",
            f"Failed: {self.failed_count}",
            f"Total time: {self.total_time:.2f}s",
            f"Dry run: {self.dry_run}",
        ]

        if self.successful_count > 0:
            avg_time = sum(r.total_time for r in self.results if r.success) / self.successful_count
            lines.append(f"Average time per image: {avg_time:.2f}s")

        if self.failed_count > 0:
            lines.append("\nFailed images:")
            for result in self.results:
                if not result.success:
                    lines.append(f"  - {result.input_path.name}: {result.error_message}")

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class PipelineStage:
    """Represents a processing stage.

    Attributes:
        name: Stage identifier.
        display_name: Human-readable name.
        enabled: Whether stage is enabled.
        required: Whether stage failure should halt pipeline.
        config: Stage configuration dictionary.
        processor: Processing function.
    """
    name: str
    display_name: str
    enabled: bool = True
    required: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    processor: Optional[Callable] = None


class UnifiedPipeline:
    """Unified pipeline orchestrator.

    Combines all processing stages into a single workflow driven by YAML recipes.
    Supports dry-run mode, error recovery, and comprehensive reporting.

    Features:
    - Stage-based processing architecture
    - RAG-based quality feedback loop integration
    - Rendering 4K pipeline integration for high-quality output
    - Quality metrics tracking and reporting
    - LPIPS-based perceptual quality scoring (when available)
    """

    def __init__(self, recipe: Dict[str, Any]):
        """Initialize pipeline from recipe dictionary.

        Args:
            recipe: Parsed recipe dictionary.
        """
        self.recipe = recipe
        self.name = recipe.get('name', 'Unnamed Pipeline')
        self.description = recipe.get('description', '')

        # Initialize stages from recipe
        self.stages = self._initialize_stages()

        # Pipeline state
        self._depth_pipeline = None
        self._material_engine = None
        self._4k_pipeline = None
        self._quality_bridge = None

        # Detect device
        self.device = self._detect_device()

        # Initialize quality feedback bridge (RAG integration)
        self._init_quality_feedback()

        log.info(f"Initialized pipeline: {self.name}")
        log.info(f"  Stages: {[s.name for s in self.stages if s.enabled]}")
        log.info(f"  Device: {self.device}")
        log.info(f"  RAG Quality Feedback: {self._quality_bridge is not None}")
        log.info(f"  4K Pipeline: {HAS_4K_PIPELINE}")

    def _init_quality_feedback(self) -> None:
        """Initialize RAG-based quality feedback bridge."""
        quality_config = self.recipe.get('quality_feedback', {})
        if not quality_config.get('enabled', True):
            return

        if not HAS_QUALITY_BRIDGE:
            log.info("Quality feedback bridge not available (optional dependency)")
            return

        try:
            # Create quality targets from recipe config
            targets = QualityTargets(
                perceptual_percentile_target=quality_config.get(
                    'perceptual_percentile_target', 95.0
                ),
                material_fidelity_target=quality_config.get(
                    'material_fidelity_target', 0.98
                ),
            )

            # Initialize the bridge
            self._quality_bridge = QualityFeedbackBridge(
                targets=targets,
                hybrid_mode=quality_config.get('hybrid_mode', True),
                lpips_network=quality_config.get('lpips_network', 'alex'),
                enable_material_fidelity=quality_config.get('enable_material_fidelity', True),
                rag_callback=self._create_rag_callback() if quality_config.get('rag_indexing_enabled', False) else None,
            )
            log.info("RAG-based quality feedback bridge initialized")

        except Exception as e:
            log.warning(f"Failed to initialize quality feedback bridge: {e}")
            self._quality_bridge = None

    def _create_rag_callback(self) -> Optional[Callable[[Dict], None]]:
        """Create callback for RAG indexing of quality metrics."""
        rag_config = self.recipe.get('quality_feedback', {})
        rag_index_path = rag_config.get('rag_index_path')

        if not rag_index_path:
            return None

        def rag_callback(document: Dict) -> None:
            """Callback to index quality metrics into RAG system."""
            import json
            output_path = Path(rag_index_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Append document to RAG index file (JSON Lines format)
            with open(output_path, 'a') as f:
                f.write(json.dumps(document) + '\n')
            log.debug(f"Indexed quality document to RAG: {document.get('image_id', 'unknown')}")

        return rag_callback

    @classmethod
    def from_recipe(cls, recipe_path: Union[str, Path]) -> "UnifiedPipeline":
        """Create pipeline from a YAML recipe file.

        Args:
            recipe_path: Path to the recipe YAML file.

        Returns:
            UnifiedPipeline instance.

        Raises:
            FileNotFoundError: If recipe file doesn't exist.
            ValueError: If recipe is invalid.

        Example:
            pipeline = UnifiedPipeline.from_recipe("config/recipes/signature_estate.yaml")
        """
        from .config_loader import load_recipe, validate_recipe

        recipe = load_recipe(recipe_path)
        is_valid, errors = validate_recipe(recipe)

        if not is_valid:
            raise ValueError(f"Invalid recipe: {'; '.join(errors)}")

        return cls(recipe)

    def _initialize_stages(self) -> List[PipelineStage]:
        """Initialize processing stages from recipe."""
        stages = []
        stage_order = self.recipe.get('stages', [])

        stage_definitions = {
            'depth_estimation': ('Depth Estimation', False),
            'ai_enhancement': ('AI Enhancement', False),
            'material_response': ('Material Response', False),
            'color_grading': ('Color Grading', False),
            'photo_finishing': ('Photo Finishing', False),
            'branding': ('Branding', False),
            'upscaling_4k': ('4K Upscaling', False),
            'quality_assessment': ('Quality Assessment', False),
        }

        for stage_name in stage_order:
            if stage_name in stage_definitions:
                display_name, required = stage_definitions[stage_name]
                stage_config = self.recipe.get(stage_name, {})
                enabled = stage_config.get('enabled', True)

                stages.append(PipelineStage(
                    name=stage_name,
                    display_name=display_name,
                    enabled=enabled,
                    required=required,
                    config=stage_config,
                ))

        # Always add quality assessment at the end if quality feedback is enabled
        quality_config = self.recipe.get('quality_feedback', {})
        if quality_config.get('enabled', True) and 'quality_assessment' not in stage_order:
            stages.append(PipelineStage(
                name='quality_assessment',
                display_name='Quality Assessment',
                enabled=True,
                required=False,
                config=quality_config,
            ))

        return stages

    def _detect_device(self) -> str:
        """Auto-detect best available processing device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                try:
                    if torch.backends.mps.is_available():
                        return "mps"
                except RuntimeError:
                    # MPS availability check can raise RuntimeError
                    pass
        except ImportError:
            pass
        return "cpu"

    def process_single(self, input_path: Union[str, Path]) -> ProcessingResult:
        """Process a single image through the pipeline.

        Args:
            input_path: Path to the input image.

        Returns:
            ProcessingResult with processing details.

        Example:
            result = pipeline.process_single("render.jpg")
            if result.success:
                print(f"Output: {result.output_path}")
        """
        input_path = Path(input_path)
        start_time = time.time()

        result = ProcessingResult(input_path=input_path)

        # Store original image for quality comparison
        original_image = None

        try:
            # Load image
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            log.info(f"Processing: {input_path.name}")
            image = Image.open(input_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Store original for quality comparison
            original_image = np.array(image).astype(np.float32) / 255.0

            # Execute stages
            for stage in self.stages:
                if not stage.enabled:
                    continue

                stage_start = time.time()
                try:
                    if stage.name == 'quality_assessment':
                        # Quality assessment is handled separately at the end
                        quality_result = self._apply_quality_assessment(
                            image, original_image, input_path, stage.config
                        )
                        if quality_result:
                            result.quality_metrics = quality_result.get('metrics')
                            result.rag_document = quality_result.get('rag_document')
                    else:
                        image = self._execute_stage(stage, image)

                    result.stages_executed.append(stage.name)
                    result.stage_times[stage.name] = time.time() - stage_start
                    log.info(f"  ✓ {stage.display_name}: {result.stage_times[stage.name]:.2f}s")

                except Exception as e:
                    result.stage_times[stage.name] = time.time() - stage_start
                    if stage.required:
                        raise
                    log.warning(f"  ⚠ {stage.display_name} failed: {e}")

            # Generate output
            output_path = self._generate_output(image, input_path)
            result.output_path = output_path
            result.success = True

            # Log quality score if available
            if result.quality_metrics:
                log.info(f"  Quality score: {result.quality_score:.2%}")

        except Exception as e:
            result.error_message = str(e)
            log.error(f"Pipeline failed: {e}")

        result.total_time = time.time() - start_time
        return result

    def process_batch(
        self,
        input_glob: str,
        output_dir: Union[str, Path],
        mode: str = "default",
        dry_run: bool = False
    ) -> BatchResult:
        """Process multiple images matching a glob pattern.

        Args:
            input_glob: Glob pattern for input files (e.g., "inputs/*.jpg").
            output_dir: Output directory path.
            mode: Processing mode ("default", "parallel").
            dry_run: If True, preview processing plan without executing.

        Returns:
            BatchResult with all processing results.

        Example:
            results = pipeline.process_batch(
                "renders/*.exr",
                "outputs/",
                dry_run=True  # Preview first
            )
            print(results.summary())
        """
        output_dir = Path(output_dir)
        batch_start = time.time()

        # Find matching files
        input_files = sorted([Path(p) for p in glob.glob(input_glob)])

        if not input_files:
            log.warning(f"No files matched pattern: {input_glob}")
            return BatchResult(dry_run=dry_run)

        log.info(f"Batch processing {len(input_files)} files")
        log.info(f"  Output: {output_dir}")
        log.info(f"  Mode: {mode}")
        log.info(f"  Dry run: {dry_run}")

        if dry_run:
            return self._dry_run_batch(input_files, output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store original output config and override
        original_output_dir = self.recipe.get('_output_dir')
        self.recipe['_output_dir'] = str(output_dir)

        batch_result = BatchResult(dry_run=dry_run)

        try:
            for i, input_path in enumerate(input_files, 1):
                log.info(f"\n[{i}/{len(input_files)}] {input_path.name}")
                result = self.process_single(input_path)
                batch_result.results.append(result)

                if result.success:
                    batch_result.successful_count += 1
                else:
                    batch_result.failed_count += 1

        finally:
            # Restore original output config
            if original_output_dir:
                self.recipe['_output_dir'] = original_output_dir

        batch_result.total_time = time.time() - batch_start

        log.info(batch_result.summary())
        return batch_result

    def _dry_run_batch(self, input_files: List[Path], output_dir: Path) -> BatchResult:
        """Generate dry-run preview of batch processing.

        Args:
            input_files: List of input file paths.
            output_dir: Output directory path.

        Returns:
            BatchResult with processing plan preview.
        """
        log.info("\n" + "=" * 70)
        log.info("DRY RUN - Processing Plan Preview")
        log.info("=" * 70)
        log.info(f"Pipeline: {self.name}")
        log.info(f"Description: {self.description}")
        log.info(f"Input files: {len(input_files)}")
        log.info(f"Output directory: {output_dir}")
        log.info("")
        log.info("Stages to execute:")
        for stage in self.stages:
            status = "✓ Enabled" if stage.enabled else "✗ Disabled"
            required = " (required)" if stage.required else ""
            log.info(f"  {stage.display_name}: {status}{required}")

        log.info("")
        log.info("Files to process:")
        for i, f in enumerate(input_files[:10], 1):
            output_name = self._get_output_name(f)
            log.info(f"  {i}. {f.name} → {output_name}")
        if len(input_files) > 10:
            log.info(f"  ... and {len(input_files) - 10} more files")

        log.info("")
        log.info("Output configuration:")
        output_config = self.recipe.get('output', {})
        log.info(f"  Format: {output_config.get('format', 'tiff')}")
        log.info(f"  Quality: {output_config.get('quality', 95)}")

        log.info("=" * 70)
        log.info("To execute, run again without dry_run=True")
        log.info("=" * 70)

        return BatchResult(
            results=[ProcessingResult(input_path=f, success=True) for f in input_files],
            dry_run=True,
            successful_count=len(input_files),
        )

    def _execute_stage(self, stage: PipelineStage, image: Image.Image) -> Image.Image:
        """Execute a processing stage.

        Args:
            stage: Stage to execute.
            image: Input image.

        Returns:
            Processed image.
        """
        if stage.name == 'depth_estimation':
            return self._apply_depth_estimation(image, stage.config)
        elif stage.name == 'ai_enhancement':
            return self._apply_ai_enhancement(image, stage.config)
        elif stage.name == 'material_response':
            return self._apply_material_response(image, stage.config)
        elif stage.name == 'color_grading':
            return self._apply_color_grading(image, stage.config)
        elif stage.name == 'photo_finishing':
            return self._apply_photo_finishing(image, stage.config)
        elif stage.name == 'branding':
            return self._apply_branding(image, stage.config)
        elif stage.name == 'upscaling_4k':
            return self._apply_upscaling_4k(image, stage.config)
        elif stage.name == 'quality_assessment':
            # Quality assessment is handled separately in process_single
            return image
        else:
            log.warning(f"Unknown stage: {stage.name}")
            return image

    def _apply_upscaling_4k(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply 4K upscaling using the Rendering4KPipeline if available.

        Args:
            image: Input image.
            config: Upscaling configuration.

        Returns:
            Upscaled image.
        """
        if not HAS_4K_PIPELINE:
            log.info("    4K upscaling not available (optional dependency)")
            # Fallback to basic Lanczos upscaling
            target_w = config.get('target_width', 3840)
            target_h = config.get('target_height', 2160)
            current_w, current_h = image.size

            if current_w >= target_w and current_h >= target_h:
                return image

            # Scale to fit within target while maintaining aspect ratio
            scale_w = target_w / current_w
            scale_h = target_h / current_h
            scale = min(scale_w, scale_h)

            new_w = int(current_w * scale)
            new_h = int(current_h * scale)

            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Use Rendering4KPipeline for high-quality upscaling
        try:
            from .pipelines.rendering_4k_pipeline import apply_upscaling, UpscalingConfig

            upscale_config = UpscalingConfig(
                enabled=True,
                target_resolution=(
                    config.get('target_width', 3840),
                    config.get('target_height', 2160)
                ),
                method=config.get('method', 'lanczos'),
                scale_factor=config.get('scale_factor', 4),
                preserve_sharpness=config.get('preserve_sharpness', True),
            )

            return apply_upscaling(image, upscale_config)

        except Exception as e:
            log.warning(f"    4K pipeline upscaling failed: {e}")
            return image

    def _apply_quality_assessment(
        self,
        image: Image.Image,
        original_image: Optional[np.ndarray],
        input_path: Path,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply quality assessment using RAG-based quality feedback bridge.

        Args:
            image: Processed image.
            original_image: Original image as numpy array (for comparison).
            input_path: Path to input file.
            config: Quality assessment configuration.

        Returns:
            Dictionary with metrics and RAG document, or None if unavailable.
        """
        if self._quality_bridge is None:
            # Fallback to basic quality metrics
            return self._compute_basic_quality_metrics(image)

        try:
            # Convert processed image to numpy
            enhanced_np = np.array(image).astype(np.float32) / 255.0

            # Use quality feedback bridge for comprehensive assessment
            unified_metrics = self._quality_bridge.assess(
                enhanced=enhanced_np,
                original=original_image,
                image_id=input_path.stem,
                pipeline_config_name=self.name,
            )

            # Log quality summary
            log.info(f"    Hybrid Score: {unified_metrics.hybrid_score:.1f}/100")
            if unified_metrics.lpips_available:
                log.info(f"    Perceptual: {unified_metrics.perceptual_composite:.1f}/100")
                log.info(f"    Material Fidelity: {unified_metrics.material_fidelity.overall_fidelity:.1%}")
            log.info(f"    {unified_metrics.targets_summary}")

            return {
                'metrics': {
                    'overall_score': unified_metrics.hybrid_score / 100.0,
                    'perceptual_score': unified_metrics.perceptual_composite,
                    'heuristic_score': unified_metrics.heuristic_composite,
                    'sharpness': unified_metrics.heuristic.sharpness,
                    'contrast': unified_metrics.heuristic.contrast,
                    'colorfulness': unified_metrics.heuristic.colorfulness,
                    'lpips_available': unified_metrics.lpips_available,
                },
                'rag_document': unified_metrics.to_rag_document(),
            }

        except Exception as e:
            log.warning(f"    Quality assessment failed: {e}")
            return self._compute_basic_quality_metrics(image)

    def _compute_basic_quality_metrics(
        self,
        image: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """Compute basic quality metrics without RAG bridge.

        Args:
            image: Processed image.

        Returns:
            Dictionary with basic metrics.
        """
        try:
            arr = np.array(image).astype(np.float32) / 255.0

            # Sharpness (Laplacian variance)
            gray = np.mean(arr, axis=2)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            # Simple convolution
            h, w = gray.shape
            padded = np.pad(gray, 1, mode='reflect')
            laplacian = (
                kernel[0, 1] * padded[0:h, 1:w+1] +
                kernel[1, 0] * padded[1:h+1, 0:w] +
                kernel[1, 1] * padded[1:h+1, 1:w+1] +
                kernel[1, 2] * padded[1:h+1, 2:w+2] +
                kernel[2, 1] * padded[2:h+2, 1:w+1]
            )
            sharpness = float(np.clip(np.var(laplacian) * 50, 0, 1))

            # Contrast
            lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            contrast = float(np.clip(np.std(lum) * 3, 0, 1))

            # Colorfulness
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            rg, yb = r - g, 0.5 * (r + g) - b
            colorfulness = float(np.clip(
                (np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)) * 2,
                0, 1
            ))

            # Overall score
            overall = 0.35 * sharpness + 0.30 * contrast + 0.35 * colorfulness

            return {
                'metrics': {
                    'overall_score': overall,
                    'sharpness': sharpness,
                    'contrast': contrast,
                    'colorfulness': colorfulness,
                    'lpips_available': False,
                },
                'rag_document': None,
            }

        except Exception as e:
            log.warning(f"    Basic quality metrics failed: {e}")
            return None

    def _apply_depth_estimation(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply depth-aware processing."""
        # Depth estimation stage (placeholder for integration)
        log.info("    Depth estimation stage (integration pending)")
        return image

    def _apply_ai_enhancement(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply AI enhancement (SDXL/ControlNet)."""
        # AI enhancement stage (placeholder for heavy ML integration)
        log.info("    AI enhancement stage (integration pending)")
        return image

    def _apply_material_response(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply Material Response processing."""
        from .processors.material_response.engine import MaterialResponseEngine

        if self._material_engine is None:
            self._material_engine = MaterialResponseEngine.from_config(config)

        return self._material_engine.apply(image)

    def _apply_color_grading(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply color grading with LUT."""
        arr = np.array(image).astype(np.float32) / 255.0

        # Apply exposure
        exposure = config.get('exposure', 0.0)
        if abs(exposure) > 0.001:
            arr = arr * (2.0 ** exposure)

        # Apply contrast
        contrast = config.get('contrast', 1.0)
        if abs(contrast - 1.0) > 0.001:
            midpoint = 0.5
            arr = (arr - midpoint) * contrast + midpoint

        # Apply saturation
        saturation = config.get('saturation', 1.0)
        if abs(saturation - 1.0) > 0.001:
            luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            arr = luminance[..., np.newaxis] + (arr - luminance[..., np.newaxis]) * saturation

        # Apply warmth
        warmth = config.get('warmth', 0.0)
        if abs(warmth) > 0.001:
            arr[..., 0] = arr[..., 0] + warmth  # Red
            arr[..., 2] = arr[..., 2] - warmth  # Blue

        # LUT application placeholder
        lut_path = config.get('lut')
        lut_strength = config.get('lut_strength', 0.7)
        if lut_path and Path(lut_path).exists():
            log.info(f"    LUT: {Path(lut_path).name} @ {lut_strength:.0%}")
            # LUT application would go here

        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8), 'RGB')

    def _apply_photo_finishing(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply photo finishing (ACES, bloom, vignette, grain)."""
        from scipy.ndimage import gaussian_filter

        arr = np.array(image).astype(np.float32) / 255.0

        # ACES tone mapping
        if config.get('aces', True):
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            arr = np.clip((arr * (a * arr + b)) / (arr * (c * arr + d) + e), 0.0, 1.0)

        # Bloom
        bloom_config = config.get('bloom', {})
        if bloom_config.get('enabled', True):
            threshold = bloom_config.get('threshold', 0.8)
            intensity = bloom_config.get('intensity', 0.25)

            lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            mask = (lum > threshold).astype(np.float32)
            glow = np.stack([gaussian_filter(arr[..., i] * mask, 9) for i in range(3)], axis=-1)
            arr = np.clip(arr + intensity * glow, 0.0, 1.0)

        # Vignette
        vignette_config = config.get('vignette', {})
        if vignette_config.get('enabled', True):
            strength = vignette_config.get('strength', 0.18)
            h, w = arr.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w / 2.0, h / 2.0
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            r = r / (r.max() + 1e-6)
            mask = 1.0 - strength * (r ** 2)
            arr = np.clip(arr * mask[..., np.newaxis], 0.0, 1.0)

        # Grain
        grain_config = config.get('grain', {})
        if grain_config.get('enabled', True):
            amount = grain_config.get('amount', 0.012)
            rng = np.random.default_rng(42)
            noise = rng.normal(0.0, amount, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)

        return Image.fromarray((arr * 255).astype(np.uint8), 'RGB')

    def _apply_branding(
        self,
        image: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Apply branding overlay (logo/text)."""
        if not config.get('enabled', False):
            return image

        from PIL import ImageDraw, ImageFont

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        width_px, height_px = canvas.size
        margin = 36

        # Text overlay
        text = config.get('text')
        if text:
            try:
                font = ImageFont.truetype("arial.ttf", size=max(22, height_px // 40))
            except (OSError, IOError):
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = margin
            y = height_px - th - margin
            pad = 14
            draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 160))
            draw.text((x, y), text, fill=(255, 255, 255, 230), font=font)

        # Logo overlay
        logo_path = config.get('logo')
        if logo_path and Path(logo_path).exists():
            logo = Image.open(logo_path).convert("RGBA")
            target_h = int(min(width_px, height_px) * 0.12)
            scale = target_h / logo.height
            new_size = (int(logo.width * scale), target_h)
            logo = logo.resize(new_size, Image.Resampling.LANCZOS)
            lx = width_px - logo.width - margin
            ly = height_px - logo.height - margin
            if canvas.mode == "RGBA":
                canvas.alpha_composite(logo, (lx, ly))
            else:
                canvas.paste(logo, (lx, ly), logo)

        return canvas

    def _generate_output(self, image: Image.Image, input_path: Path) -> Path:
        """Generate output file.

        Args:
            image: Processed image.
            input_path: Original input path.

        Returns:
            Path to output file.
        """
        output_config = self.recipe.get('output', {})
        output_format = output_config.get('format', 'tiff').lower()
        quality = output_config.get('quality', 95)

        # Determine output directory
        output_dir = self.recipe.get('_output_dir')
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = input_path.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_name = self._get_output_name(input_path)
        output_path = output_dir / output_name

        # Save based on format
        if output_format == 'jpeg':
            image.save(output_path, quality=quality, subsampling=0, optimize=True)
        elif output_format == 'png':
            image.save(output_path, compress_level=6)
        elif output_format == 'tiff':
            # Try to use tifffile for 16-bit support
            try:
                import tifffile
                arr = np.array(image).astype(np.float32) / 255.0
                arr_16bit = (np.clip(arr, 0.0, 1.0) * 65535).astype(np.uint16)
                tifffile.imwrite(output_path, arr_16bit, compression='lzw')
            except ImportError:
                image.save(output_path, compression='lzw')
        else:
            image.save(output_path)

        log.info(f"  Output: {output_path}")
        return output_path

    def _get_output_name(self, input_path: Path) -> str:
        """Generate output filename.

        Args:
            input_path: Input file path.

        Returns:
            Output filename.
        """
        output_config = self.recipe.get('output', {})
        output_format = output_config.get('format', 'tiff').lower()

        ext_map = {
            'jpeg': '.jpg',
            'png': '.png',
            'tiff': '.tif',
            'exr': '.exr',
        }
        ext = ext_map.get(output_format, '.tif')

        # Use recipe name in output filename
        recipe_slug = self.name.lower().replace(' ', '_').replace('-', '_')
        return f"{input_path.stem}_{recipe_slug}{ext}"


__all__ = [
    'UnifiedPipeline',
    'ProcessingResult',
    'BatchResult',
    'PipelineStage',
    'HAS_QUALITY_BRIDGE',
    'HAS_4K_PIPELINE',
]
