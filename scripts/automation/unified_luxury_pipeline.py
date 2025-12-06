#!/usr/bin/env python3
"""
Unified Luxury Rendering Pipeline - Production Grade
=====================================================

Integrates three advanced processing stages for maximum quality:
1. Advanced Upscaling (SwinIR/Real-ESRGAN with 16-bit precision)
2. Depth-Aware Processing (Depth Anything V2 with Apple Neural Engine)
3. Luxury Enhancements (Material Response, Color Grading, LUTs)

Key Features:
- Intelligent quality/speed optimization based on scene analysis
- 16-bit TIFF workflow (end-to-end precision preservation)
- Batch processing with progress tracking
- Memory-efficient tile-based processing
- Comprehensive quality metrics and reporting
- Multiple preset workflows (photo, architectural, archival)

Usage:
    # Single image with auto-detection
    python unified_luxury_pipeline.py input.tif --preset photo_realistic
    
    # Batch processing
    python unified_luxury_pipeline.py input_dir/ --batch --preset architectural
    
    # Custom configuration
    python unified_luxury_pipeline.py input.tif --upscale-model swinir_real_4x \\
        --enable-depth --material-response 0.8 --lut signature_estate
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from tifffile import TiffFile, imwrite as tiff_write
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

# Import pipeline components
try:
    from utils.upscaling_engine import (
        UpscalingEngine,
        UpscalingConfig,
        UpscalingModel,
        UpscalingMetrics
    )
    UPSCALING_AVAILABLE = True
except ImportError:
    UPSCALING_AVAILABLE = False
    logging.warning("Upscaling engine not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - ML features disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelinePreset(Enum):
    """Pre-configured pipeline presets for common use cases."""
    PHOTO_REALISTIC = "photo_realistic"
    ARCHITECTURAL = "architectural"
    ARCHIVAL_QUALITY = "archival_quality"
    FAST_BATCH = "fast_batch"
    SIGNATURE_ESTATE = "signature_estate"
    INTERIOR_LUXURY = "interior_luxury"
    EXTERIOR_SHOWCASE = "exterior_showcase"
    
    @property
    def description(self) -> str:
        """Get preset description."""
        descriptions = {
            self.PHOTO_REALISTIC: "Maximum quality for photographic images (SwinIR + full depth)",
            self.ARCHITECTURAL: "Optimized for architectural renders (balanced quality/speed)",
            self.ARCHIVAL_QUALITY: "Museum-grade 16-bit preservation (SwinIR + strict validation)",
            self.FAST_BATCH: "Speed-optimized for large batches (Real-ESRGAN + minimal processing)",
            self.SIGNATURE_ESTATE: "Luxury estate marketing (full enhancement suite)",
            self.INTERIOR_LUXURY: "Interior spaces (depth + material response emphasis)",
            self.EXTERIOR_SHOWCASE: "Exterior views (atmospheric effects + color grading)",
        }
        return descriptions[self]


@dataclass
class UnifiedPipelineConfig:
    """Configuration for unified luxury rendering pipeline."""
    
    # Input/Output
    input_path: Path
    output_dir: Path
    preset: PipelinePreset = PipelinePreset.PHOTO_REALISTIC
    
    # Stage Enablement
    enable_upscaling: bool = True
    enable_depth_processing: bool = True
    enable_material_response: bool = True
    enable_color_grading: bool = True
    
    # Upscaling Configuration
    upscale_model: UpscalingModel = UpscalingModel.SWINIR_REAL_4X
    upscale_factor: int = 4
    tile_size: int = 0  # Auto-detect
    
    # Depth Processing
    depth_model: str = "depth_anything_v2"
    depth_tile_size: int = 518
    zone_based_processing: bool = True
    
    # Material Response
    material_strength: float = 0.75
    surface_types: List[str] = field(default_factory=lambda: ["wood", "metal", "glass", "stone"])
    
    # Color Grading
    lut_name: Optional[str] = "signature_estate"
    lut_strength: float = 0.70
    color_temperature_shift: float = 0.0
    saturation_boost: float = 1.08
    
    # Quality Settings
    preserve_16bit: bool = True
    validate_colors: bool = True
    color_tolerance: float = 0.02
    
    # Performance
    device: str = "auto"
    batch_size: int = 1
    cache_models: bool = True
    
    # Output Options
    save_intermediate: bool = False
    generate_report: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["tiff", "png"])
    
    def __post_init__(self):
        """Apply preset configurations."""
        self.apply_preset(self.preset)
    
    def apply_preset(self, preset: PipelinePreset):
        """Apply preset configuration."""
        if preset == PipelinePreset.PHOTO_REALISTIC:
            self.upscale_model = UpscalingModel.SWINIR_REAL_4X
            self.enable_depth_processing = True
            self.material_strength = 0.80
            self.validate_colors = True
            
        elif preset == PipelinePreset.ARCHITECTURAL:
            self.upscale_model = UpscalingModel.REALESRGAN_4X
            self.enable_depth_processing = True
            self.zone_based_processing = True
            self.material_strength = 0.70
            
        elif preset == PipelinePreset.ARCHIVAL_QUALITY:
            self.upscale_model = UpscalingModel.SWINIR_REAL_4X
            self.preserve_16bit = True
            self.validate_colors = True
            self.color_tolerance = 0.015  # Strict
            self.save_intermediate = True
            
        elif preset == PipelinePreset.FAST_BATCH:
            self.upscale_model = UpscalingModel.REALESRGAN_4X
            self.enable_depth_processing = False
            self.material_strength = 0.60
            self.validate_colors = False
            
        elif preset == PipelinePreset.SIGNATURE_ESTATE:
            self.upscale_model = UpscalingModel.SWINIR_REAL_4X
            self.enable_depth_processing = True
            self.material_strength = 0.85
            self.lut_name = "signature_estate"
            self.saturation_boost = 1.10
            
        elif preset == PipelinePreset.INTERIOR_LUXURY:
            self.upscale_model = UpscalingModel.SWINIR_REAL_4X
            self.enable_depth_processing = True
            self.zone_based_processing = True
            self.material_strength = 0.80
            self.surface_types = ["wood", "metal", "glass", "fabric"]
            
        elif preset == PipelinePreset.EXTERIOR_SHOWCASE:
            self.upscale_model = UpscalingModel.SWINIR_REAL_4X
            self.enable_depth_processing = True
            self.material_strength = 0.70
            self.color_temperature_shift = 0.03  # Warmer


@dataclass
class PipelineResult:
    """Results from pipeline processing."""
    input_path: Path
    output_path: Path
    processing_time: float
    
    # Stage-specific results
    upscaling_metrics: Optional[UpscalingMetrics] = None
    depth_map_generated: bool = False
    material_response_applied: bool = False
    lut_applied: bool = False
    color_grading_applied: bool = False
    
    # Quality metrics
    final_size: Tuple[int, int] = (0, 0)
    color_deviation: float = 0.0
    bit_depth: int = 8
    file_size_mb: float = 0.0
    
    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Pipeline Processing Complete",
            f"{'=' * 60}",
            f"Input:  {self.input_path.name}",
            f"Output: {self.output_path.name}",
            f"Time:   {self.processing_time:.2f}s",
            f"",
            f"Pipeline Stages:",
            f"  Upscaling:         {'✓' if self.upscaling_metrics else '✗'}",
            f"  Depth Processing:  {'✓' if self.depth_map_generated else '✗'}",
            f"  Material Response: {'✓' if self.material_response_applied else '✗'}",
            f"  LUT Application:   {'✓' if self.lut_applied else '✗'}",
            f"  Color Grading:     {'✓' if self.color_grading_applied else '✗'}",
            f"",
            f"Output Quality:",
            f"  Resolution:   {self.final_size[0]}x{self.final_size[1]}",
            f"  Bit Depth:    {self.bit_depth}-bit",
            f"  File Size:    {self.file_size_mb:.1f} MB",
            f"  Color Dev:    {self.color_deviation:.4f}",
        ]
        
        if self.upscaling_metrics:
            lines.extend([
                f"",
                f"Upscaling Details:",
                f"  Model:    {self.upscaling_metrics.model_name}",
                f"  Tiles:    {self.upscaling_metrics.tiles_processed}",
                f"  Memory:   {self.upscaling_metrics.memory_peak_mb:.0f} MB",
            ])
        
        if self.warnings:
            lines.extend([
                f"",
                f"Warnings:",
            ])
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        return "\n".join(lines)


class UnifiedLuxuryPipeline:
    """
    Production-grade unified rendering pipeline.
    
    Orchestrates advanced upscaling, depth-aware processing, and luxury
    enhancements into a cohesive workflow optimized for luxury real estate.
    """
    
    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        
        # Initialize components
        self.upscaler = None
        self.depth_processor = None
        self.material_responder = None
        self.lut_processor = None
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_time': 0.0,
            'stage_times': {},
        }
        
        logger.info(f"Initializing Unified Luxury Pipeline")
        logger.info(f"Preset: {config.preset.value} - {config.preset.description}")
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components based on configuration."""
        
        # 1. Upscaling Engine
        if self.config.enable_upscaling and UPSCALING_AVAILABLE:
            logger.info(f"Initializing upscaling: {self.config.upscale_model.value}")
            upscale_config = UpscalingConfig(
                model=self.config.upscale_model,
                tile_size=self.config.tile_size,
                preserve_16bit=self.config.preserve_16bit,
                validate_colors=self.config.validate_colors,
                color_tolerance=self.config.color_tolerance,
                device=self.config.device,
                cache_model=self.config.cache_models
            )
            self.upscaler = UpscalingEngine(upscale_config)
            logger.info(f"✓ Upscaling ready on device: {self.upscaler.device}")
        else:
            logger.info("Upscaling disabled or unavailable")
        
        # 2. Depth Processor
        if self.config.enable_depth_processing and TORCH_AVAILABLE:
            logger.info(f"Initializing depth processing: {self.config.depth_model}")
            try:
                from utils.depth_processor import DepthProcessor, DepthConfig
                
                depth_config = DepthConfig(
                    model_name=self.config.depth_model,
                    tile_size=self.config.depth_tile_size,
                    enable_zone_processing=self.config.zone_based_processing,
                    device=self.config.device
                )
                self.depth_processor = DepthProcessor(depth_config)
                logger.info("✓ Depth processing ready")
            except Exception as e:
                logger.warning(f"Depth processor initialization failed: {e}")
                self.depth_processor = None
        else:
            logger.info("Depth processing disabled or unavailable")
        
        # 3. Material Response
        if self.config.enable_material_response:
            logger.info(f"Initializing material response (strength: {self.config.material_strength})")
            try:
                from utils.material_responder import MaterialResponder, MaterialResponseConfig
                
                material_config = MaterialResponseConfig(
                    strength=self.config.material_strength,
                    surface_types=self.config.surface_types,
                    depth_aware=True
                )
                self.material_responder = MaterialResponder(material_config)
                logger.info("✓ Material response ready")
            except Exception as e:
                logger.warning(f"Material responder initialization failed: {e}")
                self.material_responder = None
        else:
            logger.info("Material response disabled")
        
        # 4. LUT Processor (Phase 3)
        if self.config.enable_color_grading and self.config.lut_name:
            logger.info(f"Initializing LUT processor: {self.config.lut_name}")
            try:
                from utils.lut_processor import LUTProcessor, LUTConfig, LUTCategory
                
                # Find LUT file
                lut_path = self._find_lut_file(self.config.lut_name)
                if lut_path:
                    lut_config = LUTConfig(
                        lut_path=lut_path,
                        strength=self.config.lut_strength,
                        preserve_highlights=True,
                        preserve_blacks=True
                    )
                    self.lut_processor = LUTProcessor(lut_config)
                    logger.info(f"✓ LUT ready: {lut_path.name}")
                else:
                    logger.warning(f"LUT not found: {self.config.lut_name}")
                    self.lut_processor = None
            except Exception as e:
                logger.warning(f"LUT processor initialization failed: {e}")
                self.lut_processor = None
        else:
            logger.info("LUT processing disabled or no LUT specified")
    
    def _find_lut_file(self, lut_name: str) -> Optional[Path]:
        """
        Find LUT file in assets directory.
        
        Args:
            lut_name: LUT name or filename
        
        Returns:
            Path to LUT file or None
        """
        luts_dir = Path(__file__).parent / "assets" / "luts"
        
        if not luts_dir.exists():
            return None
        
        # Preset name mappings
        preset_mappings = {
            'signature_estate': 'Montecito_Golden_Hour_HDR',
            'photo_realistic': 'Kodak_2393_D55',
            'archival': 'Kodak_2393_D55',
            'film': 'FilmConvert_Nitrate_LuxuryRE',
            'warm': 'Spanish_Colonial_Warm_HDR',
            'golden_hour': 'Montecito_Golden_Hour_HDR'
        }
        
        # Check preset mappings first
        mapped_name = preset_mappings.get(lut_name.lower(), lut_name)
        
        # Search in all subdirectories
        for lut_path in luts_dir.rglob("*.cube"):
            stem_lower = lut_path.stem.lower()
            search_lower = mapped_name.lower()
            
            # Exact match
            if stem_lower == search_lower:
                return lut_path
            
            # Partial match
            if search_lower in stem_lower or stem_lower in search_lower:
                return lut_path
        
        # Try exact filename match
        lut_file = luts_dir / f"{mapped_name}.cube"
        if lut_file.exists():
            return lut_file
        
        return None
    
    def process_image(
        self,
        input_path: Union[Path, str],
        output_path: Optional[Path] = None
    ) -> PipelineResult:
        """
        Process single image through unified pipeline.
        
        Args:
            input_path: Input image path
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            PipelineResult with metrics and paths
        """
        input_path = Path(input_path)
        start_time = time.time()
        
        logger.info(f"\nProcessing: {input_path.name}")
        logger.info("=" * 60)
        
        # Generate output path
        if output_path is None:
            output_name = f"{input_path.stem}_{self.config.preset.value}"
            output_path = self.config.output_dir / f"{output_name}.tif"
        
        # Load input image
        logger.info("Stage 1: Loading image...")
        image = self._load_image(input_path)
        original_size = (image.shape[1], image.shape[0])
        logger.info(f"  Input size: {original_size[0]}x{original_size[1]}")
        
        # Initialize result
        result = PipelineResult(
            input_path=input_path,
            output_path=output_path,
            processing_time=0.0
        )
        
        # Stage 2: Upscaling
        upscaling_metrics = None
        if self.config.enable_upscaling and self.upscaler:
            logger.info("\nStage 2: AI Upscaling...")
            stage_start = time.time()
            
            image, upscaling_metrics = self.upscaler.upscale_image(
                image,
                output_path=None  # Don't save yet
            )
            
            stage_time = time.time() - stage_start
            self.stats['stage_times']['upscaling'] = stage_time
            result.upscaling_metrics = upscaling_metrics
            
            logger.info(f"  ✓ Upscaled to {image.shape[1]}x{image.shape[0]} in {stage_time:.2f}s")
            logger.info(f"  Color deviation: {upscaling_metrics.color_deviation:.4f}")
        
        # Stage 3: Depth Processing
        depth_map = None
        if self.config.enable_depth_processing and self.depth_processor:
            logger.info("\nStage 3: Depth-Aware Processing...")
            stage_start = time.time()
            
            try:
                # Estimate depth and apply zone-based adjustments
                image, depth_map = self.depth_processor.process(image)
                result.depth_map_generated = (depth_map is not None)
                
                if depth_map is not None:
                    logger.info(f"  ✓ Depth processing complete")
                    
                    # Optionally save depth map visualization
                    if self.config.save_intermediate:
                        depth_viz_path = output_path.parent / f"{output_path.stem}_depth.png"
                        self.depth_processor.save_depth_visualization(depth_map, depth_viz_path)
                else:
                    logger.info("  ⚠️  Depth estimation unavailable")
                    
            except Exception as e:
                logger.error(f"  ✗ Depth processing failed: {e}")
                result.depth_map_generated = False
            
            stage_time = time.time() - stage_start
            self.stats['stage_times']['depth'] = stage_time
        elif self.config.enable_depth_processing:
            logger.info("\nStage 3: Depth processing skipped (processor not available)")
        
        # Stage 4: Material Response
        if self.config.enable_material_response and self.material_responder:
            logger.info("\nStage 4: Material Response Enhancement...")
            stage_start = time.time()
            
            try:
                # Apply material-aware enhancements
                image = self.material_responder.enhance(
                    image,
                    surfaces=self.config.surface_types,
                    depth_map=depth_map
                )
                result.material_response_applied = True
                logger.info(f"  ✓ Material response applied")
                
            except Exception as e:
                logger.error(f"  ✗ Material response failed: {e}")
                result.material_response_applied = False
            
            stage_time = time.time() - stage_start
            self.stats['stage_times']['material_response'] = stage_time
        elif self.config.enable_material_response:
            logger.info("\nStage 4: Material response skipped (responder not available)")
        
        # Stage 5: Color Grading & LUTs
        if self.config.enable_color_grading:
            logger.info("\nStage 5: Professional Color Grading...")
            stage_start = time.time()
            
            try:
                # Apply LUT if available (Phase 3)
                if self.lut_processor:
                    logger.info(f"  Applying LUT: {self.config.lut_name} (strength: {self.config.lut_strength})")
                    image = self.lut_processor.apply(image, strength=self.config.lut_strength)
                    result.lut_applied = True
                    logger.info(f"  ✓ LUT applied")
                
                # Apply basic adjustments
                if self.config.saturation_boost != 1.0:
                    image = self._adjust_saturation(image, self.config.saturation_boost)
                    logger.info(f"  ✓ Saturation: {self.config.saturation_boost:.2f}x")
                
                if self.config.color_temperature_shift != 0.0:
                    image = self._adjust_temperature(image, self.config.color_temperature_shift)
                    logger.info(f"  ✓ Temperature shift: {self.config.color_temperature_shift:+.3f}")
                
                result.color_grading_applied = True
                logger.info(f"  ✓ Color grading complete")
            except Exception as e:
                logger.error(f"  ✗ Color grading failed: {e}")
                result.color_grading_applied = False
            
            stage_time = time.time() - stage_start
            self.stats['stage_times']['color_grading'] = stage_time
        
        # Stage 6: Final Export
        logger.info("\nStage 6: Exporting final image...")
        self._save_image(image, output_path, self.config.preserve_16bit)
        
        # Collect final metrics
        result.final_size = (image.shape[1], image.shape[0])
        result.bit_depth = 16 if self.config.preserve_16bit else 8
        result.file_size_mb = output_path.stat().st_size / (1024 * 1024)
        result.processing_time = time.time() - start_time
        
        if upscaling_metrics:
            result.color_deviation = upscaling_metrics.color_deviation
        
        # Update statistics
        self.stats['images_processed'] += 1
        self.stats['total_time'] += result.processing_time
        
        logger.info(f"\n✓ Complete in {result.processing_time:.2f}s")
        logger.info(f"  Output: {output_path}")
        
        return result
    
    def batch_process(
        self,
        input_paths: List[Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[Path, PipelineResult]:
        """
        Process multiple images with progress tracking.
        
        Args:
            input_paths: List of input image paths
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Dictionary of path -> PipelineResult
        """
        logger.info(f"\nBatch Processing {len(input_paths)} images")
        logger.info(f"Preset: {self.config.preset.value}")
        logger.info("=" * 60)
        
        results = {}
        start_time = time.time()
        
        for idx, input_path in enumerate(input_paths, 1):
            try:
                result = self.process_image(input_path)
                results[input_path] = result
                
                if progress_callback:
                    progress_callback(idx, len(input_paths), input_path.name)
                
                # Log progress
                avg_time = self.stats['total_time'] / self.stats['images_processed']
                remaining = len(input_paths) - idx
                eta = avg_time * remaining
                
                logger.info(f"\nProgress: {idx}/{len(input_paths)} ({idx/len(input_paths)*100:.1f}%)")
                logger.info(f"Avg time: {avg_time:.1f}s/image, ETA: {eta/60:.1f}min")
                
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                continue
        
        total_time = time.time() - start_time
        throughput = len(results) / (total_time / 3600)  # images/hour
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Batch Complete: {len(results)}/{len(input_paths)} images")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Throughput: {throughput:.1f} images/hour")
        
        # Generate batch report
        if self.config.generate_report:
            self._generate_batch_report(results)
        
        return results
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image preserving bit depth."""
        if TIFFFILE_AVAILABLE and path.suffix.lower() in ('.tif', '.tiff'):
            with TiffFile(path) as tif:
                image = tif.asarray()
        else:
            image = np.array(Image.open(path))
        
        # Normalize to float32 [0, 1]
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        return image
    
    def _save_image(self, image: np.ndarray, path: Path, preserve_16bit: bool):
        """Save image with bit depth preservation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if preserve_16bit and path.suffix.lower() in ('.tif', '.tiff'):
            if TIFFFILE_AVAILABLE:
                image_16bit = (image * 65535).clip(0, 65535).astype(np.uint16)
                tiff_write(path, image_16bit, photometric='rgb')
            else:
                logger.warning("tifffile not available. Saving as 8-bit PNG.")
                image_8bit = (image * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(image_8bit).save(path.with_suffix('.png'))
        else:
            if preserve_16bit:
                image_16bit = (image * 65535).clip(0, 65535).astype(np.uint16)
                Image.fromarray(image_16bit).save(path)
            else:
                image_8bit = (image * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(image_8bit).save(path)
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation."""
        # Convert to HSV
        from colorsys import rgb_to_hsv, hsv_to_rgb
        
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j]
                h, s, v = rgb_to_hsv(r, g, b)
                s = np.clip(s * factor, 0, 1)
                result[i, j] = hsv_to_rgb(h, s, v)
        
        return result
    
    def _adjust_temperature(self, image: np.ndarray, shift: float) -> np.ndarray:
        """Adjust color temperature (shift > 0 = warmer, < 0 = cooler)."""
        result = image.copy()
        
        if shift > 0:  # Warmer
            result[..., 0] = np.clip(result[..., 0] + shift, 0, 1)  # More red
            result[..., 2] = np.clip(result[..., 2] - shift * 0.5, 0, 1)  # Less blue
        else:  # Cooler
            result[..., 0] = np.clip(result[..., 0] + shift, 0, 1)  # Less red
            result[..., 2] = np.clip(result[..., 2] - shift * 0.5, 0, 1)  # More blue
        
        return result
    
    def _generate_batch_report(self, results: Dict[Path, PipelineResult]):
        """Generate comprehensive batch processing report."""
        report_path = self.config.output_dir / "batch_report.md"
        
        lines = [
            "# Batch Processing Report",
            f"Preset: {self.config.preset.value}",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total images: {len(results)}",
            f"- Total time: {self.stats['total_time']/60:.1f} minutes",
            f"- Throughput: {len(results)/(self.stats['total_time']/3600):.1f} images/hour",
            "",
            "## Stage Performance",
        ]
        
        for stage, stage_time in self.stats['stage_times'].items():
            lines.append(f"- {stage}: {stage_time:.2f}s average")
        
        lines.extend([
            "",
            "## Individual Results",
            ""
        ])
        
        for path, result in results.items():
            lines.extend([
                f"### {path.name}",
                f"- Output: {result.output_path.name}",
                f"- Time: {result.processing_time:.2f}s",
                f"- Size: {result.final_size[0]}x{result.final_size[1]}",
                f"- File: {result.file_size_mb:.1f} MB",
                ""
            ])
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Report saved: {report_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Unified Luxury Rendering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with preset
  python unified_luxury_pipeline.py input.tif --preset photo_realistic
  
  # Batch processing
  python unified_luxury_pipeline.py input_dir/ --batch --preset architectural
  
  # Custom configuration
  python unified_luxury_pipeline.py input.tif \\
      --upscale-model swinir_real_4x \\
      --enable-depth \\
      --material-response 0.8
        """
    )
    
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--preset", type=str, default="photo_realistic",
                        choices=[p.value for p in PipelinePreset],
                        help="Pipeline preset")
    
    # Stage toggles
    parser.add_argument("--no-upscaling", action="store_true", help="Disable upscaling")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth processing")
    parser.add_argument("--no-material", action="store_true", help="Disable material response")
    parser.add_argument("--no-color-grading", action="store_true", help="Disable color grading")
    
    # Upscaling options
    parser.add_argument("--upscale-model", type=str,
                        choices=[m.value for m in UpscalingModel],
                        help="Upscaling model")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size (0=auto)")
    
    # Quality options
    parser.add_argument("--no-16bit", action="store_true", help="Disable 16-bit output")
    parser.add_argument("--material-response", type=float, help="Material response strength (0-1)")
    parser.add_argument("--saturation", type=float, help="Saturation boost (0.5-2.0)")
    
    # Batch options
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = args.input
    output_dir = args.output_dir or Path(f"output_{args.preset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = UnifiedPipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        preset=PipelinePreset(args.preset),
        enable_upscaling=not args.no_upscaling,
        enable_depth_processing=not args.no_depth,
        enable_material_response=not args.no_material,
        enable_color_grading=not args.no_color_grading,
        preserve_16bit=not args.no_16bit,
        device=args.device
    )
    
    # Apply custom overrides
    if args.upscale_model:
        config.upscale_model = UpscalingModel(args.upscale_model)
    if args.tile_size:
        config.tile_size = args.tile_size
    if args.material_response is not None:
        config.material_strength = args.material_response
    if args.saturation is not None:
        config.saturation_boost = args.saturation
    
    # Initialize pipeline
    pipeline = UnifiedLuxuryPipeline(config)
    
    # Process
    if args.batch or input_path.is_dir():
        input_paths = []
        for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
            input_paths.extend(input_path.glob(f"*{ext}"))
        
        results = pipeline.batch_process(input_paths)
        
        print(f"\n{'=' * 60}")
        print(f"Processed {len(results)}/{len(input_paths)} images")
        print(f"Output: {output_dir}")
    else:
        result = pipeline.process_image(input_path)
        print(f"\n{result.summary()}")


if __name__ == "__main__":
    main()
