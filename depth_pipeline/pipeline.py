"""
Architectural Depth Pipeline - Main Orchestration

Production-ready depth-aware image processing pipeline for architectural rendering.
Integrates Depth Anything V2 with multiple depth-guided enhancement modules.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from tqdm import tqdm

from .models import DepthAnythingV2Model, ModelBackend, ModelVariant
from .processors import (
    DepthAwareDenoise,
    ZoneToneMapping,
    AtmosphericEffects,
    DepthGuidedFilters,
)
from .utils import (
    DepthCache,
    load_image,
    save_image,
    compute_image_hash,
    visualize_depth,
    depth_statistics,
)

logger = logging.getLogger(__name__)


class ArchitecturalDepthPipeline:
    """
    Production depth-aware enhancement pipeline for architectural rendering.

    Features:
    - Monocular depth estimation (Depth Anything V2)
    - Depth-aware denoising
    - Zone-based tone mapping
    - Atmospheric effects
    - Depth-guided clarity enhancement
    - LRU caching for iterative workflows
    - Batch processing support

    Example:
        >>> pipeline = ArchitecturalDepthPipeline.from_config('config/default_config.yaml')
        >>> result = pipeline.process_render('render.jpg')
        >>> pipeline.save_result(result, 'output/')
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline from configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize depth model
        self.depth_model = self._init_depth_model()

        # Initialize cache
        self.cache = self._init_cache()

        # Initialize processors
        self.processors = self._init_processors()

        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        logger.info("Initialized ArchitecturalDepthPipeline")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'ArchitecturalDepthPipeline':
        """
        Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized pipeline
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        return cls(config)

    def _init_depth_model(self) -> DepthAnythingV2Model:
        """Initialize depth estimation model."""
        model_config = self.config['depth_model']

        # Map variant string to enum
        variant_map = {
            'small': ModelVariant.SMALL,
            'base': ModelVariant.BASE,
            'large': ModelVariant.LARGE,
        }
        variant = variant_map.get(model_config['variant'], ModelVariant.SMALL)

        # Map backend string to enum
        backend_map = {
            'pytorch_cpu': ModelBackend.PYTORCH_CPU,
            'pytorch_mps': ModelBackend.PYTORCH_MPS,
            'coreml': ModelBackend.COREML,
        }
        backend = backend_map.get(model_config.get('backend'), None)

        model = DepthAnythingV2Model(
            variant=variant,
            backend=backend,
            precision=model_config.get('precision', 'fp16'),
        )

        return model

    def _init_cache(self) -> DepthCache:
        """Initialize depth cache."""
        model_config = self.config['depth_model']

        cache = DepthCache(
            max_size=model_config.get('cache_size', 100),
            enable_disk_cache=model_config.get('enable_disk_cache', False),
        )

        return cache

    def _init_processors(self) -> Dict:
        """Initialize all processing modules."""
        proc_config = self.config['processing']
        processors = {}

        # Depth-aware denoising
        if proc_config['depth_aware_denoise']['enabled']:
            params = proc_config['depth_aware_denoise']
            processors['denoise'] = DepthAwareDenoise(
                sigma_spatial=params.get('sigma_spatial', 3.0),
                sigma_range=params.get('sigma_range', 0.1),
                edge_threshold=params.get('edge_threshold', 0.05),
                preserve_strength=params.get('preserve_strength', 0.8),
            )

        # Zone tone mapping
        if proc_config['zone_tone_mapping']['enabled']:
            params = proc_config['zone_tone_mapping']
            processors['tone_mapping'] = ZoneToneMapping(
                num_zones=params.get('num_zones', 3),
                zone_params=params.get('zone_params'),
                transition_sigma=params.get('transition_sigma', 2.0),
                method=params.get('method', 'agx'),
            )

        # Atmospheric effects
        if proc_config['atmospheric_effects']['enabled']:
            params = proc_config['atmospheric_effects']
            processors['atmospheric'] = AtmosphericEffects(
                haze_density=params.get('haze_density', 0.015),
                haze_color=tuple(params.get('haze_color', [0.7, 0.8, 0.9])),
                desaturation_strength=params.get('desaturation_strength', 0.3),
                depth_scale=params.get('depth_scale', 100.0),
                enable_color_shift=params.get('enable_color_shift', True),
            )

        # Depth-guided filters
        if proc_config['depth_guided_filters']['enabled']:
            params = proc_config['depth_guided_filters']
            processors['filters'] = DepthGuidedFilters(
                clarity_strength=params.get('clarity_strength', 0.5),
                edge_preserve_threshold=params.get('edge_preserve_threshold', 0.05),
                scale_count=params.get('scale_count', 3),
                adaptive_to_depth=params.get('adaptive_to_depth', True),
            )

        return processors

    def process_render(
        self,
        image_path: Union[str, Path],
        override_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Process single architectural render.

        Args:
            image_path: Path to input render
            override_config: Optional config overrides

        Returns:
            Result dictionary with:
                - 'image': Enhanced image
                - 'depth': Depth map
                - 'metadata': Processing metadata
        """
        start_time = time.time()

        # Load image
        logger.info(f"Processing: {image_path}")
        image = load_image(image_path, normalize=True)

        # Estimate depth (with caching)
        depth_result = self.cache.get_or_compute(
            image,
            lambda: self.depth_model.estimate_depth(image)
        )
        depth = depth_result['depth']

        # Apply processing pipeline
        result_image = image.copy()

        # 1. Depth-aware denoising
        if 'denoise' in self.processors:
            logger.debug("Applying depth-aware denoising")
            result_image = self.processors['denoise'](result_image, depth)

        # 2. Zone-based tone mapping
        if 'tone_mapping' in self.processors:
            logger.debug("Applying zone tone mapping")
            result_image = self.processors['tone_mapping'](result_image, depth)

        # 3. Atmospheric effects
        if 'atmospheric' in self.processors:
            logger.debug("Applying atmospheric effects")
            result_image = self.processors['atmospheric'](result_image, depth)

        # 4. Depth-guided filters
        if 'filters' in self.processors:
            logger.debug("Applying depth-guided filters")
            result_image = self.processors['filters'](result_image, depth)

        # Compute processing time
        processing_time = time.time() - start_time

        # Collect metadata
        metadata = {
            'input_path': str(image_path),
            'input_shape': image.shape,
            'processing_time_sec': processing_time,
            'depth_inference_time_ms': depth_result['metadata']['inference_time_ms'],
            'processors_applied': list(self.processors.keys()),
            'depth_stats': depth_statistics(depth),
        }

        # Update global stats
        self.stats['images_processed'] += 1
        self.stats['total_time'] += processing_time

        logger.info(f"Processed in {processing_time:.2f}s")

        return {
            'image': result_image,
            'depth': depth,
            'metadata': metadata,
        }

    def batch_process(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_depth: bool = True,
        save_visualization: bool = True,
    ) -> List[Dict]:
        """
        Process multiple renders in batch.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            save_depth: Save depth maps as numpy arrays
            save_visualization: Save depth visualizations

        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        logger.info(f"Batch processing {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc="Processing renders"):
            try:
                # Process image
                result = self.process_render(image_path)

                # Save results
                self.save_result(
                    result,
                    output_dir,
                    save_depth=save_depth,
                    save_visualization=save_visualization,
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        # Print summary
        self._print_batch_summary(results)

        return results

    def save_result(
        self,
        result: Dict,
        output_dir: Union[str, Path],
        save_depth: bool = True,
        save_visualization: bool = True,
    ):
        """
        Save processing results.

        Args:
            result: Result dictionary from process_render
            output_dir: Output directory
            save_depth: Save depth map (.npy)
            save_visualization: Save depth visualization
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get input filename
        input_path = Path(result['metadata']['input_path'])
        stem = input_path.stem

        # Save enhanced image
        output_config = self.config.get('output', {})
        output_format = output_config.get('output_format', 'png')
        quality = output_config.get('jpeg_quality', 95)

        output_image_path = output_dir / f"{stem}_enhanced.{output_format}"
        save_image(result['image'], output_image_path, quality=quality)
        logger.info(f"Saved enhanced image: {output_image_path}")

        # Save depth map
        if save_depth:
            depth_path = output_dir / f"{stem}_depth.npy"
            np.save(depth_path, result['depth'])
            logger.debug(f"Saved depth map: {depth_path}")

        # Save depth visualization
        if save_visualization:
            colormap = output_config.get('depth_colormap', 'turbo')
            viz_path = output_dir / f"{stem}_depth_viz.png"
            visualize_depth(result['depth'], colormap=colormap, save_path=str(viz_path))

    def _print_batch_summary(self, results: List[Dict]):
        """Print batch processing summary."""
        if not results:
            logger.warning("No images processed successfully")
            return

        total_time = sum(r['metadata']['processing_time_sec'] for r in results)
        avg_time = total_time / len(results)
        avg_depth_time = np.mean([
            r['metadata']['depth_inference_time_ms'] for r in results
        ])

        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Images processed: {len(results)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per image: {avg_time:.2f}s")
        logger.info(f"Average depth inference: {avg_depth_time:.1f}ms")
        logger.info(f"Throughput: {len(results) / (total_time / 3600):.1f} images/hour")

        # Cache stats
        cache_stats = self.cache.get_stats()
        logger.info(f"\nCache statistics:")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        logger.info(f"  Entries: {cache_stats['size']}/{cache_stats['max_size']}")

        logger.info("=" * 60 + "\n")

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        stats['cache_stats'] = self.cache.get_stats()

        if stats['images_processed'] > 0:
            stats['avg_time_per_image'] = stats['total_time'] / stats['images_processed']

        return stats

    def clear_cache(self, clear_disk: bool = False):
        """Clear depth cache."""
        self.cache.clear(clear_disk=clear_disk)
        logger.info("Cache cleared")

    def __repr__(self) -> str:
        return (
            f"ArchitecturalDepthPipeline("
            f"model={self.depth_model.variant.name}, "
            f"processors={list(self.processors.keys())}, "
            f"images_processed={self.stats['images_processed']})"
        )
