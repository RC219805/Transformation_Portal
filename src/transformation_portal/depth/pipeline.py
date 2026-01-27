"""
Architectural Depth Pipeline - Main Orchestration

Production-ready depth-aware image processing pipeline for architectural rendering.
Integrates Depth Anything V2 with multiple depth-guided enhancement modules.
"""

import logging
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

from .models import DepthAnythingV2Model, ModelBackend, ModelVariant
from .processors import (
    AtmosphericEffects,
    DepthAwareDenoise,
    DepthGuidedFilters,
    ZoneToneMapping,
)
from .utils import (
    DepthCache,
    depth_statistics,
    load_image,
    save_image,
    visualize_depth,
)
from .utils.depth_utils import smooth_depth

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
        proc_config = self.config.get('processing', {})
        processors: Dict[str, object] = {}

        # Depth-aware denoising
        if proc_config.get('depth_aware_denoise', {}).get('enabled', False):
            params = proc_config['depth_aware_denoise']
            processors['denoise'] = DepthAwareDenoise(
                sigma_spatial=params.get('sigma_spatial', 3.0),
                sigma_range=params.get('sigma_range', 0.1),
                edge_threshold=params.get('edge_threshold', 0.05),
                preserve_strength=params.get('preserve_strength', 0.8),
            )

        # Zone tone mapping
        if proc_config.get('zone_tone_mapping', {}).get('enabled', False):
            params = proc_config['zone_tone_mapping']
            processors['tone_mapping'] = ZoneToneMapping(
                num_zones=params.get('num_zones', 3),
                zone_params=params.get('zone_params'),
                transition_sigma=params.get('transition_sigma', 2.0),
                method=params.get('method', 'agx'),
            )

        # Atmospheric effects
        if proc_config.get('atmospheric_effects', {}).get('enabled', False):
            params = proc_config['atmospheric_effects']
            processors['atmospheric'] = AtmosphericEffects(
                haze_density=params.get('haze_density', 0.015),
                haze_color=tuple(params.get('haze_color', [0.7, 0.8, 0.9])),
                desaturation_strength=params.get('desaturation_strength', 0.3),
                depth_scale=params.get('depth_scale', 100.0),
                enable_color_shift=params.get('enable_color_shift', True),
            )

        # Depth-guided filters
        if proc_config.get('depth_guided_filters', {}).get('enabled', False):
            params = proc_config['depth_guided_filters']
            processors['filters'] = DepthGuidedFilters(
                clarity_strength=params.get('clarity_strength', 0.5),
                edge_preserve_threshold=params.get('edge_preserve_threshold', 0.05),
                scale_count=params.get('scale_count', 3),
                adaptive_to_depth=params.get('adaptive_to_depth', True),
            )

        return processors

    def _postprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Optional depth-map postprocessing applied after inference/cache.

        This is intentionally *opt-in* and config-driven, so existing configs remain unchanged.

        Config:
          processing:
            depth_postprocessing:
              enabled: true
              method: bilateral   # gaussian | bilateral | median
              sigma: 5.0
              edge_preserve: 0.1
              preserve_scale: true  # (recommended for bilateral; keeps original min/max)

        Notes:
        - Postprocessing is applied to the depth used downstream and saved to disk.
        - Cache semantics remain unchanged: the cache stores raw inference output.
        """
        proc_cfg = self.config.get('processing', {})
        if not isinstance(proc_cfg, dict):
            return depth

        cfg = proc_cfg.get('depth_postprocessing', {})
        if not isinstance(cfg, dict) or not cfg.get('enabled', False):
            return depth

        method = str(cfg.get('method', 'bilateral')).strip().lower()
        allowed = {'gaussian', 'bilateral', 'median'}
        if method not in allowed:
            logger.warning(
                "Depth postprocessing disabled: unknown method '%s' (expected: %s)",
                method,
                ', '.join(sorted(allowed)),
            )
            return depth

        # Normalize/guard inputs
        try:
            sigma = float(cfg.get('sigma', 5.0))
        except (TypeError, ValueError):
            logger.warning("Depth postprocessing disabled: invalid sigma value")
            return depth

        if sigma <= 0.0:
            return depth

        try:
            edge_preserve = float(cfg.get('edge_preserve', 0.1))
        except (TypeError, ValueError):
            edge_preserve = 0.1
        edge_preserve = max(edge_preserve, 0.0)

        # Parse preserve_scale with proper boolean handling (supports bool, str, int, float)
        default_preserve_scale = method == 'bilateral'
        raw_preserve = cfg.get('preserve_scale', default_preserve_scale)
        if isinstance(raw_preserve, bool):
            preserve_scale = raw_preserve
        elif isinstance(raw_preserve, str):
            normalized = raw_preserve.strip().lower()
            if normalized in {'true', '1', 'yes', 'y', 'on'}:
                preserve_scale = True
            elif normalized in {'false', '0', 'no', 'n', 'off'}:
                preserve_scale = False
            else:
                logger.warning(
                    "Depth postprocessing: invalid preserve_scale value '%s', using default %s",
                    raw_preserve,
                    default_preserve_scale,
                )
                preserve_scale = default_preserve_scale
        elif isinstance(raw_preserve, Integral):
            if raw_preserve in (0, 1):
                preserve_scale = bool(raw_preserve)
            else:
                logger.warning(
                    "Depth postprocessing: unexpected preserve_scale integer '%s', using default %s",
                    raw_preserve,
                    default_preserve_scale,
                )
                preserve_scale = default_preserve_scale
        elif isinstance(raw_preserve, Real):
            if raw_preserve in (0.0, 1.0):
                preserve_scale = bool(int(raw_preserve))
            else:
                logger.warning(
                    "Depth postprocessing: unexpected preserve_scale float '%s', using default %s",
                    raw_preserve,
                    default_preserve_scale,
                )
                preserve_scale = default_preserve_scale
        else:
            preserve_scale = default_preserve_scale

        # Nothing to do if depth isn't a usable ndarray
        if not isinstance(depth, np.ndarray):
            return depth
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = np.squeeze(depth, axis=-1)

        # Expected exceptions only; unexpected bugs should surface.
        try:
            if preserve_scale and method == 'bilateral':
                d_min = float(np.nanmin(depth))
                d_max = float(np.nanmax(depth))
                if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
                    return depth

                smoothed = smooth_depth(
                    depth,
                    method=method,
                    sigma=sigma,
                    edge_preserve=edge_preserve,
                )

                # depth_utils.smooth_depth('bilateral') is expected to return normalized [0,1]
                # when using the OpenCV bilateral path. However, the fallback implementation
                # (e.g., gaussian_filter) can operate on the original depth scale. To avoid
                # rescaling twice in the fallback case, only apply the [0,1] -> [d_min,d_max]
                # mapping if the smoothed output actually looks normalized.
                smoothed = smoothed.astype(np.float32, copy=False)
                if smoothed.size == 0:
                    return depth

                s_min = float(np.nanmin(smoothed))
                s_max = float(np.nanmax(smoothed))

                # Treat as normalized if it lies within [0, 1] up to a small numerical epsilon.
                eps = 1e-3
                if (
                    np.isfinite(s_min)
                    and np.isfinite(s_max)
                    and s_min >= -eps
                    and s_max <= 1.0 + eps
                ):
                    return smoothed * (d_max - d_min) + d_min

                # Fallback path: assume smooth_depth returned data in the original scale.
                # In this case, preserve_scale means we should not reapply the range scaling.
                return smoothed

            return smooth_depth(
                depth,
                method=method,
                sigma=sigma,
                edge_preserve=edge_preserve,
            )

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning(
                "Depth postprocessing failed (%s): %s; using raw depth",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return depth
        except Exception as e:
            # Only swallow OpenCV-specific runtime errors if present; otherwise re-raise.
            try:
                import cv2  # type: ignore

                if isinstance(e, cv2.error):  # type: ignore[attr-defined]
                    logger.warning(
                        "Depth postprocessing failed (cv2.error): %s; using raw depth",
                        e,
                        exc_info=True,
                    )
                    return depth
            except ImportError:
                # OpenCV is optional; if unavailable, just re-raise the original exception.
                logger.debug(
                    "OpenCV not available while handling depth postprocessing error; re-raising.",
                    exc_info=True,
                )
                pass
            raise

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
        depth = self._postprocess_depth(depth_result['depth'])

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

    def _async_load_images(
        self,
        image_paths: List[Union[str, Path]],
        max_workers: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load images asynchronously using thread pool.

        Args:
            image_paths: List of image paths to load
            max_workers: Maximum number of threads (default: None = auto)

        Returns:
            Dictionary mapping path to loaded image array
        """
        loaded_images: Dict[str, np.ndarray] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_image, path, normalize=True): str(path)
                for path in image_paths
            }

            for future in tqdm(
                as_completed(future_to_path),
                total=len(image_paths),
                desc="Loading images",
            ):
                path = future_to_path[future]
                try:
                    loaded_images[path] = future.result()
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        return loaded_images

    def _process_single_image(
        self,
        image_path: Union[str, Path],
        preloaded_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Process single image (helper for parallel processing).

        Args:
            image_path: Path to input image
            preloaded_image: Optional preloaded image array

        Returns:
            Result dictionary
        """
        start_time = time.time()

        # Use preloaded image or load now
        if preloaded_image is not None:
            image = preloaded_image
        else:
            image = load_image(image_path, normalize=True)

        # Estimate depth (with caching)
        depth_result = self.cache.get_or_compute(
            image,
            lambda: self.depth_model.estimate_depth(image)
        )
        depth = self._postprocess_depth(depth_result['depth'])

        # Apply processing pipeline
        result_image = image.copy()

        if 'denoise' in self.processors:
            result_image = self.processors['denoise'](result_image, depth)

        if 'tone_mapping' in self.processors:
            result_image = self.processors['tone_mapping'](result_image, depth)

        if 'atmospheric' in self.processors:
            result_image = self.processors['atmospheric'](result_image, depth)

        if 'filters' in self.processors:
            result_image = self.processors['filters'](result_image, depth)

        processing_time = time.time() - start_time

        metadata = {
            'input_path': str(image_path),
            'input_shape': image.shape,
            'processing_time_sec': processing_time,
            'depth_inference_time_ms': depth_result['metadata']['inference_time_ms'],
            'processors_applied': list(self.processors.keys()),
            'depth_stats': depth_statistics(depth),
        }

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
        parallel: bool = True,
        max_workers: Optional[int] = None,
        preload_images: bool = True,
    ) -> List[Dict]:
        """
        Process multiple renders in batch with parallel processing.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            save_depth: Save depth maps as numpy arrays
            save_visualization: Save depth visualizations
            parallel: Enable parallel processing (default: True)
            max_workers: Max parallel workers (default: None = CPU count)
            preload_images: Preload images async for faster access (default: True)

        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch processing {len(image_paths)} images")
        logger.info(f"Parallel processing: {parallel}, Preload: {preload_images}")

        # Async image loading (Phase 1 optimization)
        preloaded_images: Dict[str, np.ndarray] = {}
        if preload_images:
            preloaded_images = self._async_load_images(image_paths, max_workers)

        results: List[Dict] = []

        if parallel:
            # Parallel processing (Phase 1 optimization)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {}
                for image_path in image_paths:
                    path_str = str(image_path)
                    preloaded = preloaded_images.get(path_str)
                    future = executor.submit(
                        self._process_single_image,
                        image_path,
                        preloaded,
                    )
                    future_to_path[future] = image_path

                for future in tqdm(
                    as_completed(future_to_path),
                    total=len(image_paths),
                    desc="Processing renders",
                ):
                    image_path = future_to_path[future]
                    try:
                        result = future.result()

                        # Save results
                        self.save_result(
                            result,
                            output_dir,
                            save_depth=save_depth,
                            save_visualization=save_visualization,
                        )

                        results.append(result)

                        # Update stats
                        self.stats['images_processed'] += 1
                        self.stats['total_time'] += result['metadata']['processing_time_sec']

                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                        continue
        else:
            # Sequential processing (fallback)
            for image_path in tqdm(image_paths, desc="Processing renders"):
                try:
                    path_str = str(image_path)
                    preloaded = preloaded_images.get(path_str)
                    result = self._process_single_image(image_path, preloaded)

                    # Save results
                    self.save_result(
                        result,
                        output_dir,
                        save_depth=save_depth,
                        save_visualization=save_visualization,
                    )

                    results.append(result)

                    # Update stats
                    self.stats['images_processed'] += 1
                    self.stats['total_time'] += result['metadata']['processing_time_sec']

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

    def batch_process_streaming(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_depth: bool = True,
        save_visualization: bool = True,
    ) -> Iterator[Dict]:
        """
        Process images with streaming results (Phase 3 optimization).

        This variant yields results one at a time instead of accumulating them
        in memory, so memory usage remains essentially constant regardless of
        batch size. Results are also written to disk as they are produced.

        Args:
            image_paths: List of image paths to process. Each path can be a string
                or ``pathlib.Path`` and is passed directly to :meth:`process_render`.
            output_dir: Directory where enhanced images, depth maps, and depth
                visualizations will be saved. Created if it does not already exist.
            save_depth: If True, save the raw depth map as a ``.npy`` file for each
                input image.
            save_visualization: If True, save a depth visualization PNG for each
                input image using the configured colormap.

        Yields:
            Result dictionary for each processed image in the same format
            as returned by :meth:`process_render`. Each result is yielded after it
            has been successfully written to disk via :meth:`save_result`.

        Example:
            >>> from pathlib import Path
            >>> pipeline = ArchitecturalDepthPipeline.from_config("config/interior_preset.yaml")
            >>> image_paths = list(Path("input/").glob("*.jpg"))
            >>> for result in pipeline.batch_process_streaming(
            ...     image_paths,
            ...     output_dir="output/",
            ...     save_depth=True,
            ...     save_visualization=True,
            ... ):
            ...     depth_stats = result["metadata"].get("depth_stats")
            ...     print("Processed:", result["metadata"]["input_path"], depth_stats)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Streaming batch processing {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc="Processing renders (streaming)"):
            try:
                result = self.process_render(image_path)

                self.save_result(
                    result,
                    output_dir,
                    save_depth=save_depth,
                    save_visualization=save_visualization,
                )

                yield result

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

    def batch_process_pipelined(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_depth: bool = True,
        save_visualization: bool = True,
        pipeline_workers: int = 3,
    ) -> Iterator[Dict]:
        """
        Process images using a multi-stage, pipelined batch executor (Phase 3 optimization).

        This is the highest-throughput batch API for the depth pipeline. Work is decomposed
        into several stages connected by bounded in-memory queues so that disk I/O,
        depth estimation, image processing, and saving can overlap:

        1. **Load stage (I/O bound)** – ``loader_worker`` reads images from ``image_paths``,
           normalizes them, and enqueues ``(path, image)`` into ``load_queue``.
        2. **Depth stage (ML bound)** – ``depth_worker`` consumes items from ``load_queue``,
           computes depth using :class:`DepthCache` and :attr:`depth_model`, and enqueues
           ``(path, image, depth_result)`` into ``depth_queue``.
        3. **Processing stage (CPU/GPU bound)** – ``process_worker`` consumes items from
           ``depth_queue``, applies the depth-aware enhancement stack (tone mapping,
           denoising, atmospheric effects, filters) and enqueues the processed result
           into ``process_queue``.
        4. **Save / emit stage (I/O bound)** – a saver worker consumes items from
           ``process_queue``, writes outputs into ``output_dir`` (optionally including
           depth maps and depth visualizations), and streams result dictionaries back
           to the caller via this generator.

        The queues are bounded so that memory usage remains approximately constant
        regardless of the total batch size. Stages run in separate threads, enabling
        overlap between disk I/O and depth / enhancement computation. This is especially
        beneficial when processing large batches or when depth estimation is expensive.

        Args:
            image_paths: Iterable of image paths to process. Each entry can be a :class:`str` or
                :class:`pathlib.Path`. The order of paths defines the logical batch.
            output_dir: Directory where processed renders (and optional depth artifacts) are written.
                The directory is created if it does not already exist.
            save_depth: If ``True``, write the raw depth map for each image alongside the processed
                render. If ``False``, depth is kept in memory only for processing.
            save_visualization: If ``True``, write a depth visualization (e.g., colored or normalized depth)
                for each image. Ignored if depth estimation fails for a given image.
            pipeline_workers: Controls the level of concurrency within the pipeline. Depending on the
                implementation of the worker threads, this may scale the number of
                depth / processing workers used for the internal stages. The default
                value is tuned for typical workstation workloads.

        Yields:
            A streaming iterator of per-image result dictionaries, one for each
            successfully processed input. Results are yielded as soon as they are
            saved, allowing the caller to consume outputs while the remaining
            images are still being processed.

        Notes:
            - Individual image failures are logged and skipped; the pipeline continues
              processing subsequent images.
            - Memory consumption is bounded by the size of the internal queues rather
              than the total number of images in the batch.
            - This method is suitable for very large batches where ``batch_process`` or
              ``batch_process_streaming`` might be limited by I/O or single-stage
              execution.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline processing {len(image_paths)} images")

        load_queue: queue.Queue[Optional[Tuple[Any, Any]]] = queue.Queue(maxsize=10)
        depth_queue: queue.Queue[Optional[Tuple[Any, Any, Any]]] = queue.Queue(maxsize=10)
        process_queue: queue.Queue[Optional[Tuple[Any, Any]]] = queue.Queue(maxsize=10)
        save_queue: queue.Queue[Optional[Tuple[Any, Any]]] = queue.Queue(maxsize=10)

        def loader_worker():
            for path in image_paths:
                try:
                    image = load_image(path, normalize=True)
                    load_queue.put((path, image))
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
            load_queue.put(None)

        def depth_worker():
            while True:
                item = load_queue.get()
                if item is None:
                    depth_queue.put(None)
                    break

                path, image = item
                try:
                    depth_result = self.cache.get_or_compute(
                        image,
                        lambda: self.depth_model.estimate_depth(image)
                    )
                    depth_queue.put((path, image, depth_result))
                except Exception as e:
                    logger.error(f"Failed depth estimation for {path}: {e}")

        def process_worker():
            while True:
                item = depth_queue.get()
                if item is None:
                    process_queue.put(None)
                    break

                path, image, depth_result = item
                try:
                    depth = self._postprocess_depth(depth_result['depth'])
                    result_image = image.copy()

                    if 'denoise' in self.processors:
                        result_image = self.processors['denoise'](result_image, depth)
                    if 'tone_mapping' in self.processors:
                        result_image = self.processors['tone_mapping'](result_image, depth)
                    if 'atmospheric' in self.processors:
                        result_image = self.processors['atmospheric'](result_image, depth)
                    if 'filters' in self.processors:
                        result_image = self.processors['filters'](result_image, depth)

                    result = {
                        'image': result_image,
                        'depth': depth,
                        'metadata': {
                            'input_path': str(path),
                            'input_shape': image.shape,
                            'depth_inference_time_ms': depth_result['metadata']['inference_time_ms'],
                            'processors_applied': list(self.processors.keys()),
                            'depth_stats': depth_statistics(depth),
                        }
                    }

                    process_queue.put((path, result))
                except Exception as e:
                    logger.error(f"Failed processing {path}: {e}")

        def save_worker():
            while True:
                item = process_queue.get()
                if item is None:
                    save_queue.put(None)
                    break

                path, result = item
                try:
                    self.save_result(
                        result,
                        output_dir,
                        save_depth=save_depth,
                        save_visualization=save_visualization,
                    )
                    save_queue.put((path, result))
                except Exception as e:
                    logger.error(f"Failed to save {path}: {e}")

        threads = [
            threading.Thread(target=loader_worker, name="Loader"),
            threading.Thread(target=depth_worker, name="DepthEstimator"),
            threading.Thread(target=process_worker, name="Processor"),
            threading.Thread(target=save_worker, name="Saver"),
        ]

        for t in threads:
            t.daemon = True
            t.start()

        processed_count = 0
        with tqdm(total=len(image_paths), desc="Pipeline processing") as pbar:
            while True:
                item = save_queue.get()
                if item is None:
                    break

                _, result = item
                processed_count += 1
                pbar.update(1)

                self.stats['images_processed'] += 1
                yield result

        for t in threads:
            t.join()

        logger.info(f"Pipeline processing complete: {processed_count}/{len(image_paths)} images")

    def process_render_progressive(
        self,
        image_path: Union[str, Path],
        quality_levels: List[float] = [0.25, 0.5, 1.0],
        return_all_levels: bool = False,
    ) -> Union[Dict, List[Dict]]:
        """
        Process image progressively at multiple quality levels (Phase 3 optimization).

        Note:
        - Depth postprocessing is applied once per level (after inference) to avoid double-smoothing.
        """
        from .utils.image_utils import resize_image

        logger.info(f"Progressive processing: {image_path} at levels {quality_levels}")

        image_full = load_image(image_path, normalize=True)
        h_full, w_full = image_full.shape[:2]

        results: List[Dict] = []

        for scale in quality_levels:
            start_time = time.time()

            if scale < 1.0:
                h_scaled = int(h_full * scale)
                w_scaled = int(w_full * scale)
                image_scaled = resize_image(
                    image_full,
                    size=(h_scaled, w_scaled),
                    interpolation='bilinear'
                )
                logger.info(f"Processing at {scale:.0%} resolution: {h_scaled}x{w_scaled}")
            else:
                image_scaled = image_full
                logger.info(f"Processing at full resolution: {h_full}x{w_full}")

            depth_result = self.cache.get_or_compute(
                image_scaled,
                lambda: self.depth_model.estimate_depth(image_scaled)
            )
            depth = self._postprocess_depth(depth_result['depth'])

            result_image = image_scaled.copy()

            if 'denoise' in self.processors:
                result_image = self.processors['denoise'](result_image, depth)
            if 'tone_mapping' in self.processors:
                result_image = self.processors['tone_mapping'](result_image, depth)
            if 'atmospheric' in self.processors:
                result_image = self.processors['atmospheric'](result_image, depth)
            if 'filters' in self.processors:
                result_image = self.processors['filters'](result_image, depth)

            if scale < 1.0:
                result_image = resize_image(
                    result_image,
                    size=(h_full, w_full),
                    interpolation='bicubic'
                )
                depth = resize_image(
                    depth,
                    size=(h_full, w_full),
                    interpolation='bilinear'
                )

            processing_time = time.time() - start_time

            result = {
                'image': result_image,
                'depth': depth,
                'metadata': {
                    'input_path': str(image_path),
                    'input_shape': image_full.shape,
                    'processing_scale': scale,
                    'processing_time_sec': processing_time,
                    'depth_inference_time_ms': depth_result['metadata']['inference_time_ms'],
                    'processors_applied': list(self.processors.keys()),
                    'depth_stats': depth_statistics(depth),
                }
            }

            results.append(result)
            logger.info(f"Level {scale:.0%} complete in {processing_time:.2f}s")

        return results if return_all_levels else results[-1]

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

        cache_stats = self.cache.get_stats()
        logger.info("\nCache statistics:")
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
            "ArchitecturalDepthPipeline("
            f"model={self.depth_model.variant.name}, "
            f"processors={list(self.processors.keys())}, "
            f"images_processed={self.stats['images_processed']})"
        )
