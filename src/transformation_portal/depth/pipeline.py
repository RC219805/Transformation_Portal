"""
Architectural Depth Pipeline - Main Orchestration

Production-ready depth-aware image processing pipeline for architectural rendering.
Integrates Depth Anything V2 with multiple depth-guided enhancement modules.
"""

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

from .models import DepthAnythingV2Model, ModelBackend, ModelVariant
from .processors import AtmosphericEffects, DepthAwareDenoise, DepthGuidedFilters, ZoneToneMapping
from .utils import DepthCache, depth_statistics, load_image, save_image, smooth_depth, visualize_depth

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
            "images_processed": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("Initialized ArchitecturalDepthPipeline")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "ArchitecturalDepthPipeline":
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

        with open(config_path, "r", encoding="utf-8") as f:
            # YAML_GOVERNANCE_EXEMPT: internal depth pipeline config, not config/presets/**.
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        return cls(config)

    def _init_depth_model(self) -> DepthAnythingV2Model:
        """Initialize depth estimation model."""
        model_config = self.config["depth_model"]

        # Map variant string to enum
        variant_map = {
            "small": ModelVariant.SMALL,
            "base": ModelVariant.BASE,
            "large": ModelVariant.LARGE,
        }
        variant_name = model_config["variant"]
        if variant_name not in variant_map:
            # Fail closed: an unknown variant used to load the SMALL model silently.
            raise ValueError(f"Unknown depth_model.variant {variant_name!r}; expected one of {sorted(variant_map)}")
        variant = variant_map[variant_name]

        # Map backend string to enum (absent: auto-detect)
        backend_map = {
            "pytorch_cpu": ModelBackend.PYTORCH_CPU,
            "pytorch_mps": ModelBackend.PYTORCH_MPS,
            "coreml": ModelBackend.COREML,
        }
        backend_name = model_config.get("backend")
        if backend_name is not None and backend_name not in backend_map:
            # Fail closed: an unknown backend used to fall back to auto-detection silently.
            raise ValueError(
                f"Unknown depth_model.backend {backend_name!r}; expected one of {sorted(backend_map)} or no value (auto)"
            )
        backend = backend_map.get(backend_name)

        model = DepthAnythingV2Model(
            variant=variant,
            backend=backend,
            device=model_config.get("device"),
            precision=model_config.get("precision", "fp16"),
            model_revision=model_config.get("model_revision"),
            onnx_revision=model_config.get("onnx_revision"),
            coreml_revision=model_config.get("coreml_revision"),
            strict_model_lock=model_config.get("strict_model_lock"),
        )

        return model

    def _init_cache(self) -> DepthCache:
        """Initialize depth cache."""
        model_config = self.config["depth_model"]

        cache = DepthCache(
            max_size=model_config.get("cache_size", 100),
            enable_disk_cache=model_config.get("enable_disk_cache", False),
            namespace="da2-float-prediction-v2",
        )

        return cache

    def _init_processors(self) -> Dict:
        """Initialize all processing modules."""
        proc_config = self.config.get("processing", {})
        processors = {}

        # Depth-aware denoising
        if proc_config.get("depth_aware_denoise", {}).get("enabled", False):
            params = proc_config["depth_aware_denoise"]
            processors["denoise"] = DepthAwareDenoise(
                sigma_spatial=params.get("sigma_spatial", 3.0),
                sigma_range=params.get("sigma_range", 0.1),
                edge_threshold=params.get("edge_threshold", 0.05),
                preserve_strength=params.get("preserve_strength", 0.8),
            )

        # Zone tone mapping
        if proc_config.get("zone_tone_mapping", {}).get("enabled", False):
            params = proc_config["zone_tone_mapping"]
            processors["tone_mapping"] = ZoneToneMapping(
                num_zones=params.get("num_zones", 3),
                zone_params=params.get("zone_params"),
                transition_sigma=params.get("transition_sigma", 2.0),
                method=params.get("method", "agx"),
            )

        # Atmospheric effects
        if proc_config.get("atmospheric_effects", {}).get("enabled", False):
            params = proc_config["atmospheric_effects"]
            processors["atmospheric"] = AtmosphericEffects(
                haze_density=params.get("haze_density", 0.015),
                haze_color=tuple(params.get("haze_color", [0.7, 0.8, 0.9])),
                desaturation_strength=params.get("desaturation_strength", 0.3),
                depth_scale=params.get("depth_scale", 100.0),
                enable_color_shift=params.get("enable_color_shift", True),
            )

        # Depth-guided filters
        if proc_config.get("depth_guided_filters", {}).get("enabled", False):
            params = proc_config["depth_guided_filters"]
            processors["filters"] = DepthGuidedFilters(
                clarity_strength=params.get("clarity_strength", 0.5),
                edge_preserve_threshold=params.get("edge_preserve_threshold", 0.05),
                scale_count=params.get("scale_count", 3),
                adaptive_to_depth=params.get("adaptive_to_depth", True),
            )

        return processors

    def _apply_depth_postprocessing(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing to depth map (smoothing with optional scale preservation).

        Args:
            depth: Raw depth map

        Returns:
            Postprocessed depth map
        """
        postproc_config = self.config.get("processing", {}).get("depth_postprocessing", {})

        if not postproc_config.get("enabled", False):
            return depth

        # Check if method is valid
        method = postproc_config.get("method", "bilateral")
        valid_methods = ["gaussian", "bilateral", "median"]
        if method not in valid_methods:
            logger.warning(f"Unknown smoothing method '{method}', skipping postprocessing")
            return depth

        # Store original scale if preserve_scale is enabled
        preserve_scale_raw = postproc_config.get("preserve_scale", False)
        # Parse boolean from various formats (string, int, bool)
        if isinstance(preserve_scale_raw, str):
            preserve_scale = preserve_scale_raw.lower() not in ("false", "0", "no", "off", "")
        else:
            preserve_scale = bool(preserve_scale_raw)

        if preserve_scale:
            original_min = float(depth.min())
            original_max = float(depth.max())

        # Apply smoothing
        sigma = postproc_config.get("sigma", 5.0)
        edge_preserve = postproc_config.get("edge_preserve", 0.1)

        try:
            smoothed = smooth_depth(depth, method=method, sigma=sigma, edge_preserve=edge_preserve)
        except Exception as e:
            logger.error(f"Error during depth smoothing: {e}. Returning original depth.")
            return depth

        # Restore original scale if requested
        if preserve_scale and (original_max - original_min) > 1e-8:
            smoothed_min = float(smoothed.min())
            smoothed_max = float(smoothed.max())

            # Check if smoothed output is already in original scale
            # (e.g., gaussian filter preserves scale)
            scale_ratio = (smoothed_max - smoothed_min) / (original_max - original_min)
            is_already_scaled = abs(scale_ratio - 1.0) < 0.5  # Within 50% of original range

            if is_already_scaled:
                # Output is already in original scale, don't rescale
                pass
            elif smoothed_max - smoothed_min > 1e-8:
                # Rescale smoothed depth to original range
                smoothed = (smoothed - smoothed_min) / (smoothed_max - smoothed_min)
                smoothed = smoothed * (original_max - original_min) + original_min
            else:
                # Smoothed is uniform - map to midpoint of original range
                midpoint = (original_min + original_max) / 2.0
                smoothed = np.full_like(smoothed, midpoint)

        return smoothed

    def _postprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Alias for _apply_depth_postprocessing for backward compatibility.

        Args:
            depth: Raw depth map

        Returns:
            Postprocessed depth map
        """
        return self._apply_depth_postprocessing(depth)

    def _estimate_depth(self, image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Estimate depth for one image (with caching) and apply the configured depth postprocessing.

        Every processing path uses this helper, so a configured ``depth_postprocessing`` step
        applies to batch, pipelined and progressive runs exactly as it does to ``process_render``.

        Args:
            image: Normalised input image

        Returns:
            Tuple of (raw depth result from the model or cache, postprocessed depth map)
        """
        depth_result = self.cache.get_or_compute(image, lambda: self.depth_model.estimate_depth(image))
        return depth_result, self._apply_depth_postprocessing(depth_result["depth"])

    def _apply_processors(self, image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Run the enabled depth-aware processors, in pipeline order, on a copy of ``image``."""
        result_image = image.copy()

        # 1. Depth-aware denoising
        if "denoise" in self.processors:
            logger.debug("Applying depth-aware denoising")
            result_image = self.processors["denoise"](result_image, depth)

        # 2. Zone-based tone mapping
        if "tone_mapping" in self.processors:
            logger.debug("Applying zone tone mapping")
            result_image = self.processors["tone_mapping"](result_image, depth)

        # 3. Atmospheric effects
        if "atmospheric" in self.processors:
            logger.debug("Applying atmospheric effects")
            result_image = self.processors["atmospheric"](result_image, depth)

        # 4. Depth-guided filters
        if "filters" in self.processors:
            logger.debug("Applying depth-guided filters")
            result_image = self.processors["filters"](result_image, depth)

        return result_image

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

        # Estimate depth (with caching) and apply depth postprocessing if configured
        depth_result, depth = self._estimate_depth(image)

        # Apply processing pipeline
        result_image = self._apply_processors(image, depth)

        # Compute processing time
        processing_time = time.time() - start_time

        # Collect metadata
        metadata = {
            "input_path": str(image_path),
            "input_shape": image.shape,
            "processing_time_sec": processing_time,
            "depth_inference_time_ms": depth_result["metadata"]["inference_time_ms"],
            "processors_applied": list(self.processors.keys()),
            "depth_stats": depth_statistics(depth),
        }

        # Update global stats
        self.stats["images_processed"] += 1
        self.stats["total_time"] += processing_time

        logger.info(f"Processed in {processing_time:.2f}s")

        return {
            "image": result_image,
            "depth": depth,
            "metadata": metadata,
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
        loaded_images = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_image, path, normalize=True): str(path) for path in image_paths}

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

        # Estimate depth (with caching) and apply depth postprocessing if configured
        depth_result, depth = self._estimate_depth(image)

        # Apply processing pipeline
        result_image = self._apply_processors(image, depth)

        processing_time = time.time() - start_time

        metadata = {
            "input_path": str(image_path),
            "input_shape": image.shape,
            "processing_time_sec": processing_time,
            "depth_inference_time_ms": depth_result["metadata"]["inference_time_ms"],
            "processors_applied": list(self.processors.keys()),
            "depth_stats": depth_statistics(depth),
        }

        return {
            "image": result_image,
            "depth": depth,
            "metadata": metadata,
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

        Parallel mode uses worker threads that share this pipeline: the pipeline (model, cache,
        processors) is never pickled or copied. The workers decode the images, at most
        ``max_workers`` at a time, while depth estimation and the depth-guided processors handle
        one image at a time and the outputs are written by the calling thread.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            save_depth: Save depth maps as numpy arrays
            save_visualization: Save depth visualizations
            parallel: Enable parallel processing (default: True)
            max_workers: Max worker threads (default: None = CPU count - 1, at most 8)
            preload_images: Sequential mode only: decode every image up front on a thread pool
                (default: True). Parallel mode always decodes inside its workers.

        Returns:
            List of result dictionaries, in input order; images that fail are logged and skipped

        Raises:
            RuntimeError: If every image of a non-empty batch fails
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch processing {len(image_paths)} images")
        logger.info(f"Parallel processing: {parallel}, Preload: {preload_images}")

        # Async image loading (Phase 1 optimization). Parallel mode decodes inside its workers
        # instead, so that only ``max_workers`` decoded images are in flight at any time.
        preloaded_images = {}
        if preload_images and not parallel:
            preloaded_images = self._async_load_images(image_paths, max_workers)

        results = []
        first_error: Optional[Exception] = None

        if parallel:
            # Parallel processing (Phase 1 optimization) on threads that share this pipeline.
            # A process pool would pickle ``self`` (model weights, cache, processors) once per
            # image, and fail every image when the model holds a native handle.
            if max_workers is None:
                max_workers = min(max(1, (os.cpu_count() or 1) - 1), 8)

            # The model, the unsynchronised depth cache and the processors (Numba's default
            # threading layer aborts on concurrent parallel kernels) see one image at a time.
            compute_lock = threading.Lock()

            def process_in_worker(image_path: Union[str, Path]) -> Dict:
                image = load_image(image_path, normalize=True)
                with compute_lock:
                    return self._process_single_image(image_path, image)

            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="depth-batch")
            try:
                futures = [executor.submit(process_in_worker, image_path) for image_path in image_paths]

                for image_path, future in tqdm(
                    zip(image_paths, futures),
                    total=len(image_paths),
                    desc="Processing renders",
                ):
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
                        self.stats["images_processed"] += 1
                        self.stats["total_time"] += result["metadata"]["processing_time_sec"]

                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                        if first_error is None:
                            first_error = e
                        continue
            finally:
                # An interrupt must not run the rest of the queue before it propagates.
                executor.shutdown(wait=True, cancel_futures=True)
        else:
            # Sequential processing (fallback)
            for image_path in tqdm(image_paths, desc="Processing renders"):
                try:
                    path_str = str(image_path)
                    # Release each preloaded image once it has been processed.
                    preloaded = preloaded_images.pop(path_str, None)
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
                    self.stats["images_processed"] += 1
                    self.stats["total_time"] += result["metadata"]["processing_time_sec"]

                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    if first_error is None:
                        first_error = e
                    continue

        # Print summary
        self._print_batch_summary(results)

        if first_error is not None and not results:
            # Fail closed: a batch that produced nothing must not look like a successful run.
            raise RuntimeError(
                f"batch_process failed for all {len(image_paths)} images; first error: {first_error}"
            ) from first_error

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
        input_path = Path(result["metadata"]["input_path"])
        stem = input_path.stem

        # Save enhanced image
        output_config = self.config.get("output", {})
        output_format = output_config.get("output_format", "png")
        quality = output_config.get("jpeg_quality", 95)

        output_image_path = output_dir / f"{stem}_enhanced.{output_format}"
        save_image(result["image"], output_image_path, quality=quality)
        logger.info(f"Saved enhanced image: {output_image_path}")

        # Save depth map
        if save_depth:
            depth_path = output_dir / f"{stem}_depth.npy"
            np.save(depth_path, result["depth"])
            logger.debug(f"Saved depth map: {depth_path}")

        # Save depth visualization
        if save_visualization:
            colormap = output_config.get("depth_colormap", "turbo")
            viz_path = output_dir / f"{stem}_depth_viz.png"
            visualize_depth(result["depth"], colormap=colormap, save_path=str(viz_path))

    def batch_process_streaming(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_depth: bool = True,
        save_visualization: bool = True,
    ) -> Iterator[Dict]:
        """
        Process images with streaming results (Phase 3 optimization).

        Yields results one at a time instead of accumulating in memory.
        Memory usage remains constant regardless of batch size.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            save_depth: Save depth maps as numpy arrays
            save_visualization: Save depth visualizations

        Yields:
            Result dictionary for each processed image

        Example:
            >>> for result in pipeline.batch_process_streaming(paths, 'output/'):
            ...     print(f"Processed: {result['metadata']['input_path']}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Streaming batch processing {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc="Processing renders (streaming)"):
            try:
                # Process image
                result = self.process_render(image_path)

                # Save results immediately
                self.save_result(
                    result,
                    output_dir,
                    save_depth=save_depth,
                    save_visualization=save_visualization,
                )

                # Yield result (can be garbage collected after this)
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
        Process images with pipeline parallelism (Phase 3 optimization).

        Uses producer-consumer pattern to overlap I/O, depth estimation,
        and post-processing stages for maximum hardware utilization.

        Pipeline stages:
        1. Load images (I/O bound → ThreadPool)
        2. Depth estimation (GPU bound → sequential or GPU batch)
        3. Post-processing (CPU bound → ThreadPool)
        4. Save results (I/O bound → ThreadPool)

        Args:
            image_paths: List of input image paths
            output_dir: Output directory
            save_depth: Save depth maps as numpy arrays
            save_visualization: Save depth visualizations
            pipeline_workers: Number of worker threads per stage

        Yields:
            Result dictionary for each processed image

        Example:
            >>> for result in pipeline.batch_process_pipelined(paths, 'output/'):
            ...     print(f"Processed: {result['metadata']['input_path']}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline processing {len(image_paths)} images")

        # Create queues for each stage (bounded to prevent memory issues)
        load_queue = queue.Queue(maxsize=10)
        depth_queue = queue.Queue(maxsize=10)
        process_queue = queue.Queue(maxsize=10)
        save_queue = queue.Queue(maxsize=10)

        # Stage 1: Load images
        def loader_worker():
            """Load images asynchronously."""
            for path in image_paths:
                try:
                    image = load_image(path, normalize=True)
                    load_queue.put((path, image))
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
            load_queue.put(None)  # Sentinel to signal completion

        # Stage 2: Depth estimation
        def depth_worker():
            """Estimate depth for loaded images."""
            while True:
                item = load_queue.get()
                if item is None:
                    depth_queue.put(None)
                    break

                path, image = item
                try:
                    # Cached depth estimation plus the configured depth postprocessing
                    depth_result, depth = self._estimate_depth(image)
                    depth_queue.put((path, image, depth_result, depth))
                except Exception as e:
                    logger.error(f"Failed depth estimation for {path}: {e}")

        # Stage 3: Post-processing
        def process_worker():
            """Apply depth-aware processing."""
            while True:
                item = depth_queue.get()
                if item is None:
                    process_queue.put(None)
                    break

                path, image, depth_result, depth = item
                try:
                    # Apply all processors
                    result_image = self._apply_processors(image, depth)

                    result = {
                        "image": result_image,
                        "depth": depth,
                        "metadata": {
                            "input_path": str(path),
                            "input_shape": image.shape,
                            "depth_inference_time_ms": depth_result["metadata"]["inference_time_ms"],
                            "processors_applied": list(self.processors.keys()),
                            "depth_stats": depth_statistics(depth),
                        },
                    }

                    process_queue.put((path, result))
                except Exception as e:
                    logger.error(f"Failed processing {path}: {e}")

        # Stage 4: Save results
        def save_worker():
            """Save processed results."""
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

        # Start all worker threads
        threads = [
            threading.Thread(target=loader_worker, name="Loader"),
            threading.Thread(target=depth_worker, name="DepthEstimator"),
            threading.Thread(target=process_worker, name="Processor"),
            threading.Thread(target=save_worker, name="Saver"),
        ]

        for t in threads:
            t.daemon = True
            t.start()

        # Yield results as they complete (streaming!)
        processed_count = 0
        with tqdm(total=len(image_paths), desc="Pipeline processing") as pbar:
            while True:
                item = save_queue.get()
                if item is None:
                    break

                path, result = item
                processed_count += 1
                pbar.update(1)

                # Update stats
                self.stats["images_processed"] += 1

                yield result

        # Wait for all threads to complete
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

        Provides fast preview at low resolution, then optionally high-res final.
        Ideal for interactive workflows and parameter tuning.

        Args:
            image_path: Path to input image
            quality_levels: Scaling factors for progressive processing (e.g., [0.25, 1.0])
            return_all_levels: Return all quality levels (default: only highest)

        Returns:
            Result dictionary (or list of results if return_all_levels=True)

        Example:
            >>> # Fast preview
            >>> result = pipeline.process_render_progressive('render.jpg', [0.25])
            >>> # Progressive refinement
            >>> results = pipeline.process_render_progressive(
            ...     'render.jpg',
            ...     [0.25, 0.5, 1.0],
            ...     return_all_levels=True
            ... )
        """
        from .utils.image_utils import resize_image

        logger.info(f"Progressive processing: {image_path} at levels {quality_levels}")

        # Load full resolution image
        image_full = load_image(image_path, normalize=True)
        h_full, w_full = image_full.shape[:2]

        results = []

        for scale in quality_levels:
            start_time = time.time()

            if scale < 1.0:
                # Downsample for speed
                h_scaled = int(h_full * scale)
                w_scaled = int(w_full * scale)
                image_scaled = resize_image(image_full, size=(h_scaled, w_scaled), interpolation="bilinear")
                logger.info(f"Processing at {scale:.0%} resolution: {h_scaled}x{w_scaled}")
            else:
                image_scaled = image_full
                logger.info(f"Processing at full resolution: {h_full}x{w_full}")

            # Estimate depth at current scale and apply depth postprocessing if configured
            depth_result, depth = self._estimate_depth(image_scaled)

            # Apply processing pipeline
            result_image = self._apply_processors(image_scaled, depth)

            # Upscale back to full resolution if needed
            if scale < 1.0:
                result_image = resize_image(result_image, size=(h_full, w_full), interpolation="bicubic")
                depth = resize_image(depth, size=(h_full, w_full), interpolation="bilinear")

            processing_time = time.time() - start_time

            result = {
                "image": result_image,
                "depth": depth,
                "metadata": {
                    "input_path": str(image_path),
                    "input_shape": image_full.shape,
                    "processing_scale": scale,
                    "processing_time_sec": processing_time,
                    "depth_inference_time_ms": depth_result["metadata"]["inference_time_ms"],
                    "processors_applied": list(self.processors.keys()),
                    "depth_stats": depth_statistics(depth),
                },
            }

            results.append(result)
            logger.info(f"Level {scale:.0%} complete in {processing_time:.2f}s")

        if return_all_levels:
            return results
        else:
            return results[-1]  # Return highest quality level

    def _print_batch_summary(self, results: List[Dict]):
        """Print batch processing summary."""
        if not results:
            logger.warning("No images processed successfully")
            return

        total_time = sum(r["metadata"]["processing_time_sec"] for r in results)
        avg_time = total_time / len(results)
        avg_depth_time = np.mean([r["metadata"]["depth_inference_time_ms"] for r in results])

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
        logger.info("\nCache statistics:")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        logger.info(f"  Entries: {cache_stats['size']}/{cache_stats['max_size']}")

        logger.info("=" * 60 + "\n")

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        stats["cache_stats"] = self.cache.get_stats()

        if stats["images_processed"] > 0:
            stats["avg_time_per_image"] = stats["total_time"] / stats["images_processed"]

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
