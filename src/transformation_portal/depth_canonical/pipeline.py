"""Depth processing pipeline with PBR integration.

This module provides the main DepthPipeline orchestrator for processing images.
Phase 2: Full depth estimation integration with caching.
"""

from pathlib import Path
from typing import Optional, Dict, Union, List
import hashlib
import numpy as np
from PIL import Image

from .config import UnifiedDepthConfig
from .models import ModelRegistry
from .processing import generate_pbr_maps
from .io import write_pbr_maps


class DepthPipelineResult:
    """Result container for depth pipeline processing.

    Attributes:
        depth_map: Depth map as numpy array (if generated)
        depth_path: Path to saved depth map (if saved)
        pbr_maps: Dictionary of PBR maps (normal, roughness, ao) if generated
        pbr_paths: Dictionary of paths to saved PBR maps if saved
    """

    def __init__(self):
        self.depth_map: Optional[np.ndarray] = None
        self.depth_path: Optional[Path] = None
        self.pbr_maps: Optional[Dict[str, np.ndarray]] = None
        self.pbr_paths: Optional[Dict[str, Path]] = None


class DepthPipeline:
    """Production depth processing pipeline with PBR integration.

    Capabilities:
    - Depth estimation (full in Phase 2)
    - PBR map generation (normal, roughness, ambient occlusion)
    - Atomic file writes with cleanup
    - Path validation and security checks
    - LRU caching for depth maps
    - Disk caching for computed depths

    Example:
        >>> config = UnifiedDepthConfig(
        ...     processing=ProcessingConfig(
        ...         pbr=PBRConfig(enabled=True, normal_strength=1.2)
        ...     )
        ... )
        >>> pipeline = DepthPipeline(config)
        >>> result = pipeline.process(
        ...     image_path="render.jpg",
        ...     output_dir="output/"
        ... )
        >>> print(result.pbr_paths)
    """

    def __init__(self, config: UnifiedDepthConfig):
        """Initialize the pipeline.

        Args:
            config: Unified depth processing configuration
        """
        self.config = config
        self.model_registry = ModelRegistry()
        self._depth_cache_dir = None

        # Initialize disk cache directory if enabled
        if config.io.cache_enabled:
            cache_root = Path.home() / ".cache" / "transformation_portal"
            self._depth_cache_dir = cache_root / "depth_maps"
            self._depth_cache_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        image: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
        output_dir: Optional[Path] = None,
        depth_map: Optional[np.ndarray] = None,
        basename: Optional[str] = None,
        # Backward compatibility
        image_path: Optional[Path] = None,
    ) -> DepthPipelineResult:
        """Process image to generate depth and/or PBR maps.

        Phase 2: Full depth estimation from image if depth_map not provided.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            output_dir: Directory to save outputs (optional)
            depth_map: Pre-computed depth map (optional - will estimate if not provided)
            basename: Base name for output files (defaults to image filename)
            image_path: (Deprecated) Use 'image' parameter instead

        Returns:
            DepthPipelineResult containing generated maps and paths

        Raises:
            ValueError: If required inputs are missing or invalid
        """
        # Backward compatibility: image_path -> image
        if image_path is not None:
            if image is not None:
                raise ValueError("Cannot specify both 'image' and 'image_path'")
            image = image_path

        result = DepthPipelineResult()

        # Validate inputs
        if image is None and depth_map is None:
            raise ValueError("Either image or depth_map must be provided")

        # Estimate depth if not provided
        if depth_map is None:
            if image is None:
                raise ValueError("image required for depth estimation")

            depth_map = self._estimate_depth(image)

        if depth_map.ndim != 2:
            raise ValueError(f"Depth map must be 2D, got shape {depth_map.shape}")

        # Path validation if output_dir provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            if self.config.security.validate_paths:
                # Ensure output_dir is absolute and safe
                output_dir = output_dir.resolve()
                output_dir.mkdir(parents=True, exist_ok=True)

        # Determine basename for output files
        if basename is None:
            if image is not None:
                if isinstance(image, (str, Path)):
                    basename = Path(image).stem
                else:
                    basename = "depth_output"
            else:
                basename = "depth_output"

        # Store depth map in result
        result.depth_map = depth_map

        # Generate PBR maps if enabled
        if self.config.processing.pbr.enabled:
            normal_map, roughness_map, ao_map = generate_pbr_maps(
                depth_map,
                config=self.config.processing.pbr
            )

            result.pbr_maps = {
                "normal": normal_map,
                "roughness": roughness_map,
                "ao": ao_map,
            }

            # Save PBR maps if output_dir provided
            if output_dir is not None:
                result.pbr_paths = write_pbr_maps(
                    normal_map,
                    roughness_map,
                    ao_map,
                    output_dir,
                    basename
                )

        return result

    def _estimate_depth(
        self,
        image: Union[Path, str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Estimate depth from image with caching.

        Args:
            image: Input image

        Returns:
            Depth map as numpy array [0, 1]
        """
        # Generate cache key
        cache_key = self._generate_cache_key(image)

        # Check disk cache if enabled
        if self.config.io.cache_enabled and self._depth_cache_dir:
            cached_path = self._depth_cache_dir / f"{cache_key}.npy"
            if cached_path.exists():
                try:
                    return np.load(cached_path)
                except Exception:
                    # Cache corrupted, regenerate
                    pass

        # Get model
        model = self.model_registry.get_model(
            variant=self.config.model.variant,
            device=self.config.model.device,
            dtype=self.config.model.dtype
        )

        # Estimate depth
        result = model.estimate(image)
        depth_map = result["depth"]

        # Save to disk cache if enabled
        if self.config.io.cache_enabled and self._depth_cache_dir:
            cached_path = self._depth_cache_dir / f"{cache_key}.npy"
            try:
                np.save(cached_path, depth_map)
            except Exception:
                # Cache write failed, continue without caching
                pass

        return depth_map

    def _generate_cache_key(
        self,
        image: Union[Path, str, Image.Image, np.ndarray]
    ) -> str:
        """Generate cache key for image and config.

        Args:
            image: Input image

        Returns:
            Cache key string
        """
        # Convert image to bytes for hashing
        if isinstance(image, (str, Path)):
            # Use file path and mtime
            path = Path(image)
            if path.exists():
                mtime = path.stat().st_mtime
                key_data = f"{path}_{mtime}".encode()
            else:
                # Read file content
                with open(path, "rb") as f:
                    key_data = f.read()
        elif isinstance(image, Image.Image):
            # Convert to bytes
            key_data = image.tobytes()
        else:
            # Numpy array
            key_data = image.tobytes()

        # Hash image data
        image_hash = hashlib.sha256(key_data).hexdigest()[:16]

        # Include model config in key
        config_str = f"{self.config.model.variant.value}_{self.config.model.device.value}"

        return f"{image_hash}_{config_str}"

    def batch_process(
        self,
        images: Optional[List[Union[Path, str]]] = None,
        output_dir: Optional[Path] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
        # Backward compatibility
        image_paths: Optional[List[Path]] = None,
    ) -> List[DepthPipelineResult]:
        """Process multiple images in batch.

        Phase 2: Full depth estimation from images if depth_maps not provided.

        Args:
            images: List of input images (paths or PIL Images)
            output_dir: Directory to save outputs
            depth_maps: Pre-computed depth maps (optional)
            image_paths: (Deprecated) Use 'images' parameter instead

        Returns:
            List of DepthPipelineResult objects
        """
        # Backward compatibility: image_paths -> images
        if image_paths is not None:
            if images is not None:
                raise ValueError("Cannot specify both 'images' and 'image_paths'")
            images = image_paths

        if images is None:
            raise ValueError("images parameter is required")

        if depth_maps is not None and len(depth_maps) != len(images):
            raise ValueError(
                f"Length mismatch: {len(images)} images, "
                f"{len(depth_maps)} depth maps"
            )

        results = []
        for i, image in enumerate(images):
            depth_map = depth_maps[i] if depth_maps else None
            result = self.process(
                image=image,
                output_dir=output_dir,
                depth_map=depth_map,
            )
            results.append(result)

        return results

    # Alias for backward compatibility
    def process_batch(self, *args, **kwargs) -> List[DepthPipelineResult]:
        """Deprecated alias for batch_process. Use batch_process instead."""
        return self.batch_process(*args, **kwargs)
