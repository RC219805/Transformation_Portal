"""Concrete async pipeline stage implementations for image processing.

This module provides ready-to-use async stages for common image processing
operations that can be composed into async pipelines.

Stages:
- ImageLoadStage: Async image loading with format detection
- ImageSaveStage: Async image saving with format options
- DepthEstimationStage: Depth map estimation (GPU-accelerated)
- MaterialResponseStage: Material-aware enhancement
- ColorGradingStage: LUT-based color grading
- ResizeStage: Image resizing with quality options
- DenoiseStage: Depth-aware denoising

Example:
    >>> from transformation_portal.streaming.stages import (
    ...     ImageLoadStage, DepthEstimationStage, ImageSaveStage
    ... )
    >>> from transformation_portal.streaming.async_pipeline import AsyncPipeline
    >>>
    >>> pipeline = AsyncPipeline()
    >>> pipeline.add_stage(ImageLoadStage())
    >>> pipeline.add_stage(DepthEstimationStage())
    >>> pipeline.add_stage(ImageSaveStage(output_dir="./output"))
"""

from __future__ import annotations

import asyncio
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .async_pipeline import (
    AsyncStage,
    DeviceType,
    WorkerPool,
)


@dataclass
class ImageData:
    """Container for image data flowing through pipeline.

    Attributes:
        array: Image as numpy array
        path: Original file path
        depth_map: Optional depth map array
        metadata: Processing metadata
    """
    array: Any  # numpy array
    path: Path
    depth_map: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape."""
        return self.array.shape if self.array is not None else ()

    @property
    def dtype(self):
        """Get image dtype."""
        return self.array.dtype if self.array is not None else None


class ImageLoadStage(AsyncStage[Path, ImageData]):
    """Async stage for loading images with format detection.

    Supports: JPEG, PNG, TIFF (8/16-bit), WebP, BMP

    Example:
        >>> stage = ImageLoadStage(max_concurrent=4)
        >>> result = await stage(Path("image.jpg"))
        >>> print(result.data.shape)
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        load_exif: bool = True,
        convert_16bit: bool = True,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize image load stage.

        Args:
            max_concurrent: Maximum concurrent loads
            load_exif: Extract EXIF metadata
            convert_16bit: Convert 16-bit to float32
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="image_load",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=True
        )
        self._load_exif = load_exif
        self._convert_16bit = convert_16bit
        self._worker_pool = worker_pool
        self._owns_pool = worker_pool is None

    async def startup(self) -> None:
        """Initialize resources."""
        await super().startup()
        if self._owns_pool:
            self._worker_pool = WorkerPool(io_workers=self.max_concurrent * 2)
            await self._worker_pool.startup()

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._owns_pool and self._worker_pool:
            await self._worker_pool.shutdown()
        await super().shutdown()

    def _load_sync(self, path: Path) -> ImageData:
        """Synchronous image loading (runs in thread pool)."""
        import numpy as np
        from PIL import Image

        metadata = {
            'original_path': str(path),
            'filename': path.name,
        }

        # Check for TIFF with tifffile
        if path.suffix.lower() in ('.tiff', '.tif'):
            try:
                import tifffile
                array = tifffile.imread(str(path))
                metadata['format'] = 'TIFF'
                metadata['loaded_with'] = 'tifffile'
            except ImportError:
                # Fall back to PIL
                with Image.open(path) as img:
                    array = np.array(img)
                    metadata['format'] = img.format
                    metadata['loaded_with'] = 'PIL'
        else:
            with Image.open(path) as img:
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                metadata['size'] = img.size

                if self._load_exif and hasattr(img, '_getexif'):
                    try:
                        exif = img._getexif()
                        if exif:
                            metadata['exif'] = {
                                k: v for k, v in exif.items()
                                if isinstance(v, (str, int, float, bytes))
                            }
                    except Exception:
                        pass  # EXIF extraction optional

                array = np.array(img)
                metadata['loaded_with'] = 'PIL'

        # Handle 16-bit images
        if array.dtype == np.uint16 and self._convert_16bit:
            array = array.astype(np.float32) / 65535.0
            metadata['converted_from'] = 'uint16'
            metadata['dtype'] = 'float32'
        else:
            metadata['dtype'] = str(array.dtype)

        metadata['shape'] = array.shape
        metadata['memory_mb'] = array.nbytes / (1024 * 1024)

        return ImageData(
            array=array,
            path=path,
            metadata=metadata
        )

    async def process(self, item: Path) -> ImageData:
        """Load image asynchronously.

        Args:
            item: Path to image file

        Returns:
            ImageData with loaded array
        """
        if self._worker_pool:
            return await self._worker_pool.run_io(self._load_sync, item)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_sync, item)


class ImageSaveStage(AsyncStage[ImageData, ImageData]):
    """Async stage for saving images with format options.

    Supports multiple output formats with quality settings.

    Example:
        >>> stage = ImageSaveStage(
        ...     output_dir="./output",
        ...     output_format="TIFF",
        ...     quality=95
        ... )
        >>> result = await stage(image_data)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        output_format: str = "TIFF",
        quality: int = 95,
        suffix: str = "_processed",
        max_concurrent: int = 4,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize image save stage.

        Args:
            output_dir: Output directory
            output_format: Output format (TIFF, JPEG, PNG, WebP)
            quality: Quality for lossy formats (1-100)
            suffix: Suffix to add to filename
            max_concurrent: Maximum concurrent saves
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="image_save",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=True
        )
        self._output_dir = Path(output_dir)
        self._format = output_format.upper()
        self._quality = quality
        self._suffix = suffix
        self._worker_pool = worker_pool
        self._owns_pool = worker_pool is None

    async def startup(self) -> None:
        """Initialize resources."""
        await super().startup()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self._owns_pool:
            self._worker_pool = WorkerPool(io_workers=self.max_concurrent * 2)
            await self._worker_pool.startup()

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._owns_pool and self._worker_pool:
            await self._worker_pool.shutdown()
        await super().shutdown()

    def _save_sync(self, image_data: ImageData) -> Path:
        """Synchronous image saving (runs in thread pool)."""
        import numpy as np
        from PIL import Image

        # Determine output path
        stem = image_data.path.stem + self._suffix
        ext_map = {
            'TIFF': '.tiff',
            'JPEG': '.jpg',
            'PNG': '.png',
            'WEBP': '.webp',
        }
        ext = ext_map.get(self._format, '.tiff')
        output_path = self._output_dir / f"{stem}{ext}"

        array = image_data.array

        # Convert float to appropriate int type
        if array.dtype in (np.float32, np.float64):
            if self._format == 'TIFF':
                # Save as 16-bit TIFF
                array = (np.clip(array, 0, 1) * 65535).astype(np.uint16)
            else:
                # Save as 8-bit for other formats
                array = (np.clip(array, 0, 1) * 255).astype(np.uint8)

        if self._format == 'TIFF':
            try:
                import tifffile
                tifffile.imwrite(str(output_path), array, compression='lzw')
            except ImportError:
                img = Image.fromarray(array)
                img.save(str(output_path), format='TIFF')
        else:
            img = Image.fromarray(array)
            if self._format in ('JPEG', 'WEBP'):
                img.save(str(output_path), quality=self._quality)
            else:
                img.save(str(output_path))

        return output_path

    async def process(self, item: ImageData) -> ImageData:
        """Save image asynchronously.

        Args:
            item: Image data to save

        Returns:
            ImageData with updated output path in metadata
        """
        if self._worker_pool:
            output_path = await self._worker_pool.run_io(
                self._save_sync, item
            )
        else:
            loop = asyncio.get_event_loop()
            output_path = await loop.run_in_executor(
                None, self._save_sync, item
            )

        item.metadata['output_path'] = str(output_path)
        return item


class DepthEstimationStage(AsyncStage[ImageData, ImageData]):
    """Async stage for depth map estimation.

    Uses Depth Anything V2 or compatible depth models.
    GPU-accelerated with automatic device detection.

    Example:
        >>> stage = DepthEstimationStage(
        ...     device=DeviceType.CUDA,
        ...     model_size="base"
        ... )
        >>> result = await stage(image_data)
        >>> depth_map = result.data.depth_map
    """

    def __init__(
        self,
        device: DeviceType = DeviceType.AUTO,
        model_size: str = "base",
        max_concurrent: int = 1,
        cache_model: bool = True,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize depth estimation stage.

        Args:
            device: Device for inference (AUTO, CUDA, MPS, CPU)
            model_size: Model size (small, base, large)
            max_concurrent: Maximum concurrent estimations
            cache_model: Keep model loaded in memory
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="depth_estimation",
            device=device,
            max_concurrent=max_concurrent,
            required=False  # Depth is often optional
        )
        self._model_size = model_size
        self._cache_model = cache_model
        self._model = None
        self._torch_device = None
        self._worker_pool = worker_pool

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                return "cuda"
            # Check for MPS (Apple Silicon) - verify attribute chain exists
            if (hasattr(torch, 'backends')
                    and hasattr(torch.backends, 'mps')
                    and torch.backends.mps.is_available()):
                return "mps"
        except (ImportError, AttributeError):
            # ImportError: torch not installed
            # AttributeError: torch.cuda/backends/mps attributes missing (e.g., mock torch)
            pass
        return "cpu"

    async def startup(self) -> None:
        """Initialize depth model."""
        await super().startup()

        # Determine device
        if self.device == DeviceType.AUTO:
            self._torch_device = self._detect_device()
        elif self.device == DeviceType.CUDA:
            self._torch_device = "cuda"
        elif self.device == DeviceType.MPS:
            self._torch_device = "mps"
        else:
            self._torch_device = "cpu"

        if self._cache_model:
            self._load_model()

    def _load_model(self) -> None:
        """Load depth model (lazy initialization)."""
        if self._model is not None:
            return

        try:
            # Try to import from transformation_portal depth module
            from transformation_portal.depth.models import load_depth_model
            self._model = load_depth_model(
                model_size=self._model_size,
                device=self._torch_device
            )
        except ImportError:
            # Fallback: create a placeholder that returns mock depth
            self._model = self._create_mock_model()

    def _create_mock_model(self) -> Callable:
        """Create mock depth model for testing/fallback."""
        import numpy as np

        def mock_estimate(image: Any) -> Any:
            """Generate mock depth map based on image gradient."""
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            # Create gradient-based mock depth
            h, w = gray.shape[:2]
            y_grad = np.linspace(0, 1, h).reshape(-1, 1)
            depth = np.tile(y_grad, (1, w))

            return depth.astype(np.float32)

        return mock_estimate

    async def shutdown(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                # torch is optional; if not installed, skip GPU cache cleanup
                pass

        await super().shutdown()

    def _estimate_sync(self, image_data: ImageData) -> ImageData:
        """Synchronous depth estimation."""
        if self._model is None:
            self._load_model()

        depth_map = self._model(image_data.array)

        image_data.depth_map = depth_map
        image_data.metadata['depth_estimated'] = True
        image_data.metadata['depth_device'] = self._torch_device

        return image_data

    async def process(self, item: ImageData) -> ImageData:
        """Estimate depth map asynchronously.

        Args:
            item: Image data

        Returns:
            ImageData with depth_map populated
        """
        loop = asyncio.get_event_loop()

        if self._worker_pool:
            return await self._worker_pool.run_cpu(
                self._estimate_sync, item
            )
        else:
            return await loop.run_in_executor(
                None, self._estimate_sync, item
            )


class MaterialResponseStage(AsyncStage[ImageData, ImageData]):
    """Async stage for material-aware image enhancement.

    Applies physics-based surface enhancement for different materials.

    Example:
        >>> stage = MaterialResponseStage(
        ...     materials=["wood", "metal", "glass"],
        ...     intensity=0.8
        ... )
        >>> result = await stage(image_data)
    """

    def __init__(
        self,
        materials: Optional[List[str]] = None,
        intensity: float = 1.0,
        use_depth: bool = True,
        max_concurrent: int = 2,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize material response stage.

        Args:
            materials: List of material types to enhance
            intensity: Enhancement intensity (0.0-2.0)
            use_depth: Use depth map for material detection
            max_concurrent: Maximum concurrent processes
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="material_response",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=False
        )
        self._materials = materials or ["wood", "metal", "glass", "textile"]
        self._intensity = intensity
        self._use_depth = use_depth
        self._worker_pool = worker_pool

    def _enhance_sync(self, image_data: ImageData) -> ImageData:
        """Synchronous material enhancement with physics-based rendering.

        Implements Material Response technology following three core tenets:
        1. Respect energy conservation in highlights (preserve specular)
        2. Preserve midtone texture (keep organic materials tactile)
        3. Blend transitions between materials (smooth, authored feel)
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter, sobel

        array = image_data.array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        h, w = array.shape[:2]
        intensity = self._intensity

        # Compute luminance and saturation for material detection
        if len(array.shape) == 3 and array.shape[2] >= 3:
            luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
            saturation = np.maximum(array.max(axis=2) - array.min(axis=2), 1e-6)
        else:
            luminance = array[..., 0] if len(array.shape) == 3 else array
            saturation = np.zeros_like(luminance)

        # ============================================================
        # MATERIAL DETECTION via color/intensity heuristics
        # ============================================================

        # Floor region mask (lower portion of image, typical perspective)
        y_norm = np.linspace(0, 1, h).reshape(-1, 1)
        floor_mask = np.clip((y_norm - 0.5) / 0.5, 0.0, 1.0)
        floor_mask = np.broadcast_to(floor_mask, (h, w))

        # Specular/highlight mask (bright regions - energy conservation)
        highlight_mask = np.clip((luminance - 0.7) / 0.3, 0.0, 1.0)
        highlight_mask = gaussian_filter(highlight_mask, sigma=2.0)

        # Midtone mask (preserve texture)
        midtone_mask = np.clip(1.0 - np.abs(luminance - 0.5) / 0.4, 0.0, 1.0)
        midtone_mask = gaussian_filter(midtone_mask, sigma=1.5)

        # Wood detection (warm mid-tones, moderate saturation)
        wood_mask = np.zeros((h, w), dtype=np.float32)
        if "wood" in self._materials:
            warm_bias = array[..., 0] - 0.5 * (array[..., 1] + array[..., 2]) if len(array.shape) == 3 else np.zeros((h, w))
            wood_mask = (
                np.clip((warm_bias + 0.1) / 0.2, 0.0, 1.0) *
                np.clip((saturation - 0.08) / 0.25, 0.0, 1.0) *
                np.clip((luminance - 0.2) / 0.5, 0.0, 1.0)
            )
            wood_mask *= floor_mask  # Wood typically on floors
            wood_mask = gaussian_filter(wood_mask, sigma=3.0)

        # Metal detection (neutral hue, high local contrast)
        metal_mask = np.zeros((h, w), dtype=np.float32)
        if "metal" in self._materials:
            neutral = np.clip((0.15 - saturation) / 0.15, 0.0, 1.0)
            edge_mag = np.abs(sobel(luminance, axis=0)) + np.abs(sobel(luminance, axis=1))
            edge_mag = gaussian_filter(edge_mag, sigma=1.0)
            if edge_mag.max() > 0:
                edge_mag = edge_mag / edge_mag.max()
            metal_mask = neutral * edge_mag * np.clip(luminance, 0.3, 0.9)
            metal_mask = gaussian_filter(metal_mask, sigma=2.0)

        # Glass detection (high brightness, low saturation, transparency regions)
        glass_mask = np.zeros((h, w), dtype=np.float32)
        if "glass" in self._materials:
            glass_mask = (
                np.clip((luminance - 0.6) / 0.4, 0.0, 1.0) *
                np.clip((0.12 - saturation) / 0.12, 0.0, 1.0)
            )
            glass_mask = gaussian_filter(glass_mask, sigma=3.0)

        # Textile detection (soft edges, mid brightness, low-moderate saturation)
        textile_mask = np.zeros((h, w), dtype=np.float32)
        if "textile" in self._materials:
            textile_mask = (
                np.clip((luminance - 0.35) / 0.4, 0.0, 1.0) *
                np.clip((0.3 - saturation) / 0.3, 0.0, 1.0) *
                np.clip(1.0 - floor_mask, 0.0, 1.0)  # Typically not on floor
            )
            textile_mask = gaussian_filter(textile_mask, sigma=2.0)

        # ============================================================
        # PHYSICS-BASED ENHANCEMENTS
        # ============================================================
        enhanced = array.copy()

        # 1. High-frequency texture boost (reveals grain and fabric weave)
        blurred = gaussian_filter(array, sigma=(1.2, 1.2, 0) if len(array.shape) == 3 else 1.2)
        texture_detail = array - blurred
        texture_boost = 0.25 * intensity * midtone_mask[..., np.newaxis]
        enhanced = np.clip(enhanced + texture_boost * texture_detail, 0.0, 1.0)

        # 2. Wood grain enhancement (warm tone, directional grain)
        if "wood" in self._materials and wood_mask.max() > 0.01:
            # Horizontal grain detection
            grain = np.abs(sobel(luminance * wood_mask, axis=1))
            grain = gaussian_filter(grain, sigma=(0.8, 2.0))
            if grain.max() > 0:
                grain = grain / grain.max()
            warm_wood = np.array([0.88, 0.76, 0.60], dtype=np.float32)
            wood_enhance = 0.15 * intensity * wood_mask[..., np.newaxis] * grain[..., np.newaxis]
            enhanced = np.clip(enhanced + wood_enhance * (warm_wood - enhanced), 0.0, 1.0)

        # 3. Metal specular enhancement (preserve reflections, add sheen)
        if "metal" in self._materials and metal_mask.max() > 0.01:
            # Specular highlight recovery
            specular = gaussian_filter(luminance * metal_mask, sigma=2.0)
            specular = np.clip((specular - 0.4) / 0.5, 0.0, 1.0)
            cool_metal = np.array([0.92, 0.94, 0.98], dtype=np.float32)
            metal_enhance = 0.12 * intensity * metal_mask[..., np.newaxis] * specular[..., np.newaxis]
            enhanced = np.clip(enhanced + metal_enhance * (cool_metal - enhanced), 0.0, 1.0)

        # 4. Glass clarity enhancement (subtle edge sharpening, reflection)
        if "glass" in self._materials and glass_mask.max() > 0.01:
            glass_edges = np.abs(sobel(luminance, axis=0)) + np.abs(sobel(luminance, axis=1))
            glass_edges = gaussian_filter(glass_edges, sigma=0.8)
            if glass_edges.max() > 0:
                glass_edges = glass_edges / glass_edges.max()
            glass_enhance = 0.08 * intensity * glass_mask[..., np.newaxis]
            enhanced = np.clip(enhanced + glass_enhance * glass_edges[..., np.newaxis], 0.0, 1.0)

        # 5. Textile softness (micro-contrast for fabric texture)
        if "textile" in self._materials and textile_mask.max() > 0.01:
            textile_detail = array - gaussian_filter(array, sigma=(1.5, 1.5, 0) if len(array.shape) == 3 else 1.5)
            textile_enhance = 0.18 * intensity * textile_mask[..., np.newaxis]
            enhanced = np.clip(enhanced + textile_enhance * textile_detail, 0.0, 1.0)

        # 6. Highlight energy conservation (prevent clipping, preserve sheen)
        # Roll off aggressive enhancements in highlight regions
        highlight_rolloff = 1.0 - 0.5 * highlight_mask[..., np.newaxis]
        enhanced = array + highlight_rolloff * (enhanced - array)

        # 7. Ambient occlusion (ground objects with subtle contact shadows)
        edge_mag = np.abs(sobel(luminance, axis=0)) + np.abs(sobel(luminance, axis=1))
        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()
        occlusion = gaussian_filter(edge_mag, sigma=1.5)
        ao_strength = 0.1 * intensity
        enhanced = np.clip(enhanced * (1.0 - ao_strength * occlusion[..., np.newaxis]), 0.0, 1.0)

        # 8. Transition blending (smooth material boundaries)
        # Apply final gaussian smoothing to mask edges
        final_blend = gaussian_filter(enhanced, sigma=0.3)
        blend_factor = 0.15  # Subtle blending
        enhanced = enhanced * (1 - blend_factor) + final_blend * blend_factor

        # Use depth map for depth-aware processing if available
        if self._use_depth and image_data.depth_map is not None:
            depth = image_data.depth_map.astype(np.float32)
            if depth.max() > 1.0:
                depth = depth / depth.max()
            # Reduce enhancement intensity in distant regions
            depth_factor = np.clip(1.0 - depth * 0.3, 0.7, 1.0)
            enhanced = array + depth_factor[..., np.newaxis] * (enhanced - array)

        image_data.array = np.clip(enhanced, 0.0, 1.0)
        image_data.metadata['material_enhanced'] = True
        image_data.metadata['materials'] = self._materials
        image_data.metadata['material_response_version'] = '2.0'
        image_data.metadata['enhancement_intensity'] = intensity

        return image_data

    async def process(self, item: ImageData) -> ImageData:
        """Apply material response enhancement.

        Args:
            item: Image data

        Returns:
            Enhanced ImageData
        """
        loop = asyncio.get_event_loop()

        if self._worker_pool:
            return await self._worker_pool.run_cpu(
                self._enhance_sync, item
            )
        else:
            return await loop.run_in_executor(
                None, self._enhance_sync, item
            )


class ColorGradingStage(AsyncStage[ImageData, ImageData]):
    """Async stage for LUT-based color grading.

    Applies 3D LUT color transformations for professional color grading.

    Example:
        >>> stage = ColorGradingStage(
        ...     lut_path="assets/luts/film_emulation/kodak_portra.cube"
        ... )
        >>> result = await stage(image_data)
    """

    def __init__(
        self,
        lut_path: Optional[Union[str, Path]] = None,
        intensity: float = 1.0,
        max_concurrent: int = 4,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize color grading stage.

        Args:
            lut_path: Path to .cube LUT file
            intensity: Grading intensity (0.0-1.0)
            max_concurrent: Maximum concurrent processes
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="color_grading",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=False
        )
        self._lut_path = Path(lut_path) if lut_path else None
        self._intensity = intensity
        self._lut_data = None
        self._worker_pool = worker_pool

    async def startup(self) -> None:
        """Load LUT data."""
        await super().startup()
        if self._lut_path and self._lut_path.exists():
            self._load_lut()

    def _load_lut(self) -> None:
        """Load LUT from .cube file."""
        if self._lut_path is None or not self._lut_path.exists():
            return

        try:
            import numpy as np

            lut_size = 0
            lut_data = []

            with open(self._lut_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('LUT_3D_SIZE'):
                        lut_size = int(line.split()[-1])
                    elif line and not line.startswith('#') and not line.startswith('TITLE'):
                        parts = line.split()
                        if len(parts) == 3:
                            try:
                                r, g, b = map(float, parts)
                                lut_data.append([r, g, b])
                            except ValueError:
                                continue

            if lut_size > 0 and len(lut_data) == lut_size ** 3:
                # pylint: disable=too-many-function-args  # numpy reshape accepts multiple positional args
                self._lut_data = np.array(lut_data).reshape(
                    lut_size, lut_size, lut_size, 3
                )
        except Exception:
            self._lut_data = None

    def _apply_lut_sync(self, image_data: ImageData) -> ImageData:
        """Synchronous LUT application."""
        import numpy as np

        array = image_data.array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        if self._lut_data is not None:
            # Trilinear interpolation for LUT application
            lut_size = self._lut_data.shape[0]
            indices = array * (lut_size - 1)
            indices = np.clip(indices, 0, lut_size - 1.001)

            # Floor indices for nearest-neighbor lookup
            # (trilinear interpolation would use idx1 and frac)
            idx0 = np.floor(indices).astype(np.int32)

            # Simple nearest-neighbor for now (full trilinear is more complex)
            r, g, b = idx0[..., 0], idx0[..., 1], idx0[..., 2]
            graded = self._lut_data[r, g, b]

            # Blend with original based on intensity
            array = array * (1 - self._intensity) + graded * self._intensity
        else:
            # Fallback: simple contrast/saturation adjustment
            # Increase saturation slightly
            if len(array.shape) == 3 and array.shape[2] == 3:
                gray = np.mean(array, axis=2, keepdims=True)
                array = gray + (array - gray) * (1 + 0.1 * self._intensity)

        array = np.clip(array, 0, 1)
        image_data.array = array
        image_data.metadata['color_graded'] = True

        return image_data

    async def process(self, item: ImageData) -> ImageData:
        """Apply color grading.

        Args:
            item: Image data

        Returns:
            Color graded ImageData
        """
        loop = asyncio.get_event_loop()

        if self._worker_pool:
            return await self._worker_pool.run_cpu(
                self._apply_lut_sync, item
            )
        else:
            return await loop.run_in_executor(
                None, self._apply_lut_sync, item
            )


class ResizeStage(AsyncStage[ImageData, ImageData]):
    """Async stage for image resizing.

    Supports multiple interpolation methods and aspect ratio handling.

    Example:
        >>> stage = ResizeStage(
        ...     target_size=(3840, 2160),
        ...     method="lanczos"
        ... )
        >>> result = await stage(image_data)
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        method: str = "lanczos",
        maintain_aspect: bool = True,
        max_concurrent: int = 4,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize resize stage.

        Args:
            target_size: Target (width, height)
            scale_factor: Scale factor (alternative to target_size)
            method: Interpolation method (lanczos, bilinear, nearest)
            maintain_aspect: Maintain aspect ratio
            max_concurrent: Maximum concurrent processes
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="resize",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=False
        )
        self._target_size = target_size
        self._scale_factor = scale_factor
        self._method = method
        self._maintain_aspect = maintain_aspect
        self._worker_pool = worker_pool

    def _resize_sync(self, image_data: ImageData) -> ImageData:
        """Synchronous image resizing."""
        from PIL import Image
        import numpy as np

        array = image_data.array
        original_dtype = array.dtype
        original_shape = array.shape

        # Convert to PIL Image
        if array.dtype in (np.float32, np.float64):
            pil_array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        else:
            pil_array = array

        img = Image.fromarray(pil_array)

        # Determine target size
        if self._target_size:
            target_w, target_h = self._target_size
        elif self._scale_factor:
            target_w = int(img.width * self._scale_factor)
            target_h = int(img.height * self._scale_factor)
        else:
            return image_data  # No resize needed

        # Maintain aspect ratio if requested
        if self._maintain_aspect:
            ratio = min(target_w / img.width, target_h / img.height)
            target_w = int(img.width * ratio)
            target_h = int(img.height * ratio)

        # Select resampling method
        method_map = {
            'lanczos': Image.Resampling.LANCZOS,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'nearest': Image.Resampling.NEAREST,
        }
        resample = method_map.get(self._method, Image.Resampling.LANCZOS)

        # Resize
        resized = img.resize((target_w, target_h), resample=resample)
        result_array = np.array(resized)

        # Convert back to original dtype
        if original_dtype in (np.float32, np.float64):
            result_array = result_array.astype(np.float32) / 255.0

        image_data.array = result_array
        image_data.metadata['resized'] = True
        image_data.metadata['original_size'] = original_shape[:2]
        image_data.metadata['new_size'] = result_array.shape[:2]

        return image_data

    async def process(self, item: ImageData) -> ImageData:
        """Resize image.

        Args:
            item: Image data

        Returns:
            Resized ImageData
        """
        loop = asyncio.get_event_loop()

        if self._worker_pool:
            return await self._worker_pool.run_cpu(
                self._resize_sync, item
            )
        else:
            return await loop.run_in_executor(
                None, self._resize_sync, item
            )


class DenoiseStage(AsyncStage[ImageData, ImageData]):
    """Async stage for depth-aware denoising.

    Uses depth information to apply adaptive denoising that
    preserves edges and detail in foreground objects.

    Example:
        >>> stage = DenoiseStage(strength=0.5, use_depth=True)
        >>> result = await stage(image_data)
    """

    def __init__(
        self,
        strength: float = 0.5,
        use_depth: bool = True,
        max_concurrent: int = 2,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize denoise stage.

        Args:
            strength: Denoising strength (0.0-1.0)
            use_depth: Use depth map for adaptive denoising
            max_concurrent: Maximum concurrent processes
            worker_pool: Shared worker pool
        """
        super().__init__(
            name="denoise",
            device=DeviceType.CPU,
            max_concurrent=max_concurrent,
            required=False
        )
        self._strength = strength
        self._use_depth = use_depth
        self._worker_pool = worker_pool

    def _denoise_sync(self, image_data: ImageData) -> ImageData:
        """Synchronous denoising."""
        import numpy as np

        array = image_data.array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        # Base sigma for gaussian blur
        base_sigma = self._strength * 2.0

        if self._use_depth and image_data.depth_map is not None:
            array = self._apply_depth_adaptive_denoise(array, image_data.depth_map, base_sigma)
        else:
            array = self._apply_simple_denoise(array, base_sigma)

        array = np.clip(array, 0, 1)
        image_data.array = array
        image_data.metadata['denoised'] = True
        image_data.metadata['denoise_strength'] = self._strength

        return image_data

    def _apply_depth_adaptive_denoise(self, array, depth_map, base_sigma):
        """Apply depth-adaptive denoising with more blur in background."""
        import numpy as np

        depth = depth_map
        if depth.max() > 1.0:
            depth = depth / depth.max()

        # Create per-pixel sigma map (more blur for distant pixels)
        sigma_map = base_sigma * (1 + depth * 2)

        # Apply spatially varying blur (simplified - use uniform regions)
        denoised = np.zeros_like(array)
        for sigma_level in [0.5, 1.0, 1.5, 2.0]:
            mask = (sigma_map >= sigma_level - 0.25) & (sigma_map < sigma_level + 0.25)
            if not mask.any():
                continue
            sigma = base_sigma * sigma_level
            self._apply_blur_to_masked_region(array, denoised, mask, sigma)

        # Fill any remaining pixels
        remaining = denoised.sum(axis=-1 if len(denoised.shape) == 3 else None) == 0
        if remaining.any():
            self._fill_remaining_pixels(array, denoised, remaining)

        return denoised

    def _apply_blur_to_masked_region(self, array, denoised, mask, sigma):
        """Apply gaussian blur to masked region of array."""
        from scipy.ndimage import gaussian_filter

        if len(array.shape) == 3:
            for c in range(array.shape[2]):
                blurred = gaussian_filter(array[..., c], sigma=sigma)
                effective_mask = mask if len(mask.shape) == 2 else mask[..., 0]
                denoised[..., c][effective_mask] = blurred[effective_mask]
        else:
            blurred = gaussian_filter(array, sigma=sigma)
            denoised[mask] = blurred[mask]

    def _fill_remaining_pixels(self, array, denoised, remaining):
        """Fill remaining unprocessed pixels with original values."""
        if len(array.shape) == 3:
            for c in range(array.shape[2]):
                denoised[..., c][remaining] = array[..., c][remaining]
        else:
            denoised[remaining] = array[remaining]

    def _apply_simple_denoise(self, array, base_sigma):
        """Apply simple gaussian denoise without depth awareness."""
        from scipy.ndimage import gaussian_filter

        if len(array.shape) == 3:
            for c in range(array.shape[2]):
                array[..., c] = gaussian_filter(array[..., c], sigma=base_sigma)
        else:
            array = gaussian_filter(array, sigma=base_sigma)
        return array

    async def process(self, item: ImageData) -> ImageData:
        """Apply denoising.

        Args:
            item: Image data

        Returns:
            Denoised ImageData
        """
        loop = asyncio.get_event_loop()

        if self._worker_pool:
            return await self._worker_pool.run_cpu(
                self._denoise_sync, item
            )
        else:
            return await loop.run_in_executor(
                None, self._denoise_sync, item
            )


# Factory function to create common pipeline configurations
def create_luxury_pipeline_stages(
    output_dir: Union[str, Path],
    enable_depth: bool = True,
    enable_material: bool = True,
    enable_color_grading: bool = True,
    lut_path: Optional[Union[str, Path]] = None
) -> List[AsyncStage]:
    """Create stages for luxury image processing pipeline.

    Args:
        output_dir: Output directory for processed images
        enable_depth: Enable depth estimation
        enable_material: Enable material response
        enable_color_grading: Enable color grading
        lut_path: Optional LUT file path

    Returns:
        List of configured AsyncStage instances
    """
    stages: List[AsyncStage] = [
        ImageLoadStage(max_concurrent=4),
    ]

    if enable_depth:
        stages.append(DepthEstimationStage(
            device=DeviceType.AUTO,
            max_concurrent=1
        ))

    if enable_material:
        stages.append(MaterialResponseStage(
            use_depth=enable_depth,
            max_concurrent=2
        ))

    if enable_color_grading:
        stages.append(ColorGradingStage(
            lut_path=lut_path,
            max_concurrent=4
        ))

    stages.append(ImageSaveStage(
        output_dir=output_dir,
        output_format="TIFF",
        max_concurrent=4
    ))

    return stages


# Export public API
__all__ = [
    'ImageData',
    'ImageLoadStage',
    'ImageSaveStage',
    'DepthEstimationStage',
    'MaterialResponseStage',
    'ColorGradingStage',
    'ResizeStage',
    'DenoiseStage',
    'create_luxury_pipeline_stages',
]
