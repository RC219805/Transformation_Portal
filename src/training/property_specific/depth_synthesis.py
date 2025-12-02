#!/usr/bin/env python3
"""
Depth Synthesis Pipeline for Property-Specific Training.

This module provides high-quality depth map synthesis using Depth Anything V2
Large model ensemble, optimized for architectural renderings.

Features:
- Multi-model ensemble for robust depth estimation
- Architectural priors for improved accuracy
- High-quality pseudo ground truth generation
- 16-bit PNG and float32 TIFF export formats

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import logging

import numpy as np
from PIL import Image

# Optional scipy for image processing
try:
    from scipy.ndimage import gaussian_filter, uniform_filter, convolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_filter = None
    uniform_filter = None
    convolve = None

# Optional imports for ML
try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any

logger = logging.getLogger(__name__)


class DepthModelVariant(Enum):
    """Available Depth Anything V2 model variants."""
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


class DepthBackend(Enum):
    """Available compute backends."""
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_MPS = "pytorch_mps"
    PYTORCH_CUDA = "pytorch_cuda"
    COREML = "coreml"
    ONNX = "onnx"


@dataclass
class DepthSynthesisConfig:
    """Configuration for depth synthesis pipeline."""
    # Model configuration
    primary_model: DepthModelVariant = DepthModelVariant.LARGE
    ensemble_models: List[DepthModelVariant] = field(
        default_factory=lambda: [DepthModelVariant.LARGE, DepthModelVariant.BASE]
    )
    use_ensemble: bool = True
    ensemble_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])

    # Backend configuration
    backend: DepthBackend = DepthBackend.PYTORCH_MPS
    precision: str = "fp16"
    batch_size: int = 1

    # Processing configuration
    target_size: Tuple[int, int] = (518, 518)  # Optimal for Depth Anything V2
    normalize_output: bool = True
    apply_architectural_priors: bool = True
    edge_enhancement: float = 0.15

    # Output configuration
    output_16bit_png: bool = True
    output_float32_tiff: bool = True
    colorize_depth: bool = True
    output_dir: Optional[Path] = None

    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_model": self.primary_model.value,
            "ensemble_models": [m.value for m in self.ensemble_models],
            "use_ensemble": self.use_ensemble,
            "ensemble_weights": self.ensemble_weights,
            "backend": self.backend.value,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "target_size": list(self.target_size),
            "normalize_output": self.normalize_output,
            "apply_architectural_priors": self.apply_architectural_priors,
            "edge_enhancement": self.edge_enhancement,
            "output_16bit_png": self.output_16bit_png,
            "output_float32_tiff": self.output_float32_tiff,
        }


@dataclass
class SynthesizedDepth:
    """Result of depth synthesis for a single image."""
    source_path: Path = field(default_factory=Path)
    depth_map: Optional[np.ndarray] = None  # (H, W) float32, 0=near, 1=far
    confidence_map: Optional[np.ndarray] = None  # (H, W) float32
    edge_map: Optional[np.ndarray] = None  # (H, W) edge strength
    resolution: Tuple[int, int] = (0, 0)
    min_depth: float = 0.0
    max_depth: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_16bit_png(self) -> np.ndarray:
        """Convert depth to 16-bit PNG format."""
        if self.depth_map is None:
            raise ValueError("No depth map available")

        depth_normalized = (self.depth_map - self.depth_map.min()) / (
            self.depth_map.max() - self.depth_map.min() + 1e-8
        )
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        return depth_16bit

    def to_float32_tiff(self) -> np.ndarray:
        """Get depth as float32 for TIFF export."""
        if self.depth_map is None:
            raise ValueError("No depth map available")
        return self.depth_map.astype(np.float32)

    def to_colorized(self, colormap: str = "viridis") -> np.ndarray:
        """Convert depth to colorized RGB image."""
        if self.depth_map is None:
            raise ValueError("No depth map available")

        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
        except ImportError:
            # Fallback to simple grayscale-to-color
            return self._simple_colorize()

        depth_normalized = (self.depth_map - self.depth_map.min()) / (
            self.depth_map.max() - self.depth_map.min() + 1e-8
        )
        colored = cmap(depth_normalized)[:, :, :3]
        return (colored * 255).astype(np.uint8)

    def _simple_colorize(self) -> np.ndarray:
        """Simple colorization without matplotlib."""
        depth_normalized = (self.depth_map - self.depth_map.min()) / (
            self.depth_map.max() - self.depth_map.min() + 1e-8
        )
        # Blue (near) to Yellow (far)
        r = (depth_normalized * 255).astype(np.uint8)
        g = (depth_normalized * 255).astype(np.uint8)
        b = ((1 - depth_normalized) * 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    def save(
        self,
        output_dir: Path,
        prefix: str = "",
        save_16bit: bool = True,
        save_float32: bool = True,
        save_colorized: bool = True
    ) -> Dict[str, Path]:
        """Save depth maps in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        stem = self.source_path.stem if self.source_path else "depth"
        if prefix:
            stem = f"{prefix}_{stem}"

        if save_16bit:
            png_path = output_dir / f"{stem}_depth_16bit.png"
            Image.fromarray(self.to_16bit_png()).save(png_path)
            saved_paths["16bit_png"] = png_path

        if save_float32:
            try:
                import tifffile
                tiff_path = output_dir / f"{stem}_depth_float32.tiff"
                tifffile.imwrite(tiff_path, self.to_float32_tiff())
                saved_paths["float32_tiff"] = tiff_path
            except ImportError:
                logger.warning("tifffile not available, skipping float32 TIFF export")

        if save_colorized:
            color_path = output_dir / f"{stem}_depth_colorized.png"
            Image.fromarray(self.to_colorized()).save(color_path)
            saved_paths["colorized"] = color_path

        return saved_paths


class DepthSynthesis:
    """
    Depth synthesis pipeline using Depth Anything V2 Large model ensemble.

    Generates high-quality pseudo ground truth depth maps for property-specific
    training, with architectural priors for improved accuracy on luxury real
    estate imagery.

    Attributes:
        config: Depth synthesis configuration
        models: Loaded depth estimation models
        device: Compute device (CPU/MPS/CUDA)
    """

    def __init__(self, config: Optional[DepthSynthesisConfig] = None):
        """
        Initialize depth synthesis pipeline.

        Args:
            config: Configuration for depth synthesis
        """
        self.config = config or DepthSynthesisConfig()
        self.models: Dict[DepthModelVariant, Any] = {}
        self.device = self._get_device()
        self._initialized = False

    def _get_device(self) -> str:
        """Determine compute device based on configuration."""
        if not TORCH_AVAILABLE:
            return "cpu"

        if self.config.backend == DepthBackend.PYTORCH_CUDA:
            if torch.cuda.is_available():
                return "cuda"
        elif self.config.backend == DepthBackend.PYTORCH_MPS:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"

        return "cpu"

    def initialize(self) -> None:
        """Initialize and load depth models."""
        if self._initialized:
            return

        logger.info("Initializing depth synthesis pipeline...")
        logger.info(f"  Backend: {self.config.backend.value}")
        logger.info(f"  Device: {self.device}")

        # Load models based on configuration
        models_to_load = (
            self.config.ensemble_models if self.config.use_ensemble
            else [self.config.primary_model]
        )

        for variant in models_to_load:
            try:
                self.models[variant] = self._load_model(variant)
                logger.info(f"  ✓ Loaded {variant.value} model")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {variant.value}: {e}")

        if not self.models:
            logger.warning("No depth models loaded. Using fallback estimation.")

        self._initialized = True

    def _load_model(self, variant: DepthModelVariant) -> Any:
        """Load a specific depth model variant."""
        # Try to load from transformation_portal depth infrastructure
        try:
            from transformation_portal.depth.models.depth_anything_v2 import (
                DepthAnythingV2Model,
                ModelVariant as DAModelVariant,
                ModelBackend as DABackend
            )

            variant_map = {
                DepthModelVariant.SMALL: DAModelVariant.SMALL,
                DepthModelVariant.BASE: DAModelVariant.BASE,
                DepthModelVariant.LARGE: DAModelVariant.LARGE,
            }

            backend_map = {
                DepthBackend.PYTORCH_CPU: DABackend.PYTORCH_CPU,
                DepthBackend.PYTORCH_MPS: DABackend.PYTORCH_MPS,
            }

            return DepthAnythingV2Model(
                variant=variant_map.get(variant, DAModelVariant.SMALL),
                backend=backend_map.get(self.config.backend, DABackend.PYTORCH_CPU),
                precision=self.config.precision
            )

        except ImportError:
            # Fallback: try loading from transformers
            try:
                from transformers import pipeline
                model_names = {
                    DepthModelVariant.SMALL: "depth-anything/Depth-Anything-V2-Small-hf",
                    DepthModelVariant.BASE: "depth-anything/Depth-Anything-V2-Base-hf",
                    DepthModelVariant.LARGE: "depth-anything/Depth-Anything-V2-Large-hf",
                }
                return pipeline(
                    "depth-estimation",
                    model=model_names[variant],
                    device=self.device if self.device != "mps" else -1
                )
            except Exception as e:
                logger.warning(f"Could not load model {variant.value}: {e}")
                return None

    def synthesize(
        self,
        image: Union[Path, Image.Image, np.ndarray],
        apply_priors: bool = True
    ) -> SynthesizedDepth:
        """
        Synthesize depth map for a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            apply_priors: Whether to apply architectural priors

        Returns:
            SynthesizedDepth with depth map and metadata
        """
        if not self._initialized:
            self.initialize()

        # Load image
        source_path = image if isinstance(image, Path) else Path("input")
        pil_image = self._load_image(image)
        original_size = pil_image.size

        # Estimate depth using model(s)
        if self.models:
            depth_map = self._estimate_depth_ensemble(pil_image)
        else:
            depth_map = self._fallback_depth_estimation(pil_image)

        # Resize to original size
        if depth_map.shape != (original_size[1], original_size[0]):
            depth_map = self._resize_depth(depth_map, original_size)

        # Apply architectural priors if requested
        if apply_priors and self.config.apply_architectural_priors:
            depth_map = self._apply_architectural_priors(depth_map, pil_image)

        # Apply edge enhancement
        if self.config.edge_enhancement > 0:
            edge_map = self._compute_edges(pil_image)
            depth_map = self._enhance_depth_edges(depth_map, edge_map)
        else:
            edge_map = None

        # Compute confidence map
        confidence_map = self._compute_confidence(depth_map)

        return SynthesizedDepth(
            source_path=source_path if isinstance(source_path, Path) else Path(str(source_path)),
            depth_map=depth_map,
            confidence_map=confidence_map,
            edge_map=edge_map,
            resolution=(original_size[0], original_size[1]),
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            metadata={
                "models_used": [m.value for m in self.models.keys()],
                "ensemble": self.config.use_ensemble,
                "architectural_priors": apply_priors,
                "edge_enhancement": self.config.edge_enhancement,
            }
        )

    def synthesize_all(
        self,
        images: List[Union[Path, Image.Image]],
        output_dir: Optional[Path] = None
    ) -> List[SynthesizedDepth]:
        """
        Synthesize depth maps for multiple images.

        Args:
            images: List of input images
            output_dir: Optional directory to save results

        Returns:
            List of SynthesizedDepth results
        """
        if not self._initialized:
            self.initialize()

        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i + 1}/{len(images)}")
            try:
                result = self.synthesize(image)
                results.append(result)

                if output_dir:
                    result.save(output_dir)

            except Exception as e:
                logger.error(f"Failed to process image {i + 1}: {e}")

        return results

    def _load_image(self, image: Union[Path, Image.Image, np.ndarray]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image, Path):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _estimate_depth_ensemble(self, image: Image.Image) -> np.ndarray:
        """Estimate depth using model ensemble."""
        depths = []
        weights = []

        # Create variant to weight mapping
        variant_weights = {
            DepthModelVariant.LARGE: self.config.ensemble_weights[0] if len(
                self.config.ensemble_weights) > 0 else 0.7,
            DepthModelVariant.BASE: self.config.ensemble_weights[1] if len(
                self.config.ensemble_weights) > 1 else 0.3,
            DepthModelVariant.SMALL: self.config.ensemble_weights[2] if len(
                self.config.ensemble_weights) > 2 else 0.2,
        }

        for variant, model in self.models.items():
            try:
                depth = self._estimate_single_model(model, image)
                depths.append(depth)

                # Get weight for this model using dictionary lookup
                weight = variant_weights.get(variant, 1.0)
                weights.append(weight)

            except Exception as e:
                logger.warning(f"Failed to estimate with {variant.value}: {e}")

        if not depths:
            return self._fallback_depth_estimation(image)

        # Weighted average of depth maps
        weights = np.array(weights) / sum(weights)
        ensemble_depth = np.zeros_like(depths[0])
        for depth, weight in zip(depths, weights):
            # Resize if needed
            if depth.shape != ensemble_depth.shape:
                depth = self._resize_depth(depth, ensemble_depth.shape[::-1])
            ensemble_depth += depth * weight

        return ensemble_depth

    def _estimate_single_model(self, model: Any, image: Image.Image) -> np.ndarray:
        """Estimate depth using a single model."""
        # Check if it's a transformation_portal model
        if hasattr(model, "estimate_depth"):
            result = model.estimate_depth(image)
            return result["depth"]

        # Check if it's a HuggingFace pipeline
        if hasattr(model, "__call__"):
            result = model(image)
            depth = np.array(result["depth"])
            # Normalize to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth

        raise ValueError(f"Unknown model type: {type(model)}")

    def _fallback_depth_estimation(self, image: Image.Image) -> np.ndarray:
        """Fallback depth estimation using image gradients."""
        logger.info("Using fallback depth estimation (gradient-based)")

        img_array = np.array(image).astype(np.float32)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Simple depth from intensity (darker = farther)
        depth = 1.0 - gray

        # Apply smoothing
        if SCIPY_AVAILABLE and gaussian_filter is not None:
            depth = gaussian_filter(depth, sigma=3)
        else:
            # Simple box blur fallback
            depth = self._simple_blur(depth, kernel_size=5)

        return depth.astype(np.float32)

    def _simple_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Simple box blur fallback when scipy is not available."""
        from PIL import ImageFilter
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        blurred = pil_img.filter(ImageFilter.BoxBlur(kernel_size // 2))
        return np.array(blurred).astype(np.float32) / 255.0

    def _resize_depth(
        self,
        depth: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize depth map to target size."""
        pil_depth = Image.fromarray(depth)
        pil_depth = pil_depth.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(pil_depth)

    def _apply_architectural_priors(
        self,
        depth: np.ndarray,
        image: Image.Image
    ) -> np.ndarray:
        """Apply architectural priors to improve depth estimation."""
        h, w = depth.shape

        # Prior 1: Vertical gradient (floors closer than ceilings)
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        y_coords = np.broadcast_to(y_coords, (h, w))

        # Prior 2: Vignette (edges often farther in interior shots)
        x_coords = np.linspace(-1, 1, w)[np.newaxis, :]
        y_norm = np.linspace(-1, 1, h)[:, np.newaxis]
        vignette = np.sqrt(x_coords ** 2 + y_norm ** 2) / np.sqrt(2)

        # Combine priors with learned depth
        # Weight: 80% learned, 10% vertical, 10% vignette
        depth_prior = depth * 0.8 + y_coords * 0.1 + (1 - vignette) * 0.1

        # Renormalize
        depth_prior = (depth_prior - depth_prior.min()) / (
            depth_prior.max() - depth_prior.min() + 1e-8
        )

        return depth_prior.astype(np.float32)

    def _compute_edges(self, image: Image.Image) -> np.ndarray:
        """Compute edge map from image."""
        img_array = np.array(image).astype(np.float32)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Sobel edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        if SCIPY_AVAILABLE and convolve is not None:
            gx = convolve(gray, sobel_x)
            gy = convolve(gray, sobel_y)
        else:
            # Simple numpy-based convolution fallback
            gx = self._simple_convolve(gray, sobel_x)
            gy = self._simple_convolve(gray, sobel_y)

        edges = np.sqrt(gx ** 2 + gy ** 2)
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        return edges.astype(np.float32)

    def _simple_convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple convolution fallback when scipy is not available."""
        kh, kw = kernel.shape
        h, w = image.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        # Simple loop-based convolution for small kernels
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                patch = padded[i:i + kh, j:j + kw]
                result[i, j] = np.sum(patch * kernel)

        return result

    def _enhance_depth_edges(
        self,
        depth: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """Enhance depth discontinuities at detected edges."""
        strength = self.config.edge_enhancement

        # Increase depth contrast at edges
        enhanced = depth.copy()

        # Apply edge-aware sharpening
        depth_gradient_x = np.gradient(depth, axis=1)
        depth_gradient_y = np.gradient(depth, axis=0)

        # Enhance gradients at edge locations
        enhanced += strength * edges * depth_gradient_x
        enhanced += strength * edges * depth_gradient_y

        # Renormalize
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)

        return enhanced.astype(np.float32)

    def _compute_confidence(self, depth: np.ndarray) -> np.ndarray:
        """Compute confidence map for depth estimates."""
        # Use local variance as confidence metric
        if SCIPY_AVAILABLE and uniform_filter is not None:
            # Compute local mean and variance
            local_mean = uniform_filter(depth, size=5)
            local_sq_mean = uniform_filter(depth ** 2, size=5)
            local_var = local_sq_mean - local_mean ** 2
        else:
            # Simple fallback: use gradient magnitude as inverse confidence
            gx = np.gradient(depth, axis=1)
            gy = np.gradient(depth, axis=0)
            local_var = gx ** 2 + gy ** 2

        # Invert: low variance = high confidence
        confidence = 1.0 - np.clip(local_var * 10, 0, 1)

        return confidence.astype(np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "models_loaded": [m.value for m in self.models.keys()],
            "config": self.config.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"DepthSynthesis(backend={self.config.backend.value}, "
            f"models={len(self.models)}, device={self.device})"
        )
