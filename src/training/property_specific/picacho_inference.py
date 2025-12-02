#!/usr/bin/env python3
"""
Production Inference Pipeline for 750 Picacho Lane.

This module provides production-ready inference for processing full 4K images
with property-specific enhancements, outputting 16-bit TIFF files with
comprehensive metadata.

Features:
- Full 4K image processing (4096x3072)
- Material-specific enhancement application
- 16-bit TIFF output with metadata
- Batch processing support
- Memory-efficient tiled processing for large images

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
import logging
import time

import numpy as np
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS

# Optional scipy for advanced processing
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_filter = None

# Optional ML imports
try:
    import torch
    from torch import nn, Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None  # Type stub for when PyTorch is not available

# Optional TIFF support
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    TIFF_16BIT = "16bit_tiff"
    TIFF_32BIT = "32bit_tiff"
    PNG_16BIT = "16bit_png"
    PNG_8BIT = "8bit_png"
    JPEG_HIGH = "jpeg_high"
    JPEG_WEB = "jpeg_web"


class EnhancementLevel(Enum):
    """Enhancement intensity levels."""
    SUBTLE = "subtle"
    BALANCED = "balanced"
    STRONG = "strong"
    MAXIMUM = "maximum"


@dataclass
class InferenceConfig:
    """Configuration for production inference."""
    # Model configuration
    model_path: Path = field(default_factory=lambda: Path("weights/750_picacho/best_model.pth"))
    device: str = "auto"

    # Processing configuration
    tile_size: int = 1024
    tile_overlap: int = 128
    use_tiling: bool = True
    max_resolution: Tuple[int, int] = (4096, 4096)

    # Enhancement configuration
    enhancement_level: EnhancementLevel = EnhancementLevel.BALANCED
    apply_depth_enhancement: bool = True
    apply_material_enhancement: bool = True
    apply_color_grading: bool = True

    # Material-specific strengths
    material_strengths: Dict[str, float] = field(default_factory=lambda: {
        "stone": 0.8,
        "glass": 0.9,
        "water": 0.85,
        "wood": 0.75,
        "metal": 0.85,
        "fabric": 0.7,
    })

    # Output configuration
    output_format: OutputFormat = OutputFormat.TIFF_16BIT
    output_dir: Path = field(default_factory=lambda: Path("output/750_picacho"))
    preserve_metadata: bool = True
    add_processing_metadata: bool = True

    # Quality settings
    jpeg_quality: int = 98
    png_compression: int = 6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "tile_size": self.tile_size,
            "enhancement_level": self.enhancement_level.value,
            "output_format": self.output_format.value,
            "output_dir": str(self.output_dir),
        }


@dataclass
class EnhancedOutput:
    """Result of enhancement processing."""
    source_path: Path = field(default_factory=Path)
    output_path: Path = field(default_factory=Path)
    image: Optional[np.ndarray] = None  # (H, W, C) in 0-65535 for 16-bit
    bit_depth: int = 16
    resolution: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    enhancement_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        output_path: Optional[Path] = None,
        format: Optional[OutputFormat] = None
    ) -> Path:
        """Save enhanced image to file."""
        output_path = output_path or self.output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format = format or OutputFormat.TIFF_16BIT

        if format == OutputFormat.TIFF_16BIT:
            return self._save_tiff_16bit(output_path)
        elif format == OutputFormat.TIFF_32BIT:
            return self._save_tiff_32bit(output_path)
        elif format == OutputFormat.PNG_16BIT:
            return self._save_png_16bit(output_path)
        elif format == OutputFormat.PNG_8BIT:
            return self._save_png_8bit(output_path)
        elif format in (OutputFormat.JPEG_HIGH, OutputFormat.JPEG_WEB):
            quality = 98 if format == OutputFormat.JPEG_HIGH else 85
            return self._save_jpeg(output_path, quality)

        return self._save_tiff_16bit(output_path)

    def _save_tiff_16bit(self, output_path: Path) -> Path:
        """Save as 16-bit TIFF."""
        output_path = output_path.with_suffix(".tiff")

        if TIFFFILE_AVAILABLE:
            # Use tifffile for proper 16-bit support
            image_16bit = self._to_16bit()

            # Add metadata
            metadata = {
                "ImageDescription": json.dumps(self.metadata),
                "Software": "Transformation_Portal 750 Picacho Enhancement",
            }

            tifffile.imwrite(
                output_path,
                image_16bit,
                photometric="rgb",
                metadata=metadata,
                compression="lzw"
            )
        else:
            # Fallback to PIL (limited 16-bit support)
            image_16bit = self._to_16bit()
            # PIL doesn't support 16-bit RGB well, so we save as mode 'I;16'
            # for each channel separately or just save 8-bit
            logger.warning("tifffile not available, saving as 8-bit TIFF")
            image_8bit = (image_16bit / 256).astype(np.uint8)
            Image.fromarray(image_8bit).save(output_path)

        return output_path

    def _save_tiff_32bit(self, output_path: Path) -> Path:
        """Save as 32-bit float TIFF."""
        output_path = output_path.with_suffix(".tiff")

        if TIFFFILE_AVAILABLE:
            image_float = self.image.astype(np.float32) / 65535.0
            tifffile.imwrite(
                output_path,
                image_float,
                photometric="rgb",
                compression="lzw"
            )
        else:
            logger.warning("tifffile not available for 32-bit TIFF")
            return self._save_tiff_16bit(output_path)

        return output_path

    def _save_png_16bit(self, output_path: Path) -> Path:
        """Save as 16-bit PNG."""
        output_path = output_path.with_suffix(".png")
        image_16bit = self._to_16bit()

        # PIL supports 16-bit PNG for grayscale, for RGB we need to use a workaround
        # Using imageio or cv2 would be better, but we'll save as 8-bit for now
        logger.warning("16-bit PNG RGB not well supported, saving as 8-bit")
        image_8bit = (image_16bit / 256).astype(np.uint8)
        Image.fromarray(image_8bit).save(output_path)

        return output_path

    def _save_png_8bit(self, output_path: Path) -> Path:
        """Save as 8-bit PNG."""
        output_path = output_path.with_suffix(".png")
        image_8bit = self._to_8bit()
        Image.fromarray(image_8bit).save(output_path, optimize=True)
        return output_path

    def _save_jpeg(self, output_path: Path, quality: int = 98) -> Path:
        """Save as JPEG."""
        output_path = output_path.with_suffix(".jpg")
        image_8bit = self._to_8bit()

        pil_image = Image.fromarray(image_8bit)

        # Preserve original EXIF if available
        exif = None
        if "original_exif" in self.metadata:
            exif = self.metadata["original_exif"]

        pil_image.save(
            output_path,
            quality=quality,
            progressive=True,
            exif=exif
        )

        return output_path

    def _to_16bit(self) -> np.ndarray:
        """Convert image to 16-bit."""
        if self.image is None:
            raise ValueError("No image data available")

        if self.image.dtype == np.uint16:
            return self.image
        elif self.image.dtype == np.uint8:
            return (self.image.astype(np.uint16) * 256).astype(np.uint16)
        elif self.image.dtype == np.float32 or self.image.dtype == np.float64:
            # Assume 0-1 range
            image_clipped = np.clip(self.image, 0, 1)
            return (image_clipped * 65535).astype(np.uint16)
        else:
            return self.image.astype(np.uint16)

    def _to_8bit(self) -> np.ndarray:
        """Convert image to 8-bit."""
        if self.image is None:
            raise ValueError("No image data available")

        if self.image.dtype == np.uint8:
            return self.image
        elif self.image.dtype == np.uint16:
            return (self.image / 256).astype(np.uint8)
        elif self.image.dtype == np.float32 or self.image.dtype == np.float64:
            image_clipped = np.clip(self.image, 0, 1)
            return (image_clipped * 255).astype(np.uint8)
        else:
            return self.image.astype(np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "resolution": list(self.resolution),
            "bit_depth": self.bit_depth,
            "processing_time": self.processing_time,
            "enhancement_params": self.enhancement_params,
            "metadata": self.metadata,
        }


class PicachoInference:
    """
    Production inference pipeline for 750 Picacho Lane property.

    Processes full 4K images with property-specific enhancements and
    outputs 16-bit TIFF files suitable for professional printing and
    display.

    Attributes:
        config: Inference configuration
        model: Loaded enhancement model
        device: Compute device
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize inference pipeline.

        Args:
            config: Inference configuration
            model_path: Path to model checkpoint
        """
        self.config = config or InferenceConfig()
        if model_path:
            self.config.model_path = Path(model_path)

        self.device = self._get_device()
        self.model = None
        self._initialized = False

        logger.info(f"Initialized PicachoInference on device: {self.device}")

    def _get_device(self) -> str:
        """Determine compute device."""
        if self.config.device != "auto":
            return self.config.device

        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize(self) -> None:
        """Initialize and load model."""
        if self._initialized:
            return

        logger.info("Initializing inference pipeline...")

        if TORCH_AVAILABLE and self.config.model_path.exists():
            self._load_model()
        else:
            logger.warning("Model not found or PyTorch not available. Using fallback enhancement.")
            self.model = None

        self._initialized = True

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {self.config.model_path}")

        try:
            checkpoint = torch.load(
                self.config.model_path,
                map_location=self.device,
                weights_only=True
            )

            # Try to reconstruct model architecture
            model_state = checkpoint.get("model_state", checkpoint)

            # Check if it's a ModuleDict-style model
            if isinstance(model_state, dict) and all(
                isinstance(v, dict) for v in model_state.values()
            ):
                # Load multi-module model
                self._load_multi_module_model(model_state)
            else:
                # Load simple model
                self._load_simple_model(model_state)

            logger.info("✓ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _load_multi_module_model(self, model_state: Dict[str, Dict]) -> None:
        """Load multi-module (ModuleDict) model."""
        try:
            from enhancements.hyper_reality_enhancement import (
                CausticGenerator,
                AtmosphericSynthesizer,
                MaterialTranscendence,
                SpatialHarmonics,
                EnhancementConfig
            )

            config = EnhancementConfig()
            self.model = nn.ModuleDict({
                "caustics": CausticGenerator(config.quantum_caustics),
                "atmosphere": AtmosphericSynthesizer(config.neural_atmosphere),
                "materials": MaterialTranscendence(config.material_transcendence),
                "harmonics": SpatialHarmonics(config.spatial_harmonics),
            })

            # Load state dicts
            for name, state in model_state.items():
                if name in self.model:
                    self.model[name].load_state_dict(state)

            self.model = self.model.to(self.device)
            self.model.eval()

        except ImportError:
            logger.warning("Could not import enhancement modules")
            self.model = None

    def _load_simple_model(self, model_state: Dict) -> None:
        """Load simple enhancement model."""
        # Create simple model architecture
        self.model = self._create_simple_model()
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _create_simple_model(self) -> "nn.Module":
        """Create simple enhancement model."""
        class SimpleEnhancer(nn.Module):
            def __init__(self, in_channels: int = 3, features: int = 64):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels, features, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, features * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(features * 2, features, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features, in_channels, 3, padding=1),
                )

            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return x + output * 0.3

        return SimpleEnhancer()

    def process(
        self,
        image: Union[Path, Image.Image, np.ndarray],
        output_path: Optional[Path] = None,
        materials: Optional[List[str]] = None
    ) -> EnhancedOutput:
        """
        Process a single image with property-specific enhancements.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            output_path: Optional output path
            materials: Optional list of materials in the image

        Returns:
            EnhancedOutput with enhanced image and metadata
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Load and prepare image
        source_path = image if isinstance(image, Path) else Path("input")
        pil_image, original_metadata = self._load_image(image)
        img_array = np.array(pil_image)

        # Process image
        if self.config.use_tiling and max(img_array.shape[:2]) > self.config.tile_size:
            enhanced = self._process_tiled(img_array, materials)
        else:
            enhanced = self._process_full(img_array, materials)

        # Convert to 16-bit
        if enhanced.dtype == np.float32 or enhanced.dtype == np.float64:
            enhanced_16bit = (np.clip(enhanced, 0, 1) * 65535).astype(np.uint16)
        elif enhanced.dtype == np.uint8:
            enhanced_16bit = (enhanced.astype(np.uint16) * 256).astype(np.uint16)
        else:
            enhanced_16bit = enhanced.astype(np.uint16)

        processing_time = time.time() - start_time

        # Determine output path
        if output_path is None:
            output_path = self.config.output_dir / f"{source_path.stem}_enhanced.tiff"

        # Build metadata
        metadata = {
            "source": str(source_path),
            "processing_time_seconds": processing_time,
            "enhancement_level": self.config.enhancement_level.value,
            "materials_detected": materials or [],
            "model_path": str(self.config.model_path),
            "device": self.device,
        }
        if self.config.preserve_metadata and original_metadata:
            metadata["original_metadata"] = original_metadata

        return EnhancedOutput(
            source_path=source_path,
            output_path=output_path,
            image=enhanced_16bit,
            bit_depth=16,
            resolution=(pil_image.width, pil_image.height),
            processing_time=processing_time,
            enhancement_params={
                "level": self.config.enhancement_level.value,
                "materials": materials or [],
                "depth_enhancement": self.config.apply_depth_enhancement,
                "material_enhancement": self.config.apply_material_enhancement,
            },
            metadata=metadata,
        )

    def process_batch(
        self,
        images: List[Union[Path, Image.Image]],
        output_dir: Optional[Path] = None
    ) -> List[EnhancedOutput]:
        """
        Process multiple images.

        Args:
            images: List of input images
            output_dir: Output directory for results

        Returns:
            List of EnhancedOutput results
        """
        if not self._initialized:
            self.initialize()

        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i + 1}/{len(images)}")
            try:
                result = self.process(image)
                result.save()
                results.append(result)
                logger.info(f"  ✓ Saved: {result.output_path.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")

        return results

    def _load_image(
        self,
        image: Union[Path, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, Optional[Dict]]:
        """Load image and extract metadata."""
        metadata = None

        if isinstance(image, Path):
            pil_image = Image.open(image).convert("RGB")

            # Extract EXIF metadata
            try:
                exif = pil_image._getexif()
                if exif:
                    metadata = {TAGS.get(k, k): v for k, v in exif.items()}
            except Exception:
                pass

        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return pil_image, metadata

    def _process_full(
        self,
        image: np.ndarray,
        materials: Optional[List[str]] = None
    ) -> np.ndarray:
        """Process entire image at once."""
        if self.model is not None and TORCH_AVAILABLE:
            return self._process_with_model(image, materials)
        else:
            return self._process_fallback(image, materials)

    def _process_tiled(
        self,
        image: np.ndarray,
        materials: Optional[List[str]] = None
    ) -> np.ndarray:
        """Process image in tiles for memory efficiency."""
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap

        # Calculate number of tiles
        step = tile_size - overlap
        n_tiles_y = max(1, (h - overlap) // step + 1)
        n_tiles_x = max(1, (w - overlap) // step + 1)

        # Initialize output
        output = np.zeros_like(image, dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        # Create weight mask for blending
        weight_mask = self._create_weight_mask(tile_size)

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Calculate tile bounds
                y1 = min(ty * step, h - tile_size)
                x1 = min(tx * step, w - tile_size)
                y2 = y1 + tile_size
                x2 = x1 + tile_size

                # Extract and process tile
                tile = image[y1:y2, x1:x2]
                enhanced_tile = self._process_full(tile, materials)

                # Convert to float if needed
                if enhanced_tile.dtype != np.float32:
                    if enhanced_tile.dtype == np.uint8:
                        enhanced_tile = enhanced_tile.astype(np.float32) / 255.0
                    elif enhanced_tile.dtype == np.uint16:
                        enhanced_tile = enhanced_tile.astype(np.float32) / 65535.0

                # Blend tile into output
                tile_h, tile_w = enhanced_tile.shape[:2]
                mask = weight_mask[:tile_h, :tile_w]

                output[y1:y2, x1:x2] += enhanced_tile * mask[:, :, np.newaxis]
                weights[y1:y2, x1:x2] += mask

        # Normalize by weights
        weights = np.maximum(weights, 1e-8)
        output = output / weights[:, :, np.newaxis]

        return output

    def _create_weight_mask(self, size: int) -> np.ndarray:
        """Create weight mask for tile blending."""
        # Create smooth falloff at edges
        ramp = np.linspace(0, 1, self.config.tile_overlap)
        flat = np.ones(size - 2 * self.config.tile_overlap)
        profile = np.concatenate([ramp, flat, ramp[::-1]])

        # Make 2D
        mask = np.outer(profile, profile)
        return mask.astype(np.float32)

    def _process_with_model(
        self,
        image: np.ndarray,
        materials: Optional[List[str]] = None
    ) -> np.ndarray:
        """Process image using loaded model."""
        # Convert to tensor
        img_tensor = torch.from_numpy(image).float()
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0

        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Estimate depth
        depth = self._estimate_depth(img_tensor)

        # Forward pass
        with torch.no_grad():
            if isinstance(self.model, nn.ModuleDict):
                enhanced = self._forward_multi_module(img_tensor, depth)
            else:
                enhanced = self.model(img_tensor)

        # Convert back to numpy
        enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0, 1)

        # Apply material-specific adjustments
        if materials and self.config.apply_material_enhancement:
            enhanced = self._apply_material_enhancements(enhanced, materials)

        return enhanced

    def _forward_multi_module(self, image: Tensor, depth: Tensor) -> Tensor:
        """Forward pass through multi-module model."""
        enhanced = image

        # Compute normals from depth
        normals = self._compute_normals(depth)

        # Stage 1: Caustics
        if "caustics" in self.model:
            caustics = self.model["caustics"](enhanced, depth)
            enhanced = enhanced + caustics * 0.3

        # Stage 2: Atmosphere
        if "atmosphere" in self.model:
            enhanced = self.model["atmosphere"](enhanced)

        # Stage 3: Materials
        if "materials" in self.model:
            enhanced = self.model["materials"](enhanced)

        # Stage 4: Harmonics
        if "harmonics" in self.model:
            illumination = self.model["harmonics"](normals)
            enhanced = enhanced * (1 + illumination * 0.3)

        return enhanced

    def _estimate_depth(self, image: Tensor) -> Tensor:
        """Estimate depth from image."""
        gray = torch.mean(image, dim=1, keepdim=True)
        return 1.0 - gray

    def _compute_normals(self, depth: Tensor) -> Tensor:
        """Compute surface normals from depth."""
        import torch.nn.functional as F

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth.dtype,
            device=depth.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth.dtype,
            device=depth.device
        ).view(1, 1, 3, 3)

        dx = F.conv2d(depth, sobel_x, padding=1)
        dy = F.conv2d(depth, sobel_y, padding=1)
        dz = torch.ones_like(dx) * 0.5

        normals = torch.cat([dx, dy, dz], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals

    def _process_fallback(
        self,
        image: np.ndarray,
        materials: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fallback processing without model."""
        # Convert to float
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.astype(np.float32)

        # Apply basic enhancements
        level_strength = {
            EnhancementLevel.SUBTLE: 0.3,
            EnhancementLevel.BALANCED: 0.5,
            EnhancementLevel.STRONG: 0.7,
            EnhancementLevel.MAXIMUM: 0.9,
        }
        strength = level_strength.get(self.config.enhancement_level, 0.5)

        # Enhance contrast
        mean = img_float.mean()
        enhanced = (img_float - mean) * (1 + strength * 0.2) + mean

        # Enhance saturation
        gray = np.mean(enhanced, axis=2, keepdims=True)
        enhanced = gray + (enhanced - gray) * (1 + strength * 0.15)

        # Apply material enhancements
        if materials and self.config.apply_material_enhancement:
            enhanced = self._apply_material_enhancements(enhanced, materials)

        # Clip and return
        return np.clip(enhanced, 0, 1)

    def _apply_material_enhancements(
        self,
        image: np.ndarray,
        materials: List[str]
    ) -> np.ndarray:
        """Apply material-specific enhancements."""
        enhanced = image.copy()

        for material in materials:
            strength = self.config.material_strengths.get(material, 0.5)

            if material == "stone":
                # Enhance texture and depth
                enhanced = self._enhance_texture(enhanced, strength * 0.3)

            elif material == "glass":
                # Enhance clarity and reflections
                enhanced = self._enhance_clarity(enhanced, strength * 0.25)

            elif material == "water":
                # Enhance blue tones and reflections
                enhanced[:, :, 2] *= (1 + strength * 0.1)  # Blue channel

            elif material == "wood":
                # Enhance warmth and grain
                enhanced[:, :, 0] *= (1 + strength * 0.08)  # Red channel
                enhanced[:, :, 1] *= (1 + strength * 0.05)  # Green channel

            elif material == "metal":
                # Enhance contrast and highlights
                mean = enhanced.mean()
                enhanced = (enhanced - mean) * (1 + strength * 0.15) + mean

            elif material == "fabric":
                # Subtle texture enhancement
                enhanced = self._enhance_texture(enhanced, strength * 0.15)

        return np.clip(enhanced, 0, 1)

    def _enhance_texture(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Enhance texture using high-pass filter."""
        if SCIPY_AVAILABLE and gaussian_filter is not None:
            # High-pass filter
            low_freq = gaussian_filter(image, sigma=2)
        else:
            # Fallback: simple box blur
            low_freq = self._simple_blur(image, sigma=2)

        high_freq = image - low_freq

        # Add back enhanced high frequencies
        return image + high_freq * strength

    def _enhance_clarity(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Enhance local contrast (clarity)."""
        if SCIPY_AVAILABLE and gaussian_filter is not None:
            # Local contrast enhancement
            local_mean = gaussian_filter(image, sigma=10)
        else:
            local_mean = self._simple_blur(image, sigma=10)

        enhanced = image + (image - local_mean) * strength

        return enhanced

    def _simple_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Simple blur fallback when scipy is not available."""
        # Handle multi-channel images
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = Image.fromarray((image[:, :, c] * 255).astype(np.uint8))
                blurred = channel.filter(ImageFilter.GaussianBlur(radius=sigma))
                result[:, :, c] = np.array(blurred).astype(np.float32) / 255.0
            return result
        else:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
            return np.array(blurred).astype(np.float32) / 255.0

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return {
            "initialized": self._initialized,
            "model_loaded": self.model is not None,
            "model_path": str(self.config.model_path),
            "device": self.device,
            "config": self.config.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"PicachoInference(model_loaded={self.model is not None}, "
            f"device={self.device})"
        )
