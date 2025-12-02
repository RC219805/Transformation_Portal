#!/usr/bin/env python3
"""
Training Dataset Generator for Property-Specific Models.

This module generates augmented training datasets from property images,
creating multi-scale crops with depth-image correspondence and
material-aware augmentation.

Features:
- 600+ augmented training samples from 6 property images
- Multi-scale crops (512, 1024, 2048)
- Depth-image correspondence maintenance
- Material-aware augmentation strategies
- Training/validation split generation

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging
import random

import numpy as np
from PIL import Image, ImageFilter

# Optional scipy for advanced processing
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_filter = None

from .picacho_analyzer import (
    PicachoAnalyzer,
    ImageAnalysis
)
from .depth_synthesis import DepthSynthesis, SynthesizedDepth

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of augmentation strategies."""
    GEOMETRIC = "geometric"
    COLOR = "color"
    MATERIAL = "material"
    LIGHTING = "lighting"
    QUALITY = "quality"


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("data/training_750picacho"))
    dataset_name: str = "750_picacho_lane_training"

    # Sample generation
    total_samples: int = 600
    samples_per_image: int = 100
    random_seed: int = 42

    # Multi-scale configuration
    crop_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    crop_size_weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])

    # Augmentation configuration
    augmentation_enabled: bool = True
    augmentation_types: List[AugmentationType] = field(
        default_factory=lambda: list(AugmentationType)
    )

    # Geometric augmentations
    horizontal_flip_prob: float = 0.5
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    scale_range: Tuple[float, float] = (0.8, 1.2)

    # Color augmentations
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    saturation_range: Tuple[float, float] = (0.85, 1.15)
    hue_shift_range: Tuple[float, float] = (-0.05, 0.05)

    # Quality augmentations
    noise_sigma_range: Tuple[float, float] = (0.0, 0.02)
    blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    jpeg_quality_range: Tuple[int, int] = (75, 100)

    # Depth configuration
    include_depth: bool = True
    depth_noise_sigma: float = 0.01

    # Training split
    validation_split: float = 0.1
    test_split: float = 0.05

    # Metadata
    include_metadata: bool = True
    save_augmentation_params: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "samples_per_image": self.samples_per_image,
            "crop_sizes": self.crop_sizes,
            "augmentation_enabled": self.augmentation_enabled,
            "include_depth": self.include_depth,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
        }


@dataclass
class TrainingSample:
    """A single training sample with image and optional depth."""
    sample_id: str = ""
    image: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    source_image: str = ""
    room_type: str = ""
    crop_size: int = 512
    crop_position: Tuple[int, int] = (0, 0)
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    materials: List[str] = field(default_factory=list)

    def save(self, output_dir: Path, split: str = "train") -> Dict[str, Path]:
        """Save sample to disk."""
        output_dir = Path(output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save image
        if self.image is not None:
            img_path = output_dir / f"{self.sample_id}.png"
            Image.fromarray(self.image).save(img_path)
            saved_paths["image"] = img_path

        # Save depth
        if self.depth is not None:
            depth_dir = output_dir / "depth"
            depth_dir.mkdir(exist_ok=True)
            depth_path = depth_dir / f"{self.sample_id}_depth.png"
            # Save as 16-bit
            depth_16bit = (self.depth * 65535).astype(np.uint16)
            Image.fromarray(depth_16bit).save(depth_path)
            saved_paths["depth"] = depth_path

        return saved_paths

    def to_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "sample_id": self.sample_id,
            "source_image": self.source_image,
            "room_type": self.room_type,
            "crop_size": self.crop_size,
            "crop_position": list(self.crop_position),
            "augmentation_params": self.augmentation_params,
            "materials": self.materials,
        }


class DatasetGenerator:
    """
    Training dataset generator for property-specific models.

    Generates 600+ augmented training samples from 6 property images,
    with multi-scale crops and material-aware augmentation.

    Attributes:
        analyzer: Property analyzer with image analyses
        depth_synthesis: Depth synthesis pipeline
        config: Dataset generation configuration
    """

    def __init__(
        self,
        analyzer: Optional[PicachoAnalyzer] = None,
        depth_synthesis: Optional[DepthSynthesis] = None,
        config: Optional[DatasetConfig] = None
    ):
        """
        Initialize dataset generator.

        Args:
            analyzer: Property analyzer instance
            depth_synthesis: Depth synthesis pipeline
            config: Dataset generation configuration
        """
        self.analyzer = analyzer or PicachoAnalyzer()
        self.depth_synthesis = depth_synthesis or DepthSynthesis()
        self.config = config or DatasetConfig()

        # Set random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Pre-computed depth maps
        self._depth_maps: Dict[Path, SynthesizedDepth] = {}

        # Generated samples
        self.samples: List[TrainingSample] = []

    def generate_dataset(
        self,
        num_samples: Optional[int] = None
    ) -> List[TrainingSample]:
        """
        Generate complete training dataset.

        Args:
            num_samples: Override total number of samples

        Returns:
            List of generated TrainingSample objects
        """
        num_samples = num_samples or self.config.total_samples

        # Analyze property if not already done
        if not self.analyzer.analyses:
            logger.info("Running property analysis...")
            self.analyzer.analyze_property()

        # Generate depth maps
        if self.config.include_depth:
            logger.info("Generating depth maps...")
            self._generate_depth_maps()

        # Generate samples
        logger.info(f"Generating {num_samples} training samples...")
        self.samples = []

        # Calculate samples per image
        num_images = len(self.analyzer.image_paths)
        if num_images == 0:
            logger.error("No images found for dataset generation")
            return []

        samples_per_image = num_samples // num_images

        for analysis in self.analyzer.analyses:
            image_samples = self._generate_samples_from_image(
                analysis, samples_per_image
            )
            self.samples.extend(image_samples)

        # Shuffle samples
        random.shuffle(self.samples)

        logger.info(f"Generated {len(self.samples)} training samples")
        return self.samples

    def save_dataset(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Save generated dataset to disk.

        Args:
            output_dir: Output directory (uses config if not provided)

        Returns:
            Dataset statistics and metadata
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset
        train_samples, val_samples, test_samples = self._split_dataset()

        # Save each split
        stats = {
            "train": self._save_split(train_samples, output_dir, "train"),
            "val": self._save_split(val_samples, output_dir, "val"),
            "test": self._save_split(test_samples, output_dir, "test"),
        }

        # Save metadata
        metadata = self._generate_metadata(stats)
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"  Train: {stats['train']['count']} samples")
        logger.info(f"  Val: {stats['val']['count']} samples")
        logger.info(f"  Test: {stats['test']['count']} samples")

        return metadata

    def _generate_depth_maps(self) -> None:
        """Generate depth maps for all property images."""
        for image_path in self.analyzer.image_paths:
            if image_path not in self._depth_maps:
                try:
                    depth = self.depth_synthesis.synthesize(image_path)
                    self._depth_maps[image_path] = depth
                    logger.info(f"  ✓ Depth map generated: {image_path.name}")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to generate depth for {image_path.name}: {e}")

    def _generate_samples_from_image(
        self,
        analysis: ImageAnalysis,
        num_samples: int
    ) -> List[TrainingSample]:
        """Generate training samples from a single image."""
        samples = []
        image_path = analysis.image_path

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return []

        # Get depth map if available
        depth_map = None
        if self.config.include_depth and image_path in self._depth_maps:
            depth_map = self._depth_maps[image_path].depth_map

        # Get materials for this image
        materials = [m.value for m in analysis.materials.primary_materials]

        # Generate samples with different crop sizes
        for i in range(num_samples):
            sample = self._generate_single_sample(
                img_array=img_array,
                depth_map=depth_map,
                analysis=analysis,
                sample_idx=i,
                materials=materials
            )
            if sample is not None:
                samples.append(sample)

        return samples

    def _generate_single_sample(
        self,
        img_array: np.ndarray,
        depth_map: Optional[np.ndarray],
        analysis: ImageAnalysis,
        sample_idx: int,
        materials: List[str]
    ) -> Optional[TrainingSample]:
        """Generate a single training sample with augmentation."""
        h, w = img_array.shape[:2]

        # Select crop size based on weights
        crop_size = random.choices(
            self.config.crop_sizes,
            weights=self.config.crop_size_weights
        )[0]

        # Ensure crop fits in image
        if crop_size > min(h, w):
            crop_size = min(h, w)

        # Random crop position
        max_x = max(0, w - crop_size)
        max_y = max(0, h - crop_size)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Extract crop
        img_crop = img_array[y:y + crop_size, x:x + crop_size].copy()

        # Extract corresponding depth crop
        depth_crop = None
        if depth_map is not None:
            # Resize depth to match image if needed
            if depth_map.shape[:2] != (h, w):
                # Convert float32 depth to uint16 for PIL, then back
                if depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
                    depth_uint16 = (depth_map * 65535).astype(np.uint16)
                else:
                    depth_uint16 = depth_map.astype(np.uint16)
                depth_pil = Image.fromarray(depth_uint16, mode='I;16')
                depth_resized_pil = depth_pil.resize((w, h), Image.Resampling.BILINEAR)
                depth_resized = np.array(depth_resized_pil).astype(np.float32) / 65535.0
            else:
                depth_resized = depth_map.astype(np.float32)
            depth_crop = depth_resized[y:y + crop_size, x:x + crop_size].copy()

        # Apply augmentations
        augmentation_params = {}
        if self.config.augmentation_enabled:
            img_crop, depth_crop, augmentation_params = self._apply_augmentations(
                img_crop, depth_crop, materials
            )

        # Resize to standard size for training (512x512)
        target_size = 512
        if img_crop.shape[0] != target_size:
            img_pil = Image.fromarray(img_crop)
            img_crop = np.array(img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS))

            if depth_crop is not None:
                depth_pil = Image.fromarray(depth_crop)
                depth_crop = np.array(
                    depth_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
                )

        # Create sample
        sample_id = f"{analysis.image_path.stem}_{sample_idx:04d}"

        return TrainingSample(
            sample_id=sample_id,
            image=img_crop,
            depth=depth_crop,
            source_image=analysis.image_path.name,
            room_type=analysis.room_type.value,
            crop_size=crop_size,
            crop_position=(x, y),
            augmentation_params=augmentation_params,
            materials=materials,
        )

    def _apply_augmentations(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        materials: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply augmentation pipeline to image and depth."""
        params = {}

        # Geometric augmentations
        if AugmentationType.GEOMETRIC in self.config.augmentation_types:
            image, depth, geom_params = self._apply_geometric_augmentations(image, depth)
            params.update(geom_params)

        # Color augmentations
        if AugmentationType.COLOR in self.config.augmentation_types:
            image, color_params = self._apply_color_augmentations(image)
            params.update(color_params)

        # Material-aware augmentations
        if AugmentationType.MATERIAL in self.config.augmentation_types:
            image, mat_params = self._apply_material_augmentations(image, materials)
            params.update(mat_params)

        # Quality augmentations
        if AugmentationType.QUALITY in self.config.augmentation_types:
            image, qual_params = self._apply_quality_augmentations(image)
            params.update(qual_params)

        # Add noise to depth if enabled
        if depth is not None and self.config.depth_noise_sigma > 0:
            noise = np.random.randn(*depth.shape) * self.config.depth_noise_sigma
            depth = np.clip(depth + noise, 0, 1).astype(np.float32)
            params["depth_noise"] = self.config.depth_noise_sigma

        return image, depth, params

    def _apply_geometric_augmentations(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply geometric augmentations (flip, rotation, scale)."""
        params = {}

        # Horizontal flip
        if random.random() < self.config.horizontal_flip_prob:
            image = np.fliplr(image).copy()
            if depth is not None:
                depth = np.fliplr(depth).copy()
            params["horizontal_flip"] = True
        else:
            params["horizontal_flip"] = False

        # Rotation (simplified - using PIL)
        rot_min, rot_max = self.config.rotation_range
        angle = random.uniform(rot_min, rot_max)
        if abs(angle) > 1.0:
            img_pil = Image.fromarray(image)
            image = np.array(img_pil.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False))
            if depth is not None:
                # Convert depth to uint16 for PIL rotation, then back to float32
                depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 65535).astype(np.uint16)
                depth_pil = Image.fromarray(depth_normalized, mode='I;16')
                depth_rotated = np.array(depth_pil.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False))
                # Restore original depth range
                depth = (depth_rotated.astype(np.float32) / 65535) * (depth.max() - depth.min()) + depth.min()
            params["rotation"] = angle
        else:
            params["rotation"] = 0

        return image, depth, params

    def _apply_color_augmentations(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply color augmentations (brightness, contrast, saturation)."""
        params = {}
        img_float = image.astype(np.float32) / 255.0

        # Brightness
        b_min, b_max = self.config.brightness_range
        brightness = random.uniform(b_min, b_max)
        img_float = img_float * brightness
        params["brightness"] = brightness

        # Contrast
        c_min, c_max = self.config.contrast_range
        contrast = random.uniform(c_min, c_max)
        mean = img_float.mean()
        img_float = (img_float - mean) * contrast + mean
        params["contrast"] = contrast

        # Saturation
        s_min, s_max = self.config.saturation_range
        saturation = random.uniform(s_min, s_max)
        gray = np.mean(img_float, axis=2, keepdims=True)
        img_float = gray + (img_float - gray) * saturation
        params["saturation"] = saturation

        # Clip and convert back
        img_float = np.clip(img_float, 0, 1)
        image = (img_float * 255).astype(np.uint8)

        return image, params

    def _apply_material_augmentations(
        self,
        image: np.ndarray,
        materials: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply material-aware augmentations."""
        params = {"material_adjustments": {}}

        img_float = image.astype(np.float32) / 255.0

        # Material-specific adjustments
        for material in materials:
            if material == "water":
                # Enhance blue channel slightly for water
                adjustment = random.uniform(1.0, 1.08)
                img_float[:, :, 2] *= adjustment
                params["material_adjustments"]["water_blue_boost"] = adjustment

            elif material == "wood":
                # Warm up wood tones
                warm_factor = random.uniform(1.0, 1.05)
                img_float[:, :, 0] *= warm_factor  # Red
                img_float[:, :, 1] *= warm_factor * 0.95  # Green (less)
                params["material_adjustments"]["wood_warmth"] = warm_factor

            elif material == "metal":
                # Enhance contrast for metal
                local_contrast = random.uniform(1.0, 1.1)
                mean = img_float.mean()
                img_float = (img_float - mean) * local_contrast + mean
                params["material_adjustments"]["metal_contrast"] = local_contrast

            elif material == "stone":
                # Subtle texture enhancement (via micro-contrast)
                texture_boost = random.uniform(0.0, 0.05)
                noise = np.random.randn(*img_float.shape) * texture_boost
                img_float += noise
                params["material_adjustments"]["stone_texture"] = texture_boost

        # Clip and convert back
        img_float = np.clip(img_float, 0, 1)
        image = (img_float * 255).astype(np.uint8)

        return image, params

    def _apply_quality_augmentations(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply quality-related augmentations (noise, blur, compression)."""
        params = {}

        img_float = image.astype(np.float32) / 255.0

        # Gaussian noise
        n_min, n_max = self.config.noise_sigma_range
        noise_sigma = random.uniform(n_min, n_max)
        if noise_sigma > 0:
            noise = np.random.randn(*img_float.shape) * noise_sigma
            img_float += noise
            params["noise_sigma"] = noise_sigma

        # Gaussian blur
        b_min, b_max = self.config.blur_sigma_range
        blur_sigma = random.uniform(b_min, b_max)
        if blur_sigma > 0.3:
            if SCIPY_AVAILABLE and gaussian_filter is not None:
                for c in range(3):
                    img_float[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=blur_sigma)
            else:
                # Fallback to PIL blur
                img_uint8 = (img_float * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
                img_float = np.array(pil_img).astype(np.float32) / 255.0
            params["blur_sigma"] = blur_sigma

        # Clip and convert
        img_float = np.clip(img_float, 0, 1)
        image = (img_float * 255).astype(np.uint8)

        # JPEG compression simulation
        q_min, q_max = self.config.jpeg_quality_range
        jpeg_quality = random.randint(q_min, q_max)
        if jpeg_quality < 100:
            from io import BytesIO
            pil_img = Image.fromarray(image)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=jpeg_quality)
            buffer.seek(0)
            image = np.array(Image.open(buffer))
            params["jpeg_quality"] = jpeg_quality

        return image, params

    def _split_dataset(
        self
    ) -> Tuple[List[TrainingSample], List[TrainingSample], List[TrainingSample]]:
        """Split dataset into train/val/test."""
        n_samples = len(self.samples)
        n_test = int(n_samples * self.config.test_split)
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val - n_test

        # Shuffle samples
        samples = self.samples.copy()
        random.shuffle(samples)

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        return train_samples, val_samples, test_samples

    def _save_split(
        self,
        samples: List[TrainingSample],
        output_dir: Path,
        split_name: str
    ) -> Dict[str, Any]:
        """Save a dataset split to disk."""
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        images_dir = split_dir / "images"
        depth_dir = split_dir / "depth"
        images_dir.mkdir(exist_ok=True)
        if self.config.include_depth:
            depth_dir.mkdir(exist_ok=True)

        # Save samples
        sample_metadata = []
        for sample in samples:
            # Save image
            if sample.image is not None:
                img_path = images_dir / f"{sample.sample_id}.png"
                Image.fromarray(sample.image).save(img_path)

            # Save depth
            if sample.depth is not None:
                depth_path = depth_dir / f"{sample.sample_id}_depth.png"
                depth_16bit = (sample.depth * 65535).astype(np.uint16)
                Image.fromarray(depth_16bit).save(depth_path)

            sample_metadata.append(sample.to_dict())

        # Save split metadata
        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)

        return {
            "count": len(samples),
            "directory": str(split_dir),
        }

    def _generate_metadata(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dataset metadata."""
        return {
            "dataset_name": self.config.dataset_name,
            "property_name": "750 Picacho Lane",
            "project_number": "24098.00",
            "config": self.config.to_dict(),
            "splits": stats,
            "total_samples": len(self.samples),
            "source_images": len(self.analyzer.image_paths),
            "source_image_names": [p.name for p in self.analyzer.image_paths],
            "room_types": list(set(s.room_type for s in self.samples)),
            "materials_covered": list(set(
                mat for s in self.samples for mat in s.materials
            )),
            "crop_sizes_used": self.config.crop_sizes,
            "augmentation_types": [t.value for t in self.config.augmentation_types],
            "depth_included": self.config.include_depth,
        }

    def __repr__(self) -> str:
        return (
            f"DatasetGenerator(samples={len(self.samples)}, "
            f"config={self.config.dataset_name})"
        )
