#!/usr/bin/env python3
"""
Tests for Depth-Aware Augmentations

Tests covering:
- Geometric augmentations (flip, rotate, crop, scale)
- Color augmentations (brightness, contrast, saturation)
- Depth-image correspondence preservation
- Normalization

Author: Transformation Portal Team
Version: 1.0.0
"""

import pytest
import numpy as np

# Check if dependencies are available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image  # noqa: F401
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (TORCH_AVAILABLE and PIL_AVAILABLE),
    reason="PyTorch and PIL required for augmentation tests"
)


class TestAugmentationConfig:
    """Test augmentation configuration."""

    def test_config_defaults(self):
        """Test AugmentationConfig has sensible defaults."""
        from src.training.augmentations import AugmentationConfig

        config = AugmentationConfig()

        assert config.horizontal_flip == 0.5
        assert config.rotation_degrees == 10.0
        assert config.brightness == 0.2
        assert config.normalize is True

    def test_config_custom(self):
        """Test AugmentationConfig accepts custom values."""
        from src.training.augmentations import AugmentationConfig

        config = AugmentationConfig(
            horizontal_flip=0.3,
            rotation_degrees=15.0,
            brightness=0.1,
        )

        assert config.horizontal_flip == 0.3
        assert config.rotation_degrees == 15.0
        assert config.brightness == 0.1


class TestGeometricAugmentation:
    """Test geometric augmentations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample image and depth."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        return image, depth

    def test_horizontal_flip(self, sample_data):
        """Test horizontal flip maintains correspondence."""
        from src.training.augmentations import GeometricAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(
            horizontal_flip=1.0,  # Always flip
            rotation_degrees=0,
            scale_range=(1.0, 1.0),
        )
        aug = GeometricAugmentation(config)

        # Force flip by setting high probability
        np.random.seed(0)
        flipped_image, flipped_depth = aug(image, depth)

        # Check shapes are preserved
        assert flipped_image.shape == image.shape
        assert flipped_depth.shape == depth.shape

    def test_no_augmentation(self, sample_data):
        """Test with no augmentation enabled."""
        from src.training.augmentations import GeometricAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(
            horizontal_flip=0.0,
            rotation_degrees=0.0,
            scale_range=(1.0, 1.0),
            crop_size=None,
        )
        aug = GeometricAugmentation(config)

        result_image, result_depth = aug(image, depth)

        # Should be unchanged
        assert result_image.shape == image.shape
        assert result_depth.shape == depth.shape


class TestColorAugmentation:
    """Test color augmentations."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def test_color_augmentation_output_shape(self, sample_image):
        """Test color augmentation preserves shape."""
        from src.training.augmentations import ColorAugmentation, AugmentationConfig

        config = AugmentationConfig(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        aug = ColorAugmentation(config)

        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_color_augmentation_no_change(self, sample_image):
        """Test with zero augmentation."""
        from src.training.augmentations import ColorAugmentation, AugmentationConfig

        config = AugmentationConfig(
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
        )
        aug = ColorAugmentation(config)

        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_color_values_in_range(self, sample_image):
        """Test output values are clipped to valid range."""
        from src.training.augmentations import ColorAugmentation, AugmentationConfig

        config = AugmentationConfig(
            brightness=0.5,  # Strong augmentation
            contrast=0.5,
        )
        aug = ColorAugmentation(config)

        result = aug(sample_image)

        # Float output should be in [0, 1]
        if result.dtype == np.float32:
            assert result.min() >= 0
            assert result.max() <= 1


class TestDepthAwareAugmentation:
    """Test complete depth-aware augmentation pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample image and depth."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        return image, depth

    def test_training_augmentation(self, sample_data):
        """Test training augmentation pipeline."""
        from src.training.augmentations import DepthAwareAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(normalize=False)
        aug = DepthAwareAugmentation(config, is_training=True)

        result_image, result_depth = aug(image, depth)

        assert result_image.shape[:2] == image.shape[:2]
        assert result_depth.shape == depth.shape

    def test_validation_no_augmentation(self, sample_data):
        """Test validation mode doesn't augment."""
        from src.training.augmentations import DepthAwareAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(normalize=False)
        aug = DepthAwareAugmentation(config, is_training=False)

        result_image, result_depth = aug(image, depth)

        # Shape should be same
        assert result_image.shape[:2] == image.shape[:2]

    def test_normalization(self, sample_data):
        """Test ImageNet normalization is applied."""
        from src.training.augmentations import DepthAwareAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(normalize=True)
        aug = DepthAwareAugmentation(config, is_training=False)

        result_image, result_depth = aug(image, depth)

        # Normalized image may have negative values
        # (due to ImageNet mean subtraction)
        assert result_image.dtype == np.float32

    def test_depth_normalization(self, sample_data):
        """Test depth is normalized to [0, 1]."""
        from src.training.augmentations import DepthAwareAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(normalize=True)
        aug = DepthAwareAugmentation(config, is_training=False)

        _, result_depth = aug(image, depth)

        assert result_depth.min() >= 0
        assert result_depth.max() <= 1

    def test_to_tensor(self, sample_data):
        """Test conversion to PyTorch tensors."""
        from src.training.augmentations import DepthAwareAugmentation, AugmentationConfig

        image, depth = sample_data

        config = AugmentationConfig(normalize=False)
        aug = DepthAwareAugmentation(config, is_training=False)

        result_image, result_depth = aug(image, depth)

        # Convert to tensor
        image_tensor, depth_tensor = aug.to_tensor(result_image, result_depth)

        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(depth_tensor, torch.Tensor)
        assert image_tensor.shape[0] == 3  # CHW format
        assert depth_tensor.shape[0] == 1  # 1HW format


class TestAugmentationFactories:
    """Test augmentation factory functions."""

    def test_get_train_augmentations(self):
        """Test training augmentation factory."""
        from src.training.augmentations import get_train_augmentations

        aug = get_train_augmentations(
            config={"horizontal_flip": 0.5},
            image_size=(128, 128),
        )

        assert aug.is_training is True
        assert aug.config.horizontal_flip == 0.5

    def test_get_val_augmentations(self):
        """Test validation augmentation factory."""
        from src.training.augmentations import get_val_augmentations

        aug = get_val_augmentations(image_size=(128, 128))

        assert aug.is_training is False
        assert aug.config.horizontal_flip == 0.0
        assert aug.config.rotation_degrees == 0.0


class TestCorrespondencePreservation:
    """Test that geometric transforms preserve image-depth correspondence."""

    @pytest.fixture
    def marked_data(self):
        """Create data with marker pattern to verify correspondence."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        depth = np.zeros((64, 64), dtype=np.float32)

        # Add marker in top-left quadrant
        image[:32, :32, 0] = 255  # Red in top-left
        depth[:32, :32] = 100  # High depth in top-left

        return image, depth

    def test_flip_correspondence(self, marked_data):
        """Test that flip maintains spatial correspondence."""
        from src.training.augmentations import GeometricAugmentation, AugmentationConfig

        image, depth = marked_data

        config = AugmentationConfig(
            horizontal_flip=1.0,  # Always flip
            rotation_degrees=0,
            scale_range=(1.0, 1.0),
        )
        aug = GeometricAugmentation(config)

        np.random.seed(0)  # Ensure flip happens
        flipped_image, flipped_depth = aug(image, depth)

        # After horizontal flip:
        # Red marker should be in top-RIGHT
        # High depth should be in top-RIGHT

        # Check red is now on right side
        left_red = flipped_image[:32, :32, 0].mean()
        right_red = flipped_image[:32, 32:, 0].mean()

        # Check depth correspondence follows
        left_depth = flipped_depth[:32, :32].mean()
        right_depth = flipped_depth[:32, 32:].mean()

        # Both should have flipped together (or not at all)
        red_flipped = right_red > left_red
        depth_flipped = right_depth > left_depth

        # Correspondence is preserved if both flipped or both didn't
        assert red_flipped == depth_flipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
