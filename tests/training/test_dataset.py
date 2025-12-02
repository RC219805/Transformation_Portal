#!/usr/bin/env python3
"""
Tests for Depth Dataset

Tests covering:
- Dataset loading from directory structure
- Image-depth pair matching
- Different depth formats (PNG, TIFF, NPY)
- Caching functionality
- Train/val split

Author: Transformation Portal Team
Version: 1.0.0
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import numpy as np

# Check if dependencies are available
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (TORCH_AVAILABLE and PIL_AVAILABLE),
    reason="PyTorch and PIL required for dataset tests"
)


class TestDepthDataConfig:
    """Test dataset configuration."""

    def test_config_defaults(self):
        """Test DepthDataConfig has sensible defaults."""
        from src.training.depth_dataset import DepthDataConfig

        config = DepthDataConfig()

        assert config.image_size == (518, 518)
        assert config.augmentation is True
        assert config.normalize is True
        assert config.cache_images is False

    def test_config_custom(self):
        """Test DepthDataConfig accepts custom values."""
        from src.training.depth_dataset import DepthDataConfig

        config = DepthDataConfig(
            train_dir="custom/train",
            image_size=(256, 256),
            cache_images=True,
        )

        assert config.train_dir == "custom/train"
        assert config.image_size == (256, 256)
        assert config.cache_images is True


class TestArchitecturalDepthDataset:
    """Test dataset class."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data."""
        tmpdir = tempfile.mkdtemp()

        # Create train directory structure
        train_dir = Path(tmpdir) / "train"
        images_dir = train_dir / "images"
        depth_dir = train_dir / "depth"

        images_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        # Create sample images and depth maps
        for i in range(10):
            # Image
            image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / f"image_{i:03d}.png")

            # Depth (NPY format)
            depth = np.random.rand(128, 128).astype(np.float32) * 100
            np.save(depth_dir / f"image_{i:03d}.npy", depth)

        # Create val directory
        val_dir = Path(tmpdir) / "val"
        val_images = val_dir / "images"
        val_depth = val_dir / "depth"

        val_images.mkdir(parents=True)
        val_depth.mkdir(parents=True)

        for i in range(3):
            image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(image).save(val_images / f"val_{i:03d}.png")

            depth = np.random.rand(128, 128).astype(np.float32) * 100
            np.save(val_depth / f"val_{i:03d}.npy", depth)

        yield tmpdir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_dataset_creation(self, sample_data_dir):
        """Test dataset can be created."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")

        assert len(dataset) == 10

    def test_dataset_getitem(self, sample_data_dir):
        """Test dataset __getitem__ returns correct format."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")

        image, depth = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(depth, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert depth.shape == (1, 64, 64)

    def test_dataset_val_split(self, sample_data_dir):
        """Test validation split."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        train_dataset = ArchitecturalDepthDataset(config, split="train")
        val_dataset = ArchitecturalDepthDataset(config, split="val")

        assert len(train_dataset) == 10
        assert len(val_dataset) == 3

    def test_dataset_with_dataloader(self, sample_data_dir):
        """Test dataset works with DataLoader."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(loader))
        images, depths = batch

        assert images.shape == (4, 3, 64, 64)
        assert depths.shape == (4, 1, 64, 64)

    def test_dataset_stats(self, sample_data_dir):
        """Test statistics computation."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")
        stats = dataset.get_stats()

        assert stats["num_samples"] == 10
        assert "depth_min" in stats
        assert "depth_max" in stats
        assert "depth_mean" in stats

    def test_missing_directory(self):
        """Test error on missing directory."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir="/nonexistent/path/train",
            val_dir="/nonexistent/path/val",
        )

        with pytest.raises(FileNotFoundError):
            ArchitecturalDepthDataset(config, split="train")


class TestDepthFormats:
    """Test different depth file formats."""

    @pytest.fixture
    def multi_format_data(self):
        """Create data with different depth formats."""
        tmpdir = tempfile.mkdtemp()
        train_dir = Path(tmpdir) / "train"
        images_dir = train_dir / "images"
        depth_dir = train_dir / "depth"

        images_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        # Create common image
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # NPY format
        Image.fromarray(image).save(images_dir / "npy_test.png")
        depth = np.random.rand(64, 64).astype(np.float32) * 100
        np.save(depth_dir / "npy_test.npy", depth)

        # PNG format (16-bit)
        Image.fromarray(image).save(images_dir / "png_test.png")
        depth_16bit = (np.random.rand(64, 64) * 65535).astype(np.uint16)
        Image.fromarray(depth_16bit).save(depth_dir / "png_test.png")

        yield tmpdir

        shutil.rmtree(tmpdir)

    def test_npy_format(self, multi_format_data):
        """Test NPY depth format loading."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(multi_format_data) / "train"),
            val_dir=str(Path(multi_format_data) / "train"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")

        # Should load at least the NPY file
        assert len(dataset) >= 1

    def test_png_format(self, multi_format_data):
        """Test PNG depth format loading."""
        from src.training.depth_dataset import ArchitecturalDepthDataset, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(multi_format_data) / "train"),
            val_dir=str(Path(multi_format_data) / "train"),
            image_size=(64, 64),
        )

        dataset = ArchitecturalDepthDataset(config, split="train")

        # Should find both files
        assert len(dataset) >= 2


class TestDataLoaderFactory:
    """Test data loader factory function."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample data."""
        tmpdir = tempfile.mkdtemp()

        for split in ["train", "val"]:
            split_dir = Path(tmpdir) / split
            images_dir = split_dir / "images"
            depth_dir = split_dir / "depth"

            images_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)

            num_samples = 20 if split == "train" else 5

            for i in range(num_samples):
                image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(image).save(images_dir / f"{split}_{i:03d}.png")

                depth = np.random.rand(64, 64).astype(np.float32) * 100
                np.save(depth_dir / f"{split}_{i:03d}.npy", depth)

        yield tmpdir

        shutil.rmtree(tmpdir)

    def test_create_data_loaders(self, sample_data_dir):
        """Test data loader factory."""
        from src.training.depth_dataset import create_data_loaders, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        loaders = create_data_loaders(
            config,
            batch_size=4,
            num_workers=0,
        )

        assert "train" in loaders
        assert "val" in loaders
        assert len(loaders["train"]) > 0
        assert len(loaders["val"]) > 0

    def test_data_loaders_batch_size(self, sample_data_dir):
        """Test data loaders return correct batch size."""
        from src.training.depth_dataset import create_data_loaders, DepthDataConfig

        config = DepthDataConfig(
            train_dir=str(Path(sample_data_dir) / "train"),
            val_dir=str(Path(sample_data_dir) / "val"),
            image_size=(64, 64),
        )

        loaders = create_data_loaders(
            config,
            batch_size=8,
            num_workers=0,
        )

        images, depths = next(iter(loaders["train"]))

        assert images.shape[0] == 8


class TestDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_split_dataset(self):
        """Test random split of dataset."""
        from src.training.depth_dataset import split_dataset, ArchitecturalDepthDataset, DepthDataConfig

        tmpdir = tempfile.mkdtemp()

        try:
            # Create sample data
            train_dir = Path(tmpdir) / "train"
            images_dir = train_dir / "images"
            depth_dir = train_dir / "depth"

            images_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)

            for i in range(100):
                image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(image).save(images_dir / f"img_{i:03d}.png")
                depth = np.random.rand(32, 32).astype(np.float32)
                np.save(depth_dir / f"img_{i:03d}.npy", depth)

            config = DepthDataConfig(
                train_dir=str(train_dir),
                val_dir=str(train_dir),
                image_size=(32, 32),
            )

            dataset = ArchitecturalDepthDataset(config, split="train")

            train, val, test = split_dataset(
                dataset,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
            )

            assert len(train) == 80
            assert len(val) == 10
            assert len(test) == 10

        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
